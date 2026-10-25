//! Stage 4: validated Semantic Neural IR to LLVM NVPTX kernels.
//!
//! This backend is executable but *not certified*: LLVM, PTX JIT, CUDA driver,
//! hardware and IEEE-754 behavior are outside the current formal proof chain.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::translation::{LOG2_E_F32_LLVM, TranslationKernel, TranslationKind, TranslationPlan};
use crate::{
    DType, Domain, LeanProgram, NeuralError, NeuralGraph, NodeId, NodeStorage, Operation, Shape,
    Tensor, TensorValue,
};

use super::cuda::CudaSession;

/// Fusion is opt-in and experimental: Lean proves the ideal Int rewrite,
/// while equivalence of LLVM/PTX/IEEE-754 machine execution is not proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReluFusion {
    Disabled,
    Experimental,
}

impl ReluFusion {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Experimental => "experimental",
        }
    }
}

/// One-dimensional CUDA launch policy, validated before compilation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuLaunchConfig {
    threads_per_block: u32,
    relu_fusion: ReluFusion,
    approximate_exp: bool,
    pointwise_fusion: bool,
    internal_pointwise_fusion: bool,
}

impl GpuLaunchConfig {
    /// A deliberately narrow set of supported thread counts.
    pub fn new(threads_per_block: u32, relu_fusion: ReluFusion) -> Result<Self, GpuError> {
        if !matches!(threads_per_block, 64 | 128 | 256 | 512) {
            return Err(GpuError::InvalidLaunch { threads_per_block });
        }
        Ok(Self {
            threads_per_block,
            relu_fusion,
            approximate_exp: false,
            pointwise_fusion: false,
            internal_pointwise_fusion: false,
        })
    }

    pub const fn threads_per_block(self) -> u32 {
        self.threads_per_block
    }

    pub const fn relu_fusion(self) -> ReluFusion {
        self.relu_fusion
    }

    /// Explicit opt-in to ex2.approx.f32 for Exp in executable GPU graphs.
    /// This is NOT the strict FP32 or certified numerical profile.
    pub const fn with_approximate_exp(mut self) -> Self {
        self.approximate_exp = true;
        self
    }

    pub const fn approximate_exp(self) -> bool {
        self.approximate_exp
    }

    /// Experimental strict-order Mul -> Add fusion. Never contracts to FMA.
    /// Selected producers, shared operands and unsupported shapes remain unfused.
    pub const fn with_pointwise_fusion(mut self) -> Self {
        self.pointwise_fusion = true;
        self
    }

    pub const fn pointwise_fusion(self) -> bool {
        self.pointwise_fusion
    }

    /// Experimental horizontal fusion for internal, non-public Add/Mul
    /// results with later consumers. Opt in separately from basic fusion.
    pub const fn with_internal_pointwise_fusion(mut self) -> Self {
        self.pointwise_fusion = true;
        self.internal_pointwise_fusion = true;
        self
    }

    pub const fn internal_pointwise_fusion(self) -> bool {
        self.internal_pointwise_fusion
    }
}

impl Default for GpuLaunchConfig {
    fn default() -> Self {
        Self {
            threads_per_block: 256,
            relu_fusion: ReluFusion::Disabled,
            approximate_exp: false,
            pointwise_fusion: false,
            internal_pointwise_fusion: false,
        }
    }
}

#[derive(Debug)]
pub enum GpuError {
    InvalidGraph(NeuralError),
    UnsupportedOperation {
        node: usize,
        operation: &'static str,
    },
    UnsupportedSize {
        elements: usize,
    },
    InvalidInput {
        node: usize,
        reason: &'static str,
    },
    InvalidLaunch {
        threads_per_block: u32,
    },
    InvalidBenchmark {
        iterations: usize,
    },
    NoKernelGraph,
    /// A prior CUDA device transfer/sync failed; contents may be partially updated.
    ExecutionPoisoned,
    /// No successful replay has produced results for the current inputs.
    OutputNotReady,
    /// A CUDA Graph launch is queued on this execution's non-blocking stream;
    /// no readback, feedback or input update is permitted until completion.
    ReplayPending,
    MissingInput {
        node: usize,
    },
    Io(std::io::Error),
    LlvmFailed {
        diagnostics: String,
    },
    Fp32PtxRejected {
        reason: String,
    },
    Driver(String),
    Cuda {
        function: &'static str,
        code: i32,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(err) => write!(f, "invalid neural graph: {err}"),
            Self::UnsupportedOperation { node, operation } => {
                write!(f, "GPU node {node}: {operation} is not supported")
            }
            Self::UnsupportedSize { elements } => {
                write!(
                    f,
                    "GPU tensor length {elements} exceeds supported i32 range"
                )
            }
            Self::InvalidInput { node, reason } => write!(f, "GPU input {node}: {reason}"),
            Self::MissingInput { node } => write!(f, "GPU input {node} is missing"),
            Self::InvalidLaunch { threads_per_block } => write!(
                f,
                "unsupported CUDA threads per block: {threads_per_block} (use 64, 128, 256 or 512)"
            ),
            Self::InvalidBenchmark { iterations } => write!(
                f,
                "GPU benchmark needs at least one measured graph replay, got {iterations}"
            ),
            Self::NoKernelGraph => write!(f, "GPU benchmark requires at least one CUDA kernel"),
            Self::ExecutionPoisoned => write!(
                f,
                "CUDA execution is poisoned; discard and prepare a new execution"
            ),
            Self::OutputNotReady => write!(
                f,
                "GPU outputs are not ready; run a successful CUDA Graph replay first"
            ),
            Self::ReplayPending => write!(
                f,
                "CUDA Graph replay is pending; synchronize before accessing this execution"
            ),
            Self::Io(err) => write!(f, "GPU I/O failure: {err}"),
            Self::LlvmFailed { diagnostics } => write!(f, "NVPTX lowering failed: {diagnostics}"),
            Self::Fp32PtxRejected { reason } => {
                write!(f, "strict FP32 PTX audit rejected lowering: {reason}")
            }
            Self::Driver(message) => write!(f, "CUDA driver unavailable: {message}"),
            Self::Cuda { function, code } => write!(f, "{function} returned CUDA error {code}"),
        }
    }
}

impl Error for GpuError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidGraph(err) => Some(err),
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<NeuralError> for GpuError {
    fn from(err: NeuralError) -> Self {
        Self::InvalidGraph(err)
    }
}

impl From<std::io::Error> for GpuError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// The source-buffer extents required by the emitted kernel template. This
/// describes *logical* element indices, not device pointer validity or PTX.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuMemoryContract {
    pub kernel_node: usize,
    /// (source graph node, required readable 32-bit logical elements).
    /// F32 and Index are both 32-bit in the currently supported GPU profile.
    pub reads: Vec<(usize, usize)>,
    /// Output is written once per active lane, at [0, writes).
    pub writes: usize,
    /// Additional separately allocated, observable buffers written by a
    /// bounded multi-output kernel; never masquerade as read-only arguments.
    pub additional_writes: Vec<(usize, usize)>,
    pub pattern: GpuMemoryPattern,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuMemoryPattern {
    Pointwise,
    BiasAdd {
        columns: usize,
    },
    MatMul {
        rows: usize,
        inner: usize,
        columns: usize,
    },
    Transpose2D {
        rows: usize,
        columns: usize,
    },
    ReverseRows2D {
        rows: usize,
        columns: usize,
    },
    ReverseColumns2D {
        rows: usize,
        columns: usize,
    },
    ConcatAxis0 {
        left: usize,
    },
    /// The contiguous last-axis slice profile for shape [1, N].
    SliceLastRow {
        start: usize,
        source_width: usize,
    },
    /// Replicate one scalar to a [1, N] output.
    BroadcastScalar,
    ArgMaxLast {
        width: usize,
    },
    TakeAxis0 {
        lane: usize,
        choices: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuAssuranceStatus {
    /// A deterministic Lean obligation is generated, but must still be checked.
    GeneratedLeanObligation,
    /// The check is mandatory when LLVM is lowered to PTX.
    RequiredAtLowering,
    /// A separate numeric proof/profile is required for this semantic link.
    SeparateNumericProofRequired,
    /// This layer is established only by an explicit runtime acceptance test.
    RuntimeAcceptanceRequired,
    /// No proof or acceptance claim is currently established.
    NotEstablished,
}

impl GpuAssuranceStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeneratedLeanObligation => "generated-lean-obligation",
            Self::RequiredAtLowering => "required-at-lowering",
            Self::SeparateNumericProofRequired => "separate-numeric-proof-required",
            Self::RuntimeAcceptanceRequired => "runtime-acceptance-required",
            Self::NotEstablished => "not-established",
        }
    }
}

/// Explicit trust boundary for one GPU capability plan. These are obligations
/// and verification stages, not claims that every layer has already been proved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuAssuranceBoundary {
    pub memory_address_arithmetic: GpuAssuranceStatus,
    pub llvm_ptx_structure: GpuAssuranceStatus,
    pub ideal_to_fp32: GpuAssuranceStatus,
    pub cuda_runtime: GpuAssuranceStatus,
    pub jit_sass_correspondence: GpuAssuranceStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuKernelCapability {
    pub node: usize,
    pub operation: &'static str,
    pub memory_pattern: GpuMemoryPattern,
    pub translation_kind: TranslationKind,
    pub reads: Vec<(usize, usize)>,
    pub writes: usize,
    pub additional_writes: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuOptimizationKind {
    ReluProducerFusion,
    PointwiseBiasReluFusion,
    MulAddFusion,
    MulAddMulFusion,
    BinaryPointwiseFusion,
    SqrtAddDivFusion,
    MulMulAddDivFusion,
    MulDiamondAddFusion,
    DualPointwiseOutputFusion,
}

impl GpuOptimizationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReluProducerFusion => "relu-producer-fusion",
            Self::PointwiseBiasReluFusion => "pointwise-bias-relu-fusion",
            Self::MulAddFusion => "mul-add-fusion",
            Self::MulAddMulFusion => "mul-add-mul-fusion",
            Self::BinaryPointwiseFusion => "binary-pointwise-fusion",
            Self::SqrtAddDivFusion => "sqrt-add-div-fusion",
            Self::MulMulAddDivFusion => "mul-mul-add-div-fusion",
            Self::MulDiamondAddFusion => "mul-diamond-add-fusion",
            Self::DualPointwiseOutputFusion => "dual-pointwise-output-fusion",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOptimizationCapability {
    pub root: usize,
    pub kind: GpuOptimizationKind,
    /// Semantic IR nodes whose standalone kernels are removed by the rewrite.
    pub removed_nodes: Vec<usize>,
    /// Checked against the reachable graph when the capability plan is built.
    pub exclusive_reachable_use: bool,
    /// Selected architecture outputs are never among removed nodes.
    pub selected_outputs_preserved: bool,
}

/// Typed explanation of what the GPU compiler can actually lower for this
/// concrete program. Construction is fail-closed: every launched kernel must
/// have exactly one matching memory contract and translation rule, and every
/// removed fusion node must satisfy the recorded structural preconditions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCapabilityPlan {
    pub kernels: Vec<GpuKernelCapability>,
    pub optimizations: Vec<GpuOptimizationCapability>,
    pub boundary: GpuAssuranceBoundary,
}

impl GpuCapabilityPlan {
    pub fn to_text(&self) -> String {
        let mut text = String::from("verineural-gpu-capability-plan-v1\n");
        text.push_str(&format!(
            "memory_address_arithmetic={}\nllvm_ptx_structure={}\nideal_to_fp32={}\ncuda_runtime={}\njit_sass_correspondence={}\n",
            self.boundary.memory_address_arithmetic.as_str(),
            self.boundary.llvm_ptx_structure.as_str(),
            self.boundary.ideal_to_fp32.as_str(),
            self.boundary.cuda_runtime.as_str(),
            self.boundary.jit_sass_correspondence.as_str(),
        ));
        for kernel in &self.kernels {
            text.push_str(&format!(
                "kernel=node:{};op={};memory={:?};translation={};reads={:?};writes={};additional_writes={:?}\n",
                kernel.node,
                kernel.operation,
                kernel.memory_pattern,
                kernel.translation_kind.name(),
                kernel.reads,
                kernel.writes,
                kernel.additional_writes,
            ));
        }
        for optimization in &self.optimizations {
            text.push_str(&format!(
                "optimization=root:{};kind={};removed={:?};exclusive_reachable_use={};selected_outputs_preserved={}\n",
                optimization.root,
                optimization.kind.as_str(),
                optimization.removed_nodes,
                optimization.exclusive_reachable_use,
                optimization.selected_outputs_preserved,
            ));
        }
        text
    }
}

/// A checked, immutable GPU compilation plan. No claims about machine-code proof.
/// Static inspection of separate, rounded arithmetic in emitted PTX.
/// This does not establish LLVM translation or machine/SASS correctness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp32PtxAudit {
    pub rounded_adds: usize,
    pub rounded_muls: usize,
}

/// The audit compares explicit RNE inline PTX in LLVM with PTX instructions.
/// It fails closed on contraction, FTZ, unapproved FP32 arithmetic or missing
/// instructions. PTX itself, CUDA JIT and hardware remain in the TCB.
pub fn audit_fp32_ptx(llvm_ir: &str, ptx: &str) -> Result<Fp32PtxAudit, GpuError> {
    let rejected = |reason: String| GpuError::Fp32PtxRejected { reason };
    if llvm_ir.contains("fadd float")
        || llvm_ir.contains("fmul float")
        || llvm_ir.contains("fast ")
        || llvm_ir.contains("contract ")
    {
        return Err(rejected(
            "non-explicit/fast LLVM floating point operation".into(),
        ));
    }
    const ADD_ASM: &str = "call float asm sideeffect \"add.rn.f32";
    const MUL_ASM: &str = "call float asm sideeffect \"mul.rn.f32";
    let expected_adds = llvm_ir.matches(ADD_ASM).count();
    let expected_muls = llvm_ir.matches(MUL_ASM).count();
    // Explicitly authored inline PTX, not arbitrary machine substitutions.
    let expected_subs = llvm_ir
        .matches("call float asm sideeffect \"sub.rn.f32")
        .count();
    let expected_divs = llvm_ir
        .matches("call float asm sideeffect \"div.rn.f32")
        .count();
    let expected_exp2s = llvm_ir
        .matches("call float asm sideeffect \"ex2.approx.f32")
        .count();
    let expected_sqrts = llvm_ir
        .matches("call float asm sideeffect \"sqrt.rn.f32")
        .count();
    let mut rounded_subs = 0usize;
    let mut rounded_divs = 0usize;
    let mut approximate_exp2s = 0usize;
    let mut rounded_sqrts = 0usize;
    let mut rounded_adds = 0usize;
    let mut rounded_muls = 0usize;
    let mut block_comment = false;
    for line in ptx.lines() {
        let mut code = String::new();
        let mut remaining = line;
        while !remaining.is_empty() {
            if block_comment {
                if let Some(end) = remaining.find("*/") {
                    remaining = &remaining[end + 2..];
                    block_comment = false;
                } else {
                    break;
                }
            } else {
                let line_comment = remaining.find("//").unwrap_or(remaining.len());
                let block = remaining.find("/*").unwrap_or(remaining.len());
                let end = line_comment.min(block);
                code.push_str(&remaining[..end]);
                if end == remaining.len() || end == line_comment {
                    break;
                }
                remaining = &remaining[end + 2..];
                block_comment = true;
            }
        }
        let mut instruction = code.trim();
        if instruction.starts_with('@') {
            instruction = instruction
                .split_once(char::is_whitespace)
                .map_or("", |(_, rest)| rest.trim_start());
        }
        let opcode = instruction.split_whitespace().next().unwrap_or("");
        if !opcode.ends_with(".f32") {
            continue;
        }
        if opcode.contains(".ftz.")
            || (opcode.contains(".approx.") && opcode != "ex2.approx.f32")
            || opcode.starts_with("fma.")
            || opcode.starts_with("mad.")
        {
            return Err(rejected(format!("prohibited PTX FP32 opcode: {opcode}")));
        }
        match opcode {
            "add.rn.f32" => rounded_adds += 1,
            "mul.rn.f32" => rounded_muls += 1,
            "sub.rn.f32" => rounded_subs += 1,
            "div.rn.f32" => rounded_divs += 1,
            "ex2.approx.f32" => approximate_exp2s += 1,
            "sqrt.rn.f32" => rounded_sqrts += 1,
            _ if opcode.starts_with("setp.")
                || opcode.starts_with("selp.")
                || opcode.starts_with("mov.")
                || opcode.starts_with("ld.")
                || opcode.starts_with("st.") => {}
            _ => return Err(rejected(format!("unapproved PTX FP32 opcode: {opcode}"))),
        }
    }
    if rounded_subs != expected_subs
        || rounded_divs != expected_divs
        || approximate_exp2s != expected_exp2s
        || rounded_sqrts != expected_sqrts
    {
        return Err(rejected(format!(
            "pointwise arithmetic count mismatch: LLVM sub/div/ex2/sqrt={expected_subs}/{expected_divs}/{expected_exp2s}/{expected_sqrts}; PTX={rounded_subs}/{rounded_divs}/{approximate_exp2s}/{rounded_sqrts}"
        )));
    }
    if rounded_adds != expected_adds || rounded_muls != expected_muls {
        return Err(rejected(format!(
            "rounded arithmetic count mismatch: LLVM add/mul={expected_adds}/{expected_muls}, PTX add/mul={rounded_adds}/{rounded_muls}"
        )));
    }
    Ok(Fp32PtxAudit {
        rounded_adds,
        rounded_muls,
    })
}

/// An intentionally bounded executable subset of the versioned pointwise DSL.
/// Signature and expression must both match: arbitrary custom ASTs fail closed.
/// No architecture-specific primitive enters the compiler.
fn supported_pointwise(spec: &crate::PointwiseSpec) -> Option<TranslationKind> {
    match (
        spec.id(),
        spec.version(),
        spec.arity(),
        spec.expression().canonical().as_str(),
    ) {
        ("verineural.sub", "1", 2, "sub(arg:0,arg:1)") => Some(TranslationKind::Sub),
        ("verineural.div", "1", 2, "div(arg:0,arg:1)") => Some(TranslationKind::DivRne),
        ("verineural.exp", "1", 1, "exp(arg:0)") => Some(TranslationKind::ExpApprox),
        ("verineural.sqrt", "1", 1, "sqrt(arg:0)") => Some(TranslationKind::SqrtRne),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct GpuProgram {
    graph: NeuralGraph,
    output: NodeId,
    outputs: Vec<NodeId>,
    live_nodes: Vec<usize>,
    source_inputs: Vec<NodeId>,
    llvm_ir: String,
    launch: GpuLaunchConfig,
    // Relu index -> exclusive immediate producer index.
    fused_relus: BTreeMap<usize, usize>,
    // Relu root -> [exclusive Add/Mul producer, exclusive BiasAdd producer].
    fused_pointwise_chains: BTreeMap<usize, Vec<usize>>,
    // Add root -> (exclusive Mul producer, root operand position: 0 or 1).
    fused_mul_adds: BTreeMap<usize, (usize, usize)>,
    // final Mul root -> (first Mul, intermediate Add, Add position,
    // outer Mul position, reused external source index 0..=2).
    fused_mul_add_muls: BTreeMap<usize, (usize, usize, usize, usize, usize)>,
    // Binary Add/Mul root -> (exclusive Add/Mul producer, root operand slot).
    fused_binary_pairs: BTreeMap<usize, (usize, usize)>,
    // Div root -> (Sqrt inner, Add middle, Sqrt's Add operand slot).
    fused_sqrt_add_divs: BTreeMap<usize, (usize, usize, usize)>,
    // Div root -> (first Mul, second Mul, Add, reused first input
    // selector 0/1, second Mul internal slot 0/1, Add internal slot 0/1).
    fused_mul_mul_add_divs: BTreeMap<usize, (usize, usize, usize, usize, usize, usize)>,
    // Add root -> (left Mul, right-first Mul, right-second Mul,
    // reused first-input slot, right internal slot, Add left slot).
    fused_mul_diamond_adds: BTreeMap<usize, (usize, usize, usize, usize, usize, usize)>,
    // Later selected Add/Mul root -> independent earlier selected Add/Mul
    // root. Both outputs remain individually allocated and observable.
    fused_dual_outputs: BTreeMap<usize, usize>,
}

impl GpuProgram {
    pub fn llvm_ir(&self) -> &str {
        &self.llvm_ir
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.source_inputs
    }

    pub fn output(&self) -> NodeId {
        self.output
    }

    /// Ordered, distinct architecture outputs served by the same kernel DAG.
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Exactly one kernel per live Semantic IR producer, even when multiple
    /// selected outputs depend on it. Zero-copy Reshape retains its alias.
    pub fn kernel_count(&self) -> Result<usize, GpuError> {
        Ok(self.memory_contracts()?.len())
    }

    pub const fn launch_config(&self) -> GpuLaunchConfig {
        self.launch
    }

    /// (consumer Relu node, producer binary node) pairs for the selected IR rewrite.
    pub fn fused_relus(&self) -> &BTreeMap<usize, usize> {
        &self.fused_relus
    }

    pub fn fused_pointwise_chains(&self) -> &BTreeMap<usize, Vec<usize>> {
        &self.fused_pointwise_chains
    }

    pub fn fused_mul_adds(&self) -> &BTreeMap<usize, (usize, usize)> {
        &self.fused_mul_adds
    }

    pub fn fused_mul_add_muls(&self) -> &BTreeMap<usize, (usize, usize, usize, usize, usize)> {
        &self.fused_mul_add_muls
    }

    /// Other strict-order binary pointwise pairs fused into one CUDA kernel.
    pub fn fused_binary_pairs(&self) -> &BTreeMap<usize, (usize, usize)> {
        &self.fused_binary_pairs
    }

    /// Opt-in rounded Sqrt→Add→Div chains with exclusive intermediate uses.
    pub fn fused_sqrt_add_divs(&self) -> &BTreeMap<usize, (usize, usize, usize)> {
        &self.fused_sqrt_add_divs
    }

    pub fn fused_mul_mul_add_divs(
        &self,
    ) -> &BTreeMap<usize, (usize, usize, usize, usize, usize, usize)> {
        &self.fused_mul_mul_add_divs
    }

    /// Four-op pointwise DAG fusion which preserves the selected Add root.
    pub fn fused_mul_diamond_adds(
        &self,
    ) -> &BTreeMap<usize, (usize, usize, usize, usize, usize, usize)> {
        &self.fused_mul_diamond_adds
    }

    pub fn removed_kernel_count(&self) -> usize {
        self.skipped_nodes().len()
    }

    /// Bounded, multi-output, strict-RNE fusion of independent live pointwise
    /// roots, including non-public intermediates whose consumers occur later.
    /// Both buffers remain materialized; IR and observable values are unchanged.
    pub fn fused_dual_outputs(&self) -> &BTreeMap<usize, usize> {
        &self.fused_dual_outputs
    }

    fn skipped_nodes(&self) -> BTreeSet<usize> {
        let mut skipped = self.fused_relus.values().copied().collect::<BTreeSet<_>>();
        for chain in self.fused_pointwise_chains.values() {
            skipped.extend(chain.iter().copied());
        }
        skipped.extend(self.fused_mul_adds.values().map(|&(mul, _)| mul));
        skipped.extend(
            self.fused_binary_pairs
                .values()
                .map(|&(producer, _)| producer),
        );
        for &(mul, add, _, _, _) in self.fused_mul_add_muls.values() {
            skipped.insert(mul);
            skipped.insert(add);
        }
        for &(sqrt, add, _) in self.fused_sqrt_add_divs.values() {
            skipped.insert(sqrt);
            skipped.insert(add);
        }
        for &(inner, middle, add, _, _, _) in self.fused_mul_mul_add_divs.values() {
            skipped.extend([inner, middle, add]);
        }
        for &(left, first, second, _, _, _) in self.fused_mul_diamond_adds.values() {
            skipped.extend([left, first, second]);
        }
        skipped.extend(self.fused_dual_outputs.values().copied());
        skipped
    }

    /// Independently describe and check every launched kernel's element
    /// addresses. The same input selection drives CUDA graph capture.
    pub fn memory_contracts(&self) -> Result<Vec<GpuMemoryContract>, GpuError> {
        let mut result = Vec::new();
        let skipped = self.skipped_nodes();
        for &index in &self.live_nodes {
            if skipped.contains(&index) {
                continue;
            }
            let node = &self.graph.nodes()[index];
            if matches!(node.operation, Operation::Input) || self.graph.storage(node.id)?.is_alias()
            {
                continue;
            }
            let writes = node.shape.element_count();
            let (read_nodes, pattern): (Vec<usize>, GpuMemoryPattern) = if let Some(&earlier) =
                self.fused_dual_outputs.get(&index)
            {
                let first = &self.graph.nodes()[earlier];
                (
                    first
                        .inputs
                        .iter()
                        .chain(&node.inputs)
                        .map(|id| id.index())
                        .collect(),
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(left, first, _, _, _, _)) = self.fused_mul_diamond_adds.get(&index)
            {
                let lhs = &self.graph.nodes()[left];
                let rhs = &self.graph.nodes()[first];
                (
                    vec![
                        lhs.inputs[0].index(),
                        lhs.inputs[1].index(),
                        rhs.inputs[0].index(),
                        rhs.inputs[1].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(inner, _, add, _, _, add_slot)) =
                self.fused_mul_mul_add_divs.get(&index)
            {
                let first = &self.graph.nodes()[inner];
                let sum = &self.graph.nodes()[add];
                (
                    vec![
                        first.inputs[0].index(),
                        first.inputs[1].index(),
                        sum.inputs[1 - add_slot].index(),
                        node.inputs[1].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(sqrt, add, slot)) = self.fused_sqrt_add_divs.get(&index) {
                let inner = &self.graph.nodes()[sqrt];
                let middle = &self.graph.nodes()[add];
                (
                    vec![
                        inner.inputs[0].index(),
                        middle.inputs[1 - slot].index(),
                        node.inputs[0].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(mul, add, slot, _, _)) = self.fused_mul_add_muls.get(&index) {
                let inner = &self.graph.nodes()[mul];
                let middle = &self.graph.nodes()[add];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        middle.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(mul, slot)) = self.fused_mul_adds.get(&index) {
                let producer = &self.graph.nodes()[mul];
                (
                    vec![
                        producer.inputs[0].index(),
                        producer.inputs[1].index(),
                        node.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(&(producer, slot)) = self.fused_binary_pairs.get(&index) {
                let inner = &self.graph.nodes()[producer];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        node.inputs[1 - slot].index(),
                    ],
                    GpuMemoryPattern::Pointwise,
                )
            } else if let Some(chain) = self.fused_pointwise_chains.get(&index) {
                let inner = &self.graph.nodes()[chain[0]];
                let bias = &self.graph.nodes()[chain[1]];
                (
                    vec![
                        inner.inputs[0].index(),
                        inner.inputs[1].index(),
                        bias.inputs[1].index(),
                    ],
                    GpuMemoryPattern::BiasAdd {
                        columns: node.shape.dims()[1],
                    },
                )
            } else {
                let producer = self.fused_relus.get(&index).copied().unwrap_or(index);
                let op_node = &self.graph.nodes()[producer];
                let pattern = match &op_node.operation {
                    Operation::BiasAdd => GpuMemoryPattern::BiasAdd {
                        columns: node.shape.dims()[1],
                    },
                    Operation::MatMul => {
                        let left = &self.graph.nodes()[op_node.inputs[0].index()];
                        GpuMemoryPattern::MatMul {
                            rows: left.shape.dims()[0],
                            inner: left.shape.dims()[1],
                            columns: node.shape.dims()[1],
                        }
                    }
                    Operation::Permute { axes } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2 || axes.as_slice() != [1, 0] {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::Transpose2D {
                            rows: input.shape.dims()[0],
                            columns: input.shape.dims()[1],
                        }
                    }
                    Operation::ReverseAxis { axis } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        match axis {
                            0 => GpuMemoryPattern::ReverseRows2D {
                                rows: input.shape.dims()[0],
                                columns: input.shape.dims()[1],
                            },
                            1 => GpuMemoryPattern::ReverseColumns2D {
                                rows: input.shape.dims()[0],
                                columns: input.shape.dims()[1],
                            },
                            _ => {
                                return Err(GpuError::UnsupportedOperation {
                                    node: index,
                                    operation: op_node.operation.name(),
                                });
                            }
                        }
                    }
                    Operation::Concat { axis } => {
                        let left = &self.graph.nodes()[op_node.inputs[0].index()];
                        let right = &self.graph.nodes()[op_node.inputs[1].index()];
                        if *axis != 0
                            && !(*axis == 1
                                && left.shape.rank() == 2
                                && left.shape.dims()[0] == 1
                                && right.shape.dims()[0] == 1)
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::ConcatAxis0 {
                            left: left.shape.element_count(),
                        }
                    }
                    Operation::SliceAxis {
                        axis: 1,
                        start,
                        len,
                    } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2
                            || input.shape.dims()[0] != 1
                            || *len != writes
                            || start
                                .checked_add(*len)
                                .is_none_or(|end| end > input.shape.dims()[1])
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::SliceLastRow {
                            start: *start,
                            source_width: input.shape.dims()[1],
                        }
                    }
                    Operation::BroadcastTo(_) => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        // Broadcasting a one-element tensor uses source
                        // address zero for every output lane, independently
                        // of output rank. General axis-wise broadcasting
                        // still has no GPU lowering and must fail closed.
                        if input.shape.element_count() != 1 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::BroadcastScalar
                    }
                    Operation::ArgMax { axis, tie_break } => {
                        let input = &self.graph.nodes()[op_node.inputs[0].index()];
                        if input.shape.rank() != 2
                            || *axis != 1
                            || !matches!(tie_break, crate::ArgMaxTieBreak::First)
                        {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: op_node.operation.name(),
                            });
                        }
                        GpuMemoryPattern::ArgMaxLast {
                            width: input.shape.dims()[1],
                        }
                    }
                    Operation::Take { axis: 0 } => {
                        let data = &self.graph.nodes()[op_node.inputs[0].index()];
                        GpuMemoryPattern::TakeAxis0 {
                            lane: writes,
                            choices: data.shape.dims()[0],
                        }
                    }
                    _ => GpuMemoryPattern::Pointwise,
                };
                (
                    op_node.inputs.iter().map(|id| id.index()).collect(),
                    pattern,
                )
            };
            let required = match pattern {
                GpuMemoryPattern::Pointwise => vec![writes; read_nodes.len()],
                GpuMemoryPattern::BiasAdd { columns } => {
                    if read_nodes.len() == 3 {
                        vec![writes, writes, columns]
                    } else {
                        vec![writes, columns]
                    }
                }
                GpuMemoryPattern::MatMul {
                    rows,
                    inner,
                    columns,
                } => {
                    if rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "MatMul output extent does not match row-major indexing",
                        });
                    }
                    vec![
                        rows.checked_mul(inner)
                            .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                        inner
                            .checked_mul(columns)
                            .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                    ]
                }
                GpuMemoryPattern::Transpose2D { rows, columns } => {
                    if rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "transpose2d output extent mismatch",
                        });
                    }
                    vec![writes]
                }
                GpuMemoryPattern::ReverseRows2D { rows, columns }
                | GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(writes) {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "reverse2d output extent mismatch",
                        });
                    }
                    vec![writes]
                }
                GpuMemoryPattern::SliceLastRow {
                    start,
                    source_width,
                } => {
                    if read_nodes.len() != 1
                        || start
                            .checked_add(writes)
                            .is_none_or(|end| end > source_width)
                    {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "slice source extent mismatch",
                        });
                    }
                    vec![source_width]
                }
                GpuMemoryPattern::BroadcastScalar => vec![1],
                GpuMemoryPattern::ConcatAxis0 { left } => {
                    if read_nodes.len() != 2 {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "concat_axis0 requires two source tensors",
                        });
                    }
                    let right = writes.checked_sub(left).ok_or(GpuError::InvalidInput {
                        node: index,
                        reason: "concat_axis0 left extent exceeds output",
                    })?;
                    vec![left, right]
                }
                GpuMemoryPattern::ArgMaxLast { width } => vec![
                    writes
                        .checked_mul(width)
                        .ok_or(GpuError::UnsupportedSize { elements: writes })?,
                ],
                GpuMemoryPattern::TakeAxis0 { lane, choices } => {
                    let data = &self.graph.nodes()[read_nodes[0]];
                    if lane != writes
                        || choices == 0
                        || choices.checked_mul(lane) != Some(data.shape.element_count())
                    {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "take_axis0 source extent mismatch",
                        });
                    }
                    vec![data.shape.element_count(), 1]
                }
            };
            // CUDA templates form idx = block * width + lane with u32
            // wrap semantics, then test idx < length as unsigned. Bound the
            // *whole* grid so even unused trailing lanes cannot wrap.
            let width = self.launch.threads_per_block as usize;
            let launch_extent = writes.div_ceil(width).checked_mul(width);
            if !matches!(read_nodes.len(), 1..=4)
                || read_nodes.len() != required.len()
                || writes == 0
                || writes > i32::MAX as usize
                || launch_extent.is_none_or(|n| n > u32::MAX as usize)
                || read_nodes.iter().any(|&source| source >= index)
            {
                return Err(GpuError::InvalidInput {
                    node: index,
                    reason: "unsupported CUDA kernel memory access or launch bounds",
                });
            }
            if let GpuMemoryPattern::BiasAdd { columns } = pattern {
                if columns == 0 || writes % columns != 0 {
                    return Err(GpuError::InvalidInput {
                        node: index,
                        reason: "BiasAdd requires nonempty aligned rows",
                    });
                }
            }
            let mut reads = Vec::with_capacity(read_nodes.len());
            for (source, needed) in read_nodes.into_iter().zip(required) {
                if self.graph.nodes()[source].shape.element_count() != needed
                    || needed > i32::MAX as usize
                {
                    return Err(GpuError::InvalidInput {
                        node: index,
                        reason: "kernel input extent differs from its indexed access",
                    });
                }
                reads.push((source, needed));
            }
            result.push(GpuMemoryContract {
                kernel_node: index,
                reads,
                writes,
                additional_writes: self
                    .fused_dual_outputs
                    .get(&index)
                    .map_or_else(Vec::new, |&earlier| vec![(earlier, writes)]),
                pattern,
            });
        }
        Ok(result)
    }

    /// Derive the expected kernel composition from the original reachable
    /// graph, checked fusion decisions and exact memory contracts, not PTX.
    /// Each plan row is later cross-checked against LLVM and PTX per kernel.
    pub fn translation_plan(&self) -> Result<TranslationPlan, GpuError> {
        let mut kernels = Vec::new();
        for contract in self.memory_contracts()? {
            let node = &self.graph.nodes()[contract.kernel_node];
            let kind = if let Some(&earlier) = self.fused_dual_outputs.get(&contract.kernel_node) {
                TranslationKind::DualPointwise {
                    earlier,
                    first_mul: matches!(self.graph.nodes()[earlier].operation, Operation::Mul),
                    second_mul: matches!(node.operation, Operation::Mul),
                }
            } else if let Some(&(_, _, _, reused, right_slot, left_slot)) =
                self.fused_mul_diamond_adds.get(&contract.kernel_node)
            {
                TranslationKind::MulDiamondAdd {
                    reused_source: reused as u8,
                    right_mul_on_right: right_slot == 1,
                    left_mul_on_right: left_slot == 1,
                }
            } else if let Some(&(_, _, _, reused, middle_slot, add_slot)) =
                self.fused_mul_mul_add_divs.get(&contract.kernel_node)
            {
                TranslationKind::MulMulAddDiv {
                    reused_source: reused as u8,
                    multiply_on_right: middle_slot == 1,
                    add_on_right: add_slot == 1,
                }
            } else if let Some(&(_, _, slot)) = self.fused_sqrt_add_divs.get(&contract.kernel_node)
            {
                TranslationKind::SqrtAddDiv {
                    sqrt_on_right: slot == 1,
                }
            } else if let Some(&(_, _, add_slot, outer_slot, reuse)) =
                self.fused_mul_add_muls.get(&contract.kernel_node)
            {
                TranslationKind::MulAddMul {
                    add_mul_on_right: add_slot == 1,
                    mul_add_on_right: outer_slot == 1,
                    reused_source: reuse as u8,
                }
            } else if let Some(&(_, slot)) = self.fused_mul_adds.get(&contract.kernel_node) {
                TranslationKind::MulAdd {
                    multiply_on_right: slot == 1,
                }
            } else if let Some(&(producer, slot)) =
                self.fused_binary_pairs.get(&contract.kernel_node)
            {
                TranslationKind::BinaryChain {
                    first_add: matches!(self.graph.nodes()[producer].operation, Operation::Add),
                    second_add: matches!(node.operation, Operation::Add),
                    producer_on_right: slot == 1,
                }
            } else if let Some(chain) = self.fused_pointwise_chains.get(&contract.kernel_node) {
                let inner = &self.graph.nodes()[chain[0]].operation;
                let columns = match contract.pattern {
                    GpuMemoryPattern::BiasAdd { columns } => columns,
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "fused chain lacks BiasAdd memory contract".into(),
                        });
                    }
                };
                match inner {
                    Operation::Add => TranslationKind::AddBiasRelu { columns },
                    Operation::Mul => TranslationKind::MulBiasRelu { columns },
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unsupported fused inner operation".into(),
                        });
                    }
                }
            } else if let Some(&producer) = self.fused_relus.get(&contract.kernel_node) {
                match self.graph.nodes()[producer].operation {
                    Operation::Add => TranslationKind::AddRelu,
                    Operation::Mul => TranslationKind::MulRelu,
                    Operation::BiasAdd => {
                        let columns = match contract.pattern {
                            GpuMemoryPattern::BiasAdd { columns } => columns,
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "fused bias lacks broadcast memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::BiasAddRelu { columns }
                    }
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unsupported fused producer operation".into(),
                        });
                    }
                }
            } else {
                match node.operation {
                    Operation::Add => TranslationKind::Add,
                    Operation::Mul => TranslationKind::Mul,
                    Operation::Pointwise(ref spec) => {
                        supported_pointwise(spec).ok_or(GpuError::UnsupportedOperation {
                            node: contract.kernel_node,
                            operation: node.operation.name(),
                        })?
                    }
                    Operation::BiasAdd => {
                        let columns = match contract.pattern {
                            GpuMemoryPattern::BiasAdd { columns } => columns,
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "bias lacks broadcast memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::BiasAdd { columns }
                    }
                    Operation::MatMul => {
                        let (inner, columns) = match contract.pattern {
                            GpuMemoryPattern::MatMul { inner, columns, .. } => (inner, columns),
                            _ => {
                                return Err(GpuError::Fp32PtxRejected {
                                    reason: "MatMul lacks reduction memory contract".into(),
                                });
                            }
                        };
                        TranslationKind::MatMul { inner, columns }
                    }
                    Operation::Permute { .. } => match contract.pattern {
                        GpuMemoryPattern::Transpose2D { rows, columns } => {
                            TranslationKind::Transpose2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Permute lacks transpose2d memory contract".into(),
                            });
                        }
                    },
                    Operation::ReverseAxis { axis: 0 } => match contract.pattern {
                        GpuMemoryPattern::ReverseRows2D { rows, columns } => {
                            TranslationKind::ReverseRows2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ReverseAxis(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::ReverseAxis { axis: 1 } => match contract.pattern {
                        GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                            TranslationKind::ReverseColumns2D { rows, columns }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ReverseAxis(1) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::BroadcastTo(_) => match contract.pattern {
                        GpuMemoryPattern::BroadcastScalar => TranslationKind::BroadcastScalar,
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "BroadcastTo lacks scalar memory contract".into(),
                            });
                        }
                    },
                    Operation::SliceAxis { axis: 1, .. } => match contract.pattern {
                        GpuMemoryPattern::SliceLastRow {
                            start,
                            source_width,
                        } => TranslationKind::SliceLastRow {
                            start,
                            source_width,
                        },
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "SliceAxis(1) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Concat { .. } => match contract.pattern {
                        GpuMemoryPattern::ConcatAxis0 { left } => {
                            TranslationKind::ConcatAxis0 { left }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Concat(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::ArgMax { axis: 1, .. } => match contract.pattern {
                        GpuMemoryPattern::ArgMaxLast { width } => {
                            TranslationKind::ArgMaxLast { width }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "ArgMax(last) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Take { axis: 0 } => match contract.pattern {
                        GpuMemoryPattern::TakeAxis0 { lane, .. } => {
                            TranslationKind::TakeAxis0 { lane }
                        }
                        _ => {
                            return Err(GpuError::Fp32PtxRejected {
                                reason: "Take(0) lacks memory contract".into(),
                            });
                        }
                    },
                    Operation::Relu => TranslationKind::Relu,
                    _ => {
                        return Err(GpuError::Fp32PtxRejected {
                            reason: "unexpected operation in kernel translation plan".into(),
                        });
                    }
                }
            };
            kernels.push(TranslationKernel {
                node: contract.kernel_node,
                kind,
                reads: contract.reads,
                writes: contract.writes,
            });
        }
        Ok(TranslationPlan {
            threads_per_block: self.launch.threads_per_block(),
            kernels,
        })
    }

    /// Cross-check the concrete lowering profile before it can be called
    /// supported by higher layers. This validates evidence coverage and fusion
    /// preconditions only; it does not execute Lean, LLVM, CUDA or the GPU.
    pub fn capability_plan(&self) -> Result<GpuCapabilityPlan, GpuError> {
        let contracts = self.memory_contracts()?;
        let translation = self.translation_plan()?;
        if contracts.len() != translation.kernels.len() {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan kernel/memory coverage mismatch".into(),
            });
        }

        let mut kernels = Vec::with_capacity(contracts.len());
        for (contract, rule) in contracts.iter().zip(&translation.kernels) {
            if contract.kernel_node != rule.node
                || contract.reads != rule.reads
                || contract.writes != rule.writes
                || contract.additional_writes != rule.kind.extra_writes(contract.writes)
            {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!(
                        "capability evidence mismatch for kernel node {}",
                        contract.kernel_node
                    ),
                });
            }
            kernels.push(GpuKernelCapability {
                node: contract.kernel_node,
                operation: self.graph.nodes()[contract.kernel_node].operation.name(),
                memory_pattern: contract.pattern,
                translation_kind: rule.kind,
                reads: contract.reads.clone(),
                writes: contract.writes,
                additional_writes: contract.additional_writes.clone(),
            });
        }

        let skipped = self.skipped_nodes();
        let launched = kernels
            .iter()
            .map(|kernel| kernel.node)
            .collect::<BTreeSet<_>>();
        let mut expected = BTreeSet::new();
        for &index in &self.live_nodes {
            let node = &self.graph.nodes()[index];
            if !skipped.contains(&index)
                && !matches!(node.operation, Operation::Input)
                && !self.graph.storage(node.id)?.is_alias()
            {
                expected.insert(index);
            }
        }
        if launched != expected {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan does not cover every launched semantic node".into(),
            });
        }

        let mut uses = vec![0usize; self.graph.nodes().len()];
        for &index in &self.live_nodes {
            for input in &self.graph.nodes()[index].inputs {
                uses[input.index()] += 1;
            }
        }
        for output in &self.outputs {
            uses[output.index()] += 1;
        }
        let selected = self
            .outputs
            .iter()
            .map(|id| id.index())
            .collect::<BTreeSet<_>>();
        let mut optimizations = Vec::new();
        let mut covered_removed = BTreeSet::new();

        for (&root, chain) in &self.fused_pointwise_chains {
            if chain.len() != 2 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: "fused pointwise chain has invalid length".into(),
                });
            }
            let inner = chain[0];
            let bias = chain[1];
            let root_node = &self.graph.nodes()[root];
            let inner_node = &self.graph.nodes()[inner];
            let bias_node = &self.graph.nodes()[bias];
            let exclusive = uses[inner] == 1 && uses[bias] == 1;
            let preserves = !selected.contains(&inner) && !selected.contains(&bias);
            let compatible = matches!(root_node.operation, Operation::Relu)
                && root_node.inputs.len() == 1
                && root_node.inputs[0].index() == bias
                && matches!(bias_node.operation, Operation::BiasAdd)
                && bias_node.inputs[0].index() == inner
                && matches!(inner_node.operation, Operation::Add | Operation::Mul)
                && self.fused_relus.get(&root).copied() == Some(bias);
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid pointwise fusion capability at node {root}"),
                });
            }
            covered_removed.insert(inner);
            covered_removed.insert(bias);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::PointwiseBiasReluFusion,
                removed_nodes: chain.clone(),
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &producer) in &self.fused_relus {
            if self.fused_pointwise_chains.contains_key(&root) {
                continue;
            }
            let root_node = &self.graph.nodes()[root];
            let producer_node = &self.graph.nodes()[producer];
            let exclusive = uses[producer] == 1;
            let preserves = !selected.contains(&producer);
            let compatible = matches!(root_node.operation, Operation::Relu)
                && root_node.inputs.len() == 1
                && root_node.inputs[0].index() == producer
                && matches!(
                    producer_node.operation,
                    Operation::Add | Operation::Mul | Operation::BiasAdd
                );
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid relu fusion capability at node {root}"),
                });
            }
            covered_removed.insert(producer);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::ReluProducerFusion,
                removed_nodes: vec![producer],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(mul, add, add_slot, outer_slot, reuse)) in &self.fused_mul_add_muls {
            let root_node = &self.graph.nodes()[root];
            let middle = &self.graph.nodes()[add];
            let inner = &self.graph.nodes()[mul];
            let valid_slots = add_slot < 2 && outer_slot < 2 && reuse < 3;
            let external = if valid_slots {
                [
                    inner.inputs[0],
                    inner.inputs[1],
                    middle.inputs[1 - add_slot],
                ]
            } else {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add-mul slots at node {root}"),
                });
            };
            let compatible = matches!(root_node.operation, Operation::Mul)
                && matches!(middle.operation, Operation::Add)
                && matches!(inner.operation, Operation::Mul)
                && root_node.inputs[outer_slot].index() == add
                && middle.inputs[add_slot].index() == mul
                && root_node.inputs[1 - outer_slot] == external[reuse]
                && root_node.shape == middle.shape
                && middle.shape == inner.shape
                && external
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == root_node.shape)
                && uses[mul] == 1
                && uses[add] == 1
                && !selected.contains(&mul)
                && !selected.contains(&add)
                && !covered_removed.contains(&mul)
                && !covered_removed.contains(&add)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root)
                && !self.fused_mul_adds.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add-mul fusion capability at node {root}"),
                });
            }
            covered_removed.insert(mul);
            covered_removed.insert(add);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulAddMulFusion,
                removed_nodes: vec![mul, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(mul, slot)) in &self.fused_mul_adds {
            let root_node = &self.graph.nodes()[root];
            let producer = &self.graph.nodes()[mul];
            let exclusive = uses[mul] == 1;
            let preserves = !selected.contains(&mul);
            let compatible = slot < 2
                && matches!(root_node.operation, Operation::Add)
                && matches!(producer.operation, Operation::Mul)
                && root_node.inputs[slot].index() == mul
                && root_node.shape == producer.shape
                && producer
                    .inputs
                    .iter()
                    .all(|&id| self.graph.nodes()[id.index()].shape == root_node.shape)
                && self.graph.nodes()[root_node.inputs[1 - slot].index()].shape == root_node.shape
                && !covered_removed.contains(&mul)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root);
            if !exclusive || !preserves || !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid mul-add fusion capability at node {root}"),
                });
            }
            covered_removed.insert(mul);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulAddFusion,
                removed_nodes: vec![mul],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(left, right_first, right_second, reused, right_slot, left_slot)) in
            &self.fused_mul_diamond_adds
        {
            if reused > 1 || right_slot > 1 || left_slot > 1 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid MulDiamondAdd selectors at node {root}"),
                });
            }
            let output = &self.graph.nodes()[root];
            let lhs = &self.graph.nodes()[left];
            let rhs = &self.graph.nodes()[right_first];
            let repeated = &self.graph.nodes()[right_second];
            let compatible = matches!(output.operation, Operation::Add)
                && matches!(lhs.operation, Operation::Mul)
                && matches!(rhs.operation, Operation::Mul)
                && matches!(repeated.operation, Operation::Mul)
                && output.inputs[left_slot].index() == left
                && output.inputs[1 - left_slot].index() == right_second
                && repeated.inputs[right_slot].index() == right_first
                && repeated.inputs[1 - right_slot] == rhs.inputs[reused]
                && [left, right_first, right_second]
                    .iter()
                    .all(|&id| uses[id] == 1)
                && [left, right_first, right_second]
                    .iter()
                    .all(|id| !selected.contains(id))
                && [left, right_first, right_second]
                    .iter()
                    .all(|id| !covered_removed.contains(id))
                && lhs.shape == output.shape
                && rhs.shape == output.shape
                && repeated.shape == output.shape
                && lhs
                    .inputs
                    .iter()
                    .chain(&rhs.inputs)
                    .all(|&id| self.graph.nodes()[id.index()].shape == output.shape)
                && !self.fused_mul_adds.contains_key(&root)
                && !self.fused_binary_pairs.contains_key(&root)
                && !self.fused_mul_add_muls.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid MulDiamondAdd fusion at node {root}"),
                });
            }
            covered_removed.extend([left, right_first, right_second]);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulDiamondAddFusion,
                removed_nodes: vec![left, right_first, right_second],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(inner, middle, add, reused, middle_slot, add_slot)) in
            &self.fused_mul_mul_add_divs
        {
            if reused > 1 || middle_slot > 1 || add_slot > 1 {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid four-operation fusion selectors at {root}"),
                });
            }
            let output = &self.graph.nodes()[root];
            let first = &self.graph.nodes()[inner];
            let second = &self.graph.nodes()[middle];
            let sum = &self.graph.nodes()[add];
            let div = matches!(&output.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::DivRne));
            let sources = [
                first.inputs[0],
                first.inputs[1],
                sum.inputs[1 - add_slot],
                output.inputs[1],
            ];
            let compatible = div
                && matches!(first.operation, Operation::Mul)
                && matches!(second.operation, Operation::Mul)
                && matches!(sum.operation, Operation::Add)
                && output.inputs[0].index() == add
                && sum.inputs[add_slot].index() == middle
                && second.inputs[middle_slot].index() == inner
                && second.inputs[1 - middle_slot] == first.inputs[reused]
                && output.shape == first.shape
                && output.shape == second.shape
                && output.shape == sum.shape
                && sources
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == output.shape)
                && [inner, middle, add].iter().all(|&id| uses[id] == 1)
                && [inner, middle, add].iter().all(|id| !selected.contains(id))
                && [inner, middle, add]
                    .iter()
                    .all(|id| !covered_removed.contains(id))
                && !self.fused_mul_adds.contains_key(&add)
                && !self.fused_binary_pairs.contains_key(&add);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid four-operation fusion at node {root}"),
                });
            }
            covered_removed.extend([inner, middle, add]);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::MulMulAddDivFusion,
                removed_nodes: vec![inner, middle, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(sqrt, add, slot)) in &self.fused_sqrt_add_divs {
            let output = &self.graph.nodes()[root];
            let middle = &self.graph.nodes()[add];
            let inner = &self.graph.nodes()[sqrt];
            let div = matches!(&output.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::DivRne));
            let square_root = matches!(&inner.operation, Operation::Pointwise(spec)
                if supported_pointwise(spec) == Some(TranslationKind::SqrtRne));
            let compatible = slot < 2
                && div
                && square_root
                && matches!(middle.operation, Operation::Add)
                && output.inputs[1].index() == add
                && middle.inputs[slot].index() == sqrt
                && output.shape == middle.shape
                && middle.shape == inner.shape
                && [inner.inputs[0], middle.inputs[1 - slot], output.inputs[0]]
                    .iter()
                    .all(|&id| self.graph.nodes()[id.index()].shape == output.shape)
                && uses[sqrt] == 1
                && uses[add] == 1
                && !selected.contains(&sqrt)
                && !selected.contains(&add)
                && !covered_removed.contains(&sqrt)
                && !covered_removed.contains(&add)
                && !self.fused_mul_adds.contains_key(&add)
                && !self.fused_mul_add_muls.contains_key(&add)
                && !self.fused_binary_pairs.contains_key(&add);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid sqrt-add-div fusion at node {root}"),
                });
            }
            covered_removed.insert(sqrt);
            covered_removed.insert(add);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::SqrtAddDivFusion,
                removed_nodes: vec![sqrt, add],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &(producer, slot)) in &self.fused_binary_pairs {
            let output = &self.graph.nodes()[root];
            let inner = &self.graph.nodes()[producer];
            let compatible = slot < 2
                && matches!(output.operation, Operation::Add | Operation::Mul)
                && matches!(inner.operation, Operation::Add | Operation::Mul)
                && !(matches!(output.operation, Operation::Add)
                    && matches!(inner.operation, Operation::Mul))
                && output.inputs[slot].index() == producer
                && output.shape == inner.shape
                && inner
                    .inputs
                    .iter()
                    .all(|id| self.graph.nodes()[id.index()].shape == output.shape)
                && self.graph.nodes()[output.inputs[1 - slot].index()].shape == output.shape
                && uses[producer] == 1
                && !selected.contains(&producer)
                && !covered_removed.contains(&producer)
                && !self.fused_mul_adds.contains_key(&root)
                && !self.fused_mul_add_muls.contains_key(&root)
                && !self.fused_mul_adds.contains_key(&producer)
                && !self.fused_mul_add_muls.contains_key(&producer)
                && !self.fused_relus.contains_key(&root)
                && !self.fused_pointwise_chains.contains_key(&root);
            if !compatible {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid binary pointwise fusion at node {root}"),
                });
            }
            covered_removed.insert(producer);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::BinaryPointwiseFusion,
                removed_nodes: vec![producer],
                exclusive_reachable_use: true,
                selected_outputs_preserved: true,
            });
        }

        for (&root, &earlier) in &self.fused_dual_outputs {
            let left = &self.graph.nodes()[earlier];
            let right = &self.graph.nodes()[root];
            let valid = earlier < root
                && self.live_nodes.binary_search(&earlier).is_ok()
                && self.live_nodes.binary_search(&root).is_ok()
                && uses[earlier] >= 1
                && uses[root] >= 1
                && self.live_nodes.iter().all(|&consumer| {
                    consumer > root
                        || self.graph.nodes()[consumer]
                            .inputs
                            .iter()
                            .all(|input| input.index() != earlier)
                })
                && matches!(left.operation, Operation::Add | Operation::Mul)
                && matches!(right.operation, Operation::Add | Operation::Mul)
                && left.shape == right.shape
                && left.dtype == DType::F32
                && right.dtype == DType::F32
                && left.inputs.iter().chain(&right.inputs).all(|id| {
                    self.graph.nodes()[id.index()].shape == right.shape
                        && self.graph.nodes()[id.index()].dtype == DType::F32
                        && !skipped.contains(&id.index())
                })
                && !covered_removed.contains(&earlier);
            if !valid {
                return Err(GpuError::Fp32PtxRejected {
                    reason: format!("invalid dual-output pointwise fusion at node {root}"),
                });
            }
            covered_removed.insert(earlier);
            optimizations.push(GpuOptimizationCapability {
                root,
                kind: GpuOptimizationKind::DualPointwiseOutputFusion,
                removed_nodes: vec![earlier],
                // The standalone kernel is removed, not the materialized
                // value. Downstream fanout is permitted after this root.
                exclusive_reachable_use: uses[earlier] == 1,
                selected_outputs_preserved: true,
            });
        }
        if covered_removed != skipped {
            return Err(GpuError::Fp32PtxRejected {
                reason: "capability plan does not explain every removed fusion kernel".into(),
            });
        }

        Ok(GpuCapabilityPlan {
            kernels,
            optimizations,
            boundary: GpuAssuranceBoundary {
                memory_address_arithmetic: GpuAssuranceStatus::GeneratedLeanObligation,
                llvm_ptx_structure: GpuAssuranceStatus::RequiredAtLowering,
                ideal_to_fp32: GpuAssuranceStatus::SeparateNumericProofRequired,
                cuda_runtime: GpuAssuranceStatus::RuntimeAcceptanceRequired,
                jit_sass_correspondence: GpuAssuranceStatus::NotEstablished,
            },
        })
    }

    /// Named Lean theorems for natural-number address bounds of the actual
    /// launched kernel contracts. Generation is not proof; certified builders
    /// must kernel-check and axiom-audit this program.
    pub fn memory_lean_program(&self) -> Result<LeanProgram, GpuError> {
        let mut source = format!(
            "{}\n\n{}\n",
            include_str!("../../../lean/VeriNeural/Basic.lean"),
            include_str!("../../../lean/VeriNeural/Memory.lean")
        );
        source.push_str("\nnamespace VeriNeural.Generated\n");
        let mut theorems = Vec::new();
        for contract in self.memory_contracts()? {
            source.push_str(&format!(
                "-- IR kernel {}: {} source tensors, {} output elements.\n",
                contract.kernel_node,
                contract.reads.len(),
                contract.writes
            ));
            let output_name = format!("memory_node_{}_output_bound", contract.kernel_node);
            source.push_str(&format!(
                "theorem {output_name} (index : Nat) (active : index < {}) : index < {} := by\n  exact VeriNeural.pointwise_address_bound active\n\n",
                contract.writes, contract.writes
            ));
            theorems.push(output_name);
            for &(other, extent) in &contract.additional_writes {
                let name = format!(
                    "memory_node_{}_additional_{}_output_bound",
                    contract.kernel_node, other
                );
                source.push_str(&format!(
                    "theorem {name} (index : Nat) (active : index < {extent}) : index < {extent} := by\n  exact VeriNeural.pointwise_address_bound active\n\n"
                ));
                theorems.push(name);
            }
            match contract.pattern {
                GpuMemoryPattern::Pointwise => {}
                GpuMemoryPattern::BiasAdd { columns } => {
                    let name = format!("memory_node_{}_bias_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) : index % {columns} < {columns} := by\n  exact VeriNeural.bias_address_bound (by decide)\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::MatMul {
                    rows,
                    inner,
                    columns,
                } => {
                    let name = format!("memory_node_{}_matmul_bounds", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index j : Nat) (active : index < {rows} * {columns}) (reduction : j < {inner}) :\n    (index / {columns}) * {inner} + j < {rows} * {inner} ∧\n    j * {columns} + index % {columns} < {inner} * {columns} := by\n  exact VeriNeural.matmul_address_bounds (by decide) active reduction\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::Transpose2D { rows, columns } => {
                    let name = format!("memory_node_{}_transpose_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    (index % {rows}) * {columns} + index / {rows} < {rows} * {columns} := by\n  exact VeriNeural.transpose2d_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ReverseRows2D { rows, columns } => {
                    let name = format!("memory_node_{}_reverse_rows_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    ({rows} - 1 - index / {columns}) * {columns} + index % {columns} < {rows} * {columns} := by\n  exact VeriNeural.reverse_rows2d_address_bound (by decide) (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ReverseColumns2D { rows, columns } => {
                    let name =
                        format!("memory_node_{}_reverse_columns_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {rows} * {columns}) :\n    (index / {columns}) * {columns} + ({columns} - 1 - index % {columns}) < {rows} * {columns} := by\n  exact VeriNeural.reverse_columns2d_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::BroadcastScalar => {
                    let name = format!(
                        "memory_node_{}_broadcast_scalar_bound",
                        contract.kernel_node
                    );
                    source.push_str(&format!("theorem {name} : 0 < 1 := by decide\n\n"));
                    theorems.push(name);
                }
                GpuMemoryPattern::SliceLastRow {
                    start,
                    source_width,
                } => {
                    let name = format!("memory_node_{}_slice_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {name} (index : Nat) (active : index < {}) : index + {start} < {source_width} := by\n  exact VeriNeural.slice_last_row_address_bound active (by decide)\n\n",
                        contract.writes
                    ));
                    theorems.push(name);
                }
                GpuMemoryPattern::ConcatAxis0 { left } => {
                    let right = contract.reads[1].1;
                    let left_name =
                        format!("memory_node_{}_concat_left_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {left_name} (index : Nat) (active : index < {left}) : index < {left} := by\n  exact VeriNeural.pointwise_address_bound active\n\n"
                    ));
                    theorems.push(left_name);
                    let right_name =
                        format!("memory_node_{}_concat_right_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {right_name} (index : Nat) (branch : {left} ≤ index) (active : index < {left} + {right}) : index - {left} < {right} := by\n  exact VeriNeural.concat_axis0_right_address_bound branch active\n\n"
                    ));
                    theorems.push(right_name);
                }
                GpuMemoryPattern::ArgMaxLast { width } => {
                    let rows = contract.writes;
                    let first_name =
                        format!("memory_node_{}_argmax_first_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {first_name} (index : Nat) (active : index < {rows}) : index * {width} < {rows} * {width} := by\n  exact VeriNeural.argmax_first_address_bound (by decide) active\n\n"
                    ));
                    theorems.push(first_name);
                    let candidate_name = format!(
                        "memory_node_{}_argmax_candidate_bound",
                        contract.kernel_node
                    );
                    source.push_str(&format!(
                        "theorem {candidate_name} (index lane : Nat) (active : index < {rows}) (candidate : lane < {width}) : index * {width} + lane < {rows} * {width} := by\n  exact VeriNeural.argmax_candidate_address_bound active candidate\n\n"
                    ));
                    theorems.push(candidate_name);
                }
                GpuMemoryPattern::TakeAxis0 { lane, choices } => {
                    let index_name =
                        format!("memory_node_{}_take_index_nonempty", contract.kernel_node);
                    source.push_str(&format!("theorem {index_name} : 0 < 1 := by decide\n\n"));
                    theorems.push(index_name);
                    let bound_name = format!("memory_node_{}_take_bound", contract.kernel_node);
                    source.push_str(&format!(
                        "theorem {bound_name} (selected index : Nat) (selectedBound : selected < {choices}) (active : index < {lane}) : selected * {lane} + index < {choices} * {lane} := by\n  exact VeriNeural.take_axis0_address_bound selectedBound active\n\n"
                    ));
                    theorems.push(bound_name);
                }
            }
        }
        source.push_str("end VeriNeural.Generated\n");
        Ok(LeanProgram::generated(source, theorems, Vec::new()))
    }

    /// Backwards-compatible source view over the named memory obligations.
    pub fn memory_lean_source(&self) -> Result<String, GpuError> {
        Ok(self.memory_lean_program()?.source().to_owned())
    }

    /// The Lean proof of the *ideal integer* fusion rule, not PTX/FP32 equivalence.
    /// None when no fusion was actually applied.
    pub fn fusion_lean_source(&self) -> Option<String> {
        (!self.fused_relus.is_empty()).then(|| {
            format!(
                "{}\n\n{}",
                include_str!("../../../lean/VeriNeural/Basic.lean"),
                include_str!("../../../lean/VeriNeural/Fusion.lean"),
            )
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuBenchmark {
    warmup_launches: usize,
    measured_launches: usize,
    elapsed: Duration,
}

impl GpuBenchmark {
    pub const fn warmup_launches(self) -> usize {
        self.warmup_launches
    }
    pub const fn measured_launches(self) -> usize {
        self.measured_launches
    }
    pub const fn elapsed(self) -> Duration {
        self.elapsed
    }
    pub fn average(self) -> Duration {
        Duration::from_secs_f64(self.elapsed.as_secs_f64() / self.measured_launches as f64)
    }
}

/// Host wall-clock breakdown of checked GPU feedback. `validation_d2h`
/// includes a CUDA Graph domain guard, one 32-bit pinned D2H status read,
/// synchronization, and exceptional host revalidation when necessary;
/// `snapshot_d2d` and `feedback_d2d` include their host enqueue overhead.
/// Values are observations, not a CUDA kernel-time or proof guarantee.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuFeedbackProfile {
    /// Host time to look up or fully validate the exact feedback plan.
    pub plan_host: Duration,
    pub validation_d2h: Duration,
    pub snapshot_d2d: Duration,
    pub feedback_d2d: Duration,
    pub stream_sync: Duration,
    /// CUDA-event elapsed time for one grouped feedback graph launch.
    /// None unless explicitly requested; this changes synchronization timing.
    pub device_copy: Option<Duration>,
    pub checked_values: usize,
    pub feedback_values: usize,
}

// Compiled IR, NodeId identities, and device bindings are immutable during a
// GpuExecution. Cache static plan metadata, never validated FP32 values.
struct PreparedFeedback {
    links: Vec<(NodeId, NodeId)>,
    runtime_domains: bool,
    copies: Vec<(u64, u64, usize)>,
    transfers: Vec<(u64, u64, usize)>,
    checked_sources: Vec<(u64, usize, bool)>,
    checked_requirements: Vec<(Domain, Shape)>,
    checked_values: usize,
    feedback_values: usize,
}

/// One session-wide fail-closed gate for GPU launches, copies and sync.
/// Cell preserves existing replay(&self) and benchmark(&self) signatures.
#[derive(Default)]
struct ExecutionHealth {
    poisoned: Cell<bool>,
    outputs_ready: Cell<bool>,
    replay_pending: Cell<bool>,
}

impl ExecutionHealth {
    fn ensure_ready(&self) -> Result<(), GpuError> {
        if self.poisoned.get() {
            Err(GpuError::ExecutionPoisoned)
        } else if self.replay_pending.get() {
            Err(GpuError::ReplayPending)
        } else {
            Ok(())
        }
    }

    fn begin_replay(&self) -> Result<(), GpuError> {
        self.ensure_ready()?;
        self.outputs_ready.set(false);
        self.replay_pending.set(true);
        Ok(())
    }

    fn finish_replay(&self) -> Result<(), GpuError> {
        if self.poisoned.get() {
            Err(GpuError::ExecutionPoisoned)
        } else if !self.replay_pending.get() {
            Err(GpuError::OutputNotReady)
        } else {
            self.replay_pending.set(false);
            self.outputs_ready.set(true);
            Ok(())
        }
    }

    fn ensure_pending(&self) -> Result<(), GpuError> {
        if self.poisoned.get() {
            Err(GpuError::ExecutionPoisoned)
        } else if self.replay_pending.get() {
            Ok(())
        } else {
            Err(GpuError::OutputNotReady)
        }
    }

    fn ensure_outputs_ready(&self) -> Result<(), GpuError> {
        self.ensure_ready()?;
        if self.outputs_ready.get() {
            Ok(())
        } else {
            Err(GpuError::OutputNotReady)
        }
    }

    fn invalidate_outputs(&self) {
        self.outputs_ready.set(false);
    }

    fn outputs_completed(&self) {
        self.outputs_ready.set(true);
    }

    /// Call only after metadata preflight, for operations involving CUDA.
    fn record_io<T>(&self, result: Result<T, GpuError>) -> Result<T, GpuError> {
        if result.is_err() {
            self.poisoned.set(true);
            self.replay_pending.set(false);
            self.invalidate_outputs();
        }
        result
    }
}

pub struct GpuExecution {
    session: CudaSession,
    program: GpuProgram,
    input_addresses: BTreeMap<usize, u64>,
    output_shape: Shape,
    output_address: u64,
    output_addresses: Vec<(NodeId, Shape, u64)>,
    // Only public outputs sharing storage with state inputs need snapshots.
    // Buffers are allocated once at prepare, before any CUDA Graph replay.
    preserved_output_addresses: BTreeMap<usize, u64>,
    // A named ModelExecution exposes only public readouts; raw GpuExecution
    // considers every selected output observable during feedback.
    feedback_observable_outputs: Option<Vec<NodeId>>,
    // State destinations whose selected-output aliases already have device
    // snapshots allocated; avoid rescanning every output on each replay.
    feedback_snapshot_destinations: BTreeSet<u64>,
    // Exact feedback links and domain policy key the immutable plan cache.
    prepared_feedback: Option<Rc<PreparedFeedback>>,
    // replay() and benchmark() keep their shared-reference API, so validity
    // must be invalidated through interior mutability on a new GPU epoch.
    preserved_output_valid: Cell<bool>,
    execution_health: ExecutionHealth,
    has_graph: bool,
}

impl GpuExecution {
    /// Replace all reachable graph inputs without rebuilding the CUDA Graph.
    /// Validate the full batch before the first transfer, so rejected input
    /// metadata cannot partially replace the previous input set.
    /// Transfers finish on the owned non-blocking stream before returning.
    pub fn update_inputs(&mut self, inputs: &[(NodeId, Tensor<f32>)]) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let given = validate_inputs(&self.program, inputs)?;
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        for id in self.program.source_inputs() {
            self.execution_health
                .record_io(self.session.copy_to_device(
                    self.input_addresses[&id.index()],
                    given[&id.index()].values(),
                ))?;
        }
        self.execution_health
            .record_io(self.session.flush_input_copies())
    }

    /// Update an arbitrary *subset* of reachable F32 source tensors.
    /// All IDs, uniqueness, shapes and domains are checked before enqueueing
    /// the first H2D copy. Pinned host staging is kept alive until a single
    /// synchronization of the owned non-blocking stream. All other inputs
    /// retain their previous device values.
    pub fn update_input_subset(
        &mut self,
        inputs: &[(NodeId, &Tensor<f32>)],
    ) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let mut seen = BTreeSet::new();
        for &(id, value) in inputs {
            if !seen.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate partial input update",
                });
            }
            if !self.program.source_inputs().contains(&id) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "source is outside the compiled output slice",
                });
            }
            let node = self.program.graph.node(id)?;
            if value.shape() != &node.shape {
                return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                    expected: node.shape.clone(),
                    actual: value.shape().clone(),
                }));
            }
            value.validate_domain(node.domain)?;
        }
        if inputs.is_empty() {
            return Ok(());
        }
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        for &(id, value) in inputs {
            self.execution_health.record_io(
                self.session
                    .copy_to_device(self.input_addresses[&id.index()], value.values()),
            )?;
        }
        self.execution_health
            .record_io(self.session.flush_input_copies())
    }

    /// Rebind one named source without uploading unchanged parameters or state.
    /// Metadata and domain are checked before touching GPU memory.
    pub fn update_input(&mut self, id: NodeId, value: &Tensor<f32>) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        let node = self.program.graph.node(id)?;
        if !self.program.source_inputs().contains(&id) {
            return Err(GpuError::InvalidInput {
                node: id.index(),
                reason: "source is outside the compiled output slice",
            });
        }
        if value.shape() != &node.shape {
            return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                expected: node.shape.clone(),
                actual: value.shape().clone(),
            }));
        }
        value.validate_domain(node.domain)?;
        let address = self.input_addresses[&id.index()];
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        self.execution_health
            .record_io(self.session.copy_to_device(address, value.values()))?;
        self.execution_health
            .record_io(self.session.flush_input_copies())
    }

    /// Establish the static state feedback memory contract before any replay.
    /// Public outputs that alias state inputs receive persistent device
    /// snapshots so step() cannot overwrite observable pure-step results.
    pub(crate) fn configure_model_feedback(
        &mut self,
        public_outputs: &[NodeId],
        links: &[(NodeId, NodeId)],
    ) -> Result<(), GpuError> {
        self.configure_feedback(public_outputs, links, false)
    }

    fn configure_feedback(
        &mut self,
        public_outputs: &[NodeId],
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<(), GpuError> {
        self.execution_health.ensure_ready()?;
        self.feedback_copy_plan(links, runtime_domains)?;
        let destinations = links
            .iter()
            .map(|&(state, _)| self.input_addresses[&state.index()])
            .collect::<BTreeSet<_>>();
        // Internal state roots are not observable to ModelExecution clients.
        let mut checked = BTreeSet::new();
        for &output in public_outputs {
            if !checked.insert(output.index()) {
                return Err(GpuError::InvalidInput {
                    node: output.index(),
                    reason: "duplicate observable feedback output",
                });
            }
            let (shape, address) = self
                .output_addresses
                .iter()
                .find(|(node, _, _)| *node == output)
                .map(|(_, shape, address)| (shape.clone(), *address))
                .ok_or(GpuError::InvalidInput {
                    node: output.index(),
                    reason: "public output is not in the compiled GPU program",
                })?;
            if destinations.contains(&address)
                && !self
                    .preserved_output_addresses
                    .contains_key(&output.index())
            {
                let snapshot = self.session.allocate(shape.element_count())?;
                self.preserved_output_addresses
                    .insert(output.index(), snapshot);
            }
        }
        self.feedback_snapshot_destinations.extend(destinations);
        self.feedback_observable_outputs = Some(public_outputs.to_vec());
        Ok(())
    }

    /// Save the pure step's observable values before writing feedback state.
    /// All copies stay on the owned non-blocking stream; no host round-trip.
    pub(crate) fn snapshot_outputs(&mut self) -> Result<(), GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        self.preserved_output_valid.set(false);
        for (&index, &snapshot) in &self.preserved_output_addresses {
            let (_, shape, source) = self
                .output_addresses
                .iter()
                .find(|(node, _, _)| node.index() == index)
                .expect("configured public output has a live buffer");
            self.execution_health
                .record_io(self.session.copy_device_to_device(
                    snapshot,
                    *source,
                    shape.element_count(),
                ))?;
        }
        if !self.preserved_output_addresses.is_empty() {
            self.execution_health
                .record_io(self.session.synchronize())?;
        }
        self.preserved_output_valid.set(true);
        Ok(())
    }

    fn read_value_at(
        &mut self,
        id: NodeId,
        shape: &Shape,
        address: u64,
    ) -> Result<TensorValue, GpuError> {
        match self.program.graph.node(id)?.dtype {
            DType::F32 => {
                let mut values = vec![0.0f32; shape.element_count()];
                self.execution_health
                    .record_io(self.session.copy_to_host(&mut values, address))?;
                Ok(TensorValue::F32(Tensor::new(shape.clone(), values)?))
            }
            DType::Index => {
                let mut raw = vec![0i32; shape.element_count()];
                self.execution_health
                    .record_io(self.session.copy_indices_to_host(&mut raw, address))?;
                let values = raw
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| GpuError::InvalidInput {
                            node: id.index(),
                            reason: "device index output is negative",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(TensorValue::Index(Tensor::new(shape.clone(), values)?))
            }
        }
    }

    /// Read one selected output while preserving its Semantic IR dtype.
    pub(crate) fn read_selected_value(&mut self, id: NodeId) -> Result<TensorValue, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let (_, shape, live) = self
            .output_addresses
            .iter()
            .find(|(node, _, _)| *node == id)
            .cloned()
            .ok_or(GpuError::InvalidInput {
                node: id.index(),
                reason: "requested output is not selected",
            })?;
        let address = if self.preserved_output_valid.get() {
            self.preserved_output_addresses
                .get(&id.index())
                .copied()
                .unwrap_or(live)
        } else {
            live
        };
        self.read_value_at(id, &shape, address)
    }

    /// Backward-compatible F32 reader. Index outputs must use the typed API.
    pub(crate) fn read_selected_output(&mut self, id: NodeId) -> Result<Tensor<f32>, GpuError> {
        match self.read_selected_value(id)? {
            TensorValue::F32(tensor) => Ok(tensor),
            TensorValue::Index(_) => Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                operation: "read_selected_output",
                expected: DType::F32,
                actual: DType::Index,
            })),
        }
    }

    /// Validate feedback shape, dtype, domain, liveness and alias order.
    fn feedback_copy_plan(
        &self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<Vec<(u64, u64, usize)>, GpuError> {
        let mut destinations = BTreeSet::new();
        let output_lookup = self
            .output_addresses
            .iter()
            .map(|(id, _, address)| (id.index(), *address))
            .collect::<BTreeMap<_, _>>();
        let mut copies = Vec::with_capacity(links.len());
        for &(state, next) in links {
            let old = self.program.graph.node(state)?;
            let new = self.program.graph.node(next)?;
            if !destinations.insert(state.index()) {
                return Err(GpuError::InvalidInput {
                    node: state.index(),
                    reason: "duplicate state feedback destination",
                });
            }
            if old.shape != new.shape {
                return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                    expected: old.shape.clone(),
                    actual: new.shape.clone(),
                }));
            }
            if new.dtype != old.dtype {
                return Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                    operation: "feedback_state",
                    expected: old.dtype,
                    actual: new.dtype,
                }));
            }
            if !runtime_domains && !new.domain.satisfies(old.domain) {
                return Err(GpuError::InvalidGraph(NeuralError::DomainNotEstablished {
                    required: old.domain,
                    actual: new.domain,
                }));
            }
            let &dest = self
                .input_addresses
                .get(&state.index())
                .ok_or(GpuError::InvalidInput {
                    node: state.index(),
                    reason: "feedback destination is not a live source",
                })?;
            let &src = output_lookup
                .get(&next.index())
                .ok_or(GpuError::InvalidInput {
                    node: next.index(),
                    reason: "feedback source is not a selected output",
                })?;
            copies.push((dest, src, old.shape.element_count()));
        }
        // Feedback is a parallel, simultaneous assignment. Reject duplicate
        // storage roots and cross-transfer read/write aliasing in O(n log n)
        // rather than comparing every pair of hundreds of optimizer tensors.
        let destinations = copies
            .iter()
            .map(|&(dest, _, _)| dest)
            .collect::<BTreeSet<_>>();
        if destinations.len() != copies.len()
            || copies
                .iter()
                .any(|&(dest, src, _)| dest != src && destinations.contains(&src))
        {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "overlapping state feedback sources require staging",
            });
        }
        Ok(copies)
    }

    fn prepared_feedback_plan(
        &mut self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
    ) -> Result<Rc<PreparedFeedback>, GpuError> {
        if let Some(cached) = &self.prepared_feedback {
            if cached.runtime_domains == runtime_domains && cached.links == links {
                return Ok(Rc::clone(cached));
            }
        }
        // A miss must run the original complete preflight before publishing
        // the cache entry or enqueuing a single GPU state transfer.
        let copies = self.feedback_copy_plan(links, runtime_domains)?;
        let mut checked_sources = Vec::new();
        let mut checked_requirements = Vec::new();
        let mut checked_values = 0usize;
        if runtime_domains {
            let output_lookup = self
                .output_addresses
                .iter()
                .map(|(id, _, address)| (id.index(), *address))
                .collect::<BTreeMap<_, _>>();
            for &(state, next) in links {
                let old = self.program.graph.node(state)?;
                let new = self.program.graph.node(next)?;
                if new.domain.satisfies(old.domain) {
                    continue;
                }
                if old.dtype != DType::F32 {
                    return Err(GpuError::InvalidInput {
                        node: state.index(),
                        reason: "runtime feedback domain check requires F32",
                    });
                }
                let elements = old.shape.element_count();
                let &address = output_lookup.get(&next.index()).ok_or(
                    GpuError::InvalidInput {
                        node: next.index(),
                        reason: "feedback source is not a selected output",
                    },
                )?;
                checked_sources.push((
                    address,
                    elements,
                    old.domain == Domain::NonNegativeFinite,
                ));
                checked_requirements.push((old.domain, old.shape.clone()));
                checked_values += elements;
            }
        }
        let transfers = copies
            .iter()
            .copied()
            .filter(|(dst, src, _)| dst != src)
            .collect::<Vec<_>>();
        let feedback_values = transfers.iter().map(|&(_, _, n)| n).sum();
        let prepared = Rc::new(PreparedFeedback {
            links: links.to_vec(),
            runtime_domains,
            copies,
            transfers,
            checked_sources,
            checked_requirements,
            checked_values,
            feedback_values,
        });
        self.prepared_feedback = Some(Rc::clone(&prepared));
        Ok(prepared)
    }

    /// Strict feedback for statically established source domains.
    /// Selected outputs aliasing state destinations are device-snapshotted.
    pub fn feedback_state(&mut self, links: &[(NodeId, NodeId)]) -> Result<(), GpuError> {
        self.feedback_state_impl(links, false, false).map(|_| ())
    }

    /// Experimental feedback for outputs with weaker *static* domain
    /// contracts. A separate CUDA Graph checks their FP32 bit patterns and
    /// returns one 32-bit failure flag before **any** state copy. On rejection,
    /// full tensors are read solely to report the exact host-domain error.
    /// No assumption that arbitrary FP32 arithmetic remains finite is
    /// smuggled into IR. The guard is runtime hardware validation, not proof.
    pub fn feedback_state_checked(&mut self, links: &[(NodeId, NodeId)]) -> Result<(), GpuError> {
        self.feedback_state_checked_profiled(links).map(|_| ())
    }

    /// Checked feedback plus stage-separated host timings. Validation is
    /// mandatory and happens before any state mutation, just as in the
    /// unprofiled API. No extra device readback is performed for profiling.
    pub fn feedback_state_checked_profiled(
        &mut self,
        links: &[(NodeId, NodeId)],
    ) -> Result<GpuFeedbackProfile, GpuError> {
        self.feedback_state_impl(links, true, false)
    }

    /// Diagnostic-only variant: measures the copy CUDA Graph with stream
    /// events. This introduces an extra barrier and must never be used to
    /// claim end-to-end speedups. Runtime domain checks remain mandatory.
    pub fn feedback_state_checked_device_profiled(
        &mut self,
        links: &[(NodeId, NodeId)],
    ) -> Result<GpuFeedbackProfile, GpuError> {
        self.feedback_state_impl(links, true, true)
    }

    fn feedback_state_impl(
        &mut self,
        links: &[(NodeId, NodeId)],
        runtime_domains: bool,
        device_events: bool,
    ) -> Result<GpuFeedbackProfile, GpuError> {
        let plan_started = Instant::now();
        self.execution_health.ensure_outputs_ready()?;
        let plan = self.prepared_feedback_plan(links, runtime_domains)?;
        let mut profile = GpuFeedbackProfile {
            plan_host: plan_started.elapsed(),
            checked_values: plan.checked_values,
            feedback_values: plan.feedback_values,
            ..GpuFeedbackProfile::default()
        };
        if plan.copies.is_empty() {
            return Ok(profile);
        }
        if runtime_domains && !plan.checked_sources.is_empty() {
            let validation_started = Instant::now();
            // Descriptors may be cached, but current FP32 output bytes must
            // be checked before every state mutation.
            let domains_valid = self
                .execution_health
                .record_io(self.session.validate_output_domains(&plan.checked_sources))?;
            if !domains_valid {
                // Reject before writing state; exact host oracle is reserved
                // for the exceptional path and retains source order.
                let buffers = plan
                    .checked_sources
                    .iter()
                    .map(|&(address, size, _)| (address, size))
                    .collect::<Vec<_>>();
                let copied = self
                    .execution_health
                    .record_io(self.session.copy_f32_batch_to_host(&buffers))?;
                for ((required, shape), values) in
                    plan.checked_requirements.iter().zip(copied)
                {
                    Tensor::new(shape.clone(), values)?.validate_domain(*required)?;
                }
                return Err(GpuError::Driver(
                    "GPU domain guard rejected outputs not rejected by the host oracle".into(),
                ));
            }
            profile.validation_d2h = validation_started.elapsed();
        }
        if plan
            .copies
            .iter()
            .any(|&(dst, _, _)| !self.feedback_snapshot_destinations.contains(&dst))
        {
            let observable = self
                .feedback_observable_outputs
                .clone()
                .unwrap_or_else(|| self.program.outputs().to_vec());
            self.configure_feedback(&observable, links, runtime_domains)?;
        }
        let snapshot_started = Instant::now();
        self.snapshot_outputs()?;
        profile.snapshot_d2d = snapshot_started.elapsed();
        let feedback_started = Instant::now();
        if device_events {
            profile.device_copy = Some(self.execution_health.record_io(
                self.session.launch_feedback_copies_device_timed(&plan.transfers),
            )?);
        } else {
            self.execution_health
                .record_io(self.session.launch_feedback_copies(&plan.transfers))?;
        }
        profile.feedback_d2d = feedback_started.elapsed();
        let sync_started = Instant::now();
        self.execution_health
            .record_io(self.session.synchronize())?;
        profile.stream_sync = sync_started.elapsed();
        Ok(profile)
    }

    /// Read a current source (for state inspection) without another replay.
    pub fn read_source(&mut self, id: NodeId) -> Result<Tensor<f32>, GpuError> {
        self.execution_health.ensure_ready()?;
        let node = self.program.graph.node(id)?;
        let address =
            self.input_addresses
                .get(&id.index())
                .copied()
                .ok_or(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "source is outside the compiled output slice",
                })?;
        let mut values = vec![0.0f32; node.shape.element_count()];
        self.execution_health
            .record_io(self.session.copy_to_host(&mut values, address))?;
        Ok(Tensor::new(node.shape.clone(), values)?)
    }

    /// Queue a CUDA Graph replay without synchronizing the stream.
    ///
    /// Other independent executions can enqueue work on their own
    /// non-blocking streams before any synchronization. The outputs remain
    /// inaccessible until `complete_replay` returns successfully. Every
    /// data transfer, feedback and second replay fail closed while pending.
    pub fn enqueue_replay(&self) -> Result<(), GpuError> {
        self.execution_health.begin_replay()?;
        self.preserved_output_valid.set(false);
        if self.has_graph {
            self.execution_health
                .record_io(self.session.launch_graph())?;
        }
        Ok(())
    }

    /// Synchronize this execution's stream and publish the replay outputs.
    pub fn complete_replay(&self) -> Result<(), GpuError> {
        self.execution_health.ensure_pending()?;
        self.execution_health
            .record_io(self.session.synchronize())?;
        self.execution_health.finish_replay()
    }

    /// Legacy synchronous single-execution path, preserving its contract.
    pub fn replay(&self) -> Result<(), GpuError> {
        self.enqueue_replay()?;
        self.complete_replay()
    }

    /// Launch independent CUDA Graph executions on distinct non-blocking
    /// streams before waiting for any of them. No sharing or aliasing of
    /// mutable source buffers between these owned executions is allowed by
    /// their separate CUDA allocations. This is independent inference/
    /// training-task parallelism, **not** gradient accumulation.
    ///
    /// Synchronize every already-launched stream even on an enqueue or
    /// completion error, preventing a stranded pending execution. CUDA
    /// hardware may serialize kernels; concurrency is not guaranteed.
    pub fn replay_independent_batch(executions: &mut [Self]) -> Result<(), GpuError> {
        if executions.is_empty() {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "independent GPU execution batch must not be empty",
            });
        }
        // Fail before launching any new work when another member is still
        // pending or has an unrecoverable CUDA failure.
        for execution in executions.iter() {
            execution.execution_health.ensure_ready()?;
        }
        let mut queued = 0usize;
        for execution in executions.iter() {
            if let Err(err) = execution.enqueue_replay() {
                for previous in &executions[..queued] {
                    let _ = previous.complete_replay();
                }
                return Err(err);
            }
            queued += 1;
        }
        let mut first_error = None;
        for execution in executions.iter().take(queued) {
            if let Err(err) = execution.complete_replay() {
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
        if let Some(err) = first_error {
            Err(err)
        } else {
            Ok(())
        }
    }

    /// Read the primary output while preserving its Semantic IR dtype.
    pub fn read_value(&mut self) -> Result<TensorValue, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let id = self.program.output;
        let shape = self.output_shape.clone();
        let address = if self.preserved_output_valid.get() {
            self.preserved_output_addresses
                .get(&id.index())
                .copied()
                .unwrap_or(self.output_address)
        } else {
            self.output_address
        };
        self.read_value_at(id, &shape, address)
    }

    /// Backward-compatible F32 primary-output reader.
    pub fn read_output(&mut self) -> Result<Tensor<f32>, GpuError> {
        match self.read_value()? {
            TensorValue::F32(tensor) => Ok(tensor),
            TensorValue::Index(_) => Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                operation: "read_output",
                expected: DType::F32,
                actual: DType::Index,
            })),
        }
    }

    /// Read each selected output in declaration order, preserving every dtype.
    pub fn read_values(&mut self) -> Result<Vec<(NodeId, TensorValue)>, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let outputs = self.output_addresses.clone();
        let mut result = Vec::with_capacity(outputs.len());
        for (node, shape, pointer) in outputs {
            let address = if self.preserved_output_valid.get() {
                self.preserved_output_addresses
                    .get(&node.index())
                    .copied()
                    .unwrap_or(pointer)
            } else {
                pointer
            };
            result.push((node, self.read_value_at(node, &shape, address)?));
        }
        Ok(result)
    }

    /// Read a requested subset of selected F32 outputs in caller order.
    /// All source IDs, duplicate IDs and dtypes are prevalidated before DMA.
    /// One stream barrier completes all queued D2H copies, with pinned staging
    /// retained until then. Results aliasing preserved state snapshots use the
    /// same snapshot selection as read_selected_value().
    pub fn read_outputs_selected(
        &mut self,
        selected: &[NodeId],
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        self.execution_health.ensure_outputs_ready()?;
        let mut seen = BTreeSet::new();
        let mut sources = Vec::with_capacity(selected.len());
        for &id in selected {
            if !seen.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate selected read output",
                });
            }
            let (_, shape, original) = self
                .output_addresses
                .iter()
                .find(|(node, _, _)| *node == id)
                .ok_or(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "requested output is not selected",
                })?;
            let node = self.program.graph.node(id)?;
            if node.dtype != DType::F32 {
                return Err(GpuError::InvalidGraph(NeuralError::DTypeMismatch {
                    operation: "read_outputs_selected",
                    expected: DType::F32,
                    actual: node.dtype,
                }));
            }
            let address = if self.preserved_output_valid.get() {
                self.preserved_output_addresses
                    .get(&id.index())
                    .copied()
                    .unwrap_or(*original)
            } else {
                *original
            };
            sources.push((id, shape.clone(), address));
        }
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        let buffers = sources
            .iter()
            .map(|(_, shape, address)| (*address, shape.element_count()))
            .collect::<Vec<_>>();
        let copied = self
            .execution_health
            .record_io(self.session.copy_f32_batch_to_host(&buffers))?;
        sources
            .into_iter()
            .zip(copied)
            .map(|((id, shape, _), values)| Ok((id, Tensor::new(shape, values)?)))
            .collect()
    }

    /// Backward-compatible all-F32 reader, now batched in declaration order.
    /// Mixed/Index output sets fail closed.
    pub fn read_outputs(&mut self) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        let selected = self.program.outputs().to_vec();
        self.read_outputs_selected(&selected)
    }

    pub fn benchmark(
        &self,
        warmup_launches: usize,
        measured_launches: usize,
    ) -> Result<GpuBenchmark, GpuError> {
        self.execution_health.ensure_ready()?;
        if measured_launches == 0 {
            return Err(GpuError::InvalidBenchmark { iterations: 0 });
        }
        if !self.has_graph {
            return Err(GpuError::NoKernelGraph);
        }
        self.execution_health.invalidate_outputs();
        self.preserved_output_valid.set(false);
        let milliseconds = self.execution_health.record_io(
            self.session
                .benchmark_graph(warmup_launches, measured_launches),
        )?;
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err(GpuError::Driver(
                "CUDA event benchmark returned an invalid duration".into(),
            ));
        }
        // benchmark_graph synchronizes the stop event, so the final measured
        // replay has completed and its output buffers are safe to read.
        self.execution_health.outputs_completed();
        Ok(GpuBenchmark {
            warmup_launches,
            measured_launches,
            elapsed: Duration::from_secs_f64(f64::from(milliseconds) / 1000.0),
        })
    }
}

#[derive(Clone, Debug)]
pub struct GpuEngine {
    llc: PathBuf,
    arch: String,
    launch: GpuLaunchConfig,
}

impl Default for GpuEngine {
    fn default() -> Self {
        Self::new("llc", "sm_52")
    }
}

impl GpuEngine {
    /// CUDA architecture must be an explicit LLVM -mcpu value (e.g. sm_86).
    pub fn new(llc: impl Into<PathBuf>, arch: impl Into<String>) -> Self {
        Self {
            llc: llc.into(),
            arch: arch.into(),
            launch: GpuLaunchConfig::default(),
        }
    }

    /// Select a checked launch policy. Fusion remains experimental.
    pub fn with_launch_config(mut self, launch: GpuLaunchConfig) -> Self {
        self.launch = launch;
        self
    }

    /// Backward-compatible single-output entrypoint, using the same shared
    /// compilation path as multi-output graphs.
    pub fn compile(&self, graph: &NeuralGraph, output: NodeId) -> Result<GpuProgram, GpuError> {
        self.compile_outputs(graph, &[output])
    }

    /// Compile the union of all requested outputs into one LLVM/PTX module.
    /// No duplicated producer kernels, no disconnected output slices, and
    /// no fusion that discards an explicitly selected observable output.
    pub fn compile_outputs(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
    ) -> Result<GpuProgram, GpuError> {
        if outputs.is_empty() {
            return Err(GpuError::InvalidInput {
                node: 0,
                reason: "multi-output GPU program requires at least one output",
            });
        }
        let mut seen_outputs = BTreeSet::new();
        for &id in outputs {
            graph.node(id)?;
            if !seen_outputs.insert(id.index()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "multi-output GPU program has duplicate output",
                });
            }
        }
        let output = outputs[0];
        let mut reachable = BTreeSet::new();
        let mut pending = outputs.to_vec();
        while let Some(id) = pending.pop() {
            let node = graph.node(id)?;
            if !reachable.insert(id.index()) {
                continue;
            }
            if node.shape.element_count() > i32::MAX as usize {
                return Err(GpuError::UnsupportedSize {
                    elements: node.shape.element_count(),
                });
            }
            if !matches!(
                node.operation,
                Operation::Input
                    | Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Permute { .. }
                    | Operation::ReverseAxis { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::BroadcastTo(_)
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. }
                    | Operation::Reshape(_)
                    | Operation::Pointwise(_)
            ) {
                return Err(GpuError::UnsupportedOperation {
                    node: id.index(),
                    operation: node.operation.name(),
                });
            }
            if let Operation::Pointwise(ref spec) = node.operation {
                let kind = supported_pointwise(spec);
                if kind.is_none()
                    || (matches!(kind, Some(TranslationKind::ExpApprox))
                        && !self.launch.approximate_exp())
                {
                    return Err(GpuError::UnsupportedOperation {
                        node: id.index(),
                        operation: node.operation.name(),
                    });
                }
            }
            pending.extend(node.inputs.iter().copied());
        }
        let live_nodes = reachable.into_iter().collect::<Vec<_>>();
        let source_inputs = live_nodes
            .iter()
            .filter_map(|&index| {
                let node = &graph.nodes()[index];
                matches!(node.operation, Operation::Input).then_some(node.id)
            })
            .collect::<Vec<_>>();
        // A producer can be skipped only when its sole reachable use is the
        // consuming Relu. Shared DAG nodes and Reshape aliases are left intact.
        let mut uses = vec![0usize; graph.nodes().len()];
        for &index in &live_nodes {
            for id in &graph.nodes()[index].inputs {
                uses[id.index()] += 1;
            }
        }
        // Selected roots are observable even when an in-graph consumer also
        // reads them. Never fold one into a different output's kernel.
        for &selected in outputs {
            uses[selected.index()] += 1;
        }
        // A materialized additional output may have later DAG consumers.
        // It can move to a later kernel only if no live consumer executes
        // before that kernel. This is independent of output observability.
        let mut earliest_consumer = vec![usize::MAX; graph.nodes().len()];
        for &index in &live_nodes {
            for id in &graph.nodes()[index].inputs {
                earliest_consumer[id.index()] = earliest_consumer[id.index()].min(index);
            }
        }
        let mut fused_relus = BTreeMap::new();
        let mut fused_pointwise_chains = BTreeMap::new();
        let mut fused_mul_adds = BTreeMap::new();
        let mut fused_mul_add_muls = BTreeMap::new();
        let mut fused_binary_pairs = BTreeMap::new();
        let mut fused_sqrt_add_divs = BTreeMap::new();
        let mut fused_mul_mul_add_divs = BTreeMap::new();
        let mut fused_mul_diamond_adds = BTreeMap::new();
        let mut fused_dual_outputs = BTreeMap::new();
        if self.launch.relu_fusion == ReluFusion::Experimental {
            for &index in &live_nodes {
                let node = &graph.nodes()[index];
                if !matches!(node.operation, Operation::Relu) {
                    continue;
                }
                let producer_id = node.inputs[0];
                let producer = graph.node(producer_id)?;
                if uses[producer_id.index()] != 1 {
                    continue;
                }
                match producer.operation {
                    Operation::Add | Operation::Mul => {
                        fused_relus.insert(index, producer_id.index());
                    }
                    Operation::BiasAdd => {
                        fused_relus.insert(index, producer_id.index());
                        let inner_id = producer.inputs[0];
                        let inner = graph.node(inner_id)?;
                        if uses[inner_id.index()] == 1
                            && matches!(inner.operation, Operation::Add | Operation::Mul)
                        {
                            fused_pointwise_chains
                                .insert(index, vec![inner_id.index(), producer_id.index()]);
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut skipped = fused_relus.values().copied().collect::<BTreeSet<_>>();
        for chain in fused_pointwise_chains.values() {
            skipped.extend(chain.iter().copied());
        }
        if self.launch.pointwise_fusion() {
            // A four-stage rounded pointwise chain, limited by four
            // full-shape external inputs and explicit reuse of a multiplicand:
            // Div(Add(Mul(Mul(a,b), a|b), c), d).
            // Walk roots in reverse topological order and reserve all
            // exclusive intermediates before selecting shorter rewrites.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                let div = matches!(&root.operation, Operation::Pointwise(spec)
                    if supported_pointwise(spec) == Some(TranslationKind::DivRne));
                if !div || skipped.contains(&index) {
                    continue;
                }
                let sum = graph.node(root.inputs[0])?;
                if !matches!(sum.operation, Operation::Add)
                    || uses[sum.id.index()] != 1
                    || skipped.contains(&sum.id.index())
                    || sum.shape != root.shape
                {
                    continue;
                }
                'candidate: for add_slot in 0..2 {
                    let second = graph.node(sum.inputs[add_slot])?;
                    if !matches!(second.operation, Operation::Mul)
                        || uses[second.id.index()] != 1
                        || skipped.contains(&second.id.index())
                        || second.shape != root.shape
                    {
                        continue;
                    }
                    for middle_slot in 0..2 {
                        let first = graph.node(second.inputs[middle_slot])?;
                        if !matches!(first.operation, Operation::Mul)
                            || uses[first.id.index()] != 1
                            || skipped.contains(&first.id.index())
                            || first.shape != root.shape
                        {
                            continue;
                        }
                        let externals = [
                            first.inputs[0],
                            first.inputs[1],
                            sum.inputs[1 - add_slot],
                            root.inputs[1],
                        ];
                        if externals
                            .iter()
                            .any(|id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reused) = first
                            .inputs
                            .iter()
                            .position(|&id| id == second.inputs[1 - middle_slot])
                        {
                            fused_mul_mul_add_divs.insert(
                                index,
                                (
                                    first.id.index(),
                                    second.id.index(),
                                    sum.id.index(),
                                    reused,
                                    middle_slot,
                                    add_slot,
                                ),
                            );
                            skipped.extend([first.id.index(), second.id.index(), sum.id.index()]);
                            break 'candidate;
                        }
                    }
                }
            }
            // Opt-in architecture-neutral strict-order fusion:
            // Sqrt(x) -> Add(sqrt(x), epsilon) -> Div(numerator, sum).
            // Sources must have exactly the same shape and both internal
            // nodes exactly one reachable use (including selected outputs).
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                let is_div = matches!(&root.operation, Operation::Pointwise(spec)
                    if supported_pointwise(spec) == Some(TranslationKind::DivRne));
                if !is_div || skipped.contains(&index) {
                    continue;
                }
                let middle = graph.node(root.inputs[1])?;
                if !matches!(middle.operation, Operation::Add)
                    || uses[middle.id.index()] != 1
                    || skipped.contains(&middle.id.index())
                    || root.shape != middle.shape
                {
                    continue;
                }
                for slot in 0..2 {
                    let inner = graph.node(middle.inputs[slot])?;
                    let is_sqrt = matches!(&inner.operation, Operation::Pointwise(spec)
                        if supported_pointwise(spec) == Some(TranslationKind::SqrtRne));
                    if !is_sqrt
                        || uses[inner.id.index()] != 1
                        || skipped.contains(&inner.id.index())
                        || inner.shape != root.shape
                        || [inner.inputs[0], middle.inputs[1 - slot], root.inputs[0]]
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                    {
                        continue;
                    }
                    fused_sqrt_add_divs.insert(index, (inner.id.index(), middle.id.index(), slot));
                    skipped.insert(inner.id.index());
                    skipped.insert(middle.id.index());
                    break;
                }
            }
            // Four-stage pointwise diamond: Add(Mul(a,b), Mul(Mul(c,d), c|d)).
            // The root Add may be selected, unlike its three Mul producers.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Add) || skipped.contains(&index) {
                    continue;
                }
                'diamond: for left_slot in 0..2 {
                    let left = graph.node(root.inputs[left_slot])?;
                    let second = graph.node(root.inputs[1 - left_slot])?;
                    if !matches!(left.operation, Operation::Mul)
                        || !matches!(second.operation, Operation::Mul)
                        || uses[left.id.index()] != 1
                        || uses[second.id.index()] != 1
                        || skipped.contains(&left.id.index())
                        || skipped.contains(&second.id.index())
                        || left.id == second.id
                        || left.shape != root.shape
                        || second.shape != root.shape
                    {
                        continue;
                    }
                    for right_slot in 0..2 {
                        let first = graph.node(second.inputs[right_slot])?;
                        if !matches!(first.operation, Operation::Mul)
                            || uses[first.id.index()] != 1
                            || skipped.contains(&first.id.index())
                            || first.id == left.id
                            || first.shape != root.shape
                        {
                            continue;
                        }
                        let sources = [
                            left.inputs[0],
                            left.inputs[1],
                            first.inputs[0],
                            first.inputs[1],
                        ];
                        if sources
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reused) = first
                            .inputs
                            .iter()
                            .position(|&id| id == second.inputs[1 - right_slot])
                        {
                            fused_mul_diamond_adds.insert(
                                index,
                                (
                                    left.id.index(),
                                    first.id.index(),
                                    second.id.index(),
                                    reused,
                                    right_slot,
                                    left_slot,
                                ),
                            );
                            skipped.extend([left.id.index(), first.id.index(), second.id.index()]);
                            break 'diamond;
                        }
                    }
                }
            }
            // Longer chains take priority, but never cross observed values,
            // shared nodes or existing ReLU fusion boundaries.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Mul) || skipped.contains(&index) {
                    continue;
                }
                'outer: for outer_slot in 0..2 {
                    let middle = graph.node(root.inputs[outer_slot])?;
                    if !matches!(middle.operation, Operation::Add)
                        || uses[middle.id.index()] != 1
                        || skipped.contains(&middle.id.index())
                        || middle.shape != root.shape
                    {
                        continue;
                    }
                    for add_slot in 0..2 {
                        let inner = graph.node(middle.inputs[add_slot])?;
                        if !matches!(inner.operation, Operation::Mul)
                            || uses[inner.id.index()] != 1
                            || skipped.contains(&inner.id.index())
                            || inner.shape != root.shape
                        {
                            continue;
                        }
                        let externals = [
                            inner.inputs[0],
                            inner.inputs[1],
                            middle.inputs[1 - add_slot],
                        ];
                        if externals
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                        {
                            continue;
                        }
                        if let Some(reuse) = externals
                            .iter()
                            .position(|&id| id == root.inputs[1 - outer_slot])
                        {
                            fused_mul_add_muls.insert(
                                index,
                                (
                                    inner.id.index(),
                                    middle.id.index(),
                                    add_slot,
                                    outer_slot,
                                    reuse,
                                ),
                            );
                            skipped.insert(inner.id.index());
                            skipped.insert(middle.id.index());
                            break 'outer;
                        }
                    }
                }
            }
            for &index in &live_nodes {
                let node = &graph.nodes()[index];
                if !matches!(node.operation, Operation::Add) || skipped.contains(&index) {
                    continue;
                }
                for slot in 0..2 {
                    let producer = graph.node(node.inputs[slot])?;
                    if !matches!(producer.operation, Operation::Mul)
                        || uses[producer.id.index()] != 1
                        || skipped.contains(&producer.id.index())
                        || producer.shape != node.shape
                        || graph.node(node.inputs[1 - slot])?.shape != node.shape
                        || producer
                            .inputs
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != node.shape)
                    {
                        continue;
                    }
                    fused_mul_adds.insert(index, (producer.id.index(), slot));
                    skipped.insert(producer.id.index());
                    break;
                }
            }
            // Extend strict-order pair fusion to Add→Add, Add→Mul and
            // Mul→Mul. Existing Mul→Add has priority via fused_mul_adds.
            // No shared/selected intermediate, no alias-driven shape
            // broadcast, and no overlapping fusion roots.
            for &index in live_nodes.iter().rev() {
                let root = &graph.nodes()[index];
                if !matches!(root.operation, Operation::Add | Operation::Mul)
                    || skipped.contains(&index)
                    || fused_mul_adds.contains_key(&index)
                    || fused_mul_add_muls.contains_key(&index)
                {
                    continue;
                }
                for slot in 0..2 {
                    let inner = graph.node(root.inputs[slot])?;
                    if !matches!(inner.operation, Operation::Add | Operation::Mul)
                        || (matches!(root.operation, Operation::Add)
                            && matches!(inner.operation, Operation::Mul))
                        || uses[inner.id.index()] != 1
                        || skipped.contains(&inner.id.index())
                        || fused_mul_adds.contains_key(&inner.id.index())
                        || fused_mul_add_muls.contains_key(&inner.id.index())
                        || root.shape != inner.shape
                        || graph.nodes()[root.inputs[1 - slot].index()].shape != root.shape
                        || inner
                            .inputs
                            .iter()
                            .any(|&id| graph.nodes()[id.index()].shape != root.shape)
                    {
                        continue;
                    }
                    fused_binary_pairs.insert(index, (inner.id.index(), slot));
                    skipped.insert(inner.id.index());
                    break;
                }
            }
        }
        // Horizontal fusion of two independent live Add/Mul nodes of the same
        // shape. Both results are materialized, even for non-public nodes.
        // No consumer of the earlier result may run before the second node.
        // A bounded lookback avoids quadratic compilation on large BPTT DAGs.
        // FP32 operation order and all observable values are preserved.
        if self.launch.pointwise_fusion() {
            let candidates = if self.launch.internal_pointwise_fusion() {
                live_nodes
                    .iter()
                    .copied()
                    .filter(|&index| {
                        matches!(
                            graph.nodes()[index].operation,
                            Operation::Add | Operation::Mul
                        )
                    })
                    .collect::<Vec<_>>()
            } else {
                let mut selected = outputs.iter().map(|id| id.index()).collect::<Vec<_>>();
                selected.sort_unstable();
                selected
            };
            let mut reserved = BTreeSet::new();
            for (later_pos, &later) in candidates.iter().enumerate() {
                let second = &graph.nodes()[later];
                if !matches!(second.operation, Operation::Add | Operation::Mul)
                    || uses[later] == 0
                    || skipped.contains(&later)
                    || reserved.contains(&later)
                    || second.dtype != DType::F32
                    || second.inputs.iter().any(|id| skipped.contains(&id.index()))
                    || fused_mul_adds.contains_key(&later)
                    || fused_binary_pairs.contains_key(&later)
                    || fused_mul_add_muls.contains_key(&later)
                    || fused_mul_mul_add_divs.contains_key(&later)
                    || fused_mul_diamond_adds.contains_key(&later)
                {
                    continue;
                }
                for &earlier in candidates[..later_pos].iter().rev().take(64) {
                    let first = &graph.nodes()[earlier];
                    if reserved.contains(&earlier)
                        || skipped.contains(&earlier)
                        || uses[earlier] == 0
                        || earliest_consumer[earlier] <= later
                        || !matches!(first.operation, Operation::Add | Operation::Mul)
                        || first.dtype != DType::F32
                        || first.shape != second.shape
                        || first.inputs.iter().chain(&second.inputs).any(|id| {
                            skipped.contains(&id.index())
                                || graph.nodes()[id.index()].dtype != DType::F32
                                || graph.nodes()[id.index()].shape != second.shape
                        })
                        || fused_mul_adds.contains_key(&earlier)
                        || fused_binary_pairs.contains_key(&earlier)
                        || fused_mul_add_muls.contains_key(&earlier)
                        || fused_mul_mul_add_divs.contains_key(&earlier)
                        || fused_mul_diamond_adds.contains_key(&earlier)
                    {
                        continue;
                    }
                    fused_dual_outputs.insert(later, earlier);
                    reserved.extend([earlier, later]);
                    skipped.insert(earlier);
                    break;
                }
            }
        }
        let mut llvm_ir = String::from(
            "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\n",
        );
        for &index in &live_nodes {
            let node = &graph.nodes()[index];
            if matches!(node.operation, Operation::Input)
                || graph.storage(node.id)?.is_alias()
                || skipped.contains(&index)
            {
                continue;
            }
            if let Some(&earlier) = fused_dual_outputs.get(&index) {
                llvm_ir.push_str(&dual_pointwise_kernel_ir(
                    index,
                    matches!(graph.nodes()[earlier].operation, Operation::Mul),
                    matches!(node.operation, Operation::Mul),
                ));
            } else if let Some(&(_, _, _, reused, right_slot, left_slot)) =
                fused_mul_diamond_adds.get(&index)
            {
                llvm_ir.push_str(&mul_diamond_add_kernel_ir(
                    index,
                    reused,
                    right_slot == 1,
                    left_slot == 1,
                ));
            } else if let Some(&(_, _, _, reused, middle_slot, add_slot)) =
                fused_mul_mul_add_divs.get(&index)
            {
                llvm_ir.push_str(&mul_mul_add_div_kernel_ir(
                    index,
                    reused,
                    middle_slot == 1,
                    add_slot == 1,
                ));
            } else if let Some(&(_, _, slot)) = fused_sqrt_add_divs.get(&index) {
                llvm_ir.push_str(&sqrt_add_div_kernel_ir(index, slot == 1));
            } else if let Some(&(_, _, add_slot, outer_slot, reuse)) =
                fused_mul_add_muls.get(&index)
            {
                llvm_ir.push_str(&mul_add_mul_kernel_ir(
                    index,
                    add_slot == 1,
                    outer_slot == 1,
                    reuse,
                ));
            } else if let Some(&(mul, slot)) = fused_mul_adds.get(&index) {
                debug_assert!(matches!(graph.nodes()[mul].operation, Operation::Mul));
                llvm_ir.push_str(&mul_add_kernel_ir(index, slot == 1));
            } else if let Some(&(producer, slot)) = fused_binary_pairs.get(&index) {
                llvm_ir.push_str(&binary_chain_kernel_ir(
                    index,
                    matches!(graph.nodes()[producer].operation, Operation::Add),
                    matches!(node.operation, Operation::Add),
                    slot == 1,
                ));
            } else if let Some(chain) = fused_pointwise_chains.get(&index) {
                let inner = &graph.nodes()[chain[0]];
                let bias_add = &graph.nodes()[chain[1]];
                llvm_ir.push_str(&pointwise_chain_kernel_ir(
                    index,
                    &inner.operation,
                    bias_add.shape.dims()[1],
                ));
            } else if let Some(&producer) = fused_relus.get(&index) {
                let first = &graph.nodes()[producer];
                let columns = if matches!(first.operation, Operation::BiasAdd) {
                    first.shape.dims()[1]
                } else {
                    0
                };
                llvm_ir.push_str(&kernel_ir(index, &first.operation, columns, true));
            } else if matches!(node.operation, Operation::MatMul) {
                let left = &graph.nodes()[node.inputs[0].index()];
                llvm_ir.push_str(&matmul_kernel_ir(
                    index,
                    left.shape.dims()[1],
                    node.shape.dims()[1],
                ));
            } else {
                match &node.operation {
                    Operation::Permute { axes } if axes.as_slice() == [1, 0] => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        llvm_ir.push_str(&transpose2d_kernel_ir(
                            index,
                            input.shape.dims()[0],
                            input.shape.dims()[1],
                        ));
                    }
                    Operation::ReverseAxis { axis } => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 || *axis > 1 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        llvm_ir.push_str(&reverse2d_kernel_ir(
                            index,
                            input.shape.dims()[0],
                            input.shape.dims()[1],
                            *axis,
                        ));
                    }
                    Operation::Concat { axis }
                        if *axis == 0
                            || (*axis == 1
                                && node.shape.rank() == 2
                                && node.shape.dims()[0] == 1) =>
                    {
                        let left = &graph.nodes()[node.inputs[0].index()];
                        llvm_ir
                            .push_str(&concat_axis0_kernel_ir(index, left.shape.element_count()));
                    }
                    Operation::SliceAxis { axis: 1, start, .. } => {
                        llvm_ir.push_str(&slice_last_row_kernel_ir(index, *start));
                    }
                    Operation::BroadcastTo(_) => {
                        llvm_ir.push_str(&broadcast_scalar_kernel_ir(index));
                    }
                    Operation::ArgMax {
                        axis: 1,
                        tie_break: crate::ArgMaxTieBreak::First,
                    } => {
                        let input = &graph.nodes()[node.inputs[0].index()];
                        if input.shape.rank() != 2 {
                            return Err(GpuError::UnsupportedOperation {
                                node: index,
                                operation: node.operation.name(),
                            });
                        }
                        llvm_ir.push_str(&argmax_last_kernel_ir(index, input.shape.dims()[1]));
                    }
                    Operation::Take { axis: 0 } => {
                        llvm_ir.push_str(&take_axis0_kernel_ir(index, node.shape.element_count()));
                    }
                    Operation::Permute { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. } => {
                        return Err(GpuError::UnsupportedOperation {
                            node: index,
                            operation: node.operation.name(),
                        });
                    }
                    _ => {
                        let columns = if matches!(node.operation, Operation::BiasAdd) {
                            node.shape.dims()[1]
                        } else {
                            0
                        };
                        llvm_ir.push_str(&kernel_ir(index, &node.operation, columns, false));
                    }
                }
            }
        }
        let program = GpuProgram {
            graph: graph.clone(),
            output,
            outputs: outputs.to_vec(),
            live_nodes,
            source_inputs,
            llvm_ir,
            launch: self.launch,
            fused_relus,
            fused_pointwise_chains,
            fused_mul_adds,
            fused_mul_add_muls,
            fused_binary_pairs,
            fused_sqrt_add_divs,
            fused_mul_mul_add_divs,
            fused_mul_diamond_adds,
            fused_dual_outputs,
        };
        // Fail closed unless every launched kernel has matching memory and
        // translation evidence and every applied optimization has valid
        // structural preconditions.
        program.capability_plan()?;
        Ok(program)
    }

    /// Write the inspectable LLVM IR and lower it with LLVM's NVPTX backend.
    /// LLVM must be available in the devShell; a nonzero exit is an error.
    pub fn lower(&self, program: &GpuProgram, dir: &Path) -> Result<PathBuf, GpuError> {
        fs::create_dir_all(dir)?;
        let ir = dir.join("graph.ll");
        let ptx = dir.join("graph.ptx");
        fs::write(&ir, program.llvm_ir())?;
        let output = Command::new(&self.llc)
            .arg("-march=nvptx64")
            .arg(format!("-mcpu={}", self.arch))
            .arg("-O0")
            .arg("-o")
            .arg(&ptx)
            .arg(&ir)
            .output()?;
        if !output.status.success() {
            return Err(GpuError::LlvmFailed {
                diagnostics: format!(
                    "status={:?}\nstdout={}\nstderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ),
            });
        }
        // Translation validation must succeed before the PTX reaches CUDA JIT.
        // The generated plan is derived from Semantic IR and fusion/memory
        // contracts, while the report is checked from both emitted artifacts.
        let ptx_source = fs::read_to_string(&ptx)?;
        let plan = program.translation_plan()?;
        let validated = plan
            .validate(program.llvm_ir(), &ptx_source)
            .map_err(|err| GpuError::Fp32PtxRejected {
                reason: format!("per-kernel translation validation: {err}"),
            });
        let report = match validated {
            Ok(report) => report,
            Err(err) => {
                let _ = fs::remove_file(&ptx);
                return Err(err);
            }
        };
        fs::write(dir.join("translation.plan"), plan.to_text())?;
        fs::write(dir.join("translation.report"), report.to_text())?;
        fs::write(
            dir.join("capability.plan"),
            program.capability_plan()?.to_text(),
        )?;
        Ok(ptx)
    }

    pub fn prepare(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<GpuExecution, GpuError> {
        let contracts = program.memory_contracts()?;
        let given = validate_inputs(program, inputs)?;
        let ptx_path = self.lower(program, output_dir)?;
        let ptx = fs::read(ptx_path)?;
        let mut session = CudaSession::new(&ptx)?;
        let mut buffers: BTreeMap<usize, u64> = BTreeMap::new();
        let mut input_addresses = BTreeMap::new();
        let skipped = program.skipped_nodes();

        for &index in &program.live_nodes {
            if skipped.contains(&index) && !program.fused_dual_outputs.values().any(|&n| n == index)
            {
                continue;
            }
            let node = &program.graph.nodes()[index];
            let storage = program.graph.storage(node.id)?;
            match (&node.operation, storage) {
                (Operation::Input, _) => {
                    let value = given[&index];
                    let ptr = session.allocate(value.values().len())?;
                    session.copy_to_device(ptr, value.values())?;
                    input_addresses.insert(index, ptr);
                    buffers.insert(index, ptr);
                }
                (_, NodeStorage::Alias { source, .. }) => {
                    let ptr = buffers[&source.index()];
                    buffers.insert(index, ptr);
                }
                (
                    Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Permute { .. }
                    | Operation::ReverseAxis { .. }
                    | Operation::Concat { .. }
                    | Operation::SliceAxis { .. }
                    | Operation::BroadcastTo(_)
                    | Operation::ArgMax { .. }
                    | Operation::Take { .. }
                    | Operation::Pointwise(_),
                    NodeStorage::OwnedContiguous { .. },
                ) => {
                    let ptr = session.allocate(node.shape.element_count())?;
                    buffers.insert(index, ptr);
                }
                _ => unreachable!("the compilation pass rejected unsupported nodes"),
            }
        }

        // Validate the *allocated* buffers as well as the abstract shapes
        // before capturing any kernel into the CUDA Graph. Each output has
        // its own allocation; aliases are allowed only for zero-copy Reshape
        // inputs, never for an output being written by this kernel.
        let mut written_buffers = BTreeSet::new();
        for contract in &contracts {
            let output_ptr = buffers[&contract.kernel_node];
            session.require_allocation(output_ptr, contract.writes)?;
            for &(other, elements) in &contract.additional_writes {
                let ptr = buffers[&other];
                session.require_allocation(ptr, elements)?;
                if ptr == output_ptr || !written_buffers.insert(ptr) {
                    return Err(GpuError::InvalidInput {
                        node: contract.kernel_node,
                        reason: "multi-output CUDA kernel has overlapping output allocations",
                    });
                }
            }
            if !written_buffers.insert(output_ptr) {
                return Err(GpuError::InvalidInput {
                    node: contract.kernel_node,
                    reason: "two CUDA kernel nodes write the same output allocation",
                });
            }
            for &(source, elements) in &contract.reads {
                let input_ptr = buffers[&source];
                session.require_allocation(input_ptr, elements)?;
                if input_ptr == output_ptr
                    || contract
                        .additional_writes
                        .iter()
                        .any(|&(other, _)| input_ptr == buffers[&other])
                {
                    return Err(GpuError::InvalidInput {
                        node: contract.kernel_node,
                        reason: "in-place CUDA kernel write is not supported",
                    });
                }
            }
        }

        let kernel_count = contracts.len();
        let has_graph = kernel_count > 0;
        if has_graph {
            for contract in &contracts {
                session.prepare_kernel(&format!("vn_node_{}", contract.kernel_node))?;
            }
            session.begin_graph()?;
            for contract in &contracts {
                let index = contract.kernel_node;
                let inputs = contract
                    .reads
                    .iter()
                    .map(|&(node, _)| buffers[&node])
                    .collect::<Vec<_>>();
                if let Some(&(other, _)) = contract.additional_writes.first() {
                    if contract.additional_writes.len() != 1 || inputs.len() != 4 {
                        return Err(GpuError::InvalidInput {
                            node: index,
                            reason: "invalid dual-output kernel ABI",
                        });
                    }
                    session.launch4_dual(
                        &format!("vn_node_{index}"),
                        [inputs[0], inputs[1], inputs[2], inputs[3]],
                        [buffers[&other], buffers[&index]],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else if inputs.len() == 4 {
                    session.launch4(
                        &format!("vn_node_{index}"),
                        [inputs[0], inputs[1], inputs[2], inputs[3]],
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else if inputs.len() == 3 {
                    session.launch3(
                        &format!("vn_node_{index}"),
                        [inputs[0], inputs[1], inputs[2]],
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                } else {
                    session.launch(
                        &format!("vn_node_{index}"),
                        inputs[0],
                        inputs.get(1).copied().unwrap_or(0),
                        buffers[&index],
                        contract.writes,
                        program.launch.threads_per_block,
                    )?;
                }
            }
            session.finish_graph()?;
        }

        let last = program.graph.node(program.output)?;
        let output_addresses = program
            .outputs
            .iter()
            .map(|&id| {
                let node = program.graph.node(id)?;
                let address = *buffers.get(&id.index()).ok_or(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "selected output does not own a live CUDA buffer",
                })?;
                Ok((id, node.shape.clone(), address))
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(GpuExecution {
            session,
            program: program.clone(),
            input_addresses,
            output_shape: last.shape.clone(),
            output_address: buffers[&program.output.index()],
            output_addresses,
            preserved_output_addresses: BTreeMap::new(),
            feedback_observable_outputs: None,
            feedback_snapshot_destinations: BTreeSet::new(),
            prepared_feedback: None,
            preserved_output_valid: Cell::new(false),
            execution_health: ExecutionHealth::default(),
            has_graph,
        })
    }

    /// Prepare independent executions of one compiled graph, each with its
    /// own CUDA allocations, non-blocking stream and source inventory.
    /// Validate every member's source set before preparing *any* session,
    /// so a malformed late member cannot trigger partial CUDA allocation.
    /// Each prepared session may be fed different tokens/state independently;
    /// this does not aggregate gradients or share trainable weights.
    pub fn prepare_independent(
        &self,
        program: &GpuProgram,
        batches: &[Vec<(NodeId, Tensor<f32>)>],
        output_dir: &Path,
    ) -> Result<Vec<GpuExecution>, GpuError> {
        if batches.is_empty() {
            return Err(GpuError::InvalidInput {
                node: program.output.index(),
                reason: "independent GPU batch must not be empty",
            });
        }
        for inputs in batches {
            validate_inputs(program, inputs)?;
        }
        batches
            .iter()
            .enumerate()
            .map(|(i, inputs)| {
                self.prepare(
                    program,
                    inputs,
                    &output_dir.join(format!("instance_{i:04}")),
                )
            })
            .collect()
    }

    /// Evaluate all outputs with one shared CUDA Graph capture and replay,
    /// preserving each output's Semantic IR dtype.
    pub fn execute_values(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Vec<(NodeId, TensorValue)>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_values()
    }

    /// Backward-compatible all-F32 shared execution.
    pub fn execute_outputs(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_outputs()
    }

    /// Evaluate the primary output while preserving its Semantic IR dtype.
    pub fn execute_value(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<TensorValue, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_value()
    }

    /// Backward-compatible F32 primary execution.
    pub fn execute(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Tensor<f32>, GpuError> {
        let mut execution = self.prepare(program, inputs, output_dir)?;
        execution.replay()?;
        execution.read_output()
    }
}

fn validate_inputs<'a>(
    program: &GpuProgram,
    inputs: &'a [(NodeId, Tensor<f32>)],
) -> Result<BTreeMap<usize, &'a Tensor<f32>>, GpuError> {
    let mut given = BTreeMap::new();
    for (id, value) in inputs {
        let node = program.graph.node(*id)?;
        if !matches!(node.operation, Operation::Input) {
            return Err(GpuError::InvalidInput {
                node: id.index(),
                reason: "handle is not a graph input",
            });
        }
        if !program.source_inputs.contains(id) {
            return Err(GpuError::InvalidInput {
                node: id.index(),
                reason: "input is outside the compiled output slice",
            });
        }
        if value.shape() != &node.shape {
            return Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch {
                expected: node.shape.clone(),
                actual: value.shape().clone(),
            }));
        }
        value.validate_domain(node.domain)?;
        if given.insert(id.index(), value).is_some() {
            return Err(GpuError::InvalidInput {
                node: id.index(),
                reason: "duplicate input",
            });
        }
    }
    for id in &program.source_inputs {
        if !given.contains_key(&id.index()) {
            return Err(GpuError::MissingInput { node: id.index() });
        }
    }
    Ok(given)
}

fn kernel_ir(index: usize, op: &Operation, columns: usize, fuse_relu: bool) -> String {
    // This is a single output per CUDA thread, without shared memory or
    // synchronization. The optional rewrite consumes two source buffers.
    let computation = match op {
        Operation::Add => String::from(
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Mul => String::from(
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Pointwise(spec) => match supported_pointwise(spec) {
            Some(TranslationKind::Sub) => String::from(
                "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
                   %y = load float, ptr addrspace(1) %bp, align 4\n\
                   %value = call float asm sideeffect \"sub.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
            ),
            Some(TranslationKind::DivRne) => String::from(
                "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
                   %y = load float, ptr addrspace(1) %bp, align 4\n\
                   %value = call float asm sideeffect \"div.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
            ),
            Some(TranslationKind::SqrtRne) => String::from(
                "  %value = call float asm sideeffect \"sqrt.rn.f32 $0, $1;\", \"=f,f\"(float %x)\n",
            ),
            Some(TranslationKind::ExpApprox) => format!(
                "  %scaled = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float {LOG2_E_F32_LLVM})\n\
                   %value = call float asm sideeffect \"ex2.approx.f32 $0, $1;\", \"=f,f\"(float %scaled)\n",
            ),
            _ => unreachable!("pointwise signatures were validated before codegen"),
        },
        Operation::BiasAdd => format!(
            "  %col = urem i32 %idx, {columns}\n\
               %bp = getelementptr float, ptr addrspace(1) %b, i32 %col\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n",
        ),
        Operation::Relu => String::from(
            "  %nan = fcmp uno float %x, %x\n\
               %positive = fcmp ogt float %x, 0.000000e+00\n\
               %keep = or i1 %nan, %positive\n\
               %value = select i1 %keep, float %x, float 0.000000e+00\n",
        ),
        _ => unreachable!("only supported pointwise operations require this kernel"),
    };
    let tail = if fuse_relu {
        "  %nan_fused = fcmp uno float %value, %value\n\
           %positive_fused = fcmp ogt float %value, 0.000000e+00\n\
           %keep_fused = or i1 %nan_fused, %positive_fused\n\
           %fused_result = select i1 %keep_fused, float %value, float 0.000000e+00\n"
    } else {
        ""
    };
    let result = if fuse_relu { "%fused_result" } else { "%value" };
    pointwise_kernel_ir(index, &format!("{computation}{tail}"), result)
}

/// Opt-in strict-order fusion of a pointwise Mul followed by Add.
/// Two explicit PTX RN instructions prevent LLVM from contracting the pair
/// into FMA. %a/%b are the Mul operands; %bias is the other Add operand.
/// The original Add operand order is preserved, including signed zero/NaN
/// cases where commutation must not be assumed.
/// 3-op, 3-source fusion: Mul(a,b) -> Add(product,c) -> Mul(sum,one-of-a/b/c).
/// Strictly preserve BOTH binary operand orders and two explicit mul.rn
/// instructions; no reassociation or implicit FMA.
/// Strictly rounded, architecture-neutral 3-op pointwise chain:
/// sqrt.rn(x) → add.rn(sqrt(x), epsilon) → div.rn(numerator, sum).
/// Never contract, reorder the Add operands, or skip intermediate rounding.
/// Four sequential, individually rounded FP32 operations. Operand reuse
/// is explicit and cannot be replaced with reassociation or FMA.
/// Four separate RN operations on a full-shape, four-source pointwise
/// diamond with explicit repeated input. No reassociation or FMA.
fn mul_diamond_add_kernel_ir(
    index: usize,
    reused: usize,
    right_mul_on_right: bool,
    left_mul_on_right: bool,
) -> String {
    let factor = if reused == 0 { "%z" } else { "%w" };
    let repeat_args = if right_mul_on_right {
        format!("(float {factor}, float %right)")
    } else {
        format!("(float %right, float {factor})")
    };
    let add_args = if left_mul_on_right {
        "(float %repeated, float %left)"
    } else {
        "(float %left, float %repeated)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %left = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %right = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %z, float %w)
  %repeated = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{repeat_args}
  %value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_args}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn mul_mul_add_div_kernel_ir(
    index: usize,
    reused: usize,
    multiply_on_right: bool,
    add_on_right: bool,
) -> String {
    let factor = if reused == 0 { "%x" } else { "%y" };
    let product_args = if multiply_on_right {
        format!("(float {factor}, float %first)")
    } else {
        format!("(float %first, float {factor})")
    };
    let add_args = if add_on_right {
        "(float %z, float %second)"
    } else {
        "(float %second, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %first = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %second = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{product_args}
  %third = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_args}
  %value = call float asm sideeffect "div.rn.f32 $0, $1, $2;", "=f,f,f"(float %third, float %w)
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn sqrt_add_div_kernel_ir(index: usize, sqrt_on_right: bool) -> String {
    let add_operands = if sqrt_on_right {
        "(float %y, float %root)"
    } else {
        "(float %root, float %y)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %root = call float asm sideeffect "sqrt.rn.f32 $0, $1;", "=f,f"(float %x)
  %shifted = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_operands}
  %value = call float asm sideeffect "div.rn.f32 $0, $1, $2;", "=f,f,f"(float %z, float %shifted)
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn mul_add_mul_kernel_ir(index: usize, add_right: bool, outer_right: bool, reuse: usize) -> String {
    let add_operands = if add_right {
        "(float %z, float %product)"
    } else {
        "(float %product, float %z)"
    };
    let factor = ["%x", "%y", "%z"][reuse];
    let outer_operands = if outer_right {
        format!("(float {factor}, float %middle)")
    } else {
        format!("(float %middle, float {factor})")
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %product = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %middle = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{add_operands}
  %value = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"{outer_operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

/// Three-input pointwise chain with two explicit independent RN
/// instructions, preserving both operand orders and intermediate rounding.
/// One physical CUDA Graph kernel for two independent public F32 results.
/// The four source pointers preserve the exact source operand order for both
/// operations. No FMA, fast-math, reassociation or output aliasing is allowed.
fn dual_pointwise_kernel_ir(index: usize, first_mul: bool, second_mul: bool) -> String {
    let first = if first_mul {
        "mul.rn.f32"
    } else {
        "add.rn.f32"
    };
    let second = if second_mul {
        "mul.rn.f32"
    } else {
        "add.rn.f32"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, ptr addrspace(1) %extra, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i32 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %cp, align 4
  %w = load float, ptr addrspace(1) %dp, align 4
  %first = call float asm sideeffect "{first} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %second = call float asm sideeffect "{second} $0, $1, $2;", "=f,f,f"(float %z, float %w)
  %extra_ptr = getelementptr float, ptr addrspace(1) %extra, i32 %idx
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %first, ptr addrspace(1) %extra_ptr, align 4
  store float %second, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn binary_chain_kernel_ir(
    index: usize,
    first_add: bool,
    second_add: bool,
    producer_on_right: bool,
) -> String {
    let first = if first_add {
        "add.rn.f32"
    } else {
        "mul.rn.f32"
    };
    let second = if second_add {
        "add.rn.f32"
    } else {
        "mul.rn.f32"
    };
    let operands = if producer_on_right {
        "(float %z, float %middle)"
    } else {
        "(float %middle, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %middle = call float asm sideeffect "{first} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %value = call float asm sideeffect "{second} $0, $1, $2;", "=f,f,f"{operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn mul_add_kernel_ir(index: usize, multiply_on_right: bool) -> String {
    let operands = if multiply_on_right {
        "(float %z, float %product)"
    } else {
        "(float %product, float %z)"
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %z = load float, ptr addrspace(1) %biasp, align 4
  %product = call float asm sideeffect "mul.rn.f32 $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %value = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"{operands}
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn pointwise_chain_kernel_ir(index: usize, inner_op: &Operation, columns: usize) -> String {
    let inner = match inner_op {
        Operation::Add => "add.rn.f32",
        Operation::Mul => "mul.rn.f32",
        _ => unreachable!("only Add/Mul start the supported pointwise chain"),
    };
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %bias, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %y = load float, ptr addrspace(1) %bp, align 4
  %inner = call float asm sideeffect "{inner} $0, $1, $2;", "=f,f,f"(float %x, float %y)
  %col = urem i32 %idx, {columns}
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %col
  %bias_value = load float, ptr addrspace(1) %biasp, align 4
  %biased = call float asm sideeffect "add.rn.f32 $0, $1, $2;", "=f,f,f"(float %inner, float %bias_value)
  %nan = fcmp uno float %biased, %biased
  %positive = fcmp ogt float %biased, 0.000000e+00
  %keep = or i1 %nan, %positive
  %result = select i1 %keep, float %biased, float 0.000000e+00
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %result, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn pointwise_kernel_ir(index: usize, computation: &str, result: &str) -> String {
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
         {computation}\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float {result}, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

// Stage 6: simple one-output-per-thread row-major MatMul.
// Deliberately no fusion, tiling, TF32 or fast-math. LLVM/CUDA remain TCB.
fn matmul_kernel_ir(index: usize, inner: usize, columns: usize) -> String {
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
           %row = udiv i32 %idx, {columns}\n\
           %col = urem i32 %idx, {columns}\n\
           br label %loop\n\
         loop:\n\
           %j = phi i32 [ 0, %compute ], [ %next, %body ]\n\
           %sum = phi float [ 0.000000e+00, %compute ], [ %updated, %body ]\n\
           %more = icmp ult i32 %j, {inner}\n\
           br i1 %more, label %body, label %store\n\
         body:\n\
           %lhsBase = mul i32 %row, {inner}\n\
           %lhsIdx = add i32 %lhsBase, %j\n\
           %rhsBase = mul i32 %j, {columns}\n\
           %rhsIdx = add i32 %rhsBase, %col\n\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %lhsIdx\n\
           %bp = getelementptr float, ptr addrspace(1) %b, i32 %rhsIdx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
           %y = load float, ptr addrspace(1) %bp, align 4\n\
           %product = call float asm sideeffect \"mul.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %x, float %y)\n\
           %updated = call float asm sideeffect \"add.rn.f32 $0, $1, $2;\", \"=f,f,f\"(float %sum, float %product)\n\
           %next = add i32 %j, 1\n\
           br label %loop\n\
         store:\n\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float %sum, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

fn transpose2d_kernel_ir(index: usize, rows: usize, columns: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %in_row = urem i32 %idx, {rows}
  %in_col = udiv i32 %idx, {rows}
  %row_base = mul i32 %in_row, {columns}
  %source_idx = add i32 %row_base, %in_col
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn reverse2d_kernel_ir(index: usize, rows: usize, columns: usize, axis: usize) -> String {
    let source = if axis == 0 {
        format!(
            "  %row = udiv i32 %idx, {columns}\n\
             %col = urem i32 %idx, {columns}\n\
             %rows_minus_one = add i32 0, {}\n\
             %reversed_row = sub i32 %rows_minus_one, %row\n\
             %row_base = mul i32 %reversed_row, {columns}\n\
             %source_idx = add i32 %row_base, %col\n",
            rows - 1
        )
    } else {
        format!(
            "  %row = udiv i32 %idx, {columns}\n\
             %col = urem i32 %idx, {columns}\n\
             %columns_minus_one = add i32 0, {}\n\
             %reversed_col = sub i32 %columns_minus_one, %col\n\
             %row_base = mul i32 %row, {columns}\n\
             %source_idx = add i32 %row_base, %reversed_col\n",
            columns - 1
        )
    };
    format!(
        "define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{\n\
         entry:\n\
           %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
           %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
           %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\
           %base = mul i32 %block, %width\n\
           %idx = add i32 %base, %tid\n\
           %valid = icmp ult i32 %idx, %length\n\
           br i1 %valid, label %compute, label %done\n\
         compute:\n\
         {source}\
           %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx\n\
           %x = load float, ptr addrspace(1) %ap, align 4\n\
           %op = getelementptr float, ptr addrspace(1) %out, i32 %idx\n\
           store float %x, ptr addrspace(1) %op, align 4\n\
           br label %done\n\
         done:\n\
           ret void\n\
         }}\n\n"
    )
}

// Contiguous structural kernels. No neural-architecture operation is hidden here.
fn slice_last_row_kernel_ir(index: usize, start: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %source_idx = add i32 %idx, {start}
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn broadcast_scalar_kernel_ir(index: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 0
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn concat_axis0_kernel_ir(index: usize, left: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %from_left = icmp ult i32 %idx, {left}
  br i1 %from_left, label %left, label %right
left:
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %idx
  %left_value = load float, ptr addrspace(1) %ap, align 4
  br label %store
right:
  %right_idx = sub i32 %idx, {left}
  %bp = getelementptr float, ptr addrspace(1) %b, i32 %right_idx
  %right_value = load float, ptr addrspace(1) %bp, align 4
  br label %store
store:
  %value = phi float [ %left_value, %left ], [ %right_value, %right ]
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %value, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn argmax_last_kernel_ir(index: usize, width: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %threads = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %grid_base = mul i32 %block, %threads
  %idx = add i32 %grid_base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %row_base = mul i32 %idx, {width}
  %first_ptr = getelementptr float, ptr addrspace(1) %a, i32 %row_base
  %first = load float, ptr addrspace(1) %first_ptr, align 4
  br label %loop
loop:
  %lane = phi i32 [ 1, %compute ], [ %next, %body ]
  %best_lane = phi i32 [ 0, %compute ], [ %chosen_lane, %body ]
  %best_value = phi float [ %first, %compute ], [ %chosen_value, %body ]
  %more = icmp ult i32 %lane, {width}
  br i1 %more, label %body, label %store
body:
  %candidate_idx = add i32 %row_base, %lane
  %candidate_ptr = getelementptr float, ptr addrspace(1) %a, i32 %candidate_idx
  %candidate = load float, ptr addrspace(1) %candidate_ptr, align 4
  %candidate_numeric = fcmp ord float %candidate, %candidate
  %best_nan = fcmp uno float %best_value, %best_value
  %greater = fcmp ogt float %candidate, %best_value
  %better_than_best = or i1 %best_nan, %greater
  %replace = and i1 %candidate_numeric, %better_than_best
  %chosen_lane = select i1 %replace, i32 %lane, i32 %best_lane
  %chosen_value = select i1 %replace, float %candidate, float %best_value
  %next = add i32 %lane, 1
  br label %loop
store:
  %op = getelementptr i32, ptr addrspace(1) %out, i32 %idx
  store i32 %best_lane, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

fn take_axis0_kernel_ir(index: usize, lane: usize) -> String {
    format!(
        r#"define ptx_kernel void @vn_node_{index}(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %length) {{
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %idx = add i32 %base, %tid
  %valid = icmp ult i32 %idx, %length
  br i1 %valid, label %compute, label %done
compute:
  %bp = getelementptr i32, ptr addrspace(1) %b, i32 0
  %selected = load i32, ptr addrspace(1) %bp, align 4
  %selected_base = mul i32 %selected, {lane}
  %source_idx = add i32 %selected_base, %idx
  %ap = getelementptr float, ptr addrspace(1) %a, i32 %source_idx
  %x = load float, ptr addrspace(1) %ap, align 4
  %op = getelementptr float, ptr addrspace(1) %out, i32 %idx
  store float %x, ptr addrspace(1) %op, align 4
  br label %done
done:
  ret void
}}

"#
    )
}

#[cfg(test)]
mod execution_health_tests {
    use super::*;

    #[test]
    fn outputs_not_ready_until_successful_replay_and_invalidated_by_h2d() {
        let health = ExecutionHealth::default();
        assert!(matches!(
            health.ensure_outputs_ready(),
            Err(GpuError::OutputNotReady)
        ));
        health.outputs_completed();
        assert!(health.ensure_outputs_ready().is_ok());
        // A validated input update begins a new logical input epoch.
        health.invalidate_outputs();
        assert!(health.ensure_ready().is_ok());
        assert!(matches!(
            health.ensure_outputs_ready(),
            Err(GpuError::OutputNotReady)
        ));
        // An invalid input contract is rejected before any IO or invalidation.
        health.outputs_completed();
        let _invalid = GpuError::InvalidInput {
            node: 0,
            reason: "shape mismatch",
        };
        assert!(health.ensure_outputs_ready().is_ok());
    }

    #[test]
    fn successful_synchronized_benchmark_restores_output_readiness() {
        let health = ExecutionHealth::default();
        health.outputs_completed();
        assert!(health.ensure_outputs_ready().is_ok());

        // GpuExecution::benchmark starts a new replay epoch.
        health.invalidate_outputs();
        assert!(matches!(
            health.ensure_outputs_ready(),
            Err(GpuError::OutputNotReady)
        ));

        // A successful benchmark_graph return means its stop event completed.
        health.outputs_completed();
        assert!(health.ensure_outputs_ready().is_ok());
    }

    #[test]
    fn failed_h2d_launch_and_readback_all_fail_closed() {
        for function in [
            "cuMemcpyHtoDAsync_v2",
            "cuGraphLaunch",
            "cuMemcpyDtoHAsync_v2",
        ] {
            let health = ExecutionHealth::default();
            health.outputs_completed();
            let error: Result<(), GpuError> = health.record_io(Err(GpuError::Cuda {
                function,
                code: 700,
            }));
            assert!(matches!(error, Err(GpuError::Cuda { .. })));
            assert!(matches!(
                health.ensure_ready(),
                Err(GpuError::ExecutionPoisoned)
            ));
            assert!(matches!(
                health.ensure_outputs_ready(),
                Err(GpuError::ExecutionPoisoned)
            ));
            health.outputs_completed();
            assert!(matches!(
                health.ensure_ready(),
                Err(GpuError::ExecutionPoisoned)
            ));
        }
    }

    #[test]
    fn failed_feedback_transfer_poison_is_permanent_until_new_execution() {
        let health = ExecutionHealth::default();
        assert!(health.ensure_ready().is_ok());
        assert!(health.record_io(Ok(())).is_ok());
        assert!(health.ensure_ready().is_ok());
        // Simulates a CUDA enqueue failure after another D2D copy succeeded.
        let failure: Result<(), GpuError> = health.record_io(Err(GpuError::Cuda {
            function: "cuMemcpyDtoDAsync_v2",
            code: 719,
        }));
        assert!(matches!(failure, Err(GpuError::Cuda { .. })));
        assert!(matches!(
            health.ensure_ready(),
            Err(GpuError::ExecutionPoisoned)
        ));
        assert!(health.record_io(Ok(())).is_ok());
        assert!(matches!(
            health.ensure_ready(),
            Err(GpuError::ExecutionPoisoned)
        ));
        assert!(ExecutionHealth::default().ensure_ready().is_ok());
    }

    #[test]
    fn empty_parallel_replay_is_a_preflight_error() {
        assert!(matches!(
            GpuExecution::replay_independent_batch(&mut []),
            Err(GpuError::InvalidInput {
                reason: "independent GPU execution batch must not be empty",
                ..
            })
        ));
    }

    #[test]
    fn queued_replay_is_unobservable_until_barrier_and_errors_poison() {
        let health = ExecutionHealth::default();
        health.outputs_completed();
        health.begin_replay().unwrap();
        assert!(matches!(
            health.ensure_ready(),
            Err(GpuError::ReplayPending)
        ));
        assert!(matches!(
            health.ensure_outputs_ready(),
            Err(GpuError::ReplayPending)
        ));
        assert!(matches!(
            health.begin_replay(),
            Err(GpuError::ReplayPending)
        ));
        health.ensure_pending().unwrap();
        health.finish_replay().unwrap();
        health.ensure_outputs_ready().unwrap();
        assert!(matches!(
            health.finish_replay(),
            Err(GpuError::OutputNotReady)
        ));
        health.begin_replay().unwrap();
        let failure: Result<(), GpuError> = health.record_io(Err(GpuError::Cuda {
            function: "cuStreamSynchronize",
            code: 719,
        }));
        assert!(matches!(failure, Err(GpuError::Cuda { .. })));
        assert!(matches!(
            health.ensure_pending(),
            Err(GpuError::ExecutionPoisoned)
        ));
        assert!(matches!(
            health.begin_replay(),
            Err(GpuError::ExecutionPoisoned)
        ));
    }

    #[test]
    fn cuda_sync_error_also_poisoned_but_validation_does_not() {
        let health = ExecutionHealth::default();
        // Preflight contract failures don't call record_io.
        let _validation = GpuError::InvalidInput {
            node: 0,
            reason: "bad feedback mapping",
        };
        assert!(health.ensure_ready().is_ok());
        let failure: Result<(), GpuError> = health.record_io(Err(GpuError::Cuda {
            function: "cuStreamSynchronize",
            code: 700,
        }));
        assert!(matches!(failure, Err(GpuError::Cuda { .. })));
        assert!(matches!(
            health.ensure_ready(),
            Err(GpuError::ExecutionPoisoned)
        ));
    }
}
