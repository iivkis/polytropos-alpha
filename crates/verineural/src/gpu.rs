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

// Keep public types and contracts in this facade. Implementations are split by
// responsibility; all lowering, memory checks and CUDA Graph behavior are unchanged.
mod engine;
mod execution;
mod kernels;
mod program;

use kernels::*;

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

/// LLVM NVPTX code-generation optimization level. This is deliberately
/// independent of the graph's FP32 policy: optimizing machine instructions
/// does not constitute a proof of numeric or hardware equivalence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuCodegenOptLevel {
    #[default]
    O0,
    /// Experimental: only use after translation validation and a GPU parity run.
    O2,
}

impl GpuCodegenOptLevel {
    pub const fn llc_flag(self) -> &'static str {
        match self {
            Self::O0 => "-O0",
            Self::O2 => "-O2",
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
    // Each live kernel node resolves to the first identical generated function.
    // The graph retains distinct nodes, buffers, arguments and dependencies.
    kernel_entries: BTreeMap<usize, usize>,
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

// Program lowering, fusion planning and memory/proof contracts live in gpu/program.rs.
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

/// Host-side phases of one selected F32 D2H batch. Not device-event timing:
/// `synchronize` includes completion of queued DMA; replay is separately
/// synchronized before outputs become readable.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuReadbackProfile {
    pub preflight: Duration,
    pub enqueue: Duration,
    pub synchronize: Duration,
    pub host_copy: Duration,
    pub tensor_construct: Duration,
    pub transfers: usize,
    pub bytes: usize,
}

/// Selected F32 tensors and host-side D2H timings returned by profiled readback.
/// The alias preserves the existing tuple result and its public API semantics.
pub type GpuProfiledReadback = (Vec<(NodeId, Tensor<f32>)>, GpuReadbackProfile);

pub struct GpuExecution {
    session: CudaSession,
    program: GpuProgram,
    input_addresses: BTreeMap<usize, u64>,
    output_shape: Shape,
    output_address: u64,
    output_addresses: Vec<(NodeId, Shape, u64)>,
    output_address_lookup: BTreeMap<usize, usize>,
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

// CUDA Graph replay, feedback and readback live in gpu/execution.rs.
#[derive(Clone, Debug)]
pub struct GpuEngine {
    llc: PathBuf,
    arch: String,
    launch: GpuLaunchConfig,
    codegen_opt_level: GpuCodegenOptLevel,
    allocation_slab: bool,
}

impl Default for GpuEngine {
    fn default() -> Self {
        Self::new("llc", "sm_52")
    }
}

// GPU compilation and session setup live in gpu/engine.rs.
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

// Architecture-neutral NVPTX templates live in gpu/kernels.rs.
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
