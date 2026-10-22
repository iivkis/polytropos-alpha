//! Stage 4: validated Semantic Neural IR to LLVM NVPTX kernels.
//!
//! This backend is executable but *not certified*: LLVM, PTX JIT, CUDA driver,
//! hardware and IEEE-754 behavior are outside the current formal proof chain.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::{NeuralError, NeuralGraph, NodeId, Operation, Shape, Tensor};

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
        })
    }

    pub const fn threads_per_block(self) -> u32 {
        self.threads_per_block
    }

    pub const fn relu_fusion(self) -> ReluFusion {
        self.relu_fusion
    }
}

impl Default for GpuLaunchConfig {
    fn default() -> Self {
        Self {
            threads_per_block: 256,
            relu_fusion: ReluFusion::Disabled,
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
    /// (source graph node, required readable FP32 elements).
    pub reads: Vec<(usize, usize)>,
    /// Output is written once per active lane, at [0, writes).
    pub writes: usize,
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
            || opcode.contains(".approx.")
            || opcode.starts_with("fma.")
            || opcode.starts_with("mad.")
        {
            return Err(rejected(format!("prohibited PTX FP32 opcode: {opcode}")));
        }
        match opcode {
            "add.rn.f32" => rounded_adds += 1,
            "mul.rn.f32" => rounded_muls += 1,
            _ if opcode.starts_with("setp.")
                || opcode.starts_with("selp.")
                || opcode.starts_with("mov.")
                || opcode.starts_with("ld.")
                || opcode.starts_with("st.") => {}
            _ => return Err(rejected(format!("unapproved PTX FP32 opcode: {opcode}"))),
        }
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

#[derive(Clone, Debug)]
pub struct GpuProgram {
    graph: NeuralGraph,
    output: NodeId,
    live_nodes: Vec<usize>,
    source_inputs: Vec<NodeId>,
    llvm_ir: String,
    launch: GpuLaunchConfig,
    // Relu index -> exclusive immediate producer index.
    fused_relus: BTreeMap<usize, usize>,
    // Relu root -> [exclusive Add/Mul producer, exclusive BiasAdd producer].
    fused_pointwise_chains: BTreeMap<usize, Vec<usize>>,
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

    pub fn removed_kernel_count(&self) -> usize {
        self.skipped_nodes().len()
    }

    fn skipped_nodes(&self) -> BTreeSet<usize> {
        let mut skipped = self.fused_relus.values().copied().collect::<BTreeSet<_>>();
        for chain in self.fused_pointwise_chains.values() {
            skipped.extend(chain.iter().copied());
        }
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
            if matches!(node.operation, Operation::Input | Operation::Reshape(_)) {
                continue;
            }
            let writes = node.shape.element_count();
            let (read_nodes, pattern): (Vec<usize>, GpuMemoryPattern) =
                if let Some(chain) = self.fused_pointwise_chains.get(&index) {
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
                    let pattern = match op_node.operation {
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
            };
            // CUDA templates form idx = block * width + lane with u32
            // wrap semantics, then test idx < length as unsigned. Bound the
            // *whole* grid so even unused trailing lanes cannot wrap.
            let width = self.launch.threads_per_block as usize;
            let launch_extent = writes.div_ceil(width).checked_mul(width);
            if !matches!(read_nodes.len(), 1..=3)
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
                pattern,
            });
        }
        Ok(result)
    }

    /// Lean proofs of the natural-number address bounds for the actual shapes.
    /// The proof does not validate the LLVM/PTX instructions or GPU pointers.
    pub fn memory_lean_source(&self) -> Result<String, GpuError> {
        let mut source = format!(
            "{}\n\n{}\n",
            include_str!("../../../lean/VeriNeural/Basic.lean"),
            include_str!("../../../lean/VeriNeural/Memory.lean")
        );
        source.push_str("\nnamespace VeriNeural\n");
        for contract in self.memory_contracts()? {
            source.push_str(&format!(
                "-- IR kernel {}: {} source tensors, {} output elements.\n",
                contract.kernel_node,
                contract.reads.len(),
                contract.writes
            ));
            source.push_str(&format!(
                "example (index : Nat) (active : index < {}) : index < {} :=\n  pointwise_address_bound active\n",
                contract.writes, contract.writes
            ));
            match contract.pattern {
                GpuMemoryPattern::Pointwise => {}
                GpuMemoryPattern::BiasAdd { columns } => {
                    source.push_str(&format!(
                        "example (index : Nat) : index % {columns} < {columns} :=\n  bias_address_bound (by decide)\n"
                    ));
                }
                GpuMemoryPattern::MatMul {
                    rows,
                    inner,
                    columns,
                } => {
                    source.push_str(&format!(
                        "example (index j : Nat) (active : index < {rows} * {columns}) (reduction : j < {inner}) :\n    (index / {columns}) * {inner} + j < {rows} * {inner} ∧\n    j * {columns} + index % {columns} < {inner} * {columns} :=\n  matmul_address_bounds (by decide) active reduction\n"
                    ));
                }
            }
        }
        source.push_str("end VeriNeural\n");
        Ok(source)
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

pub struct GpuExecution {
    session: CudaSession,
    program: GpuProgram,
    input_addresses: BTreeMap<usize, u64>,
    output_shape: Shape,
    output_address: u64,
    has_graph: bool,
}

impl GpuExecution {
    /// Replace all reachable graph inputs without rebuilding the CUDA Graph.
    /// Validate the full batch before the first transfer, so rejected input
    /// metadata cannot partially replace the previous input set.
    /// Transfers finish on the owned non-blocking stream before returning.
    pub fn update_inputs(&mut self, inputs: &[(NodeId, Tensor<f32>)]) -> Result<(), GpuError> {
        let given = validate_inputs(&self.program, inputs)?;
        for id in self.program.source_inputs() {
            self.session.copy_to_device(
                self.input_addresses[&id.index()],
                given[&id.index()].values(),
            )?;
        }
        self.session.flush_input_copies()
    }

    pub fn replay(&self) -> Result<(), GpuError> {
        if self.has_graph {
            self.session.launch_graph()?;
        }
        self.session.synchronize()
    }

    pub fn read_output(&mut self) -> Result<Tensor<f32>, GpuError> {
        let mut values = vec![0.0f32; self.output_shape.element_count()];
        self.session
            .copy_to_host(&mut values, self.output_address)?;
        Ok(Tensor::new(self.output_shape.clone(), values)?)
    }

    pub fn benchmark(
        &self,
        warmup_launches: usize,
        measured_launches: usize,
    ) -> Result<GpuBenchmark, GpuError> {
        if measured_launches == 0 {
            return Err(GpuError::InvalidBenchmark { iterations: 0 });
        }
        if !self.has_graph {
            return Err(GpuError::NoKernelGraph);
        }
        let milliseconds = self
            .session
            .benchmark_graph(warmup_launches, measured_launches)?;
        if !milliseconds.is_finite() || milliseconds < 0.0 {
            return Err(GpuError::Driver(
                "CUDA event benchmark returned an invalid duration".into(),
            ));
        }
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

    /// Generate one kernel per arithmetic node in the reachable DAG.
    /// Reshape is zero-copy, with no additional GPU operation.
    pub fn compile(&self, graph: &NeuralGraph, output: NodeId) -> Result<GpuProgram, GpuError> {
        graph.node(output)?;
        let mut reachable = BTreeSet::new();
        let mut pending = vec![output];
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
                    | Operation::Reshape(_)
            ) {
                return Err(GpuError::UnsupportedOperation {
                    node: id.index(),
                    operation: node.operation.name(),
                });
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
        let mut fused_relus = BTreeMap::new();
        let mut fused_pointwise_chains = BTreeMap::new();
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
        let mut llvm_ir = String::from(
            "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\n",
        );
        for &index in &live_nodes {
            let node = &graph.nodes()[index];
            if matches!(node.operation, Operation::Input | Operation::Reshape(_))
                || skipped.contains(&index)
            {
                continue;
            }
            if let Some(chain) = fused_pointwise_chains.get(&index) {
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
                let columns = if matches!(node.operation, Operation::BiasAdd) {
                    node.shape.dims()[1]
                } else {
                    0
                };
                llvm_ir.push_str(&kernel_ir(index, &node.operation, columns, false));
            }
        }
        let program = GpuProgram {
            graph: graph.clone(),
            output,
            live_nodes,
            source_inputs,
            llvm_ir,
            launch: self.launch,
            fused_relus,
            fused_pointwise_chains,
        };
        // Fail closed on unsupported memory addressing before an LLVM program
        // or CUDA session can be accepted.
        program.memory_contracts()?;
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
        // Every llc output is checked before CUDA can load it.
        let ptx_source = fs::read_to_string(&ptx)?;
        if let Err(err) = audit_fp32_ptx(program.llvm_ir(), &ptx_source) {
            let _ = fs::remove_file(&ptx);
            return Err(err);
        }
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
            if skipped.contains(&index) {
                continue;
            }
            let node = &program.graph.nodes()[index];
            match node.operation {
                Operation::Input => {
                    let value = given[&index];
                    let ptr = session.allocate(value.values().len())?;
                    session.copy_to_device(ptr, value.values())?;
                    input_addresses.insert(index, ptr);
                    buffers.insert(index, ptr);
                }
                Operation::Reshape(_) => {
                    let ptr = buffers[&node.inputs[0].index()];
                    buffers.insert(index, ptr);
                }
                Operation::Add
                | Operation::Mul
                | Operation::MatMul
                | Operation::BiasAdd
                | Operation::Relu => {
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
            if !written_buffers.insert(output_ptr) {
                return Err(GpuError::InvalidInput {
                    node: contract.kernel_node,
                    reason: "two CUDA kernel nodes write the same output allocation",
                });
            }
            for &(source, elements) in &contract.reads {
                let input_ptr = buffers[&source];
                session.require_allocation(input_ptr, elements)?;
                if input_ptr == output_ptr {
                    return Err(GpuError::InvalidInput {
                        node: contract.kernel_node,
                        reason: "in-place CUDA kernel write is not supported",
                    });
                }
            }
        }

        let kernel_count = program
            .live_nodes
            .iter()
            .filter(|&&index| {
                !skipped.contains(&index)
                    && !matches!(
                        program.graph.nodes()[index].operation,
                        Operation::Input | Operation::Reshape(_)
                    )
            })
            .count();
        let has_graph = kernel_count > 0;
        if has_graph {
            for &index in &program.live_nodes {
                if skipped.contains(&index) {
                    continue;
                }
                let op = &program.graph.nodes()[index].operation;
                if !matches!(op, Operation::Input | Operation::Reshape(_)) {
                    session.prepare_kernel(&format!("vn_node_{index}"))?;
                }
            }
            session.begin_graph()?;
            for contract in &contracts {
                let index = contract.kernel_node;
                let inputs = contract
                    .reads
                    .iter()
                    .map(|&(node, _)| buffers[&node])
                    .collect::<Vec<_>>();
                if inputs.len() == 3 {
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
        Ok(GpuExecution {
            session,
            program: program.clone(),
            input_addresses,
            output_shape: last.shape.clone(),
            output_address: buffers[&program.output.index()],
            has_graph,
        })
    }

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
