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

/// A checked, immutable GPU compilation plan. No claims about machine-code proof.
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
    pub const fn warmup_launches(self) -> usize { self.warmup_launches }
    pub const fn measured_launches(self) -> usize { self.measured_launches }
    pub const fn elapsed(self) -> Duration { self.elapsed }
    pub fn average(self) -> Duration {
        Duration::from_secs_f64(self.elapsed.as_secs_f64() / self.measured_launches as f64)
    }
}

pub struct GpuExecution {
    session: CudaSession,
    output_shape: Shape,
    output_address: u64,
    has_graph: bool,
}

impl GpuExecution {
    pub fn replay(&self) -> Result<(), GpuError> {
        if self.has_graph {
            self.session.launch_graph()?;
        }
        self.session.synchronize()
    }

    pub fn read_output(&mut self) -> Result<Tensor<f32>, GpuError> {
        let mut values = vec![0.0f32; self.output_shape.element_count()];
        self.session.copy_to_host(&mut values, self.output_address)?;
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
        let milliseconds = self.session.benchmark_graph(warmup_launches, measured_launches)?;
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
        Ok(GpuProgram {
            graph: graph.clone(),
            output,
            live_nodes,
            source_inputs,
            llvm_ir,
            launch: self.launch,
            fused_relus,
            fused_pointwise_chains,
        })
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
        Ok(ptx)
    }

    pub fn prepare(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<GpuExecution, GpuError> {
        let given = validate_inputs(program, inputs)?;
        let ptx_path = self.lower(program, output_dir)?;
        let ptx = fs::read(ptx_path)?;
        let mut session = CudaSession::new(&ptx)?;
        let mut buffers: BTreeMap<usize, u64> = BTreeMap::new();
        let skipped = program.skipped_nodes();

        for &index in &program.live_nodes {
            if skipped.contains(&index) { continue; }
            let node = &program.graph.nodes()[index];
            match node.operation {
                Operation::Input => {
                    let value = given[&index];
                    let ptr = session.allocate(value.values().len())?;
                    session.copy_to_device(ptr, value.values())?;
                    buffers.insert(index, ptr);
                }
                Operation::Reshape(_) => {
                    let ptr = buffers[&node.inputs[0].index()];
                    buffers.insert(index, ptr);
                }
                Operation::Add | Operation::Mul | Operation::MatMul
                | Operation::BiasAdd | Operation::Relu => {
                    let ptr = session.allocate(node.shape.element_count())?;
                    buffers.insert(index, ptr);
                }
                _ => unreachable!("the compilation pass rejected unsupported nodes"),
            }
        }

        let kernel_count = program.live_nodes.iter().filter(|&&index| {
            !skipped.contains(&index)
                && !matches!(
                    program.graph.nodes()[index].operation,
                    Operation::Input | Operation::Reshape(_)
                )
        }).count();
        let has_graph = kernel_count > 0;
        if has_graph {
            for &index in &program.live_nodes {
                if skipped.contains(&index) { continue; }
                let op = &program.graph.nodes()[index].operation;
                if !matches!(op, Operation::Input | Operation::Reshape(_)) {
                    session.prepare_kernel(&format!("vn_node_{index}"))?;
                }
            }
            session.begin_graph()?;
            for &index in &program.live_nodes {
                if skipped.contains(&index) { continue; }
                let node = &program.graph.nodes()[index];
                if matches!(node.operation, Operation::Input | Operation::Reshape(_)) {
                    continue;
                }
                if let Some(chain) = program.fused_pointwise_chains.get(&index) {
                    let inner = &program.graph.nodes()[chain[0]];
                    let bias_add = &program.graph.nodes()[chain[1]];
                    session.launch3(
                        &format!("vn_node_{index}"),
                        buffers[&inner.inputs[0].index()],
                        buffers[&inner.inputs[1].index()],
                        buffers[&bias_add.inputs[1].index()],
                        buffers[&index],
                        node.shape.element_count(),
                        program.launch.threads_per_block,
                    )?;
                    continue;
                }
                let kernel_inputs = if let Some(&producer) = program.fused_relus.get(&index) {
                    &program.graph.nodes()[producer].inputs
                } else {
                    &node.inputs
                };
                let first = buffers[&kernel_inputs[0].index()];
                let second = if kernel_inputs.len() == 2 {
                    buffers[&kernel_inputs[1].index()]
                } else { 0 };
                session.launch(
                    &format!("vn_node_{index}"),
                    first, second, buffers[&index],
                    node.shape.element_count(),
                    program.launch.threads_per_block,
                )?;
            }
            session.finish_graph()?;
        }

        let last = program.graph.node(program.output)?;
        Ok(GpuExecution {
            session,
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
               %value = fadd float %x, %y\n",
        ),
        Operation::Mul => String::from(
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = fmul float %x, %y\n",
        ),
        Operation::BiasAdd => format!(
            "  %col = urem i32 %idx, {columns}\n\
               %bp = getelementptr float, ptr addrspace(1) %b, i32 %col\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %value = fadd float %x, %y\n",
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
        Operation::Add => "fadd",
        Operation::Mul => "fmul",
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
  %inner = {inner} float %x, %y
  %col = urem i32 %idx, {columns}
  %biasp = getelementptr float, ptr addrspace(1) %bias, i32 %col
  %bias_value = load float, ptr addrspace(1) %biasp, align 4
  %biased = fadd float %inner, %bias_value
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
           %product = fmul float %x, %y\n\
           %updated = fadd float %sum, %product\n\
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
