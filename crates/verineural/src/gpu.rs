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

use crate::{NeuralError, NeuralGraph, NodeId, Operation, Tensor};

use super::cuda::CudaSession;

const THREADS: usize = 256;

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
}

#[derive(Clone, Debug)]
pub struct GpuEngine {
    llc: PathBuf,
    arch: String,
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
        }
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
        let mut llvm_ir = String::from(
            "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64\"\n\
             target triple = \"nvptx64-nvidia-cuda\"\n\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
             declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()\n\n",
        );
        for &index in &live_nodes {
            let node = &graph.nodes()[index];
            if matches!(node.operation, Operation::Input | Operation::Reshape(_)) {
                continue;
            }
            if matches!(node.operation, Operation::MatMul) {
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
                llvm_ir.push_str(&kernel_ir(index, &node.operation, columns));
            }
        }
        Ok(GpuProgram {
            graph: graph.clone(),
            output,
            live_nodes,
            source_inputs,
            llvm_ir,
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

    /// Execute the plan using libcuda.so.1. Inputs must match every reachable
    /// source exactly once. All declared input domains are checked on host.
    pub fn execute(
        &self,
        program: &GpuProgram,
        inputs: &[(NodeId, Tensor<f32>)],
        output_dir: &Path,
    ) -> Result<Tensor<f32>, GpuError> {
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

        // Compile before creating a CUDA context, so compiler failures do not
        // leave device allocations behind.
        let ptx_path = self.lower(program, output_dir)?;
        let ptx = fs::read(ptx_path)?;
        let mut session = CudaSession::new(&ptx)?;
        let mut buffers: BTreeMap<usize, u64> = BTreeMap::new();

        for &index in &program.live_nodes {
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
                Operation::Add
                | Operation::Mul
                | Operation::MatMul
                | Operation::BiasAdd
                | Operation::Relu => {
                    let first = buffers[&node.inputs[0].index()];
                    let second = if node.inputs.len() == 2 {
                        buffers[&node.inputs[1].index()]
                    } else {
                        0
                    };
                    let ptr = session.allocate(node.shape.element_count())?;
                    session.launch(
                        &format!("vn_node_{index}"),
                        first,
                        second,
                        ptr,
                        node.shape.element_count(),
                        THREADS as u32,
                    )?;
                    buffers.insert(index, ptr);
                }
                _ => unreachable!("the compilation pass rejected unsupported nodes"),
            }
        }
        session.synchronize()?;
        let last = program.graph.node(program.output)?;
        let mut values = vec![0.0f32; last.shape.element_count()];
        session.copy_to_host(&mut values, buffers[&program.output.index()])?;
        Ok(Tensor::new(last.shape.clone(), values)?)
    }
}

fn kernel_ir(index: usize, op: &Operation, columns: usize) -> String {
    let computation = match op {
        Operation::Add => {
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %result = fadd float %x, %y\n"
        }
        Operation::Mul => {
            "  %bp = getelementptr float, ptr addrspace(1) %b, i32 %idx\n\
               %y = load float, ptr addrspace(1) %bp, align 4\n\
               %result = fmul float %x, %y\n"
        }
        Operation::BiasAdd => {
            return pointwise_kernel_ir(
                index,
                &format!(
                    "  %col = urem i32 %idx, {columns}\n\
                 %bp = getelementptr float, ptr addrspace(1) %b, i32 %col\n\
                 %y = load float, ptr addrspace(1) %bp, align 4\n\
                 %result = fadd float %x, %y\n"
                ),
            );
        }
        Operation::Relu => {
            "  %nan = fcmp uno float %x, %x\n\
               %positive = fcmp ogt float %x, 0.000000e+00\n\
               %keep = or i1 %nan, %positive\n\
               %result = select i1 %keep, float %x, float 0.000000e+00\n"
        }
        _ => unreachable!("only arithmetic operations require kernels"),
    };
    pointwise_kernel_ir(index, computation)
}

fn pointwise_kernel_ir(index: usize, computation: &str) -> String {
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
           store float %result, ptr addrspace(1) %op, align 4\n\
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
