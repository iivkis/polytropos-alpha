//! VeriNeural: an importable Rust API and Semantic Neural IR.
//!
//! Stage 3 adds Lean 4 proof generation and checker integration for a bounded
//! ideal-integer fragment (Add, Mul, Relu, Reshape). Shape, pointwise semantics
//! and compositional nonnegativity are verified under explicit input hypotheses.
//! Stage 4 adds experimental LLVM NVPTX + CUDA Driver execution on NVIDIA GPUs.
//! Stage 5 emits an integrity-checked evidence bundle with separate assurance levels.
//! Stage 6 adds MatMul, row-wise bias and composable Dense/Residual FFN blocks.
//! Stage 7 begins controllable CUDA launches and experimental local Relu fusion,
//! with ideal-Int Lean rewrite lemmas, not certified FP32/PTX equivalence.
//! Stage 8 adds exact-integer range proofs plus concrete dyadic binary32
//! RNE traces with Lean-composed error bounds for Add/Mul/MatMul/BiasAdd/Relu/
//! Reshape, including explicit separate multiply/add MatMul reductions.
//! The LLVM/PTX/GPU link is still trusted rather than formally certified.

mod certificate;
mod cuda;
mod error;
mod gpu;
mod ir;
mod modules;
mod numeric;
mod proof;
mod tensor;

pub use certificate::{CertificateEngine, CertificateError, EvidenceStatus, Stage5Report};
pub use error::NeuralError;
pub use gpu::{
    GpuBenchmark, GpuEngine, GpuError, GpuExecution, GpuLaunchConfig, GpuMemoryContract,
    GpuMemoryPattern, GpuProgram, ReluFusion,
};
pub use ir::{
    Domain, FP32Policy, FmaMode, GpuSemantics, MathContract, ModuleInvocation, NeuralGraph,
    NeuralModule, Node, NodeId, Operation, OperationSemantics, PortContract, ReductionOrder,
    RoundingMode, SubnormalMode,
};
pub use modules::{Dense, ResidualAdd, ResidualFfn};
pub use numeric::{ExactFp32Plan, NumericError, RneFp32Plan, RneFp32Trace};
pub use proof::{CheckedLean, LeanProgram, LeanProofEngine, ProofAssumption, ProofError};
pub use tensor::{Shape, Tensor, TensorShape};

#[cfg(test)]
mod tests;
