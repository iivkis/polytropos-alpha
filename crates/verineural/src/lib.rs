//! VeriNeural: an importable Rust API and Semantic Neural IR.
//!
//! Stage 3 adds Lean 4 proof generation and checker integration for a bounded
//! ideal-integer fragment (Add, Mul, Relu, Reshape). Shape, pointwise semantics
//! and compositional nonnegativity are verified under explicit input hypotheses.
//! No IEEE-754, Rust executable, or GPU correctness is claimed.

mod error;
mod ir;
mod proof;
mod tensor;

pub use error::NeuralError;
pub use ir::{
    Domain, FP32Policy, FmaMode, GpuSemantics, MathContract, ModuleInvocation, NeuralGraph,
    NeuralModule, Node, NodeId, Operation, OperationSemantics, PortContract, ReductionOrder,
    RoundingMode, SubnormalMode,
};
pub use proof::{CheckedLean, LeanProgram, LeanProofEngine, ProofAssumption, ProofError};
pub use tensor::{Shape, Tensor, TensorShape};

#[cfg(test)]
mod tests;
