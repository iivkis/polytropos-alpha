//! VeriNeural: an importable Rust API and Semantic Neural IR.
//!
//! Stage 2: validated tensor shapes, pure CPU reference operations, a symbolic
//! neural graph, explicit numerical *requirements*, and composable contracts.
//! Nothing in this crate is a Lean proof or a GPU implementation.

mod error;
mod ir;
mod tensor;

pub use error::NeuralError;
pub use ir::{
    Domain, FP32Policy, FmaMode, GpuSemantics, MathContract, ModuleInvocation, NeuralGraph,
    NeuralModule, Node, NodeId, Operation, OperationSemantics, PortContract, ReductionOrder,
    RoundingMode, SubnormalMode,
};
pub use tensor::{Shape, Tensor, TensorShape};

#[cfg(test)]
mod tests;
