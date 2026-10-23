use std::fmt;

use crate::{Domain, Shape};

/// A construction, shape, domain, or module-composition error.
///
/// These are Rust-side validation failures, not Lean proof failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeuralError {
    ZeroDimension,
    ElementCountOverflow,
    DataLengthMismatch {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch {
        expected: Shape,
        actual: Shape,
    },
    RankMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InnerDimensionMismatch {
        left: usize,
        right: usize,
    },
    BiasWidthMismatch {
        columns: usize,
        bias_length: usize,
    },
    ElementCountMismatch {
        expected: usize,
        actual: usize,
    },
    WrongArity {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidNode,
    InvalidAxis { axis: usize, rank: usize },
    InputCannotBeApplied,
    DomainNotEstablished {
        required: Domain,
        actual: Domain,
    },
    ValueOutsideDomain {
        index: usize,
        required: Domain,
    },
    ContractViolation {
        module: &'static str,
        detail: &'static str,
    },
}

impl fmt::Display for NeuralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "tensor dimensions must be nonzero"),
            Self::ElementCountOverflow => write!(f, "tensor element count overflows usize"),
            Self::DataLengthMismatch { expected, actual } => {
                write!(f, "tensor needs {expected} elements, received {actual}")
            }
            Self::ShapeMismatch { expected, actual } => write!(
                f,
                "shape mismatch: expected {expected:?}, received {actual:?}"
            ),
            Self::RankMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation} expects rank {expected}, received rank {actual}"
            ),
            Self::InnerDimensionMismatch { left, right } => {
                write!(f, "matmul inner dimensions disagree: {left} vs {right}")
            }
            Self::BiasWidthMismatch {
                columns,
                bias_length,
            } => {
                write!(
                    f,
                    "bias length {bias_length} does not equal matrix width {columns}"
                )
            }
            Self::ElementCountMismatch { expected, actual } => {
                write!(f, "reshape expects {expected} elements, received {actual}")
            }
            Self::WrongArity {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation} expects {expected} inputs, received {actual}"
            ),
            Self::InvalidNode => write!(f, "node belongs to a different graph or does not exist"),
            Self::InvalidAxis { axis, rank } => write!(f, "tensor axis {axis} is outside rank {rank}"),
            Self::InputCannotBeApplied => write!(f, "create source nodes with graph.input()"),
            Self::DomainNotEstablished { required, actual } => write!(
                f,
                "required domain {required:?} is not established by {actual:?}"
            ),
            Self::ValueOutsideDomain { index, required } => {
                write!(f, "value at index {index} violates domain {required:?}")
            }
            Self::ContractViolation { module, detail } => {
                write!(f, "module {module} violates contract: {detail}")
            }
        }
    }
}

impl std::error::Error for NeuralError {}
