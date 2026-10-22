//! Versioned consumer-defined composite primitives lowered through typed Semantic IR.
//! This DSL deliberately cannot wrap opaque host callbacks or pretend to certify
//! unsupported backend operations.
use crate::{MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, Operation, Shape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveExpr {
    Port(usize),
    Apply(Operation, Vec<PrimitiveExpr>),
}

impl PrimitiveExpr {
    pub const fn port(index: usize) -> Self {
        Self::Port(index)
    }

    pub fn apply(operation: Operation, inputs: impl IntoIterator<Item = Self>) -> Self {
        Self::Apply(operation, inputs.into_iter().collect())
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::apply(Operation::Add, [self, rhs])
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::apply(Operation::Mul, [self, rhs])
    }

    pub fn matmul(self, rhs: Self) -> Self {
        Self::apply(Operation::MatMul, [self, rhs])
    }

    pub fn bias_add(self, bias: Self) -> Self {
        Self::apply(Operation::BiasAdd, [self, bias])
    }

    pub fn relu(self) -> Self {
        Self::apply(Operation::Relu, [self])
    }

    pub fn positive_mask(self) -> Self {
        Self::apply(Operation::PositiveMask, [self])
    }

    pub fn select_positive(self, when_positive: Self, otherwise: Self) -> Self {
        Self::apply(Operation::SelectPositive, [self, when_positive, otherwise])
    }

    pub fn sqrt(self) -> Self {
        Self::apply(Operation::Sqrt, [self])
    }

    pub fn reshape(self, shape: Shape) -> Self {
        Self::apply(Operation::Reshape(shape), [self])
    }

    fn lower(&self, graph: &mut NeuralGraph, ports: &[NodeId]) -> Result<NodeId, NeuralError> {
        match self {
            Self::Port(index) => ports.get(*index).copied().ok_or(NeuralError::InvalidNode),
            Self::Apply(operation, inputs) => {
                let ids = inputs
                    .iter()
                    .map(|e| e.lower(graph, ports))
                    .collect::<Result<Vec<_>, _>>()?;
                graph.apply(operation.clone(), &ids)
            }
        }
    }
}

/// A versioned composite declaration lowered entirely through existing typed
/// operations. It reduces consumer boilerplate, but does not extend the atomic
/// operation set.
#[derive(Clone, Debug)]
pub struct PrimitiveSpec {
    pub id: &'static str,
    pub version: &'static str,
    pub contract: MathContract,
    pub expression: PrimitiveExpr,
}

impl PrimitiveSpec {
    pub fn new(
        id: &'static str,
        version: &'static str,
        contract: MathContract,
        expression: PrimitiveExpr,
    ) -> Result<Self, NeuralError> {
        if id.is_empty() || version.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: id,
                detail: "primitive id and version must be nonempty",
            });
        }
        Ok(Self {
            id,
            version,
            contract,
            expression,
        })
    }

    pub fn apply(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.apply_module(self, inputs)
    }
}

impl NeuralModule for PrimitiveSpec {
    fn name(&self) -> &'static str {
        self.id
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn semantic_version(&self) -> &'static str {
        self.version
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        self.expression.lower(graph, inputs)
    }
}
