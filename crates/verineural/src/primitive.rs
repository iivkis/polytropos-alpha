//! Versioned consumer-defined composite primitives lowered through typed Semantic IR.
//! This DSL deliberately cannot wrap opaque host callbacks or pretend to certify
//! unsupported backend operations.
use crate::{MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, Operation};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveExpr {
    Port(usize),
    Apply(Operation, Vec<PrimitiveExpr>),
}

impl PrimitiveExpr {
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

/// A versioned semantic declaration. The expression, rather than a new
/// closed Operation variant, is the mathematical source for every backend.
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
