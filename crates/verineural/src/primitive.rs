//! Versioned consumer-defined composite primitives lowered through typed Semantic IR.
//! This DSL deliberately cannot wrap opaque host callbacks or pretend to certify
//! unsupported backend operations.
use crate::{MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, Operation, Shape};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveExpr {
    Port(usize),
    Apply(Operation, Vec<PrimitiveExpr>),
}

// Inherent fluent methods are part of the consumer DSL's stable API.
// std::ops traits would require consumers to import the traits for method syntax.
#[allow(clippy::should_implement_trait)]
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

    /// General row-major axis reduction, not a specialized neural block.
    pub fn reduce_sum(self, axis: usize) -> Self {
        Self::apply(Operation::ReduceSum { axis }, [self])
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

/// A small total elementwise mathematical language. Every argument has the
/// same shape. This AST, not a Rust callback, is the semantic source of truth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PointwiseExpr {
    Input(usize),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Exp(Box<Self>),
    Relu(Box<Self>),
}

// Preserve the same fluent expression-building API as PrimitiveExpr.
#[allow(clippy::should_implement_trait)]
impl PointwiseExpr {
    pub fn input(index: usize) -> Self {
        Self::Input(index)
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(rhs))
    }
    pub fn div(self, rhs: Self) -> Self {
        Self::Div(Box::new(self), Box::new(rhs))
    }
    pub fn exp(self) -> Self {
        Self::Exp(Box::new(self))
    }
    pub fn relu(self) -> Self {
        Self::Relu(Box::new(self))
    }

    pub(crate) fn has_ideal_int_semantics(&self) -> bool {
        match self {
            Self::Input(_) => true,
            Self::Add(a, b) | Self::Mul(a, b) => {
                a.has_ideal_int_semantics() && b.has_ideal_int_semantics()
            }
            Self::Relu(a) => a.has_ideal_int_semantics(),
            Self::Sub(..) | Self::Div(..) | Self::Exp(..) => false,
        }
    }

    fn valid(&self, arity: usize, remaining_depth: usize) -> bool {
        if remaining_depth == 0 {
            return false;
        }
        match self {
            Self::Input(index) => *index < arity,
            Self::Add(a, b) | Self::Mul(a, b) | Self::Sub(a, b) | Self::Div(a, b) => {
                a.valid(arity, remaining_depth - 1) && b.valid(arity, remaining_depth - 1)
            }
            Self::Relu(a) | Self::Exp(a) => a.valid(arity, remaining_depth - 1),
        }
    }

    /// Canonical, unambiguous expression encoding for semantic fingerprints.
    pub fn canonical(&self) -> String {
        match self {
            Self::Input(port) => format!("arg:{port}"),
            Self::Add(a, b) => format!("add({},{})", a.canonical(), b.canonical()),
            Self::Mul(a, b) => format!("mul({},{})", a.canonical(), b.canonical()),
            Self::Sub(a, b) => format!("sub({},{})", a.canonical(), b.canonical()),
            Self::Div(a, b) => format!("div({},{})", a.canonical(), b.canonical()),
            Self::Exp(a) => format!("exp({})", a.canonical()),
            Self::Relu(a) => format!("relu({})", a.canonical()),
        }
    }

    /// The Lean interpretation is derived entirely from this same AST.
    /// The caller provides graph-specific, validated tensor references.
    pub(crate) fn lean(&self, input: &impl Fn(usize) -> String) -> String {
        match self {
            Self::Input(port) => input(*port),
            Self::Add(a, b) => {
                format!("(VeriNeural.add {} {})", a.lean(input), b.lean(input))
            }
            Self::Mul(a, b) => {
                format!("(VeriNeural.mul {} {})", a.lean(input), b.lean(input))
            }
            Self::Relu(a) => format!("(VeriNeural.relu {})", a.lean(input)),
            Self::Sub(..) | Self::Div(..) | Self::Exp(..) => {
                unreachable!(
                    "unsupported pointwise expression must be rejected before Lean generation"
                )
            }
        }
    }

    /// Independent ideal-Int theorem rule for an unconditional final ReLU.
    /// All other expressions require a different proof rule: fail closed.
    pub(crate) fn nonnegative_lean(&self, input: &impl Fn(usize) -> String) -> Option<String> {
        if !self.has_ideal_int_semantics() {
            return None;
        }
        match self {
            Self::Relu(inner) => Some(format!("VeriNeural.relu_nonnegative {}", inner.lean(input))),
            _ => None,
        }
    }

    fn evaluate(&self, input: &impl Fn(usize) -> f32) -> f32 {
        match self {
            Self::Input(port) => input(*port),
            Self::Add(a, b) => a.evaluate(input) + b.evaluate(input),
            Self::Mul(a, b) => a.evaluate(input) * b.evaluate(input),
            Self::Sub(a, b) => a.evaluate(input) - b.evaluate(input),
            Self::Div(a, b) => a.evaluate(input) / b.evaluate(input),
            Self::Exp(a) => a.evaluate(input).exp(),
            Self::Relu(inner) => {
                let value = inner.evaluate(input);
                if value.is_nan() || value > 0.0 {
                    value
                } else {
                    0.0
                }
            }
        }
    }
}

/// A genuine single IR operation defined by a consumer, not a pre-expanded
/// neural module. Only supported pointwise AST semantics can be registered.
/// Lean models the ideal integer expression; GPU and FP32 proof backends reject
/// this operation until they implement corresponding verified lowerings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointwiseSpec {
    id: &'static str,
    version: &'static str,
    arity: usize,
    expression: PointwiseExpr,
}

impl PointwiseSpec {
    pub fn new(
        id: &'static str,
        version: &'static str,
        arity: usize,
        expression: PointwiseExpr,
    ) -> Result<Self, NeuralError> {
        if id.is_empty() || version.is_empty() || arity == 0 {
            return Err(NeuralError::ContractViolation {
                module: id,
                detail: "pointwise primitive requires id, version and at least one input",
            });
        }
        // Bounds recursion and artifact sizes; never execute opaque Rust code.
        if !expression.valid(arity, 64) {
            return Err(NeuralError::ContractViolation {
                module: id,
                detail: "expression refers to an invalid input or exceeds 63 levels",
            });
        }
        Ok(Self {
            id,
            version,
            arity,
            expression,
        })
    }

    pub fn id(&self) -> &'static str {
        self.id
    }
    pub fn version(&self) -> &'static str {
        self.version
    }
    pub const fn arity(&self) -> usize {
        self.arity
    }
    pub fn expression(&self) -> &PointwiseExpr {
        &self.expression
    }

    pub fn apply(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.apply(Operation::Pointwise(self.clone()), inputs)
    }

    /// Host reference only, not a floating-point certification or GPU backend.
    pub fn evaluate_host(
        &self,
        inputs: &[crate::Tensor<f32>],
    ) -> Result<crate::Tensor<f32>, NeuralError> {
        let shapes = inputs
            .iter()
            .map(|tensor| tensor.shape().clone())
            .collect::<Vec<_>>();
        let output_shape = Operation::Pointwise(self.clone()).infer_shape(&shapes)?;
        let values = (0..output_shape.element_count())
            .map(|element| {
                self.expression
                    .evaluate(&|port| inputs[port].values()[element])
            })
            .collect::<Vec<_>>();
        crate::Tensor::new(output_shape, values)
    }
}
