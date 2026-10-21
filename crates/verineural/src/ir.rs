use std::sync::atomic::{AtomicU64, Ordering};

use crate::{NeuralError, Shape};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// A *declared* value domain; source-node assertions are not proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Any,
    Finite,
    NonNegativeFinite,
}

impl Domain {
    /// Whether this domain is at least as restrictive as the required domain.
    pub fn satisfies(self, required: Self) -> bool {
        matches!(
            (self, required),
            (_, Self::Any)
                | (Self::Finite, Self::Finite)
                | (
                    Self::NonNegativeFinite,
                    Self::Finite | Self::NonNegativeFinite
                )
        )
    }
}

/// FP32 rounding requirement (not yet checked against executable code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundingMode {
    NearestTiesToEven,
}

/// Whether multiplication and addition may be silently contracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmaMode {
    SeparateOperations,
}

/// Subnormal treatment required by a future backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubnormalMode {
    Preserve,
}

/// Order required for accumulation within a reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReductionOrder {
    LeftToRight,
}

/// Explicit *target* numerical policy. It carries no proof of IEEE compliance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FP32Policy {
    pub rounding: RoundingMode,
    pub fma: FmaMode,
    pub subnormals: SubnormalMode,
    pub reduction: ReductionOrder,
}

impl FP32Policy {
    pub const fn strict() -> Self {
        Self {
            rounding: RoundingMode::NearestTiesToEven,
            fma: FmaMode::SeparateOperations,
            subnormals: SubnormalMode::Preserve,
            reduction: ReductionOrder::LeftToRight,
        }
    }
}

impl Default for FP32Policy {
    fn default() -> Self {
        Self::strict()
    }
}

/// GPUs are not lowered or verified in stage 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuSemantics {
    NotLowered,
}

/// The supported Rust IR DSL. No general Rust parser or arbitrary closures yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Input,
    Add,
    Mul,
    MatMul,
    BiasAdd,
    Relu,
    Sqrt,
    Reshape(Shape),
}

impl Operation {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::MatMul => "matmul",
            Self::BiasAdd => "bias_add",
            Self::Relu => "relu",
            Self::Sqrt => "sqrt",
            Self::Reshape(_) => "reshape",
        }
    }

    pub const fn arity(&self) -> usize {
        match self {
            Self::Input => 0,
            Self::Add | Self::Mul | Self::MatMul | Self::BiasAdd => 2,
            Self::Relu | Self::Sqrt | Self::Reshape(_) => 1,
        }
    }

    /// One source of truth for graph and concrete host-operation shape rules.
    pub fn infer_shape(&self, inputs: &[Shape]) -> Result<Shape, NeuralError> {
        if matches!(self, Self::Input) {
            return Err(NeuralError::InputCannotBeApplied);
        }
        if inputs.len() != self.arity() {
            return Err(NeuralError::WrongArity {
                operation: self.name(),
                expected: self.arity(),
                actual: inputs.len(),
            });
        }
        match self {
            Self::Add | Self::Mul => {
                if inputs[0] != inputs[1] {
                    return Err(NeuralError::ShapeMismatch {
                        expected: inputs[0].clone(),
                        actual: inputs[1].clone(),
                    });
                }
                Ok(inputs[0].clone())
            }
            Self::MatMul => {
                for shape in inputs {
                    if shape.rank() != 2 {
                        return Err(NeuralError::RankMismatch {
                            operation: self.name(),
                            expected: 2,
                            actual: shape.rank(),
                        });
                    }
                }
                let a = inputs[0].dims();
                let b = inputs[1].dims();
                if a[1] != b[0] {
                    return Err(NeuralError::InnerDimensionMismatch {
                        left: a[1],
                        right: b[0],
                    });
                }
                Shape::new(vec![a[0], b[1]])
            }
            Self::BiasAdd => {
                if inputs[0].rank() != 2 {
                    return Err(NeuralError::RankMismatch {
                        operation: self.name(),
                        expected: 2,
                        actual: inputs[0].rank(),
                    });
                }
                if inputs[1].rank() != 1 {
                    return Err(NeuralError::RankMismatch {
                        operation: self.name(),
                        expected: 1,
                        actual: inputs[1].rank(),
                    });
                }
                if inputs[0].dims()[1] != inputs[1].dims()[0] {
                    return Err(NeuralError::InnerDimensionMismatch {
                        left: inputs[0].dims()[1],
                        right: inputs[1].dims()[0],
                    });
                }
                Ok(inputs[0].clone())
            }
            Self::Relu | Self::Sqrt => Ok(inputs[0].clone()),
            Self::Reshape(output) => {
                if inputs[0].element_count() != output.element_count() {
                    return Err(NeuralError::ElementCountMismatch {
                        expected: inputs[0].element_count(),
                        actual: output.element_count(),
                    });
                }
                Ok(output.clone())
            }
            Self::Input => unreachable!("handled before shape inference"),
        }
    }

    fn required_domain(&self) -> Domain {
        if matches!(self, Self::Sqrt) {
            Domain::NonNegativeFinite
        } else {
            Domain::Any
        }
    }

    /// Conservative domain propagation; finite inputs may overflow in FP32.
    fn infer_domain(&self, inputs: &[Domain]) -> Domain {
        match self {
            Self::Relu => {
                if inputs[0].satisfies(Domain::Finite) {
                    Domain::NonNegativeFinite
                } else {
                    Domain::Any
                }
            }
            Self::Sqrt => Domain::NonNegativeFinite,
            Self::Reshape(_) => inputs[0],
            Self::Input | Self::Add | Self::Mul | Self::MatMul | Self::BiasAdd => Domain::Any,
        }
    }
}

/// A port's structural and value-domain requirements or guarantees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortContract {
    pub shape: Shape,
    pub domain: Domain,
}

impl PortContract {
    pub fn new(shape: Shape, domain: Domain) -> Self {
        Self { shape, domain }
    }
}

/// A machine-checkable *structural* contract. It is NOT a Lean proof.
///
/// Output guarantees are accepted only if the graph's conservative inference
/// supports them. Source input domains remain caller-supplied assumptions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathContract {
    pub requires: Vec<PortContract>,
    pub ensures: PortContract,
}

impl MathContract {
    pub fn new(requires: Vec<PortContract>, ensures: PortContract) -> Self {
        Self { requires, ensures }
    }
}

/// The three semantic layers: ideal operator, FP32 target, and missing GPU lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperationSemantics {
    pub ideal: Operation,
    pub fp32: FP32Policy,
    pub gpu: GpuSemantics,
}

/// Node identifiers are scoped to one graph, including transactional clones.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    graph: u64,
    index: usize,
}

impl NodeId {
    pub fn index(self) -> usize {
        self.index
    }
}

/// Immutable public view of a validated symbolic operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub id: NodeId,
    pub operation: Operation,
    pub inputs: Vec<NodeId>,
    pub shape: Shape,
    pub domain: Domain,
    pub contract: MathContract,
    pub semantics: OperationSemantics,
}

/// A successful module application, with its declared checked interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInvocation {
    pub name: &'static str,
    pub inputs: Vec<NodeId>,
    pub output: NodeId,
    pub contract: MathContract,
}

/// A user-defined, composable symbolic module.
///
/// The body builds graph nodes, rather than executing arbitrary Rust code.
/// In stage 2, contracts are checked for structural and conservative domain
/// compatibility only; they are not independently proven.
pub trait NeuralModule {
    fn name(&self) -> &'static str;
    fn contract(&self) -> &MathContract;
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError>;
}

/// Append-only Semantic Neural IR with graph-scoped edges and checked contracts.
#[derive(Clone, Debug)]
pub struct NeuralGraph {
    identity: u64,
    policy: FP32Policy,
    nodes: Vec<Node>,
    modules: Vec<ModuleInvocation>,
}

impl NeuralGraph {
    pub fn new(policy: FP32Policy) -> Self {
        Self {
            identity: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            policy,
            nodes: Vec::new(),
            modules: Vec::new(),
        }
    }

    pub fn policy(&self) -> FP32Policy {
        self.policy
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn modules(&self) -> &[ModuleInvocation] {
        &self.modules
    }

    pub fn node(&self, id: NodeId) -> Result<&Node, NeuralError> {
        if id.graph != self.identity {
            return Err(NeuralError::InvalidNode);
        }
        self.nodes.get(id.index).ok_or(NeuralError::InvalidNode)
    }

    /// Declare a symbolic input and its caller-supplied domain assumption.
    /// The assumption must be checked for actual values before execution.
    pub fn input(&mut self, shape: Shape, domain: Domain) -> NodeId {
        let id = NodeId {
            graph: self.identity,
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            id,
            operation: Operation::Input,
            inputs: Vec::new(),
            shape: shape.clone(),
            domain,
            contract: MathContract::new(Vec::new(), PortContract::new(shape, domain)),
            semantics: OperationSemantics {
                ideal: Operation::Input,
                fp32: self.policy,
                gpu: GpuSemantics::NotLowered,
            },
        });
        id
    }

    /// Append an operation only after checking arity, shape, source and domain.
    pub fn apply(&mut self, op: Operation, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        if matches!(op, Operation::Input) {
            return Err(NeuralError::InputCannotBeApplied);
        }
        if inputs.len() != op.arity() {
            return Err(NeuralError::WrongArity {
                operation: op.name(),
                expected: op.arity(),
                actual: inputs.len(),
            });
        }
        let source_nodes = inputs
            .iter()
            .map(|&id| self.node(id))
            .collect::<Result<Vec<_>, _>>()?;
        let shapes = source_nodes
            .iter()
            .map(|n| n.shape.clone())
            .collect::<Vec<_>>();
        let domains = source_nodes.iter().map(|n| n.domain).collect::<Vec<_>>();
        let shape = op.infer_shape(&shapes)?;
        let required = op.required_domain();
        for &actual in &domains {
            if !actual.satisfies(required) {
                return Err(NeuralError::DomainNotEstablished { required, actual });
            }
        }
        let contract = MathContract::new(
            shapes
                .into_iter()
                .map(|shape| PortContract::new(shape, required))
                .collect(),
            PortContract::new(shape.clone(), op.infer_domain(&domains)),
        );
        let id = NodeId {
            graph: self.identity,
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            id,
            operation: op.clone(),
            inputs: inputs.to_vec(),
            shape,
            domain: contract.ensures.domain,
            contract,
            semantics: OperationSemantics {
                ideal: op,
                fp32: self.policy,
                gpu: GpuSemantics::NotLowered,
            },
        });
        Ok(id)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Add, &[a, b])
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Mul, &[a, b])
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::MatMul, &[a, b])
    }

    /// Add a length-N bias to each row of an MxN tensor (no arbitrary broadcast).
    pub fn bias_add(&mut self, matrix: NodeId, bias: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::BiasAdd, &[matrix, bias])
    }

    pub fn relu(&mut self, a: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Relu, &[a])
    }

    pub fn sqrt(&mut self, a: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Sqrt, &[a])
    }

    pub fn reshape(&mut self, a: NodeId, shape: Shape) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Reshape(shape), &[a])
    }

    /// Validate the module interface, build in a clone, then commit atomically.
    /// Failed modules never leave partially appended nodes behind.
    pub fn apply_module<M: NeuralModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<NodeId, NeuralError> {
        let contract = module.contract();
        if inputs.len() != contract.requires.len() {
            return Err(NeuralError::ContractViolation {
                module: module.name(),
                detail: "incorrect number of inputs",
            });
        }
        for (&id, required) in inputs.iter().zip(&contract.requires) {
            let actual = self.node(id)?;
            if actual.shape != required.shape {
                return Err(NeuralError::ShapeMismatch {
                    expected: required.shape.clone(),
                    actual: actual.shape.clone(),
                });
            }
            if !actual.domain.satisfies(required.domain) {
                return Err(NeuralError::DomainNotEstablished {
                    required: required.domain,
                    actual: actual.domain,
                });
            }
        }

        let mut candidate = self.clone();
        let output = module.build(&mut candidate, inputs)?;
        let actual = candidate.node(output)?;
        if actual.shape != contract.ensures.shape {
            return Err(NeuralError::ShapeMismatch {
                expected: contract.ensures.shape.clone(),
                actual: actual.shape.clone(),
            });
        }
        if !actual.domain.satisfies(contract.ensures.domain) {
            return Err(NeuralError::DomainNotEstablished {
                required: contract.ensures.domain,
                actual: actual.domain,
            });
        }
        candidate.modules.push(ModuleInvocation {
            name: module.name(),
            inputs: inputs.to_vec(),
            output,
            contract: contract.clone(),
        });
        *self = candidate;
        Ok(output)
    }
}

impl Default for NeuralGraph {
    fn default() -> Self {
        Self::new(FP32Policy::strict())
    }
}
