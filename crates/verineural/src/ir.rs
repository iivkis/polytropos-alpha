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
    /// Elementwise 1 when x > 0, otherwise 0 (including NaN in host f32).
    PositiveMask,
    /// Choose true/false tensors by mask > 0, preserving selected element bits.
    SelectPositive,
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
            Self::PositiveMask => "positive_mask",
            Self::SelectPositive => "select_positive",
            Self::Sqrt => "sqrt",
            Self::Reshape(_) => "reshape",
        }
    }

    pub const fn arity(&self) -> usize {
        match self {
            Self::Input => 0,
            Self::Add | Self::Mul | Self::MatMul | Self::BiasAdd => 2,
            Self::SelectPositive => 3,
            Self::Relu | Self::PositiveMask | Self::Sqrt | Self::Reshape(_) => 1,
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
            Self::Add | Self::Mul | Self::SelectPositive => {
                for other in &inputs[1..] {
                    if inputs[0] != *other {
                        return Err(NeuralError::ShapeMismatch {
                            expected: inputs[0].clone(),
                            actual: other.clone(),
                        });
                    }
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
                    return Err(NeuralError::BiasWidthMismatch {
                        columns: inputs[0].dims()[1],
                        bias_length: inputs[1].dims()[0],
                    });
                }
                Ok(inputs[0].clone())
            }
            Self::Relu | Self::PositiveMask | Self::Sqrt => Ok(inputs[0].clone()),
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
            Self::Sqrt | Self::PositiveMask => Domain::NonNegativeFinite,
            Self::SelectPositive => {
                if inputs[1].satisfies(Domain::NonNegativeFinite)
                    && inputs[2].satisfies(Domain::NonNegativeFinite)
                {
                    Domain::NonNegativeFinite
                } else if inputs[1].satisfies(Domain::Finite)
                    && inputs[2].satisfies(Domain::Finite)
                {
                    Domain::Finite
                } else {
                    Domain::Any
                }
            }
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

/// Independent mathematical claims about a module's output (ideal-Int semantics).
/// Declaration is never proof. Claims create mandatory Lean obligations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractClaim {
    Nonnegative,
    EqualsInput(usize),
}

/// A machine-checkable *structural* contract. It is NOT a Lean proof.
///
/// Output guarantees are accepted only if the graph's conservative inference
/// supports them. Source input domains remain caller-supplied assumptions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathContract {
    pub requires: Vec<PortContract>,
    pub ensures: PortContract,
    pub claims: Vec<ContractClaim>,
}

impl MathContract {
    pub fn new(requires: Vec<PortContract>, ensures: PortContract) -> Self {
        Self {
            requires,
            ensures,
            claims: Vec::new(),
        }
    }

    /// The declared claim is checked in the Lean backend, not in this builder.
    pub fn with_claim(mut self, claim: ContractClaim) -> Self {
        self.claims.push(claim);
        self
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
    /// First node index that did not exist before this module body was built.
    /// Proof generators use it to reject undeclared captures from the outer graph.
    pub body_start: usize,
    pub output: NodeId,
    pub contract: MathContract,
    /// Versioned semantic declaration, not a proof identifier by itself.
    pub semantic_version: &'static str,
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
    fn semantic_version(&self) -> &'static str {
        "unversioned"
    }
}

/// Named output contracts for a multi-output module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamedPortContract {
    pub name: &'static str,
    pub port: PortContract,
    pub claims: Vec<ContractClaim>,
}

impl NamedPortContract {
    pub fn new(name: &'static str, port: PortContract) -> Self {
        Self {
            name,
            port,
            claims: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiOutputContract {
    pub inputs: Vec<NamedPortContract>,
    pub outputs: Vec<NamedPortContract>,
}

pub trait MultiOutputModule {
    fn name(&self) -> &'static str;
    fn contract(&self) -> &MultiOutputContract;
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId])
    -> Result<Vec<NodeId>, NeuralError>;
    fn semantic_version(&self) -> &'static str {
        "unversioned"
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiModuleInvocation {
    pub name: &'static str,
    pub semantic_version: &'static str,
    pub inputs: Vec<NodeId>,
    pub body_start: usize,
    pub body_end: usize,
    pub outputs: Vec<NodeId>,
    pub contract: MultiOutputContract,
}

impl MultiModuleInvocation {
    /// Resolve an output by its declared stable port name, rather than relying
    /// on the caller to remember a positional result index.
    pub fn output_named(&self, name: &str) -> Option<NodeId> {
        self.contract
            .outputs
            .iter()
            .zip(&self.outputs)
            .find(|(port, _)| port.name == name)
            .map(|(_, &output)| output)
    }
}

/// A bounded, explicit state-transition chain. `states[0]` is the initial
/// value and `states[t + 1]` is the `next_state` returned at step `t`.
/// This represents finite unrolling, not inductive verification for all time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedScan {
    pub states: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    /// Indices of instantiated step modules in `NeuralGraph::multi_modules`.
    pub module_indices: Vec<usize>,
}

impl FixedScan {
    /// Select every state transition and readout for a single Lean program.
    /// Proof generation still requires a separate Lean checker invocation.
    pub fn proof_outputs(&self) -> Vec<NodeId> {
        self.states
            .iter()
            .skip(1)
            .chain(self.outputs.iter())
            .copied()
            .collect()
    }
}

/// Append-only Semantic Neural IR with graph-scoped edges and checked contracts.
#[derive(Clone, Debug)]
pub struct NeuralGraph {
    identity: u64,
    policy: FP32Policy,
    nodes: Vec<Node>,
    modules: Vec<ModuleInvocation>,
    multi_modules: Vec<MultiModuleInvocation>,
}

impl NeuralGraph {
    pub fn new(policy: FP32Policy) -> Self {
        Self {
            identity: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            policy,
            nodes: Vec::new(),
            modules: Vec::new(),
            multi_modules: Vec::new(),
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

    pub fn multi_modules(&self) -> &[MultiModuleInvocation] {
        &self.multi_modules
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

    /// Pointwise 0/1 mask for the strict positive branch (NaN takes false branch).
    pub fn positive_mask(&mut self, a: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::PositiveMask, &[a])
    }

    /// Elementwise conditional selection, no implicit broadcast or arithmetic.
    pub fn select_positive(
        &mut self,
        mask: NodeId,
        yes: NodeId,
        no: NodeId,
    ) -> Result<NodeId, NeuralError> {
        self.apply(Operation::SelectPositive, &[mask, yes, no])
    }

    pub fn sqrt(&mut self, a: NodeId) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Sqrt, &[a])
    }

    pub fn reshape(&mut self, a: NodeId, shape: Shape) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Reshape(shape), &[a])
    }

    /// Check every edge crossing a module boundary, including nested invocations.
    /// An output must be owned by this invocation, not borrowed from its caller.
    fn validate_body(
        &self,
        body_start: usize,
        modules_start: usize,
        multi_start: usize,
        allowed: &[NodeId],
        outputs: &[NodeId],
        name: &'static str,
    ) -> Result<(), NeuralError> {
        if self.nodes.len() == body_start || outputs.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "empty body or outputs",
            });
        }
        for &output in outputs {
            self.node(output)?;
            if output.index < body_start {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "output not owned by module",
                });
            }
        }
        let authorized = |id: NodeId| id.index >= body_start || allowed.contains(&id);
        for node in &self.nodes[body_start..] {
            if matches!(node.operation, Operation::Input) {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "module declared a hidden source input",
                });
            }
            if node.inputs.iter().any(|&id| !authorized(id)) {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "undeclared external capture",
                });
            }
        }
        for nested in &self.modules[modules_start..] {
            if nested.inputs.iter().any(|&id| !authorized(id)) {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "nested module captured undeclared input",
                });
            }
        }
        for nested in &self.multi_modules[multi_start..] {
            if nested.inputs.iter().any(|&id| !authorized(id)) {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "nested multi-module captured undeclared input",
                });
            }
        }
        Ok(())
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

        let body_start = self.nodes.len();
        let module_start = self.modules.len();
        let multi_start = self.multi_modules.len();
        let mut candidate = self.clone();
        let output = module.build(&mut candidate, inputs)?;
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            inputs,
            &[output],
            module.name(),
        )?;
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
            body_start,
            output,
            contract: contract.clone(),
            semantic_version: module.semantic_version(),
        });
        *self = candidate;
        Ok(output)
    }

    /// Unroll a statically bounded state transition, without introducing a
    /// built-in recurrent architecture or an implicit GPU execution mode.
    /// The transition's interface must be exactly:
    /// inputs: `state`, `input`; outputs: `next_state`, `output`.
    /// Step instances use `apply_multi_module` with the same shape/domain
    /// checks and named proof obligations as any other consumer module.
    ///
    /// All steps are committed atomically, including when a later input
    /// violates its contract. The final state and every readout are explicit
    /// IR nodes usable by Lean generation and CUDA Graph lowering.
    pub fn scan_fixed<M: MultiOutputModule>(
        &mut self,
        transition: &M,
        initial_state: NodeId,
        inputs: &[NodeId],
    ) -> Result<FixedScan, NeuralError> {
        let contract = transition.contract();
        if contract.inputs.len() != 2
            || contract.outputs.len() != 2
            || contract.inputs[0].name != "state"
            || contract.inputs[1].name != "input"
            || contract.outputs[0].name != "next_state"
            || contract.outputs[1].name != "output"
        {
            return Err(NeuralError::ContractViolation {
                module: transition.name(),
                detail: "fixed scan requires state/input and next_state/output ports",
            });
        }
        if inputs.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: transition.name(),
                detail: "fixed scan requires at least one step",
            });
        }
        if contract.inputs[0].port.shape != contract.outputs[0].port.shape
            || !contract.outputs[0]
                .port
                .domain
                .satisfies(contract.inputs[0].port.domain)
        {
            return Err(NeuralError::ContractViolation {
                module: transition.name(),
                detail: "next_state does not satisfy the next step's state port",
            });
        }
        self.node(initial_state)?;
        for &input in inputs {
            self.node(input)?;
        }
        let mut candidate = self.clone();
        let mut states = vec![initial_state];
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut module_indices = Vec::with_capacity(inputs.len());
        for &input in inputs {
            let state = *states.last().expect("initialized with initial_state");
            let result = candidate.apply_multi_module(transition, &[state, input])?;
            module_indices.push(candidate.multi_modules.len() - 1);
            states.push(result[0]);
            outputs.push(result[1]);
        }
        *self = candidate;
        Ok(FixedScan {
            states,
            outputs,
            module_indices,
        })
    }

    /// Multi-output module invocation is atomic; each returned port has its own
    /// structural contract. Formal proof obligations remain per output.
    pub fn apply_multi_module<M: MultiOutputModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        let contract = module.contract();
        if inputs.len() != contract.inputs.len() || contract.outputs.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: module.name(),
                detail: "incorrect multi-output arity",
            });
        }
        for ports in [&contract.inputs, &contract.outputs] {
            let mut names = std::collections::BTreeSet::new();
            for port in ports {
                if port.name.is_empty() || !names.insert(port.name) {
                    return Err(NeuralError::ContractViolation {
                        module: module.name(),
                        detail: "empty or duplicated named port",
                    });
                }
            }
        }
        // Input-port claims cannot be silently treated as proved output
        // properties. Until predicate-valued preconditions are supported,
        // callers may constrain inputs only via the checked shape/domain port.
        if contract.inputs.iter().any(|port| !port.claims.is_empty()) {
            return Err(NeuralError::ContractViolation {
                module: module.name(),
                detail: "input port claims are unsupported",
            });
        }
        for (&input, port) in inputs.iter().zip(&contract.inputs) {
            let node = self.node(input)?;
            if node.shape != port.port.shape {
                return Err(NeuralError::ShapeMismatch {
                    expected: port.port.shape.clone(),
                    actual: node.shape.clone(),
                });
            }
            if !node.domain.satisfies(port.port.domain) {
                return Err(NeuralError::DomainNotEstablished {
                    required: port.port.domain,
                    actual: node.domain,
                });
            }
        }
        let body_start = self.nodes.len();
        let module_start = self.modules.len();
        let multi_start = self.multi_modules.len();
        let mut candidate = self.clone();
        let outputs = module.build(&mut candidate, inputs)?;
        if outputs.len() != contract.outputs.len() {
            return Err(NeuralError::ContractViolation {
                module: module.name(),
                detail: "incorrect multi-output result count",
            });
        }
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            inputs,
            &outputs,
            module.name(),
        )?;
        if outputs
            .iter()
            .map(|id| id.index())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != outputs.len()
        {
            return Err(NeuralError::ContractViolation {
                module: module.name(),
                detail: "duplicate output identity",
            });
        }
        for (&id, port) in outputs.iter().zip(&contract.outputs) {
            let node = candidate.node(id)?;
            if node.shape != port.port.shape {
                return Err(NeuralError::ShapeMismatch {
                    expected: port.port.shape.clone(),
                    actual: node.shape.clone(),
                });
            }
            if !node.domain.satisfies(port.port.domain) {
                return Err(NeuralError::DomainNotEstablished {
                    required: port.port.domain,
                    actual: node.domain,
                });
            }
        }
        candidate.multi_modules.push(MultiModuleInvocation {
            name: module.name(),
            semantic_version: module.semantic_version(),
            inputs: inputs.to_vec(),
            body_start,
            body_end: candidate.nodes.len(),
            outputs: outputs.clone(),
            contract: contract.clone(),
        });
        *self = candidate;
        Ok(outputs)
    }
}

impl Default for NeuralGraph {
    fn default() -> Self {
        Self::new(FP32Policy::strict())
    }
}
