use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{NeuralError, PointwiseExpr, PointwiseSpec, Shape};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

/// A *declared* value domain; source-node assertions are not proofs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Any,
    Finite,
    NonNegativeFinite,
}

/// Semantic element type carried by a graph node. Source tensors are currently
/// binary32; index tensors are produced by indexing primitives such as argmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Index,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "binary32",
            Self::Index => "index",
        }
    }
}

/// Deterministic tie handling for argmax. The first maximal lane wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgMaxTieBreak {
    First,
}

impl ArgMaxTieBreak {
    pub const fn name(self) -> &'static str {
        match self {
            Self::First => "first",
        }
    }
}

impl Domain {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Finite => "finite",
            Self::NonNegativeFinite => "nonnegative_finite",
        }
    }

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
    /// A versioned external atomic operation with declarative pointwise semantics.
    Pointwise(PointwiseSpec),
    /// Reduce one row-major tensor axis with an explicit left-to-right sum.
    ReduceSum {
        axis: usize,
    },
    /// Deterministic maximum along one axis, with first-tie policy.
    ReduceMax {
        axis: usize,
    },
    /// Right-aligned, singleton-axis broadcasting.
    BroadcastTo(Shape),
    /// Reorder tensor axes. `axes[output_axis] = input_axis`.
    Permute {
        axes: Vec<usize>,
    },
    /// Reverse one tensor axis without changing its shape.
    ReverseAxis {
        axis: usize,
    },
    /// Concatenate two same-rank tensors along one axis.
    Concat {
        axis: usize,
    },
    /// Reduce one axis to deterministic index values. F32 NaN/tie behavior is
    /// fixed by the selected policy instead of delegated to a backend accident.
    ArgMax {
        axis: usize,
        tie_break: ArgMaxTieBreak,
    },
    /// Select one lane from `data` using a one-element Index tensor.
    Take {
        axis: usize,
    },
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
            Self::Pointwise(_) => "pointwise",
            Self::ReduceSum { .. } => "reduce_sum",
            Self::ReduceMax { .. } => "reduce_max",
            Self::BroadcastTo(_) => "broadcast_to",
            Self::Permute { .. } => "permute",
            Self::ReverseAxis { .. } => "reverse_axis",
            Self::Concat { .. } => "concat",
            Self::ArgMax { .. } => "argmax",
            Self::Take { .. } => "take",
        }
    }

    pub const fn arity(&self) -> usize {
        match self {
            Self::Input => 0,
            Self::Add | Self::Mul | Self::MatMul | Self::BiasAdd => 2,
            Self::SelectPositive => 3,
            Self::Relu | Self::PositiveMask | Self::Sqrt | Self::Reshape(_) | Self::BroadcastTo(_) => 1,
            Self::Pointwise(spec) => spec.arity(),
            Self::ReduceSum { .. }
            | Self::ReduceMax { .. }
            | Self::Permute { .. }
            | Self::ReverseAxis { .. }
            | Self::ArgMax { .. } => 1,
            Self::Concat { .. } | Self::Take { .. } => 2,
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
            Self::Add | Self::Mul | Self::SelectPositive | Self::Pointwise(_) => {
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
            Self::ReduceSum { axis } | Self::ReduceMax { axis } => {
                let input = &inputs[0];
                if *axis >= input.rank() {
                    return Err(NeuralError::InvalidAxis {
                        axis: *axis,
                        rank: input.rank(),
                    });
                }
                let mut dims = input.dims().to_vec();
                dims.remove(*axis);
                Shape::new(dims)
            }
            Self::BroadcastTo(output) => {
                let input = &inputs[0];
                if input.rank() > output.rank()
                    || input.dims().iter().rev().zip(output.dims().iter().rev())
                        .any(|(&source, &target)| source != 1 && source != target)
                {
                    return Err(NeuralError::ShapeMismatch {
                        expected: output.clone(),
                        actual: input.clone(),
                    });
                }
                Ok(output.clone())
            }
            Self::Permute { axes } => {
                let input = &inputs[0];
                if axes.len() != input.rank() {
                    return Err(NeuralError::InvalidPermutation {
                        rank: input.rank(),
                        axes: axes.clone(),
                    });
                }
                let mut seen = vec![false; input.rank()];
                for &axis in axes {
                    if axis >= input.rank() || seen[axis] {
                        return Err(NeuralError::InvalidPermutation {
                            rank: input.rank(),
                            axes: axes.clone(),
                        });
                    }
                    seen[axis] = true;
                }
                Shape::new(
                    axes.iter()
                        .map(|&axis| input.dims()[axis])
                        .collect::<Vec<_>>(),
                )
            }
            Self::ReverseAxis { axis } => {
                if *axis >= inputs[0].rank() {
                    return Err(NeuralError::InvalidAxis {
                        axis: *axis,
                        rank: inputs[0].rank(),
                    });
                }
                Ok(inputs[0].clone())
            }
            Self::Concat { axis } => {
                let left = &inputs[0];
                let right = &inputs[1];
                if left.rank() != right.rank() {
                    return Err(NeuralError::RankMismatch {
                        operation: self.name(),
                        expected: left.rank(),
                        actual: right.rank(),
                    });
                }
                if *axis >= left.rank() {
                    return Err(NeuralError::InvalidAxis {
                        axis: *axis,
                        rank: left.rank(),
                    });
                }
                for dim in 0..left.rank() {
                    if dim != *axis && left.dims()[dim] != right.dims()[dim] {
                        return Err(NeuralError::ShapeMismatch {
                            expected: left.clone(),
                            actual: right.clone(),
                        });
                    }
                }
                let mut dims = left.dims().to_vec();
                dims[*axis] = dims[*axis]
                    .checked_add(right.dims()[*axis])
                    .ok_or(NeuralError::ElementCountOverflow)?;
                Shape::new(dims)
            }
            Self::ArgMax { axis, .. } => {
                let input = &inputs[0];
                if *axis >= input.rank() {
                    return Err(NeuralError::InvalidAxis {
                        axis: *axis,
                        rank: input.rank(),
                    });
                }
                let mut dims = input.dims().to_vec();
                dims.remove(*axis);
                Shape::new(dims)
            }
            Self::Take { axis } => {
                let data = &inputs[0];
                let index = &inputs[1];
                if *axis >= data.rank() {
                    return Err(NeuralError::InvalidAxis {
                        axis: *axis,
                        rank: data.rank(),
                    });
                }
                if index.element_count() != 1 {
                    return Err(NeuralError::IndexElementCountMismatch {
                        expected: 1,
                        actual: index.element_count(),
                    });
                }
                let mut dims = data.dims().to_vec();
                dims.remove(*axis);
                Shape::new(dims)
            }
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
        match self {
            Self::Sqrt => Domain::NonNegativeFinite,
            _ => Domain::Any,
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> Result<DType, NeuralError> {
        let require = |position: usize, expected: DType| -> Result<(), NeuralError> {
            let actual = inputs[position];
            if actual != expected {
                return Err(NeuralError::DTypeMismatch {
                    operation: self.name(),
                    expected,
                    actual,
                });
            }
            Ok(())
        };
        match self {
            Self::Input => Err(NeuralError::InputCannotBeApplied),
            Self::Take { .. } => {
                require(0, DType::F32)?;
                require(1, DType::Index)?;
                Ok(DType::F32)
            }
            Self::ArgMax { .. } => {
                require(0, DType::F32)?;
                Ok(DType::Index)
            }
            _ => {
                for position in 0..inputs.len() {
                    require(position, DType::F32)?;
                }
                Ok(DType::F32)
            }
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
                } else if inputs[1].satisfies(Domain::Finite) && inputs[2].satisfies(Domain::Finite)
                {
                    Domain::Finite
                } else {
                    Domain::Any
                }
            }
            Self::Reshape(_) | Self::BroadcastTo(_) | Self::ReduceMax { .. }
            | Self::Permute { .. } | Self::ReverseAxis { .. } => inputs[0],
            Self::Concat { .. } => {
                if inputs
                    .iter()
                    .all(|domain| domain.satisfies(Domain::NonNegativeFinite))
                {
                    Domain::NonNegativeFinite
                } else if inputs.iter().all(|domain| domain.satisfies(Domain::Finite)) {
                    Domain::Finite
                } else {
                    Domain::Any
                }
            }
            Self::Take { .. } => inputs[0],
            Self::ArgMax { .. } => Domain::Any,
            Self::Input
            | Self::Add
            | Self::Mul
            | Self::MatMul
            | Self::BiasAdd
            | Self::Pointwise(_)
            | Self::ReduceSum { .. } => Domain::Any,
        }
    }
}

/// A port's structural and value-domain requirements or guarantees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortContract {
    pub shape: Shape,
    pub dtype: DType,
    pub domain: Domain,
}

impl PortContract {
    pub fn new(shape: Shape, domain: Domain) -> Self {
        Self {
            shape,
            dtype: DType::F32,
            domain,
        }
    }

    pub fn index(shape: Shape) -> Self {
        Self {
            shape,
            dtype: DType::Index,
            domain: Domain::Any,
        }
    }

    fn typed(shape: Shape, dtype: DType, domain: Domain) -> Self {
        Self {
            shape,
            dtype,
            domain,
        }
    }

    pub fn any(shape: Shape) -> Self {
        Self::new(shape, Domain::Any)
    }

    pub fn finite(shape: Shape) -> Self {
        Self::new(shape, Domain::Finite)
    }

    pub fn nonnegative(shape: Shape) -> Self {
        Self::new(shape, Domain::NonNegativeFinite)
    }
}

/// Independent mathematical claims about a module's output (ideal-Int semantics).
/// Declaration is never proof. Claims create mandatory Lean obligations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContractClaim {
    Nonnegative,
    EqualsInput(usize),
    /// A user-declared, independently proved ideal-Int predicate.
    Predicate(Predicate),
}

/// A reference in a proof predicate. Module contracts use Input/Output,
/// architecture goals use Node; mixing these contexts is rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueRef {
    Input(usize),
    Output,
    Node(NodeId),
}

/// Typed, data-only proof obligations. No injected Lean text or trusted callback.
/// The supported prover currently handles tensor equality, nonnegativity and
/// pointwise ordering. Unsupported claims fail closed at proof generation/check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Predicate {
    Nonnegative(ValueRef),
    Equal(ValueRef, ValueRef),
    PointwiseLe(ValueRef, ValueRef),
}

impl Predicate {
    pub fn nonnegative(value: ValueRef) -> Self {
        Self::Nonnegative(value)
    }

    pub fn equal(left: ValueRef, right: ValueRef) -> Self {
        Self::Equal(left, right)
    }

    pub fn pointwise_le(lower: ValueRef, upper: ValueRef) -> Self {
        Self::PointwiseLe(lower, upper)
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

    pub fn nonnegative(self) -> Self {
        self.with_claim(ContractClaim::Nonnegative)
    }

    pub fn equals_input(self, port: usize) -> Self {
        self.with_claim(ContractClaim::EqualsInput(port))
    }

    /// Declare an independent ideal-Int relational contract.
    pub fn predicate(self, predicate: Predicate) -> Self {
        self.with_claim(ContractClaim::Predicate(predicate))
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
    pub dtype: DType,
    pub domain: Domain,
    pub contract: MathContract,
    pub semantics: OperationSemantics,
}

/// A successful module application, with its declared checked interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInvocation {
    pub name: &'static str,
    pub inputs: Vec<NodeId>,
    /// Half-open interval `[body_start, body_end)` owned by this invocation.
    /// It includes every node created by the body, even nodes created after the
    /// returned output. This is the single source of truth for module ownership.
    pub body_start: usize,
    pub body_end: usize,
    pub output: NodeId,
    pub contract: MathContract,
    /// Versioned semantic declaration, not a proof identifier by itself.
    pub semantic_version: &'static str,
}

impl ModuleInvocation {
    pub fn body_range(&self) -> Range<usize> {
        self.body_start..self.body_end
    }
}

/// Borrowed view over either module kind, in real application order.
#[derive(Clone, Copy, Debug)]
pub enum ModuleInstanceRef<'a> {
    Single(&'a ModuleInvocation),
    Multi(&'a MultiModuleInvocation),
}

impl<'a> ModuleInstanceRef<'a> {
    pub fn name(self) -> &'static str {
        match self {
            Self::Single(module) => module.name,
            Self::Multi(module) => module.name,
        }
    }

    pub fn semantic_version(self) -> &'static str {
        match self {
            Self::Single(module) => module.semantic_version,
            Self::Multi(module) => module.semantic_version,
        }
    }

    pub fn inputs(self) -> &'a [NodeId] {
        match self {
            Self::Single(module) => &module.inputs,
            Self::Multi(module) => &module.inputs,
        }
    }

    pub fn outputs(self) -> Vec<NodeId> {
        match self {
            Self::Single(module) => vec![module.output],
            Self::Multi(module) => module.outputs.clone(),
        }
    }

    pub fn contains_output(self, output: NodeId) -> bool {
        match self {
            Self::Single(module) => module.output == output,
            Self::Multi(module) => module.outputs.contains(&output),
        }
    }

    pub fn body_range(self) -> Range<usize> {
        match self {
            Self::Single(module) => module.body_range(),
            Self::Multi(module) => module.body_range(),
        }
    }
}

/// A user-defined, composable symbolic module.
///
/// The body builds graph nodes rather than executing arbitrary Rust code.
/// Contracts are validated structurally; mathematical claims create separate
/// proof obligations in the Lean backend.
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

    pub fn with_claim(mut self, claim: ContractClaim) -> Self {
        self.claims.push(claim);
        self
    }

    pub fn nonnegative(self) -> Self {
        self.with_claim(ContractClaim::Nonnegative)
    }

    pub fn equals_input(self, port: usize) -> Self {
        self.with_claim(ContractClaim::EqualsInput(port))
    }

    /// Declare an independent ideal-Int relational contract.
    pub fn predicate(self, predicate: Predicate) -> Self {
        self.with_claim(ContractClaim::Predicate(predicate))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiOutputContract {
    pub inputs: Vec<NamedPortContract>,
    pub outputs: Vec<NamedPortContract>,
}

impl MultiOutputContract {
    pub fn new(inputs: Vec<NamedPortContract>, outputs: Vec<NamedPortContract>) -> Self {
        Self { inputs, outputs }
    }
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
    pub fn body_range(&self) -> Range<usize> {
        self.body_start..self.body_end
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModuleInstanceKey {
    Single(usize),
    Multi(usize),
}

/// Append-only Semantic Neural IR with graph-scoped edges and checked contracts.
#[derive(Clone, Debug)]
pub struct NeuralGraph {
    identity: u64,
    policy: FP32Policy,
    nodes: Vec<Node>,
    modules: Vec<ModuleInvocation>,
    multi_modules: Vec<MultiModuleInvocation>,
    module_order: Vec<ModuleInstanceKey>,
}

impl NeuralGraph {
    pub fn new(policy: FP32Policy) -> Self {
        Self {
            identity: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            policy,
            nodes: Vec::new(),
            modules: Vec::new(),
            multi_modules: Vec::new(),
            module_order: Vec::new(),
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

    /// Iterate all module invocations in the order in which they were applied.
    /// This is the canonical hierarchy order for architecture tooling.
    pub fn module_instances(&self) -> impl Iterator<Item = ModuleInstanceRef<'_>> {
        self.module_order.iter().map(|key| match *key {
            ModuleInstanceKey::Single(index) => ModuleInstanceRef::Single(&self.modules[index]),
            ModuleInstanceKey::Multi(index) => ModuleInstanceRef::Multi(&self.multi_modules[index]),
        })
    }

    /// Dependency closure of graph roots, including declared module inputs even
    /// when a module body does not read every port directly. Artifact writers
    /// and proof planning use this one definition of semantic reachability.
    pub fn dependency_scope(&self, roots: &[NodeId]) -> Result<BTreeSet<usize>, NeuralError> {
        let mut used = BTreeSet::new();
        let mut pending = roots.to_vec();
        while let Some(id) = pending.pop() {
            let node = self.node(id)?;
            if !used.insert(id.index()) {
                continue;
            }
            pending.extend(node.inputs.iter().copied());
            for module in self.module_instances() {
                if module.contains_output(id) {
                    pending.extend(module.inputs().iter().copied());
                }
            }
        }
        Ok(used)
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
            dtype: DType::F32,
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
        let dtypes = source_nodes.iter().map(|n| n.dtype).collect::<Vec<_>>();
        let shape = op.infer_shape(&shapes)?;
        let dtype = op.infer_dtype(&dtypes)?;
        if let Operation::Take { axis } = &op {
            let axis = *axis;
            let index_node = source_nodes[1];
            let upper_bound = match index_node.operation {
                Operation::ArgMax {
                    axis: source_axis, ..
                } => {
                    let argmax_input = self.node(index_node.inputs[0])?;
                    argmax_input.shape.dims()[source_axis]
                }
                _ => {
                    return Err(NeuralError::IndexRangeUnknown {
                        node: index_node.id.index(),
                    });
                }
            };
            let extent = source_nodes[0].shape.dims()[axis];
            if upper_bound > extent {
                return Err(NeuralError::IndexRangeExceedsAxis {
                    upper_bound,
                    axis,
                    extent,
                });
            }
        }
        let required = op.required_domain();
        for &actual in &domains {
            if !actual.satisfies(required) {
                return Err(NeuralError::DomainNotEstablished { required, actual });
            }
        }
        let contract = MathContract::new(
            shapes
                .into_iter()
                .zip(dtypes)
                .map(|(shape, dtype)| PortContract::typed(shape, dtype, required))
                .collect(),
            PortContract::typed(shape.clone(), dtype, op.infer_domain(&domains)),
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
            dtype,
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

    /// Sum one tensor axis in row-major order. Reductions are not implicitly
    /// reassociated: both the host reference and ideal Lean semantics fold left.
    pub fn reduce_sum(&mut self, input: NodeId, axis: usize) -> Result<NodeId, NeuralError> {
        self.apply(Operation::ReduceSum { axis }, &[input])
    }

    pub fn reduce_max(&mut self, input: NodeId, axis: usize) -> Result<NodeId, NeuralError> {
        self.apply(Operation::ReduceMax { axis }, &[input])
    }

    pub fn broadcast_to(&mut self, input: NodeId, shape: Shape) -> Result<NodeId, NeuralError> {
        self.apply(Operation::BroadcastTo(shape), &[input])
    }

    /// Elementwise subtraction defined as a versioned pointwise AST.
    pub fn sub(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, NeuralError> {
        let spec = PointwiseSpec::new(
            "verineural.sub", "1", 2,
            PointwiseExpr::input(0).sub(PointwiseExpr::input(1)),
        )?;
        spec.apply(self, &[a, b])
    }

    /// Elementwise division: IEEE f32 host reference, ideal-Int proof unsupported.
    pub fn div(&mut self, a: NodeId, b: NodeId) -> Result<NodeId, NeuralError> {
        let spec = PointwiseSpec::new(
            "verineural.div", "1", 2,
            PointwiseExpr::input(0).div(PointwiseExpr::input(1)),
        )?;
        spec.apply(self, &[a, b])
    }

    /// Elementwise natural exponential: host reference only, GPU/proof unsupported.
    pub fn exp(&mut self, input: NodeId) -> Result<NodeId, NeuralError> {
        let spec = PointwiseSpec::new(
            "verineural.exp", "1", 1, PointwiseExpr::input(0).exp(),
        )?;
        spec.apply(self, &[input])
    }

    pub fn permute(&mut self, input: NodeId, axes: &[usize]) -> Result<NodeId, NeuralError> {
        self.apply(
            Operation::Permute {
                axes: axes.to_vec(),
            },
            &[input],
        )
    }

    pub fn reverse_axis(&mut self, input: NodeId, axis: usize) -> Result<NodeId, NeuralError> {
        self.apply(Operation::ReverseAxis { axis }, &[input])
    }

    /// Concatenate one or more tensors by lowering the public variadic API to
    /// the binary semantic primitive. This keeps the core operation minimal.
    pub fn concat(&mut self, axis: usize, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let (&first, rest) = inputs.split_first().ok_or(NeuralError::EmptyConcat)?;
        let first_node = self.node(first)?;
        if axis >= first_node.shape.rank() {
            return Err(NeuralError::InvalidAxis {
                axis,
                rank: first_node.shape.rank(),
            });
        }
        let mut current = first;
        for &next in rest {
            current = self.apply(Operation::Concat { axis }, &[current, next])?;
        }
        Ok(current)
    }

    pub fn argmax(
        &mut self,
        input: NodeId,
        axis: usize,
        tie_break: ArgMaxTieBreak,
    ) -> Result<NodeId, NeuralError> {
        self.apply(Operation::ArgMax { axis, tie_break }, &[input])
    }

    pub fn take(
        &mut self,
        data: NodeId,
        index: NodeId,
        axis: usize,
    ) -> Result<NodeId, NeuralError> {
        self.apply(Operation::Take { axis }, &[data, index])
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

    /// Compose an ad-hoc single-output module without defining a Rust type.
    /// Existing `NeuralModule` implementations delegate to this same path.
    pub fn compose<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        contract: &MathContract,
        inputs: &[NodeId],
        build: F,
    ) -> Result<NodeId, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<NodeId, NeuralError>,
    {
        self.apply_module_with(name, semantic_version, contract, inputs, build)
    }

    /// Validate the module interface, build in a clone, then commit atomically.
    /// Failed modules never leave partially appended nodes behind.
    pub fn apply_module<M: NeuralModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<NodeId, NeuralError> {
        self.apply_module_with(
            module.name(),
            module.semantic_version(),
            module.contract(),
            inputs,
            |graph, inputs| module.build(graph, inputs),
        )
    }

    fn apply_module_with<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        contract: &MathContract,
        inputs: &[NodeId],
        build: F,
    ) -> Result<NodeId, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<NodeId, NeuralError>,
    {
        if inputs.len() != contract.requires.len() {
            return Err(NeuralError::ContractViolation {
                module: name,
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
            if actual.dtype != required.dtype {
                return Err(NeuralError::DTypeMismatch {
                    operation: "module_input",
                    expected: required.dtype,
                    actual: actual.dtype,
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
        let output = build(&mut candidate, inputs)?;
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            inputs,
            &[output],
            name,
        )?;
        let actual = candidate.node(output)?;
        if actual.shape != contract.ensures.shape {
            return Err(NeuralError::ShapeMismatch {
                expected: contract.ensures.shape.clone(),
                actual: actual.shape.clone(),
            });
        }
        if actual.dtype != contract.ensures.dtype {
            return Err(NeuralError::DTypeMismatch {
                operation: "module_output",
                expected: contract.ensures.dtype,
                actual: actual.dtype,
            });
        }
        if !actual.domain.satisfies(contract.ensures.domain) {
            return Err(NeuralError::DomainNotEstablished {
                required: contract.ensures.domain,
                actual: actual.domain,
            });
        }
        let body_end = candidate.nodes.len();
        let module_index = candidate.modules.len();
        candidate.modules.push(ModuleInvocation {
            name,
            inputs: inputs.to_vec(),
            body_start,
            body_end,
            output,
            contract: contract.clone(),
            semantic_version,
        });
        candidate
            .module_order
            .push(ModuleInstanceKey::Single(module_index));
        *self = candidate;
        Ok(output)
    }

    /// Structural composition without a caller-written duplicate MathContract.
    /// Only shapes and conservative domains are inferred, never mathematical
    /// claims or Lean certification. Failed bodies leave this graph unchanged.
    pub fn compose_inferred<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        inputs: &[NodeId],
        build: F,
    ) -> Result<NodeId, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<NodeId, NeuralError>,
    {
        let requires = inputs
            .iter()
            .map(|&id| {
                let node = self.node(id)?;
                Ok(PortContract::typed(
                    node.shape.clone(),
                    node.dtype,
                    node.domain,
                ))
            })
            .collect::<Result<Vec<_>, NeuralError>>()?;
        let body_start = self.nodes.len();
        let module_start = self.modules.len();
        let multi_start = self.multi_modules.len();
        let mut candidate = self.clone();
        let output = build(&mut candidate, inputs)?;
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            inputs,
            &[output],
            name,
        )?;
        let result = candidate.node(output)?;
        let contract = MathContract::new(
            requires,
            PortContract::typed(result.shape.clone(), result.dtype, result.domain),
        );
        let module_index = candidate.modules.len();
        let body_end = candidate.nodes.len();
        candidate.modules.push(ModuleInvocation {
            name,
            inputs: inputs.to_vec(),
            body_start,
            body_end,
            output,
            contract,
            semantic_version,
        });
        candidate
            .module_order
            .push(ModuleInstanceKey::Single(module_index));
        *self = candidate;
        Ok(output)
    }

    /// Infer an atomic named multi-output block from its semantic IR.
    /// Input names and output names are part of the structural contract;
    /// mathematical postconditions still require explicit proof obligations.
    pub fn compose_multi_inferred<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        inputs: &[(&'static str, NodeId)],
        output_names: &[&'static str],
        build: F,
    ) -> Result<Vec<NodeId>, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<Vec<NodeId>, NeuralError>,
    {
        if output_names.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "multi-output block requires at least one output",
            });
        }
        for names in [
            inputs.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            output_names.to_vec(),
        ] {
            let mut seen = BTreeSet::new();
            if names.iter().any(|label| label.is_empty() || !seen.insert(*label)) {
                return Err(NeuralError::ContractViolation {
                    module: name,
                    detail: "empty or duplicate inferred multi-port name",
                });
            }
        }
        let input_ids: Vec<_> = inputs.iter().map(|(_, node)| *node).collect();
        let requires = inputs
            .iter()
            .map(|(label, node)| {
                let value = self.node(*node)?;
                Ok(NamedPortContract::new(
                    label,
                    PortContract::typed(value.shape.clone(), value.dtype, value.domain),
                ))
            })
            .collect::<Result<Vec<_>, NeuralError>>()?;
        let body_start = self.nodes.len();
        let module_start = self.modules.len();
        let multi_start = self.multi_modules.len();
        let mut candidate = self.clone();
        let outputs = build(&mut candidate, &input_ids)?;
        if outputs.len() != output_names.len() {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "incorrect inferred multi-output result count",
            });
        }
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            &input_ids,
            &outputs,
            name,
        )?;
        if outputs.iter().map(|id| id.index()).collect::<BTreeSet<_>>().len()
            != outputs.len()
        {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "duplicate inferred multi-output identity",
            });
        }
        let ensures = outputs
            .iter()
            .zip(output_names)
            .map(|(&id, &label)| {
                let value = candidate.node(id)?;
                Ok(NamedPortContract::new(
                    label,
                    PortContract::typed(value.shape.clone(), value.dtype, value.domain),
                ))
            })
            .collect::<Result<Vec<_>, NeuralError>>()?;
        let module_index = candidate.multi_modules.len();
        candidate.multi_modules.push(MultiModuleInvocation {
            name,
            semantic_version,
            inputs: input_ids,
            body_start,
            body_end: candidate.nodes.len(),
            outputs: outputs.clone(),
            contract: MultiOutputContract::new(requires, ensures),
        });
        candidate.module_order.push(ModuleInstanceKey::Multi(module_index));
        *self = candidate;
        Ok(outputs)
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
            || contract.inputs[0].port.dtype != contract.outputs[0].port.dtype
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

    /// Compose an ad-hoc multi-output module without defining a Rust type.
    pub fn compose_multi<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        contract: &MultiOutputContract,
        inputs: &[NodeId],
        build: F,
    ) -> Result<Vec<NodeId>, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<Vec<NodeId>, NeuralError>,
    {
        self.apply_multi_module_with(name, semantic_version, contract, inputs, build)
    }

    /// Multi-output module invocation is atomic; each returned port has its own
    /// structural contract. Formal proof obligations remain per output.
    pub fn apply_multi_module<M: MultiOutputModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        self.apply_multi_module_with(
            module.name(),
            module.semantic_version(),
            module.contract(),
            inputs,
            |graph, inputs| module.build(graph, inputs),
        )
    }

    fn apply_multi_module_with<F>(
        &mut self,
        name: &'static str,
        semantic_version: &'static str,
        contract: &MultiOutputContract,
        inputs: &[NodeId],
        build: F,
    ) -> Result<Vec<NodeId>, NeuralError>
    where
        F: FnOnce(&mut NeuralGraph, &[NodeId]) -> Result<Vec<NodeId>, NeuralError>,
    {
        if inputs.len() != contract.inputs.len() || contract.outputs.is_empty() {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "incorrect multi-output arity",
            });
        }
        for ports in [&contract.inputs, &contract.outputs] {
            let mut names = std::collections::BTreeSet::new();
            for port in ports {
                if port.name.is_empty() || !names.insert(port.name) {
                    return Err(NeuralError::ContractViolation {
                        module: name,
                        detail: "empty or duplicated named port",
                    });
                }
            }
        }
        if contract.inputs.iter().any(|port| !port.claims.is_empty()) {
            return Err(NeuralError::ContractViolation {
                module: name,
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
            if node.dtype != port.port.dtype {
                return Err(NeuralError::DTypeMismatch {
                    operation: "multi_module_input",
                    expected: port.port.dtype,
                    actual: node.dtype,
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
        let outputs = build(&mut candidate, inputs)?;
        if outputs.len() != contract.outputs.len() {
            return Err(NeuralError::ContractViolation {
                module: name,
                detail: "incorrect multi-output result count",
            });
        }
        candidate.validate_body(
            body_start,
            module_start,
            multi_start,
            inputs,
            &outputs,
            name,
        )?;
        if outputs
            .iter()
            .map(|id| id.index())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != outputs.len()
        {
            return Err(NeuralError::ContractViolation {
                module: name,
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
            if node.dtype != port.port.dtype {
                return Err(NeuralError::DTypeMismatch {
                    operation: "multi_module_output",
                    expected: port.port.dtype,
                    actual: node.dtype,
                });
            }
            if !node.domain.satisfies(port.port.domain) {
                return Err(NeuralError::DomainNotEstablished {
                    required: port.port.domain,
                    actual: node.domain,
                });
            }
        }
        let module_index = candidate.multi_modules.len();
        candidate.multi_modules.push(MultiModuleInvocation {
            name,
            semantic_version,
            inputs: inputs.to_vec(),
            body_start,
            body_end: candidate.nodes.len(),
            outputs: outputs.clone(),
            contract: contract.clone(),
        });
        candidate
            .module_order
            .push(ModuleInstanceKey::Multi(module_index));
        *self = candidate;
        Ok(outputs)
    }
}

impl Default for NeuralGraph {
    fn default() -> Self {
        Self::new(FP32Policy::strict())
    }
}
