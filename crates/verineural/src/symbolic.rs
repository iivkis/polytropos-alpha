//! Symbolic shape contracts resolved to validated concrete binary32 Tensor IR.
//!
//! This is static specialization, not dynamic tensor shapes or polymorphic GPU
//! kernels. An instance is lowered to the existing concrete NeuralModule DAG:
//! Lean/GPU never receive unresolved symbolic dimensions.

use std::collections::BTreeMap;

use crate::{
    ContractClaim, Domain, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId,
    PortContract, Shape,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dimension {
    Fixed(usize),
    Symbol(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicShape {
    pub dims: Vec<Dimension>,
}

impl SymbolicShape {
    pub fn new(dims: Vec<Dimension>) -> Self {
        Self { dims }
    }
}

/// Binary32, contiguous row-major port; shape is specialized before lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicPort {
    pub shape: SymbolicShape,
    pub domain: Domain,
}

impl SymbolicPort {
    pub fn new(shape: SymbolicShape, domain: Domain) -> Self {
        Self { shape, domain }
    }
}

/// Restrictions on a named dimension; all bounds refer to positive sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimensionConstraint {
    AtLeast(&'static str, usize),
    AtMost(&'static str, usize),
    MultipleOf(&'static str, usize),
}

/// A declared, reusable contract for concrete specializations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParametricSignature {
    pub inputs: Vec<SymbolicPort>,
    pub output: SymbolicPort,
    pub constraints: Vec<DimensionConstraint>,
    pub claims: Vec<ContractClaim>,
}

/// Bindings are immutable records of the actual positive dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DimensionBindings {
    values: BTreeMap<&'static str, usize>,
}

impl DimensionBindings {
    pub fn get(&self, symbol: &str) -> Option<usize> {
        self.values.get(symbol).copied()
    }

    pub fn values(&self) -> &BTreeMap<&'static str, usize> {
        &self.values
    }
}

fn valid_symbol(name: &str) -> bool {
    let Some(first) = name.bytes().next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn violation(module: &'static str, detail: &'static str) -> NeuralError {
    NeuralError::ContractViolation { module, detail }
}

impl ParametricSignature {
    /// Bind every repeated symbol across input axes, check explicit constraints,
    /// and resolve the output. Unbound outputs or unknown constraint symbols
    /// are errors, not implicit free dimensions.
    pub fn specialize(
        &self,
        owner: &'static str,
        actual_inputs: &[Shape],
    ) -> Result<(MathContract, DimensionBindings), NeuralError> {
        if actual_inputs.len() != self.inputs.len() {
            return Err(violation(owner, "symbolic module input arity mismatch"));
        }
        let mut bindings = BTreeMap::new();
        for (actual, expected) in actual_inputs.iter().zip(&self.inputs) {
            if actual.rank() != expected.shape.dims.len() {
                return Err(violation(owner, "symbolic tensor rank mismatch"));
            }
            for (&size, dimension) in actual.dims().iter().zip(&expected.shape.dims) {
                match dimension {
                    Dimension::Fixed(0) => {
                        return Err(violation(owner, "zero symbolic constant dimension"));
                    }
                    Dimension::Fixed(expected_size) if size != *expected_size => {
                        return Err(violation(owner, "fixed dimension mismatch"));
                    }
                    Dimension::Symbol(name) => {
                        if !valid_symbol(name) {
                            return Err(violation(owner, "invalid dimension symbol name"));
                        }
                        if let Some(&previous) = bindings.get(name) {
                            if previous != size {
                                return Err(violation(owner, "symbolic dimension mismatch"));
                            }
                        } else {
                            bindings.insert(*name, size);
                        }
                    }
                    Dimension::Fixed(_) => {}
                }
            }
        }

        for constraint in &self.constraints {
            let (symbol, limit) = match *constraint {
                DimensionConstraint::AtLeast(symbol, limit)
                | DimensionConstraint::AtMost(symbol, limit)
                | DimensionConstraint::MultipleOf(symbol, limit) => (symbol, limit),
            };
            let Some(&actual) = bindings.get(symbol) else {
                return Err(violation(owner, "constraint uses unbound dimension symbol"));
            };
            if limit == 0 {
                return Err(violation(
                    owner,
                    "dimension constraint requires positive bound",
                ));
            }
            let satisfied = match *constraint {
                DimensionConstraint::AtLeast(..) => actual >= limit,
                DimensionConstraint::AtMost(..) => actual <= limit,
                DimensionConstraint::MultipleOf(..) => actual % limit == 0,
            };
            if !satisfied {
                return Err(violation(owner, "symbolic dimension constraint violated"));
            }
        }

        let output_dims = self
            .output
            .shape
            .dims
            .iter()
            .map(|dim| match dim {
                Dimension::Fixed(0) => Err(violation(owner, "zero symbolic constant dimension")),
                Dimension::Fixed(size) => Ok(*size),
                Dimension::Symbol(symbol) => {
                    if !valid_symbol(symbol) {
                        return Err(violation(owner, "invalid dimension symbol name"));
                    }
                    bindings
                        .get(symbol)
                        .copied()
                        .ok_or_else(|| violation(owner, "output uses unbound dimension symbol"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output = Shape::new(output_dims)?;
        let requires = actual_inputs
            .iter()
            .zip(&self.inputs)
            .map(|(actual, declared)| PortContract::new(actual.clone(), declared.domain))
            .collect::<Vec<_>>();
        let mut contract =
            MathContract::new(requires, PortContract::new(output, self.output.domain));
        contract.claims.clone_from(&self.claims);
        Ok((contract, DimensionBindings { values: bindings }))
    }
}

/// One consumer module with one symbolic interface, specialized per invocation.
/// Only the supported concrete graph DSL can be emitted by build().
pub trait ParametricModule {
    fn name(&self) -> &'static str;
    fn semantic_version(&self) -> &'static str;
    fn signature(&self) -> &ParametricSignature;
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError>;
}

struct BoundModule<'a, M: ParametricModule> {
    module: &'a M,
    contract: MathContract,
}

impl<M: ParametricModule> NeuralModule for BoundModule<'_, M> {
    fn name(&self) -> &'static str {
        self.module.name()
    }
    fn semantic_version(&self) -> &'static str {
        self.module.semantic_version()
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        self.module.build(graph, inputs)
    }
}

impl NeuralGraph {
    /// Resolve a symbolic consumer interface to a concrete module instance.
    /// All failures preserve the graph, including hidden captures and post-
    /// condition failures, because apply_module commits transactionally.
    pub fn apply_parametric_module<M: ParametricModule>(
        &mut self,
        module: &M,
        inputs: &[NodeId],
    ) -> Result<(NodeId, DimensionBindings), NeuralError> {
        let shapes = inputs
            .iter()
            .map(|&input| Ok(self.node(input)?.shape.clone()))
            .collect::<Result<Vec<_>, NeuralError>>()?;
        let (contract, bindings) = module.signature().specialize(module.name(), &shapes)?;
        let bounded = BoundModule { module, contract };
        let output = self.apply_module(&bounded, inputs)?;
        Ok((output, bindings))
    }
}
