//! Bounded reverse-mode differentiation of the *same* typed Semantic Neural IR.
//! Creates a second ordinary NeuralGraph by appending derivatives to a clone.
//! No opaque callbacks, no architecture-specific kernels and no fictitious FP32
//! certificate: mathematical reverse rules and NVIDIA execution are separate.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{DType, Domain, NeuralError, NeuralGraph, NodeId, Operation, Shape, Tensor};

#[derive(Debug)]
pub enum DifferentiationError {
    Neural(NeuralError),
    Invalid(&'static str),
    Unsupported {
        node: usize,
        operation: &'static str,
    },
}

impl fmt::Display for DifferentiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neural(err) => write!(f, "{err}"),
            Self::Invalid(s) => write!(f, "invalid derivative request: {s}"),
            Self::Unsupported { node, operation } => write!(
                f,
                "reverse-mode node {node}: unsupported derivative for {operation}"
            ),
        }
    }
}
impl Error for DifferentiationError {}
impl From<NeuralError> for DifferentiationError {
    fn from(err: NeuralError) -> Self {
        Self::Neural(err)
    }
}

/// Differentiable graph contains the *primal* as an unchanged prefix.
/// Input/output NodeId values therefore retain their graph identity.
/// Added source tensors are explicit and appear in 'constants'.
#[derive(Clone, Debug)]
pub struct ReverseGraph {
    graph: NeuralGraph,
    root: NodeId,
    seed: Option<NodeId>,
    gradients: Vec<(NodeId, NodeId)>,
    constants: Vec<(NodeId, Tensor<f32>)>,
}

impl ReverseGraph {
    pub fn graph(&self) -> &NeuralGraph {
        &self.graph
    }
    pub fn root(&self) -> NodeId {
        self.root
    }
    pub fn seed(&self) -> NodeId {
        self.seed.expect("graph-provided VJP has no external seed")
    }
    pub fn gradients(&self) -> &[(NodeId, NodeId)] {
        &self.gradients
    }
    pub fn constants(&self) -> &[(NodeId, Tensor<f32>)] {
        &self.constants
    }
    pub fn gradient_of(&self, node: NodeId) -> Option<NodeId> {
        self.gradients
            .iter()
            .find(|(source, _)| *source == node)
            .map(|(_, grad)| *grad)
    }

    /// Add seed and internally generated zero/one tensors to primal bindings.
    /// Filter this result to GpuProgram::source_inputs before a sliced GPU run.
    pub fn bindings(
        &self,
        primal: &[(NodeId, Tensor<f32>)],
        seed: Tensor<f32>,
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, DifferentiationError> {
        let source = self.seed.ok_or(DifferentiationError::Invalid(
            "graph-provided VJP does not accept an external seed",
        ))?;
        if seed.shape() != &self.graph.node(self.root)?.shape {
            return Err(DifferentiationError::Invalid(
                "seed shape differs from differentiated output",
            ));
        }
        if seed.values().iter().any(|v| !v.is_finite()) {
            return Err(DifferentiationError::Invalid("seed must be finite"));
        }
        let mut result = primal.to_vec();
        result.push((source, seed));
        result.extend(self.constants.iter().cloned());
        Ok(result)
    }
}

struct ReverseBuilder {
    graph: NeuralGraph,
    constants: Vec<(NodeId, Tensor<f32>)>,
    fills: BTreeMap<(Vec<usize>, u32), NodeId>,
}

impl ReverseBuilder {
    fn fill(&mut self, shape: &Shape, value: f32) -> Result<NodeId, DifferentiationError> {
        let key = (shape.dims().to_vec(), value.to_bits());
        if let Some(id) = self.fills.get(&key) {
            return Ok(*id);
        }
        let tensor = Tensor::from_vec(key.0.clone(), vec![value; shape.element_count()])?;
        let id = self.graph.input(shape.clone(), Domain::Finite);
        self.constants.push((id, tensor));
        self.fills.insert(key, id);
        Ok(id)
    }

    fn neg(&mut self, value: NodeId) -> Result<NodeId, DifferentiationError> {
        let shape = self.graph.node(value)?.shape.clone();
        let zero = self.fill(&shape, 0.)?;
        Ok(self.graph.sub(zero, value)?)
    }

    fn reduce_scalar(&mut self, grad: NodeId) -> Result<NodeId, DifferentiationError> {
        let shape = self.graph.node(grad)?.shape.clone();
        if shape.dims() == [1, 1] {
            return Ok(grad);
        }
        if shape.rank() != 2 || shape.dims()[0] != 1 {
            return Err(DifferentiationError::Invalid(
                "scalar broadcast adjoint supports one row",
            ));
        }
        let ones = self.fill(&Shape::new(vec![shape.dims()[1], 1])?, 1.)?;
        Ok(self.graph.matmul(grad, ones)?)
    }

    fn scatter_slice(
        &mut self,
        grad: NodeId,
        start: usize,
        source_width: usize,
    ) -> Result<NodeId, DifferentiationError> {
        let written = self.graph.node(grad)?.shape.element_count();
        let mut parts = Vec::new();
        if start != 0 {
            parts.push(self.fill(&Shape::new(vec![1, start])?, 0.)?);
        }
        parts.push(grad);
        if start + written < source_width {
            parts.push(self.fill(&Shape::new(vec![1, source_width - start - written])?, 0.)?);
        }
        Ok(self.graph.concat(1, &parts)?)
    }
}

fn pointwise_kind(operation: &Operation) -> Option<&'static str> {
    let Operation::Pointwise(spec) = operation else {
        return None;
    };
    match (
        spec.id(),
        spec.version(),
        spec.arity(),
        spec.expression().canonical().as_str(),
    ) {
        ("verineural.sub", "1", 2, "sub(arg:0,arg:1)") => Some("sub"),
        ("verineural.div", "1", 2, "div(arg:0,arg:1)") => Some("div"),
        ("verineural.exp", "1", 1, "exp(arg:0)") => Some("exp"),
        ("verineural.sqrt", "1", 1, "sqrt(arg:0)") => Some("sqrt"),
        _ => None,
    }
}

fn supported(operation: &Operation, shape: &Shape, inputs: &[Shape]) -> bool {
    match operation {
        Operation::Input
        | Operation::Add
        | Operation::Mul
        | Operation::MatMul
        | Operation::BiasAdd
        | Operation::Reshape(_) => true,
        Operation::Permute { axes } => axes.as_slice() == [1, 0] && shape.rank() == 2,
        Operation::ReverseAxis { axis } => shape.rank() == 2 && *axis < 2,
        Operation::SliceAxis { axis: 1, .. } => {
            shape.rank() == 2 && shape.dims()[0] == 1 && inputs[0].dims()[0] == 1
        }
        Operation::Concat { axis: 1 } => {
            shape.rank() == 2 && shape.dims()[0] == 1 && inputs.iter().all(|s| s.dims()[0] == 1)
        }
        Operation::BroadcastTo(_) => {
            inputs[0] == *shape
                || (inputs[0].dims() == [1, 1] && shape.rank() == 2 && shape.dims()[0] == 1)
        }
        Operation::Pointwise(_) => pointwise_kind(operation).is_some(),
        _ => false,
    }
}

/// The seed is dL/d(root), supplied by the consumer. The returned graph
/// computes vector-Jacobian products for arbitrary requested primal inputs.
/// Input sources not depending on 'root' receive exact zero gradient.
/// All *reachable* primal operations are checked before derivative construction.
pub fn differentiate(
    primal: &NeuralGraph,
    root: NodeId,
    wrt: &[NodeId],
) -> Result<ReverseGraph, DifferentiationError> {
    let output = primal.node(root)?;
    let mut seeded = primal.clone();
    let seed = seeded.input(output.shape.clone(), Domain::Finite);
    differentiate_impl(&seeded, &[(root, seed)], wrt, Some(seed))
}

/// Compute the sum of vector-Jacobian products for multiple output roots.
/// Each seed is an existing IR expression, treated as a terminal adjoint.
/// Gradients do NOT propagate through the expression producing a seed.
/// This allows GPU loss-gradient construction and parameter VJPs in one
/// executable CUDA Graph; no machine-code/FP32 certificate is claimed.
pub fn differentiate_with_seeds(
    primal: &NeuralGraph,
    roots: &[(NodeId, NodeId)],
    wrt: &[NodeId],
) -> Result<ReverseGraph, DifferentiationError> {
    differentiate_impl(primal, roots, wrt, None)
}

fn differentiate_impl(
    primal: &NeuralGraph,
    roots: &[(NodeId, NodeId)],
    wrt: &[NodeId],
    external_seed: Option<NodeId>,
) -> Result<ReverseGraph, DifferentiationError> {
    let Some(&(root, _)) = roots.first() else {
        return Err(DifferentiationError::Invalid("at least one output/seed pair required"));
    };
    let mut unique_roots = BTreeSet::new();
    for &(output, seed) in roots {
        let y = primal.node(output)?;
        let s = primal.node(seed)?;
        if !unique_roots.insert(output.index())
            || y.dtype != DType::F32
            || s.dtype != DType::F32
            || y.shape != s.shape
        {
            return Err(DifferentiationError::Invalid(
                "output/seed pairs must be unique, matching F32 tensors",
            ));
        }
    }
    if wrt.is_empty() {
        return Err(DifferentiationError::Invalid(
            "at least one gradient target required",
        ));
    }
    let mut seen = BTreeSet::new();
    for &id in wrt {
        let node = primal.node(id)?;
        if !seen.insert(id.index())
            || node.dtype != DType::F32
            || !matches!(node.operation, Operation::Input)
        {
            return Err(DifferentiationError::Invalid(
                "gradient targets must be distinct F32 source nodes",
            ));
        }
    }

    let mut reachable = BTreeSet::new();
    let mut pending = roots.iter().map(|&(y, _)| y).collect::<Vec<_>>();
    while let Some(id) = pending.pop() {
        let node = primal.node(id)?;
        if !reachable.insert(id.index()) {
            continue;
        }
        let inputs = node
            .inputs
            .iter()
            .map(|x| primal.node(*x).map(|n| n.shape.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        if !supported(&node.operation, &node.shape, &inputs) {
            return Err(DifferentiationError::Unsupported {
                node: id.index(),
                operation: node.operation.name(),
            });
        }
        pending.extend(node.inputs.iter().copied());
    }

    // Retain only primal nodes that can transmit an adjoint to requested
    // targets. This matters for a 1,920-node network when training one small
    // parameter: backprop must not materialize irrelevant derivative kernels.
    let mut relevant = wrt.iter().map(|id| id.index()).collect::<BTreeSet<_>>();
    for &index in &reachable {
        if primal.nodes()[index]
            .inputs
            .iter()
            .any(|id| relevant.contains(&id.index()))
        {
            relevant.insert(index);
        }
    }
    let mut b = ReverseBuilder {
        graph: primal.clone(),
        constants: Vec::new(),
        fills: BTreeMap::new(),
    };
    let mut adjoints = BTreeMap::<usize, NodeId>::new();
    for &(y, seed) in roots {
        adjoints.insert(y.index(), seed);
    }
    for index in reachable.into_iter().rev() {
        if !relevant.contains(&index) {
            continue;
        }
        let Some(delta) = adjoints.get(&index).copied() else {
            continue;
        };
        let node = primal.nodes()[index].clone();
        let ids = &node.inputs;
        let contributions: Vec<(NodeId, NodeId)> = match &node.operation {
            Operation::Input => vec![],
            Operation::Add => vec![(ids[0], delta), (ids[1], delta)],
            Operation::Mul => vec![
                (ids[0], b.graph.mul(delta, ids[1])?),
                (ids[1], b.graph.mul(delta, ids[0])?),
            ],
            Operation::MatMul => {
                let a_t = b.graph.permute(ids[0], &[1, 0])?;
                let b_t = b.graph.permute(ids[1], &[1, 0])?;
                vec![
                    (ids[0], b.graph.matmul(delta, b_t)?),
                    (ids[1], b.graph.matmul(a_t, delta)?),
                ]
            }
            Operation::BiasAdd => {
                let bias_shape = &primal.node(ids[1])?.shape;
                let mut db = delta;
                let rows = node.shape.dims()[0];
                if rows > 1 {
                    let ones = b.fill(&Shape::new(vec![1, rows])?, 1.)?;
                    db = b.graph.matmul(ones, delta)?;
                }
                db = b.graph.reshape(db, bias_shape.clone())?;
                vec![(ids[0], delta), (ids[1], db)]
            }
            Operation::Reshape(_) => vec![(
                ids[0],
                b.graph.reshape(delta, primal.node(ids[0])?.shape.clone())?,
            )],
            Operation::Permute { .. } => vec![(ids[0], b.graph.permute(delta, &[1, 0])?)],
            Operation::ReverseAxis { axis } => vec![(ids[0], b.graph.reverse_axis(delta, *axis)?)],
            Operation::Concat { axis: 1 } => {
                let left = primal.node(ids[0])?.shape.dims()[1];
                let right = primal.node(ids[1])?.shape.dims()[1];
                vec![
                    (ids[0], b.graph.slice_axis(delta, 1, 0, left)?),
                    (ids[1], b.graph.slice_axis(delta, 1, left, right)?),
                ]
            }
            Operation::SliceAxis { axis: 1, start, .. } => {
                let width = primal.node(ids[0])?.shape.dims()[1];
                vec![(ids[0], b.scatter_slice(delta, *start, width)?)]
            }
            Operation::BroadcastTo(_) => {
                let source_shape = &primal.node(ids[0])?.shape;
                let grad = if source_shape == &node.shape {
                    delta
                } else {
                    b.reduce_scalar(delta)?
                };
                vec![(ids[0], grad)]
            }
            Operation::Pointwise(_) => match pointwise_kind(&node.operation) {
                Some("sub") => vec![(ids[0], delta), (ids[1], b.neg(delta)?)],
                Some("div") => {
                    let da = b.graph.div(delta, ids[1])?;
                    let scaled = b.graph.mul(delta, node.id)?;
                    let db = b.graph.div(scaled, ids[1])?;
                    vec![(ids[0], da), (ids[1], b.neg(db)?)]
                }
                Some("exp") => vec![(ids[0], b.graph.mul(delta, node.id)?)],
                Some("sqrt") => {
                    let two_y = b.graph.add(node.id, node.id)?;
                    vec![(ids[0], b.graph.div(delta, two_y)?)]
                }
                _ => unreachable!("prevalidated pointwise signature"),
            },
            _ => unreachable!("prevalidated differentiability rule"),
        };
        for (source, grad) in contributions {
            if !relevant.contains(&source.index()) {
                continue;
            }
            if let Some(existing) = adjoints.get(&source.index()).copied() {
                let summed = b.graph.add(existing, grad)?;
                adjoints.insert(source.index(), summed);
            } else {
                adjoints.insert(source.index(), grad);
            }
        }
    }
    let gradients = wrt
        .iter()
        .map(|&target| {
            let grad = match adjoints.get(&target.index()).copied() {
                Some(id) => id,
                None => b.fill(&primal.node(target)?.shape, 0.)?,
            };
            Ok((target, grad))
        })
        .collect::<Result<Vec<_>, DifferentiationError>>()?;
    Ok(ReverseGraph {
        graph: b.graph,
        root,
        seed: external_seed,
        gradients,
        constants: b.constants,
    })
}
