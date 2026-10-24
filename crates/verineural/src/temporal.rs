//! Finite-window unroll of any named recurrent Model into one ordinary IR DAG.
//! Shared parameters, explicit state edges and one supervised scalar VJP.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use crate::{DType, Domain, Model, NeuralError, NeuralGraph, NodeId, Operation, SourceRole, Tensor};

#[derive(Debug)]
pub enum TemporalError { Neural(NeuralError), Invalid(String) }
impl From<NeuralError> for TemporalError {
    fn from(e: NeuralError) -> Self { Self::Neural(e) }
}
impl fmt::Display for TemporalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neural(e) => write!(f, "{e}"),
            Self::Invalid(e) => write!(f, "invalid temporal unroll: {e}"),
        }
    }
}
impl Error for TemporalError {}

pub struct TemporalUnroll {
    graph: NeuralGraph,
    parameters: BTreeMap<String, NodeId>,
    initial_states: BTreeMap<String, NodeId>,
    inputs: Vec<BTreeMap<String, NodeId>>,
    outputs: Vec<NodeId>,
    cotangents: Vec<NodeId>,
    terminal_states: BTreeMap<String, NodeId>,
    objective: NodeId,
}
impl TemporalUnroll {
    pub fn graph(&self) -> &NeuralGraph { &self.graph }
    pub fn parameters(&self) -> &BTreeMap<String, NodeId> { &self.parameters }
    pub fn initial_states(&self) -> &BTreeMap<String, NodeId> { &self.initial_states }
    pub fn inputs(&self) -> &[BTreeMap<String, NodeId>] { &self.inputs }
    pub fn outputs(&self) -> &[NodeId] { &self.outputs }
    pub fn cotangents(&self) -> &[NodeId] { &self.cotangents }
    pub fn terminal_states(&self) -> &BTreeMap<String, NodeId> { &self.terminal_states }
    pub fn objective(&self) -> NodeId { self.objective }

    /// Strictly checks all named parameter and state sources, every timestep
    /// input and cotangent. No silent default tensors or duplicate bindings.
    pub fn bindings(
        &self,
        base: &[(String, Tensor<f32>)],
        steps: &[Vec<(String, Tensor<f32>)>],
        cotangents: &[Tensor<f32>],
    ) -> Result<Vec<(NodeId, Tensor<f32>)>, TemporalError> {
        if steps.len() != self.inputs.len() || cotangents.len() != self.inputs.len() {
            return Err(TemporalError::Invalid("step/cotangent count mismatch".into()));
        }
        let mut base_values = BTreeMap::new();
        for (name, value) in base {
            if base_values.insert(name.as_str(), value).is_some() {
                return Err(TemporalError::Invalid("duplicate base binding".into()));
            }
        }
        let expected = self.parameters.keys().chain(self.initial_states.keys())
            .map(String::as_str).collect::<BTreeSet<_>>();
        if base_values.keys().copied().collect::<BTreeSet<_>>() != expected {
            return Err(TemporalError::Invalid("missing or extra base binding".into()));
        }
        let mut bound = Vec::new();
        for (name, &node) in self.parameters.iter().chain(self.initial_states.iter()) {
            bound.push((node, (*base_values[name.as_str()]).clone()));
        }
        for (t, values) in steps.iter().enumerate() {
            let mut supplied = BTreeMap::new();
            for (name, value) in values {
                if supplied.insert(name.as_str(), value).is_some() {
                    return Err(TemporalError::Invalid("duplicate step input".into()));
                }
            }
            let expected = self.inputs[t].keys().map(String::as_str).collect::<BTreeSet<_>>();
            if supplied.keys().copied().collect::<BTreeSet<_>>() != expected {
                return Err(TemporalError::Invalid(format!("missing or extra timestep source at {t}")));
            }
            for (name, &node) in &self.inputs[t] {
                bound.push((node, (*supplied[name.as_str()]).clone()));
            }
            bound.push((self.cotangents[t], cotangents[t].clone()));
        }
        for (id, tensor) in &bound {
            let node = self.graph.node(*id)?;
            if tensor.shape() != &node.shape {
                return Err(TemporalError::Invalid("source tensor shape mismatch".into()));
            }
            tensor.validate_domain(node.domain)?;
        }
        Ok(bound)
    }
}

/// Unroll 1..=16 steps. Parameter nodes are shared; state feedback connects
/// pure prior-step producers to new consumers. A new initial state is an explicit
/// truncated-BPTT boundary (no gradient into a previous window).
/// The scalar objective is sum_t dot(output[t], cotangent[t]).
pub fn unroll_model(model: &Model, output_name: &str, steps: usize)
    -> Result<TemporalUnroll, TemporalError>
{
    if !(1..=16).contains(&steps) {
        return Err(TemporalError::Invalid("window length must be 1..=16".into()));
    }
    let root = model.output(output_name)
        .ok_or_else(|| TemporalError::Invalid("unknown supervised output".into()))?;
    let output = model.graph().node(root)?;
    if output.dtype != DType::F32 || output.shape.rank() != 2 || output.shape.dims()[0] != 1 {
        return Err(TemporalError::Invalid("supervised output must be F32 [1,N]".into()));
    }
    let old_nodes = model.graph().nodes();
    let source_by_id = model.sources().iter().map(|(name, &(id, role))|
        (id.index(), (name.clone(), role))).collect::<BTreeMap<_,_>>();
    if source_by_id.len() != model.sources().len() {
        return Err(TemporalError::Invalid("duplicate named source nodes".into()));
    }
    let mut graph = NeuralGraph::new(model.graph().policy());
    let mut params = BTreeMap::new();
    let mut initial = BTreeMap::new();
    for (name, &(id, role)) in model.sources() {
        if role == SourceRole::Input { continue; }
        let node = model.graph().node(id)?;
        let src = graph.input(node.shape.clone(), node.domain);
        match role {
            SourceRole::Parameter => { params.insert(name.clone(), src); }
            SourceRole::State => { initial.insert(name.clone(), src); }
            SourceRole::Input => unreachable!(),
        }
    }
    let mut previous = initial.clone();
    let mut step_inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut cotangents = Vec::new();
    let mut objective = None;
    for _ in 0..steps {
        let mut mapping: Vec<Option<NodeId>> = vec![None; old_nodes.len()];
        let mut inputs = BTreeMap::new();
        for node in old_nodes {
            let new = if matches!(node.operation, Operation::Input) {
                let (name, role) = source_by_id.get(&node.id.index()).ok_or_else(||
                    TemporalError::Invalid("unregistered original input".into()))?;
                match role {
                    SourceRole::Parameter => params[name],
                    SourceRole::State => previous[name],
                    SourceRole::Input => {
                        let src = graph.input(node.shape.clone(), node.domain);
                        inputs.insert(name.clone(), src);
                        src
                    }
                }
            } else {
                let ids = node.inputs.iter().map(|id|
                    mapping[id.index()].ok_or_else(|| TemporalError::Invalid(
                        "original nodes not in topological order".into()))
                ).collect::<Result<Vec<_>,_>>()?;
                graph.apply(node.operation.clone(), &ids)?
            };
            mapping[node.id.index()] = Some(new);
        }
        let logits = mapping[root.index()].ok_or_else(||
            TemporalError::Invalid("missing remapped model output".into()))?;
        let seed = graph.input(output.shape.clone(), Domain::Finite);
        let transposed = graph.permute(seed, &[1,0])?;
        let scalar = graph.matmul(logits, transposed)?;
        objective = Some(match objective {
            None => scalar,
            Some(sum) => graph.add(sum, scalar)?,
        });
        let mut next = BTreeMap::new();
        for (name, &old) in model.state_links() {
            next.insert(name.clone(), mapping[old.index()].ok_or_else(||
                TemporalError::Invalid("missing remapped transition".into()))?);
        }
        if next.len() != initial.len() {
            return Err(TemporalError::Invalid("missing temporal state transition".into()));
        }
        previous = next;
        step_inputs.push(inputs);
        outputs.push(logits);
        cotangents.push(seed);
    }
    Ok(TemporalUnroll {
        graph, parameters: params, initial_states: initial,
        inputs: step_inputs, outputs, cotangents, terminal_states: previous,
        objective: objective.expect("nonempty checked window"),
    })
}
