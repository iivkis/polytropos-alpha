//! Named GPU execution over the one Semantic Neural IR and the existing
//! CUDA Graph runtime. State feedback is an explicit on-device transfer.
use std::error::Error;
use std::fmt;
use std::path::Path;

use crate::{GpuEngine, GpuError, GpuExecution, Model, ModelError, NodeId, SourceRole, Tensor};

#[derive(Debug)]
pub enum ModelExecutionError {
    Model(ModelError),
    Gpu(GpuError),
    Invalid(String),
}

impl fmt::Display for ModelExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(err) => write!(f, "{err}"),
            Self::Gpu(err) => write!(f, "{err}"),
            Self::Invalid(detail) => write!(f, "invalid model execution: {detail}"),
        }
    }
}

impl Error for ModelExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(err) => Some(err),
            Self::Gpu(err) => Some(err),
            Self::Invalid(_) => None,
        }
    }
}

impl From<ModelError> for ModelExecutionError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<GpuError> for ModelExecutionError {
    fn from(err: GpuError) -> Self {
        Self::Gpu(err)
    }
}

type StateFeedbackLink = (NodeId, NodeId);
type ExecutionRootsAndStateLinks = (Vec<NodeId>, Vec<StateFeedbackLink>);

// A pure next_state may ignore its old state; the GPU execution slice still
// must include that source so the writable feedback destination is allocated.
fn execution_roots_with_states(
    model: &Model,
) -> Result<ExecutionRootsAndStateLinks, ModelExecutionError> {
    let links = model
        .state_links()
        .iter()
        .map(|(name, &next)| {
            model
                .source(name)
                .map(|(state, _)| (state, next))
                .ok_or_else(|| ModelExecutionError::Invalid(format!("unknown state: {name}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut roots = model.execution_roots();
    for &(state, _) in &links {
        if !roots.contains(&state) {
            roots.push(state);
        }
    }
    Ok((roots, links))
}

/// Maintains parameter and state buffers between replays. No extra IR is
/// introduced; every step is the same compiled pure graph followed by
/// explicit device-to-device state assignment.
pub struct ModelExecution<'m> {
    model: &'m Model,
    gpu: GpuExecution,
}

impl GpuEngine {
    /// Requires a complete named initial binding, then compiles all public
    /// outputs and next-state roots into a shared CUDA Graph.
    /// Any unsupported GPU operation fails closed before the CUDA session.
    pub fn prepare_model<'m, K: Into<String>>(
        &self,
        model: &'m Model,
        bindings: impl IntoIterator<Item = (K, Tensor<f32>)>,
        output_dir: &Path,
    ) -> Result<ModelExecution<'m>, ModelExecutionError> {
        let values = model.bind(bindings)?;
        let (roots, links) = execution_roots_with_states(model)?;
        let program = self.compile_outputs(model.graph(), &roots)?;
        // A model may declare inactive named parameters. The GPU program
        // binds only sources reachable from the selected outputs.
        let live = program.source_inputs();
        let active = values
            .into_iter()
            .filter(|(id, _)| live.contains(id))
            .collect::<Vec<_>>();
        let mut gpu = self.prepare(&program, &active, output_dir)?;
        let public = model.outputs().values().copied().collect::<Vec<_>>();
        // Check the full D2D plan before the first step and allocate stable
        // device snapshots for public outputs that alias writable state.
        gpu.configure_model_feedback(&public, &links)?;
        Ok(ModelExecution { model, gpu })
    }
}

impl ModelExecution<'_> {
    fn named_source(
        &self,
        name: &str,
        expected: SourceRole,
    ) -> Result<NodeId, ModelExecutionError> {
        match self.model.source(name) {
            Some((node, role)) if role == expected => Ok(node),
            Some((_, actual)) => Err(ModelExecutionError::Invalid(format!(
                "{name} is {}, expected {}",
                actual.as_str(),
                expected.as_str()
            ))),
            None => Err(ModelExecutionError::Invalid(format!(
                "unknown source: {name}"
            ))),
        }
    }

    /// Update one ephemeral input without rebinding parameters or other inputs.
    pub fn set_input(
        &mut self,
        name: &str,
        value: &Tensor<f32>,
    ) -> Result<(), ModelExecutionError> {
        let id = self.named_source(name, SourceRole::Input)?;
        Ok(self.gpu.update_input(id, value)?)
    }

    /// Persistently replace one parameter, validating its declared domain.
    pub fn set_parameter(
        &mut self,
        name: &str,
        value: &Tensor<f32>,
    ) -> Result<(), ModelExecutionError> {
        let id = self.named_source(name, SourceRole::Parameter)?;
        Ok(self.gpu.update_input(id, value)?)
    }

    /// Explicit reset/seed of a state buffer before a replay.
    pub fn set_state(
        &mut self,
        name: &str,
        value: &Tensor<f32>,
    ) -> Result<(), ModelExecutionError> {
        let id = self.named_source(name, SourceRole::State)?;
        Ok(self.gpu.update_input(id, value)?)
    }

    /// Run the pure model once, without changing temporal state.
    pub fn run(&mut self) -> Result<(), ModelExecutionError> {
        self.gpu.replay()?;
        Ok(self.gpu.snapshot_outputs()?)
    }

    /// Compute and then feed every declared next_state into its matching
    /// state input on-device. Next step reuses the same CUDA Graph.
    pub fn step(&mut self) -> Result<(), ModelExecutionError> {
        self.gpu.replay()?;
        // Preserve the last pure step's public values before feedback mutates
        // any state input (including zero-copy reshape aliases).
        self.gpu.snapshot_outputs()?;
        let links = self
            .model
            .state_links()
            .iter()
            .map(|(name, next)| {
                self.named_source(name, SourceRole::State)
                    .map(|source| (source, *next))
            })
            .collect::<Result<Vec<_>, ModelExecutionError>>()?;
        Ok(self.gpu.feedback_state(&links)?)
    }

    /// Read a public named output after run()/step().
    pub fn output(&mut self, name: &str) -> Result<Tensor<f32>, ModelExecutionError> {
        let id = self
            .model
            .output(name)
            .ok_or_else(|| ModelExecutionError::Invalid(format!("unknown output: {name}")))?;
        Ok(self.gpu.read_selected_output(id)?)
    }

    /// Read the current state input, including the feedback of the last step.
    pub fn state(&mut self, name: &str) -> Result<Tensor<f32>, ModelExecutionError> {
        let id = self.named_source(name, SourceRole::State)?;
        Ok(self.gpu.read_source(id)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, NeuralError, Shape};

    #[test]
    fn named_transition_and_multi_block_keep_one_ir_and_explicit_roles() {
        let mut builder = crate::ModelBuilder::new("temporal", "1");
        let state = builder.state("state", [2], Domain::Any).unwrap();
        let input = builder
            .input("token", [2], Domain::NonNegativeFinite)
            .unwrap();
        let weight = builder
            .parameter("weight", [2], Domain::NonNegativeFinite)
            .unwrap();
        let ports = builder
            .block_multi(
                "transition",
                &[("state", state), ("token", input), ("weight", weight)],
                &["next_state", "readout"],
                |graph, values| {
                    let sum = graph.add(values[0], values[1])?;
                    let next = graph.relu(sum)?;
                    let output = graph.mul(next, values[2])?;
                    Ok(vec![next, output])
                },
            )
            .unwrap();
        let next = ports["next_state"];
        assert!(builder.next_state("state", next).is_ok());
        assert!(matches!(
            builder.next_state("state", next),
            Err(ModelError::Invalid(_))
        ));
        builder.output("readout", ports["readout"]).unwrap();
        let model = builder.finish().unwrap();
        assert_eq!(model.source("state").unwrap().1, SourceRole::State);
        assert_eq!(model.state_links()["state"], next);
        assert_eq!(model.graph().multi_modules().len(), 1);
        assert_eq!(model.execution_roots().len(), 2);
        assert_eq!(
            model
                .metadata("architecture_global_nonnegative")
                .input_roles[&state.index()],
            "state"
        );
        // Compiling before CUDA launch validates that all executable roots
        // are reachable from one graph.
        assert!(
            GpuEngine::default()
                .compile_outputs(model.graph(), &model.execution_roots())
                .is_ok()
        );
    }

    #[test]
    fn replacement_state_is_a_live_gpu_source_even_without_read_dependencies() {
        let mut builder = crate::ModelBuilder::new("replacement", "1");
        let memory = builder.state("memory", [2], Domain::Any).unwrap();
        let token = builder.input("token", [2], Domain::Finite).unwrap();
        let out = builder.graph_mut().relu(token).unwrap();
        builder.output("readout", out).unwrap();
        builder.next_state("memory", token).unwrap();
        let model = builder.finish().unwrap();

        assert!(!model.execution_roots().contains(&memory));
        let (roots, links) = execution_roots_with_states(&model).unwrap();
        assert!(roots.contains(&memory));
        assert_eq!(links, vec![(memory, token)]);
        let program = GpuEngine::default()
            .compile_outputs(model.graph(), &roots)
            .unwrap();
        assert!(program.source_inputs().contains(&memory));
        assert!(program.outputs().contains(&memory));
        assert!(program.source_inputs().contains(&token));
    }

    #[test]
    fn invalid_temporal_connections_and_capture_leave_builder_consistent() {
        let mut builder = crate::ModelBuilder::new("bad_temporal", "1");
        let state = builder
            .state("state", [2], Domain::NonNegativeFinite)
            .unwrap();
        let input = builder.input("input", [2], Domain::Finite).unwrap();
        let bad = builder
            .graph_mut()
            .reshape(input, Shape::new(vec![1, 2]).unwrap())
            .unwrap();
        assert!(matches!(
            builder.next_state("state", bad),
            Err(ModelError::Neural(NeuralError::ShapeMismatch { .. }))
        ));
        assert!(matches!(
            builder.next_state("state", input),
            Err(ModelError::Neural(NeuralError::DomainNotEstablished { .. }))
        ));
        let before = builder.graph_mut().nodes().len();
        assert!(matches!(
            builder.block_multi(
                "broken",
                &[("state", state)],
                &["one", "two"],
                |graph, ids| Ok(vec![graph.relu(ids[0])?]),
            ),
            Err(ModelError::Neural(NeuralError::ContractViolation { .. }))
        ));
        assert_eq!(builder.graph_mut().nodes().len(), before);
        let next = builder.graph_mut().relu(state).unwrap();
        builder.next_state("state", next).unwrap();
        builder.output("out", next).unwrap();
        assert!(builder.finish().is_ok());
    }
}
