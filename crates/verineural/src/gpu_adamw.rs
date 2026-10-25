//! Architecture-neutral AdamW assembled from the public Semantic Neural IR.
//!
//! All tensor math remains in the same CUDA Graph as the consumer's backward
//! graph. Parameters and moments are fed back D2D; the CPU uploads only two
//! step-dependent bias-correction scalars and reads state for checkpoints.
//! This GPU numerical path is experimental and is not Lean/FP32 certified.
use std::collections::BTreeMap;

use crate::{
    AdamW, AdamWCheckpoint, Domain, GpuError, GpuExecution, NeuralGraph, NodeId, OptimizerError,
    Shape, Tensor,
};

/// A reusable, opt-in graph extension, not an EMA4-specific optimizer kernel.
/// All output/source handles belong to the supplied Semantic Neural IR.
#[derive(Clone, Debug)]
pub struct GpuAdamWPlan {
    bindings: Vec<(NodeId, Tensor<f32>)>,
    feedback: Vec<(NodeId, NodeId)>,
    parameters: BTreeMap<String, NodeId>,
    moments: BTreeMap<String, (NodeId, NodeId)>,
    corrections: (NodeId, NodeId),
    config: AdamWCheckpoint,
}

impl GpuAdamWPlan {
    /// Append one AdamW update for every named parameter and its gradient.
    ///
    /// Inputs are exact named inventories, not all graph inputs (fixed model
    /// constants and recurrent memory must never receive weight decay).
    /// Uses separate pointwise operations to retain f32 intermediate rounding.
    pub fn append(
        graph: &mut NeuralGraph,
        pairs: &BTreeMap<String, (NodeId, NodeId)>,
        initial: &BTreeMap<String, Tensor<f32>>,
        checkpoint: &AdamWCheckpoint,
    ) -> Result<Self, OptimizerError> {
        // A failed optimizer extension must not leave an incomplete CUDA/
        // proof graph behind for the consumer to accidentally compile.
        let mut proposed = graph.clone();
        let plan = Self::append_inner(&mut proposed, pairs, initial, checkpoint)?;
        *graph = proposed;
        Ok(plan)
    }

    fn append_inner(
        graph: &mut NeuralGraph,
        pairs: &BTreeMap<String, (NodeId, NodeId)>,
        initial: &BTreeMap<String, Tensor<f32>>,
        checkpoint: &AdamWCheckpoint,
    ) -> Result<Self, OptimizerError> {
        AdamW::from_checkpoint(checkpoint.clone(), initial)?;
        if pairs.is_empty() || pairs.keys().ne(initial.keys()) {
            return Err(OptimizerError::Invalid(
                "GPU AdamW requires the exact nonempty parameter inventory",
            ));
        }
        let scalar_shape =
            Shape::new(vec![]).map_err(|e| OptimizerError::Parameter(e.to_string()))?;
        let constants = [
            checkpoint.beta1,
            1. - checkpoint.beta1,
            checkpoint.beta2,
            1. - checkpoint.beta2,
            checkpoint.learning_rate,
            checkpoint.weight_decay,
            checkpoint.epsilon,
            1.,
            1.,
        ];
        let mut bindings = Vec::new();
        let scalar_nodes = constants
            .iter()
            .map(|&value| {
                let id = graph.input(scalar_shape.clone(), Domain::Finite);
                bindings.push((
                    id,
                    Tensor::from_slice(vec![], &[value])
                        .map_err(|e| OptimizerError::Parameter(e.to_string()))?,
                ));
                Ok(id)
            })
            .collect::<Result<Vec<_>, OptimizerError>>()?;
        let corrections = (scalar_nodes[7], scalar_nodes[8]);
        let mut broadcasts = BTreeMap::<(Vec<usize>, usize), NodeId>::new();
        let mut feedback = Vec::with_capacity(pairs.len() * 3);
        let mut parameters = BTreeMap::new();
        let mut moments = BTreeMap::new();
        for (name, &(parameter, gradient)) in pairs {
            let p = graph
                .node(parameter)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            let shape = p.shape.clone();
            let grad_shape = graph
                .node(gradient)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?
                .shape
                .clone();
            if grad_shape != shape || initial[name].shape() != &shape {
                return Err(OptimizerError::Parameter(format!(
                    "GPU AdamW parameter/gradient shape mismatch: {name}"
                )));
            }
            let (saved_first, saved_second) = if checkpoint.steps == 0 {
                let size = shape.element_count();
                (vec![0.; size], vec![0.; size])
            } else {
                checkpoint.moments[name].clone()
            };
            // FP32 operations can overflow, so the next moment's IR
            // domain cannot be strengthened to Finite without a proof.
            // Checkpoint export validates both moments before persistence.
            let m = graph.input(shape.clone(), Domain::Any);
            let v = graph.input(shape.clone(), Domain::Any);
            bindings.push((
                m,
                Tensor::new(shape.clone(), saved_first)
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))?,
            ));
            bindings.push((
                v,
                Tensor::new(shape.clone(), saved_second)
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))?,
            ));
            let mut scalar =
                |index: usize, graph: &mut NeuralGraph| -> Result<NodeId, OptimizerError> {
                    let key = (shape.dims().to_vec(), index);
                    if let Some(&id) = broadcasts.get(&key) {
                        return Ok(id);
                    }
                    let id = graph
                        .broadcast_to(scalar_nodes[index], shape.clone())
                        .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
                    broadcasts.insert(key, id);
                    Ok(id)
                };
            let b1 = scalar(0, graph)?;
            let b1_rest = scalar(1, graph)?;
            let b2 = scalar(2, graph)?;
            let b2_rest = scalar(3, graph)?;
            let lr = scalar(4, graph)?;
            let decay = scalar(5, graph)?;
            let epsilon = scalar(6, graph)?;
            let c1 = scalar(7, graph)?;
            let c2 = scalar(8, graph)?;
            let mul = |g: &mut NeuralGraph, a, b| {
                g.mul(a, b)
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))
            };
            let add = |g: &mut NeuralGraph, a, b| {
                g.add(a, b)
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))
            };
            let step_m_left = mul(graph, b1, m)?;
            let step_m_right = mul(graph, b1_rest, gradient)?;
            let step_m = add(graph, step_m_left, step_m_right)?;
            let step_v_left = mul(graph, b2, v)?;
            // Host AdamW evaluates (1-beta2)*g*g in that order.
            let step_v_right_first = mul(graph, b2_rest, gradient)?;
            let step_v_right = mul(graph, step_v_right_first, gradient)?;
            let step_v = add(graph, step_v_left, step_v_right)?;
            let m_hat = graph
                .div(step_m, c1)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            let v_hat = graph
                .div(step_v, c2)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            let sqrt_v = graph
                .sqrt_ieee(v_hat)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            let denom = add(graph, sqrt_v, epsilon)?;
            let update = graph
                .div(m_hat, denom)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            let wd = mul(graph, decay, parameter)?;
            let total_update = add(graph, update, wd)?;
            let scaled = mul(graph, lr, total_update)?;
            let next = graph
                .sub(parameter, scaled)
                .map_err(|e| OptimizerError::Parameter(e.to_string()))?;
            feedback.extend([(parameter, next), (m, step_m), (v, step_v)]);
            parameters.insert(name.clone(), next);
            moments.insert(name.clone(), (step_m, step_v));
        }
        Ok(Self {
            bindings,
            feedback,
            parameters,
            moments,
            corrections,
            config: checkpoint.clone(),
        })
    }

    pub fn bindings(&self) -> &[(NodeId, Tensor<f32>)] {
        &self.bindings
    }

    pub fn feedback(&self) -> &[(NodeId, NodeId)] {
        &self.feedback
    }

    pub fn outputs(&self) -> Vec<NodeId> {
        let mut ids = self.parameters.values().copied().collect::<Vec<_>>();
        for &(first, second) in self.moments.values() {
            ids.extend([first, second]);
        }
        ids
    }

    /// The CPU computes only the two bias corrections, using the same powf
    /// expression as host AdamW. No GPU tensor or moment is read each step.
    pub fn correction_inputs(
        &self,
        next_step: u64,
    ) -> Result<[(NodeId, Tensor<f32>); 2], OptimizerError> {
        let b1 = 1. - self.config.beta1.powf(next_step as f32);
        let b2 = 1. - self.config.beta2.powf(next_step as f32);
        if !b1.is_finite() || !b2.is_finite() || b1 <= 0. || b2 <= 0. {
            return Err(OptimizerError::Invalid(
                "GPU AdamW bias correction underflow",
            ));
        }
        Ok([
            (
                self.corrections.0,
                Tensor::from_slice(vec![], &[b1])
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))?,
            ),
            (
                self.corrections.1,
                Tensor::from_slice(vec![], &[b2])
                    .map_err(|e| OptimizerError::Parameter(e.to_string()))?,
            ),
        ])
    }

    /// Read final updated parameters and moments as one batch at checkpoint
    /// boundaries. CUDA Graph replay and D2D feedback do not change ownership
    /// of these selected outputs.
    pub fn checkpoint(
        &self,
        gpu: &mut GpuExecution,
        steps: u64,
    ) -> Result<(BTreeMap<String, Tensor<f32>>, AdamWCheckpoint), GpuError> {
        let outputs = self.outputs();
        let read = gpu.read_outputs_selected(&outputs)?;
        let mut values = read
            .into_iter()
            .map(|(id, value)| (id.index(), value))
            .collect::<BTreeMap<_, _>>();
        let mut params = BTreeMap::new();
        let mut moments = BTreeMap::new();
        for (name, &id) in &self.parameters {
            let value = values
                .remove(&id.index())
                .expect("selected parameter update");
            if value.values().iter().any(|v| !v.is_finite()) {
                return Err(GpuError::InvalidInput {
                    node: id.index(),
                    reason: "GPU AdamW generated non-finite parameter",
                });
            }
            params.insert(name.clone(), value);
        }
        for (name, &(first, second)) in &self.moments {
            let a = values
                .remove(&first.index())
                .expect("selected first moment");
            let b = values
                .remove(&second.index())
                .expect("selected second moment");
            if a.values().iter().chain(b.values()).any(|v| !v.is_finite())
                || b.values().iter().any(|&v| v < 0.)
            {
                return Err(GpuError::InvalidInput {
                    node: first.index(),
                    reason: "GPU AdamW generated invalid optimizer moments",
                });
            }
            moments.insert(name.clone(), (a.values().to_vec(), b.values().to_vec()));
        }
        let mut saved = self.config.clone();
        saved.steps = steps;
        saved.moments = moments;
        Ok((params, saved))
    }
}
