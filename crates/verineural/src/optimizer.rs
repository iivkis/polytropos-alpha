//! Architecture-neutral host optimizer over explicit named f32 parameter tensors.
//! GPU kernels and training graphs remain in VeriNeural; this CPU update is
//! a deliberately separate *trusted numerical* step, not a Lean certificate.
use std::collections::BTreeMap;
use std::fmt;

use crate::Tensor;

#[derive(Clone, Debug, PartialEq)]
pub enum OptimizerError {
    Invalid(&'static str),
    Parameter(String),
}
impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid AdamW configuration: {s}"),
            Self::Parameter(s) => write!(f, "invalid AdamW parameter: {s}"),
        }
    }
}
impl std::error::Error for OptimizerError {}

#[derive(Clone, Debug)]
struct Moments {
    first: Vec<f32>,
    second: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct AdamW {
    learning_rate: f32,
    weight_decay: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    steps: u64,
    states: BTreeMap<String, Moments>,
}

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Result<Self, OptimizerError> {
        if !learning_rate.is_finite()
            || learning_rate <= 0.
            || !weight_decay.is_finite()
            || weight_decay < 0.
        {
            return Err(OptimizerError::Invalid(
                "learning rate must be positive, weight decay nonnegative and finite",
            ));
        }
        Ok(Self {
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            steps: 0,
            states: BTreeMap::new(),
        })
    }
    pub fn steps(&self) -> u64 {
        self.steps
    }
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// One atomic step for an explicitly selected *exact* parameter set.
    /// Reject missing/extra gradients, mismatched shapes and non-finite values
    /// before mutating state or any parameter. No implicit decoupled weight
    /// decay on unselected/frozen model weights.
    pub fn step(
        &mut self,
        parameters: &mut BTreeMap<String, Tensor<f32>>,
        gradients: &BTreeMap<String, Tensor<f32>>,
    ) -> Result<(), OptimizerError> {
        if parameters.is_empty()
            || parameters.len() != gradients.len()
            || parameters.keys().ne(gradients.keys())
        {
            return Err(OptimizerError::Invalid(
                "parameters and gradients require matching nonempty names",
            ));
        }
        let next_step = self
            .steps
            .checked_add(1)
            .ok_or(OptimizerError::Invalid("optimizer step counter overflow"))?;
        let b1_correction = 1. - self.beta1.powf(next_step as f32);
        let b2_correction = 1. - self.beta2.powf(next_step as f32);
        if b1_correction <= 0. || b2_correction <= 0. {
            return Err(OptimizerError::Invalid("bias correction underflow"));
        }

        let mut new_params = BTreeMap::new();
        let mut new_states = self.states.clone();
        for (name, parameter) in parameters.iter() {
            let gradient = &gradients[name];
            if parameter.shape() != gradient.shape()
                || parameter
                    .values()
                    .iter()
                    .chain(gradient.values())
                    .any(|v| !v.is_finite())
            {
                return Err(OptimizerError::Parameter(format!(
                    "invalid shape or non-finite tensor: {name}"
                )));
            }
            let count = parameter.values().len();
            let state = new_states.entry(name.clone()).or_insert_with(|| Moments {
                first: vec![0.; count],
                second: vec![0.; count],
            });
            if state.first.len() != count || state.second.len() != count {
                return Err(OptimizerError::Parameter(format!(
                    "optimizer state shape changed: {name}"
                )));
            }
            let mut next = Vec::with_capacity(count);
            for (i, (&value, &gradient)) in
                parameter.values().iter().zip(gradient.values()).enumerate()
            {
                state.first[i] = self.beta1 * state.first[i] + (1. - self.beta1) * gradient;
                state.second[i] =
                    self.beta2 * state.second[i] + (1. - self.beta2) * gradient * gradient;
                let m_hat = state.first[i] / b1_correction;
                let v_hat = state.second[i] / b2_correction;
                let update = m_hat / (v_hat.sqrt() + self.epsilon);
                let next_value = value - self.learning_rate * (update + self.weight_decay * value);
                if !next_value.is_finite() {
                    return Err(OptimizerError::Parameter(format!(
                        "non-finite update: {name}"
                    )));
                }
                next.push(next_value);
            }
            new_params.insert(
                name.clone(),
                Tensor::new(parameter.shape().clone(), next)
                    .map_err(|_| OptimizerError::Parameter(format!("invalid tensor: {name}")))?,
            );
        }
        *parameters = new_params;
        self.states = new_states;
        self.steps = next_step;
        Ok(())
    }
}
