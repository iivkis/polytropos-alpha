//! Independent scalar streaming reference for EMA4-LM regression tests.
//! This is a host *oracle*, never an alternate CUDA/IR/backend implementation.
use super::ema4::{BlockWeights, Ema4Weights, EmaMode, HISTORY, LAYERS};
use crate::{WIDTH, reference_project};
use verineural::NeuralError;

#[derive(Clone, Debug)]
pub struct BlockState {
    pub memory: [f32; WIDTH],
    pub history1: [[f32; WIDTH]; HISTORY],
    pub history2: [[f32; WIDTH]; HISTORY],
}

impl Default for BlockState {
    fn default() -> Self {
        Self {
            memory: [0.; WIDTH],
            history1: [[0.; WIDTH]; HISTORY],
            history2: [[0.; WIDTH]; HISTORY],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ema4State(pub [BlockState; LAYERS]);

impl Default for Ema4State {
    fn default() -> Self {
        Self(std::array::from_fn(|_| BlockState::default()))
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn norm(x: &[f32; WIDTH], gamma: &[f32; WIDTH], beta: &[f32; WIDTH]) -> [f32; WIDTH] {
    let mean = x.iter().sum::<f32>() / WIDTH as f32;
    let variance = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / WIDTH as f32;
    let inv_std = 1. / (variance + 1e-5).sqrt();
    std::array::from_fn(|lane| (x[lane] - mean) * inv_std * gamma[lane] + beta[lane])
}

fn conv(
    x: &[f32; WIDTH],
    history: &mut [[f32; WIDTH]; HISTORY],
    filter: &[[f32; WIDTH]; HISTORY + 1],
    scale: f32,
) -> [f32; WIDTH] {
    let y = std::array::from_fn(|lane| {
        let mut sum = x[lane] * filter[HISTORY][lane];
        for tap in 0..HISTORY {
            sum += history[tap][lane] * filter[tap][lane];
        }
        x[lane] + scale * sum
    });
    history.copy_within(1..HISTORY, 0);
    history[HISTORY - 1] = *x;
    y
}

fn step_block(
    x: [f32; WIDTH],
    p: &BlockWeights,
    state: &mut BlockState,
    ema: EmaMode,
    residual: bool,
) -> Result<[f32; WIDTH], NeuralError> {
    let a = norm(&x, &p.norm1_gamma, &p.norm1_beta);
    let a = conv(&a, &mut state.history1, &p.depthwise1, p.conv_scale1);
    let gate = reference_project(&a, &p.operator, 0)?;
    let key = reference_project(&a, &p.operator, 1)?;
    let value = reference_project(&a, &p.operator, 2)?;
    for lane in 0..WIDTH {
        let decay_logit = p.decay_logit[lane]
            + if ema == EmaMode::Selective {
                a[lane] * p.decay_gate[lane]
            } else {
                0.
            };
        let decay = sigmoid(decay_logit);
        state.memory[lane] =
            decay * state.memory[lane] + (1. - decay) * sigmoid(key[lane]) * value[lane];
    }
    let gated: [f32; WIDTH] =
        std::array::from_fn(|lane| sigmoid(gate[lane] * p.gate_scale) * state.memory[lane]);
    let projected = reference_project(&gated, &p.operator, 3)?;
    let x = std::array::from_fn(|lane| {
        let update = 0.5 * projected[lane];
        x[lane] + if residual { update * p.residual_mem_scale } else { update }
    });

    let a = norm(&x, &p.norm2_gamma, &p.norm2_beta);
    let f = conv(&a, &mut state.history2, &p.depthwise2, p.conv_scale2);
    let projections = [
        reference_project(&f, &p.operator, 4)?,
        reference_project(&f, &p.operator, 5)?,
        reference_project(&f, &p.operator, 6)?,
        reference_project(&f, &p.operator, 7)?,
    ];
    let hidden: [f32; 2 * WIDTH] = std::array::from_fn(|i| {
        let (head, lane) = (i / WIDTH, i % WIDTH);
        let up = projections[head][lane];
        let gate = projections[head + 2][lane];
        up * sigmoid(up) * gate
    });
    let h0: [f32; WIDTH] = std::array::from_fn(|lane| hidden[lane]);
    let h1: [f32; WIDTH] = std::array::from_fn(|lane| hidden[WIDTH + lane]);
    let d0 = reference_project(&h0, &p.operator, 0)?;
    let d1 = reference_project(&h1, &p.operator, 1)?;
    Ok(std::array::from_fn(|lane| {
        let update = 0.5 * (d0[lane] + d1[lane]);
        x[lane] + if residual { update * p.residual_ff_scale } else { update }
    }))
}

/// One autoregressive symbol with tied embedding and no sequence cache growth.
pub fn reference_step(
    parameters: &Ema4Weights,
    token: usize,
    states: &mut Ema4State,
) -> Result<Vec<f32>, NeuralError> {
    reference_step_for_ema(parameters, token, states, EmaMode::Static)
}

/// Scalar independent reference for an input-conditioned EMA transition.
pub fn reference_step_for_ema(
    parameters: &Ema4Weights,
    token: usize,
    states: &mut Ema4State,
    ema: EmaMode,
) -> Result<Vec<f32>, NeuralError> {
    assert!(
        token < parameters.vocabulary
            && parameters.embedding.len() == parameters.vocabulary * WIDTH
    );
    let mut x: [f32; WIDTH] =
        std::array::from_fn(|lane| parameters.embedding[token * WIDTH + lane]);
    for layer in 0..LAYERS {
        x = step_block(x, &parameters.blocks[layer], &mut states.0[layer], ema, false)?;
    }
    let x = norm(&x, &parameters.final_gamma, &parameters.final_beta);
    Ok((0..parameters.vocabulary)
        .map(|word| {
            (0..WIDTH)
                .map(|lane| x[lane] * parameters.embedding[word * WIDTH + lane])
                .sum::<f32>()
        })
        .collect())
}

/// Scalar P2 state with an explicitly configured number of blocks.
#[derive(Clone, Debug)]
pub struct DepthState(pub Vec<BlockState>);

impl DepthState {
    pub fn new(depth: usize) -> Self {
        Self((0..depth).map(|_| BlockState::default()).collect())
    }
}

/// Independent streaming scalar oracle for bounded-depth P2.
pub fn reference_step_configuration(
    weights: &Ema4Weights,
    token: usize,
    states: &mut DepthState,
    ema: EmaMode,
    residual: bool,
) -> Result<Vec<f32>, NeuralError> {
    assert!(
        token < weights.vocabulary
            && states.0.len() == weights.layers()
            && weights.embedding.len() == weights.vocabulary * WIDTH
    );
    let mut x: [f32; WIDTH] =
        std::array::from_fn(|lane| weights.embedding[token * WIDTH + lane]);
    for (block, state) in weights.all_blocks().zip(states.0.iter_mut()) {
        x = step_block(x, block, state, ema, residual)?;
    }
    let x = norm(&x, &weights.final_gamma, &weights.final_beta);
    Ok((0..weights.vocabulary)
        .map(|word| {
            (0..WIDTH)
                .map(|lane| x[lane] * weights.embedding[word * WIDTH + lane])
                .sum::<f32>()
        })
        .collect())
}
