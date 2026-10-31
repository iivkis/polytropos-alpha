//! Four-layer, streaming EMA4-LM consumer built from VeriNeural primitives.
//!
//! Dataflow: one-hot token -> tied embedding -> 4 x (LayerNorm, Conv9,
//! 8-role Torus16-Mix/EMA/SwiGLU) -> final LayerNorm -> tied logits.
//! All temporal memory is explicit ModelBuilder state/next_state.
//! There are no lab CUDA kernels, opaque forward callbacks or extra IR.
use crate::{Parameters, ROLES, WIDTH, projection_graph, projection_graph_with_delta};
use verineural::{
    Domain, Model, ModelBuilder, ModelError, NeuralError, NeuralGraph, NodeId, Shape, Tensor,
};

pub const LAYERS: usize = 4;
pub const KERNEL: usize = 9;
pub const HISTORY: usize = KERNEL - 1;
pub const RESEARCH_VOCAB: usize = 104;

/// Controlled Torus16-Mix weight tying experiment. Both variants keep the
/// same role-specific rotations, gates, scales, biases and recurrent LM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixMode {
    /// One independent 48x48 matrix for every role within each of four blocks.
    Independent,
    /// One 48x48 matrix shared by all eight roles in each block (Torus16-Mix).
    TorusShared,
    /// Shared W plus role-specific rank-2 adapter.
    RoleDelta2,
    /// Shared W plus role-specific rank-4 adapter.
    RoleDelta4,
    /// Shared W plus role-specific rank-8 adapter.
    RoleDelta8,
}

impl MatrixMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::TorusShared => "torus_shared",
            Self::RoleDelta2 => "role_delta_2",
            Self::RoleDelta4 => "role_delta_4",
            Self::RoleDelta8 => "role_delta_8",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "independent" => Some(Self::Independent),
            "torus_shared" => Some(Self::TorusShared),
            "role_delta_2" => Some(Self::RoleDelta2),
            "role_delta" | "role_delta_4" => Some(Self::RoleDelta4),
            "role_delta_8" => Some(Self::RoleDelta8),
            _ => None,
        }
    }

    pub const fn matrix_count(self) -> usize {
        match self {
            Self::Independent => LAYERS * ROLES,
            Self::TorusShared | Self::RoleDelta2 | Self::RoleDelta4 | Self::RoleDelta8 => LAYERS,
        }
    }

    pub const fn role_delta_rank(self) -> Option<usize> {
        match self {
            Self::RoleDelta2 => Some(2),
            Self::RoleDelta4 => Some(4),
            Self::RoleDelta8 => Some(8),
            Self::Independent | Self::TorusShared => None,
        }
    }

    pub const fn adapter_parameter_count(self) -> usize {
        match self.role_delta_rank() {
            Some(rank) => LAYERS * ROLES * 2 * WIDTH * rank,
            None => 0,
        }
    }
}

/// EMA memory update policy. Static preserves the historical Semantic IR
/// and checkpoint identities; Selective introduces input-dependent decay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmaMode {
    Static,
    Selective,
}

impl EmaMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Selective => "selective",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "static" => Some(Self::Static),
            "selective" => Some(Self::Selective),
            _ => None,
        }
    }
}

/// Initialization policy for independent role matrices. The matched mode
/// isolates weight tying; the diverse mode is the unconstrained baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoleInit {
    Matched,
    Diverse,
}

impl RoleInit {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "matched" => Some(Self::Matched),
            "diverse" => Some(Self::Diverse),
            _ => None,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Matched => "matched",
            Self::Diverse => "diverse",
        }
    }
}

// An integer-only, reproducible sampler; neither backend GPU RNG nor global
// PRNG state affects the experiment's initial FP32 tensor values.
fn matrix_sample(seed: u64, layer: usize, role: usize, index: usize) -> f32 {
    let mut x = seed
        ^ ((layer as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15))
        ^ ((role as u64 + 1).wrapping_mul(0xbf58476d1ce4e5b9))
        ^ ((index as u64 + 1).wrapping_mul(0x94d049bb133111eb));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.04
}

#[derive(Clone, Debug)]
pub struct BlockWeights {
    pub operator: Parameters,
    pub norm1_gamma: [f32; WIDTH],
    pub norm1_beta: [f32; WIDTH],
    pub norm2_gamma: [f32; WIDTH],
    pub norm2_beta: [f32; WIDTH],
    pub decay_logit: [f32; WIDTH],
    /// Selective decay strength. Zero initializes exactly at static EMA.
    pub decay_gate: [f32; WIDTH],
    pub residual_mem_scale: f32,
    pub residual_ff_scale: f32,
    pub gate_scale: f32,
    pub depthwise1: [[f32; WIDTH]; KERNEL],
    pub depthwise2: [[f32; WIDTH]; KERNEL],
    pub conv_scale1: f32,
    pub conv_scale2: f32,
}

impl Default for BlockWeights {
    fn default() -> Self {
        Self {
            operator: Parameters::default(),
            norm1_gamma: [1.0; WIDTH],
            norm1_beta: [0.0; WIDTH],
            norm2_gamma: [1.0; WIDTH],
            norm2_beta: [0.0; WIDTH],
            decay_logit: std::array::from_fn(|i| 2.7 - i as f32 * (4.2 / (WIDTH - 1) as f32)),
            decay_gate: [0.; WIDTH],
            residual_mem_scale: 1.,
            residual_ff_scale: 1.,
            gate_scale: 1.0,
            depthwise1: std::array::from_fn(|tap| {
                std::array::from_fn(|lane| ((tap * 13 + lane * 3) % 17) as f32 * 0.002 - 0.016)
            }),
            depthwise2: std::array::from_fn(|tap| {
                std::array::from_fn(|lane| ((tap * 7 + lane * 11) % 19) as f32 * 0.002 - 0.018)
            }),
            conv_scale1: 0.15,
            conv_scale2: 0.15,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ema4Weights {
    pub vocabulary: usize,
    pub matrix_seed: u64,
    /// Fixed lambda/q multiplier of each Role-Delta correction (not trained).
    pub delta_scale: f32,
    /// Tied input/output embedding [vocabulary, 48].
    pub embedding: Vec<f32>,
    pub blocks: [BlockWeights; LAYERS],
    pub extra_blocks: Vec<BlockWeights>,
    pub final_gamma: [f32; WIDTH],
    pub final_beta: [f32; WIDTH],
}

impl Ema4Weights {
    pub fn deterministic(vocabulary: usize) -> Result<Self, ModelError> {
        Self::deterministic_seeded(vocabulary, 0)
    }

    pub fn deterministic_seeded(vocabulary: usize, seed: u64) -> Result<Self, ModelError> {
        if vocabulary == 0 || vocabulary > 4096 {
            return Err(ModelError::Invalid("vocabulary must be 1..=4096".into()));
        }
        let mut result = Self {
            vocabulary,
            matrix_seed: seed,
            delta_scale: 1.0,
            embedding: (0..vocabulary * WIDTH)
                .map(|i| ((i * 19 + 7) % 37) as f32 * 0.0011 - 0.020)
                .collect(),
            blocks: std::array::from_fn(|_| BlockWeights::default()),
            extra_blocks: Vec::new(),
            final_gamma: [1.; WIDTH],
            final_beta: [0.; WIDTH],
        };
        if seed != 0 {
            for (layer, block) in result.blocks.iter_mut().enumerate() {
                for (row, values) in block.operator.weight.iter_mut().enumerate() {
                    for (col, value) in values.iter_mut().enumerate() {
                        *value = matrix_sample(seed, layer, 0, row * WIDTH + col);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Opt-in P2 depth extension. The historical four blocks remain intact.
    pub fn with_depth(mut self, depth: usize) -> Result<Self, ModelError> {
        if !matches!(depth, 4 | 6 | 8) {
            return Err(ModelError::Invalid("P2 depth must be 4, 6 or 8".into()));
        }
        self.extra_blocks.clear();
        for layer in LAYERS..depth {
            let mut block = BlockWeights {
                residual_mem_scale: 0.,
                residual_ff_scale: 0.,
                ..BlockWeights::default()
            };
            if self.matrix_seed != 0 {
                for (row, values) in block.operator.weight.iter_mut().enumerate() {
                    for (col, value) in values.iter_mut().enumerate() {
                        *value = matrix_sample(self.matrix_seed, layer, 0, row * WIDTH + col);
                    }
                }
            }
            self.extra_blocks.push(block);
        }
        Ok(self)
    }

    pub fn layers(&self) -> usize { LAYERS + self.extra_blocks.len() }

    pub fn all_blocks(&self) -> impl Iterator<Item = &BlockWeights> {
        self.blocks.iter().chain(self.extra_blocks.iter())
    }
    pub const fn parameter_count(vocabulary: usize) -> usize {
        // Eight physical projection roles share one W per block. The head is
        // tied to the embedding rather than a second trainable matrix.
        vocabulary * WIDTH
            + LAYERS
                * (Parameters::trainable_count() + 4 * WIDTH + WIDTH + 1 + 2 * KERNEL * WIDTH + 2)
            + 2 * WIDTH
    }

    /// Matrix count is the only physical parameter-count difference.
    pub const fn parameter_count_for(vocabulary: usize, mode: MatrixMode) -> usize {
        Self::parameter_count(vocabulary) - LAYERS * WIDTH * WIDTH
            + mode.matrix_count() * WIDTH * WIDTH
            + mode.adapter_parameter_count()
    }

    /// Extended parameter inventory: one [1,48] gate per layer only for
    /// selective EMA. It must match the model's actual declared parameters.
    pub const fn parameter_count_for_ema(
        vocabulary: usize,
        mode: MatrixMode,
        ema: EmaMode,
    ) -> usize {
        Self::parameter_count_for(vocabulary, mode)
            + if matches!(ema, EmaMode::Selective) { LAYERS * WIDTH } else { 0 }
    }

    pub const fn parameter_count_for_configuration(
        vocabulary: usize, mode: MatrixMode, ema: EmaMode, depth: usize, residual: bool,
    ) -> usize {
        let base = Self::parameter_count_for_ema(vocabulary, mode, ema);
        let shared = vocabulary * WIDTH + 2 * WIDTH;
        let per_layer = (base - shared) / LAYERS;
        shared + per_layer * depth + if residual { 2 * depth } else { 0 }
    }

    /// Preserve the existing shared-matrix consumer API.
    pub fn trainable_parameters(
        &self,
    ) -> Result<std::collections::BTreeMap<String, Tensor<f32>>, ModelError> {
        self.trainable_parameters_for(MatrixMode::TorusShared)
    }

    /// Parameters for either ablation. All independent role matrices are
    /// initialized from the same W as the shared model so that the initial
    /// forward function is equal, including strict FP32 operation order.
    pub fn trainable_parameters_for(
        &self,
        mode: MatrixMode,
    ) -> Result<std::collections::BTreeMap<String, Tensor<f32>>, ModelError> {
        self.trainable_parameters_with_init(mode, RoleInit::Matched)
    }

    pub fn trainable_parameters_with_init(
        &self,
        mode: MatrixMode,
        role_init: RoleInit,
    ) -> Result<std::collections::BTreeMap<String, Tensor<f32>>, ModelError> {
        self.trainable_parameters_with_init_and_ema(mode, role_init, EmaMode::Static)
    }

    pub fn trainable_parameters_with_init_and_ema(
        &self,
        mode: MatrixMode,
        role_init: RoleInit,
        ema: EmaMode,
    ) -> Result<std::collections::BTreeMap<String, Tensor<f32>>, ModelError> {
        self.trainable_parameters_with_configuration(mode, role_init, ema, false)
    }

    pub fn trainable_parameters_with_configuration(
        &self,
        mode: MatrixMode,
        role_init: RoleInit,
        ema: EmaMode,
        residual: bool,
    ) -> Result<std::collections::BTreeMap<String, Tensor<f32>>, ModelError> {
        let fixed = [
            "ones_column",
            "inv_width",
            "inv_sqrt_width",
            "one_scalar",
            "epsilon_scalar",
            "one_vec",
            "minus_vec",
            "minus_scalar",
            "half_vec",
            "one_up",
            "minus_up",
            "delta_scale",
        ];
        let parameters = self
            .bindings_with_configuration(0, mode, role_init, ema, residual)?
            .into_iter()
            .filter(|(name, _)| {
                name != "token"
                    && !name.starts_with("history_")
                    && !name.starts_with("mem_")
                    && !fixed.contains(&name.as_str())
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let elements: usize = parameters.values().map(|t| t.values().len()).sum();
        if elements != Self::parameter_count_for_configuration(
            self.vocabulary, mode, ema, self.layers(), residual
        ) {
            return Err(ModelError::Invalid(format!(
                "trainable parameter inventory mismatch: {elements}"
            )));
        }
        Ok(parameters)
    }

    /// Named weights, fixed numeric constants and the zero initial recurrent
    /// state. This is setup data, not a CPU implementation of the architecture.
    pub fn bindings(&self, token: usize) -> Result<Vec<(String, Tensor<f32>)>, ModelError> {
        self.bindings_for(token, MatrixMode::TorusShared)
    }

    pub fn bindings_for(
        &self,
        token: usize,
        mode: MatrixMode,
    ) -> Result<Vec<(String, Tensor<f32>)>, ModelError> {
        self.bindings_with_init(token, mode, RoleInit::Matched)
    }

    pub fn bindings_with_init(
        &self,
        token: usize,
        mode: MatrixMode,
        role_init: RoleInit,
    ) -> Result<Vec<(String, Tensor<f32>)>, ModelError> {
        self.bindings_with_init_and_ema(token, mode, role_init, EmaMode::Static)
    }

    pub fn bindings_with_init_and_ema(
        &self,
        token: usize,
        mode: MatrixMode,
        role_init: RoleInit,
        ema: EmaMode,
    ) -> Result<Vec<(String, Tensor<f32>)>, ModelError> {
        self.bindings_with_configuration(token, mode, role_init, ema, false)
    }

    pub fn bindings_with_configuration(
        &self,
        token: usize,
        mode: MatrixMode,
        role_init: RoleInit,
        ema: EmaMode,
        residual: bool,
    ) -> Result<Vec<(String, Tensor<f32>)>, ModelError> {
        if token >= self.vocabulary || self.embedding.len() != self.vocabulary * WIDTH {
            return Err(ModelError::Invalid(
                "token outside vocabulary or bad embedding".into(),
            ));
        }
        let mut result = Vec::new();
        let mut token_onehot = vec![0.; self.vocabulary];
        token_onehot[token] = 1.;
        insert(&mut result, "token", vec![1, self.vocabulary], token_onehot)?;
        insert(
            &mut result,
            "embedding",
            vec![self.vocabulary, WIDTH],
            self.embedding.clone(),
        )?;
        insert(&mut result, "ones_column", vec![WIDTH, 1], vec![1.; WIDTH])?;
        insert(
            &mut result,
            "inv_width",
            vec![1, 1],
            vec![1. / (WIDTH as f32)],
        )?;
        insert(
            &mut result,
            "inv_sqrt_width",
            vec![1, 1],
            vec![1. / (WIDTH as f32).sqrt()],
        )?;
        insert(&mut result, "one_scalar", vec![1, 1], vec![1.])?;
        insert(&mut result, "epsilon_scalar", vec![1, 1], vec![1e-5])?;
        insert(&mut result, "one_vec", vec![1, WIDTH], vec![1.; WIDTH])?;
        insert(&mut result, "minus_vec", vec![1, WIDTH], vec![-1.; WIDTH])?;
        insert(&mut result, "minus_scalar", vec![1, 1], vec![-1.])?;
        insert(&mut result, "half_vec", vec![1, WIDTH], vec![0.5; WIDTH])?;
        insert(
            &mut result,
            "one_up",
            vec![1, 2 * WIDTH],
            vec![1.; 2 * WIDTH],
        )?;
        insert(
            &mut result,
            "minus_up",
            vec![1, 2 * WIDTH],
            vec![-1.; 2 * WIDTH],
        )?;
        insert(
            &mut result,
            "final_gamma",
            vec![1, WIDTH],
            self.final_gamma.to_vec(),
        )?;
        insert(
            &mut result,
            "final_beta",
            vec![1, WIDTH],
            self.final_beta.to_vec(),
        )?;
        if mode.role_delta_rank().is_some() {
            if !self.delta_scale.is_finite() || self.delta_scale <= 0.0 {
                return Err(ModelError::Invalid(
                    "Role-Delta scale must be positive and finite".into(),
                ));
            }
            insert(
                &mut result,
                "delta_scale",
                vec![1, 1],
                vec![self.delta_scale],
            )?;
        }
        for (layer, b) in self.all_blocks().enumerate() {
            for role_weight in 0..(if mode == MatrixMode::Independent {
                ROLES
            } else {
                1
            }) {
                let name = if mode == MatrixMode::Independent {
                    format!("weight_{layer}_{role_weight}")
                } else {
                    format!("weight_{layer}")
                };
                insert(
                    &mut result,
                    &name,
                    vec![WIDTH, WIDTH],
                    if mode == MatrixMode::Independent && role_init == RoleInit::Diverse {
                        (0..WIDTH * WIDTH)
                            .map(|index| {
                                matrix_sample(
                                    self.matrix_seed ^ 0x544f52555331364d,
                                    layer,
                                    role_weight,
                                    index,
                                )
                            })
                            .collect()
                    } else {
                        b.operator.weight.iter().flatten().copied().collect()
                    },
                )?;
            }
            for role in 0..ROLES {
                let p = &b.operator.roles[role];
                for (label, values) in [
                    ("row", &p.row),
                    ("col", &p.col),
                    ("bias", &p.bias),
                    ("mix_gate", &p.mix_gate),
                ] {
                    insert(
                        &mut result,
                        &format!("{label}_{layer}_{role}"),
                        vec![1, WIDTH],
                        values.to_vec(),
                    )?;
                }
                insert(
                    &mut result,
                    &format!("mix_bias_{layer}_{role}"),
                    vec![1, 1],
                    vec![p.mix_bias],
                )?;
                if let Some(rank) = mode.role_delta_rank() {
                    insert(
                        &mut result,
                        &format!("delta_a_{layer}_{role}"),
                        vec![WIDTH, rank],
                        (0..WIDTH * rank)
                            .map(|index| {
                                matrix_sample(
                                    self.matrix_seed ^ 0x524f4c455f444c54,
                                    layer,
                                    role,
                                    index,
                                )
                            })
                            .collect(),
                    )?;
                    insert(
                        &mut result,
                        &format!("delta_b_{layer}_{role}"),
                        vec![rank, WIDTH],
                        vec![0.0; rank * WIDTH],
                    )?;
                }
            }
            for (label, values) in [
                ("norm1_gamma", &b.norm1_gamma),
                ("norm1_beta", &b.norm1_beta),
                ("norm2_gamma", &b.norm2_gamma),
                ("norm2_beta", &b.norm2_beta),
                ("decay_logit", &b.decay_logit),
            ] {
                insert(
                    &mut result,
                    &format!("{label}_{layer}"),
                    vec![1, WIDTH],
                    values.to_vec(),
                )?;
            }
            if ema == EmaMode::Selective {
                insert(
                    &mut result,
                    &format!("decay_gate_{layer}"),
                    vec![1, WIDTH],
                    b.decay_gate.to_vec(),
                )?;
            }
            if residual {
                for (name, value) in [
                    ("residual_mem_scale", b.residual_mem_scale),
                    ("residual_ff_scale", b.residual_ff_scale),
                ] {
                    insert(&mut result, &format!("{name}_{layer}"), vec![1, 1], vec![value])?;
                }
            }
            for (label, value) in [
                ("gate_scale", b.gate_scale),
                ("conv_scale_1", b.conv_scale1),
                ("conv_scale_2", b.conv_scale2),
            ] {
                insert(
                    &mut result,
                    &format!("{label}_{layer}"),
                    vec![1, 1],
                    vec![value],
                )?;
            }
            for stage in 1..=2 {
                for tap in 0..KERNEL {
                    insert(
                        &mut result,
                        &format!("dw_{layer}_{stage}_{tap}"),
                        vec![1, WIDTH],
                        (if stage == 1 {
                            &b.depthwise1
                        } else {
                            &b.depthwise2
                        })[tap]
                            .to_vec(),
                    )?;
                }
                for offset in 0..HISTORY {
                    insert(
                        &mut result,
                        &history_name(layer, stage, offset),
                        vec![1, WIDTH],
                        vec![0.; WIDTH],
                    )?;
                }
            }
            insert(
                &mut result,
                &format!("mem_{layer}"),
                vec![1, WIDTH],
                vec![0.; WIDTH],
            )?;
        }
        Ok(result)
    }
}

fn insert(
    values: &mut Vec<(String, Tensor<f32>)>,
    name: &str,
    shape: Vec<usize>,
    data: Vec<f32>,
) -> Result<(), ModelError> {
    values.push((name.to_owned(), Tensor::from_vec(shape, data)?));
    Ok(())
}

fn history_name(layer: usize, stage: usize, offset: usize) -> String {
    format!("history_{layer}_{stage}_{offset}")
}

#[derive(Clone, Copy)]
struct NormPorts {
    ones: NodeId,
    inv_width: NodeId,
    one: NodeId,
    epsilon: NodeId,
}

fn layer_norm(
    g: &mut NeuralGraph,
    x: NodeId,
    gamma: NodeId,
    beta: NodeId,
    ports: NormPorts,
) -> Result<NodeId, NeuralError> {
    let sum = g.matmul(x, ports.ones)?;
    let mean = g.mul(sum, ports.inv_width)?;
    let mean = g.broadcast_to(mean, Shape::new(vec![1, WIDTH])?)?;
    let centered = g.sub(x, mean)?;
    let squared = g.mul(centered, centered)?;
    let var = g.matmul(squared, ports.ones)?;
    let var = g.mul(var, ports.inv_width)?;
    let denominator = g.add(var, ports.epsilon)?;
    // The versioned pointwise sqrt permits IEEE NaN on invalid arguments;
    // no undocumented positivity assumption is inserted in the graph.
    let sigma = g.sqrt_ieee(denominator)?;
    let inv_sigma = g.div(ports.one, sigma)?;
    let inv_sigma = g.broadcast_to(inv_sigma, Shape::new(vec![1, WIDTH])?)?;
    let normalized = g.mul(centered, inv_sigma)?;
    let scaled = g.mul(normalized, gamma)?;
    g.add(scaled, beta)
}

fn sigmoid(
    g: &mut NeuralGraph,
    x: NodeId,
    one: NodeId,
    minus: NodeId,
) -> Result<NodeId, NeuralError> {
    let negative = g.mul(x, minus)?;
    let e = g.exp(negative)?;
    let denom = g.add(one, e)?;
    g.div(one, denom)
}

struct RoleNodes {
    weight: NodeId,
    transposed: NodeId,
    row: NodeId,
    col: NodeId,
    bias: NodeId,
    gate: NodeId,
    gate_bias: NodeId,
    delta: Option<(NodeId, NodeId)>,
}

struct ProjectionPorts {
    roles: Vec<RoleNodes>,
    ones_column: NodeId,
    one_scalar: NodeId,
    minus_scalar: NodeId,
    inv_sqrt: NodeId,
    delta_scale: Option<NodeId>,
}

fn project(
    model: &mut ModelBuilder,
    x: NodeId,
    role: usize,
    p: &ProjectionPorts,
) -> Result<NodeId, ModelError> {
    let ports = &p.roles[role];
    let w = if role.is_multiple_of(2) {
        ports.transposed
    } else {
        ports.weight
    };
    let mut inputs = vec![
        x,
        w,
        ports.row,
        ports.col,
        ports.bias,
        ports.gate,
        ports.gate_bias,
        p.ones_column,
        p.one_scalar,
        p.minus_scalar,
        p.inv_sqrt,
    ];
    let with_delta = ports.delta.is_some();
    if let Some((a, b)) = ports.delta {
        inputs.extend([a, b, p.delta_scale.expect("Role-Delta scalar exists")]);
    }
    model.block(
        if with_delta {
            "torus16_role_delta_projection"
        } else {
            "torus16_mix_projection"
        },
        &inputs,
        |graph, inputs| {
            if with_delta {
                projection_graph_with_delta(graph, inputs, role)
            } else {
                projection_graph(graph, inputs, role)
            }
        },
    )
}

fn conv_step(
    model: &mut ModelBuilder,
    x: NodeId,
    layer: usize,
    stage: usize,
    scale: NodeId,
) -> Result<NodeId, ModelError> {
    let mut history = Vec::with_capacity(HISTORY);
    for offset in 0..HISTORY {
        let name = history_name(layer, stage, offset);
        let id = model.state(name, vec![1, WIDTH], Domain::Any)?;
        history.push(id);
    }
    let mut conv = None;
    // The last convolution tap consumes the current token; the preceding
    // HISTORY taps consume their matching recurrent history entries.
    // HISTORY = KERNEL - 1, so this iterator contains exactly KERNEL taps.
    for (tap, &source) in history.iter().chain(std::iter::once(&x)).enumerate() {
        let weight = model.parameter(
            format!("dw_{layer}_{stage}_{tap}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let term = model.graph_mut().mul(source, weight)?;
        conv = Some(if let Some(sum) = conv {
            model.graph_mut().add(sum, term)?
        } else {
            term
        });
    }
    for offset in 0..HISTORY {
        // ModelExecution rejects overlapping D2D state feedback sources.
        // Materialize each predecessor through a general full-row SliceAxis:
        // the new value is a distinct buffer, while its mathematics stays
        // the identity. No host copy, special state kernel or unsafe alias.
        let next = if offset + 1 == HISTORY {
            x
        } else {
            model
                .graph_mut()
                .slice_axis(history[offset + 1], 1, 0, WIDTH)?
        };
        model.next_state(&history_name(layer, stage, offset), next)?;
    }
    let scale = model
        .graph_mut()
        .broadcast_to(scale, Shape::new(vec![1, WIDTH])?)?;
    let scaled = model
        .graph_mut()
        .mul(scale, conv.expect("nonempty convolution"))?;
    Ok(model.graph_mut().add(x, scaled)?)
}

#[derive(Clone, Copy)]
struct BlockStepPorts {
    norm: NormPorts,
    one_vec: NodeId,
    minus_vec: NodeId,
    half_vec: NodeId,
    one_up: NodeId,
    minus_up: NodeId,
    inv_sqrt: NodeId,
    minus_scalar: NodeId,
    delta_scale: Option<NodeId>,
    ema: EmaMode,
}

fn block_step(
    model: &mut ModelBuilder,
    mut x: NodeId,
    layer: usize,
    BlockStepPorts {
        norm,
        one_vec,
        minus_vec,
        half_vec,
        one_up,
        minus_up,
        inv_sqrt,
        minus_scalar,
        delta_scale,
        ema,
    }: BlockStepPorts,
    mode: MatrixMode,
) -> Result<NodeId, ModelError> {
    let shared_matrix = if mode != MatrixMode::Independent {
        Some(model.parameter(
            format!("weight_{layer}"),
            vec![WIDTH, WIDTH],
            Domain::Finite,
        )?)
    } else {
        None
    };
    let shared_transposed = shared_matrix
        .map(|weight| model.graph_mut().permute(weight, &[1, 0]))
        .transpose()?;
    let mut roles = Vec::with_capacity(ROLES);
    for role in 0..ROLES {
        let weight = if let Some(shared) = shared_matrix {
            shared
        } else {
            model.parameter(
                format!("weight_{layer}_{role}"),
                vec![WIDTH, WIDTH],
                Domain::Finite,
            )?
        };
        let transposed = if let Some(tied) = shared_transposed {
            tied
        } else {
            model.graph_mut().permute(weight, &[1, 0])?
        };
        let row = model.parameter(
            format!("row_{layer}_{role}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let col = model.parameter(
            format!("col_{layer}_{role}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let bias = model.parameter(
            format!("bias_{layer}_{role}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let gate = model.parameter(
            format!("mix_gate_{layer}_{role}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let gate_bias = model.parameter(
            format!("mix_bias_{layer}_{role}"),
            vec![1, 1],
            Domain::Finite,
        )?;
        let delta = if let Some(rank) = mode.role_delta_rank() {
            Some((
                model.parameter(
                    format!("delta_a_{layer}_{role}"),
                    vec![WIDTH, rank],
                    Domain::Finite,
                )?,
                model.parameter(
                    format!("delta_b_{layer}_{role}"),
                    vec![rank, WIDTH],
                    Domain::Finite,
                )?,
            ))
        } else {
            None
        };
        roles.push(RoleNodes {
            weight,
            transposed,
            row,
            col,
            bias,
            gate,
            gate_bias,
            delta,
        });
    }
    let p = ProjectionPorts {
        roles,
        ones_column: norm.ones,
        one_scalar: norm.one,
        minus_scalar,
        inv_sqrt,
        delta_scale,
    };
    let norm1_gamma = model.parameter(
        format!("norm1_gamma_{layer}"),
        vec![1, WIDTH],
        Domain::Finite,
    )?;
    let norm1_beta = model.parameter(
        format!("norm1_beta_{layer}"),
        vec![1, WIDTH],
        Domain::Finite,
    )?;
    let norm2_gamma = model.parameter(
        format!("norm2_gamma_{layer}"),
        vec![1, WIDTH],
        Domain::Finite,
    )?;
    let norm2_beta = model.parameter(
        format!("norm2_beta_{layer}"),
        vec![1, WIDTH],
        Domain::Finite,
    )?;
    let decay = model.parameter(
        format!("decay_logit_{layer}"),
        vec![1, WIDTH],
        Domain::Finite,
    )?;
    let gate_scale = model.parameter(format!("gate_scale_{layer}"), vec![1, 1], Domain::Finite)?;
    let conv_scale_1 =
        model.parameter(format!("conv_scale_1_{layer}"), vec![1, 1], Domain::Finite)?;
    let conv_scale_2 =
        model.parameter(format!("conv_scale_2_{layer}"), vec![1, 1], Domain::Finite)?;

    let a = layer_norm(model.graph_mut(), x, norm1_gamma, norm1_beta, norm)?;
    let a = conv_step(model, a, layer, 1, conv_scale_1)?;
    let r = project(model, a, 0, &p)?;
    let scale = model
        .graph_mut()
        .broadcast_to(gate_scale, Shape::new(vec![1, WIDTH])?)?;
    let r = model.graph_mut().mul(r, scale)?;
    let r = sigmoid(model.graph_mut(), r, one_vec, minus_vec)?;
    let k = project(model, a, 1, &p)?;
    let k = sigmoid(model.graph_mut(), k, one_vec, minus_vec)?;
    let v = project(model, a, 2, &p)?;
    let kv = model.graph_mut().mul(k, v)?;
    let state = model.state(format!("mem_{layer}"), vec![1, WIDTH], Domain::Any)?;
    // Selective alpha_t = sigmoid(decay_logit + a_t * decay_gate).
    // a_t is the normalized/Conv9 token representation. The zero gate
    // starts at the same decay as historical Static EMA, and the extra
    // parameters are assembled solely from shared VeriNeural primitives.
    let decay_logit = if ema == EmaMode::Selective {
        let strength = model.parameter(
            format!("decay_gate_{layer}"),
            vec![1, WIDTH],
            Domain::Finite,
        )?;
        let conditioned = model.graph_mut().mul(a, strength)?;
        model.graph_mut().add(decay, conditioned)?
    } else {
        decay
    };
    let decay = sigmoid(model.graph_mut(), decay_logit, one_vec, minus_vec)?;
    let old = model.graph_mut().mul(decay, state)?;
    let one_minus = model.graph_mut().sub(one_vec, decay)?;
    let injected = model.graph_mut().mul(one_minus, kv)?;
    let new_state = model.graph_mut().add(old, injected)?;
    model.next_state(&format!("mem_{layer}"), new_state)?;
    let rm = model.graph_mut().mul(r, new_state)?;
    let delta = project(model, rm, 3, &p)?;
    let delta = model.graph_mut().mul(delta, half_vec)?;
    x = model.graph_mut().add(x, delta)?;

    let f = layer_norm(model.graph_mut(), x, norm2_gamma, norm2_beta, norm)?;
    let f = conv_step(model, f, layer, 2, conv_scale_2)?;
    let up4 = project(model, f, 4, &p)?;
    let up5 = project(model, f, 5, &p)?;
    let up6 = project(model, f, 6, &p)?;
    let up7 = project(model, f, 7, &p)?;
    let activation = model.graph_mut().concat(1, &[up4, up5])?;
    let gating = model.graph_mut().concat(1, &[up6, up7])?;
    let sig = sigmoid(model.graph_mut(), activation, one_up, minus_up)?;
    let silu = model.graph_mut().mul(activation, sig)?;
    let hidden = model.graph_mut().mul(silu, gating)?;
    let h0 = model.graph_mut().slice_axis(hidden, 1, 0, WIDTH)?;
    let h1 = model.graph_mut().slice_axis(hidden, 1, WIDTH, WIDTH)?;
    let down0 = project(model, h0, 0, &p)?;
    let down1 = project(model, h1, 1, &p)?;
    let down = model.graph_mut().add(down0, down1)?;
    let down = model.graph_mut().mul(down, half_vec)?;
    Ok(model.graph_mut().add(x, down)?)
}

/// All four blocks, tied embedding/head, Conv9 histories and EMA state.
pub fn architecture_model(vocabulary: usize) -> Result<Model, ModelError> {
    architecture_model_for(vocabulary, MatrixMode::TorusShared)
}

pub fn architecture_model_for(vocabulary: usize, mode: MatrixMode) -> Result<Model, ModelError> {
    architecture_model_for_ema(vocabulary, mode, EmaMode::Static)
}

pub fn architecture_model_for_ema(
    vocabulary: usize,
    mode: MatrixMode,
    ema: EmaMode,
) -> Result<Model, ModelError> {
    if vocabulary == 0 || vocabulary > 4096 {
        return Err(ModelError::Invalid("vocabulary must be 1..=4096".into()));
    }
    let model_name = match mode {
        MatrixMode::Independent => "torus16-mix-ema4-lm-independent",
        MatrixMode::TorusShared => "torus16-mix-ema4-lm",
        MatrixMode::RoleDelta2 | MatrixMode::RoleDelta4 | MatrixMode::RoleDelta8 => {
            "torus16-mix-ema4-lm-role-delta"
        }
    };
    let model_name = if ema == EmaMode::Selective {
        "torus16-mix-ema4-lm-selective"
    } else {
        model_name
    };
    let mut model = ModelBuilder::new(model_name, "0.2.0");
    let token = model.input("token", vec![1, vocabulary], Domain::NonNegativeFinite)?;
    let embedding = model.parameter("embedding", vec![vocabulary, WIDTH], Domain::Finite)?;
    let ones = model.parameter("ones_column", vec![WIDTH, 1], Domain::Finite)?;
    let inv_width = model.parameter("inv_width", vec![1, 1], Domain::Finite)?;
    let one = model.parameter("one_scalar", vec![1, 1], Domain::Finite)?;
    let epsilon = model.parameter("epsilon_scalar", vec![1, 1], Domain::Finite)?;
    let one_vec = model.parameter("one_vec", vec![1, WIDTH], Domain::Finite)?;
    let minus_vec = model.parameter("minus_vec", vec![1, WIDTH], Domain::Finite)?;
    let minus_scalar = model.parameter("minus_scalar", vec![1, 1], Domain::Finite)?;
    let half_vec = model.parameter("half_vec", vec![1, WIDTH], Domain::Finite)?;
    let one_up = model.parameter("one_up", vec![1, 2 * WIDTH], Domain::Finite)?;
    let minus_up = model.parameter("minus_up", vec![1, 2 * WIDTH], Domain::Finite)?;
    let inv_sqrt = model.parameter("inv_sqrt_width", vec![1, 1], Domain::Finite)?;
    let delta_scale = if mode.role_delta_rank().is_some() {
        Some(model.parameter("delta_scale", vec![1, 1], Domain::Finite)?)
    } else {
        None
    };
    let norm = NormPorts {
        ones,
        inv_width,
        one,
        epsilon,
    };
    let mut x = model.graph_mut().matmul(token, embedding)?;
    let block_ports = BlockStepPorts {
        norm,
        one_vec,
        minus_vec,
        half_vec,
        one_up,
        minus_up,
        inv_sqrt,
        minus_scalar,
        delta_scale,
        ema,
    };
    for layer in 0..LAYERS {
        x = block_step(&mut model, x, layer, block_ports, mode)?;
    }
    let gamma = model.parameter("final_gamma", vec![1, WIDTH], Domain::Finite)?;
    let beta = model.parameter("final_beta", vec![1, WIDTH], Domain::Finite)?;
    let final_hidden = layer_norm(model.graph_mut(), x, gamma, beta, norm)?;
    let head = model.graph_mut().permute(embedding, &[1, 0])?;
    let logits = model.graph_mut().matmul(final_hidden, head)?;
    model.output("logits", logits)?;
    model.finish()
}
