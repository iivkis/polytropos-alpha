//! Two 64x64 rotating cores, a trainable eight-action router, and a sequential
//! recurrent state. The computational graphs are defined using the public
//! VeriNeural API; host code owns tokenizer/dataset, discrete action sampling,
//! parameter updates, and stochastic route sampling. Deterministic state feedback stays on GPU.
//!
//! The evidence model and deterministic runtime share one Semantic IR containing
//! the rotation bank and first-tie selection. Stochastic training remains an
//! explicit laboratory policy and may choose a route on the host; BPTT/training
//! and arbitrary-length recurrence are not claimed certified.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use verineural::{
    ArgMaxTieBreak, Domain, GpuEngine, Model, ModelBuilder, ModelError, ModelExecution,
    NeuralError, NeuralGraph, NodeId, Shape, Tensor,
};

pub const WIDTH: usize = 64;
pub const CORES: usize = 2;
pub const TURNS: usize = 4;
pub const ACTIONS: usize = CORES * TURNS;
pub const OP_COUNT: usize = 40;

const ROOT_NAMES: [&str; 8] = ["+", "*", "XOR", "AND", "OR", "SUB", "SHL", "SHR"];
const COMBINE_NAMES: [&str; 4] = ["+", "XOR", "AND", "OR"];

#[derive(Debug)]
pub struct LabError(pub String);
impl fmt::Display for LabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for LabError {}

/// Temporary backend workspace. The only persistent laboratory output lives
/// directly in labs/rotating-matrices-0.1.0/artifacts/.
/// The staged backend trees are deleted even when an operation fails.
pub struct ArtifactWorkspace {
    path: PathBuf,
}

impl ArtifactWorkspace {
    pub fn new() -> std::io::Result<Self> {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(std::io::Error::other)?
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "verineural-rotating-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ArtifactWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// The declared 40-operation grammar. Operations and compositions work on
/// nonnegative eight-bit integer values, with an explicit modulo-256 policy
/// for combined operations. Simple addition/multiplication keep their usual
/// value for the dataset operand range (0..=15).
#[derive(Clone, Debug)]
pub struct Example {
    pub op: usize,
    pub a: u16,
    pub b: u16,
    pub text: String,
    pub target: u16,
}

fn root(op: usize, a: u16, b: u16) -> u16 {
    match op {
        0 => a + b,
        1 => a * b,
        2 => a ^ b,
        3 => a & b,
        4 => a | b,
        5 => a.abs_diff(b),
        6 => (a << (b % 4)) & 255,
        7 => a >> (b % 4),
        _ => unreachable!("root operation outside 0..8"),
    }
}

pub fn operation(op: usize, a: u16, b: u16) -> Result<Example, LabError> {
    if op >= OP_COUNT || a > 15 || b > 15 {
        return Err(LabError(
            "operation or operand outside laboratory domain".into(),
        ));
    }
    let (text, target) = if op < ROOT_NAMES.len() {
        (format!("{a} {} {b}", ROOT_NAMES[op]), root(op, a, b))
    } else {
        let group = (op - 8) / 8;
        let first = (op - 8) % 8;
        let second = (first + 3) % 8;
        let x = root(first, a, b);
        let y = root(second, a, b);
        let value = match group {
            0 => (x + y) & 255,
            1 => x ^ y,
            2 => x & y,
            3 => x | y,
            _ => unreachable!(),
        };
        (
            format!(
                "({a} {} {b}) {} ({a} {} {b})",
                ROOT_NAMES[first], COMBINE_NAMES[group], ROOT_NAMES[second]
            ),
            value,
        )
    };
    Ok(Example {
        op,
        a,
        b,
        text,
        target,
    })
}

/// Hold out operand *pairs* across all 40 operation types. This prevents the
/// same operand pair appearing in the train and validation partitions.
#[derive(Debug)]
pub struct Dataset {
    pub train: Vec<Example>,
    pub validation: Vec<Example>,
}
impl Dataset {
    pub fn new() -> Self {
        let mut train = Vec::new();
        let mut validation = Vec::new();
        for op in 0..OP_COUNT {
            for a in 0..=15 {
                for b in 0..=15 {
                    let case = operation(op, a, b).expect("enumeration stays in the domain");
                    if (a * 17 + b * 31) % 5 == 0 {
                        validation.push(case);
                    } else {
                        train.push(case);
                    }
                }
            }
        }
        Self { train, validation }
    }
}
impl Default for Dataset {
    fn default() -> Self {
        Self::new()
    }
}

/// A training-partition byte dictionary, not a text/subword tokenizer.
/// All input is UTF-8 encoded bytes; unseen bytes fail closed.
#[derive(Clone, Debug)]
pub struct Dictionary {
    bytes: Vec<u8>,
    ids: [Option<u16>; 256],
}
impl Dictionary {
    pub fn fit(examples: &[Example]) -> Self {
        let mut counts = [0u32; 256];
        for case in examples {
            for byte in case.text.bytes() {
                counts[byte as usize] += 1;
            }
        }
        let mut bytes = (0..=255u16)
            .filter(|&byte| counts[byte as usize] > 0)
            .map(|byte| byte as u8)
            .collect::<Vec<_>>();
        bytes.sort_by_key(|&byte| (std::cmp::Reverse(counts[byte as usize]), byte));
        let mut ids = [None; 256];
        for (index, &byte) in bytes.iter().enumerate() {
            ids[byte as usize] = Some(index as u16);
        }
        Self { bytes, ids }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn byte_order(&self) -> &[u8] {
        &self.bytes
    }
    pub fn encode(&self, input: &str) -> Result<Vec<u16>, LabError> {
        input
            .as_bytes()
            .iter()
            .map(|&byte| {
                self.ids[byte as usize]
                    .ok_or_else(|| LabError(format!("unseen UTF-8 byte 0x{byte:02x}")))
            })
            .collect()
    }

    /// Dictionary state: normalized byte-id histogram hashed into 64 cells.
    /// It is available to the router before token-by-token core execution.
    pub fn state(&self, tokens: &[u16]) -> [f32; WIDTH] {
        let mut state = [0f32; WIDTH];
        for &token in tokens {
            state[token as usize % WIDTH] += 1.0;
        }
        if !tokens.is_empty() {
            for value in &mut state {
                *value /= tokens.len() as f32;
            }
        }
        state
    }

    /// Base-2 token value (dictionary index + 1) in the low nine slots of a
    /// fixed 64-dimensional vector. The nonzero offset reserves zero for pad.
    pub fn bits(token: u16) -> [f32; WIDTH] {
        let mut bits = [0f32; WIDTH];
        for (i, bit) in bits.iter_mut().enumerate().take(9) {
            *bit = (((token as u32 + 1) >> i) & 1) as f32;
        }
        bits
    }
}

/// Row-major clockwise quarter turns using the same generic host tensor
/// primitives as the Semantic IR construction. This is a host oracle/helper
/// for stochastic training and tests; deterministic inference routes and
/// rotates inside the unified Semantic IR GPU graph.
pub fn rotate(matrix: &[f32], turns: usize) -> Result<Vec<f32>, LabError> {
    if matrix.len() != WIDTH * WIDTH || turns >= TURNS {
        return Err(LabError(
            "invalid rotation: expected 64x64 and turns 0..4".into(),
        ));
    }
    let shape = Shape::new(vec![WIDTH, WIDTH]).map_err(|error| LabError(error.to_string()))?;
    let base = Tensor::new(shape, matrix.to_vec()).map_err(|error| LabError(error.to_string()))?;
    let rotated = match turns {
        0 => Ok(base),
        1 => base
            .permute(&[1, 0])
            .and_then(|tensor| tensor.reverse_axis(1)),
        2 => base
            .reverse_axis(0)
            .and_then(|tensor| tensor.reverse_axis(1)),
        3 => base
            .permute(&[1, 0])
            .and_then(|tensor| tensor.reverse_axis(0)),
        _ => unreachable!(),
    }
    .map_err(|error| LabError(error.to_string()))?;
    Ok(rotated.into_values())
}

pub fn route_parts(action: usize) -> Result<(usize, usize), LabError> {
    if action >= ACTIONS {
        return Err(LabError("router action outside 0..8".into()));
    }
    Ok((action / TURNS, action % TURNS))
}

fn recurrent_transition(g: &mut NeuralGraph, x: &[NodeId]) -> Result<NodeId, NeuralError> {
    let combined = g.add(x[0], x[1])?;
    let mixed = g.matmul(combined, x[2])?;
    let shifted = g.bias_add(mixed, x[3])?;
    g.relu(shifted)
}

fn routing_logits(g: &mut NeuralGraph, x: &[NodeId]) -> Result<NodeId, NeuralError> {
    let features = g.add(x[0], x[1])?;
    let product = g.matmul(features, x[2])?;
    g.bias_add(product, x[3])
}

fn scalar_readout(g: &mut NeuralGraph, x: &[NodeId]) -> Result<NodeId, NeuralError> {
    let projection = g.matmul(x[0], x[1])?;
    g.bias_add(projection, x[2])
}

/// Learned logits for 2 cores x 4 orientations. This standalone model is kept
/// for stochastic-training policy, where sampling intentionally remains on the
/// host. Deterministic inference uses architecture_model() and GPU ArgMax/Take.
pub fn router_model() -> Result<Model, ModelError> {
    let mut m = ModelBuilder::new("rotating_router", "0.1.0");
    let dictionary = m.input("dictionary_state", [1, WIDTH], Domain::NonNegativeFinite)?;
    let state = m.input("state", [1, WIDTH], Domain::NonNegativeFinite)?;
    let weights = m.parameter("weights", [WIDTH, ACTIONS], Domain::Finite)?;
    let bias = m.parameter("bias", [ACTIONS], Domain::Finite)?;
    let logits = m.block(
        "routing_logits",
        &[dictionary, state, weights, bias],
        routing_logits,
    )?;
    m.output("logits", logits)?;
    m.finish()
}

/// Scalar normalized regression head; decoding rounds to ONE integer value.
pub fn head_model() -> Result<Model, ModelError> {
    let mut m = ModelBuilder::new("rotating_scalar_head", "0.1.0");
    let state = m.input("final_state", [1, WIDTH], Domain::NonNegativeFinite)?;
    let weights = m.parameter("weights", [WIDTH, 1], Domain::Finite)?;
    let bias = m.parameter("bias", [1], Domain::Finite)?;
    let scalar = m.block("scalar_readout", &[state, weights, bias], scalar_readout)?;
    m.output("normalized_result", scalar)?;
    m.finish()
}

/// One fixed training-policy route. The route choice is host policy, but the
/// spatial rotation itself remains an ordinary Semantic IR transform and GPU
/// lowering; no host-rotated tensor is bound as a derived parameter.
pub fn routed_core_model(action: usize) -> Result<Model, ModelError> {
    if action >= ACTIONS {
        return Err(ModelError::Invalid("route action outside 0..8".into()));
    }
    let turn = action % TURNS;
    let mut m = ModelBuilder::new(format!("rotating_core_route_{action}"), "0.1.0");
    let state = m.input("state", [1, WIDTH], Domain::Finite)?;
    let token = m.input("token_bits", [1, WIDTH], Domain::NonNegativeFinite)?;
    let base = m.parameter("base_weight", [WIDTH, WIDTH], Domain::Finite)?;
    let bias = m.parameter("bias", [WIDTH], Domain::Finite)?;
    let rotated = rotated_graph_weight(m.graph_mut(), base, turn)?;
    let next = m.block(
        "recurrent_transition",
        &[state, token, rotated, bias],
        recurrent_transition,
    )?;
    m.output("next_state", next)?;
    m.finish()
}

fn rotated_graph_weight(
    g: &mut NeuralGraph,
    core: NodeId,
    turn: usize,
) -> Result<NodeId, NeuralError> {
    match turn {
        0 => Ok(core),
        1 => {
            let transposed = g.permute(core, &[1, 0])?;
            g.reverse_axis(transposed, 1)
        }
        2 => {
            let rows = g.reverse_axis(core, 0)?;
            g.reverse_axis(rows, 1)
        }
        3 => {
            let transposed = g.permute(core, &[1, 0])?;
            g.reverse_axis(transposed, 0)
        }
        _ => Err(NeuralError::ContractViolation {
            module: "rotated_graph_weight",
            detail: "turn must be in 0..4",
        }),
    }
}

fn rotating_bank(g: &mut NeuralGraph, core0: NodeId, core1: NodeId) -> Result<NodeId, NeuralError> {
    let mut variants = Vec::with_capacity(ACTIONS);
    for core in [core0, core1] {
        for turn in 0..TURNS {
            let rotated = rotated_graph_weight(g, core, turn)?;
            variants.push(g.reshape(rotated, Shape::new(vec![1, WIDTH, WIDTH])?)?);
        }
    }
    g.concat(0, &variants)
}

/// One evidence model for the whole laboratory architecture.
///
/// Router, rotation bank, deterministic first-tie argmax selection, recurrent
/// transition and terminal readout share ONE Semantic Neural IR. The P0
/// structural/indexing path is therefore visible to canonical IR and Lean.
/// GPU lowering supports the deliberately narrow structural/indexing profile
/// used here; unsupported axes/ranks remain fail-closed.
pub fn architecture_model() -> Result<Model, ModelError> {
    let mut m = ModelBuilder::new("rotating-matrices", "0.1.0");
    let dictionary = m.input("dictionary_state", [1, WIDTH], Domain::NonNegativeFinite)?;
    let state = m.state("recurrent_state", [1, WIDTH], Domain::NonNegativeFinite)?;
    let token = m.input("token_bits", [1, WIDTH], Domain::NonNegativeFinite)?;

    let router_weights = m.parameter("router_weights", [WIDTH, ACTIONS], Domain::Finite)?;
    let router_bias = m.parameter("router_bias", [ACTIONS], Domain::Finite)?;
    let core0 = m.parameter("core0", [WIDTH, WIDTH], Domain::Finite)?;
    let core1 = m.parameter("core1", [WIDTH, WIDTH], Domain::Finite)?;
    let core_bias = m.parameter("core_bias", [WIDTH], Domain::Finite)?;
    let head_weights = m.parameter("head_weights", [WIDTH, 1], Domain::Finite)?;
    let head_bias = m.parameter("head_bias", [1], Domain::Finite)?;

    let logits = m.block(
        "routing_logits",
        &[dictionary, state, router_weights, router_bias],
        routing_logits,
    )?;
    let action = m.graph_mut().argmax(logits, 1, ArgMaxTieBreak::First)?;
    let bank = rotating_bank(m.graph_mut(), core0, core1)?;
    let selected_weight = m.graph_mut().take(bank, action, 0)?;
    let next_state = m.block(
        "routed_recurrent_transition",
        &[state, token, selected_weight, core_bias],
        recurrent_transition,
    )?;
    let scalar = m.block(
        "terminal_scalar_readout",
        &[next_state, head_weights, head_bias],
        scalar_readout,
    )?;

    m.next_state("recurrent_state", next_state)?;
    m.output("router_logits", logits)?;
    m.output("next_state", next_state)?;
    m.output("normalized_result", scalar)?;
    m.finish()
}

pub struct Architectures {
    pub architecture: Model,
    pub routes: Vec<Model>,
    pub router: Model,
    pub head: Model,
}
impl Architectures {
    pub fn new() -> Result<Self, ModelError> {
        let routes = (0..ACTIONS)
            .map(routed_core_model)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            architecture: architecture_model()?,
            routes,
            router: router_model()?,
            head: head_model()?,
        })
    }
}

/// Fixed-seed, dependency-free RNG. Training data ordering and initialization
/// are reproducible, but GPU floating-point parity is not asserted here.
#[derive(Clone)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16_777_216.0
    }
    pub fn signed(&mut self) -> f32 {
        self.uniform() * 2.0 - 1.0
    }
}

/// The two canonical matrices own all 8192 weights. Rotated matrices are
/// derived views at each invocation; never eight independent parameter sets.
pub struct Parameters {
    pub cores: [Vec<f32>; CORES],
    pub router: Vec<f32>,
    pub router_bias: [f32; ACTIONS],
    pub head: [f32; WIDTH],
    pub head_bias: f32,
    pub core_bias: [f32; WIDTH],
}
impl Parameters {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let cores = std::array::from_fn(|_| {
            let mut w = vec![0.0; WIDTH * WIDTH];
            for r in 0..WIDTH {
                for c in 0..WIDTH {
                    w[r * WIDTH + c] = if r == c { 0.65 } else { 0.018 * rng.signed() };
                }
            }
            w
        });
        let router = (0..WIDTH * ACTIONS).map(|_| rng.signed() * 0.02).collect();
        let head = std::array::from_fn(|_| rng.signed() * 0.01);
        Self {
            cores,
            router,
            router_bias: [0.0; ACTIONS],
            head,
            head_bias: 0.18,
            core_bias: [0.008; WIDTH],
        }
    }
}

/// Each GPU program is compiled once and replayed with refreshed input/weight
/// tensors on an explicit nonblocking CUDA Graph stream (VeriNeural's runtime).
/// Without CUDA/LLVM/Lean dependencies, the appropriate action is an error,
/// not a hidden CPU implementation of the same neural forward pass.
pub struct Runner {
    pub models: Architectures,
    deterministic: ModelExecution,
    routes: Vec<ModelExecution>,
    router: ModelExecution,
    head: ModelExecution,
}
impl Runner {
    pub fn new(p: &Parameters) -> Result<Self, Box<dyn Error>> {
        let workspace = ArtifactWorkspace::new()?;
        let models = Architectures::new()?;
        let gpu = GpuEngine::default();
        // P1 named state execution owns the runtime metadata copied from the
        // model, so Runner can own both without self-referential lifetimes.
        // The CUDA Graph is captured once; state feedback is D2D after step().
        let deterministic = gpu.prepare_model(
            &models.architecture,
            [
                (
                    "dictionary_state",
                    Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?,
                ),
                (
                    "recurrent_state",
                    Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?,
                ),
                ("token_bits", Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?),
                (
                    "router_weights",
                    Tensor::from_slice([WIDTH, ACTIONS], &p.router)?,
                ),
                (
                    "router_bias",
                    Tensor::from_slice([ACTIONS], &p.router_bias)?,
                ),
                ("core0", Tensor::from_slice([WIDTH, WIDTH], &p.cores[0])?),
                ("core1", Tensor::from_slice([WIDTH, WIDTH], &p.cores[1])?),
                ("core_bias", Tensor::from_slice([WIDTH], &p.core_bias)?),
                ("head_weights", Tensor::from_slice([WIDTH, 1], &p.head)?),
                ("head_bias", Tensor::from_slice([1], &[p.head_bias])?),
            ],
            &workspace.path().join("deterministic"),
        )?;
        let mut routes = Vec::with_capacity(ACTIONS);
        for (action, model) in models.routes.iter().enumerate() {
            let (core, _) = route_parts(action)?;
            routes.push(gpu.prepare_model(
                model,
                [
                    ("state", Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?),
                    ("token_bits", Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?),
                    (
                        "base_weight",
                        Tensor::from_slice([WIDTH, WIDTH], &p.cores[core])?,
                    ),
                    ("bias", Tensor::from_slice([WIDTH], &p.core_bias)?),
                ],
                &workspace.path().join(format!("route_{action}")),
            )?);
        }
        let router = gpu.prepare_model(
            &models.router,
            [
                (
                    "dictionary_state",
                    Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?,
                ),
                ("state", Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?),
                ("weights", Tensor::from_slice([WIDTH, ACTIONS], &p.router)?),
                ("bias", Tensor::from_slice([ACTIONS], &p.router_bias)?),
            ],
            &workspace.path().join("router"),
        )?;
        let head = gpu.prepare_model(
            &models.head,
            [
                (
                    "final_state",
                    Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?,
                ),
                ("weights", Tensor::from_slice([WIDTH, 1], &p.head)?),
                ("bias", Tensor::from_slice([1], &[p.head_bias])?),
            ],
            &workspace.path().join("head"),
        )?;
        Ok(Self {
            models,
            deterministic,
            routes,
            router,
            head,
        })
    }

    fn prepare_router(
        &mut self,
        p: &Parameters,
        dict: &[f32; WIDTH],
    ) -> Result<(), Box<dyn Error>> {
        self.router
            .set_input("dictionary_state", &Tensor::from_slice([1, WIDTH], dict)?)?;
        self.router
            .set_parameter("weights", &Tensor::from_slice([WIDTH, ACTIONS], &p.router)?)?;
        self.router
            .set_parameter("bias", &Tensor::from_slice([ACTIONS], &p.router_bias)?)?;
        Ok(())
    }

    fn prepare_routes(&mut self, p: &Parameters) -> Result<(), Box<dyn Error>> {
        for action in 0..ACTIONS {
            let (core, _) = route_parts(action)?;
            self.routes[action].set_parameter(
                "base_weight",
                &Tensor::from_slice([WIDTH, WIDTH], &p.cores[core])?,
            )?;
            self.routes[action]
                .set_parameter("bias", &Tensor::from_slice([WIDTH], &p.core_bias)?)?;
        }
        Ok(())
    }

    fn logits(&mut self, state: &[f32; WIDTH]) -> Result<[f32; ACTIONS], Box<dyn Error>> {
        self.router
            .set_input("state", &Tensor::from_slice([1, WIDTH], state)?)?;
        self.router.run()?;
        Ok(self
            .router
            .output("logits")?
            .values()
            .try_into()
            .map_err(|_| LabError("router shape".into()))?)
    }

    fn step(
        &mut self,
        state: &[f32; WIDTH],
        token: u16,
        action: usize,
    ) -> Result<[f32; WIDTH], Box<dyn Error>> {
        route_parts(action)?;
        let execution = self
            .routes
            .get_mut(action)
            .ok_or_else(|| LabError("route execution missing".into()))?;
        execution.set_input("state", &Tensor::from_slice([1, WIDTH], state)?)?;
        execution.set_input(
            "token_bits",
            &Tensor::from_slice([1, WIDTH], &Dictionary::bits(token))?,
        )?;
        execution.run()?;
        let next: [f32; WIDTH] = execution
            .output("next_state")?
            .values()
            .try_into()
            .map_err(|_| LabError("core shape".into()))?;
        if next.iter().any(|v| !v.is_finite()) {
            return Err(Box::new(LabError("nonfinite recurrent state".into())));
        }
        Ok(next)
    }

    fn readout(&mut self, p: &Parameters, state: &[f32; WIDTH]) -> Result<f32, Box<dyn Error>> {
        self.head
            .set_input("final_state", &Tensor::from_slice([1, WIDTH], state)?)?;
        self.head
            .set_parameter("weights", &Tensor::from_slice([WIDTH, 1], &p.head)?)?;
        self.head
            .set_parameter("bias", &Tensor::from_slice([1], &[p.head_bias])?)?;
        self.head.run()?;
        let value = self.head.output("normalized_result")?.values()[0];
        if !value.is_finite() {
            return Err(Box::new(LabError("nonfinite readout".into())));
        }
        Ok(value)
    }

    fn deterministic_forward(
        &mut self,
        p: &Parameters,
        dictionary: &Dictionary,
        input: &str,
    ) -> Result<Trace, Box<dyn Error>> {
        let tokens = dictionary.encode(input)?;
        if tokens.is_empty() {
            let state = [0.0; WIDTH];
            return Ok(Trace {
                normalized: self.readout(p, &state)?,
                state,
                decisions: Vec::new(),
            });
        }
        let dict = dictionary.state(&tokens);
        let mut state = [0f32; WIDTH];
        let mut decisions = Vec::with_capacity(tokens.len());

        // Parameters and sequence-wide dictionary features are uploaded once
        // per forward call. Per token only token_bits changes; state feedback
        // stays on-device through ModelExecution::step().
        self.deterministic
            .set_input("dictionary_state", &Tensor::from_slice([1, WIDTH], &dict)?)?;
        self.deterministic.set_state(
            "recurrent_state",
            &Tensor::from_slice([1, WIDTH], &[0.0; WIDTH])?,
        )?;
        self.deterministic.set_parameter(
            "router_weights",
            &Tensor::from_slice([WIDTH, ACTIONS], &p.router)?,
        )?;
        self.deterministic.set_parameter(
            "router_bias",
            &Tensor::from_slice([ACTIONS], &p.router_bias)?,
        )?;
        self.deterministic
            .set_parameter("core0", &Tensor::from_slice([WIDTH, WIDTH], &p.cores[0])?)?;
        self.deterministic
            .set_parameter("core1", &Tensor::from_slice([WIDTH, WIDTH], &p.cores[1])?)?;
        self.deterministic
            .set_parameter("core_bias", &Tensor::from_slice([WIDTH], &p.core_bias)?)?;
        self.deterministic
            .set_parameter("head_weights", &Tensor::from_slice([WIDTH, 1], &p.head)?)?;
        self.deterministic
            .set_parameter("head_bias", &Tensor::from_slice([1], &[p.head_bias])?)?;

        for &token in &tokens {
            let features = std::array::from_fn(|i| dict[i] + state[i]);
            self.deterministic.set_input(
                "token_bits",
                &Tensor::from_slice([1, WIDTH], &Dictionary::bits(token))?,
            )?;
            self.deterministic.step()?;

            let logits: [f32; ACTIONS] = self
                .deterministic
                .output("router_logits")?
                .values()
                .try_into()
                .map_err(|_| LabError("router shape".into()))?;
            state = self
                .deterministic
                .output("next_state")?
                .values()
                .try_into()
                .map_err(|_| LabError("core shape".into()))?;
            if state.iter().any(|v| !v.is_finite()) {
                return Err(Box::new(LabError(
                    "nonfinite deterministic recurrent state".into(),
                )));
            }
            decisions.push(Decision {
                action: argmax_first(&logits),
                probs: probabilities(&logits),
                features,
            });
        }

        let normalized = self.deterministic.output("normalized_result")?.values()[0];
        if !normalized.is_finite() {
            return Err(Box::new(LabError(
                "nonfinite deterministic architecture output".into(),
            )));
        }
        Ok(Trace {
            normalized,
            state,
            decisions,
        })
    }

    pub fn forward(
        &mut self,
        p: &Parameters,
        dictionary: &Dictionary,
        input: &str,
        rng: &mut Rng,
        sample_routes: bool,
    ) -> Result<Trace, Box<dyn Error>> {
        if sample_routes {
            self.forward_with_route(p, dictionary, input, rng, true, None)
        } else {
            self.deterministic_forward(p, dictionary, input)
        }
    }

    /// Fixed-action counterfactual for measuring the effect of learned routing.
    pub fn forward_with_route(
        &mut self,
        p: &Parameters,
        dictionary: &Dictionary,
        input: &str,
        rng: &mut Rng,
        sample_routes: bool,
        fixed: Option<usize>,
    ) -> Result<Trace, Box<dyn Error>> {
        if fixed.is_none() && !sample_routes {
            return self.deterministic_forward(p, dictionary, input);
        }
        if let Some(action) = fixed {
            route_parts(action)?;
        }
        let tokens = dictionary.encode(input)?;
        let dict = dictionary.state(&tokens);
        self.prepare_routes(p)?;
        if fixed.is_none() {
            self.prepare_router(p, &dict)?;
        }
        let mut state = [0f32; WIDTH];
        let mut decisions = Vec::with_capacity(tokens.len());
        for &token in &tokens {
            let probs = if fixed.is_some() {
                [1.0 / ACTIONS as f32; ACTIONS]
            } else {
                probabilities(&self.logits(&state)?)
            };
            let action = fixed.unwrap_or_else(|| choose(&probs, rng, sample_routes));
            let features = std::array::from_fn(|i| dict[i] + state[i]);
            decisions.push(Decision {
                action,
                probs,
                features,
            });
            state = self.step(&state, token, action)?;
        }
        let normalized = self.readout(p, &state)?;
        Ok(Trace {
            normalized,
            state,
            decisions,
        })
    }

    /// Fixed route trace for common-random-number SPSA core updates. No
    /// duplicated host implementation of the verified transition formula.
    fn replay_decisions(
        &mut self,
        p: &Parameters,
        dictionary: &Dictionary,
        input: &str,
        decisions: &[Decision],
    ) -> Result<f32, Box<dyn Error>> {
        let tokens = dictionary.encode(input)?;
        if tokens.len() != decisions.len() {
            return Err(Box::new(LabError("bad route trace".into())));
        }
        self.prepare_routes(p)?;
        let mut state = [0f32; WIDTH];
        for (&token, decision) in tokens.iter().zip(decisions) {
            state = self.step(&state, token, decision.action)?;
        }
        self.readout(p, &state)
    }
}

#[derive(Clone)]
pub struct Decision {
    pub action: usize,
    pub probs: [f32; ACTIONS],
    pub features: [f32; WIDTH],
}
pub struct Trace {
    pub normalized: f32,
    pub state: [f32; WIDTH],
    pub decisions: Vec<Decision>,
}
pub fn decoded(normalized: f32) -> u16 {
    (normalized * 256.0).round().clamp(0.0, 255.0) as u16
}

fn probabilities(logits: &[f32; ACTIONS]) -> [f32; ACTIONS] {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = logits.map(|v| (v - maximum).clamp(-80.0, 0.0).exp());
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        return [1.0 / ACTIONS as f32; ACTIONS];
    }
    for v in &mut weights {
        *v /= sum;
    }
    weights
}
fn argmax_first(values: &[f32; ACTIONS]) -> usize {
    let mut best = 0;
    for index in 1..ACTIONS {
        if values[index].is_nan() {
            continue;
        }
        if values[best].is_nan() || values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn choose(probs: &[f32; ACTIONS], rng: &mut Rng, stochastic: bool) -> usize {
    if !stochastic {
        return argmax_first(probs);
    }
    let mut sample = rng.uniform();
    for (i, &prob) in probs.iter().enumerate() {
        sample -= prob;
        if sample <= 0.0 {
            return i;
        }
    }
    ACTIONS - 1
}

#[derive(Clone, Copy)]
pub struct TrainingConfig {
    pub updates: usize,
    pub head_lr: f32,
    pub router_lr: f32,
    pub core_lr: f32,
    pub spsa_interval: usize,
    pub spsa_epsilon: f32,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            updates: 128,
            head_lr: 0.02,
            router_lr: 0.002,
            core_lr: 0.0001,
            spsa_interval: 8,
            spsa_epsilon: 0.005,
        }
    }
}

#[derive(Debug)]
pub struct Metrics {
    pub count: usize,
    pub exact: usize,
    pub absolute_error: f64,
    pub squared_error: f64,
    pub action_counts: [usize; ACTIONS],
    pub per_op_exact: [usize; OP_COUNT],
    pub per_op_count: [usize; OP_COUNT],
}
impl Default for Metrics {
    fn default() -> Self {
        Self {
            count: 0,
            exact: 0,
            absolute_error: 0.0,
            squared_error: 0.0,
            action_counts: [0; ACTIONS],
            per_op_exact: [0; OP_COUNT],
            per_op_count: [0; OP_COUNT],
        }
    }
}
impl Metrics {
    fn record(&mut self, case: &Example, prediction: u16) {
        let difference = prediction as f64 - case.target as f64;
        self.count += 1;
        self.exact += usize::from(prediction == case.target);
        self.absolute_error += difference.abs();
        self.squared_error += difference * difference;
        self.per_op_count[case.op] += 1;
        self.per_op_exact[case.op] += usize::from(prediction == case.target);
    }

    pub fn accuracy(&self) -> f64 {
        self.exact as f64 / self.count.max(1) as f64
    }
    pub fn mae(&self) -> f64 {
        self.absolute_error / self.count.max(1) as f64
    }
    pub fn rmse(&self) -> f64 {
        (self.squared_error / self.count.max(1) as f64).sqrt()
    }
}

/// Online regression for the scalar head, REINFORCE for the discrete router,
/// SPSA (common routing trace) for both core matrices. This is an experimental
/// training rule, not proof of convergence or a library-level optimizer.
pub fn train(
    runner: &mut Runner,
    p: &mut Parameters,
    data: &Dataset,
    dict: &Dictionary,
    config: TrainingConfig,
    seed: u64,
) -> Result<(), Box<dyn Error>> {
    if data.train.is_empty() || config.spsa_interval == 0 || config.spsa_epsilon <= 0.0 {
        return Err(Box::new(LabError("invalid training configuration".into())));
    }
    let mut rng = Rng::new(seed);
    let mut baseline = 0.05f32;
    for iteration in 0..config.updates {
        // Cycle operation types to avoid accidentally training only a subset.
        let op = iteration % OP_COUNT;
        let a = (rng.next_u64() % 16) as u16;
        let b = (rng.next_u64() % 16) as u16;
        let case = if !(a * 17 + b * 31).is_multiple_of(5) {
            operation(op, a, b)?
        } else {
            // Deterministic rejection sampling: never train on a held-out pair.
            let mut found = None;
            for offset in 1..=16u16 {
                let candidate = (b + offset) % 16;
                if !(a * 17 + candidate * 31).is_multiple_of(5) {
                    found = Some(operation(op, a, candidate)?);
                    break;
                }
            }
            found.ok_or_else(|| LabError("no training operand pair".into()))?
        };
        let trace = runner.forward(p, dict, &case.text, &mut rng, true)?;
        let target = case.target as f32 / 256.0;
        let error = (trace.normalized - target).clamp(-1.0, 1.0);
        let loss = error * error * 0.5;

        // The same perturbation signs affect both trainable 64x64 bases.
        if config.core_lr > 0.0 && iteration % config.spsa_interval == 0 {
            let signs: [Vec<f32>; CORES] = std::array::from_fn(|_| {
                (0..WIDTH * WIDTH)
                    .map(|_| if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 })
                    .collect()
            });
            for (weights, core_signs) in p.cores.iter_mut().zip(&signs) {
                for (weight, sign) in weights.iter_mut().zip(core_signs) {
                    *weight += config.spsa_epsilon * sign;
                }
            }
            let plus = runner.replay_decisions(p, dict, &case.text, &trace.decisions)?;
            for (weights, core_signs) in p.cores.iter_mut().zip(&signs) {
                for (weight, sign) in weights.iter_mut().zip(core_signs) {
                    *weight -= 2.0 * config.spsa_epsilon * sign;
                }
            }
            let minus = runner.replay_decisions(p, dict, &case.text, &trace.decisions)?;
            let gradient =
                ((plus - target).powi(2) - (minus - target).powi(2)) / (4.0 * config.spsa_epsilon);
            for (weights, core_signs) in p.cores.iter_mut().zip(&signs) {
                for (weight, sign) in weights.iter_mut().zip(core_signs) {
                    // We are currently at W - epsilon*S; restore W first.
                    *weight = (*weight + config.spsa_epsilon * sign
                        - config.core_lr * gradient.clamp(-1.0, 1.0) * sign)
                        .clamp(-1.0, 1.0);
                }
            }
        }
        for (w, &h) in p.head.iter_mut().zip(&trace.state) {
            *w = (*w - config.head_lr * error * h).clamp(-2.0, 2.0);
        }
        p.head_bias = (p.head_bias - config.head_lr * error).clamp(-1.0, 1.0);
        // Loss-as-negative-reward with moving baseline.
        let advantage = (baseline - loss).clamp(-0.5, 0.5);
        for decision in &trace.decisions {
            for action in 0..ACTIONS {
                let score = (usize::from(action == decision.action) as f32
                    - decision.probs[action])
                    * config.router_lr
                    * advantage;
                for feature in 0..WIDTH {
                    p.router[feature * ACTIONS + action] = (p.router[feature * ACTIONS + action]
                        + score * decision.features[feature])
                        .clamp(-2.0, 2.0);
                }
                p.router_bias[action] = (p.router_bias[action] + score).clamp(-2.0, 2.0);
            }
        }
        baseline = baseline * 0.95 + loss * 0.05;
    }
    Ok(())
}

/// Stratified held-out sampling: cycle all 40 operators and distribute operand
/// pairs instead of selecting the all-zero (0,0) pair for each first sample.
/// For our 52 held-out pairs per operation, stride 19 visits every pair.
fn validation_case_index(example_count: usize, index: usize) -> usize {
    let per_op = example_count / OP_COUNT;
    if per_op > 0 && example_count.is_multiple_of(OP_COUNT) {
        let op = index % OP_COUNT;
        let round = index / OP_COUNT;
        op * per_op + (7 + 11 * op + 19 * round) % per_op
    } else {
        (index * 7919) % example_count
    }
}

/// No GPU required. Supply a prediction selected ONLY from training labels,
/// so the baseline does not tune parameters on the validation partition.
pub fn constant_baseline(examples: &[Example], limit: usize, prediction: u16) -> Metrics {
    let mut metrics = Metrics::default();
    if examples.is_empty() {
        return metrics;
    }
    for index in 0..limit.min(examples.len()) {
        let case = &examples[validation_case_index(examples.len(), index)];
        metrics.record(case, prediction);
    }
    metrics
}

pub fn evaluate(
    runner: &mut Runner,
    p: &Parameters,
    dict: &Dictionary,
    examples: &[Example],
    limit: usize,
    seed: u64,
) -> Result<Metrics, Box<dyn Error>> {
    evaluate_with_route(runner, p, dict, examples, limit, seed, None)
}

/// Same dataset and weights, but optionally lock every token to a single
/// (core_id, rotate) choice. This is a true routing ablation on the GPU model.
pub fn evaluate_with_route(
    runner: &mut Runner,
    p: &Parameters,
    dict: &Dictionary,
    examples: &[Example],
    limit: usize,
    seed: u64,
    fixed: Option<usize>,
) -> Result<Metrics, Box<dyn Error>> {
    if examples.is_empty() {
        return Err(Box::new(LabError("empty validation split".into())));
    }
    let mut rng = Rng::new(seed);
    let mut metrics = Metrics::default();
    // Balanced operator coverage with distinct held-out operand pairs.
    for index in 0..limit.min(examples.len()) {
        let case = &examples[validation_case_index(examples.len(), index)];
        let trace = runner.forward_with_route(p, dict, &case.text, &mut rng, false, fixed)?;
        metrics.record(case, decoded(trace.normalized));
        for decision in &trace.decisions {
            metrics.action_counts[decision.action] += 1;
        }
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_sampling_is_stratified_and_representative() {
        let data = Dataset::new();
        let mut seen = std::collections::HashSet::new();
        let mut first_window_nonzero = 0;
        for i in 0..data.validation.len() {
            let index = validation_case_index(data.validation.len(), i);
            assert_eq!(data.validation[index].op, i % OP_COUNT);
            assert!(seen.insert(index), "repeated held-out sample {index}");
            if i < OP_COUNT && data.validation[index].target != 0 {
                first_window_nonzero += 1;
            }
        }
        assert_eq!(seen.len(), data.validation.len());
        assert!(first_window_nonzero >= OP_COUNT / 2);
        let mut targets = data
            .train
            .iter()
            .map(|case| case.target)
            .collect::<Vec<_>>();
        targets.sort_unstable();
        let baseline = constant_baseline(&data.validation, 80, targets[targets.len() / 2]);
        assert_eq!(baseline.count, 80);
        assert_eq!(baseline.per_op_count, [2; OP_COUNT]);
        assert_eq!(baseline.action_counts, [0; ACTIONS]);
    }

    #[test]
    fn requested_examples_and_40_operations() {
        assert_eq!(operation(0, 10, 9).unwrap().target, 19);
        assert_eq!(operation(3, 1, 0).unwrap().target, 0);
        let dataset = Dataset::new();
        assert_eq!(
            dataset.train.len() + dataset.validation.len(),
            OP_COUNT * 256
        );
        assert!(
            dataset
                .validation
                .iter()
                .all(|case| (case.a * 17 + case.b * 31) % 5 == 0)
        );
        assert!(
            dataset
                .train
                .iter()
                .all(|case| (case.a * 17 + case.b * 31) % 5 != 0)
        );
        let mut signatures = std::collections::BTreeSet::new();
        for op in 0..OP_COUNT {
            assert!(dataset.train.iter().any(|case| case.op == op));
            assert!(dataset.validation.iter().any(|case| case.op == op));
            // The 40 operator labels must denote 40 distinct truth tables
            // over the complete experimental operand domain.
            let signature = (0..=15)
                .flat_map(|a| (0..=15).map(move |b| operation(op, a, b).unwrap().target))
                .collect::<Vec<_>>();
            assert!(signatures.insert(signature), "duplicate operation {op}");
        }
    }

    #[test]
    fn dictionary_and_binary_tokens() {
        let dataset = Dataset::new();
        let dictionary = Dictionary::fit(&dataset.train);
        assert!(dictionary.len() >= 10);
        let ids = dictionary.encode("10 + 9").unwrap();
        assert_eq!(ids.len(), "10 + 9".len());
        let bits = Dictionary::bits(ids[0]);
        let reconstructed = bits
            .iter()
            .take(9)
            .enumerate()
            .fold(0u16, |value, (i, bit)| value | ((*bit as u16) << i));
        assert_eq!(reconstructed, ids[0] + 1);
        assert_eq!(dictionary.state(&ids).iter().sum::<f32>().round() as i32, 1);
        let non_ascii = Dictionary::fit(&[Example {
            op: 0,
            a: 0,
            b: 0,
            text: "é".into(),
            target: 0,
        }]);
        assert_eq!(non_ascii.encode("é").unwrap().len(), 2);
        assert!(non_ascii.encode("a").is_err());
    }

    #[test]
    fn four_true_spatial_rotations_of_two_separate_matrices() {
        let mut first = vec![0.0; WIDTH * WIDTH];
        first[0] = 1.0;
        first[WIDTH - 1] = 2.0;
        first[WIDTH * (WIDTH - 1)] = 3.0;
        first[WIDTH * WIDTH - 1] = 4.0;
        assert_eq!(rotate(&first, 0).unwrap(), first);
        assert_eq!(rotate(&first, 1).unwrap()[WIDTH - 1], 1.0);
        assert_eq!(rotate(&first, 2).unwrap()[WIDTH * WIDTH - 1], 1.0);
        assert_eq!(rotate(&first, 3).unwrap()[WIDTH * (WIDTH - 1)], 1.0);
        let mut rolled = first.clone();
        for _ in 0..4 {
            rolled = rotate(&rolled, 1).unwrap();
        }
        assert_eq!(rolled, first);
        for action in 0..ACTIONS {
            assert_eq!(route_parts(action).unwrap(), (action / 4, action % 4));
        }
        assert!(route_parts(8).is_err());
        for action in 0..ACTIONS {
            let model = routed_core_model(action).unwrap();
            let ir = model.graph().canonical_ir();
            assert!(!ir.contains("rotated_weight"));
            if action % TURNS != 0 {
                assert!(ir.contains("reverse_axis:"));
            }
        }
        assert_eq!(
            argmax_first(&[f32::NAN, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1
        );
        assert_eq!(argmax_first(&[f32::NAN; ACTIONS]), 0);

        let mut rng = Rng::new(7);
        let mut probs = [0.0; ACTIONS];
        probs[1] = 0.5;
        probs[2] = 0.5;
        assert_eq!(choose(&probs, &mut rng, false), 1);
    }

    #[test]
    fn one_semantic_ir_contains_router_core_and_head() {
        let model = architecture_model().unwrap();
        assert_eq!(model.graph().modules().len(), 3);
        assert_eq!(model.outputs().len(), 3);
        assert_eq!(
            model
                .graph()
                .node(model.output("router_logits").unwrap())
                .unwrap()
                .shape
                .dims(),
            &[1, ACTIONS]
        );
        assert_eq!(
            model
                .graph()
                .node(model.output("next_state").unwrap())
                .unwrap()
                .shape
                .dims(),
            &[1, WIDTH]
        );
        assert_eq!(
            model
                .graph()
                .node(model.output("normalized_result").unwrap())
                .unwrap()
                .shape
                .dims(),
            &[1, 1]
        );
        let ir = model.graph().canonical_ir();
        assert!(ir.contains("routing_logits"));
        assert!(ir.contains("permute_axes:1,0"));
        assert!(ir.contains("reverse_axis:"));
        assert!(ir.contains("concat_axis:0"));
        assert!(ir.contains("argmax_axis:1:tie=first"));
        assert!(ir.contains("take_axis:0"));
        assert!(!ir.contains("selected_rotated_weight"));
        assert!(ir.contains("routed_recurrent_transition"));
        assert!(ir.contains("terminal_scalar_readout"));
        assert_eq!(model.state_links().len(), 1);
        let outputs = [
            model.output("router_logits").unwrap(),
            model.output("next_state").unwrap(),
            model.output("normalized_result").unwrap(),
        ];
        let gpu = GpuEngine::default()
            .compile_outputs(model.graph(), &outputs)
            .unwrap();
        let plan = gpu.translation_plan().unwrap().to_text();
        assert!(plan.contains("transpose2d"));
        assert!(plan.contains("argmax_last"));
        assert!(plan.contains("take_axis0"));
        let memory = gpu.memory_lean_source().unwrap();
        assert!(memory.contains("transpose2d_address_bound"));
        assert!(memory.contains("argmax_candidate_address_bound"));
        assert!(memory.contains("take_axis0_address_bound"));
        assert_eq!(model.proof_goals().len(), 0);
    }
}
