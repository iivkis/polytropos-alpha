//! Atomic, bounded research checkpoint. Not a Lean/FP32 certificate.
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Cursor, Read, Write};
use std::path::Path;
use verineural::{AdamW, AdamWCheckpoint, Model, NeuralGraph, Tensor};

const MAGIC: &[u8] = b"VN-EMA4-BPTT-CKPT-v3\0";
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTRIES: usize = 1024;
const MAX_ELEMENTS: usize = 1_000_000;

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Non-cryptographic integrity/corpus fingerprint; not an authenticity proof.
pub fn fingerprint(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

/// Fail-closed *compatibility* fingerprint of the public Semantic IR. The
/// digest is deliberately non-cryptographic; it is not an artifact signature
/// or a proof of GPU/kernel correctness. Ignore transient graph NodeId IDs.
/// Checkpoint identity of the actual training IR and evaluation protocol.
/// This is a non-cryptographic compatibility checksum, not a formal GPU proof.
pub fn architecture_fingerprint(
    model: &Model,
    training_graph: &NeuralGraph,
    seed: u64,
    role_init: &str,
    initial: &[(String, Tensor<f32>)],
    val_limit: usize,
    test_limit: usize,
    eval_every: usize,
    optimizer_backend: &str,
) -> u64 {
    let mut description = format!(
        "ema4-training-graph-v3|seed={seed}|role_init={role_init}|validation={val_limit}|test={test_limit}|eval_every={eval_every}|optimizer={optimizer_backend}\n",
    );
    for (scope, graph) in [("model", model.graph()), ("training", training_graph)] {
        description.push_str(&format!("scope={scope}|policy={:?}\n", graph.policy()));
        for node in graph.nodes() {
            description.push_str(&format!(
                "node={}|op={:?}|shape={:?}|dtype={:?}|domain={:?}|contract={:?}|semantics={:?}|inputs=",
                node.id.index(), node.operation, node.shape, node.dtype, node.domain,
                node.contract, node.semantics,
            ));
            for input in &node.inputs {
                description.push_str(&format!("{},", input.index()));
            }
            description.push('\n');
        }
    }
    for (name, node) in model.outputs() {
        description.push_str(&format!("output={name}:{}\n", node.index()));
    }
    for (name, node) in model.state_links() {
        description.push_str(&format!("next_state={name}:{}\n", node.index()));
    }
    // Node-order alone cannot detect swapped *named* sources of equal shape.
    // Include role, source name, and stable graph-local index explicitly.
    let mut bytes = description.into_bytes();
    for (name, tensor) in initial {
        if let Some((node, role)) = model.source(name) {
            bytes.extend_from_slice(
                format!("source={name}|id={}|role={role:?}\n", node.index()).as_bytes(),
            );
        } else {
            bytes.extend_from_slice(format!("unknown_source={name}\n").as_bytes());
        }
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(tensor.shape().dims().len() as u64).to_le_bytes());
        for &dim in tensor.shape().dims() {
            bytes.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        for &value in tensor.values() {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    fingerprint(&bytes)
}

fn u64w(out: &mut Vec<u8>, n: u64) {
    out.extend_from_slice(&n.to_le_bytes());
}
fn f32w(out: &mut Vec<u8>, x: f32) -> io::Result<()> {
    if !x.is_finite() {
        return Err(invalid("non-finite checkpoint value"));
    }
    out.extend_from_slice(&x.to_bits().to_le_bytes());
    Ok(())
}
fn stringw(out: &mut Vec<u8>, s: &str) -> io::Result<()> {
    if s.is_empty() || s.len() > 512 {
        return Err(invalid("invalid checkpoint key"));
    }
    u64w(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
    Ok(())
}
fn floatsw(out: &mut Vec<u8>, xs: &[f32]) -> io::Result<()> {
    if xs.len() > MAX_ELEMENTS {
        return Err(invalid("oversized checkpoint tensor"));
    }
    u64w(out, xs.len() as u64);
    for &x in xs {
        f32w(out, x)?;
    }
    Ok(())
}
fn tensorsw(out: &mut Vec<u8>, xs: &BTreeMap<String, Tensor<f32>>) -> io::Result<()> {
    if xs.len() > MAX_ENTRIES {
        return Err(invalid("too many checkpoint tensors"));
    }
    u64w(out, xs.len() as u64);
    for (name, t) in xs {
        stringw(out, name)?;
        u64w(out, t.shape().dims().len() as u64);
        for &n in t.shape().dims() {
            u64w(out, n as u64);
        }
        floatsw(out, t.values())?;
    }
    Ok(())
}
struct Reader<'a> {
    data: Cursor<&'a [u8]>,
}
impl Reader<'_> {
    fn u64(&mut self) -> io::Result<u64> {
        let mut b = [0; 8];
        self.data.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn bounded(&mut self, max: usize) -> io::Result<usize> {
        let n = usize::try_from(self.u64()?).map_err(|_| invalid("checkpoint count overflow"))?;
        if n > max {
            return Err(invalid("checkpoint count out of range"));
        }
        Ok(n)
    }
    fn f32(&mut self) -> io::Result<f32> {
        let mut b = [0; 4];
        self.data.read_exact(&mut b)?;
        let value = f32::from_bits(u32::from_le_bytes(b));
        if !value.is_finite() {
            return Err(invalid("non-finite checkpoint value"));
        }
        Ok(value)
    }
    fn name(&mut self) -> io::Result<String> {
        let len = self.bounded(512)?;
        if len == 0 {
            return Err(invalid("empty checkpoint name"));
        }
        let mut bytes = vec![0; len];
        self.data.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| invalid("invalid checkpoint UTF-8"))
    }
    fn floats(&mut self) -> io::Result<Vec<f32>> {
        let n = self.bounded(MAX_ELEMENTS)?;
        (0..n).map(|_| self.f32()).collect()
    }
    fn tensors(&mut self) -> io::Result<BTreeMap<String, Tensor<f32>>> {
        let count = self.bounded(MAX_ENTRIES)?;
        let mut out = BTreeMap::new();
        for _ in 0..count {
            let name = self.name()?;
            let rank = self.bounded(6)?;
            if rank == 0 {
                return Err(invalid("invalid checkpoint rank"));
            }
            let dims = (0..rank)
                .map(|_| self.bounded(MAX_ELEMENTS))
                .collect::<io::Result<Vec<_>>>()?;
            let expected = dims
                .iter()
                .try_fold(1usize, |p, &n| p.checked_mul(n))
                .ok_or_else(|| invalid("checkpoint tensor length overflow"))?;
            let values = self.floats()?;
            if expected != values.len() {
                return Err(invalid("checkpoint shape mismatch"));
            }
            let tensor =
                Tensor::from_vec(dims, values).map_err(|_| invalid("invalid checkpoint tensor"))?;
            if out.insert(name, tensor).is_some() {
                return Err(invalid("duplicate checkpoint tensor"));
            }
        }
        Ok(out)
    }
}

fn same_schema(a: &BTreeMap<String, Tensor<f32>>, b: &BTreeMap<String, Tensor<f32>>) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|((an, av), (bn, bv))| {
            an == bn && av.shape() == bv.shape() && av.values().iter().all(|x| x.is_finite())
        })
}

#[derive(Clone, Debug)]
pub struct Ema4Checkpoint {
    pub completed_windows: usize,
    pub bptt: usize,
    pub vocab: usize,
    pub corpus_fingerprint: u64,
    pub architecture_fingerprint: u64,
    pub validation_limit: usize,
    pub test_limit: usize,
    pub validation_every: usize,
    /// Lowest validation BPC recorded at completed evaluation boundaries.
    pub best_validation_bpc: Option<f32>,
    pub parameters: BTreeMap<String, Tensor<f32>>,
    pub states: BTreeMap<String, Tensor<f32>>,
    pub optimizer: AdamWCheckpoint,
}
impl Ema4Checkpoint {
    pub fn validate(
        &self,
        bptt: usize,
        vocab: usize,
        corpus: u64,
        architecture: u64,
        val_limit: usize,
        test_limit: usize,
        eval_every: usize,
        max_windows: usize,
        params: &BTreeMap<String, Tensor<f32>>,
        states: &BTreeMap<String, Tensor<f32>>,
    ) -> io::Result<()> {
        if self.bptt != bptt
            || self.vocab != vocab
            || self.corpus_fingerprint != corpus
            || self.architecture_fingerprint != architecture
            || self.validation_limit != val_limit
            || self.test_limit != test_limit
            || self.validation_every != eval_every
            || self
                .best_validation_bpc
                .is_some_and(|v| !v.is_finite() || v < 0.)
            || self.completed_windows > max_windows
            || u64::try_from(self.completed_windows).ok() != Some(self.optimizer.steps)
            || !same_schema(&self.parameters, params)
            || !same_schema(&self.states, states)
        {
            return Err(invalid(
                "checkpoint incompatible with architecture/seed/corpus/BPTT/window cursor",
            ));
        }
        let o = &self.optimizer;
        if o.learning_rate.to_bits() != 0.002f32.to_bits()
            || o.weight_decay.to_bits() != 0.01f32.to_bits()
            || o.beta1.to_bits() != 0.9f32.to_bits()
            || o.beta2.to_bits() != 0.999f32.to_bits()
            || o.epsilon.to_bits() != 1e-8f32.to_bits()
        {
            return Err(invalid("checkpoint AdamW configuration mismatch"));
        }
        AdamW::from_checkpoint(o.clone(), &self.parameters)
            .map_err(|_| invalid("invalid checkpoint AdamW moments"))?;
        Ok(())
    }

    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut out = MAGIC.to_vec();
        for n in [
            self.completed_windows as u64,
            self.bptt as u64,
            self.vocab as u64,
            self.corpus_fingerprint,
            self.architecture_fingerprint,
            self.validation_limit as u64,
            self.test_limit as u64,
            self.validation_every as u64,
        ] {
            u64w(&mut out, n);
        }
        if let Some(best) = self.best_validation_bpc {
            u64w(&mut out, 1);
            f32w(&mut out, best)?;
        } else {
            u64w(&mut out, 0);
        }
        tensorsw(&mut out, &self.parameters)?;
        tensorsw(&mut out, &self.states)?;
        let o = &self.optimizer;
        for n in [o.learning_rate, o.weight_decay, o.beta1, o.beta2, o.epsilon] {
            f32w(&mut out, n)?;
        }
        u64w(&mut out, o.steps);
        if o.moments.len() > MAX_ENTRIES {
            return Err(invalid("too many AdamW entries"));
        }
        u64w(&mut out, o.moments.len() as u64);
        for (name, (first, second)) in &o.moments {
            stringw(&mut out, name)?;
            floatsw(&mut out, first)?;
            floatsw(&mut out, second)?;
        }
        let hash = fingerprint(&out);
        u64w(&mut out, hash);
        if out.len() > MAX_BYTES {
            return Err(invalid("checkpoint size too large"));
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() > MAX_BYTES || bytes.len() < MAGIC.len() + 8 {
            return Err(invalid("invalid checkpoint byte length"));
        }
        let (body, hash) = bytes.split_at(bytes.len() - 8);
        let actual = u64::from_le_bytes(hash.try_into().map_err(|_| invalid("invalid checksum"))?);
        if fingerprint(body) != actual {
            return Err(invalid("checkpoint checksum mismatch"));
        }
        let mut reader = Reader {
            data: Cursor::new(body),
        };
        let mut magic = vec![0; MAGIC.len()];
        reader.data.read_exact(&mut magic)?;
        if magic.as_slice() != MAGIC {
            return Err(invalid("unsupported checkpoint version"));
        }
        let completed_windows = reader.bounded(1_000_000)?;
        let bptt = reader.bounded(16)?;
        let vocab = reader.bounded(4096)?;
        let corpus_fingerprint = reader.u64()?;
        let architecture_fingerprint = reader.u64()?;
        let validation_limit = reader.bounded(100_000)?;
        let test_limit = reader.bounded(100_000)?;
        let validation_every = reader.bounded(1_000_000)?;
        let best_validation_bpc = match reader.bounded(1)? {
            0 => None,
            1 => Some(reader.f32()?),
            _ => return Err(invalid("invalid validation metric flag")),
        };
        let parameters = reader.tensors()?;
        let states = reader.tensors()?;
        let learning_rate = reader.f32()?;
        let weight_decay = reader.f32()?;
        let beta1 = reader.f32()?;
        let beta2 = reader.f32()?;
        let epsilon = reader.f32()?;
        let steps = reader.u64()?;
        let count = reader.bounded(MAX_ENTRIES)?;
        let mut moments = BTreeMap::new();
        for _ in 0..count {
            let name = reader.name()?;
            let first = reader.floats()?;
            let second = reader.floats()?;
            if moments.insert(name, (first, second)).is_some() {
                return Err(invalid("duplicate checkpoint optimizer state"));
            }
        }
        if reader.data.position() != body.len() as u64 {
            return Err(invalid("trailing checkpoint data"));
        }
        Ok(Self {
            completed_windows,
            bptt,
            vocab,
            corpus_fingerprint,
            architecture_fingerprint,
            validation_limit,
            test_limit,
            validation_every,
            best_validation_bpc,
            parameters,
            states,
            optimizer: AdamWCheckpoint {
                learning_rate,
                weight_decay,
                beta1,
                beta2,
                epsilon,
                steps,
                moments,
            },
        })
    }

    pub fn read(path: &Path) -> io::Result<Self> {
        Self::decode(&fs::read(path)?)
    }

    /// Atomic replacement: completed temporary file and directory are synced.
    pub fn write_atomic(&self, path: &Path) -> io::Result<()> {
        let bytes = self.encode()?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .ok_or_else(|| invalid("checkpoint path has no filename"))?;
        let temp = parent.join(format!(
            ".{}.tmp-{}",
            name.to_string_lossy(),
            std::process::id()
        ));
        let result = (|| -> io::Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temp, path)?;
            fs::File::open(parent)?.sync_all()
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(n: f32) -> Tensor<f32> {
        Tensor::from_slice(vec![1, 1], &[n]).unwrap()
    }

    #[test]
    fn checkpoint_roundtrip_corruption_schema_and_optimizer_continuity() {
        let expected = BTreeMap::from([("a".into(), scalar(1.))]);
        let states = BTreeMap::from([("memory".into(), scalar(0.25))]);
        let gradients = BTreeMap::from([("a".into(), scalar(0.3))]);
        let mut optimizer = AdamW::new(0.002, 0.01).unwrap();
        let mut trained = expected.clone();
        optimizer.step(&mut trained, &gradients).unwrap();
        let snapshot = Ema4Checkpoint {
            completed_windows: 1,
            bptt: 2,
            vocab: 104,
            corpus_fingerprint: fingerprint(b"corpus"),
            architecture_fingerprint: 12345,
            validation_limit: 64,
            test_limit: 64,
            validation_every: 2,
            best_validation_bpc: Some(3.4),
            parameters: trained.clone(),
            states: states.clone(),
            optimizer: optimizer.checkpoint(),
        };
        snapshot
            .validate(
                2,
                104,
                fingerprint(b"corpus"),
                12345,
                64,
                64,
                2,
                4,
                &expected,
                &states,
            )
            .unwrap();
        let bytes = snapshot.encode().unwrap();
        let restored = Ema4Checkpoint::decode(&bytes).unwrap();
        assert_eq!(restored.best_validation_bpc, Some(3.4));
        assert_eq!(restored.parameters, trained);
        assert_eq!(restored.states, states);
        assert_eq!(restored.optimizer, optimizer.checkpoint());
        assert!(
            restored
                .validate(
                    3,
                    104,
                    fingerprint(b"corpus"),
                    12345,
                    64,
                    64,
                    2,
                    4,
                    &expected,
                    &states
                )
                .is_err()
        );
        assert!(
            restored
                .validate(
                    2,
                    104,
                    fingerprint(b"changed"),
                    12345,
                    64,
                    64,
                    2,
                    4,
                    &expected,
                    &states
                )
                .is_err()
        );
        assert!(
            restored
                .validate(
                    2,
                    104,
                    fingerprint(b"corpus"),
                    12345,
                    64,
                    64,
                    2,
                    0,
                    &expected,
                    &states
                )
                .is_err()
        );
        assert!(
            restored
                .validate(
                    2,
                    104,
                    fingerprint(b"corpus"),
                    12346,
                    64,
                    64,
                    2,
                    4,
                    &expected,
                    &states
                )
                .is_err()
        );
        let mut bad = bytes.clone();
        bad[9] ^= 0x7f;
        assert!(Ema4Checkpoint::decode(&bad).is_err());
        assert!(Ema4Checkpoint::decode(&bytes[..bytes.len() - 2]).is_err());
        let mut trailing = bytes.clone();
        trailing.extend_from_slice(b"noise");
        assert!(Ema4Checkpoint::decode(&trailing).is_err());

        let mut continued = trained.clone();
        optimizer.step(&mut continued, &gradients).unwrap();
        let mut resumed = trained;
        let mut resumed_optimizer = AdamW::from_checkpoint(restored.optimizer, &resumed).unwrap();
        resumed_optimizer.step(&mut resumed, &gradients).unwrap();
        assert_eq!(resumed, continued);
        assert_eq!(optimizer.checkpoint(), resumed_optimizer.checkpoint());
    }

    #[test]
    fn architecture_checksum_is_stable_and_rejects_changed_ir_seed_and_role_init() {
        use crate::ema4::{Ema4Weights, MatrixMode, RoleInit, architecture_model_for};
        let weights = Ema4Weights::deterministic_seeded(16, 17).unwrap();
        let w = |mode, role_init| weights.bindings_with_init(0, mode, role_init).unwrap();
        let shared = architecture_model_for(16, MatrixMode::TorusShared).unwrap();
        let shared_again = architecture_model_for(16, MatrixMode::TorusShared).unwrap();
        let independent = architecture_model_for(16, MatrixMode::Independent).unwrap();
        let initial = w(MatrixMode::TorusShared, RoleInit::Matched);
        let model_graph = shared.graph();
        let id = architecture_fingerprint(
            &shared,
            model_graph,
            17,
            "matched",
            &initial,
            64,
            64,
            2,
            "GPU_IR",
        );
        assert_eq!(
            id,
            architecture_fingerprint(
                &shared_again,
                shared_again.graph(),
                17,
                "matched",
                &initial,
                64,
                64,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                18,
                "matched",
                &initial,
                64,
                64,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "diverse",
                &initial,
                64,
                64,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &independent,
                independent.graph(),
                17,
                "matched",
                &w(MatrixMode::Independent, RoleInit::Matched),
                64,
                64,
                2,
                "GPU_IR"
            )
        );
        let mut altered = initial.clone();
        let i = altered
            .iter()
            .position(|(name, _)| name == "one_scalar")
            .unwrap();
        altered[i].1 = scalar(0.5);
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "matched",
                &altered,
                64,
                64,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "matched",
                &initial,
                32,
                64,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "matched",
                &initial,
                64,
                32,
                2,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "matched",
                &initial,
                64,
                64,
                4,
                "GPU_IR"
            )
        );
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                model_graph,
                17,
                "matched",
                &initial,
                64,
                64,
                2,
                "HOST"
            )
        );
        let mut changed_training_graph = model_graph.clone();
        let source = shared.source("weight_0").unwrap().0;
        changed_training_graph.mul(source, source).unwrap();
        assert_ne!(
            id,
            architecture_fingerprint(
                &shared,
                &changed_training_graph,
                17,
                "matched",
                &initial,
                64,
                64,
                2,
                "GPU_IR"
            )
        );
    }

    #[test]
    fn checkpoint_rejects_legacy_v2_and_other_matrix_mode_even_with_valid_schema() {
        use crate::ema4::{Ema4Weights, MatrixMode, RoleInit, architecture_model_for};
        let weights = Ema4Weights::deterministic(16).unwrap();
        let model = architecture_model_for(16, MatrixMode::TorusShared).unwrap();
        let bindings = weights
            .bindings_with_init(0, MatrixMode::TorusShared, RoleInit::Matched)
            .unwrap();
        let identity = architecture_fingerprint(
            &model,
            model.graph(),
            0,
            "matched",
            &bindings,
            64,
            64,
            2,
            "GPU_IR",
        );
        let params = weights
            .trainable_parameters_for(MatrixMode::TorusShared)
            .unwrap();
        let snapshot = Ema4Checkpoint {
            completed_windows: 0,
            bptt: 2,
            vocab: 16,
            corpus_fingerprint: fingerprint(b"corpus"),
            architecture_fingerprint: identity,
            validation_limit: 64,
            test_limit: 64,
            validation_every: 2,
            best_validation_bpc: None,
            parameters: params.clone(),
            states: BTreeMap::new(),
            optimizer: AdamW::new(0.002, 0.01).unwrap().checkpoint(),
        };
        let encoded = snapshot.encode().unwrap();
        let decoded = Ema4Checkpoint::decode(&encoded).unwrap();
        decoded
            .validate(
                2,
                16,
                fingerprint(b"corpus"),
                identity,
                64,
                64,
                2,
                4,
                &params,
                &BTreeMap::new(),
            )
            .unwrap();
        assert!(
            decoded
                .validate(
                    2,
                    16,
                    fingerprint(b"corpus"),
                    identity + 1,
                    64,
                    64,
                    2,
                    4,
                    &params,
                    &BTreeMap::new()
                )
                .is_err()
        );
        assert!(
            decoded
                .validate(
                    2,
                    16,
                    fingerprint(b"corpus"),
                    identity,
                    32,
                    64,
                    2,
                    4,
                    &params,
                    &BTreeMap::new()
                )
                .is_err()
        );
        assert!(
            decoded
                .validate(
                    2,
                    16,
                    fingerprint(b"corpus"),
                    identity,
                    64,
                    32,
                    2,
                    4,
                    &params,
                    &BTreeMap::new()
                )
                .is_err()
        );
        assert!(
            decoded
                .validate(
                    2,
                    16,
                    fingerprint(b"corpus"),
                    identity,
                    64,
                    64,
                    4,
                    4,
                    &params,
                    &BTreeMap::new()
                )
                .is_err()
        );
        let other = architecture_model_for(16, MatrixMode::Independent).unwrap();
        let other_initial = weights
            .bindings_with_init(0, MatrixMode::Independent, RoleInit::Matched)
            .unwrap();
        let other_id = architecture_fingerprint(
            &other,
            other.graph(),
            0,
            "matched",
            &other_initial,
            64,
            64,
            2,
            "GPU_IR",
        );
        let other_params = weights
            .trainable_parameters_for(MatrixMode::Independent)
            .unwrap();
        assert!(
            decoded
                .validate(
                    2,
                    16,
                    fingerprint(b"corpus"),
                    other_id,
                    64,
                    64,
                    2,
                    4,
                    &other_params,
                    &BTreeMap::new()
                )
                .is_err()
        );
        let mut old_magic = encoded.clone();
        old_magic[..MAGIC.len()].copy_from_slice(b"VN-EMA4-BPTT-CKPT-v2\0");
        let checksum = fingerprint(&old_magic[..old_magic.len() - 8]);
        let n = old_magic.len();
        old_magic[n - 8..].copy_from_slice(&checksum.to_le_bytes());
        assert!(Ema4Checkpoint::decode(&old_magic).is_err());
    }

    #[test]
    fn atomic_file_roundtrip() {
        let params = BTreeMap::from([("a".into(), scalar(0.25))]);
        let checkpoint = Ema4Checkpoint {
            completed_windows: 0,
            bptt: 2,
            vocab: 104,
            corpus_fingerprint: 17,
            architecture_fingerprint: 9876,
            validation_limit: 64,
            test_limit: 64,
            validation_every: 2,
            best_validation_bpc: None,
            parameters: params.clone(),
            states: BTreeMap::new(),
            optimizer: AdamW::new(0.002, 0.01).unwrap().checkpoint(),
        };
        let path = std::env::temp_dir().join(format!(
            "vn_ema4_checkpoint_test_{}_{}.bin",
            std::process::id(),
            fingerprint(b"fixed")
        ));
        checkpoint.write_atomic(&path).unwrap();
        let loaded = Ema4Checkpoint::read(&path).unwrap();
        assert_eq!(loaded.parameters, params);
        let _ = fs::remove_file(&path);
    }
}
