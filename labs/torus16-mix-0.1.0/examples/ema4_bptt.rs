//! Experimental all-parameter finite-window BPTT for four-layer EMA4-LM.
//! Corpus defaults to this lab's checked-in research snapshot.
//! Softmax CE cotangents and VJPs share one CUDA Graph replay per window;
//! CE reporting and AdamW are on host. CUDA FP32 parity is not certified. Validation is reset-state streaming BPC.
//! This is not equivalence with the trained handoff PyTorch checkpoint.
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;
use torus16_mix_lab::corpus::CharCorpus;
use torus16_mix_lab::ema4::{Ema4Weights, MatrixMode, RoleInit, architecture_model_for};
use torus16_mix_lab::ema4_checkpoint::{
    CheckpointRequirements, Ema4Checkpoint, FingerprintContext, architecture_fingerprint,
    fingerprint,
};
use verineural::{
    AdamW, ArgMaxTieBreak, Domain, GpuAdamWPlan, GpuCodegenOptLevel, GpuEngine, GpuLaunchConfig,
    Model, ModelExecution, NodeId, ReluFusion, Shape, SourceRole, Tensor, differentiate_with_seeds,
    unroll_model,
};

fn onehot(token: usize, width: usize) -> Result<Tensor<f32>, Box<dyn Error>> {
    if token >= width {
        return Err("token index outside charset".into());
    }
    let mut values = vec![0.; width];
    values[token] = 1.;
    Ok(Tensor::from_vec(vec![1, width], values)?)
}
fn cross_entropy(logits: &[f32], target: usize) -> Result<(f32, Vec<f32>), Box<dyn Error>> {
    if target >= logits.len() || logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("invalid logits/target".into());
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits.iter().map(|v| (v - max).exp()).collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0. {
        return Err("softmax overflow".into());
    }
    let loss = max + sum.ln() - logits[target];
    let grad = exp
        .iter()
        .enumerate()
        .map(|(i, v)| v / sum - if i == target { 1. } else { 0. })
        .collect::<Vec<_>>();
    if !loss.is_finite() {
        return Err("non-finite CE".into());
    }
    Ok((loss, grad))
}
fn option_usize(
    name: &str,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, Box<dyn Error>> {
    let value = std::env::var(name)
        .ok()
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(default);
    if value < min || value > max {
        return Err(format!("{name} must be in {min}..={max}").into());
    }
    Ok(value)
}
/// Shared, read-only evaluation inputs for validation and final test.
struct HoldoutContext<'a> {
    engine: &'a GpuEngine,
    model: &'a Model,
    weights: &'a Ema4Weights,
    mode: MatrixMode,
    role_init: RoleInit,
    corpus: &'a CharCorpus,
}

/// Each evaluation starts with zero EMA/Conv9 state. The caller selects a
/// disjoint validation or test interval; no holdout tokens are used to train.
fn holdout_bpc(
    context: &HoldoutContext<'_>,
    params: &BTreeMap<String, Tensor<f32>>,
    interval: Range<usize>,
) -> Result<(f64, usize), Box<dyn Error>> {
    let HoldoutContext {
        engine,
        model,
        weights,
        mode,
        role_init,
        corpus,
    } = context;
    if interval.start >= interval.end || interval.end >= corpus.validation.len() {
        return Err("holdout interval outside validation corpus".into());
    }
    let count = interval.len();
    let mut bindings = weights.bindings_with_init(0, *mode, *role_init)?;
    for (name, tensor) in &mut bindings {
        if let Some(updated) = params.get(name) {
            *tensor = updated.clone();
        }
    }
    let dir = std::env::temp_dir().join(format!("vn_ema4_holdout_{}", std::process::id()));
    let mut execution = engine.prepare_model(model, bindings, &dir)?;
    let mut ce = 0f64;
    for t in interval {
        execution.set_input(
            "token",
            &onehot(corpus.validation[t], corpus.alphabet.len())?,
        )?;
        execution.step()?;
        let logits = execution.output("logits")?;
        ce += f64::from(cross_entropy(logits.values(), corpus.validation[t + 1])?.0);
    }
    Ok((ce / (count as f64 * std::f64::consts::LN_2), count))
}

/// Opt-in reuse of a holdout inference CUDA Graph across scoring intervals.
/// Parameter and state assignments follow the original fresh model bindings.
struct ReusedHoldout<'a> {
    execution: ModelExecution,
    corpus: &'a CharCorpus,
    reset_states: Vec<(String, Tensor<f32>)>,
    initial_parameters: Vec<(String, Tensor<f32>)>,
    onehots: Vec<Tensor<f32>>,
}

impl<'a> ReusedHoldout<'a> {
    fn prepare(context: &HoldoutContext<'a>) -> Result<Self, Box<dyn Error>> {
        let initial = context
            .weights
            .bindings_with_init(0, context.mode, context.role_init)?;
        // Reconstruct the execution slice through the public Model API.
        // Include next-state roots and the writable state sources, even if a
        // next-state computation does not read its previous value.
        let mut roots = context
            .model
            .outputs()
            .values()
            .copied()
            .collect::<Vec<_>>();
        roots.extend(context.model.state_links().values().copied());
        for name in context.model.state_links().keys() {
            let (state, role) = context
                .model
                .source(name)
                .ok_or_else(|| format!("unknown holdout state source: {name}"))?;
            if role != SourceRole::State {
                return Err(format!("holdout state {name} does not have state role").into());
            }
            roots.push(state);
        }
        let mut seen = BTreeSet::new();
        roots.retain(|id| seen.insert(id.index()));
        // Restrict updates to the source tensors actually materialized by
        // ModelExecution, not every declared but potentially unused parameter.
        let active = context
            .engine
            .compile_outputs(context.model.graph(), &roots)?
            .source_inputs()
            .iter()
            .map(|id| id.index())
            .collect::<BTreeSet<_>>();
        let mut reset_states = Vec::new();
        let mut initial_parameters = Vec::new();
        for (name, tensor) in &initial {
            if let Some((node, role)) = context.model.source(name) {
                match role {
                    SourceRole::State => reset_states.push((name.clone(), tensor.clone())),
                    SourceRole::Parameter if active.contains(&node.index()) => {
                        initial_parameters.push((name.clone(), tensor.clone()));
                    }
                    _ => {}
                }
            }
        }
        let dir =
            std::env::temp_dir().join(format!("vn_ema4_holdout_cached_{}", std::process::id()));
        let execution = context.engine.prepare_model(context.model, initial, &dir)?;
        let onehots = (0..context.corpus.alphabet.len())
            .map(|token| onehot(token, context.corpus.alphabet.len()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            execution,
            corpus: context.corpus,
            reset_states,
            initial_parameters,
            onehots,
        })
    }

    fn score(
        &mut self,
        params: &BTreeMap<String, Tensor<f32>>,
        interval: Range<usize>,
    ) -> Result<(f64, usize), Box<dyn Error>> {
        if interval.start >= interval.end || interval.end >= self.corpus.validation.len() {
            return Err("holdout interval outside validation corpus".into());
        }
        // Missing parameters use initialization, just as fresh model.bind.
        for (name, initial) in &self.initial_parameters {
            self.execution
                .set_parameter(name, params.get(name).unwrap_or(initial))?;
        }
        // Important: no recurrent information may leak between holdout runs.
        for (name, value) in &self.reset_states {
            self.execution.set_state(name, value)?;
        }
        let count = interval.len();
        let mut ce = 0f64;
        for t in interval {
            self.execution
                .set_input("token", &self.onehots[self.corpus.validation[t]])?;
            self.execution.step()?;
            let logits = self.execution.output("logits")?;
            ce += f64::from(cross_entropy(logits.values(), self.corpus.validation[t + 1])?.0);
        }
        Ok((ce / (count as f64 * std::f64::consts::LN_2), count))
    }
}

fn score_holdout(
    cached: &mut Option<ReusedHoldout<'_>>,
    context: &HoldoutContext<'_>,
    params: &BTreeMap<String, Tensor<f32>>,
    interval: Range<usize>,
) -> Result<(f64, usize), Box<dyn Error>> {
    match cached {
        Some(evaluator) => evaluator.score(params, interval),
        None => holdout_bpc(context, params, interval),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // The research handoff corpus is versioned with this lab. A custom
    // TORUS16_CORPUS remains optional for explicit alternative experiments.
    let corpus_path = std::env::var_os("TORUS16_CORPUS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/corpus_snapshot.txt")
        });
    let corpus = CharCorpus::read(&corpus_path)?;
    let window_len = option_usize("TORUS16_BPTT", 2, 2, 8)?;
    let train_windows = option_usize("TORUS16_TRAIN_WINDOWS", 2, 1, 1_000_000)?;
    let val_limit = option_usize("TORUS16_VAL_LIMIT", 64, 2, 100000)?;
    let test_limit = option_usize("TORUS16_TEST_LIMIT", 64, 2, 100000)?;
    if val_limit
        .checked_add(test_limit)
        .and_then(|sum| sum.checked_add(2))
        .is_none_or(|needed| needed > corpus.validation.len())
    {
        return Err(
            "validation and test subsets must be disjoint and fit in the held-out corpus".into(),
        );
    }
    let mode_name = std::env::var("TORUS16_MATRIX_MODE").unwrap_or_else(|_| "torus_shared".into());
    let matrix_mode = MatrixMode::parse(&mode_name)
        .ok_or("TORUS16_MATRIX_MODE must be independent, torus_shared, role_delta_2, role_delta_4 or role_delta_8")?;
    let role_init_name = std::env::var("TORUS16_ROLE_INIT").unwrap_or_else(|_| "matched".into());
    let role_init =
        RoleInit::parse(&role_init_name).ok_or("TORUS16_ROLE_INIT must be matched or diverse")?;
    if matrix_mode != MatrixMode::Independent && role_init != RoleInit::Matched {
        return Err("Shared/Role-Delta modes require TORUS16_ROLE_INIT=matched".into());
    }
    let matrix_seed = std::env::var("TORUS16_MATRIX_SEED")
        .ok()
        .map(|seed| seed.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    let eval_every = option_usize("TORUS16_EVAL_EVERY", 4096, 1, 131072)?;
    let checkpoint_every = option_usize("TORUS16_CHECKPOINT_EVERY", 128, 1, 4096)?;
    let checkpoint_path = std::env::var_os("TORUS16_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            match matrix_mode {
                MatrixMode::TorusShared => PathBuf::from("target/torus16-ema4-bptt.ckpt"),
                MatrixMode::Independent => PathBuf::from("target/torus16-ema4-independent-bptt.ckpt"),
                _ => PathBuf::from(format!("target/torus16-ema4-{}-bptt.ckpt", matrix_mode.as_str())),
            }
        });
    let fresh = std::env::var_os("TORUS16_FRESH").is_some();
    let corpus_hash = fingerprint(&fs::read(&corpus_path)?);
    if train_windows
        .checked_mul(window_len)
        .ok_or("training count overflow")?
        >= corpus.train.len()
    {
        return Err("not enough chronological train characters for requested windows".into());
    }
    let compile_clock = Instant::now();
    let vocab = corpus.alphabet.len();
    // Immutable one-hot host tensors: do not allocate two fresh vectors for
    // every BPTT token and target in the hot path.
    let onehots = (0..vocab)
        .map(|token| onehot(token, vocab))
        .collect::<Result<Vec<_>, _>>()?;
    let mut weights = Ema4Weights::deterministic_seeded(vocab, matrix_seed)?;
    if matrix_mode.role_delta_rank().is_some() {
        let scale = std::env::var("TORUS16_DELTA_SCALE")
            .ok()
            .map(|value| value.parse::<f32>())
            .transpose()?
            .unwrap_or(1.0);
        if !scale.is_finite() || scale <= 0.0 || scale > 16.0 {
            return Err("TORUS16_DELTA_SCALE must be finite and within (0,16]".into());
        }
        weights.delta_scale = scale;
    } else if std::env::var_os("TORUS16_DELTA_SCALE").is_some() {
        return Err("TORUS16_DELTA_SCALE is only supported in Role-Delta modes".into());
    }
    let model = architecture_model_for(vocab, matrix_mode)?;
    let initial_bindings = weights.bindings_with_init(0, matrix_mode, role_init)?;
    let trainable_initial = weights.trainable_parameters_with_init(matrix_mode, role_init)?;
    let parameter_count: usize = trainable_initial.values().map(|v| v.values().len()).sum();
    let window = unroll_model(&model, "logits", window_len)?;
    let named_parameter = window.parameters();
    let names = trainable_initial.keys().cloned().collect::<Vec<_>>();
    let targets = names
        .iter()
        .map(|name| {
            named_parameter
                .get(name)
                .copied()
                .ok_or_else(|| format!("training parameter {name} absent from unrolled model"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Graph-provided CE cotangents are terminal VJP adjoints, not
    // differentiated as ordinary logits functions (no spurious Hessian).
    let mut train_graph = window.graph().clone();
    let ones_id = train_graph.input(Shape::new(vec![vocab, 1])?, Domain::Finite);
    let scale_id = train_graph.input(Shape::new(vec![1, 1])?, Domain::Finite);
    let row = Shape::new(vec![1, vocab])?;
    let scale_row = train_graph.broadcast_to(scale_id, row.clone())?;
    let mut supervised = Vec::<(NodeId, NodeId)>::new();
    for (&logits, &target) in window.outputs().iter().zip(window.cotangents()) {
        // Stable softmax max = take(transpose(logits), argmax(logits)).
        let max_index = train_graph.argmax(logits, 1, ArgMaxTieBreak::First)?;
        let column = train_graph.permute(logits, &[1, 0])?;
        let maximum = train_graph.take(column, max_index, 0)?;
        let maximum = train_graph.reshape(maximum, Shape::new(vec![1, 1])?)?;
        let max_row = train_graph.broadcast_to(maximum, row.clone())?;
        let shifted = train_graph.sub(logits, max_row)?;
        let exps = train_graph.exp(shifted)?;
        let denom = train_graph.matmul(exps, ones_id)?;
        let denom_row = train_graph.broadcast_to(denom, row.clone())?;
        let normalized = train_graph.div(exps, denom_row)?;
        let difference = train_graph.sub(normalized, target)?;
        let seed = train_graph.mul(difference, scale_row)?;
        supervised.push((logits, seed));
    }
    let reverse = differentiate_with_seeds(&train_graph, &supervised, &targets)?;
    let gradients = targets
        .iter()
        .map(|&p| {
            reverse
                .gradient_of(p)
                .ok_or("missing differentiated parameter")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let gpu_adamw_requested = std::env::var_os("TORUS16_GPU_ADAMW").is_some();
    // Explicit, checkpointed fine-tuning experiment. Do not change the
    // established 0.002 baseline or silently reset AdamW moments/steps.
    let lr_decay_5x = std::env::var_os("TORUS16_LR_DECAY_5X").is_some();
    let baseline_lr = 0.002f32;
    let training_lr = if lr_decay_5x { 0.0004f32 } else { baseline_lr };
    if lr_decay_5x && !gpu_adamw_requested {
        return Err("TORUS16_LR_DECAY_5X requires TORUS16_GPU_ADAMW".into());
    }
    let metric_every = option_usize("TORUS16_METRIC_EVERY", 1, 1, 1_000_000)?;
    if metric_every > 1 && !gpu_adamw_requested {
        return Err("TORUS16_METRIC_EVERY>1 requires TORUS16_GPU_ADAMW=1: host optimizer still needs gradient readback".into());
    }
    let llc_opt = option_usize("TORUS16_LLC_OPT", 0, 0, 2)?;
    let codegen_opt_level = match llc_opt {
        0 => GpuCodegenOptLevel::O0,
        2 => GpuCodegenOptLevel::O2,
        _ => return Err("TORUS16_LLC_OPT supports only 0 or 2".into()),
    };
    let threads_per_block = option_usize("TORUS16_THREADS_PER_BLOCK", 256, 64, 512)?;
    if !matches!(threads_per_block, 64 | 128 | 256 | 512) {
        return Err("TORUS16_THREADS_PER_BLOCK must be one of 64,128,256,512".into());
    }
    let verify_fusion = std::env::var_os("TORUS16_VERIFY_FUSION").is_some();
    let mut selected = window.outputs().to_vec();
    // CPU AdamW needs host gradients; the explicit fusion-parity oracle
    // still observes them. Ordinary GPU training keeps them internal.
    if !gpu_adamw_requested || verify_fusion {
        selected.extend(gradients.iter().copied());
    }
    let train_output_count = selected.len();
    let terminal_outputs = window
        .terminal_states()
        .values()
        .copied()
        .collect::<Vec<_>>();
    // Next states are selected for device-to-device feedback, not for
    // readback on every training window.
    selected.extend(terminal_outputs.iter().copied());
    let state_feedback = window
        .initial_states()
        .iter()
        .map(|(name, &source)| {
            window
                .terminal_states()
                .get(name)
                .copied()
                .map(|next| (source, next))
                .ok_or_else(|| format!("missing terminal state for {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Retain the pristine named bindings for the compatibility fingerprint.
    let all = initial_bindings.clone();
    let expected_states = window
        .initial_states()
        .keys()
        .map(|name| {
            all.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| (name.clone(), value.clone()))
                .ok_or_else(|| format!("missing initial recurrent state: {name}"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    // Construct the *actual* training graph first, then validate its
    // full fingerprint against the checkpoint before any GPU replay.
    let pending_checkpoint = if !fresh && checkpoint_path.exists() {
        Some(Ema4Checkpoint::read(&checkpoint_path)?)
    } else {
        None
    };
    if lr_decay_5x && pending_checkpoint.is_none() {
        return Err(
            "TORUS16_LR_DECAY_5X requires a resumable checkpoint; fresh training is forbidden"
                .into(),
        );
    }
    if let Some(saved) = pending_checkpoint.as_ref() {
        let saved_lr = saved.optimizer.learning_rate;
        let matches_requested = saved_lr.to_bits() == training_lr.to_bits();
        let approved_transition = lr_decay_5x && saved_lr.to_bits() == baseline_lr.to_bits();
        if !matches_requested && !approved_transition {
            return Err(format!(
                "AdamW learning rate mismatch: checkpoint={saved_lr} requested={training_lr}; only explicit 0.002 -> 0.0004 transition is supported"
            ).into());
        }
    }
    let (mut params, states, mut optimizer, first_window, mut best_validation) =
        if let Some(ckpt) = pending_checkpoint.as_ref() {
            let mut optimizer_state = ckpt.optimizer.clone();
            optimizer_state.learning_rate = training_lr;
            let opt = AdamW::from_checkpoint(optimizer_state, &ckpt.parameters)?;
            (
                ckpt.parameters.clone(),
                ckpt.states.clone(),
                opt,
                ckpt.completed_windows,
                ckpt.best_validation_bpc,
            )
        } else {
            (
                trainable_initial.clone(),
                expected_states.clone(),
                AdamW::new(training_lr, 0.01)?,
                0usize,
                None,
            )
        };
    // Experimental architecture-neutral GPU optimizer; defaults to the
    // proven host path until parity and end-to-end throughput are checked.
    let mut execution_graph = reverse.graph().clone();
    let gpu_optimizer = if gpu_adamw_requested {
        let pairs = names
            .iter()
            .cloned()
            .zip(targets.iter().copied().zip(gradients.iter().copied()))
            .collect::<BTreeMap<_, _>>();
        Some(GpuAdamWPlan::append(
            &mut execution_graph,
            &pairs,
            &params,
            &optimizer.checkpoint(),
        )?)
    } else {
        None
    };
    // Full unrolled BPTT, CE cotangents, reverse AD and optimizer IR are
    // checked for compatibility, as are the named source bindings, initial
    // values and the complete held-out evaluation protocol.
    let model_fingerprint = architecture_fingerprint(
        &model,
        &execution_graph,
        FingerprintContext {
            seed: matrix_seed,
            role_init: role_init.as_str(),
            initial: &initial_bindings,
            val_limit,
            test_limit,
            eval_every,
            optimizer_backend: if gpu_optimizer.is_some() {
                "GPU_IR_ADAMW_V1"
            } else {
                "HOST_ADAMW_V1"
            },
        },
    );
    if let Some(ckpt) = pending_checkpoint.as_ref() {
        ckpt.validate_with_learning_rate(
            CheckpointRequirements {
                bptt: window_len,
                vocab,
                corpus: corpus_hash,
                architecture: model_fingerprint,
                val_limit,
                test_limit,
                eval_every,
                max_windows: train_windows,
                params: &trainable_initial,
                states: &expected_states,
            },
            ckpt.optimizer.learning_rate,
        )?;
        println!(
            "checkpoint_resumed={} completed_windows={}",
            checkpoint_path.display(),
            ckpt.completed_windows,
        );
    }
    println!(
        "training_adamw_lr={training_lr:.8} checkpoint_lr_transition={} optimizer_moments=preserved",
        if lr_decay_5x {
            "explicit_5x_decay"
        } else {
            "unchanged"
        }
    );
    println!(
        "experiment_matrix_mode={} role_init={} matrix_seed={} architecture_fingerprint={} projection_matrices={} matrix_values={} trainable_values={} adapter_rank={:?} delta_scale={} optimizer=AdamW corpus_hash={} bptt={} train_windows={} val_limit={} test_limit={} eval_every={}",
        matrix_mode.as_str(),
        role_init.as_str(),
        matrix_seed,
        model_fingerprint,
        matrix_mode.matrix_count(),
        matrix_mode.matrix_count() * torus16_mix_lab::WIDTH * torus16_mix_lab::WIDTH,
        parameter_count,
        matrix_mode.role_delta_rank(),
        weights.delta_scale,
        corpus_hash,
        window_len,
        train_windows,
        val_limit,
        test_limit,
        eval_every
    );

    if let Some(plan) = &gpu_optimizer {
        selected.extend(plan.outputs());
    }
    let mut all_feedback = state_feedback.clone();
    if let Some(plan) = &gpu_optimizer {
        all_feedback.extend_from_slice(plan.feedback());
    }
    let mut launch = GpuLaunchConfig::new(threads_per_block as u32, ReluFusion::Disabled)?
        .with_approximate_exp()
        .with_pointwise_fusion();
    if gpu_optimizer.is_some() {
        launch = launch.with_internal_pointwise_fusion();
    }
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(launch)
    .with_codegen_opt_level(codegen_opt_level);
    let holdout = HoldoutContext {
        engine: &engine,
        model: &model,
        weights: &weights,
        mode: matrix_mode,
        role_init,
        corpus: &corpus,
    };
    // Before/after reports count actual CUDA Graph kernel nodes, not the
    // number of semantic IR nodes. Baseline is compiled but NOT executed.
    let old_kernels = {
        let baseline = GpuEngine::new(
            "llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
        )
        .with_launch_config(GpuLaunchConfig::default().with_approximate_exp())
        .compile_outputs(&execution_graph, &selected)?;
        baseline.kernel_count()?
    };
    // Release the unfused program before constructing the GPU execution:
    // large BPTT graphs should not retain two IR/LLVM copies simultaneously.
    let program = engine.compile_outputs(&execution_graph, &selected)?;
    let new_kernels = program.kernel_count()?;
    if new_kernels > old_kernels {
        return Err("experimental pointwise fusion unexpectedly increased kernel count".into());
    }
    println!(
        "cuda_graph_before={} cuda_graph_after={} removed={} mul_add_fusions={} mul_add_mul_fusions={} binary_pointwise_fusions={} four_op_fusions={} diamond_fusions={} dual_output_fusions={} reduction_percent={:.2}",
        old_kernels,
        new_kernels,
        old_kernels - new_kernels,
        program.fused_mul_adds().len(),
        program.fused_mul_add_muls().len(),
        program.fused_binary_pairs().len(),
        program.fused_mul_mul_add_divs().len(),
        program.fused_mul_diamond_adds().len(),
        program.fused_dual_outputs().len(),
        100. * (old_kernels - new_kernels) as f64 / old_kernels as f64
    );

    let base = all
        .into_iter()
        .filter(|(name, _)| name != "token")
        .map(|(name, value)| {
            let value = params
                .get(&name)
                .or_else(|| states.get(&name))
                .cloned()
                .unwrap_or(value);
            (name, value)
        })
        .collect::<Vec<_>>();
    let step_tokens = (0..window_len)
        .map(|step| {
            Ok(vec![(
                "token".to_string(),
                onehot(corpus.train[step], vocab)?,
            )])
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let blank_seeds = (0..window_len)
        .map(|_| Tensor::zeros(vec![1, vocab]))
        .collect::<Result<Vec<_>, _>>()?;
    let supplied = window.bindings(&base, &step_tokens, &blank_seeds)?;
    let mut supplied = supplied;
    supplied.extend(reverse.constants().iter().cloned());
    if let Some(plan) = &gpu_optimizer {
        supplied.extend(plan.bindings().iter().cloned());
    }
    supplied.push((ones_id, Tensor::from_vec(vec![vocab, 1], vec![1.; vocab])?));
    supplied.push((
        scale_id,
        Tensor::from_slice(vec![1, 1], &[1. / window_len as f32])?,
    ));
    let supplied = supplied
        .into_iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .collect::<Vec<_>>();
    let artifact = std::env::temp_dir().join(format!("vn_ema4_bptt_{}", std::process::id()));
    // P2 acceptance uses the *real* supervised EMA4 reverse graph, with
    // identical parameters, recurrent state, training tokens and CE labels.
    // The baseline session is dropped before allocating the fused session to
    // avoid requiring two large CUDA graphs in GPU memory simultaneously.
    let compare_replay = std::env::var_os("TORUS16_BENCH_COMPARE").is_some();
    let verify_updates = if verify_fusion && first_window < train_windows {
        let mut updates = Vec::with_capacity(window_len * 2);
        let start = first_window * window_len;
        for t in 0..window_len {
            updates.push((
                window.inputs()[t]["token"],
                onehot(corpus.train[start + t], vocab)?,
            ));
            updates.push((
                window.cotangents()[t],
                onehot(corpus.train[start + t + 1], vocab)?,
            ));
        }
        if let Some(plan) = &gpu_optimizer {
            updates.extend(plan.correction_inputs(first_window as u64 + 1)?);
        }
        Some(updates)
    } else {
        None
    };
    let (baseline_replay_ms, baseline_outputs) = if compare_replay || verify_updates.is_some() {
        let baseline_engine = GpuEngine::new(
            "llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
        )
        .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
        let baseline = baseline_engine.compile_outputs(&execution_graph, &selected)?;
        let baseline_sources = supplied
            .iter()
            .filter(|(id, _)| baseline.source_inputs().contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let baseline_dir = artifact.join("baseline");
        let mut execution = baseline_engine.prepare(&baseline, &baseline_sources, &baseline_dir)?;
        let replay_ms = if compare_replay {
            Some(execution.benchmark(3, 10)?.average().as_secs_f64() * 1000.)
        } else {
            None
        };
        let outputs = if let Some(updates) = &verify_updates {
            let borrowed = updates
                .iter()
                .map(|(id, tensor)| (*id, tensor))
                .collect::<Vec<_>>();
            execution.update_input_subset(&borrowed)?;
            execution.replay()?;
            Some(execution.read_outputs_selected(&selected)?)
        } else {
            None
        };
        (replay_ms, outputs)
    } else {
        (None, None)
    };
    let mut gpu = engine.prepare(&program, &supplied, &artifact)?;
    if fresh {
        let stale_best = checkpoint_path.with_extension("best.ckpt");
        if stale_best.exists() {
            fs::remove_file(stale_best)?;
        }
    }
    if let Some(reference) = baseline_outputs {
        let updates = verify_updates
            .as_ref()
            .ok_or("missing fused parity input bindings")?;
        let borrowed = updates
            .iter()
            .map(|(id, tensor)| (*id, tensor))
            .collect::<Vec<_>>();
        gpu.update_input_subset(&borrowed)?;
        gpu.replay()?;
        let actual = gpu.read_outputs_selected(&selected)?;
        if reference.len() != actual.len() {
            return Err("fused/unfused GPU output inventory mismatch".into());
        }
        let mut values_checked = 0usize;
        for ((expected_id, expected), (actual_id, found)) in reference.iter().zip(&actual) {
            if expected_id != actual_id || expected.shape() != found.shape() {
                return Err("fused/unfused GPU node or shape mismatch".into());
            }
            for (lane, (&left, &right)) in expected.values().iter().zip(found.values()).enumerate()
            {
                if left.to_bits() != right.to_bits() {
                    return Err(format!(
                        "fused/unfused FP32 mismatch node={} lane={lane}: baseline={:#010x} fused={:#010x}",
                        expected_id.index(),
                        left.to_bits(),
                        right.to_bits()
                    )
                    .into());
                }
                values_checked += 1;
            }
        }
        println!(
            "fusion_parity=PASS selected_tensors={} fp32_values={} scope=EMA4_logits_gradients_terminal_states",
            actual.len(),
            values_checked
        );
    }
    println!(
        "setup_compile_and_cuda_prepare_ms={:.3}",
        compile_clock.elapsed().as_secs_f64() * 1000.
    );

    if let Some(baseline_ms) = baseline_replay_ms {
        let fused_ms = gpu.benchmark(3, 10)?.average().as_secs_f64() * 1000.;
        println!(
            "cuda_graph_replay_baseline_ms={baseline_ms:.5} cuda_graph_replay_fused_ms={fused_ms:.5} cuda_graph_replay_speedup={:.3} scope=GPU_only",
            baseline_ms / fused_ms
        );
    }
    // Device CUDA-event timing of the fused graph alone. Compilation,
    // HtoD copies, host CE, AdamW and holdout scoring are excluded.
    if std::env::var_os("TORUS16_BENCH_FUSED").is_some() {
        let event = gpu.benchmark(3, 10)?;
        println!(
            "fused_cuda_graph_event_average_ms={:.5} measured_launches=10",
            event.average().as_secs_f64() * 1000.
        );
    }
    // Only logits and gradients cross D2H on every window. Recurrent
    // state feedback remains device-resident until a checkpoint boundary.
    let output_count = window.outputs().len();
    println!(
        "llc_codegen_optimization={} translation_validation=required numeric_equivalence=not_proven",
        codegen_opt_level.llc_flag()
    );
    println!(
        "throughput_policy_threads_per_block={} metric_every={} optimizer_backend={} metric_sampling=reporting_only",
        threads_per_block,
        metric_every,
        if gpu_optimizer.is_some() {
            "GPU_IR"
        } else {
            "CPU"
        }
    );
    println!(
        "BPTT={} trainable_tensors={} trainable_values={} vocab={} IR_nodes={} CUDA_graph_kernels={}",
        window_len,
        names.len(),
        parameter_count,
        vocab,
        execution_graph.nodes().len(),
        program.kernel_count()?
    );
    let holdout_prepare_started = Instant::now();
    let mut cached_holdout = if std::env::var_os("TORUS16_REUSE_HOLDOUT").is_some() {
        Some(ReusedHoldout::prepare(&holdout)?)
    } else {
        None
    };
    println!(
        "holdout_session_mode={} holdout_session_prepare_s={:.6} scope=outside_training_timer",
        if cached_holdout.is_some() {
            "reused"
        } else {
            "fresh"
        },
        holdout_prepare_started.elapsed().as_secs_f64()
    );
    let train_start = Instant::now();
    // Aggregated wall times across the entire measured training loop, so a
    // single slow checkpoint or CUDA synchronization cannot disappear among
    // the three early-window samples.
    let mut profile_totals = [0.0f64; 13];
    // Event profiling modifies synchronization and is intentionally off in
    // the authoritative end-to-end throughput benchmark.
    let gpu_feedback_events =
        gpu_optimizer.is_some() && std::env::var_os("TORUS16_GPU_FEEDBACK_EVENTS").is_some();
    let mut feedback_checked_values = 0usize;
    let mut feedback_copied_values = 0usize;
    let mut validation_total_seconds = 0.0f64;
    // Test-only probe of the first *training* BPTT replay. The inputs and
    // cotangents are identical between independently and jointly tied W.
    let initial_probe_path = std::env::var_os("TORUS16_INITIAL_PROBE_PATH").map(PathBuf::from);
    // Training-only sampled learning curve (all windows at the default
    // metric_every=1). It is never used for gradient updates or checkpoint selection.
    // The held-out BPC below is the generalization metric, not this moving mean.
    let mut curve_bpc_sum = 0.0f64;
    let mut curve_windows = 0usize;
    for iteration in first_window..train_windows {
        let step_start = Instant::now();
        let start = iteration * window_len;
        let mut input_updates = Vec::with_capacity(window_len * 2 + 2);
        for t in 0..window_len {
            input_updates.push((
                window.inputs()[t]["token"],
                &onehots[corpus.train[start + t]],
            ));
            input_updates.push((
                window.cotangents()[t],
                &onehots[corpus.train[start + t + 1]],
            ));
        }
        // Only two scalar bias corrections cross H2D on the GPU path.
        let correction_updates = gpu_optimizer
            .as_ref()
            .map(|plan| plan.correction_inputs(iteration as u64 + 1))
            .transpose()?;
        if let Some(inputs) = &correction_updates {
            for (id, tensor) in inputs {
                input_updates.push((*id, tensor));
            }
        }
        let timing = Instant::now();
        gpu.update_input_subset(&input_updates)?;
        let upload_inputs_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        gpu.replay()?;
        let graph_replay_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        // One D2H batch for logits and gradients, no recurrent state
        // readback except at checkpoint boundaries.
        let read_count = if gpu_optimizer.is_some() {
            output_count
        } else {
            train_output_count
        };
        // No host-logit readback is needed for gradient/AdamW updates:
        // the VJP and optimizer are already in the CUDA Graph. Sampling
        // changes only reporting, never loss seeds or parameter updates.
        let observe_loss = gpu_optimizer.is_none()
            || iteration == first_window
            || (iteration + 1) % metric_every == 0
            || iteration + 1 == train_windows;
        let outputs = if observe_loss {
            gpu.read_outputs_selected(&selected[..read_count])?
        } else {
            Vec::new()
        };
        if iteration == 0
            && let Some(path) = &initial_probe_path
        {
            let states_snapshot = gpu.read_outputs_selected(&terminal_outputs)?;
            let mut bytes = Vec::new();
            for (_, tensor) in outputs
                .iter()
                .take(window_len)
                .chain(states_snapshot.iter())
            {
                for &value in tensor.values() {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
            fs::write(path, &bytes)?;
            println!(
                "initial_training_probe_bytes={} matrix_mode={}",
                bytes.len(),
                matrix_mode.as_str()
            );
        }
        let read_outputs_ms = timing.elapsed().as_secs_f64() * 1000.;
        let timing = Instant::now();
        let total = if observe_loss {
            let mut loss = 0f32;
            for (t, (_, logits)) in outputs.iter().take(window_len).enumerate() {
                loss += cross_entropy(logits.values(), corpus.train[start + t + 1])?.0;
            }
            Some(loss)
        } else {
            None
        };
        let ce_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        if gpu_optimizer.is_none() {
            let derivative = names
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), outputs[output_count + i].1.clone()))
                .collect::<BTreeMap<_, _>>();
            optimizer.step(&mut params, &derivative)?;
        }
        let adam_ms = timing.elapsed().as_secs_f64() * 1000.;

        // All recurrent states and, in GPU mode, trainable weights and
        // AdamW moments move simultaneously on-device after the replay.
        let timing = Instant::now();
        let feedback_details = if gpu_optimizer.is_some() {
            Some(if gpu_feedback_events {
                gpu.feedback_state_checked_device_profiled(&all_feedback)?
            } else {
                gpu.feedback_state_checked_profiled(&all_feedback)?
            })
        } else {
            gpu.feedback_state(&all_feedback)?;
            None
        };
        let state_feedback_d2d_ms = timing.elapsed().as_secs_f64() * 1000.;
        let feedback_plan_ms = feedback_details.map_or(0., |p| p.plan_host.as_secs_f64() * 1000.);
        let feedback_validation_ms =
            feedback_details.map_or(0., |p| p.validation_d2h.as_secs_f64() * 1000.);
        let feedback_snapshot_ms =
            feedback_details.map_or(0., |p| p.snapshot_d2d.as_secs_f64() * 1000.);
        let feedback_copy_ms =
            feedback_details.map_or(0., |p| p.feedback_d2d.as_secs_f64() * 1000.);
        let feedback_sync_ms = feedback_details.map_or(0., |p| p.stream_sync.as_secs_f64() * 1000.);
        let feedback_device_ms = feedback_details
            .and_then(|p| p.device_copy)
            .map_or(0., |duration| duration.as_secs_f64() * 1000.);
        if let Some(p) = feedback_details {
            feedback_checked_values += p.checked_values;
            feedback_copied_values += p.feedback_values;
        }

        let eval_due = (iteration + 1) % eval_every == 0 || iteration + 1 == train_windows;
        // An evaluation boundary also saves a resumable snapshot so the
        // measured model and the selected best checkpoint are identical.
        let checkpoint_due = (iteration + 1) % checkpoint_every == 0 || eval_due;
        let timing = Instant::now();
        let checkpoint_states = if checkpoint_due {
            let copied = gpu.read_outputs_selected(&terminal_outputs)?;
            Some(
                window
                    .terminal_states()
                    .keys()
                    .cloned()
                    .zip(copied.into_iter().map(|(_, value)| value))
                    .collect::<BTreeMap<_, _>>(),
            )
        } else {
            None
        };
        let checkpoint_state_d2h_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        let checkpoint_optimizer = if let Some(plan) = &gpu_optimizer {
            if checkpoint_due {
                let (updated, saved) = plan.checkpoint(&mut gpu, iteration as u64 + 1)?;
                params = updated;
                Some(saved)
            } else {
                None
            }
        } else {
            // Legacy host AdamW needs to upload every named trainable tensor.
            // The GPU optimizer needs no such transfers between checkpoints.
            let updates = names
                .iter()
                .zip(&targets)
                .map(|(name, &id)| (id, &params[name]))
                .collect::<Vec<_>>();
            gpu.update_input_subset(&updates)?;
            None
        };
        let upload_parameters_ms = timing.elapsed().as_secs_f64() * 1000.;
        if let Some(states) = checkpoint_states {
            let validation = if eval_due {
                let validation_started = Instant::now();
                let (bpc, count) =
                    score_holdout(&mut cached_holdout, &holdout, &params, 0..val_limit)?;
                let validation_elapsed = validation_started.elapsed().as_secs_f64();
                validation_total_seconds += validation_elapsed;
                println!(
                    "validation_window={} holdout_reset_streaming_bpc={bpc:.6} scored_characters={count} role_init={} matrix_seed={} elapsed_s={validation_elapsed:.4}",
                    iteration + 1,
                    role_init.as_str(),
                    matrix_seed,
                );
                Some(bpc as f32)
            } else {
                None
            };
            let improved =
                validation.is_some_and(|bpc| best_validation.is_none_or(|previous| bpc < previous));
            if improved {
                best_validation = validation;
            }
            let snapshot = Ema4Checkpoint {
                completed_windows: iteration + 1,
                bptt: window_len,
                vocab,
                corpus_fingerprint: corpus_hash,
                architecture_fingerprint: model_fingerprint,
                validation_limit: val_limit,
                test_limit,
                validation_every: eval_every,
                best_validation_bpc: best_validation,
                parameters: params.clone(),
                states,
                optimizer: checkpoint_optimizer.unwrap_or_else(|| optimizer.checkpoint()),
            };
            if improved {
                let best_path = checkpoint_path.with_extension("best.ckpt");
                snapshot.write_atomic(&best_path)?;
                println!(
                    "best_validation_window={} holdout_bpc={:.6} checkpoint={}",
                    iteration + 1,
                    best_validation.unwrap_or(0.),
                    best_path.display()
                );
            }
            snapshot.write_atomic(&checkpoint_path)?;
        }

        for (total, sample) in profile_totals.iter_mut().zip([
            upload_inputs_ms,
            graph_replay_ms,
            read_outputs_ms,
            ce_ms,
            adam_ms,
            state_feedback_d2d_ms,
            checkpoint_state_d2h_ms,
            upload_parameters_ms,
            feedback_validation_ms,
            feedback_snapshot_ms,
            feedback_copy_ms + feedback_sync_ms,
            feedback_device_ms,
            feedback_plan_ms,
        ]) {
            *total += sample;
        }
        if let Some(total) = total {
            println!(
                "train_window={} ce_nats={:.6} train_bpc={:.6} optimizer_step={} end_to_end_step_ms={:.3}",
                iteration,
                total / window_len as f32,
                total / (window_len as f32 * std::f32::consts::LN_2),
                if gpu_optimizer.is_some() {
                    iteration as u64 + 1
                } else {
                    optimizer.steps()
                },
                step_start.elapsed().as_secs_f64() * 1000.
            );
            curve_bpc_sum += f64::from(total / (window_len as f32 * std::f32::consts::LN_2));
            curve_windows += 1;
        }
        if curve_windows == 1024 || iteration + 1 == train_windows {
            println!(
                "train_curve_window={} train_bpc_mean={:.6} window_count={} matrix_mode={}",
                iteration + 1,
                curve_bpc_sum / curve_windows as f64,
                curve_windows,
                matrix_mode.as_str(),
            );
            curve_bpc_sum = 0.0;
            curve_windows = 0;
        }
        if iteration < 3 || (iteration + 1) % 128 == 0 {
            println!(
                "step_profile_window={} input_h2d_ms={upload_inputs_ms:.3} graph_replay_ms={graph_replay_ms:.3} logits_gradients_d2h_ms={read_outputs_ms:.3} ce_ms={ce_ms:.3} host_adamw_ms={adam_ms:.3} feedback_total_ms={state_feedback_d2d_ms:.3} feedback_validation_d2h_ms={feedback_validation_ms:.3} feedback_snapshot_d2d_ms={feedback_snapshot_ms:.3} feedback_copy_sync_ms={:.3} feedback_plan_host_ms={feedback_plan_ms:.3} feedback_device_event_ms={feedback_device_ms:.3} feedback_events={gpu_feedback_events} checkpoint_state_d2h_ms={checkpoint_state_d2h_ms:.3} parameters_h2d_ms={upload_parameters_ms:.3}",
                iteration,
                feedback_copy_ms + feedback_sync_ms,
            );
        }
    }
    let profile_windows = train_windows - first_window;
    // A completed checkpoint can be inspected again without generating NaN
    // timing summaries from a zero-window no-op training loop.
    let profile_divisor = profile_windows.max(1) as f64;
    println!(
        "training_profile_avg_ms windows={} input_h2d={:.4} graph_replay={:.4} selected_d2h={:.4} host_ce={:.4} host_adamw={:.4} feedback_total={:.4} checkpoint_state_d2h={:.4} parameters_or_checkpoint_io={:.4} feedback_validation_d2h={:.4} feedback_snapshot_d2d={:.4} feedback_copy_sync={:.4} feedback_plan_host={:.4} feedback_device_event_ms={:.4} feedback_events={} feedback_checked_values={} feedback_copied_values={} timing=host_wall_including_sync",
        profile_windows,
        profile_totals[0] / profile_divisor,
        profile_totals[1] / profile_divisor,
        profile_totals[2] / profile_divisor,
        profile_totals[3] / profile_divisor,
        profile_totals[4] / profile_divisor,
        profile_totals[5] / profile_divisor,
        profile_totals[6] / profile_divisor,
        profile_totals[7] / profile_divisor,
        profile_totals[8] / profile_divisor,
        profile_totals[9] / profile_divisor,
        profile_totals[10] / profile_divisor,
        profile_totals[12] / profile_divisor,
        profile_totals[11] / profile_divisor,
        gpu_feedback_events,
        feedback_checked_values,
        feedback_copied_values,
    );
    let including_validation = train_start.elapsed().as_secs_f64();
    // Validation creates a separate GPU inference session and must not be
    // silently counted as training throughput. Checkpoint and feedback costs
    // remain in the training clock.
    let train_elapsed = (including_validation - validation_total_seconds).max(f64::EPSILON);
    println!(
        "training_total_s={:.6} training_tokens={} training_tokens_per_second={:.2} validation_total_s={:.6} training_with_validation_s={:.6} scope=GPU_replay_plus_host_CE_AdamW_transfers optimizer_backend={}",
        train_elapsed,
        (train_windows - first_window) * window_len,
        ((train_windows - first_window) * window_len) as f64 / train_elapsed,
        validation_total_seconds,
        including_validation,
        if gpu_optimizer.is_some() {
            "GPU_IR"
        } else {
            "CPU"
        }
    );
    // Final metric is recomputed from the last saved parameters, including
    // when a finished run is reopened without further training.
    let validation_start = Instant::now();
    let (bpc, count) = score_holdout(&mut cached_holdout, &holdout, &params, 0..val_limit)?;
    println!(
        "holdout_reset_streaming_bpc={bpc:.6} matrix_mode={} role_init={} matrix_seed={} scored_characters={count} validation_split=last_10pct eval_score_seconds={:.5} best_validation_bpc={:.6}",
        matrix_mode.as_str(),
        role_init.as_str(),
        matrix_seed,
        validation_start.elapsed().as_secs_f64(),
        best_validation.unwrap_or(bpc as f32),
    );
    // Only the checkpoint selected by validation is evaluated on test.
    // No training, early stopping or best-checkpoint choice sees test data.
    let best_path = checkpoint_path.with_extension("best.ckpt");
    let best_snapshot = Ema4Checkpoint::read(&best_path)?;
    let best_checkpoint_lr = best_snapshot.optimizer.learning_rate;
    if best_checkpoint_lr.to_bits() != training_lr.to_bits()
        && !(lr_decay_5x && best_checkpoint_lr.to_bits() == baseline_lr.to_bits())
    {
        return Err("best checkpoint has an unexpected AdamW learning rate".into());
    }
    best_snapshot.validate_with_learning_rate(
        CheckpointRequirements {
            bptt: window_len,
            vocab,
            corpus: corpus_hash,
            architecture: model_fingerprint,
            val_limit,
            test_limit,
            eval_every,
            max_windows: train_windows,
            params: &trainable_initial,
            states: &expected_states,
        },
        best_checkpoint_lr,
    )?;
    let test_start = corpus.validation.len() - test_limit - 1;
    let (test_bpc, test_count) = score_holdout(
        &mut cached_holdout,
        &holdout,
        &best_snapshot.parameters,
        test_start..test_start + test_limit,
    )?;
    println!(
        "test_reset_streaming_bpc={test_bpc:.6} matrix_mode={} role_init={} matrix_seed={} scored_characters={test_count} test_split=validation_suffix_no_model_selection selection=best_validation_checkpoint checkpoint={}",
        matrix_mode.as_str(),
        role_init.as_str(),
        matrix_seed,
        best_path.display(),
    );
    println!(
        "NOTE: GPU approximate-exp softmax CE seed + VJP, host CE metric; GPU AdamW experimental when opted in; no formal FP32 gradient/optimizer proof."
    );
    Ok(())
}
