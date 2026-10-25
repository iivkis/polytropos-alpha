//! Experimental all-parameter finite-window BPTT for four-layer EMA4-LM.
//! Corpus defaults to this lab's checked-in research snapshot.
//! Softmax CE cotangents and VJPs share one CUDA Graph replay per window;
//! CE reporting and AdamW are on host. CUDA FP32 parity is not certified. Validation is reset-state streaming BPC.
//! This is not equivalence with the trained handoff PyTorch checkpoint.
use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use torus16_mix_lab::corpus::CharCorpus;
use torus16_mix_lab::ema4::{Ema4Weights, architecture_model};
use torus16_mix_lab::ema4_checkpoint::{Ema4Checkpoint, fingerprint};
use verineural::{
    AdamW, ArgMaxTieBreak, Domain, GpuAdamWPlan, GpuEngine, GpuLaunchConfig, NodeId, Shape, Tensor,
    differentiate_with_seeds, unroll_model,
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
    let train_windows = option_usize("TORUS16_TRAIN_WINDOWS", 2, 1, 4096)?;
    let val_limit = option_usize("TORUS16_VAL_LIMIT", 64, 2, 100000)?;
    let checkpoint_every = option_usize("TORUS16_CHECKPOINT_EVERY", 128, 1, 4096)?;
    let checkpoint_path = std::env::var_os("TORUS16_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/torus16-ema4-bptt.ckpt"));
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
    let weights = Ema4Weights::deterministic(vocab)?;
    let model = architecture_model(vocab)?;
    let trainable_initial = weights.trainable_parameters()?;
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
    let mut selected = window.outputs().to_vec();
    // CPU AdamW reads gradients on the host. GPU AdamW consumes them as
    // internal IR dependencies, so exporting them needlessly blocks fusion.
    if !gpu_adamw_requested {
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
    let all = weights.bindings(0)?;
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
    let (mut params, states, mut optimizer, first_window) = if !fresh && checkpoint_path.exists() {
        let ckpt = Ema4Checkpoint::read(&checkpoint_path)?;
        ckpt.validate(
            window_len,
            vocab,
            corpus_hash,
            train_windows,
            &trainable_initial,
            &expected_states,
        )?;
        let opt = AdamW::from_checkpoint(ckpt.optimizer, &ckpt.parameters)?;
        println!(
            "checkpoint_resumed={} completed_windows={}",
            checkpoint_path.display(),
            ckpt.completed_windows
        );
        (ckpt.parameters, ckpt.states, opt, ckpt.completed_windows)
    } else {
        (
            trainable_initial,
            expected_states,
            AdamW::new(0.002, 0.01)?,
            0usize,
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
    if let Some(plan) = &gpu_optimizer {
        selected.extend(plan.outputs());
    }
    let mut all_feedback = state_feedback.clone();
    if let Some(plan) = &gpu_optimizer {
        all_feedback.extend_from_slice(plan.feedback());
    }
    let mut launch = GpuLaunchConfig::default()
        .with_approximate_exp()
        .with_pointwise_fusion();
    if gpu_optimizer.is_some() {
        launch = launch.with_internal_pointwise_fusion();
    }
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(launch);
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
    let verify_fusion = std::env::var_os("TORUS16_VERIFY_FUSION").is_some();
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
        "BPTT={} trainable_tensors={} trainable_values={} vocab={} IR_nodes={} CUDA_graph_kernels={}",
        window_len,
        names.len(),
        parameter_count,
        vocab,
        execution_graph.nodes().len(),
        program.kernel_count()?
    );
    let train_start = Instant::now();
    // Aggregated wall times across the entire measured training loop, so a
    // single slow checkpoint or CUDA synchronization cannot disappear among
    // the three early-window samples.
    let mut profile_totals = [0.0f64; 11];
    let mut feedback_checked_values = 0usize;
    let mut feedback_copied_values = 0usize;
    for iteration in first_window..train_windows {
        let step_start = Instant::now();
        let start = iteration * window_len;
        let tokens = (0..window_len)
            .map(|t| onehot(corpus.train[start + t], vocab))
            .collect::<Result<Vec<_>, _>>()?;
        let labels = (0..window_len)
            .map(|t| onehot(corpus.train[start + t + 1], vocab))
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_updates = Vec::with_capacity(window_len * 2);
        for (t, tensor) in tokens.iter().enumerate() {
            input_updates.push((window.inputs()[t]["token"], tensor));
            input_updates.push((window.cotangents()[t], &labels[t]));
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
        let outputs = gpu.read_outputs_selected(&selected[..read_count])?;
        let read_outputs_ms = timing.elapsed().as_secs_f64() * 1000.;
        let timing = Instant::now();
        let mut total = 0f32;
        for (t, (_, logits)) in outputs.iter().take(window_len).enumerate() {
            total += cross_entropy(logits.values(), corpus.train[start + t + 1])?.0;
        }
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
            Some(gpu.feedback_state_checked_profiled(&all_feedback)?)
        } else {
            gpu.feedback_state(&all_feedback)?;
            None
        };
        let state_feedback_d2d_ms = timing.elapsed().as_secs_f64() * 1000.;
        let feedback_validation_ms =
            feedback_details.map_or(0., |p| p.validation_d2h.as_secs_f64() * 1000.);
        let feedback_snapshot_ms =
            feedback_details.map_or(0., |p| p.snapshot_d2d.as_secs_f64() * 1000.);
        let feedback_copy_ms =
            feedback_details.map_or(0., |p| p.feedback_d2d.as_secs_f64() * 1000.);
        let feedback_sync_ms = feedback_details.map_or(0., |p| p.stream_sync.as_secs_f64() * 1000.);
        if let Some(p) = feedback_details {
            feedback_checked_values += p.checked_values;
            feedback_copied_values += p.feedback_values;
        }

        let checkpoint_due =
            (iteration + 1) % checkpoint_every == 0 || iteration + 1 == train_windows;
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
            Ema4Checkpoint {
                completed_windows: iteration + 1,
                bptt: window_len,
                vocab,
                corpus_fingerprint: corpus_hash,
                parameters: params.clone(),
                states,
                optimizer: checkpoint_optimizer.unwrap_or_else(|| optimizer.checkpoint()),
            }
            .write_atomic(&checkpoint_path)?;
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
        ]) {
            *total += sample;
        }
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
        if iteration < 3 || (iteration + 1) % 128 == 0 {
            println!(
                "step_profile_window={} input_h2d_ms={upload_inputs_ms:.3} graph_replay_ms={graph_replay_ms:.3} logits_gradients_d2h_ms={read_outputs_ms:.3} ce_ms={ce_ms:.3} host_adamw_ms={adam_ms:.3} feedback_total_ms={state_feedback_d2d_ms:.3} feedback_validation_d2h_ms={feedback_validation_ms:.3} feedback_snapshot_d2d_ms={feedback_snapshot_ms:.3} feedback_copy_sync_ms={:.3} checkpoint_state_d2h_ms={checkpoint_state_d2h_ms:.3} parameters_h2d_ms={upload_parameters_ms:.3}",
                iteration,
                feedback_copy_ms + feedback_sync_ms,
            );
        }
    }
    let profile_windows = train_windows - first_window;
    println!(
        "training_profile_avg_ms windows={} input_h2d={:.4} graph_replay={:.4} selected_d2h={:.4} host_ce={:.4} host_adamw={:.4} feedback_total={:.4} checkpoint_state_d2h={:.4} parameters_or_checkpoint_io={:.4} feedback_validation_d2h={:.4} feedback_snapshot_d2d={:.4} feedback_copy_sync={:.4} feedback_checked_values={} feedback_copied_values={} timing=host_wall_including_sync",
        profile_windows,
        profile_totals[0] / profile_windows as f64,
        profile_totals[1] / profile_windows as f64,
        profile_totals[2] / profile_windows as f64,
        profile_totals[3] / profile_windows as f64,
        profile_totals[4] / profile_windows as f64,
        profile_totals[5] / profile_windows as f64,
        profile_totals[6] / profile_windows as f64,
        profile_totals[7] / profile_windows as f64,
        profile_totals[8] / profile_windows as f64,
        profile_totals[9] / profile_windows as f64,
        profile_totals[10] / profile_windows as f64,
        feedback_checked_values,
        feedback_copied_values,
    );
    let train_elapsed = train_start.elapsed().as_secs_f64();
    println!(
        "training_total_s={:.6} training_tokens={} training_tokens_per_second={:.2} scope=GPU_replay_plus_host_CE_AdamW_transfers optimizer_backend={}",
        train_elapsed,
        (train_windows - first_window) * window_len,
        ((train_windows - first_window) * window_len) as f64 / train_elapsed,
        if gpu_optimizer.is_some() {
            "GPU_IR"
        } else {
            "CPU"
        }
    );
    // Real holdout: chronological final 10% only. One recurrent inference
    // step per symbol, with state reset to zero at the validation boundary.
    let mut eval_bindings = weights.bindings(0)?;
    for (name, value) in &mut eval_bindings {
        if let Some(updated) = params.get(name) {
            *value = updated.clone();
        }
    }
    let eval_dir = std::env::temp_dir().join(format!("vn_ema4_holdout_{}", std::process::id()));
    let validation_start = Instant::now();
    let mut eval = engine.prepare_model(&model, eval_bindings, &eval_dir)?;
    let count = val_limit.min(corpus.validation.len() - 1);
    let mut val_ce = 0f64;
    for t in 0..count {
        let input = onehot(corpus.validation[t], vocab)?;
        eval.set_input("token", &input)?;
        eval.step()?;
        let logits = eval.output("logits")?;
        val_ce += f64::from(cross_entropy(logits.values(), corpus.validation[t + 1])?.0);
    }
    let bpc = val_ce / (count as f64 * std::f64::consts::LN_2);
    println!(
        "holdout_reset_streaming_bpc={bpc:.6} scored_characters={count} validation_split=last_10pct eval_prepare_plus_stream_seconds={:.5}",
        validation_start.elapsed().as_secs_f64()
    );
    println!(
        "NOTE: GPU approximate-exp softmax CE seed + VJP, host CE metric; GPU AdamW experimental when opted in; no formal FP32 gradient/optimizer proof."
    );
    Ok(())
}
