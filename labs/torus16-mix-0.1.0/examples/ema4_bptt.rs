//! Experimental all-parameter finite-window BPTT for four-layer EMA4-LM.
//! Corpus defaults to this lab's checked-in research snapshot.
//! Training CE seeds and AdamW are on host; forward and all VJPs use one
//! compiled VeriNeural CUDA Graph. Validation is reset-state streaming BPC.
//! This is not equivalence with the trained handoff PyTorch checkpoint.
use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;
use torus16_mix_lab::corpus::CharCorpus;
use torus16_mix_lab::ema4::{Ema4Weights, architecture_model};
use verineural::{AdamW, GpuEngine, GpuLaunchConfig, Tensor, differentiate, unroll_model};

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
    let reverse = differentiate(window.graph(), window.objective(), &targets)?;
    let gradients = targets
        .iter()
        .map(|&p| {
            reverse
                .gradient_of(p)
                .ok_or("missing differentiated parameter")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut selected = window.outputs().to_vec();
    selected.extend(gradients.iter().copied());
    selected.extend(window.terminal_states().values().copied());
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(
        GpuLaunchConfig::default()
            .with_approximate_exp()
            .with_pointwise_fusion(),
    );
    // Before/after reports count actual CUDA Graph kernel nodes, not the
    // number of semantic IR nodes. Baseline is compiled but NOT executed.
    let old_kernels = {
        let baseline = GpuEngine::new(
            "llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
        )
        .with_launch_config(GpuLaunchConfig::default().with_approximate_exp())
        .compile_outputs(reverse.graph(), &selected)?;
        baseline.kernel_count()?
    };
    // Release the unfused program before constructing the GPU execution:
    // large BPTT graphs should not retain two IR/LLVM copies simultaneously.
    let program = engine.compile_outputs(reverse.graph(), &selected)?;
    let new_kernels = program.kernel_count()?;
    if new_kernels > old_kernels {
        return Err("experimental pointwise fusion unexpectedly increased kernel count".into());
    }
    println!(
        "cuda_graph_before={} cuda_graph_after={} removed={} mul_add_fusions={} mul_add_mul_fusions={} reduction_percent={:.2}",
        old_kernels,
        new_kernels,
        old_kernels - new_kernels,
        program.fused_mul_adds().len(),
        program.fused_mul_add_muls().len(),
        100. * (old_kernels - new_kernels) as f64 / old_kernels as f64
    );

    let all = weights.bindings(0)?;
    let base = all
        .into_iter()
        .filter(|(name, _)| name != "token")
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
    let supplied = reverse.bindings(&supplied, Tensor::from_slice(vec![1, 1], &[1.])?)?;
    let supplied = supplied
        .into_iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .collect::<Vec<_>>();
    let artifact = std::env::temp_dir().join(format!("vn_ema4_bptt_{}", std::process::id()));
    let baseline_replay_ms = if std::env::var_os("TORUS16_BENCH_COMPARE").is_some() {
        // Compile/allocate/benchmark/drop the baseline first, so the full
        // training model never requires two complete CUDA executions in VRAM.
        let baseline_engine = GpuEngine::new(
            "llc",
            std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
        )
        .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
        let baseline = baseline_engine.compile_outputs(reverse.graph(), &selected)?;
        let baseline_sources = supplied
            .iter()
            .filter(|(id, _)| baseline.source_inputs().contains(id))
            .cloned()
            .collect::<Vec<_>>();
        let baseline_dir = artifact.join("baseline");
        let execution = baseline_engine.prepare(&baseline, &baseline_sources, &baseline_dir)?;
        Some(execution.benchmark(3, 10)?.average().as_secs_f64() * 1000.)
    } else {
        None
    };
    let mut gpu = engine.prepare(&program, &supplied, &artifact)?;
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
    let mut params = trainable_initial;
    let mut optimizer = AdamW::new(0.002, 0.01)?;
    let mut backward_selected = gradients.clone();
    backward_selected.extend(window.terminal_states().values().copied());
    println!(
        "BPTT={} trainable_tensors={} trainable_values={} vocab={} IR_nodes={} CUDA_graph_kernels={}",
        window_len,
        names.len(),
        parameter_count,
        vocab,
        reverse.graph().nodes().len(),
        program.kernel_count()?
    );
    let train_start = Instant::now();
    for iteration in 0..train_windows {
        let step_start = Instant::now();
        let start = iteration * window_len;
        let tokens = (0..window_len)
            .map(|t| onehot(corpus.train[start + t], vocab))
            .collect::<Result<Vec<_>, _>>()?;
        let mut input_updates = Vec::with_capacity(window_len * 2);
        for (t, tensor) in tokens.iter().enumerate() {
            input_updates.push((window.inputs()[t]["token"], tensor));
        }
        for (t, seed) in blank_seeds.iter().enumerate() {
            input_updates.push((window.cotangents()[t], seed));
        }
        let timing = Instant::now();
        gpu.update_input_subset(&input_updates)?;
        let upload_inputs_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        gpu.replay()?;
        let forward_replay_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        // Only logits are needed for CE; gradients and all 68 states are not
        // downloaded during the seed-building forward pass.
        let forward = gpu.read_outputs_selected(window.outputs())?;
        let read_logits_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        let mut total = 0f32;
        let mut seed_tensors = Vec::with_capacity(window_len);
        for t in 0..window_len {
            let (loss, mut gradient) =
                cross_entropy(forward[t].1.values(), corpus.train[start + t + 1])?;
            total += loss;
            for v in &mut gradient {
                *v /= window_len as f32;
            }
            seed_tensors.push(Tensor::from_vec(vec![1, vocab], gradient)?);
        }
        let ce_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        let seed_updates = window
            .cotangents()
            .iter()
            .copied()
            .zip(seed_tensors.iter())
            .collect::<Vec<_>>();
        gpu.update_input_subset(&seed_updates)?;
        let upload_seed_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        gpu.replay()?;
        let backward_replay_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        // Queue all gradient/state D2H copies and wait for them once, rather
        // than synchronizing for each of 271 gradients and 68 states.
        let backward = gpu.read_outputs_selected(&backward_selected)?;
        let read_backward_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        let derivative = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), backward[i].1.clone()))
            .collect::<BTreeMap<_, _>>();
        // Host staging of all next states before the single H2D update batch
        // preserves recurrent TBPTT semantics even with aliased GPU buffers.
        let next_states = window
            .terminal_states()
            .keys()
            .enumerate()
            .map(|(i, name)| (name.clone(), backward[names.len() + i].1.clone()))
            .collect::<BTreeMap<_, _>>();
        optimizer.step(&mut params, &derivative)?;
        let adam_ms = timing.elapsed().as_secs_f64() * 1000.;

        let timing = Instant::now();
        // Update ONLY AdamW's trainable inventory. window.parameters()
        // additionally contains fixed numeric sources such as ones_column,
        // inv_width and epsilon_scalar, which are not AdamW parameters.
        // 'targets' was validated against 'names' before autodiff.
        let mut updates = Vec::with_capacity(targets.len() + window.initial_states().len());
        for (name, &id) in names.iter().zip(&targets) {
            updates.push((id, &params[name]));
        }
        for (name, &id) in window.initial_states() {
            updates.push((id, &next_states[name]));
        }
        gpu.update_input_subset(&updates)?;
        let upload_parameters_states_ms = timing.elapsed().as_secs_f64() * 1000.;

        println!(
            "train_window={} ce_nats={:.6} train_bpc={:.6} optimizer_step={} end_to_end_step_ms={:.3}",
            iteration,
            total / window_len as f32,
            total / (window_len as f32 * std::f32::consts::LN_2),
            optimizer.steps(),
            step_start.elapsed().as_secs_f64() * 1000.
        );
        if iteration < 3 || (iteration + 1) % 128 == 0 {
            println!(
                "step_profile_window={} input_h2d_ms={upload_inputs_ms:.3} forward_replay_ms={forward_replay_ms:.3} logits_d2h_ms={read_logits_ms:.3} ce_ms={ce_ms:.3} seed_h2d_ms={upload_seed_ms:.3} backward_replay_ms={backward_replay_ms:.3} gradients_states_d2h_ms={read_backward_ms:.3} host_adamw_ms={adam_ms:.3} parameters_states_h2d_ms={upload_parameters_states_ms:.3}",
                iteration
            );
        }
    }
    let train_elapsed = train_start.elapsed().as_secs_f64();
    println!(
        "training_total_s={:.6} training_tokens={} training_tokens_per_second={:.2} scope=GPU_replay_plus_host_CE_AdamW_transfers",
        train_elapsed,
        train_windows * window_len,
        (train_windows * window_len) as f64 / train_elapsed
    );
    // Real holdout: chronological final 10% only. One recurrent inference
    // step per symbol, with state reset to zero at the validation boundary.
    let mut eval_bindings = weights.bindings(0)?;
    for (name, value) in &mut eval_bindings {
        if let Some(updated) = params.get(name) {
            *value = updated.clone();
        }
    }
    let eval_dir =
        PathBuf::from(std::env::temp_dir()).join(format!("vn_ema4_holdout_{}", std::process::id()));
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
        "NOTE: deterministic untrained initial values; host AdamW/CE, GPU forward/VJP; no checkpoint parity or formal FP32 gradient proof."
    );
    Ok(())
}
