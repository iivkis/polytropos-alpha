//! Experimental all-parameter finite-window BPTT for four-layer EMA4-LM.
//! Corpus defaults to this lab's checked-in research snapshot.
//! Training CE seeds and AdamW are on host; forward and all VJPs use one
//! compiled VeriNeural CUDA Graph. Validation is reset-state streaming BPC.
//! This is not equivalence with the trained handoff PyTorch checkpoint.
use std::collections::BTreeMap;
use std::error::Error;
use std::path::PathBuf;
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
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let program = engine.compile_outputs(reverse.graph(), &selected)?;
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
    let mut gpu = engine.prepare(&program, &supplied, &artifact)?;
    let mut params = trainable_initial;
    let mut optimizer = AdamW::new(0.002, 0.01)?;
    let grad_begin = window_len;
    let state_begin = grad_begin + names.len();
    println!(
        "BPTT={} trainable_tensors={} trainable_values={} vocab={} IR_nodes={} CUDA_graph_kernels={}",
        window_len,
        names.len(),
        parameter_count,
        vocab,
        reverse.graph().nodes().len(),
        program.kernel_count()?
    );
    for iteration in 0..train_windows {
        let start = iteration * window_len;
        for t in 0..window_len {
            let input = onehot(corpus.train[start + t], vocab)?;
            gpu.update_input(window.inputs()[t]["token"], &input)?;
            gpu.update_input(window.cotangents()[t], &blank_seeds[t])?;
        }
        gpu.replay()?;
        let forward = gpu.read_outputs()?;
        let mut total = 0f32;
        for t in 0..window_len {
            let (loss, mut gradient) =
                cross_entropy(forward[t].1.values(), corpus.train[start + t + 1])?;
            total += loss;
            for v in &mut gradient {
                *v /= window_len as f32;
            }
            let seed = Tensor::from_vec(vec![1, vocab], gradient)?;
            gpu.update_input(window.cotangents()[t], &seed)?;
        }
        gpu.replay()?;
        let backward = gpu.read_outputs()?;
        let derivative = names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), backward[grad_begin + i].1.clone()))
            .collect::<BTreeMap<_, _>>();
        // Snapshot ALL states before writing ANY initial-state address.
        // This is explicit host staging across truncated windows, not
        // unsafe overlapping CUDA D2D feedback.
        let next_states = window
            .terminal_states()
            .keys()
            .enumerate()
            .map(|(i, name)| (name.clone(), backward[state_begin + i].1.clone()))
            .collect::<BTreeMap<_, _>>();
        optimizer.step(&mut params, &derivative)?;
        for (name, &id) in named_parameter {
            if let Some(updated) = params.get(name) {
                gpu.update_input(id, updated)?;
            }
        }
        for (name, &id) in window.initial_states() {
            gpu.update_input(id, &next_states[name])?;
        }
        println!(
            "train_window={} ce_nats={:.6} train_bpc={:.6} optimizer_step={}",
            iteration,
            total / window_len as f32,
            total / (window_len as f32 * std::f32::consts::LN_2),
            optimizer.steps()
        );
    }
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
        "holdout_reset_streaming_bpc={bpc:.6} scored_characters={count} validation_split=last_10pct"
    );
    println!(
        "NOTE: deterministic untrained initial values; host AdamW/CE, GPU forward/VJP; no checkpoint parity or formal FP32 gradient proof."
    );
    Ok(())
}
