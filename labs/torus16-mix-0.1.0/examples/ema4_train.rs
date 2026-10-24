//! First genuine GPU training step for EMA4-LM: token CE -> GPU VJP ->
//! generic AdamW -> parameter update -> reuse the same captured CUDA Graph.
//! Gradient is truncated at 1 token. Only final LayerNorm beta is optimized.
//! This deliberately does NOT claim trained 24,908 weights, sequence BPTT,
//! all-parameter GPU training or research-quality BPC.
use std::collections::BTreeMap;
use std::error::Error;

use torus16_mix_lab::ema4::{Ema4Weights, architecture_model};
use verineural::{AdamW, GpuEngine, GpuLaunchConfig, Tensor, differentiate};

fn cross_entropy_gradient(
    logits: &[f32],
    target: usize,
) -> Result<(f32, Tensor<f32>), Box<dyn Error>> {
    if target >= logits.len() || logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("non-finite logits or out-of-range target".into());
    }
    let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let values = logits
        .iter()
        .map(|x| (x - largest).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    let loss = sum.ln() + largest - logits[target];
    let seed = values
        .iter()
        .enumerate()
        .map(|(i, &v)| v / sum - if i == target { 1. } else { 0. })
        .collect::<Vec<_>>();
    if !loss.is_finite() {
        return Err("non-finite cross-entropy".into());
    }
    Ok((loss, Tensor::from_vec(vec![1, logits.len()], seed)?))
}

fn main() -> Result<(), Box<dyn Error>> {
    let vocab = std::env::var("TORUS16_TRAIN_VOCAB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16);
    let mut weights = Ema4Weights::deterministic(vocab)?;
    let model = architecture_model(vocab)?;
    let logits = model.output("logits").ok_or("missing logits")?;
    let (parameter, _) = model.source("final_beta").ok_or("missing final_beta")?;
    let reverse = differentiate(model.graph(), logits, &[parameter])?;
    let gradient = reverse
        .gradient_of(parameter)
        .ok_or("missing final_beta gradient")?;

    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let program = engine.compile_outputs(reverse.graph(), &[logits, gradient])?;
    let source = model.bind(weights.bindings(0)?)?;
    let zero = Tensor::zeros(vec![1, vocab])?;
    let bindings = reverse.bindings(&source, zero)?;
    let active = bindings
        .into_iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .collect::<Vec<_>>();
    let path = std::env::temp_dir().join(format!("vn_ema4_train_{}", std::process::id()));
    let mut gpu = engine.prepare(&program, &active, &path)?;
    let mut optimizer = AdamW::new(0.1, 0.)?;
    let mut params = BTreeMap::from([(
        "final_beta".to_string(),
        Tensor::from_vec(vec![1, 48], weights.final_beta.to_vec())?,
    )]);
    let target = 1;
    for step in 0..8 {
        gpu.replay()?;
        let outputs = gpu.read_outputs()?;
        let (loss, seed) = cross_entropy_gradient(outputs[0].1.values(), target)?;
        gpu.update_input(reverse.seed(), &seed)?;
        gpu.replay()?;
        let outputs = gpu.read_outputs()?;
        let gradient = outputs[1].1.clone();
        let gradients = BTreeMap::from([("final_beta".to_string(), gradient)]);
        optimizer.step(&mut params, &gradients)?;
        gpu.update_input(parameter, &params["final_beta"])?;
        println!(
            "step={step} ce_nats={loss:.6} toy_bpc={:.6}",
            loss / std::f32::consts::LN_2
        );
    }
    weights
        .final_beta
        .copy_from_slice(params["final_beta"].values());
    println!(
        "EMA4 training slice: GPU reverse graph + AdamW, frozen other parameters; \
              TBPTT=1, state reset, not comparable to trained research EMA4-LM."
    );
    Ok(())
}
