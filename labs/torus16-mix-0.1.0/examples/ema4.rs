//! Full four-layer EMA4-LM single-token streaming inference.
//! Training/quality measurement is intentionally not implied.
use std::error::Error;
use torus16_mix_lab::ema4::{Ema4Weights, RESEARCH_VOCAB, architecture_model};
use torus16_mix_lab::ema4_reference::{Ema4State, reference_step};
use verineural::{GpuEngine, GpuLaunchConfig, Tensor};

fn onehot(token: usize, vocabulary: usize) -> Result<Tensor<f32>, Box<dyn Error>> {
    let mut data = vec![0.; vocabulary];
    data[token] = 1.;
    Ok(Tensor::from_vec(vec![1, vocabulary], data)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "--inspect".into());
    let vocabulary = std::env::var("TORUS16_VOCAB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(RESEARCH_VOCAB);
    let weights = Ema4Weights::deterministic(vocabulary)?;
    let model = architecture_model(vocabulary)?;
    match mode.as_str() {
        "--inspect" => {
            let bound = model.bind(weights.bindings(0)?)?;
            println!("Torus16-Mix EMA4-LM: layers=4 width=48 vocab={vocabulary}");
            println!(
                "parameters={}  recurrent_state_tensors={}",
                Ema4Weights::parameter_count(vocabulary),
                model.state_links().len()
            );
            println!(
                "IR nodes={} operator_blocks={} named_sources={} logits_shape=[1,{vocabulary}]",
                model.graph().nodes().len(),
                model.graph().modules().len(),
                bound.len()
            );
            println!(
                "Model: tied embedding/head, 2 x Conv9 and 2 x LayerNorm per layer, EMA, SwiGLU."
            );
            println!("No autograd/training, BPC, strict Exp FP32 certificate or PyTorch parity.");
        }
        "--gpu" => {
            let engine = GpuEngine::new(
                "llc",
                std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
            )
            .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
            let directory =
                std::env::temp_dir().join(format!("vn_ema4_demo_{}", std::process::id()));
            let mut execution = engine.prepare_model(&model, weights.bindings(0)?, &directory)?;
            let mut reference_state = Ema4State::default();
            for token in [0, vocabulary / 2, vocabulary - 1] {
                execution.set_input("token", &onehot(token, vocabulary)?)?;
                execution.step()?;
                let actual = execution.output("logits")?;
                let expected = reference_step(&weights, token, &mut reference_state)?;
                let error = actual
                    .values()
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                if !error.is_finite() || error > 1e-3 {
                    return Err(format!("GPU/reference mismatch at token {token}: {error}").into());
                }
                println!("stream token={token} max_logit_abs_error={error:e}");
            }
            println!("EMA4-LM CUDA Graph: 3 autoregressive steps with on-device state feedback OK");
            println!("Experimental ex2.approx; verified sample values, not IEEE/SASS or training.");
        }
        _ => return Err(format!("usage: --inspect or --gpu (got {mode})").into()),
    }
    Ok(())
}
