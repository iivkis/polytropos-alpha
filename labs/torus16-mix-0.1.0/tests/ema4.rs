//! EMA4-LM consumer acceptance: architecture shape, explicit state and
//! numerical autoregressive GPU steps. No PyTorch is involved.
use torus16_mix_lab::WIDTH;
use torus16_mix_lab::ema4::{Ema4Weights, HISTORY, LAYERS, RESEARCH_VOCAB, architecture_model};
use torus16_mix_lab::ema4_reference::{Ema4State, reference_step};
use verineural::{GpuEngine, GpuLaunchConfig, Model, Tensor};

fn onehot(token: usize, vocabulary: usize) -> Tensor<f32> {
    let mut values = vec![0.; vocabulary];
    values[token] = 1.;
    Tensor::from_vec(vec![1, vocabulary], values).unwrap()
}

fn validate_model(model: &Model, vocab: usize) {
    assert_eq!(model.outputs().len(), 1);
    assert_eq!(model.graph().modules().len(), 10 * LAYERS);
    assert_eq!(model.state_links().len(), LAYERS * (1 + 2 * HISTORY));
    let logits = model.graph().node(model.output("logits").unwrap()).unwrap();
    assert_eq!(logits.shape.dims(), &[1, vocab]);
    assert!(model.source("embedding").is_some());
    for layer in 0..LAYERS {
        assert!(model.source(&format!("weight_{layer}")).is_some());
        assert!(model.state_links().contains_key(&format!("mem_{layer}")));
        for stage in 1..=2 {
            for i in 0..HISTORY {
                let name = format!("history_{layer}_{stage}_{i}");
                let &next = model
                    .state_links()
                    .get(&name)
                    .expect("explicit history transition");
                let next_node = model.graph().node(next).unwrap();
                if i + 1 < HISTORY {
                    // Materialized address-only copy; simultaneous GPU feedback
                    // must not alias another state input that is being overwritten.
                    assert!(matches!(
                        next_node.operation,
                        verineural::Operation::SliceAxis {
                            axis: 1,
                            start: 0,
                            len: WIDTH
                        }
                    ));
                }
            }
        }
    }
}

#[test]
fn research_shape_and_named_bindings() {
    let weights = Ema4Weights::deterministic(RESEARCH_VOCAB).unwrap();
    assert_eq!(Ema4Weights::parameter_count(RESEARCH_VOCAB), 24_908);
    let model = architecture_model(RESEARCH_VOCAB).unwrap();
    validate_model(&model, RESEARCH_VOCAB);
    let bound = model.bind(weights.bindings(0).unwrap()).unwrap();
    assert_eq!(
        bound.len(),
        model
            .graph()
            .nodes()
            .iter()
            .filter(|n| matches!(n.operation, verineural::Operation::Input))
            .count()
    );
    assert!(Ema4Weights::deterministic(0).is_err());
    assert!(weights.bindings(RESEARCH_VOCAB).is_err());
}

#[test]
fn reference_streaming_is_causal_and_resets() {
    let weights = Ema4Weights::deterministic(16).unwrap();
    let mut first = Ema4State::default();
    let mut second = Ema4State::default();
    let output_a = reference_step(&weights, 3, &mut first).unwrap();
    let output_b = reference_step(&weights, 3, &mut second).unwrap();
    assert_eq!(output_a, output_b);
    assert_eq!(first.0[0].memory, second.0[0].memory);
    let next = reference_step(&weights, 7, &mut first).unwrap();
    assert!(next.iter().all(|v| v.is_finite()));
    assert_eq!(first.0[0].history1[HISTORY - 1].len(), WIDTH);
    assert_ne!(
        first.0[0].history1[HISTORY - 1],
        second.0[0].history1[HISTORY - 1]
    );
}

#[test]
#[ignore = "needs LLVM NVPTX and an NVIDIA GPU; full 4-layer CUDA Graph"]
fn gpu_streaming_logits_and_state_match_cpu_oracle() {
    let vocabulary = 16;
    let weights = Ema4Weights::deterministic(vocabulary).unwrap();
    let model = architecture_model(vocabulary).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_string()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let workspace = std::env::temp_dir().join(format!("vn_ema4_{}", std::process::id()));
    let mut runner = engine
        .prepare_model(&model, weights.bindings(3).unwrap(), &workspace)
        .unwrap();
    let mut state = Ema4State::default();
    for token in [3, 7, 2] {
        runner
            .set_input("token", &onehot(token, vocabulary))
            .unwrap();
        runner.step().unwrap();
        let actual = runner.output("logits").unwrap();
        let expected = reference_step(&weights, token, &mut state).unwrap();
        for (i, (a, b)) in actual.values().iter().zip(expected.iter()).enumerate() {
            assert!(a.is_finite());
            assert!(
                (a - b).abs() <= 1e-3 + b.abs() * 1e-3,
                "token={token} logit={i} actual={a} expected={b}"
            );
        }
        let gpu_memory = runner.state("mem_0").unwrap();
        for lane in 0..WIDTH {
            assert!((gpu_memory.values()[lane] - state.0[0].memory[lane]).abs() < 1e-3);
        }
    }
}


#[test]
fn ema4_single_step_gradient_stays_on_public_semantic_ir() {
    let model=architecture_model(16).unwrap();
    let logits=model.output("logits").unwrap();
    let final_bias=model.source("final_beta").unwrap().0;
    let reverse=verineural::differentiate(model.graph(),logits,&[final_bias]).unwrap();
    let gradient=reverse.gradient_of(final_bias).unwrap();
    assert_eq!(reverse.graph().node(gradient).unwrap().shape.dims(),&[1,WIDTH]);
    assert!(reverse.graph().nodes().len()>model.graph().nodes().len());
    let engine=GpuEngine::default().with_launch_config(
        GpuLaunchConfig::default().with_approximate_exp());
    let program=engine.compile_outputs(reverse.graph(),&[logits,gradient]).unwrap();
    assert!(program.kernel_count().unwrap()>0);
    let values=reverse.bindings(
        &model.bind(Ema4Weights::deterministic(16).unwrap().bindings(0).unwrap()).unwrap(),
        Tensor::zeros(vec![1,16]).unwrap()).unwrap();
    let active=values.iter().filter(|(id,_)|program.source_inputs().contains(id)).count();
    assert_eq!(active,program.source_inputs().len());
}
