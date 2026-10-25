//! EMA4-LM consumer acceptance: architecture shape, explicit state and
//! numerical autoregressive GPU steps. No PyTorch is involved.
use torus16_mix_lab::WIDTH;
use torus16_mix_lab::ema4::{
    Ema4Weights, HISTORY, LAYERS, MatrixMode, RESEARCH_VOCAB, architecture_model,
    architecture_model_for,
};
use torus16_mix_lab::ema4_reference::{Ema4State, reference_step};
use verineural::{GpuEngine, GpuLaunchConfig, Model, Tensor, unroll_model};

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
fn independent_matrices_only_untie_torus16_role_weights() {
    assert_eq!(MatrixMode::parse("independent"), Some(MatrixMode::Independent));
    assert_eq!(MatrixMode::parse("torus_shared"), Some(MatrixMode::TorusShared));
    assert_eq!(MatrixMode::parse("global_shared"), None);
    assert_eq!(MatrixMode::Independent.matrix_count(), LAYERS * 8);
    assert_eq!(MatrixMode::TorusShared.matrix_count(), LAYERS);
    let weights = Ema4Weights::deterministic(RESEARCH_VOCAB).unwrap();
    let shared = weights.trainable_parameters_for(MatrixMode::TorusShared).unwrap();
    let independent = weights.trainable_parameters_for(MatrixMode::Independent).unwrap();
    assert_eq!(shared.len() + LAYERS * 7, independent.len());
    assert_eq!(
        Ema4Weights::parameter_count_for(RESEARCH_VOCAB, MatrixMode::Independent),
        24_908 + 28 * WIDTH * WIDTH
    );
    assert_eq!(shared.values().map(|w| w.values().len()).sum::<usize>(), 24_908);
    assert_eq!(independent.values().map(|w| w.values().len()).sum::<usize>(), 89_420);
    // Each independent role starts with exactly the same W as the shared
    // candidate, including negative/subnormal FP32 bits.
    for layer in 0..LAYERS {
        let reference = &shared[&format!("weight_{layer}")];
        for role in 0..8 {
            let actual = &independent[&format!("weight_{layer}_{role}")];
            assert_eq!(reference.shape(), actual.shape());
            assert!(reference.values().iter().zip(actual.values()).all(|(a,b)| a.to_bits() == b.to_bits()));
        }
    }
    for (name, expected) in &shared {
        if name.starts_with("weight_") {
            continue;
        }
        let actual = &independent[name];
        assert_eq!(expected.shape(), actual.shape());
        assert!(expected.values().iter().zip(actual.values()).all(|(a,b)| a.to_bits() == b.to_bits()), "{name}");
    }
    // Both variants use the same public VeriNeural Model API and recurrent
    // transitions; invalid cross-mode bindings must be rejected.
    let shared_model = architecture_model_for(RESEARCH_VOCAB, MatrixMode::TorusShared).unwrap();
    let independent_model = architecture_model_for(RESEARCH_VOCAB, MatrixMode::Independent).unwrap();
    assert_eq!(shared_model.state_links().len(), independent_model.state_links().len());
    assert_eq!(shared_model.graph().modules().len(), independent_model.graph().modules().len());
    assert_eq!(shared_model.graph().modules().len(), 10 * LAYERS);
    for layer in 0..LAYERS {
        assert!(shared_model.source(&format!("weight_{layer}")).is_some());
        assert!(independent_model.source(&format!("weight_{layer}")).is_none());
        for role in 0..8 {
            assert!(independent_model.source(&format!("weight_{layer}_{role}")).is_some());
        }
    }
    assert!(shared_model.bind(weights.bindings_for(0, MatrixMode::TorusShared).unwrap()).is_ok());
    assert!(independent_model.bind(weights.bindings_for(0, MatrixMode::Independent).unwrap()).is_ok());
    assert!(shared_model.bind(weights.bindings_for(0, MatrixMode::Independent).unwrap()).is_err());
    assert!(independent_model.bind(weights.bindings_for(0, MatrixMode::TorusShared).unwrap()).is_err());
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph and LLVM NVPTX; compares initial independent/shared Torus16-Mix FP32 outputs bitwise"]
fn torus_matrix_modes_initial_gpu_logits_and_state_match_bits() {
    let vocabulary = 16;
    let weights = Ema4Weights::deterministic(vocabulary).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let root = std::env::temp_dir().join(format!("vn_torus_matrix_ablation_{}", std::process::id()));
    let tokens = [3, 7, 2];
    // Deliberately drop the baseline session before compiling independent
    // weights, because both four-layer CUDA Graphs can be large.
    let baseline = {
        let model = architecture_model_for(vocabulary, MatrixMode::TorusShared).unwrap();
        let mut execution = engine
            .prepare_model(
                &model,
                weights.bindings_for(0, MatrixMode::TorusShared).unwrap(),
                &root.join("shared"),
            )
            .unwrap();
        let mut observed = Vec::new();
        for &token in &tokens {
            execution.set_input("token", &onehot(token, vocabulary)).unwrap();
            execution.step().unwrap();
            observed.push((
                execution.output("logits").unwrap(),
                execution.state("mem_0").unwrap(),
            ));
        }
        observed
    };
    let model = architecture_model_for(vocabulary, MatrixMode::Independent).unwrap();
    let mut execution = engine
        .prepare_model(
            &model,
            weights.bindings_for(0, MatrixMode::Independent).unwrap(),
            &root.join("independent"),
        )
        .unwrap();
    for (step, &token) in tokens.iter().enumerate() {
        execution.set_input("token", &onehot(token, vocabulary)).unwrap();
        execution.step().unwrap();
        let actual_logits = execution.output("logits").unwrap();
        let actual_state = execution.state("mem_0").unwrap();
        for (name, actual, expected) in [
            ("logits", &actual_logits, &baseline[step].0),
            ("EMA state", &actual_state, &baseline[step].1),
        ] {
            assert_eq!(actual.shape(), expected.shape());
            for (lane, (&a, &b)) in actual.values().iter().zip(expected.values()).enumerate() {
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "step={step} {name} lane={lane} mismatched FP32 bits"
                );
            }
        }
    }
}

#[test]
fn seeded_matrix_ablation_preserves_matched_initialization_and_varies_diverse_roles() {
    use torus16_mix_lab::ema4::RoleInit;
    let a = Ema4Weights::deterministic_seeded(16, 123).unwrap();
    let b = Ema4Weights::deterministic_seeded(16, 123).unwrap();
    let c = Ema4Weights::deterministic_seeded(16, 124).unwrap();
    assert_eq!(a.blocks[0].operator.weight, b.blocks[0].operator.weight);
    assert_ne!(a.blocks[0].operator.weight, c.blocks[0].operator.weight);
    let shared = a.trainable_parameters_with_init(MatrixMode::TorusShared, RoleInit::Matched).unwrap();
    let matched = a.trainable_parameters_with_init(MatrixMode::Independent, RoleInit::Matched).unwrap();
    let diverse = a.trainable_parameters_with_init(MatrixMode::Independent, RoleInit::Diverse).unwrap();
    assert_eq!(matched["weight_0_0"], shared["weight_0"]);
    assert_eq!(matched["weight_0_7"], shared["weight_0"]);
    assert_ne!(diverse["weight_0_0"], diverse["weight_0_7"]);
    assert_ne!(diverse["weight_0_0"], shared["weight_0"]);
    assert_eq!(
        diverse.values().map(|v| v.values().len()).sum::<usize>(),
        matched.values().map(|v| v.values().len()).sum::<usize>()
    );
}

#[test]
#[ignore = "requires NVIDIA and LLVM NVPTX; compare summed independent-role VJPs against shared-W VJP"]
fn torus_role_matrix_gradients_sum_to_shared_gradient() {
    use torus16_mix_lab::ema4::RoleInit;
    let vocab = 16;
    let weights = Ema4Weights::deterministic_seeded(vocab, 13).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    ).with_launch_config(GpuLaunchConfig::default().with_approximate_exp());

    let gradients_for = |mode: MatrixMode| -> Vec<Vec<f32>> {
        let model = architecture_model_for(vocab, mode).unwrap();
        let logits = model.output("logits").unwrap();
        let names = if mode == MatrixMode::TorusShared {
            (0..LAYERS).map(|layer| format!("weight_{layer}")).collect::<Vec<_>>()
        } else {
            (0..LAYERS)
                .flat_map(|layer| (0..8).map(move |role| format!("weight_{layer}_{role}")))
                .collect::<Vec<_>>()
        };
        let sources = names.iter().map(|name| model.source(name).unwrap().0).collect::<Vec<_>>();
        let reverse = verineural::differentiate(model.graph(), logits, &sources).unwrap();
        let grads = sources.iter().map(|&id| reverse.gradient_of(id).unwrap()).collect::<Vec<_>>();
        let program = engine.compile_outputs(reverse.graph(), &grads).unwrap();
        let source_bindings = model.bind(
            weights.bindings_with_init(3, mode, RoleInit::Matched).unwrap()
        ).unwrap();
        let seed = Tensor::from_vec(
            vec![1, vocab], (0..vocab).map(|i| (i as f32 + 1.) / 16.).collect()
        ).unwrap();
        let supplied = reverse.bindings(&source_bindings, seed).unwrap();
        let supplied = supplied.into_iter()
            .filter(|(id, _)| program.source_inputs().contains(id))
            .collect::<Vec<_>>();
        let root = std::env::temp_dir().join(format!(
            "vn_role_vjp_{}_{}_{}", std::process::id(), vocab, mode.as_str()
        ));
        let mut gpu = engine.prepare(&program, &supplied, &root).unwrap();
        gpu.replay().unwrap();
        gpu.read_outputs_selected(&grads).unwrap().into_iter()
            .map(|(_, tensor)| tensor.values().to_vec()).collect()
    };
    // Running sequentially limits peak GPU memory usage.
    let shared = gradients_for(MatrixMode::TorusShared);
    let independent = gradients_for(MatrixMode::Independent);
    assert_eq!(shared.len(), LAYERS);
    assert_eq!(independent.len(), LAYERS * 8);
    for layer in 0..LAYERS {
        for (lane, &expected) in shared[layer].iter().enumerate() {
            let actual: f32 = independent[layer * 8..(layer + 1) * 8]
                .iter().map(|gradient| gradient[lane]).sum();
            // Different GPU reduction orders permit a bounded FP32 error;
            // this does not claim bitwise equivalence or formal proof.
            assert!(
                actual.is_finite() && expected.is_finite()
                    && (actual - expected).abs() <= 1e-4 + 0.02 * expected.abs(),
                "layer={layer} lane={lane}: shared={expected}, sum={actual}"
            );
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
fn gpu_training_upload_inventory_excludes_fixed_model_parameters() {
    let weights = Ema4Weights::deterministic(RESEARCH_VOCAB).unwrap();
    let trainable = weights.trainable_parameters().unwrap();
    let model = architecture_model(RESEARCH_VOCAB).unwrap();
    let window = unroll_model(&model, "logits", 2).unwrap();
    let named = window.parameters();

    // These source nodes are needed by the forward/backward program but
    // never belong to the AdamW parameter inventory or upload set.
    for fixed in ["ones_column", "inv_width", "epsilon_scalar", "one_vec"] {
        assert!(named.contains_key(fixed), "model source missing: {fixed}");
        assert!(
            !trainable.contains_key(fixed),
            "fixed source became trainable: {fixed}"
        );
    }
    assert!(named.len() > trainable.len());

    // Mirror the GPU trainer's exact zip over trainable names and target IDs.
    let names = trainable.keys().collect::<Vec<_>>();
    let targets = names
        .iter()
        .map(|name| {
            *named
                .get(*name)
                .expect("trainable name must have an IR source")
        })
        .collect::<Vec<_>>();
    let uploads = names
        .iter()
        .zip(&targets)
        .map(|(name, &id)| (id, &trainable[*name]))
        .collect::<Vec<_>>();
    assert_eq!(uploads.len(), trainable.len());
    assert_eq!(uploads.len(), targets.len());
    assert_eq!(
        uploads
            .iter()
            .map(|(_, tensor)| tensor.values().len())
            .sum::<usize>(),
        Ema4Weights::parameter_count(RESEARCH_VOCAB)
    );
    for fixed in ["ones_column", "inv_width", "epsilon_scalar", "one_vec"] {
        assert!(!uploads.iter().any(|(id, _)| *id == named[fixed]));
    }
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
    let model = architecture_model(16).unwrap();
    let logits = model.output("logits").unwrap();
    let final_bias = model.source("final_beta").unwrap().0;
    let reverse = verineural::differentiate(model.graph(), logits, &[final_bias]).unwrap();
    let gradient = reverse.gradient_of(final_bias).unwrap();
    assert_eq!(
        reverse.graph().node(gradient).unwrap().shape.dims(),
        &[1, WIDTH]
    );
    assert!(reverse.graph().nodes().len() > model.graph().nodes().len());
    let engine =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let program = engine
        .compile_outputs(reverse.graph(), &[logits, gradient])
        .unwrap();
    assert!(program.kernel_count().unwrap() > 0);
    let values = reverse
        .bindings(
            &model
                .bind(Ema4Weights::deterministic(16).unwrap().bindings(0).unwrap())
                .unwrap(),
            Tensor::zeros(vec![1, 16]).unwrap(),
        )
        .unwrap();
    let active = values
        .iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .count();
    assert_eq!(active, program.source_inputs().len());
}

#[test]
fn all_physical_ema4_parameters_are_identified_without_constants_or_states() {
    let weights = Ema4Weights::deterministic(RESEARCH_VOCAB).unwrap();
    let trainable = weights.trainable_parameters().unwrap();
    let count: usize = trainable.values().map(|v| v.values().len()).sum();
    assert_eq!(count, 24_908);
    assert_eq!(
        trainable["embedding"].values().len(),
        RESEARCH_VOCAB * WIDTH
    );
    assert!(trainable.contains_key("weight_0"));
    assert!(trainable.contains_key("weight_3"));
    assert!(trainable.contains_key("mix_gate_0_7"));
    assert!(trainable.contains_key("dw_3_2_8"));
    assert!(trainable.contains_key("final_beta"));
    for fixed in [
        "ones_column",
        "epsilon_scalar",
        "one_vec",
        "minus_vec",
        "inv_sqrt_width",
    ] {
        assert!(!trainable.contains_key(fixed));
    }
    assert!(
        trainable
            .keys()
            .all(|key| !key.starts_with("history_") && !key.starts_with("mem_"))
    );
}

#[test]
#[ignore = "expensive full two-step reverse graph; may require ample host memory"]
fn full_bptt_gradient_graph_targets_every_trainable_parameter() {
    let weights = Ema4Weights::deterministic(16).unwrap();
    let model = architecture_model(16).unwrap();
    let unrolled = verineural::unroll_model(&model, "logits", 2).unwrap();
    let trainable = weights.trainable_parameters().unwrap();
    let targets = trainable
        .keys()
        .map(|name| unrolled.parameters()[name])
        .collect::<Vec<_>>();
    let reverse =
        verineural::differentiate(unrolled.graph(), unrolled.objective(), &targets).unwrap();
    assert_eq!(reverse.gradients().len(), trainable.len());
    for &id in &targets {
        assert_eq!(
            reverse
                .graph()
                .node(reverse.gradient_of(id).unwrap())
                .unwrap()
                .shape,
            unrolled.graph().node(id).unwrap().shape
        );
    }
}
