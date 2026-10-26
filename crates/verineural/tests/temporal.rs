//! Small architecture-neutral temporal BPTT acceptance model.
use verineural::{
    Domain, GpuEngine, Model, ModelBuilder, Operation, Tensor, differentiate, unroll_model,
    unroll_model_hoisted,
};

fn recurrent() -> verineural::Model {
    let mut m = ModelBuilder::new("test-temporal", "1");
    let s = m.state("s", [1, 1], Domain::Any).unwrap();
    let x = m.input("x", [1, 1], Domain::Finite).unwrap();
    let w = m.parameter("w", [1, 1], Domain::Finite).unwrap();
    let y = m.graph_mut().mul(s, w).unwrap();
    let y = m.graph_mut().add(y, x).unwrap();
    m.next_state("s", y).unwrap();
    m.output("y", y).unwrap();
    m.finish().unwrap()
}
fn scalar(v: f32) -> Tensor<f32> {
    Tensor::from_slice(vec![1, 1], &[v]).unwrap()
}

#[test]
fn temporal_graph_has_shared_parameters_and_actual_state_edges() {
    let model = recurrent();
    assert!(unroll_model(&model, "bad", 2).is_err());
    assert!(unroll_model(&model, "y", 0).is_err());
    assert!(unroll_model(&model, "y", 17).is_err());
    let window = unroll_model(&model, "y", 2).unwrap();
    assert_eq!(window.outputs().len(), 2);
    assert_eq!(window.cotangents().len(), 2);
    assert_eq!(window.parameters().len(), 1);
    assert_eq!(window.initial_states().len(), 1);
    assert_eq!(window.terminal_states().len(), 1);
    let wrt = window.parameters()["w"];
    let grad = differentiate(window.graph(), window.objective(), &[wrt]).unwrap();
    assert_eq!(
        grad.graph()
            .node(grad.gradient_of(wrt).unwrap())
            .unwrap()
            .shape
            .dims(),
        &[1, 1]
    );
    let base = vec![
        ("s".to_owned(), scalar(0.5)),
        ("w".to_owned(), scalar(0.75)),
    ];
    let steps = vec![
        vec![("x".to_owned(), scalar(1.))],
        vec![("x".to_owned(), scalar(2.))],
    ];
    let seeds = vec![scalar(1.), scalar(1.)];
    assert_eq!(window.bindings(&base, &steps, &seeds).unwrap().len(), 6);
    assert!(window.bindings(&base, &steps, &seeds[0..1]).is_err());
    assert!(
        window
            .bindings(&base, &steps, &[scalar(1.), scalar(f32::NAN)])
            .is_err()
    );
    assert!(window.bindings(&base[0..1], &steps, &seeds).is_err());
    assert!(
        window
            .bindings(
                &base,
                &steps,
                &[scalar(1.), Tensor::zeros(vec![1, 2]).unwrap()]
            )
            .is_err()
    );
}

#[test]
#[ignore = "requires NVIDIA GPU + LLVM NVPTX, checks through-time gradient"]
fn temporal_gpu_vjp_matches_hand_derived_gradient() {
    let window = unroll_model(&recurrent(), "y", 2).unwrap();
    let parameter = window.parameters()["w"];
    let derivative = differentiate(window.graph(), window.objective(), &[parameter]).unwrap();
    let root = derivative.gradient_of(parameter).unwrap();
    let mut targets = window.outputs().to_vec();
    targets.push(root);
    let gpu = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = gpu.compile_outputs(derivative.graph(), &targets).unwrap();
    let base = vec![
        ("s".to_owned(), scalar(0.5)),
        ("w".to_owned(), scalar(0.75)),
    ];
    let steps = vec![
        vec![("x".to_owned(), scalar(1.))],
        vec![("x".to_owned(), scalar(2.))],
    ];
    let binding = window
        .bindings(&base, &steps, &[scalar(1.), scalar(1.)])
        .unwrap();
    let binding = derivative.bindings(&binding, scalar(1.)).unwrap();
    let sources = binding
        .into_iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .collect::<Vec<_>>();
    let path = std::env::temp_dir().join(format!("vn_temporal_vjp_{}", std::process::id()));
    let values = gpu.execute_outputs(&program, &sources, &path).unwrap();
    assert_eq!(values.len(), 3);
    assert!((values[0].1.values()[0] - 1.375).abs() < 1e-5);
    assert!((values[1].1.values()[0] - 3.03125).abs() < 1e-5);
    // d(s1+s2)/dw = s0 + s1 + w*s0 = 2.25.
    // Omitting the temporal edge yields 1.875 and must fail this test.
    assert!((values[2].1.values()[0] - 2.25).abs() < 1e-5);
}

// Pure parameter expressions need only one materialization per BPTT window.
// The state-dependent graph must still be rebuilt and differentiated per step.
fn invariant_recurrent() -> Model {
    let mut m = ModelBuilder::new("temporal-hoist", "1");
    let state = m.state("s", [1, 2], Domain::Any).unwrap();
    let x = m.input("x", [1, 2], Domain::Finite).unwrap();
    let w = m.parameter("w", [2, 2], Domain::Finite).unwrap();
    let b = m.parameter("b", [1, 2], Domain::Finite).unwrap();
    let wt = m.graph_mut().permute(w, &[1, 0]).unwrap();
    let b2 = m.graph_mut().mul(b, b).unwrap();
    let input = m.graph_mut().add(state, x).unwrap();
    let mixed = m.graph_mut().matmul(input, wt).unwrap();
    let y = m.graph_mut().add(mixed, b2).unwrap();
    m.next_state("s", y).unwrap();
    m.output("y", y).unwrap();
    m.finish().unwrap()
}

#[test]
fn temporal_parameter_only_hoisting_reuses_immutable_nodes_but_not_states() {
    let model = invariant_recurrent();
    let baseline = unroll_model(&model, "y", 4).unwrap();
    let optimized = unroll_model_hoisted(&model, "y", 4).unwrap();
    let count_permute = |graph: &verineural::NeuralGraph| {
        graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.operation, Operation::Permute { .. }))
            .count()
    };
    // Each timestep also appends one transpose for the dot-product
    // objective. Hoisting removes only the four repeated W transposes,
    // reducing those to one; objective transposes must remain per-step.
    assert_eq!(count_permute(baseline.graph()), 8);
    assert_eq!(count_permute(optimized.graph()), 5);
    assert_eq!(
        count_permute(baseline.graph()) - count_permute(optimized.graph()),
        3
    );
    assert_eq!(
        baseline.graph().nodes().len() - optimized.graph().nodes().len(),
        6, // Three redundant transposes and three redundant bias products.
    );
    assert_eq!(
        baseline.parameters().keys().collect::<Vec<_>>(),
        optimized.parameters().keys().collect::<Vec<_>>()
    );
    assert_eq!(baseline.inputs().len(), optimized.inputs().len());
    assert_eq!(optimized.outputs().len(), 4);
    assert_eq!(optimized.cotangents().len(), 4);
    assert_ne!(optimized.outputs()[0], optimized.outputs()[1]);
    assert_eq!(optimized.terminal_states()["s"], optimized.outputs()[3]);
}

#[test]
#[ignore = "requires NVIDIA GPU + LLVM NVPTX; checks hoisted temporal logits and VJPs"]
fn temporal_parameter_hoisting_gpu_forward_and_gradients_match_baseline() {
    let model = invariant_recurrent();
    let base = vec![
        (
            "w".to_owned(),
            Tensor::from_slice(vec![2, 2], &[0.3, -0.2, 0.25, 0.4]).unwrap(),
        ),
        (
            "b".to_owned(),
            Tensor::from_slice(vec![1, 2], &[0.1, 0.2]).unwrap(),
        ),
        (
            "s".to_owned(),
            Tensor::from_slice(vec![1, 2], &[0.6, -0.1]).unwrap(),
        ),
    ];
    let steps = vec![
        vec![(
            "x".to_owned(),
            Tensor::from_slice(vec![1, 2], &[0.4, 0.2]).unwrap(),
        )],
        vec![(
            "x".to_owned(),
            Tensor::from_slice(vec![1, 2], &[-0.1, 0.3]).unwrap(),
        )],
        vec![(
            "x".to_owned(),
            Tensor::from_slice(vec![1, 2], &[0.2, -0.4]).unwrap(),
        )],
    ];
    let seeds = vec![
        Tensor::from_slice(vec![1, 2], &[1.0, 0.5]).unwrap(),
        Tensor::from_slice(vec![1, 2], &[-0.2, 0.75]).unwrap(),
        Tensor::from_slice(vec![1, 2], &[0.1, 0.6]).unwrap(),
    ];
    // Run the same GPU forward/VJP parity at the historical three-step
    // horizon and at the first v0.2 target horizon of eight timesteps.
    // Repeated step values are deterministic; state carries through all eight.
    let short_steps = steps.clone();
    let short_seeds = seeds.clone();
    let mut steps = steps;
    let mut seeds = seeds;
    for i in 3..8 {
        steps.push(short_steps[i % short_steps.len()].clone());
        seeds.push(short_seeds[i % short_seeds.len()].clone());
    }
    let gpu = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let evaluate = |hoisted: bool, horizon: usize| {
        let window = if hoisted {
            unroll_model_hoisted(&model, "y", horizon).unwrap()
        } else {
            unroll_model(&model, "y", horizon).unwrap()
        };
        let wrt = [window.parameters()["w"], window.parameters()["b"]];
        let reverse = differentiate(window.graph(), window.objective(), &wrt).unwrap();
        let mut roots = window.outputs().to_vec();
        for &id in &wrt {
            roots.push(reverse.gradient_of(id).unwrap());
        }
        let program = gpu.compile_outputs(reverse.graph(), &roots).unwrap();
        let bindings = window
            .bindings(&base, &steps[..horizon], &seeds[..horizon])
            .unwrap();
        let bindings = reverse
            .bindings(&bindings, Tensor::from_slice(vec![1, 1], &[1.0]).unwrap())
            .unwrap();
        let sources = bindings
            .into_iter()
            .filter(|(id, _)| program.source_inputs().contains(id))
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!(
            "vn_temporal_hoist_{}_{}_{}",
            std::process::id(),
            horizon,
            if hoisted { "shared" } else { "baseline" },
        ));
        gpu.execute_outputs(&program, &sources, &path)
            .unwrap()
            .into_iter()
            .map(|(_, tensor)| tensor)
            .collect::<Vec<_>>()
    };
    for horizon in [3usize, 8] {
        let original = evaluate(false, horizon);
        let hoisted = evaluate(true, horizon);
        assert_eq!(original.len(), horizon + 2);
        assert_eq!(original.len(), hoisted.len());
        for (index, (expected, found)) in original.iter().zip(&hoisted).enumerate() {
            assert_eq!(expected.shape(), found.shape());
            for (&a, &b) in expected.values().iter().zip(found.values()) {
                if index < horizon {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "horizon={horizon} forward timestep {index}"
                    );
                } else {
                    // Floating-point reverse reduction topology can change
                    // under hoisting, so compare VJPs numerically, not bitwise.
                    let tolerance = if horizon == 3 { 1e-4 } else { 1e-3 };
                    assert!(
                        (a - b).abs() <= tolerance,
                        "horizon={horizon} parameter VJP {index}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
