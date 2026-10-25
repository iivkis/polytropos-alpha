//! Architecture-neutral CUDA Graph AdamW acceptance.
//! Host/GPU comparison is a numerical oracle, not IEEE/JIT/SASS proof.
use std::collections::BTreeMap;
use verineural::{
    AdamW, Domain, GpuAdamWPlan, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor,
};

fn tensor(v: &[f32; 4]) -> Tensor<f32> {
    Tensor::from_slice(vec![4], v).unwrap()
}
fn fixture(
    previous: &verineural::AdamWCheckpoint,
    values: &BTreeMap<String, Tensor<f32>>,
) -> (NeuralGraph, NodeId, NodeId, GpuAdamWPlan) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let p = graph.input(shape.clone(), Domain::Finite);
    let g = graph.input(shape, Domain::Finite);
    let pairs = BTreeMap::from([("weights".to_string(), (p, g))]);
    let plan = GpuAdamWPlan::append(&mut graph, &pairs, values, previous).unwrap();
    (graph, p, g, plan)
}
fn check_close(a: &[f32], b: &[f32], label: &str) {
    assert_eq!(a.len(), b.len());
    for (lane, (&left, &right)) in a.iter().zip(b).enumerate() {
        assert!(left.is_finite() && right.is_finite());
        let diff = (left - right).abs();
        let bound = 0.00002 * left.abs().max(right.abs()).max(1.);
        assert!(
            diff <= bound,
            "{label}[{lane}]: host={left:?} gpu={right:?} difference={diff} limit={bound}"
        );
    }
}
fn compare_checkpoints(
    actual: &verineural::AdamWCheckpoint,
    expected: &verineural::AdamWCheckpoint,
) {
    assert_eq!(actual.steps, expected.steps);
    for (name, (m, v)) in &expected.moments {
        let (actual_m, actual_v) = &actual.moments[name];
        check_close(actual_m, m, &format!("{name}:m"));
        check_close(actual_v, v, &format!("{name}:v"));
    }
}

#[test]
fn optimizer_is_a_generic_ir_extension_and_excludes_other_sources() {
    let initial = BTreeMap::from([("weights".into(), tensor(&[1., -2., 0.5, 3.]))]);
    let checkpoint = AdamW::new(0.002, 0.01).unwrap().checkpoint();
    let (graph, _parameter, _gradient, plan) = fixture(&checkpoint, &initial);
    assert_eq!(plan.feedback().len(), 3);
    assert_eq!(plan.outputs().len(), 3);
    assert_eq!(plan.correction_inputs(1).unwrap().len(), 2);
    let engine =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let program = engine.compile_outputs(&graph, &plan.outputs()).unwrap();
    assert!(program.kernel_count().unwrap() > 0);
    assert!(program.capability_plan().is_ok());
    // Universal Sqrt→Add→Div fusion is exercised by AdamW's v_hat
    // denominator, without an optimizer-specific compiler path.
    assert_eq!(program.fused_sqrt_add_divs().len(), 1);
    // Four-op fusion applies to AdamW's repeated gradient product
    // before bias correction, without a model- or optimizer-specific rule.
    assert_eq!(program.fused_mul_mul_add_divs().len(), 1);
    // Scalar inputs and two moment buffers are pure graph sources. They
    // do not appear in the list of optimizer-decayed parameters.
    assert_eq!(plan.bindings().len(), 11);
}

#[test]
fn scalar_gpu_broadcast_accepts_vector_matrix_and_higher_ranks_only_for_singletons() {
    let engine = GpuEngine::default();
    for dims in [vec![4], vec![2, 3], vec![2, 1, 3]] {
        let mut graph = NeuralGraph::default();
        let scalar = graph.input(Shape::new(vec![]).unwrap(), Domain::Finite);
        let copied = graph
            .broadcast_to(scalar, Shape::new(dims.clone()).unwrap())
            .unwrap();
        let program = engine.compile(&graph, copied).unwrap();
        assert_eq!(program.kernel_count().unwrap(), 1);
        assert_eq!(
            program.memory_contracts().unwrap()[0].reads,
            vec![(scalar.index(), 1)]
        );
        assert_eq!(
            program.translation_plan().unwrap().kernels[0].kind,
            verineural::TranslationKind::BroadcastScalar
        );
    }
    // The IR can describe general rank expansion, but the bounded GPU
    // implementation must not silently reinterpret a non-scalar source.
    let mut graph = NeuralGraph::default();
    let source = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let expanded = graph
        .broadcast_to(source, Shape::new(vec![2, 2]).unwrap())
        .unwrap();
    assert!(matches!(
        engine.compile(&graph, expanded),
        Err(verineural::GpuError::UnsupportedOperation {
            operation: "broadcast_to",
            ..
        })
    ));
}

#[test]
#[ignore = "requires NVIDIA CUDA and LLVM NVPTX; scalar broadcasting preserves bit patterns across ranks"]
fn scalar_broadcast_gpu_bits_match_source_across_ranks() {
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let directory =
        std::env::temp_dir().join(format!("vn_scalar_broadcast_{}", std::process::id()));
    for (i, dims) in [vec![4], vec![2, 3], vec![2, 1, 3]].into_iter().enumerate() {
        let mut graph = NeuralGraph::default();
        let scalar = graph.input(Shape::new(vec![]).unwrap(), Domain::Finite);
        let broadcast = graph
            .broadcast_to(scalar, Shape::new(dims.clone()).unwrap())
            .unwrap();
        let program = engine.compile(&graph, broadcast).unwrap();
        let initial = Tensor::from_slice(vec![], &[-0.0_f32]).unwrap();
        let output = engine
            .execute(
                &program,
                &[(scalar, initial)],
                &directory.join(format!("rank_{}", i + 1)),
            )
            .unwrap();
        assert_eq!(output.shape().dims(), dims);
        for &value in output.values() {
            assert_eq!(value.to_bits(), (-0.0_f32).to_bits());
        }
    }
}

#[test]
fn failed_optimizer_extension_does_not_modify_consumer_graph() {
    let mut graph = NeuralGraph::default();
    let p = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let wrong_gradient = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let invalid = BTreeMap::from([("weights".to_string(), (p, wrong_gradient))]);
    let values = BTreeMap::from([("weights".to_string(), tensor(&[1., 2., 3., 4.]))]);
    let saved = AdamW::new(0.002, 0.01).unwrap().checkpoint();
    let old_len = graph.nodes().len();
    assert!(GpuAdamWPlan::append(&mut graph, &invalid, &values, &saved).is_err());
    assert_eq!(graph.nodes().len(), old_len);
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph and LLVM NVPTX; two AdamW steps vs host plus bitwise restart"]
fn gpu_adamw_matches_host_and_resumes_bitwise() {
    let initial = BTreeMap::from([("weights".into(), tensor(&[1., -2., 0.5, 3.]))]);
    let adam = AdamW::new(0.002, 0.01).unwrap();
    let (graph, p, grad, plan) = fixture(&adam.checkpoint(), &initial);
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let program = engine.compile_outputs(&graph, &plan.outputs()).unwrap();
    let mut bindings = plan.bindings().to_vec();
    bindings.push((p, initial["weights"].clone()));
    bindings.push((grad, tensor(&[0.; 4])));
    let dir = std::env::temp_dir().join(format!("vn_gpu_adamw_{}", std::process::id()));
    let mut device = engine
        .prepare(&program, &bindings, &dir.join("first"))
        .unwrap();
    let mut cpu_params = initial;
    let mut cpu_adam = adam;
    let gradients = [
        tensor(&[0.1, -0.25, 1.5, -2.]),
        tensor(&[-0.5, 0.75, 0.125, 3.]),
        tensor(&[1., 1.25, -0.75, 0.25]),
    ];
    let mut snapshot = None;
    for (step, derivative) in gradients.iter().enumerate() {
        let correction = plan.correction_inputs(step as u64 + 1).unwrap();
        device.update_input(grad, derivative).unwrap();
        device
            .update_input_subset(&[
                (correction[0].0, &correction[0].1),
                (correction[1].0, &correction[1].1),
            ])
            .unwrap();
        device.replay().unwrap();
        device.feedback_state_checked(plan.feedback()).unwrap();
        let (params, moments) = plan.checkpoint(&mut device, step as u64 + 1).unwrap();
        cpu_adam
            .step(
                &mut cpu_params,
                &BTreeMap::from([("weights".into(), derivative.clone())]),
            )
            .unwrap();
        check_close(
            params["weights"].values(),
            cpu_params["weights"].values(),
            "param",
        );
        compare_checkpoints(&moments, &cpu_adam.checkpoint());
        if step == 1 {
            snapshot = Some((params.clone(), moments.clone()));
        }
        if step == 2 {
            // A new execution reconstructed from the 2-step checkpoint must
            // match the uninterrupted third GPU step exactly, even if its
            // results differ by a few ULP from the trusted host oracle.
            let (resumed_params, resumed_opt) = snapshot.take().unwrap();
            let (restart_graph, restart_p, restart_grad, restart_plan) =
                fixture(&resumed_opt, &resumed_params);
            let restart_program = engine
                .compile_outputs(&restart_graph, &restart_plan.outputs())
                .unwrap();
            let mut bound = restart_plan.bindings().to_vec();
            bound.push((restart_p, resumed_params["weights"].clone()));
            bound.push((restart_grad, derivative.clone()));
            let mut resumed = engine
                .prepare(&restart_program, &bound, &dir.join("restart"))
                .unwrap();
            let corrections = restart_plan.correction_inputs(3).unwrap();
            resumed
                .update_input_subset(&[
                    (corrections[0].0, &corrections[0].1),
                    (corrections[1].0, &corrections[1].1),
                ])
                .unwrap();
            resumed.replay().unwrap();
            resumed
                .feedback_state_checked(restart_plan.feedback())
                .unwrap();
            let (after, after_opt) = restart_plan.checkpoint(&mut resumed, 3).unwrap();
            for (name, result) in after {
                for (&a, &b) in result.values().iter().zip(params[&name].values()) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{name}: GPU restart param");
                }
            }
            assert_eq!(after_opt.steps, moments.steps);
            for (name, (m, v)) in &after_opt.moments {
                let (want_m, want_v) = &moments.moments[name];
                for (a, b) in m.iter().zip(want_m) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{name}: m");
                }
                for (a, b) in v.iter().zip(want_v) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{name}: v");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA/LLVM; domain-checked feedback rejects non-finite FP32 before mutating weights"]
fn checked_feedback_rejects_nonfinite_parameters_atomically() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let parameter = graph.input(shape.clone(), Domain::Finite);
    let denominator = graph.input(shape, Domain::Finite);
    let next = graph.div(parameter, denominator).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine.compile_outputs(&graph, &[next]).unwrap();
    let mut gpu = engine
        .prepare(
            &program,
            &[
                (parameter, tensor(&[1., 2., 3., 4.])),
                (denominator, tensor(&[0., 0., 0., 0.])),
            ],
            &std::env::temp_dir().join(format!("vn_feedback_finite_{}", std::process::id())),
        )
        .unwrap();
    gpu.replay().unwrap();
    assert!(gpu.feedback_state_checked(&[(parameter, next)]).is_err());
    for (&original, &actual) in [1.0_f32, 2., 3., 4.]
        .iter()
        .zip(gpu.read_source(parameter).unwrap().values())
    {
        assert_eq!(original.to_bits(), actual.to_bits());
    }
    gpu.update_input(denominator, &tensor(&[2., 2., 2., 2.]))
        .unwrap();
    gpu.replay().unwrap();
    gpu.feedback_state_checked(&[(parameter, next)]).unwrap();
    for (&expected, &actual) in [0.5_f32, 1., 1.5, 2.]
        .iter()
        .zip(gpu.read_source(parameter).unwrap().values())
    {
        assert_eq!(expected.to_bits(), actual.to_bits());
    }
}
