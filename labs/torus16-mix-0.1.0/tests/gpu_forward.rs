//! Torus16-Mix executable multi-output forward: the same Semantic IR that
//! defines the architecture is lowered to one CUDA Graph. No PyTorch,
//! host-side basis routing, rotated weight materialization or custom PTX.
use torus16_mix_lab::{Parameters, ROLES, WIDTH, architecture_model, reference_project};
use verineural::{GpuEngine, GpuLaunchConfig, Tensor, TranslationKind};

fn input() -> [f32; WIDTH] {
    std::array::from_fn(|i| ((i * 17 + 11) % 43) as f32 * 0.013 - 0.27)
}

fn parameters() -> Parameters {
    let mut p = Parameters::default();
    for role in 0..ROLES {
        for lane in 0..WIDTH {
            p.roles[role].row[lane] = 0.85 + (lane % 7) as f32 * 0.03;
            p.roles[role].col[lane] = 0.80 + (lane % 5) as f32 * 0.04;
            p.roles[role].bias[lane] = (lane % 3) as f32 * 0.001;
            p.roles[role].mix_gate[lane] = ((role * 13 + lane * 7) % 17) as f32 * 0.04 - 0.3;
        }
        p.roles[role].mix_bias = (role as f32 - 3.5) * 2.0;
    }
    p
}

#[test]
fn full_forward_requires_explicit_approx_exponential() {
    let model = architecture_model().unwrap();
    let outputs: Vec<_> = (0..ROLES)
        .map(|role| model.output(&format!("role_{role}")).unwrap())
        .collect();
    assert!(GpuEngine::default().compile_outputs(model.graph(), &outputs).is_err());
    let engine = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let compiled = engine.compile_outputs(model.graph(), &outputs).unwrap();
    assert_eq!(compiled.outputs(), outputs.as_slice());
    let plan = compiled.translation_plan().unwrap();
    for kind in [
        TranslationKind::Sub,
        TranslationKind::DivRne,
        TranslationKind::ExpApprox,
    ] {
        assert_eq!(plan.kernels.iter().filter(|k| k.kind == kind).count(), ROLES);
    }
    let text = plan.to_text();
    assert_eq!(verineural::TranslationPlan::from_text(&text).unwrap(), plan);
    assert!(compiled.memory_lean_source().unwrap().contains("slice_last_row_address_bound"));
}

#[test]
#[ignore = "requires NVIDIA CUDA, llc and a nonblocking CUDA Graph context"]
fn all_eight_gpu_outputs_match_independent_tensor_reference() {
    let model = architecture_model().unwrap();
    let outputs: Vec<_> = (0..ROLES)
        .map(|role| model.output(&format!("role_{role}")).unwrap())
        .collect();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_owned()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let graph = engine.compile_outputs(model.graph(), &outputs).unwrap();
    let p = parameters();
    let x = input();
    let bindings = model.bind(p.bindings(&x).unwrap()).unwrap();
    let path = std::env::temp_dir().join(format!("vn_torus16_gpu_{}", std::process::id()));
    let mut gpu = engine.prepare(&graph, &bindings, &path).unwrap();
    gpu.replay().unwrap();
    let results = gpu.read_outputs().unwrap();
    assert_eq!(results.len(), ROLES);
    for (role, (id, actual)) in results.iter().enumerate() {
        assert_eq!(*id, outputs[role]);
        let expected = reference_project(&x, &p, role).unwrap();
        for lane in 0..WIDTH {
            let error = (actual.values()[lane] - expected[lane]).abs();
            assert!(
                error <= 5e-5 + 2e-5 * expected[lane].abs(),
                "GPU/Tensor reference differs: role={role} lane={lane} \
                 actual={} expected={} abs_error={error}",
                actual.values()[lane],
                expected[lane]
            );
        }
    }
}


#[test]
#[ignore = "requires NVIDIA CUDA and LLVM NVPTX"]
fn persistent_named_execution_reuses_graph_after_new_input_and_parameter() {
    let model = architecture_model().unwrap();
    let mut p = parameters();
    let mut x = input();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_owned()),
    ).with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let path = std::env::temp_dir().join(format!("vn_torus16_named_{}", std::process::id()));
    let mut runner = engine.prepare_model(&model, p.bindings(&x).unwrap(), &path).unwrap();
    runner.run().unwrap();
    let original = runner.output("role_0").unwrap();
    let expected = reference_project(&x, &p, 0).unwrap();
    for lane in 0..WIDTH {
        assert!((original.values()[lane] - expected[lane]).abs() < 5e-5);
    }

    x = std::array::from_fn(|i| (i as f32 - 20.0) * 0.02);
    p.roles[0].mix_bias = -2.5;
    runner.set_input("x", &Tensor::from_slice(vec![1, WIDTH], &x).unwrap()).unwrap();
    runner.set_parameter(
        "mix_bias_0",
        &Tensor::from_slice(vec![1, 1], &[p.roles[0].mix_bias]).unwrap(),
    ).unwrap();
    runner.run().unwrap();
    let result = runner.output("role_0").unwrap();
    let expected = reference_project(&x, &p, 0).unwrap();
    assert!((0..WIDTH).any(|i| (result.values()[i] - original.values()[i]).abs() > 1e-5));
    for lane in 0..WIDTH {
        assert!((result.values()[lane] - expected[lane]).abs() < 5e-5,
            "role=0 lane={lane}");
    }
}
