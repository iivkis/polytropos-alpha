//! Small architecture-neutral temporal BPTT acceptance model.
use verineural::{Domain, GpuEngine, ModelBuilder, Tensor, differentiate, unroll_model};

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
