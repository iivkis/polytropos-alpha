//! Device-side feedback invariants for raw GpuExecution: readiness,
//! preflight-only rejection, and observable state/reshape aliases.
use verineural::{
    Domain, GpuEngine, GpuError, NeuralGraph, Shape, Tensor, TensorValue,
};

fn tensor(data: &[f32; 4]) -> Tensor<f32> {
    Tensor::from_slice(vec![4], data).unwrap()
}

fn expect_bits(actual: &Tensor<f32>, expected: &[f32; 4]) {
    assert_eq!(
        actual.values().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

fn expect_f32(value: &TensorValue, expected: &[f32; 4]) {
    match value {
        TensorValue::F32(value) => expect_bits(value, expected),
        TensorValue::Index(_) => panic!("feedback readout must remain F32"),
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX; checks raw CUDA Graph feedback and output alias snapshots"]
fn direct_feedback_requires_fresh_outputs_and_preserves_selected_aliases() {
    let mut graph = NeuralGraph::default();
    let state = graph.input(Shape::new(vec![4]).unwrap(), Domain::Any);
    let token = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let view = graph
        .reshape(state, Shape::new(vec![2, 2]).unwrap())
        .unwrap();
    let next = graph.add(state, token).unwrap();
    // Unlike ModelExecution, raw GpuExecution treats every selected root
    // as an observable value, including Input and its zero-copy view.
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine
        .compile_outputs(&graph, &[state, view, next])
        .unwrap();
    let state_initial = [3., 5., 7., 9.];
    let token_initial = [1., 2., 3., 4.];
    let workdir = std::env::temp_dir()
        .join(format!("vn_feedback_guard_{}", std::process::id()));
    let mut gpu = engine
        .prepare(
            &program,
            &[(state, tensor(&state_initial)), (token, tensor(&token_initial))],
            &workdir,
        )
        .unwrap();

    // No CUDA Graph replay has produced next yet. Reject before D2D.
    assert!(matches!(
        gpu.feedback_state(&[(state, next)]),
        Err(GpuError::OutputNotReady)
    ));
    expect_bits(&gpu.read_source(state).unwrap(), &state_initial);

    gpu.replay().unwrap();
    // Bad mappings fail before a single state is mutated; rejection does
    // not poison valid output readiness or CUDA execution.
    assert!(matches!(
        gpu.feedback_state(&[(state, next), (state, view)]),
        Err(GpuError::InvalidInput {
            reason: "duplicate state feedback destination",
            ..
        })
    ));
    expect_bits(&gpu.read_source(state).unwrap(), &state_initial);
    gpu.feedback_state(&[(state, next)]).unwrap();

    let first_state = [4., 7., 10., 13.];
    expect_bits(&gpu.read_source(state).unwrap(), &first_state);
    expect_f32(&gpu.read_value().unwrap(), &state_initial);
    let preserved = gpu.read_outputs_selected(&[view, state, next]).unwrap();
    assert_eq!(preserved[0].1.shape().dims(), &[2, 2]);
    expect_bits(&preserved[0].1, &state_initial);
    expect_bits(&preserved[1].1, &state_initial);
    expect_bits(&preserved[2].1, &first_state);

    let new_token = [2., 4., 6., 8.];
    gpu.update_input(token, &tensor(&new_token)).unwrap();
    assert!(matches!(
        gpu.feedback_state(&[(state, next)]),
        Err(GpuError::OutputNotReady)
    ));
    expect_bits(&gpu.read_source(state).unwrap(), &first_state);

    // A new replay invalidates the previous snapshot, even if no feedback
    // has happened yet. Reading primary and all outputs must see new data.
    gpu.replay().unwrap();
    expect_f32(&gpu.read_value().unwrap(), &first_state);
    let current = gpu.read_values().unwrap();
    expect_f32(&current[0].1, &first_state);
    expect_f32(&current[1].1, &first_state);
    let second_state = [6., 11., 16., 21.];
    expect_f32(&current[2].1, &second_state);

    gpu.feedback_state(&[(state, next)]).unwrap();
    expect_bits(&gpu.read_source(state).unwrap(), &second_state);
    let preserved_again = gpu.read_values().unwrap();
    expect_f32(&preserved_again[0].1, &first_state);
    expect_f32(&preserved_again[1].1, &first_state);
    expect_f32(&preserved_again[2].1, &second_state);
}
