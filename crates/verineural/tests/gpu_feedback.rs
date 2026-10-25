//! Device-side feedback invariants for raw GpuExecution: readiness,
//! preflight-only rejection, and observable state/reshape aliases.
use verineural::{Domain, GpuEngine, GpuError, NeuralGraph, Shape, Tensor, TensorValue};

fn tensor(data: &[f32; 4]) -> Tensor<f32> {
    Tensor::from_slice(vec![4], data).unwrap()
}

fn expect_bits(actual: &Tensor<f32>, expected: &[f32; 4]) {
    assert_eq!(
        actual
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
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
    let workdir = std::env::temp_dir().join(format!("vn_feedback_guard_{}", std::process::id()));
    let mut gpu = engine
        .prepare(
            &program,
            &[
                (state, tensor(&state_initial)),
                (token, tensor(&token_initial)),
            ],
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

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX; checks D2D feedback parity with host readback/rebind over multiple graph replays"]
fn device_feedback_matches_host_staged_state_bitwise() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let state_a = graph.input(shape.clone(), Domain::Any);
    let state_b = graph.input(shape.clone(), Domain::Any);
    let token = graph.input(shape, Domain::Finite);
    let next_a = graph.add(state_a, token).unwrap();
    let next_b = graph.mul(state_b, token).unwrap();
    let logits = graph.add(next_a, next_b).unwrap();
    let outputs = [next_a, next_b, logits];
    let links = [(state_a, next_a), (state_b, next_b)];
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine.compile_outputs(&graph, &outputs).unwrap();
    let sources = [
        (state_a, tensor(&[0.5, -1.5, 2.0, 0.25])),
        (state_b, tensor(&[1., -0.5, 0., 4.])),
        (token, tensor(&[0., 1., 2., 3.])),
    ];
    let dir = std::env::temp_dir().join(format!("vn_feedback_parity_{}", std::process::id()));
    let mut device = engine
        .prepare(&program, &sources, &dir.join("device"))
        .unwrap();
    let mut host = engine
        .prepare(&program, &sources, &dir.join("host"))
        .unwrap();

    // Different tokens and two independent recurrent states exercise more
    // than a one-step identity feedback. Both sessions execute the same DAG;
    // only the state transport between windows differs.
    let tokens = [
        [0.25, -0.5, 2., 1.],
        [1., 0.25, -0.5, 2.],
        [0., 1.5, 1., -0.25],
        [-1., 0.5, 2., 0.75],
    ];
    for (step, values) in tokens.iter().enumerate() {
        let input = tensor(values);
        device.update_input(token, &input).unwrap();
        host.update_input(token, &input).unwrap();
        device.replay().unwrap();
        host.replay().unwrap();

        let left = device.read_outputs_selected(&outputs).unwrap();
        let right = host.read_outputs_selected(&outputs).unwrap();
        assert_eq!(left.len(), right.len());
        for ((id, actual), (expected_id, expected)) in left.iter().zip(&right) {
            assert_eq!(*id, *expected_id);
            for (lane, (&a, &b)) in actual.values().iter().zip(expected.values()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step={step} output={} lane={lane}",
                    id.index()
                );
            }
        }

        // Old TBPTT strategy: selected next states D2H, then both source
        // inputs H2D. New strategy: selected next states directly D2D.
        // State updates happen after comparing the pure DAG output.
        device.feedback_state(&links).unwrap();
        let staged = host.read_outputs_selected(&[next_a, next_b]).unwrap();
        host.update_input_subset(&[(state_a, &staged[0].1), (state_b, &staged[1].1)])
            .unwrap();
        for &state in &[state_a, state_b] {
            let new = device.read_source(state).unwrap();
            let old = host.read_source(state).unwrap();
            for (lane, (&a, &b)) in new.values().iter().zip(old.values()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step={step} state={} lane={lane}",
                    state.index()
                );
            }
        }
    }
    // An unchanged plan must reuse its CUDA Graph, but a different (valid)
    // simultaneous assignment must invalidate the descriptor cache.
    let previous = device.read_outputs_selected(&[next_a, next_b]).unwrap();
    let swapped = [(state_a, next_b), (state_b, next_a)];
    device.feedback_state(&swapped).unwrap();
    let updated_a = device.read_source(state_a).unwrap();
    let updated_b = device.read_source(state_b).unwrap();
    for (actual, expected) in [(updated_a, &previous[1].1), (updated_b, &previous[0].1)] {
        for (&a, &b) in actual.values().iter().zip(expected.values()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph; grouped guard must reject before any grouped feedback write"]
fn grouped_feedback_rejects_one_invalid_tensor_without_mutating_other_sizes() {
    let mut graph = NeuralGraph::default();
    let mut bindings = Vec::new();
    let mut states = Vec::new();
    let mut tokens = Vec::new();
    let mut nexts = Vec::new();
    let mut originals = Vec::new();
    for &length in &[1usize, 257, 1025] {
        let shape = Shape::new(vec![length]).unwrap();
        let state = graph.input(shape.clone(), Domain::Finite);
        let token = graph.input(shape, Domain::Finite);
        let next = graph.add(state, token).unwrap();
        let mut initial = vec![1.0f32; length];
        let mut update = vec![2.0f32; length];
        if length == 257 {
            initial[0] = f32::MAX;
            update[0] = f32::MAX;
        }
        originals.push(initial.clone());
        bindings.push((state, Tensor::from_slice(vec![length], &initial).unwrap()));
        bindings.push((token, Tensor::from_slice(vec![length], &update).unwrap()));
        states.push(state);
        tokens.push(token);
        nexts.push(next);
    }
    let links = states
        .iter()
        .copied()
        .zip(nexts.iter().copied())
        .collect::<Vec<_>>();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine.compile_outputs(&graph, &nexts).unwrap();
    let dir = std::env::temp_dir().join(format!("vn_grouped_feedback_{}", std::process::id()));
    let mut gpu = engine.prepare(&program, &bindings, &dir).unwrap();
    gpu.replay().unwrap();
    assert!(gpu.feedback_state_checked(&links).is_err());
    for (i, &state) in states.iter().enumerate() {
        let unchanged = gpu.read_source(state).unwrap();
        assert_eq!(
            unchanged
                .values()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            originals[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let mut corrected = vec![2.0f32; 257];
    corrected[0] = -f32::MAX;
    gpu.update_input(
        tokens[1],
        &Tensor::from_slice(vec![257], &corrected).unwrap(),
    )
    .unwrap();
    gpu.replay().unwrap();
    // Optional event profiling launches exactly one feedback graph and
    // retains the same mandatory guard/atomicity semantics.
    let profiled = gpu.feedback_state_checked_device_profiled(&links).unwrap();
    assert!(profiled.device_copy.is_some());
    for (i, &state) in states.iter().enumerate() {
        let updated = gpu.read_source(state).unwrap();
        assert_eq!(updated.values().len(), originals[i].len());
        for (lane, &value) in updated.values().iter().enumerate() {
            let want = if i == 1 && lane == 0 { 0.0 } else { 3.0 };
            assert_eq!(value.to_bits(), (want as f32).to_bits());
        }
    }
}
