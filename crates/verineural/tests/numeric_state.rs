//! P1 fixed-horizon recurrent FP32 error composition.
//! Previous next_state contracts are fed into the next step as source
//! ApproxContracts. Fresh non-state inputs remain exact per step.

use verineural::{
    Domain, LeanProofEngine, NeuralGraph, NumericError, RneStateRolloutPlan, Shape,
};

fn additive_transition() -> (NeuralGraph, verineural::NodeId, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let state = graph.input(shape.clone(), Domain::Finite);
    let token = graph.input(shape, Domain::Finite);
    let sum = graph.add(state, token).unwrap();
    let next = graph.relu(sum).unwrap();
    (graph, state, next)
}

#[test]
fn recurrent_rollout_carries_error_into_the_next_state_source() {
    let (graph, state, next) = additive_transition();
    let rollout =
        RneStateRolloutPlan::compile_fixed(&graph, state, next, 20, 8_388_608, 4).unwrap();

    assert_eq!(rollout.steps().len(), 4);
    assert_eq!(rollout.initial_contract().epsilon_ticks(), 0);
    assert_eq!(rollout.state(), state);
    assert_eq!(rollout.next_state(), next);

    for pair in rollout.steps().windows(2) {
        assert_eq!(pair[1].input_state(), pair[0].output_state());
        assert!(
            pair[1].output_state().epsilon_ticks() >= pair[1].input_state().epsilon_ticks(),
            "recurrent error budget must not silently reset"
        );
    }
    assert!(rollout.steps()[0].output_state().epsilon_ticks() > 0);
    assert!(
        rollout.final_contract().epsilon_ticks()
            >= rollout.steps()[0].output_state().epsilon_ticks()
    );
}

#[test]
fn second_step_lean_source_uses_the_previous_nonzero_state_contract() {
    let (graph, state, next) = additive_transition();
    let rollout =
        RneStateRolloutPlan::compile_fixed(&graph, state, next, 20, 8_388_608, 2).unwrap();
    let previous = rollout.steps()[0].output_state();
    assert!(previous.epsilon_ticks() > 0);
    assert_eq!(rollout.steps()[1].input_state(), previous);

    let sources = rollout.lean_sources().unwrap();
    assert_eq!(sources.len(), 2);
    let expected = format!(
        "VeriNeural.ApproxContract {} {} ideal actual",
        previous.ideal_bound_ticks(),
        previous.epsilon_ticks()
    );
    assert!(
        sources[1].contains(&expected),
        "step two must assume the carried state contract rather than epsilon=0"
    );
}

#[test]
fn recurrent_feedback_rejects_a_transition_that_changes_q_scale() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let state = graph.input(shape.clone(), Domain::Finite);
    let gain = graph.input(shape, Domain::Finite);
    let next = graph.mul(state, gain).unwrap();

    assert!(matches!(
        RneStateRolloutPlan::compile_fixed(&graph, state, next, 20, 16, 2),
        Err(NumericError::InvalidInput { reason, .. })
            if reason == "RNE recurrent feedback changes the state q scale"
    ));
}

/// Explicit P1 acceptance. Run with: task verify-numeric-state
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_kernel_checks_each_composed_recurrent_step() {
    let (graph, state, next) = additive_transition();
    let rollout =
        RneStateRolloutPlan::compile_fixed(&graph, state, next, 20, 8_388_608, 4).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "verineural-numeric-state-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let engine = LeanProofEngine::default();

    for (step, source) in rollout.lean_sources().unwrap().iter().enumerate() {
        let path = dir.join(format!("step_{step}.lean"));
        std::fs::write(&path, source).unwrap();
        engine.check_file(&path).unwrap();
    }

    std::fs::remove_dir_all(dir).unwrap();
}
