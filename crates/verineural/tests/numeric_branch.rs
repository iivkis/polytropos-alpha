//! P1 numeric branch composition for PositiveMask -> SelectPositive.
//! Fast tests exercise interval classification and fail-closed behavior.
//! The ignored test asks the real Lean kernel + axiom audit to check the
//! generated stable-branch theorem.

use std::path::Path;
use verineural::{
    Domain, LeanProofEngine, NeuralGraph, NumericError, RneSelectPlan, RneThresholdSide, Shape,
};

fn graph() -> (NeuralGraph, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape.clone(), Domain::Finite);
    let yes_input = graph.input(shape.clone(), Domain::Finite);
    let no_left = graph.input(shape.clone(), Domain::Finite);
    let no_right = graph.input(shape, Domain::Finite);

    // q=20 and source bound 2^23 ticks gives the condition Add a two-tick
    // worst-case epsilon at its supported magnitude. The two branches have
    // deliberately different contracts so inheritance is observable.
    let condition = graph.add(x, y).unwrap();
    let mask = graph.positive_mask(condition).unwrap();
    let yes = graph.relu(yes_input).unwrap();
    let no = graph.add(no_left, no_right).unwrap();
    let selected = graph.select_positive(mask, yes, no).unwrap();
    (graph, selected)
}

#[test]
fn stable_select_inherits_only_the_chosen_branch_contract() {
    let (graph, selected) = graph();
    let plan = RneSelectPlan::compile(&graph, selected, 20, 8_388_608).unwrap();

    let positive = plan.classify(4.0).unwrap();
    assert_eq!(positive.side(), RneThresholdSide::Positive);
    assert_eq!(positive.selected(), Some(plan.yes()));
    assert_eq!(positive.contract().unwrap(), plan.yes_plan().output_contract());
    assert_eq!(positive.contract().unwrap().epsilon_ticks(), 0);

    let nonpositive = plan.classify(-4.0).unwrap();
    assert_eq!(nonpositive.side(), RneThresholdSide::NonPositive);
    assert_eq!(nonpositive.selected(), Some(plan.no()));
    assert_eq!(
        nonpositive.contract().unwrap(),
        plan.no_plan().output_contract()
    );
    assert!(nonpositive.contract().unwrap().epsilon_ticks() > 0);
}

#[test]
fn zero_crossing_condition_has_no_numeric_select_contract() {
    let (graph, selected) = graph();
    let plan = RneSelectPlan::compile(&graph, selected, 20, 8_388_608).unwrap();

    let uncertain = plan.classify(0.0).unwrap();
    assert_eq!(uncertain.side(), RneThresholdSide::Indeterminate);
    assert_eq!(uncertain.selected(), None);
    assert_eq!(uncertain.contract(), None);
    let (lower, upper) = uncertain.condition_evidence().ideal_interval_ticks();
    assert!(lower < 0 && upper > 0);
    assert!(matches!(
        plan.stable_contract(0.0),
        Err(NumericError::UnstableBranch { node, .. }) if node == selected.index()
    ));
    assert!(matches!(
        plan.lean_program(0.0),
        Err(NumericError::UnstableBranch { node, .. }) if node == selected.index()
    ));
}

#[test]
fn select_numeric_rule_requires_the_canonical_positive_mask_boundary() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let condition = graph.input(shape.clone(), Domain::Finite);
    let yes = graph.input(shape.clone(), Domain::Finite);
    let no = graph.input(shape, Domain::Finite);
    let selected = graph.select_positive(condition, yes, no).unwrap();
    assert!(matches!(
        RneSelectPlan::compile(&graph, selected, 20, 16),
        Err(NumericError::InvalidInput { .. })
    ));
}

#[test]
fn generated_rule_records_margin_and_selected_contract() {
    let (graph, selected) = graph();
    let plan = RneSelectPlan::compile(&graph, selected, 20, 8_388_608).unwrap();

    let positive = plan.lean_program(4.0).unwrap();
    assert!(positive.source().contains(
        "VeriNeural.approx_contract_positive_mask_select_positive"
    ));
    assert!(positive.source().contains("have hmargin"));
    assert_eq!(positive.theorem_names().len(), 1);

    let nonpositive = plan.lean_program(-4.0).unwrap();
    assert!(nonpositive.source().contains(
        "VeriNeural.approx_contract_positive_mask_select_nonpositive"
    ));
    assert_eq!(nonpositive.theorem_names().len(), 1);
}

/// Explicit P1 acceptance. Run with: task verify-numeric-branch
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_checks_both_stable_select_rules() {
    let (graph, selected) = graph();
    let plan = RneSelectPlan::compile(&graph, selected, 20, 8_388_608).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "verineural-numeric-branch-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let engine = LeanProofEngine::default();

    for (name, observed) in [("positive", 4.0), ("nonpositive", -4.0)] {
        let program = plan.lean_program(observed).unwrap();
        engine
            .verify_certified(&program, &Path::new(&dir).join(format!("{name}.lean")))
            .unwrap();
    }

    std::fs::remove_dir_all(dir).unwrap();
}
