//! Independent consumer of the typed relational predicate API.
//! Fast assertions validate the generated obligations. The ignored Lean test
//! checks the positive/negative pair with the real external kernel.
use std::path::Path;
use verineural::{
    Domain, LeanProofEngine, MathContract, NeuralGraph, PortContract, Predicate, ProofError,
    ProofGoal, ProofGoalKind, Shape, ValueRef,
};

fn case(reverse: bool) -> (NeuralGraph, ProofGoal) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let predicate = if reverse {
        Predicate::pointwise_le(ValueRef::Output, ValueRef::Input(0))
    } else {
        Predicate::pointwise_le(ValueRef::Input(0), ValueRef::Output)
    };
    let contract = MathContract::new(
        vec![PortContract::finite(shape.clone())],
        PortContract::nonnegative(shape),
    )
    .predicate(predicate);
    let y = graph
        .compose("consumer_relu", "1", &contract, &[x], |g, inputs| {
            g.relu(inputs[0])
        })
        .unwrap();
    let predicate = if reverse {
        Predicate::pointwise_le(ValueRef::Node(y), ValueRef::Node(x))
    } else {
        Predicate::pointwise_le(ValueRef::Node(x), ValueRef::Node(y))
    };
    (
        graph,
        ProofGoal {
            name: if reverse {
                "relu_may_decrease"
            } else {
                "relu_preserves_order"
            }
            .into(),
            output: y,
            kind: ProofGoalKind::Predicate(predicate),
        },
    )
}

#[test]
fn consumer_claims_are_in_canonical_ir_and_generate_nontrivial_goals() {
    let (graph, goal) = case(false);
    let ir = graph.canonical_ir();
    assert!(ir.contains("pointwise_le(input:0,output)"));
    let proof = LeanProofEngine::default()
        .generate_goal(&graph, &goal)
        .unwrap();
    assert!(proof.source().contains("VeriNeural.PointwiseLe"));
    assert!(proof.source().contains("VeriNeural.pointwise_le_relu"));
    assert!(
        proof
            .theorem_names()
            .iter()
            .any(|n| n == "module_0_claim_0")
    );
    assert!(
        proof
            .theorem_names()
            .iter()
            .any(|n| n == "architecture_property_relu_preserves_order")
    );
    assert!(!proof.source().contains("sorry"));

    let (bad, bad_goal) = case(true);
    assert_ne!(ir, bad.canonical_ir());
    let bad_proof = LeanProofEngine::default()
        .generate_goal(&bad, &bad_goal)
        .unwrap();
    // Shape correctness does not validate the user's assertion.
    // The kernel must reject the generated proof with the reversed relation.
    assert!(bad_proof.source().contains("VeriNeural.PointwiseLe"));
}

#[test]
fn rejects_foreign_or_unscoped_predicate_references() {
    let (graph, goal) = case(false);
    let mut foreign = NeuralGraph::default();
    let alien = foreign.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let invalid = ProofGoal {
        name: "foreign".into(),
        output: goal.output,
        kind: ProofGoalKind::Predicate(Predicate::pointwise_le(
            ValueRef::Node(alien),
            ValueRef::Output,
        )),
    };
    assert!(matches!(
        LeanProofEngine::default().generate_goal(&graph, &invalid),
        Err(ProofError::InvalidGraph(_))
    ));
}

fn certify(graph: &NeuralGraph, goal: &ProofGoal, path: &Path) -> Result<(), ProofError> {
    let lean = LeanProofEngine::default();
    let generated = lean.generate_goal(graph, goal)?;
    lean.verify_certified(&generated, path)?;
    Ok(())
}

/// Explicit, slower acceptance: requires pinned Lean 4 from the Nix devShell.
/// Run with: task verify-relational-predicates
#[test]
#[ignore = "requires Lean 4 from Nix devShell"]
fn lean_accepts_true_relation_and_rejects_false_one() {
    let dir = std::env::temp_dir().join(format!(
        "verineural-relational-predicates-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (good_graph, good_goal) = case(false);
    certify(&good_graph, &good_goal, &dir.join("true.lean")).unwrap();
    let (bad_graph, bad_goal) = case(true);
    assert!(matches!(
        certify(&bad_graph, &bad_goal, &dir.join("false.lean")),
        Err(ProofError::LeanRejected { .. })
    ));
    std::fs::remove_dir_all(dir).unwrap();
}
