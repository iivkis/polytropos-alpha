//! P1 arbitrary-horizon ideal-Int state invariant.
//!
//! One consumer-defined transition is instantiated once. Lean then reuses its
//! checked next_state claim in an induction over Nat rather than unrolling a
//! fixed list of states.

use verineural::{
    Domain, LeanProofEngine, MultiOutputContract, MultiOutputModule, NamedPortContract,
    NeuralError, NeuralGraph, NodeId, PortContract, ProofError, Shape,
};

struct Accumulate {
    contract: MultiOutputContract,
}

impl MultiOutputModule for Accumulate {
    fn name(&self) -> &'static str {
        "consumer_inductive_accumulate"
    }

    fn contract(&self) -> &MultiOutputContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        let sum = graph.add(inputs[0], inputs[1])?;
        let next = graph.relu(sum)?;
        Ok(vec![next, next])
    }
}

fn transition(with_claim: bool) -> Accumulate {
    let shape = Shape::new(vec![3]).unwrap();
    let any = PortContract::new(shape, Domain::Any);
    let next = if with_claim {
        NamedPortContract::new("next_state", any.clone()).nonnegative()
    } else {
        NamedPortContract::new("next_state", any.clone())
    };
    Accumulate {
        contract: MultiOutputContract {
            inputs: vec![
                NamedPortContract::new("state", any.clone()),
                NamedPortContract::new("input", any.clone()),
            ],
            outputs: vec![next, NamedPortContract::new("output", any)],
        },
    }
}

fn graph(with_claim: bool) -> (NeuralGraph, usize) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![3]).unwrap();
    let state = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let input = graph.input(shape, Domain::NonNegativeFinite);
    graph
        .apply_multi_module(&transition(with_claim), &[state, input])
        .unwrap();
    let index = graph.multi_modules().len() - 1;
    (graph, index)
}

#[test]
fn induction_reuses_one_transition_claim_without_fixed_unroll() {
    let (graph, module_index) = graph(true);
    let program = LeanProofEngine::default()
        .generate_inductive_state_nonnegative(&graph, module_index, "accumulator")
        .unwrap();

    assert!(program.source().contains("def scan_accumulator_state"));
    assert!(
        program
            .source()
            .contains("theorem scan_accumulator_inductive_nonnegative")
    );
    assert!(program.source().contains("induction t with"));
    assert!(
        program
            .source()
            .contains(&format!("multi_{module_index}_port_0_claim_0"))
    );
    assert!(program.source().contains("h_input : ∀ t"));
    assert!(!program.source().contains("no arbitrary-horizon induction"));
    assert!(
        program
            .theorem_names()
            .iter()
            .any(|name| name == "scan_accumulator_inductive_nonnegative")
    );
}

#[test]
fn induction_requires_a_consumer_owned_next_state_claim() {
    let (graph, module_index) = graph(false);
    assert!(matches!(
        LeanProofEngine::default().generate_inductive_state_nonnegative(
            &graph,
            module_index,
            "missing_claim"
        ),
        Err(ProofError::InvalidScan)
    ));
}

#[test]
fn induction_rejects_non_nonnegative_state_hypothesis() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![3]).unwrap();
    let state = graph.input(shape.clone(), Domain::Any);
    let input = graph.input(shape, Domain::NonNegativeFinite);
    graph
        .apply_multi_module(&transition(true), &[state, input])
        .unwrap();
    let module_index = graph.multi_modules().len() - 1;
    assert!(matches!(
        LeanProofEngine::default().generate_inductive_state_nonnegative(
            &graph,
            module_index,
            "bad_state"
        ),
        Err(ProofError::InvalidScan)
    ));
}

/// Explicit P1 acceptance. Run with: task verify-inductive-invariant
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_kernel_checks_arbitrary_horizon_invariant() {
    let (graph, module_index) = graph(true);
    let program = LeanProofEngine::default()
        .generate_inductive_state_nonnegative(&graph, module_index, "accumulator")
        .unwrap();
    let dir = std::env::temp_dir().join(format!(
        "verineural-inductive-invariant-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inductive_invariant.lean");
    let (_checked, audit) = LeanProofEngine::default()
        .verify_certified(&program, &path)
        .unwrap();
    assert!(
        audit
            .theorem_names
            .iter()
            .any(|name| name == "scan_accumulator_inductive_nonnegative")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
