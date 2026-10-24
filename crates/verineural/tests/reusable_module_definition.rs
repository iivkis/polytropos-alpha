//! Reusable module-definition proof slice.
//!
//! One consumer module definition gets one generic Lean theorem. Concrete
//! instances get separate application theorems that must discharge their own
//! input hypotheses. Same identity with different normalized semantics fails.

use verineural::{
    Domain, LeanProofEngine, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId,
    PortContract, ProofError, Shape,
};

fn shape() -> Shape {
    Shape::new(vec![4]).unwrap()
}

#[derive(Clone, Copy)]
enum Body {
    Add,
    Mul,
}

struct PositiveJoin {
    contract: MathContract,
    body: Body,
}

impl PositiveJoin {
    fn new(body: Body) -> Self {
        let required = PortContract::nonnegative(shape());
        Self {
            contract: MathContract::new(
                vec![required.clone(), required],
                PortContract::any(shape()),
            )
            .nonnegative(),
            body,
        }
    }
}

impl NeuralModule for PositiveJoin {
    fn name(&self) -> &'static str {
        "consumer_positive_join"
    }

    fn semantic_version(&self) -> &'static str {
        "1"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let mixed = match self.body {
            Body::Add => graph.add(inputs[0], inputs[1])?,
            Body::Mul => graph.mul(inputs[0], inputs[1])?,
        };
        graph.relu(mixed)
    }
}

// The final consumer accepts domain-unknown outputs of the earlier
// modules. Its global ideal-Int theorem still needs both nonnegative
// premises, which must come from the producer applications.
struct GlobalJoin {
    contract: MathContract,
}

impl GlobalJoin {
    fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![PortContract::any(shape()), PortContract::any(shape())],
                PortContract::any(shape()),
            ),
        }
    }
}

impl NeuralModule for GlobalJoin {
    fn name(&self) -> &'static str {
        "consumer_global_join"
    }

    fn semantic_version(&self) -> &'static str {
        "1"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

fn repeated_graph() -> NeuralGraph {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(), Domain::NonNegativeFinite);
    let y = graph.input(shape(), Domain::NonNegativeFinite);
    let z = graph.input(shape(), Domain::NonNegativeFinite);
    let module = PositiveJoin::new(Body::Add);
    graph.apply_module(&module, &[x, y]).unwrap();
    graph.apply_module(&module, &[x, z]).unwrap();
    graph
}

#[test]
fn one_definition_theorem_is_reused_by_two_concrete_instances() {
    let graph = repeated_graph();
    let program = LeanProofEngine::default()
        .generate_reusable_module_nonnegative(&graph, &[0, 1], "positive_join")
        .unwrap();

    assert_eq!(
        program
            .source()
            .matches("def module_definition_positive_join ")
            .count(),
        1
    );
    assert_eq!(
        program
            .source()
            .matches("theorem module_definition_positive_join_nonnegative ")
            .count(),
        1
    );
    assert!(
        program
            .source()
            .contains("theorem module_definition_positive_join_instance_0 ")
    );
    assert!(
        program
            .source()
            .contains("theorem module_definition_positive_join_instance_1 ")
    );

    let start = program
        .source()
        .find("def module_definition_positive_join ")
        .unwrap();
    let end = program.source()[start..]
        .find("theorem module_definition_positive_join_instance_0 ")
        .map(|offset| start + offset)
        .unwrap();
    let shared = &program.source()[start..end];
    assert!(shared.contains("p_0"));
    assert!(shared.contains("p_1"));
    assert!(shared.contains("VeriNeural.relu (VeriNeural.add (p_0) (p_1))"));
    assert!(shared.contains("VeriNeural.relu_nonnegative (VeriNeural.add (p_0) (p_1))"));
    assert!(!shared.contains("node_"));
    // The output ReLU is nonnegative unconditionally, but the reusable
    // interface still retains both declared NonNegativeFinite requires.
    assert!(shared.contains("(_h_port_0 : VeriNeural.Nonnegative p_0)"));
    assert!(shared.contains("(_h_port_1 : VeriNeural.Nonnegative p_1)"));

    let instance0 = program
        .source()
        .split("theorem module_definition_positive_join_instance_0 ")
        .nth(1)
        .unwrap()
        .split("theorem module_definition_positive_join_instance_1 ")
        .next()
        .unwrap();
    assert!(instance0.contains("(h_0 : VeriNeural.Nonnegative x_0)"));
    assert!(instance0.contains("(h_1 : VeriNeural.Nonnegative x_1)"));
    assert!(instance0.contains("module_definition_positive_join_nonnegative"));
    assert!(instance0.contains("h_0 h_1"));

    assert!(
        program
            .theorem_names()
            .contains(&"module_definition_positive_join_nonnegative".to_owned())
    );
    assert!(
        program
            .theorem_names()
            .contains(&"module_definition_positive_join_instance_0".to_owned())
    );
    assert!(
        program
            .theorem_names()
            .contains(&"module_definition_positive_join_instance_1".to_owned())
    );
}

#[test]
fn global_composition_reuses_definition_theorem_and_checks_each_invocation() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(), Domain::NonNegativeFinite);
    let y = graph.input(shape(), Domain::NonNegativeFinite);
    let z = graph.input(shape(), Domain::NonNegativeFinite);
    let module = PositiveJoin::new(Body::Add);
    let left = graph.apply_module(&module, &[x, y]).unwrap();
    let right = graph.apply_module(&module, &[x, z]).unwrap();
    let output = graph
        .apply_module(&GlobalJoin::new(), &[left, right])
        .unwrap();

    let program = LeanProofEngine::default()
        .generate_global_nonnegative(&graph, output, &[x, y, z])
        .unwrap();
    let source = program.source();

    assert_eq!(
        source
            .matches("theorem architecture_module_definition_0_nonnegative ")
            .count(),
        1
    );
    assert!(!source.contains("theorem architecture_module_definition_1_nonnegative "));
    assert!(source.contains("theorem architecture_module_definition_2_nonnegative "));
    assert!(source.contains("theorem architecture_module_instance_0 "));
    assert!(source.contains("theorem architecture_module_instance_1 "));
    assert!(source.contains("theorem architecture_module_instance_2 "));
    assert!(source.contains("have arch_module_0 := architecture_module_instance_0"));
    assert!(source.contains("have arch_module_1 := architecture_module_instance_1"));
    assert!(source.contains("have arch_module_2 := architecture_module_instance_2"));
}

#[test]
fn global_composition_rejects_same_identity_with_different_body() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(), Domain::NonNegativeFinite);
    let y = graph.input(shape(), Domain::NonNegativeFinite);
    let z = graph.input(shape(), Domain::NonNegativeFinite);
    let left = graph
        .apply_module(&PositiveJoin::new(Body::Add), &[x, y])
        .unwrap();
    let right = graph
        .apply_module(&PositiveJoin::new(Body::Mul), &[x, z])
        .unwrap();
    let output = graph
        .apply_module(&GlobalJoin::new(), &[left, right])
        .unwrap();

    assert!(matches!(
        LeanProofEngine::default().generate_global_nonnegative(&graph, output, &[x, y, z]),
        Err(ProofError::ModuleDefinitionMismatch {
            representative: 0,
            other: 1
        })
    ));
}

#[test]
fn same_identity_with_different_body_cannot_reuse_definition_proof() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(), Domain::NonNegativeFinite);
    let y = graph.input(shape(), Domain::NonNegativeFinite);
    let z = graph.input(shape(), Domain::NonNegativeFinite);
    graph
        .apply_module(&PositiveJoin::new(Body::Add), &[x, y])
        .unwrap();
    graph
        .apply_module(&PositiveJoin::new(Body::Mul), &[x, z])
        .unwrap();

    assert!(matches!(
        LeanProofEngine::default().generate_reusable_module_nonnegative(
            &graph,
            &[0, 1],
            "lying_identity"
        ),
        Err(ProofError::ModuleDefinitionMismatch {
            representative: 0,
            other: 1
        })
    ));
}

#[test]
fn duplicate_instance_request_is_rejected_instead_of_counted_as_reuse() {
    let graph = repeated_graph();
    assert!(matches!(
        LeanProofEngine::default().generate_reusable_module_nonnegative(
            &graph,
            &[0, 0],
            "duplicate"
        ),
        Err(ProofError::InvalidOutputs)
    ));
}

/// Explicit acceptance. Run with: task verify-reusable-module-proof
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_checks_shared_definition_and_each_instance_application() {
    let graph = repeated_graph();
    let program = LeanProofEngine::default()
        .generate_reusable_module_nonnegative(&graph, &[0, 1], "positive_join")
        .unwrap();
    let dir =
        std::env::temp_dir().join(format!("verineural-reusable-module-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    LeanProofEngine::default()
        .verify_certified(&program, &dir.join("reusable_module.lean"))
        .unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
