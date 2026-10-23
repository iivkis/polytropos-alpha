//! Standalone consumer for arbitrary-axis row-major reduction.
//! Proofs concern ideal Int sums; host f32 left-fold is not an FP32 guarantee.
use std::path::Path;
use verineural::{
    Domain, GpuEngine, GpuError, LeanProofEngine, MathContract, NeuralError, NeuralGraph, NodeId,
    Operation, PointwiseExpr, PointwiseSpec, PortContract, PrimitiveExpr, PrimitiveSpec,
    ProofError, Shape, Tensor,
};

fn consumer(positive: bool) -> (NeuralGraph, NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2, 3, 2]).unwrap();
    let output_shape = Shape::new(vec![2, 2]).unwrap();
    let source = graph.input(shape.clone(), Domain::Finite);
    let expression = if positive {
        PointwiseExpr::input(0).relu()
    } else {
        PointwiseExpr::input(0)
    };
    let spec = PointwiseSpec::new("consumer.preprocess", "1", 1, expression).unwrap();
    let contract = MathContract::new(
        vec![PortContract::finite(shape)],
        PortContract::any(output_shape),
    )
    .nonnegative();

    let result = graph
        .compose(
            "consumer_middle_axis_sum",
            "1",
            &contract,
            &[source],
            |g, ports| {
                let transformed = spec.apply(g, ports)?;
                g.reduce_sum(transformed, 1)
            },
        )
        .unwrap();
    (graph, result)
}

#[test]
fn every_axis_and_scalar_output_have_explicit_row_major_reference() {
    let shape = Shape::new(vec![2, 3, 2]).unwrap();
    let tensor = Tensor::new(shape, (1..=12).map(|i| i as f32).collect()).unwrap();

    let middle = tensor.reduce_sum(1).unwrap();
    assert_eq!(middle.shape().dims(), &[2, 2]);
    assert_eq!(middle.values(), &[9.0, 12.0, 27.0, 30.0]);

    let first = tensor.reduce_sum(0).unwrap();
    assert_eq!(first.shape().dims(), &[3, 2]);
    assert_eq!(first.values(), &[8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);

    let last = tensor.reduce_sum(2).unwrap();
    assert_eq!(last.shape().dims(), &[2, 3]);
    assert_eq!(last.values(), &[3.0, 7.0, 11.0, 15.0, 19.0, 23.0]);

    let vector = Tensor::new(Shape::new(vec![3]).unwrap(), vec![1.0, 2.0, 3.0]).unwrap();
    let scalar = vector.reduce_sum(0).unwrap();
    assert_eq!(scalar.shape().dims(), &[] as &[usize]);
    assert_eq!(scalar.values(), &[6.0]);

    // Observable IEEE-754 order: ((0 + 1e20) + (-1e20)) + 1 = 1,
    // whereas silently grouping the last two terms can produce 0.
    let sensitive =
        Tensor::new(Shape::new(vec![3]).unwrap(), vec![1e20f32, -1e20f32, 1.0]).unwrap();
    assert_eq!(sensitive.reduce_sum(0).unwrap().values(), &[1.0]);
}

#[test]
fn invalid_axis_is_rejected_before_mutating_ir() {
    let mut graph = NeuralGraph::default();
    let tensor = graph.input(Shape::new(vec![2, 3, 2]).unwrap(), Domain::Any);
    assert!(matches!(
        graph.reduce_sum(tensor, 3),
        Err(NeuralError::InvalidAxis { axis: 3, rank: 3 })
    ));
    assert_eq!(graph.nodes().len(), 1);

    let scalar = graph.input(Shape::new(Vec::<usize>::new()).unwrap(), Domain::Any);
    assert!(matches!(
        graph.reduce_sum(scalar, 0),
        Err(NeuralError::InvalidAxis { axis: 0, rank: 0 })
    ));
    assert_eq!(graph.nodes().len(), 2);
}

#[test]
fn external_pointwise_then_reduce_has_one_int_model_and_checked_claim() {
    let (graph, output) = consumer(true);
    assert_eq!(graph.nodes().len(), 3);
    assert!(matches!(
        graph.node(output).unwrap().operation,
        Operation::ReduceSum { axis: 1 }
    ));
    assert!(graph.canonical_ir().contains("reduce_sum_axis:1"));
    let (other, _) = {
        let mut g = NeuralGraph::default();
        let x = g.input(Shape::new(vec![2, 3, 2]).unwrap(), Domain::Finite);
        let y = g.reduce_sum(x, 0).unwrap();
        (g, y)
    };
    assert_ne!(graph.canonical_ir(), other.canonical_ir());

    let proof = LeanProofEngine::default().generate(&graph, output).unwrap();
    assert!(proof.source().contains("VeriNeural.reduceSumAxis"));
    assert!(
        proof
            .source()
            .contains("VeriNeural.reduceSumAxis_nonnegative")
    );
    assert!(proof.source().contains("module_0_claim_0"));
    assert!(!proof.source().contains("sorry"));
    assert!(matches!(
        GpuEngine::default().compile(&graph, output),
        Err(GpuError::UnsupportedOperation { .. })
    ));
}

#[test]
fn composition_dsl_reuses_shape_rules_and_proof_semantics() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2, 3, 2]).unwrap();
    let output_shape = Shape::new(vec![2, 2]).unwrap();
    let input = graph.input(shape.clone(), Domain::Finite);
    let block = PrimitiveSpec::new(
        "composed_relu_sum",
        "1",
        MathContract::new(
            vec![PortContract::finite(shape)],
            PortContract::any(output_shape),
        )
        .nonnegative(),
        PrimitiveExpr::port(0).relu().reduce_sum(1),
    )
    .unwrap();
    let output = block.apply(&mut graph, &[input]).unwrap();
    assert_eq!(graph.nodes().len(), 3);
    assert!(graph.canonical_ir().contains("reduce_sum_axis:1"));
    assert!(LeanProofEngine::default().generate(&graph, output).is_ok());
}

#[test]
fn unjustified_nonnegativity_is_not_a_certificate() {
    let (graph, output) = consumer(false);
    assert!(matches!(
        LeanProofEngine::default().generate(&graph, output),
        Err(ProofError::UnsupportedGuarantee { .. })
    ));
}

/// Explicit expensive acceptance, requiring pinned Lean in Nix devShell.
#[test]
#[ignore = "requires Lean 4"]
fn lean_checks_middle_axis_sum_and_nonnegative_claim() {
    let (graph, output) = consumer(true);
    let engine = LeanProofEngine::default();
    let program = engine.generate(&graph, output).unwrap();
    let path = std::env::temp_dir().join(format!(
        "verineural-tensor-reduction-{}.lean",
        std::process::id()
    ));
    engine.verify_certified(&program, Path::new(&path)).unwrap();
    std::fs::remove_file(path).unwrap();
}
