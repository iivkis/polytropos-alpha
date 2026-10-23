//! External user-defined *atomic* Semantic IR operation.
//! Nothing in verineural must be edited to register this particular formula.
//! Supported ideal-Int Lean model, explicit unsupported GPU/FP32 certification.
use std::path::Path;
use verineural::{
    Domain, GpuEngine, GpuError, LeanProofEngine, MathContract, NeuralError,
    NeuralGraph, Operation, PointwiseExpr, PointwiseSpec, PortContract,
    ProofError, Shape, Tensor,
};

fn make_spec(version: &'static str, covered: bool) -> PointwiseSpec {
    let x = PointwiseExpr::input(0);
    let y = PointwiseExpr::input(1);
    let expr = x.clone().mul(x).add(y);
    PointwiseSpec::new(
        "consumer.square_plus_relu",
        version,
        2,
        if covered { expr.relu() } else { expr },
    )
    .unwrap()
}

fn architecture(spec: &PointwiseSpec) -> (NeuralGraph, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let bias = graph.input(shape.clone(), Domain::Finite);
    let contract = MathContract::new(
        vec![PortContract::finite(shape.clone()); 2],
        PortContract::any(shape),
    )
    .nonnegative();
    let output = graph.compose(
        "external_pointwise",
        "1",
        &contract,
        &[x, bias],
        |graph, args| spec.apply(graph, args),
    ).unwrap();
    (graph, output)
}

#[test]
fn consumer_defines_atomic_semantics_and_reference_without_library_edits() {
    let spec = make_spec("1.0.0", true);
    let (graph, output) = architecture(&spec);
    assert_eq!(graph.nodes().len(), 3); // two inputs and exactly one custom op
    assert!(matches!(&graph.node(output).unwrap().operation, Operation::Pointwise(_)));
    let canonical = graph.canonical_ir();
    assert!(canonical.contains("consumer.square_plus_relu@1.0.0"));
    assert!(canonical.contains("relu(add(mul(arg:0,arg:0),arg:1))"));
    let (other, _) = architecture(&make_spec("1.0.1", true));
    assert_ne!(canonical, other.canonical_ir());
    let (redefined, _) = architecture(&make_spec("1.0.0", false));
    assert_ne!(canonical, redefined.canonical_ir()); // same name/version, changed semantics

    let shape = Shape::new(vec![2]).unwrap();
    let x = Tensor::new(shape.clone(), vec![-2.0f32, 3.0]).unwrap();
    let bias = Tensor::new(shape.clone(), vec![-4.0f32, 1.0]).unwrap();
    let result = spec.evaluate_host(&[x, bias]).unwrap();
    assert_eq!(result.values(), &[0.0, 10.0]);
    // Unknown CUDA lowering fails before generating PTX or opening a CUDA driver.
    assert!(matches!(
        GpuEngine::default().compile(&graph, output),
        Err(GpuError::UnsupportedOperation { .. })
    ));
    let lean = LeanProofEngine::default().generate(&graph, output).unwrap();
    assert!(lean.source().contains("VeriNeural.mul"));
    assert!(lean.source().contains("VeriNeural.add"));
    assert!(lean.source().contains("VeriNeural.relu"));
    assert!(lean.source().contains("module_0_claim_0"));
}

#[test]
fn unsupported_or_bad_semantics_fail_closed() {
    assert!(PointwiseSpec::new(
        "bad", "1", 1, PointwiseExpr::input(1)
    ).is_err());
    assert!(PointwiseSpec::new(
        "bad", "1", 0, PointwiseExpr::input(0)
    ).is_err());

    let mut graph = NeuralGraph::default();
    let a = graph.input(Shape::new(vec![2]).unwrap(), Domain::Any);
    let b = graph.input(Shape::new(vec![3]).unwrap(), Domain::Any);
    let spec = make_spec("1", true);
    assert!(matches!(spec.apply(&mut graph, &[a, b]), Err(NeuralError::ShapeMismatch { .. })));
    assert_eq!(graph.nodes().len(), 2);

    // Same well-typed structure, but no ReLU proof rule for a*a+b:
    // certification must fail even if sample inputs happen to be nonnegative.
    let (unsupported, output) = architecture(&make_spec("1", false));
    assert!(matches!(
        LeanProofEngine::default().generate(&unsupported, output),
        Err(ProofError::UnsupportedGuarantee { .. })
    ));
}

fn certify(graph: &NeuralGraph, output: verineural::NodeId, path: &Path)
    -> Result<(), ProofError>
{
    let engine = LeanProofEngine::default();
    let proof = engine.generate(graph, output)?;
    engine.verify_certified(&proof, path)?;
    Ok(())
}

/// Run explicitly with: task verify-pointwise-primitive
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_kernel_accepts_external_semantics_and_nonnegative_claim() {
    let dir = std::env::temp_dir().join(format!(
        "verineural-pointwise-consumer-{}", std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (graph, output) = architecture(&make_spec("1.0.0", true));
    certify(&graph, output, &dir.join("pointwise.lean")).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
