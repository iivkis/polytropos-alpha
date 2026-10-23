//! P2 arithmetic primitives and ergonomic tensor construction.
//! Numeric host references are not Lean proofs or CUDA certification.
use verineural::{
    Domain, GpuEngine, GpuError, LeanProofEngine, NeuralError, NeuralGraph, ProofError, Shape,
    Tensor,
};

#[test]
fn constructors_validate_shape_and_payload_without_boilerplate() {
    let v = Tensor::<f32>::from_vec([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let s = Tensor::<f32>::from_slice([2, 2], v.values()).unwrap();
    assert_eq!(v, s);
    let z = Tensor::<f32>::zeros([2, 2]).unwrap();
    assert_eq!(z.values(), &[0.0; 4]);
    assert!(matches!(
        Tensor::<f32>::from_slice([2, 2], &[1.0]),
        Err(NeuralError::DataLengthMismatch { .. })
    ));
    assert!(matches!(
        Tensor::<f32>::zeros([0, 2]),
        Err(NeuralError::ZeroDimension)
    ));
}

#[test]
fn host_broadcast_max_and_arithmetic_are_row_major() {
    let x = Tensor::<f32>::from_slice([2, 1], &[2.0, 3.0]).unwrap();
    let expanded = x.broadcast_to(Shape::new([2, 3]).unwrap()).unwrap();
    assert_eq!(expanded.values(), &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    assert_eq!(expanded.reduce_max(1).unwrap().values(), &[2.0, 3.0]);
    let scalar = Tensor::<f32>::from_vec([], vec![4.0]).unwrap();
    assert_eq!(
        scalar
            .broadcast_to(Shape::new([2, 2]).unwrap())
            .unwrap()
            .values(),
        &[4.0; 4]
    );
    let a = Tensor::<f32>::from_vec([2], vec![5.0, 8.0]).unwrap();
    let b = Tensor::<f32>::from_vec([2], vec![2.0, 4.0]).unwrap();
    assert_eq!(a.sub(&b).unwrap().values(), &[3.0, 4.0]);
    assert_eq!(a.div(&b).unwrap().values(), &[2.5, 2.0]);
    assert_eq!(
        Tensor::<f32>::from_vec([1], vec![0.0])
            .unwrap()
            .exp()
            .unwrap()
            .values(),
        &[1.0]
    );
    let nan = Tensor::<f32>::from_vec([2], vec![5.0, f32::NAN]).unwrap();
    assert!(nan.reduce_max(0).unwrap().values()[0].is_nan());
}

#[test]
fn softmax_composes_in_one_ir_and_unsupported_backends_fail_closed() {
    let mut g = NeuralGraph::default();
    let input = g.input(Shape::new([2, 3]).unwrap(), Domain::Finite);
    let maximum = g.reduce_max(input, 1).unwrap();
    // Reduction removes axis 1; restore its singleton dimension to broadcast
    // each row maximum across that row instead of right-aligning [2] to [2, 3].
    let maximum = g.reshape(maximum, Shape::new([2, 1]).unwrap()).unwrap();
    let maximum = g
        .broadcast_to(maximum, Shape::new([2, 3]).unwrap())
        .unwrap();
    let centered = g.sub(input, maximum).unwrap();
    let exponentials = g.exp(centered).unwrap();
    let sums = g.reduce_sum(exponentials, 1).unwrap();
    let sums = g.reshape(sums, Shape::new([2, 1]).unwrap()).unwrap();
    let sums = g.broadcast_to(sums, Shape::new([2, 3]).unwrap()).unwrap();
    let probabilities = g.div(exponentials, sums).unwrap();

    assert_eq!(g.node(probabilities).unwrap().shape.dims(), &[2, 3]);
    let canonical = g.canonical_ir();
    assert!(canonical.contains("reduce_max_axis:1"));
    assert!(canonical.contains("broadcast_to:2x3"));
    assert!(canonical.contains("sub(arg:0,arg:1)"));
    assert!(canonical.contains("div(arg:0,arg:1)"));
    assert!(canonical.contains("exp(arg:0)"));
    assert!(matches!(
        LeanProofEngine::default().generate(&g, probabilities),
        Err(ProofError::UnsupportedOperation { .. })
    ));
    assert!(matches!(
        GpuEngine::default().compile(&g, probabilities),
        Err(GpuError::UnsupportedOperation { .. })
    ));
}

#[test]
fn bad_axes_and_broadcast_leave_semantic_ir_unchanged() {
    let mut g = NeuralGraph::default();
    let id = g.input(Shape::new([2, 3]).unwrap(), Domain::Finite);
    let before = g.canonical_ir();
    assert!(matches!(
        g.reduce_max(id, 2),
        Err(NeuralError::InvalidAxis { .. })
    ));
    assert!(matches!(
        g.broadcast_to(id, Shape::new([3, 2]).unwrap()),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    assert_eq!(g.canonical_ir(), before);
}

#[test]
fn extended_pointwise_is_not_forged_as_ideal_int_proof() {
    let mut g = NeuralGraph::default();
    let x = g.input(Shape::new([2]).unwrap(), Domain::Finite);
    let e = g.exp(x).unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate(&g, e),
        Err(ProofError::UnsupportedOperation {
            operation: "pointwise",
            ..
        })
    ));
}
