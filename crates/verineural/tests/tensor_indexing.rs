//! P0 structural tensor/indexing slice used by routing-style consumers.
//! Host references are concrete f32/index calculations; Lean models the ideal
//! integer structure, while GPU lowering intentionally fails closed for now.
use verineural::{
    ArgMaxTieBreak, DType, Domain, GpuEngine, GpuError, LeanProofEngine, NeuralError,
    NeuralGraph, Operation, ProofError, Shape, Tensor,
};

#[test]
fn host_permute_reverse_concat_argmax_and_take_are_row_major() {
    let matrix = Tensor::new(
        Shape::new(vec![2, 3]).unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    )
    .unwrap();

    let transposed = matrix.permute(&[1, 0]).unwrap();
    assert_eq!(transposed.shape().dims(), &[3, 2]);
    assert_eq!(transposed.values(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);

    let reversed = matrix.reverse_axis(1).unwrap();
    assert_eq!(reversed.shape().dims(), &[2, 3]);
    assert_eq!(reversed.values(), &[2.0, 1.0, 0.0, 5.0, 4.0, 3.0]);

    let top = Tensor::new(Shape::new(vec![1, 3]).unwrap(), vec![10.0, 11.0, 12.0]).unwrap();
    let joined = top.concat(&matrix, 0).unwrap();
    assert_eq!(joined.shape().dims(), &[3, 3]);
    assert_eq!(
        joined.values(),
        &[10.0, 11.0, 12.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );

    let logits = Tensor::new(
        Shape::new(vec![2, 4]).unwrap(),
        vec![1.0, 5.0, 5.0, 2.0, 9.0, 8.0, 7.0, 9.0],
    )
    .unwrap();
    let action = logits.argmax(1, ArgMaxTieBreak::First).unwrap();
    assert_eq!(action.shape().dims(), &[2]);
    assert_eq!(action.values(), &[1, 0]);

    let nonfinite = Tensor::new(
        Shape::new(vec![2, 3]).unwrap(),
        vec![f32::NAN, 4.0, 3.0, f32::NAN, f32::NAN, f32::NAN],
    )
    .unwrap();
    assert_eq!(
        nonfinite
            .argmax(1, ArgMaxTieBreak::First)
            .unwrap()
            .values(),
        &[1, 0]
    );

    let bank = Tensor::new(
        Shape::new(vec![3, 2]).unwrap(),
        vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
    )
    .unwrap();
    let selected = Tensor::new(Shape::new(vec![1]).unwrap(), vec![1usize]).unwrap();
    let lane = bank.take(&selected, 0).unwrap();
    assert_eq!(lane.shape().dims(), &[2]);
    assert_eq!(lane.values(), &[20.0, 21.0]);
}

#[test]
fn graph_uses_a_real_index_dtype_and_rejects_float_indices() {
    let mut graph = NeuralGraph::default();
    let logits = graph.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let action = graph
        .argmax(logits, 1, ArgMaxTieBreak::First)
        .unwrap();
    assert_eq!(graph.node(action).unwrap().dtype, DType::Index);
    assert_eq!(graph.node(action).unwrap().shape.dims(), &[1]);

    let bank = graph.input(Shape::new(vec![4, 2, 2]).unwrap(), Domain::Finite);
    let selected = graph.take(bank, action, 0).unwrap();
    assert_eq!(graph.node(selected).unwrap().dtype, DType::F32);
    assert_eq!(graph.node(selected).unwrap().shape.dims(), &[2, 2]);

    let fake_index = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
    assert!(matches!(
        graph.take(bank, fake_index, 0),
        Err(NeuralError::DTypeMismatch {
            operation: "take",
            expected: DType::Index,
            actual: DType::F32,
        })
    ));

    let small_bank = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    assert!(matches!(
        graph.take(small_bank, action, 0),
        Err(NeuralError::IndexRangeExceedsAxis {
            upper_bound: 4,
            axis: 0,
            extent: 2,
        })
    ));
}

#[test]
fn rotating_bank_slice_is_canonical_and_has_lean_structure_but_no_fake_gpu_support() {
    let mut graph = NeuralGraph::default();
    let core0 = graph.input(
        Shape::new(vec![2, 2]).unwrap(),
        Domain::NonNegativeFinite,
    );
    let core1 = graph.input(
        Shape::new(vec![2, 2]).unwrap(),
        Domain::NonNegativeFinite,
    );
    let logits = graph.input(Shape::new(vec![1, 2]).unwrap(), Domain::Finite);

    let r0 = graph.permute(core0, &[1, 0]).unwrap();
    let r1 = graph.reverse_axis(core1, 1).unwrap();
    let r0 = graph.reshape(r0, Shape::new(vec![1, 2, 2]).unwrap()).unwrap();
    let r1 = graph.reshape(r1, Shape::new(vec![1, 2, 2]).unwrap()).unwrap();
    let bank = graph.concat(0, &[r0, r1]).unwrap();
    let action = graph
        .argmax(logits, 1, ArgMaxTieBreak::First)
        .unwrap();
    let selected = graph.take(bank, action, 0).unwrap();

    assert!(matches!(
        graph.node(selected).unwrap().operation,
        Operation::Take { axis: 0 }
    ));
    assert_eq!(graph.node(selected).unwrap().shape.dims(), &[2, 2]);
    assert_eq!(graph.node(selected).unwrap().domain, Domain::NonNegativeFinite);

    let canonical = graph.canonical_ir();
    assert!(canonical.starts_with("verineural-semantic-ir-v3\n"));
    assert!(canonical.contains("permute_axes:1,0"));
    assert!(canonical.contains("reverse_axis:1"));
    assert!(canonical.contains("concat_axis:0"));
    assert!(canonical.contains("argmax_axis:1:tie=first"));
    assert!(canonical.contains("dtype=index"));
    assert!(canonical.contains("take_axis:0"));

    let proof = LeanProofEngine::default().generate(&graph, selected).unwrap();
    let source = proof.source();
    assert!(source.contains("VeriNeural.IndexTensor"));
    assert!(source.contains("VeriNeural.transpose2D"));
    assert!(source.contains("VeriNeural.reverseAxis"));
    assert!(source.contains("VeriNeural.concatAxis"));
    assert!(source.contains("VeriNeural.argMaxFirstAxis"));
    assert!(source.contains("VeriNeural.takeAxis"));
    assert!(source.contains("VeriNeural.takeAxis_nonnegative"));
    assert!(!source.contains("sorry"));

    assert!(matches!(
        GpuEngine::default().compile(&graph, selected),
        Err(GpuError::UnsupportedOperation {
            operation: "take",
            ..
        })
    ));
}

#[test]
fn lean_backend_rejects_permutations_it_does_not_model() {
    let mut graph = NeuralGraph::default();
    let source = graph.input(Shape::new(vec![2, 3, 4]).unwrap(), Domain::Finite);
    let permuted = graph.permute(source, &[2, 0, 1]).unwrap();

    assert!(matches!(
        LeanProofEngine::default().generate(&graph, permuted),
        Err(ProofError::UnsupportedOperation {
            operation: "permute",
            ..
        })
    ));
}
