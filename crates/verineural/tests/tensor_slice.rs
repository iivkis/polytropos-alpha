//! P1 static axis slicing: host semantics, Semantic IR, Lean ideal-Int
//! preservation and explicit GPU fail-closed behavior.

use std::path::Path;
use verineural::{
    Domain, GpuEngine, GpuError, LeanProofEngine, NeuralError, NeuralGraph, Shape, Tensor,
};

#[test]
fn host_slice_axis_preserves_row_major_coordinates() {
    let tensor = Tensor::from_vec([2, 4, 2], (0..16).collect::<Vec<i32>>()).unwrap();
    let sliced = tensor.slice_axis(1, 1, 2).unwrap();
    assert_eq!(sliced.shape().dims(), &[2, 2, 2]);
    assert_eq!(sliced.values(), &[2, 3, 4, 5, 10, 11, 12, 13]);
}

#[test]
fn graph_slice_records_bounds_and_rejects_invalid_intervals_atomically() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new(vec![2, 4, 2]).unwrap(), Domain::NonNegativeFinite);
    let sliced = graph.slice_axis(input, 1, 1, 2).unwrap();
    assert_eq!(graph.node(sliced).unwrap().shape.dims(), &[2, 2, 2]);
    assert!(
        graph
            .canonical_ir()
            .contains("slice_axis:1:start=1:len=2")
    );

    let before = graph.nodes().len();
    assert!(matches!(
        graph.slice_axis(input, 1, 4, 1),
        Err(NeuralError::InvalidSlice {
            axis: 1,
            start: 4,
            len: 1,
            extent: 4
        })
    ));
    assert!(matches!(
        graph.slice_axis(input, 1, 0, 0),
        Err(NeuralError::InvalidSlice { len: 0, .. })
    ));
    assert_eq!(graph.nodes().len(), before);
}

#[test]
fn lean_models_slice_but_gpu_remains_explicitly_unsupported() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new(vec![2, 4]).unwrap(), Domain::NonNegativeFinite);
    let sliced = graph.slice_axis(input, 1, 1, 2).unwrap();

    let lean = LeanProofEngine::default().generate(&graph, sliced).unwrap();
    assert!(lean.source().contains("VeriNeural.sliceAxis"));
    assert!(lean.source().contains("VeriNeural.sliceAxis_nonnegative"));
    assert!(matches!(
        GpuEngine::default().compile(&graph, sliced),
        Err(GpuError::UnsupportedOperation {
            operation: "slice_axis",
            ..
        })
    ));
}

/// Explicit P1 acceptance. Run with: task verify-tensor-slice
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_kernel_checks_static_slice_semantics() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(Shape::new(vec![2, 4]).unwrap(), Domain::NonNegativeFinite);
    let sliced = graph.slice_axis(input, 1, 1, 2).unwrap();
    let program = LeanProofEngine::default().generate(&graph, sliced).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "verineural-tensor-slice-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    LeanProofEngine::default()
        .verify_certified(&program, &Path::new(&dir).join("slice.lean"))
        .unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
