//! Generic structural GPU profile consumed by labs; no architecture-specific kernel.
use verineural::{
    Domain, GpuEngine, GpuError, NeuralGraph, Shape, Tensor, TranslationKind, TranslationPlan,
};

fn fixture() -> (
    NeuralGraph,
    verineural::NodeId,
    verineural::NodeId,
    verineural::NodeId,
) {
    let mut g = NeuralGraph::default();
    let data = g.input(Shape::new(vec![1, 6]).unwrap(), Domain::Finite);
    let scalar = g.input(Shape::new(vec![1, 1]).unwrap(), Domain::Finite);
    let tail = g.slice_axis(data, 1, 2, 3).unwrap();
    let joined = g.concat(1, &[tail, tail]).unwrap();
    let replicated = g
        .broadcast_to(scalar, Shape::new(vec![1, 6]).unwrap())
        .unwrap();
    let result = g.add(joined, replicated).unwrap();
    (g, data, scalar, result)
}

#[test]
fn last_row_profile_has_distinct_checked_kernels() {
    let (g, _, _, result) = fixture();
    let program = GpuEngine::default().compile(&g, result).unwrap();
    assert_eq!(program.kernel_count().unwrap(), 4);
    let plan = program.translation_plan().unwrap();
    assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
    assert!(plan.kernels.iter().any(|k| matches!(
        k.kind,
        TranslationKind::SliceLastRow {
            start: 2,
            source_width: 6
        }
    )));
    assert!(
        plan.kernels
            .iter()
            .any(|k| matches!(k.kind, TranslationKind::BroadcastScalar))
    );
    assert!(
        plan.kernels
            .iter()
            .any(|k| matches!(k.kind, TranslationKind::ConcatAxis0 { left: 3 }))
    );
    assert!(
        program
            .memory_lean_source()
            .unwrap()
            .contains("slice_last_row_address_bound")
    );
}

#[test]
fn unsupported_nonflat_shapes_still_fail_closed() {
    let mut g = NeuralGraph::default();
    let source = g.input(Shape::new(vec![2, 6]).unwrap(), Domain::Finite);
    let row_slice = g.slice_axis(source, 1, 2, 3).unwrap();
    assert!(matches!(
        GpuEngine::default().compile(&g, row_slice),
        Err(GpuError::UnsupportedOperation {
            operation: "slice_axis",
            ..
        })
    ));
    let other = g.input(Shape::new(vec![2, 6]).unwrap(), Domain::Finite);
    let concatenated = g.concat(1, &[source, other]).unwrap();
    assert!(matches!(
        GpuEngine::default().compile(&g, concatenated),
        Err(GpuError::UnsupportedOperation {
            operation: "concat",
            ..
        })
    ));
    let scalar = g.input(Shape::new(vec![1, 1]).unwrap(), Domain::Finite);
    let broadcast = g
        .broadcast_to(scalar, Shape::new(vec![2, 6]).unwrap())
        .unwrap();
    assert!(matches!(
        GpuEngine::default().compile(&g, broadcast),
        Err(GpuError::UnsupportedOperation {
            operation: "broadcast_to",
            ..
        })
    ));
}

#[test]
#[ignore = "requires NVIDIA CUDA and LLVM NVPTX"]
fn gpu_graph_replay_matches_independent_literal_oracle() {
    let (g, data, scalar, result) = fixture();
    let engine = GpuEngine::default();
    let program = engine.compile(&g, result).unwrap();
    let workspace =
        std::env::temp_dir().join(format!("vn_structural_last_row_{}", std::process::id()));
    let mut execution = engine
        .prepare(
            &program,
            &[
                (
                    data,
                    Tensor::from_slice(vec![1, 6], &[1., 2., 3., 4., 5., 6.]).unwrap(),
                ),
                (scalar, Tensor::from_slice(vec![1, 1], &[2.]).unwrap()),
            ],
            &workspace,
        )
        .unwrap();
    execution.replay().unwrap();
    assert_eq!(
        execution.read_output().unwrap().values(),
        &[5., 6., 7., 5., 6., 7.]
    );
}
