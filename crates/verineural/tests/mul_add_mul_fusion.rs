//! Three-op strict-order Mul -> Add -> Mul GPU fusion acceptance.
//! At most three external buffers. Arithmetic retains independent RN instructions.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, Shape, Tensor, TranslationKind,
    TranslationPlan,
};

fn fixture(
    add_right: bool,
    outer_right: bool,
    reuse: usize,
) -> (
    NeuralGraph,
    [verineural::NodeId; 3],
    [verineural::NodeId; 3],
) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let a = graph.input(shape.clone(), Domain::Finite);
    let b = graph.input(shape.clone(), Domain::Finite);
    let c = graph.input(shape, Domain::Finite);
    let mul = graph.mul(a, b).unwrap();
    let add = if add_right {
        graph.add(c, mul).unwrap()
    } else {
        graph.add(mul, c).unwrap()
    };
    let other = [a, b, c][reuse];
    let root = if outer_right {
        graph.mul(other, add).unwrap()
    } else {
        graph.mul(add, other).unwrap()
    };
    (graph, [a, b, c], [mul, add, root])
}

#[test]
fn three_op_fusion_is_structurally_checked_and_roundtrippable() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    for add_right in [false, true] {
        for outer_right in [false, true] {
            for reuse in 0..3 {
                let (g, _, [mul, add, root]) = fixture(add_right, outer_right, reuse);
                let ordinary = GpuEngine::default().compile(&g, root).unwrap();
                let optimized = GpuEngine::default()
                    .with_launch_config(config)
                    .compile(&g, root)
                    .unwrap();
                assert_eq!(ordinary.kernel_count().unwrap(), 3);
                assert_eq!(optimized.kernel_count().unwrap(), 1);
                assert_eq!(optimized.removed_kernel_count(), 2);
                assert_eq!(
                    optimized.fused_mul_add_muls()[&root.index()],
                    (
                        mul.index(),
                        add.index(),
                        usize::from(add_right),
                        usize::from(outer_right),
                        reuse
                    )
                );
                let plan = optimized.translation_plan().unwrap();
                assert_eq!(
                    plan.kernels[0].kind,
                    TranslationKind::MulAddMul {
                        add_mul_on_right: add_right,
                        mul_add_on_right: outer_right,
                        reused_source: reuse as u8,
                    }
                );
                assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
                assert_eq!(optimized.capability_plan().unwrap().optimizations.len(), 1);
                assert!(!optimized.llvm_ir().contains("fma.rn.f32"));
                assert_eq!(
                    optimized
                        .llvm_ir()
                        .matches("call float asm sideeffect")
                        .count(),
                    3
                );
                // Observable intermediates are never removed.
                let observed = GpuEngine::default()
                    .with_launch_config(config)
                    .compile_outputs(&g, &[root, add])
                    .unwrap();
                assert!(observed.fused_mul_add_muls().is_empty());
                let observed_inner = GpuEngine::default()
                    .with_launch_config(config)
                    .compile_outputs(&g, &[root, mul])
                    .unwrap();
                assert!(observed_inner.fused_mul_add_muls().is_empty());
            }
        }
    }
}

#[test]
fn fanout_or_fourth_distinct_buffer_prevents_three_way_fusion() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    let (mut graph, [a, _, _], [mul, add, root]) = fixture(false, false, 2);
    let extra = graph.add(add, a).unwrap();
    let selected = graph.add(root, extra).unwrap();
    let shared = GpuEngine::default()
        .with_launch_config(config)
        .compile(&graph, selected)
        .unwrap();
    assert!(shared.fused_mul_add_muls().is_empty());
    let (mut graph, _, [mul, add, _]) = fixture(false, false, 2);
    let fourth = graph.input(Shape::new(vec![1, 8]).unwrap(), Domain::Finite);
    let root = graph.mul(add, fourth).unwrap();
    let fourth_source = GpuEngine::default()
        .with_launch_config(config)
        .compile(&graph, root)
        .unwrap();
    assert!(fourth_source.fused_mul_add_muls().is_empty());
    assert_eq!(fourth_source.kernel_count().unwrap(), 2);
    assert!(fourth_source.fused_mul_adds().contains_key(&add.index()));
    assert!(mul.index() < add.index());
}

#[test]
#[ignore = "requires LLVM NVPTX/NVIDIA GPU to verify bitwise fused/unfused parity"]
fn gpu_three_op_fusion_matches_unfused_bitwise() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    let values = [
        [0.25, -1.25, 2.5, 0., -0., 7., -4., 1.25],
        [2., -0.75, 0.5, 1., -2., 0.125, 3., -7.],
        [1., 0.25, -0.5, -3., 0., 6., 1.25, 2.],
    ];
    let directory = std::env::temp_dir().join(format!("vn_mul_add_mul_{}", std::process::id()));
    let arch = std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into());
    let engine = GpuEngine::new("llc", arch.clone());
    let fused_engine = GpuEngine::new("llc", arch.clone()).with_launch_config(config);
    for add_right in [false, true] {
        for outer_right in [false, true] {
            for reuse in 0..3 {
                let (graph, ids, [_, _, root]) = fixture(add_right, outer_right, reuse);
                let bindings = ids
                    .iter()
                    .zip(values.iter())
                    .map(|(&id, v)| (id, Tensor::from_slice(vec![1, 8], v).unwrap()))
                    .collect::<Vec<_>>();
                let ordinary = engine.compile(&graph, root).unwrap();
                let optimized = fused_engine.compile(&graph, root).unwrap();
                let key = format!("add{}_outer{}_reuse{}", add_right, outer_right, reuse);
                let left = engine
                    .execute(
                        &ordinary,
                        &bindings,
                        &directory.join(format!("{key}_plain")),
                    )
                    .unwrap();
                let right = fused_engine
                    .execute(
                        &optimized,
                        &bindings,
                        &directory.join(format!("{key}_fused")),
                    )
                    .unwrap();
                for (i, (&x, &y)) in left.values().iter().zip(right.values()).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "{key} lane={i}: {x} vs {y}");
                }
            }
        }
    }
}
