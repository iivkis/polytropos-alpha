//! Strict-order two-op Add/Mul CUDA kernel fusion: structural contracts,
//! observable-producer barriers and opt-in GPU bitwise parity.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor, TranslationKind,
    TranslationPlan,
};

fn fixture(
    first_add: bool,
    second_add: bool,
    producer_on_right: bool,
) -> (NeuralGraph, [NodeId; 3], NodeId, NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let a = graph.input(shape.clone(), Domain::Finite);
    let b = graph.input(shape.clone(), Domain::Finite);
    let c = graph.input(shape, Domain::Finite);
    let inner = if first_add {
        graph.add(a, b).unwrap()
    } else {
        graph.mul(a, b).unwrap()
    };
    let root = match (second_add, producer_on_right) {
        (true, true) => graph.add(c, inner).unwrap(),
        (true, false) => graph.add(inner, c).unwrap(),
        (false, true) => graph.mul(c, inner).unwrap(),
        (false, false) => graph.mul(inner, c).unwrap(),
    };
    (graph, [a, b, c], inner, root)
}

#[test]
fn binary_pointwise_fusion_preserves_order_and_selected_intermediates() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    // Mul->Add is covered by the existing dedicated MulAdd rule.
    for (first_add, second_add) in [(true, true), (true, false), (false, false)] {
        for producer_on_right in [false, true] {
            let (mut graph, [a, _, _], producer, root) =
                fixture(first_add, second_add, producer_on_right);
            let ordinary = GpuEngine::default().compile(&graph, root).unwrap();
            let optimized = GpuEngine::default()
                .with_launch_config(config)
                .compile(&graph, root)
                .unwrap();
            assert_eq!(ordinary.kernel_count().unwrap(), 2);
            assert_eq!(optimized.kernel_count().unwrap(), 1);
            assert_eq!(
                optimized.fused_binary_pairs()[&root.index()],
                (producer.index(), usize::from(producer_on_right))
            );
            let plan = optimized.translation_plan().unwrap();
            assert_eq!(
                plan.kernels[0].kind,
                TranslationKind::BinaryChain {
                    first_add,
                    second_add,
                    producer_on_right
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
                2
            );

            let selected = GpuEngine::default()
                .with_launch_config(config)
                .compile_outputs(&graph, &[root, producer])
                .unwrap();
            assert!(selected.fused_binary_pairs().is_empty());
            let extra = graph.add(producer, a).unwrap();
            let combined = graph.add(root, extra).unwrap();
            let shared = GpuEngine::default()
                .with_launch_config(config)
                .compile(&graph, combined)
                .unwrap();
            assert!(!shared.fused_binary_pairs().contains_key(&root.index()));
        }
    }
}

#[test]
#[ignore = "requires LLVM NVPTX and NVIDIA CUDA; compares fused/unfused FP32 bits"]
fn binary_pointwise_gpu_matches_unfused_bitwise() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let optimized_engine = engine.clone().with_launch_config(config);
    let data = [
        [0., -0., 1.5, -0.5, 1000., -13., 0.25, -12.5],
        [2., -1., -0.25, 8., 0.001, 17., -2., 0.5],
        [-1., 0., 4., -3., 0.25, -0., 0.5, 2.],
    ];
    let dir = std::env::temp_dir().join(format!("vn_binary_chain_{}", std::process::id()));
    for (first_add, second_add) in [(true, true), (true, false), (false, false)] {
        for producer_on_right in [false, true] {
            let (graph, ids, _, root) = fixture(first_add, second_add, producer_on_right);
            let bindings = ids
                .iter()
                .zip(&data)
                .map(|(&id, values)| (id, Tensor::from_slice(vec![1, 8], values).unwrap()))
                .collect::<Vec<_>>();
            let ordinary = engine.compile(&graph, root).unwrap();
            let fused = optimized_engine.compile(&graph, root).unwrap();
            let key = format!("{first_add}_{second_add}_{producer_on_right}");
            let reference = engine
                .execute(&ordinary, &bindings, &dir.join(format!("{key}_plain")))
                .unwrap();
            let actual = optimized_engine
                .execute(&fused, &bindings, &dir.join(format!("{key}_fused")))
                .unwrap();
            for (lane, (&want, &got)) in reference.values().iter().zip(actual.values()).enumerate()
            {
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "{key} lane={lane}: {want} vs {got}"
                );
            }
        }
    }
}
