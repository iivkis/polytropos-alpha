//! Structural and NVIDIA parity acceptance for strict-order Sqrt→Add→Div.
//! The rule is independent of any optimizer or model architecture.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor, TranslationKind,
    TranslationPlan,
};

fn fixture(sqrt_on_right: bool) -> (NeuralGraph, [NodeId; 3], NodeId, NodeId, NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let x = graph.input(shape.clone(), Domain::Any);
    let epsilon = graph.input(shape.clone(), Domain::Any);
    let numerator = graph.input(shape, Domain::Any);
    let sqrt = graph.sqrt_ieee(x).unwrap();
    let sum = if sqrt_on_right {
        graph.add(epsilon, sqrt).unwrap()
    } else {
        graph.add(sqrt, epsilon).unwrap()
    };
    let output = graph.div(numerator, sum).unwrap();
    (graph, [x, epsilon, numerator], sqrt, sum, output)
}

#[test]
fn rounded_sqrt_add_div_rule_needs_unique_unobserved_intermediates() {
    let config = GpuLaunchConfig::default().with_pointwise_fusion();
    for right in [false, true] {
        let (mut graph, ids, sqrt, sum, root) = fixture(right);
        let plain = GpuEngine::default().compile(&graph, root).unwrap();
        let optimized = GpuEngine::default()
            .with_launch_config(config)
            .compile(&graph, root)
            .unwrap();
        assert_eq!(plain.kernel_count().unwrap(), 3);
        assert_eq!(optimized.kernel_count().unwrap(), 1);
        assert_eq!(
            optimized.fused_sqrt_add_divs()[&root.index()],
            (sqrt.index(), sum.index(), usize::from(right))
        );
        let plan = optimized.translation_plan().unwrap();
        assert_eq!(plan.kernels.len(), 1);
        assert_eq!(
            plan.kernels[0].kind,
            TranslationKind::SqrtAddDiv {
                sqrt_on_right: right
            }
        );
        assert_eq!(plan.kernels[0].reads.len(), 3);
        assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
        assert_eq!(optimized.capability_plan().unwrap().optimizations.len(), 1);
        assert!(!optimized.llvm_ir().contains("fma.rn.f32"));

        let observed = GpuEngine::default()
            .with_launch_config(config)
            .compile_outputs(&graph, &[root, sum])
            .unwrap();
        assert!(observed.fused_sqrt_add_divs().is_empty());
        let second = graph.add(sum, ids[0]).unwrap();
        let shared = GpuEngine::default()
            .with_launch_config(config)
            .compile_outputs(&graph, &[root, second])
            .unwrap();
        assert!(shared.fused_sqrt_add_divs().is_empty());
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph + NVPTX LLVM; fused/unfused FP32 bit equality"]
fn rounded_sqrt_add_div_matches_unfused_gpu_bits() {
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let fused_engine = engine
        .clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let data = [
        [0.25, 1., 2., 4., 16., 0.001, -0., 64.],
        [1e-8, 1., 0.1, 0., 1e-6, 0.001, 0., 1.],
        [1., -0.25, 2., -4., 0.5, 10., 0., 3.],
    ];
    let directory = std::env::temp_dir().join(format!("vn_sqrt_add_div_{}", std::process::id()));
    for right in [false, true] {
        let (graph, ids, _, _, root) = fixture(right);
        let bound = ids
            .iter()
            .zip(&data)
            .map(|(&id, values)| (id, Tensor::from_slice(vec![1, 8], values).unwrap()))
            .collect::<Vec<_>>();
        let plain = engine.compile(&graph, root).unwrap();
        let optimized = fused_engine.compile(&graph, root).unwrap();
        let reference = engine
            .execute(
                &plain,
                &bound,
                &directory.join(format!("reference_{right}")),
            )
            .unwrap();
        let actual = fused_engine
            .execute(
                &optimized,
                &bound,
                &directory.join(format!("fused_{right}")),
            )
            .unwrap();
        for (lane, (&want, &got)) in reference.values().iter().zip(actual.values()).enumerate() {
            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "sqrt-add-div operand order {right} lane {lane}"
            );
        }
    }
}
