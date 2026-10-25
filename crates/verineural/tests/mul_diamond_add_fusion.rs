//! Four-source DAG fusion: Add(Mul(a,b), Mul(Mul(c,d), c|d)).
//! The selected Add output remains observable. Internal Mul values must not.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor,
    TranslationKind, TranslationPlan,
};

type Fixture = (NeuralGraph, [NodeId; 4], [NodeId; 3], NodeId);

fn fixture(reused: usize, right_mul_on_right: bool, left_mul_on_right: bool) -> Fixture {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let a = graph.input(shape.clone(), Domain::Any);
    let b = graph.input(shape.clone(), Domain::Any);
    let c = graph.input(shape.clone(), Domain::Any);
    let d = graph.input(shape, Domain::Any);
    let left = graph.mul(a, b).unwrap();
    let right = graph.mul(c, d).unwrap();
    let repeat = if reused == 0 { c } else { d };
    let repeated = if right_mul_on_right {
        graph.mul(repeat, right).unwrap()
    } else {
        graph.mul(right, repeat).unwrap()
    };
    let output = if left_mul_on_right {
        graph.add(repeated, left).unwrap()
    } else {
        graph.add(left, repeated).unwrap()
    };
    (graph, [a, b, c, d], [left, right, repeated], output)
}

#[test]
fn diamond_fusion_reduces_kernels_and_preserves_observable_output() {
    let baseline = GpuEngine::default();
    let optimized = baseline.clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    for reused in 0..2 {
        for right_mul_on_right in [false, true] {
            for left_mul_on_right in [false, true] {
                let (mut graph, inputs, [left, right, repeated], root) =
                    fixture(reused, right_mul_on_right, left_mul_on_right);
                let plain = baseline.compile(&graph, root).unwrap();
                let fused = optimized.compile(&graph, root).unwrap();
                assert_eq!(plain.kernel_count().unwrap(), 4);
                assert_eq!(fused.kernel_count().unwrap(), 1);
                assert_eq!(
                    fused.fused_mul_diamond_adds()[&root.index()],
                    (left.index(), right.index(), repeated.index(), reused,
                     usize::from(right_mul_on_right), usize::from(left_mul_on_right))
                );
                let contracts = fused.memory_contracts().unwrap();
                assert_eq!(contracts.len(), 1);
                assert_eq!(contracts[0].reads.len(), 4);
                assert!(contracts[0].reads.iter().all(|&(_, n)| n == 8));
                let plan = fused.translation_plan().unwrap();
                assert_eq!(plan.kernels[0].kind, TranslationKind::MulDiamondAdd {
                    reused_source: reused as u8, right_mul_on_right, left_mul_on_right,
                });
                assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
                let capability = fused.capability_plan().unwrap();
                assert_eq!(capability.optimizations.len(), 1);
                assert_eq!(capability.optimizations[0].removed_nodes.len(), 3);
                assert!(!fused.llvm_ir().contains("fma.rn.f32"));

                for &observed in &[left, right, repeated] {
                    let selected = optimized.compile_outputs(&graph, &[root, observed]).unwrap();
                    assert!(selected.fused_mul_diamond_adds().is_empty());
                }
                let extra = graph.add(right, inputs[1]).unwrap();
                let shared = optimized.compile_outputs(&graph, &[root, extra]).unwrap();
                assert!(shared.fused_mul_diamond_adds().is_empty());
            }
        }
    }
}

#[test]
#[ignore = "NVIDIA CUDA Graph + NVPTX LLVM: all eight RN operand-order variants and PTX rejection"]
fn diamond_gpu_matches_unfused_bits() {
    let baseline = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let optimized = baseline.clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let data = [
        [0.125, -0.25, 0.5, 1., 3., 4., 8., -0.0],
        [0.5, -1., 2., -3., 0.25, 0.75, -0.5, 1.],
        [2., -3., 0.5, 1.25, 0., -0.0, 2., 4.],
        [1., 3., 0.25, 1e-8, 2., 0.5, -1., 0.75],
    ];
    let dir = std::env::temp_dir().join(format!("vn_diamond_{}", std::process::id()));
    for reused in 0..2 {
        for right_mul_on_right in [false, true] {
            for left_mul_on_right in [false, true] {
                let (graph, sources, _, root) = fixture(
                    reused, right_mul_on_right, left_mul_on_right
                );
                let bound = sources.iter().zip(&data)
                    .map(|(&id, array)| (id, Tensor::from_slice(vec![1, 8], array).unwrap()))
                    .collect::<Vec<_>>();
                let a = baseline.compile(&graph, root).unwrap();
                let b = optimized.compile(&graph, root).unwrap();
                let suffix = format!("{reused}_{}_{left_mul_on_right}", right_mul_on_right);
                let expected = baseline.execute(&a, &bound, &dir.join(format!("old_{suffix}"))).unwrap();
                let actual = optimized.execute(&b, &bound, &dir.join(format!("new_{suffix}"))).unwrap();
                for (lane, (&want, &got)) in expected.values().iter().zip(actual.values()).enumerate() {
                    assert_eq!(want.to_bits(), got.to_bits(), "variant={suffix} lane={lane}");
                }
                if reused == 0 && !right_mul_on_right && !left_mul_on_right {
                    let ptx_path = optimized.lower(&b, &dir.join("audit")).unwrap();
                    let ptx = std::fs::read_to_string(ptx_path).unwrap();
                    let modified = ptx.replacen("mul.rn.f32", "mul.f32", 1);
                    assert_ne!(modified, ptx);
                    assert!(b.translation_plan().unwrap()
                        .validate(b.llvm_ir(), &modified).is_err());
                }
            }
        }
    }
}
