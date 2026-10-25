//! Generic four-op rounded fusion: Mul→Mul(reuse)→Add→Div.
//! Each template preserves operand order, intermediate rounding and fanout.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor, TranslationKind,
    TranslationPlan,
};

type Fixture = (NeuralGraph, [NodeId; 4], [NodeId; 3], NodeId);

fn fixture(reused: usize, multiply_on_right: bool, add_on_right: bool) -> Fixture {
    let mut g = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let a = g.input(shape.clone(), Domain::Any);
    let b = g.input(shape.clone(), Domain::Any);
    let c = g.input(shape.clone(), Domain::Any);
    let d = g.input(shape, Domain::Any);
    let first = g.mul(a, b).unwrap();
    let repeated = if reused == 0 { a } else { b };
    let second = if multiply_on_right {
        g.mul(repeated, first).unwrap()
    } else {
        g.mul(first, repeated).unwrap()
    };
    let sum = if add_on_right {
        g.add(c, second).unwrap()
    } else {
        g.add(second, c).unwrap()
    };
    let root = g.div(sum, d).unwrap();
    (g, [a, b, c, d], [first, second, sum], root)
}

#[test]
fn four_op_fusion_is_structurally_audited_and_preserves_all_selected_values() {
    let optimized = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let plain = GpuEngine::default();
    for reused in 0..2 {
        for multiply_on_right in [false, true] {
            for add_on_right in [false, true] {
                let (mut g, ids, [first, second, sum], root) =
                    fixture(reused, multiply_on_right, add_on_right);
                let baseline = plain.compile(&g, root).unwrap();
                let fused = optimized.compile(&g, root).unwrap();
                assert_eq!(baseline.kernel_count().unwrap(), 4);
                assert_eq!(fused.kernel_count().unwrap(), 1);
                assert_eq!(
                    fused.fused_mul_mul_add_divs()[&root.index()],
                    (
                        first.index(),
                        second.index(),
                        sum.index(),
                        reused,
                        usize::from(multiply_on_right),
                        usize::from(add_on_right),
                    )
                );
                let contract = fused.memory_contracts().unwrap();
                assert_eq!(contract.len(), 1);
                assert_eq!(contract[0].reads.len(), 4);
                assert!(contract[0].reads.iter().all(|&(_, len)| len == 8));
                let plan = fused.translation_plan().unwrap();
                assert_eq!(
                    plan.kernels[0].kind,
                    TranslationKind::MulMulAddDiv {
                        reused_source: reused as u8,
                        multiply_on_right,
                        add_on_right,
                    }
                );
                assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
                let capability = fused.capability_plan().unwrap();
                assert_eq!(capability.optimizations.len(), 1);
                assert_eq!(capability.optimizations[0].removed_nodes.len(), 3);
                assert!(!fused.llvm_ir().contains("fma.rn.f32"));
                assert_eq!(
                    fused.llvm_ir().matches("call float asm sideeffect").count(),
                    4
                );

                for &observed in &[first, second, sum] {
                    let selected = optimized.compile_outputs(&g, &[root, observed]).unwrap();
                    assert!(selected.fused_mul_mul_add_divs().is_empty());
                }
                let extra = g.add(first, ids[2]).unwrap();
                let shared = optimized.compile_outputs(&g, &[root, extra]).unwrap();
                assert!(shared.fused_mul_mul_add_divs().is_empty());
            }
        }
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph, LLVM NVPTX; strict FP32 bit-parity for all eight variants"]
fn four_op_gpu_matches_unfused_bits() {
    let plain = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let fused = plain
        .clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let data = [
        [0.125, -0.25, 0.5, 1., 3., 4., 8., -0.0],
        [0.5, -1., 2., -3., 0.25, 0.75, -0.5, 1.],
        [2., -3., 0.5, 1.25, 0., -0.0, 2., 4.],
        [1., 3., 0.25, 1e-8, 2., 0.5, -1., 0.75],
    ];
    let dir = std::env::temp_dir()
        .join(format!("vn_four_op_{}", std::process::id()));
    for reused in 0..2 {
        for multiply_on_right in [false, true] {
            for add_on_right in [false, true] {
                let (graph, inputs, _, root) =
                    fixture(reused, multiply_on_right, add_on_right);
                let bound = inputs
                    .iter()
                    .zip(&data)
                    .map(|(&id, values)| (id, Tensor::from_slice(vec![1, 8], values).unwrap()))
                    .collect::<Vec<_>>();
                let ordinary = plain.compile(&graph, root).unwrap();
                let optimized = fused.compile(&graph, root).unwrap();
                let suffix = format!("{reused}_{}_{add_on_right}", multiply_on_right);
                let expected = plain
                    .execute(&ordinary, &bound, &dir.join(format!("base_{suffix}")))
                    .unwrap();
                let actual = fused
                    .execute(&optimized, &bound, &dir.join(format!("fused_{suffix}")))
                    .unwrap();
                if reused == 0 && !multiply_on_right && !add_on_right {
                    // Fail closed even when PTX still contains the same
                    // number of arithmetic instructions with altered
                    // rounding. The plan is derived from Semantic IR.
                    let pt_xpath = fused.lower(
                        &optimized,
                        &dir.join("ptx_audit"),
                    ).unwrap();
                    let trusted_ptx = std::fs::read_to_string(pt_xpath).unwrap();
                    let changed = trusted_ptx.replacen("mul.rn.f32", "mul.f32", 1);
                    assert_ne!(trusted_ptx, changed);
                    assert!(
                        optimized.translation_plan().unwrap()
                            .validate(optimized.llvm_ir(), &changed)
                            .is_err()
                    );
                }
                for (lane, (&x, &y)) in expected
                    .values()
                    .iter()
                    .zip(actual.values())
                    .enumerate()
                {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "four-op variant={suffix} lane={lane}"
                    );
                }
            }
        }
    }
}
