use verineural::{
    Domain, Fp32PtxAudit, GpuEngine, GpuError, GpuLaunchConfig, NeuralGraph, ReluFusion, Shape,
    audit_fp32_ptx,
};

#[test]
fn static_ptx_audit_rejects_contraction_and_non_rne_instructions() {
    let llvm = "call float asm sideeffect \"mul.rn.f32";
    let good = "mul.rn.f32 %f1, %f2, %f3;";
    assert_eq!(
        audit_fp32_ptx(llvm, good).unwrap(),
        Fp32PtxAudit {
            rounded_adds: 0,
            rounded_muls: 1,
        }
    );
    for ptx in [
        "fma.rn.f32 %f1, %f2, %f3, %f4;",
        "mad.rn.f32 %f1, %f2, %f3, %f4;",
        "mul.ftz.rn.f32 %f1, %f2, %f3;",
        "mul.f32 %f1, %f2, %f3;",
        "add.rn.f32 %f1, %f2, %f3;",
        "",
    ] {
        assert!(matches!(
            audit_fp32_ptx(llvm, ptx),
            Err(GpuError::Fp32PtxRejected { .. })
        ));
    }
    assert!(matches!(
        audit_fp32_ptx("fmul float %x, %y", good),
        Err(GpuError::Fp32PtxRejected { .. })
    ));
}

#[test]
fn separate_f32_rounding_differs_from_fused_multiply_add() {
    // (1 + 2^-23)^2 rounds to 1 + 2^-22, while the fused residual is 2^-46.
    let a = f32::from_bits(0x3f80_0001);
    let rounded_product = a * a;
    let bias = -rounded_product;
    assert_eq!((rounded_product + bias).to_bits(), 0.0f32.to_bits());
    assert_ne!(a.mul_add(a, bias).to_bits(), 0.0f32.to_bits());

    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![1, 1]).unwrap();
    let bias_shape = Shape::new(vec![1]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let b = graph.input(bias_shape, Domain::Finite);
    let product = graph.mul(x, y).unwrap();
    let biased = graph.bias_add(product, b).unwrap();
    let output = graph.relu(biased).unwrap();
    let baseline = GpuEngine::default().compile(&graph, output).unwrap();
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, output)
        .unwrap();
    assert_eq!(fused.removed_kernel_count(), 2);
    for program in [&baseline, &fused] {
        let ir = program.llvm_ir();
        assert!(ir.contains("mul.rn.f32 $0, $1, $2;"));
        assert!(ir.contains("add.rn.f32 $0, $1, $2;"));
        assert!(!ir.contains("fmul float"));
        assert!(!ir.contains("fadd float"));
    }
}
