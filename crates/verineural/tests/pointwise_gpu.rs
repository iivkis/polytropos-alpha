//! Executable versioned pointwise signatures are universal mathematical
//! primitives. Unrecognized expressions and implicit exp approximation fail closed.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, PointwiseExpr, PointwiseSpec, Shape,
    TranslationKind, TranslationPlan, audit_fp32_ptx,
};

#[test]
fn strict_rne_arithmetic_and_explicit_approximation() {
    let mut g = NeuralGraph::default();
    let a = g.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let b = g.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let difference = g.sub(a, b).unwrap();
    let quotient = g.div(difference, b).unwrap();
    let exp = g.exp(quotient).unwrap();

    let strict = GpuEngine::default();
    assert!(strict.compile(&g, exp).is_err());
    let exact_profile = strict.compile(&g, quotient).unwrap();
    let exact_plan = exact_profile.translation_plan().unwrap();
    assert!(
        exact_plan
            .kernels
            .iter()
            .any(|k| k.kind == TranslationKind::Sub)
    );
    assert!(
        exact_plan
            .kernels
            .iter()
            .any(|k| k.kind == TranslationKind::DivRne)
    );

    let engine =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    let code = engine.compile(&g, exp).unwrap();
    let plan = code.translation_plan().unwrap();
    assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
    assert!(
        plan.kernels
            .iter()
            .any(|k| k.kind == TranslationKind::ExpApprox)
    );
    assert!(code.llvm_ir().contains("ex2.approx.f32"));
}

#[test]
fn primitive_id_does_not_license_different_expression() {
    let mut g = NeuralGraph::default();
    let x = g.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let spoof = PointwiseSpec::new("verineural.exp", "1", 1, PointwiseExpr::input(0).relu())
        .unwrap()
        .apply(&mut g, &[x])
        .unwrap();
    assert!(
        GpuEngine::default()
            .with_launch_config(GpuLaunchConfig::default().with_approximate_exp())
            .compile(&g, spoof)
            .is_err()
    );
}

#[test]
fn ptx_audit_rejects_unexpected_exponential() {
    let fake_ptx = "ex2.approx.f32 %f0, %f1;\n";
    assert!(audit_fp32_ptx("", fake_ptx).is_err());
    assert!(audit_fp32_ptx("call float asm sideeffect \"ex2.approx.f32", fake_ptx).is_ok());
    assert!(audit_fp32_ptx("", "div.approx.f32 %f0, %f1, %f2;\n").is_err());
}
