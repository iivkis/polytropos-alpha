//! P1/P12 explicit GPU capability boundary.
//! Compile success is not enough: every launched kernel must be paired with
//! one memory contract and one translation rule, and every removed fusion node
//! must have recorded structural preconditions.

use verineural::{
    Domain, GpuAssuranceStatus, GpuEngine, GpuLaunchConfig, GpuOptimizationKind, LeanProofEngine,
    audit_fp32_ptx,
    NeuralGraph, ReluFusion, Shape, TranslationKind,
};

fn concat_program() -> verineural::GpuProgram {
    let mut graph = NeuralGraph::default();
    let left = graph.input(Shape::new(vec![1, 2]).unwrap(), Domain::Finite);
    let right = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let output = graph.concat(0, &[left, right]).unwrap();
    GpuEngine::default().compile(&graph, output).unwrap()
}

fn fused_program() -> verineural::GpuProgram {
    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let output = graph.relu(biased).unwrap();
    GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, output)
        .unwrap()
}

#[test]
fn altered_ptx_fp32_policy_cannot_pass_numeric_opcode_audit() {
    // A narrow independent mutation fixture: counts and opcodes are checked,
    // but a successful audit does NOT prove operand-level semantics or SASS.
    let llvm = "call float asm sideeffect \"add.rn.f32\"\ncall float asm sideeffect \"mul.rn.f32\"";
    let ptx = "add.rn.f32 %f1, %f2, %f3;\nmul.rn.f32 %f4, %f5, %f6;";
    let ok = audit_fp32_ptx(llvm, ptx).unwrap();
    assert_eq!(ok.rounded_adds, 1);
    assert_eq!(ok.rounded_muls, 1);

    for corrupted in [
        ptx.replace("add.rn.f32", "add.ftz.f32"),
        ptx.replace("mul.rn.f32", "fma.rn.f32"),
        ptx.replace("mul.rn.f32", "mul.approx.f32"),
        ptx.replace("mul.rn.f32", "add.rn.f32"),
    ] {
        assert!(audit_fp32_ptx(llvm, &corrupted).is_err());
    }
    assert!(audit_fp32_ptx("fadd float %a, %b", ptx).is_err());
}

#[test]
fn capability_plan_covers_fused_kernel_and_explicit_trust_boundaries() {
    let program = fused_program();
    let plan = program.capability_plan().unwrap();

    assert_eq!(plan.kernels.len(), 1);
    assert_eq!(plan.kernels[0].operation, "relu");
    assert_eq!(
        plan.kernels[0].translation_kind,
        TranslationKind::AddBiasRelu { columns: 3 }
    );
    assert_eq!(plan.optimizations.len(), 1);
    assert_eq!(
        plan.optimizations[0].kind,
        GpuOptimizationKind::PointwiseBiasReluFusion
    );
    assert_eq!(plan.optimizations[0].removed_nodes.len(), 2);
    assert!(plan.optimizations[0].exclusive_reachable_use);
    assert!(plan.optimizations[0].selected_outputs_preserved);

    assert_eq!(
        plan.boundary.memory_address_arithmetic,
        GpuAssuranceStatus::GeneratedLeanObligation
    );
    assert_eq!(
        plan.boundary.llvm_ptx_structure,
        GpuAssuranceStatus::RequiredAtLowering
    );
    assert_eq!(
        plan.boundary.ideal_to_fp32,
        GpuAssuranceStatus::SeparateNumericProofRequired
    );
    assert_eq!(
        plan.boundary.cuda_runtime,
        GpuAssuranceStatus::RuntimeAcceptanceRequired
    );
    assert_eq!(
        plan.boundary.jit_sass_correspondence,
        GpuAssuranceStatus::NotEstablished
    );

    let text = plan.to_text();
    assert!(text.starts_with("verineural-gpu-capability-plan-v1\n"));
    assert!(text.contains("translation=add_bias_relu"));
    assert!(text.contains("kind=pointwise-bias-relu-fusion"));
    assert!(text.contains("jit_sass_correspondence=not-established"));
}

#[test]
fn selected_intermediate_output_prevents_illegal_fusion() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let relu = graph.relu(sum).unwrap();

    let program = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile_outputs(&graph, &[sum, relu])
        .unwrap();
    let plan = program.capability_plan().unwrap();
    assert_eq!(plan.optimizations.len(), 0);
    assert_eq!(plan.kernels.len(), 2);
    assert!(
        plan.kernels
            .iter()
            .any(|kernel| kernel.translation_kind == TranslationKind::Add)
    );
    assert!(
        plan.kernels
            .iter()
            .any(|kernel| kernel.translation_kind == TranslationKind::Relu)
    );
}

#[test]
fn memory_capability_has_named_auditable_theorems() {
    let program = fused_program();
    let memory = program.memory_lean_program().unwrap();
    assert!(memory.source().contains("namespace VeriNeural.Generated"));
    assert!(!memory.source().contains("\nexample "));
    assert_eq!(memory.theorem_names().len(), 2);
    assert!(
        memory
            .theorem_names()
            .iter()
            .any(|name| name.ends_with("_output_bound"))
    );
    assert!(
        memory
            .theorem_names()
            .iter()
            .any(|name| name.ends_with("_bias_bound"))
    );
}

/// Explicit P12 acceptance. Run with: task verify-gpu-capabilities
#[test]
#[ignore = "requires pinned Lean 4 from Nix devShell"]
fn lean_kernel_and_axiom_audit_check_memory_capability() {
    let root =
        std::env::temp_dir().join(format!("verineural-gpu-capability-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    for (name, program) in [("fused", fused_program()), ("concat", concat_program())] {
        let memory = program.memory_lean_program().unwrap();
        let (_, audit) = LeanProofEngine::default()
            .verify_certified(&memory, &root.join(format!("{name}.lean")))
            .unwrap();
        assert_eq!(audit.theorem_names.as_slice(), memory.theorem_names());
        if name == "concat" {
            assert!(
                memory
                    .theorem_names()
                    .iter()
                    .any(|theorem| theorem.ends_with("_concat_right_bound"))
            );
        }
    }

    std::fs::remove_dir_all(root).unwrap();
}
