use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, ReluFusion, Shape, TranslationKind,
    TranslationPlan,
};

/// A minimal PTX skeleton for unit-testing structural validation without
/// llc or CUDA. Actual NVPTX assembly is exercised by the Stage 8 example.
fn synthetic_ptx(plan: &TranslationPlan) -> String {
    let mut ptx = ".version 7.0\n.target sm_52\n.address_size 64\n".to_owned();
    for kernel in &plan.kernels {
        ptx.push_str(&format!(
            ".visible .entry vn_node_{}(\n)\n{{\n",
            kernel.node
        ));
        let (adds, muls) = match kernel.kind {
            TranslationKind::Add
            | TranslationKind::BiasAdd { .. }
            | TranslationKind::AddRelu
            | TranslationKind::BiasAddRelu { .. } => (1, 0),
            TranslationKind::Mul | TranslationKind::MulRelu => (0, 1),
            TranslationKind::MatMul { .. } | TranslationKind::MulBiasRelu { .. } => (1, 1),
            TranslationKind::AddBiasRelu { .. } => (2, 0),
            TranslationKind::Relu
            | TranslationKind::Transpose2D { .. }
            | TranslationKind::ReverseRows2D { .. }
            | TranslationKind::ReverseColumns2D { .. }
            | TranslationKind::ConcatAxis0 { .. }
            | TranslationKind::ArgMaxLast { .. }
            | TranslationKind::TakeAxis0 { .. } => (0, 0),
        };
        for _ in 0..muls {
            ptx.push_str(" mul.rn.f32 %f1, %f2, %f3;\n");
        }
        for _ in 0..adds {
            ptx.push_str(" add.rn.f32 %f1, %f2, %f3;\n");
        }
        ptx.push_str("}\n");
    }
    ptx
}

fn dense_chain() -> (NeuralGraph, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let left = graph.input(Shape::new(vec![2, 3]).unwrap(), Domain::Finite);
    let right = graph.input(Shape::new(vec![3, 2]).unwrap(), Domain::Finite);
    let bias = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let matrix = graph.matmul(left, right).unwrap();
    let biased = graph.bias_add(matrix, bias).unwrap();
    let output = graph.relu(biased).unwrap();
    (graph, output)
}

fn fused_chain() -> (NeuralGraph, verineural::NodeId) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2, 3]).unwrap();
    let left = graph.input(shape.clone(), Domain::Finite);
    let right = graph.input(shape, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let product = graph.mul(left, right).unwrap();
    let biased = graph.bias_add(product, bias).unwrap();
    let activated = graph.relu(biased).unwrap();
    let output = graph
        .reshape(activated, Shape::new(vec![3, 2]).unwrap())
        .unwrap();
    (graph, output)
}

#[test]
fn translation_plan_cross_checks_real_generated_llvm_for_matmul_bias_relu() {
    let (graph, output) = dense_chain();
    for fusion in [ReluFusion::Disabled, ReluFusion::Experimental] {
        let program = GpuEngine::default()
            .with_launch_config(GpuLaunchConfig::new(256, fusion).unwrap())
            .compile(&graph, output)
            .unwrap();
        let plan = program.translation_plan().unwrap();
        assert_eq!(plan, TranslationPlan::from_text(&plan.to_text()).unwrap());
        let ptx = synthetic_ptx(&plan);
        let report = plan.validate(program.llvm_ir(), &ptx).unwrap();
        assert_eq!(report.kernel_count, plan.kernels.len());
        assert_eq!(report.rounded_adds, 2); // MatMul + BiasAdd
        assert_eq!(report.rounded_muls, 1);
        assert_eq!(report.relu_guards, 1);
        assert!(
            report
                .to_text()
                .contains("semantic_fp32_equivalence=not-proven")
        );
        assert!(report.to_text().contains("cuda_jit_sass_hardware=trusted"));
    }
}

#[test]
fn translation_plan_tracks_fused_sources_and_rejects_wrong_kernel_assignment() {
    let (graph, output) = fused_chain();
    for fusion in [ReluFusion::Disabled, ReluFusion::Experimental] {
        let program = GpuEngine::default()
            .with_launch_config(GpuLaunchConfig::new(256, fusion).unwrap())
            .compile(&graph, output)
            .unwrap();
        let plan = program.translation_plan().unwrap();
        let ptx = synthetic_ptx(&plan);
        let report = plan.validate(program.llvm_ir(), &ptx).unwrap();
        assert_eq!(report.rounded_adds, 1);
        assert_eq!(report.rounded_muls, 1);
        if fusion == ReluFusion::Experimental {
            assert_eq!(plan.kernels.len(), 1);
            assert!(matches!(
                plan.kernels[0].kind,
                TranslationKind::MulBiasRelu { columns: 3 }
            ));
            assert_eq!(program.removed_kernel_count(), 2);
            assert_eq!(plan.kernels[0].reads.len(), 3);
        } else {
            assert_eq!(plan.kernels.len(), 3);
        }
        let missing = ptx.replace(
            &format!(".entry vn_node_{}(", plan.kernels[0].node),
            ".entry vn_node_999(",
        );
        assert!(plan.validate(program.llvm_ir(), &missing).is_err());
        let duplicate = format!("{ptx}{ptx}");
        assert!(plan.validate(program.llvm_ir(), &duplicate).is_err());
    }
}

#[test]
fn translation_validator_fail_closed_on_llvm_and_ptx_tampering() {
    let (graph, output) = fused_chain();
    let program = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, output)
        .unwrap();
    let plan = program.translation_plan().unwrap();
    let llvm = program.llvm_ir();
    let ptx = synthetic_ptx(&plan);
    for changed in [
        llvm.replace("mul.rn.f32", "add.rn.f32"),
        llvm.replace("add.rn.f32", "mul.rn.f32"),
        llvm.replace("fcmp uno float", "fcmp oeq float"),
        llvm.replace("urem i32 %idx, 3", "urem i32 %idx, 2"),
        llvm.replace("%bias_value = load float", "%other = load float"),
        llvm.replace("icmp ult i32 %idx, %length", "icmp ule i32 %idx, %length"),
        llvm.replace("store float", "ret float"),
    ] {
        assert_ne!(changed, llvm, "test mutation must actually change LLVM");
        assert!(plan.validate(&changed, &ptx).is_err());
    }
    for changed in [
        ptx.replace("mul.rn.f32", "fma.rn.f32"),
        ptx.replace("mul.rn.f32", "mul.ftz.rn.f32"),
        ptx.replace("mul.rn.f32", "add.rn.f32"),
        ptx.replace("add.rn.f32", "add.f32"),
        ptx.replace("add.rn.f32", ""),
    ] {
        assert_ne!(changed, ptx);
        assert!(plan.validate(llvm, &changed).is_err());
    }
}

#[test]
fn translation_validator_rejects_swapped_per_kernel_operations_even_if_global_counts_match() {
    let (graph, output) = dense_chain();
    let program = GpuEngine::default().compile(&graph, output).unwrap();
    let plan = program.translation_plan().unwrap();
    let ptx = synthetic_ptx(&plan);
    assert!(plan.validate(program.llvm_ir(), &ptx).is_ok());
    // Both the MatMul and BiasAdd use an add, but only MatMul has a multiply.
    let bias_node = plan
        .kernels
        .iter()
        .find(|kernel| matches!(kernel.kind, TranslationKind::BiasAdd { .. }))
        .unwrap()
        .node;
    let prefix = format!(".entry vn_node_{bias_node}(\n)\n{{\n");
    let misplaced = ptx
        .replacen("mul.rn.f32 %f1, %f2, %f3;", "mov.f32 %f1, %f2;", 1)
        .replace(&prefix, &format!("{prefix} mul.rn.f32 %f1, %f2, %f3;\n"));
    assert_ne!(misplaced, ptx);
    // Global mul/add counts are unchanged, but the multiply moved to BiasAdd.
    assert_eq!(
        misplaced.matches("mul.rn.f32").count(),
        ptx.matches("mul.rn.f32").count()
    );
    assert!(plan.validate(program.llvm_ir(), &misplaced).is_err());
}
