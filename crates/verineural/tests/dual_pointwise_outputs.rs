//! Bounded multi-output fusion: two independent public F32 Add/Mul roots.
//! Both outputs must retain their original names, buffers and strict RNE bits.
use verineural::{
    Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, NodeId, Shape, Tensor, TranslationKind,
    TranslationPlan,
};

fn fixture() -> (NeuralGraph, [NodeId; 4], [NodeId; 2]) {
    let mut g = NeuralGraph::default();
    let shape = Shape::new(vec![1, 8]).unwrap();
    let a = g.input(shape.clone(), Domain::Any);
    let b = g.input(shape.clone(), Domain::Any);
    let c = g.input(shape.clone(), Domain::Any);
    let d = g.input(shape, Domain::Any);
    let left = g.add(a, b).unwrap();
    let right = g.mul(c, d).unwrap();
    (g, [a, b, c, d], [left, right])
}

#[test]
fn independent_public_outputs_share_one_kernel_without_losing_observability() {
    let (graph, _, [left, right]) = fixture();
    let base = GpuEngine::default();
    let optimized = base
        .clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    assert_eq!(
        base.compile_outputs(&graph, &[left, right])
            .unwrap()
            .kernel_count()
            .unwrap(),
        2
    );
    for outputs in [[left, right], [right, left]] {
        let fused = optimized.compile_outputs(&graph, &outputs).unwrap();
        assert_eq!(fused.outputs(), &outputs);
        assert_eq!(fused.kernel_count().unwrap(), 1);
        assert_eq!(
            fused.fused_dual_outputs().get(&right.index()),
            Some(&left.index())
        );
        let contracts = fused.memory_contracts().unwrap();
        assert_eq!(contracts.len(), 1);
        assert_eq!(contracts[0].reads.len(), 4);
        assert_eq!(contracts[0].writes, 8);
        assert_eq!(contracts[0].additional_writes, vec![(left.index(), 8)]);
        let plan = fused.translation_plan().unwrap();
        assert_eq!(
            plan.kernels[0].kind,
            TranslationKind::DualPointwise {
                earlier: left.index(),
                first_mul: false,
                second_mul: true,
            }
        );
        assert_eq!(plan, TranslationPlan::from_text(&plan.to_text()).unwrap());
        assert_eq!(fused.capability_plan().unwrap().optimizations.len(), 1);
        assert_eq!(fused.llvm_ir().matches("store float %first").count(), 1);
        assert_eq!(fused.llvm_ir().matches("store float %second").count(), 1);
        assert!(!fused.llvm_ir().contains("fma.rn.f32"));
    }
}

#[test]
fn internal_fusion_pairs_live_unselected_nodes_without_erasing_fanout() {
    let (mut graph, [a, b, c, d], [left, right]) = fixture();
    let first_a = graph.add(left, c).unwrap();
    let first_b = graph.mul(left, d).unwrap();
    let second_a = graph.add(right, a).unwrap();
    let second_b = graph.mul(right, b).unwrap();
    let downstream = graph.add(first_a, first_b).unwrap();
    let selected = [downstream, second_a, second_b];
    assert!(!selected.contains(&left) && !selected.contains(&right));
    let old_config = GpuLaunchConfig::default().with_pointwise_fusion();
    let new_config = old_config.with_internal_pointwise_fusion();
    let base = GpuEngine::default()
        .with_launch_config(old_config)
        .compile_outputs(&graph, &selected)
        .unwrap();
    let fused = GpuEngine::default()
        .with_launch_config(new_config)
        .compile_outputs(&graph, &selected)
        .unwrap();
    assert!(base.fused_dual_outputs().get(&right.index()) != Some(&left.index()));
    assert_eq!(
        fused.fused_dual_outputs().get(&right.index()),
        Some(&left.index())
    );
    assert!(fused.kernel_count().unwrap() < base.kernel_count().unwrap());
    assert_eq!(fused.outputs(), &selected);
    let contract = fused.memory_contracts().unwrap();
    assert!(contract.iter().any(|c| {
        c.kernel_node == right.index()
            && c.additional_writes == vec![(left.index(), 8)]
            && c.reads.len() == 4
    }));
    assert!(fused.capability_plan().is_ok());
    assert!(fused.translation_plan().is_ok());
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph and LLVM NVPTX; bitwise parity of internal dual-output fusion"]
fn internal_fusion_matches_unfused_ir_on_gpu() {
    let (mut graph, inputs, [left, right]) = fixture();
    let a = graph.add(left, inputs[2]).unwrap();
    let b = graph.mul(left, inputs[3]).unwrap();
    let c = graph.add(right, inputs[0]).unwrap();
    let d = graph.mul(right, inputs[1]).unwrap();
    let result = graph.add(a, b).unwrap();
    let selected = [result, c, d];
    let inputs = inputs
        .iter()
        .enumerate()
        .map(|(i, &node)| {
            let value = [0.125, -0.0, 1.5, -3., 2., 0.5, 1e-12, 7.];
            (
                node,
                Tensor::from_slice(vec![1, 8], &value.map(|x| x + i as f32)).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let dir = std::env::temp_dir().join(format!("vn_internal_fusion_{}", std::process::id()));
    let base = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let optimized = base
        .clone()
        .with_launch_config(GpuLaunchConfig::default().with_internal_pointwise_fusion());
    let plain = base.compile_outputs(&graph, &selected).unwrap();
    let fused = optimized.compile_outputs(&graph, &selected).unwrap();
    let mut plain_gpu = base.prepare(&plain, &inputs, &dir.join("plain")).unwrap();
    let mut fused_gpu = optimized
        .prepare(&fused, &inputs, &dir.join("fused"))
        .unwrap();
    plain_gpu.replay().unwrap();
    fused_gpu.replay().unwrap();
    let expected = plain_gpu.read_outputs_selected(&selected).unwrap();
    let actual = fused_gpu.read_outputs_selected(&selected).unwrap();
    for ((expected_id, a), (actual_id, b)) in expected.iter().zip(actual.iter()) {
        assert_eq!(expected_id, actual_id);
        for (&x, &y) in a.values().iter().zip(b.values()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

#[test]
fn dependent_or_differently_shaped_outputs_are_not_paired() {
    let (mut graph, [_, _, c, _], [left, right]) = fixture();
    let dependent = graph.mul(left, c).unwrap();
    let opt =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let plan = opt.compile_outputs(&graph, &[left, dependent]).unwrap();
    assert_eq!(plan.kernel_count().unwrap(), 2);
    assert!(plan.fused_dual_outputs().is_empty());

    let source = graph.input(Shape::new(vec![2, 4]).unwrap(), Domain::Any);
    let other = graph.add(source, source).unwrap();
    let different = opt.compile_outputs(&graph, &[right, other]).unwrap();
    assert!(different.fused_dual_outputs().is_empty());
}

#[test]
fn dual_output_fusion_preserves_later_consumers_but_not_earlier_consumers() {
    let (mut graph, [_, _, _, _], [left, right]) = fixture();
    let downstream = graph.add(left, right).unwrap();
    let engine =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let safe = engine
        .compile_outputs(&graph, &[left, right, downstream])
        .unwrap();
    assert_eq!(
        safe.fused_dual_outputs().get(&right.index()),
        Some(&left.index())
    );
    assert_eq!(safe.kernel_count().unwrap(), 2);
    assert_eq!(safe.outputs(), &[left, right, downstream]);

    // A consumer scheduled between the two roots forbids moving the
    // first result's computation into the second root's kernel.
    let (mut graph, [_, _, c, _], [left, _]) = fixture();
    let intervening = graph.mul(left, c).unwrap();
    let later = graph.add(c, c).unwrap();
    let unsafe_pair = engine
        .compile_outputs(&graph, &[left, intervening, later])
        .unwrap();
    assert!(
        unsafe_pair.fused_dual_outputs().get(&later.index()) != Some(&left.index()),
        "moving a producer past an existing consumer violates DAG order"
    );
}

#[test]
fn translation_checks_public_output_bindings_with_reordered_ptx_arithmetic() {
    let (graph, _, outputs) = fixture();
    let program = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion())
        .compile_outputs(&graph, &outputs)
        .unwrap();
    let plan = program.translation_plan().unwrap();
    let node = outputs[1].index();
    // Scheduling the independent multiply before the add is legal. The first
    // store must still contain the add and the second must contain the mul.
    let reordered = format!(
        ".version 7.0\n.target sm_52\n.address_size 64\n\
         .visible .entry vn_node_{node}(\n)\n{{\n\
           mul.rn.f32 %f2, %f3, %f4;\n\
           add.rn.f32 %f1, %f5, %f6;\n\
           st.global.f32 [%rd10], %f1;\n\
           st.global.f32 [%rd11], %f2;\n}}\n"
    );
    // Both scalar PTX store encodings must preserve the original binary32
    // result bits. Reordering independent arithmetic is permitted.
    for store_kind in ["st.global.f32", "st.global.b32"] {
        let reordered = reordered.replace("st.global.f32", store_kind);
        plan.validate(program.llvm_ir(), &reordered).unwrap();

        let swapped = reordered
            .replace("[%rd10], %f1;", "[%rd10], %f2;")
            .replace("[%rd11], %f2;", "[%rd11], %f1;");
        assert_ne!(swapped, reordered);
        assert!(
            plan.validate(program.llvm_ir(), &swapped).is_err(),
            "a correctly counted but incorrectly bound public output is not valid"
        );

        let duplicate = reordered.replace("[%rd11], %f2;", "[%rd11], %f1;");
        assert_ne!(duplicate, reordered);
        assert!(plan.validate(program.llvm_ir(), &duplicate).is_err());

        let changed_width = reordered.replace(store_kind, "st.global.b64");
        assert_ne!(changed_width, reordered);
        assert!(plan.validate(program.llvm_ir(), &changed_width).is_err());
    }
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph and LLVM NVPTX; compares both selected FP32 outputs bitwise"]
fn gpu_dual_outputs_match_independent_unfused_kernels() {
    let (mut graph, inputs, [left, right]) = fixture();
    let downstream = graph.add(left, right).unwrap();
    let data = [
        [0.125, -0.25, 0.5, 1., 3., 4., 8., -0.0],
        [0.5, -1., 2., -3., 0.25, 0.75, -0.5, 1.],
        [2., -3., 0.5, 1.25, 0., -0.0, 2., 4.],
        [1., 3., 0.25, 1e-8, 2., 0.5, -1., 0.75],
    ];
    let bound = inputs
        .iter()
        .zip(&data)
        .map(|(&id, values)| (id, Tensor::from_slice(vec![1, 8], values).unwrap()))
        .collect::<Vec<_>>();
    let base = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let optimized = base
        .clone()
        .with_launch_config(GpuLaunchConfig::default().with_pointwise_fusion());
    let dir = std::env::temp_dir().join(format!("vn_dual_output_{}", std::process::id()));
    let a = base
        .compile_outputs(&graph, &[left, right, downstream])
        .unwrap();
    let b = optimized
        .compile_outputs(&graph, &[left, right, downstream])
        .unwrap();
    let mut plain = base.prepare(&a, &bound, &dir.join("plain")).unwrap();
    let mut fused = optimized.prepare(&b, &bound, &dir.join("fused")).unwrap();
    for step in 0..2 {
        if step == 1 {
            let updated = Tensor::from_slice(vec![1, 8], &data[2]).unwrap();
            plain.update_input_subset(&[(inputs[0], &updated)]).unwrap();
            fused.update_input_subset(&[(inputs[0], &updated)]).unwrap();
        }
        plain.replay().unwrap();
        fused.replay().unwrap();
        for ((want_id, expected), (got_id, actual)) in plain
            .read_outputs_selected(&[right, left, downstream])
            .unwrap()
            .iter()
            .zip(
                fused
                    .read_outputs_selected(&[right, left, downstream])
                    .unwrap()
                    .iter(),
            )
        {
            assert_eq!(want_id, got_id);
            for (&want, &got) in expected.values().iter().zip(actual.values()) {
                assert_eq!(want.to_bits(), got.to_bits());
            }
        }
    }
    let ptx = std::fs::read_to_string(optimized.lower(&b, &dir.join("audit")).unwrap()).unwrap();
    let invalid = ptx.replacen("mul.rn.f32", "mul.f32", 1);
    assert_ne!(invalid, ptx);
    assert!(
        b.translation_plan()
            .unwrap()
            .validate(b.llvm_ir(), &invalid)
            .is_err()
    );
}
