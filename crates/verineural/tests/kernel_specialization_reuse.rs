//! NVPTX function interning must not alter graph scheduling or the translation
//! obligations for every logical producer. No NVIDIA driver needed here.
use std::collections::BTreeSet;
use verineural::{Domain, GpuEngine, NeuralGraph, Shape, TranslationPlan};

#[test]
fn identical_templates_share_entrypoints_without_removing_graph_nodes() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let a = graph.add(x, y).unwrap();
    let b = graph.add(a, y).unwrap();
    let c = graph.add(b, x).unwrap();
    let d = graph.mul(a, y).unwrap();
    let e = graph.mul(b, x).unwrap();
    let program = GpuEngine::default()
        .compile_outputs(&graph, &[c, d, e])
        .unwrap();
    let plan = program.translation_plan().unwrap();
    assert_eq!(program.kernel_count().unwrap(), 5);
    assert_eq!(program.unique_kernel_count(), 2);
    assert_eq!(plan.kernels.len(), 5);
    assert_eq!(program.kernel_entry(a.index()), Some(a.index()));
    assert_eq!(program.kernel_entry(b.index()), Some(a.index()));
    assert_eq!(program.kernel_entry(c.index()), Some(a.index()));
    assert_eq!(program.kernel_entry(d.index()), Some(d.index()));
    assert_eq!(program.kernel_entry(e.index()), Some(d.index()));
    assert_eq!(program.kernel_entry(x.index()), None);
    assert_eq!(
        program.llvm_ir().matches("define ptx_kernel void").count(),
        2
    );
    assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
    let owners = plan
        .kernels
        .iter()
        .map(|k| k.entry_node)
        .collect::<BTreeSet<_>>();
    assert_eq!(owners, BTreeSet::from([a.index(), d.index()]));

    // Equivalent PTX opcodes, but different per-node LLVM and PTX bindings,
    // must still fail validation if an entrypoint is omitted or substituted.
    let ptx = format!(
        ".version 7.0\n.target sm_52\n.address_size 64\n\
         .visible .entry vn_node_{}(\n)\n{{\nadd.rn.f32 %f1, %f2, %f3;\n}}\n\
         .visible .entry vn_node_{}(\n)\n{{\nmul.rn.f32 %f1, %f2, %f3;\n}}\n",
        a.index(),
        d.index()
    );
    let report = plan.validate(program.llvm_ir(), &ptx).unwrap();
    assert_eq!(report.kernel_count, 5);
    assert_eq!(report.rounded_adds, 3);
    assert_eq!(report.rounded_muls, 2);
    assert!(plan
        .validate(
            program.llvm_ir(),
            &ptx.replace(&format!("vn_node_{}(", d.index()), "vn_node_999(")
        )
        .is_err());
    let mut invalid_plan = plan.clone();
    invalid_plan.kernels[1].entry_node = d.index(); // future entry
    assert!(TranslationPlan::from_text(&invalid_plan.to_text()).is_err());
    invalid_plan = plan;
    invalid_plan.kernels[3].entry_node = a.index(); // wrong operation
    assert!(invalid_plan.validate(program.llvm_ir(), &ptx).is_err());
}

#[test]
fn structural_concat_reuses_only_identical_addressing_templates() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![1, 3]).unwrap(), Domain::Finite);
    let y = graph.input(Shape::new(vec![1, 3]).unwrap(), Domain::Finite);
    let z = graph.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let first = graph.concat(1, &[x, y]).unwrap();
    let second = graph.concat(1, &[x, z]).unwrap();
    let different_offset = graph.concat(1, &[z, x]).unwrap();
    let program = GpuEngine::default()
        .compile_outputs(&graph, &[first, second, different_offset])
        .unwrap();
    assert_eq!(program.kernel_count().unwrap(), 3);
    assert_eq!(program.unique_kernel_count(), 2);
    assert_eq!(program.kernel_entry(first.index()), Some(first.index()));
    assert_eq!(program.kernel_entry(second.index()), Some(first.index()));
    assert_eq!(
        program.kernel_entry(different_offset.index()),
        Some(different_offset.index())
    );
    let plan = program.translation_plan().unwrap();
    assert_eq!(plan.kernels[0].writes, 6);
    assert_eq!(plan.kernels[1].writes, 7);
    assert_eq!(plan.kernels[2].writes, 7);
    assert_eq!(TranslationPlan::from_text(&plan.to_text()).unwrap(), plan);
}

#[test]
fn specialization_key_preserves_shape_dependent_constants_and_abi() {
    let mut graph = NeuralGraph::default();
    let lhs1 = graph.input(Shape::new(vec![2, 3]).unwrap(), Domain::Finite);
    let rhs1 = graph.input(Shape::new(vec![3, 2]).unwrap(), Domain::Finite);
    let lhs2 = graph.input(Shape::new(vec![2, 4]).unwrap(), Domain::Finite);
    let rhs2 = graph.input(Shape::new(vec![4, 2]).unwrap(), Domain::Finite);
    let a = graph.matmul(lhs1, rhs1).unwrap();
    let b = graph.matmul(lhs1, rhs1).unwrap();
    let c = graph.matmul(lhs2, rhs2).unwrap();
    let plan = GpuEngine::default()
        .compile_outputs(&graph, &[a, b, c])
        .unwrap();
    assert_eq!(plan.kernel_count().unwrap(), 3);
    assert_eq!(plan.unique_kernel_count(), 2);
    assert_eq!(plan.kernel_entry(b.index()), Some(a.index()));
    assert_eq!(plan.kernel_entry(c.index()), Some(c.index()));
    assert!(plan.llvm_ir().contains("icmp ult i32 %j, 3"));
    assert!(plan.llvm_ir().contains("icmp ult i32 %j, 4"));
}

#[test]
fn identical_code_with_different_output_lengths_is_reusable() {
    let mut graph = NeuralGraph::default();
    let left = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let right = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let bigger = graph.input(Shape::new(vec![8]).unwrap(), Domain::Finite);
    let add4 = graph.add(left, right).unwrap();
    let add8 = graph.add(bigger, bigger).unwrap();
    let program = GpuEngine::default()
        .compile_outputs(&graph, &[add4, add8])
        .unwrap();
    assert_eq!(program.kernel_count().unwrap(), 2);
    assert_eq!(program.unique_kernel_count(), 1);
    assert_eq!(program.kernel_entry(add8.index()), Some(add4.index()));
    assert_eq!(program.translation_plan().unwrap().kernels[1].writes, 8);
}

#[test]
#[ignore = "requires NVIDIA CUDA Graph and LLVM NVPTX; compares reused entrypoints bitwise"]
fn cuda_reused_functions_preserve_independent_outputs() {
    use verineural::Tensor;
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let y = graph.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let a = graph.add(x, y).unwrap();
    let b = graph.add(a, y).unwrap();
    let c = graph.mul(b, x).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_owned()),
    );
    let program = engine.compile_outputs(&graph, &[a, b, c]).unwrap();
    assert_eq!(program.unique_kernel_count(), 2);
    let inputs = [
        (x, Tensor::from_slice(vec![4], &[1., -2., -0., 4.]).unwrap()),
        (y, Tensor::from_slice(vec![4], &[2., 3., 5., -1.]).unwrap()),
    ];
    let dir = std::env::temp_dir().join(format!(
        "verineural_kernel_specialization_{}",
        std::process::id()
    ));
    let mut execution = engine.prepare(&program, &inputs, &dir).unwrap();
    execution.replay().unwrap();
    let outputs = execution.read_outputs().unwrap();
    let expected_x = [1_f32, -2., -0., 4.];
    let expected_y = [2_f32, 3., 5., -1.];
    for lane in 0..4 {
        let sum = expected_x[lane] + expected_y[lane];
        let twice = sum + expected_y[lane];
        let product = twice * expected_x[lane];
        for (tensor, expected) in outputs.iter().zip([sum, twice, product]) {
            assert_eq!(tensor.1.values()[lane].to_bits(), expected.to_bits());
        }
    }
}
