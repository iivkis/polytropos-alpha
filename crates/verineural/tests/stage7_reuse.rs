use verineural::{Domain, GpuEngine, GpuLaunchConfig, NeuralGraph, ReluFusion, Shape};

#[test]
fn stage7_chain_fusion_removes_two_kernels() {
    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let result = graph.relu(biased).unwrap();

    let program = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, result)
        .unwrap();

    assert_eq!(program.removed_kernel_count(), 2);
    assert_eq!(
        program.llvm_ir().matches("define ptx_kernel void").count(),
        1
    );
    assert_eq!(
        program
            .fused_pointwise_chains()
            .get(&result.index())
            .map(Vec::as_slice),
        Some(&[sum.index(), biased.index()][..])
    );
    let proof = program.fusion_lean_source().unwrap();
    assert!(proof.contains("theorem relu_add_bias_fusion"));
    assert!(proof.contains("theorem relu_mul_bias_fusion"));
}

#[test]
fn stage7_shared_inner_node_blocks_chain_fusion() {
    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let positive = graph.relu(biased).unwrap();
    let output = graph.add(sum, positive).unwrap();

    let program = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, output)
        .unwrap();

    assert!(program.fused_pointwise_chains().is_empty());
    assert_eq!(program.removed_kernel_count(), 1);
}
