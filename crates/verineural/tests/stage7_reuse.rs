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

#[test]
fn cuda_sessions_rebind_their_owned_context() {
    let source = include_str!("../src/cuda.rs");
    assert!(source.contains("cuCtxPushCurrent_v2"));
    assert!(source.contains("cuCtxPopCurrent_v2"));
    assert!(source.contains("fn enter_context"));
    assert!(source.matches("self.enter_context()?;").count() >= 7);
    assert!(source.contains("Keep this session context current for the entire capture sequence."));
}

#[test]
fn stage7_replay_inputs_are_validated_before_cuda_setup() {
    use std::path::Path;
    use verineural::{GpuError, Tensor};

    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix.clone(), Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let output = graph.relu(biased).unwrap();

    let engine = GpuEngine::default();
    let plan = engine.compile(&graph, output).unwrap();
    let valid = [
        (x, verineural::Tensor::new(matrix.clone(), vec![1.0; 6]).unwrap()),
        (y, verineural::Tensor::new(matrix.clone(), vec![2.0; 6]).unwrap()),
        (bias, verineural::Tensor::new(Shape::new(vec![3]).unwrap(), vec![1.0; 3]).unwrap()),
    ];
    let wrong_shape = [
        (x, Tensor::new(Shape::new(vec![6]).unwrap(), vec![1.0; 6]).unwrap()),
        valid[1].clone(),
        valid[2].clone(),
    ];
    // The path does not need to exist: input checks precede LLVM or CUDA.
    let output_dir = Path::new("target/verineural/should-not-run-stage7");
    assert!(matches!(
        engine.prepare(&plan, &wrong_shape, output_dir),
        Err(GpuError::InvalidGraph(_))
    ));

    let mut nonfinite = valid.clone();
    let mut values = nonfinite[0].1.values().to_vec();
    values[0] = f32::NAN;
    nonfinite[0].1 = Tensor::new(matrix, values).unwrap();
    assert!(matches!(
        engine.prepare(&plan, &nonfinite, output_dir),
        Err(GpuError::InvalidGraph(_))
    ));
    assert!(matches!(
        engine.prepare(&plan, &valid[1..], output_dir),
        Err(GpuError::MissingInput { .. })
    ));
}
