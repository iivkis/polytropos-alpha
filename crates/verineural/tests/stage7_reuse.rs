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
    // Sessions retain the one primary context per device, but own distinct
    // explicit non-blocking streams and release their own retain references.
    assert!(source.contains("cuDevicePrimaryCtxRetain"));
    assert!(source.contains("cuDevicePrimaryCtxRelease"));
    assert!(!source.contains("cuCtxCreate_v2"));
    assert!(!source.contains("cuCtxDestroy_v2"));
}

#[test]
fn staging_buffers_are_recycled_only_after_stream_sync() {
    let source = include_str!("../src/cuda.rs");
    assert!(source.contains("staging_pending: Vec<PinnedBuffer>"));
    assert!(source.contains("staging_available: Vec<PinnedBuffer>"));
    assert!(source.contains("self.staging_available.append(&mut self.staging_pending)"));
    let start = source.find("pub fn flush_input_copies").unwrap();
    let end = source[start..].find("/// Begin capturing").unwrap() + start;
    let flush = &source[start..end];
    assert!(
        flush.find("self.synchronize()?").unwrap()
            < flush.find("self.release_completed_staging()").unwrap()
    );
    assert!(source.contains("self.stream,"));
    assert!(source.contains("CU_STREAM_NON_BLOCKING"));
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
        (
            x,
            verineural::Tensor::new(matrix.clone(), vec![1.0; 6]).unwrap(),
        ),
        (
            y,
            verineural::Tensor::new(matrix.clone(), vec![2.0; 6]).unwrap(),
        ),
        (
            bias,
            verineural::Tensor::new(Shape::new(vec![3]).unwrap(), vec![1.0; 3]).unwrap(),
        ),
    ];
    let wrong_shape = [
        (
            x,
            Tensor::new(Shape::new(vec![6]).unwrap(), vec![1.0; 6]).unwrap(),
        ),
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

#[test]
fn memory_contracts_cover_baseline_and_fused_read_extents() {
    use verineural::GpuMemoryPattern;

    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let output = graph.relu(biased).unwrap();

    let base = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Disabled).unwrap())
        .compile(&graph, output)
        .unwrap();
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap())
        .compile(&graph, output)
        .unwrap();

    let before = base.memory_contracts().unwrap();
    assert_eq!(before.len(), 3);
    assert_eq!(before[0].kernel_node, sum.index());
    assert_eq!(before[0].reads, vec![(x.index(), 6), (y.index(), 6)]);
    assert_eq!(before[0].writes, 6);
    assert_eq!(before[0].pattern, GpuMemoryPattern::Pointwise);
    assert_eq!(before[1].reads, vec![(sum.index(), 6), (bias.index(), 3)]);
    assert_eq!(before[1].pattern, GpuMemoryPattern::BiasAdd { columns: 3 });
    assert_eq!(before[2].reads, vec![(biased.index(), 6)]);

    let after = fused.memory_contracts().unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].kernel_node, output.index());
    assert_eq!(
        after[0].reads,
        vec![(x.index(), 6), (y.index(), 6), (bias.index(), 3)]
    );
    assert_eq!(after[0].writes, 6);
    assert_eq!(after[0].pattern, GpuMemoryPattern::BiasAdd { columns: 3 });

    let source = fused.memory_lean_source().unwrap();
    assert!(source.contains("theorem bias_address_bound"));
    assert!(source.contains("index % 3 < 3"));
    assert!(source.contains("index < 6"));
    assert!(source.contains("IR kernel 5: 3 source tensors"));
}

#[test]
fn matmul_memory_contract_uses_row_major_source_spans() {
    use verineural::GpuMemoryPattern;

    let mut graph = NeuralGraph::default();
    let left = graph.input(Shape::new(vec![2, 4]).unwrap(), Domain::Finite);
    let right = graph.input(Shape::new(vec![4, 3]).unwrap(), Domain::Finite);
    let result = graph.matmul(left, right).unwrap();

    let program = GpuEngine::default().compile(&graph, result).unwrap();
    let contracts = program.memory_contracts().unwrap();
    assert_eq!(contracts.len(), 1);
    assert_eq!(
        contracts[0].reads,
        vec![(left.index(), 8), (right.index(), 12)]
    );
    assert_eq!(contracts[0].writes, 6);
    assert_eq!(
        contracts[0].pattern,
        GpuMemoryPattern::MatMul {
            rows: 2,
            inner: 4,
            columns: 3,
        }
    );
    let source = program.memory_lean_source().unwrap();
    assert!(source.contains("matmul_address_bounds"));
    assert!(source.contains("index / 3) * 4 + j < 2 * 4"));
    assert!(source.contains("j * 3 + index % 3 < 4 * 3"));
}

#[test]
fn memory_contract_does_not_consider_reshape_a_new_allocation() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![2, 3]).unwrap(), Domain::Finite);
    let reshaped = graph.reshape(x, Shape::new(vec![3, 2]).unwrap()).unwrap();
    let y = graph.input(Shape::new(vec![3, 2]).unwrap(), Domain::Finite);
    let output = graph.add(reshaped, y).unwrap();
    let contracts = GpuEngine::default()
        .compile(&graph, output)
        .unwrap()
        .memory_contracts()
        .unwrap();

    assert_eq!(contracts.len(), 1);
    assert_eq!(
        contracts[0].reads,
        vec![(reshaped.index(), 6), (y.index(), 6)]
    );
    assert_eq!(contracts[0].writes, 6);
}

#[test]
fn memory_proof_files_are_pinned_in_stage7_manifest() {
    let source = include_str!("../examples/stage7_fusion.rs");
    assert!(source.contains("\"memory_baseline.lean\""));
    assert!(source.contains("\"memory_fused.lean\""));
    assert!(source.contains("program.memory_lean_source()?"));
    assert!(source.contains("proof.check_file(&root.join(\"memory_baseline.lean\"))"));
    assert!(source.contains("proof.check_file(&root.join(\"memory_fused.lean\"))"));
}

#[test]
fn cuda_driver_loader_checks_nixos_host_driver_without_toolkit_stub() {
    let source = include_str!("../src/cuda.rs");
    assert!(source.contains("c\"libcuda.so.1\""));
    assert!(source.contains("c\"/run/opengl-driver/lib/libcuda.so.1\""));
    assert!(source.contains("dlopen(name.as_ptr(), RTLD_NOW)"));
    // The CUDA toolkit stub is not a replacement for the host driver.
    assert!(!source.contains("lib64/stubs/libcuda.so"));
}
