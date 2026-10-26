//! A shared GPU program is a union of reachable Semantic IR nodes, not one
//! separately compiled PTX graph per requested result.
use verineural::{
    Domain, GpuEngine, GpuError, GpuLaunchConfig, NeuralGraph, NodeId, ReluFusion, Shape,
};

fn make() -> (NeuralGraph, [NodeId; 2], [NodeId; 4]) {
    let mut g = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let x = g.input(shape.clone(), Domain::Finite);
    let y = g.input(shape, Domain::Finite);
    let sum = g.add(x, y).unwrap();
    let left = g.relu(sum).unwrap();
    let right = g.mul(sum, y).unwrap();
    let final_sum = g.add(left, right).unwrap();
    (g, [x, y], [sum, left, right, final_sum])
}

#[test]
fn shared_gpu_program_emits_common_kernel_only_once_and_orders_results() {
    let (graph, sources, outputs) = make();
    let engine = GpuEngine::default();
    let program = engine.compile_outputs(&graph, &outputs).unwrap();
    assert_eq!(program.outputs(), &outputs);
    assert_eq!(program.output(), outputs[0]);
    assert_eq!(program.source_inputs(), &sources);
    assert_eq!(program.kernel_count().unwrap(), 4);
    // Two Add producers now share one LLVM/PTX entrypoint, but remain
    // separate CUDA Graph nodes with independent buffers and arguments.
    assert_eq!(program.unique_kernel_count(), 3);
    assert_eq!(
        program
            .llvm_ir()
            .matches("define ptx_kernel void @vn_node_")
            .count(),
        program.unique_kernel_count()
    );
    let root_nodes = program
        .translation_plan()
        .unwrap()
        .kernels
        .iter()
        .map(|k| k.node)
        .collect::<Vec<_>>();
    assert_eq!(
        root_nodes,
        outputs.iter().map(|id| id.index()).collect::<Vec<_>>()
    );

    // Without shared compilation, all four root programs repeat producers.
    let independent_kernels: usize = outputs
        .iter()
        .map(|&id| engine.compile(&graph, id).unwrap().kernel_count().unwrap())
        .sum();
    assert!(independent_kernels > program.kernel_count().unwrap());

    let reversed = engine
        .compile_outputs(&graph, &[outputs[3], outputs[0]])
        .unwrap();
    assert_eq!(reversed.outputs(), &[outputs[3], outputs[0]]);
    assert_eq!(reversed.kernel_count().unwrap(), 4);
}

#[test]
fn shared_gpu_fusion_preserves_selected_intermediate_outputs() {
    let (graph, _sources, outputs) = make();
    let engine = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(128, ReluFusion::Experimental).unwrap());
    let observable = engine
        .compile_outputs(&graph, &[outputs[0], outputs[1]])
        .unwrap();
    assert_eq!(observable.kernel_count().unwrap(), 2);
    assert!(!observable.fused_relus().contains_key(&outputs[1].index()));

    // With no separately observable producer, the exclusive Add/Relu pair
    // may fuse, subject to existing experimental FP32 fusion assumptions.
    let unobserved = engine.compile_outputs(&graph, &[outputs[1]]).unwrap();
    assert_eq!(unobserved.kernel_count().unwrap(), 1);
    assert_eq!(
        unobserved.fused_relus().get(&outputs[1].index()),
        Some(&outputs[0].index())
    );
}

#[test]
fn shared_gpu_compilation_rejects_empty_duplicate_foreign_and_unsupported_roots() {
    let (graph, _sources, outputs) = make();
    let engine = GpuEngine::default();
    assert!(matches!(
        engine.compile_outputs(&graph, &[]),
        Err(GpuError::InvalidInput {
            reason: "multi-output GPU program requires at least one output",
            ..
        })
    ));
    assert!(matches!(
        engine.compile_outputs(&graph, &[outputs[0], outputs[0]]),
        Err(GpuError::InvalidInput {
            reason: "multi-output GPU program has duplicate output",
            ..
        })
    ));
    let mut foreign = NeuralGraph::default();
    let external = foreign.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    assert!(matches!(
        engine.compile_outputs(&graph, &[outputs[0], external]),
        Err(GpuError::InvalidGraph(_))
    ));

    let mut next = NeuralGraph::default();
    let input = next.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let bad = next.positive_mask(input).unwrap();
    assert!(matches!(
        engine.compile_outputs(&next, &[input, bad]),
        Err(GpuError::UnsupportedOperation {
            operation: "positive_mask",
            ..
        })
    ));
}

#[test]
fn shared_gpu_keeps_reshape_alias_and_live_input_union() {
    let mut g = NeuralGraph::default();
    let x = g.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let y = g.input(Shape::new(vec![4]).unwrap(), Domain::Finite);
    let reshaped = g.reshape(x, Shape::new(vec![4]).unwrap()).unwrap();
    let sum = g.add(reshaped, y).unwrap();
    let plan = GpuEngine::default()
        .compile_outputs(&g, &[reshaped, sum])
        .unwrap();
    assert_eq!(plan.source_inputs(), &[x, y]);
    assert_eq!(plan.kernel_count().unwrap(), 1);
    assert_eq!(plan.outputs(), &[reshaped, sum]);
    assert_eq!(plan.translation_plan().unwrap().kernels.len(), 1);
}

#[test]
fn shared_gpu_supports_standalone_inputs_without_any_kernel() {
    let mut g = NeuralGraph::default();
    let shape = Shape::new(vec![2]).unwrap();
    let a = g.input(shape.clone(), Domain::Finite);
    let b = g.input(shape, Domain::Finite);
    let program = GpuEngine::default().compile_outputs(&g, &[a, b]).unwrap();
    assert_eq!(program.kernel_count().unwrap(), 0);
    assert_eq!(program.outputs(), &[a, b]);
}
