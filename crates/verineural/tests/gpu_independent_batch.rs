//! Concurrent independent CUDA Graph replay on owned non-blocking streams.
//! Independent programs share no mutable CUDA buffers and do not aggregate
//! gradients. This test requires explicit NVIDIA hardware acceptance.
use std::time::Instant;

use verineural::{Domain, GpuEngine, GpuError, GpuExecution, NeuralGraph, Shape, Tensor};

fn inputs(value: f32, lanes: usize) -> [Tensor<f32>; 2] {
    [
        Tensor::from_vec(vec![lanes], vec![value; lanes]).unwrap(),
        Tensor::from_vec(vec![lanes], vec![value + 1.; lanes]).unwrap(),
    ]
}

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX; proves independent-stream results equal serial replays"]
fn independent_replay_matches_serial_and_rejects_pending_access() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![32768]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let product = graph.mul(sum, x).unwrap();

    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_owned()),
    );
    let program = engine.compile_outputs(&graph, &[sum, product]).unwrap();
    let directory = std::env::temp_dir()
        .join(format!("vn_parallel_graph_{}", std::process::id()));
    let batches = (0..3usize)
        .map(|i| {
            let values = inputs(i as f32 + 1., 32768);
            vec![(x, values[0].clone()), (y, values[1].clone())]
        })
        .collect::<Vec<_>>();
    let mut sessions = engine
        .prepare_independent(&program, &batches, &directory)
        .unwrap();

    // Batch preparation rejects every member's metadata before touching
    // CUDA. This must also work in non-GPU builds for malformed inventories.
    let empty: [Vec<(verineural::NodeId, Tensor<f32>)>; 0] = [];
    assert!(matches!(
        engine.prepare_independent(&program, &empty, &directory),
        Err(GpuError::InvalidInput {
            reason: "independent GPU batch must not be empty",
            ..
        })
    ));
    // Synchronous reference on the same independent sessions, before
    // any state/inputs change. No accuracy tolerance: bits must agree.
    for execution in &sessions {
        execution.replay().unwrap();
    }
    let reference = sessions
        .iter_mut()
        .map(|gpu| gpu.read_outputs_selected(&[sum, product]).unwrap())
        .collect::<Vec<_>>();

    // A manual outstanding launch must block all ordinary public IO and
    // replay until synchronization publishes the corresponding epoch.
    sessions[0].enqueue_replay().unwrap();
    assert!(matches!(
        sessions[0].read_outputs(),
        Err(GpuError::ReplayPending)
    ));
    assert!(matches!(
        sessions[0].update_input(x, &inputs(1., 32768)[0]),
        Err(GpuError::ReplayPending)
    ));
    assert!(matches!(
        sessions[0].replay(),
        Err(GpuError::ReplayPending)
    ));
    sessions[0].complete_replay().unwrap();
    assert!(matches!(
        sessions[0].complete_replay(),
        Err(GpuError::OutputNotReady)
    ));

    // An already pending member makes the batch reject. Earlier launches
    // are drained, not left permanently in ReplayPending.
    sessions[1].enqueue_replay().unwrap();
    assert!(matches!(
        GpuExecution::replay_independent_batch(&mut sessions),
        Err(GpuError::ReplayPending)
    ));
    sessions[0].read_outputs().unwrap();
    sessions[1].complete_replay().unwrap();

    GpuExecution::replay_independent_batch(&mut sessions).unwrap();
    for (execution, expected) in sessions.iter_mut().zip(&reference) {
        let result = execution.read_outputs_selected(&[sum, product]).unwrap();
        for ((id, actual), (expected_id, want)) in result.iter().zip(expected) {
            assert_eq!(*id, *expected_id);
            assert_eq!(
                actual.values().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.values().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    // A second batch of *different* inputs must not reuse stale results.
    for (i, execution) in sessions.iter_mut().enumerate() {
        let values = inputs(4. + i as f32, 32768);
        execution
            .update_input_subset(&[(x, &values[0]), (y, &values[1])])
            .unwrap();
    }
    GpuExecution::replay_independent_batch(&mut sessions).unwrap();
    for (i, gpu) in sessions.iter_mut().enumerate() {
        let a = 4. + i as f32;
        let want_sum = a + (a + 1.);
        let want_product = want_sum * a;
        let outputs = gpu.read_outputs_selected(&[sum, product]).unwrap();
        for &lane in &[0, 1, 4095, 32767] {
            assert_eq!(outputs[0].1.values()[lane].to_bits(), want_sum.to_bits());
            assert_eq!(outputs[1].1.values()[lane].to_bits(), want_product.to_bits());
        }
    }
}

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX; benchmark is informational, never a speedup guarantee"]
fn independent_streams_report_serial_vs_queued_wall_time() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![131072]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let product = graph.mul(sum, x).unwrap();
    let result = graph.add(product, y).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_owned()),
    );
    let program = engine.compile(&graph, result).unwrap();
    let directory = std::env::temp_dir()
        .join(format!("vn_concurrent_bench_{}", std::process::id()));
    let batches = (0..4usize)
        .map(|i| {
            let values = inputs(i as f32 + 1., 131072);
            vec![(x, values[0].clone()), (y, values[1].clone())]
        })
        .collect::<Vec<_>>();
    let mut sessions = engine
        .prepare_independent(&program, &batches, &directory)
        .unwrap();
    for execution in &sessions {
        execution.replay().unwrap();
    }
    const REPEATS: usize = 32;
    let serial_start = Instant::now();
    for _ in 0..REPEATS {
        for execution in &sessions {
            execution.replay().unwrap();
        }
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.;
    let queued_start = Instant::now();
    for _ in 0..REPEATS {
        GpuExecution::replay_independent_batch(&mut sessions).unwrap();
    }
    let queued_ms = queued_start.elapsed().as_secs_f64() * 1000.;
    println!(
        "independent_replay_serial_ms={serial_ms:.3} queued_ms={queued_ms:.3} executions={} rounds={} speedup={:.3} scope=host_wall_GPU_graphs_only",
        sessions.len(), REPEATS, serial_ms / queued_ms
    );
}
