//! Real CUDA Graph subset H2D and D2H batch parity.
//! One explicit non-blocking stream, queued pinned transfers and one barrier.
use verineural::{Domain, GpuEngine, GpuError, NeuralGraph, Shape, Tensor};

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX; checks batch parity and fail-before-DMA validation"]
fn batched_subset_transfers_match_single_updates_and_reads() {
    let mut g = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let x = g.input(shape.clone(), Domain::Finite);
    let y = g.input(shape, Domain::Finite);
    let sum = g.add(x, y).unwrap();
    let product = g.mul(sum, y).unwrap();
    let arch = std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_string());
    let engine = GpuEngine::new("llc", arch);
    let program = engine.compile_outputs(&g, &[sum, product]).unwrap();
    let initial = [
        (x, Tensor::from_slice(vec![4], &[1., -2., 0., 8.]).unwrap()),
        (y, Tensor::from_slice(vec![4], &[3., 4., -0.5, 2.]).unwrap()),
    ];
    let dir = std::env::temp_dir().join(format!("vn_batch_transfers_{}", std::process::id()));
    let mut gpu = engine.prepare(&program, &initial, &dir).unwrap();
    gpu.replay().unwrap();

    let old = gpu.read_outputs().unwrap();
    let (reversed, profile) = gpu.read_outputs_selected_profiled(&[product, sum]).unwrap();
    assert_eq!(profile.transfers, 2);
    assert_eq!(profile.bytes, 2 * 4 * std::mem::size_of::<f32>());
    let (packed, packed_profile) = gpu
        .read_outputs_selected_packed_profiled(&[product, sum])
        .unwrap();
    assert_eq!(packed_profile.transfers, 1);
    assert_eq!(packed_profile.bytes, profile.bytes);
    assert_eq!(packed.len(), reversed.len());
    for ((packed_id, packed_tensor), (id, tensor)) in packed.iter().zip(&reversed) {
        assert_eq!(packed_id, id);
        for (&x, &y) in packed_tensor.values().iter().zip(tensor.values()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
    let (empty, empty_profile) = gpu.read_outputs_selected_profiled(&[]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty_profile.bytes, 0);
    assert_eq!(empty_profile.transfers, 0);
    let (packed_empty, packed_empty_profile) =
        gpu.read_outputs_selected_packed_profiled(&[]).unwrap();
    assert!(packed_empty.is_empty());
    assert_eq!(packed_empty_profile.transfers, 0);
    assert_eq!(old[0].0, sum);
    assert_eq!(old[1].0, product);
    assert_eq!(reversed[0].0, product);
    assert_eq!(reversed[1].0, sum);
    for (left, right) in old.iter().zip(reversed.iter().rev()) {
        assert_eq!(left.0, right.0);
        for (a, b) in left.1.values().iter().zip(right.1.values()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    assert!(matches!(
        gpu.read_outputs_selected_packed_profiled(&[sum, sum]),
        Err(GpuError::InvalidInput { .. })
    ));
    assert!(matches!(
        gpu.read_outputs_selected_profiled(&[sum, sum]),
        Err(GpuError::InvalidInput {
            reason: "duplicate selected read output",
            ..
        })
    ));
    assert!(matches!(
        gpu.read_outputs_selected(&[x]),
        Err(GpuError::InvalidInput {
            reason: "requested output is not selected",
            ..
        })
    ));

    let new_x = Tensor::from_slice(vec![4], &[-2., 1., 4., 0.]).unwrap();
    let new_y = Tensor::from_slice(vec![4], &[3., -4., 0.25, 2.]).unwrap();
    let invalid = Tensor::from_slice(vec![2], &[5., 6.]).unwrap();
    assert!(matches!(
        gpu.update_input_subset(&[(x, &new_x), (x, &new_y)]),
        Err(GpuError::InvalidInput {
            reason: "duplicate partial input update",
            ..
        })
    ));
    assert!(
        gpu.update_input_subset(&[(x, &new_x), (y, &invalid)])
            .is_err()
    );
    // A rejected batch must not invalidate the prior outputs.
    let unmodified = gpu.read_outputs_selected(&[sum, product]).unwrap();
    for (a, b) in unmodified.iter().zip(&old) {
        for (v, w) in a.1.values().iter().zip(b.1.values()) {
            assert_eq!(v.to_bits(), w.to_bits());
        }
    }
    gpu.update_input_subset(&[(x, &new_x), (y, &new_y)])
        .unwrap();
    gpu.replay().unwrap();
    let actual = gpu.read_outputs_selected(&[sum, product]).unwrap();
    for i in 0..4 {
        let sum_expected = new_x.values()[i] + new_y.values()[i];
        let product_expected = sum_expected * new_y.values()[i];
        assert_eq!(actual[0].1.values()[i].to_bits(), sum_expected.to_bits());
        assert_eq!(
            actual[1].1.values()[i].to_bits(),
            product_expected.to_bits()
        );
    }
    // Exercise the cached packed graph against *new* GPU output values:
    // matching pointer descriptors must not return the previous epoch.
    let (packed_updated, cached_profile) = gpu
        .read_outputs_selected_packed_profiled(&[product, sum])
        .unwrap();
    assert_eq!(cached_profile.transfers, 1);
    for ((packed_id, packed_tensor), (regular_id, regular_tensor)) in
        packed_updated.iter().zip(actual.iter().rev())
    {
        assert_eq!(packed_id, regular_id);
        for (&packed_value, &regular_value) in
            packed_tensor.values().iter().zip(regular_tensor.values())
        {
            assert_eq!(packed_value.to_bits(), regular_value.to_bits());
        }
    }
    // Packed H2D scatter must update every source bit-for-bit, and
    // a cached scatter graph must accept subsequent input epochs.
    let packed_x = Tensor::from_slice(vec![4], &[6., -1., -3., 2.]).unwrap();
    let packed_y = Tensor::from_slice(vec![4], &[2., 0.25, -4., -1.]).unwrap();
    assert!(matches!(
        gpu.update_input_subset_packed(&[(x, &packed_x), (x, &packed_y)]),
        Err(GpuError::InvalidInput {
            reason: "duplicate partial input update",
            ..
        })
    ));
    gpu.update_input_subset_packed(&[(x, &packed_x), (y, &packed_y)])
        .unwrap();
    gpu.replay().unwrap();
    let packed_values = gpu.read_outputs_selected(&[sum, product]).unwrap();
    for i in 0..4 {
        let expected_sum = packed_x.values()[i] + packed_y.values()[i];
        let expected_product = expected_sum * packed_y.values()[i];
        assert_eq!(
            packed_values[0].1.values()[i].to_bits(),
            expected_sum.to_bits()
        );
        assert_eq!(
            packed_values[1].1.values()[i].to_bits(),
            expected_product.to_bits()
        );
    }
    gpu.update_input_subset_packed(&[(x, &new_x), (y, &new_y)])
        .unwrap();
    gpu.replay().unwrap();
    let repeated = gpu.read_outputs_selected(&[sum, product]).unwrap();
    for (now, before) in repeated.iter().zip(&actual) {
        assert_eq!(now.0, before.0);
        for (&left, &right) in now.1.values().iter().zip(before.1.values()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    // Existing pointwise single-update API still yields the same results.
    gpu.update_input(x, &new_x).unwrap();
    gpu.update_input(y, &new_y).unwrap();
    gpu.replay().unwrap();
    let individual = gpu.read_outputs().unwrap();
    for (a, b) in actual.iter().zip(&individual) {
        for (v, w) in a.1.values().iter().zip(b.1.values()) {
            assert_eq!(v.to_bits(), w.to_bits());
        }
    }
}

/// Compare two independent CUDA sessions using the same Semantic IR.
#[test]
#[ignore = "requires NVIDIA + LLVM NVPTX; verifies slab boundaries, replay, packed H2D/D2H"]
fn slab_allocation_preserves_cuda_graph_outputs_and_packed_io() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let product = graph.mul(sum, y).unwrap();
    let result = graph.add(product, x).unwrap();
    let arch = std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".to_string());
    let ordinary_engine = GpuEngine::new("llc", arch.clone());
    let slab_engine = GpuEngine::new("llc", arch).with_allocation_slab(true);
    let program = ordinary_engine
        .compile_outputs(&graph, &[sum, product, result])
        .unwrap();
    let source = vec![
        (x, Tensor::from_slice(vec![4], &[1., 0., -3., 9.]).unwrap()),
        (
            y,
            Tensor::from_slice(vec![4], &[2., -0., -4., 0.25]).unwrap(),
        ),
    ];
    let root = std::env::temp_dir().join(format!("vn_allocation_slab_{}", std::process::id()));
    let mut normal = ordinary_engine
        .prepare(&program, &source, &root.join("ordinary"))
        .unwrap();
    let mut slab = slab_engine
        .prepare(&program, &source, &root.join("slab"))
        .unwrap();
    for epoch in 0..3 {
        if epoch != 0 {
            let x_values =
                Tensor::from_slice(vec![4], &[epoch as f32, -0., 0.125, -(epoch as f32)]).unwrap();
            let y_values = Tensor::from_slice(vec![4], &[0.5, 7., -1., epoch as f32]).unwrap();
            normal
                .update_input_subset(&[(x, &x_values), (y, &y_values)])
                .unwrap();
            slab.update_input_subset_packed(&[(x, &x_values), (y, &y_values)])
                .unwrap();
        }
        normal.replay().unwrap();
        slab.replay().unwrap();
        let expected = normal
            .read_outputs_selected(&[result, sum, product])
            .unwrap();
        let (actual, profile) = slab
            .read_outputs_selected_packed_profiled(&[result, sum, product])
            .unwrap();
        assert_eq!(profile.transfers, 1);
        assert_eq!(expected.len(), actual.len());
        for ((expected_id, expected_tensor), (actual_id, actual_tensor)) in
            expected.iter().zip(&actual)
        {
            assert_eq!(expected_id, actual_id);
            assert_eq!(expected_tensor.shape(), actual_tensor.shape());
            for (&left, &right) in expected_tensor.values().iter().zip(actual_tensor.values()) {
                assert_eq!(
                    left.to_bits(),
                    right.to_bits(),
                    "epoch={epoch}, output={}",
                    expected_id.index()
                );
            }
        }
    }
}
