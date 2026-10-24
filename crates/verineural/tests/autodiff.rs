//! Reusable reverse-mode and AdamW acceptance, independent of neural architectures.
use std::collections::BTreeMap;
use verineural::{
    AdamW, Domain, GpuEngine, NeuralGraph, PointwiseExpr, PointwiseSpec, Shape, Tensor,
    differentiate, differentiate_with_seeds,
};

fn fixture() -> (
    NeuralGraph,
    verineural::NodeId,
    verineural::NodeId,
    verineural::NodeId,
    verineural::NodeId,
) {
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![1, 3]).unwrap(), Domain::Finite);
    let w = graph.input(Shape::new(vec![3, 2]).unwrap(), Domain::Finite);
    let bias = graph.input(Shape::new(vec![1, 2]).unwrap(), Domain::Finite);
    let z = graph.matmul(x, w).unwrap();
    let z = graph.add(z, bias).unwrap();
    let root = graph.mul(z, z).unwrap();
    (graph, x, w, bias, root)
}

fn oracle(x: &[f32; 3], w: &[f32; 6], bias: &[f32; 2], seed: &[f32; 2]) -> f32 {
    (0..2)
        .map(|col| {
            let z = (0..3).map(|row| x[row] * w[row * 2 + col]).sum::<f32>() + bias[col];
            z * z * seed[col]
        })
        .sum()
}

#[test]
fn fanout_gradient_and_finite_difference_reference() {
    let (g, x, w, bias, root) = fixture();
    let derived = differentiate(&g, root, &[w, bias, x]).unwrap();
    assert_eq!(derived.gradients().len(), 3);
    let seed = [0.25, -1.2];
    let a = [0.3, -0.8, 1.1];
    let b = [0.2, 0.1, -0.25, 0.7, -0.4, 0.09];
    let c = [-0.1, 0.3];
    let z: [f32; 2] =
        std::array::from_fn(|j| (0..3).map(|i| a[i] * b[i * 2 + j]).sum::<f32>() + c[j]);
    for k in 0..6 {
        let mut plus = b;
        let mut minus = b;
        plus[k] += 1e-3;
        minus[k] -= 1e-3;
        let numeric = (oracle(&a, &plus, &c, &seed) - oracle(&a, &minus, &c, &seed)) / 2e-3;
        let expected = 2. * z[k % 2] * seed[k % 2] * a[k / 2];
        assert!(
            (numeric - expected).abs() < 5e-4,
            "weight={k} numeric={numeric} analytic={expected}"
        );
    }
    assert!(derived.graph().nodes().len() > g.nodes().len());
    assert_eq!(
        derived
            .graph()
            .node(derived.gradient_of(w).unwrap())
            .unwrap()
            .shape
            .dims(),
        &[3, 2]
    );
    assert_eq!(
        derived
            .graph()
            .node(derived.gradient_of(bias).unwrap())
            .unwrap()
            .shape
            .dims(),
        &[1, 2]
    );
}

#[test]
fn differentiation_is_explicit_and_fail_closed() {
    let mut g = NeuralGraph::default();
    let source = g.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let y = g.relu(source).unwrap();
    assert!(differentiate(&g, y, &[source]).is_err());
    let spoof = PointwiseSpec::new("verineural.div", "1", 1, PointwiseExpr::input(0).exp())
        .unwrap()
        .apply(&mut g, &[source])
        .unwrap();
    assert!(differentiate(&g, spoof, &[source]).is_err());
    assert!(differentiate(&g, source, &[]).is_err());
    assert!(differentiate(&g, source, &[source, source]).is_err());
}

#[test]
fn reverse_structural_and_broadcast_jacobians_have_source_shapes() {
    let mut g = NeuralGraph::default();
    let x = g.input(Shape::new(vec![1, 4]).unwrap(), Domain::Finite);
    let alpha = g.input(Shape::new(vec![1, 1]).unwrap(), Domain::Finite);
    let left = g.slice_axis(x, 1, 1, 2).unwrap();
    let right = g.slice_axis(x, 1, 0, 1).unwrap();
    let mid = g.concat(1, &[right, left]).unwrap();
    let scaled = g
        .broadcast_to(alpha, Shape::new(vec![1, 3]).unwrap())
        .unwrap();
    let product = g.mul(mid, scaled).unwrap();
    let out = g.exp(product).unwrap();
    let diff = differentiate(&g, out, &[x, alpha]).unwrap();
    assert_eq!(
        diff.graph()
            .node(diff.gradient_of(x).unwrap())
            .unwrap()
            .shape
            .dims(),
        &[1, 4]
    );
    assert_eq!(
        diff.graph()
            .node(diff.gradient_of(alpha).unwrap())
            .unwrap()
            .shape
            .dims(),
        &[1, 1]
    );
    assert!(!diff.constants().is_empty());
}

#[test]
fn adamw_validates_before_any_updates_and_keeps_optimizer_state() {
    let mut optimizer = AdamW::new(0.01, 0.01).unwrap();
    let mut parameters = BTreeMap::from([(
        "weight".to_string(),
        Tensor::from_vec(vec![1, 2], vec![1., -2.]).unwrap(),
    )]);
    let mut gradients = BTreeMap::from([(
        "weight".to_string(),
        Tensor::from_vec(vec![1, 2], vec![0.5, -0.5]).unwrap(),
    )]);
    optimizer.step(&mut parameters, &gradients).unwrap();
    assert_eq!(optimizer.steps(), 1);
    assert!(parameters["weight"].values()[0] < 1.);
    let old = parameters.clone();
    gradients.insert(
        "extra".into(),
        Tensor::from_vec(vec![1, 2], vec![1., 1.]).unwrap(),
    );
    assert!(optimizer.step(&mut parameters, &gradients).is_err());
    assert_eq!(parameters, old);
    assert_eq!(optimizer.steps(), 1);
    assert!(AdamW::new(f32::NAN, 0.).is_err());
}

#[test]
#[ignore = "requires LLVM NVPTX and NVIDIA CUDA"]
fn gpu_reverse_graph_matches_scalar_matmul_finite_difference() {
    let (g, x, w, bias, root) = fixture();
    let reverse = differentiate(&g, root, &[w, bias]).unwrap();
    let gradient_w = reverse.gradient_of(w).unwrap();
    let gradient_b = reverse.gradient_of(bias).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine
        .compile_outputs(reverse.graph(), &[root, gradient_w, gradient_b])
        .unwrap();
    let a = [0.3, -0.8, 1.1];
    let b = [0.2, 0.1, -0.25, 0.7, -0.4, 0.09];
    let c = [-0.1, 0.3];
    let seed = [0.25, -1.2];
    let values = reverse
        .bindings(
            &[
                (x, Tensor::from_slice(vec![1, 3], &a).unwrap()),
                (w, Tensor::from_slice(vec![3, 2], &b).unwrap()),
                (bias, Tensor::from_slice(vec![1, 2], &c).unwrap()),
            ],
            Tensor::from_slice(vec![1, 2], &seed).unwrap(),
        )
        .unwrap();
    let active = values
        .into_iter()
        .filter(|(id, _)| program.source_inputs().contains(id))
        .collect::<Vec<_>>();
    let path = std::env::temp_dir().join(format!("vn_reverse_gpu_{}", std::process::id()));
    let outputs = engine.execute_outputs(&program, &active, &path).unwrap();
    let z: [f32; 2] =
        std::array::from_fn(|j| (0..3).map(|i| a[i] * b[i * 2 + j]).sum::<f32>() + c[j]);
    for (row, &a_row) in a.iter().enumerate() {
        for col in 0..2 {
            let actual = outputs[1].1.values()[row * 2 + col];
            let expected = 2. * z[col] * seed[col] * a_row;
            assert!(
                (actual - expected).abs() < 2e-5,
                "{row},{col}: {actual} vs {expected}"
            );
        }
    }
    for col in 0..2 {
        assert!((outputs[2].1.values()[col] - 2. * z[col] * seed[col]).abs() < 2e-5);
    }
}

#[test]
fn graph_provided_multi_output_vjp_rejects_bad_seeds() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 1]).unwrap();
    let weight = graph.input(shape.clone(), Domain::Finite);
    let input = graph.input(shape.clone(), Domain::Finite);
    let square = graph.mul(weight, weight).unwrap();
    let output = graph.add(square, input).unwrap();
    let first_seed = graph.mul(square, input).unwrap();
    let second_seed = graph.sub(input, weight).unwrap();
    let reverse = differentiate_with_seeds(
        &graph,
        &[(square, first_seed), (output, second_seed)],
        &[weight],
    )
    .unwrap();
    assert_eq!(
        reverse
            .graph()
            .node(reverse.gradient_of(weight).unwrap())
            .unwrap()
            .shape,
        shape
    );
    assert!(
        reverse
            .bindings(&[], Tensor::from_slice(vec![1, 1], &[1.]).unwrap())
            .is_err()
    );
    assert!(differentiate_with_seeds(&graph, &[], &[weight]).is_err());
    assert!(
        differentiate_with_seeds(&graph, &[(square, input), (square, input)], &[weight]).is_err()
    );
}

#[test]
#[ignore = "requires NVIDIA GPU and LLVM NVPTX for graph-provided seed parity"]
fn gpu_graph_provided_multi_output_vjp_stops_gradient_at_cotangents() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1, 1]).unwrap();
    let weight = graph.input(shape.clone(), Domain::Finite);
    let input = graph.input(shape.clone(), Domain::Finite);
    let square = graph.mul(weight, weight).unwrap();
    let output = graph.add(square, input).unwrap();
    let first_seed = graph.mul(square, input).unwrap();
    let second_seed = graph.sub(input, weight).unwrap();
    let reverse = differentiate_with_seeds(
        &graph,
        &[(square, first_seed), (output, second_seed)],
        &[weight],
    )
    .unwrap();
    let gradient = reverse.gradient_of(weight).unwrap();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    );
    let program = engine
        .compile_outputs(reverse.graph(), &[gradient])
        .unwrap();
    let values = vec![
        (weight, Tensor::from_slice(vec![1, 1], &[2.]).unwrap()),
        (input, Tensor::from_slice(vec![1, 1], &[3.]).unwrap()),
    ];
    let path = std::env::temp_dir().join(format!("vn_seeded_vjp_{}", std::process::id()));
    let result = engine.execute(&program, &values, &path).unwrap();
    // seeds: 2^2 * 3=12 and 3-2=1; VJP=4*(12+1)=52.
    assert!((result.values()[0] - 52.).abs() < 1e-5);
}
