use verineural::{Domain, ExactFp32Plan, NeuralGraph, NumericError, Shape, Tensor};

fn plan() -> (
    NeuralGraph,
    verineural::NodeId,
    [verineural::NodeId; 3],
) {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2, 3]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let bias = graph.input(Shape::new(vec![3]).unwrap(), Domain::Finite);
    let s = graph.add(x, y).unwrap();
    let biased = graph.bias_add(s, bias).unwrap();
    let relu = graph.relu(biased).unwrap();
    let output = graph
        .reshape(relu, Shape::new(vec![3, 2]).unwrap())
        .unwrap();
    (graph, output, [x, y, bias])
}

fn inputs(ids: [verineural::NodeId; 3]) -> [(verineural::NodeId, Tensor<f32>); 3] {
    [
        (
            ids[0],
            Tensor::new(
                Shape::new(vec![2, 3]).unwrap(),
                vec![-1_048_576.0, 4.0, 0.0, -10.0, 2.0, 1.0],
            )
            .unwrap(),
        ),
        (
            ids[1],
            Tensor::new(
                Shape::new(vec![2, 3]).unwrap(),
                vec![1_048_576.0, -1.0, 3.0, 1.0, 1.0, 4.0],
            )
            .unwrap(),
        ),
        (
            ids[2],
            Tensor::new(
                Shape::new(vec![3]).unwrap(),
                vec![1.0, -5.0, 2.0],
            )
            .unwrap(),
        ),
    ]
}

#[test]
fn stage8_integer_range_composes_and_integer_oracle_is_independent_of_float_ops() {
    let (graph, output, ids) = plan();
    let plan = ExactFp32Plan::compile(&graph, output).unwrap();
    assert_eq!(plan.source_inputs(), &ids);
    assert_eq!(plan.output_bound(), 3 * ExactFp32Plan::input_bound());
    let values = inputs(ids);
    let reference = plan.integer_reference(&values).unwrap();
    assert_eq!(reference.shape().dims(), &[3, 2]);
    assert_eq!(reference.values(), &[1.0, 0.0, 5.0, 0.0, 0.0, 7.0]);
    plan.compare_gpu(&values, &reference).unwrap();
    let source = plan.lean_source().unwrap();
    assert!(source.contains("theorem output_integer_range"));
    assert!(source.contains("tensor_within_biasAdd"));
    assert!(source.contains("tensor_within_relu"));
    assert!(source.contains("tensor_within_reshape"));
    assert!(source.contains("3145728"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_unsupported_fp32_inputs_fail_closed() {
    let (graph, output, ids) = plan();
    let plan = ExactFp32Plan::compile(&graph, output).unwrap();
    for value in [
        -0.0f32,
        f32::from_bits(1),
        0.25f32,
        f32::INFINITY,
        f32::NAN,
        1_048_577.0f32,
    ] {
        let mut batch = inputs(ids);
        let mut values = batch[0].1.values().to_vec();
        values[0] = value;
        batch[0].1 = Tensor::new(Shape::new(vec![2, 3]).unwrap(), values).unwrap();
        assert!(
            plan.integer_reference(&batch).is_err(),
            "value should be rejected: bits={:08x}",
            value.to_bits()
        );
    }
}

#[test]
fn stage8_rejects_unsupported_ops_and_unrepresentable_intermediates() {
    let shape = Shape::new(vec![2]).unwrap();
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let product = graph.mul(x, y).unwrap();
    assert!(matches!(
        ExactFp32Plan::compile(&graph, product),
        Err(NumericError::UnsupportedOperation { .. })
    ));

    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
    let mut current = x;
    for _ in 0..16 {
        current = graph.add(current, x).unwrap();
    }
    assert!(matches!(
        ExactFp32Plan::compile(&graph, current),
        Err(NumericError::UnsupportedRange { .. })
    ));
}

#[test]
fn stage8_invalid_batches_and_gpu_outputs_fail_closed() {
    let (graph, output, ids) = plan();
    let plan = ExactFp32Plan::compile(&graph, output).unwrap();
    let valid = inputs(ids);
    assert!(matches!(
        plan.integer_reference(&valid[..2]),
        Err(NumericError::MissingInput { .. })
    ));
    let invalid = [valid[0].clone(), valid[0].clone(), valid[2].clone()];
    assert!(matches!(
        plan.integer_reference(&invalid),
        Err(NumericError::InvalidInput { .. })
    ));
    let mut incorrect = plan.integer_reference(&valid).unwrap();
    let mut values = incorrect.values().to_vec();
    values[0] = -0.0;
    incorrect = Tensor::new(incorrect.shape().clone(), values).unwrap();
    assert!(matches!(
        plan.compare_gpu(&valid, &incorrect),
        Err(NumericError::OutputBits { index: 0, .. })
    ));
}
