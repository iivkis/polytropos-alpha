use verineural::{
    Domain, ExactFp32Plan, NeuralGraph, NumericError, ResidualFfn, RneFp32Plan, RneRangePlan, Shape,
    Tensor,
};

fn plan() -> (NeuralGraph, verineural::NodeId, [verineural::NodeId; 3]) {
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
            Tensor::new(Shape::new(vec![3]).unwrap(), vec![1.0, -5.0, 2.0]).unwrap(),
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
        Err(NumericError::UnsupportedRange { .. })
    ));
    let nonneg = graph.input(Shape::new(vec![2]).unwrap(), Domain::NonNegativeFinite);
    let sqrt = graph.sqrt(nonneg).unwrap();
    assert!(matches!(
        ExactFp32Plan::compile(&graph, sqrt),
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

#[test]
fn stage8_lean_uses_only_live_hypotheses_and_explicit_bias_shapes() {
    let (graph, output, _) = plan();
    let source = ExactFp32Plan::compile(&graph, output)
        .unwrap()
        .lean_source()
        .unwrap();

    let input_header = source
        .lines()
        .find(|line| line.starts_with("theorem node_0_integer_range "))
        .unwrap();
    assert!(input_header.contains("(h_0 :"));
    assert!(!input_header.contains("(h_1 :"));
    assert!(!input_header.contains("(h_2 :"));

    let sum_header = source
        .lines()
        .find(|line| line.starts_with("theorem node_3_integer_range "))
        .unwrap();
    assert!(sum_header.contains("(h_0 :"));
    assert!(sum_header.contains("(h_1 :"));
    assert!(!sum_header.contains("(h_2 :"));

    let bias_header = source
        .lines()
        .find(|line| line.starts_with("theorem node_4_integer_range "))
        .unwrap();
    for hypothesis in ["(h_0 :", "(h_1 :", "(h_2 :"] {
        assert!(bias_header.contains(hypothesis));
    }
    assert!(source.contains("tensor_within_biasAdd (m := 2) (n := 3)"));
    assert!(source.contains("(a := (node_3 x_0 x_1 x_2))"));
    assert!(source.contains("(b := (node_2 x_0 x_1 x_2))"));

    let library = include_str!("../../../lean/VeriNeural/Numeric.lean");
    assert!(library.contains("Int.max_eq_right"));
    assert!(library.contains("Int.max_eq_left"));
}

#[test]
fn stage8_mul_matmul_range_proofs_and_i64_oracle_compose() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(Shape::new(vec![2, 3]).unwrap(), Domain::Finite);
    let b = graph.input(Shape::new(vec![3, 2]).unwrap(), Domain::Finite);
    let scale = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let bias = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let dot = graph.matmul(a, b).unwrap();
    let product = graph.mul(dot, scale).unwrap();
    let biased = graph.bias_add(product, bias).unwrap();
    let output = graph.relu(biased).unwrap();

    let plan = ExactFp32Plan::compile_with_input_bound(&graph, output, 8).unwrap();
    assert_eq!(plan.max_input_bound(), 8);
    // Bound: k * input^2 = 3*64=192, then Mul by 8, then BiasAdd 8.
    assert_eq!(plan.output_bound(), 1544);
    let src = plan.lean_source().unwrap();
    assert!(src.contains("tensor_within_mul"));
    assert!(src.contains("tensor_within_matmul (m := 2) (k := 3) (n := 2)"));
    assert!(src.contains("theorem output_integer_range"));
    assert!(src.contains("1544"));

    let tensors = [
        (
            a,
            Tensor::new(
                Shape::new(vec![2, 3]).unwrap(),
                vec![2.0, -1.0, 3.0, 0.0, 4.0, -2.0],
            )
            .unwrap(),
        ),
        (
            b,
            Tensor::new(
                Shape::new(vec![3, 2]).unwrap(),
                vec![1.0, 2.0, -3.0, 1.0, 2.0, -1.0],
            )
            .unwrap(),
        ),
        (
            scale,
            Tensor::new(Shape::new(vec![2, 2]).unwrap(), vec![2.0, 3.0, 2.0, 4.0]).unwrap(),
        ),
        (
            bias,
            Tensor::new(Shape::new(vec![2]).unwrap(), vec![1.0, -2.0]).unwrap(),
        ),
    ];
    let result = plan.integer_reference(&tensors).unwrap();
    assert_eq!(result.values(), &[23.0, 0.0, 0.0, 22.0]);
    plan.compare_gpu(&tensors, &result).unwrap();
}

#[test]
fn stage8_multiplication_reduction_fail_closed_when_not_exactly_bounded() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let y = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let product = graph.mul(x, y).unwrap();
    assert!(matches!(
        ExactFp32Plan::compile(&graph, product),
        Err(NumericError::UnsupportedRange { .. })
    ));
    let bounded = ExactFp32Plan::compile_with_input_bound(&graph, product, 100).unwrap();
    assert_eq!(bounded.output_bound(), 10_000);
    assert!(ExactFp32Plan::compile_with_input_bound(&graph, product, 0).is_err());
    assert!(ExactFp32Plan::compile_with_input_bound(&graph, product, 1_048_577).is_err());

    let mut g = NeuralGraph::default();
    let lhs = g.input(Shape::new(vec![1, 16]).unwrap(), Domain::Finite);
    let rhs = g.input(Shape::new(vec![16, 1]).unwrap(), Domain::Finite);
    let dot = g.matmul(lhs, rhs).unwrap();
    assert!(matches!(
        ExactFp32Plan::compile_with_input_bound(&g, dot, 2048),
        Err(NumericError::UnsupportedRange { .. })
    ));
    assert_eq!(
        ExactFp32Plan::compile_with_input_bound(&g, dot, 1024)
            .unwrap()
            .output_bound(),
        16_777_216
    );
}

/// Regression: the first Stage 8 graph must not attempt to check the second
/// graph's Lean file before that file has been generated. The example has
/// deliberately separate generation and saved-artifact verification paths.
#[test]
fn stage8_example_checks_each_lean_artifact_only_after_generation() {
    let source = include_str!("../examples/stage8_exact.rs");
    let main = source
        .split_once("fn main() ->")
        .expect("Stage 8 example main")
        .1;
    let (first_graph, second_graph) = main
        .split_once("let mat_plan =")
        .expect("Stage 8 second graph");
    assert!(!first_graph.contains("numeric_matmul.lean"));
    assert!(second_graph.contains(
        "write_checked_lean(&root.join(\"numeric_matmul.lean\"), &mat_plan.lean_source()?)?"
    ));
    assert!(
        first_graph
            .contains("write_checked_lean(&root.join(\"numeric.lean\"), &plan.lean_source()?)?")
    );
    let (before_rne, after_rne) = second_graph
        .split_once("let rne_plan =")
        .expect("Stage 8 RNE graph");
    assert!(!before_rne.contains("numeric_rne.lean"));
    assert!(after_rne.contains(
        "write_checked_lean(&root.join(\"numeric_rne.lean\"), rne_trace.lean_source())?"
    ));
    assert!(after_rne.contains("numeric_rne_matmul.lean"));
    assert!(after_rne.contains("rne_mat_trace.lean_source()"));
    assert!(after_rne.contains("numeric_range.lean"));
    assert!(after_rne.contains("range_plan.lean_source()?"));
}

#[test]
fn stage8_fractional_rne_trace_accumulates_nonzero_epsilon() {
    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![1, 1]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
    let sum = graph.add(x, y).unwrap();
    let biased = graph.bias_add(sum, bias).unwrap();
    let output = graph.relu(biased).unwrap();

    // q = 20. 0x41000001 is 8 + 2^-20, 0x35800000 is 2^-20.
    // First add: 16 + 2^-20 is exactly halfway between two binary32 values
    // and ties to even at 16. The bias repeats the same loss.
    let inputs = [
        (
            x,
            Tensor::new(
                Shape::new(vec![1, 1]).unwrap(),
                vec![f32::from_bits(0x4100_0001)],
            )
            .unwrap(),
        ),
        (
            y,
            Tensor::new(
                Shape::new(vec![1, 1]).unwrap(),
                vec![f32::from_bits(0x4100_0000)],
            )
            .unwrap(),
        ),
        (
            bias,
            Tensor::new(
                Shape::new(vec![1]).unwrap(),
                vec![f32::from_bits(0x3580_0000)],
            )
            .unwrap(),
        ),
    ];

    let plan = RneFp32Plan::compile(&graph, output, 20).unwrap();
    let trace = plan.trace(&inputs).unwrap();
    assert_eq!(plan.fraction_bits(), 20);
    assert_eq!(trace.ideal_output_ticks(), &[16_777_218]);
    assert_eq!(trace.rounded_output_ticks(), &[16_777_216]);
    assert_eq!(trace.observed_error_ticks(), 2);
    // The Lean composition deliberately uses a conservative one-ulp budget
    // for each rounding site, so two sites produce four ticks.
    assert_eq!(trace.epsilon_ticks(), 4);
    assert_eq!(trace.expected().values()[0].to_bits(), 0x4180_0000);
    trace.compare_gpu(trace.expected()).unwrap();

    let source = trace.lean_source();
    assert!(source.contains("Binary32RneTick 16777217 16777216 1"));
    assert!(source.contains("VeriNeural.approx_add_round"));
    assert!(source.contains("theorem output_elem_0_fp32_error"));
    assert!(source.contains("ApproxWithin 4 16777218 16777216"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_fractional_rne_ties_choose_even_for_both_signs() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![2]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape.clone(), Domain::Finite);
    let output = graph.add(x, y).unwrap();
    let inputs = [
        (
            x,
            Tensor::new(
                shape.clone(),
                vec![f32::from_bits(0x4100_0002), f32::from_bits(0xc100_0002)],
            )
            .unwrap(),
        ),
        (
            y,
            Tensor::new(
                shape,
                vec![f32::from_bits(0x4100_0001), f32::from_bits(0xc100_0001)],
            )
            .unwrap(),
        ),
    ];

    let trace = RneFp32Plan::compile(&graph, output, 20)
        .unwrap()
        .trace(&inputs)
        .unwrap();
    assert_eq!(trace.ideal_output_ticks(), &[16_777_219, -16_777_219]);
    assert_eq!(trace.rounded_output_ticks(), &[16_777_220, -16_777_220]);
    assert_eq!(
        trace
            .expected()
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![0x4180_0002, 0xc180_0002]
    );
    assert!(
        trace
            .lean_source()
            .contains("Binary32RneTick 16777219 16777220 1")
    );
    assert!(
        trace
            .lean_source()
            .contains("Binary32RneTick -16777219 -16777220 1")
    );
}

#[test]
fn stage8_fractional_mul_scales_lattice_and_composes_error() {
    let mut graph = NeuralGraph::default();
    let matrix = Shape::new(vec![1, 1]).unwrap();
    let x = graph.input(matrix.clone(), Domain::Finite);
    let y = graph.input(matrix, Domain::Finite);
    let bias = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
    let product = graph.mul(x, y).unwrap();
    let output = graph.bias_add(product, bias).unwrap();

    let inputs = [
        (
            x,
            Tensor::new(
                Shape::new(vec![1, 1]).unwrap(),
                vec![f32::from_bits(0x4100_0001)],
            )
            .unwrap(),
        ),
        (
            y,
            Tensor::new(
                Shape::new(vec![1, 1]).unwrap(),
                vec![f32::from_bits(0x4100_0001)],
            )
            .unwrap(),
        ),
        (
            bias,
            Tensor::new(
                Shape::new(vec![1]).unwrap(),
                vec![f32::from_bits(0x3580_0000)],
            )
            .unwrap(),
        ),
    ];

    let plan = RneFp32Plan::compile(&graph, output, 20).unwrap();
    assert_eq!(plan.output_fraction_bits(), 40);
    let trace = plan.trace(&inputs).unwrap();
    assert_eq!(trace.fraction_bits(), 40);
    assert_eq!(trace.ideal_output_ticks(), &[70_368_762_003_457]);
    assert_eq!(trace.rounded_output_ticks(), &[70_368_760_954_880]);
    assert_eq!(trace.observed_error_ticks(), 1_048_577);
    assert_eq!(trace.epsilon_ticks(), 16_777_216);
    assert_eq!(trace.expected().values()[0].to_bits(), 0x4280_0002);

    let source = trace.lean_source();
    assert!(source.contains("VeriNeural.approx_mul_round"));
    assert!(source.contains("VeriNeural.approx_scale"));
    assert!(source.contains("Binary32RneTick 70368760954881 70368760954880 23"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_fractional_matmul_models_separate_mul_add_reduction() {
    let mut graph = NeuralGraph::default();
    let lhs_shape = Shape::new(vec![1, 2]).unwrap();
    let rhs_shape = Shape::new(vec![2, 1]).unwrap();
    let lhs = graph.input(lhs_shape.clone(), Domain::Finite);
    let rhs = graph.input(rhs_shape.clone(), Domain::Finite);
    let output = graph.matmul(lhs, rhs).unwrap();

    let value = f32::from_bits(0x4100_0001);
    let inputs = [
        (lhs, Tensor::new(lhs_shape, vec![value, value]).unwrap()),
        (rhs, Tensor::new(rhs_shape, vec![value, value]).unwrap()),
    ];

    let plan = RneFp32Plan::compile(&graph, output, 20).unwrap();
    assert_eq!(plan.output_fraction_bits(), 40);
    let trace = plan.trace(&inputs).unwrap();
    assert_eq!(trace.ideal_output_ticks(), &[140_737_521_909_762]);
    assert_eq!(trace.rounded_output_ticks(), &[140_737_521_909_760]);
    assert_eq!(trace.observed_error_ticks(), 2);
    assert_eq!(trace.epsilon_ticks(), 41_943_040);
    assert_eq!(trace.expected().values()[0].to_bits(), 0x4300_0002);

    let source = trace.lean_source();
    assert!(source.contains("product_0_error"));
    assert!(source.contains("sum_1_error"));
    assert!(source.contains("VeriNeural.approx_mul_round"));
    assert!(source.contains("VeriNeural.approx_add_round"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_rne_range_contract_is_reusable_over_an_input_interval() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape, Domain::Finite);
    let output = graph.add(x, y).unwrap();

    // Any exact source ticks in [-2^23, 2^23] at q=20. The worst possible
    // pre-round magnitude is 2^24 ticks, so the reusable RNE budget is 2 ticks.
    let plan = RneRangePlan::compile(&graph, output, 20, 8_388_608).unwrap();
    let contract = plan.output_contract();
    assert_eq!(contract.fraction_bits(), 20);
    assert_eq!(contract.ideal_bound_ticks(), 16_777_216);
    assert_eq!(contract.epsilon_ticks(), 2);
    assert_eq!(contract.actual_bound_ticks(), 16_777_218);

    let source = plan.lean_source().unwrap();
    assert!(source.contains("theorem node_2_range_rule"));
    assert!(source.contains("ApproxContract 16777216 2"));
    assert!(source.contains("approx_contract_add_rne (maxShift := 1)"));
    assert!(source.contains("binary32_rne_shift_le_of_bound"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_rne_range_contract_crosses_residual_ffn_module_boundary() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![1, 2]).unwrap(), Domain::Finite);
    let w1 = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let b1 = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let w2 = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
    let b2 = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
    let module = ResidualFfn::new(1, 2, 2).unwrap();
    let output = graph.apply_module(&module, &[x, w1, b1, w2, b2]).unwrap();

    // q=10 sources with |ticks| <= 16. Products refine the lattice through
    // q=20 and q=30; the final residual add aligns X to q=30.
    let plan = RneRangePlan::compile(&graph, output, 10, 16).unwrap();
    let contract = plan.output_contract();
    assert_eq!(contract.fraction_bits(), 30);
    assert_eq!(contract.ideal_bound_ticks(), 34_095_104);
    assert_eq!(contract.epsilon_ticks(), 170);
    assert_eq!(contract.actual_bound_ticks(), 34_095_274);

    let modules = plan.module_contracts();
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0].name(), "residual_ffn");
    assert_eq!(modules[0].inputs().len(), 5);
    assert_eq!(modules[0].output(), output);
    assert_eq!(modules[0].contract(), contract);

    let source = plan.lean_source().unwrap();
    assert!(source.contains("module_0_contract_well_formed"));
    assert!(source.contains("exports q=30, ideal bound 34095104, epsilon 170 ticks"));
    assert!(source.contains("approx_contract_mul_rne"));
    assert!(source.contains("approx_contract_add_rne"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn stage8_fractional_rne_slice_fails_closed_outside_its_model() {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![1]).unwrap();
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape.clone(), Domain::Finite);
    let first = graph.mul(x, y).unwrap();
    let second = graph.mul(first, x).unwrap();
    let too_fine = graph.mul(second, y).unwrap();
    assert!(matches!(
        RneFp32Plan::compile(&graph, too_fine, 20),
        Err(NumericError::InvalidInput { .. })
    ));

    let mut unsupported = NeuralGraph::default();
    let sx = unsupported.input(shape, Domain::NonNegativeFinite);
    let sqrt = unsupported.sqrt(sx).unwrap();
    assert!(matches!(
        RneFp32Plan::compile(&unsupported, sqrt, 20),
        Err(NumericError::UnsupportedOperation { .. })
    ));

    let mut graph = NeuralGraph::default();
    let x = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
    let output = graph.relu(x).unwrap();
    let plan = RneFp32Plan::compile(&graph, output, 20).unwrap();

    for value in [-0.0f32, f32::from_bits(1), 0.1f32, f32::INFINITY] {
        let input = [(
            x,
            Tensor::new(Shape::new(vec![1]).unwrap(), vec![value]).unwrap(),
        )];
        assert!(
            plan.trace(&input).is_err(),
            "value should be rejected by the q=20 RNE slice: {:08x}",
            value.to_bits()
        );
    }
    assert!(RneFp32Plan::compile(&graph, output, 0).is_err());
    assert!(RneFp32Plan::compile(&graph, output, 31).is_err());
}
