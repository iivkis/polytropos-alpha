use crate::*;

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec()).unwrap()
}

fn port(dims: &[usize], domain: Domain) -> PortContract {
    PortContract::new(shape(dims), domain)
}

struct ReluModule {
    contract: MathContract,
}

impl ReluModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: MathContract::new(
                vec![port(dims, Domain::Finite)],
                port(dims, Domain::NonNegativeFinite),
            ),
        }
    }
}

impl NeuralModule for ReluModule {
    fn name(&self) -> &'static str {
        "relu_module"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.relu(inputs[0])
    }
}

struct SqrtModule {
    contract: MathContract,
}

impl SqrtModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: MathContract::new(
                vec![port(dims, Domain::NonNegativeFinite)],
                port(dims, Domain::NonNegativeFinite),
            ),
        }
    }
}

impl NeuralModule for SqrtModule {
    fn name(&self) -> &'static str {
        "sqrt_module"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.sqrt(inputs[0])
    }
}

struct IncorrectClaim {
    contract: MathContract,
}

impl NeuralModule for IncorrectClaim {
    fn name(&self) -> &'static str {
        "incorrect_claim"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

#[test]
fn rejects_zero_dimensions_and_overflow() {
    assert_eq!(Shape::new(vec![2, 0]), Err(NeuralError::ZeroDimension));
    assert_eq!(
        Shape::new(vec![usize::MAX, 2]),
        Err(NeuralError::ElementCountOverflow)
    );
    assert_eq!(shape(&[]).element_count(), 1);
}

#[test]
fn tensor_shape_and_payload_must_agree() {
    let err = Tensor::<f32>::new(shape(&[2, 2]), vec![1.0, 2.0]).unwrap_err();
    assert!(matches!(
        err,
        NeuralError::DataLengthMismatch {
            expected: 4,
            actual: 2
        }
    ));
    let t = Tensor::new(shape(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(t.shape().dims(), &[2, 2]);
    assert_eq!(t.values().len(), 4);
}

#[test]
fn reference_operations_reuse_graph_shape_rules() {
    let a = Tensor::new(shape(&[2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::new(shape(&[2, 2]), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();
    assert_eq!(a.add(&b).unwrap().values(), &[6.0, 8.0, 10.0, 12.0]);
    assert_eq!(a.mul(&b).unwrap().values(), &[5.0, 12.0, 21.0, 32.0]);
    assert_eq!(a.matmul(&b).unwrap().values(), &[19.0, 22.0, 43.0, 50.0]);
    let wrong = Tensor::new(shape(&[4]), vec![1.0f32; 4]).unwrap();
    assert!(matches!(
        a.add(&wrong),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        a.matmul(&wrong),
        Err(NeuralError::RankMismatch { .. })
    ));
}

#[test]
fn concrete_domain_validation_rejects_invalid_values() {
    let t = Tensor::new(shape(&[3]), vec![0.0f32, 4.0, -1.0]).unwrap();
    assert_eq!(
        t.sqrt().unwrap_err(),
        NeuralError::ValueOutsideDomain {
            index: 2,
            required: Domain::NonNegativeFinite
        }
    );
    let nan = Tensor::new(shape(&[1]), vec![f32::NAN]).unwrap();
    assert!(nan.relu().values()[0].is_nan());
    assert!(nan.validate_domain(Domain::Finite).is_err());
    assert!(nan.validate_domain(Domain::Any).is_ok());
}

#[test]
fn graph_checks_shapes_before_appending_nodes() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 3]), Domain::Any);
    let b = graph.input(shape(&[3, 2]), Domain::Any);
    assert!(matches!(
        graph.add(a, b),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    assert_eq!(graph.nodes().len(), 2);
    let c = graph.matmul(a, b).unwrap();
    assert_eq!(graph.node(c).unwrap().shape, shape(&[2, 2]));
    assert_eq!(graph.node(c).unwrap().inputs, vec![a, b]);
    assert_eq!(
        graph.node(c).unwrap().semantics.gpu,
        GpuSemantics::NotLowered
    );
    assert_eq!(graph.node(c).unwrap().semantics.fp32, FP32Policy::strict());
}

#[test]
fn graph_rejects_foreign_handles_and_unknown_domain() {
    let mut a = NeuralGraph::default();
    let mut b = NeuralGraph::default();
    let external = a.input(shape(&[2]), Domain::Finite);
    let _ = b.input(shape(&[2]), Domain::Finite);
    assert_eq!(b.relu(external), Err(NeuralError::InvalidNode));
    let arbitrary = a.input(shape(&[2]), Domain::Any);
    assert_eq!(
        a.sqrt(arbitrary),
        Err(NeuralError::DomainNotEstablished {
            required: Domain::NonNegativeFinite,
            actual: Domain::Any,
        })
    );
    assert_eq!(a.nodes().len(), 2);
}

#[test]
fn modules_compose_by_output_contracts() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(shape(&[2]), Domain::Finite);
    let first = graph
        .apply_module(&ReluModule::new(&[2]), &[input])
        .unwrap();
    let second = graph
        .apply_module(&SqrtModule::new(&[2]), &[first])
        .unwrap();
    assert_eq!(graph.node(first).unwrap().domain, Domain::NonNegativeFinite);
    assert_eq!(
        graph.node(second).unwrap().domain,
        Domain::NonNegativeFinite
    );
    assert_eq!(graph.modules().len(), 2);
    assert_eq!(graph.nodes().len(), 3);
}

#[test]
fn module_failure_rolls_back_and_false_guarantees_fail_closed() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2]), Domain::Finite);
    let b = graph.input(shape(&[2]), Domain::Finite);
    let claim = IncorrectClaim {
        contract: MathContract::new(
            vec![port(&[2], Domain::Finite), port(&[2], Domain::Finite)],
            port(&[2], Domain::Finite),
        ),
    };
    assert_eq!(
        graph.apply_module(&claim, &[a, b]),
        Err(NeuralError::DomainNotEstablished {
            required: Domain::Finite,
            actual: Domain::Any,
        })
    );
    assert_eq!(graph.nodes().len(), 2);
    assert!(graph.modules().is_empty());
}

#[test]
fn reshape_keeps_count_and_domain() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2, 3]), Domain::NonNegativeFinite);
    let y = graph.reshape(x, shape(&[3, 2])).unwrap();
    assert_eq!(graph.node(y).unwrap().domain, Domain::NonNegativeFinite);
    assert!(matches!(
        graph.reshape(x, shape(&[4, 2])),
        Err(NeuralError::ElementCountMismatch { .. })
    ));
    let tensor = Tensor::new(shape(&[2, 3]), vec![1.0f32; 6]).unwrap();
    assert_eq!(
        tensor.reshape(shape(&[3, 2])).unwrap().shape().dims(),
        &[3, 2]
    );
}

#[test]
fn proof_generator_emits_node_and_module_theorems() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(shape(&[2, 2]), Domain::Finite);
    let hidden = graph
        .apply_module(&ReluModule::new(&[2, 2]), &[input])
        .unwrap();
    let flat = graph.reshape(hidden, shape(&[4])).unwrap();
    let summed = graph.add(flat, flat).unwrap();
    let out = graph.mul(summed, flat).unwrap();

    let proof = LeanProofEngine::default().generate(&graph, out).unwrap();
    assert!(proof.source().contains("import Init"));
    assert!(!proof.source().contains("sorry"));
    assert!(!proof.source().contains("axiom "));
    assert!(proof.source().contains("theorem node_1_nonnegative"));
    assert!(proof.source().contains("theorem node_2_input_0_shape"));
    assert!(
        proof
            .source()
            .contains("theorem module_0_output_nonnegative")
    );
    assert!(
        proof
            .source()
            .contains("VeriNeural.add (node_2 x_0) (node_2 x_0)")
    );
    assert!(proof.source().contains("theorem node_4_semantics"));
    assert!(proof.assumptions().is_empty());
    assert!(proof.theorem_names().len() >= 10);
}

#[test]
fn proof_generator_marks_unused_graph_inputs_without_hiding_lean_warnings() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2]), Domain::Finite);
    let b = graph.input(shape(&[2]), Domain::Finite);
    let result = graph.add(a, b).unwrap();

    let generated = LeanProofEngine::default().generate(&graph, result).unwrap();
    let lean = generated.source();
    assert!(lean.contains("def node_0 (x_0 : VeriNeural.Tensor 2) (_x_1 : VeriNeural.Tensor 2)"));
    assert!(lean.contains("def node_1 (_x_0 : VeriNeural.Tensor 2) (x_1 : VeriNeural.Tensor 2)"));
    // Calling conventions and theorem statements still use the real inputs.
    assert!(lean.contains("def node_2 (x_0 : VeriNeural.Tensor 2) (x_1 : VeriNeural.Tensor 2)"));
    assert!(lean.contains(
        "theorem node_0_semantics (x_0 : VeriNeural.Tensor 2) (x_1 : VeriNeural.Tensor 2)"
    ));
    assert!(!lean.contains("set_option linter.unusedVariables false"));
}

#[test]
fn proof_generator_keeps_source_domain_assumptions_explicit() {
    let mut graph = NeuralGraph::default();
    let src = graph.input(shape(&[3]), Domain::NonNegativeFinite);
    let output = graph.reshape(src, shape(&[1, 3])).unwrap();
    let proof = LeanProofEngine::default().generate(&graph, output).unwrap();
    assert!(
        proof
            .source()
            .contains("(h_0 : VeriNeural.Nonnegative x_0)")
    );
    assert!(proof.source().contains("exact h_0"));
    assert_eq!(
        proof.assumptions(),
        &[ProofAssumption {
            input_node: 0,
            domain: Domain::NonNegativeFinite
        }]
    );
}

#[test]
fn proof_generator_rejects_unsupported_operations() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let out = graph.sqrt(input).unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate(&graph, out),
        Err(ProofError::UnsupportedOperation {
            operation: "sqrt",
            ..
        })
    ));
}

#[test]
fn proof_generator_checks_only_reachable_subgraph() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(shape(&[2]), Domain::Finite);
    let verified_path = graph.relu(input).unwrap();
    let a = graph.input(shape(&[2, 2]), Domain::NonNegativeFinite);
    let _unsupported = graph.sqrt(a).unwrap();
    // An unsupported sibling outside the proof slice must be ignored.
    let proof = LeanProofEngine::default()
        .generate(&graph, verified_path)
        .unwrap();
    assert!(proof.source().contains("theorem node_1_nonnegative"));
    assert!(!proof.source().contains("theorem node_2_shape"));
}

#[test]
fn proof_generator_rejects_handles_from_other_graphs() {
    let mut first = NeuralGraph::default();
    let input = first.input(shape(&[2]), Domain::Any);
    let second = NeuralGraph::default();
    assert!(matches!(
        LeanProofEngine::default().generate(&second, input),
        Err(ProofError::InvalidGraph(NeuralError::InvalidNode))
    ));
}

#[test]
fn gpu_compiler_lowers_arithmetic_and_reshape_to_a_reachable_kernel_dag() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 2]), Domain::Finite);
    let b = graph.input(shape(&[2, 2]), Domain::Finite);
    let active = graph.relu(a).unwrap();
    let flat = graph.reshape(active, shape(&[4])).unwrap();
    let restored = graph.reshape(flat, shape(&[2, 2])).unwrap();
    let sum = graph.add(restored, b).unwrap();
    let product = graph.mul(sum, active).unwrap();

    let program = GpuEngine::default().compile(&graph, product).unwrap();
    assert_eq!(program.source_inputs(), &[a, b]);
    let ir = program.llvm_ir();
    assert!(ir.contains("target triple = \"nvptx64-nvidia-cuda\""));
    assert!(ir.contains("define ptx_kernel void @vn_node_2"));
    assert!(ir.contains("define ptx_kernel void @vn_node_5"));
    assert!(ir.contains("define ptx_kernel void @vn_node_6"));
    assert!(!ir.contains("define ptx_kernel void @vn_node_3"));
    assert!(!ir.contains("define ptx_kernel void @vn_node_4"));
    assert!(ir.contains("icmp ult i32 %idx, %length"));
    assert!(ir.contains("fcmp uno float %x, %x"));
    assert!(ir.contains("fadd float %x, %y"));
    assert!(ir.contains("fmul float %x, %y"));
    assert!(!ir.contains("fast "));
}

#[test]
fn gpu_compiler_rejects_unsupported_operations_and_foreign_handles() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 2]), Domain::NonNegativeFinite);
    let sqrt = graph.sqrt(a).unwrap();
    assert!(matches!(
        GpuEngine::default().compile(&graph, sqrt),
        Err(GpuError::UnsupportedOperation {
            operation: "sqrt",
            ..
        })
    ));
    let mut other = NeuralGraph::default();
    let foreign = other.input(shape(&[2, 2]), Domain::Any);
    assert!(matches!(
        GpuEngine::default().compile(&graph, foreign),
        Err(GpuError::InvalidGraph(NeuralError::InvalidNode))
    ));
}

#[test]
fn gpu_input_validation_fails_before_toolchain_or_driver_is_required() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let result = graph.relu(a).unwrap();
    let engine = GpuEngine::new("missing-llc", "sm_52");
    let program = engine.compile(&graph, result).unwrap();

    let empty: Vec<(NodeId, Tensor<f32>)> = Vec::new();
    assert!(matches!(
        engine.execute(&program, &empty, std::path::Path::new("unused")),
        Err(GpuError::MissingInput { .. })
    ));
    let invalid = Tensor::new(shape(&[2]), vec![-1.0f32, 2.0]).unwrap();
    assert!(matches!(
        engine.execute(&program, &[(a, invalid)], std::path::Path::new("unused")),
        Err(GpuError::InvalidGraph(
            NeuralError::ValueOutsideDomain { .. }
        ))
    ));
    let wrong = Tensor::new(shape(&[1]), vec![1.0f32]).unwrap();
    assert!(matches!(
        engine.execute(&program, &[(a, wrong)], std::path::Path::new("unused")),
        Err(GpuError::InvalidGraph(NeuralError::ShapeMismatch { .. }))
    ));
}


#[test]
fn stage6_bias_shape_validation_and_cpu_broadcast() {
    let matrix = Tensor::new(shape(&[2, 3]), vec![1_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bias = Tensor::new(shape(&[3]), vec![-1_f32, 0.0, 2.0]).unwrap();
    assert_eq!(matrix.bias_add(&bias).unwrap().values(), &[0., 2., 5., 3., 5., 8.]);
    assert!(matches!(
        matrix.bias_add(&Tensor::new(shape(&[2]), vec![1_f32, 2.0]).unwrap()),
        Err(NeuralError::InnerDimensionMismatch { .. })
    ));
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 3]), Domain::Finite);
    let wrong = graph.input(shape(&[2]), Domain::Finite);
    let old_len = graph.nodes().len();
    assert!(matches!(graph.bias_add(a, wrong), Err(NeuralError::InnerDimensionMismatch { .. })));
    assert_eq!(graph.nodes().len(), old_len);
}

#[test]
fn stage6_dense_residual_ffn_composes_and_emits_matmul_proofs() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2, 2]), Domain::Finite);
    let w1 = graph.input(shape(&[2, 3]), Domain::Finite);
    let b1 = graph.input(shape(&[3]), Domain::Finite);
    let w2 = graph.input(shape(&[3, 2]), Domain::Finite);
    let b2 = graph.input(shape(&[2]), Domain::Finite);
    let block = ResidualFfn::new(2, 2, 3).unwrap();
    let output = graph.apply_module(&block, &[x, w1, b1, w2, b2]).unwrap();
    assert_eq!(graph.node(output).unwrap().shape.dims(), &[2, 2]);
    assert_eq!(graph.modules().len(), 1);

    let lean = LeanProofEngine::default().generate(&graph, output).unwrap();
    assert!(lean.source().contains("VeriNeural.matmul (m := 2) (k := 2) (n := 3)"));
    assert!(lean.source().contains("VeriNeural.biasAdd (m := 2) (n := 3)"));
    assert!(lean.source().contains("theorem module_0_output_shape"));

    let gpu = GpuEngine::default().compile(&graph, output).unwrap();
    assert!(gpu.llvm_ir().contains("%sum = phi float"));
    assert!(gpu.llvm_ir().contains("urem i32 %idx, 3"));
    assert!(!gpu.llvm_ir().contains("fast "));
}

#[test]
fn stage6_dense_and_residual_contracts_are_checked() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 2]), Domain::Any);
    let w = graph.input(shape(&[2, 3]), Domain::Any);
    let bias = graph.input(shape(&[3]), Domain::Any);
    let dense = Dense::new(2, 2, 3).unwrap();
    let out = graph.apply_module(&dense, &[a, w, bias]).unwrap();
    assert_eq!(graph.node(out).unwrap().shape.dims(), &[2, 3]);

    let residual = ResidualAdd::new(shape(&[2, 3]));
    let twice = graph.apply_module(&residual, &[out, out]).unwrap();
    assert_eq!(graph.node(twice).unwrap().shape.dims(), &[2, 3]);

    let bad_bias = graph.input(shape(&[2]), Domain::Any);
    let previous_nodes = graph.nodes().len();
    assert!(graph.apply_module(&dense, &[a, w, bad_bias]).is_err());
    assert_eq!(graph.nodes().len(), previous_nodes);
    assert!(ResidualFfn::new(2, 0, 3).is_err());
}

#[test]
fn stage6_cpu_ffn_oracle_has_expected_concrete_values() {
    let x = Tensor::new(shape(&[2, 2]), vec![1_f32, 2., 3., 4.]).unwrap();
    let w1 = Tensor::new(shape(&[2, 3]), vec![1_f32, 0., 1., 0., 1., 1.]).unwrap();
    let b1 = Tensor::new(shape(&[3]), vec![-1_f32, 0., 1.]).unwrap();
    let w2 = Tensor::new(shape(&[3, 2]), vec![1_f32, 0., 0., 1., 1., 1.]).unwrap();
    let b2 = Tensor::new(shape(&[2]), vec![0_f32, 1.]).unwrap();
    let hidden = x.matmul(&w1).unwrap().bias_add(&b1).unwrap().relu();
    let y = x.add(&hidden.matmul(&w2).unwrap().bias_add(&b2).unwrap()).unwrap();
    assert_eq!(y.values(), &[5., 9., 13., 17.]);
}
