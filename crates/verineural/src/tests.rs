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

struct AddModule {
    contract: MathContract,
}

impl AddModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: MathContract::new(
                vec![port(dims, Domain::Any), port(dims, Domain::Any)],
                port(dims, Domain::Any),
            ),
        }
    }
}

impl NeuralModule for AddModule {
    fn name(&self) -> &'static str {
        "add_module"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

struct MulModule {
    contract: MathContract,
}

impl MulModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: MathContract::new(
                vec![port(dims, Domain::Any), port(dims, Domain::Any)],
                port(dims, Domain::Any),
            ),
        }
    }
}

impl NeuralModule for MulModule {
    fn name(&self) -> &'static str {
        "mul_module"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.mul(inputs[0], inputs[1])
    }
}

struct NonnegativeMatMulModule {
    contract: MathContract,
}

impl NonnegativeMatMulModule {
    fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![port(&[2, 3], Domain::Any), port(&[3, 2], Domain::Any)],
                port(&[2, 2], Domain::Any),
            ),
        }
    }
}

impl NeuralModule for NonnegativeMatMulModule {
    fn name(&self) -> &'static str {
        "nonnegative_matmul"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.matmul(inputs[0], inputs[1])
    }
}

struct NonnegativeBiasAddModule {
    contract: MathContract,
}

impl NonnegativeBiasAddModule {
    fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![port(&[2, 2], Domain::Any), port(&[2], Domain::Any)],
                port(&[2, 2], Domain::Any),
            ),
        }
    }
}

impl NeuralModule for NonnegativeBiasAddModule {
    fn name(&self) -> &'static str {
        "nonnegative_bias_add"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.bias_add(inputs[0], inputs[1])
    }
}

struct PropertyInnerModule {
    contract: MathContract,
}

impl PropertyInnerModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: MathContract::new(vec![port(dims, Domain::Any); 3], port(dims, Domain::Any)),
        }
    }
}

impl NeuralModule for PropertyInnerModule {
    fn name(&self) -> &'static str {
        "property_inner"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let add = AddModule::new(&[3]);
        let mul = MulModule::new(&[3]);
        let left = graph.apply_module(&add, &[inputs[0], inputs[1]])?;
        let right = graph.apply_module(&mul, &[inputs[0], inputs[2]])?;
        graph.add(left, right)
    }
}

struct PropertyOuterModule {
    contract: MathContract,
}

impl PropertyOuterModule {
    fn new(dims: &[usize]) -> Self {
        Self {
            contract: PropertyInnerModule::new(dims).contract,
        }
    }
}

impl NeuralModule for PropertyOuterModule {
    fn name(&self) -> &'static str {
        "property_outer"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.apply_module(&PropertyInnerModule::new(&[3]), inputs)
    }
}

struct CapturingPropertyModule {
    contract: MathContract,
    undeclared: NodeId,
}

impl NeuralModule for CapturingPropertyModule {
    fn name(&self) -> &'static str {
        "capturing_property"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.apply_module(&AddModule::new(&[3]), &[inputs[0], self.undeclared])
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
    assert_eq!(graph.modules()[0].body_start, 1);
    assert_eq!(graph.modules()[1].body_start, 2);
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
fn proof_generator_composes_explicit_global_nonnegative_property_over_module_dag() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[3]), Domain::Finite);
    let y = graph.input(shape(&[3]), Domain::Finite);
    let z = graph.input(shape(&[3]), Domain::Finite);
    let add = AddModule::new(&[3]);
    let mul = MulModule::new(&[3]);

    let left = graph.apply_module(&add, &[x, y]).unwrap();
    let right = graph.apply_module(&mul, &[x, z]).unwrap();
    let output = graph.apply_module(&add, &[left, right]).unwrap();

    let proof = LeanProofEngine::default()
        .generate_global_nonnegative(&graph, output, &[x, y, z])
        .unwrap();
    let lean = proof.source();

    assert!(lean.contains("theorem module_0_global_nonnegative"));
    assert!(lean.contains("theorem module_1_global_nonnegative"));
    assert!(lean.contains("theorem module_2_global_nonnegative"));
    assert!(lean.contains("theorem architecture_global_nonnegative"));
    assert!(lean.contains("VeriNeural.add_nonnegative"));
    assert!(lean.contains("VeriNeural.mul_nonnegative"));
    assert!(lean.contains("have arch_module_0 := module_0_global_nonnegative"));
    assert!(lean.contains("have arch_module_1 := module_1_global_nonnegative"));
    assert!(lean.contains("have arch_module_2 := module_2_global_nonnegative"));
    assert!(lean.contains("exact arch_module_2"));
    assert!(!lean.contains("sorry"));
    assert!(!lean.contains("axiom "));

    assert_eq!(
        proof.assumptions(),
        &[
            ProofAssumption {
                input_node: x.index(),
                domain: Domain::NonNegativeFinite,
            },
            ProofAssumption {
                input_node: y.index(),
                domain: Domain::NonNegativeFinite,
            },
            ProofAssumption {
                input_node: z.index(),
                domain: Domain::NonNegativeFinite,
            },
        ]
    );
}

#[test]
fn proof_generator_global_nonnegative_handles_matmul_reduction_and_bias_broadcast() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2, 3]), Domain::Finite);
    let weight = graph.input(shape(&[3, 2]), Domain::Finite);
    let bias = graph.input(shape(&[2]), Domain::Finite);
    let product = graph
        .apply_module(&NonnegativeMatMulModule::new(), &[x, weight])
        .unwrap();
    let output = graph
        .apply_module(&NonnegativeBiasAddModule::new(), &[product, bias])
        .unwrap();

    let program = LeanProofEngine::default()
        .generate_global_nonnegative(&graph, output, &[x, weight, bias])
        .unwrap();
    let lean = program.source();
    assert!(lean.contains("VeriNeural.matmul_nonnegative (m := 2) (k := 3) (n := 2)"));
    assert!(lean.contains("VeriNeural.biasAdd_nonnegative (m := 2) (n := 2)"));
    assert!(lean.contains("theorem module_0_global_nonnegative"));
    assert!(lean.contains("theorem module_1_global_nonnegative"));
    assert!(lean.contains("have arch_module_0 := module_0_global_nonnegative"));
    assert!(lean.contains("have arch_module_1 := module_1_global_nonnegative"));
    assert!(lean.contains("exact arch_module_1"));
    assert!(!lean.contains("sorry"));
    assert!(!lean.contains("admit"));
}

#[test]
fn proof_generator_matmul_bias_nonnegative_rejects_missing_bias_assumption() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2, 3]), Domain::Finite);
    let weight = graph.input(shape(&[3, 2]), Domain::Finite);
    let bias = graph.input(shape(&[2]), Domain::Finite);
    let product = graph
        .apply_module(&NonnegativeMatMulModule::new(), &[x, weight])
        .unwrap();
    let output = graph
        .apply_module(&NonnegativeBiasAddModule::new(), &[product, bias])
        .unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate_global_nonnegative(&graph, output, &[x, weight]),
        Err(ProofError::UnsupportedGuarantee { node }) if node == bias.index()
    ));
}

#[test]
fn proof_generator_composes_global_nonnegative_recursively_through_nested_dag() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[3]), Domain::Finite);
    let y = graph.input(shape(&[3]), Domain::Finite);
    let z = graph.input(shape(&[3]), Domain::Finite);
    let output = graph
        .apply_module(&PropertyOuterModule::new(&[3]), &[x, y, z])
        .unwrap();
    assert_eq!(graph.modules().len(), 4);
    assert_eq!(graph.modules()[0].name, "add_module");
    assert_eq!(graph.modules()[1].name, "mul_module");
    assert_eq!(graph.modules()[2].name, "property_inner");
    assert_eq!(graph.modules()[3].name, "property_outer");

    let program = LeanProofEngine::default()
        .generate_global_nonnegative(&graph, output, &[x, y, z])
        .unwrap();
    let source = program.source();
    let inner = source
        .split("theorem module_2_global_nonnegative")
        .nth(1)
        .unwrap()
        .split("theorem module_3_global_nonnegative")
        .next()
        .unwrap();
    let outer = source
        .split("theorem module_3_global_nonnegative")
        .nth(1)
        .unwrap();
    assert!(inner.contains("module_0_global_nonnegative"));
    assert!(inner.contains("module_1_global_nonnegative"));
    assert!(outer.contains("module_2_global_nonnegative"));
    assert!(!outer.contains("module_0_global_nonnegative"));
    assert!(!outer.contains("module_1_global_nonnegative"));
    assert!(source.contains("have arch_module_3 := module_3_global_nonnegative"));
    assert!(!source.contains("have arch_module_0 :="));
    assert!(!source.contains("have arch_module_1 :="));
    assert!(!source.contains("have arch_module_2 :="));
    assert!(source.contains("exact arch_module_3"));
    assert!(!source.contains("sorry"));
    assert!(!source.contains("admit"));
}

#[test]
fn proof_generator_nested_global_nonnegative_rejects_undeclared_capture() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[3]), Domain::Finite);
    let captured = graph.input(shape(&[3]), Domain::Finite);
    let module = CapturingPropertyModule {
        contract: MathContract::new(vec![port(&[3], Domain::Any)], port(&[3], Domain::Any)),
        undeclared: captured,
    };
    let before = graph.nodes().len();
    assert!(matches!(
        graph.apply_module(&module, &[x]),
        Err(NeuralError::ContractViolation {
            detail: "undeclared external capture",
            ..
        }) | Err(NeuralError::ContractViolation {
            detail: "nested module captured undeclared input",
            ..
        })
    ));
    assert_eq!(graph.nodes().len(), before);
    assert!(graph.modules().is_empty());
}

#[test]
fn proof_generator_global_nonnegative_fails_closed_without_required_source_assumption() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Finite);
    let y = graph.input(shape(&[2]), Domain::Finite);
    let add = AddModule::new(&[2]);
    let output = graph.apply_module(&add, &[x, y]).unwrap();

    assert!(matches!(
        LeanProofEngine::default().generate_global_nonnegative(&graph, output, &[x]),
        Err(ProofError::UnsupportedGuarantee { node }) if node == y.index()
    ));
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
    assert!(ir.contains("add.rn.f32 $0, $1, $2;"));
    assert!(ir.contains("mul.rn.f32 $0, $1, $2;"));
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
    assert_eq!(
        matrix.bias_add(&bias).unwrap().values(),
        &[0., 2., 5., 3., 5., 8.]
    );
    assert!(matches!(
        matrix.bias_add(&Tensor::new(shape(&[2]), vec![1_f32, 2.0]).unwrap()),
        Err(NeuralError::BiasWidthMismatch { .. })
    ));
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 3]), Domain::Finite);
    let wrong = graph.input(shape(&[2]), Domain::Finite);
    let old_len = graph.nodes().len();
    assert!(matches!(
        graph.bias_add(a, wrong),
        Err(NeuralError::BiasWidthMismatch { .. })
    ));
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
    assert!(
        lean.source()
            .contains("VeriNeural.matmul (m := 2) (k := 2) (n := 3)")
    );
    assert!(
        lean.source()
            .contains("VeriNeural.biasAdd (m := 2) (n := 3)")
    );
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
    let y = x
        .add(&hidden.matmul(&w2).unwrap().bias_add(&b2).unwrap())
        .unwrap();
    assert_eq!(y.values(), &[5., 9., 13., 17.]);
}

#[test]
fn stage7_launch_config_rejects_invalid_threads() {
    for threads in [0, 1, 63, 65, 1024, u32::MAX] {
        assert!(matches!(
            GpuLaunchConfig::new(threads, ReluFusion::Experimental),
            Err(GpuError::InvalidLaunch { .. })
        ));
    }
    for threads in [64, 128, 256, 512] {
        let launch = GpuLaunchConfig::new(threads, ReluFusion::Disabled).unwrap();
        assert_eq!(launch.threads_per_block(), threads);
        assert_eq!(launch.relu_fusion(), ReluFusion::Disabled);
    }
    assert_eq!(GpuLaunchConfig::default().threads_per_block(), 256);
}

#[test]
fn stage7_exclusive_bias_relu_fusion_removes_only_safe_producer() {
    let mut graph = NeuralGraph::default();
    let matrix = graph.input(shape(&[2, 3]), Domain::Finite);
    let bias = graph.input(shape(&[3]), Domain::Finite);
    let sum = graph.bias_add(matrix, bias).unwrap();
    let result = graph.relu(sum).unwrap();

    let ordinary = GpuEngine::default().compile(&graph, result).unwrap();
    assert!(ordinary.fused_relus().is_empty());
    assert!(
        ordinary
            .llvm_ir()
            .contains(&format!("void @vn_node_{}", sum.index()))
    );

    let launch = GpuLaunchConfig::new(128, ReluFusion::Experimental).unwrap();
    let fused = GpuEngine::default()
        .with_launch_config(launch)
        .compile(&graph, result)
        .unwrap();
    assert_eq!(fused.launch_config(), launch);
    assert_eq!(fused.fused_relus().get(&result.index()), Some(&sum.index()));
    assert!(
        !fused
            .llvm_ir()
            .contains(&format!("void @vn_node_{}", sum.index()))
    );
    assert!(
        fused
            .llvm_ir()
            .contains(&format!("void @vn_node_{}", result.index()))
    );
    assert!(fused.llvm_ir().contains("urem i32 %idx, 3"));
    assert!(fused.llvm_ir().contains("%fused_result = select"));
    assert!(!fused.llvm_ir().contains("fast "));
    let proof = fused.fusion_lean_source().unwrap();
    assert!(proof.contains("theorem relu_bias_fusion"));
    assert!(!proof.contains("sorry"));
    assert!(!proof.contains("admit"));
    assert!(ordinary.fusion_lean_source().is_none());
}

#[test]
fn stage7_shared_producer_is_not_fused_or_discarded() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[4]), Domain::Finite);
    let y = graph.input(shape(&[4]), Domain::Finite);
    let product = graph.mul(x, y).unwrap();
    let positive = graph.relu(product).unwrap();
    let output = graph.add(product, positive).unwrap();
    let launch = GpuLaunchConfig::new(256, ReluFusion::Experimental).unwrap();
    let gpu = GpuEngine::default()
        .with_launch_config(launch)
        .compile(&graph, output)
        .unwrap();
    assert!(gpu.fused_relus().is_empty());
    assert!(
        gpu.llvm_ir()
            .contains(&format!("void @vn_node_{}", product.index()))
    );
    assert!(
        gpu.llvm_ir()
            .contains(&format!("void @vn_node_{}", positive.index()))
    );
}

#[test]
fn stage7_fusion_applies_to_add_and_mul_but_not_across_reshape() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 2]), Domain::Finite);
    let b = graph.input(shape(&[2, 2]), Domain::Finite);
    let sum = graph.add(a, b).unwrap();
    let add_relu = graph.relu(sum).unwrap();
    let product = graph.mul(add_relu, b).unwrap();
    let reshaped = graph.reshape(product, shape(&[4])).unwrap();
    let mul_relu = graph.relu(reshaped).unwrap();
    let gpu = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(64, ReluFusion::Experimental).unwrap())
        .compile(&graph, mul_relu)
        .unwrap();
    assert_eq!(gpu.fused_relus().len(), 1);
    assert_eq!(gpu.fused_relus().get(&add_relu.index()), Some(&sum.index()));
    assert!(!gpu.fused_relus().contains_key(&mul_relu.index()));

    let direct = graph.relu(product).unwrap();
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(64, ReluFusion::Experimental).unwrap())
        .compile(&graph, direct)
        .unwrap();
    assert_eq!(
        fused.fused_relus().get(&direct.index()),
        Some(&product.index())
    );
}

#[test]
fn cuda_graph_only_runtime_regression_guard() {
    let source = include_str!("cuda.rs");
    assert!(source.contains("CU_STREAM_NON_BLOCKING"));
    assert!(source.contains("cuStreamBeginCapture_v2"));
    assert!(source.contains("cuStreamEndCapture"));
    assert!(source.contains("cuGraphInstantiateWithFlags"));
    assert!(source.contains("cuGraphLaunch"));
    assert!(source.contains("cuStreamSynchronize"));
    assert!(source.contains("cuMemcpyHtoDAsync_v2"));
    assert!(source.contains("cuMemcpyDtoHAsync_v2"));
    // The GPU launch uses an owned stream, never null / stream 0.
    assert!(
        source.contains("                    self.stream,\n                    args.as_mut_ptr()")
    );
    assert!(
        !source.contains(
            "                    ptr::null_mut(),\n                    args.as_mut_ptr()"
        )
    );
    assert!(!source.contains("cuCtxSynchronize"));
    assert!(!source.contains("cuMemcpyHtoD_v2"));
    assert!(!source.contains("cuMemcpyDtoH_v2"));
}

#[test]
fn p0_module_rejects_foreign_output_even_with_matching_shape() {
    struct Borrowed {
        contract: MathContract,
        borrowed: NodeId,
    }
    impl NeuralModule for Borrowed {
        fn name(&self) -> &'static str {
            "borrowed"
        }
        fn contract(&self) -> &MathContract {
            &self.contract
        }
        fn build(&self, _: &mut NeuralGraph, _: &[NodeId]) -> Result<NodeId, NeuralError> {
            Ok(self.borrowed)
        }
    }
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Any);
    let module = Borrowed {
        contract: MathContract::new(vec![port(&[2], Domain::Any)], port(&[2], Domain::Any)),
        borrowed: x,
    };
    assert!(matches!(
        graph.apply_module(&module, &[x]),
        Err(NeuralError::ContractViolation { .. })
    ));
    assert_eq!(graph.nodes().len(), 1);
}

#[test]
fn p0_independent_claims_are_theorems_not_structural_inference() {
    struct Identity {
        contract: MathContract,
    }
    impl NeuralModule for Identity {
        fn name(&self) -> &'static str {
            "reshape_identity"
        }
        fn contract(&self) -> &MathContract {
            &self.contract
        }
        fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
            graph.reshape(inputs[0], shape(&[2]))
        }
    }
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Any);
    let module = Identity {
        contract: MathContract::new(vec![port(&[2], Domain::Any)], port(&[2], Domain::Any))
            .with_claim(crate::ContractClaim::EqualsInput(0)),
    };
    let out = graph.apply_module(&module, &[x]).unwrap();
    let proof = LeanProofEngine::default().generate(&graph, out).unwrap();
    assert!(proof.source().contains("theorem module_0_claim_0"));
    assert!(proof.source().contains("(node_1 x_0) = (node_0 x_0)"));
    let goal = crate::ProofGoal {
        name: "identity".into(),
        output: out,
        kind: crate::ProofGoalKind::EqualTo { other: x },
    };
    assert!(
        LeanProofEngine::default()
            .generate_goal(&graph, &goal)
            .unwrap()
            .source()
            .contains("architecture_property_identity")
    );
}

#[test]
fn p0_claim_is_not_marked_proven_for_unknown_nonnegativity() {
    struct Claimed {
        contract: MathContract,
    }
    impl NeuralModule for Claimed {
        fn name(&self) -> &'static str {
            "false_nonnegative"
        }
        fn contract(&self) -> &MathContract {
            &self.contract
        }
        fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
            graph.add(inputs[0], inputs[1])
        }
    }
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Any);
    let y = graph.input(shape(&[2]), Domain::Any);
    let contract = MathContract::new(vec![port(&[2], Domain::Any); 2], port(&[2], Domain::Any))
        .with_claim(crate::ContractClaim::Nonnegative);
    let output = graph.apply_module(&Claimed { contract }, &[x, y]).unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate(&graph, output),
        Err(ProofError::UnsupportedGuarantee { .. })
    ));
}

#[test]
fn p0_extensible_composite_primitive_reuses_ir_and_proof() {
    let mut graph = NeuralGraph::default();
    // Finite input is necessary for the FP32 NonNegativeFinite guarantee.
    let x = graph.input(shape(&[2]), Domain::Finite);
    let expr = crate::PrimitiveExpr::Apply(Operation::Relu, vec![crate::PrimitiveExpr::Port(0)]);
    let spec = crate::PrimitiveSpec::new(
        "consumer_relu",
        "1.0",
        MathContract::new(
            vec![port(&[2], Domain::Finite)],
            port(&[2], Domain::NonNegativeFinite),
        )
        .with_claim(crate::ContractClaim::Nonnegative),
        expr,
    )
    .unwrap();
    let out = spec.apply(&mut graph, &[x]).unwrap();
    assert_eq!(graph.modules()[0].semantic_version, "1.0");
    // Invalid inputs still fail closed instead of inventing FP32 finiteness.
    let unspecified = graph.input(shape(&[2]), Domain::Any);
    assert!(matches!(
        spec.apply(&mut graph, &[unspecified]),
        Err(NeuralError::DomainNotEstablished {
            required: Domain::Finite,
            actual: Domain::Any
        })
    ));
    assert!(
        LeanProofEngine::default()
            .generate(&graph, out)
            .unwrap()
            .source()
            .contains("module_0_claim_0")
    );
}

#[test]
fn p1_multi_output_ports_check_shapes_and_rollback() {
    struct Fork {
        contract: crate::MultiOutputContract,
    }
    impl crate::MultiOutputModule for Fork {
        fn name(&self) -> &'static str {
            "fork"
        }
        fn contract(&self) -> &crate::MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            g: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            Ok(vec![g.relu(inputs[0])?, g.mul(inputs[0], inputs[0])?])
        }
    }
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Any);
    let named = |name| crate::NamedPortContract::new(name, port(&[2], Domain::Any));
    let spec = Fork {
        contract: crate::MultiOutputContract {
            inputs: vec![named("x")],
            outputs: vec![named("activated"), named("square")],
        },
    };
    let outs = graph.apply_multi_module(&spec, &[x]).unwrap();
    assert_eq!(outs.len(), 2);
    assert_eq!(graph.multi_modules().len(), 1);
    assert_eq!(graph.multi_modules()[0].outputs, outs);
    assert_eq!(graph.nodes().len(), 3);
}

#[test]
fn p1_multi_output_rejects_hidden_capture_and_reuses_port_namespaces() {
    struct Hidden {
        contract: crate::MultiOutputContract,
        captured: NodeId,
    }
    impl crate::MultiOutputModule for Hidden {
        fn name(&self) -> &'static str {
            "hidden_multi"
        }
        fn contract(&self) -> &crate::MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            g: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            Ok(vec![g.add(inputs[0], self.captured)?])
        }
    }
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::Any);
    let foreign = graph.input(shape(&[2]), Domain::Any);
    let nc = || crate::NamedPortContract::new("state", port(&[2], Domain::Any));
    let module = Hidden {
        contract: crate::MultiOutputContract {
            inputs: vec![nc()],
            outputs: vec![nc()],
        },
        captured: foreign,
    };
    assert!(matches!(
        graph.apply_multi_module(&module, &[x]),
        Err(NeuralError::ContractViolation {
            detail: "undeclared external capture",
            ..
        })
    ));
    assert_eq!(graph.nodes().len(), 2);
    assert!(graph.multi_modules().is_empty());
}

#[test]
fn p1_multi_output_single_lean_artifact_checks_named_ports_and_downstream() {
    struct Transition {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for Transition {
        fn name(&self) -> &'static str {
            "consumer_transition"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            Ok(vec![
                graph.relu(inputs[0])?,
                graph.reshape(inputs[0], shape(&[2]))?,
            ])
        }
    }

    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let state = NamedPortContract {
        name: "next_state",
        port: port(&[2], Domain::NonNegativeFinite),
        claims: vec![ContractClaim::Nonnegative],
    };
    let readout = NamedPortContract {
        name: "readout",
        port: port(&[2], Domain::NonNegativeFinite),
        claims: vec![ContractClaim::EqualsInput(0)],
    };
    let transition = Transition {
        contract: MultiOutputContract {
            inputs: vec![NamedPortContract::new(
                "state",
                port(&[2], Domain::NonNegativeFinite),
            )],
            outputs: vec![state, readout],
        },
    };
    let outputs = graph.apply_multi_module(&transition, &[x]).unwrap();
    let instance = &graph.multi_modules()[0];
    assert_eq!(instance.output_named("next_state"), Some(outputs[0]));
    assert_eq!(instance.output_named("readout"), Some(outputs[1]));
    assert_eq!(instance.output_named("unknown"), None);

    // Both outputs are independently usable in a larger consumer DAG.
    let combined = graph.add(outputs[0], outputs[1]).unwrap();
    let program = LeanProofEngine::default()
        .generate_outputs(&graph, &[outputs[0], outputs[1], combined])
        .unwrap();
    for theorem in [
        "multi_0_input_0_shape",
        "multi_0_input_0_nonnegative",
        "multi_0_port_0_shape",
        "multi_0_port_0_nonnegative",
        "multi_0_port_0_claim_0",
        "multi_0_port_1_shape",
        "multi_0_port_1_nonnegative",
        "multi_0_port_1_claim_0",
        "node_3_semantics",
    ] {
        assert!(program.theorem_names().iter().any(|name| name == theorem), "{theorem}");
    }
    assert!(program.source().contains("(node_2 x_0) = (node_0 x_0)"));
    assert_eq!(
        program.theorem_names().len(),
        program.theorem_names().iter().collect::<std::collections::BTreeSet<_>>().len(),
    );
}

#[test]
fn p1_multi_output_proof_rejects_empty_duplicate_foreign_and_unchecked_input_claims() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let other_graph = NeuralGraph::default();
    let foreign = {
        let mut other_graph = other_graph;
        other_graph.input(shape(&[2]), Domain::NonNegativeFinite)
    };
    let proof = LeanProofEngine::default();
    assert!(matches!(
        proof.generate_outputs(&graph, &[]),
        Err(ProofError::InvalidOutputs)
    ));
    assert!(matches!(
        proof.generate_outputs(&graph, &[x, x]),
        Err(ProofError::InvalidOutputs)
    ));
    assert!(matches!(
        proof.generate_outputs(&graph, &[x, foreign]),
        Err(ProofError::InvalidGraph(NeuralError::InvalidNode))
    ));

    struct ClaimedInput {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for ClaimedInput {
        fn name(&self) -> &'static str {
            "unverified_input_predicate"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            Ok(vec![graph.relu(inputs[0])?])
        }
    }
    let module = ClaimedInput {
        contract: MultiOutputContract {
            inputs: vec![NamedPortContract {
                name: "x",
                port: port(&[2], Domain::NonNegativeFinite),
                claims: vec![ContractClaim::Nonnegative],
            }],
            outputs: vec![NamedPortContract::new(
                "result",
                port(&[2], Domain::NonNegativeFinite),
            )],
        },
    };
    assert!(matches!(
        graph.apply_multi_module(&module, &[x]),
        Err(NeuralError::ContractViolation {
            detail: "input port claims are unsupported",
            ..
        })
    ));
    assert_eq!(graph.nodes().len(), 1);
    assert!(graph.multi_modules().is_empty());
}
