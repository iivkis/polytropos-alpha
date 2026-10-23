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
    assert_eq!(graph.modules()[0].body_end, 2);
    assert_eq!(graph.modules()[1].body_start, 2);
    assert_eq!(graph.modules()[1].body_end, 3);
    assert_eq!(graph.nodes().len(), 3);
}

#[test]
fn closure_composition_owns_the_full_body_not_just_nodes_before_output() {
    let mut graph = NeuralGraph::default();
    let tensor_shape = shape(&[2]);
    let input = graph.input(tensor_shape.clone(), Domain::Finite);
    let declared_but_unused = graph.input(tensor_shape.clone(), Domain::Finite);
    let contract = MathContract::new(
        vec![
            PortContract::finite(tensor_shape.clone()),
            PortContract::finite(tensor_shape.clone()),
        ],
        PortContract::nonnegative(tensor_shape),
    );
    let output = graph
        .compose(
            "post_output",
            "1",
            &contract,
            &[input, declared_but_unused],
            |graph, inputs| {
                let output = graph.relu(inputs[0])?;
                let _diagnostic = graph.relu(output)?;
                Ok(output)
            },
        )
        .unwrap();

    let module = &graph.modules()[0];
    assert_eq!(module.output, output);
    assert_eq!(module.body_range(), 2..4);
    assert_eq!(graph.nodes().len(), 4);
    assert_eq!(
        graph.dependency_scope(&[output]).unwrap(),
        std::collections::BTreeSet::from([0, 1, 2])
    );
    check_architecture_connectivity(&graph).unwrap();
    assert!(graph.canonical_ir().contains("body=2..4"));
}

#[test]
fn module_instances_preserve_single_and_multi_application_order() {
    let mut graph = NeuralGraph::default();
    let tensor_shape = shape(&[2]);
    let input = graph.input(tensor_shape.clone(), Domain::Finite);
    let fork_contract = MultiOutputContract::new(
        vec![NamedPortContract::new(
            "input",
            PortContract::finite(tensor_shape.clone()),
        )],
        vec![
            NamedPortContract::new("left", PortContract::nonnegative(tensor_shape.clone())),
            NamedPortContract::new("right", PortContract::nonnegative(tensor_shape.clone())),
        ],
    );
    let fork = graph
        .compose_multi("fork", "1", &fork_contract, &[input], |graph, inputs| {
            Ok(vec![graph.relu(inputs[0])?, graph.relu(inputs[0])?])
        })
        .unwrap();
    let merge_contract = MathContract::new(
        vec![PortContract::any(tensor_shape.clone()); 2],
        PortContract::any(tensor_shape),
    );
    graph
        .compose("merge", "1", &merge_contract, &fork, |graph, inputs| {
            graph.add(inputs[0], inputs[1])
        })
        .unwrap();

    assert_eq!(
        graph
            .module_instances()
            .map(ModuleInstanceRef::name)
            .collect::<Vec<_>>(),
        vec!["fork", "merge"]
    );
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
    let expr = crate::PrimitiveExpr::port(0).relu();
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
        assert!(
            program.theorem_names().iter().any(|name| name == theorem),
            "{theorem}"
        );
    }
    assert!(program.source().contains("(node_2 x_0) = (node_0 x_0)"));
    assert_eq!(
        program.theorem_names().len(),
        program
            .theorem_names()
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
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

#[test]
fn p1_fixed_scan_unrolls_explicit_state_and_checks_each_step_in_lean() {
    struct Step {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for Step {
        fn name(&self) -> &'static str {
            "bounded_consumer_step"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            let next_state = graph.relu(inputs[0])?;
            let output = graph.add(next_state, inputs[1])?;
            Ok(vec![next_state, output])
        }
    }
    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape(&[2]), Domain::Finite);
    let x0 = graph.input(shape(&[2]), Domain::Any);
    let x1 = graph.input(shape(&[2]), Domain::Any);
    let transition = Step {
        contract: MultiOutputContract {
            inputs: vec![
                NamedPortContract::new("state", port(&[2], Domain::Finite)),
                NamedPortContract::new("input", port(&[2], Domain::Any)),
            ],
            outputs: vec![
                NamedPortContract {
                    name: "next_state",
                    port: port(&[2], Domain::NonNegativeFinite),
                    claims: vec![ContractClaim::Nonnegative],
                },
                NamedPortContract::new("output", port(&[2], Domain::Any)),
            ],
        },
    };
    let scan = graph.scan_fixed(&transition, initial, &[x0, x1]).unwrap();
    assert_eq!(scan.states.len(), 3);
    assert_eq!(scan.outputs.len(), 2);
    assert_eq!(scan.module_indices, vec![0, 1]);
    assert_eq!(scan.states[0], initial);
    assert_eq!(graph.multi_modules()[0].inputs, vec![initial, x0]);
    assert_eq!(graph.multi_modules()[1].inputs, vec![scan.states[1], x1]);

    // Every per-step state assertion is a separate Lean theorem. No claim
    // of an unbounded inductive invariant or GPU machine-code proof is made.
    let lean = LeanProofEngine::default()
        .generate_outputs(&graph, &scan.proof_outputs())
        .unwrap();
    for name in [
        "multi_0_port_0_claim_0",
        "multi_1_port_0_claim_0",
        "multi_0_port_1_shape",
        "multi_1_port_1_shape",
    ] {
        assert!(lean.theorem_names().iter().any(|theorem| theorem == name));
    }
}

#[test]
fn p1_fixed_scan_rolls_back_when_later_step_violates_contract() {
    struct Step {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for Step {
        fn name(&self) -> &'static str {
            "fail_on_later_input"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            Ok(vec![graph.relu(inputs[0])?, graph.relu(inputs[1])?])
        }
    }
    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape(&[2]), Domain::Finite);
    let first = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let second = graph.input(shape(&[2]), Domain::Any);
    let transition = Step {
        contract: MultiOutputContract {
            inputs: vec![
                NamedPortContract::new("state", port(&[2], Domain::Finite)),
                NamedPortContract::new("input", port(&[2], Domain::NonNegativeFinite)),
            ],
            outputs: vec![
                NamedPortContract::new("next_state", port(&[2], Domain::NonNegativeFinite)),
                NamedPortContract::new("output", port(&[2], Domain::NonNegativeFinite)),
            ],
        },
    };
    let count = graph.nodes().len();
    assert!(matches!(
        graph.scan_fixed(&transition, initial, &[first, second]),
        Err(NeuralError::DomainNotEstablished {
            required: Domain::NonNegativeFinite,
            actual: Domain::Any,
        })
    ));
    assert_eq!(graph.nodes().len(), count);
    assert!(graph.multi_modules().is_empty());

    assert!(matches!(
        graph.scan_fixed(&transition, initial, &[]),
        Err(NeuralError::ContractViolation {
            detail: "fixed scan requires at least one step",
            ..
        })
    ));
}

struct ConsumerAccumulation {
    contract: MultiOutputContract,
}

impl ConsumerAccumulation {
    fn new() -> Self {
        let any = port(&[2], Domain::Any);
        Self {
            contract: MultiOutputContract {
                inputs: vec![
                    NamedPortContract::new("state", any.clone()),
                    NamedPortContract::new("input", any.clone()),
                ],
                outputs: vec![
                    NamedPortContract {
                        name: "next_state",
                        port: any.clone(),
                        claims: vec![ContractClaim::Nonnegative],
                    },
                    NamedPortContract {
                        name: "output",
                        port: any,
                        claims: vec![ContractClaim::Nonnegative],
                    },
                ],
            },
        }
    }
}

impl MultiOutputModule for ConsumerAccumulation {
    fn name(&self) -> &'static str {
        "consumer_accumulation"
    }
    fn contract(&self) -> &MultiOutputContract {
        &self.contract
    }
    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        let next = graph.add(inputs[0], inputs[1])?;
        let readout = graph.relu(next)?;
        Ok(vec![next, readout])
    }
}

#[test]
fn p0_p1_ideal_claim_and_bounded_scan_invariant_compose_without_fp32_claim() {
    let mut graph = NeuralGraph::default();
    let state = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let x0 = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let x1 = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let scan = graph
        .scan_fixed(&ConsumerAccumulation::new(), state, &[x0, x1])
        .unwrap();
    // The graph conservatively refuses to treat finite source addition as
    // proven finite in FP32. The ideal-Int property is independently provable.
    assert_eq!(graph.node(scan.states[1]).unwrap().domain, Domain::Any);
    assert_eq!(graph.node(scan.states[2]).unwrap().domain, Domain::Any);

    let proof = LeanProofEngine::default()
        .generate_fixed_scan_nonnegative(&graph, &scan, "accumulate")
        .unwrap();
    for name in [
        "multi_0_port_0_claim_0",
        "multi_0_port_1_claim_0",
        "multi_1_port_0_claim_0",
        "multi_1_port_1_claim_0",
        "scan_accumulate_global_nonnegative",
    ] {
        assert!(
            proof.theorem_names().iter().any(|theorem| theorem == name),
            "{name}"
        );
    }
    assert!(proof.source().contains("VeriNeural.add_nonnegative"));
    assert!(proof.source().contains("exact multi_0_port_0_claim_0"));
    assert!(proof.source().contains("exact multi_1_port_0_claim_0"));
    assert!(proof.source().contains("And.intro scan_state_0"));
    assert_eq!(proof.assumptions().len(), 3);
    assert!(!proof.source().contains("sorry"));
    assert!(!proof.source().contains("admit"));
}

#[test]
fn p0_p1_bounded_scan_invariant_fails_closed_without_actual_proof() {
    let mut graph = NeuralGraph::default();
    let state = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let allowed = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let unknown = graph.input(shape(&[2]), Domain::Any);
    let transition = ConsumerAccumulation::new();
    // Construction succeeds because the numerical domain is Any. The
    // user-claimed ideal nonnegativity is *not* granted automatically.
    let scan = graph
        .scan_fixed(&transition, state, &[allowed, unknown])
        .unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate_fixed_scan_nonnegative(&graph, &scan, "bad"),
        Err(ProofError::UnsupportedGuarantee { .. })
    ));

    let mut invalid = scan.clone();
    invalid.module_indices[1] = invalid.module_indices[0];
    assert!(matches!(
        LeanProofEngine::default().generate_fixed_scan_nonnegative(&graph, &invalid, "bad"),
        Err(ProofError::InvalidScan)
    ));

    // A transition must explicitly request its invariant. Structural ports
    // with Domain::Any are not substitutes for mathematical claims.
    let mut no_claim = ConsumerAccumulation::new();
    no_claim.contract.outputs[0].claims.clear();
    let mut second_graph = NeuralGraph::default();
    let second_state = second_graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let second_x = second_graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let second_scan = second_graph
        .scan_fixed(&no_claim, second_state, &[second_x])
        .unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate_fixed_scan_nonnegative(
            &second_graph,
            &second_scan,
            "missing",
        ),
        Err(ProofError::InvalidScan)
    ));
    assert!(matches!(
        LeanProofEngine::default().generate_fixed_scan_nonnegative(
            &second_graph,
            &second_scan,
            "not-valid!",
        ),
        Err(ProofError::InvalidGoalName)
    ));
}

struct ParametricMatMul {
    signature: ParametricSignature,
}

impl ParametricMatMul {
    fn new() -> Self {
        let symbolic = |dims: Vec<Dimension>| {
            SymbolicPort::new(SymbolicShape::new(dims), Domain::NonNegativeFinite)
        };
        Self {
            signature: ParametricSignature {
                inputs: vec![
                    symbolic(vec![Dimension::Symbol("batch"), Dimension::Symbol("inner")]),
                    symbolic(vec![Dimension::Symbol("inner"), Dimension::Symbol("width")]),
                ],
                output: SymbolicPort::new(
                    SymbolicShape::new(vec![
                        Dimension::Symbol("batch"),
                        Dimension::Symbol("width"),
                    ]),
                    Domain::Any,
                ),
                constraints: vec![
                    DimensionConstraint::AtLeast("batch", 2),
                    DimensionConstraint::AtMost("inner", 16),
                    DimensionConstraint::MultipleOf("width", 2),
                ],
                claims: vec![ContractClaim::Nonnegative],
            },
        }
    }
}

impl ParametricModule for ParametricMatMul {
    fn name(&self) -> &'static str {
        "consumer_generic_matmul"
    }

    fn semantic_version(&self) -> &'static str {
        "1.0"
    }

    fn signature(&self) -> &ParametricSignature {
        &self.signature
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.matmul(inputs[0], inputs[1])
    }
}

#[test]
fn p1_symbolic_specializations_share_lean_and_gpu_lowering_without_source_edits() {
    let mut graph = NeuralGraph::default();
    let module = ParametricMatMul::new();
    let a0 = graph.input(shape(&[2, 3]), Domain::NonNegativeFinite);
    let b0 = graph.input(shape(&[3, 4]), Domain::NonNegativeFinite);
    let a1 = graph.input(shape(&[5, 3]), Domain::NonNegativeFinite);
    let b1 = graph.input(shape(&[3, 2]), Domain::NonNegativeFinite);
    let (out0, env0) = graph.apply_parametric_module(&module, &[a0, b0]).unwrap();
    let (out1, env1) = graph.apply_parametric_module(&module, &[a1, b1]).unwrap();
    assert_eq!(graph.node(out0).unwrap().shape, shape(&[2, 4]));
    assert_eq!(graph.node(out1).unwrap().shape, shape(&[5, 2]));
    assert_eq!(env0.get("batch"), Some(2));
    assert_eq!(env1.get("batch"), Some(5));
    assert_eq!(env0.get("inner"), Some(3));
    assert_eq!(env1.get("width"), Some(2));
    assert_eq!(env0.get("unknown"), None);
    assert_eq!(graph.modules().len(), 2);
    assert_eq!(graph.modules()[0].semantic_version, "1.0");
    assert_eq!(graph.modules()[1].semantic_version, "1.0");
    assert_eq!(graph.node(out0).unwrap().domain, Domain::Any);

    let proof = LeanProofEngine::default()
        .generate_outputs(&graph, &[out0, out1])
        .unwrap();
    for name in ["module_0_claim_0", "module_1_claim_0"] {
        assert!(proof.theorem_names().iter().any(|n| n == name));
    }
    assert!(proof.source().contains("VeriNeural.matmul_nonnegative"));
    let gpu = GpuEngine::default();
    for output in [out0, out1] {
        let program = gpu.compile(&graph, output).unwrap();
        assert!(program.llvm_ir().contains("define"));
        assert!(!program.translation_plan().unwrap().kernels.is_empty());
    }
}

#[test]
fn p1_symbolic_dimensions_fail_closed_on_axes_bounds_and_unbound_results() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(&[2, 3]), Domain::NonNegativeFinite);
    let incompatible = graph.input(shape(&[4, 4]), Domain::NonNegativeFinite);
    let too_wide = graph.input(shape(&[3, 5]), Domain::NonNegativeFinite);
    let valid = graph.input(shape(&[3, 4]), Domain::NonNegativeFinite);
    let module = ParametricMatMul::new();
    let baseline = (graph.nodes().len(), graph.modules().len());
    for inputs in [&[a, incompatible][..], &[a, too_wide][..]] {
        assert!(matches!(
            graph.apply_parametric_module(&module, inputs),
            Err(NeuralError::ContractViolation { .. })
        ));
    }
    assert_eq!((graph.nodes().len(), graph.modules().len()), baseline);

    let mut unbound = ParametricMatMul::new();
    unbound.signature.output.shape.dims[1] = Dimension::Symbol("never_bound");
    assert!(matches!(
        graph.apply_parametric_module(&unbound, &[a, valid]),
        Err(NeuralError::ContractViolation {
            detail: "output uses unbound dimension symbol",
            ..
        })
    ));
    let mut illegal_constraint = ParametricMatMul::new();
    illegal_constraint.signature.constraints = vec![DimensionConstraint::MultipleOf("width", 0)];
    assert!(matches!(
        graph.apply_parametric_module(&illegal_constraint, &[a, valid]),
        Err(NeuralError::ContractViolation {
            detail: "dimension constraint requires positive bound",
            ..
        })
    ));
    let mut unknown_constraint = ParametricMatMul::new();
    unknown_constraint.signature.constraints = vec![DimensionConstraint::AtLeast("unknown", 1)];
    assert!(matches!(
        graph.apply_parametric_module(&unknown_constraint, &[a, valid]),
        Err(NeuralError::ContractViolation {
            detail: "constraint uses unbound dimension symbol",
            ..
        })
    ));
    assert_eq!((graph.nodes().len(), graph.modules().len()), baseline);
}

#[test]
fn p1_symbolic_specialization_rolls_back_wrong_output_and_preserves_provenance() {
    struct WrongOutput {
        signature: ParametricSignature,
    }
    impl ParametricModule for WrongOutput {
        fn name(&self) -> &'static str {
            "wrong_symbolic_output"
        }
        fn semantic_version(&self) -> &'static str {
            "v1"
        }
        fn signature(&self) -> &ParametricSignature {
            &self.signature
        }
        fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
            graph.reshape(inputs[0], shape(&[6]))
        }
    }

    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[2, 3]), Domain::Any);
    let module = WrongOutput {
        signature: ParametricSignature {
            inputs: vec![SymbolicPort::new(
                SymbolicShape::new(vec![Dimension::Symbol("b"), Dimension::Fixed(3)]),
                Domain::Any,
            )],
            output: SymbolicPort::new(
                SymbolicShape::new(vec![Dimension::Symbol("b"), Dimension::Fixed(3)]),
                Domain::Any,
            ),
            constraints: vec![],
            claims: vec![],
        },
    };
    assert!(matches!(
        graph.apply_parametric_module(&module, &[x]),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    assert_eq!(graph.nodes().len(), 1);
    assert!(graph.modules().is_empty());
}

#[test]
fn p1_mask_select_primitives_validate_shapes_domains_and_host_values() {
    let tensor_shape = shape(&[5]);
    let mask = Tensor::new(tensor_shape.clone(), vec![-1.0, -0.0, 0.0, 2.0, f32::NAN]).unwrap();
    let yes = Tensor::new(
        tensor_shape.clone(),
        vec![1.0, 2.0, 3.0, f32::INFINITY, 5.0],
    )
    .unwrap();
    let no = Tensor::new(tensor_shape.clone(), vec![6.0, 7.0, 8.0, 9.0, -0.0]).unwrap();
    let binary = mask.positive_mask();
    assert_eq!(binary.values(), &[0.0, 0.0, 0.0, 1.0, 0.0]);
    let chosen = mask.select_positive(&yes, &no).unwrap();
    assert_eq!(chosen.values()[..4], [6.0, 7.0, 8.0, f32::INFINITY]);
    assert_eq!(chosen.values()[4].to_bits(), (-0.0f32).to_bits());

    let wrong = Tensor::new(shape(&[4]), vec![1.0; 4]).unwrap();
    assert!(matches!(
        mask.select_positive(&wrong, &no),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    let mut graph = NeuralGraph::default();
    let condition = graph.input(tensor_shape.clone(), Domain::Any);
    let true_branch = graph.input(tensor_shape.clone(), Domain::NonNegativeFinite);
    let false_branch = graph.input(tensor_shape.clone(), Domain::NonNegativeFinite);
    let mask_node = graph.positive_mask(condition).unwrap();
    assert_eq!(
        graph.node(mask_node).unwrap().domain,
        Domain::NonNegativeFinite
    );
    let selected = graph
        .select_positive(mask_node, true_branch, false_branch)
        .unwrap();
    assert_eq!(
        graph.node(selected).unwrap().domain,
        Domain::NonNegativeFinite
    );
    let other_shape = graph.input(shape(&[4]), Domain::Any);
    let nodes = graph.nodes().len();
    assert!(matches!(
        graph.select_positive(mask_node, true_branch, other_shape),
        Err(NeuralError::ShapeMismatch { .. })
    ));
    assert_eq!(graph.nodes().len(), nodes);

    let any_branch = graph.input(tensor_shape, Domain::Any);
    let unknown_output = graph
        .select_positive(mask_node, true_branch, any_branch)
        .unwrap();
    assert_eq!(graph.node(unknown_output).unwrap().domain, Domain::Any);
}

#[test]
fn p1_mask_select_lean_contract_and_numeric_gpu_profiles_fail_closed() {
    let mut graph = NeuralGraph::default();
    let condition = graph.input(shape(&[2]), Domain::Any);
    let yes = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let no = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let expr = PrimitiveExpr::port(0)
        .positive_mask()
        .select_positive(PrimitiveExpr::port(1), PrimitiveExpr::port(2));
    let module = PrimitiveSpec::new(
        "consumer_gated_select",
        "v1",
        MathContract::new(
            vec![
                port(&[2], Domain::Any),
                port(&[2], Domain::NonNegativeFinite),
                port(&[2], Domain::NonNegativeFinite),
            ],
            port(&[2], Domain::NonNegativeFinite),
        )
        .with_claim(ContractClaim::Nonnegative),
        expr,
    )
    .unwrap();
    let result = module.apply(&mut graph, &[condition, yes, no]).unwrap();
    let program = LeanProofEngine::default().generate(&graph, result).unwrap();
    assert!(program.source().contains("VeriNeural.positiveMask"));
    assert!(program.source().contains("VeriNeural.selectPositive"));
    assert!(
        program
            .source()
            .contains("VeriNeural.selectPositive_nonnegative")
    );
    assert!(
        program
            .theorem_names()
            .iter()
            .any(|n| n == "module_0_claim_0")
    );

    assert!(matches!(
        GpuEngine::default().compile(&graph, result),
        Err(GpuError::UnsupportedOperation {
            operation: "select_positive",
            ..
        })
    ));
    assert!(matches!(
        ExactFp32Plan::compile(&graph, result),
        Err(NumericError::UnsupportedOperation {
            operation: "select_positive",
            ..
        })
    ));
    assert!(matches!(
        RneFp32Plan::compile(&graph, result, 10),
        Err(NumericError::UnsupportedOperation {
            operation: "select_positive",
            ..
        })
    ));

    // A false ideal-Int claim may never be accepted merely because the
    // condition and one selected branch are nonnegative.
    let bad = graph.input(shape(&[2]), Domain::Any);
    let invalid_contract = MathContract::new(
        vec![
            port(&[2], Domain::Any),
            port(&[2], Domain::NonNegativeFinite),
            port(&[2], Domain::Any),
        ],
        port(&[2], Domain::Any),
    )
    .with_claim(ContractClaim::Nonnegative);
    let bad_module = PrimitiveSpec::new(
        "incorrect_gated_claim",
        "v1",
        invalid_contract,
        PrimitiveExpr::port(0).select_positive(PrimitiveExpr::port(1), PrimitiveExpr::port(2)),
    )
    .unwrap();
    let result = bad_module
        .apply(&mut graph, &[condition, yes, bad])
        .unwrap();
    assert!(matches!(
        LeanProofEngine::default().generate(&graph, result),
        Err(ProofError::UnsupportedGuarantee { .. })
    ));
}

#[test]
fn p1_recurrent_gated_select_proves_bounded_state_invariant() {
    struct Gate {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for Gate {
        fn name(&self) -> &'static str {
            "consumer_gated_transition"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            let mask = graph.positive_mask(inputs[1])?;
            let next = graph.select_positive(mask, inputs[1], inputs[0])?;
            let out = graph.relu(next)?;
            Ok(vec![next, out])
        }
    }

    let p = port(&[2], Domain::NonNegativeFinite);
    let named = |name| NamedPortContract::new(name, p.clone());
    let module = Gate {
        contract: MultiOutputContract {
            inputs: vec![named("state"), named("input")],
            outputs: vec![
                NamedPortContract {
                    name: "next_state",
                    port: p.clone(),
                    claims: vec![ContractClaim::Nonnegative],
                },
                named("output"),
            ],
        },
    };
    let mut graph = NeuralGraph::default();
    let s0 = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let x0 = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let x1 = graph.input(shape(&[2]), Domain::NonNegativeFinite);
    let scan = graph.scan_fixed(&module, s0, &[x0, x1]).unwrap();
    let proof = LeanProofEngine::default()
        .generate_fixed_scan_nonnegative(&graph, &scan, "gated")
        .unwrap();
    assert!(
        proof
            .theorem_names()
            .iter()
            .any(|n| n == "scan_gated_global_nonnegative")
    );
    assert!(
        proof
            .source()
            .contains("VeriNeural.selectPositive_nonnegative")
    );
    assert!(proof.source().contains("multi_1_port_0_claim_0"));
}
