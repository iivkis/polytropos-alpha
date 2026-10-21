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
