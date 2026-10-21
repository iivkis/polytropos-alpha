//! Stage-2 example: Rust modules produce checked symbolic IR, not GPU code.
use verineural::{
    Domain, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, PortContract, Shape,
    Tensor,
};

struct PositiveLayer {
    contract: MathContract,
}

impl NeuralModule for PositiveLayer {
    fn name(&self) -> &'static str {
        "positive_layer"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.relu(inputs[0])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2, 2])?;
    let input_values = Tensor::new(shape.clone(), vec![-1.0f32, 4.0, 9.0, 16.0])?;
    // This independently checks the assumption used to construct the IR source.
    input_values.validate_domain(Domain::Finite)?;

    let mut graph = NeuralGraph::default();
    let input = graph.input(shape.clone(), Domain::Finite);
    let positive = PositiveLayer {
        contract: MathContract::new(
            vec![PortContract::new(shape.clone(), Domain::Finite)],
            PortContract::new(shape, Domain::NonNegativeFinite),
        ),
    };
    let hidden = graph.apply_module(&positive, &[input])?;
    let output = graph.sqrt(hidden)?;

    let expected = input_values.relu().sqrt()?;
    assert_eq!(expected.values(), &[0.0, 2.0, 3.0, 4.0]);
    println!("IR nodes: {}, output: {:?}", graph.nodes().len(), graph.node(output)?.shape);
    println!("Host reference output: {:?}", expected.values());
    Ok(())
}
