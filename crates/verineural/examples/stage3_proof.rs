//! End-to-end Stage 3: build Rust IR, emit Lean 4, check all proof obligations.
use std::path::Path;

use verineural::{
    Domain, LeanProofEngine, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId,
    PortContract, Shape,
};

struct ReluLayer {
    contract: MathContract,
}

impl NeuralModule for ReluLayer {
    fn name(&self) -> &'static str {
        "relu"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, input: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.relu(input[0])
    }
}

struct Flatten {
    contract: MathContract,
}

impl NeuralModule for Flatten {
    fn name(&self) -> &'static str {
        "flatten"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, input: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.reshape(input[0], Shape::new(vec![4])?)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let square = Shape::new(vec![2, 2])?;
    let flat = Shape::new(vec![4])?;

    let mut graph = NeuralGraph::default();
    let input = graph.input(square.clone(), Domain::Finite);
    let relu = ReluLayer {
        contract: MathContract::new(
            vec![PortContract::new(square.clone(), Domain::Finite)],
            PortContract::new(square.clone(), Domain::NonNegativeFinite),
        ),
    };
    let flatten = Flatten {
        contract: MathContract::new(
            vec![PortContract::new(square, Domain::NonNegativeFinite)],
            PortContract::new(flat, Domain::NonNegativeFinite),
        ),
    };

    let hidden = graph.apply_module(&relu, &[input])?;
    let reshaped = graph.apply_module(&flatten, &[hidden])?;
    let square = graph.mul(reshaped, reshaped)?;
    let output = graph.add(square, reshaped)?;

    let engine = LeanProofEngine::default();
    let generated = engine.generate(&graph, output)?;
    let file = Path::new("target/verineural/verification/stage3.lean");
    let verified = engine.verify(&generated, file)?;
    println!(
        "Lean checked {} obligations in {}",
        verified.theorem_names.len(),
        verified.path.display()
    );
    println!(
        "Input hypotheses: {} (theorem remains conditional)",
        verified.assumptions.len()
    );
    Ok(())
}
