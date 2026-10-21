//! Stage 4 executable GPU path: one Rust module graph, CPU oracle and CUDA.
use std::path::Path;

use verineural::{
    Domain, GpuEngine, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, PortContract,
    Shape, Tensor,
};

struct ReluLayer {
    contract: MathContract,
}

impl NeuralModule for ReluLayer {
    fn name(&self) -> &'static str {
        "relu_layer"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.relu(inputs[0])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2, 3])?;
    let x = Tensor::new(shape.clone(), vec![-3.0f32, -0.0, 0.5, 2.0, 4.0, 7.0])?;
    let y = Tensor::new(shape.clone(), vec![2.0f32, 3.0, -1.0, 0.5, 2.0, 4.0])?;

    let mut graph = NeuralGraph::default();
    let input_x = graph.input(shape.clone(), Domain::Finite);
    let input_y = graph.input(shape.clone(), Domain::Finite);
    let relu = ReluLayer {
        contract: MathContract::new(
            vec![PortContract::new(shape.clone(), Domain::Finite)],
            PortContract::new(shape.clone(), Domain::NonNegativeFinite),
        ),
    };
    let active = graph.apply_module(&relu, &[input_x])?;
    let flattened = graph.reshape(active, Shape::new(vec![6])?)?;
    let restored = graph.reshape(flattened, shape)?;
    let product = graph.mul(restored, input_y)?;
    let result = graph.add(product, active)?;

    // Independent host reference, using the stage-2 Tensor operations.
    let cpu_active = x.relu();
    let cpu_result = cpu_active.mul(&y)?.add(&cpu_active)?;

    let engine = GpuEngine::default();
    let program = engine.compile(&graph, result)?;
    let output = engine.execute(
        &program,
        &[(input_x, x), (input_y, y)],
        Path::new("target/verineural/gpu-stage4"),
    )?;

    for (&actual, &expected) in output.values().iter().zip(cpu_result.values()) {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "GPU differs from CPU: actual={actual:?} expected={expected:?}"
            )
            .into());
        }
    }
    println!("GPU execution matched CPU for {} FP32 values", output.values().len());
    println!("LLVM IR and PTX: target/verineural/gpu-stage4/");
    println!("Scope: numerical validation, not a proof of the GPU machine code.");
    Ok(())
}
