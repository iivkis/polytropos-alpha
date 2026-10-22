//! One CUDA Graph computes four declared outputs with one shared producer.
//! Lean certifies only the ideal-Int model, not NVIDIA FP32 machine behavior.
use std::path::Path;

use verineural::{Domain, GpuEngine, LeanProofEngine, NeuralGraph, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = NeuralGraph::default();
    let shape = Shape::new(vec![4])?;
    let x = graph.input(shape.clone(), Domain::Finite);
    let y = graph.input(shape.clone(), Domain::Finite);
    let common = graph.add(x, y)?;
    let activated = graph.relu(common)?;
    let weighted = graph.mul(common, y)?;
    let output = graph.add(activated, weighted)?;
    let outputs = [common, activated, weighted, output];
    let gpu = GpuEngine::default();
    let shared = gpu.compile_outputs(&graph, &outputs)?;
    let independent = outputs
        .iter()
        .map(|&id| gpu.compile(&graph, id)?.kernel_count())
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .sum::<usize>();
    println!(
        "Shared CUDA Graph: {} kernels vs {} independently compiled kernels",
        shared.kernel_count()?,
        independent
    );
    assert_eq!(shared.kernel_count()?, 4);
    assert!(independent > 4);

    let proof = LeanProofEngine::default().generate_outputs(&graph, &outputs)?;
    let lean = Path::new("target/verineural/multi-output-gpu/multi_output.lean");
    let (checked, axiom_audit) = LeanProofEngine::default().verify_certified(&proof, lean)?;
    println!(
        "Lean checked {} ideal-Int statements, audited {} theorem dependencies",
        checked.theorem_names.len(),
        axiom_audit.axiom_dependencies.len()
    );

    let input = |id, data| Tensor::new(shape.clone(), data).map(|value| (id, value));
    let inputs = [
        input(x, vec![1.0, 2.0, 3.0, 4.0])?,
        input(y, vec![2.0, 3.0, 4.0, 5.0])?,
    ];
    let dir = Path::new("target/verineural/multi-output-gpu/shared");
    let mut execution = gpu.prepare(&shared, &inputs, dir)?;
    execution.replay()?;
    let observed = execution.read_outputs()?;
    let expected: [&[f32]; 4] = [
        &[3.0, 5.0, 7.0, 9.0],
        &[3.0, 5.0, 7.0, 9.0],
        &[6.0, 15.0, 28.0, 45.0],
        &[9.0, 20.0, 35.0, 54.0],
    ];
    for ((id, tensor), (expected_id, oracle)) in
        observed.iter().zip(outputs.iter().zip(expected.iter()))
    {
        assert_eq!(id, expected_id);
        assert_eq!(tensor.values(), *oracle);
    }

    // Replay the exact same captured kernel DAG with a new input batch.
    let changed = [
        input(x, vec![1.0, 1.0, 1.0, 1.0])?,
        input(y, vec![1.0, 2.0, 3.0, 4.0])?,
    ];
    execution.update_inputs(&changed)?;
    execution.replay()?;
    let changed_outputs = execution.read_outputs()?;
    let next_expected: [&[f32]; 4] = [
        &[2.0, 3.0, 4.0, 5.0],
        &[2.0, 3.0, 4.0, 5.0],
        &[2.0, 6.0, 12.0, 20.0],
        &[4.0, 9.0, 16.0, 25.0],
    ];
    for ((id, tensor), (expected_id, oracle)) in changed_outputs
        .iter()
        .zip(outputs.iter().zip(next_expected.iter()))
    {
        assert_eq!(id, expected_id);
        assert_eq!(tensor.values(), *oracle);
    }
    println!("CUDA Graph captured once, replayed twice; four outputs match exact FP32 fixtures.");
    println!(
        "Assurance: conditional ideal-Int Lean; structurally validated PTX; concrete GPU bits tested, not formally proved."
    );
    Ok(())
}
