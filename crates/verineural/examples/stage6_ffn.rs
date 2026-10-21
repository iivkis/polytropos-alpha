//! Stage 6: Residual FFN from one Rust graph to Lean + NVIDIA GPU + evidence.
use std::path::Path;

use verineural::{CertificateEngine, Domain, NeuralGraph, ResidualFfn, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2, 2])?;
    let x = Tensor::new(shape.clone(), vec![1.0_f32, 2.0, 3.0, 4.0])?;
    let w1 = Tensor::new(Shape::new(vec![2, 3])?, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
    let b1 = Tensor::new(Shape::new(vec![3])?, vec![-1.0, 0.0, 1.0])?;
    let w2 = Tensor::new(Shape::new(vec![3, 2])?, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let b2 = Tensor::new(Shape::new(vec![2])?, vec![0.0, 1.0])?;

    let mut graph = NeuralGraph::default();
    let x_id = graph.input(shape, Domain::Finite);
    let w1_id = graph.input(w1.shape().clone(), Domain::Finite);
    let b1_id = graph.input(b1.shape().clone(), Domain::Finite);
    let w2_id = graph.input(w2.shape().clone(), Domain::Finite);
    let b2_id = graph.input(b2.shape().clone(), Domain::Finite);

    let ffn = ResidualFfn::new(2, 2, 3)?;
    let output = graph.apply_module(&ffn, &[x_id, w1_id, b1_id, w2_id, b2_id])?;
    // CPU oracle is independently expressed in terms of concrete Tensor ops.
    let activated = x.matmul(&w1)?.bias_add(&b1)?.relu();
    let projected = activated.matmul(&w2)?.bias_add(&b2)?;
    let cpu = x.add(&projected)?;

    let engine = CertificateEngine::default();
    let report = engine.run(
        &graph,
        output,
        &[
            (x_id, x),
            (w1_id, w1),
            (b1_id, b1),
            (w2_id, w2),
            (b2_id, b2),
        ],
        &cpu,
        Path::new("target/verineural/stage6"),
    )?;
    engine.verify_bundle(&report.directory)?;
    println!(
        "Stage 6: Lean checked {} obligations, GPU matched {} CPU FP32 values",
        report.theorem_count, report.compared_fp32_values
    );
    println!("Report: {}/report.json", report.directory.display());
    println!("Pin manifest digest externally: {}", report.manifest_sha256);
    println!("Lean covers ideal Int/model obligations, not f32 or GPU machine code.");
    Ok(())
}
