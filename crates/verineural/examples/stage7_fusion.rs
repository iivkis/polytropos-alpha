//! Stage 7: opt-in Relu fusion and controlled CUDA launch comparison.
//!
//! Both modes compile from the same Rust graph, check Lean, and compare every
//! returned f32 bit pattern against the same independent CPU reference.
//! The experimental ideal-Int fusion theorem is checked separately; LLVM/PTX
//! and floating-point equivalence are not formally established.
use std::fs;
use std::io;
use std::path::Path;
use std::time::Instant;

use verineural::{
    CertificateEngine, Domain, GpuEngine, GpuLaunchConfig, LeanProofEngine, NeuralGraph,
    ReluFusion, Shape, Tensor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2, 3])?;
    let x = Tensor::new(shape.clone(), vec![-3.0_f32, 0.0, 1.0, 2.0, -2.0, 5.0])?;
    let y = Tensor::new(shape.clone(), vec![0.5_f32, 2.0, -2.0, 1.0, 3.0, 0.0])?;
    let bias = Tensor::new(Shape::new(vec![3])?, vec![1.0_f32, -1.0, 2.0])?;

    // A user-defined composition rather than a prescribed LLM architecture.
    let mut graph = NeuralGraph::default();
    let xid = graph.input(shape.clone(), Domain::Finite);
    let yid = graph.input(shape, Domain::Finite);
    let bid = graph.input(bias.shape().clone(), Domain::Finite);
    let mixture = graph.add(xid, yid)?;
    let biased = graph.bias_add(mixture, bid)?;
    let output = graph.relu(biased)?;
    let reference = x.add(&y)?.bias_add(&bias)?.relu();
    let inputs = [(xid, x), (yid, y), (bid, bias)];
    let root = Path::new("target/verineural/stage7");
    fs::create_dir_all(root)?;

    // The intermediate bias-add node is exclusively consumed by the Relu.
    let baseline = GpuEngine::default().with_launch_config(
        GpuLaunchConfig::new(64, ReluFusion::Disabled)?,
    );
    let fused = GpuEngine::default().with_launch_config(
        GpuLaunchConfig::new(256, ReluFusion::Experimental)?,
    );
    let unoptimized_plan = baseline.compile(&graph, output)?;
    let fused_plan = fused.compile(&graph, output)?;
    if fused_plan.fused_relus().len() != 1 {
        return Err(io::Error::other("expected exactly one exclusive Relu fusion").into());
    }

    // Check the mathematical rewrite using the real Lean executable.
    // The proof is not a proof of code generation or IEEE-754 equivalence.
    let fusion_lean = root.join("fusion.lean");
    let source = fused_plan
        .fusion_lean_source()
        .ok_or_else(|| io::Error::other("no fusion lemma emitted"))?;
    fs::write(&fusion_lean, source)?;
    let proof = LeanProofEngine::default();
    proof.check_file(&fusion_lean)?;

    // These measurements include Lean checking, NVPTX compilation, JIT,
    // host transfers and CUDA work. They are NOT kernel-only benchmarks.
    let t0 = Instant::now();
    let baseline_cert = CertificateEngine::new(LeanProofEngine::default(), baseline);
    let baseline_report = baseline_cert.run(
        &graph, output, &inputs, &reference, &root.join("baseline"),
    )?;
    baseline_cert.verify_bundle(&baseline_report.directory)?;
    let baseline_duration = t0.elapsed();

    let t1 = Instant::now();
    let fused_cert = CertificateEngine::new(LeanProofEngine::default(), fused);
    let fused_report = fused_cert.run(
        &graph, output, &inputs, &reference, &root.join("fused"),
    )?;
    fused_cert.verify_bundle(&fused_report.directory)?;
    let fused_duration = t1.elapsed();

    let count_kernels = |text: &str| text.matches("define ptx_kernel void").count();
    println!(
        "Stage 7: kernels {} -> {}, guarded fusions {}",
        count_kernels(unoptimized_plan.llvm_ir()),
        count_kernels(fused_plan.llvm_ir()),
        fused_plan.fused_relus().len(),
    );
    println!("Launch: baseline 64 threads, fused 256 threads per block");
    println!(
        "CPU comparisons: baseline {} / fused {} identical f32 values",
        baseline_report.compared_fp32_values,
        fused_report.compared_fp32_values
    );
    println!("Lean ideal-Int rule checked: {}", fusion_lean.display());
    println!(
        "End-to-end validation wall time (NOT a GPU-only benchmark): {:?} vs {:?}",
        baseline_duration, fused_duration
    );
    println!(
        "SHA256SUMS digests: baseline {}, fused {}",
        baseline_report.manifest_sha256, fused_report.manifest_sha256
    );
    println!("The standalone fusion.lean is not included in the Stage-5-format SHA256SUMS.");
    println!("No certified PTX/FP32/machine-code equivalence is claimed.");
    Ok(())
}
