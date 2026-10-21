//! Stage 7: opt-in Relu fusion and controlled CUDA launch comparison.
//!
//! Both modes compile from the same Rust graph, check Lean, and compare every
//! returned f32 bit pattern against the same independent CPU reference.
//! The experimental ideal-Int fusion theorem is checked separately; LLVM/PTX
//! and floating-point equivalence are not formally established.
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use verineural::{
    CertificateEngine, Domain, GpuEngine, GpuLaunchConfig, LeanProofEngine, NeuralGraph,
    ReluFusion, Shape, Tensor,
};

const STAGE7_ARTIFACTS: [&str; 3] = [
    "fusion.lean",
    "baseline/SHA256SUMS",
    "fused/SHA256SUMS",
];

fn digest(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(io::Error::other("sha256sum failed").into());
    }
    let text = String::from_utf8(output.stdout)?;
    let hash = text.split_whitespace().next().unwrap_or("");
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(io::Error::other("invalid sha256sum digest").into());
    }
    Ok(hash.to_owned())
}

/// No GPU run: verify all three pinned sub-artifacts, and rerun Lean.
fn verify_stage7(root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = root.join("SHA256SUMS");
    if let Ok(expected) = std::env::var("VERINEURAL_STAGE7_SHA256") {
        if digest(&manifest)? != expected {
            return Err(io::Error::other("Stage 7 pinned digest mismatch").into());
        }
    }
    let check = Command::new("sha256sum")
        .current_dir(root)
        .args(["--check", "SHA256SUMS"])
        .output()?;
    if !check.status.success() {
        return Err(io::Error::other(format!(
            "Stage 7 manifest rejected: {}",
            String::from_utf8_lossy(&check.stderr)
        )).into());
    }
    let proof = LeanProofEngine::default();
    proof.check_file(&root.join("fusion.lean"))?;
    for name in ["baseline", "fused"] {
        CertificateEngine::default().verify_bundle(&root.join(name))?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new("target/verineural/stage7");
    if std::env::args().any(|a| a == "--verify") {
        verify_stage7(root)?;
        println!("Stage 7: hashes and all saved Lean sources verified (no GPU replay)");
        return Ok(());
    }
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
    fs::create_dir_all(root)?;
    // Fail closed: a previous success manifest must not survive a bad rerun.
    for name in [
        "SHA256SUMS",
        "baseline/SHA256SUMS",
        "baseline/report.json",
        "fused/SHA256SUMS",
        "fused/report.json",
    ] {
        match fs::remove_file(root.join(name)) {
            Ok(()) => {}
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }

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
    let hashes = Command::new("sha256sum")
        .current_dir(root)
        .args(STAGE7_ARTIFACTS)
        .output()?;
    if !hashes.status.success() {
        return Err(io::Error::other("Stage 7 sha256sum generation failed").into());
    }
    fs::write(root.join("SHA256SUMS"), hashes.stdout)?;
    if let Err(err) = verify_stage7(root) {
        let _ = fs::remove_file(root.join("SHA256SUMS"));
        return Err(err);
    }

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
    println!("Pinned Stage 7 SHA256SUMS digest: {}", digest(&root.join("SHA256SUMS"))?);
    println!("Recheck without a GPU: task verify-gpu-stage7");
    println!("Stage 7 manifest covers fusion.lean and both Stage 5 evidence manifests.");
    println!("No certified PTX/FP32/machine-code equivalence is claimed.");
    Ok(())
}
