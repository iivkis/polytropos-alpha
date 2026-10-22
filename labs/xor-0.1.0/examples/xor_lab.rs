//! End-to-end XOR lab: consumer graph -> independent Lean goal -> NVPTX
//! -> CUDA Graph execution -> bitwise truth table -> inspectable artifacts.
//! --verify is offline: it does not rerun the CUDA driver.

use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;
use verineural::{
    ArchitectureAutoCompiler, GpuEngine, GpuRequirement, LeanProofEngine, snapshot_architecture,
};
use xor_lab::{Lab, build, CASES};

const EVIDENCE: [&str; 10] = [
    "architecture.identity",
    "architecture.lean",
    "semantic.ir",
    "architecture.dot",
    "architecture.png",
    "inputs.bits",
    "parameters.bits",
    "output.bits",
    "report.txt",
    "auto-build/SHA256SUMS",
];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

fn digest(path: &Path) -> Result<String, Box<dyn Error>> {
    let result = Command::new("sha256sum").arg(path).output()?;
    if !result.status.success() {
        return Err(invalid(format!("sha256sum failed for {}", path.display())).into());
    }
    let stdout = String::from_utf8(result.stdout)?;
    let hash = stdout.split_whitespace().next().unwrap_or("");
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(invalid("invalid SHA-256 output").into());
    }
    Ok(hash.to_owned())
}

fn manifest(root: &Path) -> Result<String, Box<dyn Error>> {
    let mut text = String::new();
    for name in EVIDENCE {
        text.push_str(&format!("{}  {name}\n", digest(&root.join(name))?));
    }
    Ok(text)
}

fn expected_bits() -> String {
    Lab::expected()
        .iter()
        .map(|v| format!("{:08x}\n", v.to_bits()))
        .collect()
}

fn report(ir_sha256: &str) -> String {
    format!(
        "architecture=xor-mlp\nversion=0.1.0\nsemantic_ir_sha256={ir_sha256}\n\
         global_property=architecture_global_nonnegative\n\
         goal=xor_output_nonnegative\n\
         lean=two-ideal-Int-nonnegative-kernel-checked-theorems\n\
         xor_truth_table=4-of-4-bitwise-gpu-tested\n\
         fp32=concrete-small-integers-only\n\
         translation=llvm-ptx-structurally-validated\n\
         cuda_jit_sass=trusted-not-proven\n\
         training=not-performed-fixed-parameters\n"
    )
}

fn run(verify: bool) -> Result<(), Box<dyn Error>> {
    let root = Path::new("target/verineural/xor-lab");
    let lab = build()?;
    lab.validate_inputs()?;
    let compiler = ArchitectureAutoCompiler::default();
    let tasks = lab.tasks();
    let engine = GpuEngine::default();
    let program = engine.compile(&lab.graph, lab.output)?;
    let snapshot = snapshot_architecture(&lab.graph, lab.output, &program, &lab.metadata)?;
    // This exact global theorem is what ArchitectureMetadata advertises.
    // The auto compiler separately verifies the consumer's named goal.
    let global_proof = LeanProofEngine::default()
        .generate_global_nonnegative(&lab.graph, lab.output, &[])?;
    let lean = LeanProofEngine::default();
    if verify {
        compiler.verify_bundle(
            &lab.graph,
            &[lab.output],
            &tasks,
            GpuRequirement::AllOutputs,
            root,
            None,
        )?;
        snapshot.verify(root)?;
        let saved_global = root.join("architecture.lean");
        if fs::read_to_string(&saved_global)? != global_proof.source() {
            return Err(invalid("global Lean theorem differs from current Rust graph").into());
        }
        lean.check_file(&saved_global)?;
        lean.audit_file(&saved_global, global_proof.theorem_names())?;
        if fs::read_to_string(root.join("architecture.identity"))?
            != "id=xor-mlp\nversion=0.1.0\n"
            || fs::read_to_string(root.join("inputs.bits"))? != lab.input_bits()
            || fs::read_to_string(root.join("parameters.bits"))? != lab.parameter_bits()
            || fs::read_to_string(root.join("output.bits"))? != expected_bits()
            || fs::read_to_string(root.join("report.txt"))? != report(&snapshot.ir_sha256)
            || fs::read_to_string(root.join("SHA256SUMS"))? != manifest(root)?
        {
            return Err(invalid("XOR evidence mismatch; rerun lab-xor").into());
        }
        println!("XOR offline verification: IR/DOT/PNG, Lean axioms, PTX plan, input/parameter/output bits and SHA-256 PASS");
        return Ok(());
    }

    fs::create_dir_all(root)?;
    // Invalid reruns must not retain a previous successful report/manifest.
    for name in ["SHA256SUMS", "report.txt", "output.bits"] {
        let _ = fs::remove_file(root.join(name));
    }
    if root.join("architecture.identity").exists()
        && fs::read_to_string(root.join("architecture.identity"))?
            != "id=xor-mlp\nversion=0.1.0\n"
    {
        return Err(invalid("artifact directory belongs to a different architecture").into());
    }

    compiler.build(
        &lab.graph,
        &[lab.output],
        &tasks,
        GpuRequirement::AllOutputs,
        root,
    )?;
    lean.verify_certified(&global_proof, &root.join("architecture.lean"))?;
    snapshot.write(root)?;
    fs::write(root.join("architecture.identity"), "id=xor-mlp\nversion=0.1.0\n")?;
    fs::write(root.join("inputs.bits"), lab.input_bits())?;
    fs::write(root.join("parameters.bits"), lab.parameter_bits())?;

    // CUDA Graph uses the same Semantic IR; the oracle is Boolean XOR,
    // not a second hand-written implementation of the network forward pass.
    let actual = engine.execute(&program, &lab.inputs, &root.join("gpu"))?;
    let expected = Lab::expected();
    if actual.values().len() != CASES.len() {
        return Err(invalid("unexpected XOR result length").into());
    }
    for ((&(x, y), &observed), &want) in CASES.iter().zip(actual.values()).zip(&expected) {
        if observed.to_bits() != want.to_bits() {
            return Err(invalid(format!(
                "XOR({x},{y}) expected {want} ({:08x}), got {observed} ({:08x})",
                want.to_bits(),
                observed.to_bits()
            ))
            .into());
        }
        println!("{x} XOR {y} = {observed} (expected {want})");
    }

    fs::write(root.join("output.bits"), expected_bits())?;
    fs::write(root.join("report.txt"), report(&snapshot.ir_sha256))?;
    fs::write(root.join("SHA256SUMS"), manifest(root)?)?;
    println!("XOR 4/4 exact FP32 truth-table checks PASS; artifacts: {}", root.display());
    println!("Manifest SHA-256 (pin externally): {}", digest(&root.join("SHA256SUMS"))?);
    println!("Lean proves ideal-Int nonnegativity, not Boolean XOR or CUDA/SASS equivalence; weights are fixed, not trained.");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [] => run(false),
        [flag] if flag == "--verify" => run(true),
        _ => Err(invalid("usage: xor_lab [--verify]").into()),
    }
}
