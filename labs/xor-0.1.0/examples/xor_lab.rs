//! End-to-end XOR lab: consumer graph -> independent Lean goal -> NVPTX
//! -> CUDA Graph execution -> bitwise truth table -> inspectable artifacts.
//! --verify is offline: it does not rerun the CUDA driver.

use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use verineural::{
    ArchitectureAutoCompiler, GpuEngine, GpuRequirement, LeanProofEngine, snapshot_architecture,
};
use xor_lab::{Lab, build, CASES};

const EVIDENCE: [&str; 14] = [
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
    "gpu/graph.ll",
    "gpu/graph.ptx",
    "gpu/translation.plan",
    "gpu/translation.report",
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

fn verify_gpu_files(root: &Path, program: &verineural::GpuProgram) -> Result<(), Box<dyn Error>> {
    let dir = root.join("gpu");
    let llvm = fs::read_to_string(dir.join("graph.ll"))?;
    let ptx = fs::read_to_string(dir.join("graph.ptx"))?;
    let saved_plan = fs::read_to_string(dir.join("translation.plan"))?;
    let plan = program.translation_plan()?;
    if llvm != program.llvm_ir() || saved_plan != plan.to_text() {
        return Err(invalid("executed GPU LLVM/translation plan differs from current graph").into());
    }
    let checked = plan.validate(&llvm, &ptx)?;
    if fs::read_to_string(dir.join("translation.report"))? != checked.to_text() {
        return Err(invalid("executed GPU translation evidence rejected").into());
    }
    Ok(())
}

fn run(verify: bool, root: &Path) -> Result<(), Box<dyn Error>> {
    const IDENTITY: &str = "id=xor-mlp\nversion=0.1.0\n";
    // The success marker is invalidated before any proof/compiler/GPU stage
    // runs. A failed rerun must not preserve a stale successful certificate.
    if !verify {
        if root.join("architecture.identity").exists()
            && fs::read_to_string(root.join("architecture.identity"))? != IDENTITY
        {
            return Err(invalid("artifact directory belongs to a different architecture").into());
        }
        fs::create_dir_all(root)?;
        for name in ["SHA256SUMS", "report.txt", "output.bits"] {
            let _ = fs::remove_file(root.join(name));
        }
    }

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
        verify_gpu_files(root, &program)?;
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
        println!("XOR offline verification: IR/DOT/PNG, global Lean/goal axioms, executed GPU LLVM/PTX and input/parameter/output bits, SHA-256 PASS");
        return Ok(());
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
    fs::write(root.join("architecture.identity"), IDENTITY)?;
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

    verify_gpu_files(root, &program)?;
    fs::write(root.join("output.bits"), expected_bits())?;
    fs::write(root.join("report.txt"), report(&snapshot.ir_sha256))?;
    fs::write(root.join("SHA256SUMS"), manifest(root)?)?;
    println!("XOR 4/4 exact FP32 truth-table checks PASS; artifacts: {}", root.display());
    println!("Manifest SHA-256 (pin externally): {}", digest(&root.join("SHA256SUMS"))?);
    println!("Lean proves ideal-Int nonnegativity, not Boolean XOR or CUDA/SASS equivalence; weights are fixed, not trained.");
    Ok(())
}

fn parse_cli(args: &[String]) -> Result<(bool, PathBuf), io::Error> {
    let package_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut root = package_root.clone();
    let mut verify = false;
    let mut custom_root = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--verify" if !verify => verify = true,
            "--artifact-root" if !custom_root => {
                index += 1;
                let path = args
                    .get(index)
                    .filter(|path| !path.is_empty() && !path.starts_with("--"))
                    .ok_or_else(|| invalid("--artifact-root requires a path"))?;
                root = package_root.join(path);
                custom_root = true;
            }
            other => return Err(invalid(format!("unknown or repeated XOR argument: {other}"))),
        }
        index += 1;
    }
    Ok((verify, root))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (verify, root) = parse_cli(&args)?;
    run(verify, &root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_root_defaults_to_package_and_accepts_overrides() {
        let (_, default_root) = parse_cli(&[]).unwrap();
        assert_eq!(default_root, PathBuf::from(env!("CARGO_MANIFEST_DIR")));

        let (verify, relative) = parse_cli(&[
            "--verify".to_owned(),
            "--artifact-root".to_owned(),
            "evidence".to_owned(),
        ])
        .unwrap();
        assert!(verify);
        assert_eq!(relative, default_root.join("evidence"));

        let absolute = default_root.join("absolute-test");
        let (_, selected) = parse_cli(&[
            "--artifact-root".to_owned(),
            absolute.to_string_lossy().into_owned(),
        ])
        .unwrap();
        assert_eq!(selected, absolute);

        assert!(parse_cli(&["--artifact-root".to_owned()]).is_err());
        assert!(parse_cli(&["--verify".to_owned(), "--verify".to_owned()]).is_err());
    }
}
