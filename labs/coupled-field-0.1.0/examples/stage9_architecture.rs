//! Stage 9 end-to-end research lab. All artifacts live under one explicit
//! root, defaulting to this lab package. This is an integrity-checked evidence
//! bundle, NOT a certified IEEE-754 or CUDA JIT/SASS equivalence proof.

use coupled_field_lab::build;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use verineural::{
    GpuEngine, GpuLaunchConfig, GpuProgram, LeanProgram, LeanProofEngine, ProofGoal, ReluFusion,
    TranslationPlan, snapshot_architecture,
};

const FILES: [&str; 18] = [
    "architecture.identity",
    "semantic.ir",
    "architecture.dot",
    "architecture.png",
    "architecture.lean",
    "composition.lean",
    "parameters.bits",
    "inputs.bits",
    "output.bits",
    "report.txt",
    "gpu/baseline/graph.ll",
    "gpu/baseline/graph.ptx",
    "gpu/baseline/translation.plan",
    "gpu/baseline/translation.report",
    "gpu/fused/graph.ll",
    "gpu/fused/graph.ptx",
    "gpu/fused/translation.plan",
    "gpu/fused/translation.report",
];

fn failed(reason: impl Into<String>) -> io::Error {
    io::Error::other(reason.into())
}
fn fingerprint(path: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(failed("sha256sum failed").into());
    }
    let text = String::from_utf8(output.stdout)?;
    let sum = text.split_whitespace().next().unwrap_or("");
    if sum.len() != 64 || !sum.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(failed("invalid SHA-256 checksum").into());
    }
    Ok(sum.to_owned())
}

fn checksums(root: &Path) -> Result<String, Box<dyn Error>> {
    let mut contents = String::new();
    for name in FILES {
        contents.push_str(&format!("{}  {name}\n", fingerprint(&root.join(name))?));
    }
    Ok(contents)
}

// Do not accept a syntactically valid, but unrelated, Lean theorem.
// Both the source and the transitive axiom dependencies are rechecked.
fn verify_lean_evidence(
    root: &Path,
    file: &str,
    expected: &LeanProgram,
) -> Result<(), Box<dyn Error>> {
    let path = root.join(file);
    if fs::read_to_string(&path)? != expected.source() {
        return Err(failed("saved Lean source differs from current Rust architecture").into());
    }
    let proof = LeanProofEngine::default();
    proof.check_file(&path)?;
    proof.audit_file(&path, expected.theorem_names())?;
    Ok(())
}

/// The lab declares this predicate once in ModelBuilder; consumers must not
/// reconstruct a potentially different theorem from an ad-hoc graph walk.
fn composition_goal(lab: &coupled_field_lab::Lab) -> Result<&ProofGoal, Box<dyn Error>> {
    lab.model
        .proof_goals()
        .get("coupled_branch_lower_bound")
        .ok_or_else(|| failed("missing declared coupled branch proof obligation").into())
}

fn verified_bundle(
    root: &Path,
    lab: &coupled_field_lab::Lab,
    program: &GpuProgram,
    expected: &[i64],
) -> Result<(), Box<dyn Error>> {
    let identity = format!(
        "id={}\nversion={}\n",
        lab.metadata().id,
        lab.metadata().version
    );
    if fs::read_to_string(root.join("architecture.identity"))? != identity {
        return Err(failed("bundle belongs to another architecture id/version").into());
    }
    let snapshot = snapshot_architecture(lab.graph(), lab.output(), program, &lab.metadata())?;
    snapshot.verify(root)?;
    if fs::read_to_string(root.join("inputs.bits"))? != lab.inputs_bits()
        || fs::read_to_string(root.join("parameters.bits"))? != lab.parameters_bits()
    {
        return Err(failed("saved parameter values or inputs have changed").into());
    }
    let expected_bits = expected
        .iter()
        .map(|&x| format!("{:08x}\n", (x as f32).to_bits()))
        .collect::<String>();
    if fs::read_to_string(root.join("output.bits"))? != expected_bits {
        return Err(failed("saved GPU output differs from exact Int oracle").into());
    }
    let proof = LeanProofEngine::default().generate_global_nonnegative(
        lab.graph(),
        lab.output(),
        &lab.source_nodes(),
    )?;
    if !proof
        .theorem_names()
        .iter()
        .any(|name| name == "architecture_global_nonnegative")
    {
        return Err(failed("architecture-level Lean obligation absent").into());
    }
    verify_lean_evidence(root, "architecture.lean", &proof)?;
    let relational =
        LeanProofEngine::default().generate_goal(lab.graph(), composition_goal(lab)?)?;
    verify_lean_evidence(root, "composition.lean", &relational)?;
    for label in ["baseline", "fused"] {
        let dir = root.join("gpu").join(label);
        let llvm = fs::read_to_string(dir.join("graph.ll"))?;
        let ptx = fs::read_to_string(dir.join("graph.ptx"))?;
        let saved = fs::read_to_string(dir.join("translation.plan"))?;
        let plan = TranslationPlan::from_text(&saved)?;
        let report = plan.validate(&llvm, &ptx)?;
        if fs::read_to_string(dir.join("translation.report"))? != report.to_text() {
            return Err(failed(format!("{label} translation evidence no longer validates")).into());
        }
        // Offline verification binds the full IR-derived expected plan, not
        // only a self-consistent set of saved LLVM/PTX/plan strings.
        let fusion = if label == "fused" {
            ReluFusion::Experimental
        } else {
            ReluFusion::Disabled
        };
        let original = GpuEngine::default()
            .with_launch_config(GpuLaunchConfig::new(256, fusion)?)
            .compile(lab.graph(), lab.output())?;
        if original.translation_plan()?.to_text() != saved || original.llvm_ir() != llvm {
            return Err(failed(format!(
                "{label} code/plan diverged from current Rust graph"
            ))
            .into());
        }
    }
    let report = format!(
        "architecture={}\nversion={}\nsemantic_ir_sha256={}\n\
         global_property=architecture_global_nonnegative\n\
         lean=proven-under-declared-nonnegative-input-assumptions\n\
         composition_property=architecture_property_coupled_branch_lower_bound\n\
         composition_lean=proven-ideal-Int-two-branch-bound\n\
         numeric_oracle=checked-integer-within-2^24\n\
         gpu=baseline-fused-bitwise-tested\n\
         translation=per-kernel-llvm-ptx-structural-validated\n\
         cuda_jit_sass=trusted-not-proven\n\
         parameter_values=explicitly-pinned-in-manifest\n\
         artifact_root=explicit-lab-local\n",
        lab.metadata().id,
        lab.metadata().version,
        snapshot.ir_sha256,
    );
    if fs::read_to_string(root.join("report.txt"))? != report {
        return Err(failed("assurance report mismatch").into());
    }
    if fs::read_to_string(root.join("SHA256SUMS"))? != checksums(root)? {
        return Err(failed("SHA-256 manifest is missing, incomplete or stale").into());
    }
    println!(
        "Stage 9 offline verification: Semantic IR, DOT, 15+ metadata attributes, Lean, baseline/fused translation, exact integer outputs and 18 SHA-256 artifacts PASS"
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut verify = false;
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--verify" => verify = true,
            "--artifact-root" => {
                index += 1;
                let path = args
                    .get(index)
                    .ok_or_else(|| failed("--artifact-root needs a path"))?;
                root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
            }
            other => return Err(failed(format!("unknown Stage 9 argument: {other}")).into()),
        }
        index += 1;
    }
    let lab = build()?;
    let expected = lab.exact_integer_output()?;
    // The independently declared source assumptions are validated for the
    // specific tensor values. The Lean theorem remains conditional.
    for (id, t) in &lab.inputs {
        t.validate_domain(lab.graph().node(*id)?.domain)?;
    }
    let baseline =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Disabled)?);
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental)?);
    let base_plan = baseline.compile(lab.graph(), lab.output())?;
    if verify {
        verified_bundle(&root, &lab, &base_plan, &expected)?;
        return Ok(());
    }
    fs::create_dir_all(&root)?;
    let identity = format!(
        "id={}\nversion={}\n",
        lab.metadata().id,
        lab.metadata().version
    );
    if root.join("architecture.identity").exists()
        && fs::read_to_string(root.join("architecture.identity"))? != identity
    {
        return Err(failed("different architecture already owns selected artifact root").into());
    }
    // A failed rerun must not leave a stale successful certificate.
    let _ = fs::remove_file(root.join("SHA256SUMS"));
    let _ = fs::remove_file(root.join("report.txt"));
    let proof = LeanProofEngine::default().generate_global_nonnegative(
        lab.graph(),
        lab.output(),
        &lab.source_nodes(),
    )?;
    if !proof
        .theorem_names()
        .iter()
        .any(|name| name == "architecture_global_nonnegative")
    {
        return Err(failed("architecture-level Lean obligation absent").into());
    }
    let (checked, _audit) =
        LeanProofEngine::default().verify_certified(&proof, &root.join("architecture.lean"))?;
    if checked.theorem_names.is_empty() {
        return Err(failed("Lean returned no checked theorems").into());
    }
    let relational =
        LeanProofEngine::default().generate_goal(lab.graph(), composition_goal(&lab)?)?;
    LeanProofEngine::default().verify_certified(&relational, &root.join("composition.lean"))?;
    let snapshot = snapshot_architecture(lab.graph(), lab.output(), &base_plan, &lab.metadata())?;
    snapshot.write(&root)?;
    fs::write(root.join("architecture.identity"), identity)?;
    fs::write(root.join("inputs.bits"), lab.inputs_bits())?;
    fs::write(root.join("parameters.bits"), lab.parameters_bits())?;
    let wanted = expected
        .iter()
        .map(|&x| (x as f32).to_bits())
        .collect::<Vec<_>>();
    for (label, engine, program) in [
        ("baseline", &baseline, base_plan.clone()),
        ("fused", &fused, fused.compile(lab.graph(), lab.output())?),
    ] {
        let actual = engine.execute(&program, &lab.inputs, &root.join("gpu").join(label))?;
        if actual
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
            != wanted
        {
            return Err(
                failed(format!("{label} GPU disagrees with checked integer oracle")).into(),
            );
        }
        if label == "fused" && program.removed_kernel_count() == 0 {
            return Err(failed("experimental fusion was not exercised").into());
        }
        println!(
            "Stage 9 {label}: {} exact integer values match independent i64 oracle; kernels={}, removed={}",
            wanted.len(),
            program.translation_plan()?.kernels.len(),
            program.removed_kernel_count()
        );
    }
    fs::write(
        root.join("output.bits"),
        wanted
            .iter()
            .map(|x| format!("{x:08x}\n"))
            .collect::<String>(),
    )?;
    fs::write(
        root.join("report.txt"),
        format!(
            "architecture={}\nversion={}\nsemantic_ir_sha256={}\n\
         global_property=architecture_global_nonnegative\n\
         lean=proven-under-declared-nonnegative-input-assumptions\n\
         composition_property=architecture_property_coupled_branch_lower_bound\n\
         composition_lean=proven-ideal-Int-two-branch-bound\n\
         numeric_oracle=checked-integer-within-2^24\n\
         gpu=baseline-fused-bitwise-tested\n\
         translation=per-kernel-llvm-ptx-structural-validated\n\
         cuda_jit_sass=trusted-not-proven\n\
         parameter_values=explicitly-pinned-in-manifest\n\
         artifact_root=explicit-lab-local\n",
            lab.metadata().id,
            lab.metadata().version,
            snapshot.ir_sha256
        ),
    )?;
    fs::write(root.join("SHA256SUMS"), checksums(&root)?)?;
    verified_bundle(&root, &lab, &base_plan, &expected)?;
    println!(
        "Stage 9 manifest SHA256 (pin externally): {}",
        fingerprint(&root.join("SHA256SUMS"))?
    );
    println!(
        "Stage 9 guarantee: global ideal-Int nonnegative theorem under explicit input assumptions; concrete GPU FP32 exact-integer probes; structural translation validation; CUDA JIT/SASS still trusted"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage9_rejects_swapped_lean_source_before_accepting_certificate() {
        let lab = build().unwrap();
        let expected = LeanProofEngine::default()
            .generate_global_nonnegative(lab.graph(), lab.output(), &lab.source_nodes())
            .unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-stage9-proof-replay-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("architecture.lean"),
            format!("{}\n-- swapped\n", expected.source()),
        )
        .unwrap();
        // No Lean installation is needed for this negative case: comparing
        // against the independently regenerated Rust proof fails first.
        assert!(verify_lean_evidence(&root, "architecture.lean", &expected).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
