//! Stage 10 acceptance: the architecture lives exclusively in src/lib.rs.
//! This example only selects the output location and invokes VeriNeural.
use std::error::Error;
use std::path::PathBuf;
use verineural::CertifiedBuildConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mut verify = false;
    let mut root: Option<PathBuf> = None;
    let mut pin: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--verify" if !verify => verify = true,
            "--artifact-root" if root.is_none() => {
                root = Some(args.next().ok_or("--artifact-root needs a path")?.into());
            }
            "--pinned-sha256" if pin.is_none() => {
                pin = Some(args.next().ok_or("--pinned-sha256 needs a digest")?);
            }
            _ => return Err(format!("unknown or repeated option: {arg}").into()),
        }
    }
    if pin.is_some() && !verify {
        return Err("--pinned-sha256 only applies to --verify".into());
    }
    let model = coupled_field_lab::architecture()?;
    let mut config = CertifiedBuildConfig::new(env!("CARGO_MANIFEST_DIR"))
        .require_all_proofs().require_gpu_lowering();
    if let Some(path) = root {
        config = config.artifact_root(path);
    }
    let report = if verify {
        model.verify_certified(config, pin.as_deref())?
    } else {
        model.certify(config)?
    };
    println!("{}", if verify { "Certified replay passed" } else { "Certified build published" });
    println!("Evidence: {}", report.directory.display());
    println!("Lean checker: {}", report.lean_version);
    println!("Mandatory goals verified: {}", report.goal_theorems.len());
    println!("SHA256SUMS: {} (pin externally)", report.manifest_sha256);
    println!("Assurance: ideal-Int/Lean + GPU structural translation; NOT FP32/SASS machine proof");
    Ok(())
}
