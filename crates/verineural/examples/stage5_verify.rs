//! Recheck a saved Stage 5 bundle without requiring a GPU run.
//! Pin the SHA256SUMS digest from the original run separately if authenticity matters.
use std::path::Path;

use verineural::CertificateEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = Path::new("target/verineural/stage5");
    let engine = CertificateEngine::default();
    match std::env::var("VERINEURAL_STAGE5_SHA256") {
        Ok(expected) => {
            engine.verify_bundle_with_digest(dir, &expected)?;
            println!("Pinned manifest digest: PASS");
        }
        Err(std::env::VarError::NotPresent) => {
            engine.verify_bundle(dir)?;
            println!("Unpinned manifest: integrity only, provenance NOT verified");
        }
        Err(error) => return Err(Box::new(error)),
    }
    println!("Bundle hash consistency: PASS");
    println!("Saved Lean source: rechecked by Lean 4");
    println!("No PTX semantic-preservation or GPU execution proof is implied.");
    Ok(())
}
