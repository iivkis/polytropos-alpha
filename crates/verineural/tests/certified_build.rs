//! Stage 10 tool-dependent acceptance. Run explicitly with --ignored in Nix.
//! Every test goes through the same public Model::certify API as consumers.
use std::fs;
use std::path::PathBuf;
use verineural::{
    AutoBuildError, CertifiedBuildConfig, Domain, Model, ModelBuilder, ProofGoal, ProofGoalKind,
};

fn model(id: &str) -> Model {
    let mut builder = ModelBuilder::new(id, "0.1.0");
    let x = builder.input("x", [2], Domain::NonNegativeFinite).unwrap();
    let y = builder
        .block("positive", &[x], |g, p| g.relu(p[0]))
        .unwrap();
    builder.output("y", y).unwrap();
    builder
        .proof_goal(ProofGoal::nonnegative("y_nonnegative", y, vec![]))
        .unwrap();
    builder.finish().unwrap()
}

fn temp_root(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "verineural-certified-{test_name}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

#[test]
#[ignore = "requires pinned Lean, llc, sha256sum, dot and CPU build tools"]
fn certified_build_replays_and_detects_modification() {
    let root = temp_root("replay");
    let m = model("replay");
    let config = CertifiedBuildConfig::new(&root);
    let built = m.certify(config.clone()).unwrap();
    assert_eq!(built.goal_theorems.len(), 1);
    assert!(built.directory.join("identity.txt").is_file());
    assert!(built.directory.join("architecture.dot").is_file());
    assert!(built.directory.join("architecture.png").is_file());
    assert!(root.join("semantic.ir").is_file());
    assert!(root.join("architecture.dot").is_file());
    assert!(root.join("architecture.png").is_file());
    assert!(root.join("architecture.lean").is_file());
    assert!(root.join("SHA256SUMS").is_file());
    let digest = built.manifest_sha256;
    m.verify_certified(config.clone(), Some(&digest)).unwrap();
    fs::write(built.directory.join("semantic.ir"), "tampered IR").unwrap();
    assert!(matches!(
        m.verify_certified(config, Some(&digest)),
        Err(AutoBuildError::Integrity(_))
    ));
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires pinned Lean, llc, sha256sum, dot; exercises Linux atomic reissue"]
fn certified_reissue_preserves_manifest_and_rejects_wrong_external_pin() {
    let root = temp_root("reissue");
    let _ = fs::remove_dir_all(&root);
    let architecture = model("reissue");
    let config = CertifiedBuildConfig::new(&root);
    let first = architecture.certify(config.clone()).unwrap();
    let first_digest = first.manifest_sha256;
    let first_manifest = fs::read(root.join("auto-build/SHA256SUMS")).unwrap();

    // A real second build executes the Linux renameat2 swap, unlike first publish.
    let second = architecture.certify(config.clone()).unwrap();
    assert_eq!(second.manifest_sha256, first_digest);
    assert_eq!(
        fs::read(root.join("auto-build/SHA256SUMS")).unwrap(),
        first_manifest
    );
    assert!(root.join("architecture.lean").is_file());
    assert!(root.join("architecture.png").is_file());
    assert_eq!(
        architecture
            .verify_certified(config.clone(), Some(&first_digest))
            .unwrap()
            .manifest_sha256,
        first_digest
    );
    assert!(matches!(
        architecture.verify_certified(config, Some(&"0".repeat(64))),
        Err(AutoBuildError::Integrity(_))
    ));
    assert!(fs::read_dir(&root).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".auto-build.staging-")
    }));
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires pinned Lean, llc, sha256sum and dot"]
fn foreign_identity_cannot_replace_published_bundle() {
    let root = temp_root("foreign");
    let first = model("first");
    let second = model("second");
    let config = CertifiedBuildConfig::new(&root);
    let report = first.certify(config.clone()).unwrap();
    let digest = report.manifest_sha256;
    assert!(matches!(
        second.certify(config.clone()),
        Err(AutoBuildError::Integrity(_))
    ));
    assert_eq!(
        first
            .verify_certified(config, Some(&digest))
            .unwrap()
            .manifest_sha256,
        digest
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[ignore = "requires pinned Lean/LLC/Graphviz toolchain"]
fn false_obligation_blocks_certificate_publication() {
    let root = temp_root("false-goal");
    let mut builder = ModelBuilder::new("false", "0.1.0");
    let x = builder.input("x", [2], Domain::Finite).unwrap();
    let y = builder.block("relu", &[x], |g, p| g.relu(p[0])).unwrap();
    builder.output("y", y).unwrap();
    // ReLU(x) = x is false for negative input values.
    builder
        .proof_goal(ProofGoal {
            name: "false_equality".into(),
            output: y,
            kind: ProofGoalKind::EqualTo { other: x },
        })
        .unwrap();
    let model = builder.finish().unwrap();
    assert!(model.certify(CertifiedBuildConfig::new(&root)).is_err());
    assert!(!root.join("auto-build/SHA256SUMS").exists());
    if root.exists() {
        fs::remove_dir_all(root).unwrap();
    }
}
