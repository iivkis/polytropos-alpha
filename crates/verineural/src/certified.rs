//! One proof-gated entry point for finished models. The existing auto-builder
//! owns Lean checking, Graphviz rendering, GPU translation and SHA256SUMS.
//! These are proofs about ideal-Int math and structural translation, NOT
//! formal proofs about FP32 GPU/SASS execution.
use std::path::{Path, PathBuf};

/// A selected Cargo package root owns its own certified evidence directory.
/// Relative overrides are resolved against this package, never the process CWD.
#[derive(Clone, Debug)]
pub struct CertifiedBuildConfig {
    package_root: PathBuf,
    artifact_root: Option<PathBuf>,
}

impl CertifiedBuildConfig {
    pub fn new(package_root: impl Into<PathBuf>) -> Self {
        Self {
            package_root: package_root.into(),
            artifact_root: None,
        }
    }

    pub fn artifact_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.artifact_root = Some(root.into());
        self
    }

    /// Proof goals are always mandatory. There is no weaker certified mode.
    pub fn require_all_proofs(self) -> Self {
        self
    }

    /// GPU support is required for every declared output in certified builds.
    pub fn require_gpu_lowering(self) -> Self {
        self
    }

    pub fn resolved_root(&self) -> PathBuf {
        match &self.artifact_root {
            None => self.package_root.clone(),
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => self.package_root.join(path),
        }
    }

    pub fn package_root(&self) -> &Path {
        &self.package_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_roots_are_resolved_only_relative_to_the_package() {
        let package = Path::new("labs/model-0.1.0");
        assert_eq!(CertifiedBuildConfig::new(package).resolved_root(), package);
        assert_eq!(
            CertifiedBuildConfig::new(package)
                .artifact_root("generated/run1")
                .resolved_root(),
            package.join("generated/run1")
        );
        assert_eq!(
            CertifiedBuildConfig::new(package)
                .artifact_root("/tmp/model-run1")
                .resolved_root(),
            Path::new("/tmp/model-run1")
        );
    }
}

/// Human-facing artifact paths point to the single audited auto-build bundle.
/// Previous same-id/version lab snapshots are preserved when first migrated.
const ROOT_LINKS: [(&str, &str); 7] = [
    ("semantic.ir", "auto-build/semantic.ir"),
    ("architecture.dot", "auto-build/architecture.dot"),
    ("architecture.png", "auto-build/architecture.png"),
    ("architecture.lean", "auto-build/lean/architecture.lean"),
    ("SHA256SUMS", "auto-build/SHA256SUMS"),
    ("report.txt", "auto-build/report.txt"),
    ("identity.txt", "auto-build/identity.txt"),
];

fn root_identity(identity: &str) -> Result<(&str, &str), crate::AutoBuildError> {
    let id = identity.lines().find_map(|s| s.strip_prefix("id="));
    let version = identity.lines().find_map(|s| s.strip_prefix("version="));
    match (id, version) {
        (Some(id), Some(version)) => Ok((id, version)),
        _ => Err(crate::AutoBuildError::Integrity(
            "model identity lacks id/version".into(),
        )),
    }
}

pub(crate) fn preflight_root(root: &Path, identity: &str) -> Result<(), crate::AutoBuildError> {
    let (id, version) = root_identity(identity)?;
    let legacy_identity = root.join("architecture.identity");
    let legacy_ok = if legacy_identity.is_file() {
        std::fs::read_to_string(&legacy_identity)? == format!("id={id}\nversion={version}\n")
    } else {
        false
    };
    for (name, target) in ROOT_LINKS {
        let path = root.join(name);
        match std::fs::symlink_metadata(&path) {
            Ok(meta) if meta.file_type().is_symlink() => {
                if std::fs::read_link(&path)? != Path::new(target) {
                    return Err(crate::AutoBuildError::Integrity(format!(
                        "unexpected external artifact symlink: {}",
                        path.display()
                    )));
                }
            }
            Ok(meta) if meta.is_file() && legacy_ok => {
                let backup = root.join(".verineural-pre-stage10");
                if let Ok(backup_meta) = std::fs::symlink_metadata(&backup) {
                    if !backup_meta.is_dir() || backup_meta.file_type().is_symlink() {
                        return Err(crate::AutoBuildError::Integrity(
                            "legacy backup must be a real directory".into(),
                        ));
                    }
                }
                if std::fs::symlink_metadata(backup.join(name)).is_ok() {
                    return Err(crate::AutoBuildError::Integrity(format!(
                        "legacy backup already exists; refusing partial migration of {name}"
                    )));
                }
            }
            Ok(_) => {
                return Err(crate::AutoBuildError::Integrity(format!(
                    "refusing to replace unowned artifact: {}",
                    path.display()
                )));
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

/// A stable set of relative links names one underlying bundle. Root link
/// changes cannot be one atomic filesystem operation, but an unsuccessful
/// first migration must leave no partially installed links or moved files.
#[cfg(unix)]
fn publish_root_transaction(
    root: &Path,
    identity: &str,
    // Test-only fault boundary: inject an error after a fixed number of edits.
    interrupt_after: Option<usize>,
) -> Result<(), crate::AutoBuildError> {
    use std::os::unix::fs::symlink;

    preflight_root(root, identity)?;
    let backup = root.join(".verineural-pre-stage10");
    // (root name, whether an original regular file was moved, whether our
    // symlink was installed). Only ever undo our own changes.
    let mut updated: Vec<(&str, bool, bool)> = Vec::new();
    let outcome = (|| -> Result<(), crate::AutoBuildError> {
        for (name, target) in ROOT_LINKS {
            let path = root.join(name);
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.file_type().is_symlink() => continue,
                Ok(meta) if meta.is_file() => {
                    if interrupt_after == Some(updated.len()) {
                        return Err(crate::AutoBuildError::Integrity(
                            "interrupted root artifact migration".into(),
                        ));
                    }
                    std::fs::create_dir_all(&backup)?;
                    std::fs::rename(&path, backup.join(name))?;
                    updated.push((name, true, false));
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    if interrupt_after == Some(updated.len()) {
                        return Err(crate::AutoBuildError::Integrity(
                            "interrupted root artifact migration".into(),
                        ));
                    }
                    updated.push((name, false, false));
                }
                Ok(_) => {
                    return Err(crate::AutoBuildError::Integrity(format!(
                        "root artifact changed during publication: {}",
                        path.display()
                    )));
                }
                Err(error) => return Err(error.into()),
            }
            symlink(target, &path)?;
            if let Some(last) = updated.last_mut() {
                last.2 = true;
            }
        }
        Ok(())
    })();

    if let Err(error) = outcome {
        let mut rollback_error = None;
        for (name, old_file, installed) in updated.into_iter().rev() {
            let path = root.join(name);
            if installed {
                if let Err(err) = std::fs::remove_file(&path) {
                    rollback_error.get_or_insert(err);
                    continue;
                }
            }
            if old_file {
                if let Err(err) = std::fs::rename(backup.join(name), &path) {
                    rollback_error.get_or_insert(err);
                }
            }
        }
        return if let Some(rollback_error) = rollback_error {
            Err(crate::AutoBuildError::Integrity(format!(
                "root publication failed ({error}); rollback also failed ({rollback_error})"
            )))
        } else {
            Err(error)
        };
    }
    Ok(())
}

#[cfg(unix)]
pub(crate) fn publish_root(root: &Path, identity: &str) -> Result<(), crate::AutoBuildError> {
    publish_root_transaction(root, identity, None)
}

#[cfg(not(unix))]
pub(crate) fn publish_root(_root: &Path, _identity: &str) -> Result<(), crate::AutoBuildError> {
    Err(crate::AutoBuildError::Integrity(
        "Stage 10 root-level bundle links require Unix symlink support".into(),
    ))
}

pub(crate) fn verify_root(root: &Path, identity: &str) -> Result<(), crate::AutoBuildError> {
    preflight_root(root, identity)?;
    for (name, target) in ROOT_LINKS {
        let path = root.join(name);
        if std::fs::read_link(&path).ok().as_deref() != Some(Path::new(target)) {
            return Err(crate::AutoBuildError::Integrity(format!(
                "missing or replaced certified root link: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Shared Cargo example entry point for any consumer-defined model.
///
/// Architecture definitions belong in the consumer's src/lib.rs. An example
/// can call this function without implementing proof orchestration or CLI
/// parsing a second time.
pub fn certified_example_main(
    model: &crate::Model,
    package_root: impl Into<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut verify = false;
    let mut root: Option<PathBuf> = None;
    let mut pin: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--verify" if !verify => verify = true,
            "--artifact-root" if root.is_none() => {
                root = Some(args.next().ok_or("--artifact-root requires a path")?.into());
            }
            "--pinned-sha256" if pin.is_none() => {
                let digest = args.next().ok_or("--pinned-sha256 requires a digest")?;
                if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return Err("--pinned-sha256 requires exactly 64 hexadecimal characters".into());
                }
                pin = Some(digest);
            }
            _ => return Err(format!("unknown or repeated certified option: {arg}").into()),
        }
    }
    if pin.is_some() && !verify {
        return Err("--pinned-sha256 only applies to --verify".into());
    }
    let mut config = CertifiedBuildConfig::new(package_root)
        .require_all_proofs()
        .require_gpu_lowering();
    if let Some(root) = root {
        config = config.artifact_root(root);
    }
    let report = if verify {
        model.verify_certified(config, pin.as_deref())?
    } else {
        model.certify(config)?
    };
    println!(
        "{}",
        if verify {
            "Certified replay passed"
        } else {
            "Certified build published"
        }
    );
    println!("Evidence: {}", report.directory.display());
    println!("Lean checker: {}", report.lean_version);
    println!("Mandatory goals verified: {}", report.goal_theorems.len());
    println!("SHA256SUMS: {} (pin externally)", report.manifest_sha256);
    println!(
        "Assurance: Lean ideal-Int proofs and GPU structural translation; NOT FP32/SASS machine proof"
    );
    Ok(())
}

#[cfg(all(test, unix))]
mod root_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "verineural-root-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn only_same_identity_legacy_artifacts_can_be_migrated() {
        let root = test_root("migration");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("architecture.identity"), "id=lab\nversion=1\n").unwrap();
        std::fs::write(root.join("architecture.dot"), "legacy dot").unwrap();
        let identity = "verineural-certified-model-v1\nid=lab\nversion=1\n";
        preflight_root(&root, identity).unwrap();
        assert!(
            preflight_root(
                &root,
                "verineural-certified-model-v1\nid=foreign\nversion=1\n"
            )
            .is_err()
        );
        publish_root(&root, identity).unwrap();
        assert_eq!(
            std::fs::read_link(root.join("architecture.dot")).unwrap(),
            Path::new("auto-build/architecture.dot")
        );
        assert_eq!(
            std::fs::read_to_string(root.join(".verineural-pre-stage10/architecture.dot")).unwrap(),
            "legacy dot"
        );
        verify_root(&root, identity).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn migration_fails_before_mutation_if_a_backup_conflicts() {
        let root = test_root("backup-conflict");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".verineural-pre-stage10")).unwrap();
        std::fs::write(root.join("architecture.identity"), "id=lab\nversion=1\n").unwrap();
        std::fs::write(root.join("architecture.dot"), "original").unwrap();
        std::fs::write(root.join("architecture.png"), "original png").unwrap();
        std::fs::write(
            root.join(".verineural-pre-stage10/architecture.png"),
            "previous",
        )
        .unwrap();
        let identity = "verineural-certified-model-v1\nid=lab\nversion=1\n";
        assert!(publish_root(&root, identity).is_err());
        assert_eq!(
            std::fs::read_to_string(root.join("architecture.dot")).unwrap(),
            "original"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("architecture.png")).unwrap(),
            "original png"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_legacy_migration_restores_every_original_file() {
        let root = test_root("rollback-legacy");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("architecture.identity"), "id=lab\nversion=1\n").unwrap();
        std::fs::write(root.join("architecture.dot"), "old dot").unwrap();
        std::fs::write(root.join("architecture.png"), "old png").unwrap();
        let identity = "verineural-certified-model-v1\nid=lab\nversion=1\n";
        assert!(publish_root_transaction(&root, identity, Some(2)).is_err());
        assert_eq!(std::fs::read_to_string(root.join("architecture.dot")).unwrap(), "old dot");
        assert_eq!(std::fs::read_to_string(root.join("architecture.png")).unwrap(), "old png");
        assert!(!root.join("semantic.ir").exists());
        assert!(!root.join(".verineural-pre-stage10/architecture.dot").exists());
        publish_root(&root, identity).unwrap();
        verify_root(&root, identity).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_fresh_publication_clears_all_new_links() {
        let root = test_root("rollback-fresh");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let identity = "verineural-certified-model-v1\nid=lab\nversion=1\n";
        assert!(publish_root_transaction(&root, identity, Some(3)).is_err());
        for (name, _) in ROOT_LINKS {
            assert!(std::fs::symlink_metadata(root.join(name)).is_err());
        }
        publish_root(&root, identity).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reject_external_root_link_and_unowned_file() {
        let root = test_root("rogue");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let identity = "verineural-certified-model-v1\nid=lab\nversion=1\n";
        std::fs::write(root.join("semantic.ir"), "other").unwrap();
        assert!(preflight_root(&root, identity).is_err());
        std::fs::remove_file(root.join("semantic.ir")).unwrap();
        symlink("/tmp/outside/ir", root.join("semantic.ir")).unwrap();
        assert!(preflight_root(&root, identity).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
