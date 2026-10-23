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
        Self { package_root: package_root.into(), artifact_root: None }
    }

    pub fn artifact_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.artifact_root = Some(root.into());
        self
    }

    /// Proof goals are always mandatory. There is no weaker certified mode.
    pub fn require_all_proofs(self) -> Self { self }

    /// GPU support is required for every declared output in certified builds.
    pub fn require_gpu_lowering(self) -> Self { self }

    pub fn resolved_root(&self) -> PathBuf {
        match &self.artifact_root {
            None => self.package_root.clone(),
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => self.package_root.join(path),
        }
    }

    pub fn package_root(&self) -> &Path { &self.package_root }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_roots_are_resolved_only_relative_to_the_package() {
        let package = Path::new("labs/model-0.1.0");
        assert_eq!(CertifiedBuildConfig::new(package).resolved_root(), package);
        assert_eq!(
            CertifiedBuildConfig::new(package).artifact_root("generated/run1").resolved_root(),
            package.join("generated/run1")
        );
        assert_eq!(
            CertifiedBuildConfig::new(package).artifact_root("/tmp/model-run1").resolved_root(),
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
        _ => Err(crate::AutoBuildError::Integrity("model identity lacks id/version".into())),
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
                        "unexpected external artifact symlink: {}", path.display()
                    )));
                }
            }
            Ok(meta) if meta.is_file() && legacy_ok => {}
            Ok(_) => return Err(crate::AutoBuildError::Integrity(format!(
                "refusing to replace unowned artifact: {}", path.display()
            ))),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

/// A stable set of relative links names one underlying bundle.
/// For legacy lab roots, old files are backed up on first migration.
#[cfg(unix)]
pub(crate) fn publish_root(root: &Path, identity: &str) -> Result<(), crate::AutoBuildError> {
    use std::os::unix::fs::symlink;
    preflight_root(root, identity)?;
    let backup = root.join(".verineural-pre-stage10");
    for (name, target) in ROOT_LINKS {
        let path = root.join(name);
        if let Ok(meta) = std::fs::symlink_metadata(&path) {
            if meta.file_type().is_symlink() { continue; }
            std::fs::create_dir_all(&backup)?;
            let destination = backup.join(name);
            if destination.exists() {
                return Err(crate::AutoBuildError::Integrity(format!(
                    "cannot back up existing artifact: {}", destination.display()
                )));
            }
            std::fs::rename(&path, destination)?;
        }
        symlink(target, path)?;
    }
    Ok(())
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
                "missing or replaced certified root link: {}", path.display()
            )));
        }
    }
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
        assert!(preflight_root(
            &root, "verineural-certified-model-v1\nid=foreign\nversion=1\n"
        ).is_err());
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
