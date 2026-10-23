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
