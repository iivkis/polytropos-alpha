//! Integrity and current-source binding for the developer-run auto build.
//! SHA256SUMS proves artifact integrity relative to a separately pinned digest;
//! it is not a signature and cannot prove NVIDIA machine semantics.
use std::fs;
use std::path::Path;

use crate::auto_build::{AutoBuildError, AutoBuildPlan, PlannedGpu};

const MANIFEST: &str = "SHA256SUMS";

pub(super) fn hash_file(path: &Path) -> Result<String, AutoBuildError> {
    crate::certificate::hash_file(path).map_err(|err| AutoBuildError::Integrity(err.to_string()))
}

/// The expected artifact set is derived from the current plan, never from
/// user-controlled manifest lines. Paths are relative to auto-build/.
pub(super) fn artifact_files(plan: &AutoBuildPlan) -> Vec<String> {
    let mut files = vec![
        "semantic.ir".to_owned(),
        "architecture.dot".to_owned(),
        "architecture.png".to_owned(),
        "report.txt".to_owned(),
        "lean/architecture.lean".to_owned(),
        "lean/audit.txt".to_owned(),
    ];
    if plan.identity.is_some() {
        files.push("identity.txt".to_owned());
    }
    for (index, (name, _)) in plan.goals.iter().enumerate() {
        files.push(format!("lean/goal_{index}_{name}.lean"));
    }
    for (index, item) in plan.gpu.iter().enumerate() {
        if matches!(item, PlannedGpu::Supported { .. }) {
            for name in [
                "graph.ll",
                "graph.ptx",
                "translation.plan",
                "translation.report",
                "capability.plan",
            ] {
                files.push(format!("gpu_{index}/{name}"));
            }
        }
    }
    if plan.shared_gpu.is_some() {
        for name in [
            "graph.ll",
            "graph.ptx",
            "translation.plan",
            "translation.report",
            "capability.plan",
        ] {
            files.push(format!("gpu_shared/{name}"));
        }
    }
    files.sort();
    files
}

/// Reject foreign or legacy bundles before starting a new staged build.
/// Identity is checked again during replay, and SHA256SUMS covers its bytes.
pub(super) fn check_existing_identity(dir: &Path, expected: &str) -> Result<(), AutoBuildError> {
    match fs::symlink_metadata(dir) {
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
        Ok(meta) if meta.file_type().is_symlink() => {
            return Err(AutoBuildError::Integrity(
                "certified bundle directory cannot be a symlink".into(),
            ));
        }
        Ok(_) => {}
    }
    if !dir.is_dir() {
        return Err(AutoBuildError::Integrity(format!(
            "certified output is not a directory: {}",
            dir.display()
        )));
    }
    let saved = fs::read_to_string(dir.join("identity.txt")).map_err(|_| {
        AutoBuildError::Integrity("existing bundle lacks a readable certified identity".into())
    })?;
    if saved != expected {
        return Err(AutoBuildError::Integrity(
            "existing bundle belongs to a different model identity or declaration".into(),
        ));
    }
    Ok(())
}

fn manifest_text(dir: &Path, plan: &AutoBuildPlan) -> Result<String, AutoBuildError> {
    let mut text = String::new();
    for name in artifact_files(plan) {
        let path = dir.join(&name);
        if !path.is_file() {
            return Err(AutoBuildError::Integrity(format!(
                "required auto-build artifact missing: {name}"
            )));
        }
        text.push_str(&format!("{}  {name}\n", hash_file(&path)?));
    }
    Ok(text)
}

pub(super) fn write_manifest(dir: &Path, plan: &AutoBuildPlan) -> Result<String, AutoBuildError> {
    fs::write(dir.join(MANIFEST), manifest_text(dir, plan)?)?;
    hash_file(&dir.join(MANIFEST))
}

pub(super) fn verify_manifest(
    dir: &Path,
    plan: &AutoBuildPlan,
    expected_digest: Option<&str>,
) -> Result<String, AutoBuildError> {
    let existing = fs::read_to_string(dir.join(MANIFEST))?;
    if let Some(identity) = &plan.identity {
        check_existing_identity(dir, identity)?;
    }
    let expected = manifest_text(dir, plan)?;
    if existing != expected {
        return Err(AutoBuildError::Integrity(
            "SHA256SUMS does not match the complete expected artifact set".into(),
        ));
    }
    let digest = hash_file(&dir.join(MANIFEST))?;
    if expected_digest.is_some_and(|pinned| pinned != digest) {
        return Err(AutoBuildError::Integrity(
            "externally pinned manifest digest mismatch".into(),
        ));
    }
    Ok(digest)
}

pub(super) fn verify_current_ir(dir: &Path, source: &str) -> Result<(), AutoBuildError> {
    if fs::read_to_string(dir.join("semantic.ir"))? != source {
        return Err(AutoBuildError::Integrity(
            "saved auto-build IR differs from current Rust architecture".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ArchitectureAutoCompiler, Domain, GpuRequirement, NeuralGraph, NodeId, Shape};

    fn graph() -> (NeuralGraph, NodeId) {
        let mut graph = NeuralGraph::default();
        let x = graph.input(Shape::new(vec![2]).unwrap(), Domain::NonNegativeFinite);
        let output = graph.relu(x).unwrap();
        (graph, output)
    }

    #[test]
    fn canonical_ir_is_graph_identity_independent_and_semantic() {
        let (g1, _) = graph();
        let (g2, _) = graph();
        let ir1 = g1.canonical_ir();
        assert_eq!(ir1, g2.canonical_ir());
        assert!(ir1.starts_with("verineural-semantic-ir-v3\n"));
        assert!(!ir1.contains("MathContract {"));
        assert!(!ir1.contains("FP32Policy {"));

        let mut other = NeuralGraph::default();
        let x = other.input(Shape::new(vec![3]).unwrap(), Domain::NonNegativeFinite);
        let _y = other.relu(x).unwrap();
        assert_ne!(ir1, other.canonical_ir());
    }

    #[test]
    fn auto_manifest_detects_tampering_and_rejects_unpinned_or_wrong_artifacts() {
        let (graph, output) = graph();
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &[], GpuRequirement::AllOutputs)
            .unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-auto-bundle-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        let files = artifact_files(&plan);
        assert!(files.contains(&"gpu_0/graph.ptx".to_owned()));
        assert!(files.contains(&"architecture.dot".to_owned()));
        assert!(files.contains(&"architecture.png".to_owned()));
        for file in &files {
            let path = root.join(file);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(path, file.as_bytes()).unwrap();
        }
        let digest = write_manifest(&root, &plan).unwrap();
        assert_eq!(
            verify_manifest(&root, &plan, Some(&digest)).unwrap(),
            digest
        );
        assert!(matches!(
            verify_manifest(&root, &plan, Some(&"0".repeat(64))),
            Err(AutoBuildError::Integrity(_))
        ));
        assert!(matches!(
            verify_current_ir(&root, "a different current graph"),
            Err(AutoBuildError::Integrity(_))
        ));
        std::fs::write(root.join("gpu_0/graph.ptx"), b"tampered!").unwrap();
        assert!(matches!(
            verify_manifest(&root, &plan, None),
            Err(AutoBuildError::Integrity(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn certified_identity_prevents_cross_model_overwrite() {
        let root = std::env::temp_dir().join(format!(
            "verineural-identity-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("identity.txt"), "id=xor\nversion=1\n").unwrap();
        check_existing_identity(&root, "id=xor\nversion=1\n").unwrap();
        assert!(matches!(
            check_existing_identity(&root, "id=other\nversion=1\n"),
            Err(AutoBuildError::Integrity(_))
        ));
        fs::remove_file(root.join("identity.txt")).unwrap();
        assert!(check_existing_identity(&root, "id=xor\nversion=1\n").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn existing_certified_bundle_symlink_is_rejected() {
        use std::os::unix::fs::symlink;
        let root = std::env::temp_dir().join(format!(
            "verineural-symlink-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("real")).unwrap();
        fs::write(root.join("real/identity.txt"), "id=lab\nversion=1\n").unwrap();
        symlink("real", root.join("auto-build")).unwrap();
        assert!(matches!(
            check_existing_identity(&root.join("auto-build"), "id=lab\nversion=1\n"),
            Err(AutoBuildError::Integrity(_))
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shared_output_bundle_names_are_required_when_two_outputs_are_gpu_supported() {
        let mut graph = NeuralGraph::default();
        let shape = Shape::new(vec![2]).unwrap();
        let a = graph.input(shape.clone(), Domain::Finite);
        let b = graph.input(shape, Domain::Finite);
        let common = graph.add(a, b).unwrap();
        let output = graph.relu(common).unwrap();
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[common, output], &[], GpuRequirement::AllOutputs)
            .unwrap();
        let files = artifact_files(&plan);
        assert!(plan.shared_gpu.is_some());
        assert!(files.contains(&"gpu_shared/graph.ll".to_owned()));
        assert!(files.contains(&"gpu_shared/graph.ptx".to_owned()));
        assert!(files.contains(&"gpu_shared/translation.plan".to_owned()));
        assert!(files.contains(&"gpu_shared/translation.report".to_owned()));
        assert!(files.contains(&"gpu_0/graph.ptx".to_owned()));
        assert!(files.contains(&"gpu_1/graph.ptx".to_owned()));
    }

    #[test]
    fn auto_manifest_requires_entire_planned_gpu_and_lean_artifact_set() {
        let (graph, output) = graph();
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &[], GpuRequirement::AllOutputs)
            .unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-auto-missing-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        assert!(matches!(
            write_manifest(&root, &plan),
            Err(AutoBuildError::Integrity(_))
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
