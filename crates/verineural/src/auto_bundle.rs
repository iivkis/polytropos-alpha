//! Integrity and current-source binding for the developer-run auto build.
//! SHA256SUMS proves artifact integrity relative to a separately pinned digest;
//! it is not a signature and cannot prove NVIDIA machine semantics.
use std::fs;
use std::path::Path;

use crate::{ArchitectureProofTask, GpuRequirement, NeuralGraph, NodeId, ProofGoalKind};
use crate::auto_build::{AutoBuildError, AutoBuildPlan, PlannedGpu};

const MANIFEST: &str = "SHA256SUMS";

fn hash_file(path: &Path) -> Result<String, AutoBuildError> {
    crate::certificate::hash_file(path)
        .map_err(|err| AutoBuildError::Integrity(err.to_string()))
}

pub(super) fn canonical_ir(
    graph: &NeuralGraph,
    outputs: &[NodeId],
    tasks: &[ArchitectureProofTask],
    gpu_requirement: GpuRequirement,
) -> String {
    // Canonical identifiers are node indices; graph-scoped random handles must
    // not enter a reproducible fingerprint.
    let mut text = format!(
        "verineural-auto-ir-v1\npolicy={:?}\noutputs={:?}\ngpu_requirement={gpu_requirement:?}\n",
        graph.policy(),
        outputs.iter().map(|id| id.index()).collect::<Vec<_>>()
    );
    for (index, node) in graph.nodes().iter().enumerate() {
        text.push_str(&format!(
            "node={index};operation={:?};inputs={:?};shape={:?};domain={:?};contract={:?};semantics={:?}\n",
            node.operation, node.inputs.iter().map(|id| id.index()).collect::<Vec<_>>(),
            node.shape.dims(), node.domain, node.contract, node.semantics,
        ));
    }
    for (index, module) in graph.modules().iter().enumerate() {
        text.push_str(&format!(
            "module={index};name={};version={};inputs={:?};body_start={};output={};contract={:?}\n",
            module.name, module.semantic_version,
            module.inputs.iter().map(|id| id.index()).collect::<Vec<_>>(),
            module.body_start, module.output.index(), module.contract
        ));
    }
    for (index, module) in graph.multi_modules().iter().enumerate() {
        text.push_str(&format!(
            "multi_module={index};name={};version={};inputs={:?};body_start={};body_end={};outputs={:?};contract={:?}\n",
            module.name, module.semantic_version,
            module.inputs.iter().map(|id| id.index()).collect::<Vec<_>>(),
            module.body_start, module.body_end,
            module.outputs.iter().map(|id| id.index()).collect::<Vec<_>>(),
            module.contract,
        ));
    }
    for (index, task) in tasks.iter().enumerate() {
        match task {
            ArchitectureProofTask::Goal(goal) => match &goal.kind {
                ProofGoalKind::Nonnegative { assumptions } => text.push_str(&format!(
                    "task={index};name={};output={};nonnegative_assumptions={:?}\n",
                    goal.name, goal.output.index(),
                    assumptions.iter().map(|id| id.index()).collect::<Vec<_>>(),
                )),
                ProofGoalKind::EqualTo { other } => text.push_str(&format!(
                    "task={index};name={};output={};equal_to={}\n",
                    goal.name, goal.output.index(), other.index(),
                )),
            },
            ArchitectureProofTask::FixedScanNonnegative { name, scan } => {
                text.push_str(&format!(
                    "task={index};name={name};scan_states={:?};scan_outputs={:?};scan_modules={:?}\n",
                    scan.states.iter().map(|id| id.index()).collect::<Vec<_>>(),
                    scan.outputs.iter().map(|id| id.index()).collect::<Vec<_>>(),
                    scan.module_indices,
                ));
            }
        }
    }
    text
}

/// The expected artifact set is derived from the current plan, never from
/// user-controlled manifest lines. Paths are relative to auto-build/.
pub(super) fn artifact_files(plan: &AutoBuildPlan) -> Vec<String> {
    let mut files = vec![
        "semantic.ir".to_owned(),
        "report.txt".to_owned(),
        "lean/architecture.lean".to_owned(),
        "lean/audit.txt".to_owned(),
    ];
    for (index, (name, _)) in plan.goals.iter().enumerate() {
        files.push(format!("lean/goal_{index}_{name}.lean"));
    }
    for (index, item) in plan.gpu.iter().enumerate() {
        if matches!(item, PlannedGpu::Supported { .. }) {
            for name in ["graph.ll", "graph.ptx", "translation.plan", "translation.report"] {
                files.push(format!("gpu_{index}/{name}"));
            }
        }
    }
    files.sort();
    files
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
    let expected = manifest_text(dir, plan)?;
    if existing != expected {
        return Err(AutoBuildError::Integrity(
            "SHA256SUMS does not match the complete expected artifact set".into()
        ));
    }
    let digest = hash_file(&dir.join(MANIFEST))?;
    if expected_digest.is_some_and(|pinned| pinned != digest) {
        return Err(AutoBuildError::Integrity("externally pinned manifest digest mismatch".into()));
    }
    Ok(digest)
}

pub(super) fn verify_current_ir(dir: &Path, source: &str) -> Result<(), AutoBuildError> {
    if fs::read_to_string(dir.join("semantic.ir"))? != source {
        return Err(AutoBuildError::Integrity(
            "saved auto-build IR differs from current Rust architecture or proof tasks".into()
        ));
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchitectureAutoCompiler, ArchitectureProofTask, Domain, GpuRequirement,
        NeuralGraph, ProofGoal, ProofGoalKind, Shape,
    };

    fn graph() -> (NeuralGraph, NodeId) {
        let mut graph = NeuralGraph::default();
        let x = graph.input(Shape::new(vec![2]).unwrap(), Domain::NonNegativeFinite);
        let output = graph.relu(x).unwrap();
        (graph, output)
    }

    #[test]
    fn auto_ir_is_independent_of_graph_identity_but_bound_to_contracts_and_tasks() {
        let (g1, out1) = graph();
        let (g2, out2) = graph();
        let ir1 = canonical_ir(&g1, &[out1], &[], GpuRequirement::AllOutputs);
        let ir2 = canonical_ir(&g2, &[out2], &[], GpuRequirement::AllOutputs);
        assert_eq!(ir1, ir2);

        let task = ArchitectureProofTask::Goal(ProofGoal {
            name: "consumer_claim".to_owned(),
            output: out1,
            kind: ProofGoalKind::Nonnegative { assumptions: vec![] },
        });
        assert_ne!(ir1, canonical_ir(&g1, &[out1], &[task], GpuRequirement::AllOutputs));
        assert_ne!(
            ir1,
            canonical_ir(&g1, &[out1], &[], GpuRequirement::AllowUnsupported),
        );
        let mut other = NeuralGraph::default();
        let x = other.input(Shape::new(vec![3]).unwrap(), Domain::NonNegativeFinite);
        let y = other.relu(x).unwrap();
        assert_ne!(
            ir1,
            canonical_ir(&other, &[y], &[], GpuRequirement::AllOutputs),
        );
    }

    #[test]
    fn auto_manifest_detects_tampering_and_rejects_unpinned_or_wrong_artifacts() {
        let (graph, output) = graph();
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &[], GpuRequirement::AllOutputs).unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-auto-bundle-{}-{:?}", std::process::id(),
            std::thread::current().id()
        ));
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir_all(&root).unwrap();
        let files = artifact_files(&plan);
        assert!(files.contains(&"gpu_0/graph.ptx".to_owned()));
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
    fn auto_manifest_requires_entire_planned_gpu_and_lean_artifact_set() {
        let (graph, output) = graph();
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &[], GpuRequirement::AllOutputs).unwrap();
        let root = std::env::temp_dir().join(format!(
            "verineural-auto-missing-{}-{:?}", std::process::id(),
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
