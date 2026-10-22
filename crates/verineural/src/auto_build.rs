//! One consumer-facing build path for arbitrary *supported* neural graphs.
//!
//! The graph and contracts remain the single source of truth. Binding checks
//! are structural Rust checks; the Lean kernel verifies mathematical claims
//! about the generated ideal-Int model; LLVM/PTX translation validation does
//! not prove actual binary32 or NVIDIA machine-code equivalence.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::{
    FixedScan, GpuEngine, GpuError, GpuProgram, LeanProgram, LeanProofEngine, NeuralError,
    NeuralGraph, NodeId, Operation, ProofError, ProofGoal, auto_bundle,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuRequirement {
    /// Produce Lean proofs and explicitly mark unsupported GPU outputs.
    AllowUnsupported,
    /// Fail unless *every* requested output has supported NVPTX lowering.
    AllOutputs,
}

#[derive(Clone, Debug)]
pub enum ArchitectureProofTask {
    Goal(ProofGoal),
    /// Prove every state of one explicitly unrolled finite scan.
    FixedScanNonnegative {
        name: String,
        scan: FixedScan,
    },
}

impl ArchitectureProofTask {
    fn name(&self) -> &str {
        match self {
            Self::Goal(goal) => &goal.name,
            Self::FixedScanNonnegative { name, .. } => name,
        }
    }
}

#[derive(Debug)]
pub enum AutoBuildError {
    Graph(NeuralError),
    Proof(ProofError),
    Gpu(GpuError),
    Io(io::Error),
    InvalidConnection(String),
    DuplicateTask(String),
    Integrity(String),
}

impl fmt::Display for AutoBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => write!(f, "graph validation failed: {e}"),
            Self::Proof(e) => write!(f, "Lean obligation failed: {e}"),
            Self::Gpu(e) => write!(f, "GPU lowering failed: {e}"),
            Self::Io(e) => write!(f, "artifact I/O failed: {e}"),
            Self::InvalidConnection(e) => write!(f, "invalid architecture connectivity: {e}"),
            Self::DuplicateTask(e) => write!(f, "duplicate proof task name: {e}"),
            Self::Integrity(e) => write!(f, "auto-build artifact integrity failure: {e}"),
        }
    }
}

impl Error for AutoBuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Graph(e) => Some(e),
            Self::Proof(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<NeuralError> for AutoBuildError {
    fn from(value: NeuralError) -> Self {
        Self::Graph(value)
    }
}
impl From<ProofError> for AutoBuildError {
    fn from(value: ProofError) -> Self {
        Self::Proof(value)
    }
}
impl From<GpuError> for AutoBuildError {
    fn from(value: GpuError) -> Self {
        Self::Gpu(value)
    }
}
impl From<io::Error> for AutoBuildError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// The certified scope is the dependency closure of selected roots, including
/// declared module inputs even if a body does not read a particular port.
fn selected_dependency_scope(
    graph: &NeuralGraph,
    outputs: &[NodeId],
) -> Result<BTreeSet<usize>, AutoBuildError> {
    let mut used = BTreeSet::new();
    let mut pending = outputs.to_vec();
    while let Some(id) = pending.pop() {
        let node = graph.node(id)?;
        if !used.insert(id.index()) {
            continue;
        }
        pending.extend(node.inputs.iter().copied());
        for module in graph.modules() {
            if module.output == id {
                pending.extend(module.inputs.iter().copied());
            }
        }
        for module in graph.multi_modules() {
            if module.outputs.contains(&id) {
                pending.extend(module.inputs.iter().copied());
            }
        }
    }
    Ok(used)
}

/// Rechecked graph bindings, not a formal theorem about GPU wiring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectivityEvidence {
    pub data_edges: usize,
    pub module_ports: usize,
    pub module_instances: usize,
}

fn invalid(reason: impl Into<String>) -> AutoBuildError {
    AutoBuildError::InvalidConnection(reason.into())
}

fn check_port(
    graph: &NeuralGraph,
    id: NodeId,
    port: &crate::PortContract,
    where_: &str,
) -> Result<(), AutoBuildError> {
    let node = graph.node(id)?;
    if node.shape != port.shape || !node.domain.satisfies(port.domain) {
        return Err(invalid(format!(
            "{where_}: shape/domain mismatch at node {}",
            id.index()
        )));
    }
    Ok(())
}

/// Independently repeat the structural checks for *all* nodes and instances.
/// Constructor-time validation rejects bad captures; this pass also gives the
/// build an explicit, inspectable edge/port count.
pub fn check_architecture_connectivity(
    graph: &NeuralGraph,
) -> Result<ConnectivityEvidence, AutoBuildError> {
    let mut data_edges = 0;
    let mut module_ports = 0;
    for (index, node) in graph.nodes().iter().enumerate() {
        if node.id.index() != index {
            return Err(invalid(format!("node {index} has inconsistent identity")));
        }
        graph.node(node.id)?;
        if matches!(node.operation, Operation::Input) {
            if !node.inputs.is_empty() {
                return Err(invalid(format!("source node {index} has dependencies")));
            }
        } else {
            let mut shapes = Vec::new();
            for &input in &node.inputs {
                if input.index() >= index {
                    return Err(invalid(format!("node {index} has a non-DAG edge")));
                }
                shapes.push(graph.node(input)?.shape.clone());
                data_edges += 1;
            }
            if node.operation.infer_shape(&shapes)? != node.shape {
                return Err(invalid(format!("node {index} has inconsistent shape")));
            }
        }
        if node.contract.requires.len() != node.inputs.len() {
            return Err(invalid(format!("node {index} contract arity differs")));
        }
        for (&input, port) in node.inputs.iter().zip(&node.contract.requires) {
            check_port(graph, input, port, &format!("node {index} input"))?;
        }
        check_port(
            graph,
            node.id,
            &node.contract.ensures,
            &format!("node {index} result"),
        )?;
    }

    for (index, module) in graph.modules().iter().enumerate() {
        if module.inputs.len() != module.contract.requires.len()
            || module.body_start >= graph.nodes().len()
            || module.output.index() < module.body_start
        {
            return Err(invalid(format!(
                "module {index} has malformed body or ports"
            )));
        }
        for (&input, port) in module.inputs.iter().zip(&module.contract.requires) {
            if input.index() >= module.body_start {
                return Err(invalid(format!("module {index} captures a forward input")));
            }
            check_port(graph, input, port, &format!("module {index} input"))?;
            module_ports += 1;
        }
        check_port(
            graph,
            module.output,
            &module.contract.ensures,
            &format!("module {index} output"),
        )?;
        module_ports += 1;
        check_body_edges(
            graph,
            module.body_start,
            module.output.index() + 1,
            &module.inputs,
            &format!("module {index}"),
        )?;
    }
    for (index, module) in graph.multi_modules().iter().enumerate() {
        if module.inputs.len() != module.contract.inputs.len()
            || module.outputs.len() != module.contract.outputs.len()
            || module.outputs.is_empty()
            || module.body_start >= module.body_end
            || module.body_end > graph.nodes().len()
        {
            return Err(invalid(format!(
                "multi-module {index} has malformed body or ports"
            )));
        }
        for (&input, port) in module.inputs.iter().zip(&module.contract.inputs) {
            if input.index() >= module.body_start {
                return Err(invalid(format!(
                    "multi-module {index} captures a forward input"
                )));
            }
            check_port(
                graph,
                input,
                &port.port,
                &format!("multi-module {index} input"),
            )?;
            module_ports += 1;
        }
        let mut seen = BTreeSet::new();
        for (&output, port) in module.outputs.iter().zip(&module.contract.outputs) {
            if output.index() < module.body_start
                || output.index() >= module.body_end
                || !seen.insert(output.index())
            {
                return Err(invalid(format!(
                    "multi-module {index} has invalid output identity"
                )));
            }
            check_port(
                graph,
                output,
                &port.port,
                &format!("multi-module {index} output"),
            )?;
            module_ports += 1;
        }
        check_body_edges(
            graph,
            module.body_start,
            module.body_end,
            &module.inputs,
            &format!("multi-module {index}"),
        )?;
    }
    Ok(ConnectivityEvidence {
        data_edges,
        module_ports,
        module_instances: graph.modules().len() + graph.multi_modules().len(),
    })
}

fn check_body_edges(
    graph: &NeuralGraph,
    start: usize,
    end: usize,
    declared: &[NodeId],
    instance: &str,
) -> Result<(), AutoBuildError> {
    for node in &graph.nodes()[start..end] {
        if matches!(node.operation, Operation::Input) {
            return Err(invalid(format!(
                "{instance}: hidden input {}",
                node.id.index()
            )));
        }
        for &input in &node.inputs {
            if input.index() < start && !declared.contains(&input) {
                return Err(invalid(format!(
                    "{instance}: undeclared capture {} -> {}",
                    input.index(),
                    node.id.index()
                )));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub enum PlannedGpu {
    Supported {
        output: NodeId,
        program: Box<GpuProgram>,
    },
    Unsupported {
        output: NodeId,
        reason: String,
    },
}

#[derive(Clone, Debug)]
pub struct AutoBuildPlan {
    pub connectivity: ConnectivityEvidence,
    pub outputs: Vec<NodeId>,
    pub architecture: LeanProgram,
    pub goals: Vec<(String, LeanProgram)>,
    pub gpu: Vec<PlannedGpu>,
}

#[derive(Clone, Debug)]
pub struct OutputBuildEvidence {
    pub node: usize,
    pub ptx: Option<PathBuf>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AutoBuildReport {
    pub directory: PathBuf,
    pub connectivity: ConnectivityEvidence,
    pub output_evidence: Vec<OutputBuildEvidence>,
    pub architecture_theorems: usize,
    pub goal_theorems: Vec<(String, usize)>,
    pub source_hypotheses: Vec<crate::ProofAssumption>,
    pub goal_hypotheses: Vec<(String, Vec<crate::ProofAssumption>)>,
    /// Axiom audit per theorem, namespaced by the generated Lean file.
    pub axiom_dependencies: BTreeMap<String, Vec<String>>,
    pub lean_version: String,
    /// Hash of SHA256SUMS. Authenticity requires external pinning.
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Default)]
pub struct ArchitectureAutoCompiler {
    proof: LeanProofEngine,
    gpu: GpuEngine,
}

impl ArchitectureAutoCompiler {
    pub fn new(proof: LeanProofEngine, gpu: GpuEngine) -> Self {
        Self { proof, gpu }
    }

    /// Pure planning stage; checks wiring, all selected output claims and
    /// independent user goals before running either external tool.
    pub fn plan(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
        tasks: &[ArchitectureProofTask],
        gpu_requirement: GpuRequirement,
    ) -> Result<AutoBuildPlan, AutoBuildError> {
        let connectivity = check_architecture_connectivity(graph)?;
        let architecture = self.proof.generate_outputs(graph, outputs)?;
        let scope = selected_dependency_scope(graph, outputs)?;
        let mut names = BTreeSet::new();
        let mut goals = Vec::new();
        for task in tasks {
            let name = task.name();
            if !names.insert(name.to_owned()) {
                return Err(AutoBuildError::DuplicateTask(name.to_owned()));
            }
            let goals_in_scope = match task {
                ArchitectureProofTask::Goal(goal) => {
                    scope.contains(&goal.output.index()) && graph.node(goal.output).is_ok()
                }
                ArchitectureProofTask::FixedScanNonnegative { scan, .. } => scan
                    .states
                    .iter()
                    .chain(&scan.outputs)
                    .all(|id| scope.contains(&id.index()) && graph.node(*id).is_ok()),
            };
            if !goals_in_scope {
                return Err(invalid(format!(
                    "proof task {name} is not connected to any selected architecture output"
                )));
            }
            let program = match task {
                ArchitectureProofTask::Goal(goal) => self.proof.generate_goal(graph, goal)?,
                ArchitectureProofTask::FixedScanNonnegative { name, scan } => self
                    .proof
                    .generate_fixed_scan_nonnegative(graph, scan, name)?,
            };
            goals.push((name.to_owned(), program));
        }
        let mut gpu = Vec::new();
        for &output in outputs {
            match self.gpu.compile(graph, output) {
                Ok(program) => gpu.push(PlannedGpu::Supported {
                    output,
                    program: Box::new(program),
                }),
                Err(error @ (GpuError::UnsupportedOperation { .. } | GpuError::NoKernelGraph))
                    if gpu_requirement == GpuRequirement::AllowUnsupported =>
                {
                    gpu.push(PlannedGpu::Unsupported {
                        output,
                        reason: error.to_string(),
                    });
                }
                Err(error) => return Err(AutoBuildError::Gpu(error)),
            }
        }
        Ok(AutoBuildPlan {
            connectivity,
            outputs: outputs.to_vec(),
            architecture,
            goals,
            gpu,
        })
    }

    /// One developer-invoked build. The reserved auto-build/ subdirectory is
    /// replaced each time; no stale report survives failed proof/GPU checks.
    /// There is no CUDA execution or claim of IEEE-754 machine equivalence.
    pub fn build(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
        tasks: &[ArchitectureProofTask],
        gpu_requirement: GpuRequirement,
        root: &Path,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        let directory = root.join("auto-build");
        if directory.exists() {
            fs::remove_dir_all(&directory)?;
        }
        fs::create_dir_all(&directory)?;
        let plan = self.plan(graph, outputs, tasks, gpu_requirement)?;
        fs::write(
            directory.join("semantic.ir"),
            auto_bundle::canonical_ir(graph, outputs, tasks, gpu_requirement),
        )?;
        let lean_dir = directory.join("lean");
        fs::create_dir_all(&lean_dir)?;
        let (checked, audit) = self
            .proof
            .verify_certified(&plan.architecture, &lean_dir.join("architecture.lean"))?;
        let mut goal_theorems = Vec::new();
        let mut goal_hypotheses = Vec::new();
        let mut axiom_dependencies = audit
            .axiom_dependencies
            .iter()
            .map(|(theorem, axioms)| (format!("architecture.{theorem}"), axioms.clone()))
            .collect::<BTreeMap<_, _>>();
        for (index, (name, program)) in plan.goals.iter().enumerate() {
            let file = lean_dir.join(format!("goal_{index}_{name}.lean"));
            let (proof, goal_audit) = self.proof.verify_certified(program, &file)?;
            if goal_audit.lean_version != audit.lean_version {
                return Err(invalid(
                    "Lean version changed during the architecture build",
                ));
            }
            goal_theorems.push((name.clone(), proof.theorem_names.len()));
            goal_hypotheses.push((name.clone(), proof.assumptions));
            for (theorem, axioms) in goal_audit.axiom_dependencies {
                axiom_dependencies.insert(format!("goal_{index}_{name}.{theorem}"), axioms);
            }
        }

        let mut output_evidence = Vec::new();
        for (index, target) in plan.gpu.iter().enumerate() {
            match target {
                PlannedGpu::Supported { output, program } => {
                    let ptx = self
                        .gpu
                        .lower(program, &directory.join(format!("gpu_{index}")))?;
                    output_evidence.push(OutputBuildEvidence {
                        node: output.index(),
                        ptx: Some(ptx),
                        reason: None,
                    });
                }
                PlannedGpu::Unsupported { output, reason } => {
                    output_evidence.push(OutputBuildEvidence {
                        node: output.index(),
                        ptx: None,
                        reason: Some(reason.clone()),
                    });
                }
            }
        }
        fs::write(
            lean_dir.join("audit.txt"),
            audit_text(&audit.lean_version, &axiom_dependencies),
        )?;
        let mut report = AutoBuildReport {
            directory: directory.clone(),
            connectivity: plan.connectivity,
            output_evidence,
            architecture_theorems: checked.theorem_names.len(),
            goal_theorems,
            source_hypotheses: checked.assumptions,
            goal_hypotheses,
            axiom_dependencies,
            lean_version: audit.lean_version,
            manifest_sha256: String::new(),
        };
        fs::write(directory.join("report.txt"), report_text(graph, &report))?;
        report.manifest_sha256 = match auto_bundle::write_manifest(&directory, &plan) {
            Ok(digest) => digest,
            Err(error) => {
                let _ = fs::remove_file(directory.join("report.txt"));
                let _ = fs::remove_file(directory.join("SHA256SUMS"));
                return Err(error);
            }
        };
        Ok(report)
    }
}

fn audit_text(lean_version: &str, axiom_dependencies: &BTreeMap<String, Vec<String>>) -> String {
    let mut text = format!("lean_checker_version={lean_version}\n");
    for (theorem, axioms) in axiom_dependencies {
        text.push_str(&format!("axioms={theorem};dependencies={axioms:?}\n"));
    }
    text
}

fn report_text(graph: &NeuralGraph, report: &AutoBuildReport) -> String {
    let mut text = format!(
        "architecture_assurance=ideal-Int-Lean-kernel-checked\n\
             connectivity_assurance=Rust-structural-validation\n\
             fp32_machine_equivalence=not-proven\n\
             nvidia_jit_sass_equivalence=not-proven\n\
             lean_version={}\n\
             nodes={}\ndata_edges={}\nmodule_ports={}\nmodule_instances={}\n\
             architecture_theorems={}\nsource_hypotheses={:?}\n",
        report.lean_version,
        graph.nodes().len(),
        report.connectivity.data_edges,
        report.connectivity.module_ports,
        report.connectivity.module_instances,
        report.architecture_theorems,
        report.source_hypotheses,
    );
    for (name, count) in &report.goal_theorems {
        let assumptions = report
            .goal_hypotheses
            .iter()
            .find(|(goal, _)| goal == name)
            .map(|(_, hypotheses)| hypotheses.as_slice())
            .unwrap_or(&[]);
        text.push_str(&format!(
            "goal={name};lean_theorems={count};status=proven-ideal-Int;hypotheses={assumptions:?}\n"
        ));
    }
    for (theorem, axioms) in &report.axiom_dependencies {
        text.push_str(&format!("axioms={theorem};dependencies={axioms:?}\n"));
    }
    for item in &report.output_evidence {
        if let Some(ptx) = &item.ptx {
            text.push_str(&format!(
                "gpu_output={};status=structurally-translated;ptx={}\n",
                item.node,
                ptx.display()
            ));
        } else {
            text.push_str(&format!(
                "gpu_output={};status=unsupported;reason={}\n",
                item.node,
                item.reason.as_deref().unwrap_or("unknown")
            ));
        }
    }
    text
}

impl ArchitectureAutoCompiler {
    /// Offline recheck of a previously emitted bundle against this precise
    /// graph, output selection, goal list and GPU policy. First verify the
    /// complete manifest and immutable IR, then rerun Lean and inspect the
    /// saved LLVM/PTX pairs. No CUDA driver or GPU execution is required.
    ///
    /// A caller needing authenticity must separately pin the SHA256SUMS hash:
    /// the bundle's own manifest cannot authenticate itself.
    pub fn verify_bundle(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
        tasks: &[ArchitectureProofTask],
        gpu_requirement: GpuRequirement,
        root: &Path,
        pinned_sha256: Option<&str>,
    ) -> Result<AutoBuildReport, AutoBuildError> {
        let directory = root.join("auto-build");
        let plan = self.plan(graph, outputs, tasks, gpu_requirement)?;
        let digest = auto_bundle::verify_manifest(&directory, &plan, pinned_sha256)?;
        auto_bundle::verify_current_ir(
            &directory,
            &auto_bundle::canonical_ir(graph, outputs, tasks, gpu_requirement),
        )?;
        let ir_sha256 = auto_bundle::hash_file(&directory.join("semantic.ir"))?;
        let expected_dot = crate::architecture::auto_architecture_dot(graph, &plan, &ir_sha256)?;
        let dot_file = directory.join("architecture.dot");
        if fs::read_to_string(&dot_file)? != expected_dot {
            return Err(AutoBuildError::Integrity(
                "architecture.dot does not match current Semantic IR and output status".into(),
            ));
        }
        // Re-render the exact saved DOT with the pinned Nix Graphviz tool.
        // A PNG that was swapped while retaining the manifest is rejected.
        let rerendered = directory.join("architecture.replay.png");
        crate::architecture::render_dot_png(&dot_file, &rerendered)?;
        let image_check = (|| -> Result<bool, AutoBuildError> {
            Ok(fs::read(&rerendered)? == fs::read(directory.join("architecture.png"))?)
        })();
        let _ = fs::remove_file(&rerendered);
        if !image_check? {
            return Err(AutoBuildError::Integrity(
                "saved architecture.png differs from pinned Graphviz render of DOT".into(),
            ));
        }
        let lean_dir = directory.join("lean");
        let mut axiom_dependencies = BTreeMap::new();
        let (lean_version, architecture_theorems) = {
            let file = lean_dir.join("architecture.lean");
            verify_source(&self.proof, &file, &plan.architecture)?;
            let audit = self
                .proof
                .audit_file(&file, plan.architecture.theorem_names())?;
            for (theorem, axioms) in audit.axiom_dependencies {
                axiom_dependencies.insert(format!("architecture.{theorem}"), axioms);
            }
            (audit.lean_version, plan.architecture.theorem_names().len())
        };
        let mut goal_theorems = Vec::new();
        let mut goal_hypotheses = Vec::new();
        for (index, (name, program)) in plan.goals.iter().enumerate() {
            let file = lean_dir.join(format!("goal_{index}_{name}.lean"));
            verify_source(&self.proof, &file, program)?;
            let audit = self.proof.audit_file(&file, program.theorem_names())?;
            if audit.lean_version != lean_version {
                return Err(AutoBuildError::Integrity(
                    "Lean checker version changed while replaying proofs".into(),
                ));
            }
            for (theorem, axioms) in audit.axiom_dependencies {
                axiom_dependencies.insert(format!("goal_{index}_{name}.{theorem}"), axioms);
            }
            goal_theorems.push((name.clone(), program.theorem_names().len()));
            goal_hypotheses.push((name.clone(), program.assumptions().to_vec()));
        }
        if fs::read_to_string(lean_dir.join("audit.txt"))?
            != audit_text(&lean_version, &axiom_dependencies)
        {
            return Err(AutoBuildError::Integrity(
                "saved Lean axiom dependencies or checker version no longer match".into(),
            ));
        }

        let mut output_evidence = Vec::new();
        for (index, item) in plan.gpu.iter().enumerate() {
            let gpu_dir = directory.join(format!("gpu_{index}"));
            match item {
                PlannedGpu::Supported { output, program } => {
                    if fs::read_to_string(gpu_dir.join("graph.ll"))? != program.llvm_ir() {
                        return Err(AutoBuildError::Integrity(format!(
                            "GPU output {index} LLVM differs from current IR"
                        )));
                    }
                    let translation = program.translation_plan()?;
                    if fs::read_to_string(gpu_dir.join("translation.plan"))?
                        != translation.to_text()
                    {
                        return Err(AutoBuildError::Integrity(format!(
                            "GPU output {index} translation plan mismatch"
                        )));
                    }
                    let ptx_file = gpu_dir.join("graph.ptx");
                    let ptx = fs::read_to_string(&ptx_file)?;
                    let checked = translation
                        .validate(program.llvm_ir(), &ptx)
                        .map_err(|err| {
                            AutoBuildError::Integrity(format!(
                                "GPU output {index} LLVM/PTX translation rejected: {err}"
                            ))
                        })?;
                    if fs::read_to_string(gpu_dir.join("translation.report"))? != checked.to_text()
                    {
                        return Err(AutoBuildError::Integrity(format!(
                            "GPU output {index} translation report mismatch"
                        )));
                    }
                    output_evidence.push(OutputBuildEvidence {
                        node: output.index(),
                        ptx: Some(ptx_file),
                        reason: None,
                    });
                }
                PlannedGpu::Unsupported { output, reason } => {
                    if gpu_dir.exists() {
                        return Err(AutoBuildError::Integrity(format!(
                            "unsupported GPU output {index} has unexpected compiled artifacts"
                        )));
                    }
                    output_evidence.push(OutputBuildEvidence {
                        node: output.index(),
                        ptx: None,
                        reason: Some(reason.clone()),
                    });
                }
            }
        }
        let ir_sha256 = auto_bundle::hash_file(&directory.join("semantic.ir"))?;
        let dot = crate::architecture::auto_architecture_dot(graph, &plan, &ir_sha256)?;
        fs::write(directory.join("architecture.dot"), dot)?;
        crate::architecture::render_dot_png(
            &directory.join("architecture.dot"),
            &directory.join("architecture.png"),
        )?;
        let report = AutoBuildReport {
            directory: directory.clone(),
            connectivity: plan.connectivity,
            output_evidence,
            architecture_theorems,
            goal_theorems,
            source_hypotheses: plan.architecture.assumptions().to_vec(),
            goal_hypotheses,
            axiom_dependencies,
            lean_version,
            manifest_sha256: digest,
        };
        if fs::read_to_string(directory.join("report.txt"))? != report_text(graph, &report) {
            return Err(AutoBuildError::Integrity(
                "auto-build report does not describe the freshly rechecked evidence".into(),
            ));
        }
        Ok(report)
    }
}

fn verify_source(
    proof: &LeanProofEngine,
    path: &Path,
    expected: &LeanProgram,
) -> Result<(), AutoBuildError> {
    if fs::read_to_string(path)? != expected.source() {
        return Err(AutoBuildError::Integrity(format!(
            "saved Lean source differs from the generated current proof: {}",
            path.display()
        )));
    }
    proof.check_file(path)?;
    Ok(())
}
