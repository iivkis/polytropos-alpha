//! Stage 5: one evidence bundle for a concrete IR/Lean/PTX/GPU run.
//!
//! This is an *integrity-checked validation report*, not a proof that PTX,
//! JIT-generated machine code or IEEE-754 behavior preserves the Lean model.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{GpuEngine, GpuError, LeanProofEngine, NeuralGraph, NodeId, ProofError, Tensor};

const ARTIFACTS: [&str; 7] = [
    "graph.ir",
    "proof.lean",
    "graph.ll",
    "graph.ptx",
    "inputs.bits",
    "output.bits",
    "report.json",
];
const MANIFEST: &str = "SHA256SUMS";

/// These labels describe distinct evidence levels, not a single certification grade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceStatus {
    Proven,
    Validated,
    Trusted,
    Unsupported,
}

impl EvidenceStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Proven => "proven",
            Self::Validated => "validated",
            Self::Trusted => "trusted",
            Self::Unsupported => "unsupported",
        }
    }
}

#[derive(Debug)]
pub enum CertificateError {
    Proof(ProofError),
    Gpu(GpuError),
    Io(io::Error),
    WrongOutputShape,
    NumericalMismatch {
        index: usize,
        gpu_bits: u32,
        cpu_bits: u32,
    },
    HashTool(String),
    Integrity(String),
}

impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proof(err) => write!(f, "proof step failed: {err}"),
            Self::Gpu(err) => write!(f, "GPU step failed: {err}"),
            Self::Io(err) => write!(f, "certificate I/O failed: {err}"),
            Self::WrongOutputShape => write!(f, "GPU and CPU output shapes differ"),
            Self::NumericalMismatch {
                index,
                gpu_bits,
                cpu_bits,
            } => write!(
                f,
                "FP32 mismatch at {index}: GPU={gpu_bits:08x}, CPU={cpu_bits:08x}"
            ),
            Self::HashTool(message) => write!(f, "sha256sum failed: {message}"),
            Self::Integrity(message) => write!(f, "bundle integrity failure: {message}"),
        }
    }
}

impl Error for CertificateError {}

impl From<ProofError> for CertificateError {
    fn from(value: ProofError) -> Self {
        Self::Proof(value)
    }
}

impl From<GpuError> for CertificateError {
    fn from(value: GpuError) -> Self {
        Self::Gpu(value)
    }
}

impl From<io::Error> for CertificateError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Successful validation with a SHA-256 digest that can be pinned externally.
/// Neither this structure nor the on-disk JSON is a signed certificate.
#[derive(Debug)]
pub struct Stage5Report {
    pub directory: PathBuf,
    pub manifest_sha256: String,
    pub theorem_count: usize,
    pub compared_fp32_values: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CertificateEngine {
    proof: LeanProofEngine,
    gpu: GpuEngine,
}

impl CertificateEngine {
    pub fn new(proof: LeanProofEngine, gpu: GpuEngine) -> Self {
        Self { proof, gpu }
    }

    /// Fail closed: old report and manifest are removed *before* any checking.
    ///
    /// The caller supplies a separately computed Rust f32 reference. Its
    /// correctness is trusted, not proven. A bitwise comparison is evidence
    /// for these specific input values only.
    pub fn run(
        &self,
        graph: &NeuralGraph,
        output: NodeId,
        inputs: &[(NodeId, Tensor<f32>)],
        cpu_reference: &Tensor<f32>,
        directory: &Path,
    ) -> Result<Stage5Report, CertificateError> {
        fs::create_dir_all(directory)?;
        for name in ["report.json", MANIFEST] {
            match fs::remove_file(directory.join(name)) {
                Ok(()) => {}
                Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                Err(err) => return Err(err.into()),
            }
        }

        let result: Result<Stage5Report, CertificateError> = (|| {
            let model = self.proof.generate(graph, output)?;
            let (checked, audit) = self
                .proof
                .verify_certified(&model, &directory.join("proof.lean"))?;
            let program = self.gpu.compile(graph, output)?;
            let actual = self.gpu.execute(&program, inputs, directory)?;
            if actual.shape() != cpu_reference.shape() {
                return Err(CertificateError::WrongOutputShape);
            }
            for (index, (&gpu, &cpu)) in actual
                .values()
                .iter()
                .zip(cpu_reference.values())
                .enumerate()
            {
                if gpu.to_bits() != cpu.to_bits() {
                    return Err(CertificateError::NumericalMismatch {
                        index,
                        gpu_bits: gpu.to_bits(),
                        cpu_bits: cpu.to_bits(),
                    });
                }
            }

            fs::write(directory.join("graph.ir"), ir_snapshot(graph, output))?;
            fs::write(directory.join("inputs.bits"), input_snapshot(graph, inputs))?;
            fs::write(
                directory.join("output.bits"),
                actual
                    .values()
                    .iter()
                    .zip(cpu_reference.values())
                    .enumerate()
                    .map(|(i, (gpu, cpu))| {
                        format!("{i} gpu={:08x} cpu={:08x}\n", gpu.to_bits(), cpu.to_bits())
                    })
                    .collect::<String>(),
            )?;

            let hashes = ARTIFACTS[..ARTIFACTS.len() - 1]
                .iter()
                .map(|name| Ok((*name, hash_file(&directory.join(name))?)))
                .collect::<Result<Vec<_>, CertificateError>>()?;
            let theorem_names = checked
                .theorem_names
                .iter()
                .map(|name| format!("\"{}\"", escape_json(name)))
                .collect::<Vec<_>>()
                .join(", ");
            let hypotheses = checked
                .assumptions
                .iter()
                .map(|a| {
                    format!(
                        "{{\"input_node\":{},\"domain\":\"{:?}\"}}",
                        a.input_node, a.domain
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let input_domains = program
                .source_inputs()
                .iter()
                .map(|&id| {
                    let node = &graph.nodes()[id.index()];
                    format!(
                        "{{\"node\":{},\"shape\":{:?},\"domain\":\"{:?}\"}}",
                        id.index(),
                        node.shape.dims(),
                        node.domain
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            let artifacts = hashes
                .iter()
                .map(|(name, hash)| format!("{{\"path\":\"{name}\",\"sha256\":\"{hash}\"}}"))
                .collect::<Vec<_>>()
                .join(",\n    ");
            let report = format!(
                concat!(
                    "{{\n",
                    "  \"schema\": \"verineural.stage5.v1\",\n",
                    "  \"output_node\": {},\n",
                    "  \"statuses\": {{\"mathematical\":\"{}\",\"composition\":\"{}\",",
                    "\"numerical\":\"{}\",\"gpu_execution\":\"{}\",",
                    "\"compilation\":\"{}\",\"fp32_semantics\":\"{}\",",
                    "\"machine_code_correctness\":\"{}\"}},\n",
                    "  \"proven_scope\": \"Lean 4 ideal Int model: shapes, per-node definitions, and applicable nonnegativity/module obligations only\",\n",
                    "  \"numerical_scope\": \"bitwise CPU/GPU equality for the recorded concrete f32 inputs only\",\n",
                    "  \"compilation_scope\": \"LLVM NVPTX/PTX generated and run; semantic preservation is trusted, not proved; experimental fusion is not FP32-certified\",\n",
                    "  \"certified_machine_code\": false,\n",
                    "  \"gpu_launch\": {{\"execution\":\"cuda_graph_capture_nonblocking_stream\",\"threads_per_block\":{},\"relu_fusion\":\"{}\",\"applied_fusions\":{}}},\n",
                    "  \"theorems\": [{}],\n",
                    "  \"lean_hypotheses\": [{}],\n",
                    "  \"declared_input_domains\": [{}],\n",
                    "  \"compared_fp32_values\": {},\n",
                    "  \"artifacts\": [\n    {}\n  ],\n",
                    "  \"tcb\": [\"Rust graph and certificate serializer\",",
                    "\"Lean model generator\", \"Lean 4 executable and kernel\",",
                    "\"LLVM llc NVPTX compiler\", \"CUDA driver/JIT/runtime\",",
                    "\"NVIDIA hardware\", \"Rust f32 CPU reference\",",
                    "\"coreutils sha256sum\"],\n",
                    "  \"limitations\": [\"Hashes detect changes relative to a pinned digest; they do not authenticate the author\",",
                    "\"Lean integer semantics is not linked by proof to GPU floating-point behavior\",",
                    "\"Input domain declarations are caller assumptions checked for the concrete GPU inputs\",",
                    "\"No verified GPU memory, compiler correctness, IEEE-754 or numerical error bound\"]\n",
                    "}}\n"
                ),
                output.index(),
                EvidenceStatus::Proven.as_str(),
                EvidenceStatus::Proven.as_str(),
                EvidenceStatus::Validated.as_str(),
                EvidenceStatus::Validated.as_str(),
                EvidenceStatus::Trusted.as_str(),
                EvidenceStatus::Unsupported.as_str(),
                EvidenceStatus::Unsupported.as_str(),
                program.launch_config().threads_per_block(),
                program.launch_config().relu_fusion().as_str(),
                program.fused_relus().len(),
                theorem_names,
                hypotheses,
                input_domains,
                actual.values().len(),
                artifacts,
            );
            let report = report.replace(
                "\"lean_hypotheses\":",
                &format!("\"lean_checker_version\": \"{}\",\n  \"lean_axiom_audit\": \"no-transitive-axioms\",\n  \"lean_hypotheses\":",
                    escape_json(&audit.lean_version)),
            );
            fs::write(directory.join("report.json"), report)?;
            fs::write(directory.join(MANIFEST), manifest_contents(directory)?)?;
            self.verify_bundle(directory)?;
            Ok(Stage5Report {
                directory: directory.to_path_buf(),
                manifest_sha256: hash_file(&directory.join(MANIFEST))?,
                theorem_count: checked.theorem_names.len(),
                compared_fp32_values: actual.values().len(),
            })
        })();
        if result.is_err() {
            // A partially written report must never survive any failed run.
            for name in ["report.json", MANIFEST] {
                let _ = fs::remove_file(directory.join(name));
            }
        }
        result
    }

    /// Verify the report against a SHA256SUMS digest pinned outside the bundle.
    pub fn verify_bundle_with_digest(
        &self,
        directory: &Path,
        pinned_sha256: &str,
    ) -> Result<(), CertificateError> {
        let actual = hash_file(&directory.join(MANIFEST))?;
        if actual != pinned_sha256 {
            return Err(CertificateError::Integrity(
                "pinned SHA256SUMS digest mismatch".into(),
            ));
        }
        self.verify_bundle(directory)
    }

    /// Recompute all bundled hashes and rerun the trusted Lean checker.
    ///
    /// This confirms integrity relative to SHA256SUMS and Lean acceptance,
    /// NOT authenticity of SHA256SUMS itself or GPU semantic correctness.
    /// Pin the printed digest of SHA256SUMS outside the bundle.
    pub fn verify_bundle(&self, directory: &Path) -> Result<(), CertificateError> {
        let on_disk = fs::read_to_string(directory.join(MANIFEST))?;
        let expected = manifest_contents(directory)?;
        if on_disk != expected {
            return Err(CertificateError::Integrity(
                "SHA256SUMS differs from one or more artifacts".into(),
            ));
        }
        let source_file = directory.join("proof.lean");
        self.proof.check_file(&source_file)?;
        let lean = fs::read_to_string(&source_file)?;
        let generated = lean
            .split_once("namespace VeriNeural.Generated")
            .map(|(_, tail)| tail)
            .ok_or_else(|| {
                CertificateError::Integrity("missing generated Lean namespace".into())
            })?;
        let names = generated
            .lines()
            .filter_map(|line| {
                line.trim_start()
                    .strip_prefix("theorem ")
                    .and_then(|line| line.split_whitespace().next())
                    .map(ToOwned::to_owned)
            })
            .collect::<Vec<_>>();
        let audit = self.proof.audit_file(&source_file, &names)?;
        let report = fs::read_to_string(directory.join("report.json"))?;
        if !report.contains(&format!(
            "\"lean_checker_version\": \"{}\"",
            escape_json(&audit.lean_version),
        )) {
            return Err(CertificateError::Integrity(
                "saved report Lean version differs from current audited checker".into(),
            ));
        }
        Ok(())
    }
}

/// Canonical within the v3 format: graph identity and process-global IDs do
/// not contribute to the fingerprint; complete semantics and interfaces do.
fn ir_snapshot(graph: &NeuralGraph, output: NodeId) -> String {
    let mut source = format!(
        "verineural-ir-v3\noutput={}\nfp32={:?}\n",
        output.index(),
        graph.policy()
    );
    for (index, node) in graph.nodes().iter().enumerate() {
        let edges = node
            .inputs
            .iter()
            .map(|id| id.index().to_string())
            .collect::<Vec<_>>()
            .join(",");
        source.push_str(&format!(
            "node={index};op={:?};inputs={edges};shape={:?};domain={:?};semantics={:?};requires={:?};ensures={:?};claims={:?}\n",
            node.operation,
            node.shape.dims(),
            node.domain,
            node.semantics,
            node.contract.requires,
            node.contract.ensures,
            node.contract.claims,
        ));
    }
    for module in graph.modules() {
        source.push_str(&format!(
            "module={};version={};inputs={:?};body_start={};output={};requires={:?};ensures={:?};claims={:?}\n",
            module.name,
            module.semantic_version,
            module
                .inputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module.body_start,
            module.output.index(),
            module.contract.requires,
            module.contract.ensures,
            module.contract.claims,
        ));
    }
    for module in graph.multi_modules() {
        source.push_str(&format!(
            "multi_module={};version={};body=[{},{}) inputs={:?};outputs={:?};contracts={:?}\n",
            module.name,
            module.semantic_version,
            module.body_start,
            module.body_end,
            module
                .inputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module
                .outputs
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>(),
            module.contract
        ));
    }
    source
}

fn input_snapshot(graph: &NeuralGraph, inputs: &[(NodeId, Tensor<f32>)]) -> String {
    let mut ordered = inputs.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(id, _)| id.index());
    let mut text = String::from("verineural-inputs-f32-bits-v1\n");
    for (id, value) in ordered {
        text.push_str(&format!(
            "node={};declared_domain={:?};shape={:?};bits=",
            id.index(),
            graph.nodes()[id.index()].domain,
            value.shape().dims()
        ));
        for bits in value.values() {
            text.push_str(&format!("{:08x},", bits.to_bits()));
        }
        text.push('\n');
    }
    text
}

fn escape_json(value: &str) -> String {
    let mut out = String::new();
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn hash_file(path: &Path) -> Result<String, CertificateError> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(CertificateError::HashTool(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let digest = text.split_whitespace().next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(CertificateError::HashTool("invalid SHA-256 digest".into()));
    }
    Ok(digest.to_owned())
}

fn manifest_contents(directory: &Path) -> Result<String, CertificateError> {
    let lines = ARTIFACTS
        .iter()
        .map(|name| Ok(format!("{}  {}\n", hash_file(&directory.join(name))?, name)))
        .collect::<Result<Vec<_>, CertificateError>>()?;
    Ok(lines.concat())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Domain, Shape};

    #[test]
    fn snapshots_ignore_process_graph_ids() {
        let make = || {
            let mut graph = NeuralGraph::default();
            let input = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
            let output = graph.relu(input).unwrap();
            (graph, output)
        };
        let (a, a_out) = make();
        let (b, b_out) = make();
        assert_eq!(ir_snapshot(&a, a_out), ir_snapshot(&b, b_out));
    }

    #[test]
    fn statuses_do_not_equate_validation_with_proof() {
        assert_eq!(EvidenceStatus::Proven.as_str(), "proven");
        assert_eq!(EvidenceStatus::Validated.as_str(), "validated");
        assert_eq!(EvidenceStatus::Trusted.as_str(), "trusted");
        assert_eq!(EvidenceStatus::Unsupported.as_str(), "unsupported");
        assert_eq!(escape_json("a\"b\n"), "a\\\"b\\n");
    }

    #[test]
    fn tampering_with_bundled_artifact_is_rejected_before_lean() {
        let dir = std::env::temp_dir().join(format!("verineural-stage5-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for name in ARTIFACTS {
            fs::write(dir.join(name), name).unwrap();
        }
        fs::write(dir.join(MANIFEST), manifest_contents(&dir).unwrap()).unwrap();
        fs::write(dir.join("graph.ptx"), "modified").unwrap();
        let engine = CertificateEngine::new(LeanProofEngine::new("false"), GpuEngine::default());
        assert!(matches!(
            engine.verify_bundle(&dir),
            Err(CertificateError::Integrity(_))
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn unsupported_proof_clears_stale_success_report() {
        let dir =
            std::env::temp_dir().join(format!("verineural-stage5-fail-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("report.json"), "stale").unwrap();
        fs::write(dir.join(MANIFEST), "stale").unwrap();
        let mut graph = NeuralGraph::default();
        let input = graph.input(Shape::new(vec![1]).unwrap(), Domain::NonNegativeFinite);
        let output = graph.sqrt(input).unwrap();
        let expected = Tensor::new(Shape::new(vec![1]).unwrap(), vec![1.0]).unwrap();
        let result = CertificateEngine::default().run(&graph, output, &[], &expected, &dir);
        assert!(matches!(result, Err(CertificateError::Proof(_))));
        assert!(!dir.join("report.json").exists());
        assert!(!dir.join(MANIFEST).exists());
        fs::remove_dir_all(dir).unwrap();
    }
}
