//! Lean 4 proof generation for the deliberately narrow, ideal-integer fragment.
//!
//! The verified statement concerns the generated Lean model, not Rust f32
//! execution, FP32 rounding, generated PTX, or physical GPU execution.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{Domain, NeuralError, NeuralGraph, NodeId, Operation, Shape};

const LEAN_LIBRARY: &str = include_str!("../../../lean/VeriNeural/Basic.lean");

#[derive(Debug)]
pub enum ProofError {
    InvalidGraph(NeuralError),
    UnsupportedOperation {
        node: usize,
        operation: &'static str,
    },
    UnsupportedGuarantee {
        node: usize,
    },
    Io(std::io::Error),
    LeanRejected {
        details: String,
    },
}

impl fmt::Display for ProofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGraph(error) => write!(f, "invalid neural graph: {error}"),
            Self::UnsupportedOperation { node, operation } => {
                write!(
                    f,
                    "Lean proof backend cannot model {operation} at node {node}"
                )
            }
            Self::UnsupportedGuarantee { node } => {
                write!(
                    f,
                    "Lean proof backend cannot establish domain at node {node}"
                )
            }
            Self::Io(error) => write!(f, "Lean checker I/O failure: {error}"),
            Self::LeanRejected { details } => write!(f, "Lean proof rejected: {details}"),
        }
    }
}

impl Error for ProofError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidGraph(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NeuralError> for ProofError {
    fn from(value: NeuralError) -> Self {
        Self::InvalidGraph(value)
    }
}

impl From<std::io::Error> for ProofError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// A source-node hypothesis in the theorem, never an unconditional conclusion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofAssumption {
    pub input_node: usize,
    pub domain: Domain,
}

/// Generated Lean source and a list of obligations. Generation is not proof.
#[derive(Clone, Debug)]
pub struct LeanProgram {
    source: String,
    theorems: Vec<String>,
    assumptions: Vec<ProofAssumption>,
}

impl LeanProgram {
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn theorem_names(&self) -> &[String] {
        &self.theorems
    }

    pub fn assumptions(&self) -> &[ProofAssumption] {
        &self.assumptions
    }
}

/// Produced only after the configured Lean executable exits successfully
/// without warnings or other diagnostics. This is NOT a GPU certificate.
#[derive(Clone, Debug)]
pub struct CheckedLean {
    pub path: PathBuf,
    pub theorem_names: Vec<String>,
    pub assumptions: Vec<ProofAssumption>,
}

/// An external Lean 4 checker. The executable and Lean's kernel belong to TCB.
#[derive(Clone, Debug)]
pub struct LeanProofEngine {
    executable: PathBuf,
}

impl Default for LeanProofEngine {
    fn default() -> Self {
        Self::new("lean")
    }
}

impl LeanProofEngine {
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
        }
    }

    /// Produce Lean definitions and theorems for the ancestry of one output.
    ///
    /// Only Add, Mul, Relu and Reshape have integer ideal semantics in stage 3.
    /// Sqrt and MatMul fail closed, even though they are available to the Rust
    /// reference engine. The input nonnegativity assumptions are explicit.
    pub fn generate(&self, graph: &NeuralGraph, output: NodeId) -> Result<LeanProgram, ProofError> {
        let mut used = BTreeSet::new();
        let mut pending = vec![output];

        while let Some(id) = pending.pop() {
            let node = graph.node(id)?;
            if !used.insert(id.index()) {
                continue;
            }
            match node.operation {
                Operation::Input
                | Operation::Add
                | Operation::Mul
                | Operation::Relu
                | Operation::Reshape(_) => {}
                _ => {
                    return Err(ProofError::UnsupportedOperation {
                        node: id.index(),
                        operation: node.operation.name(),
                    });
                }
            }
            pending.extend(node.inputs.iter().copied());
            // Module contracts also introduce dependencies, even when a body
            // returns a node without using every declared input.
            for module in graph.modules() {
                if module.output == id {
                    pending.extend(module.inputs.iter().copied());
                }
            }
        }

        let sources = used
            .iter()
            .filter(|&&index| matches!(graph.nodes()[index].operation, Operation::Input))
            .copied()
            .collect::<Vec<_>>();
        let assumptions = sources
            .iter()
            .filter(|&&i| graph.nodes()[i].domain == Domain::NonNegativeFinite)
            .map(|&i| ProofAssumption {
                input_node: i,
                domain: Domain::NonNegativeFinite,
            })
            .collect::<Vec<_>>();

        let params = sources
            .iter()
            .map(|&i| {
                format!(
                    "(x_{i} : VeriNeural.Tensor {})",
                    graph.nodes()[i].shape.element_count()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let arguments = sources
            .iter()
            .map(|&i| format!("x_{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let hypothesis_params = assumptions
            .iter()
            .map(|a| {
                format!(
                    "(h_{} : VeriNeural.Nonnegative x_{})",
                    a.input_node, a.input_node
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let hypothesis_args = assumptions
            .iter()
            .map(|a| format!("h_{}", a.input_node))
            .collect::<Vec<_>>()
            .join(" ");
        let proof_params = format!("{params} {hypothesis_params}");
        let proof_args = format!("{arguments} {hypothesis_args}");
        let call = |index: usize| format!("(node_{index} {arguments})");
        let proof_call = |index: usize| format!("node_{index}_nonnegative {proof_args}");

        let mut source = String::from(LEAN_LIBRARY);
        source.push_str("\n\nnamespace VeriNeural.Generated\n\n");
        let mut theorems = Vec::new();

        for &index in &used {
            let node = &graph.nodes()[index];
            let count = node.shape.element_count();
            let rhs = match &node.operation {
                Operation::Input => format!("x_{index}"),
                Operation::Add => format!(
                    "VeriNeural.add {} {}",
                    call(node.inputs[0].index()),
                    call(node.inputs[1].index())
                ),
                Operation::Mul => format!(
                    "VeriNeural.mul {} {}",
                    call(node.inputs[0].index()),
                    call(node.inputs[1].index())
                ),
                Operation::Relu => {
                    format!("VeriNeural.relu {}", call(node.inputs[0].index()))
                }
                Operation::Reshape(_) => {
                    format!("VeriNeural.reshape {}", call(node.inputs[0].index()))
                }
                _ => unreachable!("checked while collecting ancestors"),
            };

            source.push_str(&format!(
                "def node_{index} {params} : VeriNeural.Tensor {count} :=\n  {rhs}\n\n"
            ));

            let name = format!("node_{index}_shape");
            emit_decidable(
                &mut source,
                &mut theorems,
                name,
                format!("VeriNeural.shapeSize {} = {count}", shape_expr(&node.shape)),
            );

            for (port, &input_id) in node.inputs.iter().enumerate() {
                let predecessor = graph.node(input_id)?;
                let name = format!("node_{index}_input_{port}_shape");
                let premise = if matches!(node.operation, Operation::Reshape(_)) {
                    format!(
                        "VeriNeural.shapeSize {} = VeriNeural.shapeSize {}",
                        shape_expr(&predecessor.shape),
                        shape_expr(&node.shape)
                    )
                } else {
                    format!(
                        "{} = {}",
                        shape_expr(&predecessor.shape),
                        shape_expr(&node.shape)
                    )
                };
                emit_decidable(&mut source, &mut theorems, name, premise);
            }

            let name = format!("node_{index}_semantics");
            source.push_str(&format!(
                "theorem {name} {params} (i : Fin {count}) :\n  node_{index} {arguments} i = ({rhs}) i := by\n  rfl\n\n"
            ));
            theorems.push(name);

            if node.domain == Domain::NonNegativeFinite {
                let proof = match node.operation {
                    Operation::Input => format!("h_{index}"),
                    Operation::Relu => {
                        format!(
                            "VeriNeural.relu_nonnegative {}",
                            call(node.inputs[0].index())
                        )
                    }
                    Operation::Reshape(_) => {
                        let prev = node.inputs[0].index();
                        format!(
                            "VeriNeural.reshape_nonnegative {} ({})",
                            call(prev),
                            proof_call(prev)
                        )
                    }
                    _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
                };
                let name = format!("node_{index}_nonnegative");
                source.push_str(&format!(
                    "theorem {name} {proof_params} :\n  VeriNeural.Nonnegative {} := by\n  exact {proof}\n\n",
                    call(index)
                ));
                theorems.push(name);
            }
        }

        for (module_index, module) in graph.modules().iter().enumerate() {
            if !used.contains(&module.output.index()) {
                continue;
            }
            for (port, (&input, required)) in module
                .inputs
                .iter()
                .zip(&module.contract.requires)
                .enumerate()
            {
                let actual = graph.node(input)?;
                let name = format!("module_{module_index}_input_{port}_shape");
                emit_decidable(
                    &mut source,
                    &mut theorems,
                    name,
                    format!(
                        "{} = {}",
                        shape_expr(&actual.shape),
                        shape_expr(&required.shape)
                    ),
                );
                if required.domain == Domain::NonNegativeFinite {
                    if actual.domain != Domain::NonNegativeFinite {
                        return Err(ProofError::UnsupportedGuarantee {
                            node: input.index(),
                        });
                    }
                    emit_nonnegative_alias(
                        &mut source,
                        &mut theorems,
                        format!("module_{module_index}_input_{port}_nonnegative"),
                        &proof_params,
                        &call(input.index()),
                        &proof_call(input.index()),
                    );
                }
            }

            let actual = graph.node(module.output)?;
            emit_decidable(
                &mut source,
                &mut theorems,
                format!("module_{module_index}_output_shape"),
                format!(
                    "{} = {}",
                    shape_expr(&actual.shape),
                    shape_expr(&module.contract.ensures.shape)
                ),
            );
            if module.contract.ensures.domain == Domain::NonNegativeFinite {
                if actual.domain != Domain::NonNegativeFinite {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: module.output.index(),
                    });
                }
                emit_nonnegative_alias(
                    &mut source,
                    &mut theorems,
                    format!("module_{module_index}_output_nonnegative"),
                    &proof_params,
                    &call(module.output.index()),
                    &proof_call(module.output.index()),
                );
            }
        }

        source.push_str("end VeriNeural.Generated\n");
        Ok(LeanProgram {
            source,
            theorems,
            assumptions,
        })
    }

    /// Write and run Lean. Even a successful generation is *unverified* until
    /// this returns Ok. Any Lean warning or diagnostic fails the check.
    pub fn verify(
        &self,
        program: &LeanProgram,
        output_file: &Path,
    ) -> Result<CheckedLean, ProofError> {
        if let Some(parent) = output_file
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(output_file, program.source())?;
        self.check_file(output_file)?;
        Ok(CheckedLean {
            path: output_file.to_path_buf(),
            theorem_names: program.theorems.clone(),
            assumptions: program.assumptions.clone(),
        })
    }

    /// Rerun the trusted Lean checker for a saved file.
    pub fn check_file(&self, source_file: &Path) -> Result<(), ProofError> {
        let output = Command::new(&self.executable).arg(source_file).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() || !stdout.trim().is_empty() || !stderr.trim().is_empty() {
            return Err(ProofError::LeanRejected {
                details: format!(
                    "exit={:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                    output.status.code()
                ),
            });
        }
        Ok(())
    }


fn shape_expr(shape: &Shape) -> String {
    let dims = shape
        .dims()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("([{dims}] : List Nat)")
}

fn emit_decidable(
    source: &mut String,
    theorems: &mut Vec<String>,
    name: String,
    statement: String,
) {
    source.push_str(&format!(
        "theorem {name} :\n  {statement} := by\n  decide\n\n"
    ));
    theorems.push(name);
}

fn emit_nonnegative_alias(
    source: &mut String,
    theorems: &mut Vec<String>,
    name: String,
    proof_params: &str,
    term: &str,
    proof_term: &str,
) {
    source.push_str(&format!(
        "theorem {name} {proof_params} :\n  VeriNeural.Nonnegative {term} := by\n  exact {proof_term}\n\n"
    ));
    theorems.push(name);
}
