//! Lean 4 proof generation for the deliberately narrow, ideal-integer fragment.
//!
//! The verified statement concerns the generated Lean model, not Rust f32
//! execution, FP32 rounding, generated PTX, or physical GPU execution.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{ContractClaim, Domain, NeuralError, NeuralGraph, NodeId, Operation, Shape};

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
    InvalidAssumption {
        node: usize,
    },
    InvalidGoalName,
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
                    "Lean proof backend cannot establish requested guarantee at node {node}"
                )
            }
            Self::InvalidAssumption { node } => {
                write!(
                    f,
                    "global proof assumption at node {node} must be a reachable source input"
                )
            }
            Self::InvalidGoalName => write!(f, "property name must be a nonempty Lean-safe identifier"),
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

#[derive(Clone, Debug)]
struct ModuleNonnegativeProof {
    required_ports: Vec<usize>,
    proof_term: String,
}

/// A requested independent ideal-integer property (not a shape theorem).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProofGoalKind {
    Nonnegative { assumptions: Vec<NodeId> },
    EqualTo { other: NodeId },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofGoal {
    pub name: String,
    pub output: NodeId,
    pub kind: ProofGoalKind,
}

/// Exact audit result for a pinned checker invocation. Zero admitted axioms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeanAxiomAudit {
    pub lean_version: String,
    pub theorem_names: Vec<String>,
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
                | Operation::MatMul
                | Operation::BiasAdd
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
            for module in graph.multi_modules() {
                if module.outputs.contains(&id) {
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
                Operation::MatMul => {
                    let left = graph.node(node.inputs[0])?;
                    let right = graph.node(node.inputs[1])?;
                    let m = left.shape.dims()[0];
                    let k = left.shape.dims()[1];
                    let n = right.shape.dims()[1];
                    format!(
                        "VeriNeural.matmul (m := {m}) (k := {k}) (n := {n}) {} {}",
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index())
                    )
                }
                Operation::BiasAdd => {
                    let m = node.shape.dims()[0];
                    let n = node.shape.dims()[1];
                    format!(
                        "VeriNeural.biasAdd (m := {m}) (n := {n}) {} {}",
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index())
                    )
                }
                Operation::Relu => {
                    format!("VeriNeural.relu {}", call(node.inputs[0].index()))
                }
                Operation::Reshape(_) => {
                    format!("VeriNeural.reshape {}", call(node.inputs[0].index()))
                }
                _ => unreachable!("checked while collecting ancestors"),
            };

            // Every node accepts the shared graph inputs for stable calling
            // conventions. An input node consumes only its own binder; mark
            // the other binders intentionally unused to avoid Lean warnings.
            let definition_params = if matches!(node.operation, Operation::Input) {
                sources
                    .iter()
                    .map(|&source_index| {
                        let binder = if source_index == index {
                            format!("x_{source_index}")
                        } else {
                            format!("_x_{source_index}")
                        };
                        format!(
                            "({binder} : VeriNeural.Tensor {})",
                            graph.nodes()[source_index].shape.element_count()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                params.clone()
            };
            source.push_str(&format!(
                "def node_{index} {definition_params} : VeriNeural.Tensor {count} :=\n  {rhs}\n\n"
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
                let premise = if matches!(node.operation, Operation::MatMul) {
                    let left = graph.node(node.inputs[0])?;
                    let right = graph.node(node.inputs[1])?;
                    format!("{} = {}", left.shape.dims()[1], right.shape.dims()[0])
                } else if matches!(node.operation, Operation::BiasAdd) && port == 1 {
                    format!("{} = {}", predecessor.shape.dims()[0], node.shape.dims()[1])
                } else if matches!(node.operation, Operation::Reshape(_)) {
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
                // Node-level input proofs only consume their own source hypothesis;
                // ReLU nonnegativity is unconditional. Keep every positional
                // parameter so existing proof applications remain unchanged, but
                // mark only genuinely unused binders with Lean's underscore prefix.
                let node_proof_params =
                    if matches!(node.operation, Operation::Input | Operation::Relu) {
                        let hypothesis_binders = assumptions
                            .iter()
                            .map(|assumption| {
                                let source = assumption.input_node;
                                let used =
                                    matches!(node.operation, Operation::Input) && source == index;
                                let binder = if used {
                                    format!("h_{source}")
                                } else {
                                    format!("_h_{source}")
                                };
                                format!("({binder} : VeriNeural.Nonnegative x_{source})")
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        format!("{params} {hypothesis_binders}")
                    } else {
                        proof_params.clone()
                    };
                source.push_str(&format!(
                    "theorem {name} {node_proof_params} :\n  VeriNeural.Nonnegative {} := by\n  exact {proof}\n\n",
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
            for (claim_index, claim) in module.contract.claims.iter().enumerate() {
                let name = format!("module_{module_index}_claim_{claim_index}");
                match claim {
                    ContractClaim::Nonnegative => {
                        if actual.domain != Domain::NonNegativeFinite {
                            return Err(ProofError::UnsupportedGuarantee { node: module.output.index() });
                        }
                        emit_nonnegative_alias(&mut source, &mut theorems, name,
                            &proof_params, &call(module.output.index()),
                            &proof_call(module.output.index()));
                    }
                    ContractClaim::EqualsInput(port) => {
                        let Some(&input) = module.inputs.get(*port) else {
                            return Err(ProofError::UnsupportedGuarantee { node: module.output.index() });
                        };
                        if actual.shape != graph.node(input)?.shape {
                            return Err(ProofError::UnsupportedGuarantee { node: module.output.index() });
                        }
                        source.push_str(&format!(
                            "theorem {name} {params} : {} = {} := by\n  rfl\n\n",
                            call(module.output.index()), call(input.index())
                        ));
                        theorems.push(name);
                    }
                }
            }
        }

        for (module_index, module) in graph.multi_modules().iter().enumerate() {
            if !module.outputs.iter().any(|o| used.contains(&o.index())) { continue; }
            for (port_index, (&output, port)) in module.outputs.iter().zip(&module.contract.outputs).enumerate() {
                if !used.contains(&output.index()) { continue; }
                for (claim_index, claim) in port.claims.iter().enumerate() {
                    let name = format!("multi_{module_index}_port_{port_index}_claim_{claim_index}");
                    let actual = graph.node(output)?;
                    match claim {
                        ContractClaim::Nonnegative => {
                            if actual.domain != Domain::NonNegativeFinite {
                                return Err(ProofError::UnsupportedGuarantee { node: output.index() });
                            }
                            emit_nonnegative_alias(&mut source, &mut theorems, name,
                                &proof_params, &call(output.index()), &proof_call(output.index()));
                        }
                        ContractClaim::EqualsInput(index) => {
                            let Some(&input) = module.inputs.get(*index) else {
                                return Err(ProofError::UnsupportedGuarantee { node: output.index() });
                            };
                            if actual.shape != graph.node(input)?.shape {
                                return Err(ProofError::UnsupportedGuarantee { node: output.index() });
                            }
                            source.push_str(&format!(
                                "theorem {name} {params} : {} = {} := by\n  rfl\n\n",
                                call(output.index()), call(input.index())
                            ));
                            theorems.push(name);
                        }
                    }
                }
            }
        }

        source.push_str("end VeriNeural.Generated\n");
        Ok(LeanProgram {
            source,
            theorems,
            assumptions,
        })
    }

    /// Prove one explicit architecture-level ideal-Int property:
    /// the selected output is elementwise nonnegative under the listed source
    /// hypotheses. This theorem composes top-level module proofs over their
    /// dependency DAG. It does not claim FP32 finiteness or GPU equivalence.
    pub fn generate_global_nonnegative(
        &self,
        graph: &NeuralGraph,
        output: NodeId,
        assumptions: &[NodeId],
    ) -> Result<LeanProgram, ProofError> {
        let mut program = self.generate(graph, output)?;
        let used = collect_used_nodes(graph, output)?;
        let sources = used
            .iter()
            .filter(|&&index| matches!(graph.nodes()[index].operation, Operation::Input))
            .copied()
            .collect::<Vec<_>>();
        let source_set = sources.iter().copied().collect::<BTreeSet<_>>();

        let mut assumption_set = program
            .assumptions
            .iter()
            .map(|assumption| assumption.input_node)
            .collect::<BTreeSet<_>>();
        for &assumption in assumptions {
            let node = graph.node(assumption)?;
            if !matches!(node.operation, Operation::Input)
                || !source_set.contains(&assumption.index())
            {
                return Err(ProofError::InvalidAssumption {
                    node: assumption.index(),
                });
            }
            assumption_set.insert(assumption.index());
        }

        let params = sources
            .iter()
            .map(|&index| {
                format!(
                    "(x_{index} : VeriNeural.Tensor {})",
                    graph.nodes()[index].shape.element_count()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let arguments = sources
            .iter()
            .map(|&index| format!("x_{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let call = |index: usize| format!("(node_{index} {arguments})");

        let modules = graph.modules();
        let eligible = modules
            .iter()
            .enumerate()
            .filter_map(|(index, module)| used.contains(&module.output.index()).then_some(index))
            .collect::<Vec<_>>();
        let top_level = eligible
            .iter()
            .copied()
            .filter(|&child_index| {
                !eligible
                    .iter()
                    .copied()
                    .any(|parent_index| module_contains(graph, parent_index, child_index))
            })
            .collect::<Vec<_>>();

        let mut producers = BTreeMap::<usize, usize>::new();
        for module_index in top_level {
            let output_index = modules[module_index].output.index();
            if producers.insert(output_index, module_index).is_some() {
                return Err(ProofError::UnsupportedGuarantee { node: output_index });
            }
        }
        let Some(&final_module) = producers.get(&output.index()) else {
            return Err(ProofError::UnsupportedGuarantee {
                node: output.index(),
            });
        };

        let mut module_proofs = BTreeMap::<usize, ModuleNonnegativeProof>::new();
        let mut selected = BTreeSet::<usize>::new();
        let mut required_assumptions = BTreeSet::<usize>::new();
        let mut pending = vec![final_module];
        while let Some(module_index) = pending.pop() {
            if !selected.insert(module_index) {
                continue;
            }
            let proof = build_module_nonnegative_proof(
                graph,
                module_index,
                &arguments,
                &mut module_proofs,
            )?;
            for &port in &proof.required_ports {
                let input = modules[module_index].inputs[port];
                if assumption_set.contains(&input.index()) {
                    required_assumptions.insert(input.index());
                    continue;
                }
                let Some(&producer_index) = producers.get(&input.index()) else {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: input.index(),
                    });
                };
                if producer_index >= module_index {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: input.index(),
                    });
                }
                pending.push(producer_index);
            }
        }

        let namespace_end = "end VeriNeural.Generated\n";
        if !program.source.ends_with(namespace_end) {
            return Err(ProofError::LeanRejected {
                details: "generated Lean source has no namespace terminator".to_owned(),
            });
        }
        program
            .source
            .truncate(program.source.len() - namespace_end.len());

        for (&module_index, proof) in &module_proofs {
            let module = &modules[module_index];
            let proof_params = proof
                .required_ports
                .iter()
                .map(|&port| {
                    let input = module.inputs[port].index();
                    format!(
                        "(h_module_{module_index}_input_{port} : VeriNeural.Nonnegative {})",
                        call(input)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            let theorem_params = if proof_params.is_empty() {
                params.clone()
            } else {
                format!("{params} {proof_params}")
            };
            let name = format!("module_{module_index}_global_nonnegative");
            program.source.push_str(&format!(
                "-- Ideal-Int module property; this is not an FP32/GPU guarantee.\n\
                 theorem {name} {theorem_params} :\n  VeriNeural.Nonnegative {} := by\n  exact {}\n\n",
                call(module.output.index()),
                proof.proof_term
            ));
            program.theorems.push(name);
        }

        let global_hypotheses = required_assumptions
            .iter()
            .map(|&index| format!("(arch_h_{index} : VeriNeural.Nonnegative x_{index})"))
            .collect::<Vec<_>>()
            .join(" ");
        let global_params = if global_hypotheses.is_empty() {
            params.clone()
        } else {
            format!("{params} {global_hypotheses}")
        };
        program.source.push_str(&format!(
            "-- Explicit user-requested architecture property in the ideal-Int model.\n\
             theorem architecture_global_nonnegative {global_params} :\n  \
             VeriNeural.Nonnegative {} := by\n",
            call(output.index())
        ));

        let mut proofs = required_assumptions
            .iter()
            .map(|&index| (index, format!("arch_h_{index}")))
            .collect::<BTreeMap<_, _>>();
        for &module_index in &selected {
            let module = &modules[module_index];
            let proof = &module_proofs[&module_index];
            let mut args = sources
                .iter()
                .map(|&index| format!("x_{index}"))
                .collect::<Vec<_>>();
            for &port in &proof.required_ports {
                let input = module.inputs[port].index();
                let Some(input_proof) = proofs.get(&input) else {
                    return Err(ProofError::UnsupportedGuarantee { node: input });
                };
                args.push(input_proof.clone());
            }
            let have_name = format!("arch_module_{module_index}");
            program.source.push_str(&format!(
                "  have {have_name} := module_{module_index}_global_nonnegative {}\n",
                args.join(" ")
            ));
            proofs.insert(module.output.index(), have_name);
        }
        let Some(final_proof) = proofs.get(&output.index()) else {
            return Err(ProofError::UnsupportedGuarantee {
                node: output.index(),
            });
        };
        program
            .source
            .push_str(&format!("  exact {final_proof}\n\n{namespace_end}"));
        program
            .theorems
            .push("architecture_global_nonnegative".to_owned());

        for index in required_assumptions {
            if !program
                .assumptions
                .iter()
                .any(|assumption| assumption.input_node == index)
            {
                program.assumptions.push(ProofAssumption {
                    input_node: index,
                    domain: Domain::NonNegativeFinite,
                });
            }
        }
        program
            .assumptions
            .sort_by_key(|assumption| assumption.input_node);

        Ok(program)
    }

    /// Produce an independently requested theorem. Every goal gets its own
    /// program and result; invalid equality is rejected by Lean, never inferred
    /// from the output shape or a generator-created RHS.
    pub fn generate_goal(&self, graph: &NeuralGraph, goal: &ProofGoal) -> Result<LeanProgram, ProofError> {
        let valid = !goal.name.is_empty() && goal.name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            && goal.name.as_bytes()[0].is_ascii_alphabetic();
        if !valid { return Err(ProofError::InvalidGoalName); }
        match &goal.kind {
            ProofGoalKind::Nonnegative { assumptions } => {
                let mut program = self.generate_global_nonnegative(graph, goal.output, assumptions)?;
                program.source = program.source.replace("architecture_global_nonnegative",
                    &format!("architecture_property_{}", goal.name));
                if let Some(last) = program.theorems.last_mut() {
                    *last = format!("architecture_property_{}", goal.name);
                }
                Ok(program)
            }
            ProofGoalKind::EqualTo { other } => {
                let used = collect_used_nodes(graph, goal.output)?;
                if !used.contains(&other.index()) || graph.node(goal.output)?.shape != graph.node(*other)?.shape {
                    return Err(ProofError::UnsupportedGuarantee { node: other.index() });
                }
                let mut program = self.generate(graph, goal.output)?;
                let source_nodes = used.iter()
                    .filter(|&&index| matches!(graph.nodes()[index].operation, Operation::Input))
                    .copied().collect::<Vec<_>>();
                let params = source_nodes.iter().map(|&i| format!("(x_{i} : VeriNeural.Tensor {})",
                    graph.nodes()[i].shape.element_count())).collect::<Vec<_>>().join(" ");
                let args = source_nodes.iter().map(|&i| format!("x_{i}")).collect::<Vec<_>>().join(" ");
                let terminator = "end VeriNeural.Generated\n";
                program.source.truncate(program.source.len() - terminator.len());
                let name = format!("architecture_property_{}", goal.name);
                program.source.push_str(&format!(
                    "-- User-owned equality goal, independently checked by Lean.\n                     theorem {name} {params} : node_{} {args} = node_{} {args} := by\n  rfl\n\n{terminator}",
                    goal.output.index(), other.index()
                ));
                program.theorems.push(name);
                Ok(program)
            }
        }
    }

    /// Stronger checker profile: all transitive axioms must be absent. Lean's
    /// kernel, its imported library and executable version remain explicit TCB.
    pub fn verify_certified(
        &self, program: &LeanProgram, output_file: &Path,
    ) -> Result<(CheckedLean, LeanAxiomAudit), ProofError> {
        let checked = self.verify(program, output_file)?;
        let audit = self.audit_file(output_file, &program.theorems)?;
        Ok((checked, audit))
    }

    /// Recheck transitive axioms in a saved Lean artifact. No user axiom,
    /// sorryAx, native_decide assumption or other axiom can be admitted here.
    pub fn audit_file(&self, source_file: &Path, theorems: &[String]) -> Result<LeanAxiomAudit, ProofError> {
        if theorems.is_empty() || theorems.iter().any(|name| name.is_empty()
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'))
        {
            return Err(ProofError::InvalidGoalName);
        }
        let version = Command::new(&self.executable).arg("--version").output()?;
        if !version.status.success() {
            return Err(ProofError::LeanRejected { details: "Lean version unavailable".into() });
        }
        let version = String::from_utf8_lossy(&version.stdout).trim().to_owned();
        let audit_path = source_file.with_extension("axiom-audit.lean");
        let mut audit_source = fs::read_to_string(source_file)?;
        for name in theorems {
            audit_source.push_str(&format!("\n#print axioms VeriNeural.Generated.{name}\n"));
        }
        fs::write(&audit_path, audit_source)?;
        let result = Command::new(&self.executable).arg(&audit_path).output();
        let _ = fs::remove_file(&audit_path);
        let result = result?;
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        let lines = stdout.lines().filter(|line| !line.trim().is_empty()).collect::<Vec<_>>();
        if !result.status.success() || !stderr.trim().is_empty()
            || lines.len() != theorems.len()
            || lines.iter().any(|line| !line.contains("does not depend on any axioms"))
        {
            return Err(ProofError::LeanRejected {
                details: format!("certified axiom audit failed; exit={:?}\nstdout:\n{stdout}\nstderr:\n{stderr}", result.status.code()),
            });
        }
        Ok(LeanAxiomAudit { lean_version: version, theorem_names: theorems.to_vec() })
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
}

fn collect_used_nodes(graph: &NeuralGraph, output: NodeId) -> Result<BTreeSet<usize>, ProofError> {
    let mut used = BTreeSet::new();
    let mut pending = vec![output];
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

fn module_contains(graph: &NeuralGraph, parent_index: usize, child_index: usize) -> bool {
    if child_index >= parent_index {
        return false;
    }
    let modules = graph.modules();
    let parent = &modules[parent_index];
    let child = &modules[child_index];
    child.body_start >= parent.body_start
        && child.output.index() >= parent.body_start
        && child.output.index() <= parent.output.index()
}

#[derive(Default)]
struct NonnegativeTraversal {
    required_ports: BTreeSet<usize>,
    cache: BTreeMap<usize, String>,
}

fn build_module_nonnegative_proof(
    graph: &NeuralGraph,
    module_index: usize,
    arguments: &str,
    module_proofs: &mut BTreeMap<usize, ModuleNonnegativeProof>,
) -> Result<ModuleNonnegativeProof, ProofError> {
    if let Some(proof) = module_proofs.get(&module_index) {
        return Ok(proof.clone());
    }
    let module = &graph.modules()[module_index];
    let boundary = module
        .inputs
        .iter()
        .enumerate()
        .map(|(port, input)| (input.index(), port))
        .collect::<BTreeMap<_, _>>();
    let mut traversal = NonnegativeTraversal::default();
    let proof_term = prove_module_node_nonnegative(
        graph,
        module_index,
        module.output.index(),
        arguments,
        &boundary,
        &mut traversal,
        module_proofs,
    )?;
    let proof = ModuleNonnegativeProof {
        required_ports: traversal.required_ports.into_iter().collect(),
        proof_term,
    };
    module_proofs.insert(module_index, proof.clone());
    Ok(proof)
}

fn prove_module_node_nonnegative(
    graph: &NeuralGraph,
    module_index: usize,
    index: usize,
    arguments: &str,
    boundary: &BTreeMap<usize, usize>,
    traversal: &mut NonnegativeTraversal,
    module_proofs: &mut BTreeMap<usize, ModuleNonnegativeProof>,
) -> Result<String, ProofError> {
    if let Some(proof) = traversal.cache.get(&index) {
        return Ok(proof.clone());
    }
    if let Some(&port) = boundary.get(&index) {
        traversal.required_ports.insert(port);
        return Ok(format!("h_module_{module_index}_input_{port}"));
    }

    let module = &graph.modules()[module_index];
    if index < module.body_start || index > module.output.index() {
        return Err(ProofError::UnsupportedGuarantee { node: index });
    }

    // A direct child is selected by IR body ranges. Grandchildren cannot
    // bypass their immediate parent theorem even when outputs are identical.
    let children = (0..module_index)
        .filter(|&child_index| {
            module_contains(graph, module_index, child_index)
                && graph.modules()[child_index].output.index() == index
                && !(0..module_index).any(|parent_index| {
                    parent_index != child_index
                        && module_contains(graph, module_index, parent_index)
                        && module_contains(graph, parent_index, child_index)
                })
        })
        .collect::<Vec<_>>();
    if children.len() > 1 {
        return Err(ProofError::UnsupportedGuarantee { node: index });
    }
    if let Some(&child_index) = children.first() {
        let child_proof =
            build_module_nonnegative_proof(graph, child_index, arguments, module_proofs)?;
        let mut args = vec![arguments.to_owned()];
        for &port in &child_proof.required_ports {
            let child_input = graph.modules()[child_index].inputs[port].index();
            let input_proof = prove_module_node_nonnegative(
                graph,
                module_index,
                child_input,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            args.push(format!("({input_proof})"));
        }
        let result = format!(
            "(module_{child_index}_global_nonnegative {})",
            args.join(" ")
        );
        traversal.cache.insert(index, result.clone());
        return Ok(result);
    }

    let node = &graph.nodes()[index];
    let call = |node_index: usize| format!("(node_{node_index} {arguments})");
    let proof = match node.operation {
        Operation::Add | Operation::Mul | Operation::MatMul | Operation::BiasAdd => {
            let left = node.inputs[0].index();
            let right = node.inputs[1].index();
            let left_proof = prove_module_node_nonnegative(
                graph,
                module_index,
                left,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            let right_proof = prove_module_node_nonnegative(
                graph,
                module_index,
                right,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            let theorem = match node.operation {
                Operation::Add => "VeriNeural.add_nonnegative".to_owned(),
                Operation::Mul => "VeriNeural.mul_nonnegative".to_owned(),
                Operation::MatMul => {
                    let shape_a = graph.nodes()[left].shape.dims();
                    let shape_b = graph.nodes()[right].shape.dims();
                    format!(
                        "VeriNeural.matmul_nonnegative (m := {}) (k := {}) (n := {})",
                        shape_a[0], shape_a[1], shape_b[1]
                    )
                }
                Operation::BiasAdd => {
                    let shape = node.shape.dims();
                    format!(
                        "VeriNeural.biasAdd_nonnegative (m := {}) (n := {})",
                        shape[0], shape[1]
                    )
                }
                _ => unreachable!("handled two-input operations"),
            };
            format!(
                "{theorem} {} {} ({left_proof}) ({right_proof})",
                call(left),
                call(right)
            )
        }
        Operation::Relu => {
            format!(
                "VeriNeural.relu_nonnegative {}",
                call(node.inputs[0].index())
            )
        }
        Operation::Reshape(_) => {
            let input = node.inputs[0].index();
            let input_proof = prove_module_node_nonnegative(
                graph,
                module_index,
                input,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            format!(
                "VeriNeural.reshape_nonnegative {} ({input_proof})",
                call(input)
            )
        }
        _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
    };
    traversal.cache.insert(index, proof.clone());
    Ok(proof)
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
