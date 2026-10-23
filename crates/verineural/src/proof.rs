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

use crate::{
    ContractClaim, DType, Domain, FixedScan, NeuralError, NeuralGraph, NodeId, Operation,
    Predicate, Shape, ValueRef,
};

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
    InvalidOutputs,
    InvalidScan,
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
            Self::InvalidGoalName => {
                write!(f, "property name must be a nonempty Lean-safe identifier")
            }
            Self::InvalidOutputs => {
                write!(
                    f,
                    "Lean proof requires a nonempty set of distinct output nodes"
                )
            }
            Self::InvalidScan => write!(f, "invalid bounded scan or missing state invariant claim"),
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
    Nonnegative {
        assumptions: Vec<NodeId>,
    },
    EqualTo {
        other: NodeId,
    },
    /// A single independent, explicitly typed mathematical relation.
    Predicate(Predicate),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofGoal {
    pub name: String,
    pub output: NodeId,
    pub kind: ProofGoalKind,
}

/// The certified policy admits Lean's foundational `propext` axiom as TCB.
/// Additional axioms require an explicit policy review.
const ALLOWED_LEAN_AXIOMS: &[&str] = &["propext"];

/// Exact audit result, including allowed transitive axiom dependencies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeanAxiomAudit {
    pub lean_version: String,
    pub theorem_names: Vec<String>,
    pub axiom_dependencies: BTreeMap<String, Vec<String>>,
}

/// Require one exact Lean response for every expected theorem.
/// Fail closed on malformed output or any non-allowlisted dependency.
fn parse_axiom_report(
    stdout: &str,
    theorems: &[String],
) -> Result<BTreeMap<String, Vec<String>>, ProofError> {
    let lines = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if lines.len() != theorems.len() {
        return Err(ProofError::LeanRejected {
            details: "unexpected axiom report length".into(),
        });
    }
    let mut dependencies = BTreeMap::new();
    for (name, line) in theorems.iter().zip(lines) {
        let expected = format!("'VeriNeural.Generated.{name}'");
        let rest = line
            .strip_prefix(&expected)
            .ok_or_else(|| ProofError::LeanRejected {
                details: format!("unexpected axiom theorem identity: {line}"),
            })?;
        let axioms = if rest == " does not depend on any axioms" {
            Vec::new()
        } else if let Some(items) = rest
            .strip_prefix(" depends on axioms: [")
            .and_then(|s| s.strip_suffix(']'))
        {
            items.split(", ").map(str::to_owned).collect::<Vec<_>>()
        } else {
            return Err(ProofError::LeanRejected {
                details: format!("malformed axiom report: {line}"),
            });
        };
        if axioms
            .iter()
            .any(|name| name.is_empty() || !ALLOWED_LEAN_AXIOMS.contains(&name.as_str()))
        {
            return Err(ProofError::LeanRejected {
                details: format!("unapproved axiom dependencies for {name}: {axioms:?}"),
            });
        }
        if dependencies.insert(name.clone(), axioms).is_some() {
            return Err(ProofError::LeanRejected {
                details: format!("duplicate theorem: {name}"),
            });
        }
    }
    Ok(dependencies)
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
        self.generate_outputs(graph, &[output])
    }

    /// Generate one consistent Lean model and *all* reachable obligations for
    /// several outputs, including distinct named ports of a multi-output module.
    /// This is a single ideal-Int proof artifact, not a GPU or FP32 certificate.
    /// Requests are required to be nonempty and distinct; foreign nodes fail.
    pub fn generate_outputs(
        &self,
        graph: &NeuralGraph,
        outputs: &[NodeId],
    ) -> Result<LeanProgram, ProofError> {
        if outputs.is_empty() {
            return Err(ProofError::InvalidOutputs);
        }
        let mut seen = BTreeSet::new();
        for &output in outputs {
            graph.node(output)?;
            if !seen.insert(output.index()) {
                return Err(ProofError::InvalidOutputs);
            }
        }
        let used = graph.dependency_scope(outputs)?;
        for &index in &used {
            let node = &graph.nodes()[index];
            match node.operation {
                Operation::Pointwise(ref spec)
                    if !spec.expression().has_ideal_int_semantics() =>
                {
                    return Err(ProofError::UnsupportedOperation {
                        node: index,
                        operation: node.operation.name(),
                    });
                }
                Operation::Input
                | Operation::Add
                | Operation::Mul
                | Operation::MatMul
                | Operation::BiasAdd
                | Operation::Relu
                | Operation::PositiveMask
                | Operation::SelectPositive
                | Operation::Reshape(_)
                | Operation::Pointwise(_)
                | Operation::ReduceSum { .. }
                | Operation::ReverseAxis { .. }
                | Operation::Concat { .. }
                | Operation::ArgMax { .. }
                | Operation::Take { .. } => {}
                Operation::Permute { ref axes } => {
                    let input = graph.node(node.inputs[0])?;
                    let identity = axes.iter().copied().eq(0..axes.len());
                    let transpose_2d = input.shape.rank() == 2 && axes.as_slice() == [1, 0];
                    if !identity && !transpose_2d {
                        return Err(ProofError::UnsupportedOperation {
                            node: index,
                            operation: node.operation.name(),
                        });
                    }
                }
                _ => {
                    return Err(ProofError::UnsupportedOperation {
                        node: index,
                        operation: node.operation.name(),
                    });
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
                    "(x_{i} : {})",
                    lean_value_type(
                        graph.nodes()[i].dtype,
                        graph.nodes()[i].shape.element_count()
                    )
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
        let source_hypotheses = assumptions
            .iter()
            .map(|assumption| assumption.input_node)
            .collect::<BTreeSet<_>>();
        let claim_context = IdealNonnegativeClaimContext {
            params: &params,
            arguments: &arguments,
            assumptions: &assumptions,
            source_hypotheses: &source_hypotheses,
        };

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
                Operation::PositiveMask => {
                    format!("VeriNeural.positiveMask {}", call(node.inputs[0].index()))
                }
                Operation::SelectPositive => {
                    format!(
                        "VeriNeural.selectPositive {} {} {}",
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index()),
                        call(node.inputs[2].index())
                    )
                }
                Operation::Reshape(_) => {
                    format!("VeriNeural.reshape {}", call(node.inputs[0].index()))
                }
                Operation::Pointwise(spec) => spec
                    .expression()
                    .lean(&|port| call(node.inputs[port].index())),
                Operation::ReduceSum { axis } => {
                    let input = graph.node(node.inputs[0])?;
                    let (outer, width, inner) = axis_factors(&input.shape, *axis);
                    format!(
                        "VeriNeural.reduceSumAxis (outer := {outer}) (width := {width}) \
                         (inner := {inner}) {}",
                        call(node.inputs[0].index())
                    )
                }
                Operation::Permute { axes } => {
                    let input = graph.node(node.inputs[0])?;
                    if axes.iter().copied().eq(0..axes.len()) {
                        format!("VeriNeural.reshape {}", call(node.inputs[0].index()))
                    } else {
                        let dims = input.shape.dims();
                        format!(
                            "VeriNeural.transpose2D (rows := {}) (cols := {}) {}",
                            dims[0],
                            dims[1],
                            call(node.inputs[0].index())
                        )
                    }
                }
                Operation::ReverseAxis { axis } => {
                    let input = graph.node(node.inputs[0])?;
                    let (outer, width, inner) = axis_factors(&input.shape, *axis);
                    format!(
                        "VeriNeural.reverseAxis (outer := {outer}) (width := {width}) \
                         (inner := {inner}) {}",
                        call(node.inputs[0].index())
                    )
                }
                Operation::Concat { axis } => {
                    let left = graph.node(node.inputs[0])?;
                    let right = graph.node(node.inputs[1])?;
                    let (outer, left_width, inner) = axis_factors(&left.shape, *axis);
                    let (_, right_width, _) = axis_factors(&right.shape, *axis);
                    format!(
                        "VeriNeural.concatAxis (outer := {outer}) (leftWidth := {left_width}) \
                         (rightWidth := {right_width}) (inner := {inner}) {} {}",
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index())
                    )
                }
                Operation::ArgMax { axis, .. } => {
                    let input = graph.node(node.inputs[0])?;
                    let (outer, width, inner) = axis_factors(&input.shape, *axis);
                    format!(
                        "VeriNeural.argMaxFirstAxis (outer := {outer}) (width := {width}) \
                         (inner := {inner}) {}",
                        call(node.inputs[0].index())
                    )
                }
                Operation::Take { axis } => {
                    let data = graph.node(node.inputs[0])?;
                    let (outer, width, inner) = axis_factors(&data.shape, *axis);
                    format!(
                        "VeriNeural.takeAxis (outer := {outer}) (width := {width}) \
                         (inner := {inner}) {} {}",
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index())
                    )
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
                            "({binder} : {})",
                            lean_value_type(
                                graph.nodes()[source_index].dtype,
                                graph.nodes()[source_index].shape.element_count()
                            )
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                params.clone()
            };
            let value_type = lean_value_type(node.dtype, count);
            source.push_str(&format!(
                "def node_{index} {definition_params} : {value_type} :=\n  {rhs}\n\n"
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
                } else if matches!(node.operation, Operation::ReduceSum { .. }) {
                    let dims = predecessor.shape.dims();
                    let axis = match &node.operation {
                        Operation::ReduceSum { axis } => *axis,
                        _ => unreachable!(),
                    };
                    let outer = dims[..axis].iter().product::<usize>();
                    let width = dims[axis];
                    let inner = dims[axis + 1..].iter().product::<usize>();
                    format!(
                        "VeriNeural.shapeSize {} = {outer} * {width} * {inner}",
                        shape_expr(&predecessor.shape)
                    )
                } else if matches!(node.operation, Operation::Permute { .. }) {
                    format!(
                        "VeriNeural.shapeSize {} = VeriNeural.shapeSize {}",
                        shape_expr(&predecessor.shape),
                        shape_expr(&node.shape)
                    )
                } else if matches!(node.operation, Operation::Concat { .. }) {
                    let axis = match &node.operation {
                        Operation::Concat { axis } => *axis,
                        _ => unreachable!(),
                    };
                    let (outer, width, inner) = axis_factors(&predecessor.shape, axis);
                    format!(
                        "VeriNeural.shapeSize {} = {outer} * {width} * {inner}",
                        shape_expr(&predecessor.shape)
                    )
                } else if matches!(node.operation, Operation::ArgMax { .. }) {
                    let axis = match &node.operation {
                        Operation::ArgMax { axis, .. } => *axis,
                        _ => unreachable!(),
                    };
                    let (outer, width, inner) = axis_factors(&predecessor.shape, axis);
                    format!(
                        "VeriNeural.shapeSize {} = {outer} * {width} * {inner}",
                        shape_expr(&predecessor.shape)
                    )
                } else if matches!(node.operation, Operation::Take { .. }) {
                    if port == 0 {
                        let axis = match &node.operation {
                            Operation::Take { axis } => *axis,
                            _ => unreachable!(),
                        };
                        let (outer, width, inner) = axis_factors(&predecessor.shape, axis);
                        format!(
                            "VeriNeural.shapeSize {} = {outer} * {width} * {inner}",
                            shape_expr(&predecessor.shape)
                        )
                    } else {
                        format!(
                            "VeriNeural.shapeSize {} = 1",
                            shape_expr(&predecessor.shape)
                        )
                    }
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
                    Operation::PositiveMask => {
                        format!(
                            "VeriNeural.positiveMask_nonnegative {}",
                            call(node.inputs[0].index())
                        )
                    }
                    Operation::SelectPositive => {
                        let yes = node.inputs[1].index();
                        let no = node.inputs[2].index();
                        format!(
                            "VeriNeural.selectPositive_nonnegative {} {} {} ({}) ({})",
                            call(node.inputs[0].index()),
                            call(yes),
                            call(no),
                            proof_call(yes),
                            proof_call(no)
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
                    Operation::Permute { ref axes } => {
                        let prev = node.inputs[0].index();
                        if axes.iter().copied().eq(0..axes.len()) {
                            format!(
                                "VeriNeural.reshape_nonnegative {} ({})",
                                call(prev),
                                proof_call(prev)
                            )
                        } else {
                            let dims = graph.nodes()[prev].shape.dims();
                            format!(
                                "VeriNeural.transpose2D_nonnegative (rows := {}) (cols := {}) {} ({})",
                                dims[0],
                                dims[1],
                                call(prev),
                                proof_call(prev)
                            )
                        }
                    }
                    Operation::ReverseAxis { axis } => {
                        let prev = node.inputs[0].index();
                        let (outer, width, inner) = axis_factors(&graph.nodes()[prev].shape, axis);
                        format!(
                            "VeriNeural.reverseAxis_nonnegative (outer := {outer}) \
                             (width := {width}) (inner := {inner}) {} ({})",
                            call(prev),
                            proof_call(prev)
                        )
                    }
                    Operation::Concat { axis } => {
                        let left = node.inputs[0].index();
                        let right = node.inputs[1].index();
                        let (outer, left_width, inner) =
                            axis_factors(&graph.nodes()[left].shape, axis);
                        let (_, right_width, _) = axis_factors(&graph.nodes()[right].shape, axis);
                        format!(
                            "VeriNeural.concatAxis_nonnegative (outer := {outer}) \
                             (leftWidth := {left_width}) (rightWidth := {right_width}) \
                             (inner := {inner}) {} {} ({}) ({})",
                            call(left),
                            call(right),
                            proof_call(left),
                            proof_call(right)
                        )
                    }
                    Operation::Take { axis } => {
                        let data = node.inputs[0].index();
                        let selected = node.inputs[1].index();
                        let (outer, width, inner) = axis_factors(&graph.nodes()[data].shape, axis);
                        format!(
                            "VeriNeural.takeAxis_nonnegative (outer := {outer}) \
                             (width := {width}) (inner := {inner}) {} {} ({})",
                            call(data),
                            call(selected),
                            proof_call(data)
                        )
                    }
                    _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
                };
                let name = format!("node_{index}_nonnegative");
                // Node-level input proofs only consume their own source hypothesis;
                // ReLU nonnegativity is unconditional. Keep every positional
                // parameter so existing proof applications remain unchanged, but
                // mark only genuinely unused binders with Lean's underscore prefix.
                let node_proof_params = if matches!(
                    node.operation,
                    Operation::Input | Operation::Relu | Operation::PositiveMask
                ) {
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
                        emit_ideal_nonnegative_claim(
                            graph,
                            &mut source,
                            &mut theorems,
                            name,
                            module.output,
                            &claim_context,
                        )?;
                    }
                    ContractClaim::EqualsInput(port) => {
                        let Some(&input) = module.inputs.get(*port) else {
                            return Err(ProofError::UnsupportedGuarantee {
                                node: module.output.index(),
                            });
                        };
                        let input_node = graph.node(input)?;
                        if actual.shape != input_node.shape || actual.dtype != input_node.dtype {
                            return Err(ProofError::UnsupportedGuarantee {
                                node: module.output.index(),
                            });
                        }
                        source.push_str(&format!(
                            "theorem {name} {params} : {} = {} := by\n  rfl\n\n",
                            call(module.output.index()),
                            call(input.index())
                        ));
                        theorems.push(name);
                    }
                    ContractClaim::Predicate(predicate) => {
                        emit_module_predicate(
                            graph,
                            &mut source,
                            &mut theorems,
                            name,
                            predicate,
                            (&module.inputs, module.output),
                            &claim_context,
                        )?;
                    }
                }
            }
        }

        for (module_index, module) in graph.multi_modules().iter().enumerate() {
            if !module.outputs.iter().any(|o| used.contains(&o.index())) {
                continue;
            }

            // All input ports are part of the declared module interface, even
            // if a particular output does not use the associated tensor.
            for (port_index, (&input, port)) in module
                .inputs
                .iter()
                .zip(&module.contract.inputs)
                .enumerate()
            {
                let actual = graph.node(input)?;
                emit_decidable(
                    &mut source,
                    &mut theorems,
                    format!("multi_{module_index}_input_{port_index}_shape"),
                    format!(
                        "{} = {}",
                        shape_expr(&actual.shape),
                        shape_expr(&port.port.shape)
                    ),
                );
                if port.port.domain == Domain::NonNegativeFinite {
                    if actual.domain != Domain::NonNegativeFinite {
                        return Err(ProofError::UnsupportedGuarantee {
                            node: input.index(),
                        });
                    }
                    emit_nonnegative_alias(
                        &mut source,
                        &mut theorems,
                        format!("multi_{module_index}_input_{port_index}_nonnegative"),
                        &proof_params,
                        &call(input.index()),
                        &proof_call(input.index()),
                    );
                }
            }

            for (port_index, (&output, port)) in module
                .outputs
                .iter()
                .zip(&module.contract.outputs)
                .enumerate()
            {
                if !used.contains(&output.index()) {
                    continue;
                }
                let actual = graph.node(output)?;
                emit_decidable(
                    &mut source,
                    &mut theorems,
                    format!("multi_{module_index}_port_{port_index}_shape"),
                    format!(
                        "{} = {}",
                        shape_expr(&actual.shape),
                        shape_expr(&port.port.shape)
                    ),
                );
                if port.port.domain == Domain::NonNegativeFinite {
                    if actual.domain != Domain::NonNegativeFinite {
                        return Err(ProofError::UnsupportedGuarantee {
                            node: output.index(),
                        });
                    }
                    emit_nonnegative_alias(
                        &mut source,
                        &mut theorems,
                        format!("multi_{module_index}_port_{port_index}_nonnegative"),
                        &proof_params,
                        &call(output.index()),
                        &proof_call(output.index()),
                    );
                }
                for (claim_index, claim) in port.claims.iter().enumerate() {
                    let name =
                        format!("multi_{module_index}_port_{port_index}_claim_{claim_index}");
                    match claim {
                        ContractClaim::Nonnegative => {
                            emit_ideal_nonnegative_claim(
                                graph,
                                &mut source,
                                &mut theorems,
                                name,
                                output,
                                &claim_context,
                            )?;
                        }
                        ContractClaim::EqualsInput(index) => {
                            let Some(&input) = module.inputs.get(*index) else {
                                return Err(ProofError::UnsupportedGuarantee {
                                    node: output.index(),
                                });
                            };
                            let input_node = graph.node(input)?;
                            if actual.shape != input_node.shape || actual.dtype != input_node.dtype
                            {
                                return Err(ProofError::UnsupportedGuarantee {
                                    node: output.index(),
                                });
                            }
                            source.push_str(&format!(
                                "theorem {name} {params} : {} = {} := by\n  rfl\n\n",
                                call(output.index()),
                                call(input.index())
                            ));
                            theorems.push(name);
                        }
                        ContractClaim::Predicate(predicate) => {
                            emit_module_predicate(
                                graph,
                                &mut source,
                                &mut theorems,
                                name,
                                predicate,
                                (&module.inputs, output),
                                &claim_context,
                            )?;
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
    pub fn generate_goal(
        &self,
        graph: &NeuralGraph,
        goal: &ProofGoal,
    ) -> Result<LeanProgram, ProofError> {
        let valid = !goal.name.is_empty()
            && goal
                .name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_')
            && goal.name.as_bytes()[0].is_ascii_alphabetic();
        if !valid {
            return Err(ProofError::InvalidGoalName);
        }
        match &goal.kind {
            ProofGoalKind::Nonnegative { assumptions } => {
                let mut program =
                    self.generate_global_nonnegative(graph, goal.output, assumptions)?;
                program.source = program.source.replace(
                    "architecture_global_nonnegative",
                    &format!("architecture_property_{}", goal.name),
                );
                if let Some(last) = program.theorems.last_mut() {
                    *last = format!("architecture_property_{}", goal.name);
                }
                Ok(program)
            }
            // Legacy public constructors are adapters to the same predicate
            // proof path, rather than maintaining another theorem generator.
            ProofGoalKind::EqualTo { other } => self.generate_goal(
                graph,
                &ProofGoal {
                    name: goal.name.clone(),
                    output: goal.output,
                    kind: ProofGoalKind::Predicate(Predicate::equal(
                        ValueRef::Output,
                        ValueRef::Node(*other),
                    )),
                },
            ),
            ProofGoalKind::Predicate(predicate) => {
                let resolve = |value: &ValueRef| -> Result<NodeId, ProofError> {
                    let id = match value {
                        ValueRef::Node(id) => *id,
                        ValueRef::Output => goal.output,
                        ValueRef::Input(_) => {
                            return Err(ProofError::UnsupportedGuarantee {
                                node: goal.output.index(),
                            });
                        }
                    };
                    graph.node(id)?;
                    Ok(id)
                };
                match predicate {
                    Predicate::Nonnegative(value) => {
                        if resolve(value)? != goal.output {
                            return Err(ProofError::UnsupportedGuarantee {
                                node: goal.output.index(),
                            });
                        }
                        self.generate_goal(
                            graph,
                            &ProofGoal {
                                name: goal.name.clone(),
                                output: goal.output,
                                kind: ProofGoalKind::Nonnegative {
                                    assumptions: vec![],
                                },
                            },
                        )
                    }
                    Predicate::Equal(a, b) | Predicate::PointwiseLe(a, b) => {
                        let left = resolve(a)?;
                        let right = resolve(b)?;
                        let used = collect_used_nodes(graph, goal.output)?;
                        let left_node = graph.node(left)?;
                        let right_node = graph.node(right)?;
                        if !used.contains(&left.index())
                            || !used.contains(&right.index())
                            || left_node.shape != right_node.shape
                            || left_node.dtype != right_node.dtype
                            || (matches!(predicate, Predicate::PointwiseLe(_, _))
                                && left_node.dtype != DType::F32)
                        {
                            return Err(ProofError::UnsupportedGuarantee {
                                node: goal.output.index(),
                            });
                        }
                        let mut program = self.generate(graph, goal.output)?;
                        let sources = used
                            .iter()
                            .filter(|&&index| {
                                matches!(graph.nodes()[index].operation, Operation::Input)
                            })
                            .copied()
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
                        let args = sources
                            .iter()
                            .map(|&i| format!("x_{i}"))
                            .collect::<Vec<_>>()
                            .join(" ");
                        let terminator = "end VeriNeural.Generated\n";
                        program
                            .source
                            .truncate(program.source.len() - terminator.len());
                        let name = format!("architecture_property_{}", goal.name);
                        let term = |id: NodeId| format!("(node_{} {args})", id.index());
                        let (proposition, proof) = match predicate {
                            Predicate::Equal(_, _) => (
                                format!("{} = {}", term(left), term(right)),
                                "  rfl".to_owned(),
                            ),
                            Predicate::PointwiseLe(_, _) => {
                                // A compositional branch bound, not merely a
                                // restatement of the final ReLU theorem:
                                // lower <= relu(add(lower, relu(other))).
                                let upper = graph.node(right)?;
                                let composed = if matches!(upper.operation, Operation::Relu) {
                                    let sum = graph.node(upper.inputs[0])?;
                                    if matches!(sum.operation, Operation::Add)
                                        && sum.inputs[0] == left
                                    {
                                        let other = sum.inputs[1];
                                        let branch = graph.node(other)?;
                                        if matches!(branch.operation, Operation::Relu) {
                                            let pre_relu = branch.inputs[0];
                                            Some(format!(
                                                "  change VeriNeural.PointwiseLe {} (VeriNeural.relu (VeriNeural.add {} {}))\n  exact VeriNeural.pointwise_le_relu_add_of_nonnegative {} {} (VeriNeural.relu_nonnegative {})",
                                                term(left),
                                                term(left),
                                                term(other),
                                                term(left),
                                                term(other),
                                                term(pre_relu)
                                            ))
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };
                                (
                                    format!(
                                        "VeriNeural.PointwiseLe {} {}",
                                        term(left),
                                        term(right)
                                    ),
                                    composed.unwrap_or_else(|| {
                                        format!(
                                            "  exact VeriNeural.pointwise_le_relu {}",
                                            term(left)
                                        )
                                    }),
                                )
                            }
                            Predicate::Nonnegative(_) => unreachable!(),
                        };
                        program.source.push_str(&format!(
                            "theorem {name} {params} : {proposition} := by\n{proof}\n\n{terminator}"
                        ));
                        program.theorems.push(name);
                        Ok(program)
                    }
                }
            }
        }
    }

    /// Produce a single ideal-Int theorem covering every state of a *fixed*
    /// horizon, by composing per-step consumer-owned Nonnegative claims.
    /// Only the stated horizon is covered. This is not an FP32 or GPU proof.
    pub fn generate_fixed_scan_nonnegative(
        &self,
        graph: &NeuralGraph,
        scan: &FixedScan,
        name: &str,
    ) -> Result<LeanProgram, ProofError> {
        if name.is_empty()
            || !name.as_bytes()[0].is_ascii_alphabetic()
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err(ProofError::InvalidGoalName);
        }
        if scan.outputs.is_empty()
            || scan.states.len() != scan.outputs.len() + 1
            || scan.module_indices.len() != scan.outputs.len()
        {
            return Err(ProofError::InvalidScan);
        }
        let mut claim_names = Vec::new();
        let mut seen = BTreeSet::new();
        for (step, &index) in scan.module_indices.iter().enumerate() {
            let module = graph
                .multi_modules()
                .get(index)
                .ok_or(ProofError::InvalidScan)?;
            if !seen.insert(index)
                || module.inputs.len() != 2
                || module.outputs.len() != 2
                || module.contract.inputs.len() != 2
                || module.contract.outputs.len() != 2
                || module.contract.inputs[0].name != "state"
                || module.contract.inputs[1].name != "input"
                || module.contract.outputs[0].name != "next_state"
                || module.contract.outputs[1].name != "output"
                || module.inputs[0] != scan.states[step]
                || module.outputs[0] != scan.states[step + 1]
                || module.outputs[1] != scan.outputs[step]
            {
                return Err(ProofError::InvalidScan);
            }
            let claim = module.contract.outputs[0]
                .claims
                .iter()
                .position(|c| *c == ContractClaim::Nonnegative)
                .ok_or(ProofError::InvalidScan)?;
            claim_names.push(format!("multi_{index}_port_0_claim_{claim}"));
        }
        let mut targets = scan.proof_outputs();
        targets.push(scan.states[0]);
        let mut program = self.generate_outputs(graph, &targets)?;
        let mut used = BTreeSet::new();
        for &target in &targets {
            used.extend(collect_used_nodes(graph, target)?);
        }
        let sources = used
            .iter()
            .filter(|&&i| matches!(graph.nodes()[i].operation, Operation::Input))
            .copied()
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
        let args = sources
            .iter()
            .map(|i| format!("x_{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let hypotheses = program
            .assumptions()
            .iter()
            .map(|a| {
                format!(
                    "(h_{} : VeriNeural.Nonnegative x_{})",
                    a.input_node, a.input_node
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let hypothesis_args = program
            .assumptions()
            .iter()
            .map(|a| format!("h_{}", a.input_node))
            .collect::<Vec<_>>()
            .join(" ");
        let source_hypotheses = program
            .assumptions()
            .iter()
            .map(|a| a.input_node)
            .collect::<BTreeSet<_>>();
        let base = prove_ideal_nonnegative(
            graph,
            scan.states[0],
            &args,
            &source_hypotheses,
            &mut BTreeMap::new(),
        )?;
        let end = "end VeriNeural.Generated\n";
        if !program.source.ends_with(end) {
            return Err(ProofError::LeanRejected {
                details: "generated Lean source has no namespace terminator".into(),
            });
        }
        program.source.truncate(program.source.len() - end.len());
        let theorem = format!("scan_{name}_global_nonnegative");
        let propositions = scan
            .states
            .iter()
            .map(|state| format!("VeriNeural.Nonnegative (node_{} {args})", state.index()))
            .collect::<Vec<_>>()
            .join(" ∧\n    ");
        program.source.push_str(&format!(
            "-- Ideal-Int states 0..{}; no arbitrary-horizon induction.\n\
             theorem {theorem} {params} {hypotheses} :\n    {propositions} := by\n",
            scan.outputs.len()
        ));
        program.source.push_str(&format!(
            "  have scan_state_0 : VeriNeural.Nonnegative (node_{} {args}) := by\n    exact {base}\n",
            scan.states[0].index()
        ));
        for (step, state) in scan.states.iter().enumerate().skip(1) {
            program.source.push_str(&format!(
                "  have scan_state_{step} : VeriNeural.Nonnegative (node_{} {args}) := by\n    exact {} {args} {hypothesis_args}\n",
                state.index(), claim_names[step - 1]
            ));
        }
        let mut conjunction = format!("scan_state_{}", scan.outputs.len());
        for step in (0..scan.outputs.len()).rev() {
            conjunction = format!("And.intro scan_state_{step} ({conjunction})");
        }
        program
            .source
            .push_str(&format!("  exact {conjunction}\n\n{end}"));
        program.theorems.push(theorem);
        Ok(program)
    }

    /// Certified profile: only explicitly allowlisted foundational Lean axioms
    /// are admitted. Checker, kernel and imported library remain explicit TCB.
    pub fn verify_certified(
        &self,
        program: &LeanProgram,
        output_file: &Path,
    ) -> Result<(CheckedLean, LeanAxiomAudit), ProofError> {
        let checked = self.verify(program, output_file)?;
        let audit = self.audit_file(output_file, &program.theorems)?;
        Ok((checked, audit))
    }

    /// Recheck saved theorems and record actual axioms. User axioms, `sorryAx`
    /// and unreviewed axioms remain forbidden; `propext` is declared TCB.
    pub fn audit_file(
        &self,
        source_file: &Path,
        theorems: &[String],
    ) -> Result<LeanAxiomAudit, ProofError> {
        if theorems.is_empty()
            || theorems.iter().any(|name| {
                name.is_empty() || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            })
        {
            return Err(ProofError::InvalidGoalName);
        }
        let version = Command::new(&self.executable).arg("--version").output()?;
        if !version.status.success() {
            return Err(ProofError::LeanRejected {
                details: "Lean version unavailable".into(),
            });
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
        if !result.status.success() || !stderr.trim().is_empty() {
            return Err(ProofError::LeanRejected {
                details: format!(
                    "axiom audit failed: exit={:?}; stderr={stderr}",
                    result.status.code()
                ),
            });
        }
        let axiom_dependencies = parse_axiom_report(&stdout, theorems)?;
        Ok(LeanAxiomAudit {
            lean_version: version,
            theorem_names: theorems.to_vec(),
            axiom_dependencies,
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
}

fn collect_used_nodes(graph: &NeuralGraph, output: NodeId) -> Result<BTreeSet<usize>, ProofError> {
    Ok(graph.dependency_scope(&[output])?)
}

fn module_contains(graph: &NeuralGraph, parent_index: usize, child_index: usize) -> bool {
    if child_index >= parent_index {
        return false;
    }
    let modules = graph.modules();
    let parent = &modules[parent_index];
    let child = &modules[child_index];
    child.body_start >= parent.body_start && child.body_end <= parent.body_end
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
    if index < module.body_start || index >= module.body_end {
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
        Operation::PositiveMask => {
            format!(
                "VeriNeural.positiveMask_nonnegative {}",
                call(node.inputs[0].index())
            )
        }
        Operation::SelectPositive => {
            let yes = node.inputs[1].index();
            let no = node.inputs[2].index();
            let hy = prove_module_node_nonnegative(
                graph,
                module_index,
                yes,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            let hn = prove_module_node_nonnegative(
                graph,
                module_index,
                no,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            format!(
                "VeriNeural.selectPositive_nonnegative {} {} {} ({hy}) ({hn})",
                call(node.inputs[0].index()),
                call(yes),
                call(no)
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
        Operation::Permute { ref axes } => {
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
            if axes.iter().copied().eq(0..axes.len()) {
                format!(
                    "VeriNeural.reshape_nonnegative {} ({input_proof})",
                    call(input)
                )
            } else {
                let dims = graph.nodes()[input].shape.dims();
                format!(
                    "VeriNeural.transpose2D_nonnegative (rows := {}) (cols := {}) {} ({input_proof})",
                    dims[0],
                    dims[1],
                    call(input)
                )
            }
        }
        Operation::ReverseAxis { axis } => {
            let input = node.inputs[0].index();
            let (outer, width, inner) = axis_factors(&graph.nodes()[input].shape, axis);
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
                "VeriNeural.reverseAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} ({input_proof})",
                call(input)
            )
        }
        Operation::Concat { axis } => {
            let left = node.inputs[0].index();
            let right = node.inputs[1].index();
            let (outer, left_width, inner) = axis_factors(&graph.nodes()[left].shape, axis);
            let (_, right_width, _) = axis_factors(&graph.nodes()[right].shape, axis);
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
            format!(
                "VeriNeural.concatAxis_nonnegative (outer := {outer}) \
                 (leftWidth := {left_width}) (rightWidth := {right_width}) \
                 (inner := {inner}) {} {} ({left_proof}) ({right_proof})",
                call(left),
                call(right)
            )
        }
        Operation::Take { axis } => {
            let data = node.inputs[0].index();
            let selected = node.inputs[1].index();
            let (outer, width, inner) = axis_factors(&graph.nodes()[data].shape, axis);
            let data_proof = prove_module_node_nonnegative(
                graph,
                module_index,
                data,
                arguments,
                boundary,
                traversal,
                module_proofs,
            )?;
            format!(
                "VeriNeural.takeAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} {} ({data_proof})",
                call(data),
                call(selected)
            )
        }
        Operation::Pointwise(ref spec) => spec
            .expression()
            .nonnegative_lean(&|port| call(node.inputs[port].index()))
            .ok_or(ProofError::UnsupportedGuarantee { node: index })?,
        Operation::ReduceSum { .. } => {
            let input = node.inputs[0].index();
            let dims = graph.nodes()[input].shape.dims();
            let axis = match &node.operation {
                Operation::ReduceSum { axis } => *axis,
                _ => unreachable!(),
            };
            let outer = dims[..axis].iter().product::<usize>();
            let width = dims[axis];
            let inner = dims[axis + 1..].iter().product::<usize>();
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
                "VeriNeural.reduceSumAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} ({input_proof})",
                call(input)
            )
        }
        _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
    };
    traversal.cache.insert(index, proof.clone());
    Ok(proof)
}

/// A claim is about the ideal-Int interpretation of the graph, not the
/// conservative FP32 value domain. Never infer proof from \`Domain::Any\` or
/// from a user assertion: build it from trusted Lean operation lemmas and
/// explicitly declared source assumptions.
fn prove_ideal_nonnegative(
    graph: &NeuralGraph,
    id: NodeId,
    arguments: &str,
    source_hypotheses: &BTreeSet<usize>,
    cache: &mut BTreeMap<usize, String>,
) -> Result<String, ProofError> {
    let index = id.index();
    if let Some(term) = cache.get(&index) {
        return Ok(term.clone());
    }
    let node = graph.node(id)?;
    let call = |id: NodeId| format!("(node_{} {arguments})", id.index());
    let proof = match node.operation {
        Operation::Input if source_hypotheses.contains(&index) => format!("h_{index}"),
        Operation::Relu => format!("VeriNeural.relu_nonnegative {}", call(node.inputs[0])),
        Operation::PositiveMask => {
            format!(
                "VeriNeural.positiveMask_nonnegative {}",
                call(node.inputs[0])
            )
        }
        Operation::SelectPositive => {
            let yes = node.inputs[1];
            let no = node.inputs[2];
            let hy = prove_ideal_nonnegative(graph, yes, arguments, source_hypotheses, cache)?;
            let hn = prove_ideal_nonnegative(graph, no, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.selectPositive_nonnegative {} {} {} ({hy}) ({hn})",
                call(node.inputs[0]),
                call(yes),
                call(no)
            )
        }
        Operation::Reshape(_) => {
            let input = node.inputs[0];
            let h = prove_ideal_nonnegative(graph, input, arguments, source_hypotheses, cache)?;
            format!("VeriNeural.reshape_nonnegative {} ({h})", call(input))
        }
        Operation::Permute { ref axes } => {
            let input = node.inputs[0];
            let h = prove_ideal_nonnegative(graph, input, arguments, source_hypotheses, cache)?;
            if axes.iter().copied().eq(0..axes.len()) {
                format!("VeriNeural.reshape_nonnegative {} ({h})", call(input))
            } else {
                let dims = graph.node(input)?.shape.dims();
                format!(
                    "VeriNeural.transpose2D_nonnegative (rows := {}) (cols := {}) {} ({h})",
                    dims[0],
                    dims[1],
                    call(input)
                )
            }
        }
        Operation::ReverseAxis { axis } => {
            let input = node.inputs[0];
            let (outer, width, inner) = axis_factors(&graph.node(input)?.shape, axis);
            let h = prove_ideal_nonnegative(graph, input, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.reverseAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} ({h})",
                call(input)
            )
        }
        Operation::Concat { axis } => {
            let left = node.inputs[0];
            let right = node.inputs[1];
            let (outer, left_width, inner) = axis_factors(&graph.node(left)?.shape, axis);
            let (_, right_width, _) = axis_factors(&graph.node(right)?.shape, axis);
            let hl = prove_ideal_nonnegative(graph, left, arguments, source_hypotheses, cache)?;
            let hr = prove_ideal_nonnegative(graph, right, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.concatAxis_nonnegative (outer := {outer}) \
                 (leftWidth := {left_width}) (rightWidth := {right_width}) \
                 (inner := {inner}) {} {} ({hl}) ({hr})",
                call(left),
                call(right)
            )
        }
        Operation::Take { axis } => {
            let data = node.inputs[0];
            let selected = node.inputs[1];
            let (outer, width, inner) = axis_factors(&graph.node(data)?.shape, axis);
            let h = prove_ideal_nonnegative(graph, data, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.takeAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} {} ({h})",
                call(data),
                call(selected)
            )
        }
        Operation::Pointwise(ref spec) => spec
            .expression()
            .nonnegative_lean(&|port| call(node.inputs[port]))
            .ok_or(ProofError::UnsupportedGuarantee { node: index })?,
        Operation::ReduceSum { axis } => {
            let input = node.inputs[0];
            let dims = graph.node(input)?.shape.dims();
            let outer = dims[..axis].iter().product::<usize>();
            let width = dims[axis];
            let inner = dims[axis + 1..].iter().product::<usize>();
            let proof = prove_ideal_nonnegative(graph, input, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.reduceSumAxis_nonnegative (outer := {outer}) \
                 (width := {width}) (inner := {inner}) {} ({proof})",
                call(input)
            )
        }
        Operation::Add | Operation::Mul | Operation::MatMul | Operation::BiasAdd => {
            let a = node.inputs[0];
            let b = node.inputs[1];
            let ha = prove_ideal_nonnegative(graph, a, arguments, source_hypotheses, cache)?;
            let hb = prove_ideal_nonnegative(graph, b, arguments, source_hypotheses, cache)?;
            let lemma = match node.operation {
                Operation::Add => "VeriNeural.add_nonnegative".to_owned(),
                Operation::Mul => "VeriNeural.mul_nonnegative".to_owned(),
                Operation::MatMul => {
                    let left = graph.node(a)?.shape.dims();
                    let right = graph.node(b)?.shape.dims();
                    format!(
                        "VeriNeural.matmul_nonnegative (m := {}) (k := {}) (n := {})",
                        left[0], left[1], right[1]
                    )
                }
                Operation::BiasAdd => {
                    let dims = node.shape.dims();
                    format!(
                        "VeriNeural.biasAdd_nonnegative (m := {}) (n := {})",
                        dims[0], dims[1]
                    )
                }
                _ => unreachable!("handled binary primitive"),
            };
            format!("{lemma} {} {} ({ha}) ({hb})", call(a), call(b))
        }
        _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
    };
    cache.insert(index, proof.clone());
    Ok(proof)
}

/// Shared, borrowed proof signature for independently claimed ideal-Int
/// postconditions. Keeps source hypotheses explicit without bloating callers.
struct IdealNonnegativeClaimContext<'a> {
    params: &'a str,
    arguments: &'a str,
    assumptions: &'a [ProofAssumption],
    source_hypotheses: &'a BTreeSet<usize>,
}

/// Resolve module-local references to concrete graph handles, without
/// accepting globally captured nodes in an ostensibly closed contract.
fn resolve_module_value(
    graph: &NeuralGraph,
    value: &ValueRef,
    inputs: &[NodeId],
    output: NodeId,
) -> Result<NodeId, ProofError> {
    let id = match value {
        ValueRef::Input(index) => inputs.get(*index).copied(),
        ValueRef::Output => Some(output),
        ValueRef::Node(_) => None,
    }
    .ok_or(ProofError::UnsupportedGuarantee {
        node: output.index(),
    })?;
    graph.node(id)?;
    Ok(id)
}

fn emit_module_predicate(
    graph: &NeuralGraph,
    source: &mut String,
    theorems: &mut Vec<String>,
    name: String,
    predicate: &Predicate,
    module: (&[NodeId], NodeId),
    context: &IdealNonnegativeClaimContext<'_>,
) -> Result<(), ProofError> {
    let (inputs, output) = module;
    match predicate {
        Predicate::Nonnegative(value) => {
            let target = resolve_module_value(graph, value, inputs, output)?;
            emit_ideal_nonnegative_claim(graph, source, theorems, name, target, context)
        }
        Predicate::Equal(a, b) | Predicate::PointwiseLe(a, b) => {
            let left = resolve_module_value(graph, a, inputs, output)?;
            let right = resolve_module_value(graph, b, inputs, output)?;
            let left_node = graph.node(left)?;
            let right_node = graph.node(right)?;
            if left_node.shape != right_node.shape
                || left_node.dtype != right_node.dtype
                || (matches!(predicate, Predicate::PointwiseLe(_, _))
                    && left_node.dtype != DType::F32)
            {
                return Err(ProofError::UnsupportedGuarantee {
                    node: output.index(),
                });
            }
            let call = |id: NodeId| format!("(node_{} {})", id.index(), context.arguments);
            let (proposition, proof) = match predicate {
                Predicate::Equal(_, _) => (
                    format!("{} = {}", call(left), call(right)),
                    "  rfl\n".to_owned(),
                ),
                Predicate::PointwiseLe(_, _) => (
                    format!("VeriNeural.PointwiseLe {} {}", call(left), call(right)),
                    format!("  exact VeriNeural.pointwise_le_relu {}\n", call(left)),
                ),
                Predicate::Nonnegative(_) => unreachable!(),
            };
            source.push_str(&format!(
                "theorem {name} {} : {proposition} := by\n{proof}\n",
                context.params
            ));
            theorems.push(name);
            Ok(())
        }
    }
}

fn emit_ideal_nonnegative_claim(
    graph: &NeuralGraph,
    source: &mut String,
    names: &mut Vec<String>,
    theorem: String,
    output: NodeId,
    context: &IdealNonnegativeClaimContext<'_>,
) -> Result<(), ProofError> {
    let params = context.params;
    let arguments = context.arguments;
    let assumptions = context.assumptions;
    let proof = prove_ideal_nonnegative(
        graph,
        output,
        arguments,
        context.source_hypotheses,
        &mut BTreeMap::new(),
    )?;
    // Keep the full positional hypothesis interface for every theorem, but
    // mark unused binders to avoid Lean's unused-variable diagnostics.
    let hypothesis_params = assumptions
        .iter()
        .map(|a| {
            let idx = a.input_node;
            let wanted = format!("h_{idx}");
            let identifier = if proof
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|token| token == wanted)
            {
                format!("h_{idx}")
            } else {
                format!("_h_{idx}")
            };
            format!("({identifier} : VeriNeural.Nonnegative x_{idx})")
        })
        .collect::<Vec<_>>()
        .join(" ");
    source.push_str(&format!(
        "theorem {theorem} {params} {hypothesis_params} :\n  \
         VeriNeural.Nonnegative (node_{} {arguments}) := by\n  \
         exact {proof}\n\n",
        output.index()
    ));
    names.push(theorem);
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

fn lean_value_type(dtype: DType, count: usize) -> String {
    match dtype {
        DType::F32 => format!("VeriNeural.Tensor {count}"),
        DType::Index => format!("VeriNeural.IndexTensor {count}"),
    }
}

fn axis_factors(shape: &Shape, axis: usize) -> (usize, usize, usize) {
    let dims = shape.dims();
    (
        dims[..axis].iter().product(),
        dims[axis],
        dims[axis + 1..].iter().product(),
    )
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

#[cfg(test)]
mod axiom_audit_tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["node_2_nonnegative".into(), "node_3_shape".into()]
    }

    #[test]
    fn propext_is_allowed_but_visible_in_dependencies() {
        let stdout = "'VeriNeural.Generated.node_2_nonnegative' depends on axioms: [propext]\n\
                      'VeriNeural.Generated.node_3_shape' does not depend on any axioms\n";
        let deps = parse_axiom_report(stdout, &names()).unwrap();
        assert_eq!(deps["node_2_nonnegative"], vec!["propext"]);
        assert!(deps["node_3_shape"].is_empty());
    }

    #[test]
    fn forbidden_axioms_and_forged_reports_fail_closed() {
        for axiom in [
            "sorryAx",
            "user.hidden",
            "Classical.choice",
            "propext, sorryAx",
        ] {
            let stdout = format!(
                "'VeriNeural.Generated.node_2_nonnegative' depends on axioms: [{axiom}]\n\
                 'VeriNeural.Generated.node_3_shape' does not depend on any axioms\n"
            );
            assert!(parse_axiom_report(&stdout, &names()).is_err());
        }
        for stdout in [
            "'VeriNeural.Generated.unexpected' does not depend on any axioms\n\
             'VeriNeural.Generated.node_3_shape' does not depend on any axioms\n",
            "'VeriNeural.Generated.node_2_nonnegative' does not depend on any axioms\n",
            "'VeriNeural.Generated.node_2_nonnegative' does not depend on any axioms\n\
             'VeriNeural.Generated.node_3_shape' does not depend on any axioms\n\
             unexpected extra line\n",
        ] {
            assert!(parse_axiom_report(stdout, &names()).is_err());
        }
    }
}
