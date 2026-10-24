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
    ContractClaim, DType, Domain, FixedScan, ModuleInstanceRef, NeuralError, NeuralGraph, NodeId,
    Operation, Predicate, Shape, ValueRef,
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
    ModuleDefinitionMismatch {
        representative: usize,
        other: usize,
    },
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
            Self::ModuleDefinitionMismatch {
                representative,
                other,
            } => write!(
                f,
                "module instances {representative} and {other} do not share one reusable semantic definition"
            ),
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
    pub(crate) fn generated(
        source: String,
        theorems: Vec<String>,
        assumptions: Vec<ProofAssumption>,
    ) -> Self {
        Self {
            source,
            theorems,
            assumptions,
        }
    }

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
                Operation::Pointwise(ref spec) if !spec.expression().has_ideal_int_semantics() => {
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
                | Operation::SliceAxis { .. }
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
                Operation::SliceAxis { axis, start, len } => {
                    let input = graph.node(node.inputs[0])?;
                    let (outer, width, inner) = axis_factors(&input.shape, *axis);
                    format!(
                        "VeriNeural.sliceAxis (outer := {outer}) (width := {width})                          (inner := {inner}) (start := {start}) (len := {len}) {}",
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
                } else if matches!(node.operation, Operation::SliceAxis { .. }) {
                    let (axis, start, len) = match &node.operation {
                        Operation::SliceAxis { axis, start, len } => (*axis, *start, *len),
                        _ => unreachable!(),
                    };
                    let (outer, width, inner) = axis_factors(&predecessor.shape, axis);
                    format!(
                        "VeriNeural.shapeSize {} = {outer} * {width} * {inner} ∧                          {start} + {len} ≤ {width}",
                        shape_expr(&predecessor.shape)
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
                    Operation::SliceAxis { axis, start, len } => {
                        let input = node.inputs[0].index();
                        let (outer, width, inner) = axis_factors(&graph.nodes()[input].shape, axis);
                        format!(
                            "VeriNeural.sliceAxis_nonnegative (outer := {outer})                              (width := {width}) (inner := {inner}) (start := {start})                              (len := {len}) {} ({})",
                            call(input),
                            proof_call(input)
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

    /// Generate one reusable ideal-Int definition theorem for repeated
    /// single-output module instances, plus a separate application theorem for
    /// each concrete instance. The definition proof is shared; every instance
    /// must independently discharge the required input nonnegativity.
    ///
    /// First profile: F32 output with an explicit Nonnegative claim and the
    /// ideal-Int operations supported by the normal proof backend. Instances
    /// must agree on name, semantic version, contract and normalized body.
    pub fn generate_reusable_module_nonnegative(
        &self,
        graph: &NeuralGraph,
        module_indices: &[usize],
        name: &str,
    ) -> Result<LeanProgram, ProofError> {
        let valid_name = !name.is_empty()
            && name.as_bytes()[0].is_ascii_alphabetic()
            && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_');
        if !valid_name {
            return Err(ProofError::InvalidGoalName);
        }
        if module_indices.is_empty() {
            return Err(ProofError::InvalidOutputs);
        }
        let mut distinct = BTreeSet::new();
        for &index in module_indices {
            if !distinct.insert(index) {
                return Err(ProofError::InvalidOutputs);
            }
        }

        let representative_index = module_indices[0];
        let representative = graph
            .modules()
            .get(representative_index)
            .ok_or(ProofError::InvalidOutputs)?;
        let representative_output = graph.node(representative.output)?;
        if representative_output.dtype != DType::F32
            || !representative
                .contract
                .claims
                .contains(&ContractClaim::Nonnegative)
        {
            return Err(ProofError::UnsupportedGuarantee {
                node: representative.output.index(),
            });
        }

        let representative_definition =
            normalize_module_nonnegative_definition(graph, representative_index)?;
        for &index in &module_indices[1..] {
            let module = graph
                .modules()
                .get(index)
                .ok_or(ProofError::InvalidOutputs)?;
            let output = graph.node(module.output)?;
            if module.name != representative.name
                || module.semantic_version != representative.semantic_version
                || module.contract != representative.contract
                || output.dtype != DType::F32
                || !module.contract.claims.contains(&ContractClaim::Nonnegative)
                || normalize_module_nonnegative_definition(graph, index)?
                    != representative_definition
            {
                return Err(ProofError::ModuleDefinitionMismatch {
                    representative: representative_index,
                    other: index,
                });
            }
        }

        let outputs = module_indices
            .iter()
            .map(|&index| graph.modules()[index].output)
            .collect::<Vec<_>>();
        let mut program = self.generate_outputs(graph, &outputs)?;
        let namespace_end = "end VeriNeural.Generated\n";
        if !program.source.ends_with(namespace_end) {
            return Err(ProofError::LeanRejected {
                details: "generated Lean source has no namespace terminator".into(),
            });
        }
        program
            .source
            .truncate(program.source.len() - namespace_end.len());

        let definition_name = format!("module_definition_{name}");
        let theorem_name = format!("{definition_name}_nonnegative");
        let expression_tokens = representative_definition
            .expression
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .collect::<BTreeSet<_>>();
        let mut definition_port_params = Vec::new();
        let mut theorem_port_params = Vec::new();
        for (port, &input) in representative.inputs.iter().enumerate() {
            let node = graph.node(input)?;
            let value_type = lean_value_type(node.dtype, node.shape.element_count());
            let binder = if expression_tokens.contains(format!("p_{port}").as_str()) {
                format!("p_{port}")
            } else {
                format!("_p_{port}")
            };
            definition_port_params.push(format!("({binder} : {value_type})"));
            theorem_port_params.push(format!("(p_{port} : {value_type})"));
        }
        let definition_port_params = definition_port_params.join(" ");
        let theorem_port_params = theorem_port_params.join(" ");
        let proof_tokens = representative_definition
            .proof_term
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .collect::<BTreeSet<_>>();
        let required_params = representative_definition
            .required_ports
            .iter()
            .map(|&port| {
                let name = format!("h_port_{port}");
                let binder = if proof_tokens.contains(name.as_str()) {
                    name
                } else {
                    format!("_h_port_{port}")
                };
                format!("({binder} : VeriNeural.Nonnegative p_{port})")
            })
            .collect::<Vec<_>>()
            .join(" ");
        program.source.push_str(&format!(
            "-- Reusable module definition; instance compatibility is proved separately.\n             def {definition_name} {definition_port_params} : VeriNeural.Tensor {} :=\n  {}\n\n             theorem {theorem_name} {theorem_port_params} {required_params} :\n               VeriNeural.Nonnegative ({definition_name} {}) := by\n               exact {}\n\n",
            representative_output.shape.element_count(),
            representative_definition.expression,
            (0..representative.inputs.len())
                .map(|port| format!("p_{port}"))
                .collect::<Vec<_>>()
                .join(" "),
            representative_definition.proof_term,
        ));
        program.theorems.push(theorem_name.clone());

        let mut used = BTreeSet::new();
        for &output in &outputs {
            used.extend(collect_used_nodes(graph, output)?);
        }
        let sources = used
            .iter()
            .filter(|&&index| matches!(graph.nodes()[index].operation, Operation::Input))
            .copied()
            .collect::<Vec<_>>();
        let params = sources
            .iter()
            .map(|&index| {
                format!(
                    "(x_{index} : {})",
                    lean_value_type(
                        graph.nodes()[index].dtype,
                        graph.nodes()[index].shape.element_count()
                    )
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let arguments = sources
            .iter()
            .map(|&index| format!("x_{index}"))
            .collect::<Vec<_>>()
            .join(" ");
        let source_hypotheses = program
            .assumptions()
            .iter()
            .map(|assumption| assumption.input_node)
            .collect::<BTreeSet<_>>();

        for &module_index in module_indices {
            let module = &graph.modules()[module_index];
            let port_values = module
                .inputs
                .iter()
                .map(|input| format!("(node_{} {arguments})", input.index()))
                .collect::<Vec<_>>();
            let mut input_proofs = Vec::new();
            for &port in &representative_definition.required_ports {
                let input = module.inputs[port];
                input_proofs.push(prove_ideal_nonnegative(
                    graph,
                    input,
                    &arguments,
                    &source_hypotheses,
                    &mut BTreeMap::new(),
                )?);
            }
            let proof_text = input_proofs.join(" ");
            let hypothesis_params = program
                .assumptions()
                .iter()
                .map(|assumption| {
                    let wanted = format!("h_{}", assumption.input_node);
                    let used = input_proofs.iter().any(|proof| {
                        proof
                            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                            .any(|token| token == wanted)
                    });
                    let binder = if used {
                        wanted
                    } else {
                        format!("_h_{}", assumption.input_node)
                    };
                    format!(
                        "({binder} : VeriNeural.Nonnegative x_{})",
                        assumption.input_node
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            let instance_name = format!("{definition_name}_instance_{module_index}");
            program.source.push_str(&format!(
                "theorem {instance_name} {params} {hypothesis_params} :\n                   VeriNeural.Nonnegative (node_{} {arguments}) := by\n                   change VeriNeural.Nonnegative ({definition_name} {})\n                   exact {theorem_name} {}{}\n\n",
                module.output.index(),
                port_values.join(" "),
                port_values.join(" "),
                if proof_text.is_empty() {
                    String::new()
                } else {
                    format!(" {proof_text}")
                },
            ));
            program.theorems.push(instance_name);
        }

        program.source.push_str(namespace_end);
        Ok(program)
    }

    /// Reuse one ideal-Int theorem for a named multi-output module definition.
    /// A single right-nested tensor product represents every output; the
    /// theorem proves all output Nonnegative claims together. Each concrete
    /// invocation receives a separate checked application for every port.
    ///
    /// Supported profile: at least two F32 outputs, all claiming Nonnegative.
    /// All required input sign conditions must be declared in the contract.
    pub fn generate_reusable_multi_module_nonnegative(
        &self,
        graph: &NeuralGraph,
        module_indices: &[usize],
        name: &str,
    ) -> Result<LeanProgram, ProofError> {
        if name.is_empty()
            || !name.as_bytes()[0].is_ascii_alphabetic()
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err(ProofError::InvalidGoalName);
        }
        if module_indices.is_empty()
            || module_indices
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                .len()
                != module_indices.len()
        {
            return Err(ProofError::InvalidOutputs);
        }
        let representative_index = module_indices[0];
        let representative = graph
            .multi_modules()
            .get(representative_index)
            .ok_or(ProofError::InvalidOutputs)?;
        let definition = normalize_multi_nonnegative_definition(graph, representative_index)?;
        for &index in &module_indices[1..] {
            let instance = graph
                .multi_modules()
                .get(index)
                .ok_or(ProofError::InvalidOutputs)?;
            if instance.name != representative.name
                || instance.semantic_version != representative.semantic_version
                || instance.contract != representative.contract
                || normalize_multi_nonnegative_definition(graph, index)? != definition
            {
                return Err(ProofError::ModuleDefinitionMismatch {
                    representative: representative_index,
                    other: index,
                });
            }
        }
        let outputs = module_indices
            .iter()
            .flat_map(|&index| graph.multi_modules()[index].outputs.iter().copied())
            .collect::<Vec<_>>();
        let mut program = self.generate_outputs(graph, &outputs)?;
        let end = "end VeriNeural.Generated\n";
        if !program.source.ends_with(end) {
            return Err(ProofError::LeanRejected {
                details: "generated Lean source has no namespace terminator".into(),
            });
        }
        program.source.truncate(program.source.len() - end.len());

        let definition_name = format!("module_definition_{name}");
        let theorem_name = format!("{definition_name}_nonnegative");
        let expr_tokens = definition
            .outputs
            .iter()
            .flat_map(|out| {
                out.expression
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            })
            .collect::<BTreeSet<_>>();
        let proof_tokens = definition
            .outputs
            .iter()
            .flat_map(|out| {
                out.proof_term
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            })
            .collect::<BTreeSet<_>>();
        let mut def_params = Vec::new();
        let mut theorem_params = Vec::new();
        for (port, &input) in representative.inputs.iter().enumerate() {
            let node = graph.node(input)?;
            let ty = lean_value_type(node.dtype, node.shape.element_count());
            let binder = if expr_tokens.contains(format!("p_{port}").as_str()) {
                format!("p_{port}")
            } else {
                format!("_p_{port}")
            };
            def_params.push(format!("({binder} : {ty})"));
            theorem_params.push(format!("(p_{port} : {ty})"));
        }
        let required = definition
            .required_ports
            .iter()
            .map(|&port| {
                let binder = format!("h_port_{port}");
                let binder = if proof_tokens.contains(binder.as_str()) {
                    binder
                } else {
                    format!("_h_port_{port}")
                };
                format!("({binder} : VeriNeural.Nonnegative p_{port})")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let port_args = (0..representative.inputs.len())
            .map(|port| format!("p_{port}"))
            .collect::<Vec<_>>()
            .join(" ");
        let type_expr = representative
            .outputs
            .iter()
            .map(|&id| {
                let node = graph.node(id)?;
                Ok(lean_value_type(node.dtype, node.shape.element_count()))
            })
            .collect::<Result<Vec<_>, ProofError>>()?
            .into_iter()
            .rev()
            .reduce(|tail, head| format!("({head} × {tail})"))
            .ok_or(ProofError::InvalidOutputs)?;
        let value_expr = definition
            .outputs
            .iter()
            .map(|out| out.expression.clone())
            .rev()
            .reduce(|tail, head| format!("({head}, {tail})"))
            .ok_or(ProofError::InvalidOutputs)?;
        let projection = |port: usize| {
            let mut fields = vec!["2"; port];
            if port + 1 != definition.outputs.len() {
                fields.push("1");
            }
            fields.join(".")
        };
        let properties = (0..definition.outputs.len())
            .map(|port| {
                format!(
                    "VeriNeural.Nonnegative (({definition_name} {port_args}).{})",
                    projection(port)
                )
            })
            .collect::<Vec<_>>()
            .join(" ∧ ");
        let property_proof = definition
            .outputs
            .iter()
            .map(|out| out.proof_term.clone())
            .rev()
            .reduce(|tail, head| format!("And.intro ({head}) ({tail})"))
            .ok_or(ProofError::InvalidOutputs)?;
        program.source.push_str(&format!(
            "-- One normalized multi-output definition and all-port property.\n\
             def {definition_name} {} : {type_expr} :=\n  {value_expr}\n\n\
             theorem {theorem_name} {} {required} :\n  {properties} := by\n\
               exact {property_proof}\n\n",
            def_params.join(" "),
            theorem_params.join(" "),
        ));
        program.theorems.push(theorem_name.clone());

        let mut used = BTreeSet::new();
        for &output in &outputs {
            used.extend(collect_used_nodes(graph, output)?);
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
                    "(x_{i} : {})",
                    lean_value_type(
                        graph.nodes()[i].dtype,
                        graph.nodes()[i].shape.element_count()
                    )
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let args = sources
            .iter()
            .map(|&i| format!("x_{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let source_hypotheses = program
            .assumptions()
            .iter()
            .map(|a| a.input_node)
            .collect::<BTreeSet<_>>();
        for &instance_index in module_indices {
            let instance = &graph.multi_modules()[instance_index];
            let input_values = instance
                .inputs
                .iter()
                .map(|id| format!("(node_{} {args})", id.index()))
                .collect::<Vec<_>>()
                .join(" ");
            let premises = definition
                .required_ports
                .iter()
                .map(|&port| {
                    prove_ideal_nonnegative(
                        graph,
                        instance.inputs[port],
                        &args,
                        &source_hypotheses,
                        &mut BTreeMap::new(),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let hypothesis_params = program
                .assumptions()
                .iter()
                .map(|a| {
                    let needed = format!("h_{}", a.input_node);
                    let used = premises.iter().any(|term| {
                        term.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                            .any(|token| token == needed)
                    });
                    let binder = if used {
                        needed
                    } else {
                        format!("_h_{}", a.input_node)
                    };
                    format!("({binder} : VeriNeural.Nonnegative x_{})", a.input_node)
                })
                .collect::<Vec<_>>()
                .join(" ");
            let premise_args = if premises.is_empty() {
                String::new()
            } else {
                format!(" {}", premises.join(" "))
            };
            for (port, &output) in instance.outputs.iter().enumerate() {
                let property_name =
                    format!("{definition_name}_instance_{instance_index}_port_{port}");
                let path = projection(port);
                program.source.push_str(&format!(
                    "theorem {property_name} {params} {hypothesis_params} :\n\
                     VeriNeural.Nonnegative (node_{} {args}) := by\n\
                     change VeriNeural.Nonnegative (({definition_name} {input_values}).{path})\n\
                     exact ({theorem_name} {input_values}{premise_args}).{path}\n\n",
                    output.index(),
                ));
                program.theorems.push(property_name);
            }
        }

        program.source.push_str(end);
        Ok(program)
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

        // Module hierarchy is heterogeneous: one application order contains
        // both single- and multi-output invocations. Global composition reasons
        // about a concrete output port, not merely a module index.
        let ordered = ordered_global_module_keys(graph);
        let eligible = ordered
            .iter()
            .copied()
            .filter(|&(_, key)| {
                global_module_outputs(graph, key)
                    .is_ok_and(|outputs| outputs.iter().any(|id| used.contains(&id.index())))
            })
            .collect::<Vec<_>>();
        let top_level = eligible
            .iter()
            .copied()
            .filter(|&child| {
                !eligible
                    .iter()
                    .copied()
                    .any(|parent| global_module_contains(graph, parent, child))
            })
            .collect::<Vec<_>>();

        let mut producers = BTreeMap::<usize, GlobalModuleOutputKey>::new();
        for (sequence, module) in top_level {
            for (port, output) in global_module_outputs(graph, module)?
                .into_iter()
                .enumerate()
            {
                if !used.contains(&output.index()) {
                    continue;
                }
                let key = GlobalModuleOutputKey {
                    sequence,
                    module,
                    port,
                };
                if producers.insert(output.index(), key).is_some() {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: output.index(),
                    });
                }
            }
        }
        let Some(&final_module) = producers.get(&output.index()) else {
            return Err(ProofError::UnsupportedGuarantee {
                node: output.index(),
            });
        };

        let mut normalized_definitions =
            BTreeMap::<GlobalModuleOutputKey, NormalizedModuleNonnegativeDefinition>::new();
        let mut selected = BTreeSet::<GlobalModuleOutputKey>::new();
        let mut required_assumptions = BTreeSet::<usize>::new();
        let mut pending = vec![final_module];
        while let Some(module_key) = pending.pop() {
            if !selected.insert(module_key) {
                continue;
            }
            let definition = normalize_global_module_output(graph, module_key)?;
            let inputs = global_module_inputs(graph, module_key.module)?;
            for &port in &definition.required_ports {
                let input = inputs[port];
                if assumption_set.contains(&input.index()) {
                    required_assumptions.insert(input.index());
                    continue;
                }
                let Some(&producer) = producers.get(&input.index()) else {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: input.index(),
                    });
                };
                if producer.sequence >= module_key.sequence {
                    return Err(ProofError::UnsupportedGuarantee {
                        node: input.index(),
                    });
                }
                pending.push(producer);
            }
            normalized_definitions.insert(module_key, definition);
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

        // Reuse a definition theorem only for the same module kind, stable
        // name/version/full interface, target output port and normalized proof.
        let mut definition_groups = Vec::<Vec<GlobalModuleOutputKey>>::new();
        for &module_key in &selected {
            let mut matching_group = None;
            for (group_index, group) in definition_groups.iter().enumerate() {
                let representative = group[0];
                if module_key.port == representative.port
                    && global_module_same_interface(
                        graph,
                        module_key.module,
                        representative.module,
                    )?
                {
                    if normalized_definitions[&module_key]
                        != normalized_definitions[&representative]
                    {
                        return Err(ProofError::ModuleDefinitionMismatch {
                            representative: representative.sequence,
                            other: module_key.sequence,
                        });
                    }
                    matching_group = Some(group_index);
                    break;
                }
            }
            let group_index = match matching_group {
                Some(group_index) => group_index,
                None => {
                    definition_groups.push(Vec::new());
                    definition_groups.len() - 1
                }
            };
            definition_groups[group_index].push(module_key);
        }

        for group in &definition_groups {
            let representative_key = group[0];
            let inputs = global_module_inputs(graph, representative_key.module)?;
            let representative_output = graph.node(global_module_output(
                graph,
                representative_key.module,
                representative_key.port,
            )?)?;
            if representative_output.dtype != DType::F32 {
                return Err(ProofError::UnsupportedGuarantee {
                    node: representative_output.id.index(),
                });
            }
            let definition = &normalized_definitions[&representative_key];
            let definition_name = global_definition_name(representative_key);
            let theorem_name = format!("{definition_name}_nonnegative");
            let expression_tokens = definition
                .expression
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .collect::<BTreeSet<_>>();
            let proof_tokens = definition
                .proof_term
                .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .collect::<BTreeSet<_>>();
            let mut definition_port_params = Vec::new();
            let mut theorem_port_params = Vec::new();
            for (port, &input) in inputs.iter().enumerate() {
                let input_node = graph.node(input)?;
                let value_type =
                    lean_value_type(input_node.dtype, input_node.shape.element_count());
                let definition_binder = if expression_tokens.contains(format!("p_{port}").as_str())
                {
                    format!("p_{port}")
                } else {
                    format!("_p_{port}")
                };
                definition_port_params.push(format!("({definition_binder} : {value_type})"));
                theorem_port_params.push(format!("(p_{port} : {value_type})"));
            }
            let required_params = definition
                .required_ports
                .iter()
                .map(|&port| {
                    let binder = format!("h_port_{port}");
                    let binder = if proof_tokens.contains(binder.as_str()) {
                        binder
                    } else {
                        format!("_h_port_{port}")
                    };
                    format!("({binder} : VeriNeural.Nonnegative p_{port})")
                })
                .collect::<Vec<_>>()
                .join(" ");
            program.source.push_str(&format!(
                "-- One normalized ideal-Int theorem shared by every compatible module invocation.\n\
                 def {definition_name} {} : VeriNeural.Tensor {} :=\n  {}\n\n\
                 theorem {theorem_name} {} {required_params} :\n  \
                 VeriNeural.Nonnegative ({definition_name} {}) := by\n  exact {}\n\n",
                definition_port_params.join(" "),
                representative_output.shape.element_count(),
                definition.expression,
                theorem_port_params.join(" "),
                (0..inputs.len())
                    .map(|port| format!("p_{port}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                definition.proof_term,
            ));
            program.theorems.push(theorem_name.clone());

            for &module_key in group {
                let inputs = global_module_inputs(graph, module_key.module)?;
                let output = global_module_output(graph, module_key.module, module_key.port)?;
                let definition = &normalized_definitions[&module_key];
                let port_values = inputs
                    .iter()
                    .map(|input| call(input.index()))
                    .collect::<Vec<_>>();
                let proof_params = definition
                    .required_ports
                    .iter()
                    .map(|&port| {
                        format!(
                            "({} : VeriNeural.Nonnegative {})",
                            global_input_hypothesis_name(module_key, port),
                            call(inputs[port].index())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let theorem_params = if proof_params.is_empty() {
                    params.clone()
                } else {
                    format!("{params} {proof_params}")
                };
                let input_proofs = definition
                    .required_ports
                    .iter()
                    .map(|&port| global_input_hypothesis_name(module_key, port))
                    .collect::<Vec<_>>()
                    .join(" ");
                let instance_name = global_instance_name(module_key);
                program.source.push_str(&format!(
                    "-- Concrete invocation: only its premises are instance-specific.\n\
                     theorem {instance_name} {theorem_params} :\n  \
                     VeriNeural.Nonnegative {} := by\n  \
                     change VeriNeural.Nonnegative ({definition_name} {})\n  \
                     exact {theorem_name} {}{}\n\n",
                    call(output.index()),
                    port_values.join(" "),
                    port_values.join(" "),
                    if input_proofs.is_empty() {
                        String::new()
                    } else {
                        format!(" {input_proofs}")
                    },
                ));
                program.theorems.push(instance_name);
            }
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
        for &module_key in &selected {
            let definition = &normalized_definitions[&module_key];
            let inputs = global_module_inputs(graph, module_key.module)?;
            let output = global_module_output(graph, module_key.module, module_key.port)?;
            let mut args = sources
                .iter()
                .map(|&index| format!("x_{index}"))
                .collect::<Vec<_>>();
            for &port in &definition.required_ports {
                let input = inputs[port].index();
                let Some(input_proof) = proofs.get(&input) else {
                    return Err(ProofError::UnsupportedGuarantee { node: input });
                };
                args.push(input_proof.clone());
            }
            let have_name = global_have_name(module_key);
            let instance_name = global_instance_name(module_key);
            program.source.push_str(&format!(
                "  have {have_name} := {instance_name} {}\n",
                args.join(" ")
            ));
            proofs.insert(output.index(), have_name);
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
                                let derived =
                                    derived_pointwise_le(graph, left, right, &term, used.len() + 1);
                                (
                                    format!(
                                        "VeriNeural.PointwiseLe {} {}",
                                        term(left),
                                        term(right)
                                    ),
                                    format!(
                                        "  exact {}",
                                        derived.unwrap_or_else(|| format!(
                                            "VeriNeural.pointwise_le_relu {}",
                                            term(left)
                                        ))
                                    ),
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

    /// Build an arbitrary-horizon ideal-Int invariant from one reusable
    /// consumer-defined state transition. Unlike FixedScan, this theorem does
    /// not unroll time: Lean performs induction over Nat and reuses the
    /// transition's checked next_state Nonnegative claim at every successor.
    ///
    /// First supported profile: exactly two direct F32 sources named
    /// state/input, two outputs named next_state/output, state-preserving
    /// shape/dtype, and NonNegativeFinite source hypotheses.
    pub fn generate_inductive_state_nonnegative(
        &self,
        graph: &NeuralGraph,
        module_index: usize,
        name: &str,
    ) -> Result<LeanProgram, ProofError> {
        if name.is_empty()
            || !name.as_bytes()[0].is_ascii_alphabetic()
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        {
            return Err(ProofError::InvalidGoalName);
        }
        let module = graph
            .multi_modules()
            .get(module_index)
            .ok_or(ProofError::InvalidScan)?;
        if module.inputs.len() != 2
            || module.outputs.len() != 2
            || module.contract.inputs.len() != 2
            || module.contract.outputs.len() != 2
            || module.contract.inputs[0].name != "state"
            || module.contract.inputs[1].name != "input"
            || module.contract.outputs[0].name != "next_state"
            || module.contract.outputs[1].name != "output"
        {
            return Err(ProofError::InvalidScan);
        }

        let state = module.inputs[0];
        let input = module.inputs[1];
        let next_state = module.outputs[0];
        let state_node = graph.node(state)?;
        let input_node = graph.node(input)?;
        let next_node = graph.node(next_state)?;
        if state == input
            || !matches!(state_node.operation, Operation::Input)
            || !matches!(input_node.operation, Operation::Input)
            || state_node.dtype != DType::F32
            || input_node.dtype != DType::F32
            || state_node.domain != Domain::NonNegativeFinite
            || input_node.domain != Domain::NonNegativeFinite
            || state_node.shape != next_node.shape
            || state_node.dtype != next_node.dtype
        {
            return Err(ProofError::InvalidScan);
        }
        let claim = module.contract.outputs[0]
            .claims
            .iter()
            .position(|claim| *claim == ContractClaim::Nonnegative)
            .ok_or(ProofError::InvalidScan)?;
        let claim_name = format!("multi_{module_index}_port_0_claim_{claim}");

        let mut program = self.generate_outputs(graph, &module.outputs)?;
        let mut used = BTreeSet::new();
        for &target in &module.outputs {
            used.extend(collect_used_nodes(graph, target)?);
        }
        let sources = used
            .iter()
            .filter(|&&index| matches!(graph.nodes()[index].operation, Operation::Input))
            .copied()
            .collect::<Vec<_>>();
        if sources.len() != 2
            || !sources.contains(&state.index())
            || !sources.contains(&input.index())
        {
            return Err(ProofError::InvalidScan);
        }
        let assumption_nodes = program
            .assumptions()
            .iter()
            .map(|assumption| assumption.input_node)
            .collect::<BTreeSet<_>>();
        if assumption_nodes != BTreeSet::from([state.index(), input.index()]) {
            return Err(ProofError::InvalidScan);
        }

        let state_count = state_node.shape.element_count();
        let input_count = input_node.shape.element_count();
        let step_args = |state_value: &str, input_value: &str| {
            sources
                .iter()
                .map(|source| {
                    if *source == state.index() {
                        state_value.to_owned()
                    } else {
                        input_value.to_owned()
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let hypothesis_args = program
            .assumptions()
            .iter()
            .map(|assumption| {
                if assumption.input_node == state.index() {
                    "ih".to_owned()
                } else {
                    "(h_input t)".to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        let definition = format!("scan_{name}_state");
        let theorem = format!("scan_{name}_inductive_nonnegative");
        let end = "end VeriNeural.Generated\n";
        if !program.source.ends_with(end) {
            return Err(ProofError::LeanRejected {
                details: "generated Lean source has no namespace terminator".into(),
            });
        }
        program.source.truncate(program.source.len() - end.len());

        let successor_args = step_args(&format!("({definition} initial inputAt t)"), "(inputAt t)");
        program.source.push_str(&format!(
            "def {definition}\n                 (initial : VeriNeural.Tensor {state_count})\n                 (inputAt : Nat → VeriNeural.Tensor {input_count}) :\n                 Nat → VeriNeural.Tensor {state_count}\n               | 0 => initial\n               | Nat.succ t => node_{} {successor_args}\n\n",
            next_state.index()
        ));
        program.source.push_str(&format!(
            concat!(
                "theorem {theorem}\n",
                "    (initial : VeriNeural.Tensor {state_count})\n",
                "    (inputAt : Nat → VeriNeural.Tensor {input_count})\n",
                "    (h_initial : VeriNeural.Nonnegative initial)\n",
                "    (h_input : ∀ t, VeriNeural.Nonnegative (inputAt t)) :\n",
                "    ∀ t, VeriNeural.Nonnegative ({definition} initial inputAt t) := by\n",
                "  intro t\n",
                "  induction t with\n",
                "  | zero =>\n",
                "      simpa [{definition}] using h_initial\n",
                "  | succ t ih =>\n",
                "      simp only [{definition}]\n",
                "      exact {claim_name} {successor_args} {hypothesis_args}\n\n",
                "{end}",
            ),
            theorem = theorem,
            state_count = state_count,
            input_count = input_count,
            definition = definition,
            claim_name = claim_name,
            successor_args = successor_args,
            hypothesis_args = hypothesis_args,
            end = end,
        ));
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedModuleNonnegativeDefinition {
    expression: String,
    proof_term: String,
    required_ports: Vec<usize>,
}

fn normalize_module_nonnegative_definition(
    graph: &NeuralGraph,
    module_index: usize,
) -> Result<NormalizedModuleNonnegativeDefinition, ProofError> {
    normalize_module_nonnegative_definition_impl(graph, module_index, true)
}

fn normalize_module_global_nonnegative_definition(
    graph: &NeuralGraph,
    module_index: usize,
) -> Result<NormalizedModuleNonnegativeDefinition, ProofError> {
    normalize_module_nonnegative_definition_impl(graph, module_index, false)
}

fn normalize_module_nonnegative_definition_impl(
    graph: &NeuralGraph,
    module_index: usize,
    require_declared_nonnegative_inputs: bool,
) -> Result<NormalizedModuleNonnegativeDefinition, ProofError> {
    let module = graph
        .modules()
        .get(module_index)
        .ok_or(ProofError::InvalidOutputs)?;
    let boundary = module
        .inputs
        .iter()
        .enumerate()
        .map(|(port, input)| (input.index(), port))
        .collect::<BTreeMap<_, _>>();
    let mut expression_cache = BTreeMap::new();
    let expression = normalized_module_expression(
        graph,
        module.body_start,
        module.body_end,
        module.output.index(),
        &boundary,
        &mut expression_cache,
    )?;
    let mut required_ports = BTreeSet::new();
    let mut proof_cache = BTreeMap::new();
    let proof_term = normalized_module_nonnegative_proof(
        graph,
        module.body_start,
        module.body_end,
        module.output.index(),
        &boundary,
        &mut required_ports,
        &mut proof_cache,
    )?;
    if require_declared_nonnegative_inputs {
        for &port in &required_ports {
            let required = &module.contract.requires[port];
            if required.dtype != DType::F32 || required.domain != Domain::NonNegativeFinite {
                return Err(ProofError::UnsupportedGuarantee {
                    node: module.output.index(),
                });
            }
        }
    }
    // Interface requires remain obligations even when the selected property
    // happens to be provable without them (for example final ReLU).
    for (port, required) in module.contract.requires.iter().enumerate() {
        if required.domain == Domain::NonNegativeFinite {
            if required.dtype != DType::F32 {
                return Err(ProofError::UnsupportedGuarantee {
                    node: module.output.index(),
                });
            }
            required_ports.insert(port);
        }
    }
    Ok(NormalizedModuleNonnegativeDefinition {
        expression,
        proof_term,
        required_ports: required_ports.into_iter().collect(),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedMultiNonnegativeDefinition {
    outputs: Vec<NormalizedModuleNonnegativeDefinition>,
    required_ports: Vec<usize>,
}

fn normalize_multi_nonnegative_definition(
    graph: &NeuralGraph,
    index: usize,
) -> Result<NormalizedMultiNonnegativeDefinition, ProofError> {
    let module = graph
        .multi_modules()
        .get(index)
        .ok_or(ProofError::InvalidOutputs)?;
    if module.outputs.len() < 2 {
        return Err(ProofError::InvalidOutputs);
    }
    let boundary = module
        .inputs
        .iter()
        .enumerate()
        .map(|(port, input)| (input.index(), port))
        .collect::<BTreeMap<_, _>>();
    let mut required = BTreeSet::new();
    let mut outputs = Vec::new();
    for (port, &output) in module.outputs.iter().enumerate() {
        if graph.node(output)?.dtype != DType::F32
            || !module.contract.outputs[port]
                .claims
                .contains(&ContractClaim::Nonnegative)
        {
            return Err(ProofError::UnsupportedGuarantee {
                node: output.index(),
            });
        }
        let expression = normalized_module_expression(
            graph,
            module.body_start,
            module.body_end,
            output.index(),
            &boundary,
            &mut BTreeMap::new(),
        )?;
        let mut needed = BTreeSet::new();
        let proof_term = normalized_module_nonnegative_proof(
            graph,
            module.body_start,
            module.body_end,
            output.index(),
            &boundary,
            &mut needed,
            &mut BTreeMap::new(),
        )?;
        for &port in &needed {
            let contract = &module.contract.inputs[port].port;
            if contract.dtype != DType::F32 || contract.domain != Domain::NonNegativeFinite {
                return Err(ProofError::UnsupportedGuarantee {
                    node: output.index(),
                });
            }
        }
        required.extend(needed.iter().copied());
        outputs.push(NormalizedModuleNonnegativeDefinition {
            expression,
            proof_term,
            required_ports: needed.into_iter().collect(),
        });
    }
    // Even unused declared requires remain obligations of each instance.
    for (port, input) in module.contract.inputs.iter().enumerate() {
        if input.port.domain == Domain::NonNegativeFinite {
            if input.port.dtype != DType::F32 {
                return Err(ProofError::UnsupportedGuarantee {
                    node: module.outputs[0].index(),
                });
            }
            required.insert(port);
        }
    }
    Ok(NormalizedMultiNonnegativeDefinition {
        outputs,
        required_ports: required.into_iter().collect(),
    })
}

fn normalized_module_expression(
    graph: &NeuralGraph,
    body_start: usize,
    body_end: usize,
    index: usize,
    boundary: &BTreeMap<usize, usize>,
    cache: &mut BTreeMap<usize, String>,
) -> Result<String, ProofError> {
    if let Some(expression) = cache.get(&index) {
        return Ok(expression.clone());
    }
    if let Some(&port) = boundary.get(&index) {
        return Ok(format!("p_{port}"));
    }
    if index < body_start || index >= body_end {
        return Err(ProofError::UnsupportedGuarantee { node: index });
    }
    let node = &graph.nodes()[index];
    let expr = |input: NodeId, cache: &mut BTreeMap<usize, String>| {
        normalized_module_expression(graph, body_start, body_end, input.index(), boundary, cache)
    };
    let result = match &node.operation {
        Operation::Add => format!(
            "VeriNeural.add ({}) ({})",
            expr(node.inputs[0], cache)?,
            expr(node.inputs[1], cache)?
        ),
        Operation::Mul => format!(
            "VeriNeural.mul ({}) ({})",
            expr(node.inputs[0], cache)?,
            expr(node.inputs[1], cache)?
        ),
        Operation::MatMul => {
            let left = graph.node(node.inputs[0])?;
            let right = graph.node(node.inputs[1])?;
            format!(
                "VeriNeural.matmul (m := {}) (k := {}) (n := {}) ({}) ({})",
                left.shape.dims()[0],
                left.shape.dims()[1],
                right.shape.dims()[1],
                expr(node.inputs[0], cache)?,
                expr(node.inputs[1], cache)?
            )
        }
        Operation::BiasAdd => format!(
            "VeriNeural.biasAdd (m := {}) (n := {}) ({}) ({})",
            node.shape.dims()[0],
            node.shape.dims()[1],
            expr(node.inputs[0], cache)?,
            expr(node.inputs[1], cache)?
        ),
        Operation::Relu => format!("VeriNeural.relu ({})", expr(node.inputs[0], cache)?),
        Operation::PositiveMask => {
            format!("VeriNeural.positiveMask ({})", expr(node.inputs[0], cache)?)
        }
        Operation::SelectPositive => format!(
            "VeriNeural.selectPositive ({}) ({}) ({})",
            expr(node.inputs[0], cache)?,
            expr(node.inputs[1], cache)?,
            expr(node.inputs[2], cache)?
        ),
        Operation::Reshape(_) => {
            format!("VeriNeural.reshape ({})", expr(node.inputs[0], cache)?)
        }
        Operation::ReduceSum { axis } => {
            let input = graph.node(node.inputs[0])?;
            let (outer, width, inner) = axis_factors(&input.shape, *axis);
            format!(
                "VeriNeural.reduceSumAxis (outer := {outer}) (width := {width}) (inner := {inner}) ({})",
                expr(node.inputs[0], cache)?
            )
        }
        Operation::Permute { axes } => {
            let input = graph.node(node.inputs[0])?;
            if axes.iter().copied().eq(0..axes.len()) {
                format!("VeriNeural.reshape ({})", expr(node.inputs[0], cache)?)
            } else if input.shape.rank() == 2 && axes.as_slice() == [1, 0] {
                format!(
                    "VeriNeural.transpose2D (rows := {}) (cols := {}) ({})",
                    input.shape.dims()[0],
                    input.shape.dims()[1],
                    expr(node.inputs[0], cache)?
                )
            } else {
                return Err(ProofError::UnsupportedOperation {
                    node: index,
                    operation: node.operation.name(),
                });
            }
        }
        Operation::ReverseAxis { axis } => {
            let input = graph.node(node.inputs[0])?;
            let (outer, width, inner) = axis_factors(&input.shape, *axis);
            format!(
                "VeriNeural.reverseAxis (outer := {outer}) (width := {width}) (inner := {inner}) ({})",
                expr(node.inputs[0], cache)?
            )
        }
        Operation::Concat { axis } => {
            let left = graph.node(node.inputs[0])?;
            let right = graph.node(node.inputs[1])?;
            let (outer, left_width, inner) = axis_factors(&left.shape, *axis);
            let (_, right_width, _) = axis_factors(&right.shape, *axis);
            format!(
                "VeriNeural.concatAxis (outer := {outer}) (leftWidth := {left_width}) (rightWidth := {right_width}) (inner := {inner}) ({}) ({})",
                expr(node.inputs[0], cache)?,
                expr(node.inputs[1], cache)?
            )
        }
        Operation::SliceAxis { axis, start, len } => {
            let input = graph.node(node.inputs[0])?;
            let (outer, width, inner) = axis_factors(&input.shape, *axis);
            format!(
                "VeriNeural.sliceAxis (outer := {outer}) (width := {width}) (inner := {inner}) (start := {start}) (len := {len}) ({})",
                expr(node.inputs[0], cache)?
            )
        }
        Operation::ArgMax { axis, .. } => {
            let input = graph.node(node.inputs[0])?;
            let (outer, width, inner) = axis_factors(&input.shape, *axis);
            format!(
                "VeriNeural.argMaxFirstAxis (outer := {outer}) (width := {width}) (inner := {inner}) ({})",
                expr(node.inputs[0], cache)?
            )
        }
        Operation::Take { axis } => {
            let data = graph.node(node.inputs[0])?;
            let (outer, width, inner) = axis_factors(&data.shape, *axis);
            format!(
                "VeriNeural.takeAxis (outer := {outer}) (width := {width}) (inner := {inner}) ({}) ({})",
                expr(node.inputs[0], cache)?,
                expr(node.inputs[1], cache)?
            )
        }
        Operation::Pointwise(spec) if spec.expression().has_ideal_int_semantics() => {
            spec.expression().lean(&|port| {
                normalized_module_expression(
                    graph,
                    body_start,
                    body_end,
                    node.inputs[port].index(),
                    boundary,
                    &mut BTreeMap::new(),
                )
                .unwrap_or_else(|_| "__unsupported".to_owned())
            })
        }
        _ => {
            return Err(ProofError::UnsupportedOperation {
                node: index,
                operation: node.operation.name(),
            });
        }
    };
    if result.contains("__unsupported") {
        return Err(ProofError::UnsupportedGuarantee { node: index });
    }
    cache.insert(index, result.clone());
    Ok(result)
}

fn normalized_module_nonnegative_proof(
    graph: &NeuralGraph,
    body_start: usize,
    body_end: usize,
    index: usize,
    boundary: &BTreeMap<usize, usize>,
    required_ports: &mut BTreeSet<usize>,
    cache: &mut BTreeMap<usize, String>,
) -> Result<String, ProofError> {
    if let Some(proof) = cache.get(&index) {
        return Ok(proof.clone());
    }
    if let Some(&port) = boundary.get(&index) {
        required_ports.insert(port);
        return Ok(format!("h_port_{port}"));
    }
    if index < body_start || index >= body_end {
        return Err(ProofError::UnsupportedGuarantee { node: index });
    }
    let node = &graph.nodes()[index];
    let term = |input: NodeId| {
        normalized_module_expression(
            graph,
            body_start,
            body_end,
            input.index(),
            boundary,
            &mut BTreeMap::new(),
        )
    };
    let prove = |input: NodeId,
                 required_ports: &mut BTreeSet<usize>,
                 cache: &mut BTreeMap<usize, String>| {
        normalized_module_nonnegative_proof(
            graph,
            body_start,
            body_end,
            input.index(),
            boundary,
            required_ports,
            cache,
        )
    };
    let proof = match &node.operation {
        Operation::Add | Operation::Mul | Operation::MatMul | Operation::BiasAdd => {
            let left = node.inputs[0];
            let right = node.inputs[1];
            let hl = prove(left, required_ports, cache)?;
            let hr = prove(right, required_ports, cache)?;
            let lemma = match node.operation {
                Operation::Add => "VeriNeural.add_nonnegative".to_owned(),
                Operation::Mul => "VeriNeural.mul_nonnegative".to_owned(),
                Operation::MatMul => {
                    let l = graph.node(left)?.shape.dims();
                    let r = graph.node(right)?.shape.dims();
                    format!(
                        "VeriNeural.matmul_nonnegative (m := {}) (k := {}) (n := {})",
                        l[0], l[1], r[1]
                    )
                }
                Operation::BiasAdd => format!(
                    "VeriNeural.biasAdd_nonnegative (m := {}) (n := {})",
                    node.shape.dims()[0],
                    node.shape.dims()[1]
                ),
                _ => unreachable!(),
            };
            format!("{lemma} ({}) ({}) ({hl}) ({hr})", term(left)?, term(right)?)
        }
        Operation::Relu => {
            format!("VeriNeural.relu_nonnegative ({})", term(node.inputs[0])?)
        }
        Operation::PositiveMask => {
            format!(
                "VeriNeural.positiveMask_nonnegative ({})",
                term(node.inputs[0])?
            )
        }
        Operation::SelectPositive => {
            let yes = node.inputs[1];
            let no = node.inputs[2];
            let hy = prove(yes, required_ports, cache)?;
            let hn = prove(no, required_ports, cache)?;
            format!(
                "VeriNeural.selectPositive_nonnegative ({}) ({}) ({}) ({hy}) ({hn})",
                term(node.inputs[0])?,
                term(yes)?,
                term(no)?
            )
        }
        Operation::Reshape(_) => {
            let input = node.inputs[0];
            let h = prove(input, required_ports, cache)?;
            format!("VeriNeural.reshape_nonnegative ({}) ({h})", term(input)?)
        }
        Operation::Permute { axes } => {
            let input = node.inputs[0];
            let h = prove(input, required_ports, cache)?;
            let input_node = graph.node(input)?;
            if axes.iter().copied().eq(0..axes.len()) {
                format!("VeriNeural.reshape_nonnegative ({}) ({h})", term(input)?)
            } else if input_node.shape.rank() == 2 && axes.as_slice() == [1, 0] {
                format!(
                    "VeriNeural.transpose2D_nonnegative (rows := {}) (cols := {}) ({}) ({h})",
                    input_node.shape.dims()[0],
                    input_node.shape.dims()[1],
                    term(input)?
                )
            } else {
                return Err(ProofError::UnsupportedGuarantee { node: index });
            }
        }
        Operation::ReverseAxis { axis } => {
            let input = node.inputs[0];
            let (outer, width, inner) = axis_factors(&graph.node(input)?.shape, *axis);
            let h = prove(input, required_ports, cache)?;
            format!(
                "VeriNeural.reverseAxis_nonnegative (outer := {outer}) (width := {width}) (inner := {inner}) ({}) ({h})",
                term(input)?
            )
        }
        Operation::Concat { axis } => {
            let left = node.inputs[0];
            let right = node.inputs[1];
            let (outer, left_width, inner) = axis_factors(&graph.node(left)?.shape, *axis);
            let (_, right_width, _) = axis_factors(&graph.node(right)?.shape, *axis);
            let hl = prove(left, required_ports, cache)?;
            let hr = prove(right, required_ports, cache)?;
            format!(
                "VeriNeural.concatAxis_nonnegative (outer := {outer}) (leftWidth := {left_width}) (rightWidth := {right_width}) (inner := {inner}) ({}) ({}) ({hl}) ({hr})",
                term(left)?,
                term(right)?
            )
        }
        Operation::SliceAxis { axis, start, len } => {
            let input = node.inputs[0];
            let (outer, width, inner) = axis_factors(&graph.node(input)?.shape, *axis);
            let h = prove(input, required_ports, cache)?;
            format!(
                "VeriNeural.sliceAxis_nonnegative (outer := {outer}) (width := {width}) (inner := {inner}) (start := {start}) (len := {len}) ({}) ({h})",
                term(input)?
            )
        }
        Operation::ReduceSum { axis } => {
            let input = node.inputs[0];
            let (outer, width, inner) = axis_factors(&graph.node(input)?.shape, *axis);
            let h = prove(input, required_ports, cache)?;
            format!(
                "VeriNeural.reduceSumAxis_nonnegative (outer := {outer}) (width := {width}) (inner := {inner}) ({}) ({h})",
                term(input)?
            )
        }
        Operation::Take { axis } => {
            let data = node.inputs[0];
            let selected = node.inputs[1];
            let (outer, width, inner) = axis_factors(&graph.node(data)?.shape, *axis);
            let h = prove(data, required_ports, cache)?;
            format!(
                "VeriNeural.takeAxis_nonnegative (outer := {outer}) (width := {width}) (inner := {inner}) ({}) ({}) ({h})",
                term(data)?,
                term(selected)?
            )
        }
        Operation::Pointwise(spec) => spec
            .expression()
            .nonnegative_lean(&|port| {
                normalized_module_expression(
                    graph,
                    body_start,
                    body_end,
                    node.inputs[port].index(),
                    boundary,
                    &mut BTreeMap::new(),
                )
                .unwrap_or_else(|_| "__unsupported".to_owned())
            })
            .filter(|proof| !proof.contains("__unsupported"))
            .ok_or(ProofError::UnsupportedGuarantee { node: index })?,
        _ => return Err(ProofError::UnsupportedGuarantee { node: index }),
    };
    cache.insert(index, proof.clone());
    Ok(proof)
}

// Constructively prove a narrow, architecture-neutral relational fragment:
// nonnegative branches built from ReLU/Add/Mul, and transitive lower bounds
// across arbitrary reachable Add/ReLU chains. Never assume a source's
// runtime Domain is also a Lean theorem hypothesis.
fn derived_nonnegative(
    graph: &NeuralGraph,
    node: NodeId,
    term: &impl Fn(NodeId) -> String,
    fuel: usize,
) -> Option<String> {
    if fuel == 0 {
        return None;
    }
    let item = graph.node(node).ok()?;
    match &item.operation {
        Operation::Relu => Some(format!(
            "(VeriNeural.relu_nonnegative {})",
            term(item.inputs[0])
        )),
        Operation::Add | Operation::Mul => {
            let left = item.inputs[0];
            let right = item.inputs[1];
            let hl = derived_nonnegative(graph, left, term, fuel - 1)?;
            let hr = derived_nonnegative(graph, right, term, fuel - 1)?;
            let lemma = if matches!(item.operation, Operation::Add) {
                "VeriNeural.add_nonnegative"
            } else {
                "VeriNeural.mul_nonnegative"
            };
            Some(format!(
                "({lemma} {} {} {hl} {hr})",
                term(left),
                term(right)
            ))
        }
        _ => None,
    }
}

fn derived_pointwise_le(
    graph: &NeuralGraph,
    lower: NodeId,
    upper: NodeId,
    term: &impl Fn(NodeId) -> String,
    fuel: usize,
) -> Option<String> {
    if fuel == 0 {
        return None;
    }
    if lower == upper {
        return Some(format!("(VeriNeural.pointwise_le_refl {})", term(lower)));
    }
    let item = graph.node(upper).ok()?;
    match &item.operation {
        Operation::Relu => {
            let inner = item.inputs[0];
            // Direct bound permits a nonnegative sibling of arbitrary
            // supported composition depth, not only a naked ReLU.
            let source = graph.node(inner).ok()?;
            if matches!(source.operation, Operation::Add) {
                let left = source.inputs[0];
                let right = source.inputs[1];
                if lower == left {
                    if let Some(hb) = derived_nonnegative(graph, right, term, fuel - 1) {
                        return Some(format!(
                            "(VeriNeural.pointwise_le_relu_add_of_nonnegative {} {} {hb})",
                            term(left),
                            term(right)
                        ));
                    }
                }
            }
            let hnext = format!("(VeriNeural.pointwise_le_relu {})", term(inner));
            if lower == inner {
                return Some(hnext);
            }
            let previous = derived_pointwise_le(graph, lower, inner, term, fuel - 1)?;
            Some(format!(
                "(VeriNeural.pointwise_le_trans {} {} {} {previous} {hnext})",
                term(lower),
                term(inner),
                term(upper)
            ))
        }
        Operation::Add => {
            let left = item.inputs[0];
            let right = item.inputs[1];
            // Either operand may provide the lower bound if the opposite
            // branch is independently proven nonnegative.
            for (base, other, lemma) in [
                (left, right, "VeriNeural.pointwise_le_add_of_nonnegative"),
                (
                    right,
                    left,
                    "VeriNeural.pointwise_le_add_of_nonnegative_left",
                ),
            ] {
                let Some(nonnegative) = derived_nonnegative(graph, other, term, fuel - 1) else {
                    continue;
                };
                let step = format!("({lemma} {} {} {nonnegative})", term(left), term(right));
                if lower == base {
                    return Some(step);
                }
                if let Some(previous) = derived_pointwise_le(graph, lower, base, term, fuel - 1) {
                    return Some(format!(
                        "(VeriNeural.pointwise_le_trans {} {} {} {previous} {step})",
                        term(lower),
                        term(base),
                        term(upper)
                    ));
                }
            }
            None
        }
        _ => None,
    }
}

fn collect_used_nodes(graph: &NeuralGraph, output: NodeId) -> Result<BTreeSet<usize>, ProofError> {
    Ok(graph.dependency_scope(&[output])?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GlobalModuleKey {
    Single(usize),
    Multi(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct GlobalModuleOutputKey {
    sequence: usize,
    module: GlobalModuleKey,
    port: usize,
}

fn ordered_global_module_keys(graph: &NeuralGraph) -> Vec<(usize, GlobalModuleKey)> {
    let mut single = 0usize;
    let mut multi = 0usize;
    graph
        .module_instances()
        .enumerate()
        .map(|(sequence, instance)| {
            let key = match instance {
                ModuleInstanceRef::Single(_) => {
                    let key = GlobalModuleKey::Single(single);
                    single += 1;
                    key
                }
                ModuleInstanceRef::Multi(_) => {
                    let key = GlobalModuleKey::Multi(multi);
                    multi += 1;
                    key
                }
            };
            (sequence, key)
        })
        .collect()
}

fn global_module_ref(
    graph: &NeuralGraph,
    key: GlobalModuleKey,
) -> Result<ModuleInstanceRef<'_>, ProofError> {
    match key {
        GlobalModuleKey::Single(index) => graph
            .modules()
            .get(index)
            .map(ModuleInstanceRef::Single)
            .ok_or(ProofError::InvalidOutputs),
        GlobalModuleKey::Multi(index) => graph
            .multi_modules()
            .get(index)
            .map(ModuleInstanceRef::Multi)
            .ok_or(ProofError::InvalidOutputs),
    }
}

fn global_module_inputs(
    graph: &NeuralGraph,
    key: GlobalModuleKey,
) -> Result<&[NodeId], ProofError> {
    Ok(global_module_ref(graph, key)?.inputs())
}

fn global_module_outputs(
    graph: &NeuralGraph,
    key: GlobalModuleKey,
) -> Result<Vec<NodeId>, ProofError> {
    Ok(global_module_ref(graph, key)?.outputs())
}

fn global_module_output(
    graph: &NeuralGraph,
    key: GlobalModuleKey,
    port: usize,
) -> Result<NodeId, ProofError> {
    global_module_outputs(graph, key)?
        .get(port)
        .copied()
        .ok_or(ProofError::InvalidOutputs)
}

fn global_module_contains(
    graph: &NeuralGraph,
    parent: (usize, GlobalModuleKey),
    child: (usize, GlobalModuleKey),
) -> bool {
    if child.0 >= parent.0 {
        return false;
    }
    let Ok(parent) = global_module_ref(graph, parent.1) else {
        return false;
    };
    let Ok(child) = global_module_ref(graph, child.1) else {
        return false;
    };
    let parent = parent.body_range();
    let child = child.body_range();
    child.start >= parent.start && child.end <= parent.end
}

fn global_module_same_interface(
    graph: &NeuralGraph,
    left: GlobalModuleKey,
    right: GlobalModuleKey,
) -> Result<bool, ProofError> {
    Ok(match (left, right) {
        (GlobalModuleKey::Single(left), GlobalModuleKey::Single(right)) => {
            let left = graph
                .modules()
                .get(left)
                .ok_or(ProofError::InvalidOutputs)?;
            let right = graph
                .modules()
                .get(right)
                .ok_or(ProofError::InvalidOutputs)?;
            left.name == right.name
                && left.semantic_version == right.semantic_version
                && left.contract == right.contract
        }
        (GlobalModuleKey::Multi(left), GlobalModuleKey::Multi(right)) => {
            let left = graph
                .multi_modules()
                .get(left)
                .ok_or(ProofError::InvalidOutputs)?;
            let right = graph
                .multi_modules()
                .get(right)
                .ok_or(ProofError::InvalidOutputs)?;
            left.name == right.name
                && left.semantic_version == right.semantic_version
                && left.contract == right.contract
        }
        _ => false,
    })
}

fn normalize_multi_global_nonnegative_definition(
    graph: &NeuralGraph,
    index: usize,
    output_port: usize,
) -> Result<NormalizedModuleNonnegativeDefinition, ProofError> {
    let module = graph
        .multi_modules()
        .get(index)
        .ok_or(ProofError::InvalidOutputs)?;
    let output = module
        .outputs
        .get(output_port)
        .copied()
        .ok_or(ProofError::InvalidOutputs)?;
    if graph.node(output)?.dtype != DType::F32 {
        return Err(ProofError::UnsupportedGuarantee {
            node: output.index(),
        });
    }
    let boundary = module
        .inputs
        .iter()
        .enumerate()
        .map(|(port, input)| (input.index(), port))
        .collect::<BTreeMap<_, _>>();
    let expression = normalized_module_expression(
        graph,
        module.body_start,
        module.body_end,
        output.index(),
        &boundary,
        &mut BTreeMap::new(),
    )?;
    let mut required_ports = BTreeSet::new();
    let proof_term = normalized_module_nonnegative_proof(
        graph,
        module.body_start,
        module.body_end,
        output.index(),
        &boundary,
        &mut required_ports,
        &mut BTreeMap::new(),
    )?;
    // A concrete global application still owes every declared nonnegative
    // interface requirement, even if this particular output proof ignores it.
    for (port, input) in module.contract.inputs.iter().enumerate() {
        if input.port.domain == Domain::NonNegativeFinite {
            if input.port.dtype != DType::F32 {
                return Err(ProofError::UnsupportedGuarantee {
                    node: output.index(),
                });
            }
            required_ports.insert(port);
        }
    }
    Ok(NormalizedModuleNonnegativeDefinition {
        expression,
        proof_term,
        required_ports: required_ports.into_iter().collect(),
    })
}

fn normalize_global_module_output(
    graph: &NeuralGraph,
    key: GlobalModuleOutputKey,
) -> Result<NormalizedModuleNonnegativeDefinition, ProofError> {
    match key.module {
        GlobalModuleKey::Single(index) => {
            if key.port != 0 {
                return Err(ProofError::InvalidOutputs);
            }
            normalize_module_global_nonnegative_definition(graph, index)
        }
        GlobalModuleKey::Multi(index) => {
            normalize_multi_global_nonnegative_definition(graph, index, key.port)
        }
    }
}

fn global_definition_name(key: GlobalModuleOutputKey) -> String {
    match key.module {
        GlobalModuleKey::Single(_) => {
            format!("architecture_module_definition_{}", key.sequence)
        }
        GlobalModuleKey::Multi(_) => format!(
            "architecture_module_definition_{}_port_{}",
            key.sequence, key.port
        ),
    }
}

fn global_instance_name(key: GlobalModuleOutputKey) -> String {
    match key.module {
        GlobalModuleKey::Single(_) => format!("architecture_module_instance_{}", key.sequence),
        GlobalModuleKey::Multi(_) => format!(
            "architecture_module_instance_{}_port_{}",
            key.sequence, key.port
        ),
    }
}

fn global_input_hypothesis_name(key: GlobalModuleOutputKey, input_port: usize) -> String {
    match key.module {
        GlobalModuleKey::Single(_) => {
            format!("h_module_{}_input_{input_port}", key.sequence)
        }
        GlobalModuleKey::Multi(_) => format!(
            "h_module_{}_port_{}_input_{input_port}",
            key.sequence, key.port
        ),
    }
}

fn global_have_name(key: GlobalModuleOutputKey) -> String {
    match key.module {
        GlobalModuleKey::Single(_) => format!("arch_module_{}", key.sequence),
        GlobalModuleKey::Multi(_) => format!("arch_module_{}_port_{}", key.sequence, key.port),
    }
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
        Operation::SliceAxis { axis, start, len } => {
            let input = node.inputs[0];
            let (outer, width, inner) = axis_factors(&graph.node(input)?.shape, axis);
            let h = prove_ideal_nonnegative(graph, input, arguments, source_hypotheses, cache)?;
            format!(
                "VeriNeural.sliceAxis_nonnegative (outer := {outer})                  (width := {width}) (inner := {inner}) (start := {start})                  (len := {len}) {} ({h})",
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
