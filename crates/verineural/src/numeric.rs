//! Stage 8 exact-integer FP32 domain plus compositional range obligations for
//! Add/Mul/MatMul/BiasAdd/Relu/Reshape.
//!
//! All integral inputs and intermediate mathematical values are restricted to
//! magnitude <= 2^24, where every integer has a binary32 representation.
//! Lean establishes *ideal Int* bounds, and the integer oracle validates the
//! concrete GPU output bitwise. Correct IEEE RNE hardware/LLVM behavior is
//! STILL TRUSTED, not proved by this engine. Non-integral or negative-zero
//! inputs and unsupported operators fail closed.
//!
//! A second slice models concrete fractional dyadic traces under round-to-nearest,
//! ties-to-even and composes absolute error through Add/Mul/MatMul/BiasAdd/Relu/
//! Reshape. Products use finer per-node dyadic lattices; MatMul explicitly models
//! separate rounded multiply/add steps. A reusable range slice additionally
//! propagates ideal bounds and worst-case epsilon across arbitrary bounded inputs
//! and exports module-boundary summaries. The LLVM/PTX/GPU execution link remains TCB.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{
    LeanProgram, LeanProofEngine, ModuleInvocation, NeuralError, NeuralGraph, NodeId, Operation,
    ProofError, Tensor,
};

const EXACT_F32_INTEGER_LIMIT: i64 = 1 << 24;
const SOURCE_INTEGER_LIMIT: i64 = 1 << 20;
const BASIC_LIBRARY: &str = include_str!("../../../lean/VeriNeural/Basic.lean");
const MEMORY_LIBRARY: &str = include_str!("../../../lean/VeriNeural/Memory.lean");
const NUMERIC_LIBRARY: &str = include_str!("../../../lean/VeriNeural/Numeric.lean");

#[derive(Debug)]
pub enum NumericError {
    Graph(NeuralError),
    Proof(ProofError),
    UnsupportedOperation {
        node: usize,
        operation: &'static str,
    },
    UnsupportedRange {
        node: usize,
        bound: i64,
    },
    UnstableBranch {
        node: usize,
        lower_ticks: i128,
        upper_ticks: i128,
    },
    InvalidInput {
        node: usize,
        reason: &'static str,
    },
    MissingInput {
        node: usize,
    },
    OutputBits {
        index: usize,
        expected: u32,
        actual: u32,
    },
}

impl fmt::Display for NumericError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(err) => write!(f, "invalid neural graph: {err}"),
            Self::Proof(err) => write!(f, "numerical Lean model: {err}"),
            Self::UnsupportedOperation { node, operation } => {
                write!(f, "numeric FP32 subset rejects {operation} at node {node}")
            }
            Self::UnsupportedRange { node, bound } => write!(
                f,
                "node {node} needs integer bound {bound}, outside exact binary32 subset"
            ),
            Self::UnstableBranch {
                node,
                lower_ticks,
                upper_ticks,
            } => write!(
                f,
                "numeric branch at node {node} is unstable: ideal condition interval [{lower_ticks}, {upper_ticks}] crosses zero"
            ),
            Self::InvalidInput { node, reason } => {
                write!(f, "FP32 numeric input {node}: {reason}")
            }
            Self::MissingInput { node } => write!(f, "missing numeric input node {node}"),
            Self::OutputBits {
                index,
                expected,
                actual,
            } => write!(
                f,
                "FP32 oracle mismatch at {index}: expected={expected:08x} GPU={actual:08x}"
            ),
        }
    }
}

impl Error for NumericError {}

impl From<NeuralError> for NumericError {
    fn from(value: NeuralError) -> Self {
        Self::Graph(value)
    }
}

impl From<ProofError> for NumericError {
    fn from(value: ProofError) -> Self {
        Self::Proof(value)
    }
}

/// Conditional exactness under trusted IEEE binary32 round-to-nearest,
/// separate operations, preservation of the emitted arithmetic and CUDA
/// execution. No claim about the hardware is certified by this plan.
#[derive(Clone, Debug)]
pub struct ExactFp32Plan {
    graph: NeuralGraph,
    output: NodeId,
    reachable: Vec<usize>,
    inputs: Vec<NodeId>,
    bounds: BTreeMap<usize, i64>,
    max_input: i64,
}

impl ExactFp32Plan {
    pub const fn input_bound() -> i64 {
        SOURCE_INTEGER_LIMIT
    }

    pub const fn max_exact_integer() -> i64 {
        EXACT_F32_INTEGER_LIMIT
    }

    /// Backwards-compatible conservative profile for purely additive graphs.
    /// Products normally need a tighter explicit source bound.
    pub fn compile(graph: &NeuralGraph, output: NodeId) -> Result<Self, NumericError> {
        Self::compile_with_input_bound(graph, output, SOURCE_INTEGER_LIMIT)
    }

    /// Check all reachable integer intermediates and left-to-right MatMul
    /// partial sums against the exact-binary32 magnitude limit. A tighter
    /// input bound permits safe products without weakening the guarantee.
    pub fn compile_with_input_bound(
        graph: &NeuralGraph,
        output: NodeId,
        max_input: i64,
    ) -> Result<Self, NumericError> {
        graph.node(output)?;
        if !(1..=SOURCE_INTEGER_LIMIT).contains(&max_input) {
            return Err(NumericError::InvalidInput {
                node: output.index(),
                reason: "source bound must be in 1..=2^20",
            });
        }
        if graph.policy() != crate::FP32Policy::strict() {
            return Err(NumericError::InvalidInput {
                node: output.index(),
                reason: "exact-integer subset requires strict FP32 policy",
            });
        }
        let mut used = BTreeSet::new();
        let mut pending = vec![output];
        while let Some(id) = pending.pop() {
            let node = graph.node(id)?;
            if !used.insert(id.index()) {
                continue;
            }
            if !matches!(
                node.operation,
                Operation::Input
                    | Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Reshape(_)
            ) {
                return Err(NumericError::UnsupportedOperation {
                    node: id.index(),
                    operation: node.operation.name(),
                });
            }
            pending.extend(node.inputs.iter().copied());
            // Mirror the Lean proof engine's module-interface dependencies.
            for module in graph.modules() {
                if module.output == id {
                    pending.extend(module.inputs.iter().copied());
                }
            }
        }

        let reachable = used.into_iter().collect::<Vec<_>>();
        let inputs = reachable
            .iter()
            .filter_map(|&i| {
                matches!(graph.nodes()[i].operation, Operation::Input)
                    .then_some(graph.nodes()[i].id)
            })
            .collect::<Vec<_>>();
        let mut bounds = BTreeMap::<usize, i64>::new();
        for &index in &reachable {
            let node = &graph.nodes()[index];
            let parent = |position: usize| bounds[&node.inputs[position].index()];
            let bound = match node.operation {
                Operation::Input => max_input,
                Operation::Add | Operation::BiasAdd => parent(0) + parent(1),
                Operation::Mul => parent(0).checked_mul(parent(1)).unwrap_or(i64::MAX),
                Operation::MatMul => {
                    let left = graph.nodes()[node.inputs[0].index()].shape.dims()[1];
                    parent(0)
                        .checked_mul(parent(1))
                        .and_then(|product| i64::try_from(left).ok()?.checked_mul(product))
                        .unwrap_or(i64::MAX)
                }
                Operation::Relu | Operation::Reshape(_) => parent(0),
                _ => unreachable!("rejected during graph traversal"),
            };
            if bound > EXACT_F32_INTEGER_LIMIT {
                return Err(NumericError::UnsupportedRange { node: index, bound });
            }
            bounds.insert(index, bound);
        }
        Ok(Self {
            graph: graph.clone(),
            output,
            reachable,
            inputs,
            bounds,
            max_input,
        })
    }

    pub fn output_bound(&self) -> i64 {
        self.bounds[&self.output.index()]
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn max_input_bound(&self) -> i64 {
        self.max_input
    }

    /// Generate a Lean theorem for every reachable output node and its real
    /// composed input hypotheses. The result is NOT proof until Lean checks it.
    pub fn lean_source(&self) -> Result<String, NumericError> {
        let base = LeanProofEngine::default().generate(&self.graph, self.output)?;
        let mut source = base.source().to_owned();
        source.push_str("\n\n");
        source.push_str(MEMORY_LIBRARY);
        source.push_str("\n\n");
        source.push_str(NUMERIC_LIBRARY);
        source.push_str("\nnamespace VeriNeural.Generated\n\n");

        let params = self
            .inputs
            .iter()
            .map(|id| {
                let i = id.index();
                format!(
                    "(x_{i} : VeriNeural.Tensor {})",
                    self.graph.nodes()[i].shape.element_count()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        // A node's proof needs hypotheses only for its actual data ancestors,
        // not every reachable input in the enclosing graph. This prevents
        // vacuous assumptions and Lean's unused-variable warnings.
        let mut dependencies: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let mut sources = BTreeSet::new();
            if matches!(node.operation, Operation::Input) {
                sources.insert(index);
            } else {
                for parent in &node.inputs {
                    sources.extend(dependencies[&parent.index()].iter().copied());
                }
            }
            dependencies.insert(index, sources);
        }
        let hypotheses_for = |index: usize| {
            dependencies[&index]
                .iter()
                .map(|source| {
                    format!(
                        "(h_{source} : VeriNeural.TensorWithin {} x_{source})",
                        self.max_input
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let args = self
            .inputs
            .iter()
            .map(|id| format!("x_{}", id.index()))
            .collect::<Vec<_>>()
            .join(" ");
        let conditions_for = |index: usize| {
            dependencies[&index]
                .iter()
                .map(|source| format!("h_{source}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let call = |index: usize| format!("(node_{index} {args})");
        let proof = |index: usize| {
            format!(
                "(node_{index}_integer_range {args} {})",
                conditions_for(index)
            )
        };

        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let bound = self.bounds[&index];
            let derived = match node.operation {
                Operation::Input => format!("h_{index}"),
                Operation::Add => format!(
                    "VeriNeural.tensor_within_add {} {}",
                    proof(node.inputs[0].index()),
                    proof(node.inputs[1].index())
                ),
                Operation::Mul => format!(
                    "VeriNeural.tensor_within_mul {} {}",
                    proof(node.inputs[0].index()),
                    proof(node.inputs[1].index())
                ),
                Operation::MatMul => {
                    let left = &self.graph.nodes()[node.inputs[0].index()];
                    let right = &self.graph.nodes()[node.inputs[1].index()];
                    format!(
                        "VeriNeural.tensor_within_matmul (m := {}) (k := {}) (n := {}) \
                         (a := {}) (b := {}) (by decide : 0 < {}) {} {}",
                        left.shape.dims()[0],
                        left.shape.dims()[1],
                        right.shape.dims()[1],
                        call(node.inputs[0].index()),
                        call(node.inputs[1].index()),
                        right.shape.dims()[1],
                        proof(node.inputs[0].index()),
                        proof(node.inputs[1].index()),
                    )
                }
                Operation::BiasAdd => format!(
                    "VeriNeural.tensor_within_biasAdd (m := {}) (n := {}) (a := {}) (b := {}) \
                     (by decide : 0 < {}) {} {}",
                    node.shape.dims()[0],
                    node.shape.dims()[1],
                    call(node.inputs[0].index()),
                    call(node.inputs[1].index()),
                    node.shape.dims()[1],
                    proof(node.inputs[0].index()),
                    proof(node.inputs[1].index())
                ),
                Operation::Relu => format!(
                    "VeriNeural.tensor_within_relu (by decide : 0 ≤ ({bound} : Int)) {}",
                    proof(node.inputs[0].index())
                ),
                Operation::Reshape(_) => format!(
                    "VeriNeural.tensor_within_reshape {}",
                    proof(node.inputs[0].index())
                ),
                _ => unreachable!("checked in compile"),
            };
            source.push_str(&format!(
                "theorem node_{index}_integer_range {params} {hypotheses} :\n  \
                 VeriNeural.TensorWithin {bound} {} := by\n  \
                 simpa [node_{index}] using ({derived})\n\n",
                call(index),
                hypotheses = hypotheses_for(index)
            ));
        }
        let final_bound = self.output_bound();
        source.push_str(&format!(
            "-- Final output of this *particular* composition is within the\n\
             -- exact-binary32-integer magnitude, assuming input bounds.\n\
             theorem output_integer_range {params} {hypotheses} :\n  \
             VeriNeural.TensorWithin {final_bound} {} := by\n  \
             exact {}\n\nend VeriNeural.Generated\n",
            call(self.output.index()),
            proof(self.output.index()),
            hypotheses = hypotheses_for(self.output.index()),
        ));
        Ok(source)
    }

    fn checked_inputs<'a>(
        &self,
        values: &'a [(NodeId, Tensor<f32>)],
    ) -> Result<BTreeMap<usize, &'a Tensor<f32>>, NumericError> {
        let mut given = BTreeMap::new();
        for (id, tensor) in values {
            let node = self.graph.node(*id)?;
            if !matches!(node.operation, Operation::Input) || !self.inputs.contains(id) {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "input is outside this numeric output slice",
                });
            }
            if tensor.shape() != &node.shape {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "tensor shape differs from the graph input",
                });
            }
            tensor.validate_domain(node.domain)?;
            for value in tensor.values() {
                if !value.is_finite()
                    || value.abs() > self.max_input as f32
                    || value.fract() != 0.0
                    || (value.to_bits() == (-0.0f32).to_bits())
                {
                    return Err(NumericError::InvalidInput {
                        node: id.index(),
                        reason: "requires finite integral f32 values, no -0, |x| <= plan input bound",
                    });
                }
            }
            if given.insert(id.index(), tensor).is_some() {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate graph input",
                });
            }
        }
        for id in &self.inputs {
            if !given.contains_key(&id.index()) {
                return Err(NumericError::MissingInput { node: id.index() });
            }
        }
        Ok(given)
    }

    /// An independent i64 oracle; no f32 arithmetic is used until the final
    /// exact integer-to-binary32 conversion. All intermediates are <= 2^24.
    pub fn integer_reference(
        &self,
        inputs: &[(NodeId, Tensor<f32>)],
    ) -> Result<Tensor<f32>, NumericError> {
        let given = self.checked_inputs(inputs)?;
        let mut values = BTreeMap::<usize, Vec<i64>>::new();
        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let output = match node.operation {
                Operation::Input => given[&index]
                    .values()
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
                Operation::Add => {
                    let left = &values[&node.inputs[0].index()];
                    let right = &values[&node.inputs[1].index()];
                    left.iter().zip(right).map(|(&a, &b)| a + b).collect()
                }
                Operation::Mul => {
                    let left = &values[&node.inputs[0].index()];
                    let right = &values[&node.inputs[1].index()];
                    left.iter().zip(right).map(|(&a, &b)| a * b).collect()
                }
                Operation::MatMul => {
                    let left = &values[&node.inputs[0].index()];
                    let right = &values[&node.inputs[1].index()];
                    let inner = self.graph.nodes()[node.inputs[0].index()].shape.dims()[1];
                    let rows = node.shape.dims()[0];
                    let cols = node.shape.dims()[1];
                    let mut out = vec![0i64; rows * cols];
                    for row in 0..rows {
                        for col in 0..cols {
                            let mut sum = 0i64;
                            for j in 0..inner {
                                sum += left[row * inner + j] * right[j * cols + col];
                            }
                            out[row * cols + col] = sum;
                        }
                    }
                    out
                }
                Operation::BiasAdd => {
                    let left = &values[&node.inputs[0].index()];
                    let bias = &values[&node.inputs[1].index()];
                    left.iter()
                        .enumerate()
                        .map(|(i, &value)| value + bias[i % bias.len()])
                        .collect()
                }
                Operation::Relu => values[&node.inputs[0].index()]
                    .iter()
                    .map(|&value| value.max(0))
                    .collect(),
                Operation::Reshape(_) => values[&node.inputs[0].index()].clone(),
                _ => unreachable!("checked in compile"),
            };
            debug_assert_eq!(output.len(), node.shape.element_count());
            debug_assert!(output.iter().all(|x| x.abs() <= self.bounds[&index]));
            values.insert(index, output);
        }
        let out = values
            .remove(&self.output.index())
            .expect("reachable output");
        let shape = self.graph.node(self.output)?.shape.clone();
        let floats = out.into_iter().map(|i| i as f32).collect();
        Ok(Tensor::new(shape, floats)?)
    }

    pub fn compare_gpu(
        &self,
        inputs: &[(NodeId, Tensor<f32>)],
        gpu: &Tensor<f32>,
    ) -> Result<(), NumericError> {
        let expected = self.integer_reference(inputs)?;
        if expected.shape() != gpu.shape() {
            return Err(NumericError::InvalidInput {
                node: self.output.index(),
                reason: "GPU output shape differs from exact-integer reference",
            });
        }
        for (index, (&target, &actual)) in expected.values().iter().zip(gpu.values()).enumerate() {
            if target.to_bits() != actual.to_bits() {
                return Err(NumericError::OutputBits {
                    index,
                    expected: target.to_bits(),
                    actual: actual.to_bits(),
                });
            }
        }
        Ok(())
    }
}

/// A concrete Stage 8 rounding slice for fractional dyadic inputs.
///
/// Source inputs live on 2^-fraction_bits. Each reachable node carries its own
/// exact dyadic proof lattice: Add/BiasAdd lift operands to the finer lattice,
/// while Mul/MatMul add operand fractional widths. Concrete traces fail closed
/// if that lattice or any i64 oracle value would overflow. No f32 arithmetic is
/// used by the oracle itself.
#[derive(Clone, Debug)]
pub struct RneFp32Plan {
    graph: NeuralGraph,
    output: NodeId,
    reachable: Vec<usize>,
    inputs: Vec<NodeId>,
    fraction_bits: u32,
    scales: BTreeMap<usize, u32>,
}

/// Result of one concrete, independently rounded binary32 trace.
///
/// epsilon_ticks is the Lean-composed conservative absolute-error budget on
/// the output trace lattice. observed_error_ticks is the actual deviation of
/// the integer RNE oracle from exact dyadic arithmetic on the same lattice.
#[derive(Clone, Debug)]
pub struct RneFp32Trace {
    expected: Tensor<f32>,
    ideal_ticks: Vec<i64>,
    rounded_ticks: Vec<i64>,
    epsilon_ticks: i64,
    observed_error_ticks: i64,
    fraction_bits: u32,
    lean_source: String,
}

#[derive(Clone, Copy, Debug)]
struct RneWitness {
    pre_round: i64,
    rounded: i64,
    shift: u32,
}

#[derive(Clone, Copy, Debug)]
struct RneMatMulStep {
    left_element: usize,
    right_element: usize,
    product_ideal: i64,
    product_actual: i64,
    product_epsilon: i64,
    product_rounding: RneWitness,
    sum_ideal: i64,
    sum_actual: i64,
    sum_epsilon: i64,
    sum_rounding: RneWitness,
}

#[derive(Clone, Debug)]
struct RneNodeTrace {
    ideal: Vec<i64>,
    actual: Vec<i64>,
    epsilon: Vec<i64>,
    fraction_bits: u32,
    rounding: Vec<Option<RneWitness>>,
    matmul_steps: Option<Vec<Vec<RneMatMulStep>>>,
}

impl RneFp32Plan {
    /// Compile a concrete dyadic RNE trace profile.
    ///
    /// fraction_bits chooses the exact source-input lattice 2^-q. Source
    /// inputs keep q <= 30 so accepted normal binary32 values fit the i64
    /// decoder. Products may use finer internal lattices, capped at 60 bits;
    /// concrete traces still fail closed on any value/budget overflow.
    pub fn compile(
        graph: &NeuralGraph,
        output: NodeId,
        fraction_bits: u32,
    ) -> Result<Self, NumericError> {
        graph.node(output)?;
        if !(1..=30).contains(&fraction_bits) {
            return Err(NumericError::InvalidInput {
                node: output.index(),
                reason: "RNE dyadic fraction_bits must be in 1..=30",
            });
        }
        if graph.policy() != crate::FP32Policy::strict() {
            return Err(NumericError::InvalidInput {
                node: output.index(),
                reason: "RNE dyadic slice requires strict FP32 policy",
            });
        }

        let mut used = BTreeSet::new();
        let mut pending = vec![output];
        while let Some(id) = pending.pop() {
            let node = graph.node(id)?;
            if !used.insert(id.index()) {
                continue;
            }
            if !matches!(
                node.operation,
                Operation::Input
                    | Operation::Add
                    | Operation::Mul
                    | Operation::MatMul
                    | Operation::BiasAdd
                    | Operation::Relu
                    | Operation::Reshape(_)
            ) {
                return Err(NumericError::UnsupportedOperation {
                    node: id.index(),
                    operation: node.operation.name(),
                });
            }
            pending.extend(node.inputs.iter().copied());
            for module in graph.modules() {
                if module.output == id {
                    pending.extend(module.inputs.iter().copied());
                }
            }
        }

        let reachable = used.into_iter().collect::<Vec<_>>();
        let inputs = reachable
            .iter()
            .filter_map(|&i| {
                matches!(graph.nodes()[i].operation, Operation::Input)
                    .then_some(graph.nodes()[i].id)
            })
            .collect::<Vec<_>>();

        let mut scales = BTreeMap::<usize, u32>::new();
        for &index in &reachable {
            let node = &graph.nodes()[index];
            let parent = |position: usize| scales[&node.inputs[position].index()];
            let scale = match node.operation {
                Operation::Input => fraction_bits,
                Operation::Add | Operation::BiasAdd => parent(0).max(parent(1)),
                Operation::Mul | Operation::MatMul => {
                    parent(0)
                        .checked_add(parent(1))
                        .ok_or(NumericError::InvalidInput {
                            node: index,
                            reason: "RNE proof lattice fractional-width overflow",
                        })?
                }
                Operation::Relu | Operation::Reshape(_) => parent(0),
                _ => unreachable!("rejected during RNE plan compilation"),
            };
            if scale > 60 {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "RNE proof lattice exceeds 60 fractional bits",
                });
            }
            scales.insert(index, scale);
        }

        Ok(Self {
            graph: graph.clone(),
            output,
            reachable,
            inputs,
            fraction_bits,
            scales,
        })
    }

    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
    }

    pub fn output_fraction_bits(&self) -> u32 {
        self.scales[&self.output.index()]
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    fn checked_tick_inputs(
        &self,
        values: &[(NodeId, Tensor<f32>)],
    ) -> Result<BTreeMap<usize, Vec<i64>>, NumericError> {
        let mut given = BTreeMap::new();
        for (id, tensor) in values {
            let node = self.graph.node(*id)?;
            if !matches!(node.operation, Operation::Input) || !self.inputs.contains(id) {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "input is outside this RNE output slice",
                });
            }
            if tensor.shape() != &node.shape {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "tensor shape differs from the graph input",
                });
            }
            tensor.validate_domain(node.domain)?;
            let ticks = tensor
                .values()
                .iter()
                .map(|&value| {
                    f32_to_dyadic_ticks(value, self.fraction_bits).map_err(|reason| {
                        NumericError::InvalidInput {
                            node: id.index(),
                            reason,
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if given.insert(id.index(), ticks).is_some() {
                return Err(NumericError::InvalidInput {
                    node: id.index(),
                    reason: "duplicate graph input",
                });
            }
        }
        for id in &self.inputs {
            if !given.contains_key(&id.index()) {
                return Err(NumericError::MissingInput { node: id.index() });
            }
        }
        Ok(given)
    }

    /// Execute the independent dyadic/RNE oracle and build a concrete Lean
    /// proof trace. MatMul models a separate rounded multiply followed by a
    /// separate rounded add at every left-to-right reduction step.
    pub fn trace(&self, inputs: &[(NodeId, Tensor<f32>)]) -> Result<RneFp32Trace, NumericError> {
        let given = self.checked_tick_inputs(inputs)?;
        let mut nodes = BTreeMap::<usize, RneNodeTrace>::new();

        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let len = node.shape.element_count();
            let trace = match node.operation {
                Operation::Input => {
                    let ticks = given[&index].clone();
                    RneNodeTrace {
                        ideal: ticks.clone(),
                        actual: ticks,
                        epsilon: vec![0; len],
                        fraction_bits: self.scales[&index],
                        rounding: vec![None; len],
                        matmul_steps: None,
                    }
                }
                Operation::Add => {
                    let left = &nodes[&node.inputs[0].index()];
                    let right = &nodes[&node.inputs[1].index()];
                    self.trace_add(index, left, right, None)?
                }
                Operation::Mul => {
                    let left = &nodes[&node.inputs[0].index()];
                    let right = &nodes[&node.inputs[1].index()];
                    self.trace_mul(index, left, right)?
                }
                Operation::MatMul => {
                    let left = &nodes[&node.inputs[0].index()];
                    let right = &nodes[&node.inputs[1].index()];
                    self.trace_matmul(index, left, right)?
                }
                Operation::BiasAdd => {
                    let left = &nodes[&node.inputs[0].index()];
                    let bias = &nodes[&node.inputs[1].index()];
                    self.trace_add(index, left, bias, Some(bias.actual.len()))?
                }
                Operation::Relu => {
                    let parent = &nodes[&node.inputs[0].index()];
                    RneNodeTrace {
                        ideal: parent.ideal.iter().map(|&v| v.max(0)).collect(),
                        actual: parent.actual.iter().map(|&v| v.max(0)).collect(),
                        epsilon: parent.epsilon.clone(),
                        fraction_bits: parent.fraction_bits,
                        rounding: vec![None; len],
                        matmul_steps: None,
                    }
                }
                Operation::Reshape(_) => {
                    let parent = &nodes[&node.inputs[0].index()];
                    RneNodeTrace {
                        ideal: parent.ideal.clone(),
                        actual: parent.actual.clone(),
                        epsilon: parent.epsilon.clone(),
                        fraction_bits: parent.fraction_bits,
                        rounding: vec![None; len],
                        matmul_steps: None,
                    }
                }
                _ => unreachable!("rejected during RNE plan compilation"),
            };
            if trace.actual.len() != len || trace.fraction_bits != self.scales[&index] {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "internal RNE trace shape/scale mismatch",
                });
            }
            for i in 0..len {
                let observed = trace.actual[i]
                    .checked_sub(trace.ideal[i])
                    .and_then(|value| value.checked_abs())
                    .ok_or(NumericError::InvalidInput {
                        node: index,
                        reason: "RNE trace error overflow",
                    })?;
                if observed > trace.epsilon[i] {
                    return Err(NumericError::InvalidInput {
                        node: index,
                        reason: "RNE trace exceeded composed error budget",
                    });
                }
            }
            nodes.insert(index, trace);
        }

        let final_trace = nodes
            .get(&self.output.index())
            .expect("reachable RNE output");
        let epsilon_ticks = final_trace.epsilon.iter().copied().max().unwrap_or(0);
        let observed_error_ticks = final_trace
            .actual
            .iter()
            .zip(&final_trace.ideal)
            .map(|(&actual, &ideal)| {
                actual
                    .checked_sub(ideal)
                    .and_then(|value| value.checked_abs())
                    .expect("validated RNE error")
            })
            .max()
            .unwrap_or(0);
        let floats = final_trace
            .actual
            .iter()
            .map(|&ticks| {
                encode_representable_ticks(ticks, final_trace.fraction_bits)
                    .map(f32::from_bits)
                    .map_err(|reason| NumericError::InvalidInput {
                        node: self.output.index(),
                        reason,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expected = Tensor::new(self.graph.node(self.output)?.shape.clone(), floats)?;
        let lean_source = self.trace_lean_source(&nodes, epsilon_ticks)?;

        Ok(RneFp32Trace {
            expected,
            ideal_ticks: final_trace.ideal.clone(),
            rounded_ticks: final_trace.actual.clone(),
            epsilon_ticks,
            observed_error_ticks,
            fraction_bits: final_trace.fraction_bits,
            lean_source,
        })
    }

    fn trace_add(
        &self,
        node_index: usize,
        left: &RneNodeTrace,
        right: &RneNodeTrace,
        bias_len: Option<usize>,
    ) -> Result<RneNodeTrace, NumericError> {
        let len = left.actual.len();
        if matches!(bias_len, Some(0)) {
            return Err(NumericError::InvalidInput {
                node: node_index,
                reason: "bias width must be positive in RNE trace",
            });
        }
        let fraction_bits = left.fraction_bits.max(right.fraction_bits);
        let left_shift = fraction_bits - left.fraction_bits;
        let right_shift = fraction_bits - right.fraction_bits;
        let mut ideal = Vec::with_capacity(len);
        let mut actual = Vec::with_capacity(len);
        let mut epsilon = Vec::with_capacity(len);
        let mut rounding = Vec::with_capacity(len);

        for i in 0..len {
            let j = bias_len.map_or(i, |width| i % width);
            let left_ideal =
                rescale_i64(left.ideal[i], left_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "ideal dyadic addition rescale overflow",
                })?;
            let right_ideal =
                rescale_i64(right.ideal[j], right_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "ideal dyadic addition rescale overflow",
                })?;
            let left_actual =
                rescale_i64(left.actual[i], left_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "rounded dyadic addition rescale overflow",
                })?;
            let right_actual =
                rescale_i64(right.actual[j], right_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "rounded dyadic addition rescale overflow",
                })?;
            let left_epsilon =
                rescale_i64(left.epsilon[i], left_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "dyadic addition error rescale overflow",
                })?;
            let right_epsilon =
                rescale_i64(right.epsilon[j], right_shift).ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "dyadic addition error rescale overflow",
                })?;
            let ideal_value =
                left_ideal
                    .checked_add(right_ideal)
                    .ok_or(NumericError::InvalidInput {
                        node: node_index,
                        reason: "ideal dyadic addition overflow",
                    })?;
            let pre_round =
                left_actual
                    .checked_add(right_actual)
                    .ok_or(NumericError::InvalidInput {
                        node: node_index,
                        reason: "rounded dyadic addition overflow",
                    })?;
            let (rounded, _bits, shift) = round_ticks_to_binary32(pre_round, fraction_bits)
                .map_err(|reason| NumericError::InvalidInput {
                    node: node_index,
                    reason,
                })?;
            let step = binary32_step_i64(shift).ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "binary32 ulp step overflow",
            })?;
            let budget = left_epsilon
                .checked_add(right_epsilon)
                .and_then(|v| v.checked_add(step))
                .ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "RNE addition error budget overflow",
                })?;
            ideal.push(ideal_value);
            actual.push(rounded);
            epsilon.push(budget);
            rounding.push(Some(RneWitness {
                pre_round,
                rounded,
                shift,
            }));
        }

        Ok(RneNodeTrace {
            ideal,
            actual,
            epsilon,
            fraction_bits,
            rounding,
            matmul_steps: None,
        })
    }

    fn trace_mul_element(
        &self,
        node_index: usize,
        left: &RneNodeTrace,
        right: &RneNodeTrace,
        left_element: usize,
        right_element: usize,
    ) -> Result<(i64, i64, i64, RneWitness), NumericError> {
        let fraction_bits = left.fraction_bits.checked_add(right.fraction_bits).ok_or(
            NumericError::InvalidInput {
                node: node_index,
                reason: "RNE product fractional-width overflow",
            },
        )?;
        let ideal = left.ideal[left_element]
            .checked_mul(right.ideal[right_element])
            .ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "ideal dyadic product overflow",
            })?;
        let pre_round = left.actual[left_element]
            .checked_mul(right.actual[right_element])
            .ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "rounded dyadic product overflow",
            })?;
        let (rounded, _bits, shift) =
            round_ticks_to_binary32(pre_round, fraction_bits).map_err(|reason| {
                NumericError::InvalidInput {
                    node: node_index,
                    reason,
                }
            })?;
        let step = binary32_step_i64(shift).ok_or(NumericError::InvalidInput {
            node: node_index,
            reason: "binary32 product ulp step overflow",
        })?;
        let bound_ideal_left =
            left.ideal[left_element]
                .checked_abs()
                .ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "ideal product bound overflow",
                })?;
        let bound_actual_right =
            right.actual[right_element]
                .checked_abs()
                .ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "actual product bound overflow",
                })?;
        let propagated = left.epsilon[left_element]
            .checked_mul(bound_actual_right)
            .and_then(|v| {
                bound_ideal_left
                    .checked_mul(right.epsilon[right_element])
                    .and_then(|w| v.checked_add(w))
            })
            .ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "RNE product propagated-error overflow",
            })?;
        let budget = propagated
            .checked_add(step)
            .ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "RNE product error budget overflow",
            })?;

        Ok((
            ideal,
            rounded,
            budget,
            RneWitness {
                pre_round,
                rounded,
                shift,
            },
        ))
    }

    fn trace_mul(
        &self,
        node_index: usize,
        left: &RneNodeTrace,
        right: &RneNodeTrace,
    ) -> Result<RneNodeTrace, NumericError> {
        if left.actual.len() != right.actual.len() {
            return Err(NumericError::InvalidInput {
                node: node_index,
                reason: "internal RNE pointwise product shape mismatch",
            });
        }
        let len = left.actual.len();
        let fraction_bits = left.fraction_bits + right.fraction_bits;
        let mut ideal = Vec::with_capacity(len);
        let mut actual = Vec::with_capacity(len);
        let mut epsilon = Vec::with_capacity(len);
        let mut rounding = Vec::with_capacity(len);
        for i in 0..len {
            let (ideal_value, actual_value, error, witness) =
                self.trace_mul_element(node_index, left, right, i, i)?;
            ideal.push(ideal_value);
            actual.push(actual_value);
            epsilon.push(error);
            rounding.push(Some(witness));
        }
        Ok(RneNodeTrace {
            ideal,
            actual,
            epsilon,
            fraction_bits,
            rounding,
            matmul_steps: None,
        })
    }

    fn trace_matmul(
        &self,
        node_index: usize,
        left: &RneNodeTrace,
        right: &RneNodeTrace,
    ) -> Result<RneNodeTrace, NumericError> {
        let node = &self.graph.nodes()[node_index];
        let left_node = &self.graph.nodes()[node.inputs[0].index()];
        let inner = left_node.shape.dims()[1];
        let rows = node.shape.dims()[0];
        let cols = node.shape.dims()[1];
        let fraction_bits = left.fraction_bits + right.fraction_bits;
        let mut ideal = Vec::with_capacity(rows * cols);
        let mut actual = Vec::with_capacity(rows * cols);
        let mut epsilon = Vec::with_capacity(rows * cols);
        let mut all_steps = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                let mut sum_ideal = 0i64;
                let mut sum_actual = 0i64;
                let mut sum_epsilon = 0i64;
                let mut steps = Vec::with_capacity(inner);
                for j in 0..inner {
                    let left_element = row * inner + j;
                    let right_element = j * cols + col;
                    let (product_ideal, product_actual, product_epsilon, product_rounding) = self
                        .trace_mul_element(
                        node_index,
                        left,
                        right,
                        left_element,
                        right_element,
                    )?;
                    let next_ideal =
                        sum_ideal
                            .checked_add(product_ideal)
                            .ok_or(NumericError::InvalidInput {
                                node: node_index,
                                reason: "ideal MatMul reduction overflow",
                            })?;
                    let pre_round = sum_actual.checked_add(product_actual).ok_or(
                        NumericError::InvalidInput {
                            node: node_index,
                            reason: "rounded MatMul reduction overflow",
                        },
                    )?;
                    let (next_actual, _bits, shift) =
                        round_ticks_to_binary32(pre_round, fraction_bits).map_err(|reason| {
                            NumericError::InvalidInput {
                                node: node_index,
                                reason,
                            }
                        })?;
                    let step = binary32_step_i64(shift).ok_or(NumericError::InvalidInput {
                        node: node_index,
                        reason: "binary32 MatMul-add ulp step overflow",
                    })?;
                    let next_epsilon = sum_epsilon
                        .checked_add(product_epsilon)
                        .and_then(|v| v.checked_add(step))
                        .ok_or(NumericError::InvalidInput {
                            node: node_index,
                            reason: "RNE MatMul reduction error budget overflow",
                        })?;
                    let sum_rounding = RneWitness {
                        pre_round,
                        rounded: next_actual,
                        shift,
                    };
                    steps.push(RneMatMulStep {
                        left_element,
                        right_element,
                        product_ideal,
                        product_actual,
                        product_epsilon,
                        product_rounding,
                        sum_ideal: next_ideal,
                        sum_actual: next_actual,
                        sum_epsilon: next_epsilon,
                        sum_rounding,
                    });
                    sum_ideal = next_ideal;
                    sum_actual = next_actual;
                    sum_epsilon = next_epsilon;
                }
                ideal.push(sum_ideal);
                actual.push(sum_actual);
                epsilon.push(sum_epsilon);
                all_steps.push(steps);
            }
        }

        Ok(RneNodeTrace {
            ideal,
            actual,
            epsilon,
            fraction_bits,
            rounding: vec![None; rows * cols],
            matmul_steps: Some(all_steps),
        })
    }

    fn trace_lean_source(
        &self,
        traces: &BTreeMap<usize, RneNodeTrace>,
        output_epsilon: i64,
    ) -> Result<String, NumericError> {
        let base = LeanProofEngine::default().generate(&self.graph, self.output)?;
        let mut source = base.source().to_owned();
        source.push_str("\n\n");
        source.push_str(MEMORY_LIBRARY);
        source.push_str("\n\n");
        source.push_str(NUMERIC_LIBRARY);
        source.push_str("\nnamespace VeriNeural.Generated.RneTrace\n\n");
        source.push_str(&format!(
            "-- Concrete dyadic RNE trace: source inputs use 2^-{} ticks.\n\
             -- Internal products may use finer per-node dyadic lattices.\n\
             -- The hardware/LLVM/PTX link is not proved by this theorem.\n\n",
            self.fraction_bits
        ));

        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let trace = &traces[&index];
            for element in 0..trace.actual.len() {
                let error_name = format!("rne_node_{index}_elem_{element}_error");
                match node.operation {
                    Operation::Input => {
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin 0 {} {} := by\n  \
                             exact VeriNeural.approx_refl _\n\n",
                            trace.ideal[element], trace.actual[element]
                        ));
                    }
                    Operation::Add | Operation::BiasAdd => {
                        let left_index = node.inputs[0].index();
                        let right_index = node.inputs[1].index();
                        let left_trace = &traces[&left_index];
                        let right_trace = &traces[&right_index];
                        let right_element = if matches!(node.operation, Operation::BiasAdd) {
                            element % right_trace.actual.len()
                        } else {
                            element
                        };
                        let left_shift = trace.fraction_bits - left_trace.fraction_bits;
                        let right_shift = trace.fraction_bits - right_trace.fraction_bits;
                        let left_factor = binary32_step_i64(left_shift).expect("validated scale");
                        let right_factor = binary32_step_i64(right_shift).expect("validated scale");
                        let left_ideal =
                            rescale_i64(left_trace.ideal[element], left_shift).expect("validated");
                        let left_actual =
                            rescale_i64(left_trace.actual[element], left_shift).expect("validated");
                        let left_epsilon = rescale_i64(left_trace.epsilon[element], left_shift)
                            .expect("validated");
                        let right_ideal =
                            rescale_i64(right_trace.ideal[right_element], right_shift)
                                .expect("validated");
                        let right_actual =
                            rescale_i64(right_trace.actual[right_element], right_shift)
                                .expect("validated");
                        let right_epsilon =
                            rescale_i64(right_trace.epsilon[right_element], right_shift)
                                .expect("validated");
                        let witness = trace.rounding[element].expect("add rounding witness");
                        let round_name = format!("rne_node_{index}_elem_{element}_round");
                        source.push_str(&format!(
                            "theorem {round_name} :\n  \
                             VeriNeural.Binary32RneTick {} {} {} := by\n  \
                             decide\n\n",
                            witness.pre_round, witness.rounded, witness.shift
                        ));
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin {} {} {} := by\n  \
                             have hleft : VeriNeural.ApproxWithin {left_epsilon} \
                               {left_ideal} {left_actual} := by\n    \
                               simpa using (VeriNeural.approx_scale (scale := {left_factor}) \
                                 (by decide) rne_node_{left_index}_elem_{element}_error)\n  \
                             have hright : VeriNeural.ApproxWithin {right_epsilon} \
                               {right_ideal} {right_actual} := by\n    \
                               simpa using (VeriNeural.approx_scale (scale := {right_factor}) \
                                 (by decide) rne_node_{right_index}_elem_{right_element}_error)\n  \
                             simpa [VeriNeural.binary32Step] using\n    \
                               (VeriNeural.approx_add_round hleft hright\n      \
                                 (VeriNeural.binary32_rne_within_step {round_name}))\n\n",
                            trace.epsilon[element], trace.ideal[element], trace.actual[element]
                        ));
                    }
                    Operation::Mul => {
                        let left_index = node.inputs[0].index();
                        let right_index = node.inputs[1].index();
                        let left_trace = &traces[&left_index];
                        let right_trace = &traces[&right_index];
                        let witness = trace.rounding[element].expect("mul rounding witness");
                        let round_name = format!("rne_node_{index}_elem_{element}_round");
                        let bound_ideal_left =
                            left_trace.ideal[element].checked_abs().expect("validated");
                        let bound_actual_right = right_trace.actual[element]
                            .checked_abs()
                            .expect("validated");
                        source.push_str(&format!(
                            "theorem {round_name} :\n  \
                             VeriNeural.Binary32RneTick {} {} {} := by\n  \
                             decide\n\n",
                            witness.pre_round, witness.rounded, witness.shift
                        ));
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin {} {} {} := by\n  \
                             have hideal : VeriNeural.Within {bound_ideal_left} \
                               {} := by decide\n  \
                             have hactual : VeriNeural.Within {bound_actual_right} \
                               {} := by decide\n  \
                             simpa [VeriNeural.binary32Step] using\n    \
                               (VeriNeural.approx_mul_round\n      \
                                 rne_node_{left_index}_elem_{element}_error\n      \
                                 rne_node_{right_index}_elem_{element}_error\n      \
                                 hideal hactual\n      \
                                 (VeriNeural.binary32_rne_within_step {round_name}))\n\n",
                            trace.epsilon[element],
                            trace.ideal[element],
                            trace.actual[element],
                            left_trace.ideal[element],
                            right_trace.actual[element],
                        ));
                    }
                    Operation::MatMul => {
                        let left_index = node.inputs[0].index();
                        let right_index = node.inputs[1].index();
                        let left_trace = &traces[&left_index];
                        let right_trace = &traces[&right_index];
                        let steps = &trace
                            .matmul_steps
                            .as_ref()
                            .expect("MatMul reduction witnesses")[element];
                        let init_name = format!("rne_node_{index}_elem_{element}_acc_0_error");
                        source.push_str(&format!(
                            "theorem {init_name} :\n  \
                             VeriNeural.ApproxWithin 0 0 0 := by\n  \
                             exact VeriNeural.approx_refl _\n\n"
                        ));
                        let mut previous = init_name;
                        for (step_index, step) in steps.iter().enumerate() {
                            let product_round_name = format!(
                                "rne_node_{index}_elem_{element}_product_{step_index}_round"
                            );
                            let product_error_name = format!(
                                "rne_node_{index}_elem_{element}_product_{step_index}_error"
                            );
                            let sum_round_name =
                                format!("rne_node_{index}_elem_{element}_sum_{step_index}_round");
                            let sum_error_name = format!(
                                "rne_node_{index}_elem_{element}_sum_{}_error",
                                step_index + 1
                            );
                            let bound_ideal_left = left_trace.ideal[step.left_element]
                                .checked_abs()
                                .expect("validated");
                            let bound_actual_right = right_trace.actual[step.right_element]
                                .checked_abs()
                                .expect("validated");
                            source.push_str(&format!(
                                "theorem {product_round_name} :\n  \
                                 VeriNeural.Binary32RneTick {} {} {} := by\n  \
                                 decide\n\n",
                                step.product_rounding.pre_round,
                                step.product_rounding.rounded,
                                step.product_rounding.shift
                            ));
                            source.push_str(&format!(
                                "theorem {product_error_name} :\n  \
                                 VeriNeural.ApproxWithin {} {} {} := by\n  \
                                 have hideal : VeriNeural.Within {bound_ideal_left} \
                                   {} := by decide\n  \
                                 have hactual : VeriNeural.Within {bound_actual_right} \
                                   {} := by decide\n  \
                                 simpa [VeriNeural.binary32Step] using\n    \
                                   (VeriNeural.approx_mul_round\n      \
                                     rne_node_{left_index}_elem_{}_error\n      \
                                     rne_node_{right_index}_elem_{}_error\n      \
                                     hideal hactual\n      \
                                     (VeriNeural.binary32_rne_within_step \
                                       {product_round_name}))\n\n",
                                step.product_epsilon,
                                step.product_ideal,
                                step.product_actual,
                                left_trace.ideal[step.left_element],
                                right_trace.actual[step.right_element],
                                step.left_element,
                                step.right_element,
                            ));
                            source.push_str(&format!(
                                "theorem {sum_round_name} :\n  \
                                 VeriNeural.Binary32RneTick {} {} {} := by\n  \
                                 decide\n\n",
                                step.sum_rounding.pre_round,
                                step.sum_rounding.rounded,
                                step.sum_rounding.shift
                            ));
                            source.push_str(&format!(
                                "theorem {sum_error_name} :\n  \
                                 VeriNeural.ApproxWithin {} {} {} := by\n  \
                                 simpa [VeriNeural.binary32Step] using\n    \
                                   (VeriNeural.approx_add_round\n      \
                                     {previous}\n      \
                                     {product_error_name}\n      \
                                     (VeriNeural.binary32_rne_within_step {sum_round_name}))\n\n",
                                step.sum_epsilon, step.sum_ideal, step.sum_actual
                            ));
                            previous = sum_error_name;
                        }
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin {} {} {} := by\n  \
                             simpa using {previous}\n\n",
                            trace.epsilon[element], trace.ideal[element], trace.actual[element]
                        ));
                    }
                    Operation::Relu => {
                        let parent = node.inputs[0].index();
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin {} {} {} := by\n  \
                             simpa using\n    \
                             (VeriNeural.approx_relu rne_node_{parent}_elem_{element}_error)\n\n",
                            trace.epsilon[element], trace.ideal[element], trace.actual[element]
                        ));
                    }
                    Operation::Reshape(_) => {
                        let parent = node.inputs[0].index();
                        source.push_str(&format!(
                            "theorem {error_name} :\n  \
                             VeriNeural.ApproxWithin {} {} {} := by\n  \
                             simpa using rne_node_{parent}_elem_{element}_error\n\n",
                            trace.epsilon[element], trace.ideal[element], trace.actual[element]
                        ));
                    }
                    _ => unreachable!("rejected during RNE plan compilation"),
                }
            }
        }

        let output = &traces[&self.output.index()];
        for element in 0..output.actual.len() {
            source.push_str(&format!(
                "theorem output_elem_{element}_fp32_error :\n  \
                 VeriNeural.ApproxWithin {output_epsilon} {} {} := by\n  \
                 exact VeriNeural.approx_weaken (by decide) \
                 rne_node_{}_elem_{element}_error\n\n",
                output.ideal[element],
                output.actual[element],
                self.output.index()
            ));
        }
        source.push_str("end VeriNeural.Generated.RneTrace\n");
        Ok(source)
    }
}

impl RneFp32Trace {
    pub fn expected(&self) -> &Tensor<f32> {
        &self.expected
    }

    pub fn ideal_output_ticks(&self) -> &[i64] {
        &self.ideal_ticks
    }

    pub fn rounded_output_ticks(&self) -> &[i64] {
        &self.rounded_ticks
    }

    pub fn epsilon_ticks(&self) -> i64 {
        self.epsilon_ticks
    }

    pub fn observed_error_ticks(&self) -> i64 {
        self.observed_error_ticks
    }

    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon_ticks as f64 / (1u64 << self.fraction_bits) as f64
    }

    pub fn observed_error(&self) -> f64 {
        self.observed_error_ticks as f64 / (1u64 << self.fraction_bits) as f64
    }

    pub fn lean_source(&self) -> &str {
        &self.lean_source
    }

    pub fn compare_gpu(&self, gpu: &Tensor<f32>) -> Result<(), NumericError> {
        if self.expected.shape() != gpu.shape() {
            return Err(NumericError::InvalidInput {
                node: 0,
                reason: "GPU output shape differs from dyadic RNE reference",
            });
        }
        for (index, (&target, &actual)) in
            self.expected.values().iter().zip(gpu.values()).enumerate()
        {
            if target.to_bits() != actual.to_bits() {
                return Err(NumericError::OutputBits {
                    index,
                    expected: target.to_bits(),
                    actual: actual.to_bits(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RneErrorContract {
    ideal_bound_ticks: i64,
    epsilon_ticks: i64,
    actual_bound_ticks: i64,
    fraction_bits: u32,
}

impl RneErrorContract {
    pub fn ideal_bound_ticks(self) -> i64 {
        self.ideal_bound_ticks
    }

    pub fn epsilon_ticks(self) -> i64 {
        self.epsilon_ticks
    }

    pub fn actual_bound_ticks(self) -> i64 {
        self.actual_bound_ticks
    }

    pub fn fraction_bits(self) -> u32 {
        self.fraction_bits
    }

    pub fn epsilon(self) -> f64 {
        self.epsilon_ticks as f64 / (1u64 << self.fraction_bits) as f64
    }
}

/// Conditional ReLU branch in the ideal integer model, not an assertion that
/// CUDA/LLVM/PTX followed the source-level dyadic RNE contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RneThresholdSide {
    Positive,
    NonPositive,
    Indeterminate,
}

/// A pre-ReLU observation and the interval enclosing its ideal counterpart
/// if the per-node ApproxContract hypothesis holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RneThresholdEvidence {
    relu: NodeId,
    input: NodeId,
    fraction_bits: u32,
    observed_ticks: i64,
    epsilon_ticks: i64,
    ideal_lower_ticks: i128,
    ideal_upper_ticks: i128,
    side: RneThresholdSide,
}

impl RneThresholdEvidence {
    pub fn relu(&self) -> NodeId {
        self.relu
    }

    pub fn input(&self) -> NodeId {
        self.input
    }

    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
    }

    pub fn observed_ticks(&self) -> i64 {
        self.observed_ticks
    }

    pub fn epsilon_ticks(&self) -> i64 {
        self.epsilon_ticks
    }

    pub fn ideal_interval_ticks(&self) -> (i128, i128) {
        (self.ideal_lower_ticks, self.ideal_upper_ticks)
    }

    pub fn side(&self) -> RneThresholdSide {
        self.side
    }
}

/// Generic strict-positive condition classification for a reachable numeric
/// node. This is reusable by ReLU and by PositiveMask → SelectPositive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RneConditionEvidence {
    condition: NodeId,
    fraction_bits: u32,
    observed_ticks: i64,
    epsilon_ticks: i64,
    ideal_lower_ticks: i128,
    ideal_upper_ticks: i128,
    side: RneThresholdSide,
}

impl RneConditionEvidence {
    pub fn condition(&self) -> NodeId {
        self.condition
    }

    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
    }

    pub fn observed_ticks(&self) -> i64 {
        self.observed_ticks
    }

    pub fn epsilon_ticks(&self) -> i64 {
        self.epsilon_ticks
    }

    pub fn ideal_interval_ticks(&self) -> (i128, i128) {
        (self.ideal_lower_ticks, self.ideal_upper_ticks)
    }

    pub fn side(&self) -> RneThresholdSide {
        self.side
    }
}

/// Conditional numerical evidence for the canonical
/// PositiveMask(condition) → SelectPositive(mask, yes, no) pattern.
/// An indeterminate condition deliberately has no selected contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RneSelectEvidence {
    select: NodeId,
    condition: NodeId,
    selected: Option<NodeId>,
    condition_evidence: RneConditionEvidence,
    contract: Option<RneErrorContract>,
}

impl RneSelectEvidence {
    pub fn select(&self) -> NodeId {
        self.select
    }

    pub fn condition(&self) -> NodeId {
        self.condition
    }

    pub fn side(&self) -> RneThresholdSide {
        self.condition_evidence.side()
    }

    pub fn condition_evidence(&self) -> RneConditionEvidence {
        self.condition_evidence
    }

    pub fn selected(&self) -> Option<NodeId> {
        self.selected
    }

    pub fn contract(&self) -> Option<RneErrorContract> {
        self.contract
    }
}

/// P1 branch-composition slice. It does not pretend PositiveMask/SelectPositive
/// are ordinary RNE arithmetic nodes. Instead it proves a conditional rule:
/// once the condition interval lies strictly on one side of zero, the select
/// inherits exactly the already-proved error contract of that branch.
#[derive(Clone, Debug)]
pub struct RneSelectPlan {
    select: NodeId,
    condition: NodeId,
    yes: NodeId,
    no: NodeId,
    condition_plan: RneRangePlan,
    yes_plan: RneRangePlan,
    no_plan: RneRangePlan,
}

/// One fixed recurrent step in the numerical contract rollout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RneStateStepContract {
    step: usize,
    input_state: RneErrorContract,
    output_state: RneErrorContract,
}

impl RneStateStepContract {
    pub fn step(&self) -> usize {
        self.step
    }

    pub fn input_state(&self) -> RneErrorContract {
        self.input_state
    }

    pub fn output_state(&self) -> RneErrorContract {
        self.output_state
    }
}

/// Fixed-horizon recurrent numerical composition for one explicit state input.
/// Fresh non-state sources are exact dyadic inputs at every step. The feedback
/// state carries the previous step's epsilon and bound. This deliberately
/// requires an invariant q scale; transitions that change the state lattice
/// fail closed instead of silently rescaling temporal state.
#[derive(Clone, Debug)]
pub struct RneStateRolloutPlan {
    state: NodeId,
    next_state: NodeId,
    initial_state: RneErrorContract,
    steps: Vec<RneStateStepContract>,
    plans: Vec<RneRangePlan>,
}

impl RneStateRolloutPlan {
    pub fn compile_fixed(
        graph: &NeuralGraph,
        state: NodeId,
        next_state: NodeId,
        source_fraction_bits: u32,
        source_bound_ticks: i64,
        horizon: usize,
    ) -> Result<Self, NumericError> {
        let state_node = graph.node(state)?;
        let next_node = graph.node(next_state)?;
        if !matches!(state_node.operation, Operation::Input) {
            return Err(NumericError::InvalidInput {
                node: state.index(),
                reason: "RNE recurrent state must be an Input node",
            });
        }
        if state_node.shape != next_node.shape || state_node.dtype != next_node.dtype {
            return Err(NumericError::InvalidInput {
                node: next_state.index(),
                reason: "RNE recurrent next_state must preserve state shape and dtype",
            });
        }

        let initial_state =
            RneRangePlan::compile(graph, state, source_fraction_bits, source_bound_ticks)?
                .output_contract();
        let mut input_state = initial_state;
        let mut steps = Vec::with_capacity(horizon);
        let mut plans = Vec::with_capacity(horizon);

        for step in 0..horizon {
            let plan = RneRangePlan::compile_with_source_contracts(
                graph,
                next_state,
                source_fraction_bits,
                source_bound_ticks,
                &[(state, input_state)],
            )?;
            let output_state = plan.output_contract();
            if output_state.fraction_bits != source_fraction_bits {
                return Err(NumericError::InvalidInput {
                    node: next_state.index(),
                    reason: "RNE recurrent feedback changes the state q scale",
                });
            }
            steps.push(RneStateStepContract {
                step,
                input_state,
                output_state,
            });
            input_state = output_state;
            plans.push(plan);
        }

        Ok(Self {
            state,
            next_state,
            initial_state,
            steps,
            plans,
        })
    }

    pub fn state(&self) -> NodeId {
        self.state
    }

    pub fn next_state(&self) -> NodeId {
        self.next_state
    }

    pub fn initial_contract(&self) -> RneErrorContract {
        self.initial_state
    }

    pub fn steps(&self) -> &[RneStateStepContract] {
        &self.steps
    }

    pub fn final_contract(&self) -> RneErrorContract {
        self.steps
            .last()
            .map(RneStateStepContract::output_state)
            .unwrap_or(self.initial_state)
    }

    pub fn step_plan(&self, step: usize) -> Option<&RneRangePlan> {
        self.plans.get(step)
    }

    /// Each source is a standalone Lean program for one conditional step.
    /// The Rust rollout guarantees adjacent numeric contracts match exactly;
    /// arbitrary-horizon induction is intentionally not claimed here.
    pub fn lean_sources(&self) -> Result<Vec<String>, NumericError> {
        self.plans.iter().map(RneRangePlan::lean_source).collect()
    }
}

impl RneSelectPlan {
    pub fn compile(
        graph: &NeuralGraph,
        select: NodeId,
        source_fraction_bits: u32,
        source_bound_ticks: i64,
    ) -> Result<Self, NumericError> {
        let select_node = graph.node(select)?;
        if !matches!(select_node.operation, Operation::SelectPositive) {
            return Err(NumericError::InvalidInput {
                node: select.index(),
                reason: "RNE select plan requires a SelectPositive node",
            });
        }
        let mask = select_node.inputs[0];
        let mask_node = graph.node(mask)?;
        if !matches!(mask_node.operation, Operation::PositiveMask) {
            return Err(NumericError::InvalidInput {
                node: mask.index(),
                reason: "RNE select plan currently requires a direct PositiveMask producer",
            });
        }
        let condition = mask_node.inputs[0];
        let yes = select_node.inputs[1];
        let no = select_node.inputs[2];

        Ok(Self {
            select,
            condition,
            yes,
            no,
            condition_plan: RneRangePlan::compile(
                graph,
                condition,
                source_fraction_bits,
                source_bound_ticks,
            )?,
            yes_plan: RneRangePlan::compile(graph, yes, source_fraction_bits, source_bound_ticks)?,
            no_plan: RneRangePlan::compile(graph, no, source_fraction_bits, source_bound_ticks)?,
        })
    }

    pub fn select(&self) -> NodeId {
        self.select
    }

    pub fn condition(&self) -> NodeId {
        self.condition
    }

    pub fn yes(&self) -> NodeId {
        self.yes
    }

    pub fn no(&self) -> NodeId {
        self.no
    }

    pub fn condition_plan(&self) -> &RneRangePlan {
        &self.condition_plan
    }

    pub fn yes_plan(&self) -> &RneRangePlan {
        &self.yes_plan
    }

    pub fn no_plan(&self) -> &RneRangePlan {
        &self.no_plan
    }

    pub fn classify(&self, observed_condition: f32) -> Result<RneSelectEvidence, NumericError> {
        let condition_evidence = self
            .condition_plan
            .classify_condition(self.condition, observed_condition)?;
        let (selected, contract) = match condition_evidence.side() {
            RneThresholdSide::Positive => (Some(self.yes), Some(self.yes_plan.output_contract())),
            RneThresholdSide::NonPositive => (Some(self.no), Some(self.no_plan.output_contract())),
            RneThresholdSide::Indeterminate => (None, None),
        };
        Ok(RneSelectEvidence {
            select: self.select,
            condition: self.condition,
            selected,
            condition_evidence,
            contract,
        })
    }

    pub fn stable_contract(
        &self,
        observed_condition: f32,
    ) -> Result<RneErrorContract, NumericError> {
        let evidence = self.classify(observed_condition)?;
        evidence.contract().ok_or_else(|| {
            let (lower_ticks, upper_ticks) = evidence.condition_evidence().ideal_interval_ticks();
            NumericError::UnstableBranch {
                node: self.select.index(),
                lower_ticks,
                upper_ticks,
            }
        })
    }

    /// Generate the conditional Lean composition theorem for one observed
    /// condition. The theorem assumes the condition ApproxContract and the
    /// selected branch ApproxContract; the opposite branch is intentionally
    /// irrelevant. Indeterminate observations fail closed before generation.
    pub fn lean_program(&self, observed_condition: f32) -> Result<LeanProgram, NumericError> {
        let evidence = self.classify(observed_condition)?;
        let selected = evidence.contract().ok_or_else(|| {
            let (lower_ticks, upper_ticks) = evidence.condition_evidence().ideal_interval_ticks();
            NumericError::UnstableBranch {
                node: self.select.index(),
                lower_ticks,
                upper_ticks,
            }
        })?;
        let condition = self.condition_plan.output_contract();
        let observed_ticks = evidence.condition_evidence().observed_ticks();
        let theorem = match evidence.side() {
            RneThresholdSide::Positive => format!("select_{}_stable_positive", self.select.index()),
            RneThresholdSide::NonPositive => {
                format!("select_{}_stable_nonpositive", self.select.index())
            }
            RneThresholdSide::Indeterminate => unreachable!("handled above"),
        };
        let lemma = match evidence.side() {
            RneThresholdSide::Positive => {
                "VeriNeural.approx_contract_positive_mask_select_positive"
            }
            RneThresholdSide::NonPositive => {
                "VeriNeural.approx_contract_positive_mask_select_nonpositive"
            }
            RneThresholdSide::Indeterminate => unreachable!("handled above"),
        };
        let margin = match evidence.side() {
            RneThresholdSide::Positive => {
                format!("{} < ({observed_ticks} : Int)", condition.epsilon_ticks)
            }
            RneThresholdSide::NonPositive => {
                format!("({observed_ticks} : Int) + {} ≤ 0", condition.epsilon_ticks)
            }
            RneThresholdSide::Indeterminate => unreachable!("handled above"),
        };
        let source = format!(
            "{BASIC_LIBRARY}\n\n{MEMORY_LIBRARY}\n\n{NUMERIC_LIBRARY}\n\nnamespace VeriNeural.Generated\n\n             theorem {theorem}\n                 {{idealCondition selectedIdeal selectedActual otherIdeal otherActual : Int}}\n                 (hcondition : VeriNeural.ApproxContract {} {} idealCondition ({observed_ticks} : Int))\n                 (hselected : VeriNeural.ApproxContract {} {} selectedIdeal selectedActual) :\n                 VeriNeural.ApproxContract {} {}\n                   (if 0 < (if 0 < idealCondition then 1 else 0) then {} else {})\n                   (if 0 < (if 0 < ({observed_ticks} : Int) then 1 else 0) then {} else {}) := by\n               have hmargin : {margin} := by decide\n               exact {} hcondition hmargin hselected\n\nend VeriNeural.Generated\n",
            condition.ideal_bound_ticks,
            condition.epsilon_ticks,
            selected.ideal_bound_ticks,
            selected.epsilon_ticks,
            selected.ideal_bound_ticks,
            selected.epsilon_ticks,
            if matches!(evidence.side(), RneThresholdSide::Positive) {
                "selectedIdeal"
            } else {
                "otherIdeal"
            },
            if matches!(evidence.side(), RneThresholdSide::Positive) {
                "otherIdeal"
            } else {
                "selectedIdeal"
            },
            if matches!(evidence.side(), RneThresholdSide::Positive) {
                "selectedActual"
            } else {
                "otherActual"
            },
            if matches!(evidence.side(), RneThresholdSide::Positive) {
                "otherActual"
            } else {
                "selectedActual"
            },
            lemma,
        );
        Ok(LeanProgram::generated(source, vec![theorem], Vec::new()))
    }
}

#[derive(Clone, Debug)]
pub struct RneModuleErrorContract {
    name: &'static str,
    inputs: Vec<(NodeId, RneErrorContract)>,
    output: NodeId,
    contract: RneErrorContract,
}

impl RneModuleErrorContract {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn inputs(&self) -> &[(NodeId, RneErrorContract)] {
        &self.inputs
    }

    pub fn output(&self) -> NodeId {
        self.output
    }

    pub fn contract(&self) -> RneErrorContract {
        self.contract
    }
}

#[derive(Clone, Debug)]
enum RneRangeRule {
    Input,
    Add {
        left_shift: u32,
        right_shift: u32,
        max_shift: u32,
    },
    Mul {
        max_shift: u32,
    },
    MatMul {
        product: RneErrorContract,
        product_max_shift: u32,
        sums: Vec<(RneErrorContract, u32)>,
    },
    BiasAdd {
        left_shift: u32,
        right_shift: u32,
        max_shift: u32,
    },
    Relu,
    Reshape,
}

#[derive(Clone, Debug)]
struct RneGlobalElement {
    ideal: String,
    actual: String,
    proof: String,
}

#[derive(Clone, Debug)]
struct RneArchitectureStep {
    module_index: usize,
    args: Vec<String>,
    output_proofs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct RneRangePlan {
    graph: NeuralGraph,
    output: NodeId,
    reachable: Vec<usize>,
    inputs: Vec<NodeId>,
    source_fraction_bits: u32,
    source_bound_ticks: i64,
    contracts: BTreeMap<usize, RneErrorContract>,
    rules: BTreeMap<usize, RneRangeRule>,
}

impl RneRangePlan {
    /// Build a reusable worst-case dyadic RNE contract for all values satisfying
    /// the source tick bound. The theorem is conditional on the same
    /// Binary32RneTick relation as concrete traces; LLVM/PTX/GPU conformance
    /// remains outside the formal boundary.
    pub fn compile(
        graph: &NeuralGraph,
        output: NodeId,
        source_fraction_bits: u32,
        source_bound_ticks: i64,
    ) -> Result<Self, NumericError> {
        Self::compile_with_source_contracts(
            graph,
            output,
            source_fraction_bits,
            source_bound_ticks,
            &[],
        )
    }

    /// Compile the same range plan while replacing selected reachable Input
    /// nodes with already-established ApproxContracts. This is the primitive
    /// needed for recurrent error composition: a previous next_state contract
    /// becomes the current state source contract, while all other sources keep
    /// the exact default bound for this step.
    pub fn compile_with_source_contracts(
        graph: &NeuralGraph,
        output: NodeId,
        source_fraction_bits: u32,
        source_bound_ticks: i64,
        source_contracts: &[(NodeId, RneErrorContract)],
    ) -> Result<Self, NumericError> {
        if source_bound_ticks < 0 {
            return Err(NumericError::InvalidInput {
                node: output.index(),
                reason: "RNE range source bound must be nonnegative",
            });
        }
        let base = RneFp32Plan::compile(graph, output, source_fraction_bits)?;
        let mut overrides = BTreeMap::<usize, RneErrorContract>::new();
        for &(source, contract) in source_contracts {
            let node = graph.node(source)?;
            if !matches!(node.operation, Operation::Input) || !base.inputs.contains(&source) {
                return Err(NumericError::InvalidInput {
                    node: source.index(),
                    reason: "RNE source override must name a reachable Input",
                });
            }
            if contract.fraction_bits != source_fraction_bits {
                return Err(NumericError::InvalidInput {
                    node: source.index(),
                    reason: "RNE source override must use the plan source q scale",
                });
            }
            let expected_actual = contract
                .ideal_bound_ticks
                .checked_add(contract.epsilon_ticks)
                .ok_or(NumericError::InvalidInput {
                    node: source.index(),
                    reason: "RNE source override bound overflow",
                })?;
            if contract.ideal_bound_ticks < 0
                || contract.epsilon_ticks < 0
                || contract.actual_bound_ticks != expected_actual
            {
                return Err(NumericError::InvalidInput {
                    node: source.index(),
                    reason: "RNE source override is not a well-formed ApproxContract bound",
                });
            }
            if overrides.insert(source.index(), contract).is_some() {
                return Err(NumericError::InvalidInput {
                    node: source.index(),
                    reason: "duplicate RNE source override",
                });
            }
        }

        let mut contracts = BTreeMap::<usize, RneErrorContract>::new();
        let mut rules = BTreeMap::<usize, RneRangeRule>::new();

        for &index in &base.reachable {
            let node = &graph.nodes()[index];
            let source_contract = || -> Result<RneErrorContract, NumericError> {
                Self::contract(
                    index,
                    source_bound_ticks,
                    0,
                    base.scales[&index],
                    "RNE source contract overflow",
                )
            };
            let trace = match node.operation {
                Operation::Input => {
                    rules.insert(index, RneRangeRule::Input);
                    overrides
                        .get(&index)
                        .copied()
                        .map(Ok)
                        .unwrap_or_else(source_contract)?
                }
                Operation::Add | Operation::BiasAdd => {
                    let left = contracts[&node.inputs[0].index()];
                    let right = contracts[&node.inputs[1].index()];
                    let (contract, left_shift, right_shift, max_shift) =
                        Self::add_contract(index, left, right)?;
                    let rule = if matches!(node.operation, Operation::BiasAdd) {
                        RneRangeRule::BiasAdd {
                            left_shift,
                            right_shift,
                            max_shift,
                        }
                    } else {
                        RneRangeRule::Add {
                            left_shift,
                            right_shift,
                            max_shift,
                        }
                    };
                    rules.insert(index, rule);
                    contract
                }
                Operation::Mul => {
                    let left = contracts[&node.inputs[0].index()];
                    let right = contracts[&node.inputs[1].index()];
                    let (contract, max_shift) = Self::mul_contract(index, left, right)?;
                    rules.insert(index, RneRangeRule::Mul { max_shift });
                    contract
                }
                Operation::MatMul => {
                    let left = contracts[&node.inputs[0].index()];
                    let right = contracts[&node.inputs[1].index()];
                    let (product, product_max_shift) = Self::mul_contract(index, left, right)?;
                    let inner = graph.nodes()[node.inputs[0].index()].shape.dims()[1];
                    let mut accumulator = Self::contract(
                        index,
                        0,
                        0,
                        product.fraction_bits,
                        "RNE MatMul zero contract",
                    )?;
                    let mut sums = Vec::with_capacity(inner);
                    for _ in 0..inner {
                        let (next, left_shift, right_shift, max_shift) =
                            Self::add_contract(index, accumulator, product)?;
                        debug_assert_eq!(left_shift, 0);
                        debug_assert_eq!(right_shift, 0);
                        sums.push((next, max_shift));
                        accumulator = next;
                    }
                    rules.insert(
                        index,
                        RneRangeRule::MatMul {
                            product,
                            product_max_shift,
                            sums,
                        },
                    );
                    accumulator
                }
                Operation::Relu => {
                    let contract = contracts[&node.inputs[0].index()];
                    rules.insert(index, RneRangeRule::Relu);
                    contract
                }
                Operation::Reshape(_) => {
                    let contract = contracts[&node.inputs[0].index()];
                    rules.insert(index, RneRangeRule::Reshape);
                    contract
                }
                _ => unreachable!("RneFp32Plan rejected unsupported range operation"),
            };
            if trace.fraction_bits != base.scales[&index] {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "RNE range contract scale disagrees with trace plan",
                });
            }
            contracts.insert(index, trace);
        }

        Ok(Self {
            graph: graph.clone(),
            output,
            reachable: base.reachable,
            inputs: base.inputs,
            source_fraction_bits,
            source_bound_ticks,
            contracts,
            rules,
        })
    }

    fn contract(
        node: usize,
        ideal_bound_ticks: i64,
        epsilon_ticks: i64,
        fraction_bits: u32,
        reason: &'static str,
    ) -> Result<RneErrorContract, NumericError> {
        if ideal_bound_ticks < 0 || epsilon_ticks < 0 {
            return Err(NumericError::InvalidInput { node, reason });
        }
        let actual_bound_ticks = ideal_bound_ticks
            .checked_add(epsilon_ticks)
            .ok_or(NumericError::InvalidInput { node, reason })?;
        Ok(RneErrorContract {
            ideal_bound_ticks,
            epsilon_ticks,
            actual_bound_ticks,
            fraction_bits,
        })
    }

    fn scale_contract(
        node: usize,
        contract: RneErrorContract,
        target_fraction_bits: u32,
    ) -> Result<(RneErrorContract, u32), NumericError> {
        let shift = target_fraction_bits
            .checked_sub(contract.fraction_bits)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range contract cannot scale to a coarser lattice",
            })?;
        let factor = binary32_step_i64(shift).ok_or(NumericError::InvalidInput {
            node,
            reason: "RNE range scale factor overflow",
        })?;
        let ideal_bound_ticks =
            contract
                .ideal_bound_ticks
                .checked_mul(factor)
                .ok_or(NumericError::InvalidInput {
                    node,
                    reason: "RNE range ideal-bound rescale overflow",
                })?;
        let epsilon_ticks =
            contract
                .epsilon_ticks
                .checked_mul(factor)
                .ok_or(NumericError::InvalidInput {
                    node,
                    reason: "RNE range epsilon rescale overflow",
                })?;
        Ok((
            Self::contract(
                node,
                ideal_bound_ticks,
                epsilon_ticks,
                target_fraction_bits,
                "RNE range scaled contract overflow",
            )?,
            shift,
        ))
    }

    fn round_budget(node: usize, pre_round_bound: i64) -> Result<(u32, i64), NumericError> {
        if pre_round_bound < 0 {
            return Err(NumericError::InvalidInput {
                node,
                reason: "RNE range pre-round bound must be nonnegative",
            });
        }
        let magnitude = pre_round_bound as u64;
        let bit_len = if magnitude == 0 {
            0
        } else {
            64 - magnitude.leading_zeros()
        };
        let max_shift = bit_len.saturating_sub(24);
        let step = binary32_step_i64(max_shift).ok_or(NumericError::InvalidInput {
            node,
            reason: "RNE range ULP budget overflow",
        })?;
        let limit =
            8_388_608i128
                .checked_mul(1i128.checked_shl(max_shift + 1).ok_or(
                    NumericError::InvalidInput {
                        node,
                        reason: "RNE range shift limit overflow",
                    },
                )?)
                .ok_or(NumericError::InvalidInput {
                    node,
                    reason: "RNE range shift limit overflow",
                })?;
        if i128::from(pre_round_bound) >= limit {
            return Err(NumericError::InvalidInput {
                node,
                reason: "RNE range shift bound is inconsistent",
            });
        }
        Ok((max_shift, step))
    }

    fn add_contract(
        node: usize,
        left: RneErrorContract,
        right: RneErrorContract,
    ) -> Result<(RneErrorContract, u32, u32, u32), NumericError> {
        let fraction_bits = left.fraction_bits.max(right.fraction_bits);
        let (left, left_shift) = Self::scale_contract(node, left, fraction_bits)?;
        let (right, right_shift) = Self::scale_contract(node, right, fraction_bits)?;
        let ideal_bound_ticks = left
            .ideal_bound_ticks
            .checked_add(right.ideal_bound_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range addition ideal bound overflow",
            })?;
        let pre_round_bound = left
            .actual_bound_ticks
            .checked_add(right.actual_bound_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range addition actual bound overflow",
            })?;
        let (max_shift, round_budget) = Self::round_budget(node, pre_round_bound)?;
        let epsilon_ticks = left
            .epsilon_ticks
            .checked_add(right.epsilon_ticks)
            .and_then(|value| value.checked_add(round_budget))
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range addition epsilon overflow",
            })?;
        Ok((
            Self::contract(
                node,
                ideal_bound_ticks,
                epsilon_ticks,
                fraction_bits,
                "RNE range addition contract overflow",
            )?,
            left_shift,
            right_shift,
            max_shift,
        ))
    }

    fn mul_contract(
        node: usize,
        left: RneErrorContract,
        right: RneErrorContract,
    ) -> Result<(RneErrorContract, u32), NumericError> {
        let fraction_bits = left.fraction_bits.checked_add(right.fraction_bits).ok_or(
            NumericError::InvalidInput {
                node,
                reason: "RNE range product fractional-width overflow",
            },
        )?;
        if fraction_bits > 60 {
            return Err(NumericError::InvalidInput {
                node,
                reason: "RNE range product lattice exceeds 60 fractional bits",
            });
        }
        let ideal_bound_ticks = left
            .ideal_bound_ticks
            .checked_mul(right.ideal_bound_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range product ideal bound overflow",
            })?;
        let pre_round_bound = left
            .actual_bound_ticks
            .checked_mul(right.actual_bound_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range product actual bound overflow",
            })?;
        let (max_shift, round_budget) = Self::round_budget(node, pre_round_bound)?;
        let propagated_left = left
            .epsilon_ticks
            .checked_mul(right.actual_bound_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range product propagated error overflow",
            })?;
        let propagated_right = left
            .ideal_bound_ticks
            .checked_mul(right.epsilon_ticks)
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range product propagated error overflow",
            })?;
        let epsilon_ticks = propagated_left
            .checked_add(propagated_right)
            .and_then(|value| value.checked_add(round_budget))
            .ok_or(NumericError::InvalidInput {
                node,
                reason: "RNE range product epsilon overflow",
            })?;
        Ok((
            Self::contract(
                node,
                ideal_bound_ticks,
                epsilon_ticks,
                fraction_bits,
                "RNE range product contract overflow",
            )?,
            max_shift,
        ))
    }

    pub fn source_fraction_bits(&self) -> u32 {
        self.source_fraction_bits
    }

    pub fn source_bound_ticks(&self) -> i64 {
        self.source_bound_ticks
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    pub fn output_contract(&self) -> RneErrorContract {
        self.contracts[&self.output.index()]
    }

    pub fn node_contract(&self, node: NodeId) -> Result<RneErrorContract, NumericError> {
        self.graph.node(node)?;
        self.contracts
            .get(&node.index())
            .copied()
            .ok_or(NumericError::InvalidInput {
                node: node.index(),
                reason: "node is outside this RNE range output slice",
            })
    }

    /// Classify any reachable numeric condition against strict zero.
    /// This is a conditional source-level result under the node ApproxContract.
    pub fn classify_condition(
        &self,
        condition: NodeId,
        observed: f32,
    ) -> Result<RneConditionEvidence, NumericError> {
        let contract = self.node_contract(condition)?;
        let observed_ticks =
            f32_to_dyadic_ticks(observed, contract.fraction_bits).map_err(|reason| {
                NumericError::InvalidInput {
                    node: condition.index(),
                    reason,
                }
            })?;
        if observed_ticks.unsigned_abs() > contract.actual_bound_ticks as u64 {
            return Err(NumericError::InvalidInput {
                node: condition.index(),
                reason: "condition observation exceeds the node actual range bound",
            });
        }
        // i128 avoids signed overflow even near the maximum supported i64.
        let epsilon = i128::from(contract.epsilon_ticks);
        let actual = i128::from(observed_ticks);
        let ideal_lower_ticks = actual - epsilon;
        let ideal_upper_ticks = actual + epsilon;
        let side = if ideal_lower_ticks > 0 {
            RneThresholdSide::Positive
        } else if ideal_upper_ticks <= 0 {
            RneThresholdSide::NonPositive
        } else {
            RneThresholdSide::Indeterminate
        };
        Ok(RneConditionEvidence {
            condition,
            fraction_bits: contract.fraction_bits,
            observed_ticks,
            epsilon_ticks: contract.epsilon_ticks,
            ideal_lower_ticks,
            ideal_upper_ticks,
            side,
        })
    }

    /// Backwards-compatible ReLU-specific view over generic condition evidence.
    pub fn classify_relu_input(
        &self,
        relu: NodeId,
        observed: f32,
    ) -> Result<RneThresholdEvidence, NumericError> {
        let node = self.graph.node(relu)?;
        if !matches!(node.operation, Operation::Relu) || !self.contracts.contains_key(&relu.index())
        {
            return Err(NumericError::InvalidInput {
                node: relu.index(),
                reason: "threshold target must be a reachable ReLU",
            });
        }
        let input = node.inputs[0];
        let evidence = self.classify_condition(input, observed)?;
        let (ideal_lower_ticks, ideal_upper_ticks) = evidence.ideal_interval_ticks();
        Ok(RneThresholdEvidence {
            relu,
            input,
            fraction_bits: evidence.fraction_bits(),
            observed_ticks: evidence.observed_ticks(),
            epsilon_ticks: evidence.epsilon_ticks(),
            ideal_lower_ticks,
            ideal_upper_ticks,
            side: evidence.side(),
        })
    }

    pub fn module_contracts(&self) -> Vec<RneModuleErrorContract> {
        self.graph
            .modules()
            .iter()
            .filter_map(|module| {
                let contract = self.contracts.get(&module.output.index()).copied()?;
                let inputs = module
                    .inputs
                    .iter()
                    .filter_map(|&node| {
                        self.contracts
                            .get(&node.index())
                            .copied()
                            .map(|contract| (node, contract))
                    })
                    .collect::<Vec<_>>();
                (inputs.len() == module.inputs.len()).then_some(RneModuleErrorContract {
                    name: module.name,
                    inputs,
                    output: module.output,
                    contract,
                })
            })
            .collect()
    }

    fn module_in_slice(&self, module: &ModuleInvocation) -> bool {
        self.contracts.contains_key(&module.output.index())
            && module
                .inputs
                .iter()
                .all(|input| self.contracts.contains_key(&input.index()))
    }

    fn module_contains(&self, parent_index: usize, child_index: usize) -> bool {
        if child_index >= parent_index {
            return false;
        }
        let modules = self.graph.modules();
        let parent = &modules[parent_index];
        let child = &modules[child_index];
        self.module_in_slice(parent)
            && self.module_in_slice(child)
            && child.body_start >= parent.body_start
            && child.body_end <= parent.body_end
    }

    fn direct_nested_module_indices(
        &self,
        parent_index: usize,
    ) -> Result<Vec<usize>, NumericError> {
        let modules = self.graph.modules();
        let parent = &modules[parent_index];
        let parent_internal = self
            .module_internal_nodes(parent)?
            .into_iter()
            .collect::<BTreeSet<_>>();
        let nested = (0..parent_index)
            .filter(|&child_index| {
                self.module_contains(parent_index, child_index)
                    && parent_internal.contains(&modules[child_index].output.index())
            })
            .collect::<Vec<_>>();

        Ok(nested
            .iter()
            .copied()
            .filter(|&child_index| {
                !nested.iter().copied().any(|container_index| {
                    container_index != child_index
                        && self.module_contains(container_index, child_index)
                })
            })
            .collect())
    }

    fn architecture_module_indices(&self) -> Result<Option<Vec<usize>>, NumericError> {
        let modules = self.graph.modules();
        let eligible = modules
            .iter()
            .enumerate()
            .filter_map(|(index, module)| self.module_in_slice(module).then_some(index))
            .collect::<Vec<_>>();
        let top_level = eligible
            .iter()
            .copied()
            .filter(|&child_index| {
                !eligible
                    .iter()
                    .copied()
                    .any(|parent_index| self.module_contains(parent_index, child_index))
            })
            .collect::<Vec<_>>();

        let mut producers = BTreeMap::<usize, usize>::new();
        for module_index in top_level {
            let output = modules[module_index].output.index();
            if producers.insert(output, module_index).is_some() {
                return Err(NumericError::InvalidInput {
                    node: output,
                    reason: "architecture range proof has multiple top-level module producers",
                });
            }
        }
        let Some(&final_module) = producers.get(&self.output.index()) else {
            return Ok(None);
        };

        let source_inputs = self
            .inputs
            .iter()
            .map(|input| input.index())
            .collect::<BTreeSet<_>>();
        let mut selected = BTreeSet::new();
        let mut stack = vec![final_module];
        while let Some(module_index) = stack.pop() {
            if !selected.insert(module_index) {
                continue;
            }
            let module = &modules[module_index];
            for input in &module.inputs {
                if source_inputs.contains(&input.index()) {
                    continue;
                }
                let Some(&producer_index) = producers.get(&input.index()) else {
                    return Ok(None);
                };
                if producer_index >= module_index {
                    return Err(NumericError::InvalidInput {
                        node: input.index(),
                        reason: "architecture module dependency is not topologically ordered",
                    });
                }
                stack.push(producer_index);
            }
        }

        if selected.len() < 2 {
            return Ok(None);
        }
        Ok(Some(selected.into_iter().collect()))
    }

    fn module_internal_nodes(&self, module: &ModuleInvocation) -> Result<Vec<usize>, NumericError> {
        let boundary = module
            .inputs
            .iter()
            .map(|node| node.index())
            .collect::<BTreeSet<_>>();
        let mut stack = vec![module.output.index()];
        let mut internal = BTreeSet::new();

        while let Some(index) = stack.pop() {
            if boundary.contains(&index) {
                continue;
            }
            if !self.contracts.contains_key(&index) {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "module global range proof leaves the compiled output slice",
                });
            }
            if index < module.body_start || index >= module.body_end {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "module global range proof leaves the declared module body",
                });
            }
            let node = &self.graph.nodes()[index];
            if matches!(node.operation, Operation::Input) {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "module global range proof reaches an undeclared input",
                });
            }
            if internal.insert(index) {
                stack.extend(node.inputs.iter().map(|input| input.index()));
            }
        }

        Ok(internal.into_iter().collect())
    }

    fn global_element(
        elements: &BTreeMap<(usize, usize), RneGlobalElement>,
        node: NodeId,
        element: usize,
    ) -> Result<RneGlobalElement, NumericError> {
        elements
            .get(&(node.index(), element))
            .cloned()
            .ok_or(NumericError::InvalidInput {
                node: node.index(),
                reason: "module global range proof is missing an element dependency",
            })
    }

    fn lean_conjunction(items: &[String]) -> String {
        match items {
            [] => "True".to_owned(),
            [one] => one.clone(),
            [first, rest @ ..] => {
                format!("{first} /\\ ({})", Self::lean_conjunction(rest))
            }
        }
    }

    fn lean_conjunction_proof(items: &[String]) -> String {
        match items {
            [] => "True.intro".to_owned(),
            [one] => one.clone(),
            [first, rest @ ..] => {
                format!("⟨{first}, {}⟩", Self::lean_conjunction_proof(rest))
            }
        }
    }

    fn module_global_source(
        &self,
        module_index: usize,
        module: &ModuleInvocation,
    ) -> Result<String, NumericError> {
        let internal = self.module_internal_nodes(module)?;
        let nested_modules = self.direct_nested_module_indices(module_index)?;
        let mut nested_by_output = BTreeMap::<usize, usize>::new();
        let mut nested_internal = BTreeSet::new();
        for &nested_index in &nested_modules {
            let nested = &self.graph.modules()[nested_index];
            if nested_by_output
                .insert(nested.output.index(), nested_index)
                .is_some()
            {
                return Err(NumericError::InvalidInput {
                    node: nested.output.index(),
                    reason: "module range proof has ambiguous nested module output",
                });
            }
            nested_internal.extend(self.module_internal_nodes(nested)?);
        }
        let mut source = format!(
            "-- Whole-module numerical composition for {}. Every output element\n\
             -- is derived from declared module-input contracts and explicit RNE witnesses.\n\
             -- Direct nested modules are consumed through their checked whole-module theorems.\n\
             -- Dependencies before body_start={} are rejected unless declared as inputs.\n\
             theorem module_{module_index}_global_contract\n",
            module.name, module.body_start
        );
        let mut elements = BTreeMap::<(usize, usize), RneGlobalElement>::new();
        let mut seen_inputs = BTreeSet::new();

        for &input in &module.inputs {
            if !seen_inputs.insert(input.index()) {
                continue;
            }
            let node = &self.graph.nodes()[input.index()];
            let contract = self.contracts[&input.index()];
            for element in 0..node.shape.element_count() {
                let ideal = format!("ideal_{}_{}", input.index(), element);
                let actual = format!("actual_{}_{}", input.index(), element);
                let proof = format!("h_{}_{}", input.index(), element);
                source.push_str(&format!(
                    "    {{{ideal} {actual} : Int}}\n    \
                     ({proof} : VeriNeural.ApproxContract {} {} {ideal} {actual})\n",
                    contract.ideal_bound_ticks, contract.epsilon_ticks
                ));
                elements.insert(
                    (input.index(), element),
                    RneGlobalElement {
                        ideal,
                        actual,
                        proof,
                    },
                );
            }
        }

        for &index in &internal {
            let node = &self.graph.nodes()[index];
            match &self.rules[&index] {
                RneRangeRule::Input => {
                    return Err(NumericError::InvalidInput {
                        node: index,
                        reason: "module global range proof contains an internal Input node",
                    });
                }
                RneRangeRule::Add {
                    left_shift,
                    right_shift,
                    ..
                }
                | RneRangeRule::BiasAdd {
                    left_shift,
                    right_shift,
                    ..
                } => {
                    let left_factor =
                        binary32_step_i64(*left_shift).expect("validated range scale");
                    let right_factor =
                        binary32_step_i64(*right_shift).expect("validated range scale");
                    let bias_columns = node.shape.dims().get(1).copied();
                    for element in 0..node.shape.element_count() {
                        let left = Self::global_element(&elements, node.inputs[0], element)?;
                        let right_index = if matches!(node.operation, Operation::BiasAdd) {
                            element % bias_columns.expect("validated BiasAdd rank")
                        } else {
                            element
                        };
                        let right = Self::global_element(&elements, node.inputs[1], right_index)?;
                        let actual = format!("actual_{index}_{element}");
                        let shift = format!("shift_{index}_{element}");
                        let proof = format!("h_{index}_{element}");
                        source.push_str(&format!(
                            "    {{{actual} : Int}} {{{shift} : Nat}}\n    \
                             (hrne_{index}_{element} : VeriNeural.Binary32RneTick\n      \
                               (({}) * {left_factor} + ({}) * {right_factor}) \
                               {actual} {shift})\n",
                            left.actual, right.actual
                        ));
                        elements.insert(
                            (index, element),
                            RneGlobalElement {
                                ideal: format!(
                                    "(({}) * {left_factor} + ({}) * {right_factor})",
                                    left.ideal, right.ideal
                                ),
                                actual,
                                proof,
                            },
                        );
                    }
                }
                RneRangeRule::Mul { .. } => {
                    for element in 0..node.shape.element_count() {
                        let left = Self::global_element(&elements, node.inputs[0], element)?;
                        let right = Self::global_element(&elements, node.inputs[1], element)?;
                        let actual = format!("actual_{index}_{element}");
                        let shift = format!("shift_{index}_{element}");
                        let proof = format!("h_{index}_{element}");
                        source.push_str(&format!(
                            "    {{{actual} : Int}} {{{shift} : Nat}}\n    \
                             (hrne_{index}_{element} : VeriNeural.Binary32RneTick\n      \
                               (({}) * ({})) {actual} {shift})\n",
                            left.actual, right.actual
                        ));
                        elements.insert(
                            (index, element),
                            RneGlobalElement {
                                ideal: format!("(({}) * ({}))", left.ideal, right.ideal),
                                actual,
                                proof,
                            },
                        );
                    }
                }
                RneRangeRule::MatMul { .. } => {
                    let left_shape = &self.graph.nodes()[node.inputs[0].index()].shape;
                    let right_shape = &self.graph.nodes()[node.inputs[1].index()].shape;
                    let inner = left_shape.dims()[1];
                    let columns = right_shape.dims()[1];
                    for element in 0..node.shape.element_count() {
                        let row = element / columns;
                        let column = element % columns;
                        let mut ideal_sum = String::new();
                        for j in 0..inner {
                            let left_index = row * inner + j;
                            let right_index = j * columns + column;
                            let left = Self::global_element(&elements, node.inputs[0], left_index)?;
                            let right =
                                Self::global_element(&elements, node.inputs[1], right_index)?;
                            let product = format!("product_{index}_{element}_{j}");
                            let sum = format!("sum_{index}_{element}_{j}");
                            let product_shift = format!("productShift_{index}_{element}_{j}");
                            let sum_shift = format!("sumShift_{index}_{element}_{j}");
                            let previous = if j == 0 {
                                "0".to_owned()
                            } else {
                                format!("sum_{index}_{element}_{}", j - 1)
                            };
                            source.push_str(&format!(
                                "    {{{product} {sum} : Int}}\n    \
                                 {{{product_shift} {sum_shift} : Nat}}\n    \
                                 (hproduct_{index}_{element}_{j} : \
                                   VeriNeural.Binary32RneTick\n      \
                                   (({}) * ({})) {product} {product_shift})\n    \
                                 (hsum_{index}_{element}_{j} : \
                                   VeriNeural.Binary32RneTick\n      \
                                   ({previous} + {product}) {sum} {sum_shift})\n",
                                left.actual, right.actual
                            ));
                            let term = format!("(({}) * ({}))", left.ideal, right.ideal);
                            ideal_sum = if ideal_sum.is_empty() {
                                term
                            } else {
                                format!("({ideal_sum} + {term})")
                            };
                        }
                        elements.insert(
                            (index, element),
                            RneGlobalElement {
                                ideal: ideal_sum,
                                actual: format!("sum_{index}_{element}_{}", inner - 1),
                                proof: format!("h_{index}_{element}"),
                            },
                        );
                    }
                }
                RneRangeRule::Relu | RneRangeRule::Reshape => {
                    for element in 0..node.shape.element_count() {
                        let parent = Self::global_element(&elements, node.inputs[0], element)?;
                        let (ideal, actual) = if matches!(node.operation, Operation::Relu) {
                            (
                                format!("(max ({}) 0)", parent.ideal),
                                format!("(max ({}) 0)", parent.actual),
                            )
                        } else {
                            (parent.ideal, parent.actual)
                        };
                        elements.insert(
                            (index, element),
                            RneGlobalElement {
                                ideal,
                                actual,
                                proof: format!("h_{index}_{element}"),
                            },
                        );
                    }
                }
            }
        }

        let output_contract = self.contracts[&module.output.index()];
        let output_node = &self.graph.nodes()[module.output.index()];
        let mut goals = Vec::with_capacity(output_node.shape.element_count());
        let mut output_proofs = Vec::with_capacity(output_node.shape.element_count());
        for element in 0..output_node.shape.element_count() {
            let output = Self::global_element(&elements, module.output, element)?;
            goals.push(format!(
                "VeriNeural.ApproxContract {} {} ({}) ({})",
                output_contract.ideal_bound_ticks,
                output_contract.epsilon_ticks,
                output.ideal,
                output.actual
            ));
            output_proofs.push(output.proof);
        }
        source.push_str(&format!("    : {} := by\n", Self::lean_conjunction(&goals)));

        for &index in &internal {
            if let Some(&nested_index) = nested_by_output.get(&index) {
                let nested = &self.graph.modules()[nested_index];
                let mut args = Vec::new();
                let mut seen_inputs = BTreeSet::new();
                for &input in &nested.inputs {
                    if !seen_inputs.insert(input.index()) {
                        continue;
                    }
                    let input_node = &self.graph.nodes()[input.index()];
                    for element in 0..input_node.shape.element_count() {
                        args.push(Self::global_element(&elements, input, element)?.proof);
                    }
                }
                for nested_node in self.module_internal_nodes(nested)? {
                    args.extend(self.architecture_witness_args(nested_node));
                }
                let call_args = if args.is_empty() {
                    String::new()
                } else {
                    format!(" {}", args.join(" "))
                };
                let nested_result = format!("nested_module_{nested_index}");
                source.push_str(&format!(
                    "  have {nested_result} := module_{nested_index}_global_contract{call_args}\n"
                ));
                let output_node = &self.graph.nodes()[nested.output.index()];
                let output_proofs = (0..output_node.shape.element_count())
                    .map(|element| {
                        Self::global_element(&elements, nested.output, element)
                            .map(|output| output.proof)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if output_proofs.len() == 1 {
                    source.push_str(&format!("  have {} := {nested_result}\n", output_proofs[0]));
                } else {
                    source.push_str(&format!(
                        "  rcases {nested_result} with {}\n",
                        Self::lean_conjunction_pattern(&output_proofs)
                    ));
                }
                continue;
            }
            if nested_internal.contains(&index) {
                continue;
            }

            let node = &self.graph.nodes()[index];
            let contract = self.contracts[&index];
            match &self.rules[&index] {
                RneRangeRule::Input => unreachable!("rejected above"),
                RneRangeRule::Add { .. } | RneRangeRule::BiasAdd { .. } => {
                    let bias_columns = node.shape.dims().get(1).copied();
                    for element in 0..node.shape.element_count() {
                        let left = Self::global_element(&elements, node.inputs[0], element)?;
                        let right_index = if matches!(node.operation, Operation::BiasAdd) {
                            element % bias_columns.expect("validated BiasAdd rank")
                        } else {
                            element
                        };
                        let right = Self::global_element(&elements, node.inputs[1], right_index)?;
                        let output = Self::global_element(&elements, node.id, element)?;
                        source.push_str(&format!(
                            "  have {} : VeriNeural.ApproxContract {} {} ({}) ({}) := by\n    \
                             exact node_{index}_range_rule {} {} hrne_{index}_{element}\n",
                            output.proof,
                            contract.ideal_bound_ticks,
                            contract.epsilon_ticks,
                            output.ideal,
                            output.actual,
                            left.proof,
                            right.proof
                        ));
                    }
                }
                RneRangeRule::Mul { .. } => {
                    for element in 0..node.shape.element_count() {
                        let left = Self::global_element(&elements, node.inputs[0], element)?;
                        let right = Self::global_element(&elements, node.inputs[1], element)?;
                        let output = Self::global_element(&elements, node.id, element)?;
                        source.push_str(&format!(
                            "  have {} : VeriNeural.ApproxContract {} {} ({}) ({}) := by\n    \
                             exact node_{index}_range_rule {} {} hrne_{index}_{element}\n",
                            output.proof,
                            contract.ideal_bound_ticks,
                            contract.epsilon_ticks,
                            output.ideal,
                            output.actual,
                            left.proof,
                            right.proof
                        ));
                    }
                }
                RneRangeRule::MatMul { .. } => {
                    let left_shape = &self.graph.nodes()[node.inputs[0].index()].shape;
                    let right_shape = &self.graph.nodes()[node.inputs[1].index()].shape;
                    let inner = left_shape.dims()[1];
                    let columns = right_shape.dims()[1];
                    for element in 0..node.shape.element_count() {
                        let row = element / columns;
                        let column = element % columns;
                        let mut args = String::new();
                        for j in 0..inner {
                            let left =
                                Self::global_element(&elements, node.inputs[0], row * inner + j)?;
                            let right = Self::global_element(
                                &elements,
                                node.inputs[1],
                                j * columns + column,
                            )?;
                            args.push_str(&format!(
                                " {} {} hproduct_{index}_{element}_{j} hsum_{index}_{element}_{j}",
                                left.proof, right.proof
                            ));
                        }
                        let output = Self::global_element(&elements, node.id, element)?;
                        source.push_str(&format!(
                            "  have {} : VeriNeural.ApproxContract {} {} ({}) ({}) := by\n    \
                             exact node_{index}_range_rule{args}\n",
                            output.proof,
                            contract.ideal_bound_ticks,
                            contract.epsilon_ticks,
                            output.ideal,
                            output.actual
                        ));
                    }
                }
                RneRangeRule::Relu | RneRangeRule::Reshape => {
                    for element in 0..node.shape.element_count() {
                        let parent = Self::global_element(&elements, node.inputs[0], element)?;
                        let output = Self::global_element(&elements, node.id, element)?;
                        source.push_str(&format!(
                            "  have {} : VeriNeural.ApproxContract {} {} ({}) ({}) := by\n    \
                             exact node_{index}_range_rule {}\n",
                            output.proof,
                            contract.ideal_bound_ticks,
                            contract.epsilon_ticks,
                            output.ideal,
                            output.actual,
                            parent.proof
                        ));
                    }
                }
            }
        }

        source.push_str(&format!(
            "  exact {}\n\n",
            Self::lean_conjunction_proof(&output_proofs)
        ));
        Ok(source)
    }

    fn lean_conjunction_pattern(items: &[String]) -> String {
        match items {
            [] => "_".to_owned(),
            [one] => one.clone(),
            [first, rest @ ..] => {
                format!("⟨{first}, {}⟩", Self::lean_conjunction_pattern(rest))
            }
        }
    }

    fn architecture_register_node(
        &self,
        index: usize,
        elements: &mut BTreeMap<(usize, usize), RneGlobalElement>,
        source: &mut String,
    ) -> Result<(), NumericError> {
        let node = &self.graph.nodes()[index];
        if (0..node.shape.element_count()).all(|element| elements.contains_key(&(index, element))) {
            return Ok(());
        }

        match &self.rules[&index] {
            RneRangeRule::Input => {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "architecture range proof reached an undeclared Input",
                });
            }
            RneRangeRule::Add {
                left_shift,
                right_shift,
                ..
            }
            | RneRangeRule::BiasAdd {
                left_shift,
                right_shift,
                ..
            } => {
                let left_factor = binary32_step_i64(*left_shift).expect("validated range scale");
                let right_factor = binary32_step_i64(*right_shift).expect("validated range scale");
                let bias_columns = node.shape.dims().get(1).copied();
                for element in 0..node.shape.element_count() {
                    let left = Self::global_element(elements, node.inputs[0], element)?;
                    let right_index = if matches!(node.operation, Operation::BiasAdd) {
                        element % bias_columns.expect("validated BiasAdd rank")
                    } else {
                        element
                    };
                    let right = Self::global_element(elements, node.inputs[1], right_index)?;
                    let actual = format!("actual_{index}_{element}");
                    let shift = format!("shift_{index}_{element}");
                    source.push_str(&format!(
                        "    {{{actual} : Int}} {{{shift} : Nat}}\n    \
                         (hrne_{index}_{element} : VeriNeural.Binary32RneTick\n      \
                           (({}) * {left_factor} + ({}) * {right_factor}) \
                           {actual} {shift})\n",
                        left.actual, right.actual
                    ));
                    elements.insert(
                        (index, element),
                        RneGlobalElement {
                            ideal: format!(
                                "(({}) * {left_factor} + ({}) * {right_factor})",
                                left.ideal, right.ideal
                            ),
                            actual,
                            proof: String::new(),
                        },
                    );
                }
            }
            RneRangeRule::Mul { .. } => {
                for element in 0..node.shape.element_count() {
                    let left = Self::global_element(elements, node.inputs[0], element)?;
                    let right = Self::global_element(elements, node.inputs[1], element)?;
                    let actual = format!("actual_{index}_{element}");
                    let shift = format!("shift_{index}_{element}");
                    source.push_str(&format!(
                        "    {{{actual} : Int}} {{{shift} : Nat}}\n    \
                         (hrne_{index}_{element} : VeriNeural.Binary32RneTick\n      \
                           (({}) * ({})) {actual} {shift})\n",
                        left.actual, right.actual
                    ));
                    elements.insert(
                        (index, element),
                        RneGlobalElement {
                            ideal: format!("(({}) * ({}))", left.ideal, right.ideal),
                            actual,
                            proof: String::new(),
                        },
                    );
                }
            }
            RneRangeRule::MatMul { .. } => {
                let left_shape = &self.graph.nodes()[node.inputs[0].index()].shape;
                let right_shape = &self.graph.nodes()[node.inputs[1].index()].shape;
                let inner = left_shape.dims()[1];
                let columns = right_shape.dims()[1];
                for element in 0..node.shape.element_count() {
                    let row = element / columns;
                    let column = element % columns;
                    let mut ideal_sum = String::new();
                    for j in 0..inner {
                        let left = Self::global_element(elements, node.inputs[0], row * inner + j)?;
                        let right =
                            Self::global_element(elements, node.inputs[1], j * columns + column)?;
                        let product = format!("product_{index}_{element}_{j}");
                        let sum = format!("sum_{index}_{element}_{j}");
                        let product_shift = format!("productShift_{index}_{element}_{j}");
                        let sum_shift = format!("sumShift_{index}_{element}_{j}");
                        let previous = if j == 0 {
                            "0".to_owned()
                        } else {
                            format!("sum_{index}_{element}_{}", j - 1)
                        };
                        source.push_str(&format!(
                            "    {{{product} {sum} : Int}}\n    \
                             {{{product_shift} {sum_shift} : Nat}}\n    \
                             (hproduct_{index}_{element}_{j} : \
                               VeriNeural.Binary32RneTick\n      \
                               (({}) * ({})) {product} {product_shift})\n    \
                             (hsum_{index}_{element}_{j} : \
                               VeriNeural.Binary32RneTick\n      \
                               ({previous} + {product}) {sum} {sum_shift})\n",
                            left.actual, right.actual
                        ));
                        let term = format!("(({}) * ({}))", left.ideal, right.ideal);
                        ideal_sum = if ideal_sum.is_empty() {
                            term
                        } else {
                            format!("({ideal_sum} + {term})")
                        };
                    }
                    elements.insert(
                        (index, element),
                        RneGlobalElement {
                            ideal: ideal_sum,
                            actual: format!("sum_{index}_{element}_{}", inner - 1),
                            proof: String::new(),
                        },
                    );
                }
            }
            RneRangeRule::Relu | RneRangeRule::Reshape => {
                for element in 0..node.shape.element_count() {
                    let parent = Self::global_element(elements, node.inputs[0], element)?;
                    let (ideal, actual) = if matches!(node.operation, Operation::Relu) {
                        (
                            format!("(max ({}) 0)", parent.ideal),
                            format!("(max ({}) 0)", parent.actual),
                        )
                    } else {
                        (parent.ideal, parent.actual)
                    };
                    elements.insert(
                        (index, element),
                        RneGlobalElement {
                            ideal,
                            actual,
                            proof: String::new(),
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn architecture_witness_args(&self, index: usize) -> Vec<String> {
        let node = &self.graph.nodes()[index];
        match &self.rules[&index] {
            RneRangeRule::Add { .. } | RneRangeRule::BiasAdd { .. } | RneRangeRule::Mul { .. } => {
                (0..node.shape.element_count())
                    .map(|element| format!("hrne_{index}_{element}"))
                    .collect()
            }
            RneRangeRule::MatMul { .. } => {
                let inner = self.graph.nodes()[node.inputs[0].index()].shape.dims()[1];
                let mut args = Vec::new();
                for element in 0..node.shape.element_count() {
                    for j in 0..inner {
                        args.push(format!("hproduct_{index}_{element}_{j}"));
                        args.push(format!("hsum_{index}_{element}_{j}"));
                    }
                }
                args
            }
            RneRangeRule::Input | RneRangeRule::Relu | RneRangeRule::Reshape => Vec::new(),
        }
    }

    fn architecture_global_source(&self) -> Result<Option<String>, NumericError> {
        let Some(module_indices) = self.architecture_module_indices()? else {
            return Ok(None);
        };

        let mut source = String::from(
            "-- Architecture-level numerical composition over the reachable module DAG.\n\
             -- Producer module theorems feed every dependent branch and fan-in boundary.\n\
             theorem architecture_global_contract\n",
        );
        let mut elements = BTreeMap::<(usize, usize), RneGlobalElement>::new();
        for &input in &self.inputs {
            let node = &self.graph.nodes()[input.index()];
            let contract = self.contracts[&input.index()];
            for element in 0..node.shape.element_count() {
                let ideal = format!("ideal_{}_{}", input.index(), element);
                let actual = format!("actual_{}_{}", input.index(), element);
                let proof = format!("arch_h_{}_{}", input.index(), element);
                source.push_str(&format!(
                    "    {{{ideal} {actual} : Int}}\n    \
                     ({proof} : VeriNeural.ApproxContract {} {} {ideal} {actual})\n",
                    contract.ideal_bound_ticks, contract.epsilon_ticks
                ));
                elements.insert(
                    (input.index(), element),
                    RneGlobalElement {
                        ideal,
                        actual,
                        proof,
                    },
                );
            }
        }

        let mut declared_nodes = BTreeSet::new();
        let mut steps = Vec::<RneArchitectureStep>::new();
        for &module_index in &module_indices {
            let module = &self.graph.modules()[module_index];
            let mut args = Vec::new();
            let mut seen_inputs = BTreeSet::new();
            for &input in &module.inputs {
                if !seen_inputs.insert(input.index()) {
                    continue;
                }
                let node = &self.graph.nodes()[input.index()];
                for element in 0..node.shape.element_count() {
                    let input_element = Self::global_element(&elements, input, element)?;
                    if input_element.proof.is_empty() {
                        return Ok(None);
                    }
                    args.push(input_element.proof);
                }
            }

            let internal = self.module_internal_nodes(module)?;
            for &index in &internal {
                if declared_nodes.insert(index) {
                    self.architecture_register_node(index, &mut elements, &mut source)?;
                }
                args.extend(self.architecture_witness_args(index));
            }

            let output_node = &self.graph.nodes()[module.output.index()];
            let output_count = output_node.shape.element_count();
            let output_proofs = (0..output_count)
                .map(|element| format!("arch_h_{}_{}", module.output.index(), element))
                .collect::<Vec<_>>();
            for (element, proof) in output_proofs.iter().enumerate() {
                let output = elements.get_mut(&(module.output.index(), element)).ok_or(
                    NumericError::InvalidInput {
                        node: module.output.index(),
                        reason: "architecture range proof is missing a module output",
                    },
                )?;
                output.proof = proof.clone();
            }
            steps.push(RneArchitectureStep {
                module_index,
                args,
                output_proofs,
            });
        }

        let output_contract = self.output_contract();
        let output_node = &self.graph.nodes()[self.output.index()];
        let mut goals = Vec::with_capacity(output_node.shape.element_count());
        let mut final_proofs = Vec::with_capacity(output_node.shape.element_count());
        for element in 0..output_node.shape.element_count() {
            let output = Self::global_element(&elements, self.output, element)?;
            if output.proof.is_empty() {
                return Ok(None);
            }
            goals.push(format!(
                "VeriNeural.ApproxContract {} {} ({}) ({})",
                output_contract.ideal_bound_ticks,
                output_contract.epsilon_ticks,
                output.ideal,
                output.actual
            ));
            final_proofs.push(output.proof);
        }
        source.push_str(&format!("    : {} := by\n", Self::lean_conjunction(&goals)));
        for step in steps {
            let call_args = if step.args.is_empty() {
                String::new()
            } else {
                format!(" {}", step.args.join(" "))
            };
            let module_result = format!("arch_module_{}", step.module_index);
            source.push_str(&format!(
                "  have {module_result} := module_{}_global_contract{call_args}\n",
                step.module_index
            ));
            if step.output_proofs.len() == 1 {
                source.push_str(&format!(
                    "  have {} := {module_result}\n",
                    step.output_proofs[0]
                ));
            } else {
                source.push_str(&format!(
                    "  rcases {module_result} with {}\n",
                    Self::lean_conjunction_pattern(&step.output_proofs)
                ));
            }
        }
        source.push_str(&format!(
            "  exact {}\n\n",
            Self::lean_conjunction_proof(&final_proofs)
        ));
        Ok(Some(source))
    }

    fn add_rule_source(
        &self,
        index: usize,
        left_shift: u32,
        right_shift: u32,
        max_shift: u32,
    ) -> String {
        let node = &self.graph.nodes()[index];
        let left = self.contracts[&node.inputs[0].index()];
        let right = self.contracts[&node.inputs[1].index()];
        let output = self.contracts[&index];
        let left_factor = binary32_step_i64(left_shift).expect("validated range scale");
        let right_factor = binary32_step_i64(right_shift).expect("validated range scale");
        let left_scaled = Self::scale_contract(index, left, output.fraction_bits)
            .expect("validated range scale")
            .0;
        let right_scaled = Self::scale_contract(index, right, output.fraction_bits)
            .expect("validated range scale")
            .0;
        format!(
            "theorem node_{index}_range_rule\n    \
             {{idealA actualA idealB actualB rounded : Int}} {{shift : Nat}}\n    \
             (ha : VeriNeural.ApproxContract {} {} idealA actualA)\n    \
             (hb : VeriNeural.ApproxContract {} {} idealB actualB)\n    \
             (hrne : VeriNeural.Binary32RneTick\n      \
               (actualA * {left_factor} + actualB * {right_factor}) rounded shift) :\n    \
             VeriNeural.ApproxContract {} {}\n      \
               (idealA * {left_factor} + idealB * {right_factor}) rounded := by\n  \
             have haScaled : VeriNeural.ApproxContract {} {}\n      \
                 (idealA * {left_factor}) (actualA * {left_factor}) := by\n    \
               simpa using (VeriNeural.approx_contract_scale (scale := {left_factor})\n      \
                 (by decide) ha)\n  \
             have hbScaled : VeriNeural.ApproxContract {} {}\n      \
                 (idealB * {right_factor}) (actualB * {right_factor}) := by\n    \
               simpa using (VeriNeural.approx_contract_scale (scale := {right_factor})\n      \
                 (by decide) hb)\n  \
             simpa [VeriNeural.binary32Step] using\n    \
               (VeriNeural.approx_contract_add_rne (maxShift := {max_shift})\n      \
                 haScaled hbScaled (by decide) hrne)\n\n",
            left.ideal_bound_ticks,
            left.epsilon_ticks,
            right.ideal_bound_ticks,
            right.epsilon_ticks,
            output.ideal_bound_ticks,
            output.epsilon_ticks,
            left_scaled.ideal_bound_ticks,
            left_scaled.epsilon_ticks,
            right_scaled.ideal_bound_ticks,
            right_scaled.epsilon_ticks,
        )
    }

    fn mul_rule_source(&self, index: usize, max_shift: u32) -> String {
        let node = &self.graph.nodes()[index];
        let left = self.contracts[&node.inputs[0].index()];
        let right = self.contracts[&node.inputs[1].index()];
        let output = self.contracts[&index];
        format!(
            "theorem node_{index}_range_rule\n    \
             {{idealA actualA idealB actualB rounded : Int}} {{shift : Nat}}\n    \
             (ha : VeriNeural.ApproxContract {} {} idealA actualA)\n    \
             (hb : VeriNeural.ApproxContract {} {} idealB actualB)\n    \
             (hrne : VeriNeural.Binary32RneTick (actualA * actualB) rounded shift) :\n    \
             VeriNeural.ApproxContract {} {} (idealA * idealB) rounded := by\n  \
             simpa [VeriNeural.binary32Step] using\n    \
               (VeriNeural.approx_contract_mul_rne (maxShift := {max_shift})\n      \
                 ha hb (by decide) hrne)\n\n",
            left.ideal_bound_ticks,
            left.epsilon_ticks,
            right.ideal_bound_ticks,
            right.epsilon_ticks,
            output.ideal_bound_ticks,
            output.epsilon_ticks,
        )
    }

    fn matmul_rule_source(
        &self,
        index: usize,
        product: RneErrorContract,
        product_max_shift: u32,
        sums: &[(RneErrorContract, u32)],
    ) -> String {
        let node = &self.graph.nodes()[index];
        let left = self.contracts[&node.inputs[0].index()];
        let right = self.contracts[&node.inputs[1].index()];
        let output = self.contracts[&index];
        let inner = sums.len();
        let mut source = format!("theorem node_{index}_range_rule\n");
        for j in 0..inner {
            source.push_str(&format!(
                "    {{idealA_{j} actualA_{j} idealB_{j} actualB_{j} product_{j} sum_{j} : Int}}\n    \
                 {{productShift_{j} sumShift_{j} : Nat}}\n"
            ));
        }
        for j in 0..inner {
            source.push_str(&format!(
                "    (ha_{j} : VeriNeural.ApproxContract {} {} idealA_{j} actualA_{j})\n    \
                 (hb_{j} : VeriNeural.ApproxContract {} {} idealB_{j} actualB_{j})\n    \
                 (hproduct_{j} : VeriNeural.Binary32RneTick\n      \
                   (actualA_{j} * actualB_{j}) product_{j} productShift_{j})\n",
                left.ideal_bound_ticks,
                left.epsilon_ticks,
                right.ideal_bound_ticks,
                right.epsilon_ticks,
            ));
            let previous = if j == 0 {
                "0".to_owned()
            } else {
                format!("sum_{}", j - 1)
            };
            source.push_str(&format!(
                "    (hsum_{j} : VeriNeural.Binary32RneTick\n      \
                   ({previous} + product_{j}) sum_{j} sumShift_{j})\n"
            ));
        }
        let mut ideal_expression = String::new();
        for j in 0..inner {
            let term = format!("idealA_{j} * idealB_{j}");
            ideal_expression = if ideal_expression.is_empty() {
                term
            } else {
                format!("({ideal_expression} + {term})")
            };
        }
        source.push_str(&format!(
            "    : VeriNeural.ApproxContract {} {}\n      \
             {ideal_expression} sum_{} := by\n  \
             have hacc_0 : VeriNeural.ApproxContract 0 0 0 0 := by\n    \
               exact VeriNeural.approx_contract_exact (by decide) (by decide)\n",
            output.ideal_bound_ticks,
            output.epsilon_ticks,
            inner - 1,
        ));
        let mut previous_contract_name = "hacc_0".to_owned();
        for (j, (sum_contract, sum_max_shift)) in sums.iter().enumerate() {
            source.push_str(&format!(
                "  have hproductContract_{j} : VeriNeural.ApproxContract {} {}\n      \
                   (idealA_{j} * idealB_{j}) product_{j} := by\n    \
                 simpa [VeriNeural.binary32Step] using\n      \
                   (VeriNeural.approx_contract_mul_rne (maxShift := {product_max_shift})\n        \
                     ha_{j} hb_{j} (by decide) hproduct_{j})\n",
                product.ideal_bound_ticks, product.epsilon_ticks
            ));
            let ideal_sum = {
                let mut expr = String::new();
                for k in 0..=j {
                    let term = format!("idealA_{k} * idealB_{k}");
                    expr = if expr.is_empty() {
                        term
                    } else {
                        format!("({expr} + {term})")
                    };
                }
                expr
            };
            source.push_str(&format!(
                "  have hsumContract_{j} : VeriNeural.ApproxContract {} {}\n      \
                   ({ideal_sum}) sum_{j} := by\n    \
                 simpa [VeriNeural.binary32Step] using\n      \
                   (VeriNeural.approx_contract_add_rne (maxShift := {sum_max_shift})\n        \
                     {previous_contract_name} hproductContract_{j} (by decide) hsum_{j})\n",
                sum_contract.ideal_bound_ticks, sum_contract.epsilon_ticks
            ));
            previous_contract_name = format!("hsumContract_{j}");
        }
        source.push_str(&format!("  exact {previous_contract_name}\n\n"));
        source
    }

    /// Emit range-transfer theorems with arbitrary scalar values. The numeric
    /// bounds are fixed by this plan, but the values themselves are variables,
    /// so the proof is reusable for every input satisfying the source contract.
    pub fn lean_source(&self) -> Result<String, NumericError> {
        let base = LeanProofEngine::default().generate(&self.graph, self.output)?;
        let mut source = base.source().to_owned();
        source.push_str("\n\n");
        source.push_str(MEMORY_LIBRARY);
        source.push_str("\n\n");
        source.push_str(NUMERIC_LIBRARY);
        source.push_str("\nnamespace VeriNeural.Generated.RneRange\n\n");
        source.push_str(&format!(
            "-- Reusable dyadic range contract: source q={}, |ticks| <= {}.\n\
             -- Values are arbitrary; local arithmetic is conditional on Binary32RneTick.\n\
             -- LLVM/PTX/JIT/GPU conformance is not proved here.\n\n",
            self.source_fraction_bits, self.source_bound_ticks
        ));

        for &index in &self.reachable {
            let node = &self.graph.nodes()[index];
            let contract = self.contracts[&index];
            match &self.rules[&index] {
                RneRangeRule::Input => {
                    source.push_str(&format!(
                        "theorem node_{index}_range_rule\n    \
                         {{ideal actual : Int}}\n    \
                         (h : VeriNeural.ApproxContract {} {} ideal actual) :\n    \
                         VeriNeural.ApproxContract {} {} ideal actual := by\n  \
                         exact h\n\n",
                        contract.ideal_bound_ticks,
                        contract.epsilon_ticks,
                        contract.ideal_bound_ticks,
                        contract.epsilon_ticks,
                    ));
                }
                RneRangeRule::Add {
                    left_shift,
                    right_shift,
                    max_shift,
                }
                | RneRangeRule::BiasAdd {
                    left_shift,
                    right_shift,
                    max_shift,
                } => source.push_str(&self.add_rule_source(
                    index,
                    *left_shift,
                    *right_shift,
                    *max_shift,
                )),
                RneRangeRule::Mul { max_shift } => {
                    source.push_str(&self.mul_rule_source(index, *max_shift))
                }
                RneRangeRule::MatMul {
                    product,
                    product_max_shift,
                    sums,
                } => source.push_str(&self.matmul_rule_source(
                    index,
                    *product,
                    *product_max_shift,
                    sums,
                )),
                RneRangeRule::Relu => {
                    let parent = self.contracts[&node.inputs[0].index()];
                    source.push_str(&format!(
                        "theorem node_{index}_range_rule\n    \
                         {{ideal actual : Int}}\n    \
                         (h : VeriNeural.ApproxContract {} {} ideal actual) :\n    \
                         VeriNeural.ApproxContract {} {} (max ideal 0) (max actual 0) := by\n  \
                         simpa using (VeriNeural.approx_contract_relu h)\n\n",
                        parent.ideal_bound_ticks,
                        parent.epsilon_ticks,
                        contract.ideal_bound_ticks,
                        contract.epsilon_ticks,
                    ));
                    source.push_str(&format!(
                        r#"theorem node_{index}_positive_branch
    {{ideal actual : Int}}
    (h : VeriNeural.ApproxContract {} {} ideal actual)
    (hmargin : {} < actual) :
    max ideal 0 = ideal ∧ max actual 0 = actual := by
  exact VeriNeural.approx_contract_relu_positive_branch h hmargin

theorem node_{index}_nonpositive_branch
    {{ideal actual : Int}}
    (h : VeriNeural.ApproxContract {} {} ideal actual)
    (hmargin : actual + {} ≤ 0) :
    max ideal 0 = 0 ∧ max actual 0 = 0 := by
  exact VeriNeural.approx_contract_relu_nonpositive_branch h hmargin

"#,
                        parent.ideal_bound_ticks,
                        parent.epsilon_ticks,
                        parent.epsilon_ticks,
                        parent.ideal_bound_ticks,
                        parent.epsilon_ticks,
                        parent.epsilon_ticks,
                    ));
                }
                RneRangeRule::Reshape => {
                    let parent = self.contracts[&node.inputs[0].index()];
                    source.push_str(&format!(
                        "theorem node_{index}_range_rule\n    \
                         {{ideal actual : Int}}\n    \
                         (h : VeriNeural.ApproxContract {} {} ideal actual) :\n    \
                         VeriNeural.ApproxContract {} {} ideal actual := by\n  \
                         exact h\n\n",
                        parent.ideal_bound_ticks,
                        parent.epsilon_ticks,
                        contract.ideal_bound_ticks,
                        contract.epsilon_ticks,
                    ));
                }
            }
        }

        let output = self.output_contract();
        source.push_str(&format!(
            "theorem output_contract_well_formed :\n  \
             (0 : Int) <= {} /\\ (0 : Int) <= {} := by\n  \
             decide\n\n",
            output.ideal_bound_ticks, output.epsilon_ticks
        ));
        for (module_index, module) in self.graph.modules().iter().enumerate() {
            if !self.module_in_slice(module) {
                continue;
            }
            let contract = self.contracts[&module.output.index()];
            source.push_str(&format!(
                "-- Module {} (output node {}) exports q={}, ideal bound {}, epsilon {} ticks.\n\
                 theorem module_{module_index}_contract_well_formed :\n  \
                   (0 : Int) <= {} /\\ (0 : Int) <= {} := by\n  \
                   decide\n\n",
                module.name,
                module.output.index(),
                contract.fraction_bits,
                contract.ideal_bound_ticks,
                contract.epsilon_ticks,
                contract.ideal_bound_ticks,
                contract.epsilon_ticks,
            ));
            source.push_str(&self.module_global_source(module_index, module)?);
        }
        if let Some(architecture) = self.architecture_global_source()? {
            source.push_str(&architecture);
        }
        source.push_str("end VeriNeural.Generated.RneRange\n");
        Ok(source)
    }
}

fn rescale_i64(value: i64, shift: u32) -> Option<i64> {
    let factor = 1i128.checked_shl(shift)?;
    let scaled = i128::from(value).checked_mul(factor)?;
    i64::try_from(scaled).ok()
}

fn binary32_step_i64(shift: u32) -> Option<i64> {
    1i64.checked_shl(shift)
}

fn f32_to_dyadic_ticks(value: f32, fraction_bits: u32) -> Result<i64, &'static str> {
    let bits = value.to_bits();
    let sign = bits >> 31;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;

    if exponent == 0xff {
        return Err("RNE dyadic inputs must be finite");
    }
    if exponent == 0 {
        if fraction == 0 {
            if sign != 0 {
                return Err("RNE dyadic slice does not model signed zero");
            }
            return Ok(0);
        }
        return Err("RNE dyadic slice does not yet model subnormals");
    }

    let significand = (1u64 << 23) | u64::from(fraction);
    let unbiased = exponent as i32 - 127;
    let shift = unbiased - 23 + fraction_bits as i32;
    let magnitude = if shift >= 0 {
        let left = shift as u32;
        if left > 104 {
            return Err("fixed-point input overflow");
        }
        (u128::from(significand)) << left
    } else {
        let right = (-shift) as u32;
        if right >= 64 {
            return Err("input is not on the selected dyadic lattice");
        }
        let mask = if right == 0 { 0 } else { (1u64 << right) - 1 };
        if significand & mask != 0 {
            return Err("input is not on the selected dyadic lattice");
        }
        u128::from(significand >> right)
    };
    if magnitude > i64::MAX as u128 {
        return Err("fixed-point input overflow");
    }
    let magnitude = magnitude as i64;
    Ok(if sign == 0 { magnitude } else { -magnitude })
}

fn round_ticks_to_binary32(
    value: i64,
    fraction_bits: u32,
) -> Result<(i64, u32, u32), &'static str> {
    if value == 0 {
        return Ok((0, 0, 0));
    }
    let magnitude = value.unsigned_abs();
    if magnitude > i64::MAX as u64 {
        return Err("fixed-point magnitude exceeds supported i64 range");
    }
    let bit_len = 64 - magnitude.leading_zeros();
    let shift = bit_len.saturating_sub(24);
    let rounded_magnitude = if shift == 0 {
        magnitude
    } else {
        let step = 1u64 << shift;
        let quotient = magnitude >> shift;
        let remainder = magnitude & (step - 1);
        let half = step >> 1;
        let increment = remainder > half || (remainder == half && quotient & 1 == 1);
        let rounded_quotient = quotient + if increment { 1 } else { 0 };
        rounded_quotient
            .checked_shl(shift)
            .ok_or("binary32 rounding overflow")?
    };
    if rounded_magnitude > i64::MAX as u64 {
        return Err("binary32 rounded value exceeds supported i64 range");
    }
    let rounded = if value < 0 {
        -(rounded_magnitude as i64)
    } else {
        rounded_magnitude as i64
    };
    let bits = encode_representable_ticks(rounded, fraction_bits)?;
    Ok((rounded, bits, shift))
}

fn encode_representable_ticks(value: i64, fraction_bits: u32) -> Result<u32, &'static str> {
    if value == 0 {
        return Ok(0);
    }
    let sign = if value < 0 { 1 } else { 0 };
    let magnitude = value.unsigned_abs();
    if magnitude > i64::MAX as u64 {
        return Err("fixed-point magnitude exceeds supported i64 range");
    }
    let bit_len = 64 - magnitude.leading_zeros();
    let precision_shift = bit_len.saturating_sub(24);
    if precision_shift > 0 {
        let mask = (1u64 << precision_shift) - 1;
        if magnitude & mask != 0 {
            return Err("rounded tick is not exactly representable as binary32");
        }
    }
    let significand = if bit_len < 24 {
        magnitude << (24 - bit_len)
    } else {
        magnitude >> precision_shift
    };
    let exponent = bit_len as i32 - 1 - fraction_bits as i32;
    if !(-126..=127).contains(&exponent) {
        return Err("dyadic value leaves the normal finite binary32 range");
    }
    let biased = (exponent + 127) as u32;
    let fraction = (significand as u32) & 0x7f_ffff;
    Ok((sign << 31) | (biased << 23) | fraction)
}
