//! Stage 8 first vertical slice: exact-integer FP32 domain and compositional
//! range obligations for Add, BiasAdd, Relu and Reshape.
//!
//! All integral inputs and intermediate mathematical values are restricted to
//! magnitude <= 2^24, where every integer has a binary32 representation.
//! Lean establishes *ideal Int* bounds, and the integer oracle validates the
//! concrete GPU output bitwise. Correct IEEE RNE hardware/LLVM behavior is
//! STILL TRUSTED, not proved by this engine. Non-integral or negative-zero
//! inputs and unsupported operators fail closed.
//!
//! A second, narrower slice models concrete fractional dyadic traces under
//! round-to-nearest, ties-to-even and composes a nonzero absolute error budget
//! for Add/BiasAdd/Relu/Reshape. The LLVM/PTX/GPU execution link remains TCB.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{LeanProofEngine, NeuralError, NeuralGraph, NodeId, Operation, ProofError, Tensor};

const EXACT_F32_INTEGER_LIMIT: i64 = 1 << 24;
const SOURCE_INTEGER_LIMIT: i64 = 1 << 20;
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
                write!(
                    f,
                    "exact-integer FP32 subset rejects {operation} at node {node}"
                )
            }
            Self::UnsupportedRange { node, bound } => write!(
                f,
                "node {node} needs integer bound {bound}, outside exact binary32 subset"
            ),
            Self::InvalidInput { node, reason } => {
                write!(f, "FP32 exact-integer input {node}: {reason}")
            }
            Self::MissingInput { node } => write!(f, "missing numeric input node {node}"),
            Self::OutputBits {
                index,
                expected,
                actual,
            } => write!(
                f,
                "FP32 integer oracle mismatch at {index}: expected={expected:08x} GPU={actual:08x}"
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
/// Values are represented internally as exact integer ticks of 2^-fraction_bits.
/// The current slice deliberately supports only additive operators
/// (Add/BiasAdd/Relu/Reshape): multiplication changes the fixed-point scale and
/// will be handled by a later slice. No f32 arithmetic is used by the oracle.
#[derive(Clone, Debug)]
pub struct RneFp32Plan {
    graph: NeuralGraph,
    output: NodeId,
    reachable: Vec<usize>,
    inputs: Vec<NodeId>,
    fraction_bits: u32,
}

/// Result of one concrete, independently rounded binary32 trace.
///
/// epsilon_ticks is the Lean-composed conservative absolute-error budget.
/// observed_error_ticks is the actual deviation of the integer RNE oracle
/// from ideal fixed-point arithmetic for this concrete trace.
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

#[derive(Clone, Debug)]
struct RneNodeTrace {
    ideal: Vec<i64>,
    actual: Vec<i64>,
    epsilon: Vec<i64>,
    rounding: Vec<Option<RneWitness>>,
}

impl RneFp32Plan {
    /// Compile a concrete dyadic RNE trace profile.
    ///
    /// fraction_bits chooses the exact input lattice 2^-q. Restricting
    /// q <= 30 keeps every nonzero i64 tick safely in the normal finite
    /// binary32 exponent range, so this first slice can reject subnormals
    /// rather than silently inventing FTZ semantics.
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

        Ok(Self {
            graph: graph.clone(),
            output,
            reachable,
            inputs,
            fraction_bits,
        })
    }

    pub fn fraction_bits(&self) -> u32 {
        self.fraction_bits
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

    /// Execute the independent fixed-point/RNE oracle and build a concrete
    /// Lean proof trace. The returned f32 values are constructed from bits;
    /// the oracle itself never performs f32 addition or rounding.
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
                        rounding: vec![None; len],
                    }
                }
                Operation::Add => {
                    let left = &nodes[&node.inputs[0].index()];
                    let right = &nodes[&node.inputs[1].index()];
                    self.trace_add(index, left, right, None)?
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
                        rounding: vec![None; len],
                    }
                }
                Operation::Reshape(_) => {
                    let parent = &nodes[&node.inputs[0].index()];
                    RneNodeTrace {
                        ideal: parent.ideal.clone(),
                        actual: parent.actual.clone(),
                        epsilon: parent.epsilon.clone(),
                        rounding: vec![None; len],
                    }
                }
                _ => unreachable!("rejected during RNE plan compilation"),
            };
            if trace.actual.len() != len {
                return Err(NumericError::InvalidInput {
                    node: index,
                    reason: "internal RNE trace shape mismatch",
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
                encode_representable_ticks(ticks, self.fraction_bits)
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
            fraction_bits: self.fraction_bits,
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
        let mut ideal = Vec::with_capacity(len);
        let mut actual = Vec::with_capacity(len);
        let mut epsilon = Vec::with_capacity(len);
        let mut rounding = Vec::with_capacity(len);

        for i in 0..len {
            let j = bias_len.map_or(i, |width| i % width);
            let ideal_value =
                left.ideal[i]
                    .checked_add(right.ideal[j])
                    .ok_or(NumericError::InvalidInput {
                        node: node_index,
                        reason: "ideal fixed-point addition overflow",
                    })?;
            let pre_round =
                left.actual[i]
                    .checked_add(right.actual[j])
                    .ok_or(NumericError::InvalidInput {
                        node: node_index,
                        reason: "rounded fixed-point addition overflow",
                    })?;
            let (rounded, _bits, shift) = round_ticks_to_binary32(pre_round, self.fraction_bits)
                .map_err(|reason| NumericError::InvalidInput {
                    node: node_index,
                    reason,
                })?;
            let step = 1i64.checked_shl(shift).ok_or(NumericError::InvalidInput {
                node: node_index,
                reason: "binary32 ulp step overflow",
            })?;
            let budget = left.epsilon[i]
                .checked_add(right.epsilon[j])
                .and_then(|v| v.checked_add(step))
                .ok_or(NumericError::InvalidInput {
                    node: node_index,
                    reason: "RNE error budget overflow",
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
            rounding,
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
            "-- Concrete dyadic RNE trace: one tick = 2^-{}.\n\
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
                        let right_element = if matches!(node.operation, Operation::BiasAdd) {
                            element % traces[&right_index].actual.len()
                        } else {
                            element
                        };
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
                             simpa [VeriNeural.binary32Step] using\n    \
                             (VeriNeural.approx_add_round\n      \
                             rne_node_{left_index}_elem_{element}_error\n      \
                             rne_node_{right_index}_elem_{right_element}_error\n      \
                             (VeriNeural.binary32_rne_within_step {round_name}))\n\n",
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
        significand
            .checked_shl(shift as u32)
            .ok_or("fixed-point input overflow")?
    } else {
        let right = (-shift) as u32;
        if right >= 64 {
            return Err("input is not on the selected dyadic lattice");
        }
        let mask = if right == 0 { 0 } else { (1u64 << right) - 1 };
        if significand & mask != 0 {
            return Err("input is not on the selected dyadic lattice");
        }
        significand >> right
    };
    if magnitude > i64::MAX as u64 {
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
