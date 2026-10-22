//! Stage 8 first vertical slice: exact-integer FP32 domain and compositional
//! range obligations for Add, BiasAdd, Relu and Reshape.
//!
//! All integral inputs and intermediate mathematical values are restricted to
//! magnitude <= 2^24, where every integer has a binary32 representation.
//! Lean establishes *ideal Int* bounds, and the integer oracle validates the
//! concrete GPU output bitwise. Correct IEEE RNE hardware/LLVM behavior is
//! STILL TRUSTED, not proved by this engine. Non-integral or negative-zero
//! inputs and unsupported operators fail closed.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::{
    LeanProofEngine, NeuralError, NeuralGraph, NodeId, Operation, ProofError, Tensor,
};

const EXACT_F32_INTEGER_LIMIT: i64 = 1 << 24;
const SOURCE_INTEGER_LIMIT: i64 = 1 << 20;
const NUMERIC_LIBRARY: &str = include_str!("../../../lean/VeriNeural/Numeric.lean");

#[derive(Debug)]
pub enum NumericError {
    Graph(NeuralError),
    Proof(ProofError),
    UnsupportedOperation { node: usize, operation: &'static str },
    UnsupportedRange { node: usize, bound: i64 },
    InvalidInput { node: usize, reason: &'static str },
    MissingInput { node: usize },
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
                write!(f, "exact-integer FP32 subset rejects {operation} at node {node}")
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
}

impl ExactFp32Plan {
    pub const fn input_bound() -> i64 {
        SOURCE_INTEGER_LIMIT
    }

    pub const fn max_exact_integer() -> i64 {
        EXACT_F32_INTEGER_LIMIT
    }

    /// Build the numerical contract from the same Semantic Neural IR as the
    /// GPU lowering. Unsupported nonlinear/reduction ops fail closed.
    pub fn compile(graph: &NeuralGraph, output: NodeId) -> Result<Self, NumericError> {
        graph.node(output)?;
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
                Operation::Input => SOURCE_INTEGER_LIMIT,
                Operation::Add | Operation::BiasAdd => parent(0) + parent(1),
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
        })
    }

    pub fn output_bound(&self) -> i64 {
        self.bounds[&self.output.index()]
    }

    pub fn source_inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Generate a Lean theorem for every reachable output node and its real
    /// composed input hypotheses. The result is NOT proof until Lean checks it.
    pub fn lean_source(&self) -> Result<String, NumericError> {
        let base = LeanProofEngine::default().generate(&self.graph, self.output)?;
        let mut source = base.source().to_owned();
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
        let hypotheses = self
            .inputs
            .iter()
            .map(|id| {
                format!(
                    "(h_{} : VeriNeural.TensorWithin {SOURCE_INTEGER_LIMIT} x_{})",
                    id.index(),
                    id.index()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let args = self
            .inputs
            .iter()
            .map(|id| format!("x_{}", id.index()))
            .collect::<Vec<_>>()
            .join(" ");
        let conditions = self
            .inputs
            .iter()
            .map(|id| format!("h_{}", id.index()))
            .collect::<Vec<_>>()
            .join(" ");
        let call = |index: usize| format!("(node_{index} {args})");
        let proof = |index: usize| format!("(node_{index}_integer_range {args} {conditions})");

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
                Operation::BiasAdd => format!(
                    "VeriNeural.tensor_within_biasAdd (by decide : 0 < {}) {} {}",
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
                call(index)
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
            if !matches!(node.operation, Operation::Input)
                || !self.inputs.contains(id)
            {
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
                    || value.abs() > SOURCE_INTEGER_LIMIT as f32
                    || value.fract() != 0.0
                    || (value.to_bits() == (-0.0f32).to_bits())
                {
                    return Err(NumericError::InvalidInput {
                        node: id.index(),
                        reason: "requires finite integral f32 values, no -0, |x| <= 2^20",
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
            debug_assert!(
                output.iter().all(|x| x.abs() <= self.bounds[&index])
            );
            values.insert(index, output);
        }
        let out = values.remove(&self.output.index()).expect("reachable output");
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
        for (index, (&target, &actual)) in expected
            .values()
            .iter()
            .zip(gpu.values())
            .enumerate()
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
