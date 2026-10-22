//! XOR inference laboratory, built solely as a consumer of VeriNeural.
//! Parameters are fixed exactly representable FP32 values, not trained.
//! Lean's nonnegativity theorem is distinct from testing the XOR truth table.

use std::collections::BTreeMap;
use std::error::Error;
use verineural::{
    ArchitectureMetadata, ContractClaim, Domain, MathContract, NeuralError, NeuralGraph,
    NeuralModule, NodeId, PortContract, Shape, Tensor,
};

pub const CASES: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

fn port(dims: &[usize]) -> PortContract {
    PortContract::new(Shape::new(dims.to_vec()).expect("fixed positive dimensions"), Domain::Any)
}

/// Generic consumer-owned affine block: MatMul(input, weights) + bias.
struct Affine {
    contract: MathContract,
}

impl Affine {
    fn new(rows: usize, input: usize, output: usize) -> Self {
        Self {
            contract: MathContract::new(
                vec![port(&[rows, input]), port(&[input, output]), port(&[output])],
                port(&[rows, output]),
            ),
        }
    }
}

impl NeuralModule for Affine {
    fn name(&self) -> &'static str {
        "xor_affine"
    }

    fn semantic_version(&self) -> &'static str {
        "1.0"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let linear = g.matmul(inputs[0], inputs[1])?;
        g.bias_add(linear, inputs[2])
    }
}

/// Reusable activation layer, composed from a nested affine block and ReLU.
struct ActivatedAffine {
    contract: MathContract,
}

impl ActivatedAffine {
    fn new(rows: usize, input: usize, output: usize) -> Self {
        let contract = Affine::new(rows, input, output)
            .contract
            .with_claim(ContractClaim::Nonnegative);
        Self { contract }
    }
}

impl NeuralModule for ActivatedAffine {
    fn name(&self) -> &'static str {
        "xor_activated_affine"
    }

    fn semantic_version(&self) -> &'static str {
        "1.0"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let affine = g.apply_module(&Affine::new(4, 2, 2), inputs)?;
        g.relu(affine)
    }
}

/// A consumer-owned 2 -> 2 -> 1 MLP for a batch of four Boolean input rows.
///
/// Hidden weights = [[1,1], [1,1]], biases = [0,-1].
/// Output weights = [[1],[-2]], bias = [0].
/// ReLU(s) - 2*ReLU(s-1) equals XOR on s in {0,1,2}.
pub struct XorMlp {
    contract: MathContract,
}

impl Default for XorMlp {
    fn default() -> Self {
        Self::new()
    }
}

impl XorMlp {
    pub fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![
                    port(&[4, 2]),
                    port(&[2, 2]),
                    port(&[2]),
                    port(&[2, 1]),
                    port(&[1]),
                ],
                port(&[4, 1]),
            )
            .with_claim(ContractClaim::Nonnegative),
        }
    }
}

impl NeuralModule for XorMlp {
    fn name(&self) -> &'static str {
        "xor_mlp"
    }

    fn semantic_version(&self) -> &'static str {
        "0.1.0"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let hidden = g.apply_module(&ActivatedAffine::new(4, 2, 2), &inputs[..3])?;
        let logits = g.apply_module(&Affine::new(4, 2, 1), &[hidden, inputs[3], inputs[4]])?;
        g.relu(logits)
    }
}

pub struct Lab {
    pub graph: NeuralGraph,
    pub output: NodeId,
    pub sources: [NodeId; 5],
    pub inputs: Vec<(NodeId, Tensor<f32>)>,
    pub metadata: ArchitectureMetadata,
}

pub fn build() -> Result<Lab, Box<dyn Error>> {
    let mut graph = NeuralGraph::default();
    let x_shape = Shape::new(vec![4, 2])?;
    let inputs = [
        graph.input(x_shape.clone(), Domain::NonNegativeFinite),
        graph.input(Shape::new(vec![2, 2])?, Domain::Finite),
        graph.input(Shape::new(vec![2])?, Domain::Finite),
        graph.input(Shape::new(vec![2, 1])?, Domain::Finite),
        graph.input(Shape::new(vec![1])?, Domain::Finite),
    ];
    let output = graph.apply_module(&XorMlp::new(), &inputs)?;
    let samples = CASES
        .iter()
        .flat_map(|&(x, y)| [f32::from(x), f32::from(y)])
        .collect::<Vec<_>>();
    let values = [
        Tensor::new(x_shape, samples)?,
        Tensor::new(Shape::new(vec![2, 2])?, vec![1.0, 1.0, 1.0, 1.0])?,
        Tensor::new(Shape::new(vec![2])?, vec![0.0, -1.0])?,
        Tensor::new(Shape::new(vec![2, 1])?, vec![1.0, -2.0])?,
        Tensor::new(Shape::new(vec![1])?, vec![0.0])?,
    ];
    let supplied = inputs.into_iter().zip(values).collect::<Vec<_>>();
    let metadata = ArchitectureMetadata {
        id: "xor-mlp".into(),
        version: "0.1.0".into(),
        input_roles: BTreeMap::from([
            (inputs[0].index(), "input".into()),
            (inputs[1].index(), "parameter".into()),
            (inputs[2].index(), "parameter".into()),
            (inputs[3].index(), "parameter".into()),
            (inputs[4].index(), "parameter".into()),
        ]),
        global_property: "architecture_global_nonnegative".into(),
    };
    Ok(Lab {
        graph,
        output,
        sources: inputs,
        inputs: supplied,
        metadata,
    })
}

impl Lab {
    pub fn expected() -> Vec<f32> {
        CASES.iter().map(|&(x, y)| f32::from(x ^ y)).collect()
    }

    pub fn validate_inputs(&self) -> Result<(), NeuralError> {
        for (id, tensor) in &self.inputs {
            tensor.validate_domain(self.graph.node(*id)?.domain)?;
        }
        Ok(())
    }

    fn bits_by_role(&self, role: &str) -> String {
        self.inputs
            .iter()
            .filter(|(id, _)| {
                self.metadata
                    .input_roles
                    .get(&id.index())
                    .is_some_and(|value| value == role)
            })
            .map(|(id, tensor)| {
                format!(
                    "node={} shape={:?} bits={}\n",
                    id.index(),
                    tensor.shape().dims(),
                    tensor
                        .values()
                        .iter()
                        .map(|v| format!("{:08x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect()
    }

    pub fn input_bits(&self) -> String {
        self.bits_by_role("input")
    }

    pub fn parameter_bits(&self) -> String {
        self.bits_by_role("parameter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verineural::{GpuEngine, LeanProofEngine, NeuralError};

    #[test]
    fn xor_truth_table_and_exact_fixed_parameters() {
        let lab = build().unwrap();
        lab.validate_inputs().unwrap();
        assert_eq!(Lab::expected(), [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(lab.inputs[0].1.values(), &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        assert_eq!(lab.graph.modules().len(), 4);
        assert_eq!(lab.graph.nodes().len(), 11);
        assert!(lab.parameter_bits().contains("c0000000")); // -2.0
    }

    #[test]
    fn supported_graph_generates_one_global_proof_and_gpu_plan() {
        let lab = build().unwrap();
        let gpu = GpuEngine::default()
            .compile(&lab.graph, lab.output)
            .unwrap();
        assert!(!gpu.translation_plan().unwrap().kernels.is_empty());

        let snapshot =
            verineural::snapshot_architecture(&lab.graph, lab.output, &gpu, &lab.metadata)
                .unwrap();
        assert!(snapshot
            .dot
            .contains("vn_global_property=\"architecture_global_nonnegative\""));

        let proof = LeanProofEngine::default()
            .generate_global_nonnegative(&lab.graph, lab.output, &[])
            .unwrap();
        assert!(proof
            .source()
            .contains("theorem architecture_global_nonnegative"));

        let mut invalid_meta = lab.metadata.clone();
        invalid_meta.global_property = "xor_output_nonnegative".into();
        assert!(verineural::snapshot_architecture(
            &lab.graph,
            lab.output,
            &gpu,
            &invalid_meta
        )
        .is_err());
    }

    #[test]
    fn malformed_port_and_nonfinite_weight_fail_closed() {
        let mut graph = NeuralGraph::default();
        let wrong = graph.input(Shape::new(vec![4, 3]).unwrap(), Domain::Finite);
        let w1 = graph.input(Shape::new(vec![2, 2]).unwrap(), Domain::Finite);
        let b1 = graph.input(Shape::new(vec![2]).unwrap(), Domain::Finite);
        let w2 = graph.input(Shape::new(vec![2, 1]).unwrap(), Domain::Finite);
        let b2 = graph.input(Shape::new(vec![1]).unwrap(), Domain::Finite);
        assert!(matches!(
            graph.apply_module(&XorMlp::new(), &[wrong, w1, b1, w2, b2]),
            Err(NeuralError::ShapeMismatch { .. })
        ));
        assert_eq!(graph.modules().len(), 0);

        let mut lab = build().unwrap();
        lab.inputs[3].1 =
            Tensor::new(Shape::new(vec![2, 1]).unwrap(), vec![1.0, f32::NAN]).unwrap();
        assert!(lab.validate_inputs().is_err());
    }
}
