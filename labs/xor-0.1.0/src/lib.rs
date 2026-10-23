//! XOR inference laboratory, built solely as a consumer of VeriNeural.
//! Parameters are fixed exactly representable FP32 values, not trained.
//! Lean's nonnegativity theorem is distinct from testing the XOR truth table.

use std::error::Error;
use verineural::{
    ArchitectureMetadata, Domain, Model, ModelBuilder, ModelError, NeuralError, NeuralGraph,
    NodeId, ProofGoal, Shape, SourceRole, Tensor,
};

pub const CASES: [(u8, u8); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// Shared consumer-owned affine mathematics, not duplicated in CUDA or Lean.
fn affine(graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
    let linear = graph.matmul(inputs[0], inputs[1])?;
    graph.bias_add(linear, inputs[2])
}

/// The hidden affine/ReLU and the output affine/ReLU are nested inferred blocks.
/// For Boolean x and y, ReLU(x+y)-2*ReLU(x+y-1) is XOR.
fn xor_body(graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
    let hidden = graph.compose_inferred("xor_activated_affine", "1.0", &inputs[..3], |g, p| {
        let z = g.compose_inferred("xor_affine", "1.0", p, affine)?;
        g.relu(z)
    })?;
    let logits =
        graph.compose_inferred("xor_affine", "1.0", &[hidden, inputs[3], inputs[4]], affine)?;
    graph.relu(logits)
}

/// The lab declares its math, sources, module composition and obligations here.
/// The Semantic IR, GPU compiler and Lean checker remain library-owned.
pub fn architecture() -> Result<Model, ModelError> {
    let mut m = ModelBuilder::new("xor-mlp", "0.1.0");
    let x = m.input("x", [4, 2], Domain::NonNegativeFinite)?;
    let w1 = m.parameter("w1", [2, 2], Domain::Finite)?;
    let b1 = m.parameter("b1", [2], Domain::Finite)?;
    let w2 = m.parameter("w2", [2, 1], Domain::Finite)?;
    let b2 = m.parameter("b2", [1], Domain::Finite)?;
    let y = m.block("xor_mlp", &[x, w1, b1, w2, b2], xor_body)?;
    m.output("y", y)?;
    // A named claim is a Lean obligation, not a statement of proven correctness.
    m.proof_goal(ProofGoal::nonnegative("xor_output_nonnegative", y, vec![]))?;
    m.finish()
}

pub struct Lab {
    pub model: Model,
    /// Keep source id order so existing input/parameter evidence stays stable.
    pub inputs: Vec<(NodeId, Tensor<f32>)>,
}

pub fn build() -> Result<Lab, Box<dyn Error>> {
    let model = architecture()?;
    let samples = CASES
        .iter()
        .flat_map(|&(x, y)| [f32::from(x), f32::from(y)])
        .collect::<Vec<_>>();
    let mut inputs = model.bind([
        ("x", Tensor::new(Shape::new(vec![4, 2])?, samples)?),
        (
            "w1",
            Tensor::new(Shape::new(vec![2, 2])?, vec![1.0, 1.0, 1.0, 1.0])?,
        ),
        ("b1", Tensor::new(Shape::new(vec![2])?, vec![0.0, -1.0])?),
        ("w2", Tensor::new(Shape::new(vec![2, 1])?, vec![1.0, -2.0])?),
        ("b2", Tensor::new(Shape::new(vec![1])?, vec![0.0])?),
    ])?;
    inputs.sort_by_key(|(id, _)| id.index());
    Ok(Lab { model, inputs })
}

impl Lab {
    pub fn graph(&self) -> &NeuralGraph {
        self.model.graph()
    }

    pub fn output(&self) -> NodeId {
        self.model.output("y").expect("architecture declares y")
    }

    pub fn metadata(&self) -> ArchitectureMetadata {
        self.model.metadata("architecture_global_nonnegative")
    }

    pub fn expected() -> Vec<f32> {
        CASES.iter().map(|&(x, y)| f32::from(x ^ y)).collect()
    }

    /// Recheck data even if a consumer has mutated bindings since construction.
    pub fn validate_inputs(&self) -> Result<(), NeuralError> {
        for (id, tensor) in &self.inputs {
            let node = self.graph().node(*id)?;
            if tensor.shape() != &node.shape {
                return Err(NeuralError::ShapeMismatch {
                    expected: node.shape.clone(),
                    actual: tensor.shape().clone(),
                });
            }
            tensor.validate_domain(node.domain)?;
        }
        Ok(())
    }

    fn bits_by_role(&self, role: SourceRole) -> String {
        let roles = self.metadata().input_roles;
        self.inputs
            .iter()
            .filter(|(id, _)| {
                roles
                    .get(&id.index())
                    .is_some_and(|value| value == role.as_str())
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
        self.bits_by_role(SourceRole::Input)
    }

    pub fn parameter_bits(&self) -> String {
        self.bits_by_role(SourceRole::Parameter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verineural::{
        ArchitectureAutoCompiler, GpuEngine, GpuRequirement, LeanProofEngine, NeuralError,
    };

    #[test]
    fn xor_truth_table_and_named_fixed_parameters() {
        let lab = build().unwrap();
        lab.validate_inputs().unwrap();
        assert_eq!(Lab::expected(), [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(
            lab.inputs[0].1.values(),
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(lab.graph().modules().len(), 4);
        assert_eq!(lab.graph().nodes().len(), 11);
        assert!(lab.parameter_bits().contains("c0000000")); // -2.0
    }

    #[test]
    fn model_plans_named_goal_and_generates_global_proof_gpu_plan() {
        let lab = build().unwrap();
        let compiler =
            ArchitectureAutoCompiler::new(LeanProofEngine::default(), GpuEngine::default());
        let plan = lab
            .model
            .plan(&compiler, GpuRequirement::AllOutputs)
            .unwrap();
        assert_eq!(plan.outputs, vec![lab.output()]);
        assert_eq!(plan.goals.len(), 1);
        assert_eq!(plan.goals[0].0, "xor_output_nonnegative");
        assert!(
            plan.goals[0]
                .1
                .source()
                .contains("theorem architecture_property_xor_output_nonnegative")
        );

        let gpu = GpuEngine::default()
            .compile(lab.graph(), lab.output())
            .unwrap();
        assert!(!gpu.translation_plan().unwrap().kernels.is_empty());

        let snapshot =
            verineural::snapshot_architecture(lab.graph(), lab.output(), &gpu, &lab.metadata())
                .unwrap();
        assert!(
            snapshot
                .dot
                .contains("vn_global_property=\"architecture_global_nonnegative\"")
        );

        let proof = LeanProofEngine::default()
            .generate_global_nonnegative(lab.graph(), lab.output(), &[])
            .unwrap();
        assert!(
            proof
                .source()
                .contains("theorem architecture_global_nonnegative")
        );

        let mut invalid_meta = lab.metadata().clone();
        invalid_meta.global_property = "xor_output_nonnegative".into();
        assert!(
            verineural::snapshot_architecture(lab.graph(), lab.output(), &gpu, &invalid_meta)
                .is_err()
        );
    }

    #[test]
    fn malformed_port_and_nonfinite_weight_fail_closed() {
        let mut model = ModelBuilder::new("wrong_xor", "1");
        let wrong = model.input("x", [4, 3], Domain::Finite).unwrap();
        let w1 = model.parameter("w1", [2, 2], Domain::Finite).unwrap();
        let b1 = model.parameter("b1", [2], Domain::Finite).unwrap();
        let w2 = model.parameter("w2", [2, 1], Domain::Finite).unwrap();
        let b2 = model.parameter("b2", [1], Domain::Finite).unwrap();
        assert!(matches!(
            model.block("xor_mlp", &[wrong, w1, b1, w2, b2], xor_body),
            Err(ModelError::Neural(
                NeuralError::InnerDimensionMismatch { .. }
            ))
        ));
        assert_eq!(model.graph_mut().modules().len(), 0);
        assert_eq!(model.graph_mut().nodes().len(), 5);

        let mut lab = build().unwrap();
        lab.inputs[3].1 =
            Tensor::new(Shape::new(vec![2, 1]).unwrap(), vec![1.0, f32::NAN]).unwrap();
        assert!(lab.validate_inputs().is_err());
    }
}
