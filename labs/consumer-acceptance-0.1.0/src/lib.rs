//! Architecture-neutral acceptance consumers. This crate depends only on the
//! published VeriNeural API; no core implementation or bespoke Lean generator.
use verineural::{
    ArchitectureAutoCompiler, ArchitectureProofTask, AutoBuildError, Domain, FixedScan,
    GpuRequirement, MathContract, MultiOutputContract, MultiOutputModule, NamedPortContract,
    NeuralError, NeuralGraph, NeuralModule, NodeId, PointwiseExpr, PointwiseSpec, PortContract,
    ProofGoal, Shape,
};

fn shape(dims: &[usize]) -> Shape {
    Shape::new(dims.to_vec()).expect("nonzero fixture dimensions")
}

/// A reusable small consumer-owned operation, *not* a core architecture.
struct PositiveJoin {
    contract: MathContract,
}
impl PositiveJoin {
    fn new(width: usize) -> Self {
        let port = PortContract::nonnegative(shape(&[width]));
        Self {
            contract: MathContract::new(
                vec![port.clone(), port],
                PortContract::any(shape(&[width])),
            )
            .nonnegative(),
        }
    }
}
impl NeuralModule for PositiveJoin {
    fn name(&self) -> &'static str {
        "fixture_positive_join"
    }
    fn semantic_version(&self) -> &'static str {
        "1"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let added = graph.add(inputs[0], inputs[1])?;
        graph.relu(added)
    }
}

pub fn nested() -> (NeuralGraph, NodeId, ProofGoal) {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(&[4]), Domain::NonNegativeFinite);
    let y = graph.input(shape(&[4]), Domain::NonNegativeFinite);
    let z = graph.input(shape(&[4]), Domain::NonNegativeFinite);
    let join = PositiveJoin::new(4);
    let left = graph
        .component("fixture/nested/left")
        .single(&join, &[x, y])
        .unwrap();
    let right = graph
        .component("fixture/nested/right")
        .single(&join, &[x, z])
        .unwrap();
    let merged = graph.add(left, right).unwrap();
    let output = graph.relu(merged).unwrap();
    (
        graph,
        output,
        ProofGoal::bounded_above("nested_branch_bound", left, output),
    )
}

/// A consumer-owned, explicitly bounded state transition.
struct Accumulate {
    ports: MultiOutputContract,
}
impl Accumulate {
    fn new(width: usize) -> Self {
        let port = PortContract::any(shape(&[width]));
        Self {
            ports: MultiOutputContract::new(
                vec![
                    NamedPortContract::new("state", port.clone()),
                    NamedPortContract::new("input", port.clone()),
                ],
                vec![
                    NamedPortContract::new("next_state", port.clone()).nonnegative(),
                    NamedPortContract::new("output", port).nonnegative(),
                ],
            ),
        }
    }
}
impl MultiOutputModule for Accumulate {
    fn name(&self) -> &'static str {
        "fixture_accumulate"
    }
    fn semantic_version(&self) -> &'static str {
        "1"
    }
    fn contract(&self) -> &MultiOutputContract {
        &self.ports
    }
    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        let sum = graph.add(inputs[0], inputs[1])?;
        let next = graph.relu(sum)?;
        let output = graph.relu(next)?;
        Ok(vec![next, output])
    }
}

pub fn recurrent() -> (NeuralGraph, FixedScan) {
    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape(&[3]), Domain::NonNegativeFinite);
    let first = graph.input(shape(&[3]), Domain::NonNegativeFinite);
    let second = graph.input(shape(&[3]), Domain::NonNegativeFinite);
    let step = Accumulate::new(3);
    let scan = graph
        .component("fixture/recurrent/steps")
        .scan(&step, initial, &[first, second])
        .unwrap();
    (graph, scan)
}

/// Matrix mixing, custom pointwise semantics, positive mask, selection, axis
/// reduction and final ReLU. It is *not* a softmax or a ready-made Attention
/// layer; GPU lowering of the custom primitive and mask must remain unsupported.
pub fn tensor_heavy() -> (NeuralGraph, NodeId, ProofGoal) {
    let mut graph = NeuralGraph::default();
    let matrix = graph.input(shape(&[2, 2]), Domain::NonNegativeFinite);
    let values = graph.input(shape(&[2, 2]), Domain::NonNegativeFinite);
    let gates = graph.input(shape(&[2, 2]), Domain::NonNegativeFinite);
    // The global nonnegativity prover composes verified module obligations:
    // a bare graph output is not an architecture-level module theorem.
    // Keep the input/output domains conservative; this is an ideal-Int claim,
    // not a claim that floating-point reductions can never overflow.
    let contract = MathContract::new(
        vec![PortContract::any(shape(&[2, 2])); 3],
        PortContract::any(shape(&[2])),
    )
    .nonnegative();
    let output = graph
        .compose(
            "fixture_tensor_heavy",
            "1",
            &contract,
            &[matrix, values, gates],
            |graph, inputs| {
                let product = graph.matmul(inputs[0], inputs[1])?;
                let spec = PointwiseSpec::new(
                    "fixture.consumer.relu_scores",
                    "1.0.0",
                    1,
                    PointwiseExpr::input(0).relu(),
                )?;
                let custom = spec.apply(graph, &[product])?;
                let mask = graph.positive_mask(inputs[2])?;
                let gated = graph.select_positive(mask, custom, product)?;
                let reduced = graph.reduce_sum(gated, 1)?;
                graph.relu(reduced)
            },
        )
        .unwrap();
    let goal = ProofGoal::nonnegative("tensor_heavy_output", output, vec![matrix, values, gates]);
    (graph, output, goal)
}

/// Attention-like tensor flow assembled only from general primitives. The
/// packed source is statically sliced into Q/K/V-shaped regions; there is no
/// Attention/Transformer component in VeriNeural. Static SliceAxis has ideal-Int
/// proof support but intentionally no GPU lowering in this P1 slice.
pub fn attention_like_slices() -> (NeuralGraph, NodeId, ProofGoal) {
    let mut graph = NeuralGraph::default();
    let packed = graph.input(shape(&[2, 6]), Domain::NonNegativeFinite);
    let contract = MathContract::new(
        vec![PortContract::nonnegative(shape(&[2, 6]))],
        PortContract::any(shape(&[2, 2])),
    )
    .nonnegative();
    let output = graph
        .compose(
            "fixture_attention_like_slices",
            "1",
            &contract,
            &[packed],
            |graph, inputs| {
                let q = graph.slice_axis(inputs[0], 1, 0, 2)?;
                let k = graph.slice_axis(inputs[0], 1, 2, 2)?;
                let v = graph.slice_axis(inputs[0], 1, 4, 2)?;
                let kt = graph.permute(k, &[1, 0])?;
                let scores = graph.matmul(q, kt)?;
                let weights = graph.relu(scores)?;
                let context = graph.matmul(weights, v)?;
                graph.relu(context)
            },
        )
        .unwrap();
    let goal = ProofGoal::nonnegative("attention_like_output", output, vec![packed]);
    (graph, output, goal)
}

/// Build plans are inspectable evidence statements, not GPU or Lean proofs.
pub fn plan_nested() -> Result<(usize, Vec<String>), AutoBuildError> {
    let (graph, output, goal) = nested();
    let plan = ArchitectureAutoCompiler::default().plan(
        &graph,
        &[output],
        &[ArchitectureProofTask::Goal(goal)],
        GpuRequirement::AllOutputs,
    )?;
    Ok((
        plan.connectivity.module_instances,
        plan.goals[0].1.theorem_names().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use verineural::{BindingError, GpuError, LeanProofEngine, Operation, PlannedGpu, ProofError};

    #[test]
    fn nested_modules_have_independent_global_bound_and_gpu_plan() {
        let (graph, output, goal) = nested();
        let (other, _, _) = nested();
        assert_eq!(graph.canonical_ir(), other.canonical_ir());
        assert_eq!(graph.modules().len(), 2);
        assert!(graph.canonical_ir().contains("fixture_positive_join"));
        let plan = ArchitectureAutoCompiler::default()
            .plan(
                &graph,
                &[output],
                &[ArchitectureProofTask::Goal(goal.clone())],
                GpuRequirement::AllOutputs,
            )
            .unwrap();
        assert!(matches!(plan.gpu[0], PlannedGpu::Supported { .. }));
        assert!(
            plan.goals[0]
                .1
                .theorem_names()
                .contains(&"architecture_property_nested_branch_bound".to_owned())
        );
        assert!(
            plan.goals[0]
                .1
                .source()
                .contains("pointwise_le_relu_add_of_nonnegative")
        );
        assert_eq!(plan.connectivity.module_instances, 2);
    }

    #[test]
    fn binder_reports_path_port_constraint_and_preserves_graph() {
        let mut graph = NeuralGraph::default();
        let x = graph.input(shape(&[4]), Domain::NonNegativeFinite);
        let wrong = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let before = graph.nodes().len();
        let error = graph
            .component("example/left")
            .single(&PositiveJoin::new(4), &[x, wrong])
            .unwrap_err();
        assert_eq!(error.component_path, "example/left");
        assert_eq!(error.port, "input_1");
        assert_eq!(error.constraint, "input shape");
        assert!(error.to_string().contains("shape mismatch"));
        assert_eq!(graph.nodes().len(), before);
        let weak_domain = graph.input(shape(&[4]), Domain::Any);
        let error = graph
            .component("example/parameters")
            .single(&PositiveJoin::new(4), &[x, weak_domain])
            .unwrap_err();
        assert_eq!(error.port, "input_1");
        assert_eq!(error.constraint, "input value domain");
        assert!(error.to_string().contains("NonNegativeFinite"));
    }

    #[test]
    fn recurrent_named_ports_and_fixed_horizon_goal() {
        let (graph, scan) = recurrent();
        let (other, _) = recurrent();
        assert_eq!(graph.canonical_ir(), other.canonical_ir());
        let tasks = [ArchitectureProofTask::FixedScanNonnegative {
            name: "two_steps".into(),
            scan: scan.clone(),
        }];
        let plan = ArchitectureAutoCompiler::default()
            .plan(
                &graph,
                &scan.proof_outputs(),
                &tasks,
                GpuRequirement::AllOutputs,
            )
            .unwrap();
        assert_eq!(plan.connectivity.module_instances, 2);
        assert_eq!(plan.goals.len(), 1);
        assert!(
            plan.goals[0]
                .1
                .theorem_names()
                .iter()
                .any(|name| name.contains("two_steps"))
        );
        assert!(
            plan.gpu
                .iter()
                .all(|part| matches!(part, PlannedGpu::Supported { .. }))
        );
    }

    #[test]
    fn bad_recurrent_binding_and_absent_claim_are_rejected() {
        let mut graph = NeuralGraph::default();
        let initial = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let wrong = graph.input(shape(&[2]), Domain::NonNegativeFinite);
        let step = Accumulate::new(3);
        let before = graph.nodes().len();
        let error = graph
            .component("example/scan")
            .scan(&step, initial, &[wrong])
            .unwrap_err();
        assert_eq!(error.port, "input[0]");
        assert_eq!(error.component_path, "example/scan");
        assert_eq!(graph.nodes().len(), before);

        // A state transition without an explicit invariant claim may run
        // structurally, but it cannot earn the scan theorem.
        let mut weak = Accumulate::new(3);
        weak.ports.outputs[0].claims.clear();
        let mut graph = NeuralGraph::default();
        let initial = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let event = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let scan = graph
            .component("example/unproved_scan")
            .scan(&weak, initial, &[event])
            .unwrap();
        assert!(matches!(
            LeanProofEngine::default().generate_fixed_scan_nonnegative(
                &graph,
                &scan,
                "missing_invariant"
            ),
            Err(ProofError::InvalidScan)
        ));
    }

    #[test]
    fn multi_bindings_are_named_and_fail_before_mutation() {
        let mut graph = NeuralGraph::default();
        let initial = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let event = graph.input(shape(&[3]), Domain::NonNegativeFinite);
        let step = Accumulate::new(3);
        let outputs = graph
            .component("example/step")
            .multi_named(&step, &[("input", event), ("state", initial)])
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains_key("next_state"));
        assert!(outputs.contains_key("output"));
        assert_eq!(graph.multi_modules().len(), 1);
        let before = graph.nodes().len();
        let err: BindingError = graph
            .component("example/bad_step")
            .multi(&step, &[("input", event), ("input", initial)])
            .unwrap_err();
        assert_eq!(err.port, "state"); // required state has not been bound
        assert_eq!(graph.nodes().len(), before);
    }

    #[test]
    fn custom_primitive_and_tensor_heavy_backend_fail_closed() {
        let (graph, output, goal) = tensor_heavy();
        let (same, _, _) = tensor_heavy();
        assert_eq!(graph.canonical_ir(), same.canonical_ir());
        assert_eq!(graph.modules().len(), 1);
        assert!(
            graph
                .nodes()
                .iter()
                .any(|node| matches!(node.operation, Operation::Pointwise(_)))
        );
        let tasks = [ArchitectureProofTask::Goal(goal)];
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &tasks, GpuRequirement::AllowUnsupported)
            .unwrap();
        assert!(
            plan.goals[0]
                .1
                .theorem_names()
                .contains(&"architecture_property_tensor_heavy_output".to_owned())
        );
        // Nested custom pointwise ReLU consumes a MatMul application. The
        // normalized global proof must not treat named (m/k/n) arguments
        // as arguments to ReLU.
        assert!(plan.goals[0]
            .1
            .source()
            .contains("VeriNeural.relu_nonnegative (VeriNeural.matmul"));
        assert!(!plan.goals[0]
            .1
            .source()
            .contains("VeriNeural.relu_nonnegative VeriNeural.matmul"));
        assert!(matches!(plan.gpu[0], PlannedGpu::Unsupported { .. }));
        assert!(matches!(
            ArchitectureAutoCompiler::default().plan(
                &graph,
                &[output],
                &tasks,
                GpuRequirement::AllOutputs
            ),
            Err(AutoBuildError::Gpu(GpuError::UnsupportedOperation { .. }))
        ));
    }

    #[test]
    fn attention_like_dataflow_uses_static_slices_and_fails_closed_on_gpu() {
        let (graph, output, goal) = attention_like_slices();
        let ir = graph.canonical_ir();
        assert!(ir.contains("slice_axis:1:start=0:len=2"));
        assert!(ir.contains("slice_axis:1:start=2:len=2"));
        assert!(ir.contains("slice_axis:1:start=4:len=2"));
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.operation, Operation::SliceAxis { .. }))
                .count(),
            3
        );

        let tasks = [ArchitectureProofTask::Goal(goal)];
        let plan = ArchitectureAutoCompiler::default()
            .plan(&graph, &[output], &tasks, GpuRequirement::AllowUnsupported)
            .unwrap();
        assert!(
            plan.goals[0]
                .1
                .source()
                .contains("VeriNeural.sliceAxis_nonnegative")
        );
        assert!(matches!(plan.gpu[0], PlannedGpu::Unsupported { .. }));
        assert!(matches!(
            ArchitectureAutoCompiler::default().plan(
                &graph,
                &[output],
                &tasks,
                GpuRequirement::AllOutputs
            ),
            Err(AutoBuildError::Gpu(GpuError::UnsupportedOperation {
                operation: "slice_axis",
                ..
            }))
        ));
    }

    #[test]
    fn invalid_consumer_claim_cannot_be_called_a_proof() {
        let mut graph = NeuralGraph::default();
        let x = graph.input(shape(&[2]), Domain::Finite);
        let spec = PointwiseSpec::new(
            "fixture.consumer.unsafe_claim",
            "1",
            1,
            PointwiseExpr::input(0).add(PointwiseExpr::input(0)),
        )
        .unwrap();
        let output = spec.apply(&mut graph, &[x]).unwrap();
        let false_goal = ProofGoal::nonnegative("unjustified", output, vec![]);
        let error = false_goal
            .generate_source(&graph, &LeanProofEngine::default())
            .unwrap_err();
        assert_eq!(error.theorem, "architecture_property_unjustified");
        assert!(error.to_string().contains("unjustified"));
    }

    /// Slow, explicit kernel acceptance. This does not certify FP32 or SASS.
    #[test]
    #[ignore = "requires pinned Lean 4 from the Nix devShell"]
    fn lean_accepts_three_consumer_families_and_rejects_false_goal() {
        use std::fs;
        let root = std::env::temp_dir().join(format!("verineural-p2-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let lean = LeanProofEngine::default();
        let (nested_graph, _, nested_goal) = nested();
        let (_, nested_audit) = nested_goal
            .certify(&nested_graph, &lean, &root.join("nested.lean"))
            .unwrap();
        assert!(!nested_audit.theorem_names.is_empty());

        let (recurrent_graph, scan) = recurrent();
        let recurrent_goal = lean
            .generate_fixed_scan_nonnegative(&recurrent_graph, &scan, "two_steps")
            .unwrap();
        lean.verify_certified(&recurrent_goal, &root.join("recurrent.lean"))
            .unwrap();

        let (tensor_graph, _, tensor_goal) = tensor_heavy();
        let (_, tensor_audit) = tensor_goal
            .certify(&tensor_graph, &lean, &root.join("tensor_heavy.lean"))
            .unwrap();
        assert!(!tensor_audit.theorem_names.is_empty());

        let (attention_graph, _, attention_goal) = attention_like_slices();
        let (_, attention_audit) = attention_goal
            .certify(&attention_graph, &lean, &root.join("attention_like.lean"))
            .unwrap();
        assert!(!attention_audit.theorem_names.is_empty());

        // Independent property of the consumer-defined atomic operation,
        // not just of an unrelated final ReLU node.
        let mut custom_graph = NeuralGraph::default();
        let x = custom_graph.input(shape(&[2]), Domain::Finite);
        let spec = PointwiseSpec::new(
            "fixture.custom_relu",
            "1",
            1,
            PointwiseExpr::input(0).relu(),
        )
        .unwrap();
        let contract = MathContract::new(
            vec![PortContract::finite(shape(&[2]))],
            PortContract::any(shape(&[2])),
        )
        .nonnegative();
        let custom_output = custom_graph
            .compose(
                "fixture_atomic_contract",
                "1",
                &contract,
                &[x],
                |g, args| spec.apply(g, args),
            )
            .unwrap();
        let program = lean.generate(&custom_graph, custom_output).unwrap();
        assert!(
            program
                .theorem_names()
                .iter()
                .any(|name| name.contains("claim"))
        );
        lean.verify_certified(&program, &root.join("custom_operation.lean"))
            .unwrap();

        // Mathematical counterexample: final ReLU can fall below the left
        // operand when the independent right branch is negative.
        let mut false_graph = NeuralGraph::default();
        let a = false_graph.input(shape(&[2]), Domain::Finite);
        let b = false_graph.input(shape(&[2]), Domain::Finite);
        let positive_a = false_graph.relu(a).unwrap();
        let total = false_graph.add(positive_a, b).unwrap();
        let final_output = false_graph.relu(total).unwrap();
        let claim = ProofGoal::bounded_above("false_bound", positive_a, final_output);
        assert!(
            claim
                .certify(&false_graph, &lean, &root.join("false.lean"))
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
