//! Regression tests for the architecture-wide auto builder.
//! The bundle round-trip exercises the Nix-provided Lean, LLVM and Graphviz.
use verineural::{
    ArchitectureAutoCompiler, ArchitectureProofTask, AutoBuildError, ContractClaim, Domain,
    GpuRequirement, MathContract, MultiOutputContract, MultiOutputModule, NamedPortContract,
    NeuralError, NeuralGraph, NeuralModule, NodeId, PlannedGpu, PortContract, ProofError,
    ProofGoal, ProofGoalKind, Shape,
};

fn shape(n: usize) -> Shape {
    Shape::new(vec![n]).unwrap()
}

struct ConsumerBlock {
    spec: MathContract,
    claimed: bool,
}
impl NeuralModule for ConsumerBlock {
    fn name(&self) -> &'static str {
        "consumer_block"
    }
    fn contract(&self) -> &MathContract {
        &self.spec
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        if self.claimed {
            graph.add(inputs[0], inputs[1])
        } else {
            graph.relu(inputs[0])
        }
    }
}

#[test]
fn auto_build_plans_all_supported_outputs_and_separate_user_goal() {
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(4), Domain::NonNegativeFinite);
    let y = graph.input(shape(4), Domain::NonNegativeFinite);
    let relu = graph.relu(x).unwrap();
    let merged = graph.add(relu, y).unwrap();
    let task = ArchitectureProofTask::Goal(ProofGoal {
        name: "root".into(),
        output: merged,
        kind: ProofGoalKind::Nonnegative {
            assumptions: vec![x, y],
        },
    });
    let plan = ArchitectureAutoCompiler::default()
        .plan(&graph, &[relu, merged], &[task], GpuRequirement::AllOutputs)
        .unwrap_err();
    // A bare Add output does not have a top-level module theorem, so the
    // global composition API must reject the unsupported architecture goal.
    assert!(matches!(
        plan,
        AutoBuildError::Proof(ProofError::UnsupportedGuarantee { .. })
    ));

    let plan = ArchitectureAutoCompiler::default()
        .plan(&graph, &[relu, merged], &[], GpuRequirement::AllOutputs)
        .unwrap();
    assert_eq!(plan.connectivity.data_edges, 3);
    assert_eq!(plan.connectivity.module_instances, 0);
    assert_eq!(plan.gpu.len(), 2);
    assert!(
        plan.gpu
            .iter()
            .all(|x| matches!(x, PlannedGpu::Supported { .. }))
    );
    assert!(
        plan.architecture
            .theorem_names()
            .iter()
            .any(|n| n == "node_3_semantics")
    );
}

#[test]
fn auto_build_tracks_unsupported_outputs_without_mislabeling_gpu() {
    let mut graph = NeuralGraph::default();
    let condition = graph.input(shape(2), Domain::Any);
    let on_true = graph.input(shape(2), Domain::NonNegativeFinite);
    let on_false = graph.input(shape(2), Domain::NonNegativeFinite);
    let relu = graph.relu(on_true).unwrap();
    let gate = graph.positive_mask(condition).unwrap();
    let selected = graph.select_positive(gate, relu, on_false).unwrap();
    let auto = ArchitectureAutoCompiler::default();
    let plan = auto
        .plan(
            &graph,
            &[relu, selected],
            &[],
            GpuRequirement::AllowUnsupported,
        )
        .unwrap();
    assert!(matches!(plan.gpu[0], PlannedGpu::Supported { .. }));
    assert!(matches!(plan.gpu[1], PlannedGpu::Unsupported { .. }));
    assert!(
        plan.architecture
            .source()
            .contains("VeriNeural.selectPositive")
    );
    assert!(matches!(
        auto.plan(&graph, &[relu, selected], &[], GpuRequirement::AllOutputs),
        Err(AutoBuildError::Gpu(
            verineural::GpuError::UnsupportedOperation {
                operation: "select_positive",
                ..
            }
        ))
    ));
}

#[test]
fn auto_build_rejects_false_component_claims_and_invalid_output_handles() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(2), Domain::Any);
    let b = graph.input(shape(2), Domain::Any);
    let port = PortContract::new(shape(2), Domain::Any);
    let faulty = ConsumerBlock {
        spec: MathContract::new(vec![port.clone(), port.clone()], port)
            .with_claim(ContractClaim::Nonnegative),
        claimed: true,
    };
    let out = graph.apply_module(&faulty, &[a, b]).unwrap();
    assert!(matches!(
        ArchitectureAutoCompiler::default().plan(
            &graph,
            &[out],
            &[],
            GpuRequirement::AllowUnsupported,
        ),
        Err(AutoBuildError::Proof(
            ProofError::UnsupportedGuarantee { .. }
        ))
    ));

    let mut foreign_graph = NeuralGraph::default();
    let foreign = foreign_graph.input(shape(2), Domain::Any);
    assert!(matches!(
        ArchitectureAutoCompiler::default().plan(
            &graph,
            &[foreign],
            &[],
            GpuRequirement::AllowUnsupported,
        ),
        Err(AutoBuildError::Proof(ProofError::InvalidGraph(
            NeuralError::InvalidNode
        )))
    ));
}

#[test]
fn auto_build_deduplicates_goals_and_catches_invalid_module_capture() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(2), Domain::Any);
    let b = graph.input(shape(2), Domain::Any);
    let c = graph.relu(a).unwrap();
    let task = || {
        ArchitectureProofTask::Goal(ProofGoal {
            name: "same".into(),
            output: c,
            kind: ProofGoalKind::EqualTo { other: c },
        })
    };
    assert!(matches!(
        ArchitectureAutoCompiler::default().plan(
            &graph, &[c], &[task(), task()], GpuRequirement::AllowUnsupported,
        ),
        Err(AutoBuildError::DuplicateTask(name)) if name == "same"
    ));

    struct Hidden {
        spec: MathContract,
        capture: NodeId,
    }
    impl NeuralModule for Hidden {
        fn name(&self) -> &'static str {
            "hidden_capture"
        }
        fn contract(&self) -> &MathContract {
            &self.spec
        }
        fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
            graph.add(inputs[0], self.capture)
        }
    }
    let port = PortContract::new(shape(2), Domain::Any);
    let module = Hidden {
        spec: MathContract::new(vec![port.clone()], port),
        capture: b,
    };
    let before = graph.nodes().len();
    assert!(matches!(
        graph.apply_module(&module, &[a]),
        Err(NeuralError::ContractViolation {
            detail: "undeclared external capture",
            ..
        })
    ));
    assert_eq!(graph.nodes().len(), before);
}

#[test]
fn auto_build_plans_named_multi_output_state_transition_and_global_scan_theorem() {
    struct Step {
        contract: MultiOutputContract,
    }
    impl MultiOutputModule for Step {
        fn name(&self) -> &'static str {
            "external_step"
        }
        fn contract(&self) -> &MultiOutputContract {
            &self.contract
        }
        fn build(
            &self,
            graph: &mut NeuralGraph,
            inputs: &[NodeId],
        ) -> Result<Vec<NodeId>, NeuralError> {
            let mask = graph.positive_mask(inputs[1])?;
            let next = graph.select_positive(mask, inputs[1], inputs[0])?;
            let readout = graph.relu(next)?;
            Ok(vec![next, readout])
        }
    }

    let port = PortContract::new(shape(3), Domain::NonNegativeFinite);
    let named = |name| NamedPortContract::new(name, port.clone());
    let step = Step {
        contract: MultiOutputContract {
            inputs: vec![named("state"), named("input")],
            outputs: vec![
                NamedPortContract {
                    name: "next_state",
                    port: port.clone(),
                    claims: vec![ContractClaim::Nonnegative],
                },
                named("output"),
            ],
        },
    };
    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape(3), Domain::NonNegativeFinite);
    let first = graph.input(shape(3), Domain::NonNegativeFinite);
    let second = graph.input(shape(3), Domain::NonNegativeFinite);
    let scan = graph.scan_fixed(&step, initial, &[first, second]).unwrap();
    let plan = ArchitectureAutoCompiler::default()
        .plan(
            &graph,
            &scan.proof_outputs(),
            &[ArchitectureProofTask::FixedScanNonnegative {
                name: "consumer_state".into(),
                scan,
            }],
            GpuRequirement::AllowUnsupported,
        )
        .unwrap();
    assert_eq!(plan.connectivity.module_instances, 2);
    assert_eq!(plan.connectivity.module_ports, 8);
    assert_eq!(plan.goals.len(), 1);
    assert!(
        plan.goals[0]
            .1
            .theorem_names()
            .iter()
            .any(|n| { n == "scan_consumer_state_global_nonnegative" })
    );
    assert!(
        plan.gpu
            .iter()
            .all(|item| matches!(item, PlannedGpu::Unsupported { .. }))
    );
}

#[test]
fn auto_build_refuses_successful_but_disconnected_property() {
    let mut graph = NeuralGraph::default();
    let input = graph.input(shape(2), Domain::Any);
    let detached = graph.input(shape(2), Domain::Any);
    let output = graph.relu(input).unwrap();
    let task = ArchitectureProofTask::Goal(ProofGoal {
        name: "irrelevant".into(),
        output: detached,
        kind: ProofGoalKind::EqualTo { other: detached },
    });
    assert!(matches!(
        ArchitectureAutoCompiler::default().plan(
            &graph, &[output], &[task], GpuRequirement::AllOutputs,
        ),
        Err(AutoBuildError::InvalidConnection(message))
            if message.contains("not connected")
    ));
}

#[test]
#[ignore = "requires Lean 4, llc, sha256sum and Graphviz dot; run in the full Nix devShell"]
fn auto_build_creates_dot_and_png_before_manifest_and_replays_without_mutation() {
    // The previous auto-build failed after Lean/PTX because SHA256SUMS listed
    // architecture.dot even though only verify_bundle wrote it.
    let mut graph = NeuralGraph::default();
    let x = graph.input(shape(2), Domain::NonNegativeFinite);
    let output = graph.relu(x).unwrap();
    let root = std::env::temp_dir().join(format!(
        "verineural-auto-visual-regression-{}-{:?}",
        std::process::id(),
        std::thread::current().id(),
    ));
    if root.exists() {
        std::fs::remove_dir_all(&root).unwrap();
    }
    let auto = ArchitectureAutoCompiler::default();
    let generated = auto
        .build(&graph, &[output], &[], GpuRequirement::AllOutputs, &root)
        .unwrap();
    let folder = root.join("auto-build");
    let saved_dot = std::fs::read_to_string(folder.join("architecture.dot")).unwrap();
    let png = std::fs::read(folder.join("architecture.png")).unwrap();
    assert!(saved_dot.contains("vn_ir_sha256="));
    assert!(saved_dot.contains("vn_selected_output="));
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let manifest = std::fs::read_to_string(folder.join("SHA256SUMS")).unwrap();
    assert!(manifest.contains("  architecture.dot\n"));
    assert!(manifest.contains("  architecture.png\n"));
    let replayed = auto
        .verify_bundle(
            &graph,
            &[output],
            &[],
            GpuRequirement::AllOutputs,
            &root,
            Some(&generated.manifest_sha256),
        )
        .unwrap();
    assert_eq!(generated.manifest_sha256, replayed.manifest_sha256);
    assert_eq!(
        std::fs::read_to_string(folder.join("architecture.dot")).unwrap(),
        saved_dot
    );
    assert_eq!(std::fs::read(folder.join("architecture.png")).unwrap(), png);

    std::fs::write(folder.join("architecture.png"), b"tampered PNG").unwrap();
    assert!(matches!(
        auto.verify_bundle(
            &graph,
            &[output],
            &[],
            GpuRequirement::AllOutputs,
            &root,
            Some(&generated.manifest_sha256),
        ),
        Err(AutoBuildError::Integrity(_))
    ));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn auto_builder_plans_one_shared_nvptx_module_for_all_supported_outputs() {
    let mut graph = NeuralGraph::default();
    let a = graph.input(shape(4), Domain::Finite);
    let b = graph.input(shape(4), Domain::Finite);
    let common = graph.add(a, b).unwrap();
    let left = graph.relu(common).unwrap();
    let right = graph.mul(common, b).unwrap();
    let final_sum = graph.add(left, right).unwrap();
    let auto = ArchitectureAutoCompiler::default();

    let plan = auto
        .plan(
            &graph,
            &[left, right, final_sum],
            &[],
            GpuRequirement::AllOutputs,
        )
        .unwrap();
    assert_eq!(plan.gpu.len(), 3);
    let shared = plan.shared_gpu.as_ref().unwrap();
    assert_eq!(shared.outputs(), &[left, right, final_sum]);
    assert_eq!(shared.kernel_count().unwrap(), 4);
    // Two Add producers now share one LLVM/PTX entrypoint, but remain
    // separate CUDA Graph nodes with independent buffers and arguments.
    assert_eq!(shared.unique_kernel_count(), 3);
    assert_eq!(
        shared
            .llvm_ir()
            .matches("define ptx_kernel void @vn_node_")
            .count(),
        shared.unique_kernel_count()
    );
    assert_eq!(shared.source_inputs(), &[a, b]);

    let unsupported = graph.positive_mask(a).unwrap();
    let partial = auto
        .plan(
            &graph,
            &[final_sum, unsupported],
            &[],
            GpuRequirement::AllowUnsupported,
        )
        .unwrap();
    assert!(partial.shared_gpu.is_none());
    assert!(matches!(partial.gpu[0], PlannedGpu::Supported { .. }));
    assert!(matches!(partial.gpu[1], PlannedGpu::Unsupported { .. }));
}
