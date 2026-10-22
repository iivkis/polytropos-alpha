//! Independent architecture consumer: define reusable local modules, wire a
//! fork/fan-in graph, then verify all selected outputs and a global goal and
//! emit NVPTX without manually invoking either backend for each component.
use std::path::Path;

use verineural::{
    ArchitectureAutoCompiler, ArchitectureProofTask, ContractClaim, Domain, GpuRequirement,
    MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId, PortContract, ProofGoal,
    ProofGoalKind, Shape,
};

struct Activated {
    contract: MathContract,
}
impl NeuralModule for Activated {
    fn name(&self) -> &'static str {
        "consumer_activation"
    }
    fn semantic_version(&self) -> &'static str {
        "1.0"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.relu(inputs[0])
    }
}

struct Merge {
    contract: MathContract,
}
impl NeuralModule for Merge {
    fn name(&self) -> &'static str {
        "consumer_merge"
    }
    fn semantic_version(&self) -> &'static str {
        "1.0"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![4])?;
    let finite = PortContract::new(shape.clone(), Domain::Finite);
    let nonnegative = PortContract::new(shape.clone(), Domain::NonNegativeFinite);
    let any = PortContract::new(shape.clone(), Domain::Any);
    let activated = Activated {
        contract: MathContract::new(vec![finite], nonnegative.clone())
            .with_claim(ContractClaim::Nonnegative),
    };
    let merge = Merge {
        contract: MathContract::new(vec![nonnegative.clone(), nonnegative], any)
            .with_claim(ContractClaim::Nonnegative),
    };

    let mut graph = NeuralGraph::default();
    let signal = graph.input(shape.clone(), Domain::Finite);
    let parameter = graph.input(shape, Domain::NonNegativeFinite);
    let left = graph.apply_module(&activated, &[signal])?;
    let right = graph.apply_module(&activated, &[parameter])?;
    let output = graph.apply_module(&merge, &[left, right])?;

    let tasks = vec![ArchitectureProofTask::Goal(ProofGoal {
        name: "merged_nonnegative".to_owned(),
        output,
        kind: ProofGoalKind::Nonnegative {
            assumptions: Vec::new(),
        },
    })];
    let report = ArchitectureAutoCompiler::default().build(
        &graph,
        &[left, right, output],
        &tasks,
        GpuRequirement::AllOutputs,
        Path::new("target/verineural/auto-architecture"),
    )?;
    println!(
        "Wiring: {} edges, {} bound module ports, {} instances; \
         checked {} architecture theorems and {} independent goal(s)",
        report.connectivity.data_edges,
        report.connectivity.module_ports,
        report.connectivity.module_instances,
        report.architecture_theorems,
        report.goal_theorems.len(),
    );
    for artifact in &report.output_evidence {
        println!("GPU output {}: {:?}", artifact.node, artifact.ptx);
    }
    println!("Report: {}", report.directory.join("report.txt").display());
    println!(
        "Assurance: Lean ideal-Int and structural PTX translation only; GPU machine semantics not proven"
    );
    Ok(())
}
