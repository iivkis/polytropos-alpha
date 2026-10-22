//! Consumer-defined gated state update over generic mask and selection IR.
//! Produces a Lean-checked fixed-horizon ideal-Int invariant, while binary32
//! RNE and NVPTX lowering remain explicitly unsupported for these new ops.
use std::path::Path;

use verineural::{
    ContractClaim, Domain, GpuEngine, GpuError, LeanProofEngine, MultiOutputContract,
    MultiOutputModule, NamedPortContract, NeuralError, NeuralGraph, NodeId, PortContract, Shape,
    Tensor,
};

struct GatedUpdate {
    interface: MultiOutputContract,
}

impl MultiOutputModule for GatedUpdate {
    fn name(&self) -> &'static str {
        "consumer_gated_update"
    }
    fn contract(&self) -> &MultiOutputContract {
        &self.interface
    }
    fn semantic_version(&self) -> &'static str {
        "1.0"
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        let mask = graph.positive_mask(inputs[1])?;
        let next_state = graph.select_positive(mask, inputs[1], inputs[0])?;
        let readout = graph.relu(next_state)?;
        Ok(vec![next_state, readout])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![3])?;
    let positive = PortContract::new(shape.clone(), Domain::NonNegativeFinite);
    let named = |name| NamedPortContract::new(name, positive.clone());
    let module = GatedUpdate {
        interface: MultiOutputContract {
            inputs: vec![named("state"), named("input")],
            outputs: vec![
                NamedPortContract {
                    name: "next_state",
                    port: positive.clone(),
                    claims: vec![ContractClaim::Nonnegative],
                },
                named("output"),
            ],
        },
    };

    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let first = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let second = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let scan = graph.scan_fixed(&module, initial, &[first, second])?;
    let checker = LeanProofEngine::default();
    let lean = checker.generate_fixed_scan_nonnegative(&graph, &scan, "gated")?;
    let file = Path::new("target/verineural/verification/gated_select.lean");
    let (checked, audit) = checker.verify_certified(&lean, file)?;

    // An independent concrete host-reference fixture, not an FP32 theorem.
    let mask = Tensor::new(shape.clone(), vec![-1.0f32, 0.0, f32::NAN])?;
    let yes = Tensor::new(shape.clone(), vec![1.0, 2.0, 3.0])?;
    let no = Tensor::new(shape, vec![4.0, 5.0, 6.0])?;
    assert_eq!(mask.positive_mask().values(), &[0.0, 0.0, 0.0]);
    assert_eq!(mask.select_positive(&yes, &no)?.values(), &[4.0, 5.0, 6.0]);

    for output in &scan.outputs {
        assert!(matches!(
            GpuEngine::default().compile(&graph, *output),
            Err(GpuError::UnsupportedOperation { .. })
        ));
    }
    println!(
        "Lean checked {} theorems (H=2); assumptions={}; axioms={:?}",
        checked.theorem_names.len(),
        checked.assumptions.len(),
        audit
            .axiom_dependencies
            .values()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>()
    );
    println!(
        "Assurance: ideal-Int bounded invariant; f32 mask reference checked; RNE/GPU lowering unsupported"
    );
    Ok(())
}
