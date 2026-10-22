//! Consumer-defined fixed-horizon recurrence with independently checked
//! ideal-Int local claims and a global nonnegative state invariant.
use std::path::Path;

use verineural::{
    ContractClaim, Domain, LeanProofEngine, MultiOutputContract, MultiOutputModule,
    NamedPortContract, NeuralError, NeuralGraph, NodeId, PortContract, Shape,
};

struct Accumulate {
    contract: MultiOutputContract,
}

impl MultiOutputModule for Accumulate {
    fn name(&self) -> &'static str {
        "consumer_accumulate"
    }

    fn contract(&self) -> &MultiOutputContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<Vec<NodeId>, NeuralError> {
        // The f32 domain of additions stays Any: they can overflow. The
        // mathematical ideal-Int property is proved separately by Lean.
        let next = graph.add(inputs[0], inputs[1])?;
        let output = graph.relu(next)?;
        Ok(vec![next, output])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape::new(vec![2])?;
    let any = PortContract::new(shape.clone(), Domain::Any);
    let transition = Accumulate {
        contract: MultiOutputContract {
            inputs: vec![
                NamedPortContract::new("state", any.clone()),
                NamedPortContract::new("input", any.clone()),
            ],
            outputs: vec![
                NamedPortContract {
                    name: "next_state",
                    port: any.clone(),
                    claims: vec![ContractClaim::Nonnegative],
                },
                NamedPortContract::new("output", any),
            ],
        },
    };
    let mut graph = NeuralGraph::default();
    let initial = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let x0 = graph.input(shape.clone(), Domain::NonNegativeFinite);
    let x1 = graph.input(shape, Domain::NonNegativeFinite);
    let scan = graph.scan_fixed(&transition, initial, &[x0, x1])?;
    let proof =
        LeanProofEngine::default().generate_fixed_scan_nonnegative(&graph, &scan, "accumulator")?;
    let output = Path::new("target/verineural/verification/scan_invariant.lean");
    let (checked, audit) = LeanProofEngine::default().verify_certified(&proof, output)?;
    println!(
        "Lean certified {} theorems in {}; bounded horizon {}, conditional source hypotheses {}, reviewed axioms {:?}",
        checked.theorem_names.len(),
        checked.path.display(),
        scan.outputs.len(),
        checked.assumptions.len(),
        audit
            .axiom_dependencies
            .values()
            .flatten()
            .collect::<std::collections::BTreeSet<_>>(),
    );
    println!(
        "Assurance: ideal-Int bounded invariant only; no FP32, GPU or unbounded induction proof"
    );
    Ok(())
}
