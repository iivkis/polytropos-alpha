//! One consumer-owned symbolic MatMul module instantiated at different shapes.
//! Lean proves an ideal-Int output claim, LLVM NVPTX generates two PTX programs.
//! No IEEE-754 / JIT / GPU machine equivalence is claimed.
use std::path::Path;

use verineural::{
    ContractClaim, Dimension, DimensionConstraint, Domain, GpuEngine, LeanProofEngine,
    NeuralError, NeuralGraph, NodeId, ParametricModule,
    ParametricSignature, Shape, SymbolicPort, SymbolicShape,
};

struct GenericMatMul {
    signature: ParametricSignature,
}

impl ParametricModule for GenericMatMul {
    fn name(&self) -> &'static str {
        "consumer_matmul"
    }
    fn semantic_version(&self) -> &'static str {
        "1.0"
    }
    fn signature(&self) -> &ParametricSignature {
        &self.signature
    }
    fn build(&self, graph: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        graph.matmul(inputs[0], inputs[1])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = |dims| SymbolicPort::new(SymbolicShape::new(dims), Domain::NonNegativeFinite);
    let matmul = GenericMatMul {
        signature: ParametricSignature {
            inputs: vec![
                port(vec![Dimension::Symbol("m"), Dimension::Symbol("k")]),
                port(vec![Dimension::Symbol("k"), Dimension::Symbol("n")]),
            ],
            output: SymbolicPort::new(
                SymbolicShape::new(vec![Dimension::Symbol("m"), Dimension::Symbol("n")]),
                Domain::Any,
            ),
            constraints: vec![
                DimensionConstraint::AtLeast("m", 2),
                DimensionConstraint::AtMost("k", 16),
            ],
            claims: vec![ContractClaim::Nonnegative],
        },
    };
    let mut graph = NeuralGraph::default();
    let mut outputs = Vec::new();
    for (m, k, n) in [(2, 3, 4), (5, 3, 2)] {
        let a = graph.input(Shape::new(vec![m, k])?, Domain::NonNegativeFinite);
        let b = graph.input(Shape::new(vec![k, n])?, Domain::NonNegativeFinite);
        let (output, bindings) = graph.apply_parametric_module(&matmul, &[a, b])?;
        println!("specialized {bindings:?} -> {:?}", graph.node(output)?.shape.dims());
        outputs.push(output);
    }

    let checker = LeanProofEngine::default();
    let proof = checker.generate_outputs(&graph, &outputs)?;
    let root = Path::new("target/verineural/verification/symbolic");
    let (checked, audit) = checker.verify_certified(&proof, &root.join("symbols.lean"))?;
    println!(
        "Lean checked {} obligations with axiom policy {:?}",
        checked.theorem_names.len(), audit.axiom_dependencies.values()
            .flatten().collect::<std::collections::BTreeSet<_>>()
    );

    let gpu = GpuEngine::default();
    for (i, output) in outputs.into_iter().enumerate() {
        let program = gpu.compile(&graph, output)?;
        let ptx = gpu.lower(&program, &root.join(format!("profile_{i}")))?;
        println!("NVPTX translated specialization {i}: {}", ptx.display());
    }
    println!("Assurance: ideal Int under source hypotheses; GPU machine semantics remain trusted");
    Ok(())
}
