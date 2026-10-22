//! Independent Stage 9 research consumer: a coupled, two-level spatial field.
//! This is NOT a prescribed VeriNeural architecture, neural language model
//! or an implementation of training/state updates. It exercises nested module
//! boundaries, parameter sharing, fork/fan-in and a global ideal-Int theorem.

use std::collections::BTreeMap;
use std::error::Error;
use verineural::{
    ArchitectureMetadata, Domain, MathContract, NeuralError, NeuralGraph, NeuralModule, NodeId,
    PortContract, Shape, Tensor,
};

fn field() -> Shape {
    Shape::new(vec![1, 4]).expect("valid field shape")
}
fn channel() -> Shape {
    Shape::new(vec![4]).expect("valid channel shape")
}
fn any(s: Shape) -> PortContract {
    PortContract::new(s, Domain::Any)
}

pub struct BoundaryCell {
    contract: MathContract,
}
impl BoundaryCell {
    pub fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![any(field()), any(field()), any(channel())],
                any(field()),
            ),
        }
    }
}
impl Default for BoundaryCell {
    fn default() -> Self {
        Self::new()
    }
}
impl NeuralModule for BoundaryCell {
    fn name(&self) -> &'static str {
        "boundary_response"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let product = g.mul(inputs[0], inputs[1])?;
        let biased = g.bias_add(product, inputs[2])?;
        g.relu(biased)
    }
}

/// CoupledUnit(a,b,c,p,q) := Relu(Boundary(a,b,p) + Boundary(b,c,q)).
pub struct CoupledUnit {
    contract: MathContract,
}
impl CoupledUnit {
    pub fn new() -> Self {
        Self {
            contract: MathContract::new(
                vec![
                    any(field()),
                    any(field()),
                    any(field()),
                    any(channel()),
                    any(channel()),
                ],
                any(field()),
            ),
        }
    }
}
impl Default for CoupledUnit {
    fn default() -> Self {
        Self::new()
    }
}
impl NeuralModule for CoupledUnit {
    fn name(&self) -> &'static str {
        "coupled_unit"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let left = g.apply_module(&BoundaryCell::new(), &[inputs[0], inputs[1], inputs[3]])?;
        let right = g.apply_module(&BoundaryCell::new(), &[inputs[1], inputs[2], inputs[4]])?;
        let mixed = g.add(left, right)?;
        g.relu(mixed)
    }
}

/// Two overlapping coupled units share the same parameters, but rotate the
/// field variables. The resulting whole-architecture output is a fork/fan-in.
pub struct CoupledField {
    contract: MathContract,
}
impl CoupledField {
    pub fn new() -> Self {
        Self {
            contract: CoupledUnit::new().contract,
        }
    }
}
impl Default for CoupledField {
    fn default() -> Self {
        Self::new()
    }
}
impl NeuralModule for CoupledField {
    fn name(&self) -> &'static str {
        "coupled_field"
    }
    fn contract(&self) -> &MathContract {
        &self.contract
    }
    fn build(&self, g: &mut NeuralGraph, inputs: &[NodeId]) -> Result<NodeId, NeuralError> {
        let left = g.apply_module(&CoupledUnit::new(), inputs)?;
        let right = g.apply_module(
            &CoupledUnit::new(),
            &[inputs[0], inputs[2], inputs[1], inputs[4], inputs[3]],
        )?;
        let total = g.add(left, right)?;
        g.relu(total)
    }
}

/// Concrete parameters and values are tiny nonnegative integers and exactly
/// representable in binary32. All nonnegative facts used by Lean remain
/// explicit caller assumptions, verified for this particular input vector.
pub fn build() -> Result<Lab, Box<dyn Error>> {
    let mut g = NeuralGraph::default();
    let x = g.input(field(), Domain::NonNegativeFinite);
    let y = g.input(field(), Domain::NonNegativeFinite);
    let z = g.input(field(), Domain::NonNegativeFinite);
    let p = g.input(channel(), Domain::NonNegativeFinite);
    let q = g.input(channel(), Domain::NonNegativeFinite);
    let output = g.apply_module(&CoupledField::new(), &[x, y, z, p, q])?;
    let samples = [
        (x, field(), vec![1., 2., 3., 4.]),
        (y, field(), vec![2., 1., 0., 1.]),
        (z, field(), vec![0., 1., 2., 1.]),
        (p, channel(), vec![1., 0., 1., 0.]),
        (q, channel(), vec![0., 2., 0., 1.]),
    ];
    let inputs = samples
        .into_iter()
        .map(|(id, s, v)| Ok((id, Tensor::new(s, v)?)))
        .collect::<Result<Vec<_>, NeuralError>>()?;
    let metadata = ArchitectureMetadata {
        id: "coupled-field".into(),
        version: "0.1.0".into(),
        input_roles: BTreeMap::from([
            (x.index(), "input".into()),
            (y.index(), "input".into()),
            (z.index(), "input".into()),
            (p.index(), "parameter".into()),
            (q.index(), "parameter".into()),
        ]),
        global_property: "architecture_global_nonnegative".into(),
    };
    Ok(Lab {
        graph: g,
        output,
        source_nodes: [x, y, z, p, q],
        inputs,
        metadata,
    })
}

pub struct Lab {
    pub graph: NeuralGraph,
    pub output: NodeId,
    pub source_nodes: [NodeId; 5],
    pub inputs: Vec<(NodeId, Tensor<f32>)>,
    pub metadata: ArchitectureMetadata,
}

impl Lab {
    pub fn inputs_bits(&self) -> String {
        self.inputs
            .iter()
            .map(|(id, t)| {
                format!(
                    "node={} role={} shape={:?} bits={}\n",
                    id.index(),
                    self.metadata
                        .input_roles
                        .get(&id.index())
                        .expect("all source roles declared"),
                    t.shape().dims(),
                    t.values()
                        .iter()
                        .map(|x| format!("{:08x}", x.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect()
    }
    pub fn parameters_bits(&self) -> String {
        self.inputs
            .iter()
            .filter(|(id, _)| {
                self.metadata
                    .input_roles
                    .get(&id.index())
                    .is_some_and(|s| s == "parameter")
            })
            .map(|(id, t)| {
                format!(
                    "node={} bits={}\n",
                    id.index(),
                    t.values()
                        .iter()
                        .map(|x| format!("{:08x}", x.to_bits()))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect()
    }
    /// Independent Int forward pass over all live nodes, with checked +/-2^24
    /// integer-exact intermediate bounds. Not host f32 math.
    pub fn exact_integer_output(&self) -> Result<Vec<i64>, Box<dyn Error>> {
        use verineural::Operation;
        let mut values: Vec<Vec<i64>> = Vec::new();
        for (idx, node) in self.graph.nodes().iter().enumerate() {
            let result = match &node.operation {
                Operation::Input => {
                    let t = &self
                        .inputs
                        .iter()
                        .find(|(id, _)| id.index() == idx)
                        .ok_or("source input missing")?
                        .1;
                    t.values()
                        .iter()
                        .map(|&v| {
                            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > 16_777_216.0 {
                                Err("nonexact or negative source input")
                            } else {
                                Ok(v as i64)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                Operation::Add => {
                    let a = &values[node.inputs[0].index()];
                    let b = &values[node.inputs[1].index()];
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| x.checked_add(*y).ok_or("i64 add overflow"))
                        .collect::<Result<Vec<_>, _>>()?
                }
                Operation::Mul => {
                    let a = &values[node.inputs[0].index()];
                    let b = &values[node.inputs[1].index()];
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| x.checked_mul(*y).ok_or("i64 mul overflow"))
                        .collect::<Result<Vec<_>, _>>()?
                }
                Operation::BiasAdd => {
                    let a = &values[node.inputs[0].index()];
                    let b = &values[node.inputs[1].index()];
                    a.iter()
                        .enumerate()
                        .map(|(i, x)| x.checked_add(b[i % b.len()]).ok_or("i64 bias add overflow"))
                        .collect::<Result<Vec<_>, _>>()?
                }
                Operation::Relu => values[node.inputs[0].index()]
                    .iter()
                    .map(|v| (*v).max(0))
                    .collect(),
                Operation::Reshape(_) => values[node.inputs[0].index()].clone(),
                _ => return Err("unsupported lab integer operation".into()),
            };
            if result.iter().any(|&x| x.unsigned_abs() > 16_777_216) {
                return Err("lab integer operation outside exact FP32 profile".into());
            }
            values.push(result);
        }
        let output = values[self.output.index()].clone();
        if output != [4, 10, 8, 12] {
            return Err("lab oracle regression".into());
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verineural::{GpuEngine, LeanProofEngine};

    #[test]
    fn nested_architecture_has_independent_global_property_and_exact_oracle() {
        let lab = build().unwrap();
        assert_eq!(lab.graph.modules().len(), 7);
        assert_eq!(lab.exact_integer_output().unwrap(), [4, 10, 8, 12]);
        let proof = LeanProofEngine::default()
            .generate_global_nonnegative(&lab.graph, lab.output, &lab.source_nodes)
            .unwrap();
        assert!(
            proof
                .source()
                .contains("theorem architecture_global_nonnegative")
        );
        // Each of the five input lemmas must expose exactly one used
        // hypothesis, with the remaining positional binders explicitly unused.
        for source in 0..5 {
            let marker = format!("theorem node_{source}_nonnegative ");
            let declaration = proof
                .source()
                .split(&marker)
                .nth(1)
                .expect("source nonnegativity theorem")
                .split(" := by")
                .next()
                .expect("theorem declaration");
            for hypothesis in 0..5 {
                let binder = if source == hypothesis {
                    format!("(h_{hypothesis} :")
                } else {
                    format!("(_h_{hypothesis} :")
                };
                assert!(
                    declaration.contains(&binder),
                    "node {source} binder {hypothesis}"
                );
            }
        }
        assert!(
            !proof
                .source()
                .contains("set_option linter.unusedVariables false")
        );
        let gpu = GpuEngine::default()
            .compile(&lab.graph, lab.output)
            .unwrap();
        assert!(!gpu.translation_plan().unwrap().kernels.is_empty());
        let snapshot =
            verineural::snapshot_architecture(&lab.graph, lab.output, &gpu, &lab.metadata).unwrap();
        assert_eq!(snapshot.dot.matches("subgraph cluster_m").count(), 7);
        assert!(snapshot.dot.contains(r#"vn_role="parameter""#));
        assert!(snapshot.dot.contains("vn_module_inputs="));
        // 7 composed instances retain their input-port provenance without
        // the 35+ redundant dashed arrows that overwhelmed the old diagram.
        let real_edges: usize = lab.graph.nodes().iter().map(|node| node.inputs.len()).sum();
        assert_eq!(snapshot.dot.matches(" -> ").count(), real_edges);
        assert!(!snapshot.dot.contains("module-input-port"));
        assert!(snapshot.dot.contains(r#"vn_input_port_0="n0""#));
        assert!(snapshot.dot.contains("rank=same; n0; n1; n2; n3; n4;"));
        assert!(snapshot.dot.contains(r#"fillcolor="#FFF1DB""#));
        assert!(snapshot.dot.contains(r#"style="rounded,filled""#));
        assert!(snapshot.dot.contains("dpi=180"));
        assert!(snapshot.dot.contains("vn_ir_sha256="));
    }

    #[test]
    fn changed_parameter_must_revalidate_its_contract() {
        let mut lab = build().unwrap();
        let id = lab.inputs[3].0;
        lab.inputs[3].1 = Tensor::new(channel(), vec![-1., 0., 1., 0.]).unwrap();
        assert!(
            lab.inputs[3]
                .1
                .validate_domain(lab.graph.node(id).unwrap().domain)
                .is_err()
        );
        assert!(lab.exact_integer_output().is_err());
    }
}
