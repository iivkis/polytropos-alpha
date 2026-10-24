//! Torus16-Mix operator consumer. One shared 48x48 W and eight gated roles.
//! All forward operations are built from the public VeriNeural Semantic IR.
//! The operator is reusable by the EMA4 streaming LM consumer in this lab.
use verineural::{
    Domain, Model, ModelBuilder, ModelError, NeuralError, NeuralGraph, NodeId, Shape, Tensor,
};

pub mod corpus;
pub mod ema4;
pub mod ema4_reference;

pub const WIDTH: usize = 48;
pub const ROLES: usize = 8;
pub const BASES: usize = 2 * ROLES;

#[derive(Clone, Debug)]
pub struct RoleParameters {
    pub row: [f32; WIDTH],
    pub col: [f32; WIDTH],
    pub bias: [f32; WIDTH],
    pub mix_gate: [f32; WIDTH],
    pub mix_bias: f32,
}

impl Default for RoleParameters {
    fn default() -> Self {
        Self {
            row: [1.0; WIDTH],
            col: [1.0; WIDTH],
            bias: [0.0; WIDTH],
            mix_gate: [0.0; WIDTH],
            mix_bias: 3.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Parameters {
    pub weight: [[f32; WIDTH]; WIDTH],
    pub roles: [RoleParameters; ROLES],
}

impl Default for Parameters {
    fn default() -> Self {
        let weight = std::array::from_fn(|i| {
            std::array::from_fn(|j| ((i * 53 + j * 17 + 19) % 101) as f32 * 0.0004 - 0.02)
        });
        Self {
            weight,
            roles: std::array::from_fn(|_| RoleParameters::default()),
        }
    }
}

impl Parameters {
    /// 2304 shared matrix weights + 8*(4*48+1) role weights = 3848.
    pub const fn trainable_count() -> usize {
        WIDTH * WIDTH + ROLES * (4 * WIDTH + 1)
    }

    /// Bind once to the same parameter/source nodes used by the evidence model.
    /// Fixed constants are source values rather than invented runtime operations.
    pub fn bindings(&self, x: &[f32; WIDTH]) -> Result<Vec<(String, Tensor<f32>)>, NeuralError> {
        let mut values = vec![
            ("x".into(), Tensor::from_slice(vec![1, WIDTH], x)?),
            (
                "weight".into(),
                Tensor::from_vec(
                    vec![WIDTH, WIDTH],
                    self.weight
                        .iter()
                        .flat_map(|row| row.iter().copied())
                        .collect(),
                )?,
            ),
            (
                "ones_column".into(),
                Tensor::from_vec(vec![WIDTH, 1], vec![1.0; WIDTH])?,
            ),
            ("one".into(), Tensor::from_slice(vec![1, 1], &[1.0])?),
            ("minus_one".into(), Tensor::from_slice(vec![1, 1], &[-1.0])?),
            (
                "inv_sqrt_width".into(),
                Tensor::from_slice(vec![1, 1], &[1.0 / (WIDTH as f32).sqrt()])?,
            ),
        ];
        for (role, p) in self.roles.iter().enumerate() {
            for (name, data) in [
                ("row", &p.row),
                ("col", &p.col),
                ("bias", &p.bias),
                ("mix_gate", &p.mix_gate),
            ] {
                values.push((
                    format!("{name}_{role}"),
                    Tensor::from_slice(vec![1, WIDTH], data)?,
                ));
            }
            values.push((
                format!("mix_bias_{role}"),
                Tensor::from_slice(vec![1, 1], &[p.mix_bias])?,
            ));
        }
        Ok(values)
    }
}

fn roll_left_graph(g: &mut NeuralGraph, x: NodeId, shift: usize) -> Result<NodeId, NeuralError> {
    let shift = shift % WIDTH;
    if shift == 0 {
        return Ok(x);
    }
    let tail = g.slice_axis(x, 1, shift, WIDTH - shift)?;
    let head = g.slice_axis(x, 1, 0, shift)?;
    g.concat(1, &[tail, head])
}

pub(crate) fn projection_graph(
    g: &mut NeuralGraph,
    inputs: &[NodeId],
    role: usize,
) -> Result<NodeId, NeuralError> {
    // x, W (already transposed for even roles), row, col, bias,
    // mix_gate, mix_bias, ones_column, one, minus_one, inv_sqrt_width.
    let x = inputs[0];
    let weighted_gate = g.mul(x, inputs[5])?;
    let gate_sum = g.matmul(weighted_gate, inputs[7])?;
    let normalized = g.mul(gate_sum, inputs[10])?;
    let logit = g.add(normalized, inputs[6])?;
    let neg_logit = g.mul(logit, inputs[9])?;
    let exp = g.exp(neg_logit)?;
    let denominator = g.add(inputs[8], exp)?;
    let gate = g.div(inputs[8], denominator)?;
    let other = g.sub(inputs[8], gate)?;
    let gate = g.broadcast_to(gate, Shape::new(vec![1, WIDTH])?)?;
    let other = g.broadcast_to(other, Shape::new(vec![1, WIDTH])?)?;

    let z = g.mul(x, inputs[3])?;
    let r = (role * 7 + 1) % WIDTH;
    let c = (role * 11 + 3) % WIDTH;
    let first = roll_left_graph(g, z, c)?;
    let second = roll_left_graph(g, z, (c + 17) % WIDTH)?;
    let first = g.mul(gate, first)?;
    let second = g.mul(other, second)?;
    let mixed = g.add(first, second)?;

    // torch F.linear(z, W): z @ W.T; F.linear(z, W.T): z @ W.
    let projected = g.matmul(mixed, inputs[1])?;
    let unshifted = roll_left_graph(g, projected, (WIDTH - r) % WIDTH)?;
    let scaled = g.mul(unshifted, inputs[2])?;
    g.add(scaled, inputs[4])
}

/// Eight named outputs reuse one W. There is no bank of 16 materialized weights,
/// and the gate is inside the Semantic IR, not computed on the host.
pub fn architecture_model() -> Result<Model, ModelError> {
    let mut model = ModelBuilder::new("torus16-mix-operator", "0.1.0");
    let x = model.input("x", vec![1, WIDTH], Domain::Finite)?;
    let w = model.parameter("weight", vec![WIDTH, WIDTH], Domain::Finite)?;
    let ones_column = model.parameter("ones_column", vec![WIDTH, 1], Domain::Finite)?;
    let one = model.parameter("one", vec![1, 1], Domain::Finite)?;
    let minus_one = model.parameter("minus_one", vec![1, 1], Domain::Finite)?;
    let inv_sqrt = model.parameter("inv_sqrt_width", vec![1, 1], Domain::Finite)?;
    let w_transposed = model.graph_mut().permute(w, &[1, 0])?;

    for role in 0..ROLES {
        let row = model.parameter(format!("row_{role}"), vec![1, WIDTH], Domain::Finite)?;
        let col = model.parameter(format!("col_{role}"), vec![1, WIDTH], Domain::Finite)?;
        let bias = model.parameter(format!("bias_{role}"), vec![1, WIDTH], Domain::Finite)?;
        let mix_gate =
            model.parameter(format!("mix_gate_{role}"), vec![1, WIDTH], Domain::Finite)?;
        let mix_bias = model.parameter(format!("mix_bias_{role}"), vec![1, 1], Domain::Finite)?;
        let matrix = if role % 2 == 0 { w_transposed } else { w };
        let output = model.block(
            "torus16_mix_projection",
            &[
                x,
                matrix,
                row,
                col,
                bias,
                mix_gate,
                mix_bias,
                ones_column,
                one,
                minus_one,
                inv_sqrt,
            ],
            |g, inputs| projection_graph(g, inputs, role),
        )?;
        model.output(format!("role_{role}"), output)?;
    }
    model.finish()
}

fn roll_left_tensor(x: &Tensor<f32>, shift: usize) -> Result<Tensor<f32>, NeuralError> {
    let shift = shift % WIDTH;
    if shift == 0 {
        return Ok(x.clone());
    }
    let a = x.slice_axis(1, shift, WIDTH - shift)?;
    let b = x.slice_axis(1, 0, shift)?;
    a.concat(&b, 1)
}

/// VeriNeural Tensor host reference for one role. Not the GPU execution path.
pub fn reference_project(
    x: &[f32; WIDTH],
    parameters: &Parameters,
    role: usize,
) -> Result<[f32; WIDTH], NeuralError> {
    assert!(role < ROLES, "role outside 0..8");
    let p = &parameters.roles[role];
    let x = Tensor::from_slice(vec![1, WIDTH], x)?;
    let gate_weight = Tensor::from_slice(vec![1, WIDTH], &p.mix_gate)?;
    let ones = Tensor::from_vec(vec![WIDTH, 1], vec![1.0; WIDTH])?;
    let scale = Tensor::from_slice(vec![1, 1], &[1.0 / (WIDTH as f32).sqrt()])?;
    let bias = Tensor::from_slice(vec![1, 1], &[p.mix_bias])?;
    let one = Tensor::from_slice(vec![1, 1], &[1.0])?;
    let negative = Tensor::from_slice(vec![1, 1], &[-1.0])?;
    let logit = x
        .mul(&gate_weight)?
        .matmul(&ones)?
        .mul(&scale)?
        .add(&bias)?;
    let denominator = one.add(&logit.mul(&negative)?.exp()?)?;
    let gate = one.div(&denominator)?;
    let other = one.sub(&gate)?;
    let gate = gate.broadcast_to(Shape::new(vec![1, WIDTH])?)?;
    let other = other.broadcast_to(Shape::new(vec![1, WIDTH])?)?;

    let scaled = x.mul(&Tensor::from_slice(vec![1, WIDTH], &p.col)?)?;
    let r = (role * 7 + 1) % WIDTH;
    let c = (role * 11 + 3) % WIDTH;
    let a = roll_left_tensor(&scaled, c)?.mul(&gate)?;
    let b = roll_left_tensor(&scaled, c + 17)?.mul(&other)?;
    let mixed = a.add(&b)?;
    let weight = Tensor::from_vec(
        vec![WIDTH, WIDTH],
        parameters
            .weight
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect(),
    )?;
    let weight = if role % 2 == 0 {
        weight.permute(&[1, 0])?
    } else {
        weight
    };
    let result = roll_left_tensor(&mixed.matmul(&weight)?, (WIDTH - r) % WIDTH)?;
    let result = result
        .mul(&Tensor::from_slice(vec![1, WIDTH], &p.row)?)?
        .add(&Tensor::from_slice(vec![1, WIDTH], &p.bias)?)?;
    Ok(std::array::from_fn(|i| result.values()[i]))
}

/// Independent scalar oracle: explicitly computes both virtual bases without
/// using any IR implementation or PyTorch. Test-only, never used for execution.
#[cfg(test)]
fn explicit_oracle(x: &[f32; WIDTH], parameters: &Parameters, role: usize) -> [f32; WIDTH] {
    let p = &parameters.roles[role];
    let logit = x.iter().zip(p.mix_gate).map(|(a, b)| *a * b).sum::<f32>() / (WIDTH as f32).sqrt()
        + p.mix_bias;
    let gate = 1.0 / (1.0 + (-logit).exp());
    let r = (role * 7 + 1) % WIDTH;
    let c = (role * 11 + 3) % WIDTH;
    std::array::from_fn(|out| {
        let row = (out + WIDTH - r) % WIDTH;
        let basis = |shift: usize| {
            (0..WIDTH)
                .map(|j| {
                    let input = (j + shift) % WIDTH;
                    let w = if role % 2 == 0 {
                        parameters.weight[row][j]
                    } else {
                        parameters.weight[j][row]
                    };
                    x[input] * p.col[input] * w
                })
                .sum::<f32>()
        };
        (gate * basis(c) + (1.0 - gate) * basis((c + 17) % WIDTH)) * p.row[out] + p.bias[out]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use verineural::{GpuEngine, Operation};

    #[test]
    fn all_sixteen_bases_match_independent_scalar_oracle() {
        let mut p = Parameters::default();
        let x = std::array::from_fn(|i| (i as f32 - 23.0) * 0.011);
        for role in 0..ROLES {
            for i in 0..WIDTH {
                p.roles[role].col[i] = 0.85 + (i % 7) as f32 * 0.05;
                p.roles[role].row[i] = 0.7 + (i % 5) as f32 * 0.03;
                p.roles[role].mix_gate[i] = (i as f32 - 19.0) * 0.02;
                p.roles[role].bias[i] = (i % 3) as f32 * 0.003;
            }
            for gate in [-14.0, 0.0, 3.0, 14.0] {
                p.roles[role].mix_bias = gate;
                let got = reference_project(&x, &p, role).unwrap();
                let expected = explicit_oracle(&x, &p, role);
                for i in 0..WIDTH {
                    assert!(
                        (got[i] - expected[i]).abs() < 2e-5,
                        "role={role} bias={gate} lane={i} got={} expected={}",
                        got[i],
                        expected[i]
                    );
                }
            }
        }
    }

    #[test]
    fn graph_has_one_shared_weight_and_eight_gated_projections() {
        let model = architecture_model().unwrap();
        assert_eq!(model.outputs().len(), ROLES);
        assert_eq!(Parameters::trainable_count(), 3848);
        let x = [0.25; WIDTH];
        assert_eq!(
            model
                .bind(Parameters::default().bindings(&x).unwrap())
                .unwrap()
                .len(),
            46
        );
        for role in 0..ROLES {
            let node = model
                .graph()
                .node(model.output(&format!("role_{role}")).unwrap())
                .unwrap();
            assert_eq!(node.shape.dims(), &[1, WIDTH]);
        }
        assert!(
            model
                .graph()
                .nodes()
                .iter()
                .any(|n| matches!(n.operation, Operation::Pointwise(_)))
        );
        assert!(
            model
                .graph()
                .nodes()
                .iter()
                .any(|n| matches!(n.operation, Operation::SliceAxis { .. }))
        );
    }

    #[test]
    fn unsupported_gpu_operations_are_not_silently_bypassed() {
        let model = architecture_model().unwrap();
        let roots: Vec<_> = model.outputs().values().copied().collect();
        assert!(
            GpuEngine::default()
                .compile_outputs(model.graph(), &roots)
                .is_err()
        );
    }

    #[test]
    fn one_shared_weight_zeros_to_role_bias() {
        let mut p = Parameters::default();
        p.weight = [[0.0; WIDTH]; WIDTH];
        for role in 0..ROLES {
            p.roles[role].bias.fill(role as f32 * 0.125);
            let output = reference_project(&[0.37; WIDTH], &p, role).unwrap();
            for actual in output {
                assert!((actual - role as f32 * 0.125).abs() < 1e-6);
            }
        }
    }
}
