//! Independent scalar contracts for a single Role-Delta projection.
//! GPU VJP checks exercise the exact lab projection built on public VeriNeural IR.
use crate::{Parameters, WIDTH, projection_graph_with_delta};
use verineural::{Domain, GpuEngine, GpuLaunchConfig, Model, ModelBuilder, Tensor, differentiate};

const RANK: usize = 4;

#[derive(Clone)]
struct Fixture {
    p: Parameters,
    x: [f32; WIDTH],
    a: Vec<f32>,
    b: Vec<f32>,
    scale: f32,
}

impl Fixture {
    fn new(role: usize, zero_b: bool) -> Self {
        let mut p = Parameters::default();
        let q = &mut p.roles[role];
        for i in 0..WIDTH {
            q.row[i] = 0.75 + (i % 7) as f32 * 0.07;
            q.col[i] = 0.80 + (i % 5) as f32 * 0.04;
            q.bias[i] = (i % 3) as f32 * 0.003;
            q.mix_gate[i] = (i as f32 - 23.0) * 0.03;
        }
        q.mix_bias = 0.15;
        Self {
            p,
            x: std::array::from_fn(|i| (i as f32 - 20.0) * 0.017),
            a: (0..WIDTH * RANK)
                .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.015)
                .collect(),
            b: (0..RANK * WIDTH)
                .map(|i| {
                    if zero_b {
                        0.0
                    } else {
                        ((i * 13 % 31) as f32 - 15.0) * 0.021
                    }
                })
                .collect(),
            scale: 0.75,
        }
    }

    fn bindings(&self, role: usize) -> Vec<(String, Tensor<f32>)> {
        let p = &self.p.roles[role];
        let mut result = Vec::new();
        let mut put = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
            result.push((name.into(), Tensor::from_vec(shape, values).unwrap()));
        };
        put("x", vec![1, WIDTH], self.x.to_vec());
        put(
            "weight",
            vec![WIDTH, WIDTH],
            self.p.weight.iter().flatten().copied().collect(),
        );
        for (name, values) in [
            ("row", &p.row),
            ("col", &p.col),
            ("bias", &p.bias),
            ("mix_gate", &p.mix_gate),
        ] {
            put(name, vec![1, WIDTH], values.to_vec());
        }
        put("mix_bias", vec![1, 1], vec![p.mix_bias]);
        put("ones_column", vec![WIDTH, 1], vec![1.0; WIDTH]);
        put("one", vec![1, 1], vec![1.0]);
        put("minus_one", vec![1, 1], vec![-1.0]);
        put(
            "inv_sqrt_width",
            vec![1, 1],
            vec![1.0 / (WIDTH as f32).sqrt()],
        );
        put("delta_a", vec![WIDTH, RANK], self.a.clone());
        put("delta_b", vec![RANK, WIDTH], self.b.clone());
        put("delta_scale", vec![1, 1], vec![self.scale]);
        result
    }

    // Entirely independent scalar reference. No Tensor, IR or GPU execution.
    fn mixed(&self, role: usize) -> [f32; WIDTH] {
        let p = &self.p.roles[role];
        let logit = self
            .x
            .iter()
            .zip(p.mix_gate)
            .map(|(a, b)| *a * b)
            .sum::<f32>()
            / (WIDTH as f32).sqrt()
            + p.mix_bias;
        let gate = 1.0 / (1.0 + (-logit).exp());
        let c = (role * 11 + 3) % WIDTH;
        std::array::from_fn(|j| {
            let one = (j + c) % WIDTH;
            let two = (j + c + 17) % WIDTH;
            gate * self.x[one] * p.col[one] + (1.0 - gate) * self.x[two] * p.col[two]
        })
    }

    fn forward(&self, role: usize) -> [f32; WIDTH] {
        let p = &self.p.roles[role];
        let mixed = self.mixed(role);
        let r = (role * 7 + 1) % WIDTH;
        let low: [f32; RANK] =
            std::array::from_fn(|k| (0..WIDTH).map(|j| mixed[j] * self.a[j * RANK + k]).sum());
        std::array::from_fn(|out| {
            let column = (out + WIDTH - r) % WIDTH;
            let base = (0..WIDTH)
                .map(|j| {
                    let w = if role.is_multiple_of(2) {
                        self.p.weight[column][j]
                    } else {
                        self.p.weight[j][column]
                    };
                    mixed[j] * w
                })
                .sum::<f32>();
            let delta = (0..RANK)
                .map(|k| low[k] * self.b[k * WIDTH + column])
                .sum::<f32>();
            (base + self.scale * delta) * p.row[out] + p.bias[out]
        })
    }

    fn loss(&self, role: usize, seed: &[f32; WIDTH]) -> f32 {
        self.forward(role)
            .iter()
            .zip(seed)
            .map(|(&y, &dy)| y * dy)
            .sum()
    }

    // For L = sum(output * seed), dL/dprojected[column] accounts for
    // the original roll and row scale, before W and A/B branches.
    fn gradients(&self, role: usize, seed: &[f32; WIDTH]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mixed = self.mixed(role);
        let r = (role * 7 + 1) % WIDTH;
        let dprojected: [f32; WIDTH] = std::array::from_fn(|column| {
            let out = (column + r) % WIDTH;
            seed[out] * self.p.roles[role].row[out]
        });
        let mut dw = vec![0.0; WIDTH * WIDTH];
        for (j, &u) in mixed.iter().enumerate() {
            for (column, &d) in dprojected.iter().enumerate() {
                let index = if role.is_multiple_of(2) {
                    column * WIDTH + j
                } else {
                    j * WIDTH + column
                };
                dw[index] += u * d;
            }
        }
        let mut da = vec![0.0; WIDTH * RANK];
        let mut db = vec![0.0; RANK * WIDTH];
        for k in 0..RANK {
            let upstream: f32 = (0..WIDTH)
                .map(|column| self.b[k * WIDTH + column] * dprojected[column])
                .sum();
            let projected_a: f32 = (0..WIDTH).map(|j| mixed[j] * self.a[j * RANK + k]).sum();
            for (j, &u) in mixed.iter().enumerate() {
                da[j * RANK + k] = self.scale * u * upstream;
            }
            for (column, &d) in dprojected.iter().enumerate() {
                db[k * WIDTH + column] = self.scale * projected_a * d;
            }
        }
        (dw, da, db)
    }
}

fn model(role: usize) -> Model {
    let mut m = ModelBuilder::new("role-delta-gradient-oracle", "0.2.0");
    let x = m.input("x", vec![1, WIDTH], Domain::Finite).unwrap();
    let w = m
        .parameter("weight", vec![WIDTH, WIDTH], Domain::Finite)
        .unwrap();
    let w = if role.is_multiple_of(2) {
        m.graph_mut().permute(w, &[1, 0]).unwrap()
    } else {
        w
    };
    let row = m.parameter("row", vec![1, WIDTH], Domain::Finite).unwrap();
    let col = m.parameter("col", vec![1, WIDTH], Domain::Finite).unwrap();
    let bias = m.parameter("bias", vec![1, WIDTH], Domain::Finite).unwrap();
    let mix_gate = m
        .parameter("mix_gate", vec![1, WIDTH], Domain::Finite)
        .unwrap();
    let mix_bias = m.parameter("mix_bias", vec![1, 1], Domain::Finite).unwrap();
    let ones = m
        .parameter("ones_column", vec![WIDTH, 1], Domain::Finite)
        .unwrap();
    let one = m.parameter("one", vec![1, 1], Domain::Finite).unwrap();
    let minus = m
        .parameter("minus_one", vec![1, 1], Domain::Finite)
        .unwrap();
    let inv = m
        .parameter("inv_sqrt_width", vec![1, 1], Domain::Finite)
        .unwrap();
    let a = m
        .parameter("delta_a", vec![WIDTH, RANK], Domain::Finite)
        .unwrap();
    let b = m
        .parameter("delta_b", vec![RANK, WIDTH], Domain::Finite)
        .unwrap();
    let scale = m
        .parameter("delta_scale", vec![1, 1], Domain::Finite)
        .unwrap();
    let output = m
        .block(
            "role_delta_projection",
            &[
                x, w, row, col, bias, mix_gate, mix_bias, ones, one, minus, inv, a, b, scale,
            ],
            |g, nodes| projection_graph_with_delta(g, nodes, role),
        )
        .unwrap();
    m.output("projected", output).unwrap();
    m.finish().unwrap()
}

fn seed() -> [f32; WIDTH] {
    std::array::from_fn(|i| ((i * 7 % 17) as f32 - 8.0) * 0.12)
}

fn close(name: &str, got: &[f32], expected: &[f32], abs: f32, rel: f32) {
    assert_eq!(got.len(), expected.len(), "{name}");
    for (i, (&a, &b)) in got.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= abs + rel * b.abs(),
            "{name}[{i}]: GPU/analytic {a} vs {b}"
        );
    }
}

#[test]
fn role_delta_scalar_vjp_matches_finite_differences() {
    let seed = seed();
    let epsilon = 0.01;
    for role in [0, 1] {
        for zero_b in [true, false] {
            let fixture = Fixture::new(role, zero_b);
            let (dw, da, db) = fixture.gradients(role, &seed);
            for index in [0, 38, 117] {
                let (row, col) = (index / WIDTH, index % WIDTH);
                let mut minus = fixture.clone();
                let mut plus = fixture.clone();
                minus.p.weight[row][col] -= epsilon;
                plus.p.weight[row][col] += epsilon;
                let fd = (plus.loss(role, &seed) - minus.loss(role, &seed)) / (2.0 * epsilon);
                close("dW finite difference", &[dw[index]], &[fd], 3e-4, 0.01);
            }
            for index in [0, 39, 117] {
                let mut minus = fixture.clone();
                let mut plus = fixture.clone();
                minus.a[index] -= epsilon;
                plus.a[index] += epsilon;
                let fd = (plus.loss(role, &seed) - minus.loss(role, &seed)) / (2.0 * epsilon);
                close("dA finite difference", &[da[index]], &[fd], 3e-4, 0.01);
                let mut minus = fixture.clone();
                let mut plus = fixture.clone();
                minus.b[index] -= epsilon;
                plus.b[index] += epsilon;
                let fd = (plus.loss(role, &seed) - minus.loss(role, &seed)) / (2.0 * epsilon);
                close("dB finite difference", &[db[index]], &[fd], 3e-4, 0.01);
            }
            if zero_b {
                assert!(da.iter().all(|v| *v == 0.0));
                assert!(db.iter().any(|v| v.abs() > 1e-6));
            } else {
                assert!(da.iter().any(|v| v.abs() > 1e-6));
            }
        }
    }
}

#[test]
#[ignore = "requires Nix LLVM llc and NVIDIA CUDA Graph; analytical projection VJP parity"]
fn role_delta_gpu_vjp_matches_scalar_oracle() {
    let seed = seed();
    let engine = GpuEngine::new(
        "llc",
        std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
    )
    .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
    for role in [0, 1] {
        let model = model(role);
        let output = model.output("projected").unwrap();
        let sources = ["weight", "delta_a", "delta_b"].map(|name| model.source(name).unwrap().0);
        let reverse = differentiate(model.graph(), output, &sources).unwrap();
        let gradients = sources.map(|id| reverse.gradient_of(id).unwrap());
        let roots = [output, gradients[0], gradients[1], gradients[2]];
        let compiled = engine.compile_outputs(reverse.graph(), &roots).unwrap();
        for zero_b in [true, false] {
            let fixture = Fixture::new(role, zero_b);
            let bound = model.bind(fixture.bindings(role)).unwrap();
            let tensors = reverse
                .bindings(&bound, Tensor::from_slice(vec![1, WIDTH], &seed).unwrap())
                .unwrap();
            let active = tensors
                .into_iter()
                .filter(|(id, _)| compiled.source_inputs().contains(id))
                .collect::<Vec<_>>();
            let dir = std::env::temp_dir().join(format!(
                "vn_role_delta_vjp_{}_{}_{}",
                std::process::id(),
                role,
                if zero_b { "zero" } else { "active" }
            ));
            let mut execution = engine.prepare(&compiled, &active, &dir).unwrap();
            execution.replay().unwrap();
            let observed = execution.read_outputs_selected(&roots).unwrap();
            let (dw, da, db) = fixture.gradients(role, &seed);
            let expected_y = fixture.forward(role);
            for ((actual_id, _), expected_id) in observed.iter().zip(roots) {
                assert_eq!(*actual_id, expected_id);
            }
            close("forward", observed[0].1.values(), &expected_y, 5e-5, 2e-4);
            close("dW", observed[1].1.values(), &dw, 8e-5, 0.01);
            close("dA", observed[2].1.values(), &da, 8e-5, 0.01);
            close("dB", observed[3].1.values(), &db, 8e-5, 0.01);
        }
    }
}
