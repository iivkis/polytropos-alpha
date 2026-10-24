//! NVIDIA-only end-to-end acceptance for the rotating-matrices consumer.
//!
//! An independent scalar CPU oracle checks the GPU's chosen route, spatial
//! rotation, recurrent feedback and final readout. This is empirical parity,
//! NOT a proof of LLVM/PTX/JIT/SASS correctness or IEEE-754 equivalence.

use rotating_matrices_lab::{
    Dictionary, Example, Parameters, Rng, Runner, ACTIONS, CORES, TURNS, WIDTH,
};

fn dictionary() -> Dictionary {
    Dictionary::fit(&[Example {
        op: 0,
        a: 0,
        b: 0,
        text: "abcd".to_owned(),
        target: 0,
    }])
}

fn parameters() -> Parameters {
    let mut p = Parameters::new(7);
    // Nonzero, action-dependent router weights exercise the MatMul + BiasAdd
    // logits path even when the selected route is pinned by a bias margin.
    for row in 0..WIDTH {
        for action in 0..ACTIONS {
            let n = (row * 5 + action * 7) % 13 + 1;
            p.router[row * ACTIONS + action] = n as f32 / 8192.0;
        }
    }
    p.head.fill(0.0);
    p.head[0] = 0.5;
    p.head[1] = 0.25;
    p.head[2] = 0.125;
    p.head_bias = 0.0625;
    p.core_bias.fill(0.015625);

    // Nonsymmetric dense matrices make all rotations observable. Every entry
    // is an exactly representable binary fraction to limit rounding noise.
    for core in 0..CORES {
        for row in 0..WIDTH {
            for col in 0..WIDTH {
                let n = (row * 7 + col * 11 + core * 13) % 17 + 1;
                p.cores[core][row * WIDTH + col] = n as f32 / 4096.0;
            }
        }
    }
    p
}

fn set_route(p: &mut Parameters, route: Option<usize>) {
    p.router_bias.fill(-2.0);
    if let Some(route) = route {
        p.router_bias[route] = 2.0;
    } else {
        // Deliberate eight-way tie, requiring ArgMax::First on the device.
        p.router.fill(0.0);
        p.router_bias.fill(2.0);
    }
}

/// Independent indexing formula, not Tensor::rotate or Semantic IR operations.
fn matrix_at(p: &Parameters, core: usize, turn: usize, row: usize, col: usize) -> f32 {
    let (source_row, source_col) = match turn {
        0 => (row, col),
        1 => (WIDTH - 1 - col, row),
        2 => (WIDTH - 1 - row, WIDTH - 1 - col),
        3 => (col, WIDTH - 1 - row),
        _ => unreachable!(),
    };
    p.cores[core][source_row * WIDTH + source_col]
}

struct Reference {
    state: [f32; WIDTH],
    normalized: f32,
    actions: Vec<usize>,
    probabilities: Vec<[f32; ACTIONS]>,
}

fn reference(p: &Parameters, dict: &Dictionary, input: &str) -> Reference {
    let tokens = dict.encode(input).unwrap();
    let histogram = dict.state(&tokens);
    let mut state = [0.0f32; WIDTH];
    let mut actions = Vec::with_capacity(tokens.len());
    let mut probabilities = Vec::with_capacity(tokens.len());

    for token in tokens {
        let mut logits = [0.0f32; ACTIONS];
        for (row, &state_value) in state.iter().enumerate() {
            let feature = histogram[row] + state_value;
            for (action, logit) in logits.iter_mut().enumerate() {
                *logit += feature * p.router[row * ACTIONS + action];
            }
        }
        for (action, value) in logits.iter_mut().enumerate() {
            *value += p.router_bias[action];
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = logits.map(|value| (value - maximum).clamp(-80.0, 0.0).exp());
        let total: f32 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        probabilities.push(weights);
        // All logits are finite in this fixture; equality selects first.
        let mut action = 0;
        for candidate in 1..ACTIONS {
            if logits[candidate] > logits[action] {
                action = candidate;
            }
        }
        actions.push(action);
        let core = action / TURNS;
        let turn = action % TURNS;
        let bits = rotating_matrices_lab::Dictionary::bits(token);
        let mut next = [0.0f32; WIDTH];
        for col in 0..WIDTH {
            let mut accumulator = 0.0f32;
            for row in 0..WIDTH {
                accumulator += (state[row] + bits[row])
                    * matrix_at(p, core, turn, row, col);
            }
            next[col] = (accumulator + p.core_bias[col]).max(0.0);
        }
        state = next;
    }
    let mut normalized = p.head_bias;
    for (value, weight) in state.iter().zip(&p.head) {
        normalized += value * weight;
    }
    Reference {
        state,
        normalized,
        actions,
        probabilities,
    }
}

fn assert_near(actual: f32, expected: f32, context: &str) {
    let bound = 1.0e-5 * expected.abs().max(1.0);
    assert!(
        actual.is_finite() && (actual - expected).abs() <= bound,
        "{context}: GPU {actual:?}, independent CPU {expected:?}, tolerance {bound:?}"
    );
}

#[test]
fn cpu_oracle_distinguishes_all_eight_routes() {
    let dict = dictionary();
    let mut p = parameters();
    let mut signatures = Vec::new();
    for route in 0..ACTIONS {
        set_route(&mut p, Some(route));
        let observed = reference(&p, &dict, "abcda");
        assert_eq!(observed.actions, vec![route; 5]);
        let bits = observed.state.map(f32::to_bits);
        assert!(
            !signatures.contains(&bits),
            "the reference fixture cannot distinguish route {route}"
        );
        signatures.push(bits);
    }
    set_route(&mut p, None);
    assert_eq!(reference(&p, &dict, "abc").actions, vec![0; 3]);
}

/// Run explicitly in a pinned Nix devShell with NVIDIA CUDA Graph support:
/// task verify-lab-rotating-gpu
#[test]
#[ignore = "requires NVPTX llc, CUDA Graph and NVIDIA hardware from the Nix devShell"]
fn deterministic_cuda_graph_matches_independent_cpu_oracle() {
    let dict = dictionary();
    let mut p = parameters();
    let mut gpu = Runner::new(&p).expect("prepare the shared architecture CUDA Graph");
    let mut rng = Rng::new(123);

    // Reuse the same ModelExecution across changes of route and sequence
    // length. This also checks state reset instead of leaked feedback.
    for route in 0..ACTIONS {
        set_route(&mut p, Some(route));
        for input in ["abcda", "dc"] {
            let cpu = reference(&p, &dict, input);
            let actual = gpu
                .forward(&p, &dict, input, &mut rng, false)
                .expect("deterministic CUDA Graph replay");
            assert_eq!(
                actual.decisions.iter().map(|d| d.action).collect::<Vec<_>>(),
                cpu.actions,
                "GPU router decisions for route {route}, input {input}"
            );
            for (step, (decision, expected)) in
                actual.decisions.iter().zip(&cpu.probabilities).enumerate()
            {
                for (action, (&got, &want)) in decision.probs.iter().zip(expected).enumerate() {
                    assert_near(
                        got,
                        want,
                        &format!("route {route}, input {input}, step {step}, probability[{action}]"),
                    );
                }
            }
            for (slot, (&got, &expected)) in actual.state.iter().zip(&cpu.state).enumerate() {
                assert_near(got, expected, &format!("route {route}, input {input}, state[{slot}]"));
            }
            assert_near(
                actual.normalized,
                cpu.normalized,
                &format!("route {route}, input {input}, readout"),
            );
        }
    }

    // Ties must be resolved on the GPU, not merely in host diagnostics.
    set_route(&mut p, None);
    let cpu = reference(&p, &dict, "badc");
    let actual = gpu
        .forward(&p, &dict, "badc", &mut rng, false)
        .expect("first-tie CUDA Graph replay");
    assert_eq!(actual.decisions.iter().map(|d| d.action).collect::<Vec<_>>(), cpu.actions);
    for (&got, &expected) in actual.state.iter().zip(&cpu.state) {
        assert_near(got, expected, "first-tie recurrent state");
    }
    assert_near(actual.normalized, cpu.normalized, "first-tie readout");
}
