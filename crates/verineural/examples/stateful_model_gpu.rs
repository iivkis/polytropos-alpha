//! GPU acceptance for explicit named state feedback.
//!
//! One pure multi-output Model is compiled/captured once. Each step replays
//! the same CUDA Graph and transfers next_state into the state input on-device.
//! All expected values are independent, exactly representable FP32 fixtures.
//! This checks concrete GPU behavior, not Lean/IEEE/SASS equivalence.

use std::error::Error;
use std::path::Path;

use verineural::{Domain, GpuEngine, ModelBuilder, ModelExecution, Tensor};

const WIDTH: usize = 4;

fn tensor(values: [f32; WIDTH]) -> Result<Tensor<f32>, Box<dyn Error>> {
    Ok(Tensor::from_vec([WIDTH], values.to_vec())?)
}

fn expect_bits(
    phase: &str,
    name: &str,
    actual: &Tensor<f32>,
    expected: [f32; WIDTH],
) {
    let actual_bits = actual.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let expected_bits = expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(actual.shape().dims(), &[WIDTH], "{phase}: {name} shape");
    assert_eq!(actual_bits, expected_bits, "{phase}: {name} FP32 bits");
}

fn check(
    execution: &mut ModelExecution<'_>,
    phase: &str,
    state: [f32; WIDTH],
    increment: [f32; WIDTH],
    readout: [f32; WIDTH],
) -> Result<(), Box<dyn Error>> {
    expect_bits(phase, "state", &execution.state("state")?, state);
    expect_bits(
        phase,
        "increment",
        &execution.output("increment")?,
        increment,
    );
    expect_bits(phase, "readout", &execution.output("readout")?, readout);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut builder = ModelBuilder::new("stateful-gpu-fixture", "1");
    let state = builder.state("state", [WIDTH], Domain::Any)?;
    let token = builder.input("token", [WIDTH], Domain::Finite)?;
    let gain = builder.parameter("gain", [WIDTH], Domain::Finite)?;

    // next = state + token*gain; readout = next + gain.
    // This is consumer composition, not a built-in recurrent network.
    let ports = builder.block_multi(
        "affine_transition",
        &[("state", state), ("token", token), ("gain", gain)],
        &["next_state", "readout", "increment"],
        |graph, values| {
            let increment = graph.mul(values[1], values[2])?;
            let next = graph.add(values[0], increment)?;
            let readout = graph.add(next, values[2])?;
            Ok(vec![next, readout, increment])
        },
    )?;
    builder.next_state("state", ports["next_state"])?;
    builder.output("increment", ports["increment"])?;
    builder.output("readout", ports["readout"])?;

    let model = builder.finish()?;
    assert_eq!(model.state_links().len(), 1);
    assert_eq!(model.outputs().len(), 2);
    assert_eq!(model.graph().multi_modules().len(), 1);

    let gpu = GpuEngine::default();
    let mut execution = gpu.prepare_model(
        &model,
        [
            ("state", tensor([0.0, 0.0, 0.0, 0.0])?),
            ("token", tensor([1.0, 2.0, 3.0, 4.0])?),
            ("gain", tensor([2.0, 3.0, 4.0, 5.0])?),
        ],
        Path::new("target/verineural/stateful-model-gpu"),
    )?;

    // run() leaves temporal state unchanged; next_state is deliberately NOT
    // a public model output and must still be compiled for state feedback.
    execution.run()?;
    check(
        &mut execution,
        "run without feedback",
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 6.0, 12.0, 20.0],
        [4.0, 9.0, 16.0, 25.0],
    )?;

    execution.step()?;
    check(
        &mut execution,
        "step 1",
        [2.0, 6.0, 12.0, 20.0],
        [2.0, 6.0, 12.0, 20.0],
        [4.0, 9.0, 16.0, 25.0],
    )?;

    // Invalid named roles, shapes and non-finite input values must not
    // mutate the previously bound state, token or parameter buffers.
    assert!(execution.set_input("gain", &tensor([1.0; WIDTH])?).is_err());
    assert!(execution.set_parameter("token", &tensor([1.0; WIDTH])?).is_err());
    assert!(execution.set_state("gain", &tensor([1.0; WIDTH])?).is_err());
    assert!(execution.set_input("token", &Tensor::from_vec([2], vec![1.0, 2.0])?).is_err());
    assert!(execution
        .set_input("token", &tensor([1.0, f32::NAN, 1.0, 1.0])?)
        .is_err());
    expect_bits(
        "rejected writes",
        "state",
        &execution.state("state")?,
        [2.0, 6.0, 12.0, 20.0],
    );

    // Only the token is uploaded. gain and accumulated state stay on-device.
    execution.set_input("token", &tensor([2.0, 1.0, 0.0, 3.0])?)?;
    execution.step()?;
    check(
        &mut execution,
        "step 2, new token",
        [6.0, 9.0, 12.0, 35.0],
        [4.0, 3.0, 0.0, 15.0],
        [8.0, 12.0, 16.0, 40.0],
    )?;

    // No rebind whatsoever: unchanged gain and token, new accumulated state.
    execution.step()?;
    check(
        &mut execution,
        "step 3, persistent parameter",
        [10.0, 12.0, 12.0, 50.0],
        [4.0, 3.0, 0.0, 15.0],
        [12.0, 15.0, 16.0, 55.0],
    )?;

    // Explicit named parameter update must affect the very next replay
    // without recompilation or CUDA Graph reconstruction.
    execution.set_parameter("gain", &tensor([3.0, 2.0, 1.0, 4.0])?)?;
    execution.step()?;
    check(
        &mut execution,
        "step 4, replaced parameter",
        [16.0, 14.0, 12.0, 62.0],
        [6.0, 2.0, 0.0, 12.0],
        [19.0, 16.0, 13.0, 66.0],
    )?;

    // Explicit state reset and a new token use the same captured graph.
    execution.set_state("state", &tensor([5.0, 4.0, 3.0, 2.0])?)?;
    execution.set_input("token", &tensor([1.0; WIDTH])?)?;
    execution.step()?;
    check(
        &mut execution,
        "step 5, state reset",
        [8.0, 6.0, 4.0, 6.0],
        [3.0, 2.0, 1.0, 4.0],
        [11.0, 8.0, 5.0, 10.0],
    )?;

    // Exercise the P0 cases that the ordinary additive state transition
    // cannot detect: unreachable state storage and aliasing public outputs.
    check_replacement_and_aliased_outputs(&gpu)?;

    // A new session would require another prepare_model call. This fixture
    // deliberately holds one ModelExecution for all six GPU replays.
    println!(
        "Stateful CUDA Graph: one capture, run + five steps; \
         next_state feedback, persistent parameters, named updates and reset \
         match exact FP32 fixtures."
    );
    println!(
        "Assurance: concrete GPU bit checks; no formal CUDA D2D, IEEE-754, \
         JIT/SASS or recurrent convergence proof."
    );
    Ok(())
}


/// Regression: both states are replaced with token, so neither is needed
/// for the pure computation. "hidden" is not a public output at all;
/// "visible" is public both directly and through a zero-copy reshape.
fn check_replacement_and_aliased_outputs(gpu: &GpuEngine) -> Result<(), Box<dyn Error>> {
    let mut builder = ModelBuilder::new("state-replacement-and-alias", "1");
    let hidden = builder.state("hidden", [WIDTH], Domain::Any)?;
    let visible = builder.state("visible", [WIDTH], Domain::Any)?;
    let token = builder.input("token", [WIDTH], Domain::Finite)?;
    let gain = builder.parameter("gain", [WIDTH], Domain::Finite)?;
    let view = builder
        .graph_mut()
        .reshape(visible, verineural::Shape::new([2, 2])?)?;
    let scaled = builder.graph_mut().mul(token, gain)?;
    builder.next_state("hidden", token)?;
    builder.next_state("visible", token)?;
    builder.output("previous", visible)?;
    builder.output("previous_view", view)?;
    builder.output("scaled", scaled)?;
    let model = builder.finish()?;

    let mut execution = gpu.prepare_model(
        &model,
        [
            ("hidden", tensor([100.0, 101.0, 102.0, 103.0])?),
            ("visible", tensor([8.0, 9.0, 10.0, 11.0])?),
            ("token", tensor([1.0, 2.0, 3.0, 4.0])?),
            ("gain", tensor([2.0; WIDTH])?),
        ],
        Path::new("target/verineural/stateful-model-gpu/replacement"),
    )?;

    execution.run()?;
    expect_bits(
        "replacement run",
        "hidden",
        &execution.state("hidden")?,
        [100.0, 101.0, 102.0, 103.0],
    );
    expect_bits(
        "replacement run",
        "previous",
        &execution.output("previous")?,
        [8.0, 9.0, 10.0, 11.0],
    );
    execution.step()?;
    // Public output must represent the pre-feedback pure DAG evaluation,
    // while both inputs to the next step contain the new token.
    expect_bits(
        "replacement step 1",
        "hidden",
        &execution.state("hidden")?,
        [1.0, 2.0, 3.0, 4.0],
    );
    expect_bits(
        "replacement step 1",
        "visible",
        &execution.state("visible")?,
        [1.0, 2.0, 3.0, 4.0],
    );
    expect_bits(
        "replacement step 1",
        "previous",
        &execution.output("previous")?,
        [8.0, 9.0, 10.0, 11.0],
    );
    let view = execution.output("previous_view")?;
    assert_eq!(view.shape().dims(), &[2, 2]);
    assert_eq!(
        view.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [8.0f32, 9.0, 10.0, 11.0]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "zero-copy reshape alias must be a pre-feedback snapshot",
    );
    expect_bits(
        "replacement step 1",
        "scaled",
        &execution.output("scaled")?,
        [2.0, 4.0, 6.0, 8.0],
    );

    execution.set_input("token", &tensor([5.0, 6.0, 7.0, 8.0])?)?;
    execution.step()?;
    expect_bits(
        "replacement step 2",
        "previous",
        &execution.output("previous")?,
        [1.0, 2.0, 3.0, 4.0],
    );
    expect_bits(
        "replacement step 2",
        "hidden",
        &execution.state("hidden")?,
        [5.0, 6.0, 7.0, 8.0],
    );
    expect_bits(
        "replacement step 2",
        "visible",
        &execution.state("visible")?,
        [5.0, 6.0, 7.0, 8.0],
    );
    expect_bits(
        "replacement step 2",
        "scaled",
        &execution.output("scaled")?,
        [10.0, 12.0, 14.0, 16.0],
    );

    // run() refreshes the public snapshot but does not move state.
    execution.run()?;
    expect_bits(
        "replacement run after feedback",
        "previous",
        &execution.output("previous")?,
        [5.0, 6.0, 7.0, 8.0],
    );
    expect_bits(
        "replacement run after feedback",
        "hidden",
        &execution.state("hidden")?,
        [5.0, 6.0, 7.0, 8.0],
    );
    println!(
        "P0 GPU regressions: unreachable state destinations are live; \
         public state/reshape outputs preserve pre-feedback FP32 bits."
    );
    Ok(())
}
