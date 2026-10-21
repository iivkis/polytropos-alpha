//! Stage 7 workload experiment: reproducible graph replay timings, not a GPU proof.
//! Run separately from the hash-pinned Stage 7 evidence example.

use std::error::Error;
use std::io;
use std::path::Path;

use verineural::{
    Domain, GpuEngine, GpuError, GpuLaunchConfig, NeuralGraph, NodeId, ReluFusion, Shape, Tensor,
};

const COLUMNS: usize = 256;
const ELEMENT_COUNTS: [usize; 4] = [1_024, 16_384, 262_144, 1_048_576];
const SAMPLES: usize = 5;
const WARMUP: usize = 25;
const REPLAYS: usize = 1_000;

fn dataset(rows: usize, variant: usize) -> Result<[Tensor<f32>; 3], Box<dyn Error>> {
    let elements = rows * COLUMNS;
    let shape = Shape::new(vec![rows, COLUMNS])?;
    let x = (0..elements)
        .map(|i| match (i + variant) % 32 {
            0 => -0.0f32,
            1 => 0.0,
            2 => f32::MIN_POSITIVE,
            3 => -f32::MIN_POSITIVE,
            _ => ((i % 17) as i32 - 8) as f32 * 0.25 + variant as f32,
        })
        .collect();
    let y = (0..elements)
        .map(|i| ((i * 3 + variant * 5) % 13) as f32 * 0.25 - 1.5)
        .collect();
    let bias = (0..COLUMNS)
        .map(|i| ((i + variant * 3) % 11) as f32 * 0.125 - 0.625)
        .collect();
    Ok([
        Tensor::new(shape.clone(), x)?,
        Tensor::new(shape, y)?,
        Tensor::new(Shape::new(vec![COLUMNS])?, bias)?,
    ])
}

fn inputs(ids: [NodeId; 3], tensors: &[Tensor<f32>; 3]) -> [(NodeId, Tensor<f32>); 3] {
    [
        (ids[0], tensors[0].clone()),
        (ids[1], tensors[1].clone()),
        (ids[2], tensors[2].clone()),
    ]
}

fn reference(t: &[Tensor<f32>; 3]) -> Result<Tensor<f32>, Box<dyn Error>> {
    Ok(t[0].add(&t[1])?.bias_add(&t[2])?.relu())
}

fn check_bits(label: &str, observed: &Tensor<f32>, expected: &Tensor<f32>) -> io::Result<()> {
    if observed.shape() != expected.shape() {
        return Err(io::Error::other(format!("{label}: shape mismatch")));
    }
    for (i, (&actual, &cpu)) in observed.values().iter().zip(expected.values()).enumerate() {
        if actual.to_bits() != cpu.to_bits() {
            return Err(io::Error::other(format!(
                "{label}: FP32 mismatch at {i}: GPU={:08x}, CPU={:08x}",
                actual.to_bits(),
                cpu.to_bits()
            )));
        }
    }
    Ok(())
}

fn summary(samples: &mut [f64]) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    (
        samples[0],
        samples[samples.len() / 2],
        samples[samples.len() - 1],
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let baseline =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Disabled)?);
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental)?);
    let root = Path::new("target/verineural/stage7-workload");
    println!(
        "CUDA Graph workload experiment: {SAMPLES} series, {WARMUP} warmup + {REPLAYS} measured per series"
    );
    println!(
        "Results are CUDA-event stream replay intervals in microseconds, not GPU-kernel-only latency"
    );
    println!(
        "elements | baseline min/median/max us | fused min/median/max us | fused/baseline median"
    );

    for elements in ELEMENT_COUNTS {
        let rows = elements / COLUMNS;
        let shape = Shape::new(vec![rows, COLUMNS])?;
        let mut graph = NeuralGraph::default();
        let xid = graph.input(shape.clone(), Domain::Finite);
        let yid = graph.input(shape, Domain::Finite);
        let bid = graph.input(Shape::new(vec![COLUMNS])?, Domain::Finite);
        let added = graph.add(xid, yid)?;
        let biased = graph.bias_add(added, bid)?;
        let output = graph.relu(biased)?;
        let ids = [xid, yid, bid];

        let plan_a = baseline.compile(&graph, output)?;
        let plan_b = fused.compile(&graph, output)?;
        if plan_b.removed_kernel_count() != 2 {
            return Err(io::Error::other("chain fusion did not remove two kernels").into());
        }
        let first = dataset(rows, 0)?;
        let first_inputs = inputs(ids, &first);
        let first_expected = reference(&first)?;
        let path = root.join(elements.to_string());
        let mut exec_a = baseline.prepare(&plan_a, &first_inputs, &path.join("baseline"))?;
        let mut exec_b = fused.prepare(&plan_b, &first_inputs, &path.join("fused"))?;

        exec_a.replay()?;
        exec_b.replay()?;
        check_bits("initial baseline", &exec_a.read_output()?, &first_expected)?;
        check_bits("initial fused", &exec_b.read_output()?, &first_expected)?;

        // Rebind new input values to both *existing* CUgraphExec instances.
        let second = dataset(rows, 1)?;
        let second_inputs = inputs(ids, &second);
        let second_expected = reference(&second)?;
        if first_expected.values() == second_expected.values() {
            return Err(io::Error::other("test datasets produced identical outputs").into());
        }
        exec_a.update_inputs(&second_inputs)?;
        exec_b.update_inputs(&second_inputs)?;
        exec_a.replay()?;
        exec_b.replay()?;
        check_bits("updated baseline", &exec_a.read_output()?, &second_expected)?;
        check_bits("updated fused", &exec_b.read_output()?, &second_expected)?;

        // Input validation is all-or-nothing: a rejected batch must not have
        // modified the previously accepted device inputs.
        let mut invalid = second_inputs.clone();
        let mut data = invalid[0].1.values().to_vec();
        data[0] = f32::NAN;
        invalid[0].1 = Tensor::new(Shape::new(vec![rows, COLUMNS])?, data)?;
        if !matches!(
            exec_a.update_inputs(&invalid),
            Err(GpuError::InvalidGraph(_))
        ) || !matches!(
            exec_b.update_inputs(&invalid),
            Err(GpuError::InvalidGraph(_))
        ) {
            return Err(io::Error::other("non-finite input update was not rejected").into());
        }
        exec_a.replay()?;
        exec_b.replay()?;
        check_bits(
            "rejected update baseline",
            &exec_a.read_output()?,
            &second_expected,
        )?;
        check_bits(
            "rejected update fused",
            &exec_b.read_output()?,
            &second_expected,
        )?;

        // Alternate measurement order to reduce systematic drift bias.
        let mut times_a = Vec::with_capacity(SAMPLES);
        let mut times_b = Vec::with_capacity(SAMPLES);
        for trial in 0..SAMPLES {
            let (a, b) = if trial % 2 == 0 {
                (
                    exec_a.benchmark(WARMUP, REPLAYS)?,
                    exec_b.benchmark(WARMUP, REPLAYS)?,
                )
            } else {
                let b = exec_b.benchmark(WARMUP, REPLAYS)?;
                let a = exec_a.benchmark(WARMUP, REPLAYS)?;
                (a, b)
            };
            times_a.push(a.average().as_secs_f64() * 1_000_000.0);
            times_b.push(b.average().as_secs_f64() * 1_000_000.0);
        }
        let (amin, amed, amax) = summary(&mut times_a);
        let (bmin, bmed, bmax) = summary(&mut times_b);
        println!(
            "{elements:8} | {amin:7.3}/{amed:7.3}/{amax:7.3} | {bmin:7.3}/{bmed:7.3}/{bmax:7.3} | {ratio:.3}x",
            ratio = bmed / amed
        );
    }
    println!(
        "Each size verified by bitwise CPU comparison before and after graph input replacement"
    );
    println!(
        "Including signed zeros and minimum normal values does not prove general IEEE-754 equivalence"
    );
    println!("No FP32, PTX, machine code or GPU memory-safety proof is implied.");
    Ok(())
}
