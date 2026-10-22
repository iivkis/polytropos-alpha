//! Stage 8: exact-integer ranges plus concrete fractional dyadic RNE/epsilon
//! traces. Rust input guards -> Lean range/error lemmas -> baseline/fused CUDA
//! Graph -> independent integer oracles -> SHA-256 evidence.
//! The LLVM/PTX/JIT/GPU link remains trusted rather than formally proved.

use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

use verineural::{
    Domain, ExactFp32Plan, GpuEngine, GpuLaunchConfig, LeanProofEngine, NeuralGraph, ReluFusion,
    ResidualFfn, RneFp32Plan, RneRangePlan, Shape, Tensor,
};

const FILES: [&str; 32] = [
    "numeric.lean",
    "baseline/graph.ll",
    "baseline/graph.ptx",
    "fused/graph.ll",
    "fused/graph.ptx",
    "inputs.bits",
    "output.bits",
    "numeric_matmul.lean",
    "baseline_matmul/graph.ll",
    "baseline_matmul/graph.ptx",
    "fused_matmul/graph.ll",
    "fused_matmul/graph.ptx",
    "matmul_inputs.bits",
    "matmul_output.bits",
    "numeric_rne.lean",
    "baseline_rne/graph.ll",
    "baseline_rne/graph.ptx",
    "fused_rne/graph.ll",
    "fused_rne/graph.ptx",
    "rne_inputs.bits",
    "rne_output.bits",
    "rne_error.txt",
    "numeric_rne_matmul.lean",
    "baseline_rne_matmul/graph.ll",
    "baseline_rne_matmul/graph.ptx",
    "fused_rne_matmul/graph.ll",
    "fused_rne_matmul/graph.ptx",
    "rne_matmul_inputs.bits",
    "rne_matmul_output.bits",
    "rne_matmul_error.txt",
    "numeric_range.lean",
    "range_contract.txt",
];

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(io::Error::other("sha256sum failed").into());
    }
    Ok(String::from_utf8(output.stdout)?
        .split_whitespace()
        .next()
        .ok_or_else(|| io::Error::other("missing SHA-256 digest"))?
        .to_owned())
}

fn verify(root: &Path) -> Result<(), Box<dyn Error>> {
    let manifest = root.join("SHA256SUMS");
    if let Ok(pinned) = std::env::var("VERINEURAL_STAGE8_SHA256") {
        if sha256(&manifest)? != pinned {
            return Err(io::Error::other("Stage 8 externally pinned hash mismatch").into());
        }
    }
    let checked = Command::new("sha256sum")
        .arg("--check")
        .arg("SHA256SUMS")
        .current_dir(root)
        .output()?;
    if !checked.status.success() {
        return Err(io::Error::other(format!(
            "Stage 8 manifest rejected: {}",
            String::from_utf8_lossy(&checked.stderr)
        ))
        .into());
    }
    LeanProofEngine::default().check_file(&root.join("numeric.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_matmul.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_rne.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_rne_matmul.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_range.lean"))?;
    println!("Stage 8: saved Lean sources and SHA-256 artifact manifest PASS");
    println!(
        "Scope: ideal integer ranges, concrete dyadic RNE traces, and reusable bounded RNE/module contracts; not machine FP32 certification"
    );
    Ok(())
}

/// Generate and check one Lean artifact as a single ordered operation.
/// Saved-artifact verification is handled separately by verify().
fn write_checked_lean(path: &Path, source: &str) -> Result<(), Box<dyn Error>> {
    fs::write(path, source)?;
    LeanProofEngine::default().check_file(path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new("target/verineural/stage8-exact");
    if std::env::args().any(|arg| arg == "--verify") {
        return verify(root);
    }
    fs::create_dir_all(root)?;
    // A failed rerun must never leave a success manifest behind.
    match fs::remove_file(root.join("SHA256SUMS")) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }

    let shape = Shape::new(vec![2, 3])?;
    let x = Tensor::new(
        shape.clone(),
        vec![-1_048_576.0, 0.0, 1.0, 100.0, -5.0, 1_048_576.0],
    )?;
    let y = Tensor::new(
        shape.clone(),
        vec![1_048_576.0, 7.0, -3.0, -100.0, 5.0, -1_048_576.0],
    )?;
    let bias = Tensor::new(Shape::new(vec![3])?, vec![1.0, -2.0, 3.0])?;

    let mut graph = NeuralGraph::default();
    let xid = graph.input(shape.clone(), Domain::Finite);
    let yid = graph.input(shape, Domain::Finite);
    let bid = graph.input(bias.shape().clone(), Domain::Finite);
    let added = graph.add(xid, yid)?;
    let biased = graph.bias_add(added, bid)?;
    let activated = graph.relu(biased)?;
    let output = graph.reshape(activated, Shape::new(vec![3, 2])?)?;
    let inputs = [(xid, x), (yid, y), (bid, bias)];

    let plan = ExactFp32Plan::compile(&graph, output)?;
    let expected = plan.integer_reference(&inputs)?;
    write_checked_lean(&root.join("numeric.lean"), &plan.lean_source()?)?;

    let baseline =
        GpuEngine::default().with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Disabled)?);
    let fused = GpuEngine::default()
        .with_launch_config(GpuLaunchConfig::new(256, ReluFusion::Experimental)?);
    for (label, engine) in [("baseline", &baseline), ("fused", &fused)] {
        let gpu_plan = engine.compile(&graph, output)?;
        if label == "fused" && gpu_plan.removed_kernel_count() != 2 {
            return Err(io::Error::other("expected Add/BiasAdd/Relu chain fusion").into());
        }
        let actual = engine.execute(&gpu_plan, &inputs, &root.join(label))?;
        plan.compare_gpu(&inputs, &actual)?;
        println!(
            "Stage 8 {label}: {} exact-integer FP32 outputs match independent i64 oracle",
            actual.values().len()
        );
    }

    // Independent reduction graph: MatMul -> Mul -> BiasAdd -> Relu.
    // A tighter bound proves even partial dot sums remain exactly representable.
    let mut mat_graph = NeuralGraph::default();
    let lhs = Tensor::new(
        Shape::new(vec![2, 3])?,
        vec![2.0, -1.0, 3.0, 0.0, 4.0, -2.0],
    )?;
    let rhs = Tensor::new(
        Shape::new(vec![3, 2])?,
        vec![1.0, 2.0, -3.0, 1.0, 2.0, -1.0],
    )?;
    let mask = Tensor::new(Shape::new(vec![2, 2])?, vec![2.0, 3.0, 2.0, 4.0])?;
    let mbias = Tensor::new(Shape::new(vec![2])?, vec![1.0, -2.0])?;
    let aid = mat_graph.input(lhs.shape().clone(), Domain::Finite);
    let bid = mat_graph.input(rhs.shape().clone(), Domain::Finite);
    let mid = mat_graph.input(mask.shape().clone(), Domain::Finite);
    let biasid = mat_graph.input(mbias.shape().clone(), Domain::Finite);
    let product = mat_graph.matmul(aid, bid)?;
    let scaled = mat_graph.mul(product, mid)?;
    let bias_added = mat_graph.bias_add(scaled, biasid)?;
    let mat_output = mat_graph.relu(bias_added)?;
    let mat_inputs = [(aid, lhs), (bid, rhs), (mid, mask), (biasid, mbias)];
    let mat_plan = ExactFp32Plan::compile_with_input_bound(&mat_graph, mat_output, 8)?;
    let mat_expected = mat_plan.integer_reference(&mat_inputs)?;
    write_checked_lean(&root.join("numeric_matmul.lean"), &mat_plan.lean_source()?)?;
    for (label, engine) in [("baseline_matmul", &baseline), ("fused_matmul", &fused)] {
        let plan = engine.compile(&mat_graph, mat_output)?;
        if label == "fused_matmul" && plan.removed_kernel_count() != 2 {
            return Err(io::Error::other("expected Mul/BiasAdd/Relu chain fusion").into());
        }
        let actual = engine.execute(&plan, &mat_inputs, &root.join(label))?;
        mat_plan.compare_gpu(&mat_inputs, &actual)?;
        println!(
            "Stage 8 {label}: {} exact integer MatMul/Mul values match i64 oracle",
            actual.values().len()
        );
    }
    fs::write(
        root.join("matmul_inputs.bits"),
        mat_inputs
            .iter()
            .map(|(id, tensor)| {
                format!(
                    "node={} shape={:?} bits={:?}\n",
                    id.index(),
                    tensor.shape().dims(),
                    tensor
                        .values()
                        .iter()
                        .map(|x| format!("{:08x}", x.to_bits()))
                        .collect::<Vec<_>>()
                )
            })
            .collect::<String>(),
    )?;
    fs::write(
        root.join("matmul_output.bits"),
        mat_expected
            .values()
            .iter()
            .map(|x| format!("{:08x}\n", x.to_bits()))
            .collect::<String>(),
    )?;

    // Fractional dyadic RNE trace. With q=20, x=8+2^-20 and y=8
    // produce the exact midpoint 16+2^-20, which RNE sends to even 16.
    // Adding bias=2^-20 repeats the rounding loss, so epsilon is nonzero.
    let mut rne_graph = NeuralGraph::default();
    let rne_shape = Shape::new(vec![1, 1])?;
    let rne_x = Tensor::new(rne_shape.clone(), vec![f32::from_bits(0x4100_0001)])?;
    let rne_y = Tensor::new(rne_shape.clone(), vec![f32::from_bits(0x4100_0000)])?;
    let rne_bias = Tensor::new(Shape::new(vec![1])?, vec![f32::from_bits(0x3580_0000)])?;
    let rne_xid = rne_graph.input(rne_shape.clone(), Domain::Finite);
    let rne_yid = rne_graph.input(rne_shape, Domain::Finite);
    let rne_bid = rne_graph.input(rne_bias.shape().clone(), Domain::Finite);
    let rne_sum = rne_graph.add(rne_xid, rne_yid)?;
    let rne_biased = rne_graph.bias_add(rne_sum, rne_bid)?;
    let rne_output = rne_graph.relu(rne_biased)?;
    let rne_inputs = [(rne_xid, rne_x), (rne_yid, rne_y), (rne_bid, rne_bias)];
    let rne_plan = RneFp32Plan::compile(&rne_graph, rne_output, 20)?;
    let rne_trace = rne_plan.trace(&rne_inputs)?;
    if rne_trace.observed_error_ticks() == 0 || rne_trace.epsilon_ticks() == 0 {
        return Err(io::Error::other("expected a nonzero Stage 8 RNE error trace").into());
    }
    write_checked_lean(&root.join("numeric_rne.lean"), rne_trace.lean_source())?;
    for (label, engine) in [("baseline_rne", &baseline), ("fused_rne", &fused)] {
        let gpu_plan = engine.compile(&rne_graph, rne_output)?;
        if label == "fused_rne" && gpu_plan.removed_kernel_count() != 2 {
            return Err(
                io::Error::other("expected fractional Add/BiasAdd/Relu chain fusion").into(),
            );
        }
        let actual = engine.execute(&gpu_plan, &rne_inputs, &root.join(label))?;
        rne_trace.compare_gpu(&actual)?;
        println!(
            "Stage 8 {label}: {} fractional FP32 outputs match integer RNE oracle",
            actual.values().len()
        );
    }
    fs::write(
        root.join("rne_inputs.bits"),
        rne_inputs
            .iter()
            .map(|(id, tensor)| {
                format!(
                    "node={} shape={:?} bits={:?}\n",
                    id.index(),
                    tensor.shape().dims(),
                    tensor
                        .values()
                        .iter()
                        .map(|x| format!("{:08x}", x.to_bits()))
                        .collect::<Vec<_>>()
                )
            })
            .collect::<String>(),
    )?;
    fs::write(
        root.join("rne_output.bits"),
        rne_trace
            .expected()
            .values()
            .iter()
            .map(|x| format!("{:08x}\n", x.to_bits()))
            .collect::<String>(),
    )?;
    fs::write(
        root.join("rne_error.txt"),
        format!(
            "input_fraction_bits={}\noutput_fraction_bits={}\nideal_ticks={:?}\nrounded_ticks={:?}\nobserved_error_ticks={}\n\
             epsilon_ticks={}\nobserved_error={:.12e}\nepsilon={:.12e}\n",
            rne_plan.fraction_bits(),
            rne_trace.fraction_bits(),
            rne_trace.ideal_output_ticks(),
            rne_trace.rounded_output_ticks(),
            rne_trace.observed_error_ticks(),
            rne_trace.epsilon_ticks(),
            rne_trace.observed_error(),
            rne_trace.epsilon(),
        ),
    )?;

    // Fractional MatMul trace. Each product is rounded independently, then
    // each left-to-right accumulation add is rounded independently as required
    // by the strict FP32 policy. The chosen inputs leave a two-tick ideal/RNE
    // discrepancy on the q=40 output proof lattice.
    let mut rne_mat_graph = NeuralGraph::default();
    let rne_mat_value = f32::from_bits(0x4100_0001);
    let rne_mat_lhs = Tensor::new(Shape::new(vec![1, 2])?, vec![rne_mat_value, rne_mat_value])?;
    let rne_mat_rhs = Tensor::new(Shape::new(vec![2, 1])?, vec![rne_mat_value, rne_mat_value])?;
    let rne_mat_lhs_id = rne_mat_graph.input(rne_mat_lhs.shape().clone(), Domain::Finite);
    let rne_mat_rhs_id = rne_mat_graph.input(rne_mat_rhs.shape().clone(), Domain::Finite);
    let rne_mat_output = rne_mat_graph.matmul(rne_mat_lhs_id, rne_mat_rhs_id)?;
    let rne_mat_inputs = [(rne_mat_lhs_id, rne_mat_lhs), (rne_mat_rhs_id, rne_mat_rhs)];
    let rne_mat_plan = RneFp32Plan::compile(&rne_mat_graph, rne_mat_output, 20)?;
    let rne_mat_trace = rne_mat_plan.trace(&rne_mat_inputs)?;
    if rne_mat_plan.output_fraction_bits() != 40
        || rne_mat_trace.observed_error_ticks() != 2
        || rne_mat_trace.epsilon_ticks() == 0
    {
        return Err(
            io::Error::other("unexpected Stage 8 fractional MatMul RNE/error trace").into(),
        );
    }
    write_checked_lean(
        &root.join("numeric_rne_matmul.lean"),
        rne_mat_trace.lean_source(),
    )?;
    for (label, engine) in [
        ("baseline_rne_matmul", &baseline),
        ("fused_rne_matmul", &fused),
    ] {
        let gpu_plan = engine.compile(&rne_mat_graph, rne_mat_output)?;
        let actual = engine.execute(&gpu_plan, &rne_mat_inputs, &root.join(label))?;
        rne_mat_trace.compare_gpu(&actual)?;
        println!(
            "Stage 8 {label}: {} fractional MatMul output(s) match separate-mul/add integer RNE oracle",
            actual.values().len()
        );
    }
    fs::write(
        root.join("rne_matmul_inputs.bits"),
        rne_mat_inputs
            .iter()
            .map(|(id, tensor)| {
                format!(
                    "node={} shape={:?} bits={:?}\n",
                    id.index(),
                    tensor.shape().dims(),
                    tensor
                        .values()
                        .iter()
                        .map(|x| format!("{:08x}", x.to_bits()))
                        .collect::<Vec<_>>()
                )
            })
            .collect::<String>(),
    )?;
    fs::write(
        root.join("rne_matmul_output.bits"),
        rne_mat_trace
            .expected()
            .values()
            .iter()
            .map(|x| format!("{:08x}\n", x.to_bits()))
            .collect::<String>(),
    )?;
    fs::write(
        root.join("rne_matmul_error.txt"),
        format!(
            "input_fraction_bits={}\noutput_fraction_bits={}\nideal_ticks={:?}\nrounded_ticks={:?}\nobserved_error_ticks={}\n\
             epsilon_ticks={}\nobserved_error={:.12e}\nepsilon={:.12e}\n",
            rne_mat_plan.fraction_bits(),
            rne_mat_trace.fraction_bits(),
            rne_mat_trace.ideal_output_ticks(),
            rne_mat_trace.rounded_output_ticks(),
            rne_mat_trace.observed_error_ticks(),
            rne_mat_trace.epsilon_ticks(),
            rne_mat_trace.observed_error(),
            rne_mat_trace.epsilon(),
        ),
    )?;

    // Reusable range contract over a composed module. Unlike the concrete RNE
    // traces above, the values are arbitrary within the source tick interval.
    // Lean proves the scalar transfer rules under Binary32RneTick hypotheses;
    // Rust exports the resulting module boundary bound/epsilon summary.
    let mut range_graph = NeuralGraph::default();
    let range_x = range_graph.input(Shape::new(vec![1, 2])?, Domain::Finite);
    let range_w1 = range_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let range_b1 = range_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let range_w2 = range_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let range_b2 = range_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let range_module = ResidualFfn::new(1, 2, 2)?;
    let range_output = range_graph.apply_module(
        &range_module,
        &[range_x, range_w1, range_b1, range_w2, range_b2],
    )?;
    let range_plan = RneRangePlan::compile(&range_graph, range_output, 10, 16)?;
    let range_contract = range_plan.output_contract();
    if range_contract.fraction_bits() != 30
        || range_contract.ideal_bound_ticks() != 34_095_104
        || range_contract.epsilon_ticks() != 170
    {
        return Err(io::Error::other("unexpected Stage 8 reusable RNE range contract").into());
    }
    let range_modules = range_plan.module_contracts();
    if range_modules.len() != 1 || range_modules[0].name() != "residual_ffn" {
        return Err(io::Error::other("missing Stage 8 ResidualFfn range contract").into());
    }
    write_checked_lean(
        &root.join("numeric_range.lean"),
        &range_plan.lean_source()?,
    )?;
    fs::write(
        root.join("range_contract.txt"),
        format!(
            "module={}\nsource_fraction_bits={}\nsource_bound_ticks={}\n\
             output_fraction_bits={}\nideal_bound_ticks={}\nepsilon_ticks={}\n\
             actual_bound_ticks={}\nepsilon={:.12e}\n",
            range_modules[0].name(),
            range_plan.source_fraction_bits(),
            range_plan.source_bound_ticks(),
            range_contract.fraction_bits(),
            range_contract.ideal_bound_ticks(),
            range_contract.epsilon_ticks(),
            range_contract.actual_bound_ticks(),
            range_contract.epsilon(),
        ),
    )?;

    let input_snapshot = inputs
        .iter()
        .map(|(id, tensor)| {
            format!(
                "node={} shape={:?} bits={:?}\n",
                id.index(),
                tensor.shape().dims(),
                tensor
                    .values()
                    .iter()
                    .map(|x| format!("{:08x}", x.to_bits()))
                    .collect::<Vec<_>>()
            )
        })
        .collect::<String>();
    fs::write(root.join("inputs.bits"), input_snapshot)?;
    fs::write(
        root.join("output.bits"),
        expected
            .values()
            .iter()
            .map(|x| format!("{:08x}\n", x.to_bits()))
            .collect::<String>(),
    )?;
    let result = Command::new("sha256sum")
        .args(FILES)
        .current_dir(root)
        .output()?;
    if !result.status.success() {
        return Err(io::Error::other(format!(
            "SHA-256 manifest generation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        ))
        .into());
    }
    fs::write(root.join("SHA256SUMS"), result.stdout)?;
    if let Err(err) = verify(root) {
        let _ = fs::remove_file(root.join("SHA256SUMS"));
        return Err(err);
    }
    println!(
        "Stage 8 additive profile: input |x| <= {}, intermediate |value| <= {}",
        ExactFp32Plan::input_bound(),
        plan.output_bound()
    );
    println!(
        "Stage 8 MatMul/Mul profile: input |x| <= {}, intermediate |value| <= {}",
        mat_plan.max_input_bound(),
        mat_plan.output_bound()
    );
    println!(
        "Stage 8 additive dyadic RNE profile: q_in={}, q_out={}, observed error={} ticks, proved epsilon={} ticks",
        rne_plan.fraction_bits(),
        rne_trace.fraction_bits(),
        rne_trace.observed_error_ticks(),
        rne_trace.epsilon_ticks()
    );
    println!(
        "Stage 8 MatMul dyadic RNE profile: q_in={}, q_out={}, observed error={} ticks, proved epsilon={} ticks",
        rne_mat_plan.fraction_bits(),
        rne_mat_trace.fraction_bits(),
        rne_mat_trace.observed_error_ticks(),
        rne_mat_trace.epsilon_ticks()
    );
    println!(
        "Stage 8 reusable ResidualFfn range contract: q_in={}, |input ticks|<={}, q_out={}, ideal_bound={}, epsilon={} ticks",
        range_plan.source_fraction_bits(),
        range_plan.source_bound_ticks(),
        range_contract.fraction_bits(),
        range_contract.ideal_bound_ticks(),
        range_contract.epsilon_ticks()
    );
    println!(
        "Manifest SHA256 (pin externally): {}",
        sha256(&root.join("SHA256SUMS"))?
    );
    println!(
        "Lean proves ideal Int bounds, concrete dyadic RNE composition, and reusable bounded scalar transfer rules; LLVM/PTX and hardware remain trusted"
    );
    Ok(())
}
