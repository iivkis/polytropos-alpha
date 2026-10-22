//! Stage 8: conditional exact binary32-integer slice. A custom graph is
//! checked from Rust input-domain guards -> ideal-Int Lean range lemmas ->
//! baseline/fused CUDA Graph -> independent i64 oracle + SHA-256 artifacts.
//! Neither this example nor the Lean checker proves actual IEEE/LLVM/PTX.

use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

use verineural::{
    Domain, ExactFp32Plan, GpuEngine, GpuLaunchConfig, LeanProofEngine, NeuralGraph, ReluFusion,
    Shape, Tensor,
};

const FILES: [&str; 14] = [
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
    println!("Stage 8: saved Lean source and SHA-256 artifact manifest PASS");
    println!("Scope: conditional integer range proof, not machine FP32 certification");
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
    fs::write(root.join("numeric.lean"), plan.lean_source()?)?;
    LeanProofEngine::default().check_file(&root.join("numeric.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_matmul.lean"))?;

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
    let lhs = Tensor::new(Shape::new(vec![2, 3])?, vec![2.0, -1.0, 3.0, 0.0, 4.0, -2.0])?;
    let rhs = Tensor::new(Shape::new(vec![3, 2])?, vec![1.0, 2.0, -3.0, 1.0, 2.0, -1.0])?;
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
    fs::write(root.join("numeric_matmul.lean"), mat_plan.lean_source()?)?;
    LeanProofEngine::default().check_file(&root.join("numeric_matmul.lean"))?;
    for (label, engine) in [
        ("baseline_matmul", &baseline),
        ("fused_matmul", &fused),
    ] {
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
                    id.index(), tensor.shape().dims(),
                    tensor.values().iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>()
                )
            })
            .collect::<String>(),
    )?;
    fs::write(
        root.join("matmul_output.bits"),
        mat_expected.values().iter().map(|x| format!("{:08x}\n", x.to_bits())).collect::<String>(),
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
        "Manifest SHA256 (pin externally): {}",
        sha256(&root.join("SHA256SUMS"))?
    );
    println!(
        "Lean proves composed ideal Int bounds; IEEE RNE, LLVM/PTX and hardware remain trusted"
    );
    Ok(())
}
