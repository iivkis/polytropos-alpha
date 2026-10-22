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
    Domain, ExactFp32Plan, GpuEngine, GpuLaunchConfig, LeanProofEngine, MathContract, NeuralGraph,
    NeuralModule, NodeId, PortContract, ReluFusion, ResidualAdd, ResidualFfn, RneFp32Plan,
    RneRangePlan, RneThresholdSide, Shape, Tensor, audit_fp32_ptx,
    subnormal_add_exact_bits, subnormal_mul_rne_bits, subnormal_relu_bits,
};

const FILES: [&str; 63] = [
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
    "numeric_threshold.lean",
    "threshold_certificates.txt",
    "baseline_fma/graph.ll",
    "baseline_fma/graph.ptx",
    "fused_fma/graph.ll",
    "fused_fma/graph.ptx",
    "fma_probe_inputs.bits",
    "fma_probe_output.bits",
    "numeric_subnormal.lean",
    "baseline_subnormal/graph.ll",
    "baseline_subnormal/graph.ptx",
    "fused_subnormal/graph.ll",
    "fused_subnormal/graph.ptx",
    "subnormal_inputs.bits",
    "subnormal_output.bits",
    "subnormal_evidence.txt",
    "fp32_contraction_report.txt",
    "numeric_architecture.lean",
    "architecture_contract.txt",
    "numeric_architecture_dag.lean",
    "architecture_dag_contract.txt",
    "numeric_nested.lean",
    "nested_contract.txt",
    "numeric_nested_deep.lean",
    "nested_deep_contract.txt",
    "architecture_property.lean",
    "architecture_property_contract.txt",
    "architecture_property_nested.lean",
    "architecture_property_nested_contract.txt",
    "architecture_property_matmul_bias.lean",
    "architecture_property_matmul_bias_contract.txt",
];

const PTX_PROFILES: [&str; 12] = [
    "baseline",
    "fused",
    "baseline_matmul",
    "fused_matmul",
    "baseline_rne",
    "fused_rne",
    "baseline_rne_matmul",
    "fused_rne_matmul",
    "baseline_fma",
    "fused_fma",
    "baseline_subnormal",
    "fused_subnormal",
];

/// Recheck the actually saved PTX/LLVM pairs, not only the Rust generator.
/// Static opcode/rounding checks are not a PTX-to-SASS translation proof.
fn fp32_contraction_report(root: &Path) -> Result<String, Box<dyn Error>> {
    let mut report = String::from("scope=LLVM-inline-PTX-vs-emitted-PTX\n");
    report.push_str("guarantee=static-rounded-add-mul-opcodes-and-no-PTX-FMA\n");
    report.push_str("machine_fp32_sass_jit=trusted-not-proven\n");
    for profile in PTX_PROFILES {
        let dir = root.join(profile);
        let llvm = fs::read_to_string(dir.join("graph.ll"))?;
        let ptx = fs::read_to_string(dir.join("graph.ptx"))?;
        let audit = audit_fp32_ptx(&llvm, &ptx)?;
        report.push_str(&format!(
            "{profile}:add.rn.f32={} mul.rn.f32={} fma=0\n",
            audit.rounded_adds, audit.rounded_muls
        ));
    }
    Ok(report)
}

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
    LeanProofEngine::default().check_file(&root.join("numeric_threshold.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_subnormal.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_architecture.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_architecture_dag.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_nested.lean"))?;
    LeanProofEngine::default().check_file(&root.join("numeric_nested_deep.lean"))?;
    LeanProofEngine::default().check_file(&root.join("architecture_property.lean"))?;
    LeanProofEngine::default().check_file(&root.join("architecture_property_nested.lean"))?;
    LeanProofEngine::default().check_file(&root.join("architecture_property_matmul_bias.lean"))?;
    if fs::read_to_string(root.join("fp32_contraction_report.txt"))?
        != fp32_contraction_report(root)?
    {
        return Err(io::Error::other("saved Stage 8 FP32 PTX audit report mismatch").into());
    }
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

struct GlobalAddModule {
    contract: MathContract,
}

impl GlobalAddModule {
    fn new(shape: Shape) -> Self {
        let any = |shape| PortContract::new(shape, Domain::Any);
        Self {
            contract: MathContract::new(vec![any(shape.clone()), any(shape.clone())], any(shape)),
        }
    }
}

impl NeuralModule for GlobalAddModule {
    fn name(&self) -> &'static str {
        "global_add"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        graph.add(inputs[0], inputs[1])
    }
}

struct GlobalMulModule {
    contract: MathContract,
}

impl GlobalMulModule {
    fn new(shape: Shape) -> Self {
        let any = |shape| PortContract::new(shape, Domain::Any);
        Self {
            contract: MathContract::new(vec![any(shape.clone()), any(shape.clone())], any(shape)),
        }
    }
}

impl NeuralModule for GlobalMulModule {
    fn name(&self) -> &'static str {
        "global_mul"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        graph.mul(inputs[0], inputs[1])
    }
}

struct GlobalNonnegativeMatMul {
    contract: MathContract,
}

impl GlobalNonnegativeMatMul {
    fn new() -> Result<Self, verineural::NeuralError> {
        Ok(Self {
            contract: MathContract::new(
                vec![
                    PortContract::new(Shape::new(vec![2, 3])?, Domain::Any),
                    PortContract::new(Shape::new(vec![3, 2])?, Domain::Any),
                ],
                PortContract::new(Shape::new(vec![2, 2])?, Domain::Any),
            ),
        })
    }
}

impl NeuralModule for GlobalNonnegativeMatMul {
    fn name(&self) -> &'static str {
        "global_nonnegative_matmul"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        graph.matmul(inputs[0], inputs[1])
    }
}

struct GlobalNonnegativeBiasAdd {
    contract: MathContract,
}

impl GlobalNonnegativeBiasAdd {
    fn new() -> Result<Self, verineural::NeuralError> {
        Ok(Self {
            contract: MathContract::new(
                vec![
                    PortContract::new(Shape::new(vec![2, 2])?, Domain::Any),
                    PortContract::new(Shape::new(vec![2])?, Domain::Any),
                ],
                PortContract::new(Shape::new(vec![2, 2])?, Domain::Any),
            ),
        })
    }
}

impl NeuralModule for GlobalNonnegativeBiasAdd {
    fn name(&self) -> &'static str {
        "global_nonnegative_bias_add"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        graph.bias_add(inputs[0], inputs[1])
    }
}

struct NestedGlobalProperty {
    contract: MathContract,
}

impl NestedGlobalProperty {
    fn new(shape: Shape) -> Self {
        let any = |shape| PortContract::new(shape, Domain::Any);
        Self {
            contract: MathContract::new(
                vec![any(shape.clone()), any(shape.clone()), any(shape.clone())],
                any(shape),
            ),
        }
    }
}

impl NeuralModule for NestedGlobalProperty {
    fn name(&self) -> &'static str {
        "nested_global_property"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        let add = GlobalAddModule::new(Shape::new(vec![3])?);
        let mul = GlobalMulModule::new(Shape::new(vec![3])?);
        let left = graph.apply_module(&add, &[inputs[0], inputs[1]])?;
        let right = graph.apply_module(&mul, &[inputs[0], inputs[2]])?;
        graph.add(left, right)
    }
}

struct OuterGlobalProperty {
    contract: MathContract,
}

impl OuterGlobalProperty {
    fn new(shape: Shape) -> Self {
        Self {
            contract: NestedGlobalProperty::new(shape).contract,
        }
    }
}

impl NeuralModule for OuterGlobalProperty {
    fn name(&self) -> &'static str {
        "outer_global_property"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        graph.apply_module(&NestedGlobalProperty::new(Shape::new(vec![3])?), inputs)
    }
}

struct NestedResidualPair {
    contract: MathContract,
}

impl NestedResidualPair {
    fn new() -> Result<Self, verineural::NeuralError> {
        let x = Shape::new(vec![1, 2])?;
        let matrix = Shape::new(vec![2, 2])?;
        let bias = Shape::new(vec![2])?;
        let finite = |shape| PortContract::new(shape, Domain::Finite);
        Ok(Self {
            contract: MathContract::new(
                vec![
                    finite(x.clone()),
                    finite(matrix.clone()),
                    finite(bias.clone()),
                    finite(matrix.clone()),
                    finite(bias.clone()),
                    finite(matrix.clone()),
                    finite(bias.clone()),
                    finite(matrix),
                    finite(bias),
                ],
                PortContract::new(x, Domain::Any),
            ),
        })
    }
}

impl NeuralModule for NestedResidualPair {
    fn name(&self) -> &'static str {
        "nested_residual_pair"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        let block = ResidualFfn::new(1, 2, 2)?;
        let hidden = graph.apply_module(
            &block,
            &[inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]],
        )?;
        graph.apply_module(
            &block,
            &[hidden, inputs[5], inputs[6], inputs[7], inputs[8]],
        )
    }
}

struct DeepNestedWrapper {
    contract: MathContract,
}

impl DeepNestedWrapper {
    fn new() -> Result<Self, verineural::NeuralError> {
        Ok(Self {
            contract: NestedResidualPair::new()?.contract.clone(),
        })
    }
}

impl NeuralModule for DeepNestedWrapper {
    fn name(&self) -> &'static str {
        "deep_nested_wrapper"
    }

    fn contract(&self) -> &MathContract {
        &self.contract
    }

    fn build(
        &self,
        graph: &mut NeuralGraph,
        inputs: &[NodeId],
    ) -> Result<NodeId, verineural::NeuralError> {
        let pair = NestedResidualPair::new()?;
        graph.apply_module(&pair, inputs)
    }
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

    // Adversarial contraction case: separate RNE yields zero, but a
    // fused multiply-add leaves a positive 2^-46 residual after the bias.
    // The fused CUDA Graph kernel must preserve the *two* roundings.
    let a = f32::from_bits(0x3f80_0001);
    let separately_rounded_product = a * a;
    let fma_bias = -separately_rounded_product;
    if (separately_rounded_product + fma_bias).to_bits() != 0.0f32.to_bits()
        || a.mul_add(a, fma_bias).to_bits() == 0.0f32.to_bits()
    {
        return Err(io::Error::other("FMA-sensitive FP32 fixture is not discriminating").into());
    }
    let mut fma_graph = NeuralGraph::default();
    let fma_shape = Shape::new(vec![1, 1])?;
    let fma_x = fma_graph.input(fma_shape.clone(), Domain::Finite);
    let fma_y = fma_graph.input(fma_shape.clone(), Domain::Finite);
    let fma_b = fma_graph.input(Shape::new(vec![1])?, Domain::Finite);
    let fma_product = fma_graph.mul(fma_x, fma_y)?;
    let fma_biased = fma_graph.bias_add(fma_product, fma_b)?;
    let fma_result = fma_graph.relu(fma_biased)?;
    let fma_inputs = [
        (fma_x, Tensor::new(fma_shape.clone(), vec![a])?),
        (fma_y, Tensor::new(fma_shape, vec![a])?),
        (fma_b, Tensor::new(Shape::new(vec![1])?, vec![fma_bias])?),
    ];
    for (label, engine) in [("baseline_fma", &baseline), ("fused_fma", &fused)] {
        let program = engine.compile(&fma_graph, fma_result)?;
        if label == "fused_fma" && program.removed_kernel_count() != 2 {
            return Err(io::Error::other("FMA probe did not fuse Mul/BiasAdd/ReLU").into());
        }
        let output = engine.execute(&program, &fma_inputs, &root.join(label))?;
        if output.values().len() != 1 || output.values()[0].to_bits() != 0.0f32.to_bits() {
            return Err(io::Error::other(format!(
                "{label} introduced an unexpected FP32 contraction or rounding difference"
            ))
            .into());
        }
        println!("Stage 8 {label}: FMA-sensitive separate-rounding GPU probe matches +0 bits");
    }
    fs::write(
        root.join("fma_probe_inputs.bits"),
        format!(
            "a={:08x}\nb={:08x}\nbias={:08x}\n",
            a.to_bits(),
            a.to_bits(),
            fma_bias.to_bits()
        ),
    )?;
    fs::write(root.join("fma_probe_output.bits"), "00000000\n")?;

    // Gradual-underflow slice: the reference path uses integer significands
    // and the 2^-149 subnormal lattice. It never executes host f32 arithmetic
    // to obtain expected bits. Exact results exercise input denormals, output
    // denormals, halfway ties-to-even, normal/subnormal crossing, and ReLU.
    //
    // These concrete tests do NOT enlarge the old finite-normal RneFp32Plan
    // Lean proof domain. The new Lean evidence proves lattice facts only.
    let lhs_bits: [u32; 8] = [
        0x0080_0000, // min normal * 1/2 => subnormal
        0x0080_0000, // min normal * 2^-24 => halfway zero
        0x0000_0001, // min subnormal * 1/2 => halfway zero
        0x0000_0003, // 3 min subnormal * 1/2 => 2 ticks, ties-to-even
        0x0000_0001, // min subnormal * 1 => 1 tick, then bias
        0x0080_0001, // midpoint after multiplication, even tick wins
        0x8000_0001, // negative subnormal input -> ReLU +0
        0x007f_ffff, // largest subnormal plus bias -> min normal
    ];
    let rhs_bits: [u32; 8] = [
        0x3f00_0000, 0x3380_0000, 0x3f00_0000, 0x3f00_0000,
        0x3f80_0000, 0x3f00_0000, 0x3f80_0000, 0x3f80_0000,
    ];
    let bias_bits: [u32; 8] = [0, 0, 0, 0, 1, 0, 0, 1];
    let expected_fixed: [u32; 8] = [
        0x0040_0000, 0, 0, 2, 2, 0x0040_0000, 0, 0x0080_0000,
    ];
    let mut expected_subnormal = Vec::with_capacity(lhs_bits.len());
    let mut subnormal_evidence = String::from(
        "model=integer-significand-exact-product-and-subnormal-RNE-lattice\n\
         fractional_ulp=2^-149\n\
         scope=finite-binary32-mul-to-subnormal-or-smallest-normal,exact-subnormal-add,ReLU\n\
         lean_model=integer-half-tick-ties-even-examples-not-full-IEEE\n\
         observed_gpu=baseline-and-fused-CUDA-Graph-bitwise\n\
         machine_fp32_sass_jit=trusted-not-proven\n",
    );
    for lane in 0..lhs_bits.len() {
        let product = subnormal_mul_rne_bits(lhs_bits[lane], rhs_bits[lane])
            .ok_or_else(|| io::Error::other("subnormal product outside integer oracle domain"))?;
        let biased = subnormal_add_exact_bits(product, bias_bits[lane])
            .ok_or_else(|| io::Error::other("subnormal bias outside integer oracle domain"))?;
        let expected = subnormal_relu_bits(biased)
            .ok_or_else(|| io::Error::other("subnormal ReLU outside integer oracle domain"))?;
        if expected != expected_fixed[lane] {
            return Err(io::Error::other("subnormal integer oracle fixture regression").into());
        }
        expected_subnormal.push(expected);
        subnormal_evidence.push_str(&format!(
            "lane={lane} lhs={:08x} rhs={:08x} bias={:08x} product={product:08x} expected={expected:08x}\n",
            lhs_bits[lane], rhs_bits[lane], bias_bits[lane]
        ));
    }
    let mut subnormal_graph = NeuralGraph::default();
    let subnormal_shape = Shape::new(vec![1, lhs_bits.len()])?;
    let subnormal_x = subnormal_graph.input(subnormal_shape.clone(), Domain::Finite);
    let subnormal_y = subnormal_graph.input(subnormal_shape.clone(), Domain::Finite);
    let subnormal_bias = subnormal_graph.input(Shape::new(vec![lhs_bits.len()])?, Domain::Finite);
    let subnormal_product = subnormal_graph.mul(subnormal_x, subnormal_y)?;
    let subnormal_biased = subnormal_graph.bias_add(subnormal_product, subnormal_bias)?;
    let subnormal_output = subnormal_graph.relu(subnormal_biased)?;
    let subnormal_inputs = [
        (
            subnormal_x,
            Tensor::new(
                subnormal_shape.clone(),
                lhs_bits.iter().map(|&bits| f32::from_bits(bits)).collect(),
            )?,
        ),
        (
            subnormal_y,
            Tensor::new(
                subnormal_shape,
                rhs_bits.iter().map(|&bits| f32::from_bits(bits)).collect(),
            )?,
        ),
        (
            subnormal_bias,
            Tensor::new(
                Shape::new(vec![bias_bits.len()])?,
                bias_bits.iter().map(|&bits| f32::from_bits(bits)).collect(),
            )?,
        ),
    ];
    for (label, engine) in [
        ("baseline_subnormal", &baseline),
        ("fused_subnormal", &fused),
    ] {
        let program = engine.compile(&subnormal_graph, subnormal_output)?;
        if label == "fused_subnormal" && program.removed_kernel_count() != 2 {
            return Err(io::Error::other("subnormal Mul/BiasAdd/ReLU fusion missing").into());
        }
        let output = engine.execute(&program, &subnormal_inputs, &root.join(label))?;
        if output.values().len() != expected_subnormal.len() {
            return Err(io::Error::other("subnormal GPU output length mismatch").into());
        }
        for (lane, (&observed, &expected)) in output
            .values()
            .iter()
            .zip(expected_subnormal.iter())
            .enumerate()
        {
            if observed.to_bits() != expected {
                return Err(io::Error::other(format!(
                    "{label}: subnormal gradual-underflow mismatch at lane {lane}: expected={expected:08x}, observed={:08x}",
                    observed.to_bits()
                )).into());
            }
        }
        println!(
            "Stage 8 {label}: {} gradual-underflow, RNE tie and FTZ-sensitive GPU values match integer oracle",
            expected_subnormal.len()
        );
    }
    fs::write(
        root.join("subnormal_inputs.bits"),
        (0..lhs_bits.len())
            .map(|lane| {
                format!(
                    "lane={lane} lhs={:08x} rhs={:08x} bias={:08x}\n",
                    lhs_bits[lane], rhs_bits[lane], bias_bits[lane]
                )
            })
            .collect::<String>(),
    )?;
    fs::write(
        root.join("subnormal_output.bits"),
        expected_subnormal
            .iter()
            .map(|bits| format!("{bits:08x}\n"))
            .collect::<String>(),
    )?;
    fs::write(root.join("subnormal_evidence.txt"), &subnormal_evidence)?;
    write_checked_lean(
        &root.join("numeric_subnormal.lean"),
        include_str!("../../../lean/VeriNeural/Subnormal.lean"),
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
    write_checked_lean(&root.join("numeric_range.lean"), &range_plan.lean_source()?)?;
    fs::write(
        root.join("range_contract.txt"),
        format!(
            "module={}\nglobal_theorem=module_0_global_contract\n\
             source_fraction_bits={}\nsource_bound_ticks={}\n\
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

    // Synthetic, exactly representable pre-ReLU observations.
    // Classification remains conditional on the node's ApproxContract.
    let mut threshold_graph = NeuralGraph::default();
    let threshold_shape = Shape::new(vec![1])?;
    let threshold_x = threshold_graph.input(threshold_shape.clone(), Domain::Finite);
    let threshold_y = threshold_graph.input(threshold_shape, Domain::Finite);
    let threshold_sum = threshold_graph.add(threshold_x, threshold_y)?;
    let threshold_relu = threshold_graph.relu(threshold_sum)?;
    let threshold_plan = RneRangePlan::compile(&threshold_graph, threshold_relu, 10, 16)?;
    let threshold_contract = threshold_plan.node_contract(threshold_sum)?;
    if threshold_contract.ideal_bound_ticks() != 32 || threshold_contract.epsilon_ticks() != 1 {
        return Err(io::Error::other("unexpected Stage 8 threshold budget").into());
    }
    let positive = threshold_plan.classify_relu_input(threshold_relu, 2.0 / 1024.0)?;
    let nonpositive = threshold_plan.classify_relu_input(threshold_relu, -1.0 / 1024.0)?;
    let ambiguous = threshold_plan.classify_relu_input(threshold_relu, 0.0)?;
    if positive.side() != RneThresholdSide::Positive
        || nonpositive.side() != RneThresholdSide::NonPositive
        || ambiguous.side() != RneThresholdSide::Indeterminate
        || positive.ideal_interval_ticks() != (1, 3)
        || nonpositive.ideal_interval_ticks() != (-2, 0)
        || ambiguous.ideal_interval_ticks() != (-1, 1)
    {
        return Err(io::Error::other("unexpected Stage 8 threshold decision").into());
    }

    let mut threshold_source = threshold_plan.lean_source()?;
    if !threshold_source.contains("theorem node_3_positive_branch")
        || !threshold_source.contains("theorem node_3_nonpositive_branch")
    {
        return Err(io::Error::other("missing conditional ReLU branch proofs").into());
    }
    threshold_source.push_str(&format!(
        r#"
namespace VeriNeural.Generated.RneThreshold

theorem positive_observation {{ideal : Int}}
    (h : VeriNeural.ApproxContract {bound} {epsilon} ideal ({positive_ticks} : Int)) :
    max ideal 0 = ideal ∧ max ({positive_ticks} : Int) 0 = ({positive_ticks} : Int) := by
  exact VeriNeural.Generated.RneRange.node_{relu}_positive_branch h (by decide)

theorem nonpositive_observation {{ideal : Int}}
    (h : VeriNeural.ApproxContract {bound} {epsilon} ideal ({negative_ticks} : Int)) :
    max ideal 0 = 0 ∧ max ({negative_ticks} : Int) 0 = 0 := by
  exact VeriNeural.Generated.RneRange.node_{relu}_nonpositive_branch h (by decide)

-- The same actual operand allows opposite ideal branch choices.
theorem ambiguous_has_opposite_models :
    VeriNeural.ApproxContract {bound} {epsilon} (1 : Int) 0 ∧
    VeriNeural.ApproxContract {bound} {epsilon} (-1 : Int) 0 := by
  unfold VeriNeural.ApproxContract VeriNeural.ApproxWithin VeriNeural.Within
  decide

end VeriNeural.Generated.RneThreshold
"#,
        bound = threshold_contract.ideal_bound_ticks(),
        epsilon = threshold_contract.epsilon_ticks(),
        positive_ticks = positive.observed_ticks(),
        negative_ticks = nonpositive.observed_ticks(),
        relu = threshold_relu.index(),
    ));
    write_checked_lean(&root.join("numeric_threshold.lean"), &threshold_source)?;
    fs::write(
        root.join("threshold_certificates.txt"),
        format!(
            "model=dyadic-rne-int\nobservations=synthetic-not-gpu\nrelu_node={}\ninput_node={}\n\
             q={}\nepsilon_ticks={}\n\
             positive={}: {:?} {:?}\nnonpositive={}: {:?} {:?}\n\
             ambiguous={}: {:?} {:?}\n\
             proofs=positive_observation,nonpositive_observation,ambiguous_has_opposite_models\n\
             assumption=pre-relu-ApproxContract\nmachine_fp32_gpu=trusted-not-proven\n",
            threshold_relu.index(),
            threshold_sum.index(),
            positive.fraction_bits(),
            positive.epsilon_ticks(),
            positive.observed_ticks(),
            positive.ideal_interval_ticks(),
            positive.side(),
            nonpositive.observed_ticks(),
            nonpositive.ideal_interval_ticks(),
            nonpositive.side(),
            ambiguous.observed_ticks(),
            ambiguous.ideal_interval_ticks(),
            ambiguous.side(),
        ),
    )?;

    // Architecture-level composition: the second ResidualFfn consumes the
    // first module's output contract. The generated Lean theorem must compose
    // module_0_global_contract -> module_1_global_contract instead of flattening
    // both blocks into one proof body.
    let mut architecture_graph = NeuralGraph::default();
    let arch_x = architecture_graph.input(Shape::new(vec![1, 2])?, Domain::Finite);
    let arch_w1a = architecture_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let arch_b1a = architecture_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let arch_w2a = architecture_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let arch_b2a = architecture_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let arch_w1b = architecture_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let arch_b1b = architecture_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let arch_w2b = architecture_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let arch_b2b = architecture_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let arch_block = ResidualFfn::new(1, 2, 2)?;
    let arch_hidden = architecture_graph.apply_module(
        &arch_block,
        &[arch_x, arch_w1a, arch_b1a, arch_w2a, arch_b2a],
    )?;
    let arch_output = architecture_graph.apply_module(
        &arch_block,
        &[arch_hidden, arch_w1b, arch_b1b, arch_w2b, arch_b2b],
    )?;
    let architecture_plan = RneRangePlan::compile(&architecture_graph, arch_output, 10, 16)?;
    let architecture_contract = architecture_plan.output_contract();
    if architecture_contract.fraction_bits() != 50
        || architecture_contract.ideal_bound_ticks() != 53_928_163_016_704
        || architecture_contract.epsilon_ticks() != 184_963_072
    {
        return Err(io::Error::other("unexpected Stage 8 architecture RNE contract").into());
    }
    let architecture_source = architecture_plan.lean_source()?;
    if !architecture_source.contains("theorem architecture_global_contract")
        || !architecture_source.contains("have arch_module_0 := module_0_global_contract")
        || !architecture_source.contains("have arch_module_1 := module_1_global_contract")
    {
        return Err(io::Error::other("missing Stage 8 architecture composition theorem").into());
    }
    write_checked_lean(
        &root.join("numeric_architecture.lean"),
        &architecture_source,
    )?;
    fs::write(
        root.join("architecture_contract.txt"),
        format!(
            "modules=2\nglobal_theorem=architecture_global_contract\n\
             source_fraction_bits={}\nsource_bound_ticks={}\n\
             output_fraction_bits={}\nideal_bound_ticks={}\nepsilon_ticks={}\n\
             actual_bound_ticks={}\nepsilon={:.12e}\n",
            architecture_plan.source_fraction_bits(),
            architecture_plan.source_bound_ticks(),
            architecture_contract.fraction_bits(),
            architecture_contract.ideal_bound_ticks(),
            architecture_contract.epsilon_ticks(),
            architecture_contract.actual_bound_ticks(),
            architecture_contract.epsilon(),
        ),
    )?;

    // Branching architecture DAG: two independent ResidualFfn branches fan in
    // through a checked ResidualAdd module. The architecture theorem must consume
    // both producer module contracts before the join.
    let mut dag_graph = NeuralGraph::default();
    let dag_x = dag_graph.input(Shape::new(vec![1, 2])?, Domain::Finite);
    let dag_w1a = dag_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let dag_b1a = dag_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let dag_w2a = dag_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let dag_b2a = dag_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let dag_w1b = dag_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let dag_b1b = dag_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let dag_w2b = dag_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let dag_b2b = dag_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let dag_block = ResidualFfn::new(1, 2, 2)?;
    let dag_left =
        dag_graph.apply_module(&dag_block, &[dag_x, dag_w1a, dag_b1a, dag_w2a, dag_b2a])?;
    let dag_right =
        dag_graph.apply_module(&dag_block, &[dag_x, dag_w1b, dag_b1b, dag_w2b, dag_b2b])?;
    let dag_join = ResidualAdd::new(Shape::new(vec![1, 2])?);
    let dag_output = dag_graph.apply_module(&dag_join, &[dag_left, dag_right])?;
    let dag_plan = RneRangePlan::compile(&dag_graph, dag_output, 10, 16)?;
    let dag_contract = dag_plan.output_contract();
    if dag_contract.fraction_bits() != 30
        || dag_contract.ideal_bound_ticks() != 68_190_208
        || dag_contract.epsilon_ticks() != 348
    {
        return Err(io::Error::other("unexpected Stage 8 architecture DAG contract").into());
    }
    let dag_source = dag_plan.lean_source()?;
    if !dag_source
        .contains("-- Architecture-level numerical composition over the reachable module DAG.")
        || !dag_source.contains("have arch_module_0 := module_0_global_contract")
        || !dag_source.contains("have arch_module_1 := module_1_global_contract")
        || !dag_source.contains("have arch_module_2 := module_2_global_contract")
    {
        return Err(io::Error::other("missing Stage 8 fork/fan-in architecture theorem").into());
    }
    write_checked_lean(&root.join("numeric_architecture_dag.lean"), &dag_source)?;
    fs::write(
        root.join("architecture_dag_contract.txt"),
        format!(
            "modules=3\nshape=fork-fan-in\nglobal_theorem=architecture_global_contract\n\
             source_fraction_bits={}\nsource_bound_ticks={}\n\
             output_fraction_bits={}\nideal_bound_ticks={}\nepsilon_ticks={}\n\
             actual_bound_ticks={}\nepsilon={:.12e}\n",
            dag_plan.source_fraction_bits(),
            dag_plan.source_bound_ticks(),
            dag_contract.fraction_bits(),
            dag_contract.ideal_bound_ticks(),
            dag_contract.epsilon_ticks(),
            dag_contract.actual_bound_ticks(),
            dag_contract.epsilon(),
        ),
    )?;

    // Nested hierarchy: the outer user module invokes two ResidualFfn children.
    // Its generated theorem must reuse the two child whole-module theorems.
    let mut nested_graph = NeuralGraph::default();
    let nested_x = nested_graph.input(Shape::new(vec![1, 2])?, Domain::Finite);
    let nested_w1a = nested_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let nested_b1a = nested_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let nested_w2a = nested_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let nested_b2a = nested_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let nested_w1b = nested_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let nested_b1b = nested_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let nested_w2b = nested_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let nested_b2b = nested_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let nested_module = NestedResidualPair::new()?;
    let nested_output = nested_graph.apply_module(
        &nested_module,
        &[
            nested_x, nested_w1a, nested_b1a, nested_w2a, nested_b2a, nested_w1b, nested_b1b,
            nested_w2b, nested_b2b,
        ],
    )?;
    let nested_plan = RneRangePlan::compile(&nested_graph, nested_output, 10, 16)?;
    let nested_contract = nested_plan.output_contract();
    if nested_contract.fraction_bits() != 50
        || nested_contract.ideal_bound_ticks() != 53_928_163_016_704
        || nested_contract.epsilon_ticks() != 184_963_072
    {
        return Err(io::Error::other("unexpected Stage 8 nested-module contract").into());
    }
    let nested_source = nested_plan.lean_source()?;
    if !nested_source.contains("theorem module_2_global_contract")
        || !nested_source.contains("have nested_module_0 := module_0_global_contract")
        || !nested_source.contains("have nested_module_1 := module_1_global_contract")
    {
        return Err(io::Error::other("missing Stage 8 nested whole-module composition").into());
    }
    write_checked_lean(&root.join("numeric_nested.lean"), &nested_source)?;
    fs::write(
        root.join("nested_contract.txt"),
        format!(
            "outer_module=nested_residual_pair\nchild_modules=2\n\
             theorem=module_2_global_contract\nsource_fraction_bits={}\n\
             source_bound_ticks={}\noutput_fraction_bits={}\n\
             ideal_bound_ticks={}\nepsilon_ticks={}\nactual_bound_ticks={}\n\
             epsilon={:.12e}\n",
            nested_plan.source_fraction_bits(),
            nested_plan.source_bound_ticks(),
            nested_contract.fraction_bits(),
            nested_contract.ideal_bound_ticks(),
            nested_contract.epsilon_ticks(),
            nested_contract.actual_bound_ticks(),
            nested_contract.epsilon(),
        ),
    )?;

    // Recursive hierarchy: a wrapper contains NestedResidualPair, which itself
    // contains two ResidualFfn modules. The outer theorem must consume only the
    // direct child theorem; the child theorem recursively consumes its own children.
    let mut deep_graph = NeuralGraph::default();
    let deep_x = deep_graph.input(Shape::new(vec![1, 2])?, Domain::Finite);
    let deep_w1a = deep_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let deep_b1a = deep_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let deep_w2a = deep_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let deep_b2a = deep_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let deep_w1b = deep_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let deep_b1b = deep_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let deep_w2b = deep_graph.input(Shape::new(vec![2, 2])?, Domain::Finite);
    let deep_b2b = deep_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let deep_module = DeepNestedWrapper::new()?;
    let deep_output = deep_graph.apply_module(
        &deep_module,
        &[
            deep_x, deep_w1a, deep_b1a, deep_w2a, deep_b2a, deep_w1b, deep_b1b, deep_w2b, deep_b2b,
        ],
    )?;
    let deep_plan = RneRangePlan::compile(&deep_graph, deep_output, 10, 16)?;
    let deep_contract = deep_plan.output_contract();
    if deep_contract.fraction_bits() != 50
        || deep_contract.ideal_bound_ticks() != 53_928_163_016_704
        || deep_contract.epsilon_ticks() != 184_963_072
    {
        return Err(io::Error::other("unexpected Stage 8 deep nested contract").into());
    }
    let deep_source = deep_plan.lean_source()?;
    let pair_section = deep_source
        .split("theorem module_2_global_contract")
        .nth(1)
        .and_then(|rest| rest.split("theorem module_3_global_contract").next())
        .ok_or_else(|| io::Error::other("missing Stage 8 nested pair theorem section"))?;
    let outer_section = deep_source
        .split("theorem module_3_global_contract")
        .nth(1)
        .ok_or_else(|| io::Error::other("missing Stage 8 deep wrapper theorem"))?;
    if !pair_section.contains("have nested_module_0 := module_0_global_contract")
        || !pair_section.contains("have nested_module_1 := module_1_global_contract")
        || !outer_section.contains("have nested_module_2 := module_2_global_contract")
        || outer_section.contains("have nested_module_0 := module_0_global_contract")
        || outer_section.contains("have nested_module_1 := module_1_global_contract")
    {
        return Err(io::Error::other("missing recursive Stage 8 nested theorem reuse").into());
    }
    write_checked_lean(&root.join("numeric_nested_deep.lean"), &deep_source)?;
    fs::write(
        root.join("nested_deep_contract.txt"),
        format!(
            "outer_module=deep_nested_wrapper\ndirect_child_modules=1\nrecursive_depth=3\n\
             theorem=module_3_global_contract\nsource_fraction_bits={}\n\
             source_bound_ticks={}\noutput_fraction_bits={}\n\
             ideal_bound_ticks={}\nepsilon_ticks={}\nactual_bound_ticks={}\n\
             epsilon={:.12e}\n",
            deep_plan.source_fraction_bits(),
            deep_plan.source_bound_ticks(),
            deep_contract.fraction_bits(),
            deep_contract.ideal_bound_ticks(),
            deep_contract.epsilon_ticks(),
            deep_contract.actual_bound_ticks(),
            deep_contract.epsilon(),
        ),
    )?;

    // Explicit architecture-level functional property in the ideal-Int model.
    // Three user-declared nonnegative source assumptions flow through a module
    // fork/fan-in DAG: Add(x,y), Mul(x,z), then Add(left,right).
    let mut property_graph = NeuralGraph::default();
    let property_shape = Shape::new(vec![3])?;
    let property_x = property_graph.input(property_shape.clone(), Domain::Finite);
    let property_y = property_graph.input(property_shape.clone(), Domain::Finite);
    let property_z = property_graph.input(property_shape.clone(), Domain::Finite);
    let property_add = GlobalAddModule::new(property_shape.clone());
    let property_mul = GlobalMulModule::new(property_shape);
    let property_left = property_graph.apply_module(&property_add, &[property_x, property_y])?;
    let property_right = property_graph.apply_module(&property_mul, &[property_x, property_z])?;
    let property_output =
        property_graph.apply_module(&property_add, &[property_left, property_right])?;
    let property_program = LeanProofEngine::default().generate_global_nonnegative(
        &property_graph,
        property_output,
        &[property_x, property_y, property_z],
    )?;
    let property_source = property_program.source();
    if !property_source.contains("theorem architecture_global_nonnegative")
        || !property_source.contains("have arch_module_0 := module_0_global_nonnegative")
        || !property_source.contains("have arch_module_1 := module_1_global_nonnegative")
        || !property_source.contains("have arch_module_2 := module_2_global_nonnegative")
        || !property_source.contains("exact arch_module_2")
    {
        return Err(
            io::Error::other("missing Stage 8 global architecture nonnegative proof").into(),
        );
    }
    write_checked_lean(&root.join("architecture_property.lean"), property_source)?;
    fs::write(
        root.join("architecture_property_contract.txt"),
        format!(
            "property=output_nonnegative\nmodel=ideal-int\nmodules=3\nshape=fork-fan-in\n\
             assumptions=node:{},node:{},node:{}\n\
             theorem=architecture_global_nonnegative\nfp32_gpu_claim=false\n",
            property_x.index(),
            property_y.index(),
            property_z.index(),
        ),
    )?;

    // Recursive global ideal-Int property: the architecture has one outer
    // wrapper, one nested DAG module, and two leaf Add/Mul child modules.
    // Each theorem must consume only its immediate child boundary proofs.
    let mut nested_property_graph = NeuralGraph::default();
    let nested_property_shape = Shape::new(vec![3])?;
    let nested_property_x =
        nested_property_graph.input(nested_property_shape.clone(), Domain::Finite);
    let nested_property_y =
        nested_property_graph.input(nested_property_shape.clone(), Domain::Finite);
    let nested_property_z =
        nested_property_graph.input(nested_property_shape.clone(), Domain::Finite);
    let nested_property_output = nested_property_graph.apply_module(
        &OuterGlobalProperty::new(nested_property_shape),
        &[nested_property_x, nested_property_y, nested_property_z],
    )?;
    let nested_property_program = LeanProofEngine::default().generate_global_nonnegative(
        &nested_property_graph,
        nested_property_output,
        &[nested_property_x, nested_property_y, nested_property_z],
    )?;
    let nested_property_source = nested_property_program.source();
    let nested_inner = nested_property_source
        .split("theorem module_2_global_nonnegative")
        .nth(1)
        .and_then(|remainder| {
            remainder
                .split("theorem module_3_global_nonnegative")
                .next()
        })
        .ok_or_else(|| io::Error::other("missing nested property theorem"))?;
    let nested_outer = nested_property_source
        .split("theorem module_3_global_nonnegative")
        .nth(1)
        .ok_or_else(|| io::Error::other("missing outer property theorem"))?;
    if !nested_inner.contains("module_0_global_nonnegative")
        || !nested_inner.contains("module_1_global_nonnegative")
        || !nested_outer.contains("module_2_global_nonnegative")
        || nested_outer.contains("module_0_global_nonnegative")
        || nested_outer.contains("module_1_global_nonnegative")
        || !nested_property_source.contains("have arch_module_3 := module_3_global_nonnegative")
    {
        return Err(io::Error::other("invalid recursive global property composition").into());
    }
    write_checked_lean(
        &root.join("architecture_property_nested.lean"),
        nested_property_source,
    )?;
    fs::write(
        root.join("architecture_property_nested_contract.txt"),
        format!(
            "property=output_nonnegative\nmodel=ideal-int\nmodules=4\nrecursive_depth=3\n\
             hierarchy=outer_global_property>nested_global_property>global_add,global_mul\n\
             assumptions=node:{},node:{},node:{}\n\
             theorem=architecture_global_nonnegative\nfp32_gpu_claim=false\n",
            nested_property_x.index(),
            nested_property_y.index(),
            nested_property_z.index(),
        ),
    )?;

    // Nonnegative ideal-Int MatMul reduction followed by row-major BiasAdd
    // broadcast, with explicit input, matrix-weight and bias assumptions.
    let mut property_matmul_graph = NeuralGraph::default();
    let property_matmul_x = property_matmul_graph.input(Shape::new(vec![2, 3])?, Domain::Finite);
    let property_matmul_w = property_matmul_graph.input(Shape::new(vec![3, 2])?, Domain::Finite);
    let property_matmul_b = property_matmul_graph.input(Shape::new(vec![2])?, Domain::Finite);
    let property_matmul_product = property_matmul_graph.apply_module(
        &GlobalNonnegativeMatMul::new()?,
        &[property_matmul_x, property_matmul_w],
    )?;
    let property_matmul_output = property_matmul_graph.apply_module(
        &GlobalNonnegativeBiasAdd::new()?,
        &[property_matmul_product, property_matmul_b],
    )?;
    let property_matmul_program = LeanProofEngine::default().generate_global_nonnegative(
        &property_matmul_graph,
        property_matmul_output,
        &[property_matmul_x, property_matmul_w, property_matmul_b],
    )?;
    let property_matmul_source = property_matmul_program.source();
    if !property_matmul_source.contains("VeriNeural.matmul_nonnegative (m := 2) (k := 3) (n := 2)")
        || !property_matmul_source.contains("VeriNeural.biasAdd_nonnegative (m := 2) (n := 2)")
        || !property_matmul_source.contains("have arch_module_0 := module_0_global_nonnegative")
        || !property_matmul_source.contains("have arch_module_1 := module_1_global_nonnegative")
    {
        return Err(io::Error::other("missing MatMul/BiasAdd global nonnegative theorem").into());
    }
    write_checked_lean(
        &root.join("architecture_property_matmul_bias.lean"),
        property_matmul_source,
    )?;
    fs::write(
        root.join("architecture_property_matmul_bias_contract.txt"),
        format!(
            "property=output_nonnegative\nmodel=ideal-int\nmodules=2\n\
             operations=matmul,bias_add\nmatmul=m:2,k:3,n:2\n\
             bias_broadcast=rows:2,channels:2\n\
             assumptions=node:{},node:{},node:{}\n\
             theorem=architecture_global_nonnegative\nfp32_gpu_claim=false\n",
            property_matmul_x.index(),
            property_matmul_w.index(),
            property_matmul_b.index(),
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
    fs::write(
        root.join("fp32_contraction_report.txt"),
        fp32_contraction_report(root)?,
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
        "Stage 8 architecture composition: modules=2, q_out={}, ideal_bound={}, epsilon={} ticks",
        architecture_contract.fraction_bits(),
        architecture_contract.ideal_bound_ticks(),
        architecture_contract.epsilon_ticks()
    );
    println!(
        "Stage 8 architecture DAG: fork/fan-in modules=3, q_out={}, ideal_bound={}, epsilon={} ticks",
        dag_contract.fraction_bits(),
        dag_contract.ideal_bound_ticks(),
        dag_contract.epsilon_ticks()
    );
    println!(
        "Stage 8 nested hierarchy: child_modules=2, q_out={}, ideal_bound={}, epsilon={} ticks",
        nested_contract.fraction_bits(),
        nested_contract.ideal_bound_ticks(),
        nested_contract.epsilon_ticks()
    );
    println!(
        "Stage 8 deep nested hierarchy: recursive_depth=3, q_out={}, ideal_bound={}, epsilon={} ticks",
        deep_contract.fraction_bits(),
        deep_contract.ideal_bound_ticks(),
        deep_contract.epsilon_ticks()
    );
    println!(
        "Stage 8 global property: architecture_global_nonnegative over a 3-module Add/Mul fan-in DAG (ideal-Int only)"
    );
    println!(
        "Stage 8 nested global property: architecture_global_nonnegative with recursive_depth=3 and child theorem reuse (ideal-Int only)"
    );
    println!(
        "Stage 8 MatMul/BiasAdd global property: nonnegative 2x3 * 3x2 with row-wise bias broadcast (ideal-Int only)"
    );
    println!(
        "Stage 8 ReLU threshold margins: conditional positive/nonpositive proofs plus explicit ambiguity witness"
    );
    println!(
        "Stage 8 FP32 PTX audit: explicit separate RNE mul/add, no emitted FMA/FTZ (LLVM/PTX static check only)"
    );
    println!("Stage 8 gradual underflow: 8 signed/subnormal lanes in both baseline/fused kernels; integer bit oracle plus Lean half-tick model");
    println!(
        "Manifest SHA256 (pin externally): {}",
        sha256(&root.join("SHA256SUMS"))?
    );
    println!(
        "Lean proves ideal Int bounds, concrete dyadic RNE composition, reusable bounded transfer rules, whole-module theorems, fork/fan-in architecture DAG composition, recursive nested-module theorem reuse through depth 3, and global ideal-Int nonnegativity for Add/Mul/MatMul/BiasAdd and nested modules; ReLU margins are conditional with explicit ambiguity witness; a PTX opcode audit, FMA probe, and subnormal bit-oracle GPU probes have been checked; integer half-tick Lean lemmas cover only selected boundary examples; LLVM/PTX-to-machine translation and hardware remain trusted"
    );
    Ok(())
}
