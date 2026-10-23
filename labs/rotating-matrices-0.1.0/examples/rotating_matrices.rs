//! Run with `task lab-rotating` or `task train-lab-rotating -- <updates>`.
use rotating_matrices_lab::{
    ACTIONS, ArtifactWorkspace, Dataset, Dictionary, OP_COUNT, Parameters, Rng, Runner,
    TrainingConfig, architecture_model, constant_baseline, evaluate, evaluate_with_route,
    operation, train,
};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use verineural::CertifiedBuildConfig;

const ARCHITECTURE_FILES: [&str; 11] = [
    "semantic.ir",
    "architecture.lean",
    "architecture.dot",
    "architecture.png",
    "architecture.audit.txt",
    "architecture.graph.ll",
    "architecture.graph.ptx",
    "architecture.translation.plan",
    "architecture.translation.report",
    "certification.txt",
    "identity.txt",
];

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let output = Command::new("sha256sum").arg(path).output()?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let text = String::from_utf8(output.stdout)?;
    let digest = text.split_whitespace().next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("invalid SHA-256 for {}", path.display()).into());
    }
    Ok(digest.to_owned())
}

fn write_architecture_manifest(root: &Path) -> Result<(), Box<dyn Error>> {
    let mut manifest = String::new();
    for name in ARCHITECTURE_FILES {
        manifest.push_str(&format!("{}  {name}\n", sha256(&root.join(name))?));
    }
    fs::write(root.join("SHA256SUMS"), manifest)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "--inspect".into());
    let data = Dataset::new();
    let dictionary = Dictionary::fit(&data.train);
    let architecture = architecture_model()?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts");

    match mode.as_str() {
        "--inspect" => {
            let mut inspect =
                String::from("two canonical 64x64 cores, R0/R90/R180/R270 => 8 actions\n");
            println!("two canonical 64x64 cores, R0/R90/R180/R270 => 8 actions");
            println!("UTF-8 byte vocabulary={} (train-only)", dictionary.len());
            inspect.push_str(&format!(
                "UTF-8 byte vocabulary={} (train-only)\n",
                dictionary.len()
            ));
            println!(
                "40 operations; train={} validation={}",
                data.train.len(),
                data.validation.len()
            );
            inspect.push_str(&format!(
                "40 operations; train={} validation={}\n",
                data.train.len(),
                data.validation.len()
            ));
            fs::create_dir_all(&root)?;
            println!(
                "architecture: {} Semantic IR nodes, {} modules, {} outputs",
                architecture.graph().nodes().len(),
                architecture.graph().modules().len(),
                architecture.outputs().len()
            );
            inspect.push_str(&format!(
                "architecture: {} Semantic IR nodes, {} modules, {} outputs\n",
                architecture.graph().nodes().len(),
                architecture.graph().modules().len(),
                architecture.outputs().len()
            ));
            fs::write(
                root.join("semantic.ir"),
                architecture.graph().canonical_ir(),
            )?;
            for &(op, a, b) in &[(0, 10, 9), (3, 1, 0)] {
                let case = operation(op, a, b)?;
                println!("x={:?} y={}", case.text, case.target);
                inspect.push_str(&format!("x={:?} y={}\n", case.text, case.target));
            }
            println!("Only inspect/IR export performed; no Lean check, GPU run or training.");
            inspect.push_str(
                "Only inspect/IR export performed; no Lean check, GPU run or training.\n",
            );
            fs::write(root.join("inspect.txt"), inspect)?;
        }
        "--certify-architecture" => {
            // One Model -> one Semantic IR -> one Lean -> one DOT -> one PNG.
            // Host argmax/rotation and the unbounded recurrence remain explicit
            // trust boundaries instead of being mislabeled as certified.
            let workspace = ArtifactWorkspace::new()?;
            let certified_root = workspace.path().join("certified");
            architecture.certify(CertifiedBuildConfig::new(&certified_root))?;
            let bundle = certified_root.join("auto-build");
            let shared_gpu = bundle.join("gpu_shared");
            if !shared_gpu.is_dir() {
                return Err("unified architecture did not produce shared GPU lowering".into());
            }

            let staged = workspace.path().join("published");
            fs::create_dir_all(&staged)?;
            for (source, destination) in [
                ("semantic.ir", "semantic.ir"),
                ("architecture.dot", "architecture.dot"),
                ("architecture.png", "architecture.png"),
                ("lean/architecture.lean", "architecture.lean"),
                ("lean/audit.txt", "architecture.audit.txt"),
                ("report.txt", "certification.txt"),
                ("identity.txt", "identity.txt"),
            ] {
                fs::copy(bundle.join(source), staged.join(destination))?;
            }
            for (source, destination) in [
                ("graph.ll", "architecture.graph.ll"),
                ("graph.ptx", "architecture.graph.ptx"),
                ("translation.plan", "architecture.translation.plan"),
                ("translation.report", "architecture.translation.report"),
            ] {
                fs::copy(shared_gpu.join(source), staged.join(destination))?;
            }
            if fs::metadata(staged.join("architecture.png"))?.len() == 0 {
                return Err("architecture.png is empty".into());
            }
            write_architecture_manifest(&staged)?;

            fs::create_dir_all(&root)?;
            for entry in fs::read_dir(&root)? {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with("core.")
                    || name.starts_with("router.")
                    || name.starts_with("head.")
                    || matches!(
                        name.as_ref(),
                        "core"
                            | "router"
                            | "head"
                            | "gpu"
                            | "certified-core"
                            | "certified-router"
                            | "certified-head"
                    )
                {
                    let path = entry.path();
                    if path.is_dir() {
                        fs::remove_dir_all(path)?;
                    } else {
                        fs::remove_file(path)?;
                    }
                }
            }
            for entry in fs::read_dir(&staged)? {
                let entry = entry?;
                fs::copy(entry.path(), root.join(entry.file_name()))?;
            }
            println!(
                "Certified one architecture bundle: {}",
                root.join("architecture.lean").display()
            );
        }
        "--train" => {
            let updates = match args.next() {
                Some(raw) => raw.parse::<usize>()?,
                None => TrainingConfig::default().updates,
            };
            let mut parameters = Parameters::new(0x0051_A7E5);
            let mut runner = Runner::new(&parameters)?;
            let baseline = evaluate(
                &mut runner,
                &parameters,
                &dictionary,
                &data.validation,
                80,
                11,
            )?;
            println!(
                "before training: exact={:.3} MAE={:.2} RMSE={:.2}",
                baseline.accuracy(),
                baseline.mae(),
                baseline.rmse()
            );
            // Train-only median: a non-neural reference using the same held-out cases.
            let mut train_targets = data
                .train
                .iter()
                .map(|case| case.target)
                .collect::<Vec<_>>();
            train_targets.sort_unstable();
            let median = train_targets[train_targets.len() / 2];
            let constant = constant_baseline(&data.validation, 80, median);
            println!(
                "train-median constant baseline y={median}: exact={:.3} MAE={:.2} RMSE={:.2}",
                constant.accuracy(),
                constant.mae(),
                constant.rmse()
            );
            let config = TrainingConfig {
                updates,
                ..TrainingConfig::default()
            };
            train(
                &mut runner,
                &mut parameters,
                &data,
                &dictionary,
                config,
                1009,
            )?;
            let metrics = evaluate(
                &mut runner,
                &parameters,
                &dictionary,
                &data.validation,
                80,
                11,
            )?;
            println!(
                "after {updates} updates: exact={:.3} MAE={:.2} RMSE={:.2}",
                metrics.accuracy(),
                metrics.mae(),
                metrics.rmse()
            );
            println!(
                "routes (core0 R0/R90/R180/R270, core1 R0/R90/R180/R270): {:?}",
                metrics.action_counts
            );
            for op in 0..OP_COUNT {
                if metrics.per_op_count[op] != 0 {
                    println!(
                        "op_{op:02}: {}/{} exact",
                        metrics.per_op_exact[op], metrics.per_op_count[op]
                    );
                }
            }
            println!("fixed-route ablations (40 held-out examples each):");
            for action in 0..ACTIONS {
                let ablation = evaluate_with_route(
                    &mut runner,
                    &parameters,
                    &dictionary,
                    &data.validation,
                    40,
                    11,
                    Some(action),
                )?;
                println!(
                    "core={} R{}: exact={:.3} MAE={:.2}",
                    action / 4,
                    (action % 4) * 90,
                    ablation.accuracy(),
                    ablation.mae()
                );
            }
            // Stable plain-text FP32 bit serialization, including model seed
            // and every trained parameter. No unverified claims of accuracy.
            fs::create_dir_all(&root)?;
            let mut report = format!(
                "seed=0x51A7E5\nupdates={updates}\nvalidation_samples={}\nexact={}\nmae={}\nrmse={}\nroutes={:?}\n",
                metrics.count,
                metrics.exact,
                metrics.mae(),
                metrics.rmse(),
                metrics.action_counts,
            );
            report.push_str(&format!("dictionary_bytes={:?}\n", dictionary.byte_order()));
            for op in 0..OP_COUNT {
                report.push_str(&format!(
                    "op_{op:02}={}/{}\n",
                    metrics.per_op_exact[op], metrics.per_op_count[op]
                ));
            }
            for (name, values) in [
                ("core0", parameters.cores[0].as_slice()),
                ("core1", parameters.cores[1].as_slice()),
                ("router", parameters.router.as_slice()),
                ("router_bias", parameters.router_bias.as_slice()),
                ("core_bias", parameters.core_bias.as_slice()),
                ("head", parameters.head.as_slice()),
                ("head_bias", std::slice::from_ref(&parameters.head_bias)),
            ] {
                report.push_str(name);
                report.push('=');
                report.push_str(
                    &values
                        .iter()
                        .map(|v| format!("{:08x}", v.to_bits()))
                        .collect::<Vec<_>>()
                        .join(","),
                );
                report.push('\n');
            }
            fs::write(root.join("training-report.txt"), report)?;
            for sample in ["10 + 9", "1 AND 0"] {
                let mut rng = Rng::new(11);
                let trace = runner.forward(&parameters, &dictionary, sample, &mut rng, false)?;
                println!(
                    "x={sample:?}, predicted={} (unrounded={:.4})",
                    rotating_matrices_lab::decoded(trace.normalized),
                    trace.normalized * 256.0
                );
            }
            println!(
                "Experimental training, not evidence of arithmetic generalization or Lean-verified learning."
            );
        }
        _ => {
            return Err(format!(
                "unknown mode {mode}; use --inspect, --train [updates] or --certify-architecture"
            )
            .into());
        }
    }
    Ok(())
}
