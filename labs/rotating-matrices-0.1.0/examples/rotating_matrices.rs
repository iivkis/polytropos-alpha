//! Run with `task lab-rotating` or `task train-lab-rotating -- <updates>`.
use rotating_matrices_lab::{
    ACTIONS, Architectures, ArtifactWorkspace, Dataset, Dictionary, OP_COUNT, Parameters, Rng,
    Runner, TrainingConfig, constant_baseline, evaluate, evaluate_with_route, operation, train,
};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use verineural::CertifiedBuildConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "--inspect".into());
    let data = Dataset::new();
    let dictionary = Dictionary::fit(&data.train);
    let architectures = Architectures::new()?;
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
            for (name, graph) in [
                ("core", architectures.core.graph()),
                ("router", architectures.router.graph()),
                ("head", architectures.head.graph()),
            ] {
                println!("{name}: {} Semantic IR nodes", graph.nodes().len());
                inspect.push_str(&format!(
                    "{name}: {} Semantic IR nodes\n",
                    graph.nodes().len()
                ));
                fs::write(root.join(format!("{name}.ir")), graph.canonical_ir())?;
            }
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
        "--certify-primitives" => {
            // Three independent certified bundles remain intact *inside*
            // three .tar files, so SHA256SUMS and local relative symlinks are
            // still verifiable after extraction. No persistent subdirectories.
            // This does NOT certify rotation, selection, recurrence or training.
            let workspace = ArtifactWorkspace::new()?;
            let staged = workspace.path().join("flat");
            fs::create_dir_all(&staged)?;
            for (name, model) in [
                ("core", &architectures.core),
                ("router", &architectures.router),
                ("head", &architectures.head),
            ] {
                let folder = format!("certified-{name}");
                let path = workspace.path().join(&folder);
                model.certify(CertifiedBuildConfig::new(&path))?;
                let archive = staged.join(format!("{name}.certified.tar"));
                let status = Command::new("tar")
                    .args([
                        "--sort=name",
                        "--mtime=@0",
                        "--owner=0",
                        "--group=0",
                        "--numeric-owner",
                        "-cf",
                    ])
                    .arg(&archive)
                    .arg("-C")
                    .arg(workspace.path())
                    .arg(&folder)
                    .status()?;
                if !status.success() {
                    return Err(format!("tar failed for certified model {name}").into());
                }
                let bundle = path.join("auto-build");
                for (source, extension) in [
                    ("semantic.ir", "certified.ir"),
                    ("architecture.dot", "dot"),
                    ("architecture.png", "png"),
                    ("lean/architecture.lean", "lean"),
                    ("lean/audit.txt", "audit.txt"),
                    ("report.txt", "certification.txt"),
                ] {
                    fs::copy(
                        bundle.join(source),
                        staged.join(format!("{name}.{extension}")),
                    )?;
                }
                println!(
                    "Certified primitive graph: {name} ({} in archive)",
                    archive.display()
                );
            }
            // Do not publish partial certificates if one of the three fails.
            fs::create_dir_all(&root)?;
            for entry in fs::read_dir(&staged)? {
                let entry = entry?;
                fs::copy(entry.path(), root.join(entry.file_name()))?;
            }
            // Remove exactly the legacy generated directories only after all
            // three new proof archives have been published successfully.
            for name in [
                "gpu",
                "certified-core",
                "certified-router",
                "certified-head",
            ] {
                let legacy = root.join(name);
                if legacy.exists() {
                    fs::remove_dir_all(&legacy)?;
                }
            }
        }
        "--train" => {
            let updates = match args.next() {
                Some(raw) => raw.parse::<usize>()?,
                None => TrainingConfig::default().updates,
            };
            let mut parameters = Parameters::new(0x51A7_E5);
            let mut runner = Runner::new(&root, &parameters)?;
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
            let mut config = TrainingConfig::default();
            config.updates = updates;
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
                "unknown mode {mode}; use --inspect, --train [updates] or --certify-primitives"
            )
            .into());
        }
    }
    Ok(())
}
