//! Inspect the Torus16-Mix Semantic IR or run its VeriNeural host reference.
//! No PyTorch import, standalone CUDA kernels or separate proof generators.
use std::error::Error;
use std::path::PathBuf;
use torus16_mix_lab::{Parameters, ROLES, WIDTH, architecture_model, reference_project};
use verineural::{CertifiedBuildConfig, GpuEngine, GpuLaunchConfig};

fn main() -> Result<(), Box<dyn Error>> {
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "--inspect".into());
    let model = architecture_model()?;
    match mode.as_str() {
        "--inspect" => {
            let params = Parameters::default();
            let input = std::array::from_fn(|i| (i as f32 - 24.0) / WIDTH as f32);
            let bindings = model.bind(params.bindings(&input)?)?;
            let output = reference_project(&input, &params, 0)?;
            println!(
                "Torus16-Mix: WIDTH={WIDTH}, roles={ROLES}, bases={}, trainable_operator_parameters={}",
                ROLES * 2,
                Parameters::trainable_count()
            );
            println!(
                "IR nodes={}, module instances={}, bound sources={}",
                model.graph().nodes().len(),
                model.graph().modules().len(),
                bindings.len()
            );
            println!("reference role_0 first 6 lanes: {:?}", &output[..6]);
            println!(
                "Proof/GPU status: full operator unsupported (exp/div); one-row structural GPU slice is available."
            );
            println!("This is the operator slice; no LM training or original BPC reproduction.");
        }
        "--gpu" => {
            // ONE executable CUDA Graph: all eight public outputs are roots.
            // Exponential lowering uses the explicitly opted-in approximate profile.
            let params = Parameters::default();
            let input = std::array::from_fn(|i| ((i * 17 + 11) % 43) as f32 * 0.013 - 0.27);
            let engine = GpuEngine::new(
                "llc",
                std::env::var("VERINEURAL_CUDA_ARCH").unwrap_or_else(|_| "sm_52".into()),
            )
            .with_launch_config(GpuLaunchConfig::default().with_approximate_exp());
            let workspace =
                std::env::temp_dir().join(format!("vn_torus16_mix_gpu_{}", std::process::id()));
            let mut runner = engine.prepare_model(&model, params.bindings(&input)?, &workspace)?;
            runner.run()?;
            let mut worst_error = 0.0_f32;
            for role in 0..ROLES {
                let result = runner.output(&format!("role_{role}"))?;
                let reference = reference_project(&input, &params, role)?;
                for lane in 0..WIDTH {
                    let error = (result.values()[lane] - reference[lane]).abs();
                    let tolerance = 5e-5 + 2e-5 * reference[lane].abs();
                    if !error.is_finite() || error > tolerance {
                        return Err(format!(
                            "GPU mismatch: role={role} lane={lane} \
                             actual={} expected={} error={error} tolerance={tolerance}",
                            result.values()[lane],
                            reference[lane]
                        )
                        .into());
                    }
                    worst_error = worst_error.max(error);
                }
            }
            println!("Torus16-Mix CUDA Graph: eight roles verified; max_abs_error={worst_error:e}");
            println!(
                "Numerical boundary: exp2 APPROX opt-in; no strict expf parity, \
                 no Lean FP32 proof, no training/BPC claim."
            );
        }
        "--ir" => print!("{}", model.graph().canonical_ir()),
        "--certify" => {
            // Deliberately fail closed until the full IR has GPU + Lean support.
            let output = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts")
                .join("certified");
            model.certify(CertifiedBuildConfig::new(output))?;
            println!("Certified; verify evidence in artifacts/certified.");
        }
        _ => {
            return Err(format!(
                "unsupported mode: {mode} (use --inspect, --gpu, --ir or --certify)"
            )
            .into());
        }
    }
    Ok(())
}
