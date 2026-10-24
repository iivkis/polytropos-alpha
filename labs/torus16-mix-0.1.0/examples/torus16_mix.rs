//! Inspect the Torus16-Mix Semantic IR or run its VeriNeural host reference.
//! No PyTorch import, standalone CUDA kernels or separate proof generators.
use std::error::Error;
use std::path::PathBuf;
use torus16_mix_lab::{Parameters, ROLES, WIDTH, architecture_model, reference_project};
use verineural::CertifiedBuildConfig;

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
            return Err(
                format!("unsupported mode: {mode} (use --inspect, --ir or --certify)").into(),
            );
        }
    }
    Ok(())
}
