//! Consumer architecture is defined in src/lib.rs; the reusable library
//! owns all CLI arguments, proof checks, GPU lowering and bundle publication.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = coupled_field_lab::architecture()?;
    verineural::certified_example_main(&model, env!("CARGO_MANIFEST_DIR"))
}
