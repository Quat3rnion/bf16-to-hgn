#[path = "../tests/support/mod.rs"]
mod support;

use bf16_to_hgn::{
    conversion,
    manifest::{Options, OutputProfile},
    verify,
};

fn main() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let (source, plan) = support::matrix_fixture(temporary.path(), OutputProfile::Ht43)?;
    let options = Options {
        profile: OutputProfile::Ht43,
        seed: 7,
        ht_beam: 8,
        ..Options::default()
    };
    let output = temporary.path().join("ht43");
    conversion::convert(&source, &plan, &output, &options)?;
    let report = verify::verify(&source, &plan, &output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    // These real small HF tensors exercise HT4/HT3/Q8G64. The manual library
    // plan and container identity are deliberately not engine checkpoints.
    Ok(())
}
