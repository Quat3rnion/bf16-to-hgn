#[path = "../tests/support/mod.rs"]
mod support;

use bf16_to_hgn::{
    conversion,
    manifest::{Options, OutputProfile},
    verify,
};

fn main() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (gguf, plan) = support::gguf_matrix_fixture(temporary.path())?;
    let source = support::FixtureAssets {
        tensors: &gguf,
        assets: &assets,
    };
    let options = Options {
        profile: OutputProfile::Ht43,
        seed: 7,
        ht_beam: 8,
        ..Options::default()
    };
    let output = temporary.path().join("ht43");
    let progress = bf16_to_hgn::progress::Progress::disabled();
    conversion::convert(&source, &plan, &output, &options, &progress)?;
    let report = verify::verify(
        &source,
        &plan,
        &output,
        bf16_to_hgn::verify_gpu::DecodeBackend::Cpu,
        &progress,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    // Independently written GGUF v3 contains genuine F32/Q8_0 payloads, exact
    // PLE integer metadata, and an embedded table. No engine identity or CLI
    // architecture-validation bypass is used by this library-only fixture.
    Ok(())
}
