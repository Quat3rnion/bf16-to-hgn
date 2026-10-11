#[path = "../tests/support/mod.rs"]
mod support;

fn main() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let options = bf16_to_hgn::manifest::Options {
        seed: 7,
        threads: 1,
        ht_beam: 8,
        ..Default::default()
    };
    let output = temporary.path().join("converted");
    let progress = bf16_to_hgn::progress::Progress::stderr();
    bf16_to_hgn::conversion::convert(&snapshot, &plan, &output, &options, &progress)?;
    let report = bf16_to_hgn::verify::verify(
        &snapshot,
        &plan,
        &output,
        bf16_to_hgn::verify_gpu::DecodeBackend::Cpu,
        &progress,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    // TempDir removes both the genuine fixture input and converted output.
    Ok(())
}
