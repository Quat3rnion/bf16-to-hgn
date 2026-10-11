#[path = "../tests/support/mod.rs"]
mod support;

use bf16_to_hgn::{
    codec::Encoding,
    conversion,
    hgn_source::HgnSidecar,
    manifest::{Options, OutputProfile},
    verify,
};

fn main() -> anyhow::Result<()> {
    let temporary = tempfile::tempdir()?;
    let (hf, mut plan) = support::matrix_fixture(temporary.path(), OutputProfile::V2)?;
    let v2 = temporary.path().join("v2");
    let options = Options {
        seed: 7,
        ht_beam: 8,
        ..Options::default()
    };
    let progress = bf16_to_hgn::progress::Progress::disabled();
    conversion::convert(&hf, &plan, &v2, &options, &progress)?;

    let hgn = HgnSidecar::open(&v2.join("model.hgn"))?;
    let source = support::FixtureAssets {
        tensors: &hgn,
        assets: &hf,
    };
    plan.files[0].identity = "bf16-hgn-ht43-matrix-fixture".into();
    for tensor in &mut plan.files[0].tensors {
        if tensor.name == "layers.1.self_attn.q_proj.weight" {
            tensor.encoding = Encoding::Ht3;
        }
        if tensor.name.ends_with("hyper_connection_mixer.weight") {
            tensor.encoding = Encoding::Q8G64;
        }
    }
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options
    };
    let output = temporary.path().join("ht43");
    conversion::convert(&source, &plan, &output, &options, &progress)?;
    let report = verify::verify(
        &source,
        &plan,
        &output,
        bf16_to_hgn::verify_gpu::DecodeBackend::Cpu,
        &progress,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    // A real v2 quantized HGN is decoded and requantized. This fixture-only
    // library plan has no public CLI architecture-validation bypass.
    Ok(())
}
