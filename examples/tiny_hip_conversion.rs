//! Library-only structural fixture: real CPU/HIP encode, atomic publication,
//! independent full verification and byte-for-byte HGN container comparison.
//! `--root /tmp/tiny-hip` retains inputs/outputs; a CPU-only executable can then
//! use `--root /tmp/tiny-hip --verify-only` without a GPU or HIP feature.

#[path = "../tests/support/mod.rs"]
mod support;

use anyhow::{ensure, Context, Result};
use bf16_to_hgn::{
    codec::HtBackend,
    conversion, hf,
    manifest::{Manifest, Options, OutputProfile},
    verify,
};
use clap::Parser;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(about = "Real tiny CPU/HIP conversion, independent verification and exact HGN bytes")]
struct Args {
    /// New directory retaining genuine fixture source plus cpu/hip outputs.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Verify existing outputs and compare bytes; works in CPU-only builds.
    #[arg(long, requires = "root")]
    verify_only: bool,
    #[arg(long, default_value_t = 8)]
    ht_beam: usize,
}

fn compare_container(cpu: &Path, hip: &Path) -> Result<Value> {
    let mut cpu = fs::File::open(cpu)?;
    let mut hip = fs::File::open(hip)?;
    let size = cpu.metadata()?.len();
    ensure!(
        size == hip.metadata()?.len(),
        "CPU/HIP container sizes differ"
    );
    let mut remaining = size;
    let mut offset = 0u64;
    let mut left = [0u8; 64 * 1024];
    let mut right = [0u8; 64 * 1024];
    let mut hash = Sha256::new();
    while remaining > 0 {
        let take = remaining.min(left.len() as u64) as usize;
        cpu.read_exact(&mut left[..take])?;
        hip.read_exact(&mut right[..take])?;
        ensure!(
            left[..take] == right[..take],
            "CPU/HIP container bytes differ at chunk offset {offset}"
        );
        hash.update(&left[..take]);
        remaining -= take as u64;
        offset += take as u64;
    }
    Ok(json!({"bytes": size, "sha256": format!("{:x}", hash.finalize()), "identical_bytes": true}))
}

fn verify_and_compare(root: &Path) -> Result<Value> {
    let snapshot = hf::Snapshot::open(&root.join("source"))?;
    let mut manifests = Vec::new();
    let mut reports = Vec::new();
    for (name, backend) in [("cpu", HtBackend::Cpu), ("hip", HtBackend::Hip)] {
        let output = root.join(name);
        let manifest: Manifest =
            serde_json::from_value(hf::read_json(&output.join("conversion-manifest.json"))?)
                .context("read tiny conversion manifest")?;
        ensure!(
            manifest.options.ht_backend == backend,
            "{name} manifest records the wrong backend"
        );
        let report = verify::verify(&snapshot, &manifest.plan, &output)?;
        reports.push(json!({"backend": backend, "verification": report}));
        manifests.push(manifest);
    }
    let cpu = &manifests[0];
    let hip = &manifests[1];
    let mut hip_options = hip.options.clone();
    hip_options.ht_backend = HtBackend::Cpu;
    ensure!(
        cpu.options == hip_options,
        "CPU/HIP conversion options differ beyond backend"
    );
    ensure!(cpu.plan == hip.plan, "CPU/HIP plans differ");
    ensure!(
        cpu.sources == hip.sources && cpu.assets == hip.assets,
        "CPU/HIP source or copied-asset fingerprints differ"
    );
    ensure!(
        cpu.containers == hip.containers,
        "CPU/HIP payload layouts or fingerprints differ"
    );
    let mut containers = Vec::new();
    for container in &cpu.containers {
        let comparison = compare_container(
            &root.join("cpu").join(&container.filename),
            &root.join("hip").join(&container.filename),
        )?;
        containers.push(json!({"filename": container.filename, "comparison": comparison}));
    }
    Ok(json!({
        "fixture": "library-only; not an engine-compatible checkpoint or public CLI bypass",
        "ht_beam": cpu.options.ht_beam,
        "conversion_encoder_sha256": cpu.encoder_sha256,
        "verifier_encoder_sha256": bf16_to_hgn::ENCODER_SHA256,
        "verified_identical_hgn_bytes": true,
        "containers": containers,
        "reports": reports,
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let options = Options {
        profile: OutputProfile::Ht43,
        seed: 7,
        threads: 1,
        ht_beam: args.ht_beam,
        ht_backend: HtBackend::Hip,
    };
    if !args.verify_only {
        // Fail before creating even a fixture directory: never fake a GPU run.
        options.validate_for_execution()?;
    }
    let mut temporary = None;
    let root = if let Some(root) = args.root {
        if !args.verify_only {
            fs::create_dir(&root)
                .context("create new retained fixture directory without clobbering")?;
        }
        root
    } else {
        temporary = Some(tempfile::tempdir()?);
        temporary
            .as_ref()
            .expect("temporary directory")
            .path()
            .to_path_buf()
    };
    if !args.verify_only {
        let (snapshot, plan) = support::hip_fixture(&root)?;
        for (name, backend) in [("cpu", HtBackend::Cpu), ("hip", HtBackend::Hip)] {
            let opts = Options {
                ht_backend: backend,
                ..options.clone()
            };
            conversion::convert(&snapshot, &plan, &root.join(name), &opts)?;
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&verify_and_compare(&root)?)?
    );
    // Retained roots are never deleted; the default TempDir cleans up its own fixture.
    drop(temporary);
    Ok(())
}
