//! Library-only structural fixture: real CPU/GPU encode, atomic publication,
//! independent full verification and byte-for-byte HGN container comparison.
//! HIP remains the default; `--backend cuda` selects CUDA instead.
//! `--root /tmp/tiny-cuda --backend cuda` retains inputs and cpu/cuda outputs;
//! a CPU-only executable can use those arguments plus `--verify-only`.
//! Select real GPU verification with `--decode-backend hip` or `cuda`.

#[path = "../tests/support/mod.rs"]
mod support;

use anyhow::{ensure, Context, Result};
use bf16_to_hgn::{
    codec::HtBackend,
    conversion, hf,
    manifest::{Manifest, Options, OutputProfile},
    progress::Progress,
    verify,
    verify_gpu::DecodeBackend,
};
use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, ValueEnum)]
enum GpuBackend {
    Hip,
    Cuda,
}

impl GpuBackend {
    fn encoder(self) -> HtBackend {
        match self {
            Self::Hip => HtBackend::Hip,
            Self::Cuda => HtBackend::Cuda,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Hip => "hip",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Parser)]
#[command(about = "Real tiny CPU/GPU conversion, independent verification and exact HGN bytes")]
struct Args {
    /// GPU encoder and retained output directory; defaults to HIP.
    #[arg(long, value_enum, default_value = "hip")]
    backend: GpuBackend,
    /// New directory retaining genuine fixture source plus cpu/GPU outputs.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Verify existing outputs and compare bytes; works in CPU-only builds.
    #[arg(long, requires = "root")]
    verify_only: bool,
    /// Runtime HT3/HT4/I4R decoder; independent of recorded encoding provenance.
    #[arg(long, value_enum, default_value_t = DecodeBackend::Cpu)]
    decode_backend: DecodeBackend,
    #[arg(long, default_value_t = 8)]
    ht_beam: usize,
}

fn compare_container(cpu: &Path, gpu: &Path) -> Result<Value> {
    let mut cpu = fs::File::open(cpu)?;
    let mut gpu = fs::File::open(gpu)?;
    let size = cpu.metadata()?.len();
    ensure!(
        size == gpu.metadata()?.len(),
        "CPU/GPU container sizes differ"
    );
    let mut remaining = size;
    let mut offset = 0u64;
    let mut left = [0u8; 64 * 1024];
    let mut right = [0u8; 64 * 1024];
    let mut hash = Sha256::new();
    while remaining > 0 {
        let take = remaining.min(left.len() as u64) as usize;
        cpu.read_exact(&mut left[..take])?;
        gpu.read_exact(&mut right[..take])?;
        ensure!(
            left[..take] == right[..take],
            "CPU/GPU container bytes differ at chunk offset {offset}"
        );
        hash.update(&left[..take]);
        remaining -= take as u64;
        offset += take as u64;
    }
    Ok(json!({"bytes": size, "sha256": format!("{:x}", hash.finalize()), "identical_bytes": true}))
}

fn verify_and_compare(
    root: &Path,
    gpu_backend: GpuBackend,
    decode_backend: DecodeBackend,
    progress: &Progress,
) -> Result<Value> {
    let snapshot = hf::Snapshot::open(&root.join("source"))?;
    let mut manifests = Vec::new();
    let mut reports = Vec::<Value>::new();
    for (name, backend) in [
        ("cpu", HtBackend::Cpu),
        (gpu_backend.name(), gpu_backend.encoder()),
    ] {
        let output = root.join(name);
        let manifest: Manifest =
            serde_json::from_value(hf::read_json(&output.join("conversion-manifest.json"))?)
                .context("read tiny conversion manifest")?;
        ensure!(
            manifest.options.ht_backend == backend,
            "{name} manifest records the wrong backend"
        );
        let report = serde_json::to_value(verify::verify(
            &snapshot,
            &manifest.plan,
            &output,
            decode_backend,
            progress,
        )?)?;
        if let Some(reference) = reports.first() {
            ensure!(
                reference["verification"] == report,
                "complete verification reports differ for CPU/GPU encoded payloads"
            );
        }
        reports.push(json!({"backend": backend, "verification": report}));
        manifests.push(manifest);
    }
    let cpu = &manifests[0];
    let gpu = &manifests[1];
    let mut gpu_options = gpu.options.clone();
    gpu_options.ht_backend = HtBackend::Cpu;
    ensure!(
        cpu.options == gpu_options,
        "CPU/GPU conversion options differ beyond backend"
    );
    ensure!(cpu.plan == gpu.plan, "CPU/GPU plans differ");
    ensure!(
        cpu.sources == gpu.sources && cpu.assets == gpu.assets,
        "CPU/GPU source or copied-asset fingerprints differ"
    );
    ensure!(
        cpu.containers == gpu.containers,
        "CPU/GPU payload layouts or fingerprints differ"
    );
    let mut containers = Vec::new();
    for container in &cpu.containers {
        let comparison = compare_container(
            &root.join("cpu").join(&container.filename),
            &root.join(gpu_backend.name()).join(&container.filename),
        )?;
        containers.push(json!({"filename": container.filename, "comparison": comparison}));
    }
    let decode_backend = match decode_backend {
        DecodeBackend::Cpu => "cpu",
        DecodeBackend::Hip => "hip",
        DecodeBackend::Cuda => "cuda",
    };
    Ok(json!({
        "fixture": "library-only; not an engine-compatible checkpoint or public CLI bypass",
        "encoding_backend": gpu_backend.encoder(),
        "decode_backend": decode_backend,
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
        ht_backend: args.backend.encoder(),
    };
    if !args.verify_only {
        // Fail before creating even a fixture directory: never fake a GPU run.
        options.validate_for_execution()?;
    }
    let progress = Progress::stderr();
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
        let (snapshot, plan) = support::gpu_fixture(&root)?;
        for (name, backend) in [
            ("cpu", HtBackend::Cpu),
            (args.backend.name(), args.backend.encoder()),
        ] {
            let opts = Options {
                ht_backend: backend,
                ..options.clone()
            };
            conversion::convert(&snapshot, &plan, &root.join(name), &opts, &progress)?;
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&verify_and_compare(
            &root,
            args.backend,
            args.decode_backend,
            &progress,
        )?)?
    );
    // Retained roots are never deleted; the default TempDir cleans up its own fixture.
    drop(temporary);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_stays_hip() {
        let args = Args::parse_from(["tiny_hip_conversion"]);
        assert!(matches!(args.backend, GpuBackend::Hip));
        assert!(matches!(args.decode_backend, DecodeBackend::Cpu));
    }

    #[test]
    fn cuda_verify_only_defaults_to_cpu_decoder() {
        let args = Args::parse_from([
            "tiny_hip_conversion",
            "--backend",
            "cuda",
            "--root",
            "/tmp/tiny-cuda",
            "--verify-only",
        ]);
        assert!(args.verify_only);
        assert!(matches!(args.backend, GpuBackend::Cuda));
        assert!(matches!(args.decode_backend, DecodeBackend::Cpu));
        assert!(Args::try_parse_from(["tiny_hip_conversion", "--backend", "cpu"]).is_err());
    }

    #[test]
    fn cuda_decoder_is_selectable_independently() {
        let args = Args::parse_from([
            "tiny_hip_conversion",
            "--backend",
            "hip",
            "--decode-backend",
            "cuda",
        ]);
        assert!(matches!(args.backend, GpuBackend::Hip));
        assert!(matches!(args.decode_backend, DecodeBackend::Cuda));
    }
}
