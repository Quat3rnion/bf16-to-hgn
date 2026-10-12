//! Library-only structural fixture: real CPU/GPU encode, atomic publication,
//! independent full verification and byte-for-byte HGN container comparison.
//! HIP remains the default; `--backend cuda` selects CUDA instead.
//! `--root /tmp/tiny-hip-i4r --i4r-backend hip` retains inputs and cpu/hip outputs;
//! a CPU-only executable can use those arguments plus `--verify-only`.
//! Select real GPU verification with `--decode-backend hip` or `cuda`.
//! --i4r-backend hip or cuda enables a real I4R matrix in the fixture; combine
//! with --ht-backend cpu for I4R-only GPU encoding, or select the same --backend
//! for dual HT/I4R encoding. CPU-reference conversion always uses CPU backends.

#[path = "../tests/support/mod.rs"]
mod support;

use anyhow::{ensure, Context, Result};
use bf16_to_hgn::{
    codec::{Encoding, HtBackend, I4rBackend, Q4cpBackend},
    conversion, hf,
    manifest::{Manifest, Options, OutputProfile},
    progress::Progress,
    verify,
    verify_gpu::DecodeBackend,
    Origin,
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
    /// Override HT encoding independently; otherwise --backend selects HIP/CUDA.
    #[arg(long, value_enum)]
    ht_backend: Option<HtBackend>,
    /// Independent I4R encoder; CPU preserves the original all-HT fixture.
    #[arg(long, value_enum, default_value_t = I4rBackend::Cpu)]
    i4r_backend: I4rBackend,
    /// Independent Q4C-P encoder for the fixture's Q4C-P embedding matrix.
    #[arg(long, value_enum, default_value_t = Q4cpBackend::Cpu)]
    q4cp_backend: Q4cpBackend,
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
    ht_backend: HtBackend,
    i4r_backend: I4rBackend,
    q4cp_backend: Q4cpBackend,
    decode_backend: DecodeBackend,
    progress: &Progress,
) -> Result<Value> {
    let snapshot = hf::Snapshot::open(&root.join("source"))?;
    let mut manifests = Vec::new();
    let mut reports = Vec::<Value>::new();
    for (name, backend, i4r, q4cp) in [
        ("cpu", HtBackend::Cpu, I4rBackend::Cpu, Q4cpBackend::Cpu),
        (gpu_backend.name(), ht_backend, i4r_backend, q4cp_backend),
    ] {
        let output = root.join(name);
        let manifest: Manifest =
            serde_json::from_value(hf::read_json(&output.join("conversion-manifest.json"))?)
                .context("read tiny conversion manifest")?;
        ensure!(
            manifest.options.ht_backend == backend
                && manifest.options.i4r_backend == i4r
                && manifest.options.q4cp_backend == q4cp,
            "{name} manifest records the wrong HT/I4R/Q4C-P backend"
        );
        if i4r_backend.is_gpu() {
            ensure!(
                manifest.plan.files.iter().any(|file| file
                    .tensors
                    .iter()
                    .any(|tensor| tensor.encoding == Encoding::I4r)),
                "GPU I4R smoke must contain an actual I4R tensor"
            );
        }
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
        reports.push(json!({"backend": backend, "i4r_backend": i4r, "q4cp_backend": q4cp, "verification": report}));
        manifests.push(manifest);
    }
    let cpu = &manifests[0];
    let gpu = &manifests[1];
    let mut gpu_options = gpu.options.clone();
    gpu_options.ht_backend = HtBackend::Cpu;
    gpu_options.i4r_backend = I4rBackend::Cpu;
    gpu_options.q4cp_backend = Q4cpBackend::Cpu;
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
        "encoding_backend": ht_backend,
        "i4r_backend": i4r_backend,
        "q4cp_backend": q4cp_backend,
        "decode_backend": decode_backend,
        "ht_beam": cpu.options.ht_beam,
        "conversion_encoder_sha256": cpu.encoder_sha256,
        "verifier_encoder_sha256": bf16_to_hgn::ENCODER_SHA256,
        "verified_identical_hgn_bytes": true,
        "containers": containers,
        "reports": reports,
    }))
}

fn tiny_fixture(
    root: &Path,
    i4r_backend: I4rBackend,
) -> Result<(hf::Snapshot, bf16_to_hgn::ConversionPlan)> {
    let (snapshot, mut plan) = support::gpu_fixture(root)?;
    if i4r_backend.is_gpu() {
        // Restore the genuine source expert axes and shared sign planes for I4R.
        // Other tensors retain the original legal HT3/HT4 matrix fixture.
        for tensor in &mut plan.files[0].tensors {
            match tensor.name.as_str() {
                "fixture.experts.down.weight" => {
                    tensor.shape = vec![2, 128, 128];
                    tensor.encoding = Encoding::I4r;
                }
                "fixture.experts.down.svh" => {
                    tensor.shape = vec![128];
                    tensor.origin = Origin::Signs {
                        key: tensor.name.clone(),
                    };
                }
                _ => {}
            }
        }
    }
    Ok((snapshot, plan))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ht_backend = args.ht_backend.unwrap_or_else(|| args.backend.encoder());
    let options = Options {
        profile: OutputProfile::Ht43,
        seed: 7,
        threads: 1,
        ht_beam: args.ht_beam,
        ht_backend,
        i4r_backend: args.i4r_backend,
        q4cp_backend: args.q4cp_backend,
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
        let (snapshot, plan) = tiny_fixture(&root, args.i4r_backend)?;
        for (name, backend, i4r_backend, q4cp_backend) in [
            ("cpu", HtBackend::Cpu, I4rBackend::Cpu, Q4cpBackend::Cpu),
            (args.backend.name(), ht_backend, args.i4r_backend, args.q4cp_backend),
        ] {
            let opts = Options {
                ht_backend: backend,
                i4r_backend,
                q4cp_backend,
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
            ht_backend,
            args.i4r_backend,
            args.q4cp_backend,
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
        assert!(args.ht_backend.is_none());
        assert_eq!(args.i4r_backend, I4rBackend::Cpu);
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

    #[test]
    fn gpu_i4r_and_ht_cpu_are_independent() {
        for (selection, backend) in [("hip", I4rBackend::Hip), ("cuda", I4rBackend::Cuda)] {
            let args = Args::parse_from([
                "tiny_hip_conversion",
                "--backend",
                selection,
                "--ht-backend",
                "cpu",
                "--i4r-backend",
                selection,
            ]);
            assert_eq!(args.ht_backend, Some(HtBackend::Cpu));
            assert_eq!(args.i4r_backend, backend);
        }
        let args = Args::parse_from([
            "tiny_hip_conversion",
            "--backend",
            "cuda",
            "--i4r-backend",
            "hip",
        ]);
        assert!(matches!(args.backend, GpuBackend::Cuda));
        assert_eq!(args.i4r_backend, I4rBackend::Hip);
    }

    #[test]
    fn retained_hip_i4r_verify_only_defaults_to_cpu_decoder() {
        let args = Args::parse_from([
            "tiny_hip_conversion",
            "--root",
            "/tmp/tiny-hip-i4r",
            "--ht-backend",
            "cpu",
            "--i4r-backend",
            "hip",
            "--verify-only",
        ]);
        assert!(args.verify_only);
        assert!(matches!(args.backend, GpuBackend::Hip));
        assert_eq!(args.ht_backend, Some(HtBackend::Cpu));
        assert_eq!(args.i4r_backend, I4rBackend::Hip);
        assert!(matches!(args.decode_backend, DecodeBackend::Cpu));
    }

    #[test]
    fn gpu_i4r_fixtures_convert_and_source_verify_without_a_device() -> Result<()> {
        for backend in [I4rBackend::Hip, I4rBackend::Cuda] {
            let temporary = tempfile::tempdir()?;
            let (snapshot, plan) = tiny_fixture(temporary.path(), backend)?;
            assert!(plan.files[0]
                .tensors
                .iter()
                .any(|tensor| tensor.encoding == Encoding::I4r));
            let options = Options {
                profile: OutputProfile::Ht43,
                seed: 7,
                threads: 1,
                ht_beam: 8,
                ht_backend: HtBackend::Cpu,
                i4r_backend: I4rBackend::Cpu,
            };
            let progress = Progress::disabled();
            let output = temporary.path().join("cpu");
            conversion::convert(&snapshot, &plan, &output, &options, &progress)?;
            let report = verify::verify(&snapshot, &plan, &output, DecodeBackend::Cpu, &progress)?;
            let down = &report.errors["model.hgn:fixture.experts.down.weight"];
            assert_eq!(down.values, 32768);
            assert!(down.source_squared.is_finite() && down.source_squared > 0.0);
            // A generous 10% squared-error ceiling checks reconstructed source
            // fidelity without freezing this structural fixture's quantized bytes.
            assert!(down.error_squared.is_finite());
            assert!(
                down.relative_squared_error() < 0.1,
                "I4R source-relative squared error exceeds 10%: {}",
                down.relative_squared_error(),
            );
        }
        Ok(())
    }
}
