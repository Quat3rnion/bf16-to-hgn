//! Multi-GPU device-pool coverage: `--devices` parsing, bounded queue
//! behavior on CPU workers, and (with GPU features) the hip/cuda smoke test
//! proving device assignment never changes published bytes.

mod support;

use anyhow::Result;
use bf16_to_hgn::{
    codec::HtBackend,
    conversion,
    manifest::{Options, OutputProfile},
    pool::{parse_devices, DevicePool, GpuDevice},
    progress::Progress,
};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

fn options() -> Options {
    Options {
        seed: 7,
        threads: 1,
        ht_beam: 8,
        ..Options::default()
    }
}

#[test]
fn parses_valid_device_lists() {
    assert_eq!(
        parse_devices("cpu:0,cpu:1").unwrap(),
        vec![
            GpuDevice {
                backend: HtBackend::Cpu,
                index: 0
            },
            GpuDevice {
                backend: HtBackend::Cpu,
                index: 1
            },
        ]
    );
    assert_eq!(
        parse_devices(" CPU:1 ").unwrap(),
        vec![GpuDevice {
            backend: HtBackend::Cpu,
            index: 1
        }]
    );
    assert_eq!(
        parse_devices("cpu:0, cpu:1, cpu:2, cpu:3").unwrap().len(),
        4
    );
    let eight = "cpu:0,cpu:1,cpu:2,cpu:3,cpu:4,cpu:5,cpu:6,cpu:7";
    assert_eq!(parse_devices(eight).unwrap().len(), 8);
}

#[test]
fn rejects_invalid_device_lists() {
    assert!(parse_devices("metal:0").is_err());
    assert!(parse_devices("cpu:0,cpu:0").is_err());
    assert!(parse_devices(
        "cpu:0,cpu:1,cpu:2,cpu:3,cpu:4,cpu:5,cpu:6,cpu:7,cpu:8"
    )
    .is_err());
    assert!(parse_devices("cpu").is_err());
    assert!(parse_devices("cpu:x").is_err());
    assert!(parse_devices(",").is_err());
}

#[test]
fn gpu_backends_require_matching_features() {
    if cfg!(feature = "hip") {
        assert!(parse_devices("hip:0").is_ok());
    } else {
        assert!(parse_devices("hip:0").is_err());
    }
    if cfg!(feature = "cuda") {
        assert!(parse_devices("cuda:0").is_ok());
    } else {
        assert!(parse_devices("cuda:0").is_err());
    }
}

#[test]
fn pool_bounds_inflight_jobs() -> Result<()> {
    let devices = parse_devices("cpu:0,cpu:1")?;
    let depth = 4;
    let running = Arc::new(AtomicUsize::new(0));
    let peak_running = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let pool = DevicePool::new(scope, &devices, depth);
        for _ in 0..20u32 {
            let running = running.clone();
            let peak_running = peak_running.clone();
            pool.submit(move || {
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak_running.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(2));
                running.fetch_sub(1, Ordering::SeqCst);
            })?;
            assert!(
                pool.in_flight() <= depth,
                "queue exceeded its bounded depth of {depth}"
            );
        }
        pool.join()
    })?;
    assert!(
        peak_running.load(Ordering::SeqCst) <= 2,
        "more jobs ran concurrently than there are CPU workers"
    );
    Ok(())
}

#[test]
fn pool_collects_results_keyed_by_name() -> Result<()> {
    let devices = parse_devices("cpu:0,cpu:1,cpu:2")?;
    let results = Arc::new(Mutex::new(BTreeMap::<String, u64>::new()));
    std::thread::scope(|scope| {
        let pool = DevicePool::new(scope, &devices, 8);
        for i in 0..10u64 {
            let results = results.clone();
            pool.submit(move || {
                results.lock().unwrap().insert(format!("tensor.{i}"), i * i);
            })?;
        }
        pool.join()?;
        let map = results.lock().unwrap();
        assert_eq!(map.len(), 10);
        assert_eq!(map["tensor.7"], 49);
        Ok(())
    })
}

/// CI-runnable pool path: `--devices cpu:0,cpu:1` runs the same tensor
/// pipeline through two CPU encode workers and must publish the exact same
/// bytes as the single-device pipeline.
#[test]
fn cpu_device_pool_conversion_matches_single_device_bytes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let single = temporary.path().join("single");
    let pooled = temporary.path().join("pooled");
    conversion::convert(&snapshot, &plan, &single, &options(), &Progress::disabled())?;
    conversion::convert_with_devices(
        &snapshot,
        &plan,
        &pooled,
        &options(),
        &Progress::disabled(),
        &conversion::ResumeOptions::default(),
        &parse_devices("cpu:0,cpu:1")?,
    )?;
    for file in ["model.hgn", "ngram.hgn", "vision.hgn"] {
        let (a, b) = (single.join(file), pooled.join(file));
        if a.exists() && b.exists() {
            assert_eq!(
                fs::read(&a)?,
                fs::read(&b)?,
                "{file} bytes differ between single-device and cpu:0,cpu:1"
            );
        }
    }
    Ok(())
}

/// Materializes the safetensors fixture (assets) and the GGUF matrix fixture
/// at a fixed path so the shell can run the baseline and multi-gpu binaries
/// against the identical source.
#[test]
fn materializes_cli_spot_check_source() -> Result<()> {
    let dir = Path::new("/tmp/bf16-spot");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir)?;
    support::fixture(dir)?;
    support::gguf_matrix_fixture(dir)?;
    Ok(())
}

/// Converts the GGUF fixture to a fixed HGN directory so the shell can run
/// the baseline and multi-gpu binaries against the identical HGN source.
#[test]
fn materializes_cli_spot_check_hgn_source() -> Result<()> {
    let dir = Path::new("/tmp/bf16-spot-hgn");
    let _ = fs::remove_dir_all(dir);
    fs::create_dir_all(dir)?;
    let (assets, _) = support::fixture(dir)?;
    let (gguf, plan) = support::gguf_matrix_fixture(dir)?;
    let source = support::FixtureAssets {
        tensors: &gguf,
        assets: &assets,
    };
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    conversion::convert(
        &source,
        &plan,
        &dir.join("hgn"),
        &options,
        &Progress::disabled(),
    )?;
    Ok(())
}

/// Headline proof: encoding the same GGUF fixture on the R9700 alone and on
/// hip:0,cuda:0 concurrently publishes byte-identical output.
#[cfg(all(feature = "hip", feature = "cuda"))]
#[test]
fn hip_and_cuda_device_pool_publishes_identical_bytes() -> Result<()> {
    use sha2::{Digest, Sha256};
    fn sha256_path(path: &Path) -> Result<String> {
        let mut hasher = Sha256::new();
        std::io::copy(&mut fs::File::open(path)?, &mut hasher)?;
        Ok(format!("{:x}", hasher.finalize()))
    }
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (gguf, plan) = support::gguf_matrix_fixture(temporary.path())?;
    let source = support::FixtureAssets {
        tensors: &gguf,
        assets: &assets,
    };
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    let hip = temporary.path().join("hip");
    let dual = temporary.path().join("hip-cuda");
    conversion::convert_with_devices(
        &source,
        &plan,
        &hip,
        &options,
        &Progress::disabled(),
        &conversion::ResumeOptions::default(),
        &parse_devices("hip:0")?,
    )?;
    conversion::convert_with_devices(
        &source,
        &plan,
        &dual,
        &options,
        &Progress::disabled(),
        &conversion::ResumeOptions::default(),
        &parse_devices("hip:0,cuda:0")?,
    )?;
    let mut checked = 0;
    for entry in fs::read_dir(&hip)? {
        let path = entry?.path();
        if path.extension().map(|ext| ext == "hgn").unwrap_or(false) {
            let name = path.file_name().expect("hgn file name");
            let paired = dual.join(name);
            assert!(paired.exists(), "missing {}", paired.display());
            assert_eq!(
                sha256_path(&path)?,
                sha256_path(&paired)?,
                "{name:?} bytes differ between hip:0 and hip:0,cuda:0"
            );
            checked += 1;
        }
    }
    assert!(checked >= 2, "expected model.hgn and ngram.hgn outputs");
    Ok(())
}
