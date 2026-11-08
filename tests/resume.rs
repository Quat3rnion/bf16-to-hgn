mod support;

use anyhow::{bail, Context, Result};
use bf16_to_hgn::{
    conversion::{self, AbortHook, ResumeOptions},
    hf::Snapshot,
    manifest::{Options, OutputProfile},
    progress::Progress,
    resume, verify,
    verify_gpu::DecodeBackend,
    ConversionPlan,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

const CHILD_TEST: &str = "killed_child_resume_matches_fresh";

fn options() -> Options {
    Options {
        seed: 7,
        threads: 1,
        ht_beam: 8,
        ..Options::default()
    }
}

fn sha256_tree(root: &Path) -> Result<BTreeMap<String, String>> {
    fn walk(dir: &Path, root: &Path, tree: &mut BTreeMap<String, String>) -> Result<()> {
        let mut entries = fs::read_dir(dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, root, tree)?;
            } else {
                let mut bytes = Vec::new();
                File::open(&path)?.read_to_end(&mut bytes)?;
                let mut hash = Sha256::new();
                hash.update(&bytes);
                tree.insert(
                    path.strip_prefix(root).unwrap_or(&path).display().to_string(),
                    format!("{:x}", hash.finalize()),
                );
            }
        }
        Ok(())
    }
    let mut tree = BTreeMap::new();
    walk(root, root, &mut tree)?;
    Ok(tree)
}

/// Deterministic (staging, journal, lock) sibling paths for an output that
/// does not exist yet, exactly as `convert --resume` computes them.
fn staging_for(
    snapshot: &Snapshot,
    plan: &ConversionPlan,
    output: &Path,
    options: &Options,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let destination = conversion::destination_path(snapshot, output)?;
    let build = resume::fingerprint(snapshot, plan, options, &destination)?;
    let parent = destination.parent().context("output has no parent")?;
    Ok(resume::staging_paths(parent, &build.name_digest))
}

fn resume_leftovers(parent: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in fs::read_dir(parent)? {
        let path = entry?.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        if name.starts_with(".bf16-to-hgn-resume-") {
            found.push(path);
        }
    }
    Ok(found)
}

/// Core property: an interrupted --resume run continues and produces output
/// byte-identical to an uninterrupted fresh conversion; verify passes.
#[test]
fn interrupted_resume_reproduces_fresh_output_exactly() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();

    let out_a = temporary.path().join("a");
    conversion::convert(&snapshot, &plan, &out_a, &options(), &progress)?;
    let expected = sha256_tree(&out_a)?;

    let out_b = temporary.path().join("b");
    let hook: AbortHook = Arc::new(|kind: &str, count: u64| {
        if kind == "tensor" && count >= 3 {
            bail!("test interrupt after {count} journaled tensors");
        }
        Ok(())
    });
    let error = conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out_b,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: Some(hook),
        },
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("--resume"),
        "failure must hint rerun with --resume: {error:#}"
    );
    let (staging, journal, _lock) = staging_for(&snapshot, &plan, &out_b, &options())?;
    assert!(staging.is_dir() && staging.join("model.hgn").is_file());
    assert!(journal.is_file());
    assert!(fs::read_to_string(&journal)?.lines().count() >= 4);
    assert!(!out_b.exists(), "interrupted run must not publish");

    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out_b,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    assert_eq!(sha256_tree(&out_b)?, expected);
    verify::verify(
        &snapshot,
        &plan,
        &out_b,
        DecodeBackend::Cpu,
        &progress,
    )?;
    assert!(
        resume_leftovers(temporary.path())?.is_empty(),
        "success must remove staging, journal and lock"
    );
    Ok(())
}

/// Bonus granularity: interrupting between chunk units of a multi-unit tensor
/// resumes mid-tensor (replayed units are re-hashed from source, never
/// re-encoded) and still reproduces the fresh output exactly.
#[test]
fn chunk_journal_resumes_mid_tensor() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();
    let out_fresh = temporary.path().join("fresh");
    conversion::convert(&snapshot, &plan, &out_fresh, &options(), &progress)?;
    let expected = sha256_tree(&out_fresh)?;

    let mut interrupted_mid_chunk = false;
    for limit in 1u64..=24 {
        let out = temporary.path().join(format!("chunk{limit}"));
        let hook: AbortHook = Arc::new(move |kind: &str, count: u64| {
            if kind == "chunk" && count >= limit {
                bail!("test interrupt after {count} journaled chunks");
            }
            Ok(())
        });
        let result = conversion::convert_with_resume(
            &snapshot,
            &plan,
            &out,
            &options(),
            &progress,
            &ResumeOptions {
                enabled: true,
                abort_hook: Some(hook),
            },
        );
        if result.is_ok() {
            continue; // conversion completed before `limit` chunk units
        }
        let (staging, journal, _lock) = staging_for(&snapshot, &plan, &out, &options())?;
        let text = fs::read_to_string(&journal)?;
        let last = text.lines().last().context("empty journal")?;
        if !last.contains("\"kind\":\"chunk\"") || !staging.is_dir() {
            continue; // interrupted on a tensor/file boundary, not mid-tensor
        }
        // The tail record is a chunk unit: the tensor is partially written.
        conversion::convert_with_resume(
            &snapshot,
            &plan,
            &out,
            &options(),
            &progress,
            &ResumeOptions {
                enabled: true,
                abort_hook: None,
            },
        )?;
        assert_eq!(sha256_tree(&out)?, expected, "chunk resume at limit {limit}");
        verify::verify(&snapshot, &plan, &out, DecodeBackend::Cpu, &progress)?;
        interrupted_mid_chunk = true;
        break;
    }
    assert!(
        interrupted_mid_chunk,
        "no interruption landed mid-tensor on a chunk boundary"
    );
    Ok(())
}

/// A changed seed/beam/threads/profile alters the journal header fingerprint:
/// resume must hard-fail (no silent fresh fallback), and the original options
/// must still resume afterwards.
#[test]
fn fingerprint_mismatch_is_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();
    let out = temporary.path().join("mismatch");
    let hook: AbortHook = Arc::new(|kind: &str, count: u64| {
        if kind == "tensor" && count >= 1 {
            bail!("test interrupt");
        }
        Ok(())
    });
    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: Some(hook),
        },
    )
    .unwrap_err();

    for changed in [
        Options {
            seed: 99,
            ..options()
        },
        Options {
            ht_beam: 16,
            ..options()
        },
        Options {
            threads: 2,
            ..options()
        },
        Options {
            profile: OutputProfile::Ht43,
            ..options()
        },
    ] {
        let error = conversion::convert_with_resume(
            &snapshot,
            &plan,
            &out,
            &changed,
            &progress,
            &ResumeOptions {
                enabled: true,
                abort_hook: None,
            },
        )
        .unwrap_err();
        let text = format!("{error:#}");
        assert!(
            text.contains("fingerprint mismatch"),
            "unexpected rejection text: {text}"
        );
        assert!(
            !out.exists(),
            "a rejected resume must not publish or start over"
        );
    }

    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    assert!(out.join("conversion-manifest.json").is_file());
    Ok(())
}

/// Corruption inside the last journaled span is caught by the resume
/// verification pass: the journal falls back to before the bad unit, the
/// tensor is re-encoded, and the final output still matches a fresh run.
#[test]
fn corrupted_last_unit_falls_back_and_reencodes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();
    let out_fresh = temporary.path().join("fresh");
    conversion::convert(&snapshot, &plan, &out_fresh, &options(), &progress)?;
    let expected = sha256_tree(&out_fresh)?;

    let out = temporary.path().join("corrupt");
    let hook: AbortHook = Arc::new(|kind: &str, count: u64| {
        if kind == "tensor" && count >= 4 {
            bail!("test interrupt");
        }
        Ok(())
    });
    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: Some(hook),
        },
    )
    .unwrap_err();
    let (staging, journal, _lock) = staging_for(&snapshot, &plan, &out, &options())?;
    let text = fs::read_to_string(&journal)?;
    let lines_before = text.lines().count();
    let last_tensor = text
        .lines()
        .rev()
        .find(|line| line.contains("\"kind\":\"tensor\""))
        .context("no tensor record in journal")?;
    let record: serde_json::Value = serde_json::from_str(last_tensor)?;
    let file_index = record["file"].as_u64().context("file index")? as usize;
    let offset = record["payload_offset"].as_u64().context("offset")?;
    let size = record["payload_size"].as_u64().context("size")?;
    let path = staging.join(&plan.files[file_index].filename);
    {
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        file.seek(SeekFrom::Start(offset + size / 2))?;
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)?;
        file.seek(SeekFrom::Start(offset + size / 2))?;
        file.write_all(&[byte[0] ^ 0xFF])?;
        file.sync_all()?;
    }

    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    assert_eq!(sha256_tree(&out)?, expected);
    verify::verify(&snapshot, &plan, &out, DecodeBackend::Cpu, &progress)?;
    // The corrupt unit and everything after it were dropped and re-appended.
    let resumed = fs::read_to_string(&journal).unwrap_or_default();
    assert!(
        resumed.is_empty() || resumed.lines().count() != lines_before,
        "journal should have been rewritten after fallback (published run removes it)"
    );
    Ok(())
}

/// A truncated (wrong-sized) staging file invalidates its journal records:
/// they are dropped and the file is recreated from scratch.
#[test]
fn truncated_staging_file_is_recreated() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();
    let out_fresh = temporary.path().join("fresh");
    conversion::convert(&snapshot, &plan, &out_fresh, &options(), &progress)?;
    let expected = sha256_tree(&out_fresh)?;

    let out = temporary.path().join("truncated");
    let hook: AbortHook = Arc::new(|kind: &str, count: u64| {
        if kind == "tensor" && count >= 2 {
            bail!("test interrupt");
        }
        Ok(())
    });
    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: Some(hook),
        },
    )
    .unwrap_err();
    let (staging, journal, _lock) = staging_for(&snapshot, &plan, &out, &options())?;
    let model = staging.join("model.hgn");
    let size = fs::metadata(&model)?.len();
    OpenOptions::new()
        .write(true)
        .open(&model)?
        .set_len(size / 2)?;
    assert!(fs::read_to_string(&journal)?.lines().count() > 1);

    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    assert_eq!(sha256_tree(&out)?, expected);
    verify::verify(&snapshot, &plan, &out, DecodeBackend::Cpu, &progress)?;
    Ok(())
}

/// A second concurrent run on the same fingerprint fails fast while the lock
/// is held, and proceeds once it is released.
#[test]
fn concurrent_resume_fails_fast_on_lock() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let progress = Progress::disabled();
    let out = temporary.path().join("locked");
    let (_staging, _journal, lock) = staging_for(&snapshot, &plan, &out, &options())?;
    let held = resume::LockGuard::acquire(&lock)?;
    let error = conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )
    .unwrap_err();
    assert!(
        format!("{error:#}").contains("already running"),
        "unexpected lock error: {error:#}"
    );
    drop(held);
    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &out,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    assert!(out.join("conversion-manifest.json").is_file());
    Ok(())
}

/// Hard-kill equivalence: a child process exits(9) mid-conversion (no
/// destructors, no cleanup — exactly what SIGKILL leaves behind); the rerun
/// resumes and the published tree is byte-identical to a fresh conversion.
#[test]
fn killed_child_resume_matches_fresh() -> Result<()> {
    if let Some(dir) = std::env::var_os("HGN_RESUME_CHILD_DIR") {
        return child_main(Path::new(&dir));
    }
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    fs::write(
        temporary.path().join("plan.json"),
        serde_json::to_vec(&plan)?,
    )?;
    let exe = std::env::current_exe()?;
    let child_out = temporary.path().join("child_out");

    let abort = std::process::Command::new(&exe)
        .arg(CHILD_TEST)
        .arg("--exact")
        .arg("--nocapture")
        .env("HGN_RESUME_CHILD_DIR", temporary.path())
        .env("HGN_RESUME_CHILD_MODE", "abort")
        .output()?;
    assert_eq!(
        abort.status.code(),
        Some(9),
        "child abort exit status; stderr: {}",
        String::from_utf8_lossy(&abort.stderr)
    );
    let (staging, journal, _lock) = staging_for(&snapshot, &plan, &child_out, &options())?;
    assert!(staging.is_dir() && journal.is_file());
    assert!(
        fs::read_to_string(&journal)?.lines().count() >= 3,
        "journal must hold the header plus the two completed tensors"
    );
    assert!(!child_out.exists());

    let finish = std::process::Command::new(&exe)
        .arg(CHILD_TEST)
        .arg("--exact")
        .arg("--nocapture")
        .env("HGN_RESUME_CHILD_DIR", temporary.path())
        .env("HGN_RESUME_CHILD_MODE", "finish")
        .output()?;
    assert!(
        finish.status.success(),
        "child finish failed; stderr: {}",
        String::from_utf8_lossy(&finish.stderr)
    );
    let stdout = String::from_utf8(finish.stdout)?;
    let line = stdout
        .lines()
        .find(|line| line.starts_with('{'))
        .context("child printed no output tree")?;
    let child_tree: BTreeMap<String, String> = serde_json::from_str(line)?;

    let progress = Progress::disabled();
    let out_a = temporary.path().join("parent_fresh");
    conversion::convert(&snapshot, &plan, &out_a, &options(), &progress)?;
    assert_eq!(sha256_tree(&out_a)?, child_tree);
    verify::verify(
        &snapshot,
        &plan,
        &child_out,
        DecodeBackend::Cpu,
        &progress,
    )?;
    assert!(!journal.exists() && !staging.exists());
    Ok(())
}

fn child_main(dir: &Path) -> Result<()> {
    let snapshot = Snapshot::open(&dir.join("source"))?;
    let plan: ConversionPlan = serde_json::from_slice(&fs::read(dir.join("plan.json"))?)?;
    let output = dir.join("child_out");
    let progress = Progress::disabled();
    if std::env::var("HGN_RESUME_CHILD_MODE").as_deref() == Ok("abort") {
        let hook: AbortHook = Arc::new(|kind: &str, count: u64| {
            if kind == "tensor" && count >= 2 {
                // Hard exit: skips destructors exactly like SIGKILL.
                std::process::exit(9);
            }
            Ok(())
        });
        let _ = conversion::convert_with_resume(
            &snapshot,
            &plan,
            &output,
            &options(),
            &progress,
            &ResumeOptions {
                enabled: true,
                abort_hook: Some(hook),
            },
        );
        bail!("abort hook did not fire");
    }
    conversion::convert_with_resume(
        &snapshot,
        &plan,
        &output,
        &options(),
        &progress,
        &ResumeOptions {
            enabled: true,
            abort_hook: None,
        },
    )?;
    println!("{}", serde_json::to_string(&sha256_tree(&output)?)?);
    Ok(())
}

/// The CLI surface documents the flag.
#[test]
fn cli_convert_documents_resume_flag() -> Result<()> {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
        .args(["convert", "--help"])
        .output()?;
    let help = String::from_utf8(output.stdout)?;
    assert!(
        help.contains("--resume"),
        "convert --help must document --resume:\n{help}"
    );
    Ok(())
}
