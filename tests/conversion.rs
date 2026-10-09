mod support;

use anyhow::Result;
use bf16_to_hgn::{architecture, container, conversion, hf::Snapshot, manifest::Options, verify};
use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
};

fn options() -> Options {
    Options {
        seed: 7,
        threads: 1,
        ht_beam: 8,
    }
}

#[test]
fn actual_fixture_converts_and_verifies_every_source_and_output() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest = conversion::convert(&snapshot, &plan, &output, &options())?;
    let report = verify::verify(&snapshot, &plan, &output)?;
    assert_eq!(report.source_tensors, snapshot.tensors.len());
    assert_eq!(
        report.output_tensors,
        plan.files.iter().map(|f| f.tensors.len()).sum::<usize>()
    );
    assert_eq!(report.errors.len(), manifest.sources.len() - 1); // Two ngram shards become one output.
    assert!(report
        .errors
        .values()
        .all(|error| error.error_squared.is_finite() && error.values > 0));
    assert!(report
        .errors
        .values()
        .any(|error| error.error_squared > 0.0));
    assert_eq!(
        fs::read(output.join("config.json"))?,
        fs::read(snapshot.root.join("config.json"))?
    );
    assert_eq!(
        fs::read(output.join("tokenizer/config.json"))?,
        fs::read(snapshot.root.join("config.json"))?
    );
    let gate = manifest.containers[0]
        .tensors
        .iter()
        .find(|t| t.entry.name == "fixture.experts.gate_up.weight")
        .expect("gate/up entry");
    assert_eq!(gate.entry.encoding.code(), 23);
    assert_eq!(gate.entry.encoding.variant(), 128);
    assert_eq!(gate.entry.shape, [2, 256, 128]);
    Ok(())
}

#[test]
fn same_source_seed_and_compiled_encoder_produce_identical_bytes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let first = temporary.path().join("first");
    let second = temporary.path().join("second");
    let a = conversion::convert(&snapshot, &plan, &first, &options())?;
    let b = conversion::convert(&snapshot, &plan, &second, &options())?;
    assert_eq!(a, b);
    for file in &plan.files {
        assert_eq!(
            fs::read(first.join(&file.filename))?,
            fs::read(second.join(&file.filename))?
        );
    }
    Ok(())
}

#[test]
fn existing_destination_is_not_clobbered() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    fs::create_dir(&output)?;
    fs::write(output.join("sentinel"), b"untouched")?;
    assert!(conversion::convert(&snapshot, &plan, &output, &options()).is_err());
    assert_eq!(fs::read(output.join("sentinel"))?, b"untouched");
    assert_eq!(fs::read_dir(&output)?.count(), 1);
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[test]
fn output_inside_input_is_rejected_without_writes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = snapshot.root.join("converted");
    assert!(conversion::convert(&snapshot, &plan, &output, &options()).is_err());
    assert!(!output.exists());
    assert_no_staging(&snapshot.root)?;
    Ok(())
}

#[test]
fn failed_encoding_cleans_staging_and_never_publishes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let info = &snapshot.tensors["fixture.affine6.weight"];
    let mut file = OpenOptions::new()
        .write(true)
        .open(snapshot.root.join(&info.shard))?;
    file.seek(SeekFrom::Start(info.offset))?;
    file.write_all(&0x7fc0u16.to_le_bytes())?;
    file.sync_all()?;
    drop(file);
    let snapshot = Snapshot::open(&snapshot.root)?;
    let output = temporary.path().join("converted");
    let error = conversion::convert(&snapshot, &plan, &output, &options())
        .expect_err("nonfinite BF16 must fail");
    assert!(format!("{error:#}").contains("nonfinite"));
    assert!(!output.exists());
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[test]
fn modified_payload_and_manifest_coverage_are_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest = conversion::convert(&snapshot, &plan, &output, &options())?;
    let index = manifest.containers[0]
        .tensors
        .iter()
        .position(|t| t.entry.name == "fixture.affine6.weight")
        .expect("affine entry");
    let offset = manifest.containers[0].tensors[index].entry.offset;
    let path = output.join("model.hgn");
    let original = fs::read(&path)?;
    let mut file = OpenOptions::new().write(true).open(&path)?;
    file.seek(SeekFrom::Start(offset))?;
    file.write_all(&[original[offset as usize] ^ 1])?;
    file.sync_all()?;
    assert!(verify::verify(&snapshot, &plan, &output).is_err());
    drop(file);
    fs::write(&path, original)?;
    let mut corrupt = manifest;
    corrupt.sources.pop();
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec(&corrupt)?,
    )?;
    assert!(verify::verify(&snapshot, &plan, &output).is_err());
    Ok(())
}

#[test]
fn unmanifested_output_and_source_mutations_are_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    conversion::convert(&snapshot, &plan, &output, &options())?;
    fs::write(output.join("unknown.hgn"), b"unexpected")?;
    assert!(verify::verify(&snapshot, &plan, &output).is_err());
    fs::remove_file(output.join("unknown.hgn"))?;
    let info = &snapshot.tensors["fixture.raw.weight"];
    let mut file = OpenOptions::new()
        .write(true)
        .open(snapshot.root.join(&info.shard))?;
    file.seek(SeekFrom::Start(info.offset))?;
    file.write_all(&0x3f80u16.to_le_bytes())?;
    file.sync_all()?;
    drop(file);
    let changed = Snapshot::open(&snapshot.root)?;
    assert!(verify::verify(&changed, &plan, &output).is_err());
    Ok(())
}

#[test]
fn malformed_hgn_geometry_and_unused_shape_slots_are_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    conversion::convert(&snapshot, &plan, &output, &options())?;
    let path = output.join("model.hgn");
    let mut bytes = fs::read(&path)?;
    bytes[container::HEADER_SIZE as usize + 120..container::HEADER_SIZE as usize + 128]
        .copy_from_slice(&1i64.to_le_bytes());
    fs::write(&path, bytes)?;
    assert!(container::Reader::open(&path).is_err());
    Ok(())
}

#[test]
fn public_cli_has_no_structural_fixture_bypass() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, _) = support::fixture(temporary.path())?;
    assert!(architecture::plan(&snapshot).is_err());
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
        .arg("plan")
        .arg("--source")
        .arg(&snapshot.root)
        .output()?;
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
    Ok(())
}

fn assert_no_staging(parent: &std::path::Path) -> Result<()> {
    assert!(fs::read_dir(parent)?.all(|entry| !entry
        .expect("directory entry")
        .file_name()
        .to_string_lossy()
        .starts_with(".bf16-to-hgn-")));
    Ok(())
}
