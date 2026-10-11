mod support;

use anyhow::Result;
use bf16_to_hgn::{
    architecture,
    codec::{Encoding, HtBackend},
    container, conversion,
    hf::Snapshot,
    manifest::{Options, OutputProfile},
    progress::Progress,
    source::{self, SourceKind, SourceOptions, TensorSource},
    verify,
    verify_gpu::DecodeBackend,
};
use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
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
fn actual_fixture_converts_and_verifies_every_source_and_output() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    let report = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
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
fn eta_fixture_converts_and_verifies_mixed_encodings_and_coverage() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::eta_fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let progress = Progress::disabled();
    let manifest = conversion::convert(&snapshot, &plan, &output, &options(), &progress)?;
    let report = verify::verify(&snapshot, &plan, &output, DecodeBackend::Cpu, &progress)?;
    assert_eq!(manifest.sources.len(), snapshot.tensors.len());
    assert_eq!(report.source_tensors, snapshot.tensors.len());
    assert_eq!(
        report.output_tensors,
        plan.files
            .iter()
            .map(|file| file.tensors.len())
            .sum::<usize>()
    );
    assert_eq!(report.errors.len(), manifest.sources.len() - 1);
    assert!(report
        .errors
        .values()
        .all(|error| error.error_squared.is_finite() && error.values > 0));
    for (name, encoding, bits) in [
        ("fixture.dense.weight", Encoding::Ht, 4),
        ("fixture.dense_ht3.weight", Encoding::Ht3, 3),
    ] {
        let entry = &manifest.containers[0]
            .tensors
            .iter()
            .find(|record| record.entry.name == name)
            .expect("encoded HT matrix")
            .entry;
        assert_eq!(entry.encoding, encoding);
        assert_eq!(entry.shape, [128, 128]);
        assert_eq!(entry.size, 128 * 128 * bits / 8);
        assert_eq!(
            report.errors[&format!("model.hgn:{name}")].values,
            128 * 128
        );
    }
    for encoding in [Encoding::I4r, Encoding::Q8G64] {
        assert_eq!(
            manifest.containers[0]
                .tensors
                .iter()
                .filter(|record| record.entry.encoding == encoding)
                .count(),
            2
        );
    }
    assert_eq!(
        report.errors["qwen38-flash-next-ngram.hgn:layers.1.ple.ngram_embedding.weight"].values,
        2 * 3 * 160
    );
    Ok(())
}

#[test]
fn same_source_seed_and_compiled_encoder_produce_identical_bytes() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let first = temporary.path().join("first");
    let second = temporary.path().join("second");
    let a = conversion::convert(&snapshot, &plan, &first, &options(), &Progress::disabled())?;
    let b = conversion::convert(&snapshot, &plan, &second, &options(), &Progress::disabled())?;
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
    assert!(
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled()).is_err()
    );
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
    assert!(
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled()).is_err()
    );
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
    let error = conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())
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
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
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
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    drop(file);
    fs::write(&path, original)?;
    let mut corrupt = manifest;
    corrupt.sources.pop();
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec(&corrupt)?,
    )?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    Ok(())
}

#[test]
fn unmanifested_output_and_source_mutations_are_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    fs::write(output.join("unknown.hgn"), b"unexpected")?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
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
    assert!(verify::verify(
        &changed,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    Ok(())
}

#[test]
fn malformed_hgn_geometry_and_unused_shape_slots_are_rejected() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
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
    assert!(architecture::plan(&snapshot, &Options::default()).is_err());
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
        .arg("plan")
        .arg("--source")
        .arg(&snapshot.root)
        .output()?;
    assert_eq!(result.status.code(), Some(2));
    assert!(result.stdout.is_empty());
    Ok(())
}

#[test]
fn legacy_options_default_to_v2_and_profiles_serialize_explicitly() -> Result<()> {
    let old: Options = serde_json::from_str(r#"{"seed":7,"threads":1,"ht_beam":8}"#)?;
    assert_eq!(old.profile, OutputProfile::V2);
    assert_eq!(old.ht_backend, HtBackend::Cpu);
    assert_eq!(serde_json::to_value(old)?["profile"], "v2");
    let hip = Options {
        ht_backend: HtBackend::Hip,
        ..options()
    };
    hip.validate()?; // Semantic validation must not require a GPU or the feature.
    assert_eq!(serde_json::to_value(hip)?["ht_backend"], "hip");
    assert!(serde_json::from_str::<Options>(
        r#"{"seed":7,"threads":1,"ht_beam":8,"ht_backend":"cuda"}"#
    )
    .is_err());
    assert_ne!(OutputProfile::V2.policy(), OutputProfile::Ht43.policy());
    Ok(())
}

#[test]
fn source_factory_detects_hf_and_rejects_unused_sidecars() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, _) = support::fixture(temporary.path())?;
    let opened = source::open(&snapshot.root, None, &SourceOptions::default())?;
    assert_eq!(opened.kind(), SourceKind::Hf);
    assert_eq!(opened.tensors(), &snapshot.tensors);
    assert_eq!(opened.config(), &snapshot.config);
    assert!(source::open(
        &snapshot.root,
        Some(SourceKind::Hf),
        &SourceOptions {
            mtp: Some(temporary.path().join("unused.hgn")),
            ..SourceOptions::default()
        },
    )
    .is_err());
    assert!(opened.asset_path("../config.json").is_err());
    assert_eq!(
        opened.asset_path("vocab.json")?,
        snapshot.root.join("vocab.json")
    );
    Ok(())
}

#[test]
fn external_asset_root_is_protected_from_output_overlap() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let first = temporary.path().join("first");
    let second = temporary.path().join("second");
    fs::create_dir(&first)?;
    fs::create_dir(&second)?;
    let (snapshot, _) = support::fixture(&first)?;
    let (assets, _) = support::fixture(&second)?;
    let source = support::FixtureAssets {
        tensors: &snapshot,
        assets: &assets,
    };
    let output = assets.root.join("converted");
    assert!(conversion::destination_path(&source, &output).is_err());
    assert!(!output.exists());
    assert_no_staging(&assets.root)?;
    Ok(())
}

#[test]
fn ht43_hf_matrix_fixture_converts_with_explicit_canonical_provenance() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::matrix_fixture(temporary.path(), OutputProfile::Ht43)?;
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    let output = temporary.path().join("ht43");
    let manifest = conversion::convert(&snapshot, &plan, &output, &options, &Progress::disabled())?;
    let report = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(manifest.schema, 2);
    assert_eq!(manifest.source_kind, SourceKind::Hf);
    assert!(manifest.index_sha256.is_some());
    assert_eq!(report.source_tensors, snapshot.tensors.len());
    let ht3 = manifest.containers[0]
        .tensors
        .iter()
        .find(|record| record.entry.name == "layers.1.self_attn.q_proj.weight")
        .expect("HT3 fixture");
    assert_eq!(
        (ht3.entry.encoding.code(), ht3.entry.encoding.variant()),
        (16, 0x1206)
    );
    assert_eq!(ht3.entry.size, 128 * 128 * 3 / 8);
    for record in &manifest.sources {
        assert_eq!(record.sha256, snapshot.hash_tensor(&record.tensor.name)?);
    }
    let shard = manifest.source_provenance["dependencies"]
        .as_array()
        .expect("dependency list")
        .iter()
        .find(|record| record["file"] == "model.safetensors")
        .expect("shard dependency");
    assert!(shard.get("header_sha256").is_some());
    assert!(shard.get("sha256").is_none()); // No falsely claimed whole-shard digest.
    assert!(
        !serde_json::to_string(&manifest)?.contains(&snapshot.root.to_string_lossy().to_string())
    );
    Ok(())
}

#[test]
fn imported_f32_bf16_boundary_preserves_source_and_rounds_once() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let values = [1.00390625f32, 1.005859375, -0.0];
    let (hgn, plan) = support::imported_norm_fixture(temporary.path(), &values)?;
    let source = support::FixtureAssets {
        tensors: &hgn,
        assets: &assets,
    };
    let name = "model.language_model.layers.0.input_layernorm.weight";
    let canonical = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(source.read_bytes(name, 0, canonical.len())?, canonical);
    assert_eq!(
        source
            .read_f32(name, 0, 3)?
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let output = temporary.path().join("ht43");
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    let mut manifest =
        conversion::convert(&source, &plan, &output, &options, &Progress::disabled())?;
    let report = verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(manifest.source_kind, SourceKind::Hgn);
    assert!(manifest.index_sha256.is_none());
    assert!(serde_json::to_value(&manifest)?
        .get("index_sha256")
        .is_none());
    assert_eq!(
        manifest.sources[0].tensor.dtype,
        bf16_to_hgn::hf::Dtype::F32
    );
    assert_eq!(manifest.sources[0].sha256, source.hash_tensor(name)?);
    let mut reader = container::Reader::open(&output.join("model.hgn"))?;
    assert_eq!(
        reader.read_at(0, 0, 6)?,
        [0x80, 0x3f, 0x81, 0x3f, 0x00, 0x80]
    );
    assert!(report
        .errors
        .values()
        .any(|totals| totals.error_squared > 0.0));
    manifest.source_provenance = serde_json::json!({ "wrong": "physical source" });
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec(&manifest)?,
    )?;
    assert!(verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    Ok(())
}

#[test]
fn imported_f32_overflow_is_rejected_before_publication() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (hgn, plan) = support::imported_norm_fixture(temporary.path(), &[f32::MAX])?;
    let source = support::FixtureAssets {
        tensors: &hgn,
        assets: &assets,
    };
    let output = temporary.path().join("ht43");
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    assert!(conversion::convert(&source, &plan, &output, &options, &Progress::disabled()).is_err());
    assert!(!output.exists());
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[test]
fn profile_and_source_flags_do_not_bypass_public_architecture_validation() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, _) = support::fixture(temporary.path())?;
    for command in ["inspect", "plan", "convert"] {
        let mut cli = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"));
        cli.arg(command).arg("--source").arg(&snapshot.root).args([
            "--source-format",
            "hf",
            "--profile",
            "ht43",
        ]);
        if command == "convert" {
            cli.arg("--output")
                .arg(temporary.path().join("output"))
                .arg("--dry-run");
        }
        let result = cli.output()?;
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        assert!(!String::from_utf8_lossy(&result.stderr).contains("unexpected argument"));
    }
    assert!(!temporary.path().join("output").exists());
    Ok(())
}

#[test]
fn real_gguf_f32_q8_and_integer_metadata_convert_and_verify_ht43() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (gguf, plan) = support::gguf_matrix_fixture(temporary.path())?;
    let source = support::FixtureAssets {
        tensors: &gguf,
        assets: &assets,
    };
    let q8 = "model.language_model.layers.1.self_attn.q_proj.weight";
    let values = source.read_f32(q8, 0, 32)?;
    assert_eq!(values[0], -31.0 / 32.0);
    assert_eq!(values[31].to_bits(), 0.0f32.to_bits());
    let multipliers = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
    assert_eq!(
        source.read_bytes(multipliers, 0, 24)?,
        [1i64, 3, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let output = temporary.path().join("ht43");
    let options = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    let manifest = conversion::convert(&source, &plan, &output, &options, &Progress::disabled())?;
    let report = verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(manifest.source_kind, SourceKind::Gguf);
    assert!(manifest.index_sha256.is_none());
    assert_eq!(report.source_tensors, 133);
    assert_eq!(report.output_tensors, 10);
    assert_eq!(report.errors.len(), 6); // Two matrices, three exact I64 vectors, and one table.
    assert_eq!(
        manifest
            .sources
            .iter()
            .find(|r| r.tensor.name == q8)
            .expect("Q8 source")
            .sha256,
        source.hash_tensor(q8)?
    );
    assert!(!serde_json::to_string(&manifest)?
        .contains(&temporary.path().to_string_lossy().to_string()));
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

#[test]
fn verification_accepts_hip_metadata_and_schema2_without_backend() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    assert_eq!(manifest.schema, 2);
    assert_eq!(manifest.options.ht_backend, HtBackend::Cpu);
    assert_eq!(manifest.plan, plan);
    let current = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(verify::manifest_options(&output)?, manifest.options);
    let mut json = serde_json::to_value(&manifest)?;
    json["options"]["ht_backend"] = serde_json::json!("hip");
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    assert_eq!(
        verify::manifest_options(&output)?.ht_backend,
        HtBackend::Hip
    );
    verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    json["options"]
        .as_object_mut()
        .expect("options object")
        .remove("ht_backend");
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    // Exercise the full canonical manifest and fresh-plan verification gate,
    // not only Options deserialization. The genuine CPU payloads are unchanged.
    let legacy = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(legacy.source_tensors, current.source_tensors);
    assert_eq!(legacy.output_tensors, current.output_tensors);
    assert_eq!(
        serde_json::to_value(&legacy.errors)?,
        serde_json::to_value(&current.errors)?
    );
    json["options"]["unknown_backend_setting"] = serde_json::json!(1);
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    assert!(verify::manifest_options(&output).is_err());
    let error = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("unknown or noncanonical manifest fields"));
    json["options"]
        .as_object_mut()
        .expect("options object")
        .remove("unknown_backend_setting");
    json["plan"]["architecture"] = serde_json::json!("not-the-fresh-source-plan");
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    let error = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("manifest plan differs from the fresh source plan"));
    Ok(())
}

#[cfg(not(feature = "hip"))]
#[test]
fn hip_conversion_requires_feature_before_staging_or_source_open() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let hip = Options {
        ht_backend: HtBackend::Hip,
        ..options()
    };
    let error =
        conversion::convert(&snapshot, &plan, &output, &hip, &Progress::disabled()).unwrap_err();
    assert!(error.to_string().contains("--features hip"));
    assert!(!output.exists());
    assert_no_staging(temporary.path())?;
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
        .args(["convert", "--ht-backend", "hip", "--source"])
        .arg(temporary.path().join("missing-source"))
        .arg("--output")
        .arg(&output)
        .output()?;
    assert_eq!(result.status.code(), Some(2));
    assert!(String::from_utf8(result.stderr)?.contains("--features hip"));
    assert!(result.stdout.is_empty());
    assert!(!output.exists());
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[cfg(not(feature = "hip"))]
#[test]
fn hip_verification_requires_feature_even_without_eligible_tensors() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (hgn, plan) = support::imported_norm_fixture(temporary.path(), &[1.0, -0.0, 3.125])?;
    let source = support::FixtureAssets {
        tensors: &hgn,
        assets: &assets,
    };
    assert!(plan
        .files
        .iter()
        .flat_map(|file| &file.tensors)
        .all(|tensor| {
            tensor.encoding.ht_bits().is_none()
                && tensor.encoding != bf16_to_hgn::codec::Encoding::I4r
        }));
    let output = temporary.path().join("converted");
    let opts = Options {
        profile: OutputProfile::Ht43,
        ..options()
    };
    conversion::convert(&source, &plan, &output, &opts, &Progress::disabled())?;
    verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::default(),
        &Progress::default(),
    )?;
    let error = verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::Hip,
        &Progress::disabled(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("--features hip"));
    Ok(())
}

#[test]
fn describing_hip_does_not_initialize_a_device_or_bypass_architecture() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, _) = support::fixture(temporary.path())?;
    for command in ["inspect", "plan", "convert"] {
        let mut cli = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"));
        cli.args([command, "--ht-backend", "hip", "--source"])
            .arg(&snapshot.root);
        if command == "convert" {
            cli.arg("--dry-run")
                .arg("--output")
                .arg(temporary.path().join("converted"));
        }
        let result = cli.output()?;
        assert_eq!(result.status.code(), Some(2)); // Structural config remains rejected.
        let error = String::from_utf8(result.stderr)?;
        assert!(!error.contains("--features hip"));
        assert!(!error.contains("HIP HT preflight"));
        assert!(result.stdout.is_empty());
    }
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires an available gfx1151 HIP device; encodes and publishes real tiny files"]
fn hip_tiny_conversion_matches_cpu_all_containers_and_verifies() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::hip_fixture(temporary.path())?;
    assert!(plan
        .files
        .iter()
        .flat_map(|file| &file.tensors)
        .any(|t| t.encoding == Encoding::Ht3));
    assert!(plan
        .files
        .iter()
        .flat_map(|file| &file.tensors)
        .any(|t| t.encoding == Encoding::Ht));
    let mut baseline = None;
    for backend in [HtBackend::Cpu, HtBackend::Hip] {
        let name = if backend == HtBackend::Cpu {
            "cpu"
        } else {
            "hip"
        };
        let output = temporary.path().join(name);
        let opts = Options {
            profile: OutputProfile::Ht43,
            ht_backend: backend,
            ..options()
        };
        let manifest =
            conversion::convert(&snapshot, &plan, &output, &opts, &Progress::disabled())?;
        assert_eq!(manifest.options.ht_backend, backend);
        verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled(),
        )?;
        if let Some(cpu) = &baseline {
            let cpu: &bf16_to_hgn::manifest::Manifest = cpu;
            assert_eq!(cpu.containers, manifest.containers);
            assert_eq!(cpu.sources, manifest.sources);
            assert_eq!(cpu.assets, manifest.assets);
            for container in &manifest.containers {
                assert_eq!(
                    fs::read(temporary.path().join("cpu").join(&container.filename))?,
                    fs::read(output.join(&container.filename))?,
                    "{}",
                    container.filename
                );
            }
        } else {
            baseline = Some(manifest);
        }
    }
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires an available gfx1151 HIP device; compares complete verification reports"]
fn hip_full_verification_reports_match_cpu_bitwise() -> Result<()> {
    for profile in [OutputProfile::V2, OutputProfile::Ht43] {
        let temporary = tempfile::tempdir()?;
        let (snapshot, plan) = if profile == OutputProfile::V2 {
            support::fixture(temporary.path())?
        } else {
            support::hip_fixture(temporary.path())?
        };
        let opts = Options {
            profile,
            ..options()
        };
        let output = temporary.path().join("converted");
        let manifest =
            conversion::convert(&snapshot, &plan, &output, &opts, &Progress::disabled())?;
        assert_eq!(manifest.options.ht_backend, HtBackend::Cpu);
        let encodings: Vec<_> = plan
            .files
            .iter()
            .flat_map(|file| &file.tensors)
            .map(|tensor| tensor.encoding)
            .collect();
        assert!(encodings.contains(&Encoding::Ht));
        assert!(encodings.contains(&if profile == OutputProfile::V2 {
            Encoding::I4r
        } else {
            Encoding::Ht3
        }));
        let cpu = verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled(),
        )?;
        let hip = verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Hip,
            &Progress::disabled(),
        )?;
        assert_eq!(cpu.manifest_schema, 2);
        assert!(cpu.provenance_attested);
        assert_eq!(cpu.errors.len(), hip.errors.len());
        for (name, reference) in &cpu.errors {
            let actual = &hip.errors[name];
            assert_eq!(reference.values, actual.values, "{name}");
            assert_eq!(
                reference.source_squared.to_bits(),
                actual.source_squared.to_bits(),
                "{name}"
            );
            assert_eq!(
                reference.error_squared.to_bits(),
                actual.error_squared.to_bits(),
                "{name}"
            );
        }
        assert_eq!(serde_json::to_value(cpu)?, serde_json::to_value(hip)?);
    }
    Ok(())
}

#[test]
fn backend_selection_preserves_semantic_option_limits() -> Result<()> {
    for backend in [HtBackend::Cpu, HtBackend::Hip] {
        for beam in [1, 256] {
            Options {
                ht_backend: backend,
                ht_beam: beam,
                ..options()
            }
            .validate()?;
        }
        for beam in [0, 257, usize::MAX] {
            assert!(Options {
                ht_backend: backend,
                ht_beam: beam,
                ..options()
            }
            .validate()
            .is_err());
        }
        for threads in [0, 65, usize::MAX] {
            assert!(Options {
                ht_backend: backend,
                threads,
                ..options()
            }
            .validate()
            .is_err());
        }
    }
    Ok(())
}
