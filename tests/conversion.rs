mod support;

use anyhow::Result;
use bf16_to_hgn::{
    architecture,
    codec::{Encoding, HtBackend, I4rBackend},
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
    assert_eq!(old.i4r_backend, I4rBackend::Cpu);
    assert_eq!(serde_json::to_value(old)?["profile"], "v2");
    for (ht_backend, ht_name) in [
        (HtBackend::Cpu, "cpu"),
        (HtBackend::Hip, "hip"),
        (HtBackend::Cuda, "cuda"),
    ] {
        for (i4r_backend, i4r_name) in [(I4rBackend::Cpu, "cpu"), (I4rBackend::Cuda, "cuda")] {
            let selected = Options {
                ht_backend,
                i4r_backend,
                ..options()
            };
            selected.validate()?; // Semantic validation requires neither a GPU nor its feature.
            let json = serde_json::to_value(&selected)?;
            assert_eq!(json["ht_backend"], ht_name);
            assert_eq!(json["i4r_backend"], i4r_name);
            assert_eq!(serde_json::from_value::<Options>(json)?, selected);
        }
    }
    assert!(serde_json::from_str::<Options>(
        r#"{"seed":7,"threads":1,"ht_beam":8,"ht_backend":"unknown-device"}"#
    )
    .is_err());
    for invalid in ["hip", "CUDA", "gpu", "unknown-device"] {
        let mut json = serde_json::to_value(options())?;
        json["i4r_backend"] = serde_json::json!(invalid);
        assert!(
            serde_json::from_value::<Options>(json).is_err(),
            "{invalid}"
        );
    }
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
fn verification_accepts_gpu_provenance_and_schema2_without_backends() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    assert_eq!(manifest.schema, 2);
    assert_eq!(manifest.options.ht_backend, HtBackend::Cpu);
    assert_eq!(manifest.options.i4r_backend, I4rBackend::Cpu);
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
    for (ht_backend, ht_name) in [
        (HtBackend::Cpu, "cpu"),
        (HtBackend::Hip, "hip"),
        (HtBackend::Cuda, "cuda"),
    ] {
        for (i4r_backend, i4r_name) in [(I4rBackend::Cpu, "cpu"), (I4rBackend::Cuda, "cuda")] {
            json["options"]["ht_backend"] = serde_json::json!(ht_name);
            json["options"]["i4r_backend"] = serde_json::json!(i4r_name);
            fs::write(
                output.join("conversion-manifest.json"),
                serde_json::to_vec_pretty(&json)?,
            )?;
            let provenance = verify::manifest_options(&output)?;
            assert_eq!(provenance.ht_backend, ht_backend);
            assert_eq!(provenance.i4r_backend, i4r_backend);
            let report = verify::verify(
                &snapshot,
                &plan,
                &output,
                DecodeBackend::Cpu,
                &Progress::disabled(),
            )?;
            assert_eq!(
                serde_json::to_value(report)?,
                serde_json::to_value(&current)?
            );
        }
    }
    for field in ["ht_backend", "i4r_backend"] {
        json["options"][field] = serde_json::json!("unknown-device");
        fs::write(
            output.join("conversion-manifest.json"),
            serde_json::to_vec_pretty(&json)?,
        )?;
        assert!(verify::manifest_options(&output).is_err());
        assert!(verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled(),
        )
        .is_err());
        json["options"][field] = serde_json::json!("cpu");
    }
    json["options"]
        .as_object_mut()
        .expect("options object")
        .remove("ht_backend");
    json["options"]
        .as_object_mut()
        .expect("options object")
        .remove("i4r_backend");
    fs::write(
        output.join("conversion-manifest.json"),
        serde_json::to_vec_pretty(&json)?,
    )?;
    assert_eq!(verify::manifest_options(&output)?, options());
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
    conversion_requires_feature_before_staging_or_source_open(
        Options {
            ht_backend: HtBackend::Hip,
            ..options()
        },
        "--ht-backend",
        "hip",
    )
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_conversion_requires_feature_before_staging_or_source_open() -> Result<()> {
    conversion_requires_feature_before_staging_or_source_open(
        Options {
            ht_backend: HtBackend::Cuda,
            ..options()
        },
        "--ht-backend",
        "cuda",
    )
}

#[cfg(not(feature = "cuda"))]
#[test]
fn i4r_cuda_conversion_requires_feature_before_staging_or_source_open() -> Result<()> {
    conversion_requires_feature_before_staging_or_source_open(
        Options {
            i4r_backend: I4rBackend::Cuda,
            ..options()
        },
        "--i4r-backend",
        "cuda",
    )
}

#[cfg(any(not(feature = "hip"), not(feature = "cuda")))]
fn conversion_requires_feature_before_staging_or_source_open(
    opts: Options,
    flag: &str,
    feature: &str,
) -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    // A stale source must not hide the actionable missing-feature error.
    fs::write(snapshot.root.join("model.safetensors"), b"not safetensors")?;
    let expected = format!("--features {feature}");
    for occupied in [false, true] {
        if occupied {
            fs::create_dir(&output)?;
            fs::write(output.join("keep.bin"), b"existing output")?;
        }
        let error = conversion::convert(&snapshot, &plan, &output, &opts, &Progress::disabled())
            .unwrap_err();
        assert!(error.to_string().contains(&expected), "{error:#}");
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
            .args(["convert", flag, feature, "--source"])
            .arg(temporary.path().join("missing-source"))
            .arg("--output")
            .arg(&output)
            .output()?;
        assert_eq!(result.status.code(), Some(2));
        assert!(String::from_utf8(result.stderr)?.contains(&expected));
        assert!(result.stdout.is_empty());
        if occupied {
            assert_eq!(fs::read(output.join("keep.bin"))?, b"existing output");
            assert_eq!(fs::read_dir(&output)?.count(), 1);
        } else {
            assert!(!output.exists());
        }
        assert_no_staging(temporary.path())?;
    }
    Ok(())
}

#[cfg(not(feature = "hip"))]
#[test]
fn hip_verification_requires_feature_even_without_eligible_tensors() -> Result<()> {
    verification_requires_feature_even_without_eligible_tensors(DecodeBackend::Hip, "hip")
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_verification_requires_feature_even_without_eligible_tensors() -> Result<()> {
    verification_requires_feature_even_without_eligible_tensors(DecodeBackend::Cuda, "cuda")
}

#[cfg(any(not(feature = "hip"), not(feature = "cuda")))]
fn verification_requires_feature_even_without_eligible_tensors(
    backend: DecodeBackend,
    feature: &str,
) -> Result<()> {
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
    let before = publication_bytes(&output)?;
    // Backend preflight must precede stale-source checking as well as manifest I/O.
    fs::write(assets.root.join("model.safetensors"), b"not safetensors")?;
    let expected = format!("--features {feature}");
    let missing = temporary.path().join("missing-output");
    for target in [&output, &missing] {
        let error =
            verify::verify(&source, &plan, target, backend, &Progress::disabled()).unwrap_err();
        assert!(error.to_string().contains(&expected), "{error:#}");
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
            .args(["verify", "--decode-backend", feature, "--source"])
            .arg(temporary.path().join("missing-source"))
            .arg("--output")
            .arg(target)
            .output()?;
        assert_eq!(result.status.code(), Some(2));
        assert!(String::from_utf8(result.stderr)?.contains(&expected));
        assert!(result.stdout.is_empty());
    }
    assert!(!missing.exists());
    assert_eq!(publication_bytes(&output)?, before);
    assert_no_staging(temporary.path())?;
    Ok(())
}

fn publication_bytes(
    root: &std::path::Path,
) -> Result<std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>> {
    fn collect(
        root: &std::path::Path,
        directory: &std::path::Path,
        files: &mut std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    ) -> Result<()> {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_dir() {
                collect(root, &path, files)?;
            } else {
                files.insert(path.strip_prefix(root)?.to_path_buf(), fs::read(path)?);
            }
        }
        Ok(())
    }
    let mut files = std::collections::BTreeMap::new();
    collect(root, root, &mut files)?;
    Ok(files)
}

#[test]
fn describing_hip_does_not_initialize_a_device_or_bypass_architecture() -> Result<()> {
    describing_backend_does_not_initialize_a_device_or_bypass_architecture(
        "--ht-backend",
        "hip",
        "HIP HT",
    )
}

#[test]
fn describing_cuda_does_not_initialize_a_device_or_bypass_architecture() -> Result<()> {
    describing_backend_does_not_initialize_a_device_or_bypass_architecture(
        "--ht-backend",
        "cuda",
        "CUDA HT",
    )
}

#[test]
fn describing_i4r_cuda_does_not_initialize_a_device_or_bypass_architecture() -> Result<()> {
    describing_backend_does_not_initialize_a_device_or_bypass_architecture(
        "--i4r-backend",
        "cuda",
        "CUDA I4R",
    )
}

fn describing_backend_does_not_initialize_a_device_or_bypass_architecture(
    flag: &str,
    backend: &str,
    label: &str,
) -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, _) = support::fixture(temporary.path())?;
    for command in ["inspect", "plan", "convert"] {
        let mut cli = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"));
        cli.args([command, flag, backend, "--source"])
            .arg(&snapshot.root);
        if command == "convert" {
            cli.arg("--dry-run")
                .arg("--output")
                .arg(temporary.path().join("converted"));
        }
        let result = cli.output()?;
        assert_eq!(result.status.code(), Some(2)); // Structural config remains rejected.
        let error = String::from_utf8(result.stderr)?;
        assert!(!error.contains(&format!("--features {backend}")));
        assert!(!error.contains(&format!("{label} preflight")));
        assert!(result.stdout.is_empty());
    }
    assert_no_staging(temporary.path())?;
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires an available gfx1151 HIP device; encodes and publishes real tiny files"]
fn hip_tiny_conversion_matches_cpu_all_containers_and_verifies() -> Result<()> {
    tiny_conversion_matches_cpu_all_containers_and_verifies(HtBackend::Hip, DecodeBackend::Hip)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an available CUDA device; encodes and publishes real tiny files"]
fn cuda_tiny_conversion_matches_cpu_all_containers_and_verifies() -> Result<()> {
    tiny_conversion_matches_cpu_all_containers_and_verifies(HtBackend::Cuda, DecodeBackend::Cuda)
}

#[cfg(any(feature = "hip", feature = "cuda"))]
fn tiny_conversion_matches_cpu_all_containers_and_verifies(
    gpu_backend: HtBackend,
    decode_backend: DecodeBackend,
) -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::gpu_fixture(temporary.path())?;
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
    let mut baseline_report = None;
    for backend in [HtBackend::Cpu, gpu_backend] {
        let output = temporary.path().join(backend.label().to_ascii_lowercase());
        let opts = Options {
            profile: OutputProfile::Ht43,
            ht_backend: backend,
            ..options()
        };
        let manifest =
            conversion::convert(&snapshot, &plan, &output, &opts, &Progress::disabled())?;
        assert_eq!(manifest.options.ht_backend, backend);
        let report = verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled(),
        )?;
        let gpu_report = verify::verify(
            &snapshot,
            &plan,
            &output,
            decode_backend,
            &Progress::disabled(),
        )?;
        let report = serde_json::to_value(report)?;
        assert_eq!(report, serde_json::to_value(gpu_report)?);
        if let Some(cpu_report) = &baseline_report {
            assert_eq!(&report, cpu_report);
        } else {
            baseline_report = Some(report);
        }
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
    full_verification_reports_match_cpu_bitwise(DecodeBackend::Hip)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an available CUDA device; compares complete verification reports"]
fn cuda_full_verification_reports_match_cpu_bitwise() -> Result<()> {
    full_verification_reports_match_cpu_bitwise(DecodeBackend::Cuda)
}

#[cfg(any(feature = "hip", feature = "cuda"))]
fn full_verification_reports_match_cpu_bitwise(backend: DecodeBackend) -> Result<()> {
    for profile in [OutputProfile::V2, OutputProfile::Ht43] {
        let temporary = tempfile::tempdir()?;
        let (snapshot, plan) = if profile == OutputProfile::V2 {
            support::fixture(temporary.path())?
        } else {
            support::gpu_fixture(temporary.path())?
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
        let gpu = verify::verify(&snapshot, &plan, &output, backend, &Progress::disabled())?;
        assert_eq!(cpu.manifest_schema, 2);
        assert!(cpu.provenance_attested);
        assert_eq!(cpu.errors.len(), gpu.errors.len());
        for (name, reference) in &cpu.errors {
            let actual = &gpu.errors[name];
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
        assert_eq!(serde_json::to_value(&cpu)?, serde_json::to_value(gpu)?);
        // Encoding provenance never chooses the runtime decoder, including I4R.
        let mut json = serde_json::to_value(&manifest)?;
        for provenance in ["hip", "cuda"] {
            json["options"]["ht_backend"] = serde_json::json!(provenance);
            json["options"]["i4r_backend"] = serde_json::json!("cuda");
            fs::write(
                output.join("conversion-manifest.json"),
                serde_json::to_vec_pretty(&json)?,
            )?;
            for runtime in [DecodeBackend::Cpu, backend] {
                let report =
                    verify::verify(&snapshot, &plan, &output, runtime, &Progress::disabled())?;
                assert_eq!(serde_json::to_value(report)?, serde_json::to_value(&cpu)?);
            }
        }
    }
    Ok(())
}

#[test]
fn backend_selection_preserves_semantic_option_limits() -> Result<()> {
    for (backend, i4r_backend) in [
        (HtBackend::Cpu, I4rBackend::Cpu),
        (HtBackend::Hip, I4rBackend::Cpu),
        (HtBackend::Cuda, I4rBackend::Cpu),
        (HtBackend::Cpu, I4rBackend::Cuda),
        (HtBackend::Hip, I4rBackend::Cuda),
        (HtBackend::Cuda, I4rBackend::Cuda),
    ] {
        for beam in [1, 256] {
            Options {
                ht_backend: backend,
                i4r_backend,
                ht_beam: beam,
                ..options()
            }
            .validate()?;
        }
        for beam in [0, 257, usize::MAX] {
            assert!(Options {
                ht_backend: backend,
                i4r_backend,
                ht_beam: beam,
                ..options()
            }
            .validate()
            .is_err());
        }
        for threads in [0, 65, usize::MAX] {
            assert!(Options {
                ht_backend: backend,
                i4r_backend,
                threads,
                ..options()
            }
            .validate()
            .is_err());
        }
    }
    Ok(())
}

#[test]
fn backend_labels_and_defaults_are_available_independently_of_features() {
    assert_eq!(HtBackend::default(), HtBackend::Cpu);
    assert_eq!(I4rBackend::default(), I4rBackend::Cpu);
    assert_eq!(DecodeBackend::default(), DecodeBackend::Cpu);
    for (backend, label, gpu) in [
        (HtBackend::Cpu, "CPU", false),
        (HtBackend::Hip, "HIP", true),
        (HtBackend::Cuda, "CUDA", true),
    ] {
        assert_eq!(backend.label(), label);
        assert_eq!(backend.is_gpu(), gpu);
    }
    for (backend, label, gpu) in [
        (I4rBackend::Cpu, "CPU", false),
        (I4rBackend::Cuda, "CUDA", true),
    ] {
        assert_eq!(backend.label(), label);
        assert_eq!(backend.is_gpu(), gpu);
    }
    for (backend, label, gpu) in [
        (DecodeBackend::Cpu, "CPU", false),
        (DecodeBackend::Hip, "HIP", true),
        (DecodeBackend::Cuda, "CUDA", true),
    ] {
        assert_eq!(backend.label(), label);
        assert_eq!(backend.is_gpu(), gpu);
    }
}

#[cfg(feature = "cuda")]
fn i4r_batched_fixture(
    parent: &std::path::Path,
) -> Result<(Snapshot, bf16_to_hgn::ConversionPlan)> {
    use bf16_to_hgn::hf::Dtype;

    let (snapshot, mut plan) = support::fixture(parent)?;
    // Each gate/up portion is 35 bands (32 + 3); down is 33 (32 + 1).
    // Both have two experts. Keep all other codecs, n-gram/vision containers,
    // MTP tensors and copied assets from the existing real safetensors fixture.
    let gate_rows = 2 * 35 * 128;
    let down_rows = 33 * 128;
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    let mut weight_map = std::collections::BTreeMap::new();
    for (name, info) in &snapshot.tensors {
        let expert_rows = match name.as_str() {
            "fixture.experts.gate_up" => Some(gate_rows),
            "fixture.experts.down" => Some(down_rows),
            _ => None,
        };
        let (shape, bytes) = if let Some(rows) = expert_rows {
            let shape = vec![2, rows as u64, 128];
            let bytes = (0..2 * rows * 128)
                .flat_map(|i| {
                    let row = i / 128;
                    let expert = row / rows;
                    let value = ((i * 37 + row * 19 + expert * 53) % 509) as f32 - 254.0;
                    half::bf16::from_f32(value / 128.0).to_le_bytes()
                })
                .collect::<Vec<_>>();
            (shape, bytes)
        } else {
            (
                info.shape.clone(),
                snapshot.read_bytes(name, 0, usize::try_from(info.size)?)?,
            )
        };
        let start = payload.len();
        payload.extend_from_slice(&bytes);
        let dtype = match info.dtype {
            Dtype::Bf16 => "BF16",
            Dtype::F32 => "F32",
            Dtype::I64 => "I64",
        };
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]
            }),
        );
        weight_map.insert(name.clone(), "model.safetensors");
    }
    let mut header = serde_json::to_vec(&header)?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut source = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(snapshot.root.join("model.safetensors"))?;
    source.write_all(&(header.len() as u64).to_le_bytes())?;
    source.write_all(&header)?;
    source.write_all(&payload)?;
    drop(source);
    fs::write(
        snapshot.root.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": { "total_size": payload.len() }, "weight_map": weight_map
        }))?,
    )?;
    for tensor in &mut plan.files[0].tensors {
        match tensor.name.as_str() {
            "fixture.experts.gate_up.weight" => {
                tensor.name = "fixture.experts.gate_up_proj.weight".into();
                tensor.shape[1] = gate_rows as u64;
            }
            "fixture.experts.down.weight" => {
                tensor.name = "fixture.experts.down_proj.weight".into();
                tensor.shape[1] = down_rows as u64;
            }
            "fixture.experts.gate_up.svh" => tensor.shape[0] = gate_rows as u64,
            "fixture.experts.down.svh" => tensor.shape[0] = down_rows as u64,
            _ => {}
        }
    }
    plan.files[0].tensors.sort_by(|a, b| a.name.cmp(&b.name));
    // Existing I4R wire semantics intentionally share su across experts and
    // restart the expert-local sv plane. Distinct expert su is tested through
    // separate public encode_rows calls in i4r_cuda.rs, not a changed format.
    Ok((Snapshot::open(&snapshot.root)?, plan))
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an available CUDA device; publishes multi-batch I4R experts with CPU HT"]
fn cuda_i4r_conversion_matches_cpu_all_containers_and_verifies() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = i4r_batched_fixture(temporary.path())?;
    assert_eq!(plan.files.len(), 3);
    assert!(plan
        .files
        .iter()
        .flat_map(|file| &file.tensors)
        .any(|tensor| tensor.encoding == Encoding::Ht));
    let experts: Vec<_> = plan
        .files
        .iter()
        .flat_map(|file| &file.tensors)
        .filter(|tensor| tensor.encoding == Encoding::I4r)
        .collect();
    assert_eq!(experts.len(), 2);
    for tensor in experts {
        assert_eq!(tensor.shape[0], 2);
        assert!(tensor.shape[1] > 32 * 128);
        if tensor.name.ends_with(".gate_up_proj.weight") {
            assert_eq!(tensor.shape[1] / 2, 35 * 128);
        }
    }
    let mut baseline: Option<(
        bf16_to_hgn::manifest::Manifest,
        serde_json::Value,
        std::collections::BTreeMap<std::path::PathBuf, Vec<u8>>,
    )> = None;
    for i4r_backend in [I4rBackend::Cpu, I4rBackend::Cuda] {
        let output = temporary
            .path()
            .join(i4r_backend.label().to_ascii_lowercase());
        let opts = Options {
            ht_backend: HtBackend::Cpu,
            i4r_backend,
            ..options()
        };
        let manifest =
            conversion::convert(&snapshot, &plan, &output, &opts, &Progress::disabled())?;
        assert_eq!(manifest.options.ht_backend, HtBackend::Cpu);
        assert_eq!(manifest.options.i4r_backend, i4r_backend);
        assert_eq!(verify::manifest_options(&output)?, opts);
        let serialized: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("conversion-manifest.json"))?)?;
        assert_eq!(serialized["options"]["ht_backend"], "cpu");
        assert_eq!(
            serialized["options"]["i4r_backend"],
            i4r_backend.label().to_ascii_lowercase()
        );
        // Always verify against the original BF16 source with the CPU decoder,
        // including the publication carrying CUDA encoding provenance.
        let report = verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled(),
        )?;
        assert_eq!(report.source_tensors, snapshot.tensors.len());
        assert!(report.provenance_attested);
        let report = serde_json::to_value(report)?;
        let mut files = publication_bytes(&output)?;
        assert!(files
            .remove(std::path::Path::new("conversion-manifest.json"))
            .is_some());
        // Compare complete containers, including planar code/scale offsets and
        // every sign side plane, plus every copied asset, not just payload hashes.
        if let Some((cpu, cpu_report, cpu_files)) = &baseline {
            let cpu: &bf16_to_hgn::manifest::Manifest = cpu;
            assert_eq!(cpu.containers, manifest.containers);
            assert_eq!(cpu.sources, manifest.sources);
            assert_eq!(cpu.assets, manifest.assets);
            assert_eq!(cpu.plan, manifest.plan);
            assert_eq!(cpu_report, &report);
            assert_eq!(cpu_files.len(), files.len());
            for (filename, expected) in cpu_files {
                let actual = files.get(filename).expect("matching publication inventory");
                assert_eq!(actual.len(), expected.len(), "{}", filename.display());
                assert!(
                    actual == expected,
                    "{} differs from CPU bytes",
                    filename.display()
                );
            }
        } else {
            baseline = Some((manifest, report, files));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_i4r_device_unavailable_fails_before_source_open_or_staging() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let output = temporary.path().join("converted");
    for occupied in [false, true] {
        if occupied {
            fs::create_dir(&output)?;
            fs::write(output.join("keep.bin"), b"existing output")?;
        }
        // Hide devices only in this fresh CLI child, never in the test process
        // or another test's initialized CUDA context. This is a real runtime
        // availability failure, not a mocked encoder or substituted payload.
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_bf16-to-hgn"))
            .env("CUDA_VISIBLE_DEVICES", "")
            .args([
                "convert",
                "--ht-backend",
                "cpu",
                "--i4r-backend",
                "cuda",
                "--source",
            ])
            .arg(temporary.path().join("missing-source"))
            .arg("--output")
            .arg(&output)
            .output()?;
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        let error = String::from_utf8(result.stderr)?;
        assert!(error.contains("CUDA I4R preflight failed:"), "{error}");
        assert!(!error.contains("CUDA HT preflight"), "{error}");
        if occupied {
            assert_eq!(fs::read(output.join("keep.bin"))?, b"existing output");
            assert_eq!(fs::read_dir(&output)?.count(), 1);
        } else {
            assert!(!output.exists());
        }
        assert_no_staging(temporary.path())?;
    }
    Ok(())
}
