mod support;

use anyhow::Result;
use bf16_to_hgn::{
    conversion,
    manifest::{Options, OutputProfile},
    progress::Progress,
    verify,
    verify_gpu::DecodeBackend,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn options() -> Options {
    Options {
        seed: 7,
        threads: 1,
        ht_beam: 8,
        ..Options::default()
    }
}

// Keep actual converted payloads, tensor inventories and hashes. Only remove
// fields that did not exist in the original fe0804e schema-1 writer.
fn legacy_manifest(mut current: Value) -> Value {
    current["schema"] = json!(1);
    let object = current.as_object_mut().expect("manifest object");
    object.remove("source_kind");
    object.remove("source_provenance");
    let options = current["options"].as_object_mut().expect("options object");
    options.remove("profile");
    options.remove("ht_backend");
    current
}

fn save(output: &Path, manifest: &Value) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(manifest)?;
    fs::write(output.join("conversion-manifest.json"), &bytes)?;
    Ok(bytes)
}

#[test]
fn schema1_verifies_actual_v2_payloads_and_raw_hf_tensor_hashes_without_rewriting() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::fixture(temporary.path())?;
    let output = temporary.path().join("converted");
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    let current = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    for source in &manifest.sources {
        let raw =
            snapshot.read_bytes(&source.tensor.name, 0, usize::try_from(source.tensor.size)?)?;
        assert_eq!(source.sha256, format!("{:x}", Sha256::digest(&raw)));
    }
    let bytes = save(&output, &legacy_manifest(serde_json::to_value(manifest)?))?;
    assert_eq!(verify::manifest_options(&output)?, options());
    let legacy = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(current.manifest_schema, 2);
    assert!(current.provenance_attested);
    assert_eq!(legacy.manifest_schema, 1);
    assert!(!legacy.provenance_attested);
    assert_eq!(legacy.source_tensors, current.source_tensors);
    assert_eq!(legacy.output_tensors, current.output_tensors);
    assert_eq!(legacy.payload_bytes, current.payload_bytes);
    assert_eq!(
        serde_json::to_value(legacy.errors)?,
        serde_json::to_value(current.errors)?
    );
    assert_eq!(fs::read(output.join("conversion-manifest.json"))?, bytes);
    Ok(())
}

#[test]
fn schema1_matrix_payloads_keep_exact_fresh_plan_and_source_checks() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::matrix_fixture(temporary.path(), OutputProfile::V2)?;
    let output = temporary.path().join("converted");
    let manifest =
        conversion::convert(&snapshot, &plan, &output, &options(), &Progress::disabled())?;
    let legacy = legacy_manifest(serde_json::to_value(manifest)?);
    save(&output, &legacy)?;
    let report = verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    assert_eq!(report.source_tensors, snapshot.tensors.len());
    assert!(report.errors.values().all(|error| error.values > 0));

    let mut changed = legacy.clone();
    changed["plan"]["architecture"] = json!("different-architecture");
    save(&output, &changed)?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());

    let mut changed = legacy.clone();
    changed["sources"][0]["sha256"] = json!("00".repeat(32));
    save(&output, &changed)?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());

    let mut changed = legacy.clone();
    let offset = changed["sources"][0]["tensor"]["offset"].as_u64().unwrap();
    changed["sources"][0]["tensor"]["offset"] = json!(offset + 1);
    save(&output, &changed)?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());

    let mut changed = legacy;
    changed["sources"].as_array_mut().unwrap().pop();
    save(&output, &changed)?;
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
fn schema1_rejects_mixed_schemas_invalid_pins_missing_index_and_unknown_fields() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::matrix_fixture(temporary.path(), OutputProfile::V2)?;
    let output = temporary.path().join("converted");
    let current = serde_json::to_value(conversion::convert(
        &snapshot,
        &plan,
        &output,
        &options(),
        &Progress::disabled(),
    )?)?;
    let legacy = legacy_manifest(current.clone());
    let mut rejected = Vec::new();
    for field in ["source_kind", "source_provenance"] {
        let mut changed = legacy.clone();
        changed[field] = current[field].clone();
        rejected.push(changed);
    }
    for field in ["profile", "ht_backend"] {
        let mut changed = legacy.clone();
        changed["options"][field] = current["options"][field].clone();
        rejected.push(changed);
    }
    let mut changed = legacy.clone();
    changed.as_object_mut().unwrap().remove("index_sha256");
    rejected.push(changed);
    let mut changed = legacy.clone();
    changed["index_sha256"] = Value::Null;
    rejected.push(changed);
    for (field, value) in [
        ("tool", json!("other-tool")),
        ("tool_version", json!("0.2.0")),
        ("spec_version", json!("1.2.0")),
        ("spec_commit", json!("0".repeat(40))),
        ("policy", json!(OutputProfile::Ht43.policy())),
        ("encoder_sha256", json!("g".repeat(64))),
        ("config_sha256", json!("0".repeat(63))),
        ("index_sha256", json!("not-a-hash")),
        ("unexpected", json!(true)),
    ] {
        let mut changed = legacy.clone();
        changed[field] = value;
        rejected.push(changed);
    }
    for (field, value) in [("threads", json!(0)), ("ht_beam", json!(257))] {
        let mut changed = legacy.clone();
        changed["options"][field] = value;
        rejected.push(changed);
    }
    let mut changed = legacy.clone();
    changed["sources"][0]["tensor"]["unexpected"] = json!(true);
    rejected.push(changed);
    for schema in [json!(0), json!(3), json!("1"), json!(1.0), Value::Null] {
        let mut changed = legacy.clone();
        changed["schema"] = schema;
        rejected.push(changed);
    }
    for changed in rejected {
        save(&output, &changed)?;
        assert!(verify::manifest_options(&output).is_err());
        assert!(verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled()
        )
        .is_err());
    }
    let valid = serde_json::to_string(&legacy)?;
    let duplicated = format!("{{\"schema\":1,{}", &valid[1..]);
    fs::write(output.join("conversion-manifest.json"), duplicated)?;
    assert!(verify::manifest_options(&output).is_err());
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
fn schema2_requires_real_provenance_and_only_allows_the_existing_backend_omission() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (snapshot, plan) = support::matrix_fixture(temporary.path(), OutputProfile::V2)?;
    let output = temporary.path().join("converted");
    let current = serde_json::to_value(conversion::convert(
        &snapshot,
        &plan,
        &output,
        &options(),
        &Progress::disabled(),
    )?)?;
    let mut old_schema2 = current.clone();
    old_schema2["options"]
        .as_object_mut()
        .unwrap()
        .remove("ht_backend");
    save(&output, &old_schema2)?;
    verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    for field in ["source_kind", "source_provenance"] {
        let mut changed = current.clone();
        changed.as_object_mut().unwrap().remove(field);
        save(&output, &changed)?;
        assert!(verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled()
        )
        .is_err());
    }
    let mut changed = current.clone();
    changed["options"]
        .as_object_mut()
        .unwrap()
        .remove("profile");
    save(&output, &changed)?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    for provenance in [Value::Null, json!({}), json!({"unattested": true})] {
        let mut changed = current.clone();
        changed["source_provenance"] = provenance;
        save(&output, &changed)?;
        assert!(verify::verify(
            &snapshot,
            &plan,
            &output,
            DecodeBackend::Cpu,
            &Progress::disabled()
        )
        .is_err());
    }
    let mut changed = current.clone();
    let shard = changed["source_provenance"]["dependencies"]
        .as_array_mut()
        .expect("physical dependencies")
        .iter_mut()
        .find(|dependency| dependency.get("header_sha256").is_some())
        .expect("safetensors shard provenance");
    shard["header_sha256"] = json!("00".repeat(32));
    save(&output, &changed)?;
    assert!(verify::verify(
        &snapshot,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled()
    )
    .is_err());
    let mut changed = current;
    changed["sources"][0]["sha256"] = json!("00".repeat(32));
    save(&output, &changed)?;
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
fn schema1_raw_hash_domain_cannot_be_applied_to_an_imported_hgn_source() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let (assets, _) = support::fixture(temporary.path())?;
    let (hgn, plan) = support::imported_norm_fixture(temporary.path(), &[1.00390625, -0.0])?;
    let source = support::FixtureAssets {
        tensors: &hgn,
        assets: &assets,
    };
    let output = temporary.path().join("converted");
    let manifest = conversion::convert(&source, &plan, &output, &options(), &Progress::disabled())?;
    verify::verify(
        &source,
        &plan,
        &output,
        DecodeBackend::Cpu,
        &Progress::disabled(),
    )?;
    let mut legacy = legacy_manifest(serde_json::to_value(manifest)?);
    legacy["index_sha256"] = json!("00".repeat(32));
    save(&output, &legacy)?;
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
