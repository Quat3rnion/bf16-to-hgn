use crate::{
    codec::HtBackend,
    hf,
    manifest::{AssetRecord, ContainerRecord, Manifest, Options, OutputProfile, SourceRecord},
    source::SourceKind,
    ConversionPlan, SPEC_COMMIT, SPEC_VERSION,
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

// Historical pins belong to the reader, not the current writer/version. Schema 1
// was emitted by the HF-only CPU v2 converter at fe0804e.
const LEGACY_TOOL_VERSION: &str = "0.1.0";
const LEGACY_SPEC_VERSION: &str = "1.1.0";
const LEGACY_SPEC_COMMIT: &str = "79e75fd4ae7642478823525b0a3212951f584af5";
const LEGACY_POLICY: &str = "bf16-hgn-v2-rtn-nf4-signed-rms-trellis-v1";

#[derive(Debug, Serialize, Deserialize)]
struct LegacyOptions {
    seed: u64,
    threads: usize,
    ht_beam: usize,
}

/// The exact original schema. In particular, the index digest is required and
/// neither execution/profile selection nor physical source provenance existed.
#[derive(Debug, Serialize, Deserialize)]
struct LegacyManifest {
    schema: u32,
    tool: String,
    tool_version: String,
    encoder_sha256: String,
    spec_version: String,
    spec_commit: String,
    policy: String,
    options: LegacyOptions,
    plan: ConversionPlan,
    config_sha256: String,
    index_sha256: String,
    sources: Vec<SourceRecord>,
    assets: Vec<AssetRecord>,
    containers: Vec<ContainerRecord>,
}

/// Read-only verification view; it must never be serialized as an artifact.
#[derive(Debug)]
pub(crate) struct VerifiedManifest {
    pub(crate) schema: u32,
    pub(crate) options: Options,
    pub(crate) source_kind: SourceKind,
    /// None means schema 1 did not attest physical source provenance. Its tensor
    /// hashes cover raw HF safetensors ranges, not decoded imported sources.
    pub(crate) source_provenance: Option<serde_json::Value>,
    pub(crate) plan: ConversionPlan,
    pub(crate) config_sha256: String,
    pub(crate) index_sha256: Option<String>,
    pub(crate) sources: Vec<SourceRecord>,
    pub(crate) assets: Vec<AssetRecord>,
    pub(crate) containers: Vec<ContainerRecord>,
}

/// `None` is only for CLI option discovery before building the fresh plan.
/// Verification must pass `Some` and retain all source/payload checks afterward.
pub(crate) fn read_manifest(
    path: &Path,
    plan: Option<&ConversionPlan>,
) -> Result<VerifiedManifest> {
    // This reader rejects duplicate JSON keys, including nested keys.
    let json = hf::read_json(path)?;
    let manifest = match json.get("schema").and_then(serde_json::Value::as_u64) {
        Some(1) => {
            let legacy =
                LegacyManifest::deserialize(&json).context("invalid conversion manifest")?;
            ensure!(
                serde_json::to_value(&legacy)? == json,
                "unknown or noncanonical manifest fields"
            );
            ensure!(legacy.tool == "bf16-to-hgn", "unexpected manifest tool");
            ensure!(
                legacy.tool_version == LEGACY_TOOL_VERSION,
                "unsupported manifest tool version"
            );
            ensure!(
                legacy.spec_version == LEGACY_SPEC_VERSION
                    && legacy.spec_commit == LEGACY_SPEC_COMMIT,
                "manifest specification mismatch"
            );
            ensure!(
                legacy.policy == LEGACY_POLICY,
                "manifest quantization policy mismatch"
            );
            check_sha256(&legacy.encoder_sha256)?;
            VerifiedManifest {
                schema: legacy.schema,
                options: Options {
                    profile: OutputProfile::V2,
                    seed: legacy.options.seed,
                    threads: legacy.options.threads,
                    ht_beam: legacy.options.ht_beam,
                    ht_backend: HtBackend::Cpu,
                },
                source_kind: SourceKind::Hf,
                source_provenance: None,
                plan: legacy.plan,
                config_sha256: legacy.config_sha256,
                index_sha256: Some(legacy.index_sha256),
                sources: legacy.sources,
                assets: legacy.assets,
                containers: legacy.containers,
            }
        }
        Some(2) => {
            let current = Manifest::deserialize(&json).context("invalid conversion manifest")?;
            // The writer's shared serde types ignore unknown fields. Reject
            // those recursively, preserving only the existing schema-2
            // exception for manifests written before ht_backend selection.
            let mut canonical = serde_json::to_value(&current)?;
            if json["options"]
                .as_object()
                .is_some_and(|options| !options.contains_key("ht_backend"))
            {
                canonical["options"]
                    .as_object_mut()
                    .expect("serialized Options is an object")
                    .remove("ht_backend");
            }
            ensure!(canonical == json, "unknown or noncanonical manifest fields");
            ensure!(current.tool == "bf16-to-hgn", "unexpected manifest tool");
            ensure!(
                current.tool_version == env!("CARGO_PKG_VERSION"),
                "unsupported manifest tool version"
            );
            ensure!(
                current.spec_version == SPEC_VERSION && current.spec_commit == SPEC_COMMIT,
                "manifest specification mismatch"
            );
            ensure!(
                current.policy == current.options.profile.policy(),
                "manifest quantization policy mismatch"
            );
            // Encoder digests record provenance, not authorization; rebuilding
            // an unchanged format/policy must remain verifiable.
            check_sha256(&current.encoder_sha256)?;
            VerifiedManifest {
                schema: current.schema,
                options: current.options,
                source_kind: current.source_kind,
                source_provenance: Some(current.source_provenance),
                plan: current.plan,
                config_sha256: current.config_sha256,
                index_sha256: current.index_sha256,
                sources: current.sources,
                assets: current.assets,
                containers: current.containers,
            }
        }
        _ => bail!("unsupported manifest schema"),
    };
    check_sha256(&manifest.config_sha256)?;
    if let Some(hash) = &manifest.index_sha256 {
        check_sha256(hash)?;
    }
    ensure!(
        manifest.index_sha256.is_some() == (manifest.source_kind == SourceKind::Hf),
        "source index provenance does not match source format"
    );
    manifest.options.validate()?;
    if let Some(plan) = plan {
        ensure!(
            &manifest.plan == plan,
            "manifest plan differs from the fresh source plan"
        );
    }
    Ok(manifest)
}

fn check_sha256(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid SHA-256 hex digest"
    );
    Ok(())
}
