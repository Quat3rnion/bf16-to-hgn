//! Bounded canonical tensor access for supported physical source formats.
//!
//! Imported floating tensors retain their decoded F32 values; canonical tensor
//! SHA-256 digests describe `read_bytes`, not the encoded HGN/GGUF payload.

use crate::hf::{Snapshot, TensorInfo};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt::Debug,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

pub const ASSET_NAMES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "generation_config.json",
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    #[default]
    Hf,
    Hgn,
    Gguf,
}

#[derive(Debug, Clone, Default)]
pub struct SourceOptions {
    pub assets: Option<PathBuf>,
    pub ngram: Option<PathBuf>,
    pub vision: Option<PathBuf>,
    pub mtp: Option<PathBuf>,
}

pub trait TensorSource: Debug {
    /// Immutable physical source root (directory or model file).
    fn root(&self) -> &Path;
    fn tensors(&self) -> &BTreeMap<String, TensorInfo>;
    fn config(&self) -> &Value;
    fn kind(&self) -> SourceKind;
    /// Resolve a safe asset filename, including optional assets not present yet.
    fn asset_path(&self, name: &str) -> Result<PathBuf>;
    /// Read canonical little-endian dtype bytes. I64 values remain exact.
    fn read_bytes(&self, name: &str, byte_offset: u64, len: usize) -> Result<Vec<u8>>;
    fn read_f32(&self, name: &str, element_offset: u64, len: usize) -> Result<Vec<f32>>;
    /// SHA-256 of the same canonical byte stream returned by `read_bytes`.
    fn hash_tensor(&self, name: &str) -> Result<String>;
    fn check_unchanged(&self) -> Result<()>;
    /// Physical format metadata and dependency checksums, without absolute paths.
    fn provenance(&self) -> Result<Value>;
    /// All physical inputs/assets that output publication must not overlap.
    fn input_paths(&self) -> Vec<PathBuf>;
}

/// Open a source without decoding tensor payloads. An explicit format resolves
/// ambiguous directories; otherwise detect HF indexes or the file's magic.
pub fn open(
    path: &Path,
    kind: Option<SourceKind>,
    options: &SourceOptions,
) -> Result<Box<dyn TensorSource>> {
    let kind = match kind {
        Some(kind) => kind,
        None => detect(path)?,
    };
    match kind {
        SourceKind::Hf => {
            ensure!(
                options.assets.is_none()
                    && options.ngram.is_none()
                    && options.vision.is_none()
                    && options.mtp.is_none(),
                "HF snapshots contain their own assets and weights; sidecar options are for HGN/GGUF inputs"
            );
            Ok(Box::new(Snapshot::open(path)?))
        }
        SourceKind::Hgn => Ok(Box::new(crate::hgn_source::HgnSource::open(path, options)?)),
        SourceKind::Gguf => Ok(Box::new(crate::gguf_source::GgufSource::open(
            path, options,
        )?)),
    }
}

fn detect(path: &Path) -> Result<SourceKind> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading source metadata {}", path.display()))?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "source must not be a symlink"
    );
    if metadata.is_dir() {
        let hf = path.join("model.safetensors.index.json").is_file();
        let mut hgn = false;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("hgn"))
            {
                hgn = true;
            }
        }
        return match (hf, hgn) {
            (true, false) => Ok(SourceKind::Hf),
            (false, true) => Ok(SourceKind::Hgn),
            (true, true) => bail!("ambiguous source directory; select --source-format hf or hgn"),
            (false, false) => {
                bail!("source directory has no HF index or HGN model; select a model file")
            }
        };
    }
    ensure!(
        metadata.is_file(),
        "source must be a regular file or directory"
    );
    let mut file = File::open(path).context("opening source for format detection")?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .context("reading source format magic")?;
    match &magic {
        b"HGN1" => Ok(SourceKind::Hgn),
        b"GGUF" => Ok(SourceKind::Gguf),
        _ => bail!("unknown source format; expected an HF directory, HGN file, or GGUF file"),
    }
}
