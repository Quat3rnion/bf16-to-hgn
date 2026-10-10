//! Bounded GGUF v2/v3 Flash-Next input. Tensor values are decoded on demand;
//! neither an intermediate HF checkpoint nor a decoded model is materialized.
//!
//! Canonical names and transforms follow hgn-spec 1.1.0, commit
//! 79e75fd4ae7642478823525b0a3212951f584af5, GGUF-IMPORT.md. Physical GGML
//! values stay F32, including IQ group scales; only the output codec rounds.

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[cfg(unix)]
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};

use crate::gguf_decode::{decode, GgmlType};
use crate::hf::{Dtype, TensorInfo};
use crate::hgn_source::HgnSidecar;
use crate::source::{SourceKind, SourceOptions, TensorSource, ASSET_NAMES as ASSETS};

const MAX_HEADER: u64 = 128 * 1024 * 1024;
const WINDOW: usize = 1024 * 1024;
const MAX_TENSORS: u64 = 20_000;
const MAX_METADATA: u64 = 100_000;
const NGRAM_PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
const PLE: &[(&str, &str, usize)] = &[
    ("qwen4exp.ple.layer_multipliers", "layer_multipliers", 3),
    ("qwen4exp.ple.head_offsets", "ngram_heads_offsets", 16),
    (
        "qwen4exp.ple.head_vocab_sizes",
        "ngram_heads_vocab_sizes",
        16,
    ),
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Signature {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

impl Signature {
    fn metadata(metadata: &Metadata) -> Result<Self> {
        Ok(Self {
            len: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            identity: (
                metadata.dev(),
                metadata.ino(),
                metadata.ctime(),
                metadata.ctime_nsec(),
            ),
        })
    }

    fn path(path: &Path, directory: bool) -> Result<Self> {
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("reading input {}", path.display()))?;
        ensure!(
            if directory {
                metadata.is_dir()
            } else {
                metadata.is_file()
            },
            "input must be a regular {} (not a symlink): {}",
            if directory { "directory" } else { "file" },
            path.display()
        );
        Self::metadata(&metadata)
    }
}

#[derive(Debug)]
struct InputFile {
    path: PathBuf,
    file: File,
    signature: Signature,
}

impl InputFile {
    fn open(path: &Path) -> Result<Self> {
        let signature = Signature::path(path, false)?;
        let path = path.canonicalize()?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = options.open(&path)?;
        let input = Self {
            path,
            file,
            signature,
        };
        input.check()?;
        Ok(input)
    }

    fn check(&self) -> Result<()> {
        ensure!(
            Signature::path(&self.path, false)? == self.signature
                && Signature::metadata(&self.file.metadata()?)? == self.signature,
            "GGUF input changed: {}",
            self.path.display()
        );
        Ok(())
    }

    fn read_into(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        self.check()?;
        self.read_range(offset, bytes)?;
        self.check()
    }

    // Callers bracket one bounded canonical request with signature checks.
    // Positional reads, unlike mmap, also fail safely on concurrent truncation.
    fn read_range(&self, offset: u64, bytes: &mut [u8]) -> Result<()> {
        ensure!(
            offset
                .checked_add(bytes.len() as u64)
                .is_some_and(|end| end <= self.signature.len),
            "GGUF read exceeds input bounds"
        );
        #[cfg(unix)]
        self.file.read_exact_at(bytes, offset)?;
        #[cfg(not(unix))]
        {
            let mut file = self.file.try_clone()?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(bytes)?;
        }
        Ok(())
    }

    fn hash_range(&self, offset: u64, size: u64) -> Result<String> {
        ensure!(
            offset
                .checked_add(size)
                .is_some_and(|end| end <= self.signature.len),
            "GGUF hash range exceeds input"
        );
        let mut hasher = Sha256::new();
        let mut bytes = vec![0; usize::try_from(size.min(WINDOW as u64))?];
        let mut done = 0;
        while done < size {
            let take = usize::try_from((size - done).min(bytes.len() as u64))?;
            self.read_into(offset + done, &mut bytes[..take])?;
            hasher.update(&bytes[..take]);
            done += take as u64;
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

struct Header {
    input: BufReader<File>,
    position: u64,
    limit: u64,
    hash: Sha256,
}

impl Header {
    fn new(file: &InputFile) -> Result<Self> {
        let mut input = file.file.try_clone()?;
        input.seek(SeekFrom::Start(0))?;
        Ok(Self {
            input: BufReader::new(input),
            position: 0,
            limit: file.signature.len.min(MAX_HEADER),
            hash: Sha256::new(),
        })
    }

    fn bytes(&mut self, bytes: &mut [u8]) -> Result<()> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .context("GGUF header offset overflow")?;
        ensure!(
            end <= self.limit,
            "truncated or oversized GGUF header (limit {MAX_HEADER} bytes)"
        );
        self.input
            .read_exact(bytes)
            .context("reading GGUF header")?;
        self.hash.update(bytes);
        self.position = end;
        Ok(())
    }

    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.bytes(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.fixed()?))
    }

    fn skip(&mut self, count: u64) -> Result<()> {
        ensure!(
            count <= self.limit - self.position,
            "truncated or oversized GGUF metadata value"
        );
        let mut remaining = count;
        let mut bytes = [0; 8192];
        while remaining != 0 {
            let take = usize::try_from(remaining.min(bytes.len() as u64))?;
            self.bytes(&mut bytes[..take])?;
            remaining -= take as u64;
        }
        Ok(())
    }

    fn string(&mut self, maximum: u64) -> Result<String> {
        let size = self.u64()?;
        ensure!(
            size <= maximum && size <= self.limit - self.position,
            "GGUF string exceeds length bound"
        );
        let mut bytes = vec![0; usize::try_from(size)?];
        self.bytes(&mut bytes)?;
        Ok(String::from_utf8(bytes).context("non-UTF-8 GGUF string")?)
    }

    fn value(&mut self, ty: u32, keep: bool) -> Result<Option<Value>> {
        let value = match ty {
            0 => json!(u8::from_le_bytes(self.fixed()?)),
            1 => json!(i8::from_le_bytes(self.fixed()?)),
            2 => json!(u16::from_le_bytes(self.fixed()?)),
            3 => json!(i16::from_le_bytes(self.fixed()?)),
            4 => json!(self.u32()?),
            5 => json!(i32::from_le_bytes(self.fixed()?)),
            6 => {
                let value = f32::from_le_bytes(self.fixed()?);
                if keep {
                    ensure!(value.is_finite(), "non-finite GGUF metadata");
                }
                json!(value)
            }
            7 => {
                let value = self.fixed::<1>()?[0];
                ensure!(value <= 1, "invalid GGUF boolean");
                json!(value == 1)
            }
            8 => {
                if keep {
                    json!(self.string(4096)?)
                } else {
                    let size = self.u64()?;
                    self.skip(size)?;
                    return Ok(None);
                }
            }
            9 => {
                let element_type = self.u32()?;
                ensure!(
                    element_type <= 12 && element_type != 9,
                    "invalid/nested GGUF metadata array type"
                );
                let count = self.u64()?;
                ensure!(
                    count <= self.limit - self.position,
                    "GGUF metadata array exceeds header bound"
                );
                ensure!(
                    !keep || count <= 4096,
                    "retained GGUF metadata array exceeds bound"
                );
                let mut values = Vec::new();
                if keep {
                    values.try_reserve_exact(usize::try_from(count)?)?;
                }
                for _ in 0..count {
                    if let Some(value) = self.value(element_type, keep)? {
                        values.push(value);
                    }
                }
                if keep {
                    Value::Array(values)
                } else {
                    return Ok(None);
                }
            }
            10 => json!(self.u64()?),
            11 => json!(i64::from_le_bytes(self.fixed()?)),
            12 => {
                let value = f64::from_le_bytes(self.fixed()?);
                if keep {
                    ensure!(value.is_finite(), "non-finite GGUF metadata");
                }
                json!(value)
            }
            _ => bail!("unknown GGUF metadata type {ty}"),
        };
        Ok(keep.then_some(value))
    }
}

#[derive(Debug)]
struct Shard {
    input: InputFile,
    version: u32,
    header_bytes: u64,
    header_sha256: String,
    metadata: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
struct PhysicalTensor {
    name: String,
    shard: usize,
    shape: Vec<u64>,
    ty: GgmlType,
    offset: u64,
    bytes: u64,
    elements: u64,
}

fn product(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1u64, |size, &dim| {
        ensure!(dim != 0, "zero GGUF tensor dimension");
        size.checked_mul(dim).context("GGUF tensor shape overflow")
    })
}

fn keep_metadata(key: &str) -> bool {
    matches!(
        key,
        "general.architecture"
            | "general.alignment"
            | "general.file_type"
            | "qwen4exp.ssm.group_count"
            | "qwen4exp.ssm.time_step_rank"
            | "qwen4exp.ssm.state_size"
            | "qwen4exp.block_count"
            | "qwen4exp.expert_count"
            | "qwen4exp.embedding_length"
            | "split.no"
            | "split.count"
            | "split.tensors.count"
    ) || PLE.iter().any(|(name, _, _)| *name == key)
}

fn parse_shard(path: &Path, shard: usize) -> Result<(Shard, Vec<PhysicalTensor>)> {
    let input = InputFile::open(path)?;
    let mut header = Header::new(&input)?;
    ensure!(
        &header.fixed::<4>()? == b"GGUF",
        "not a little-endian GGUF file"
    );
    let version = header.u32()?;
    ensure!(
        matches!(version, 2 | 3),
        "unsupported GGUF version {version}; expected 2 or 3"
    );
    let count = header.u64()?;
    let metadata_count = header.u64()?;
    ensure!(
        count > 0 && count <= MAX_TENSORS,
        "GGUF tensor count exceeds bound or is zero"
    );
    ensure!(
        metadata_count <= MAX_METADATA,
        "GGUF metadata count exceeds bound"
    );
    let mut metadata = BTreeMap::new();
    let mut keys = BTreeSet::new();
    for _ in 0..metadata_count {
        let key = header.string(1024)?;
        ensure!(
            keys.insert(key.clone()),
            "duplicate GGUF metadata key {key}"
        );
        let ty = header.u32()?;
        if let Some(value) = header.value(ty, keep_metadata(&key))? {
            metadata.insert(key, value);
        }
    }
    let alignment = match metadata.get("general.alignment") {
        Some(value) => value
            .as_u64()
            .context("GGUF alignment must be an unsigned integer")?,
        None => 32,
    };
    ensure!(
        alignment.is_power_of_two() && alignment <= 1024 * 1024,
        "invalid GGUF alignment {alignment}"
    );
    let mut names = BTreeSet::new();
    let mut tensors = Vec::with_capacity(usize::try_from(count)?);
    for _ in 0..count {
        let name = header.string(95)?;
        ensure!(
            !name.is_empty() && !name.contains('\0') && names.insert(name.clone()),
            "empty/duplicate/invalid GGUF tensor name {name:?}"
        );
        let rank = header.u32()?;
        ensure!(
            (1..=4).contains(&rank),
            "GGUF tensor {name} must have 1 to 4 dimensions"
        );
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            shape.push(header.u64()?);
        }
        let elements = product(&shape).with_context(|| format!("GGUF tensor {name}"))?;
        let ty =
            GgmlType::from_code(header.u32()?).with_context(|| format!("GGUF tensor {name}"))?;
        let (block_elements, block_bytes) = ty.block();
        ensure!(
            shape[0] % block_elements as u64 == 0,
            "GGUF tensor {name} row width splits a {:?} block",
            ty
        );
        let bytes = (elements / block_elements as u64)
            .checked_mul(block_bytes as u64)
            .context("GGUF payload byte count overflow")?;
        let offset = header.u64()?;
        ensure!(offset % alignment == 0, "unaligned GGUF tensor {name}");
        shape.reverse();
        tensors.push(PhysicalTensor {
            name,
            shard,
            shape,
            ty,
            offset,
            bytes,
            elements,
        });
    }
    let data_start = header
        .position
        .checked_add(alignment - 1)
        .context("GGUF header alignment overflow")?
        & !(alignment - 1);
    ensure!(
        data_start <= input.signature.len,
        "GGUF data section exceeds file"
    );
    // Hash the padding as well as every byte of the metadata and directory.
    header.skip(data_start - header.position)?;
    let header_sha256 = format!("{:x}", header.hash.finalize());
    let mut ranges = Vec::with_capacity(tensors.len());
    for tensor in &mut tensors {
        tensor.offset = tensor
            .offset
            .checked_add(data_start)
            .context("GGUF tensor offset overflow")?;
        let end = tensor
            .offset
            .checked_add(tensor.bytes)
            .context("GGUF tensor extent overflow")?;
        ensure!(
            end <= input.signature.len,
            "truncated GGUF tensor {}",
            tensor.name
        );
        ranges.push((tensor.offset, end));
    }
    ranges.sort_unstable();
    ensure!(
        ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "overlapping GGUF tensor payloads"
    );
    input.check()?;
    Ok((
        Shard {
            input,
            version,
            header_bytes: data_start,
            header_sha256,
            metadata,
        },
        tensors,
    ))
}

fn shard_paths(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("GGUF filename must be UTF-8")?;
    if let Some(stem) = name.strip_suffix(".gguf") {
        if let Some((left, total)) = stem.rsplit_once("-of-") {
            if let Some((prefix, index)) = left.rsplit_once('-') {
                if total.len() == 5
                    && index.len() == 5
                    && total.bytes().all(|b| b.is_ascii_digit())
                    && index.bytes().all(|b| b.is_ascii_digit())
                {
                    let total: usize = total.parse()?;
                    let index: usize = index.parse()?;
                    ensure!(
                        (1..=1024).contains(&total) && (1..=total).contains(&index),
                        "invalid GGUF split shard numbering"
                    );
                    let parent = path.parent().unwrap_or(Path::new("."));
                    return Ok((1..=total)
                        .map(|part| parent.join(format!("{prefix}-{part:05}-of-{total:05}.gguf")))
                        .collect());
                }
            }
        }
    }
    Ok(vec![path.to_path_buf()])
}

#[derive(Debug, Clone, Copy, Default)]
enum Reorder {
    #[default]
    None,
    /// Permute 48 contiguous units after an unchanged prefix.
    Heads { prefix: u64, unit: u64 },
    /// Permute each row's 48 heads of 128 columns.
    Columns,
}

impl Reorder {
    fn span(self, position: u64, remaining: u64) -> (u64, u64) {
        match self {
            Self::None => (position, remaining),
            Self::Heads { prefix, unit } => {
                if position < prefix {
                    return (position, remaining.min(prefix - position));
                }
                let relative = position - prefix;
                let head = relative / unit;
                let within = relative % unit;
                (
                    prefix + (head / 3 + 16 * (head % 3)) * unit + within,
                    remaining.min(unit - within),
                )
            }
            Self::Columns => {
                let row = position / 6144;
                let col = position % 6144;
                let head = col / 128;
                (
                    row * 6144 + (head / 3 + 16 * (head % 3)) * 128 + col % 128,
                    remaining.min(128 - col % 128),
                )
            }
        }
    }
}

#[derive(Debug)]
enum Layout {
    Tensor {
        physical: usize,
        start: u64,
        reorder: Reorder,
        norm: bool,
        log_a: bool,
    },
    Pair {
        first: usize,
        second: usize,
        half: u64,
        experts: bool,
    },
    Integers(Vec<u8>),
    Sidecar(usize),
}

impl Layout {
    fn span(&self, position: u64, remaining: u64) -> (usize, u64, u64) {
        match *self {
            Self::Tensor {
                physical,
                start,
                reorder,
                ..
            } => {
                let (position, count) = reorder.span(position, remaining);
                (physical, start + position, count)
            }
            Self::Pair {
                first,
                second,
                half,
                experts,
            } => {
                if experts {
                    let part = position / half;
                    let within = position % half;
                    (
                        if part % 2 == 0 { first } else { second },
                        (part / 2) * half + within,
                        remaining.min(half - within),
                    )
                } else if position < half {
                    (first, position, remaining.min(half - position))
                } else {
                    (second, position - half, remaining)
                }
            }
            _ => unreachable!("nonphysical layout"),
        }
    }

    fn dependencies(&self) -> impl Iterator<Item = usize> {
        match *self {
            Self::Tensor { physical, .. } => [Some(physical), None],
            Self::Pair { first, second, .. } => [Some(first), Some(second)],
            _ => [None, None],
        }
        .into_iter()
        .flatten()
    }
}

#[derive(Debug)]
struct Assets {
    root: PathBuf,
    signature: Signature,
    files: BTreeMap<String, Signature>,
    config: Value,
}

impl Assets {
    fn open(root: &Path) -> Result<Self> {
        let signature = Signature::path(root, true)?;
        let root = root.canonicalize()?;
        let mut files = BTreeMap::new();
        for (index, &name) in ASSETS.iter().enumerate() {
            match fs::symlink_metadata(root.join(name)) {
                Ok(_) => {
                    files.insert(name.to_owned(), Signature::path(&root.join(name), false)?);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound && index >= 6 => {}
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("required GGUF --assets file {name}"))
                }
            }
        }
        let config_file = InputFile::open(&root.join("config.json"))?;
        ensure!(
            config_file.signature.len <= 16 * 1024 * 1024,
            "config.json exceeds size bound"
        );
        let mut bytes = vec![0; usize::try_from(config_file.signature.len)?];
        config_file.read_into(0, &mut bytes)?;
        let config: Value =
            serde_json::from_slice(&bytes).context("parsing supplied GGUF config.json")?;
        ensure!(config.is_object(), "GGUF config.json must be an object");
        let assets = Self {
            root,
            signature,
            files,
            config,
        };
        assets.check()?;
        Ok(assets)
    }
    fn check(&self) -> Result<()> {
        ensure!(
            Signature::path(&self.root, true)? == self.signature,
            "GGUF assets directory changed"
        );
        for (name, signature) in &self.files {
            ensure!(
                &Signature::path(&self.root.join(name), false)? == signature,
                "GGUF asset changed: {name}"
            );
        }
        Ok(())
    }
}

/// A complete import source, with explicit matching assets and HGN sidecars.
#[derive(Debug)]
pub struct GgufSource {
    root: PathBuf,
    assets: Option<Assets>,
    shards: Vec<Shard>,
    physical: Vec<PhysicalTensor>,
    tensors: BTreeMap<String, TensorInfo>,
    layouts: BTreeMap<String, Layout>,
    sidecars: Vec<HgnSidecar>,
    raw_hashes: RefCell<BTreeMap<usize, String>>,
}

/// Low-level, immutable GGUF tensor reader, without assets or auxiliary weights.
/// This is not a complete model source: normal CLI input uses `GgufSource::open`
/// and the fixed architecture planner rejects any incomplete tensor inventory.
#[derive(Debug)]
pub struct GgufSidecar(GgufSource);

impl GgufSidecar {
    pub fn open(path: &Path) -> Result<Self> {
        let root = path.canonicalize().context("opening GGUF source")?;
        let (shards, physical, metadata) = load_shards(path)?;
        let (tensors, layouts) = canonical_tensors(&physical, &metadata)?;
        let source = Self(GgufSource {
            root,
            assets: None,
            shards,
            physical,
            tensors,
            layouts,
            sidecars: Vec::new(),
            raw_hashes: RefCell::new(BTreeMap::new()),
        });
        source.check_unchanged()?;
        Ok(source)
    }
}

impl TensorSource for GgufSidecar {
    fn root(&self) -> &Path {
        self.0.root()
    }
    fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        self.0.tensors()
    }
    fn config(&self) -> &Value {
        self.0.config()
    }
    fn kind(&self) -> SourceKind {
        SourceKind::Gguf
    }
    fn asset_path(&self, _name: &str) -> Result<PathBuf> {
        bail!("GGUF tensor reader has no assets; use --assets")
    }
    fn read_bytes(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.0.read_bytes(name, offset, len)
    }
    fn read_f32(&self, name: &str, offset: u64, len: usize) -> Result<Vec<f32>> {
        self.0.read_f32(name, offset, len)
    }
    fn hash_tensor(&self, name: &str) -> Result<String> {
        self.0.hash_tensor(name)
    }
    fn check_unchanged(&self) -> Result<()> {
        self.0.check_unchanged()
    }
    fn provenance(&self) -> Result<Value> {
        self.0.provenance()
    }
    fn input_paths(&self) -> Vec<PathBuf> {
        self.0.input_paths()
    }
}

impl GgufSource {
    pub fn open(path: &Path, options: &SourceOptions) -> Result<Self> {
        let assets_path = options
            .assets
            .as_deref()
            .context("GGUF requires --assets with the matching config.json and tokenizer assets")?;
        let mtp = options
            .mtp
            .as_deref()
            .context("GGUF has no MTP head; supply the matching --mtp HGN sidecar")?;
        let vision = options
            .vision
            .as_deref()
            .context("GGUF has no vision weights; supply the matching --vision HGN sidecar")?;
        let assets = Assets::open(assets_path)?;
        let mut result = GgufSidecar::open(path)?.0;
        result.assets = Some(assets);
        result.add_sidecar(mtp, "mtp.")?;
        result.add_sidecar(vision, "model.visual.")?;
        let embedded_ngram = result
            .tensors
            .keys()
            .any(|name| name.starts_with(NGRAM_PREFIX));
        match (embedded_ngram, options.ngram.as_deref()) {
            (true, Some(_)) => bail!(
                "GGUF already contains its n-gram table; do not supply a second --ngram source"
            ),
            (false, Some(path)) => result.add_sidecar(path, NGRAM_PREFIX)?,
            (false, None) => bail!(
                "GGUF has no per_layer_token_embd.weight; supply the matching --ngram HGN sidecar"
            ),
            (true, None) => {}
        }
        result.check_unchanged()?;
        Ok(result)
    }

    fn add_sidecar(&mut self, path: &Path, prefix: &str) -> Result<()> {
        let sidecar = HgnSidecar::open(path)?;
        ensure!(
            !sidecar.tensors().is_empty(),
            "empty GGUF sidecar {}",
            path.display()
        );
        let index = self.sidecars.len();
        for (name, tensor) in sidecar.tensors() {
            ensure!(
                name.starts_with(prefix),
                "sidecar {} contains unexpected tensor {name}; expected {prefix}",
                path.display()
            );
            ensure!(
                !self.tensors.contains_key(name),
                "duplicate GGUF/sidecar tensor {name}"
            );
            self.tensors.insert(name.clone(), tensor.clone());
            self.layouts.insert(name.clone(), Layout::Sidecar(index));
        }
        self.sidecars.push(sidecar);
        Ok(())
    }

    fn tensor(&self, name: &str, offset: u64, length: u64, width: u64) -> Result<&TensorInfo> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown GGUF tensor {name}"))?;
        ensure!(
            offset
                .checked_add(length)
                .and_then(|end| end.checked_mul(width))
                .is_some_and(|end| end <= tensor.size),
            "GGUF tensor read out of bounds: {name}"
        );
        Ok(tensor)
    }

    fn decode_range(
        &self,
        index: usize,
        offset: u64,
        output: &mut [f32],
        bytes: &mut Vec<u8>,
    ) -> Result<()> {
        let tensor = &self.physical[index];
        ensure!(
            offset
                .checked_add(output.len() as u64)
                .is_some_and(|end| end <= tensor.elements),
            "GGUF physical read exceeds tensor {}",
            tensor.name
        );
        let (block_elements, block_bytes) = tensor.ty.block();
        let mut position = offset;
        let mut done = 0;
        let max_blocks = (WINDOW / block_bytes).max(1);
        let mut scratch = [0.0f32; 256];
        while done < output.len() {
            let block = position / block_elements as u64;
            let skip = (position % block_elements as u64) as usize;
            if skip != 0 || output.len() - done < block_elements {
                bytes.resize(block_bytes, 0);
                self.shards[tensor.shard]
                    .input
                    .read_range(tensor.offset + block * block_bytes as u64, bytes)?;
                decode(tensor.ty, bytes, &mut scratch[..block_elements])?;
                let take = (block_elements - skip).min(output.len() - done);
                output[done..done + take].copy_from_slice(&scratch[skip..skip + take]);
                done += take;
                position += take as u64;
            } else {
                let blocks = ((output.len() - done) / block_elements).min(max_blocks);
                bytes.resize(blocks * block_bytes, 0);
                self.shards[tensor.shard]
                    .input
                    .read_range(tensor.offset + block * block_bytes as u64, bytes)?;
                let take = blocks * block_elements;
                decode(tensor.ty, bytes, &mut output[done..done + take])?;
                done += take;
                position += take as u64;
            }
        }
        Ok(())
    }

    fn hash_physical(&self, index: usize) -> Result<()> {
        if self.raw_hashes.borrow().contains_key(&index) {
            return Ok(());
        }
        let tensor = &self.physical[index];
        let hash = self.shards[tensor.shard]
            .input
            .hash_range(tensor.offset, tensor.bytes)?;
        self.raw_hashes.borrow_mut().insert(index, hash);
        Ok(())
    }
}

impl TensorSource for GgufSource {
    fn root(&self) -> &Path {
        &self.root
    }
    fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    fn config(&self) -> &Value {
        self.assets
            .as_ref()
            .map(|assets| &assets.config)
            .unwrap_or(&Value::Null)
    }
    fn kind(&self) -> SourceKind {
        SourceKind::Gguf
    }
    fn asset_path(&self, name: &str) -> Result<PathBuf> {
        ensure!(
            ASSETS.contains(&name),
            "unknown or unsafe GGUF asset name {name}"
        );
        Ok(self
            .assets
            .as_ref()
            .context("GGUF source has no assets")?
            .root
            .join(name))
    }

    fn read_bytes(&self, name: &str, byte_offset: u64, len: usize) -> Result<Vec<u8>> {
        self.tensor(name, byte_offset, len as u64, 1)?;
        match &self.layouts[name] {
            Layout::Integers(bytes) => {
                for shard in &self.shards {
                    shard.input.check()?;
                }
                let start = usize::try_from(byte_offset)?;
                Ok(bytes[start..start + len].to_vec())
            }
            Layout::Sidecar(index) => self.sidecars[*index].read_bytes(name, byte_offset, len),
            _ => {
                if len == 0 {
                    return Ok(Vec::new());
                }
                let mut first = byte_offset / 4;
                let mut skip = (byte_offset % 4) as usize;
                let mut bytes = Vec::new();
                bytes.try_reserve_exact(len)?;
                while bytes.len() < len {
                    let elements = (len - bytes.len())
                        .checked_add(skip)
                        .and_then(|n| n.checked_add(3))
                        .context("GGUF canonical byte count overflow")?
                        / 4;
                    let elements = elements.min(WINDOW / 4);
                    let values = self.read_f32(name, first, elements)?;
                    for (index, value) in values.into_iter().enumerate() {
                        let word = value.to_le_bytes();
                        let begin = if index == 0 { skip } else { 0 };
                        let take = (4 - begin).min(len - bytes.len());
                        bytes.extend_from_slice(&word[begin..begin + take]);
                    }
                    first += elements as u64;
                    skip = 0;
                }
                Ok(bytes)
            }
        }
    }

    fn read_f32(&self, name: &str, element_offset: u64, len: usize) -> Result<Vec<f32>> {
        let tensor = self.tensor(name, element_offset, len as u64, 4)?;
        ensure!(
            tensor.dtype != Dtype::I64,
            "I64 GGUF metadata requires read_bytes"
        );
        let layout = &self.layouts[name];
        if let Layout::Sidecar(index) = layout {
            return self.sidecars[*index].read_f32(name, element_offset, len);
        }
        for dependency in layout.dependencies() {
            self.shards[self.physical[dependency].shard].input.check()?;
        }
        let mut values = Vec::new();
        values.try_reserve_exact(len)?;
        values.resize(len, 0.0);
        let mut bytes = Vec::new();
        let mut done = 0;
        let mut row = [0.0f32; 6144];
        while done < len {
            let position = element_offset + done as u64;
            // A complete output row needs every source head. Read/decode it once,
            // not through 48 tiny positional reads (especially for K-quants).
            if let Layout::Tensor {
                physical,
                reorder: Reorder::Columns,
                ..
            } = layout
            {
                if position % 6144 == 0 && len - done >= 6144 {
                    self.decode_range(*physical, position, &mut row, &mut bytes)?;
                    for head in 0..48 {
                        let source_head = head / 3 + 16 * (head % 3);
                        values[done + head * 128..done + (head + 1) * 128]
                            .copy_from_slice(&row[source_head * 128..(source_head + 1) * 128]);
                    }
                    done += 6144;
                    continue;
                }
            }
            let (physical, offset, take) = layout.span(position, (len - done) as u64);
            let take = usize::try_from(take)?;
            self.decode_range(physical, offset, &mut values[done..done + take], &mut bytes)
                .with_context(|| format!("decoding GGUF source tensor {name}"))?;
            done += take;
        }
        for dependency in layout.dependencies() {
            self.shards[self.physical[dependency].shard].input.check()?;
        }
        if let Layout::Tensor { norm, log_a, .. } = layout {
            if *norm {
                for value in &mut values {
                    *value -= 1.0;
                }
            }
            if *log_a {
                ensure!(
                    values.iter().all(|&value| value < 0.0),
                    "ssm_a must contain finite strictly negative values: {name}"
                );
                for value in &mut values {
                    *value = (-f64::from(*value)).ln() as f32;
                }
            }
        }
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "non-finite canonical GGUF tensor {name}"
        );
        Ok(values)
    }

    fn hash_tensor(&self, name: &str) -> Result<String> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown GGUF tensor {name}"))?;
        if let Layout::Sidecar(index) = &self.layouts[name] {
            return self.sidecars[*index].hash_tensor(name);
        }
        for index in self.layouts[name].dependencies() {
            self.hash_physical(index)?;
        }
        let mut hasher = Sha256::new();
        let mut offset = 0;
        while offset < tensor.size {
            let take = usize::try_from((tensor.size - offset).min(WINDOW as u64))?;
            hasher.update(self.read_bytes(name, offset, take)?);
            offset += take as u64;
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn check_unchanged(&self) -> Result<()> {
        if let Some(assets) = &self.assets {
            assets.check()?;
        }
        for shard in &self.shards {
            shard.input.check()?;
        }
        for sidecar in &self.sidecars {
            sidecar.check_unchanged()?;
        }
        Ok(())
    }

    fn provenance(&self) -> Result<Value> {
        self.check_unchanged()?;
        // Conversion and verification call this after consuming the source.
        // Physical hashes are a separate bounded pass, never an open/plan scan:
        // hashing transformed F32 would not bind discarded original bits.
        for index in 0..self.physical.len() {
            self.hash_physical(index)?;
        }
        let hashes = self.raw_hashes.borrow();
        let shards: Vec<Value> = self
            .shards
            .iter()
            .map(|shard| {
                json!({
                    "file": shard.input.path.file_name().and_then(|name| name.to_str()),
                    "version": shard.version, "file_bytes": shard.input.signature.len,
                    "header_bytes": shard.header_bytes, "header_sha256": shard.header_sha256,
                    "general_file_type": shard.metadata.get("general.file_type"),
                })
            })
            .collect();
        let tensors: Vec<Value> = self
            .physical
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                json!({
                    "name": tensor.name, "shard": tensor.shard, "type": tensor.ty as u32,
                    "shape": tensor.shape, "offset": tensor.offset, "bytes": tensor.bytes,
                    "raw_payload_sha256": hashes.get(&index),
                })
            })
            .collect();
        let sidecars = self
            .sidecars
            .iter()
            .map(TensorSource::provenance)
            .collect::<Result<Vec<_>>>()?;
        self.check_unchanged()?;
        Ok(
            json!({ "format": "gguf", "import_spec": "hgn-spec-1.1.0@79e75fd4ae7642478823525b0a3212951f584af5",
            "canonical_values": "decoded-f32-no-intermediate-bf16-rounding", "shards": shards,
            "physical_tensors": tensors, "sidecars": sidecars }),
        )
    }

    fn input_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.root.clone()];
        if let Some(assets) = &self.assets {
            paths.push(assets.root.clone());
        }
        paths.extend(self.shards.iter().map(|shard| shard.input.path.clone()));
        for sidecar in &self.sidecars {
            paths.extend(sidecar.input_paths());
        }
        paths
    }
}

fn load_shards(path: &Path) -> Result<(Vec<Shard>, Vec<PhysicalTensor>, BTreeMap<String, Value>)> {
    let paths = shard_paths(path)?;
    let mut shards = Vec::with_capacity(paths.len());
    let mut tensors = Vec::new();
    let mut names = BTreeSet::new();
    let mut metadata = BTreeMap::new();
    for (index, path) in paths.iter().enumerate() {
        let (shard, parsed) = parse_shard(path, index)?;
        for (key, value) in &shard.metadata {
            if key.starts_with("qwen4exp.") || key == "general.architecture" {
                if let Some(previous) = metadata.insert(key.clone(), value.clone()) {
                    ensure!(&previous == value, "inconsistent split GGUF metadata {key}");
                }
            }
        }
        for (key, expected) in [
            ("split.no", index as u64),
            ("split.count", paths.len() as u64),
        ] {
            if let Some(value) = shard.metadata.get(key) {
                ensure!(
                    value.as_u64() == Some(expected),
                    "GGUF {key} disagrees with shard filename"
                );
            }
        }
        for tensor in parsed {
            ensure!(
                names.insert(tensor.name.clone()),
                "GGUF tensor appears in multiple shards: {}",
                tensor.name
            );
            tensors.push(tensor);
        }
        ensure!(
            tensors.len() as u64 <= MAX_TENSORS,
            "split GGUF tensor count exceeds bound"
        );
        shards.push(shard);
    }
    for shard in &shards {
        if let Some(count) = shard.metadata.get("split.tensors.count") {
            ensure!(
                count.as_u64() == Some(tensors.len() as u64),
                "GGUF split.tensors.count mismatch"
            );
        }
    }
    ensure!(
        metadata.get("general.architecture").and_then(Value::as_str) == Some("qwen4exp"),
        "GGUF general.architecture must be qwen4exp"
    );
    for (key, value) in [
        ("qwen4exp.ssm.group_count", 16),
        ("qwen4exp.ssm.time_step_rank", 48),
        ("qwen4exp.ssm.state_size", 128),
        ("qwen4exp.block_count", 48),
        ("qwen4exp.expert_count", 512),
        ("qwen4exp.embedding_length", 2560),
    ] {
        ensure!(
            metadata.get(key).and_then(Value::as_u64) == Some(value),
            "GGUF {key} must be {value}"
        );
    }
    Ok((shards, tensors, metadata))
}

fn hf_name(name: &str) -> String {
    let suffix = if name.ends_with(".mlp.experts.gate_up_proj.weight")
        || name.ends_with(".mlp.experts.down_proj.weight")
    {
        name.strip_suffix(".weight").unwrap()
    } else {
        name
    };
    if name == "lm_head.weight" {
        suffix.to_owned()
    } else {
        format!("model.language_model.{suffix}")
    }
}

fn add_tensor(
    tensors: &mut BTreeMap<String, TensorInfo>,
    layouts: &mut BTreeMap<String, Layout>,
    name: String,
    shape: Vec<u64>,
    layout: Layout,
) -> Result<()> {
    let dtype = if matches!(layout, Layout::Integers(_)) {
        Dtype::I64
    } else {
        Dtype::F32
    };
    let size = product(&shape)?
        .checked_mul(dtype.byte_width())
        .context("GGUF canonical byte size overflow")?;
    let info = TensorInfo {
        name: name.clone(),
        shard: "gguf-canonical".to_owned(),
        dtype,
        shape,
        offset: 0,
        size,
    };
    ensure!(
        tensors.insert(name.clone(), info).is_none(),
        "duplicate canonical GGUF tensor {name}"
    );
    layouts.insert(name, layout);
    Ok(())
}

fn canonical_tensors(
    physical: &[PhysicalTensor],
    metadata: &BTreeMap<String, Value>,
) -> Result<(BTreeMap<String, TensorInfo>, BTreeMap<String, Layout>)> {
    let names: BTreeMap<&str, usize> = physical
        .iter()
        .enumerate()
        .map(|(i, tensor)| (tensor.name.as_str(), i))
        .collect();
    let mut tensors = BTreeMap::new();
    let mut layouts = BTreeMap::new();
    for (index, tensor) in physical.iter().enumerate() {
        if tensor.name == "per_layer_token_embd.weight" {
            ensure!(
                tensor.shape.len() == 2 && tensor.shape[1] == 160 && tensor.shape[0] % 128 == 0,
                "invalid GGUF n-gram shape"
            );
            let rows = tensor.shape[0] / 128;
            for shard in 0..128 {
                add_tensor(
                    &mut tensors,
                    &mut layouts,
                    format!("{NGRAM_PREFIX}{shard}.weight"),
                    vec![rows, 160],
                    Layout::Tensor {
                        physical: index,
                        start: shard * rows * 160,
                        reorder: Reorder::None,
                        norm: false,
                        log_a: false,
                    },
                )?;
            }
            continue;
        }
        let (name, suffix) = mapped_name(&tensor.name)?;
        if matches!(suffix, "ffn_up_exps.weight" | "indexer.k_proj.weight") {
            let counterpart = tensor.name.replace(
                suffix,
                if suffix == "ffn_up_exps.weight" {
                    "ffn_gate_exps.weight"
                } else {
                    "indexer.q_proj.weight"
                },
            );
            ensure!(
                names.contains_key(counterpart.as_str()),
                "missing GGUF fused tensor {counterpart}"
            );
            continue;
        }
        if matches!(suffix, "ffn_gate_exps.weight" | "indexer.q_proj.weight") {
            let experts = suffix == "ffn_gate_exps.weight";
            let counterpart = tensor.name.replace(
                suffix,
                if experts {
                    "ffn_up_exps.weight"
                } else {
                    "indexer.k_proj.weight"
                },
            );
            let &second = names
                .get(counterpart.as_str())
                .with_context(|| format!("missing GGUF fused tensor {counterpart}"))?;
            let other = &physical[second];
            ensure!(
                tensor.ty == other.ty,
                "GGUF fused pair must have the same type: {}",
                tensor.name
            );
            let (shape, half) = if experts {
                ensure!(
                    tensor.shape == [512, 640, 2560] && other.shape == tensor.shape,
                    "invalid GGUF fused expert shape"
                );
                (vec![512, 1280, 2560], 640 * 2560)
            } else {
                ensure!(
                    tensor.shape == [512, 2560] && other.shape == [128, 2560],
                    "invalid GGUF indexer pair shape"
                );
                (vec![640, 2560], 512 * 2560)
            };
            add_tensor(
                &mut tensors,
                &mut layouts,
                hf_name(&name),
                shape,
                Layout::Pair {
                    first: index,
                    second,
                    half,
                    experts,
                },
            )?;
            continue;
        }
        let mut shape = tensor.shape.clone();
        // GGUF writers may retain or omit size-one axes. These two canonical
        // reshapes restore required singleton axes without changing value order;
        // non-singleton axes must still match the model contract.
        if suffix.ends_with("conv1d.weight") || suffix == "ffn_gate_inp_shexp.weight" {
            shape.retain(|&dimension| dimension != 1);
        }
        if suffix.ends_with("conv1d.weight") {
            ensure!(
                shape.len() == 2,
                "GGUF convolution must have two non-singleton dimensions"
            );
            shape.insert(1, 1);
        }
        if suffix == "ffn_gate_inp_shexp.weight" {
            ensure!(shape == [2560], "invalid GGUF shared expert gate shape");
            shape.insert(0, 1);
        }
        let reorder = match suffix {
            "attn_qkv.weight" => {
                ensure!(shape == [10240, 2560], "invalid GGUF QKV shape");
                Reorder::Heads {
                    prefix: 4096 * 2560,
                    unit: 128 * 2560,
                }
            }
            "attn_gate.weight" => {
                ensure!(shape == [6144, 2560], "invalid GGUF DeltaNet gate shape");
                Reorder::Heads {
                    prefix: 0,
                    unit: 128 * 2560,
                }
            }
            "ssm_alpha.weight" | "ssm_beta.weight" => {
                ensure!(
                    shape == [48, 2560],
                    "invalid GGUF DeltaNet head projection shape"
                );
                Reorder::Heads {
                    prefix: 0,
                    unit: 2560,
                }
            }
            "ssm_out.weight" => {
                ensure!(shape == [2560, 6144], "invalid GGUF DeltaNet output shape");
                Reorder::Columns
            }
            "ssm_conv1d.weight" => {
                ensure!(
                    shape == [10240, 1, 4],
                    "invalid GGUF DeltaNet convolution shape"
                );
                Reorder::Heads {
                    prefix: 4096 * 4,
                    unit: 128 * 4,
                }
            }
            "ssm_a" | "ssm_dt.bias" => {
                ensure!(shape == [48], "invalid GGUF DeltaNet head vector shape");
                Reorder::Heads { prefix: 0, unit: 1 }
            }
            _ => Reorder::None,
        };
        let norm = [
            "hc_norm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "q_layernorm.weight",
            "k_layernorm.weight",
            "ple.norm_key.weight",
            "ple.norm_query.weight",
            "ple.norm_conv.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix));
        if suffix == "ssm_a" {
            ensure!(tensor.ty == GgmlType::F32, "ssm_a requires GGUF F32");
        }
        add_tensor(
            &mut tensors,
            &mut layouts,
            hf_name(&name),
            shape,
            Layout::Tensor {
                physical: index,
                start: 0,
                reorder,
                norm,
                log_a: suffix == "ssm_a",
            },
        )?;
    }
    for &(key, suffix, count) in PLE {
        let values = metadata
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("missing GGUF integer array {key}"))?;
        ensure!(
            values.len() == count,
            "GGUF {key} must have {count} integers"
        );
        let mut bytes = Vec::with_capacity(count * 8);
        for value in values {
            bytes.extend_from_slice(
                &value
                    .as_i64()
                    .with_context(|| format!("GGUF {key} contains a non-I64 integer"))?
                    .to_le_bytes(),
            );
        }
        add_tensor(
            &mut tensors,
            &mut layouts,
            format!("model.language_model.layers.1.ple.ple_embedding.{suffix}"),
            vec![count as u64],
            Layout::Integers(bytes),
        )?;
    }
    Ok((tensors, layouts))
}

fn mapped_name(name: &str) -> Result<(String, &str)> {
    let whole = match name {
        "token_embd.weight" => Some("embed_tokens.weight"),
        "output.weight" => Some("lm_head.weight"),
        "output_hc_down.weight" => Some("hyper_connection_mixer.input_mix_weight_down.weight"),
        "output_hc_up.weight" => Some("hyper_connection_mixer.input_mix_weight_up.weight"),
        "output_hc_norm.weight" => Some("hyper_connection_mixer.hc_norm.weight"),
        _ => None,
    };
    if let Some(mapped) = whole {
        return Ok((mapped.to_owned(), name));
    }
    let (layer, suffix) = name
        .strip_prefix("blk.")
        .and_then(|name| name.split_once('.'))
        .with_context(|| format!("unknown GGUF tensor name {name}"))?;
    let layer_number: u32 = layer.parse().context("invalid GGUF layer number")?;
    ensure!(
        layer_number < 48 && layer_number.to_string() == layer,
        "invalid GGUF layer number {layer}"
    );
    let mapped = match suffix {
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "indexer.q_norm.weight" => "self_attn.indexer.q_layernorm.weight",
        "indexer.k_norm.weight" => "self_attn.indexer.k_layernorm.weight",
        "indexer.q_proj.weight" | "indexer.k_proj.weight" => {
            "self_attn.indexer.index_qk_proj.weight"
        }
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" => "mlp.experts.gate_up_proj.weight",
        "ffn_down_exps.weight" => "mlp.experts.down_proj.weight",
        "hc_attn_down.weight" => "attn_hyper_connection.input_mix_weight_down.weight",
        "hc_attn_up.weight" => "attn_hyper_connection.input_mix_weight_up.weight",
        "hc_attn_inject.weight" => "attn_hyper_connection.block_inject_weight.weight",
        "hc_attn_norm.weight" => "attn_hyper_connection.hc_norm.weight",
        "hc_ffn_down.weight" => "mlp_hyper_connection.input_mix_weight_down.weight",
        "hc_ffn_up.weight" => "mlp_hyper_connection.input_mix_weight_up.weight",
        "hc_ffn_inject.weight" => "mlp_hyper_connection.block_inject_weight.weight",
        "hc_ffn_norm.weight" => "mlp_hyper_connection.hc_norm.weight",
        "ple_key.weight" => "ple.key_proj.weight",
        "ple_value.weight" => "ple.value_proj.weight",
        "ple_conv1d.weight" => "ple.conv1d.weight",
        "ple_norm_key.weight" => "ple.norm_key.weight",
        "ple_norm_query.weight" => "ple.norm_query.weight",
        "ple_norm_conv.weight" => "ple.norm_conv.weight",
        _ => bail!("unknown GGUF tensor name {name}"),
    };
    Ok((format!("layers.{layer}.{mapped}"), suffix))
}

#[cfg(test)]
#[path = "gguf_source_tests.rs"]
mod tests;
