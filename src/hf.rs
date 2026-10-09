//! Metadata-validated, bounded-window access to an immutable local HF snapshot.
//!
//! Callers must keep the source immutable until `check_unchanged` succeeds at the
//! end of conversion. Signatures detect ordinary edits and replacements; they
//! cannot make an mmap safe against a concurrent, uncooperative file truncation.

use anyhow::{bail, ensure, Context, Result};
use memmap2::MmapOptions;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const INDEX: &str = "model.safetensors.index.json";
const CONFIG: &str = "config.json";
const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const WINDOW_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    I64,
}

impl Dtype {
    pub fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub shard: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Absolute byte offset in the shard, including its prefix and JSON header.
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileSignature {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed: (i64, i64),
}

impl FileSignature {
    fn from_metadata(metadata: &Metadata) -> Result<Self> {
        Ok(Self {
            len: metadata.len(),
            modified: metadata
                .modified()
                .context("reading source modification time")?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

#[derive(Debug)]
pub struct Snapshot {
    pub root: PathBuf,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub config: Value,
    root_signature: FileSignature,
    signatures: BTreeMap<PathBuf, FileSignature>,
}

impl Snapshot {
    /// Read only directory metadata, the index/config JSON and shard headers.
    /// Architecture-specific names, shapes and required assets are checked by
    /// the planner; this reader also accepts small, structurally valid fixtures.
    pub fn open(root: &Path) -> Result<Self> {
        let root_signature = directory_signature(root)?;
        let root = root
            .canonicalize()
            .context("canonicalizing source directory")?;
        ensure!(
            directory_signature(&root)? == root_signature,
            "source directory changed while opening"
        );
        let signatures = inventory(&root)?;
        let index_signature = signatures
            .get(Path::new(INDEX))
            .context("missing model.safetensors.index.json")?;
        let config_signature = signatures
            .get(Path::new(CONFIG))
            .context("missing config.json")?;
        let index = read_json_checked(&root.join(INDEX), index_signature)?;
        let config = read_json_checked(&root.join(CONFIG), config_signature)?;
        ensure!(config.is_object(), "config.json must be a JSON object");
        let index_object = index.as_object().context("index must be a JSON object")?;
        let weight_map = index_object
            .get("weight_map")
            .and_then(Value::as_object)
            .context("index weight_map must be a JSON object")?;
        ensure!(!weight_map.is_empty(), "index weight_map must not be empty");

        let mut shards: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (name, value) in weight_map {
            ensure!(
                !name.is_empty() && name != "__metadata__" && !name.chars().any(char::is_control),
                "invalid tensor name in index: {name:?}"
            );
            let shard = value
                .as_str()
                .with_context(|| format!("shard for {name} must be a string"))?;
            ensure!(
                safe_filename(shard) && shard.ends_with(".safetensors"),
                "unsafe safetensors shard path for {name}: {shard:?}"
            );
            ensure!(
                signatures.contains_key(Path::new(shard)),
                "missing indexed shard: {shard}"
            );
            shards.entry(shard).or_default().insert(name);
        }
        for path in signatures.keys().filter(|path| is_safetensors(path)) {
            let shard = path.to_str().context("non-UTF-8 safetensors path")?;
            ensure!(
                shards.contains_key(shard),
                "unindexed safetensors shard: {shard}"
            );
        }

        let mut tensors = BTreeMap::new();
        let mut total_size = 0u64;
        for (shard, names) in &shards {
            let signature = signatures
                .get(Path::new(shard))
                .context("indexed shard disappeared from inventory")?;
            let size = read_shard(&root, shard, signature, names, &mut tensors)?;
            total_size = total_size
                .checked_add(size)
                .context("source byte count overflow")?;
        }
        ensure!(
            tensors.len() == weight_map.len(),
            "index/header tensor count mismatch"
        );
        if let Some(metadata) = index_object.get("metadata") {
            let metadata = metadata
                .as_object()
                .context("index metadata must be an object")?;
            if let Some(declared) = metadata.get("total_size") {
                ensure!(
                    index_total_size(declared)? == total_size,
                    "index metadata.total_size does not match tensor bytes"
                );
            }
        }

        let snapshot = Self {
            root,
            tensors,
            config,
            root_signature,
            signatures,
        };
        snapshot.check_unchanged()?;
        Ok(snapshot)
    }

    pub fn read_bytes(&self, name: &str, byte_offset: u64, len: usize) -> Result<Vec<u8>> {
        let len_u64 = u64::try_from(len).context("byte count exceeds u64")?;
        let (tensor, offset) = self.byte_range(name, byte_offset, len_u64)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .context("allocating tensor byte result")?;
        self.visit_range(tensor, offset, len_u64, |bytes| {
            output.extend_from_slice(bytes);
        })?;
        Ok(output)
    }

    /// Decode floating-point tensors without allocating an intermediate byte
    /// vector. I64 must use `read_bytes`, so integer precision is never lost.
    pub fn read_f32(&self, name: &str, element_offset: u64, len: usize) -> Result<Vec<f32>> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown tensor: {name}"))?;
        ensure!(
            tensor.dtype != Dtype::I64,
            "I64 tensor {name} requires read_bytes"
        );
        let width = tensor.dtype.byte_width();
        let byte_offset = element_offset
            .checked_mul(width)
            .context("element offset overflow")?;
        let byte_len = u64::try_from(len)
            .context("element count exceeds u64")?
            .checked_mul(width)
            .context("element byte count overflow")?;
        let (tensor, offset) = self.byte_range(name, byte_offset, byte_len)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(len)
            .context("allocating floating-point result")?;
        self.visit_range(tensor, offset, byte_len, |bytes| match tensor.dtype {
            Dtype::Bf16 => output.extend(bytes.chunks_exact(2).map(|value| {
                f32::from_bits(u32::from(u16::from_le_bytes([value[0], value[1]])) << 16)
            })),
            Dtype::F32 => output.extend(
                bytes
                    .chunks_exact(4)
                    .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]])),
            ),
            Dtype::I64 => unreachable!("I64 was rejected before reading"),
        })?;
        Ok(output)
    }

    /// SHA-256 over exactly the raw tensor bytes, never over the whole shard.
    pub fn hash_tensor(&self, name: &str) -> Result<String> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown tensor: {name}"))?;
        let mut hasher = Sha256::new();
        self.visit_range(tensor, tensor.offset, tensor.size, |bytes| {
            hasher.update(bytes)
        })?;
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Recheck all source files, including assets, and discover newly added
    /// files. This is metadata-only; no shard payload or whole-file hash is read.
    pub fn check_unchanged(&self) -> Result<()> {
        self.check_root()?;
        let current = inventory(&self.root)?;
        for (path, signature) in &self.signatures {
            ensure!(
                current.get(path) == Some(signature),
                "source file changed or disappeared: {}",
                path.display()
            );
        }
        ensure!(
            current.len() == self.signatures.len(),
            "files added to source snapshot"
        );
        self.check_root()
    }

    fn check_root(&self) -> Result<()> {
        ensure!(
            directory_signature(&self.root)? == self.root_signature,
            "source directory changed"
        );
        Ok(())
    }

    fn byte_range(&self, name: &str, byte_offset: u64, len: u64) -> Result<(&TensorInfo, u64)> {
        let tensor = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown tensor: {name}"))?;
        let end = byte_offset
            .checked_add(len)
            .context("tensor byte range overflow")?;
        ensure!(end <= tensor.size, "byte range exceeds tensor {name}");
        let offset = tensor
            .offset
            .checked_add(byte_offset)
            .context("shard offset overflow")?;
        Ok((tensor, offset))
    }

    fn visit_range(
        &self,
        tensor: &TensorInfo,
        offset: u64,
        len: u64,
        mut visit: impl FnMut(&[u8]),
    ) -> Result<()> {
        self.check_root()?;
        let signature = self
            .signatures
            .get(Path::new(&tensor.shard))
            .context("tensor references an unknown source shard")?;
        let end = offset
            .checked_add(len)
            .context("shard byte range overflow")?;
        ensure!(
            end <= signature.len,
            "tensor byte range exceeds source shard"
        );
        let path = self.root.join(&tensor.shard);
        let file = open_checked(&path, signature)?;
        let mut position = offset;
        while position < end {
            let size = usize::try_from((end - position).min(WINDOW_BYTES as u64))?;
            // The snapshot is required to remain immutable. Bounds and file
            // identity are checked before mapping and after every window. The
            // mapping is read-only and does not outlive this callback.
            let mapping = unsafe { MmapOptions::new().offset(position).len(size).map(&file) }
                .with_context(|| format!("mapping source window in {}", tensor.shard))?;
            visit(&mapping);
            drop(mapping);
            discard_pages(&file, position, size);
            ensure_file_unchanged(&file, &path, signature)?;
            position += size as u64;
        }
        ensure_file_unchanged(&file, &path, signature)?;
        self.check_root()
    }
}

fn safe_filename(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.ends_with('.')
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"-_.".contains(&c))
        && matches!(
            (
                Path::new(name).components().next(),
                Path::new(name).components().count()
            ),
            (Some(Component::Normal(_)), 1)
        )
}

fn is_safetensors(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

fn temporary_name(name: &str) -> bool {
    name.ends_with('~')
        || [
            ".incomplete",
            ".tmp",
            ".temp",
            ".part",
            ".partial",
            ".swp",
            ".swo",
        ]
        .iter()
        .any(|suffix| {
            name.match_indices(*suffix).any(|(position, _)| {
                let remaining = &name[position + suffix.len()..];
                remaining.is_empty() || remaining.starts_with('.')
            })
        })
}

fn inventory(root: &Path) -> Result<BTreeMap<PathBuf, FileSignature>> {
    let mut files = BTreeMap::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(&directory)
            .with_context(|| format!("listing source directory {}", directory.display()))?
        {
            let entry = entry.context("reading source directory entry")?;
            let path = entry.path();
            let filename = entry.file_name();
            let name = filename.to_str().context("non-UTF-8 source filename")?;
            ensure!(
                !temporary_name(name),
                "incomplete or temporary source entry: {}",
                path.display()
            );
            let metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("reading source metadata {}", path.display()))?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "symlink source entry: {}",
                path.display()
            );
            if metadata.is_dir() {
                directories.push(path);
            } else {
                ensure!(
                    metadata.is_file(),
                    "non-regular source file: {}",
                    path.display()
                );
                let relative = path
                    .strip_prefix(root)
                    .context("source entry escapes root")?;
                ensure!(
                    !is_safetensors(relative) || relative.components().count() == 1,
                    "safetensors shard must be flat, not nested: {}",
                    relative.display()
                );
                files.insert(
                    relative.to_path_buf(),
                    FileSignature::from_metadata(&metadata)?,
                );
            }
        }
    }
    Ok(files)
}

fn directory_signature(path: &Path) -> Result<FileSignature> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading source directory {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "source must be a real directory, not a symlink: {}",
        path.display()
    );
    FileSignature::from_metadata(&metadata)
}

fn regular_signature(path: &Path) -> Result<FileSignature> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading source file metadata {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "source must be a regular file, not a symlink: {}",
        path.display()
    );
    FileSignature::from_metadata(&metadata)
}

fn open_checked(path: &Path, signature: &FileSignature) -> Result<File> {
    ensure!(
        regular_signature(path)? == *signature,
        "source file changed: {}",
        path.display()
    );
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    let file = options
        .open(path)
        .with_context(|| format!("opening source file {}", path.display()))?;
    ensure_file_unchanged(&file, path, signature)?;
    Ok(file)
}

fn ensure_file_unchanged(file: &File, path: &Path, signature: &FileSignature) -> Result<()> {
    let metadata = file
        .metadata()
        .with_context(|| format!("checking open source {}", path.display()))?;
    ensure!(
        metadata.is_file()
            && FileSignature::from_metadata(&metadata)? == *signature
            && regular_signature(path)? == *signature,
        "source file changed: {}",
        path.display()
    );
    Ok(())
}

// Eviction is a hint, not a correctness requirement. Other targets still use
// bounded read-only mappings and the same bounds/signature checks.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
fn discard_pages(file: &File, offset: u64, len: usize) {
    use std::os::fd::AsRawFd;

    if let (Ok(offset), Ok(len)) = (libc::off_t::try_from(offset), libc::off_t::try_from(len)) {
        // SAFETY: this advises only the already-consumed range of a live file.
        // Some filesystems do not support fadvise; its failure cannot invalidate
        // successfully read bytes and must not change portable behavior.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), offset, len, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
fn discard_pages(_file: &File, _offset: u64, _len: usize) {}

fn read_shard(
    root: &Path,
    shard: &str,
    signature: &FileSignature,
    expected: &BTreeSet<&str>,
    tensors: &mut BTreeMap<String, TensorInfo>,
) -> Result<u64> {
    let path = root.join(shard);
    let mut file = open_checked(&path, signature)?;
    ensure!(
        signature.len >= 8,
        "truncated safetensors length prefix: {shard}"
    );
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("reading header length in {shard}"))?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(
        header_len > 0 && header_len <= MAX_HEADER_BYTES,
        "invalid safetensors header length in {shard}"
    );
    let data_start = header_len
        .checked_add(8)
        .context("safetensors header offset overflow")?;
    ensure!(
        data_start <= signature.len,
        "truncated safetensors header: {shard}"
    );
    let mut bytes = vec![0u8; usize::try_from(header_len)?];
    // Do not use a buffered reader: its read-ahead could touch tensor payloads.
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading safetensors header in {shard}"))?;
    ensure_file_unchanged(&file, &path, signature)?;
    ensure!(
        bytes.first() == Some(&b'{'),
        "safetensors header must start with '{{': {shard}"
    );
    let header =
        parse_json(&bytes).with_context(|| format!("invalid safetensors JSON in {shard}"))?;
    let header = header
        .as_object()
        .context("safetensors header must be an object")?;
    if let Some(metadata) = header.get("__metadata__") {
        ensure!(
            metadata
                .as_object()
                .is_some_and(|map| map.values().all(Value::is_string)),
            "safetensors __metadata__ must contain only string values: {shard}"
        );
    }
    let count = header.len() - usize::from(header.contains_key("__metadata__"));
    ensure!(
        count == expected.len(),
        "index/header tensor count mismatch in {shard}"
    );
    let mut ranges = Vec::with_capacity(count);
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        ensure!(
            expected.contains(name.as_str()),
            "index/header tensor mismatch: {name} in {shard}"
        );
        let descriptor = value
            .as_object()
            .with_context(|| format!("invalid tensor descriptor: {name}"))?;
        ensure!(
            descriptor.len() == 3
                && descriptor.contains_key("dtype")
                && descriptor.contains_key("shape")
                && descriptor.contains_key("data_offsets"),
            "tensor descriptor must contain exactly dtype, shape and data_offsets: {name}"
        );
        let dtype = match descriptor.get("dtype").and_then(Value::as_str) {
            Some("BF16") => Dtype::Bf16,
            Some("F32") => Dtype::F32,
            Some("I64") => Dtype::I64,
            _ => bail!("unsupported dtype for tensor {name}"),
        };
        let dimensions = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .with_context(|| format!("invalid shape for tensor {name}"))?;
        ensure!(
            (1..=5).contains(&dimensions.len()),
            "tensor rank must be 1..5: {name}"
        );
        let mut shape = Vec::with_capacity(dimensions.len());
        let mut elements = 1u64;
        for dimension in dimensions {
            let dimension = dimension
                .as_u64()
                .with_context(|| format!("non-integer dimension: {name}"))?;
            ensure!(dimension > 0, "tensor dimensions must be positive: {name}");
            elements = elements
                .checked_mul(dimension)
                .with_context(|| format!("shape overflow: {name}"))?;
            shape.push(dimension);
        }
        let size = elements
            .checked_mul(dtype.byte_width())
            .with_context(|| format!("tensor size overflow: {name}"))?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .with_context(|| format!("invalid data_offsets: {name}"))?;
        ensure!(
            offsets.len() == 2,
            "data_offsets must have two entries: {name}"
        );
        let start = offsets[0]
            .as_u64()
            .with_context(|| format!("invalid start offset: {name}"))?;
        let end = offsets[1]
            .as_u64()
            .with_context(|| format!("invalid end offset: {name}"))?;
        ensure!(
            end.checked_sub(start) == Some(size),
            "shape/byte length mismatch: {name}"
        );
        let absolute_end = data_start
            .checked_add(end)
            .with_context(|| format!("tensor end overflow: {name}"))?;
        ensure!(
            absolute_end <= signature.len,
            "tensor range exceeds shard bounds: {name}"
        );
        let offset = data_start
            .checked_add(start)
            .with_context(|| format!("tensor offset overflow: {name}"))?;
        let info = TensorInfo {
            name: name.clone(),
            shard: shard.to_owned(),
            dtype,
            shape,
            offset,
            size,
        };
        ensure!(
            tensors.insert(name.clone(), info).is_none(),
            "duplicate tensor across shards: {name}"
        );
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    let mut covered = 0u64;
    for (start, end) in ranges {
        ensure!(
            start == covered,
            "overlapping or gapped tensor ranges in {shard}"
        );
        covered = end;
    }
    ensure!(
        data_start.checked_add(covered) == Some(signature.len),
        "trailing bytes or incomplete tensor coverage in {shard}"
    );
    Ok(covered)
}

fn index_total_size(value: &Value) -> Result<u64> {
    if let Some(size) = value.as_u64() {
        return Ok(size);
    }
    // The actual BF16 source index writes an integral JSON float. Limit that
    // spelling to the exact-integer range, rather than rounding a large total.
    if let Some(size) = value.as_f64() {
        if size.is_finite()
            && (0.0..=9_007_199_254_740_992.0).contains(&size)
            && size.fract() == 0.0
        {
            return Ok(size as u64);
        }
    }
    bail!("index metadata.total_size must be an exact nonnegative integer")
}

/// Read at most 16 MiB of JSON, rejecting duplicate object keys at every depth.
pub fn read_json(path: &Path) -> Result<Value> {
    let signature = regular_signature(path)?;
    read_json_checked(path, &signature)
}

fn read_json_checked(path: &Path, signature: &FileSignature) -> Result<Value> {
    ensure!(
        signature.len <= MAX_JSON_BYTES,
        "JSON exceeds 16 MiB: {}",
        path.display()
    );
    let mut file = open_checked(path, signature)?;
    let mut bytes = vec![0u8; usize::try_from(signature.len)?];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading JSON {}", path.display()))?;
    ensure_file_unchanged(&file, path, signature)?;
    parse_json(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

struct UniqueValue(Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;

impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object keys")
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(value)))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
        Number::from_f64(value)
            .map(|number| UniqueValue(Value::Number(number)))
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value.to_owned())))
    }

    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(value)))
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut sequence: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut object: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = object.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key: {key:?}")));
            }
            let UniqueValue(value) = object.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

fn parse_json(bytes: &[u8]) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let UniqueValue(value) = UniqueValue::deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    use tempfile::{tempdir, TempDir};

    const SHARD: &str = "model.safetensors";
    const SIMPLE_HEADER: &str = r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    const SIMPLE_BYTES: [u8; 4] = [0x80, 0x3f, 0x00, 0xc0];

    fn write_shard(root: &Path, name: &str, header: &str, payload: &[u8]) -> u64 {
        let mut header = header.as_bytes().to_vec();
        header.resize(header.len() + (8 - header.len() % 8) % 8, b' ');
        let mut file = File::create(root.join(name)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(payload).unwrap();
        8 + header.len() as u64
    }

    fn write_index(root: &Path, index: Value) {
        fs::write(root.join(INDEX), serde_json::to_vec(&index).unwrap()).unwrap();
    }

    fn fixture(header: &str, payload: &[u8]) -> TempDir {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join(CONFIG), b"{}").unwrap();
        write_index(directory.path(), json!({"weight_map": {"x": SHARD}}));
        write_shard(directory.path(), SHARD, header, payload);
        directory
    }

    fn assert_rejected(root: &Path, message: &str) {
        let error = Snapshot::open(root).unwrap_err();
        assert!(
            format!("{error:#}").contains(message),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn reads_real_bf16_f32_and_exact_i64_bytes() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(
            root.join(CONFIG),
            br#"{"nested":{"flags":[true,null,"value"]}}"#,
        )
        .unwrap();
        let mut payload = vec![0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f];
        payload.extend_from_slice(&(-0.0f32).to_le_bytes());
        payload.extend_from_slice(&3.25f32.to_le_bytes());
        payload.extend_from_slice(&(-(1i64 << 60)).to_le_bytes());
        payload.extend_from_slice(&i64::MAX.to_le_bytes());
        let offset = write_shard(
            root,
            SHARD,
            r#"{"integer":{"dtype":"I64","shape":[2],"data_offsets":[14,30]},"float":{"dtype":"F32","shape":[2],"data_offsets":[6,14]},"bf":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},"__metadata__":{"format":"pt"}}"#,
            &payload,
        );
        write_index(
            root,
            json!({
                "metadata": {"total_size": 30.0},
                "weight_map": {"bf": SHARD, "float": SHARD, "integer": SHARD}
            }),
        );
        let snapshot = Snapshot::open(root).unwrap();
        assert!(snapshot.root.is_absolute());
        assert_eq!(snapshot.config["nested"]["flags"][0], true);
        assert_eq!(snapshot.tensors["bf"].offset, offset);
        assert_eq!(snapshot.tensors["integer"].offset, offset + 14);
        assert_eq!(snapshot.tensors["integer"].dtype, Dtype::I64);
        assert_eq!(snapshot.read_f32("bf", 0, 3).unwrap(), [1.0, -2.0, 0.5]);
        assert_eq!(snapshot.read_f32("bf", 1, 2).unwrap(), [-2.0, 0.5]);
        let floats = snapshot.read_f32("float", 0, 2).unwrap();
        assert_eq!(floats[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(floats[1], 3.25);
        assert_eq!(
            snapshot.read_bytes("integer", 0, 16).unwrap(),
            payload[14..]
        );
        assert!(snapshot.read_f32("integer", 0, 2).is_err());
        for (name, bytes) in [
            ("bf", &payload[..6]),
            ("float", &payload[6..14]),
            ("integer", &payload[14..]),
        ] {
            assert_eq!(
                snapshot.hash_tensor(name).unwrap(),
                format!("{:x}", Sha256::digest(bytes))
            );
        }
        snapshot.check_unchanged().unwrap();
    }

    #[test]
    fn reads_and_hashes_across_bounded_mapping_windows() {
        let payload: Vec<u8> = (0..WINDOW_BYTES + 16)
            .map(|index| (index % 251) as u8)
            .collect();
        let header = json!({"x": {
            "dtype": "BF16", "shape": [payload.len() / 2], "data_offsets": [0, payload.len()]
        }})
        .to_string();
        let directory = fixture(&header, &payload);
        let snapshot = Snapshot::open(directory.path()).unwrap();
        assert_eq!(
            snapshot.hash_tensor("x").unwrap(),
            format!("{:x}", Sha256::digest(&payload))
        );
        assert_eq!(
            snapshot.read_bytes("x", 3, payload.len() - 3).unwrap(),
            payload[3..]
        );
        let floats = snapshot.read_f32("x", 1, payload.len() / 2 - 1).unwrap();
        for index in [0, WINDOW_BYTES / 2 - 1, floats.len() - 1] {
            let position = (index + 1) * 2;
            let bits = u32::from(u16::from_le_bytes([
                payload[position],
                payload[position + 1],
            ])) << 16;
            assert_eq!(floats[index].to_bits(), bits);
        }
    }

    #[test]
    fn opens_large_sparse_source_from_headers_only() {
        let payload_size = 1u64 << 34;
        let header = json!({"x": {
            "dtype": "BF16", "shape": [payload_size / 2], "data_offsets": [0, payload_size]
        }})
        .to_string();
        let directory = fixture(&header, &[]);
        let path = directory.path().join(SHARD);
        let file = OpenOptions::new().write(true).open(path).unwrap();
        let data_start = file.metadata().unwrap().len();
        file.set_len(data_start + payload_size).unwrap();
        let snapshot = Snapshot::open(directory.path()).unwrap();
        assert_eq!(snapshot.tensors["x"].size, payload_size);
        assert_eq!(snapshot.tensors["x"].offset, data_start);
    }

    #[test]
    fn opens_1658_structural_tensors_without_architecture_policy() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join(CONFIG), b"{}").unwrap();
        let mut header = Map::new();
        let mut weights = Map::new();
        for index in 0..1658 {
            let name = format!("tensor_{index}");
            header.insert(
                name.clone(),
                json!({
                    "dtype": "BF16", "shape": [1], "data_offsets": [index * 2, index * 2 + 2]
                }),
            );
            weights.insert(name, Value::String(SHARD.to_owned()));
        }
        header.insert(
            "tensor_0".to_owned(),
            json!({
                "dtype": "BF16", "shape": [1,1,1,1,1], "data_offsets": [0,2]
            }),
        );
        write_shard(
            root,
            SHARD,
            &Value::Object(header).to_string(),
            &vec![0; 1658 * 2],
        );
        write_index(
            root,
            json!({"metadata": {"total_size": 3316}, "weight_map": weights}),
        );
        let snapshot = Snapshot::open(root).unwrap();
        assert_eq!(snapshot.tensors.len(), 1658);
        assert_eq!(snapshot.tensors["tensor_0"].shape, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn enforces_read_ranges_including_empty_and_overflowing_ranges() {
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let snapshot = Snapshot::open(directory.path()).unwrap();
        assert!(snapshot.read_bytes("x", 4, 0).unwrap().is_empty());
        assert!(snapshot.read_f32("x", 2, 0).unwrap().is_empty());
        assert!(snapshot.read_bytes("x", 5, 0).is_err());
        assert!(snapshot.read_bytes("x", 3, 2).is_err());
        assert!(snapshot.read_bytes("x", u64::MAX, 1).is_err());
        assert!(snapshot.read_bytes("x", 0, usize::MAX).is_err());
        assert!(snapshot.read_f32("x", 2, 1).is_err());
        assert!(snapshot.read_f32("x", u64::MAX, 0).is_err());
        assert!(snapshot.read_f32("x", 0, usize::MAX).is_err());
        assert!(snapshot.read_bytes("missing", 0, 0).is_err());
        assert!(snapshot.hash_tensor("missing").is_err());
    }

    #[test]
    fn rejects_duplicate_json_keys_at_every_depth() {
        for json in [
            r#"{"a":1,"a":2}"#,
            r#"{"a":{"b":1,"b":2}}"#,
            r#"[{"a":1,"\u0061":2}]"#,
        ] {
            let directory = tempdir().unwrap();
            let path = directory.path().join("input.json");
            fs::write(&path, json).unwrap();
            assert!(format!("{:#}", read_json(&path).unwrap_err()).contains("duplicate JSON key"));
        }
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        fs::write(
            directory.path().join(INDEX),
            r#"{"weight_map":{"x":"model.safetensors","x":"other.safetensors"}}"#,
        )
        .unwrap();
        assert_rejected(directory.path(), "duplicate JSON key");
        let directory = fixture(
            r#"{"x":{"dtype":"BF16","dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            &SIMPLE_BYTES,
        );
        assert_rejected(directory.path(), "duplicate JSON key");
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        fs::write(directory.path().join(CONFIG), r#"{"a":{"b":1,"b":2}}"#).unwrap();
        assert_rejected(directory.path(), "duplicate JSON key");
    }

    #[test]
    fn rejects_oversized_or_truncated_metadata() {
        for length in [0, MAX_HEADER_BYTES + 1, u64::MAX] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            fs::write(directory.path().join(SHARD), length.to_le_bytes()).unwrap();
            assert_rejected(directory.path(), "header length");
        }
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        fs::write(directory.path().join(SHARD), [0u8; 7]).unwrap();
        assert_rejected(directory.path(), "length prefix");
        fs::write(directory.path().join(SHARD), 100u64.to_le_bytes()).unwrap();
        assert_rejected(directory.path(), "truncated safetensors header");
        let file = File::create(directory.path().join(CONFIG)).unwrap();
        file.set_len(MAX_JSON_BYTES + 1).unwrap();
        assert_rejected(directory.path(), "JSON exceeds 16 MiB");
    }

    #[test]
    fn rejects_invalid_dtypes_shapes_and_offset_numbers() {
        for descriptor in [
            json!({"dtype":"F16","shape":[2],"data_offsets":[0,4]}),
            json!({"dtype":"U8","shape":[4],"data_offsets":[0,4]}),
            json!({"dtype":"bf16","shape":[2],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[],"data_offsets":[0,2]}),
            json!({"dtype":"BF16","shape":[0],"data_offsets":[0,0]}),
            json!({"dtype":"BF16","shape":[-1],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[2.0],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[true],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[1,1,1,1,1,1],"data_offsets":[0,2]}),
            json!({"dtype":"BF16","shape":[u64::MAX,2],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[u64::MAX],"data_offsets":[0,4]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[4,0]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[0,3]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[-1,3]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[0.0,4]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[0,4,4]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[u64::MAX-4,u64::MAX]}),
            json!({"dtype":"BF16","shape":[2],"data_offsets":[0,4],"extra":true}),
        ] {
            let header = json!({"x":descriptor}).to_string();
            let directory = fixture(&header, &SIMPLE_BYTES);
            assert!(
                Snapshot::open(directory.path()).is_err(),
                "accepted {header}"
            );
        }
    }

    #[test]
    fn rejects_holes_overlaps_trailing_bytes_and_invalid_header_metadata() {
        let directory = fixture(
            r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}}"#,
            &[0; 6],
        );
        assert_rejected(directory.path(), "gapped");
        let directory = fixture(
            r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"y":{"dtype":"BF16","shape":[2],"data_offsets":[2,6]}}"#,
            &[0; 6],
        );
        write_index(
            directory.path(),
            json!({"weight_map":{"x":SHARD,"y":SHARD}}),
        );
        assert_rejected(directory.path(), "overlapping");
        let directory = fixture(SIMPLE_HEADER, &[0; 6]);
        assert_rejected(directory.path(), "trailing bytes");
        let directory = fixture(SIMPLE_HEADER, &[0; 2]);
        assert_rejected(directory.path(), "exceeds shard bounds");
        for header in [
            r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"__metadata__":{"format":1}}"#,
            r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"__metadata__":[]}"#,
            r#" {"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            r#"{"x":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}{}"#,
        ] {
            let directory = fixture(header, &SIMPLE_BYTES);
            assert!(
                Snapshot::open(directory.path()).is_err(),
                "accepted {header}"
            );
        }
    }

    #[test]
    fn requires_exact_safe_index_and_valid_totals() {
        for shard in [
            "../model.safetensors",
            "/model.safetensors",
            "nested/model.safetensors",
            "C:\\model.safetensors",
            "model.safetensors\u{0}",
            "model.bin",
            "model.safetensors ",
        ] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            write_index(directory.path(), json!({"weight_map":{"x":shard}}));
            assert_rejected(directory.path(), "unsafe");
        }
        for index in [
            json!({"weight_map":{"x":"missing.safetensors"}}),
            json!({"weight_map":{"y":SHARD}}),
            json!({"weight_map":{"x":SHARD,"y":SHARD}}),
            json!({"weight_map":{"":SHARD}}),
            json!({"weight_map":{"__metadata__":SHARD}}),
            json!({"weight_map":{"x":42}}),
            json!({"weight_map":{}}),
            json!({"weight_map":[]}),
            json!({"metadata":4,"weight_map":{"x":SHARD}}),
            json!({"metadata":{"total_size":3},"weight_map":{"x":SHARD}}),
            json!({"metadata":{"total_size":4.5},"weight_map":{"x":SHARD}}),
            json!({"metadata":{"total_size":-4},"weight_map":{"x":SHARD}}),
            json!({"metadata":{"total_size":true},"weight_map":{"x":SHARD}}),
            json!({"metadata":{"total_size":1e20},"weight_map":{"x":SHARD}}),
        ] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            write_index(directory.path(), index.clone());
            assert!(
                Snapshot::open(directory.path()).is_err(),
                "accepted {index}"
            );
        }
        for required in [CONFIG, INDEX] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            fs::remove_file(directory.path().join(required)).unwrap();
            assert_rejected(directory.path(), "missing");
        }
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        fs::write(directory.path().join(CONFIG), b"[]").unwrap();
        assert_rejected(directory.path(), "config.json must be a JSON object");
    }

    #[test]
    fn rejects_unindexed_nested_and_incomplete_source_files() {
        for name in [
            "extra.safetensors",
            "extra.SAFETENSORS",
            "model.safetensors.incomplete",
            "download.tmp",
            "model.safetensors.part",
            "backup~",
        ] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            fs::write(directory.path().join(name), b"unused").unwrap();
            assert!(Snapshot::open(directory.path()).is_err(), "accepted {name}");
        }
        for name in ["extra.safetensors", "download.incomplete"] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            let cache = directory.path().join(".cache/huggingface/download");
            fs::create_dir_all(&cache).unwrap();
            fs::write(cache.join(name), b"unused").unwrap();
            assert!(Snapshot::open(directory.path()).is_err());
        }
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let cache = directory.path().join(".cache/huggingface/download");
        fs::create_dir_all(&cache).unwrap();
        fs::write(
            cache.join("model.safetensors.metadata"),
            b"revision\netag\n0\n",
        )
        .unwrap();
        fs::write(cache.join("model.safetensors.lock"), b"").unwrap();
        Snapshot::open(directory.path()).unwrap();
    }

    #[test]
    fn detects_source_changes_before_reads_and_at_completion() {
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let snapshot = Snapshot::open(directory.path()).unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(directory.path().join(SHARD))
            .unwrap();
        file.write_all(&[0]).unwrap();
        assert!(snapshot.read_bytes("x", 0, 0).is_err());
        assert!(snapshot.read_f32("x", 0, 1).is_err());
        assert!(snapshot.hash_tensor("x").is_err());
        assert!(snapshot.check_unchanged().is_err());

        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        fs::write(directory.path().join("tokenizer.json"), b"{}").unwrap();
        let snapshot = Snapshot::open(directory.path()).unwrap();
        fs::write(
            directory.path().join("tokenizer.json"),
            b"{\"changed\":true}",
        )
        .unwrap();
        assert!(snapshot.check_unchanged().is_err());

        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let snapshot = Snapshot::open(directory.path()).unwrap();
        fs::write(directory.path().join("added.safetensors"), b"new").unwrap();
        assert!(snapshot.check_unchanged().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinks_and_replaced_files() {
        use std::os::unix::fs::symlink;

        for name in [SHARD, CONFIG, INDEX, "tokenizer.json"] {
            let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
            let path = directory.path().join(name);
            if path.exists() {
                fs::rename(&path, directory.path().join("original")).unwrap();
            } else {
                fs::write(directory.path().join("original"), b"{}").unwrap();
            }
            symlink("original", &path).unwrap();
            assert_rejected(directory.path(), "symlink");
            assert!(read_json(&path).is_err());
        }
        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let parent = tempdir().unwrap();
        let link = parent.path().join("source");
        symlink(directory.path(), &link).unwrap();
        assert_rejected(&link, "symlink");

        let directory = fixture(SIMPLE_HEADER, &SIMPLE_BYTES);
        let snapshot = Snapshot::open(directory.path()).unwrap();
        write_shard(
            directory.path(),
            "replacement",
            SIMPLE_HEADER,
            &SIMPLE_BYTES,
        );
        fs::rename(
            directory.path().join("replacement"),
            directory.path().join(SHARD),
        )
        .unwrap();
        assert!(snapshot.read_bytes("x", 0, 4).is_err());
        assert!(snapshot.check_unchanged().is_err());
    }
}
