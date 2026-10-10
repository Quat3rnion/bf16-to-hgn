//! Canonical, bounded reads of native HGN weights and their companion files.
//!
//! The importer deliberately does not use the canonical-output `container::Reader`:
//! append-only transplants leave unreachable old payloads and move live entries out
//! of table order. Only live spans must be disjoint. Checksums and SHA-256 cover
//! their physical payloads; canonical tensor hashes cover decoded original-basis
//! F32 (or exact I64) bytes. No decoded model or full expert is materialized.

use crate::codec::{self, EncodedBlock, Encoding, PayloadLayout, QuantMeta};
use crate::container::{self, Fingerprint, ENTRY_SIZE, HEADER_SIZE};
use crate::hf::{self, Dtype, TensorInfo};
use crate::source::{SourceKind, SourceOptions, TensorSource};
use anyhow::{bail, ensure, Context, Result};
use half::f16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const WINDOW_BYTES: usize = 8 * 1024 * 1024;
// Every Flash-Next row fits this bound, including the 10240-wide HC matrices.
// A rotated read therefore needs at most 8 MiB of decoded band scratch.
const MAX_COLUMNS: u64 = 16384;
const NGRAM: &str = "layers.1.ple.ngram_embedding.weight";
const NGRAM_PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";
const REQUIRED_ASSETS: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "generation_config.json",
];
const OPTIONAL_ASSETS: &[&str] = &[
    "vocab.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Signature {
    size: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed: (i64, i64),
}

impl Signature {
    fn metadata(metadata: &Metadata) -> Result<Self> {
        Ok(Self {
            size: metadata.len(),
            modified: metadata.modified()?,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    fn file(path: &Path) -> Result<Self> {
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("reading source metadata {}", path.display()))?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "source must be a regular file, not a symlink: {}",
            path.display()
        );
        Self::metadata(&metadata)
    }

    fn check(&self, path: &Path) -> Result<()> {
        ensure!(
            Self::file(path)? == *self,
            "source file changed: {}",
            path.display()
        );
        Ok(())
    }

    fn open(&self, path: &Path) -> Result<File> {
        self.check(path)?;
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        let file = options.open(path)?;
        self.check_open(&file, path)?;
        Ok(file)
    }

    fn check_open(&self, file: &File, path: &Path) -> Result<()> {
        let metadata = file.metadata()?;
        ensure!(
            metadata.is_file() && Self::metadata(&metadata)? == *self,
            "open source file changed: {}",
            path.display()
        );
        self.check(path)
    }
}

#[derive(Debug)]
struct PhysicalTensor {
    name: String,
    encoding: Encoding,
    shape: Vec<u64>,
    rows: u64,
    cols: usize,
    layout: PayloadLayout,
    offset: u64,
    size: u64,
    checksum: u32,
}

#[derive(Debug, Clone)]
struct Rotation {
    su: usize,
    sv: usize,
    su_per_expert: bool,
    sv_per_expert: bool,
    paired: bool,
}

#[derive(Debug)]
struct View {
    physical: usize,
    element_start: u64,
    rotation: Option<Rotation>,
}

/// Reusable HGN sidecar/container decoder. It validates physical storage, not a
/// complete model architecture. `HgnSource` and the architecture planner enforce
/// the public source contract; this helper also permits non-engine test fixtures.
#[derive(Debug)]
pub struct HgnSidecar {
    path: PathBuf,
    signature: Signature,
    identity: String,
    version: u32,
    header_table_sha256: String,
    physical: Vec<PhysicalTensor>,
    tensors: BTreeMap<String, TensorInfo>,
    views: BTreeMap<String, View>,
    fingerprints: Mutex<BTreeMap<usize, Fingerprint>>,
    null_config: Value,
}

impl HgnSidecar {
    /// Only metadata is read here; ordinary reads remain bounded row probes.
    /// `hash_tensor` checksums/hashes the tensor's physical dependencies, and
    /// `provenance` validates every dependency before publication. Results are
    /// cached only while the immutable file signature is unchanged.
    pub fn open(path: &Path) -> Result<Self> {
        let signature = Signature::file(path)?;
        let path = path.canonicalize()?;
        let mut file = signature.open(&path)?;
        ensure!(signature.size >= HEADER_SIZE, "truncated HGN header");
        let mut header = [0u8; HEADER_SIZE as usize];
        file.read_exact(&mut header)?;
        ensure!(u32le(&header, 0) == 0x314e4748, "invalid HGN magic");
        let version = u32le(&header, 4);
        ensure!(
            version == 1 || version == 2,
            "unsupported HGN version {version}"
        );
        let count = u64le(&header, 8);
        ensure!((1..=1 << 20).contains(&count), "invalid HGN tensor count");
        let table_start = u64le(&header, 16);
        let table_end = add(table_start, mul(count, ENTRY_SIZE)?)?;
        let data_offset = u64le(&header, 24);
        ensure!(
            u64le(&header, 32) == signature.size,
            "HGN file size mismatch"
        );
        ensure!(
            table_start >= HEADER_SIZE && table_end <= signature.size,
            "HGN table exceeds file or overlaps header"
        );
        ensure!(
            data_offset >= table_end && data_offset <= signature.size && data_offset % 64 == 0,
            "invalid HGN data offset"
        );
        let identity = ascii(&header[40..104], "identity")?;
        file.seek(SeekFrom::Start(table_start))?;
        let mut table_hash = Sha256::new();
        table_hash.update(header);
        let mut physical = Vec::with_capacity(usize::try_from(count)?);
        let mut names = BTreeMap::new();
        for index in 0..usize::try_from(count)? {
            let mut entry = [0u8; ENTRY_SIZE as usize];
            file.read_exact(&mut entry)?;
            table_hash.update(entry);
            let name = ascii(&entry[..96], "tensor name")?;
            ensure!(
                names.insert(name.clone(), index).is_none(),
                "duplicate HGN tensor {name}"
            );
            let rank = u32le(&entry, 100) as usize;
            ensure!((1..=4).contains(&rank), "invalid HGN rank for {name}");
            let mut shape = Vec::with_capacity(rank);
            for dimension in 0..4 {
                let value = i64::from_le_bytes(
                    entry[104 + 8 * dimension..112 + 8 * dimension]
                        .try_into()
                        .expect("dimension bytes"),
                );
                if dimension < rank {
                    ensure!(value > 0, "invalid HGN dimension for {name}");
                    shape.push(value as u64);
                } else {
                    ensure!(value == 0, "nonzero unused HGN shape slot for {name}");
                }
            }
            let encoding = encoding(u32le(&entry, 96), u32le(&entry, 156))?;
            let cols_u64 = *shape.last().expect("positive rank");
            // Rotation vectors may be flattened large arrays; they are streamed
            // as scalar elements rather than allocated as matrix rows.
            let auxiliary = is_auxiliary(&name);
            ensure!(
                auxiliary || cols_u64 <= MAX_COLUMNS,
                "unsupported HGN row width {cols_u64} for {name}"
            );
            let cols = usize::try_from(cols_u64)?;
            let rows = product(&shape[..rank - 1])?;
            let layout = codec::payload_layout(encoding, usize::try_from(rows)?, cols)
                .with_context(|| format!("invalid HGN storage for {name}"))?;
            let offset = u64le(&entry, 136);
            let size = u64le(&entry, 144);
            ensure!(
                size == layout.len && size > 0,
                "HGN payload size mismatch for {name}"
            );
            ensure!(
                offset >= data_offset && offset % 64 == 0,
                "HGN payload overlaps metadata or is unaligned: {name}"
            );
            ensure!(
                add(offset, size)? <= signature.size,
                "HGN payload exceeds file: {name}"
            );
            if encoding.ht_bits().is_some() || encoding == Encoding::I4r {
                ensure!(
                    (rank == 2 || rank == 3) && shape[rank - 2] % 128 == 0,
                    "rotated HGN matrix has invalid shape: {name}"
                );
                if rank == 3 {
                    ensure!(
                        is_expert(&name),
                        "rotated rank-three tensor is not an expert: {name}"
                    );
                    if name.ends_with(".gate_up_proj.weight") {
                        ensure!(
                            shape[1] % 256 == 0,
                            "gate/up projection halves are not 128-aligned"
                        );
                    }
                } else {
                    ensure!(encoding != Encoding::I4r, "I4R requires routed experts");
                }
            }
            if encoding == Encoding::Fp8G128 {
                ensure!(
                    name == NGRAM && rank == 3,
                    "FP8G requires the stacked n-gram table"
                );
            }
            if auxiliary {
                ensure!(
                    encoding == Encoding::F16,
                    "rotation vector must use F16: {name}"
                );
            }
            physical.push(PhysicalTensor {
                name,
                encoding,
                shape,
                rows,
                cols,
                layout,
                offset,
                size,
                checksum: u32le(&entry, 152),
            });
        }
        let mut spans: Vec<_> = physical
            .iter()
            .map(|p| (p.offset, p.offset + p.size, &p.name))
            .collect();
        spans.sort_unstable_by_key(|span| span.0);
        for pair in spans.windows(2) {
            ensure!(
                pair[0].1 <= pair[1].0,
                "overlapping HGN payloads: {} and {}",
                pair[0].2,
                pair[1].2
            );
        }
        // Gaps may contain old append-only payloads; they are not padding and
        // must neither be required to be zero nor included in a tensor hash.
        signature.check_open(&file, &path)?;
        let mut result = Self {
            path,
            signature,
            identity,
            version,
            header_table_sha256: format!("{:x}", table_hash.finalize()),
            physical,
            tensors: BTreeMap::new(),
            views: BTreeMap::new(),
            fingerprints: Mutex::new(BTreeMap::new()),
            null_config: Value::Null,
        };
        result.bind(&names)?;
        Ok(result)
    }

    fn bind(&mut self, names: &BTreeMap<String, usize>) -> Result<()> {
        let mut used_auxiliary = BTreeSet::new();
        for index in 0..self.physical.len() {
            let tensor = &self.physical[index];
            if is_auxiliary(&tensor.name) {
                continue;
            }
            let rotation =
                if tensor.encoding.ht_bits().is_some() || tensor.encoding == Encoding::I4r {
                    let base = tensor
                        .name
                        .strip_suffix(".weight")
                        .context("rotated name needs .weight")?;
                    let experts = tensor.shape.len() == 3;
                    let gate_up = experts && base.ends_with(".gate_up_proj");
                    let mut su_name = if gate_up {
                        format!("{}.su", base.strip_suffix(".gate_up_proj").expect("suffix"))
                    } else {
                        format!("{base}.suh")
                    };
                    // This is the specified shared-expert input-vector alias, never
                    // a substitution of weights from another file.
                    if !names.contains_key(&su_name) {
                        for suffix in [".shared_expert.gate_proj", ".shared_expert.up_proj"] {
                            if let Some(prefix) = base.strip_suffix(suffix) {
                                su_name = format!("{prefix}.experts.su");
                            }
                        }
                    }
                    let sv_name = format!("{base}.svh");
                    let su = *names
                        .get(&su_name)
                        .with_context(|| format!("missing rotation vector {su_name}"))?;
                    let sv = *names
                        .get(&sv_name)
                        .with_context(|| format!("missing rotation vector {sv_name}"))?;
                    let per_expert = experts && tensor.encoding.ht_bits().is_some();
                    let expert_count = if per_expert { tensor.shape[0] } else { 1 };
                    let su_per_expert = per_expert && !gate_up;
                    let su_count = mul(
                        tensor.cols as u64,
                        if su_per_expert { expert_count } else { 1 },
                    )?;
                    let sv_count = mul(tensor.shape[tensor.shape.len() - 2], expert_count)?;
                    for (dependency, expected) in [(su, su_count), (sv, sv_count)] {
                        let vector = &self.physical[dependency];
                        ensure!(
                            vector.encoding == Encoding::F16 && product(&vector.shape)? == expected,
                            "rotation vector dimensions/store mismatch: {}",
                            vector.name
                        );
                        used_auxiliary.insert(dependency);
                    }
                    Some(Rotation {
                        su,
                        sv,
                        su_per_expert,
                        sv_per_expert: per_expert,
                        paired: gate_up && tensor.encoding.ht_bits().is_some(),
                    })
                } else {
                    None
                };
            if tensor.name == NGRAM {
                ensure!(
                    tensor.shape.len() == 3 && tensor.cols == 160,
                    "n-gram table must have shape [parts, rows, 160]"
                );
                let parts = tensor.shape[0];
                ensure!(parts <= 128, "unsupported n-gram partition count");
                let shard_shape = tensor.shape[1..].to_vec();
                let shard_elements = product(&shard_shape)?;
                for shard in 0..parts {
                    self.insert_view(
                        format!("{NGRAM_PREFIX}{shard}.weight"),
                        shard_shape.clone(),
                        index,
                        mul(shard, shard_elements)?,
                        None,
                    )?;
                }
            } else {
                let name = canonical_name(&tensor.name)?;
                let shape = if tensor.name == "visual.patch_embed.proj.weight" {
                    ensure!(
                        tensor.shape == [1152, 1536],
                        "invalid vision patch embedding shape"
                    );
                    vec![1152, 3, 2, 16, 16]
                } else {
                    tensor.shape.clone()
                };
                self.insert_view(name, shape, index, 0, rotation)?;
            }
        }
        for (index, tensor) in self.physical.iter().enumerate() {
            ensure!(
                !is_auxiliary(&tensor.name) || used_auxiliary.contains(&index),
                "unreferenced HGN rotation vector {}",
                tensor.name
            );
        }
        ensure!(!self.tensors.is_empty(), "HGN contains no model tensors");
        Ok(())
    }

    fn insert_view(
        &mut self,
        name: String,
        shape: Vec<u64>,
        physical: usize,
        element_start: u64,
        rotation: Option<Rotation>,
    ) -> Result<()> {
        let dtype = if self.physical[physical].encoding == Encoding::I64 {
            Dtype::I64
        } else {
            Dtype::F32
        };
        let info = TensorInfo {
            name: name.clone(),
            shard: basename(&self.path)?,
            dtype,
            size: mul(product(&shape)?, dtype.byte_width())?,
            shape,
            offset: 0,
        };
        ensure!(
            self.tensors.insert(name.clone(), info).is_none(),
            "duplicate canonical HGN tensor {name}"
        );
        self.views.insert(
            name,
            View {
                physical,
                element_start,
                rotation,
            },
        );
        Ok(())
    }

    fn integrity(&self, index: usize) -> Result<()> {
        self.signature.check(&self.path)?;
        let mut hashes = self
            .fingerprints
            .lock()
            .map_err(|_| anyhow::anyhow!("HGN hash lock poisoned"))?;
        if hashes.contains_key(&index) {
            return Ok(());
        }
        let tensor = &self.physical[index];
        let mut file = self.signature.open(&self.path)?;
        let value = container::fingerprint(&mut file, tensor.offset, tensor.size)?;
        self.signature.check_open(&file, &self.path)?;
        ensure!(
            value.xor32 == tensor.checksum,
            "HGN checksum mismatch: {}",
            tensor.name
        );
        hashes.insert(index, value);
        Ok(())
    }

    fn raw(&self, index: usize, offset: u64, len: usize) -> Result<Vec<u8>> {
        let tensor = &self.physical[index];
        ensure!(
            add(offset, len as u64)? <= tensor.size,
            "HGN read exceeds payload {}",
            tensor.name
        );
        let mut file = self.signature.open(&self.path)?;
        let position = add(tensor.offset, offset)?;
        file.seek(SeekFrom::Start(position))?;
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes)?;
        self.signature.check_open(&file, &self.path)?;
        container::advise_drop(&file, position, len as u64);
        Ok(bytes)
    }

    fn vector(&self, index: usize, start: u64, len: usize) -> Result<Vec<f32>> {
        let bytes = self.raw(
            index,
            mul(start, 2)?,
            len.checked_mul(2).context("vector size overflow")?,
        )?;
        let values: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite HGN rotation vector"
        );
        Ok(values)
    }

    fn band(&self, view: &View, row: u64, rows: usize) -> Result<Vec<f32>> {
        let tensor = &self.physical[view.physical];
        let cols = tensor.cols;
        ensure!(
            add(row, rows as u64)? <= tensor.rows,
            "HGN band exceeds tensor"
        );
        let mut codebook = [0f32; 16];
        let mut meta = QuantMeta::default();
        let su;
        let sv;
        if let Some(rotation) = &view.rotation {
            ensure!(
                rows == 128 && row % 128 == 0,
                "rotation needs a full aligned band"
            );
            let matrix_rows = tensor.shape[tensor.shape.len() - 2];
            let expert = row / matrix_rows;
            let local_row = row % matrix_rows;
            su = self.vector(
                rotation.su,
                if rotation.su_per_expert {
                    mul(expert, cols as u64)?
                } else {
                    0
                },
                cols,
            )?;
            sv = self.vector(
                rotation.sv,
                if rotation.sv_per_expert {
                    row
                } else {
                    local_row
                },
                rows,
            )?;
            meta.su = &su;
            meta.sv = &sv;
            if rotation.paired {
                let half = matrix_rows / 2;
                let up = local_row >= half;
                let pair = (local_row % half) / 128;
                let bits = tensor.encoding.ht_bits().context("paired HT encoding")? as u64;
                let expert_bytes = mul(mul(matrix_rows, cols as u64)?, bits)? / 8;
                let pair_bytes = mul(mul(256, cols as u64)?, bits)? / 8;
                let start = add(mul(expert, expert_bytes)?, mul(pair, pair_bytes)?)?;
                let bytes = self.raw(view.physical, start, usize::try_from(pair_bytes)?)?;
                return codec::decode_ht_gate_up_band(tensor.encoding, &bytes, cols, up, &meta);
            }
        }
        let values = mul(rows as u64, cols as u64)?;
        let (code_start, code_len, scales) = match tensor.encoding {
            Encoding::Q4Cp | Encoding::I4r => {
                if tensor.encoding == Encoding::Q4Cp {
                    let bytes = self.raw(view.physical, 0, 64)?;
                    for (value, bytes) in codebook.iter_mut().zip(bytes.chunks_exact(4)) {
                        *value = f32::from_le_bytes(bytes.try_into().expect("codebook value"));
                    }
                    meta.codebook = &codebook;
                    let code_end = add(
                        tensor.layout.codes_offset,
                        mul(tensor.rows, cols as u64)? / 2,
                    )?;
                    self.zero_range(
                        view.physical,
                        code_end,
                        tensor.layout.scales_offset - code_end,
                    )?;
                }
                let scale_start = add(
                    tensor.layout.scales_offset,
                    mul(row, tensor.layout.scale_row_stride)?,
                )?;
                let scales = self.raw(
                    view.physical,
                    scale_start,
                    usize::try_from(mul(rows as u64, tensor.layout.scale_row_stride)?)?,
                )?;
                (
                    add(tensor.layout.codes_offset, mul(row, cols as u64)? / 2)?,
                    values / 2,
                    scales,
                )
            }
            Encoding::Fp8G128 => {
                let bytes = self.raw(view.physical, tensor.layout.scales_offset, 4)?;
                meta.fp8_scale = f32::from_le_bytes(bytes.try_into().expect("FP8 scale"));
                let code_end = mul(tensor.rows, cols as u64)?;
                self.zero_range(
                    view.physical,
                    code_end,
                    tensor.layout.scales_offset - code_end,
                )?;
                (mul(row, cols as u64)?, values, Vec::new())
            }
            _ => {
                let row_bytes = tensor.size / tensor.rows;
                (
                    mul(row, row_bytes)?,
                    mul(rows as u64, row_bytes)?,
                    Vec::new(),
                )
            }
        };
        let block = EncodedBlock {
            codes: self.raw(view.physical, code_start, usize::try_from(code_len)?)?,
            scales,
        };
        codec::decode_rows(tensor.encoding, &block, rows, cols, &meta)
    }

    fn zero_range(&self, index: usize, offset: u64, len: u64) -> Result<()> {
        ensure!(len < 64, "invalid HGN plane padding");
        if len != 0 {
            ensure!(
                self.raw(index, offset, usize::try_from(len)?)?
                    .iter()
                    .all(|b| *b == 0),
                "nonzero HGN plane padding"
            );
        }
        Ok(())
    }
}

impl TensorSource for HgnSidecar {
    fn root(&self) -> &Path {
        &self.path
    }
    fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    fn config(&self) -> &Value {
        &self.null_config
    }
    fn kind(&self) -> SourceKind {
        SourceKind::Hgn
    }
    fn asset_path(&self, name: &str) -> Result<PathBuf> {
        bail!("HGN sidecar has no asset {name}")
    }

    fn read_bytes(&self, name: &str, byte_offset: u64, len: usize) -> Result<Vec<u8>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown HGN tensor {name}"))?;
        ensure!(
            add(byte_offset, len as u64)? <= info.size,
            "canonical HGN byte range exceeds tensor"
        );
        self.signature.check(&self.path)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let view = &self.views[name];
        if info.dtype == Dtype::I64 {
            return self.raw(
                view.physical,
                add(mul(view.element_start, 8)?, byte_offset)?,
                len,
            );
        }
        let first = byte_offset / 4;
        let end = add(add(byte_offset, len as u64)?, 3)? / 4;
        let values = self.read_f32(name, first, usize::try_from(end - first)?)?;
        let skip = (byte_offset % 4) as usize;
        let mut bytes = Vec::with_capacity(len);
        for (index, value) in values.into_iter().enumerate() {
            let data = value.to_le_bytes();
            let start = if index == 0 { skip } else { 0 };
            let take = (len - bytes.len()).min(4 - start);
            bytes.extend_from_slice(&data[start..start + take]);
        }
        Ok(bytes)
    }

    fn read_f32(&self, name: &str, element_offset: u64, len: usize) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown HGN tensor {name}"))?;
        ensure!(
            info.dtype != Dtype::I64,
            "I64 requires exact read_bytes, not read_f32"
        );
        ensure!(
            add(element_offset, len as u64)? <= info.size / 4,
            "canonical HGN element range exceeds tensor"
        );
        self.signature.check(&self.path)?;
        let view = &self.views[name];
        let tensor = &self.physical[view.physical];
        let cols = tensor.cols as u64;
        let mut current = add(view.element_start, element_offset)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(len)
            .context("allocating canonical HGN read")?;
        while result.len() < len {
            let row = current / cols;
            let band_row = if view.rotation.is_some() {
                row / 128 * 128
            } else {
                row
            };
            let rows = if view.rotation.is_some() {
                128
            } else {
                let needed = add(current % cols, (len - result.len()) as u64)?;
                let needed_rows = add(needed, cols - 1)? / cols;
                needed_rows
                    .min(tensor.rows - band_row)
                    .min((WINDOW_BYTES as u64 / 4 / cols).max(1)) as usize
            };
            let band = self.band(view, band_row, rows)?;
            let start = usize::try_from(current - mul(band_row, cols)?)?;
            let take = (len - result.len()).min(band.len() - start);
            result.extend_from_slice(&band[start..start + take]);
            current = add(current, take as u64)?;
        }
        Ok(result)
    }

    fn hash_tensor(&self, name: &str) -> Result<String> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown HGN tensor {name}"))?;
        let view = &self.views[name];
        self.integrity(view.physical)?;
        if let Some(rotation) = &view.rotation {
            self.integrity(rotation.su)?;
            self.integrity(rotation.sv)?;
        }
        let mut hash = Sha256::new();
        let mut offset = 0;
        while offset < info.size {
            let take = (info.size - offset).min(WINDOW_BYTES as u64) as usize;
            hash.update(self.read_bytes(name, offset, take)?);
            offset += take as u64;
        }
        self.check_unchanged()?;
        Ok(format!("{:x}", hash.finalize()))
    }

    fn check_unchanged(&self) -> Result<()> {
        self.signature.check(&self.path)
    }

    fn provenance(&self) -> Result<Value> {
        let mut payloads = Vec::with_capacity(self.physical.len());
        for (index, tensor) in self.physical.iter().enumerate() {
            self.integrity(index)?;
            let hashes = self
                .fingerprints
                .lock()
                .map_err(|_| anyhow::anyhow!("HGN hash lock poisoned"))?;
            let fingerprint = &hashes[&index];
            payloads.push(json!({"name": tensor.name, "store": tensor.encoding.code(),
                "variant": tensor.encoding.variant(), "shape": tensor.shape,
                "offset": tensor.offset, "size": tensor.size,
                "xor32": tensor.checksum, "sha256": fingerprint.sha256}));
        }
        self.check_unchanged()?;
        Ok(
            json!({"format": "hgn", "version": self.version, "identity": self.identity,
            "file": basename(&self.path)?, "size": self.signature.size,
            "header_table_sha256": self.header_table_sha256, "payloads": payloads,
            "canonical_dtype": "F32 except exact I64", "rotation": "inverse H128 on both axes"}),
        )
    }

    fn input_paths(&self) -> Vec<PathBuf> {
        vec![self.path.clone()]
    }
}

/// Complete HGN input assembly. No external weights are selected except explicit
/// paths or existing, standard-named siblings of the supplied checkpoint.
#[derive(Debug)]
pub struct HgnSource {
    root: PathBuf,
    parts: Vec<HgnSidecar>,
    tensors: BTreeMap<String, TensorInfo>,
    owners: BTreeMap<String, usize>,
    config: Value,
    assets_root: PathBuf,
    assets: BTreeMap<String, Signature>,
}

impl HgnSource {
    pub fn open(path: &Path, options: &SourceOptions) -> Result<Self> {
        let source_metadata = fs::symlink_metadata(path)?;
        ensure!(
            !source_metadata.file_type().is_symlink(),
            "HGN source must not be a symlink"
        );
        let root = path.canonicalize()?;
        let model_path = if source_metadata.is_dir() {
            directory_model(&root)?
        } else {
            root.clone()
        };
        let main = HgnSidecar::open(&model_path)?;
        ensure!(
            matches!(
                main.identity.as_str(),
                "qwen3.8-flash-next-v2" | "qwen3.8-flash-next-ht43"
            ),
            "unsupported HGN checkpoint identity {:?}; expected native v2 or HT43",
            main.identity
        );
        let parent = model_path.parent().context("HGN path has no parent")?;
        let assets_path = options.assets.as_deref().unwrap_or(parent);
        let metadata = fs::symlink_metadata(assets_path)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "--assets must be a real directory"
        );
        let assets_root = assets_path.canonicalize()?;
        let mut assets = BTreeMap::new();
        for &name in REQUIRED_ASSETS {
            let signature = Signature::file(&assets_root.join(name))
                .with_context(|| format!("missing required HGN asset {name}; supply --assets"))?;
            assets.insert(name.to_owned(), signature);
        }
        for &name in OPTIONAL_ASSETS {
            let file = assets_root.join(name);
            match fs::symlink_metadata(&file) {
                Ok(_) => {
                    assets.insert(name.to_owned(), Signature::file(&file)?);
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        let config = hf::read_json(&assets_root.join("config.json"))?;
        ensure!(config.is_object(), "config.json must be an object");
        let mut result = Self {
            root,
            parts: Vec::new(),
            tensors: BTreeMap::new(),
            owners: BTreeMap::new(),
            config,
            assets_root,
            assets,
        };
        result.attach(main)?;
        for (kind, supplied, filename, prefix) in [
            (
                "ngram",
                options.ngram.as_deref(),
                "qwen38-flash-next-ngram.hgn",
                NGRAM_PREFIX,
            ),
            (
                "vision",
                options.vision.as_deref(),
                "qwen38-flash-next-vision.hgn",
                "model.visual.",
            ),
            (
                "mtp",
                options.mtp.as_deref(),
                "qwen38-flash-next-mtp.hgn",
                "mtp.",
            ),
        ] {
            let present = result.tensors.keys().any(|name| name.starts_with(prefix));
            if supplied.is_none() && present {
                continue;
            }
            let sidecar_path = supplied
                .map(Path::to_path_buf)
                .unwrap_or_else(|| parent.join(filename));
            let part = HgnSidecar::open(&sidecar_path)
                .with_context(|| format!("missing/invalid {kind} weights; supply --{kind}"))?;
            ensure!(
                part.tensors.keys().all(|name| name.starts_with(prefix)),
                "{kind} sidecar contains tensors outside its role"
            );
            result.attach(part)?;
        }
        result.check_unchanged()?;
        Ok(result)
    }

    fn attach(&mut self, part: HgnSidecar) -> Result<()> {
        let index = self.parts.len();
        for (name, info) in part.tensors() {
            ensure!(
                !self.tensors.contains_key(name),
                "duplicate HGN tensor across inputs: {name}"
            );
            self.tensors.insert(name.clone(), info.clone());
            self.owners.insert(name.clone(), index);
        }
        self.parts.push(part);
        Ok(())
    }

    fn part(&self, name: &str) -> Result<&HgnSidecar> {
        let index = self
            .owners
            .get(name)
            .with_context(|| format!("unknown HGN tensor {name}"))?;
        Ok(&self.parts[*index])
    }

    fn validate_ngram_bounds(&self) -> Result<()> {
        let prefix = "model.language_model.layers.1.ple.ple_embedding.";
        let offsets_name = format!("{prefix}ngram_heads_offsets");
        let sizes_name = format!("{prefix}ngram_heads_vocab_sizes");
        let (Some(offsets), Some(sizes)) = (
            self.tensors.get(&offsets_name),
            self.tensors.get(&sizes_name),
        ) else {
            // The architecture planner reports incomplete PLE metadata.
            return Ok(());
        };
        ensure!(
            offsets.dtype == Dtype::I64
                && sizes.dtype == Dtype::I64
                && offsets.shape == [16]
                && sizes.shape == [16],
            "invalid n-gram head metadata"
        );
        let rows = self
            .tensors
            .iter()
            .filter(|(name, _)| name.starts_with(NGRAM_PREFIX))
            .try_fold(0u64, |total, (_, info)| add(total, info.shape[0]))?;
        let offsets = self.read_bytes(&offsets_name, 0, 128)?;
        let sizes = self.read_bytes(&sizes_name, 0, 128)?;
        for (offset, size) in offsets.chunks_exact(8).zip(sizes.chunks_exact(8)) {
            let offset = i64::from_le_bytes(offset.try_into().expect("I64 offset"));
            let size = i64::from_le_bytes(size.try_into().expect("I64 size"));
            ensure!(
                offset >= 0 && size > 0 && add(offset as u64, size as u64)? <= rows,
                "n-gram head exceeds supplied table"
            );
        }
        Ok(())
    }
}

impl TensorSource for HgnSource {
    fn root(&self) -> &Path {
        &self.root
    }
    fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    fn config(&self) -> &Value {
        &self.config
    }
    fn kind(&self) -> SourceKind {
        SourceKind::Hgn
    }
    fn asset_path(&self, name: &str) -> Result<PathBuf> {
        ensure!(
            REQUIRED_ASSETS.contains(&name) || OPTIONAL_ASSETS.contains(&name),
            "unknown/unsafe HGN asset {name}"
        );
        let path = self.assets_root.join(name);
        if let Some(signature) = self.assets.get(name) {
            signature.check(&path)?;
        } else {
            ensure!(
                matches!(fs::symlink_metadata(&path), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
                "source asset appeared after opening: {name}"
            );
        }
        Ok(path)
    }
    fn read_bytes(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        self.part(name)?.read_bytes(name, offset, len)
    }
    fn read_f32(&self, name: &str, offset: u64, len: usize) -> Result<Vec<f32>> {
        self.part(name)?.read_f32(name, offset, len)
    }
    fn hash_tensor(&self, name: &str) -> Result<String> {
        self.part(name)?.hash_tensor(name)
    }
    fn check_unchanged(&self) -> Result<()> {
        for part in &self.parts {
            part.check_unchanged()?;
        }
        for (name, signature) in &self.assets {
            signature.check(&self.assets_root.join(name))?;
        }
        for &name in OPTIONAL_ASSETS {
            self.asset_path(name)?;
        }
        Ok(())
    }
    fn provenance(&self) -> Result<Value> {
        self.validate_ngram_bounds()?;
        let parts = self
            .parts
            .iter()
            .map(TensorSource::provenance)
            .collect::<Result<Vec<_>>>()?;
        let mut assets = BTreeMap::new();
        for (name, signature) in &self.assets {
            let path = self.assets_root.join(name);
            let mut file = signature.open(&path)?;
            let hash = container::fingerprint(&mut file, 0, signature.size)?;
            signature.check_open(&file, &path)?;
            assets.insert(name, json!({"size": hash.size, "sha256": hash.sha256}));
        }
        self.check_unchanged()?;
        Ok(json!({"format": "hgn", "containers": parts, "assets": assets}))
    }
    fn input_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.root.clone(), self.assets_root.clone()];
        paths.extend(self.parts.iter().flat_map(TensorSource::input_paths));
        paths
    }
}

fn directory_model(root: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".hgn")
            || name.contains(".overlay")
            || matches!(
                name,
                "qwen38-flash-next-ngram.hgn"
                    | "qwen38-flash-next-vision.hgn"
                    | "qwen38-flash-next-mtp.hgn"
            )
        {
            continue;
        }
        candidates.push(path);
    }
    ensure!(
        candidates.len() == 1,
        "HGN directory must contain exactly one model; supply the model file explicitly"
    );
    Ok(candidates.remove(0))
}

fn encoding(code: u32, variant: u32) -> Result<Encoding> {
    let value = match (code, variant) {
        (0, 0) => Encoding::Bf16,
        (1, 0) => Encoding::F32,
        (2, 0) => Encoding::F16,
        (4, 0) => Encoding::I64,
        (5, 2) => Encoding::Q4Cp,
        (7, 0) => Encoding::Q8G64,
        (10, 0) => Encoding::Fp8G128,
        (16, 0x1206) => Encoding::Ht3,
        (16, 0x1208) => Encoding::Ht,
        (23, 128) => Encoding::I4r,
        (24, 64) => Encoding::Q6G64,
        _ => bail!("unsupported HGN store/variant {code}/{variant:#x}; supported stores: 0,1,2,4,5/2,7,10,16/0x1206,16/0x1208,23/128,24/64"),
    };
    Ok(value)
}

fn canonical_name(name: &str) -> Result<String> {
    let name = if is_expert(name) {
        name.strip_suffix(".weight").expect("expert suffix")
    } else {
        name
    };
    if name == "lm_head.weight" || name.starts_with("mtp.") {
        return Ok(name.to_owned());
    }
    if name.starts_with("visual.") {
        return Ok(format!("model.{name}"));
    }
    ensure!(
        name == "embed_tokens.weight"
            || name.starts_with("layers.")
            || name.starts_with("hyper_connection_mixer."),
        "unsupported HGN tensor name {name}"
    );
    Ok(format!("model.language_model.{name}"))
}
fn is_expert(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj.weight")
        || name.ends_with(".mlp.experts.down_proj.weight")
}
fn is_auxiliary(name: &str) -> bool {
    name.ends_with(".suh") || name.ends_with(".svh") || name.ends_with(".mlp.experts.su")
}
fn basename(path: &Path) -> Result<String> {
    Ok(path
        .file_name()
        .and_then(|name| name.to_str())
        .context("non-UTF-8 source filename")?
        .to_owned())
}
fn product(shape: &[u64]) -> Result<u64> {
    shape.iter().try_fold(1, |a, b| mul(a, *b))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).context("HGN size multiplication overflow")
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("HGN offset addition overflow")
}
fn u32le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}
fn u64le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}
fn ascii(bytes: &[u8], kind: &str) -> Result<String> {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .context("unterminated HGN text field")?;
    ensure!(
        end > 0
            && bytes[..end].iter().all(|byte| byte.is_ascii_graphic())
            && bytes[end..].iter().all(|byte| *byte == 0),
        "invalid HGN {kind}"
    );
    Ok(std::str::from_utf8(&bytes[..end])?.to_owned())
}

#[cfg(test)]
#[path = "hgn_source_tests.rs"]
mod tests;
