//! Resumable-conversion support: fingerprinted staging, an fsync'd JSONL
//! journal, and an exclusive run lock.
//!
//! With `--resume`, the staging directory name is deterministic:
//! `.bf16-to-hgn-resume-<16 hex>` in the destination's parent, where the hex
//! comes from a *job identity* digest (source kind, source file paths+sizes,
//! destination path). The journal's first line is a header carrying the full
//! *validity fingerprint* — everything that affects output bytes: tool/encoder
//! aggregate hash, per-file path+size+mtime stats of every physical input
//! (including sidecars and assets), the HF index digest where applicable,
//! profile, all backend choices, ht-beam, seed, threads, the planned output
//! file list, planned bytes, and a digest of the planned tensor table.
//!
//! Splitting identity from validity is deliberate: a rerun with changed
//! options must *find* the existing journal and hard-fail on the header
//! mismatch instead of silently starting a fresh staging directory elsewhere.
//!
//! The journal and lock files are siblings of the staging directory
//! (`<staging>.journal`, `<staging>.lock`), never inside it, so the published
//! tree is byte-identical to a non-resume run.
//!
//! Recovery rules (all documented behavior, no silent fallbacks):
//! - Missing staging directory with an existing journal (e.g. after a failed
//!   publication sync cleaned up): the journal is reset and the conversion
//!   starts over from scratch.
//! - Existing staging directory without a journal, or with a corrupt/torn
//!   header: unrecoverable; the user is told to remove the staging directory.
//! - Journal header fingerprint mismatch: hard error listing the differing
//!   fields; the user must remove the staging directory to convert with
//!   different options.
//! - Missing or wrong-sized staging file referenced by journal records:
//!   records from the first reference onward are dropped and the file is
//!   recreated (deterministic re-encode).
//! - Torn trailing line or unparsable record: the journal is truncated to the
//!   last valid record.
//! - Payload corruption: the last [`VERIFY_UNITS`] completed journal units
//!   (tensor/chunk records) are re-checksummed against actual staging bytes,
//!   oldest first; at the first mismatch the journal falls back (truncates) to
//!   before that unit and the affected work is redone. Units older than the
//!   verified suffix are trusted without re-reading.

use crate::{
    container,
    conversion::hash_file,
    manifest::{AssetRecord, Options},
    source::{SourceKind, TensorSource},
    ConversionPlan, SPEC_COMMIT, SPEC_VERSION,
};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

/// Bump when the journal record format changes incompatibly.
pub const JOURNAL_SCHEMA: u32 = 1;
pub const STAGING_PREFIX: &str = ".bf16-to-hgn-resume-";
/// Completed journal units re-checksummed against disk on resume.
pub const VERIFY_UNITS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceFileStat {
    /// Canonical absolute path of one physical input file.
    pub path: String,
    pub size: u64,
    /// Validity-only (never part of the staging-name identity digest).
    pub mtime_secs: u64,
    pub mtime_nanos: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub store: u32,
    pub variant: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedFile {
    pub filename: String,
    pub identity: String,
    pub bytes: u64,
    pub tensors: Vec<PlannedTensor>,
}

/// Everything that affects output bytes; serialized as the journal header.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fingerprint {
    pub journal_schema: u32,
    pub tool_version: String,
    pub encoder_sha256: String,
    pub spec_version: String,
    pub spec_commit: String,
    pub source_kind: SourceKind,
    /// Sorted per-file path+size+mtime stats of all physical inputs.
    pub source_files: Vec<SourceFileStat>,
    pub index_sha256: Option<String>,
    pub options: Options,
    pub destination: String,
    pub planned_files: Vec<PlannedFile>,
    pub planned_bytes: u64,
    /// SHA-256 of the canonical JSON of `planned_files` (names/shapes/stores in order).
    pub plan_digest: String,
}

/// Computed fingerprint plus both digests derived from it.
pub struct FingerprintBuild {
    pub fingerprint: Fingerprint,
    /// Canonical JSON of `fingerprint` (journal header `fields`).
    pub fields: Value,
    /// SHA-256 over `fields`; stored in the journal header.
    pub header_digest: String,
    /// SHA-256 over the identity subset (kind, destination, path+size);
    /// its first 16 hex chars name the staging directory.
    pub name_digest: String,
}

/// Cheap metadata-only fingerprint: stat walks and small JSON digests, never
/// full payload rehashing.
pub fn fingerprint(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    options: &Options,
    destination: &Path,
) -> Result<FingerprintBuild> {
    let source_files = source_stats(snapshot)?;
    let index_sha256 = if snapshot.kind() == SourceKind::Hf {
        Some(hash_file(&snapshot.asset_path("model.safetensors.index.json")?)?.1)
    } else {
        None
    };
    let mut planned_files = Vec::with_capacity(plan.files.len());
    let mut planned_bytes = 0u64;
    for file in &plan.files {
        let layout = container::layout(file)?;
        planned_bytes = planned_bytes
            .checked_add(layout.file_size)
            .context("planned bytes overflow")?;
        planned_files.push(PlannedFile {
            filename: file.filename.clone(),
            identity: file.identity.clone(),
            bytes: layout.file_size,
            tensors: file
                .tensors
                .iter()
                .map(|tensor| PlannedTensor {
                    name: tensor.name.clone(),
                    shape: tensor.shape.clone(),
                    store: tensor.encoding.code(),
                    variant: tensor.encoding.variant(),
                })
                .collect(),
        });
    }
    let plan_digest = hex(&sha256_bytes(&serde_json::to_vec(&planned_files)?));
    let fingerprint = Fingerprint {
        journal_schema: JOURNAL_SCHEMA,
        tool_version: env!("CARGO_PKG_VERSION").into(),
        encoder_sha256: crate::ENCODER_SHA256.into(),
        spec_version: SPEC_VERSION.into(),
        spec_commit: SPEC_COMMIT.into(),
        source_kind: snapshot.kind(),
        source_files,
        index_sha256,
        options: options.clone(),
        destination: destination.display().to_string(),
        planned_files,
        planned_bytes,
        plan_digest,
    };
    // serde_json::Value maps are BTreeMap-keyed, so serialization is canonical.
    let fields = serde_json::to_value(&fingerprint)?;
    let header_digest = hex(&sha256_bytes(&serde_json::to_vec(&fields)?));
    let identity = serde_json::json!({
        "source_kind": fingerprint.source_kind,
        "destination": fingerprint.destination,
        "source_files": fingerprint
            .source_files
            .iter()
            .map(|file| serde_json::json!({ "path": file.path, "size": file.size }))
            .collect::<Vec<_>>(),
    });
    let name_digest = hex(&sha256_bytes(&serde_json::to_vec(&identity)?));
    Ok(FingerprintBuild {
        fingerprint,
        fields,
        header_digest,
        name_digest,
    })
}

/// Where the deterministic staging dir, journal, and lock live for a job.
pub fn staging_paths(parent: &Path, name_digest: &str) -> (PathBuf, PathBuf, PathBuf) {
    let stem = format!("{}{}", STAGING_PREFIX, &name_digest[..16]);
    (
        parent.join(&stem),
        parent.join(format!("{stem}.journal")),
        parent.join(format!("{stem}.lock")),
    )
}

fn source_stats(snapshot: &dyn TensorSource) -> Result<Vec<SourceFileStat>> {
    let mut roots = snapshot.input_paths();
    roots.push(snapshot.root().to_path_buf());
    let mut files: BTreeMap<String, SourceFileStat> = BTreeMap::new();
    for root in roots {
        let root = fs::canonicalize(&root)
            .with_context(|| format!("canonicalizing source input {}", root.display()))?;
        let metadata = fs::metadata(&root)?;
        if metadata.is_file() {
            insert_stat(&mut files, &root, &metadata)?;
        } else if metadata.is_dir() {
            walk_dir(&root, &mut files)?;
        } else {
            bail!("source input is neither file nor directory: {}", root.display());
        }
    }
    Ok(files.into_values().collect())
}

fn walk_dir(dir: &Path, files: &mut BTreeMap<String, SourceFileStat>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("walking source directory {}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(fs::DirEntry::path);
    for entry in entries {
        let path = entry.path();
        // Follow symlinks to their target stats; HF cache layouts are common.
        let metadata = fs::metadata(&path)
            .with_context(|| format!("stating source file {}", path.display()))?;
        if metadata.is_dir() {
            walk_dir(&path, files)?;
        } else if metadata.is_file() {
            let canonical = fs::canonicalize(&path)?;
            insert_stat(files, &canonical, &metadata)?;
        } else {
            bail!("source input is neither file nor directory: {}", path.display());
        }
    }
    Ok(())
}

fn insert_stat(
    files: &mut BTreeMap<String, SourceFileStat>,
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<()> {
    let modified = metadata
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    files.insert(
        path.display().to_string(),
        SourceFileStat {
            path: path.display().to_string(),
            size: metadata.len(),
            mtime_secs: modified.as_secs(),
            mtime_nanos: modified.subsec_nanos(),
        },
    );
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(bytes);
    hash.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// List the differing top-level fingerprint fields between two headers.
fn differing_fields(old: &Value, new: &Value) -> Vec<String> {
    let mut keys: Vec<&String> = Vec::new();
    if let (Some(old), Some(new)) = (old.as_object(), new.as_object()) {
        keys.extend(old.keys());
        for key in new.keys() {
            if !old.contains_key(key) {
                keys.push(key);
            }
        }
    }
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter(|key| old.get(*key) != new.get(*key))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------
// Journal records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileEntryRecord {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub checksum: u32,
    pub sha256: String,
    pub xor32: u32,
}

/// One JSONL line. `kind` discriminates; the header is always the first line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    Header {
        schema: u32,
        fingerprint: String,
        fields: Value,
    },
    /// One completed output tensor: absolute payload span in the staging file,
    /// its SHA-256, and the canonical source hashes consumed for it.
    Tensor {
        file: usize,
        tensor: usize,
        name: String,
        payload_offset: u64,
        payload_size: u64,
        payload_sha256: String,
        source_sha256: BTreeMap<String, String>,
    },
    /// One completed intra-tensor write unit (row batch / expert / copy chunk /
    /// table chunk). Spans are absolute staging-file offsets, hashed in order.
    Chunk {
        file: usize,
        tensor: usize,
        unit: usize,
        spans: Vec<(u64, u64)>,
        payload_sha256: String,
    },
    /// One fully finished (header/table-written) staging file.
    File {
        file: usize,
        filename: String,
        entries: Vec<FileEntryRecord>,
    },
    /// All assets copied into staging.
    Assets {
        records: Vec<AssetRecord>,
    },
}

impl Record {
    fn is_unit(&self) -> bool {
        matches!(self, Record::Tensor { .. } | Record::Chunk { .. })
    }
}

// ---------------------------------------------------------------------------
// Resume state derived from the validated journal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct CompletedTensor {
    pub name: String,
    pub source_sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PartialTensor {
    pub file: usize,
    pub tensor: usize,
    /// Count of leading chunk units already on disk and verified.
    pub units: usize,
}

#[derive(Debug, Default)]
pub struct ResumeState {
    /// Completed tensors by (file index, tensor index).
    pub completed_tensors: BTreeMap<(usize, usize), CompletedTensor>,
    /// Finished files by index (payload+header complete; finish() skippable).
    pub file_records: BTreeMap<usize, Vec<FileEntryRecord>>,
    /// Copied assets, when their staging files all still exist.
    pub assets: Option<Vec<AssetRecord>>,
    /// Trailing partially-written tensor with verified chunk units.
    pub partial: Option<PartialTensor>,
}

// ---------------------------------------------------------------------------
// Journal
// ---------------------------------------------------------------------------

pub struct Journal {
    file: File,
    pub path: PathBuf,
    units_appended: usize,
}

impl Journal {
    /// Open (validating/resuming) or initialize the journal for a staging dir.
    /// Creates the staging directory when absent. Returns the open journal
    /// (ready for appends) and the derived resume state.
    pub fn open_or_init(
        staging: &Path,
        path: &Path,
        build: &FingerprintBuild,
        plan: &ConversionPlan,
    ) -> Result<(Journal, ResumeState)> {
        match fs::symlink_metadata(staging) {
            Ok(metadata) => ensure!(metadata.is_dir(), "resume staging path is not a directory"),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let staging_existed = staging.exists();
        if !staging_existed {
            fs::create_dir_all(staging)
                .with_context(|| format!("creating resume staging {}", staging.display()))?;
            // Recovery rule: a journal without its staging directory (e.g. a
            // failed publication sync removed the renamed tree) is stale; the
            // conversion starts over with a fresh journal.
            if path.exists() {
                fs::remove_file(path).context("removing stale journal for missing staging dir")?;
            }
        }
        let layouts = plan
            .files
            .iter()
            .map(container::layout)
            .collect::<Result<Vec<_>>>()?;
        if !path.exists() {
            ensure!(
                !staging_existed || is_empty_dir(staging)?,
                "resume staging {} exists without a journal; remove it (and its .journal/.lock siblings) to start over",
                staging.display()
            );
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("creating journal {}", path.display()))?;
            let header = Record::Header {
                schema: JOURNAL_SCHEMA,
                fingerprint: build.header_digest.clone(),
                fields: build.fields.clone(),
            };
            write_line(&mut file, &header)?;
            file.sync_all()?;
            return Ok((
                Journal {
                    file,
                    path: path.to_path_buf(),
                    units_appended: 0,
                },
                ResumeState::default(),
            ));
        }
        let (records, valid_len) = read_records(path)?;
        let Some(Record::Header {
            schema,
            fingerprint,
            fields,
        }) = records.first().map(|(record, _)| record)
        else {
            bail!(
                "journal {} has no valid header line; remove the resume staging directory {} and its .journal/.lock siblings to start over",
                path.display(),
                staging.display()
            );
        };
        ensure!(
            *schema == JOURNAL_SCHEMA,
            "journal {} has unsupported schema {schema} (expected {JOURNAL_SCHEMA}); remove the resume staging directory {} to start over",
            path.display(),
            staging.display()
        );
        if *fingerprint != build.header_digest || *fields != build.fields {
            let changed = differing_fields(fields, &build.fields);
            bail!(
                "resume fingerprint mismatch: this journal was written for a different conversion (changed fields: {}); rerun with the original options, or remove {} and its .journal/.lock siblings to start over",
                if changed.is_empty() { "header digest".to_string() } else { changed.join(", ") },
                staging.display()
            );
        }
        // Validate records linearly; truncate at the first problem.
        let mut accepted: Vec<Record> = Vec::new();
        let mut offsets: Vec<u64> = Vec::new(); // end offset per accepted record
        let mut partial: Option<(usize, usize, usize)> = None; // file, tensor, next unit
        'scan: for (record, record_end) in records {
            match &record {
                Record::Header { .. } => {
                    if !accepted.is_empty() {
                        break 'scan; // duplicate header: torn state
                    }
                }
                Record::Tensor {
                    file,
                    tensor,
                    name,
                    payload_offset,
                    payload_size,
                    ..
                } => {
                    let Some(layout) = layouts.get(*file) else {
                        break 'scan;
                    };
                    let Some(entry) = layout.entries.get(*tensor) else {
                        break 'scan;
                    };
                    if entry.name != *name
                        || *payload_offset != entry.offset
                        || *payload_size != entry.size
                    {
                        break 'scan;
                    }
                    if !staging_file_ready(
                        staging,
                        Path::new(&plan.files[*file].filename),
                        layout.file_size,
                    )? {
                        break 'scan; // recovery rule: drop entries for missing files
                    }
                    partial = None;
                }
                Record::Chunk {
                    file,
                    tensor,
                    unit,
                    spans,
                    ..
                } => {
                    let Some(layout) = layouts.get(*file) else {
                        break 'scan;
                    };
                    let Some(entry) = layout.entries.get(*tensor) else {
                        break 'scan;
                    };
                    let expected = match partial {
                        Some((pf, pt, next)) if (pf, pt) == (*file, *tensor) => next,
                        Some(_) => break 'scan, // chunk for an unfinished other tensor
                        None => 0,
                    };
                    if *unit != expected || spans.is_empty() {
                        break 'scan;
                    }
                    if spans.iter().any(|&(off, size)| {
                        size == 0
                            || off < entry.offset
                            || off.checked_add(size).is_none_or(|end| end > entry.offset + entry.size)
                    }) {
                        break 'scan;
                    }
                    if !staging_file_ready(
                        staging,
                        Path::new(&plan.files[*file].filename),
                        layout.file_size,
                    )? {
                        break 'scan;
                    }
                    partial = Some((*file, *tensor, expected + 1));
                }
                Record::File {
                    file,
                    filename,
                    entries,
                } => {
                    let (Some(layout), Some(container)) = (layouts.get(*file), plan.files.get(*file))
                    else {
                        break 'scan;
                    };
                    if container.filename != *filename
                        || entries.len() != layout.entries.len()
                        || entries.iter().zip(&layout.entries).any(|(record, entry)| {
                            record.name != entry.name
                                || record.offset != entry.offset
                                || record.size != entry.size
                                || record.sha256.len() != 64
                        })
                    {
                        break 'scan;
                    }
                    if !staging_file_ready(
                        staging,
                        Path::new(&plan.files[*file].filename),
                        layout.file_size,
                    )? {
                        break 'scan;
                    }
                    partial = None;
                }
                Record::Assets { records } => {
                    if records.is_empty() {
                        break 'scan;
                    }
                    let complete = records.iter().all(|asset| {
                        fs::metadata(staging.join(&asset.destination))
                            .is_ok_and(|metadata| metadata.len() == asset.size)
                    });
                    if !complete {
                        break 'scan; // recovery rule: recreate missing asset files
                    }
                    partial = None;
                }
            }
            accepted.push(record);
            offsets.push(record_end);
        }
        // Re-checksum the last VERIFY_UNITS completed units, oldest first; the
        // first mismatch truncates the journal to before that unit (fallback,
        // never a silent pass).
        let unit_positions: Vec<usize> = accepted
            .iter()
            .enumerate()
            .filter(|(_, record)| record.is_unit())
            .map(|(index, _)| index)
            .collect();
        let start = unit_positions.len().saturating_sub(VERIFY_UNITS);
        let mut keep = accepted.len();
        for &index in &unit_positions[start..] {
            if !verify_unit(staging, plan, &accepted[index])? {
                keep = index;
                break;
            }
        }
        accepted.truncate(keep);
        let truncated_len = if keep == 0 { 0 } else { offsets[keep - 1] };
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("opening journal {}", path.display()))?;
        if truncated_len != valid_len {
            file.set_len(truncated_len)?;
            file.sync_all()?;
        }
        let mut state = ResumeState::default();
        for record in &accepted {
            match record {
                Record::Header { .. } => {}
                Record::Tensor {
                    file,
                    tensor,
                    name,
                    source_sha256,
                    ..
                } => {
                    state.completed_tensors.insert(
                        (*file, *tensor),
                        CompletedTensor {
                            name: name.clone(),
                            source_sha256: source_sha256.clone(),
                        },
                    );
                    state.partial = None;
                }
                Record::Chunk { file, tensor, unit, .. } => {
                    state.partial = Some(PartialTensor {
                        file: *file,
                        tensor: *tensor,
                        units: unit + 1,
                    });
                }
                Record::File { file, entries, .. } => {
                    state.file_records.insert(*file, entries.clone());
                    state.partial = None;
                }
                Record::Assets { records } => {
                    state.assets = Some(records.clone());
                    state.partial = None;
                }
            }
        }
        Ok((
            Journal {
                file,
                path: path.to_path_buf(),
                units_appended: 0,
            },
            state,
        ))
    }

    /// Append one record and fsync before returning (crash-safe granularity).
    pub fn append(&mut self, record: &Record) -> Result<()> {
        write_line(&mut self.file, record)?;
        self.file.sync_all()?;
        if record.is_unit() {
            self.units_appended += 1;
        }
        Ok(())
    }
}

fn is_empty_dir(dir: &Path) -> Result<bool> {
    Ok(fs::read_dir(dir)?.next().is_none())
}

/// The staging file must exist with exactly the planned preallocated size.
fn staging_file_ready(staging: &Path, filename: &Path, planned_size: u64) -> Result<bool> {
    match fs::metadata(staging.join(filename)) {
        Ok(metadata) => Ok(metadata.len() == planned_size),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn write_line(file: &mut File, record: &Record) -> Result<()> {
    let mut line = serde_json::to_vec(record)?;
    line.push(b'\n');
    file.write_all(&line)?;
    Ok(())
}

/// Read all complete (newline-terminated, parsable) records with their end
/// byte offsets, plus the byte length of that valid prefix. A torn final
/// line, an empty line, or a parse error ends the valid prefix.
fn read_records(path: &Path) -> Result<(Vec<(Record, u64)>, u64)> {
    let bytes = fs::read(path).with_context(|| format!("reading journal {}", path.display()))?;
    let mut records = Vec::new();
    let mut position = 0usize;
    while let Some(newline) = bytes[position..].iter().position(|&byte| byte == b'\n') {
        let line = &bytes[position..position + newline];
        position += newline + 1;
        if line.is_empty() {
            break; // journals never contain blank lines
        }
        match serde_json::from_slice::<Record>(line) {
            Ok(record) => records.push((record, position as u64)),
            Err(_) => break,
        }
    }
    let valid_len = records.last().map_or(0, |(_, end)| *end);
    Ok((records, valid_len))
}

/// Re-checksum one completed unit (tensor or chunk record) against the actual
/// staging bytes. Spans are absolute offsets inside the record's staging file.
fn verify_unit(staging: &Path, plan: &ConversionPlan, record: &Record) -> Result<bool> {
    let (file, spans, expected) = match record {
        Record::Tensor {
            file,
            payload_offset,
            payload_size,
            payload_sha256,
            ..
        } => (*file, vec![(*payload_offset, *payload_size)], payload_sha256),
        Record::Chunk {
            file,
            spans,
            payload_sha256,
            ..
        } => (*file, spans.clone(), payload_sha256),
        _ => return Ok(true),
    };
    let Some(container) = plan.files.get(file) else {
        return Ok(false);
    };
    let path = staging.join(&container.filename);
    match hash_spans_in(&path, &spans) {
        Ok(actual) => Ok(actual == *expected),
        // A short/missing file fails verification; the caller truncates the
        // journal back to before this unit.
        Err(_) => Ok(false),
    }
}

pub(crate) fn hash_spans_in(file: &Path, spans: &[(u64, u64)]) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut handle = File::open(file)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    for &(offset, size) in spans {
        handle.seek(SeekFrom::Start(offset))?;
        let mut remaining = size;
        while remaining > 0 {
            let take = usize::try_from(remaining.min(buffer.len() as u64))?;
            handle.read_exact(&mut buffer[..take])?;
            hash.update(&buffer[..take]);
            remaining -= take as u64;
        }
    }
    Ok(hex(&hash.finalize()))
}

/// Actual allocated bytes (st_blocks) under a directory tree; 0 when absent.
pub fn allocated_bytes(dir: &Path) -> Result<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        fn walk(dir: &Path, total: &mut u64) -> Result<()> {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_dir() {
                    walk(&entry.path(), total)?;
                } else {
                    *total = total
                        .checked_add(metadata.blocks().saturating_mul(512))
                        .context("allocated size overflow")?;
                }
            }
            Ok(())
        }
        match fs::metadata(dir) {
            Ok(metadata) if metadata.is_dir() => {
                let mut total = metadata.blocks().saturating_mul(512);
                walk(dir, &mut total)?;
                Ok(total)
            }
            Ok(_) => Ok(0),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(0),
            Err(error) => Err(error.into()),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(0)
    }
}

/// Remove a stale staging file whose size does not match the planned
/// preallocated size (its journal entries were already dropped).
pub fn clean_stale_staging(path: &Path, planned_size: u64) -> Result<()> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.len() != planned_size => {
            fs::remove_file(path)
                .with_context(|| format!("removing stale staging file {}", path.display()))?;
        }
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lock
// ---------------------------------------------------------------------------

/// Exclusive advisory lock on the staging sibling; held for the whole run.
pub struct LockGuard {
    file: File,
}

impl LockGuard {
    pub fn acquire(path: &Path) -> Result<LockGuard> {
        use std::os::unix::io::AsRawFd;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("creating resume lock {}", path.display()))?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if result != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::EWOULDBLOCK) {
                bail!(
                    "another conversion is already running for this resume staging (lock {} is held); wait for it or remove the stale lock",
                    path.display()
                );
            }
            return Err(error).with_context(|| format!("locking {}", path.display()));
        }
        Ok(LockGuard { file })
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}
