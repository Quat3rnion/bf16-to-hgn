use crate::{
    codec::{self, EncodedBlock, Encoding, PayloadLayout, QuantMeta},
    container::{self, Entry, Layout, Reader},
    conversion,
    hf::{self, Dtype, TensorInfo},
    manifest::{AssetRecord, ErrorTotals, Manifest, Options, SourceRecord, VerifyReport},
    numel, rows_cols,
    source::{SourceKind, TensorSource},
    ConversionPlan, Origin, TensorPlan, SPEC_COMMIT, SPEC_VERSION,
};
use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

const MANIFEST_NAME: &str = "conversion-manifest.json";
const BLOCK_ROWS: usize = 128;
const RAW_CHUNK: usize = 8 * 1024 * 1024;

/// Verify every source byte and every output payload without retaining matrices.
/// Error totals describe the decoded result; they are not a quality threshold.
pub fn verify(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    output: &Path,
) -> Result<VerifyReport> {
    snapshot.check_unchanged()?;
    let expected_sources = conversion::validate_plan(snapshot, plan)?;
    for tensor in plan.files.iter().flat_map(|file| &file.tensors) {
        if let Origin::Ngram { shards } = &tensor.origin {
            check_ngram_order(shards)?;
        }
    }
    let destinations = conversion::asset_destinations(plan)?;
    check_inventory(output, plan, &destinations)?;
    let manifest = read_manifest(&output.join(MANIFEST_NAME), plan)?;
    let sources = SourceCoverage::new(snapshot, &expected_sources, &manifest.sources)?;
    ensure!(
        manifest.containers.len() == plan.files.len(),
        "manifest container set mismatch"
    );

    ensure!(
        manifest.source_kind == snapshot.kind(),
        "manifest source format mismatch"
    );
    let (_, config_hash) = conversion::hash_file(&snapshot.asset_path("config.json")?)?;
    let index_hash = if snapshot.kind() == SourceKind::Hf {
        Some(conversion::hash_file(&snapshot.asset_path("model.safetensors.index.json")?)?.1)
    } else {
        None
    };
    ensure!(
        config_hash == manifest.config_sha256,
        "source config SHA-256 mismatch"
    );
    ensure!(
        index_hash == manifest.index_sha256,
        "source index SHA-256 mismatch"
    );
    verify_assets(snapshot, output, &destinations, &manifest.assets)?;
    conversion::validate_ple(snapshot, plan)?;

    let rotations = conversion::prepare_rotations(snapshot, plan, &manifest.options)?;
    let mut payloads = PayloadVerifier {
        snapshot,
        options: &manifest.options,
        rotations,
        sources,
    };
    let mut report = VerifyReport {
        source_tensors: 0,
        output_tensors: 0,
        payload_bytes: 0,
        errors: BTreeMap::new(),
    };
    for (file_plan, record) in plan.files.iter().zip(&manifest.containers) {
        ensure!(
            record.filename == file_plan.filename,
            "manifest container filename/order mismatch"
        );
        let mut reader = Reader::open(&output.join(&file_plan.filename))
            .with_context(|| format!("read container {}", file_plan.filename))?;
        ensure!(
            reader.version == 2,
            "converter output must use HGN version 2"
        );
        check_layout(&reader.layout, &container::layout(file_plan)?)?;
        ensure!(
            record.layout == reader.layout,
            "manifest container layout mismatch: {}",
            file_plan.filename
        );
        ensure!(
            record.tensors.len() == reader.layout.entries.len(),
            "manifest output tensor set mismatch"
        );

        for (index, (tensor, output_record)) in
            file_plan.tensors.iter().zip(&record.tensors).enumerate()
        {
            let entry = &reader.layout.entries[index];
            ensure!(
                output_record.entry == *entry,
                "manifest entry mismatch: {}",
                tensor.name
            );
            check_sha256(&output_record.fingerprint.sha256)?;
            ensure!(
                output_record.fingerprint.size == entry.size,
                "manifest payload size mismatch: {}",
                tensor.name
            );
            ensure!(
                output_record.fingerprint.xor32 == entry.checksum,
                "manifest payload checksum mismatch: {}",
                tensor.name
            );

            let errors = payloads
                .verify_tensor(&mut reader, index, tensor)
                .with_context(|| format!("verify {}:{}", file_plan.filename, tensor.name))?;
            let actual = reader.fingerprint(index)?;
            ensure!(
                actual == output_record.fingerprint,
                "payload SHA-256/XOR mismatch: {}:{}",
                file_plan.filename,
                tensor.name
            );
            ensure!(
                actual.xor32 == reader.layout.entries[index].checksum,
                "actual HGN checksum mismatch: {}",
                tensor.name
            );
            report.payload_bytes = report
                .payload_bytes
                .checked_add(actual.size)
                .context("verified payload size overflow")?;
            report.output_tensors = report
                .output_tensors
                .checked_add(1)
                .context("verified tensor count overflow")?;
            if let Some(errors) = errors {
                ensure!(
                    errors.values == numel(&tensor.shape)?,
                    "incomplete decoded coverage: {}",
                    tensor.name
                );
                let key = format!("{}:{}", file_plan.filename, tensor.name);
                ensure!(
                    report.errors.insert(key, errors).is_none(),
                    "duplicate error report key"
                );
            }
        }
    }
    report.source_tensors = payloads.sources.finish()?;
    ensure!(
        snapshot.provenance()? == manifest.source_provenance,
        "source physical provenance mismatch"
    );
    snapshot.check_unchanged()?;
    Ok(report)
}

fn read_manifest(path: &Path, plan: &ConversionPlan) -> Result<Manifest> {
    let json = hf::read_json(path)?;
    let manifest = Manifest::deserialize(&json).context("invalid conversion manifest")?;
    // Shared serde types also serve the writer. Reject fields those types would
    // otherwise silently ignore, including unknown fields in nested records.
    // Schema2 predates backend selection: omit only the newly defaulted field
    // when comparing an old manifest, without relaxing any other canonicality.
    let mut canonical = serde_json::to_value(&manifest)?;
    if manifest.schema == 2
        && json["options"]
            .as_object()
            .is_some_and(|options| !options.contains_key("ht_backend"))
    {
        canonical["options"]
            .as_object_mut()
            .expect("serialized Options is an object")
            .remove("ht_backend");
    }
    ensure!(canonical == json, "unknown or noncanonical manifest fields");
    ensure!(manifest.schema == 2, "unsupported manifest schema");
    ensure!(manifest.tool == "bf16-to-hgn", "unexpected manifest tool");
    ensure!(
        manifest.tool_version == env!("CARGO_PKG_VERSION"),
        "unsupported manifest tool version"
    );
    ensure!(
        manifest.spec_version == SPEC_VERSION && manifest.spec_commit == SPEC_COMMIT,
        "manifest specification mismatch"
    );
    ensure!(
        manifest.policy == manifest.options.profile.policy(),
        "manifest quantization policy mismatch"
    );
    // The encoder digest records provenance, not authorization. An unchanged
    // format/policy remains verifiable after rebuilding the converter.
    check_sha256(&manifest.encoder_sha256)?;
    check_sha256(&manifest.config_sha256)?;
    if let Some(hash) = &manifest.index_sha256 {
        check_sha256(hash)?;
    }
    ensure!(
        manifest.index_sha256.is_some() == (manifest.source_kind == SourceKind::Hf),
        "source index provenance does not match source format"
    );
    manifest.options.validate()?;
    ensure!(
        &manifest.plan == plan,
        "manifest plan differs from the fresh source plan"
    );
    Ok(manifest)
}

fn check_sha256(hash: &str) -> Result<()> {
    ensure!(
        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid SHA-256 hex digest"
    );
    Ok(())
}

fn check_layout(actual: &Layout, planned: &Layout) -> Result<()> {
    ensure!(
        actual.identity == planned.identity,
        "HGN identity differs from plan"
    );
    ensure!(
        actual.data_offset == planned.data_offset && actual.file_size == planned.file_size,
        "HGN file geometry differs from plan"
    );
    ensure!(
        actual.entries.len() == planned.entries.len(),
        "HGN tensor set differs from plan"
    );
    for (actual, planned) in actual.entries.iter().zip(&planned.entries) {
        ensure!(
            same_geometry(actual, planned),
            "HGN tensor geometry/order differs from plan: {}",
            planned.name
        );
    }
    Ok(())
}

fn same_geometry(actual: &Entry, planned: &Entry) -> bool {
    actual.name == planned.name
        && actual.encoding == planned.encoding
        && actual.shape == planned.shape
        && actual.offset == planned.offset
        && actual.size == planned.size
}

fn safe_relative(name: &str) -> Result<PathBuf> {
    ensure!(
        !name.is_empty()
            && !name.contains('\\')
            && name
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "unsafe output path"
    );
    let path = PathBuf::from(name);
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "unsafe output path"
    );
    Ok(path)
}

fn check_inventory(
    output: &Path,
    plan: &ConversionPlan,
    destinations: &[(String, String)],
) -> Result<()> {
    let metadata = fs::symlink_metadata(output)?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "output must be a nonsymlink directory"
    );
    let mut files = BTreeSet::from([PathBuf::from(MANIFEST_NAME)]);
    for name in plan.files.iter().map(|file| file.filename.as_str()).chain(
        destinations
            .iter()
            .map(|(_, destination)| destination.as_str()),
    ) {
        ensure!(
            files.insert(safe_relative(name)?),
            "duplicate planned output path: {name}"
        );
    }
    let mut directories = BTreeSet::new();
    for file in &files {
        let mut parent = file.parent();
        while let Some(path) = parent.filter(|path| !path.as_os_str().is_empty()) {
            directories.insert(path.to_owned());
            parent = path.parent();
        }
    }
    let mut pending = vec![PathBuf::new()];
    let mut seen_files = BTreeSet::new();
    let mut seen_directories = BTreeSet::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(output.join(&directory))? {
            let entry = entry?;
            let relative = directory.join(entry.file_name());
            let kind = entry.file_type()?;
            ensure!(
                !kind.is_symlink(),
                "symlink in output: {}",
                relative.display()
            );
            if kind.is_dir() {
                ensure!(
                    directories.contains(&relative),
                    "unexpected output directory: {}",
                    relative.display()
                );
                ensure!(
                    seen_directories.insert(relative.clone()),
                    "duplicate output directory"
                );
                pending.push(relative);
            } else {
                ensure!(
                    kind.is_file(),
                    "nonregular output file: {}",
                    relative.display()
                );
                ensure!(
                    files.contains(&relative),
                    "unexpected output file: {}",
                    relative.display()
                );
                ensure!(seen_files.insert(relative), "duplicate output file");
            }
        }
    }
    ensure!(
        seen_files == files && seen_directories == directories,
        "missing planned output files/directories"
    );
    Ok(())
}

fn verify_assets(
    snapshot: &dyn TensorSource,
    output: &Path,
    destinations: &[(String, String)],
    records: &[AssetRecord],
) -> Result<()> {
    ensure!(
        records.len() == destinations.len(),
        "manifest asset set mismatch"
    );
    let mut by_destination = BTreeMap::new();
    for record in records {
        check_sha256(&record.sha256)?;
        ensure!(
            by_destination
                .insert(record.destination.as_str(), record)
                .is_none(),
            "duplicate manifest asset destination"
        );
    }
    let mut source_hashes = BTreeMap::new();
    for (source, destination) in destinations {
        let record = by_destination
            .get(destination.as_str())
            .context("missing manifest asset destination")?;
        ensure!(
            record.source == *source,
            "manifest asset source mismatch: {destination}"
        );
        let source_digest = match source_hashes.entry(source.as_str()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(conversion::hash_file(&snapshot.asset_path(source)?)?)
            }
        };
        ensure!(
            source_digest.0 == record.size && source_digest.1 == record.sha256,
            "source asset SHA-256/size mismatch: {source}"
        );
        let (size, hash) = conversion::hash_file(&output.join(destination))?;
        ensure!(
            size == record.size && hash == record.sha256,
            "copied asset SHA-256/size mismatch: {destination}"
        );
    }
    Ok(())
}

struct SourceCoverage<'a> {
    records: BTreeMap<&'a str, &'a SourceRecord>,
    seen: BTreeSet<&'a str>,
}

impl<'a> SourceCoverage<'a> {
    fn new(
        snapshot: &dyn TensorSource,
        expected: &BTreeSet<String>,
        records: &'a [SourceRecord],
    ) -> Result<Self> {
        ensure!(
            records.len() == expected.len(),
            "manifest source tensor set mismatch"
        );
        let mut indexed = BTreeMap::new();
        for record in records {
            check_sha256(&record.sha256)?;
            let name = record.tensor.name.as_str();
            ensure!(
                expected.contains(name),
                "unexpected manifest source tensor: {name}"
            );
            let actual = snapshot
                .tensors()
                .get(name)
                .context("manifest source absent from snapshot")?;
            ensure!(
                &record.tensor == actual,
                "manifest source metadata mismatch: {name}"
            );
            ensure!(
                indexed.insert(name, record).is_none(),
                "duplicate manifest source tensor: {name}"
            );
        }
        ensure!(
            expected
                .iter()
                .all(|name| indexed.contains_key(name.as_str())),
            "missing manifest source tensor"
        );
        Ok(Self {
            records: indexed,
            seen: BTreeSet::new(),
        })
    }

    fn record(&mut self, info: &TensorInfo, hash: Sha256, bytes: u64) -> Result<()> {
        let record = self
            .records
            .get(info.name.as_str())
            .context("consumed unrecorded source tensor")?;
        ensure!(
            bytes == info.size,
            "incomplete source byte coverage: {}",
            info.name
        );
        let actual = format!("{:x}", hash.finalize());
        ensure!(
            actual == record.sha256,
            "source tensor SHA-256 mismatch: {}",
            info.name
        );
        ensure!(
            self.seen.insert(record.tensor.name.as_str()),
            "source tensor consumed more than once: {}",
            info.name
        );
        Ok(())
    }

    fn finish(self) -> Result<usize> {
        ensure!(
            self.seen.len() == self.records.len()
                && self.records.keys().all(|name| self.seen.contains(name)),
            "verification did not consume the exact source tensor set"
        );
        Ok(self.seen.len())
    }
}

struct PayloadVerifier<'a> {
    snapshot: &'a dyn TensorSource,
    options: &'a Options,
    rotations: BTreeMap<String, Vec<f32>>,
    sources: SourceCoverage<'a>,
}

impl PayloadVerifier<'_> {
    fn verify_tensor(
        &mut self,
        reader: &mut Reader,
        index: usize,
        tensor: &TensorPlan,
    ) -> Result<Option<ErrorTotals>> {
        match &tensor.origin {
            Origin::Signs { .. } | Origin::HtScales { .. } => {
                self.verify_vector(reader, index, tensor)?;
                Ok(None)
            }
            Origin::Tensor { name } => {
                let info = self
                    .snapshot
                    .tensors()
                    .get(name)
                    .context("missing planned source")?;
                let totals = if conversion::direct_copy(tensor.encoding, info.dtype) {
                    self.verify_direct(reader, index, info)?
                } else {
                    self.verify_matrix(reader, index, tensor, info)?
                };
                Ok(Some(totals))
            }
            Origin::Ngram { shards } => Ok(Some(self.verify_table(reader, index, tensor, shards)?)),
        }
    }

    fn verify_vector(&self, reader: &mut Reader, index: usize, tensor: &TensorPlan) -> Result<()> {
        ensure!(
            tensor.encoding == Encoding::F16,
            "generated vector is not F16"
        );
        let values = self
            .rotations
            .get(&tensor.name)
            .context("missing regenerated rotation vector")?;
        ensure!(
            values.len() as u64 == numel(&tensor.shape)?,
            "generated vector length mismatch"
        );
        let mut offset = 0u64;
        for values in values.chunks(RAW_CHUNK / 2) {
            let bytes = reader.read_at(index, offset, values.len() * 2)?;
            for (actual, &value) in bytes.chunks_exact(2).zip(values) {
                let expected = half::f16::from_f32(value);
                ensure!(
                    value.is_finite() && expected.is_finite(),
                    "nonfinite generated vector value"
                );
                ensure!(
                    actual == expected.to_bits().to_le_bytes(),
                    "generated vector differs from deterministic source policy"
                );
            }
            offset += bytes.len() as u64;
        }
        ensure!(
            offset == reader.layout.entries[index].size,
            "incomplete generated vector coverage"
        );
        Ok(())
    }

    fn verify_direct(
        &mut self,
        reader: &mut Reader,
        index: usize,
        info: &TensorInfo,
    ) -> Result<ErrorTotals> {
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        let mut totals = ErrorTotals::default();
        while offset < info.size {
            let take = (info.size - offset).min(RAW_CHUNK as u64) as usize;
            let raw = self.snapshot.read_bytes(&info.name, offset, take)?;
            let actual = reader.read_at(index, offset, take)?;
            ensure!(
                raw == actual,
                "direct payload differs from source bytes: {}",
                info.name
            );
            accumulate_exact(&mut totals, &raw, info.dtype)?;
            hash.update(&raw);
            offset = offset
                .checked_add(raw.len() as u64)
                .context("source coverage overflow")?;
        }
        self.sources.record(info, hash, offset)?;
        Ok(totals)
    }

    fn verify_matrix(
        &mut self,
        reader: &mut Reader,
        index: usize,
        tensor: &TensorPlan,
        info: &TensorInfo,
    ) -> Result<ErrorTotals> {
        if tensor.encoding.ht_bits().is_some()
            && tensor.shape.len() == 3
            && tensor.name.ends_with(".gate_up_proj.weight")
        {
            return self.verify_gate_up(reader, index, tensor, info);
        }
        let (rows, cols) = rows_cols(&tensor.shape)?;
        let layout = codec::payload_layout(tensor.encoding, rows, cols)?;
        if tensor.encoding != Encoding::Bf16 {
            check_quant_metadata(reader, index, tensor.encoding, rows, cols, &layout)?;
        }
        let mut hash = Sha256::new();
        let mut consumed = 0u64;
        let mut totals = ErrorTotals::default();
        let mut row = 0;
        while row < rows {
            let take = BLOCK_ROWS.min(rows - row);
            let elements = take
                .checked_mul(cols)
                .context("decode block size overflow")?;
            let (raw, source) = conversion::read_source_values(
                self.snapshot,
                info,
                (row as u64) * (cols as u64),
                elements,
            )?;
            hash.update(&raw);
            consumed = consumed
                .checked_add(raw.len() as u64)
                .context("source coverage overflow")?;
            let block = read_block(reader, index, tensor.encoding, &layout, row, take, cols)?;
            let (su, sv) = verification_rotation_slice(tensor, &self.rotations, row, take)?;
            let meta = QuantMeta {
                su,
                sv,
                threads: self.options.threads,
                ht_beam: self.options.ht_beam,
                ..QuantMeta::default()
            };
            let decoded = if let Some(bits) = tensor.encoding.ht_bits() {
                decode_ht_independent(&block.codes, bits, cols, None, su, sv)?
            } else {
                codec::decode_rows(tensor.encoding, &block, take, cols, &meta)?
            };
            if tensor.encoding == Encoding::Bf16 {
                for (&value, bytes) in source.iter().zip(block.codes.chunks_exact(2)) {
                    ensure!(
                        bytes == half::bf16::from_f32(value).to_le_bytes(),
                        "BF16 output differs from rounded canonical source"
                    );
                }
            }
            accumulate_errors(&mut totals, &source, &decoded)?;
            row += take;
        }
        self.sources.record(info, hash, consumed)?;
        Ok(totals)
    }

    fn verify_gate_up(
        &mut self,
        reader: &mut Reader,
        index: usize,
        tensor: &TensorPlan,
        info: &TensorInfo,
    ) -> Result<ErrorTotals> {
        let bits = tensor
            .encoding
            .ht_bits()
            .context("paired experts require HT")?;
        let [experts, expert_rows, cols]: [u64; 3] = tensor.shape.as_slice().try_into()?;
        let (experts, expert_rows, cols) = (
            usize::try_from(experts)?,
            usize::try_from(expert_rows)?,
            usize::try_from(cols)?,
        );
        ensure!(expert_rows % 256 == 0, "invalid paired expert rows");
        let projection_rows = expert_rows / 2;
        let paired_bytes = 256usize
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(bits))
            .context("paired verification size overflow")?
            / 8;
        let buffer_bytes = projection_rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(info.dtype.byte_width() as usize))
            .context("expert source buffer overflow")?;
        let mut pending_up = Vec::with_capacity(buffer_bytes);
        let mut totals = ErrorTotals::default();
        let mut hash = Sha256::new();
        let mut consumed = 0u64;
        for expert in 0..experts {
            pending_up.clear();
            for band in 0..projection_rows / 128 {
                let offset = (expert * (projection_rows / 128) + band) as u64 * paired_bytes as u64;
                let bytes = reader.read_at(index, offset, paired_bytes)?;
                for is_up in [false, true] {
                    let row =
                        expert * expert_rows + band * 128 + if is_up { projection_rows } else { 0 };
                    let (raw, source) = conversion::read_source_values(
                        self.snapshot,
                        info,
                        (row * cols) as u64,
                        128 * cols,
                    )?;
                    if is_up {
                        pending_up.extend_from_slice(&raw);
                    } else {
                        hash.update(&raw);
                    }
                    consumed = consumed
                        .checked_add(raw.len() as u64)
                        .context("source coverage overflow")?;
                    let (su, sv) = verification_rotation_slice(tensor, &self.rotations, row, 128)?;
                    let decoded = decode_ht_independent(&bytes, bits, cols, Some(is_up), su, sv)?;
                    accumulate_errors(&mut totals, &source, &decoded)?;
                }
            }
            hash.update(&pending_up);
        }
        self.sources.record(info, hash, consumed)?;
        Ok(totals)
    }

    fn verify_table(
        &mut self,
        reader: &mut Reader,
        index: usize,
        tensor: &TensorPlan,
        shards: &[String],
    ) -> Result<ErrorTotals> {
        let (rows, cols) = rows_cols(&tensor.shape)?;
        let layout = codec::payload_layout(tensor.encoding, rows, cols)?;
        let scale = check_quant_metadata(reader, index, tensor.encoding, rows, cols, &layout)?;
        let meta = QuantMeta {
            fp8_scale: scale,
            threads: self.options.threads,
            ht_beam: self.options.ht_beam,
            ..QuantMeta::default()
        };
        let mut totals = ErrorTotals::default();
        let mut absmax = 0.0f32;
        let mut output_row = 0usize;
        for name in shards {
            let info = self
                .snapshot
                .tensors()
                .get(name)
                .context("missing n-gram source")?;
            let (shard_rows, shard_cols) = rows_cols(&info.shape)?;
            ensure!(shard_cols == cols, "n-gram column count mismatch");
            let mut hash = Sha256::new();
            let mut consumed = 0u64;
            let mut row = 0;
            while row < shard_rows {
                let take = BLOCK_ROWS.min(shard_rows - row);
                let elements = take
                    .checked_mul(cols)
                    .context("table block size overflow")?;
                let (raw, source) = conversion::read_source_values(
                    self.snapshot,
                    info,
                    (row as u64) * (cols as u64),
                    elements,
                )?;
                hash.update(&raw);
                consumed = consumed
                    .checked_add(raw.len() as u64)
                    .context("source coverage overflow")?;
                for &value in &source {
                    ensure!(value.is_finite(), "nonfinite table source value");
                    absmax = absmax.max(value.abs());
                }
                let block = read_block(
                    reader,
                    index,
                    tensor.encoding,
                    &layout,
                    output_row,
                    take,
                    cols,
                )?;
                ensure!(
                    block.codes.iter().all(|code| code & 0x7f <= 0x7e),
                    "FP8 code exceeds the encoder's 448 magnitude policy"
                );
                let decoded = codec::decode_rows(tensor.encoding, &block, take, cols, &meta)?;
                accumulate_errors(&mut totals, &source, &decoded)?;
                row += take;
                output_row = output_row
                    .checked_add(take)
                    .context("table coverage overflow")?;
            }
            self.sources.record(info, hash, consumed)?;
        }
        ensure!(output_row == rows, "incomplete n-gram output coverage");
        let expected_scale = if absmax == 0.0 { 1.0 } else { absmax / 448.0 };
        ensure!(
            expected_scale.is_finite()
                && expected_scale > 0.0
                && scale.to_bits() == expected_scale.to_bits(),
            "FP8 global scale differs from source absmax policy"
        );
        Ok(totals)
    }
}

// Deliberately separate from the encoder's rotation slicing: a per-expert down
// input vector and a full per-expert output vector must not silently restart.
fn verification_rotation_slice<'a>(
    tensor: &TensorPlan,
    vectors: &'a BTreeMap<String, Vec<f32>>,
    row: usize,
    take: usize,
) -> Result<(&'a [f32], &'a [f32])> {
    let Some(rotation) = &tensor.rotation else {
        return Ok((&[], &[]));
    };
    let (rows, cols) = rows_cols(&tensor.shape)?;
    let end = row.checked_add(take).context("verification row overflow")?;
    ensure!(take > 0 && end <= rows, "verification band exceeds matrix");
    let expert_rows = if tensor.shape.len() == 3 {
        usize::try_from(tensor.shape[1])?
    } else {
        rows
    };
    ensure!(
        row / expert_rows == (end - 1) / expert_rows,
        "verification band crosses expert"
    );
    let input = vectors
        .get(&rotation.su)
        .context("missing verification input vector")?;
    let output = vectors
        .get(&rotation.sv)
        .context("missing verification output vector")?;
    let input_first = if tensor.shape.len() == 3
        && tensor.encoding.ht_bits().is_some()
        && tensor.name.ends_with(".down_proj.weight")
    {
        row / expert_rows * cols
    } else {
        0
    };
    let output_first = if tensor.encoding == Encoding::I4r {
        row % expert_rows
    } else {
        row
    };
    Ok((
        input
            .get(input_first..input_first + cols)
            .context("short verification input vector")?,
        output
            .get(output_first..output_first + take)
            .context("short verification output vector")?,
    ))
}

static VERIFICATION_HT_CODEBOOK: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| {
    (0..=u16::MAX)
        .map(|state| {
            let mixed = u32::from(state).wrapping_mul(0x83dcd12d);
            let sum = (mixed & 255) + ((mixed >> 8) & 255) + ((mixed >> 16) & 255) + (mixed >> 24);
            let exact = (1774i64 * i64::from(sum) - 905216) as f64 / 262144.0;
            half::f16::from_f64(exact).to_f32()
        })
        .collect()
});

/// Independent output oracle: addresses each (row,column) directly, extracts
/// its circular 16-bit state from a two-word window, and performs an FP64
/// inverse transform. It does not reuse the codec's tile mapping or decoder.
fn decode_ht_independent(
    bytes: &[u8],
    bits: usize,
    cols: usize,
    paired: Option<bool>,
    su: &[f32],
    sv: &[f32],
) -> Result<Vec<f32>> {
    ensure!(bits == 3 || bits == 4, "invalid HT verifier width");
    ensure!(
        cols > 0 && cols % 128 == 0 && su.len() == cols && sv.len() == 128,
        "invalid HT verifier geometry"
    );
    let row_tiles = if paired.is_some() { 16 } else { 8 };
    let tile_bytes = 32 * bits;
    let expected = (cols / 16)
        .checked_mul(row_tiles)
        .and_then(|n| n.checked_mul(tile_bytes))
        .context("HT verifier payload overflow")?;
    ensure!(bytes.len() == expected, "invalid HT verifier band size");
    ensure!(
        su.iter().chain(sv).all(|v| v.is_finite()),
        "nonfinite HT verifier rotation"
    );
    let mut stored = vec![0.0f64; 128 * cols];
    for row in 0..128 {
        for col in 0..cols {
            let tile = col / 16 * row_tiles + row / 16 + if paired == Some(true) { 8 } else { 0 };
            let value = (col % 16 / 8) * 128 + row % 16 * 8 + col % 8;
            let start = (bits * (value + 1) + 256 * bits - 16) % (256 * bits);
            let word = start / 32;
            let next = (word + 1) % (8 * bits);
            let base = tile * tile_bytes;
            let first = u32::from_le_bytes(bytes[base + 4 * word..base + 4 * word + 4].try_into()?);
            let second =
                u32::from_le_bytes(bytes[base + 4 * next..base + 4 * next + 4].try_into()?);
            let state =
                (((u64::from(first) << 32) | u64::from(second)) >> (48 - start % 32)) as u16;
            stored[row * cols + col] = f64::from(VERIFICATION_HT_CODEBOOK[state as usize]);
        }
    }
    // Sylvester H * M * H, normalized once by 128. Alternate axis passes
    // rather than the codec's block-local transform to retain an independent
    // implementation while keeping O(N K log 128) work and one band of scratch.
    for axis in [0, 1] {
        for step in [1, 2, 4, 8, 16, 32, 64] {
            for row in 0..128 {
                for col in 0..cols {
                    let coordinate = if axis == 0 { col } else { row };
                    if coordinate & step == 0 {
                        let a = row * cols + col;
                        let b = a + if axis == 0 { step } else { step * cols };
                        let first = stored[a];
                        let second = stored[b];
                        stored[a] = first + second;
                        stored[b] = first - second;
                    }
                }
            }
        }
    }
    let values: Vec<f32> = stored
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            (value / 128.0 * f64::from(sv[index / cols]) * f64::from(su[index % cols])) as f32
        })
        .collect();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite HT verifier value"
    );
    Ok(values)
}

fn check_ngram_order(shards: &[String]) -> Result<()> {
    let mut first_prefix = None;
    let mut first_weight_suffix = None;
    for (index, name) in shards.iter().enumerate() {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let (prefix, number) = stem
            .rsplit_once("shard_")
            .context("n-gram source lacks shard index")?;
        ensure!(
            !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit()),
            "invalid n-gram shard index"
        );
        ensure!(
            number == "0" || !number.starts_with('0'),
            "noncanonical n-gram shard index"
        );
        ensure!(
            number.parse::<usize>()? == index,
            "n-gram shards are not in contiguous numeric order"
        );
        ensure!(
            *first_prefix.get_or_insert(prefix) == prefix,
            "mixed n-gram shard prefixes"
        );
        ensure!(
            *first_weight_suffix.get_or_insert(name.ends_with(".weight"))
                == name.ends_with(".weight"),
            "mixed n-gram shard suffixes"
        );
    }
    ensure!(!shards.is_empty(), "empty n-gram source list");
    Ok(())
}

fn check_quant_metadata(
    reader: &mut Reader,
    index: usize,
    encoding: Encoding,
    rows: usize,
    cols: usize,
    layout: &PayloadLayout,
) -> Result<f32> {
    let values = (rows as u64)
        .checked_mul(cols as u64)
        .context("payload element count overflow")?;
    match encoding {
        Encoding::Q4Cp => {
            let codebook = reader.read_at(index, 0, 64)?;
            for (bytes, expected) in codebook.chunks_exact(4).zip(codec::NF4_CODEBOOK) {
                let actual = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
                ensure!(
                    actual.is_finite() && actual.to_bits() == expected.to_bits(),
                    "Q4 codebook differs from the fixed NF4 policy"
                );
            }
            let code_end = layout
                .codes_offset
                .checked_add(values / 2)
                .context("code plane overflow")?;
            check_zero_padding(
                reader,
                index,
                code_end,
                layout
                    .scales_offset
                    .checked_sub(code_end)
                    .context("invalid Q4 planes")?,
            )?;
            Ok(1.0)
        }
        Encoding::Fp8G128 => {
            check_zero_padding(
                reader,
                index,
                values,
                layout
                    .scales_offset
                    .checked_sub(values)
                    .context("invalid FP8 planes")?,
            )?;
            let bytes = reader.read_at(index, layout.scales_offset, 4)?;
            let scale = f32::from_le_bytes(bytes.try_into().expect("four bytes"));
            ensure!(
                scale.is_finite() && scale > 0.0,
                "FP8 global scale must be positive and finite"
            );
            Ok(scale)
        }
        Encoding::Q8G64 | Encoding::Q6G64 | Encoding::Ht | Encoding::Ht3 | Encoding::I4r => Ok(1.0),
        _ => bail!("not a quantized matrix encoding"),
    }
}

fn check_zero_padding(reader: &mut Reader, index: usize, offset: u64, size: u64) -> Result<()> {
    ensure!(size < 64, "excessive quantized payload padding");
    if size != 0 {
        let bytes = reader.read_at(index, offset, usize::try_from(size)?)?;
        ensure!(
            bytes.iter().all(|&byte| byte == 0),
            "nonzero quantized payload padding"
        );
    }
    Ok(())
}

fn read_block(
    reader: &mut Reader,
    index: usize,
    encoding: Encoding,
    layout: &PayloadLayout,
    row: usize,
    rows: usize,
    cols: usize,
) -> Result<EncodedBlock> {
    let count = (rows as u64)
        .checked_mul(cols as u64)
        .context("block size overflow")?;
    let first = (row as u64)
        .checked_mul(cols as u64)
        .context("block offset overflow")?;
    let (codes_offset, codes_len, scales_offset, scales_len) = match encoding {
        Encoding::Q4Cp | Encoding::I4r => (
            layout
                .codes_offset
                .checked_add(first / 2)
                .context("code offset overflow")?,
            count / 2,
            layout
                .scales_offset
                .checked_add(
                    (row as u64)
                        .checked_mul(layout.scale_row_stride)
                        .context("scale offset overflow")?,
                )
                .context("scale offset overflow")?,
            (rows as u64)
                .checked_mul(layout.scale_row_stride)
                .context("scale size overflow")?,
        ),
        Encoding::Ht => (first / 2, count / 2, 0, 0),
        Encoding::Ht3 => (
            first.checked_mul(3).context("HT3 offset overflow")? / 8,
            count.checked_mul(3).context("HT3 size overflow")? / 8,
            0,
            0,
        ),
        Encoding::Bf16 => (
            first.checked_mul(2).context("BF16 offset overflow")?,
            count.checked_mul(2).context("BF16 size overflow")?,
            0,
            0,
        ),
        Encoding::Fp8G128 => (first, count, 0, 0),
        Encoding::Q8G64 | Encoding::Q6G64 => {
            let stride = codec::payload_layout(encoding, 1, cols)?.len;
            (
                (row as u64)
                    .checked_mul(stride)
                    .context("row offset overflow")?,
                (rows as u64)
                    .checked_mul(stride)
                    .context("row size overflow")?,
                0,
                0,
            )
        }
        _ => bail!("not a quantized block encoding"),
    };
    let codes = reader.read_at(index, codes_offset, usize::try_from(codes_len)?)?;
    let scales = if scales_len == 0 {
        Vec::new()
    } else {
        reader.read_at(index, scales_offset, usize::try_from(scales_len)?)?
    };
    Ok(EncodedBlock { codes, scales })
}

fn accumulate_errors(totals: &mut ErrorTotals, source: &[f32], decoded: &[f32]) -> Result<()> {
    ensure!(
        source.len() == decoded.len(),
        "decoded element count mismatch"
    );
    for (&source, &decoded) in source.iter().zip(decoded) {
        ensure!(
            source.is_finite() && decoded.is_finite(),
            "nonfinite source/decoded value"
        );
        let source = f64::from(source);
        let error = f64::from(decoded) - source;
        totals.source_squared += source * source;
        totals.error_squared += error * error;
    }
    totals.values = totals
        .values
        .checked_add(source.len() as u64)
        .context("error value count overflow")?;
    ensure!(
        totals.source_squared.is_finite() && totals.error_squared.is_finite(),
        "nonfinite error totals"
    );
    Ok(())
}

fn accumulate_exact(totals: &mut ErrorTotals, raw: &[u8], dtype: Dtype) -> Result<()> {
    let width = usize::try_from(dtype.byte_width())?;
    ensure!(raw.len() % width == 0, "partial direct source value");
    for bytes in raw.chunks_exact(width) {
        let value = match dtype {
            Dtype::Bf16 => f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16,
            )),
            Dtype::F32 => f64::from(f32::from_le_bytes(bytes.try_into().expect("four bytes"))),
            Dtype::I64 => i64::from_le_bytes(bytes.try_into().expect("eight bytes")) as f64,
        };
        ensure!(value.is_finite(), "nonfinite direct source value");
        totals.source_squared += value * value;
    }
    totals.values = totals
        .values
        .checked_add((raw.len() / width) as u64)
        .context("direct value count overflow")?;
    ensure!(
        totals.source_squared.is_finite(),
        "nonfinite exact-source total"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf::Snapshot;
    use crate::ContainerPlan;
    use serde_json::{json, Value};
    use std::{
        fs::File,
        io::{Seek, SeekFrom, Write},
    };
    use tempfile::{tempdir, TempDir};

    const TABLE_SOURCE: &str =
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";

    struct Fixture {
        _directory: TempDir,
        snapshot: Snapshot,
        plan: ConversionPlan,
        output: PathBuf,
    }

    fn fixture() -> Fixture {
        let directory = tempdir().unwrap();
        let root = directory.path().join("source");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        let specifications = [
            ("raw", "BF16", vec![2u64, 32]),
            ("quant", "BF16", vec![129, 32]),
            ("q6", "BF16", vec![2, 320]),
            ("indices", "I64", vec![3]),
            ("float", "F32", vec![4]),
            (TABLE_SOURCE, "BF16", vec![3, 160]),
        ];
        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        let mut weights = serde_json::Map::new();
        for (name, dtype, shape) in &specifications {
            let start = payload.len();
            match *dtype {
                "I64" => {
                    for value in [i64::MIN, (1i64 << 53) + 1, i64::MAX] {
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                }
                "F32" => {
                    for value in [-0.0f32, 3.25, f32::MAX, -f32::MAX] {
                        payload.extend_from_slice(&value.to_le_bytes());
                    }
                }
                _ => {
                    for index in 0..numel(shape).unwrap() {
                        let value = (index % 29) as f32 / 16.0 - 0.75;
                        payload.extend_from_slice(
                            &half::bf16::from_f32(value).to_bits().to_le_bytes(),
                        );
                    }
                }
            }
            header.insert(
                (*name).into(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
            weights.insert((*name).into(), json!("model.safetensors"));
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        header.resize(header.len() + (8 - header.len() % 8) % 8, b' ');
        let mut file = File::create(root.join("model.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload).unwrap();
        drop(file);
        fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": payload.len()}, "weight_map": weights
            }))
            .unwrap(),
        )
        .unwrap();
        let tensors = specifications[..5]
            .iter()
            .map(|(name, _, shape)| TensorPlan {
                name: (*name).into(),
                shape: shape.clone(),
                encoding: match *name {
                    "quant" => Encoding::Q4Cp,
                    "q6" => Encoding::Q6G64,
                    "indices" => Encoding::I64,
                    "float" => Encoding::F32,
                    _ => Encoding::Bf16,
                },
                origin: Origin::Tensor {
                    name: (*name).into(),
                },
                rotation: None,
            })
            .collect();
        let plan = ConversionPlan {
            architecture: "structural-fixture".into(),
            files: vec![
                ContainerPlan {
                    filename: "model.hgn".into(),
                    identity: "fixture".into(),
                    tensors,
                },
                ContainerPlan {
                    filename: "table.hgn".into(),
                    identity: "table-fixture".into(),
                    tensors: vec![TensorPlan {
                        name: "table".into(),
                        shape: vec![1, 3, 160],
                        encoding: Encoding::Fp8G128,
                        origin: Origin::Ngram {
                            shards: vec![TABLE_SOURCE.into()],
                        },
                        rotation: None,
                    }],
                },
            ],
            assets: vec!["config.json".into()],
        };
        let snapshot = Snapshot::open(&root).unwrap();
        let output = directory.path().join("output");
        conversion::convert(&snapshot, &plan, &output, &Options::default()).unwrap();
        Fixture {
            _directory: directory,
            snapshot,
            plan,
            output,
        }
    }

    fn save_manifest(fixture: &Fixture, manifest: &Manifest) {
        fs::write(
            fixture.output.join(MANIFEST_NAME),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
    }

    fn patch_payload(
        fixture: &Fixture,
        container_index: usize,
        tensor_index: usize,
        relative: u64,
        bytes: &[u8],
    ) {
        let path = fixture
            .output
            .join(&fixture.plan.files[container_index].filename);
        let reader = Reader::open(&path).unwrap();
        let offset = reader.layout.entries[tensor_index].offset + relative;
        let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(bytes).unwrap();
    }

    // Give corrupt bytes internally consistent hashes/checksums. Semantic and
    // source checks must still reject them, independently of fingerprint checks.
    fn reseal(fixture: &Fixture, container_index: usize, tensor_index: usize) {
        let mut manifest =
            read_manifest(&fixture.output.join(MANIFEST_NAME), &fixture.plan).unwrap();
        let path = fixture
            .output
            .join(&fixture.plan.files[container_index].filename);
        let mut reader = Reader::open(&path).unwrap();
        let fingerprint = reader.fingerprint(tensor_index).unwrap();
        let mut entry = reader.layout.entries[tensor_index].clone();
        entry.checksum = fingerprint.xor32;
        let mut file = fs::OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(
            container::HEADER_SIZE + tensor_index as u64 * container::ENTRY_SIZE,
        ))
        .unwrap();
        file.write_all(&container::entry_bytes(&entry).unwrap())
            .unwrap();
        let record = &mut manifest.containers[container_index];
        record.layout.entries[tensor_index] = entry.clone();
        record.tensors[tensor_index].entry = entry;
        record.tensors[tensor_index].fingerprint = fingerprint;
        save_manifest(fixture, &manifest);
    }

    fn assert_rejected(fixture: &Fixture, message: &str) {
        let error = verify(&fixture.snapshot, &fixture.plan, &fixture.output).unwrap_err();
        assert!(
            format!("{error:#}").contains(message),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn actual_conversion_is_fully_verified_across_partial_blocks() {
        let fixture = fixture();
        let report = verify(&fixture.snapshot, &fixture.plan, &fixture.output).unwrap();
        assert_eq!(report.source_tensors, 6);
        assert_eq!(report.output_tensors, 6);
        assert_eq!(report.errors.len(), 6);
        assert_eq!(report.errors["model.hgn:quant"].values, 129 * 32);
        assert_eq!(report.errors["table.hgn:table"].values, 3 * 160);
        for name in ["raw", "indices", "float"] {
            assert_eq!(
                report.errors[&format!("model.hgn:{name}")].error_squared,
                0.0
            );
        }
        assert!(report.errors["model.hgn:float"].source_squared > f64::from(f32::MAX));
        let expected: u64 = fixture
            .plan
            .files
            .iter()
            .flat_map(|file| container::layout(file).unwrap().entries)
            .map(|entry| entry.size)
            .sum();
        assert_eq!(report.payload_bytes, expected);
    }

    #[test]
    fn provenance_hash_allows_rebuilds_but_manifest_fields_are_strict() {
        let fixture = fixture();
        let mut manifest =
            read_manifest(&fixture.output.join(MANIFEST_NAME), &fixture.plan).unwrap();
        manifest.encoder_sha256 = "a".repeat(64);
        save_manifest(&fixture, &manifest);
        verify(&fixture.snapshot, &fixture.plan, &fixture.output).unwrap();
        let mut json = serde_json::to_value(&manifest).unwrap();
        json["sources"][0]["unexpected"] = json!(true);
        fs::write(
            fixture.output.join(MANIFEST_NAME),
            serde_json::to_vec(&json).unwrap(),
        )
        .unwrap();
        assert_rejected(&fixture, "unknown or noncanonical");
        fs::write(
            fixture.output.join(MANIFEST_NAME),
            br#"{"schema":1,"schema":1}"#,
        )
        .unwrap();
        assert_rejected(&fixture, "duplicate JSON key");
    }

    #[test]
    fn manifest_sources_require_exact_metadata_coverage_and_hashes() {
        let fixture = fixture();
        let original = read_manifest(&fixture.output.join(MANIFEST_NAME), &fixture.plan).unwrap();
        let mut manifest = original.clone();
        manifest.sources[0].tensor.offset += 1;
        save_manifest(&fixture, &manifest);
        assert_rejected(&fixture, "source metadata mismatch");
        let mut manifest = original.clone();
        manifest.sources[1] = manifest.sources[0].clone();
        save_manifest(&fixture, &manifest);
        assert_rejected(&fixture, "duplicate manifest source");
        let mut manifest = original;
        manifest
            .sources
            .iter_mut()
            .find(|source| source.tensor.name == "quant")
            .unwrap()
            .sha256 = "0".repeat(64);
        save_manifest(&fixture, &manifest);
        assert_rejected(&fixture, "source tensor SHA-256 mismatch");
    }

    #[test]
    fn fingerprint_checks_actual_quantized_bytes() {
        let fixture = fixture();
        patch_payload(&fixture, 0, 1, 64, &[0x88]);
        assert_rejected(&fixture, "payload SHA-256/XOR mismatch");
    }

    #[test]
    fn direct_values_require_source_bytes_even_after_resealing() {
        let fixture = fixture();
        // Flip the otherwise f64-invisible least significant bit of 2^53 + 1.
        patch_payload(&fixture, 0, 3, 8, &[0]);
        reseal(&fixture, 0, 3);
        assert_rejected(&fixture, "direct payload differs from source bytes");
    }

    #[test]
    fn q4_codebook_scale_and_global_padding_are_independently_validated() {
        let layout = codec::payload_layout(Encoding::Q4Cp, 129, 32).unwrap();
        for (offset, bytes, message) in [
            (0, vec![1], "Q4 codebook"),
            (64 + 129 * 16, vec![1], "nonzero quantized payload padding"),
            (
                layout.scales_offset + 2,
                vec![1],
                "nonzero scale-row padding",
            ),
            (
                layout.scales_offset,
                vec![0, 0x7c],
                "non-finite quantization scale",
            ),
        ] {
            let fixture = fixture();
            patch_payload(&fixture, 0, 1, offset, &bytes);
            reseal(&fixture, 0, 1);
            assert_rejected(&fixture, message);
        }
    }

    #[test]
    fn q6_padding_is_checked_after_fingerprints_are_resealed() {
        let fixture = fixture();
        patch_payload(&fixture, 0, 2, 260, &[1]);
        reseal(&fixture, 0, 2);
        assert_rejected(&fixture, "nonzero affine-row padding");
    }

    #[test]
    fn fp8_padding_scale_and_encoder_range_are_independent_checks() {
        let layout = codec::payload_layout(Encoding::Fp8G128, 3, 160).unwrap();
        for (offset, bytes, message) in [
            (480, vec![1], "nonzero quantized payload padding"),
            (
                layout.scales_offset,
                0.0f32.to_le_bytes().to_vec(),
                "positive and finite",
            ),
            (
                layout.scales_offset,
                1.0f32.to_le_bytes().to_vec(),
                "source absmax policy",
            ),
            (0, vec![0x7f], "448 magnitude policy"),
        ] {
            let fixture = fixture();
            patch_payload(&fixture, 1, 0, offset, &bytes);
            reseal(&fixture, 1, 0);
            assert_rejected(&fixture, message);
        }
    }

    #[test]
    fn inventories_and_asset_provenance_are_exact() {
        let fixture = fixture();
        let mut manifest =
            read_manifest(&fixture.output.join(MANIFEST_NAME), &fixture.plan).unwrap();
        manifest.assets[0].source = "tokenizer.json".into();
        save_manifest(&fixture, &manifest);
        assert_rejected(&fixture, "manifest asset source mismatch");
        manifest.assets[0].source = "config.json".into();
        save_manifest(&fixture, &manifest);
        fs::write(fixture.output.join("config.json"), b"{ }").unwrap();
        assert_rejected(&fixture, "copied asset SHA-256/size mismatch");
        fs::write(fixture.output.join("config.json"), b"{}").unwrap();
        fs::write(fixture.output.join("unplanned"), b"").unwrap();
        assert_rejected(&fixture, "unexpected output file");
    }

    #[cfg(unix)]
    #[test]
    fn output_symlinks_are_rejected() {
        let fixture = fixture();
        let path = fixture.output.join("config.json");
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(fixture.snapshot.root.join("config.json"), path).unwrap();
        assert_rejected(&fixture, "symlink in output");
    }

    #[test]
    fn ngram_order_is_numeric_not_lexicographic() {
        let mut shards: Vec<_> = (0..12)
            .map(|index| format!("table.shard_{index}.weight"))
            .collect();
        check_ngram_order(&shards).unwrap();
        shards.swap(2, 10);
        assert!(check_ngram_order(&shards).is_err());
        assert!(check_ngram_order(&["table.shard_00.weight".into()]).is_err());
        assert!(
            check_ngram_order(&["table.shard_0.weight".into(), "other.shard_1.weight".into()])
                .is_err()
        );
    }

    #[test]
    fn squared_errors_use_f64_and_refuse_nonfinite_values() {
        let mut totals = ErrorTotals::default();
        accumulate_errors(&mut totals, &[f32::MAX], &[-f32::MAX]).unwrap();
        assert_eq!(totals.values, 1);
        let expected = f64::from(f32::MAX);
        assert_eq!(totals.source_squared, expected * expected);
        assert_eq!(totals.error_squared, 4.0 * expected * expected);
        assert!(accumulate_errors(&mut totals, &[0.0], &[f32::NAN]).is_err());
        assert!(accumulate_errors(&mut totals, &[f32::INFINITY], &[0.0]).is_err());
        assert!(accumulate_errors(&mut totals, &[0.0], &[]).is_err());
        assert!(check_sha256(&"0".repeat(63)).is_err());
        assert!(check_sha256(&"g".repeat(64)).is_err());
        assert!(check_sha256(&"0".repeat(64)).is_ok());
    }

    #[test]
    fn manifest_plan_must_match_even_with_equal_tensor_counts() {
        let fixture = fixture();
        let mut manifest =
            read_manifest(&fixture.output.join(MANIFEST_NAME), &fixture.plan).unwrap();
        manifest.plan.files[0].tensors.swap(0, 1);
        save_manifest(&fixture, &manifest);
        assert_rejected(&fixture, "fresh source plan");
        let mut json: Value = serde_json::to_value(&manifest).unwrap();
        json["plan"] = serde_json::to_value(&fixture.plan).unwrap();
        json["containers"][0]["layout"]["identity"] = json!("wrong-identity");
        fs::write(
            fixture.output.join(MANIFEST_NAME),
            serde_json::to_vec(&json).unwrap(),
        )
        .unwrap();
        assert_rejected(&fixture, "manifest container layout mismatch");
    }
}

#[cfg(test)]
#[path = "ht43_verify_tests.rs"]
mod ht43_tests;

// Internal bridge for the combined paired-stream regression. Its canonical
// 2-D byte sink deliberately does not relax the public reader's fixed expert
// shapes; production dispatch, source coverage and independent decoding run
// unchanged against the logical expert tensor.
#[cfg(test)]
pub(crate) fn verify_tensor_fixture(
    snapshot: &dyn TensorSource,
    tensor: &TensorPlan,
    reader: &mut Reader,
    options: &Options,
    rotations: BTreeMap<String, Vec<f32>>,
    source_hash: String,
) -> Result<ErrorTotals> {
    let Origin::Tensor { name } = &tensor.origin else {
        bail!("fixture must consume one source tensor");
    };
    let records = [SourceRecord {
        tensor: snapshot
            .tensors()
            .get(name)
            .context("missing fixture source")?
            .clone(),
        sha256: source_hash,
    }];
    let expected = BTreeSet::from([name.clone()]);
    let sources = SourceCoverage::new(snapshot, &expected, &records)?;
    let mut verifier = PayloadVerifier {
        snapshot,
        options,
        rotations,
        sources,
    };
    let totals = verifier
        .verify_tensor(reader, 0, tensor)?
        .context("missing fixture error totals")?;
    ensure!(
        verifier.sources.finish()? == 1,
        "fixture source coverage mismatch"
    );
    let fingerprint = reader.fingerprint(0)?;
    ensure!(
        fingerprint.xor32 == reader.layout.entries[0].checksum,
        "fixture payload checksum mismatch"
    );
    Ok(totals)
}
