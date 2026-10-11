use crate::{
    codec::{self, Encoding, QuantMeta},
    container::{self, Writer},
    hf::{Dtype, TensorInfo},
    manifest::{AssetRecord, ContainerRecord, Manifest, Options, OutputRecord, SourceRecord},
    numel, rows_cols,
    source::{SourceKind, TensorSource, ASSET_NAMES},
    ConversionPlan, Origin, TensorPlan, ENCODER_SHA256, SPEC_COMMIT, SPEC_VERSION,
};
use anyhow::{bail, ensure, Context, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

const COPY_CHUNK: usize = 8 * 1024 * 1024;
const MATRIX_ROWS: usize = 128;
const TABLE_ROWS: usize = 8192;

pub fn validate_plan(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
) -> Result<BTreeSet<String>> {
    ensure!(!plan.files.is_empty(), "empty conversion plan");
    let mut filenames = BTreeSet::new();
    let mut sources = BTreeSet::new();
    let mut outputs = BTreeMap::new();
    for file in &plan.files {
        safe_filename(&file.filename)?;
        ensure!(
            file.filename.ends_with(".hgn") && filenames.insert(&file.filename),
            "invalid/duplicate HGN filename"
        );
        container::layout(file)?;
        for tensor in &file.tensors {
            ensure!(
                outputs.insert(tensor.name.clone(), tensor).is_none(),
                "duplicate output tensor across files: {}",
                tensor.name
            );
            match &tensor.origin {
                Origin::Tensor { name } => {
                    let info = snapshot
                        .tensors()
                        .get(name)
                        .with_context(|| format!("missing source tensor {name}"))?;
                    ensure!(
                        sources.insert(name.clone()),
                        "source tensor mapped more than once: {name}"
                    );
                    ensure!(
                        numel(&info.shape)? == numel(&tensor.shape)?,
                        "reshape changes element count: {name}"
                    );
                    let expected = match tensor.encoding {
                        Encoding::Bf16
                            if snapshot.kind() != SourceKind::Hf && info.dtype == Dtype::F32 =>
                        {
                            Dtype::F32
                        }
                        Encoding::Bf16 => Dtype::Bf16,
                        Encoding::F32 => Dtype::F32,
                        Encoding::I64 => Dtype::I64,
                        Encoding::F16 | Encoding::Fp8G128 => {
                            bail!("invalid direct source encoding")
                        }
                        _ => {
                            ensure!(info.dtype != Dtype::I64, "cannot quantize integers");
                            info.dtype
                        }
                    };
                    ensure!(
                        info.dtype == expected,
                        "source/output dtype mismatch: {name}"
                    );
                }
                Origin::Signs { key } => {
                    ensure!(
                        tensor.encoding == Encoding::F16
                            && key == &tensor.name
                            && tensor.rotation.is_none(),
                        "invalid sign plane"
                    );
                }
                Origin::HtScales {
                    tensor: source,
                    signs_key,
                } => {
                    let info = snapshot
                        .tensors()
                        .get(source)
                        .context("missing HT-scale source")?;
                    ensure!(
                        tensor.encoding == Encoding::F16
                            && info.dtype != Dtype::I64
                            && signs_key == &tensor.name
                            && tensor.rotation.is_none(),
                        "invalid HT scale plane"
                    );
                    let (rows, _) = rows_cols(&info.shape)?;
                    ensure!(
                        numel(&tensor.shape)? == rows as u64,
                        "HT scale count mismatch"
                    );
                }
                Origin::Ngram { shards } => {
                    ensure!(
                        tensor.encoding == Encoding::Fp8G128
                            && tensor.shape.len() == 3
                            && tensor.shape[0] == shards.len() as u64
                            && tensor.rotation.is_none(),
                        "invalid n-gram plan"
                    );
                    ensure!(!shards.is_empty(), "empty n-gram source");
                    for (shard_index, name) in shards.iter().enumerate() {
                        ensure!(
                            name.ends_with(&format!(".shard_{shard_index}.weight")),
                            "n-gram shards must be in contiguous numeric order"
                        );
                        let info = snapshot
                            .tensors()
                            .get(name)
                            .context("missing n-gram source")?;
                        ensure!(
                            sources.insert(name.clone()),
                            "n-gram source mapped more than once"
                        );
                        ensure!(
                            (info.dtype == Dtype::Bf16
                                || (snapshot.kind() != SourceKind::Hf && info.dtype == Dtype::F32))
                                && info.shape == tensor.shape[1..],
                            "n-gram source shape/dtype mismatch"
                        );
                    }
                }
            }
        }
    }
    for tensor in outputs.values() {
        let rotated = tensor.encoding.ht_bits().is_some() || tensor.encoding == Encoding::I4r;
        ensure!(
            tensor.rotation.is_some() == rotated,
            "missing/unexpected rotation: {}",
            tensor.name
        );
        if let Some(rotation) = &tensor.rotation {
            let (total_rows, cols) = rows_cols(&tensor.shape)?;
            let experts = tensor.shape.len() == 3;
            let rows = if tensor.encoding == Encoding::I4r {
                tensor.shape[1]
            } else {
                total_rows as u64
            };
            let input_count = if experts
                && tensor.encoding.ht_bits().is_some()
                && tensor.name.ends_with(".down_proj.weight")
            {
                tensor.shape[0]
                    .checked_mul(cols as u64)
                    .context("rotation count overflow")?
            } else {
                cols as u64
            };
            for (name, count) in [(&rotation.su, input_count), (&rotation.sv, rows)] {
                let vector = outputs.get(name).context("missing rotation plane")?;
                ensure!(
                    vector.encoding == Encoding::F16
                        && numel(&vector.shape)? == count
                        && matches!(
                            vector.origin,
                            Origin::Signs { .. } | Origin::HtScales { .. }
                        ),
                    "invalid rotation plane {name}"
                );
            }
        }
    }
    let actual: BTreeSet<_> = snapshot.tensors().keys().cloned().collect();
    ensure!(
        sources == actual,
        "plan does not cover the exact source tensor set"
    );
    asset_destinations(plan)?;
    Ok(sources)
}

pub fn prepare_rotations(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    options: &Options,
) -> Result<BTreeMap<String, Vec<f32>>> {
    let mut vectors = BTreeMap::new();
    for tensor in plan.files.iter().flat_map(|f| &f.tensors) {
        match &tensor.origin {
            Origin::Signs { key } => {
                let values = codec::generate_signs(
                    options.seed,
                    key,
                    usize::try_from(numel(&tensor.shape)?)?,
                );
                vectors.insert(tensor.name.clone(), values);
            }
            Origin::HtScales {
                tensor: source,
                signs_key,
            } => {
                let info = snapshot
                    .tensors()
                    .get(source)
                    .context("missing HT source")?;
                let (rows, cols) = rows_cols(&info.shape)?;
                let signs = codec::generate_signs(options.seed, signs_key, rows);
                let mut scales = Vec::with_capacity(rows);
                let mut row = 0;
                while row < rows {
                    let take = MATRIX_ROWS.min(rows - row);
                    let input = snapshot.read_f32(source, (row * cols) as u64, take * cols)?;
                    let block = codec::ht_row_scales(&input, take, cols, &signs[row..row + take])?;
                    ensure!(
                        block.len() == take && block.iter().all(|x| x.is_finite()),
                        "invalid generated HT row scale"
                    );
                    scales.extend(block);
                    row += take;
                }
                vectors.insert(tensor.name.clone(), scales);
            }
            _ => {}
        }
    }
    Ok(vectors)
}

pub fn read_source_values(
    snapshot: &dyn TensorSource,
    info: &TensorInfo,
    element_offset: u64,
    len: usize,
) -> Result<(Vec<u8>, Vec<f32>)> {
    ensure!(
        info.dtype != Dtype::I64,
        "integer source requires exact raw access"
    );
    let width = info.dtype.byte_width();
    let offset = element_offset
        .checked_mul(width)
        .context("source offset overflow")?;
    let byte_len = usize::try_from(
        (len as u64)
            .checked_mul(width)
            .context("source read overflow")?,
    )?;
    let raw = snapshot.read_bytes(&info.name, offset, byte_len)?;
    let values = decode_source(&raw, info.dtype)?;
    Ok((raw, values))
}

fn decode_source(raw: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    ensure!(dtype != Dtype::I64, "cannot widen i64 weights through f32");
    let width = dtype.byte_width() as usize;
    ensure!(raw.len() % width == 0, "unaligned source float bytes");
    let mut values = Vec::with_capacity(raw.len() / width);
    for bytes in raw.chunks_exact(width) {
        values.push(source_float(bytes, dtype)?);
    }
    Ok(values)
}

fn source_float(bytes: &[u8], dtype: Dtype) -> Result<f32> {
    let value = match dtype {
        Dtype::Bf16 => f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16),
        Dtype::F32 => f32::from_le_bytes(bytes.try_into().expect("four bytes")),
        Dtype::I64 => bail!("cannot widen i64 weights through f32"),
    };
    ensure!(value.is_finite(), "nonfinite source weight");
    Ok(value)
}

pub fn validate_ple(snapshot: &dyn TensorSource, plan: &ConversionPlan) -> Result<()> {
    let prefix = "model.language_model.layers.1.ple.ple_embedding.";
    let names = [
        format!("{prefix}layer_multipliers"),
        format!("{prefix}ngram_heads_offsets"),
        format!("{prefix}ngram_heads_vocab_sizes"),
    ];
    let present = names
        .iter()
        .filter(|name| snapshot.tensors().contains_key(*name))
        .count();
    if present == 0 {
        return Ok(());
    }
    ensure!(present == 3, "incomplete PLE hash vectors");
    let table = plan
        .files
        .iter()
        .flat_map(|f| &f.tensors)
        .find(|t| matches!(t.origin, Origin::Ngram { .. }))
        .context("PLE vectors need n-gram table")?;
    let rows = table.shape[0]
        .checked_mul(table.shape[1])
        .context("n-gram row count overflow")?;
    let mut vectors = Vec::new();
    for name in &names {
        let info = &snapshot.tensors()[name];
        ensure!(info.dtype == Dtype::I64, "PLE vectors must be i64");
        let bytes = snapshot.read_bytes(name, 0, usize::try_from(info.size)?)?;
        vectors.push(
            bytes
                .chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("eight bytes")))
                .collect::<Vec<_>>(),
        );
    }
    ensure!(
        vectors[0].len() == 3 && vectors[0].iter().all(|&x| x > 0),
        "invalid PLE layer multipliers"
    );
    ensure!(
        vectors[1].len() == 16 && vectors[2].len() == 16,
        "invalid PLE head vectors"
    );
    for (&offset, &size) in vectors[1].iter().zip(&vectors[2]) {
        ensure!(
            offset >= 0
                && size > 0
                && (offset as u64)
                    .checked_add(size as u64)
                    .context("PLE index overflow")?
                    <= rows,
            "PLE head exceeds n-gram table"
        );
    }
    Ok(())
}

/// A structural library entry point. The public CLI always uses architecture::plan.
pub fn convert(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    output: &Path,
    options: &Options,
) -> Result<Manifest> {
    ensure!(
        cfg!(any(target_os = "linux", target_os = "macos")),
        "atomic no-clobber publication is supported on Linux/macOS only"
    );
    options.validate_for_execution()?;
    let expected_sources = validate_plan(snapshot, plan)?;
    validate_ple(snapshot, plan)?;
    snapshot.check_unchanged()?;
    let destination = destination_path(snapshot, output)?;
    let parent = destination.parent().context("output has no parent")?;
    let required_bytes = planned_bytes(plan)?;
    check_space(parent, required_bytes)?;
    let staging = tempfile::Builder::new()
        .prefix(".bf16-to-hgn-")
        .tempdir_in(parent)?;
    let vectors = prepare_rotations(snapshot, plan, options)?;
    let mut source_hashes = BTreeMap::new();
    let mut containers = Vec::new();
    for file in &plan.files {
        let mut writer = Writer::create(&staging.path().join(&file.filename), file)?;
        for (index, tensor) in file.tensors.iter().enumerate() {
            match &tensor.origin {
                Origin::Signs { .. } | Origin::HtScales { .. } => {
                    let values = vectors
                        .get(&tensor.name)
                        .context("missing prepared rotation")?;
                    let mut bytes = Vec::with_capacity(values.len() * 2);
                    for &value in values {
                        bytes
                            .extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
                    }
                    writer.write_at(index, 0, &bytes)?;
                }
                Origin::Tensor { name } => {
                    let info = &snapshot.tensors()[name];
                    let hash = encode_tensor(
                        snapshot,
                        info,
                        tensor,
                        index,
                        &mut writer,
                        &vectors,
                        options,
                    )?;
                    source_hashes.insert(name.clone(), hash);
                }
                Origin::Ngram { shards } => encode_table(
                    snapshot,
                    tensor,
                    shards,
                    index,
                    &mut writer,
                    &mut source_hashes,
                    options,
                )?,
            }
        }
        let (layout, fingerprints) = writer.finish()?;
        let tensors = layout
            .entries
            .iter()
            .cloned()
            .zip(fingerprints)
            .map(|(entry, fingerprint)| OutputRecord { entry, fingerprint })
            .collect();
        containers.push(ContainerRecord {
            filename: file.filename.clone(),
            layout,
            tensors,
        });
    }
    ensure!(
        source_hashes.keys().cloned().collect::<BTreeSet<_>>() == expected_sources,
        "conversion did not consume every source tensor"
    );
    let sources = source_hashes
        .into_iter()
        .map(|(name, sha256)| SourceRecord {
            tensor: snapshot.tensors()[&name].clone(),
            sha256,
        })
        .collect();
    let assets = copy_assets(snapshot, plan, staging.path())?;
    let (_, config_sha256) = hash_file(&snapshot.asset_path("config.json")?)?;
    let index_sha256 = if snapshot.kind() == SourceKind::Hf {
        Some(hash_file(&snapshot.asset_path("model.safetensors.index.json")?)?.1)
    } else {
        None
    };
    let source_provenance = snapshot.provenance()?;
    let manifest = Manifest {
        schema: 2,
        tool: "bf16-to-hgn".into(),
        tool_version: env!("CARGO_PKG_VERSION").into(),
        encoder_sha256: ENCODER_SHA256.into(),
        spec_version: SPEC_VERSION.into(),
        spec_commit: SPEC_COMMIT.into(),
        policy: options.profile.policy().into(),
        options: options.clone(),
        plan: plan.clone(),
        config_sha256,
        index_sha256,
        source_kind: snapshot.kind(),
        source_provenance,
        sources,
        assets,
        containers,
    };
    let manifest_path = staging.path().join("conversion-manifest.json");
    let mut manifest_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(manifest_path)?;
    serde_json::to_writer_pretty(&mut manifest_file, &manifest)?;
    manifest_file.write_all(b"\n")?;
    manifest_file.sync_all()?;
    snapshot.check_unchanged()?;
    File::open(staging.path())?.sync_all()?;
    publish_no_clobber(staging.path(), &destination)?;
    if let Err(error) = File::open(parent).and_then(|directory| directory.sync_all()) {
        // This path was created by our successful no-replace rename, never an existing destination.
        fs::remove_dir_all(&destination)
            .context("publication sync failed and newly published output cleanup failed")?;
        return Err(error).context("publication sync failed; removed newly published output");
    }
    Ok(manifest)
}

fn encode_tensor(
    snapshot: &dyn TensorSource,
    info: &TensorInfo,
    tensor: &TensorPlan,
    index: usize,
    writer: &mut Writer,
    vectors: &BTreeMap<String, Vec<f32>>,
    options: &Options,
) -> Result<String> {
    let mut hash = Sha256::new();
    if direct_copy(tensor.encoding, info.dtype) {
        let mut offset = 0;
        while offset < info.size {
            let len = (info.size - offset).min(COPY_CHUNK as u64) as usize;
            let bytes = snapshot.read_bytes(&info.name, offset, len)?;
            if info.dtype != Dtype::I64 {
                for value in bytes.chunks_exact(info.dtype.byte_width() as usize) {
                    source_float(value, info.dtype)?;
                }
            }
            hash.update(&bytes);
            writer.write_at(index, offset, &bytes)?;
            offset += len as u64;
        }
        return Ok(format!("{:x}", hash.finalize()));
    }
    if tensor.encoding.ht_bits().is_some()
        && tensor.shape.len() == 3
        && tensor.name.ends_with(".gate_up_proj.weight")
    {
        return encode_gate_up(snapshot, info, tensor, index, writer, vectors, options);
    }
    let (rows, cols) = rows_cols(&tensor.shape)?;
    let layout = codec::payload_layout(tensor.encoding, rows, cols)?;
    let defaults = QuantMeta::default();
    if tensor.encoding == Encoding::Q4Cp {
        let mut bytes = [0u8; 64];
        for (i, &value) in defaults.codebook.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        writer.write_at(index, 0, &bytes)?;
    }
    let mut row = 0;
    while row < rows {
        let take = MATRIX_ROWS.min(rows - row);
        let (raw, values) = read_source_values(snapshot, info, (row * cols) as u64, take * cols)?;
        hash.update(raw);
        let (su, sv) = rotation_slice(tensor, vectors, row, take)?;
        let meta = QuantMeta {
            su,
            sv,
            threads: options.threads,
            ht_beam: options.ht_beam,
            ht_backend: options.ht_backend,
            ..QuantMeta::default()
        };
        let block = codec::encode_rows(tensor.encoding, &values, take, cols, &meta)
            .with_context(|| format!("encode {} rows {row}..{}", tensor.name, row + take))?;
        if matches!(tensor.encoding, Encoding::Q4Cp | Encoding::I4r) {
            ensure!(
                block.codes.len() == take * cols / 2
                    && block.scales.len() as u64 == take as u64 * layout.scale_row_stride,
                "codec planar size mismatch"
            );
            writer.write_at(
                index,
                layout.codes_offset + (row * cols / 2) as u64,
                &block.codes,
            )?;
            writer.write_at(
                index,
                layout.scales_offset + row as u64 * layout.scale_row_stride,
                &block.scales,
            )?;
        } else {
            let stride = layout.len / rows as u64;
            ensure!(
                block.scales.is_empty() && block.codes.len() as u64 == take as u64 * stride,
                "codec block size mismatch"
            );
            writer.write_at(index, row as u64 * stride, &block.codes)?;
        }
        row += take;
    }
    Ok(format!("{:x}", hash.finalize()))
}

pub(crate) fn direct_copy(encoding: Encoding, dtype: Dtype) -> bool {
    matches!(
        (encoding, dtype),
        (Encoding::Bf16, Dtype::Bf16) | (Encoding::F32, Dtype::F32) | (Encoding::I64, Dtype::I64)
    )
}

fn encode_gate_up(
    snapshot: &dyn TensorSource,
    info: &TensorInfo,
    tensor: &TensorPlan,
    index: usize,
    writer: &mut Writer,
    vectors: &BTreeMap<String, Vec<f32>>,
    options: &Options,
) -> Result<String> {
    let experts = usize::try_from(tensor.shape[0])?;
    let expert_rows = usize::try_from(tensor.shape[1])?;
    let cols = usize::try_from(tensor.shape[2])?;
    let half_rows = expert_rows / 2;
    ensure!(
        expert_rows % 256 == 0,
        "gate/up requires complete paired bands"
    );
    let pair_bytes = codec::payload_layout(tensor.encoding, 256, cols)?.len;
    let buffered = half_rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(info.dtype.byte_width() as usize))
        .context("expert source buffer size overflow")?;
    // Hash in canonical gate-then-up order, although the encoded payload pairs
    // corresponding bands. Retain only one expert's up bytes, not the matrix.
    let mut up_bytes = Vec::with_capacity(buffered);
    let mut hash = Sha256::new();
    let mut offset = 0;
    for expert in 0..experts {
        up_bytes.clear();
        for band in (0..half_rows).step_by(MATRIX_ROWS) {
            let gate_row = expert * expert_rows + band;
            let up_row = gate_row + half_rows;
            let (gate_raw, gate) =
                read_source_values(snapshot, info, (gate_row * cols) as u64, MATRIX_ROWS * cols)?;
            let (up_raw, up) =
                read_source_values(snapshot, info, (up_row * cols) as u64, MATRIX_ROWS * cols)?;
            hash.update(gate_raw);
            up_bytes.extend_from_slice(&up_raw);
            let (su, gate_sv) = rotation_slice(tensor, vectors, gate_row, MATRIX_ROWS)?;
            let (_, up_sv) = rotation_slice(tensor, vectors, up_row, MATRIX_ROWS)?;
            let gate_meta = QuantMeta {
                su,
                sv: gate_sv,
                threads: options.threads,
                ht_beam: options.ht_beam,
                ht_backend: options.ht_backend,
                ..QuantMeta::default()
            };
            let up_meta = QuantMeta {
                sv: up_sv,
                ..gate_meta
            };
            let bytes = codec::encode_ht_gate_up_bands(
                tensor.encoding,
                &gate,
                &up,
                cols,
                &gate_meta,
                &up_meta,
            )?;
            ensure!(
                bytes.len() as u64 == pair_bytes,
                "paired HT encoder size mismatch"
            );
            writer.write_at(index, offset, &bytes)?;
            offset = offset
                .checked_add(pair_bytes)
                .context("expert payload offset overflow")?;
        }
        ensure!(
            up_bytes.len() == buffered,
            "incomplete expert source coverage"
        );
        hash.update(&up_bytes);
    }
    ensure!(
        offset == writer.layout.entries[index].size,
        "incomplete paired HT payload"
    );
    Ok(format!("{:x}", hash.finalize()))
}

pub fn rotation_slice<'a>(
    tensor: &TensorPlan,
    vectors: &'a BTreeMap<String, Vec<f32>>,
    row: usize,
    take: usize,
) -> Result<(&'a [f32], &'a [f32])> {
    match &tensor.rotation {
        None => Ok((&[], &[])),
        Some(rotation) => {
            let su = vectors
                .get(&rotation.su)
                .context("missing input rotation")?;
            let sv = vectors
                .get(&rotation.sv)
                .context("missing output rotation")?;
            let (rows, cols) = rows_cols(&tensor.shape)?;
            let end = row.checked_add(take).context("rotation row overflow")?;
            ensure!(take > 0 && end <= rows, "rotation band exceeds matrix");
            let experts = tensor.shape.len() == 3;
            let expert_rows = if experts {
                usize::try_from(tensor.shape[1])?
            } else {
                rows
            };
            let expert = row / expert_rows;
            ensure!(
                (end - 1) / expert_rows == expert,
                "rotation band crosses expert boundary"
            );
            let su_offset = if experts
                && tensor.encoding.ht_bits().is_some()
                && tensor.name.ends_with(".down_proj.weight")
            {
                expert
                    .checked_mul(cols)
                    .context("input rotation offset overflow")?
            } else {
                0
            };
            let su = su
                .get(su_offset..su_offset + cols)
                .context("input rotation exceeds vector")?;
            let local_row = if tensor.encoding == Encoding::I4r {
                row % expert_rows
            } else {
                row
            };
            Ok((
                su,
                sv.get(local_row..local_row + take)
                    .context("rotation band exceeds matrix")?,
            ))
        }
    }
}

fn encode_table(
    snapshot: &dyn TensorSource,
    tensor: &TensorPlan,
    shards: &[String],
    index: usize,
    writer: &mut Writer,
    hashes: &mut BTreeMap<String, String>,
    options: &Options,
) -> Result<()> {
    let cols = usize::try_from(tensor.shape[2])?;
    let shard_rows = usize::try_from(tensor.shape[1])?;
    let mut absmax = 0.0f32;
    for name in shards {
        let info = &snapshot.tensors()[name];
        let mut row = 0;
        while row < shard_rows {
            let take = TABLE_ROWS.min(shard_rows - row);
            let (_, values) = read_source_values(snapshot, info, (row * cols) as u64, take * cols)?;
            for value in values {
                absmax = absmax.max(value.abs());
            }
            row += take;
        }
    }
    let scale = if absmax == 0.0 { 1.0 } else { absmax / 448.0 };
    ensure!(scale.is_finite() && scale > 0.0, "invalid fp8 global scale");
    let meta = QuantMeta {
        fp8_scale: scale,
        threads: options.threads,
        ht_beam: options.ht_beam,
        ht_backend: options.ht_backend,
        ..QuantMeta::default()
    };
    let mut output_offset = 0;
    for name in shards {
        let info = &snapshot.tensors()[name];
        let mut hash = Sha256::new();
        let mut row = 0;
        while row < shard_rows {
            let take = TABLE_ROWS.min(shard_rows - row);
            let (raw, values) =
                read_source_values(snapshot, info, (row * cols) as u64, take * cols)?;
            hash.update(raw);
            let block = codec::encode_rows(Encoding::Fp8G128, &values, take, cols, &meta)?;
            ensure!(
                block.codes.len() == take * cols && block.scales.is_empty(),
                "FP8 codec size mismatch"
            );
            writer.write_at(index, output_offset, &block.codes)?;
            output_offset += block.codes.len() as u64;
            row += take;
        }
        hashes.insert(name.clone(), format!("{:x}", hash.finalize()));
    }
    let layout = codec::payload_layout(
        tensor.encoding,
        shards
            .len()
            .checked_mul(shard_rows)
            .context("table rows overflow")?,
        cols,
    )?;
    writer.write_at(index, layout.len - 4, &scale.to_le_bytes())?;
    Ok(())
}

pub fn planned_bytes(plan: &ConversionPlan) -> Result<u64> {
    plan.files.iter().try_fold(0u64, |total, file| {
        total
            .checked_add(container::layout(file)?.file_size)
            .context("output size overflow")
    })
}

pub fn asset_destinations(plan: &ConversionPlan) -> Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    let mut names = BTreeSet::new();
    for name in &plan.assets {
        safe_filename(name)?;
        ensure!(names.insert(name), "duplicate source asset");
        ensure!(
            ASSET_NAMES.contains(&name.as_str()),
            "unrecognized asset: {name}"
        );
        if name == "config.json"
            || name.ends_with("processor_config.json")
            || name == "preprocessor_config.json"
        {
            result.push((name.clone(), name.clone()));
        }
        if !name.ends_with("processor_config.json") && name != "preprocessor_config.json" {
            result.push((name.clone(), format!("tokenizer/{name}")));
        }
    }
    result.sort();
    Ok(result)
}

fn copy_assets(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    staging: &Path,
) -> Result<Vec<AssetRecord>> {
    let mut records = Vec::new();
    for (source, destination) in asset_destinations(plan)? {
        let path = snapshot.asset_path(&source)?;
        ensure!(
            fs::symlink_metadata(&path)?.is_file()
                && !fs::symlink_metadata(&path)?.file_type().is_symlink(),
            "asset must be a regular non-symlink file"
        );
        let before = fs::metadata(&path)?;
        let target = staging.join(&destination);
        fs::create_dir_all(target.parent().context("asset has no parent")?)?;
        let mut input = open_no_follow(&path)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)?;
        let mut hash = Sha256::new();
        let mut bytes = vec![0u8; COPY_CHUNK];
        let mut size = 0u64;
        loop {
            let len = input.read(&mut bytes)?;
            if len == 0 {
                break;
            }
            hash.update(&bytes[..len]);
            output.write_all(&bytes[..len])?;
            size += len as u64;
            container::advise_drop(&input, size - len as u64, len as u64);
        }
        output.sync_all()?;
        container::advise_drop(&output, 0, size);
        let after = fs::metadata(&path)?;
        ensure!(
            size == before.len()
                && before.len() == after.len()
                && before.modified()? == after.modified()?,
            "source asset changed during copy"
        );
        records.push(AssetRecord {
            source,
            destination,
            size,
            sha256: format!("{:x}", hash.finalize()),
        });
    }
    let tokenizer = staging.join("tokenizer");
    if tokenizer.is_dir() {
        File::open(tokenizer)?.sync_all()?;
    }
    Ok(records)
}

pub fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut file = open_no_follow(path)?;
    ensure!(file.metadata()?.is_file(), "not a regular file");
    let before = file.metadata()?;
    let mut hash = Sha256::new();
    let mut bytes = vec![0u8; COPY_CHUNK];
    let mut size = 0u64;
    loop {
        let len = file.read(&mut bytes)?;
        if len == 0 {
            break;
        }
        hash.update(&bytes[..len]);
        container::advise_drop(&file, size, len as u64);
        size += len as u64;
    }
    let after = file.metadata()?;
    ensure!(
        size == before.len()
            && before.len() == after.len()
            && before.modified()? == after.modified()?,
        "file changed while hashing"
    );
    Ok((size, format!("{:x}", hash.finalize())))
}

fn open_no_follow(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    Ok(options.open(path)?)
}

pub fn destination_path(snapshot: &dyn TensorSource, output: &Path) -> Result<PathBuf> {
    ensure!(
        fs::symlink_metadata(output).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound),
        "output already exists or cannot be inspected"
    );
    let name = output
        .file_name()
        .context("output needs a directory name")?;
    ensure!(
        !name.is_empty() && name != "." && name != "..",
        "invalid output directory name"
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).context("output parent must already exist")?;
    let destination = parent.join(name);
    let inputs = snapshot.input_paths();
    for input in inputs
        .iter()
        .map(PathBuf::as_path)
        .chain(std::iter::once(snapshot.root()))
    {
        let input =
            fs::canonicalize(input).context("source input protection path is unavailable")?;
        ensure!(
            !destination.starts_with(&input) && !input.starts_with(&destination),
            "source and output paths must not overlap"
        );
    }
    Ok(destination)
}

fn safe_filename(name: &str) -> Result<()> {
    ensure!(
        !name.is_empty()
            && Path::new(name).components().count() == 1
            && matches!(
                Path::new(name).components().next(),
                Some(Component::Normal(_))
            )
            && !name.contains('\\'),
        "unsafe relative filename"
    );
    Ok(())
}

fn check_space(parent: &Path, need: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let path = CString::new(parent.as_os_str().as_bytes())?;
        let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        ensure!(
            unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } == 0,
            "cannot inspect output free space: {}",
            std::io::Error::last_os_error()
        );
        let stat = unsafe { stat.assume_init() };
        let available = (stat.f_bavail as u128) * (stat.f_frsize as u128);
        ensure!(
            available >= need as u128 + need as u128 / 20 + 64 * 1024 * 1024,
            "insufficient output space (need payloads plus 5% and 64 MiB)"
        );
    }
    #[cfg(not(unix))]
    {
        let _ = (parent, need);
    }
    Ok(())
}

fn publish_no_clobber(source: &Path, destination: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let source = CString::new(source.as_os_str().as_bytes())?;
        let destination = CString::new(destination.as_os_str().as_bytes())?;
        let result = unsafe {
            libc::syscall(
                libc::SYS_renameat2,
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .context("atomic no-clobber output publication");
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let source = CString::new(source.as_os_str().as_bytes())?;
        let destination = CString::new(destination.as_os_str().as_bytes())?;
        let result =
            unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        if result != 0 {
            return Err(std::io::Error::last_os_error())
                .context("atomic no-clobber output publication");
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (source, destination);
        bail!("this platform has no implemented atomic no-clobber directory publication");
    }
}

#[cfg(test)]
#[path = "core_profile_tests.rs"]
mod tests;
