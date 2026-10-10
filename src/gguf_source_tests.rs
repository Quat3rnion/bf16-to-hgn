use super::*;
use std::io::Write;
use tempfile::tempdir;

type FixtureTensor<'a> = (&'a str, Vec<u64>, GgmlType, Vec<u8>);

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn metadata() -> BTreeMap<String, Value> {
    let mut values = BTreeMap::new();
    values.insert("general.architecture".into(), json!("qwen4exp"));
    for (name, value) in [
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 48),
        ("ssm.state_size", 128),
        ("block_count", 48),
        ("expert_count", 512),
        ("embedding_length", 2560),
    ] {
        values.insert(format!("qwen4exp.{name}"), json!(value));
    }
    values.insert(
        "qwen4exp.ple.layer_multipliers".into(),
        json!([9_007_199_254_740_993i64, 3, 7]),
    );
    values.insert(
        "qwen4exp.ple.head_offsets".into(),
        json!((0..16).collect::<Vec<_>>()),
    );
    values.insert("qwen4exp.ple.head_vocab_sizes".into(), json!(vec![128; 16]));
    values
}

fn metadata_value(value: &Value) -> (u32, Vec<u8>) {
    let mut bytes = Vec::new();
    let ty = if let Some(value) = value.as_str() {
        string(&mut bytes, value);
        8
    } else if let Some(values) = value.as_array() {
        let string_elements = values.first().is_some_and(Value::is_string);
        bytes.extend_from_slice(&(if string_elements { 8u32 } else { 11u32 }).to_le_bytes());
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            if string_elements {
                string(&mut bytes, value.as_str().unwrap());
            } else {
                bytes.extend_from_slice(&value.as_i64().unwrap().to_le_bytes());
            }
        }
        9
    } else {
        bytes.extend_from_slice(&value.as_i64().unwrap().to_le_bytes());
        11
    };
    (ty, bytes)
}

struct Fixture {
    bytes: Vec<u8>,
    type_fields: Vec<usize>,
    offset_fields: Vec<usize>,
    dimension_fields: Vec<usize>,
    data_start: usize,
}

/// Serialize the actual public GGUF wire format, not the importer's structs.
fn fixture(
    version: u32,
    metadata: &BTreeMap<String, Value>,
    tensors: &[FixtureTensor<'_>],
) -> Fixture {
    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32) as usize;
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        string(&mut bytes, key);
        let (ty, payload) = metadata_value(value);
        bytes.extend_from_slice(&ty.to_le_bytes());
        bytes.extend_from_slice(&payload);
    }
    let mut type_fields = Vec::new();
    let mut offset_fields = Vec::new();
    let mut dimension_fields = Vec::new();
    let mut payload = Vec::new();
    for (name, dimensions, ty, data) in tensors {
        string(&mut bytes, name);
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        dimension_fields.push(bytes.len());
        for dimension in dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        type_fields.push(bytes.len());
        bytes.extend_from_slice(&(*ty as u32).to_le_bytes());
        offset_fields.push(bytes.len());
        let start = payload.len().div_ceil(alignment) * alignment;
        payload.resize(start, 0);
        bytes.extend_from_slice(&(start as u64).to_le_bytes());
        payload.extend_from_slice(data);
    }
    let data_start = bytes.len().div_ceil(alignment) * alignment;
    bytes.resize(data_start, 0);
    bytes.extend_from_slice(&payload);
    Fixture {
        bytes,
        type_fields,
        offset_fields,
        dimension_fields,
        data_start,
    }
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

#[test]
fn tiny_genuine_gguf_smoke() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("tiny.gguf");
    let mut meta = metadata();
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        json!(["unused", "metadata", "is skipped"]),
    );
    let fixture = fixture(
        3,
        &meta,
        &[
            (
                "blk.0.ssm_a",
                vec![48],
                GgmlType::F32,
                floats((1..=48).map(|n| -(n as f32))),
            ),
            (
                "blk.0.ssm_dt.bias",
                vec![48],
                GgmlType::F32,
                floats((0..48).map(|n| n as f32 + 0.125)),
            ),
            (
                "blk.3.attn_q_norm.weight",
                vec![256],
                GgmlType::F32,
                floats((0..256).map(|n| 1.0 + n as f32 / 1024.0)),
            ),
        ],
    );
    fs::write(&path, &fixture.bytes).unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    let a = "model.language_model.layers.0.linear_attn.A_log";
    let dt = "model.language_model.layers.0.linear_attn.dt_bias";
    let norm = "model.language_model.layers.3.self_attn.q_norm.weight";
    assert_eq!(source.tensors()[a].dtype, Dtype::F32);
    assert_eq!(source.tensors()[a].shape, [48]);
    // Explicit independent grouped-head ordering, not an inverse helper call.
    let order = [
        0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8,
        24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
    ];
    assert_eq!(
        source.read_f32(dt, 0, 48).unwrap(),
        order.map(|n| n as f32 + 0.125)
    );
    assert_eq!(
        source.read_f32(a, 0, 48).unwrap(),
        order.map(|n| ((n + 1) as f64).ln() as f32)
    );
    assert_eq!(
        source.read_f32(norm, 1, 3).unwrap(),
        [1.0 / 1024.0, 2.0 / 1024.0, 3.0 / 1024.0]
    );
    let full = source.read_f32(dt, 0, 48).unwrap();
    let canonical = floats(full);
    assert_eq!(source.read_bytes(dt, 3, 91).unwrap(), canonical[3..94]);
    assert_eq!(
        source.hash_tensor(dt).unwrap(),
        format!("{:x}", Sha256::digest(&canonical))
    );
    let ple = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
    assert_eq!(
        source.read_bytes(ple, 0, 8).unwrap(),
        9_007_199_254_740_993i64.to_le_bytes()
    );
    assert!(source.read_f32(ple, 0, 1).is_err());
    let provenance = source.provenance().unwrap();
    assert_eq!(
        provenance["shards"][0]["header_sha256"],
        format!("{:x}", Sha256::digest(&fixture.bytes[..fixture.data_start]))
    );
    let raw_dt = floats((0..48).map(|n| n as f32 + 0.125));
    assert_eq!(
        provenance["physical_tensors"][1]["raw_payload_sha256"],
        format!("{:x}", Sha256::digest(raw_dt))
    );
    assert!(!provenance
        .to_string()
        .contains(temp.path().to_str().unwrap()));
    assert_eq!(
        source.provenance().unwrap(),
        GgufSidecar::open(&path).unwrap().provenance().unwrap()
    );
    assert!(source.asset_path("config.json").is_err());
    assert!(source.read_f32(dt, 48, 1).is_err());
    assert!(source.read_f32(dt, u64::MAX, 1).is_err());
    assert!(source.read_bytes(dt, 191, 2).is_err());
    assert!(source.read_bytes(dt, 192, 0).unwrap().is_empty());
    source.check_unchanged().unwrap();
}

#[test]
fn gguf2_split_files_and_duplicate_names() {
    let temp = tempdir().unwrap();
    let first = temp.path().join("tiny-00001-of-00002.gguf");
    let second = temp.path().join("tiny-00002-of-00002.gguf");
    for (index, path) in [&first, &second].into_iter().enumerate() {
        let mut meta = metadata();
        meta.insert("split.no".into(), json!(index));
        meta.insert("split.count".into(), json!(2));
        meta.insert("split.tensors.count".into(), json!(2));
        meta.insert("general.alignment".into(), json!(64));
        let name = if index == 0 {
            "blk.0.ssm_norm.weight"
        } else {
            "blk.1.ssm_norm.weight"
        };
        fs::write(
            path,
            fixture(
                2,
                &meta,
                &[(name, vec![128], GgmlType::F32, floats([index as f32; 128]))],
            )
            .bytes,
        )
        .unwrap();
    }
    let source = GgufSidecar::open(&second).unwrap();
    assert_eq!(source.0.shards.len(), 2);
    assert_eq!(
        source
            .read_f32(
                "model.language_model.layers.1.linear_attn.norm.weight",
                127,
                1
            )
            .unwrap(),
        [1.0]
    );
    assert!(source.input_paths().contains(&first));
    let mut duplicate = metadata();
    duplicate.insert("split.no".into(), json!(1));
    duplicate.insert("split.count".into(), json!(2));
    fs::write(
        &second,
        fixture(
            2,
            &duplicate,
            &[(
                "blk.0.ssm_norm.weight",
                vec![128],
                GgmlType::F32,
                floats([0.0; 128]),
            )],
        )
        .bytes,
    )
    .unwrap();
    assert!(format!("{:#}", GgufSidecar::open(&first).unwrap_err()).contains("multiple shards"));
}

#[test]
fn indexer_fusion_reads_across_real_split_shards() {
    let temp = tempdir().unwrap();
    let first = temp.path().join("indexer-00001-of-00002.gguf");
    let second = temp.path().join("indexer-00002-of-00002.gguf");
    for (part, path) in [&first, &second].into_iter().enumerate() {
        let mut meta = metadata();
        meta.insert("split.no".into(), json!(part));
        meta.insert("split.count".into(), json!(2));
        let (name, rows, word) = if part == 0 {
            ("blk.3.indexer.q_proj.weight", 512, [0x00, 0x40])
        } else {
            ("blk.3.indexer.k_proj.weight", 128, [0x40, 0x40])
        };
        let payload = word.repeat(rows * 2560);
        fs::write(
            path,
            fixture(
                3,
                &meta,
                &[(name, vec![2560, rows as u64], GgmlType::Bf16, payload)],
            )
            .bytes,
        )
        .unwrap();
    }
    let source = GgufSidecar::open(&first).unwrap();
    let name = "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight";
    assert_eq!(source.tensors()[name].shape, [640, 2560]);
    assert_eq!(
        source.read_f32(name, 512 * 2560 - 1, 3).unwrap(),
        [2.0, 3.0, 3.0]
    );
}

#[test]
fn physical_bounds_unknown_types_names_and_metadata_are_rejected() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("bad.gguf");
    let base = fixture(
        3,
        &metadata(),
        &[
            (
                "blk.0.ssm_norm.weight",
                vec![128],
                GgmlType::F32,
                floats([0.0; 128]),
            ),
            (
                "blk.1.ssm_norm.weight",
                vec![128],
                GgmlType::F32,
                floats([0.0; 128]),
            ),
        ],
    );
    let mut variants = Vec::new();
    let mut truncated = base.bytes.clone();
    truncated.pop();
    variants.push(truncated);
    let mut unsupported = base.bytes.clone();
    unsupported[base.type_fields[0]..base.type_fields[0] + 4].copy_from_slice(&3u32.to_le_bytes());
    variants.push(unsupported);
    let mut overlapping = base.bytes.clone();
    overlapping[base.offset_fields[1]..base.offset_fields[1] + 8]
        .copy_from_slice(&0u64.to_le_bytes());
    variants.push(overlapping);
    let mut overflow = base.bytes.clone();
    overflow[base.dimension_fields[0]..base.dimension_fields[0] + 8]
        .copy_from_slice(&u64::MAX.to_le_bytes());
    variants.push(overflow);
    let mut offset_overflow = base.bytes.clone();
    offset_overflow[base.offset_fields[0]..base.offset_fields[0] + 8]
        .copy_from_slice(&(u64::MAX - 31).to_le_bytes());
    variants.push(offset_overflow);
    let mut rank = base.bytes.clone();
    rank[base.dimension_fields[0] - 4..base.dimension_fields[0]]
        .copy_from_slice(&5u32.to_le_bytes());
    variants.push(rank);
    let mut version = base.bytes.clone();
    version[4..8].copy_from_slice(&1u32.to_le_bytes());
    variants.push(version);
    for bytes in variants {
        fs::write(&path, bytes).unwrap();
        assert!(GgufSidecar::open(&path).is_err());
    }
    fs::write(
        &path,
        fixture(
            3,
            &metadata(),
            &[(
                "blk.0.made_up.weight",
                vec![1],
                GgmlType::F32,
                floats([1.0]),
            )],
        )
        .bytes,
    )
    .unwrap();
    assert!(format!("{:#}", GgufSidecar::open(&path).unwrap_err()).contains("unknown GGUF tensor"));
    let mut wrong_arch = metadata();
    wrong_arch.insert("general.architecture".into(), json!("qwen3next"));
    fs::write(
        &path,
        fixture(
            3,
            &wrong_arch,
            &[(
                "blk.0.ssm_norm.weight",
                vec![1],
                GgmlType::F32,
                floats([1.0]),
            )],
        )
        .bytes,
    )
    .unwrap();
    assert!(GgufSidecar::open(&path).is_err());
    let mut missing = metadata();
    missing.remove("qwen4exp.ple.head_offsets");
    fs::write(
        &path,
        fixture(
            3,
            &missing,
            &[(
                "blk.0.ssm_norm.weight",
                vec![1],
                GgmlType::F32,
                floats([1.0]),
            )],
        )
        .bytes,
    )
    .unwrap();
    assert!(format!("{:#}", GgufSidecar::open(&path).unwrap_err()).contains("head_offsets"));
    let mut absurd = b"GGUF".to_vec();
    absurd.extend_from_slice(&3u32.to_le_bytes());
    absurd.extend_from_slice(&u64::MAX.to_le_bytes());
    absurd.extend_from_slice(&0u64.to_le_bytes());
    fs::write(&path, absurd).unwrap();
    assert!(GgufSidecar::open(&path).is_err());
}

#[test]
fn raw_windows_cross_quant_blocks_without_rounding() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("q8.gguf");
    let mut payload = Vec::new();
    for scale in [0.5f32, 0.25, 2.0, -1.0] {
        payload.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
        payload.extend((0..32).map(|i| (i as i8 - 16) as u8));
    }
    fs::write(
        &path,
        fixture(
            3,
            &metadata(),
            &[("blk.3.attn_q.weight", vec![32, 4], GgmlType::Q8_0, payload)],
        )
        .bytes,
    )
    .unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.3.self_attn.q_proj.weight";
    assert_eq!(source.tensors()[name].shape, [4, 32]);
    assert_eq!(
        source.read_f32(name, 30, 5).unwrap(),
        [7.0, 7.5, -4.0, -3.75, -3.5]
    );
    assert_eq!(source.read_f32(name, 63, 3).unwrap(), [3.75, -32.0, -30.0]);
    let bytes = floats([7.0, 7.5, -4.0, -3.75, -3.5]);
    assert_eq!(
        source.read_bytes(name, 30 * 4 + 1, 18).unwrap(),
        bytes[1..19]
    );
}

#[test]
fn grouped_heads_row_column_and_conv_boundaries_have_golden_offsets() {
    let qkv = Reorder::Heads {
        prefix: 4096 * 2560,
        unit: 128 * 2560,
    };
    assert_eq!(qkv.span(4096 * 2560 - 3, 99), (4096 * 2560 - 3, 3));
    assert_eq!(qkv.span(4224 * 2560, 20), (6144 * 2560, 20));
    let conv = Reorder::Heads {
        prefix: 4096 * 4,
        unit: 128 * 4,
    };
    assert_eq!(conv.span(4352 * 4 + 3, 8), (8192 * 4 + 3, 8));
    let scalar = Reorder::Heads { prefix: 0, unit: 1 };
    assert_eq!(scalar.span(46, 20), (31, 1));
    assert_eq!(scalar.span(47, 20), (47, 1));
    assert_eq!(Reorder::Columns.span(128 + 7, 999), (2048 + 7, 121));
    assert_eq!(Reorder::Columns.span(6144 + 256, 128), (6144 + 4096, 128));
}

#[test]
fn fused_experts_and_indexer_have_independent_golden_spans() {
    let expert = Layout::Pair {
        first: 3,
        second: 8,
        half: 640 * 2560,
        experts: true,
    };
    assert_eq!(expert.span(640 * 2560 - 1, 8), (3, 640 * 2560 - 1, 1));
    assert_eq!(expert.span(640 * 2560, 8), (8, 0, 8));
    assert_eq!(expert.span(1280 * 2560, 8), (3, 640 * 2560, 8));
    assert_eq!(
        expert.span((511 * 1280 + 640) * 2560, 8),
        (8, 511 * 640 * 2560, 8)
    );
    let indexer = Layout::Pair {
        first: 2,
        second: 9,
        half: 512 * 2560,
        experts: false,
    };
    assert_eq!(indexer.span(512 * 2560 - 1, 2), (2, 512 * 2560 - 1, 1));
    assert_eq!(indexer.span(512 * 2560, 3), (9, 0, 3));
    let physical = |name: &str, shape: Vec<u64>| PhysicalTensor {
        name: name.into(),
        shard: 0,
        elements: product(&shape).unwrap(),
        shape,
        ty: GgmlType::Bf16,
        offset: 0,
        bytes: 0,
    };
    let input = vec![
        physical("blk.0.ffn_gate_exps.weight", vec![512, 640, 2560]),
        physical("blk.0.ffn_up_exps.weight", vec![512, 640, 2560]),
        physical("blk.3.indexer.q_proj.weight", vec![512, 2560]),
        physical("blk.3.indexer.k_proj.weight", vec![128, 2560]),
    ];
    let (tensors, layouts) = canonical_tensors(&input, &metadata()).unwrap();
    let experts = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    assert_eq!(tensors[experts].shape, [512, 1280, 2560]);
    assert_eq!(layouts[experts].span(640 * 2560, 1), (1, 0, 1));
    assert_eq!(
        tensors["model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight"].shape,
        [640, 2560]
    );
    assert!(canonical_tensors(&input[..1], &metadata()).is_err());
    let mut mismatch = input.clone();
    mismatch[1].ty = GgmlType::Q4_0;
    assert!(canonical_tensors(&mismatch, &metadata()).is_err());
}

#[test]
fn ngram_table_maps_all_128_canonical_shards() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("ngram.gguf");
    let payload = floats((0..128 * 2 * 160).map(|i| i as f32));
    fs::write(
        &path,
        fixture(
            3,
            &metadata(),
            &[(
                "per_layer_token_embd.weight",
                vec![160, 256],
                GgmlType::F32,
                payload,
            )],
        )
        .bytes,
    )
    .unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    let name = format!("{NGRAM_PREFIX}127.weight");
    assert_eq!(source.tensors()[&name].shape, [2, 160]);
    assert_eq!(source.read_f32(&name, 318, 2).unwrap(), [40958.0, 40959.0]);
    assert_eq!(
        source
            .tensors()
            .keys()
            .filter(|name| name.starts_with(NGRAM_PREFIX))
            .count(),
        128
    );
}

#[test]
fn nonnegative_a_and_source_replacement_are_rejected() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("immutable.gguf");
    let fixture = fixture(
        3,
        &metadata(),
        &[("blk.0.ssm_a", vec![48], GgmlType::F32, floats([0.0; 48]))],
    );
    fs::write(&path, &fixture.bytes).unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.0.linear_attn.A_log";
    assert!(source.read_f32(name, 0, 48).is_err());
    let replacement = temp.path().join("replacement.gguf");
    fs::write(&replacement, fixture.bytes).unwrap();
    fs::rename(replacement, &path).unwrap();
    assert!(source.check_unchanged().is_err());
    assert!(source.read_f32(name, 0, 1).is_err());
    let current = GgufSidecar::open(&path).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(24)
        .unwrap();
    assert!(current.read_f32(name, 0, 1).is_err());
}

#[test]
fn complete_source_requires_explicit_assets_and_missing_sidecars() {
    let path = Path::new("unopened.gguf");
    assert!(format!(
        "{:#}",
        GgufSource::open(path, &SourceOptions::default()).unwrap_err()
    )
    .contains("--assets"));
    let mut options = SourceOptions::default();
    options.assets = Some("assets".into());
    assert!(format!("{:#}", GgufSource::open(path, &options).unwrap_err()).contains("--mtp"));
    options.mtp = Some("mtp.hgn".into());
    assert!(format!("{:#}", GgufSource::open(path, &options).unwrap_err()).contains("--vision"));
}

#[test]
fn assets_are_bounded_immutable_and_reject_symlinks() {
    let temp = tempdir().unwrap();
    for name in &ASSETS[..6] {
        fs::write(
            temp.path().join(name),
            if *name == "config.json" {
                "{}"
            } else {
                "asset"
            },
        )
        .unwrap();
    }
    let assets = Assets::open(temp.path()).unwrap();
    OpenOptions::new()
        .append(true)
        .open(temp.path().join("tokenizer.json"))
        .unwrap()
        .write_all(b"changed")
        .unwrap();
    assert!(assets.check().is_err());
    #[cfg(unix)]
    {
        let symlink = temp.path().join("link.gguf");
        std::os::unix::fs::symlink(temp.path().join("config.json"), &symlink).unwrap();
        assert!(InputFile::open(&symlink).is_err());
    }
}

#[test]
fn q6_output_projection_reads_bounded_rows_and_reorders_split_blocks() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("out-projection.gguf");
    // Only one initialized row is needed. The other valid zero rows are sparse
    // file extents, so this test never creates or reads a decoded matrix.
    let mut row = vec![0u8; 24 * 210];
    for (superblock, bytes) in row.chunks_exact_mut(210).enumerate() {
        for element in 0..256 {
            let half = element / 128;
            let quarter = (element % 128) / 32;
            let lane = element % 32;
            let code = (2 * superblock + half + 16) as u8;
            bytes[half * 64 + (quarter % 2) * 32 + lane] |= (code & 15) << (4 * (quarter / 2));
            bytes[128 + half * 32 + lane] |= (code >> 4) << (2 * quarter);
        }
        bytes[192..208].fill(1);
        bytes[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let fixture = fixture(
        3,
        &metadata(),
        &[("blk.0.ssm_out.weight", vec![6144, 2560], GgmlType::Q6K, row)],
    );
    fs::write(&path, &fixture.bytes).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(fixture.data_start as u64 + 2560 * 24 * 210)
        .unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.0.linear_attn.out_proj.weight";
    let order = [
        0, 16, 32, 1, 17, 33, 2, 18, 34, 3, 19, 35, 4, 20, 36, 5, 21, 37, 6, 22, 38, 7, 23, 39, 8,
        24, 40, 9, 25, 41, 10, 26, 42, 11, 27, 43, 12, 28, 44, 13, 29, 45, 14, 30, 46, 15, 31, 47,
    ];
    let expected: Vec<f32> = order
        .into_iter()
        .flat_map(|head| [head as f32 - 16.0; 128])
        .collect();
    assert_eq!(source.read_f32(name, 0, 6144).unwrap(), expected);
    assert_eq!(source.read_f32(name, 123, 140).unwrap(), expected[123..263]);
    assert!(
        source.0.raw_hashes.borrow().is_empty(),
        "ordinary reads must not scan/hash the entire physical tensor"
    );
}

#[test]
fn quantized_norm_offsets_apply_but_plain_deltanet_norm_does_not() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("norms.gguf");
    let mut block = vec![0x00, 0x34]; // Q8_0 d=.25
    block.extend((0..32).map(|code| code as u8));
    fs::write(
        &path,
        fixture(
            3,
            &metadata(),
            &[
                (
                    "blk.3.attn_q_norm.weight",
                    vec![256],
                    GgmlType::Q8_0,
                    block.repeat(8),
                ),
                (
                    "blk.0.ssm_norm.weight",
                    vec![128],
                    GgmlType::Q8_0,
                    block.repeat(4),
                ),
            ],
        )
        .bytes,
    )
    .unwrap();
    let source = GgufSidecar::open(&path).unwrap();
    assert_eq!(
        source
            .read_f32(
                "model.language_model.layers.3.self_attn.q_norm.weight",
                0,
                3
            )
            .unwrap(),
        [-1.0, -0.75, -0.5]
    );
    assert_eq!(
        source
            .read_f32(
                "model.language_model.layers.0.linear_attn.norm.weight",
                0,
                3
            )
            .unwrap(),
        [0.0, 0.25, 0.5]
    );
}

#[test]
fn shared_expert_gate_normalizes_equivalent_singleton_axes() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("shared-gate.gguf");
    let canonical = "model.language_model.layers.0.mlp.shared_expert_gate.weight";
    for dimensions in [
        vec![2560],
        vec![2560, 1],
        vec![1, 2560],
        vec![1, 2560, 1],
        vec![1, 1, 2560, 1],
    ] {
        fs::write(
            &path,
            fixture(
                3,
                &metadata(),
                &[(
                    "blk.0.ffn_gate_inp_shexp.weight",
                    dimensions,
                    GgmlType::F32,
                    floats((0..2560).map(|element| element as f32 / 1024.0)),
                )],
            )
            .bytes,
        )
        .unwrap();
        let source = GgufSidecar::open(&path).unwrap();
        assert_eq!(source.tensors()[canonical].shape, [1, 2560]);
        assert_eq!(
            source.read_f32(canonical, 1279, 3).unwrap(),
            [1279.0 / 1024.0, 1280.0 / 1024.0, 1281.0 / 1024.0]
        );
    }
    // Equal element count is not sufficient: non-singleton axes cannot move,
    // merge, or silently acquire a different size.
    for dimensions in [
        vec![1280, 2],
        vec![1280, 1, 2],
        vec![2561, 1],
        vec![2560, 2],
    ] {
        let elements = product(&dimensions).unwrap() as usize;
        fs::write(
            &path,
            fixture(
                3,
                &metadata(),
                &[(
                    "blk.0.ffn_gate_inp_shexp.weight",
                    dimensions,
                    GgmlType::F32,
                    floats(vec![0.0; elements]),
                )],
            )
            .bytes,
        )
        .unwrap();
        assert!(format!("{:#}", GgufSidecar::open(&path).unwrap_err())
            .contains("shared expert gate shape"));
    }
}

#[test]
fn convolution_normalizes_only_equivalent_singleton_axes() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("convolution.gguf");
    let canonical = "model.language_model.layers.0.linear_attn.conv1d.weight";
    for dimensions in [vec![4, 10240], vec![4, 1, 10240], vec![1, 4, 1, 10240]] {
        fs::write(
            &path,
            fixture(
                3,
                &metadata(),
                &[(
                    "blk.0.ssm_conv1d.weight",
                    dimensions,
                    GgmlType::F32,
                    floats((0..40960).map(|element| element as f32)),
                )],
            )
            .bytes,
        )
        .unwrap();
        let source = GgufSidecar::open(&path).unwrap();
        assert_eq!(source.tensors()[canonical].shape, [10240, 1, 4]);
        assert_eq!(
            source.read_f32(canonical, 4224 * 4, 2).unwrap(),
            [24576.0, 24577.0]
        );
    }
    fs::write(
        &path,
        fixture(
            3,
            &metadata(),
            &[(
                "blk.0.ssm_conv1d.weight",
                vec![2, 2, 10240],
                GgmlType::F32,
                floats(vec![0.0; 40960]),
            )],
        )
        .bytes,
    )
    .unwrap();
    assert!(
        format!("{:#}", GgufSidecar::open(&path).unwrap_err()).contains("non-singleton dimensions")
    );
}
