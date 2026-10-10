use super::*;
use half::{bf16, f16};
use std::io::Write;
use tempfile::TempDir;

struct Tensor {
    name: String,
    code: u32,
    variant: u32,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

fn tensor(name: &str, code: u32, variant: u32, shape: &[u64], bytes: Vec<u8>) -> Tensor {
    Tensor {
        name: name.into(),
        code,
        variant,
        shape: shape.to_vec(),
        bytes,
    }
}

fn halves(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect()
}

fn xor(bytes: &[u8]) -> u32 {
    bytes.chunks(4).fold(0, |result, word| {
        let mut data = [0; 4];
        data[..word.len()].copy_from_slice(word);
        result ^ u32::from_le_bytes(data)
    })
}

fn padded(value: usize) -> usize {
    (value + 63) & !63
}

// Independently serialize the specified container, not container::Writer. A
// non-engine identity prevents small fixtures being mistaken for real models.
fn fixture(tensors: &[Tensor], reversed: bool) -> (TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("fixture.hgn");
    write_fixture(&path, tensors, reversed);
    (temp, path)
}

fn write_fixture(path: &Path, tensors: &[Tensor], reversed: bool) {
    let data_start = padded(104 + tensors.len() * 160);
    let mut bytes = vec![0u8; data_start];
    bytes[..4].copy_from_slice(b"HGN1");
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    bytes[8..16].copy_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes[16..24].copy_from_slice(&104u64.to_le_bytes());
    bytes[24..32].copy_from_slice(&(data_start as u64).to_le_bytes());
    let identity = b"hgn-import-fixture";
    bytes[40..40 + identity.len()].copy_from_slice(identity);
    let mut order: Vec<_> = (0..tensors.len()).collect();
    if reversed {
        order.reverse();
    }
    for index in order {
        let tensor = &tensors[index];
        bytes.resize(padded(bytes.len()), 0);
        let offset = bytes.len();
        let entry = &mut bytes[104 + index * 160..104 + (index + 1) * 160];
        entry[..tensor.name.len()].copy_from_slice(tensor.name.as_bytes());
        entry[96..100].copy_from_slice(&tensor.code.to_le_bytes());
        entry[100..104].copy_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for (dimension, value) in tensor.shape.iter().enumerate() {
            entry[104 + dimension * 8..112 + dimension * 8].copy_from_slice(&value.to_le_bytes());
        }
        entry[136..144].copy_from_slice(&(offset as u64).to_le_bytes());
        entry[144..152].copy_from_slice(&(tensor.bytes.len() as u64).to_le_bytes());
        entry[152..156].copy_from_slice(&xor(&tensor.bytes).to_le_bytes());
        entry[156..160].copy_from_slice(&tensor.variant.to_le_bytes());
        bytes.extend_from_slice(&tensor.bytes);
    }
    let size = bytes.len() as u64;
    bytes[32..40].copy_from_slice(&size.to_le_bytes());
    fs::write(path, bytes).unwrap();
}

fn f32_tensor(name: &str, values: &[f32]) -> Tensor {
    tensor(
        name,
        1,
        0,
        &[values.len() as u64],
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
    )
}

#[test]
fn tiny_importer_smoke_preserves_f32_i64_and_canonical_names() {
    let values = [1.2345678f32, -0.33333334, 0.0, 2.125];
    let integers = [i64::MAX, -(1i64 << 54) - 1, 1234567890123456789];
    let exact: Vec<_> = integers
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let tensors = [
        f32_tensor("layers.0.linear_attn.A_log", &values),
        tensor(
            "layers.1.ple.ple_embedding.layer_multipliers",
            4,
            0,
            &[3],
            exact.clone(),
        ),
        tensor(
            "mtp.layers.0.mlp.experts.down_proj.weight",
            0,
            0,
            &[1, 1, 2],
            [1.5f32, -2.0]
                .iter()
                .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
                .collect(),
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.0.linear_attn.A_log";
    assert_eq!(source.read_f32(name, 0, values.len()).unwrap(), values);
    let expected: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(source.read_bytes(name, 1, 11).unwrap(), expected[1..12]);
    assert_eq!(
        source.hash_tensor(name).unwrap(),
        format!("{:x}", Sha256::digest(&expected))
    );
    assert_eq!(source.tensors()[name].dtype, Dtype::F32);
    let integer_name = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers";
    assert_eq!(source.read_bytes(integer_name, 0, 24).unwrap(), exact);
    assert!(source.read_f32(integer_name, 0, 1).is_err());
    assert_eq!(
        source
            .read_f32("mtp.layers.0.mlp.experts.down_proj", 0, 2)
            .unwrap(),
        [1.5, -2.0]
    );
    assert!(source.read_f32(name, u64::MAX, 2).is_err());
    assert!(source.read_bytes(name, u64::MAX, 2).is_err());
    assert!(source.read_bytes(name, 16, 0).unwrap().is_empty());
    assert!(source.read_bytes(name, 17, 0).is_err());
    let provenance = source.provenance().unwrap();
    assert_eq!(provenance["payloads"].as_array().unwrap().len(), 3);
    assert!(!provenance
        .to_string()
        .contains(path.parent().unwrap().to_str().unwrap()));
    assert!(HgnSource::open(&path, &SourceOptions::default())
        .unwrap_err()
        .to_string()
        .contains("identity"));
}

#[test]
fn accepts_unordered_live_spans_and_appended_replacements() {
    let tensors = [
        f32_tensor("lm_head.weight", &[1.0]),
        f32_tensor("embed_tokens.weight", &[2.0]),
    ];
    let (_temp, path) = fixture(&tensors, true);
    let mut bytes = fs::read(&path).unwrap();
    // Leave the old live payload intact, followed by nonzero unreachable bytes.
    bytes.resize(padded(bytes.len()) + 128, 0x5a);
    let new_offset = bytes.len();
    let replacement = 7.125f32.to_le_bytes();
    bytes.extend_from_slice(&replacement);
    bytes[104 + 136..104 + 144].copy_from_slice(&(new_offset as u64).to_le_bytes());
    bytes[104 + 152..104 + 156].copy_from_slice(&xor(&replacement).to_le_bytes());
    let size = bytes.len() as u64;
    bytes[32..40].copy_from_slice(&size.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    let source = HgnSidecar::open(&path).unwrap();
    assert_eq!(source.read_f32("lm_head.weight", 0, 1).unwrap(), [7.125]);
    assert_eq!(
        source
            .read_f32("model.language_model.embed_tokens.weight", 0, 1)
            .unwrap(),
        [2.0]
    );
    assert_eq!(
        source.provenance().unwrap()["payloads"][0]["offset"],
        new_offset as u64
    );
}

#[test]
fn rejects_overlap_overflow_unknown_stores_and_damaged_checksums() {
    let tensors = [
        f32_tensor("lm_head.weight", &[1.0]),
        f32_tensor("embed_tokens.weight", &[2.0]),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let original = fs::read(&path).unwrap();
    for (offset, value) in [
        (104 + 160 + 136, u64le(&original, 104 + 136)),
        (104 + 136, u64::MAX - 63),
        (16, u64::MAX - 159),
    ] {
        let mut bytes = original.clone();
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(HgnSidecar::open(&path).is_err());
    }
    let mut bytes = original.clone();
    bytes[104 + 96..104 + 100].copy_from_slice(&9u32.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    assert!(HgnSidecar::open(&path)
        .unwrap_err()
        .to_string()
        .contains("unsupported HGN store"));
    let mut bytes = original;
    let payload = u64le(&bytes, 104 + 136) as usize;
    bytes[payload] ^= 1;
    fs::write(&path, bytes).unwrap();
    let source = HgnSidecar::open(&path).unwrap();
    assert!(source
        .hash_tensor("lm_head.weight")
        .unwrap_err()
        .to_string()
        .contains("checksum"));
    assert!(source.provenance().is_err());
}

#[test]
fn detects_mutations_replacements_and_symlinks() {
    let tensors = [f32_tensor("lm_head.weight", &[1.0])];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    source.hash_tensor("lm_head.weight").unwrap(); // Cache the original checksum.
    let replacement = path.with_extension("new");
    fs::copy(&path, &replacement).unwrap();
    fs::rename(&replacement, &path).unwrap();
    assert!(source.check_unchanged().is_err());
    assert!(source.read_f32("lm_head.weight", 0, 1).is_err());
    let source = HgnSidecar::open(&path).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(source.check_unchanged().is_err());
    #[cfg(unix)]
    {
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(HgnSidecar::open(&link).is_err());
    }
}

#[test]
fn decodes_planar_and_affine_formats_from_independent_bytes() {
    let mut q4 = vec![0u8; 144];
    for level in 0..16 {
        q4[level * 4..level * 4 + 4].copy_from_slice(&((level as f32 - 8.0) / 2.0).to_le_bytes());
    }
    for index in 0..32 {
        q4[64 + index] = ((2 * index) % 16) as u8 | (((2 * index + 1) % 16) as u8) << 4;
    }
    q4[128..132].copy_from_slice(&halves(&[-0.25, 0.5]));
    let mut q8: Vec<_> = (0..64u8).collect();
    q8.extend_from_slice(&halves(&[0.5, -2.0]));
    let mut q6 = vec![0u8; 64];
    for index in 0..64 {
        q6[index / 2] |= ((index & 15) as u8) << (4 * (index % 2));
        q6[32 + index / 4] |= ((index >> 4) as u8) << (2 * (index % 4));
    }
    q6[48..52].copy_from_slice(&halves(&[0.25, -3.0]));
    let tensors = [
        tensor("embed_tokens.weight", 5, 2, &[1, 64], q4),
        tensor("mtp.fc_hidden.weight", 7, 0, &[1, 64], q8),
        tensor(
            "hyper_connection_mixer.input_mix_weight_down.weight",
            24,
            64,
            &[1, 64],
            q6,
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    let expected: Vec<_> = (0..64)
        .map(|index| ((index % 16) as f32 - 8.0) / 2.0 * if index < 32 { -0.25 } else { 0.5 })
        .collect();
    assert_eq!(
        source
            .read_f32("model.language_model.embed_tokens.weight", 0, 64)
            .unwrap(),
        expected
    );
    let expected: Vec<_> = (0..64).map(|index| index as f32 * 0.5 - 2.0).collect();
    assert_eq!(
        source.read_f32("mtp.fc_hidden.weight", 0, 64).unwrap(),
        expected
    );
    let expected: Vec<_> = (0..64).map(|index| index as f32 * 0.25 - 3.0).collect();
    assert_eq!(
        source
            .read_f32(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                0,
                64
            )
            .unwrap(),
        expected
    );
}

#[test]
fn splits_ngram_table_into_exact_canonical_shard_ranges() {
    let mut bytes: Vec<_> = (0..640)
        .map(|index| [0, 0x38, 0xb8, 0x7f][index % 4])
        .collect();
    bytes.extend_from_slice(&0.125f32.to_le_bytes());
    let (_temp, path) = fixture(&[tensor(NGRAM, 10, 0, &[2, 2, 160], bytes)], false);
    let source = HgnSidecar::open(&path).unwrap();
    for part in 0..2 {
        let name = format!("{NGRAM_PREFIX}{part}.weight");
        assert_eq!(source.tensors()[&name].shape, [2, 160]);
        assert_eq!(
            source.read_f32(&name, 157, 7).unwrap(),
            [0.125, -0.125, 60.0, 0.0, 0.125, -0.125, 60.0]
        );
    }
    assert_eq!(
        source.provenance().unwrap()["payloads"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn restores_vision_patch_shape_without_reordering_or_rounding() {
    let mut bytes = vec![0u8; 1152 * 1536 * 2];
    bytes[0..2].copy_from_slice(&bf16::from_f32(1.5).to_le_bytes());
    let (_temp, path) = fixture(
        &[tensor(
            "visual.patch_embed.proj.weight",
            0,
            0,
            &[1152, 1536],
            bytes,
        )],
        false,
    );
    let source = HgnSidecar::open(&path).unwrap();
    let name = "model.visual.patch_embed.proj.weight";
    assert_eq!(source.tensors()[name].shape, [1152, 3, 2, 16, 16]);
    assert_eq!(source.read_f32(name, 0, 2).unwrap(), [1.5, 0.0]);
}

// Direct specification oracle: individual circular stream bits, independent of
// codec tile unpacking, native encoder and fast Walsh-transform implementation.
fn ht_stored(bytes: &[u8], tile: usize, value: usize, bits: usize) -> f64 {
    let tile_bytes = 32 * bits;
    let tile = &bytes[tile * tile_bytes..(tile + 1) * tile_bytes];
    let length = 256 * bits;
    let mut state = 0u16;
    for bit in 0..16 {
        let position = ((bits * (value + 1)) as isize - 16 + bit as isize)
            .rem_euclid(length as isize) as usize;
        let word = u32::from_le_bytes(
            tile[position / 32 * 4..position / 32 * 4 + 4]
                .try_into()
                .unwrap(),
        );
        state = (state << 1) | ((word >> (31 - position % 32)) & 1) as u16;
    }
    let hashed = u32::from(state).wrapping_mul(0x83dcd12d);
    let sum: i32 = hashed
        .to_le_bytes()
        .iter()
        .map(|value| i32::from(*value))
        .sum();
    f16::from_f32((1774 * sum - 905216) as f32 / 262144.0).to_f32() as f64
}

fn sign(row: usize, col: usize) -> f64 {
    if (row & col).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn ht_oracle(
    bytes: &[u8],
    expert: usize,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    bits: usize,
    paired: bool,
    su: f32,
    sv: f32,
) -> f32 {
    let mut sum = 0.0;
    for i in 0..128 {
        for j in 0..128 {
            let r = row / 128 * 128 + i;
            let c = col / 128 * 128 + j;
            let local_tile = if paired {
                ((r % (rows / 2) / 128) * (cols / 16) + c / 16) * 16
                    + r % 128 / 16
                    + 8 * (r / (rows / 2))
            } else {
                ((r / 128) * (cols / 16) + c / 16) * 8 + r % 128 / 16
            };
            let tile = expert * rows * cols / 256 + local_tile;
            let value = (c % 16 / 8) * 128 + r % 16 * 8 + c % 8;
            sum += sign(row % 128, i) * ht_stored(bytes, tile, value, bits) * sign(j, col % 128);
        }
    }
    (sum / 128.0 * f64::from(su) * f64::from(sv)) as f32
}

#[test]
fn ht3_ht4_inverse_uses_paired_gate_up_tiles_and_per_expert_scales() {
    let rows = 512;
    let cols = 256;
    let experts = 2;
    for bits in [3, 4] {
        let data: Vec<_> = (0..experts * rows * cols * bits / 8)
            .map(|index| ((index * 73 + index / 5 * 17 + 19) % 256) as u8)
            .collect();
        let su: Vec<_> = (0..cols)
            .map(|i| if i % 3 == 0 { -0.5 } else { 1.0 })
            .collect();
        let sv: Vec<_> = (0..experts * rows)
            .map(|i| (i % 11 + 1) as f32 / 16.0 * if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let base = "layers.1.mlp.experts";
        let tensors = [
            tensor(
                &format!("{base}.gate_up_proj.weight"),
                16,
                if bits == 3 { 0x1206 } else { 0x1208 },
                &[experts as u64, rows as u64, cols as u64],
                data.clone(),
            ),
            tensor(&format!("{base}.su"), 2, 0, &[cols as u64], halves(&su)),
            tensor(
                &format!("{base}.gate_up_proj.svh"),
                2,
                0,
                &[experts as u64, rows as u64],
                halves(&sv),
            ),
        ];
        let (_temp, path) = fixture(&tensors, false);
        let source = HgnSidecar::open(&path).unwrap();
        let name = "model.language_model.layers.1.mlp.experts.gate_up_proj";
        for expert in 0..experts {
            for (row, col) in [
                (0, 0),
                (127, 137),
                (128, 17),
                (255, 255),
                (256, 3),
                (383, 129),
                (511, 255),
            ] {
                let expected = ht_oracle(
                    &data,
                    expert,
                    row,
                    col,
                    rows,
                    cols,
                    bits,
                    true,
                    su[col],
                    sv[expert * rows + row],
                );
                let offset = ((expert * rows + row) * cols + col) as u64;
                assert_eq!(
                    source.read_f32(name, offset, 1).unwrap()[0].to_bits(),
                    expected.to_bits(),
                    "bits={bits} expert={expert} row={row} col={col}"
                );
            }
        }
        // One request crossing gate/up, expert and rotation-band boundaries.
        let start = (rows - 1) * cols + cols - 1;
        let actual = source.read_f32(name, start as u64, 3).unwrap();
        let expected = ht_oracle(&data, 1, 0, 1, rows, cols, bits, true, su[1], sv[rows]);
        assert_eq!(actual[2].to_bits(), expected.to_bits());
        assert_eq!(
            source.provenance().unwrap()["payloads"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }
}

#[test]
fn ht_down_inverse_indexes_each_experts_input_vector() {
    let rows = 128;
    let cols = 128;
    let data: Vec<_> = (0..2 * rows * cols * 3 / 8)
        .map(|i| (i * 31 + i / 7) as u8)
        .collect();
    let su: Vec<_> = (0..2 * cols)
        .map(|i| if i < cols { 1.0 } else { -0.5 })
        .collect();
    let sv: Vec<_> = (0..2 * rows).map(|i| (i % 7 + 1) as f32 / 8.0).collect();
    let tensors = [
        tensor(
            "layers.1.mlp.experts.down_proj.weight",
            16,
            0x1206,
            &[2, 128, 128],
            data.clone(),
        ),
        tensor(
            "layers.1.mlp.experts.down_proj.suh",
            2,
            0,
            &[2, 128],
            halves(&su),
        ),
        tensor(
            "layers.1.mlp.experts.down_proj.svh",
            2,
            0,
            &[2, 128],
            halves(&sv),
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.1.mlp.experts.down_proj";
    for expert in 0..2 {
        let (row, col) = (37, 63);
        let expected = ht_oracle(
            &data,
            expert,
            row,
            col,
            rows,
            cols,
            3,
            false,
            su[expert * cols + col],
            sv[expert * rows + row],
        );
        assert_eq!(
            source
                .read_f32(name, ((expert * rows + row) * cols + col) as u64, 1)
                .unwrap()[0]
                .to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
fn i4r_inverse_keeps_scale_plane_global_and_rotation_vectors_shared() {
    let rows = 256usize;
    let cols = 128usize;
    let experts = 2usize;
    let mut data = vec![0u8; experts * rows * cols / 2];
    for index in 0..experts * rows * cols {
        let row = index / cols;
        let col = index % cols;
        let code = ((row * 3 + col * 5 + row * col + row / 11) % 16) as u8;
        data[index / 2] |= code << (4 * (index % 2));
    }
    let scales: Vec<_> = (0..experts * rows)
        .map(|i| -((i % 5 + 1) as f32) / 16.0)
        .collect();
    data.extend_from_slice(&halves(&scales));
    let su: Vec<_> = (0..cols)
        .map(|i| if i % 2 == 0 { 0.5 } else { -1.0 })
        .collect();
    let sv: Vec<_> = (0..rows)
        .map(|i| if i % 7 == 0 { 0.0 } else { 1.5 })
        .collect();
    let tensors = [
        tensor(
            "layers.0.mlp.experts.down_proj.weight",
            23,
            128,
            &[2, 256, 128],
            data.clone(),
        ),
        tensor(
            "layers.0.mlp.experts.down_proj.suh",
            2,
            0,
            &[128],
            halves(&su),
        ),
        tensor(
            "layers.0.mlp.experts.down_proj.svh",
            2,
            0,
            &[256],
            halves(&sv),
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.0.mlp.experts.down_proj";
    for (expert, row, col) in [(0, 19, 3), (0, 146, 35), (1, 33, 71), (1, 201, 127)] {
        let mut sum = 0.0;
        for i in 0..128 {
            for j in 0..128 {
                let global_row = expert * rows + row / 128 * 128 + i;
                let index = global_row * cols + j;
                let nibble = (data[index / 2] >> (4 * (index % 2))) & 15;
                let stored = (f64::from(nibble) - 8.0) * f64::from(scales[global_row]);
                sum += sign(row % 128, i) * stored * sign(j, col);
            }
        }
        let expected = (sum / 128.0 * f64::from(su[col]) * f64::from(sv[row])) as f32;
        let actual = source
            .read_f32(name, ((expert * rows + row) * cols + col) as u64, 1)
            .unwrap()[0];
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn missing_wrong_and_orphan_rotation_vectors_fail_at_open() {
    let matrix = || tensor("lm_head.weight", 16, 0x1208, &[128, 128], vec![0; 8192]);
    let (_temp, path) = fixture(&[matrix()], false);
    assert!(HgnSidecar::open(&path)
        .unwrap_err()
        .to_string()
        .contains("missing rotation"));
    let (_temp, path) = fixture(
        &[
            matrix(),
            tensor("lm_head.suh", 2, 0, &[127], halves(&vec![1.0; 127])),
            tensor("lm_head.svh", 2, 0, &[128], halves(&vec![1.0; 128])),
        ],
        false,
    );
    assert!(HgnSidecar::open(&path).is_err());
    let (_temp, path) = fixture(
        &[
            f32_tensor("lm_head.weight", &[1.0]),
            tensor("lm_head.suh", 2, 0, &[1], halves(&[1.0])),
        ],
        false,
    );
    assert!(HgnSidecar::open(&path)
        .unwrap_err()
        .to_string()
        .contains("unreferenced"));
}

#[test]
fn assembles_sidecars_without_substitutions_and_protects_every_input() {
    let (main_temp, main_path) = fixture(&[f32_tensor("lm_head.weight", &[1.25])], false);
    let (_vision_temp, vision_path) =
        fixture(&[f32_tensor("visual.pos_embed.weight", &[2.5])], false);
    let (_mtp_temp, mtp_path) = fixture(&[f32_tensor("mtp.fc_hidden.weight", &[3.75])], false);
    let mut source = HgnSource {
        root: main_path.clone(),
        parts: Vec::new(),
        tensors: BTreeMap::new(),
        owners: BTreeMap::new(),
        config: json!({}),
        assets_root: main_temp.path().to_owned(),
        assets: BTreeMap::new(),
    };
    source
        .attach(HgnSidecar::open(&main_path).unwrap())
        .unwrap();
    source
        .attach(HgnSidecar::open(&vision_path).unwrap())
        .unwrap();
    source.attach(HgnSidecar::open(&mtp_path).unwrap()).unwrap();
    assert_eq!(source.read_f32("lm_head.weight", 0, 1).unwrap(), [1.25]);
    assert_eq!(
        source
            .read_f32("model.visual.pos_embed.weight", 0, 1)
            .unwrap(),
        [2.5]
    );
    assert_eq!(
        source.read_f32("mtp.fc_hidden.weight", 0, 1).unwrap(),
        [3.75]
    );
    for path in [&main_path, &vision_path, &mtp_path] {
        assert!(source.input_paths().contains(path));
    }
    assert!(source.attach(HgnSidecar::open(&mtp_path).unwrap()).is_err());
    assert!(source.asset_path("../config.json").is_err());
    assert!(!source.asset_path("vocab.json").unwrap().exists());
    fs::write(main_temp.path().join("vocab.json"), b"{}").unwrap();
    assert!(source.check_unchanged().is_err());
}

#[test]
fn ngram_heads_must_fit_the_assembled_table() {
    for (offset, size, valid) in [
        (0i64, 4i64, true),
        (3, 2, false),
        (-1, 1, false),
        (0, 0, false),
    ] {
        let metadata = [
            tensor(
                "layers.1.ple.ple_embedding.ngram_heads_offsets",
                4,
                0,
                &[16],
                [offset; 16]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
            tensor(
                "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
                4,
                0,
                &[16],
                [size; 16]
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            ),
        ];
        let (temp, path) = fixture(&metadata, false);
        let mut table = vec![0u8; 640];
        table.extend_from_slice(&1.0f32.to_le_bytes());
        let (_ngram_temp, ngram_path) =
            fixture(&[tensor(NGRAM, 10, 0, &[2, 2, 160], table)], false);
        let mut source = HgnSource {
            root: path.clone(),
            parts: Vec::new(),
            tensors: BTreeMap::new(),
            owners: BTreeMap::new(),
            config: json!({}),
            assets_root: temp.path().to_owned(),
            assets: BTreeMap::new(),
        };
        source.attach(HgnSidecar::open(&path).unwrap()).unwrap();
        source
            .attach(HgnSidecar::open(&ngram_path).unwrap())
            .unwrap();
        assert_eq!(source.validate_ngram_bounds().is_ok(), valid);
    }
}

#[test]
fn public_open_assembles_assets_and_existing_siblings_without_architecture_bypass() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let model = root.join("qwen38-flash-next-v2.hgn");
    write_fixture(&model, &[f32_tensor("lm_head.weight", &[1.25])], false);
    // Exercise the production source-assembly entrypoint without adding a test
    // flag to it. This metadata-only fixture is still rejected by the planner.
    let mut file = OpenOptions::new().write(true).open(&model).unwrap();
    file.seek(SeekFrom::Start(40)).unwrap();
    let mut identity = [0u8; 64];
    let native = b"qwen3.8-flash-next-v2";
    identity[..native.len()].copy_from_slice(native);
    file.write_all(&identity).unwrap();
    drop(file);
    for &name in REQUIRED_ASSETS {
        fs::write(root.join(name), b"{}").unwrap();
    }
    let ngram = root.join("qwen38-flash-next-ngram.hgn");
    let vision = root.join("qwen38-flash-next-vision.hgn");
    let mtp = root.join("qwen38-flash-next-mtp.hgn");
    // FP8G has 32 padding bytes before its trailing scale for a single row.
    let mut table = vec![0u8; 196];
    table[..160].fill(0x38);
    table[192..].copy_from_slice(&0.25f32.to_le_bytes());
    write_fixture(&ngram, &[tensor(NGRAM, 10, 0, &[1, 1, 160], table)], false);
    write_fixture(
        &vision,
        &[f32_tensor("visual.pos_embed.weight", &[2.5])],
        false,
    );
    write_fixture(&mtp, &[f32_tensor("mtp.fc_hidden.weight", &[3.75])], false);
    for input in [model.as_path(), root] {
        let source = HgnSource::open(input, &SourceOptions::default()).unwrap();
        assert_eq!(source.parts.len(), 4);
        assert_eq!(source.tensors().len(), 4);
        assert_eq!(source.config(), &json!({}));
        assert_eq!(
            source.asset_path("tokenizer.json").unwrap(),
            root.join("tokenizer.json")
        );
        assert_eq!(source.read_f32("lm_head.weight", 0, 1).unwrap(), [1.25]);
        assert_eq!(
            source
                .read_f32("model.visual.pos_embed.weight", 0, 1)
                .unwrap(),
            [2.5]
        );
        assert_eq!(
            source.read_f32("mtp.fc_hidden.weight", 0, 1).unwrap(),
            [3.75]
        );
        assert_eq!(
            source
                .read_f32(&format!("{NGRAM_PREFIX}0.weight"), 159, 1)
                .unwrap(),
            [0.25]
        );
        for path in [&model, &ngram, &vision, &mtp] {
            assert!(source.input_paths().contains(path));
        }
        assert!(crate::architecture::plan(&source, &crate::manifest::Options::default()).is_err());
        source.check_unchanged().unwrap();
    }
}

#[test]
fn shared_expert_gate_and_up_resolve_the_layer_experts_input_alias() {
    let rows = 128;
    let cols = 256;
    let gate: Vec<_> = (0..rows * cols / 2)
        .map(|i| (i * 29 + i / 9 + 17) as u8)
        .collect();
    let up: Vec<_> = (0..rows * cols / 2)
        .map(|i| (i * 53 + i / 7 + 31) as u8)
        .collect();
    let su: Vec<_> = (0..cols)
        .map(|i| if i % 3 == 0 { -0.5 } else { 1.0 })
        .collect();
    let gate_sv: Vec<_> = (0..rows).map(|i| (i % 7 + 1) as f32 / 8.0).collect();
    let up_sv: Vec<_> = (0..rows).map(|i| -((i % 5 + 1) as f32) / 16.0).collect();
    let prefix = "layers.2.mlp";
    let tensors = [
        tensor(
            &format!("{prefix}.shared_expert.gate_proj.weight"),
            16,
            0x1208,
            &[128, 256],
            gate.clone(),
        ),
        tensor(
            &format!("{prefix}.shared_expert.up_proj.weight"),
            16,
            0x1208,
            &[128, 256],
            up.clone(),
        ),
        tensor(&format!("{prefix}.experts.su"), 2, 0, &[256], halves(&su)),
        tensor(
            &format!("{prefix}.shared_expert.gate_proj.svh"),
            2,
            0,
            &[128],
            halves(&gate_sv),
        ),
        tensor(
            &format!("{prefix}.shared_expert.up_proj.svh"),
            2,
            0,
            &[128],
            halves(&up_sv),
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    assert_eq!(source.tensors().len(), 2);
    for (projection, data, sv) in [("gate", &gate, &gate_sv), ("up", &up, &up_sv)] {
        let name = format!("model.language_model.{prefix}.shared_expert.{projection}_proj.weight");
        for (row, col) in [(7, 63), (37, 145)] {
            let expected = ht_oracle(data, 0, row, col, rows, cols, 4, false, su[col], sv[row]);
            assert_eq!(
                source
                    .read_f32(&name, (row * cols + col) as u64, 1)
                    .unwrap()[0]
                    .to_bits(),
                expected.to_bits(),
                "{projection} row={row} col={col}"
            );
        }
    }
}

#[test]
fn i4r_gate_up_uses_flat_rows_across_experts_and_multiple_scale_groups() {
    let experts = 2usize;
    let rows = 512usize;
    let cols = 256usize;
    let groups = cols / 128;
    let mut codes = vec![0u8; experts * rows * cols / 2];
    for index in 0..experts * rows * cols {
        let mixed = (index as u32).wrapping_mul(0x9e3779b9) ^ (index as u32 >> 3);
        let code = (mixed.rotate_left((index / cols % 17) as u32) >> 24) as u8 & 15;
        codes[index / 2] |= code << (4 * (index % 2));
    }
    let scales: Vec<_> = (0..experts * rows * groups)
        .map(|i| -((i % 13 + 1) as f32) / 32.0)
        .collect();
    let mut payload = codes.clone();
    payload.extend_from_slice(&halves(&scales));
    let su: Vec<_> = (0..cols)
        .map(|i| if i % 2 == 0 { 0.5 } else { -1.0 })
        .collect();
    let sv: Vec<_> = (0..rows)
        .map(|i| (i % 5 + 1) as f32 / 4.0 * if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let tensors = [
        tensor(
            "layers.0.mlp.experts.gate_up_proj.weight",
            23,
            128,
            &[2, 512, 256],
            payload,
        ),
        tensor("layers.0.mlp.experts.su", 2, 0, &[256], halves(&su)),
        tensor(
            "layers.0.mlp.experts.gate_up_proj.svh",
            2,
            0,
            &[512],
            halves(&sv),
        ),
    ];
    let (_temp, path) = fixture(&tensors, false);
    let source = HgnSidecar::open(&path).unwrap();
    let name = "model.language_model.layers.0.mlp.experts.gate_up_proj";
    let expected = |expert: usize, row: usize, col: usize| {
        let mut sum = 0.0;
        for i in 0..128 {
            for j in 0..128 {
                let global_row = expert * rows + row / 128 * 128 + i;
                let global_col = col / 128 * 128 + j;
                let index = global_row * cols + global_col;
                let nibble = (codes[index / 2] >> (4 * (index % 2))) & 15;
                let scale = scales[global_row * groups + col / 128];
                let stored = (f64::from(nibble) - 8.0) * f64::from(scale);
                sum += sign(row % 128, i) * stored * sign(j, col % 128);
            }
        }
        (sum / 128.0 * f64::from(sv[row]) * f64::from(su[col])) as f32
    };
    for (expert, row, col) in [
        (0, 0, 0),
        (0, 127, 127),
        (0, 255, 127),
        (0, 256, 128),
        (0, 511, 255),
        (1, 19, 3),
        (1, 259, 129),
        (1, 398, 211),
    ] {
        let actual = source
            .read_f32(name, ((expert * rows + row) * cols + col) as u64, 1)
            .unwrap()[0];
        assert_eq!(
            actual.to_bits(),
            expected(expert, row, col).to_bits(),
            "expert={expert} row={row} col={col}"
        );
    }
    let boundary = source.read_f32(name, (rows * cols - 1) as u64, 3).unwrap();
    assert_eq!(boundary[1].to_bits(), expected(1, 0, 0).to_bits());
    assert_eq!(boundary[2].to_bits(), expected(1, 0, 1).to_bits());
}
