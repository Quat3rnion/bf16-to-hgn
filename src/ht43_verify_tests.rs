use super::*;
use crate::Rotation;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

fn words(count: usize) -> Vec<u8> {
    (1..=count)
        .flat_map(|w| 0x9e3779b9u32.wrapping_mul(w as u32).to_le_bytes())
        .collect()
}

#[test]
fn independent_ht_oracle_matches_published_full_rotation_golden() {
    let su: Vec<f32> = (0..128)
        .map(|c| if (5 * c) % 7 < 3 { -1.0 } else { 1.0 })
        .collect();
    let sv: Vec<f32> = (0..128)
        .map(|r| (8 + r % 8) as f32 / 512.0 * if r % 3 == 0 { -1.0 } else { 1.0 })
        .collect();
    let values = decode_ht_independent(&words(2048), 4, 128, None, &su, &sv).unwrap();
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        "1a1f0f4bee42296506b55f20ae44b69a6b558ee16bf5c70ec9519fa52cbfc1b2"
    );
    assert_eq!(values[0], -0.0005001630634069443);
    assert_eq!(values[129], 0.006169011816382408);
}

#[test]
fn independent_ht3_oracle_matches_direct_fp64_sum_at_word_and_tile_boundaries() {
    let bytes = words(1536);
    let su = vec![1.0; 128];
    let sv = vec![0.5; 128];
    let decoded = decode_ht_independent(&bytes, 3, 128, None, &su, &sv).unwrap();
    // Reference deliberately reads each of the 16 state bits individually and
    // evaluates the complete mathematical sum, not a fast Hadamard transform.
    for (row, col) in [
        (0usize, 0usize),
        (1, 1),
        (15, 8),
        (16, 16),
        (64, 64),
        (127, 127),
    ] {
        let mut sum = 0.0f64;
        for i in 0..128 {
            for j in 0..128 {
                let tile = (j / 16) * 8 + i / 16;
                let value = (j % 16 / 8) * 128 + (i % 16) * 8 + j % 8;
                let mut state = 0u32;
                for bit in 0..16 {
                    let position = (3 * (value + 1) + 768 - 16 + bit) % 768;
                    let offset = tile * 96 + position / 32 * 4;
                    let word = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                    state = (state << 1) | ((word >> (31 - position % 32)) & 1);
                }
                let mixed = state.wrapping_mul(0x83dcd12d);
                let a: i32 = mixed.to_le_bytes().iter().map(|&v| i32::from(v)).sum();
                let value = half::f16::from_f64(f64::from(1774 * a - 905216) / 262144.0).to_f64();
                let parity = ((row & i).count_ones() + (col & j).count_ones()) & 1;
                sum += if parity == 0 { value } else { -value };
            }
        }
        assert_eq!(decoded[row * 128 + col], (sum / 256.0) as f32);
    }
}

#[test]
fn independent_paired_oracle_distinguishes_gate_up_and_column_tiles() {
    for bits in [3, 4] {
        let cols = 256;
        let tile_bytes = bits * 32;
        let mut bytes = vec![0u8; 256 * cols * bits / 8];
        for column in 0..cols / 16 {
            bytes[(column * 16 + 8) * tile_bytes..(column + 1) * 16 * tile_bytes].fill(255);
        }
        let mut su = vec![1.0; cols];
        su[128] = -1.0;
        let sv = vec![0.5; 128];
        let gate = decode_ht_independent(&bytes, bits, cols, Some(false), &su, &sv).unwrap();
        let up = decode_ht_independent(&bytes, bits, cols, Some(true), &su, &sv).unwrap();
        assert_eq!((gate[0], gate[128]), (-221.0, 221.0));
        assert_eq!((up[0], up[128]), (-41.6875, 41.6875));
        assert_eq!(gate.iter().filter(|&&v| v != 0.0).count(), 2);
        assert_eq!(up.iter().filter(|&&v| v != 0.0).count(), 2);
        assert!(
            decode_ht_independent(&bytes[..bytes.len() - 1], bits, cols, Some(true), &su, &sv)
                .is_err()
        );
        assert!(decode_ht_independent(&bytes, bits, cols, None, &su, &sv).is_err());
    }
}

#[test]
fn independent_expert_rotation_selection_keeps_expert_offsets() {
    let vectors = BTreeMap::from([
        ("input".into(), (0..256).map(|i| i as f32).collect()),
        ("output".into(), (0..512).map(|i| i as f32).collect()),
    ]);
    let mut tensor = TensorPlan {
        name: "layers.1.mlp.experts.down_proj.weight".into(),
        shape: vec![2, 256, 128],
        encoding: Encoding::Ht3,
        origin: Origin::Tensor {
            name: "source".into(),
        },
        rotation: Some(Rotation {
            su: "input".into(),
            sv: "output".into(),
        }),
    };
    let (su, sv) = verification_rotation_slice(&tensor, &vectors, 384, 128).unwrap();
    assert_eq!(
        (su[0], su[127], sv[0], sv[127]),
        (128.0, 255.0, 384.0, 511.0)
    );
    assert!(verification_rotation_slice(&tensor, &vectors, 128, 256).is_err());
    tensor.encoding = Encoding::I4r;
    let (su, sv) = verification_rotation_slice(&tensor, &vectors, 384, 128).unwrap();
    assert_eq!((su[0], su[127], sv[0], sv[127]), (0.0, 127.0, 128.0, 255.0));
}

#[test]
#[ignore = "set HT43_SMOKE_MODEL to a read-only stock HT43 main file; bounded metadata/bands only"]
fn stock_ht43_expert_bands_match_independent_oracle() {
    let path =
        PathBuf::from(std::env::var_os("HT43_SMOKE_MODEL").expect("HT43_SMOKE_MODEL is required"));
    let source = crate::hgn_source::HgnSidecar::open(&path).unwrap();
    let mut file = File::open(&path).unwrap();
    let mut header = [0u8; 104];
    file.read_exact(&mut header).unwrap();
    let number = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap());
    let count = number(&header[8..16]) as usize;
    assert!(count > 0 && count < 8192);
    file.seek(SeekFrom::Start(number(&header[16..24]))).unwrap();
    let mut table = vec![0u8; count * 160];
    file.read_exact(&mut table).unwrap();
    let mut entries = BTreeMap::new();
    for entry in table.chunks_exact(160) {
        let end = entry[..96].iter().position(|&b| b == 0).unwrap();
        let name = std::str::from_utf8(&entry[..end]).unwrap().to_owned();
        let offset = number(&entry[136..144]);
        let size = number(&entry[144..152]);
        let variant = u32::from_le_bytes(entry[156..160].try_into().unwrap());
        assert!(entries.insert(name, (offset, size, variant)).is_none());
    }
    let mut read = |name: &str, relative: usize, len: usize| {
        let &(offset, size, _) = entries.get(name).unwrap();
        assert!(relative as u64 + len as u64 <= size);
        file.seek(SeekFrom::Start(offset + relative as u64))
            .unwrap();
        let mut bytes = vec![0u8; len];
        file.read_exact(&mut bytes).unwrap();
        bytes
    };
    for layer in [0, 1] {
        let prefix = format!("layers.{layer}.mlp.experts");
        let expert = 3;
        for (projection, rows, cols, band) in
            [("gate_up_proj", 1280, 2560, 2), ("down_proj", 2560, 640, 1)]
        {
            let name = format!("{prefix}.{projection}.weight");
            let bits = match entries[&name].2 {
                0x1206 => 3,
                0x1208 => 4,
                other => panic!("unexpected HT variant {other:x}"),
            };
            let paired = projection == "gate_up_proj";
            let band_rows = if paired { 256 } else { 128 };
            let offset = (expert * rows + band * band_rows) * cols * bits / 8;
            let bytes = read(&name, offset, band_rows * cols * bits / 8);
            let input_name = if paired {
                format!("{prefix}.su")
            } else {
                format!("{prefix}.down_proj.suh")
            };
            let input_offset = if paired { 0 } else { expert * cols * 2 };
            let input = read(&input_name, input_offset, cols * 2);
            let su: Vec<f32> = input
                .chunks_exact(2)
                .map(|v| half::f16::from_le_bytes(v.try_into().unwrap()).to_f32())
                .collect();
            for up in [false, true].into_iter().take(if paired { 2 } else { 1 }) {
                let row = band * 128 + if up { 640 } else { 0 };
                let output = read(
                    &format!("{prefix}.{projection}.svh"),
                    (expert * rows + row) * 2,
                    256,
                );
                let sv: Vec<f32> = output
                    .chunks_exact(2)
                    .map(|v| half::f16::from_le_bytes(v.try_into().unwrap()).to_f32())
                    .collect();
                let expected =
                    decode_ht_independent(&bytes, bits, cols, paired.then_some(up), &su, &sv)
                        .unwrap();
                let canonical = format!("model.language_model.{prefix}.{projection}");
                let actual = source
                    .read_f32(
                        &canonical,
                        ((expert * rows + row) * cols) as u64,
                        128 * cols,
                    )
                    .unwrap();
                assert_eq!(actual, expected, "{name}, expert {expert}, row {row}");
            }
        }
    }
}
