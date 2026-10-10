use super::*;

fn block(ty: GgmlType) -> Vec<u8> {
    vec![0; ty.block().1]
}
fn set_half(bytes: &mut [u8], at: usize, value: f32) {
    bytes[at..at + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}
fn values(ty: GgmlType, bytes: &[u8]) -> Vec<f32> {
    let mut output = vec![0.0; bytes.len() / ty.block().1 * ty.block().0];
    decode(ty, bytes, &mut output).unwrap();
    output
}

#[test]
fn raw_types_preserve_values_without_bf16_rounding() {
    let precise = f32::from_bits(0x3f800001);
    assert_eq!(values(GgmlType::F32, &precise.to_le_bytes()), [precise]);
    assert_eq!(
        values(GgmlType::F16, &[1, 0, 0, 0xbc]),
        [2.0f32.powi(-24), -1.0]
    );
    assert_eq!(
        values(GgmlType::Bf16, &[0x81, 0x3f, 0x80, 0xbf]),
        [1.0078125, -1.0]
    );
}

#[test]
fn q4_and_iq4_golden_nibble_order() {
    // Low nibbles address the first 16 values, NOT alternating even values.
    let mut bytes = vec![0x00, 0x40]; // d=2
    bytes.extend((0..16).map(|j| j | ((15 - j) << 4)));
    let q4 = values(GgmlType::Q4_0, &bytes);
    let expected: Vec<f32> = (0..16)
        .chain((0..16).rev())
        .map(|code| 2.0 * (code as f32 - 8.0))
        .collect();
    assert_eq!(q4, expected);
    let iq4 = values(GgmlType::Iq4Nl, &bytes);
    assert_eq!(
        &iq4[..16],
        &[
            -254.0, -208.0, -166.0, -130.0, -98.0, -70.0, -44.0, -20.0, 2.0, 26.0, 50.0, 76.0,
            106.0, 138.0, 178.0, 226.0
        ]
    );
    assert_eq!(iq4[16], 226.0);
    assert_eq!(iq4[31], -254.0);
}

#[test]
fn q5_golden_high_plane_and_affine_zero() {
    let mut symmetric = vec![0x00, 0x38, 0x01, 0x00, 0x01, 0x80]; // d=.5, high=0x80010001
    symmetric.extend([0xf0; 16]);
    let decoded = values(GgmlType::Q5_0, &symmetric);
    let mut expected = vec![-8.0; 16];
    expected[0] = 0.0;
    expected.extend([-0.5; 16]);
    expected[16] = 7.5;
    expected[31] = 7.5;
    assert_eq!(decoded, expected);
    let mut affine = vec![0x00, 0x38, 0x00, 0xc0]; // d=.5, m=-2
    affine.extend_from_slice(&symmetric[2..]);
    assert_eq!(
        values(GgmlType::Q5_1, &affine),
        expected.iter().map(|v| v + 6.0).collect::<Vec<_>>()
    );
    // The canonical F32 path has no direct-store Q5_0 zero-overflow restriction.
    set_half(&mut symmetric, 0, 4096.0);
    assert_eq!(values(GgmlType::Q5_0, &symmetric)[1], -65536.0);
}

#[test]
fn q8_signed_codes_include_minus_128() {
    let mut bytes = vec![0x00, 0x34]; // d=.25
    bytes.extend((0..32).map(|n| (n as i8 - 16) as u8));
    bytes[2] = 128;
    let result = values(GgmlType::Q8_0, &bytes);
    assert_eq!(result[0], -32.0);
    assert_eq!(result[1], -3.75);
    assert_eq!(result[31], 3.75);
}

#[test]
fn k_affine_golden_scales_and_high_planes() {
    // Independent logical scale values; pack with the published reverse rule.
    let scale = [1u8, 2, 3, 4, 17, 34, 51, 63];
    let minimum = [7u8, 8, 9, 10, 33, 18, 47, 60];
    for ty in [GgmlType::Q4K, GgmlType::Q5K] {
        let mut bytes = block(ty);
        set_half(&mut bytes, 0, 0.5);
        set_half(&mut bytes, 2, 0.25);
        for j in 0..4 {
            bytes[4 + j] = scale[j] | ((scale[j + 4] >> 4) << 6);
            bytes[8 + j] = minimum[j] | ((minimum[j + 4] >> 4) << 6);
            bytes[12 + j] = (scale[j + 4] & 15) | ((minimum[j + 4] & 15) << 4);
        }
        let start = if ty == GgmlType::Q5K {
            bytes[16..48].fill(0xa5);
            48
        } else {
            16
        };
        for pair in 0..4 {
            bytes[start + pair * 32..start + (pair + 1) * 32]
                .fill((2 * pair + 1) as u8 | (((2 * pair + 2) as u8) << 4));
        }
        let decoded = values(ty, &bytes);
        for group in 0..8 {
            let high = if ty == GgmlType::Q5K && 0xa5 & (1 << group) != 0 {
                16
            } else {
                0
            };
            let expected = 0.5 * f32::from(scale[group]) * (group + 1 + high) as f32
                - 0.25 * f32::from(minimum[group]);
            assert_eq!(&decoded[group * 32..(group + 1) * 32], &[expected; 32]);
        }
    }
}

#[test]
fn q6_golden_four_quarters_and_signed_scales() {
    let mut bytes = block(GgmlType::Q6K);
    bytes[..128].fill(0xf0);
    bytes[128..192].fill(0xe4); // high pairs 0,1,2,3
    for group in 0..16 {
        bytes[192 + group] = (group as i8 - 8) as u8;
    }
    set_half(&mut bytes, 208, 0.25);
    let decoded = values(GgmlType::Q6K, &bytes);
    for j in 0..256 {
        let level = [-32.0, -16.0, 15.0, 31.0][(j % 128) / 32];
        assert_eq!(
            decoded[j],
            0.25 * (j as i32 / 16 - 8) as f32 * level,
            "element {j}"
        );
    }
}

#[test]
fn iq4_xs_keeps_f32_group_scales() {
    let mut bytes = block(GgmlType::Iq4Xs);
    bytes[..8].copy_from_slice(&[0x55, 0x35, 0xe4, 0xe4, 0x10, 0x32, 0x54, 0x76]);
    bytes[8..].fill(0xf0);
    let decoded = values(GgmlType::Iq4Xs, &bytes);
    let d = f16::from_bits(0x3555).to_f32();
    for (group, multiplier) in [-32, -15, 2, 19, -28, -11, 6, 23].into_iter().enumerate() {
        let scale = d * multiplier as f32;
        assert_eq!(&decoded[group * 32..group * 32 + 16], &[scale * -127.0; 16]);
        assert_eq!(
            &decoded[group * 32 + 16..group * 32 + 32],
            &[scale * 113.0; 16]
        );
    }
    assert_ne!(decoded[32], f16::from_f32(d * -15.0).to_f32() * -127.0);
}

#[test]
fn iq3_s_golden_grid_extremes_signs_and_unrounded_scales() {
    let mut bytes = block(GgmlType::Iq3S);
    bytes[..2].copy_from_slice(&[0x55, 0x35]);
    for pair in bytes[2..66].chunks_exact_mut(2) {
        pair.copy_from_slice(&[1, 255]);
    }
    bytes[66..74].fill(0xaa); // alternating grid index 1 and 511
    bytes[74..106].fill(0xaa);
    bytes[106..].copy_from_slice(&[0x10, 0x32, 0x54, 0x76]);
    let decoded = values(GgmlType::Iq3S, &bytes);
    let d = f16::from_bits(0x3555).to_f32();
    // Published grid[1]=0x01010103, grid[511]=0x0f0f0101.
    let levels = [3.0, -1.0, 1.0, -1.0, 1.0, -1.0, 15.0, -15.0];
    for j in 0..256 {
        let scale = d * (2 * (j / 32) + 1) as f32;
        assert_eq!(decoded[j], scale * levels[j % 8], "element {j}");
    }
    assert_ne!(decoded[224], f16::from_f32(d * 15.0).to_f32() * 3.0);
}

#[test]
fn decoder_rejects_unknown_partial_and_nonfinite_blocks() {
    for unsupported in [
        3, 9, 10, 11, 15, 16, 17, 18, 19, 22, 24, 25, 26, 27, 28, 29, 34, 35, 39,
    ] {
        assert!(
            GgmlType::from_code(unsupported).is_err(),
            "type {unsupported}"
        );
    }
    assert!(decode(GgmlType::Q6K, &[0; 209], &mut [0.0; 256]).is_err());
    assert!(decode(GgmlType::F32, &f32::NAN.to_le_bytes(), &mut [0.0]).is_err());
    let mut bytes = block(GgmlType::Iq3S);
    bytes[..2].copy_from_slice(&[0, 0x7c]);
    assert!(decode(GgmlType::Iq3S, &bytes, &mut [0.0; 256]).is_err());
}
