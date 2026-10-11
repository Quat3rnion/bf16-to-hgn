//! Verification-only HIP decoding. Encoding provenance never selects this backend.
//!
//! I4R follows `codec::decode_rows`; HT receives the independent verifier's
//! codebook and uses its circular-state mapping, not the encoder's decoder.
//! Both native axis passes use ordered f64 butterflies and one final f32 cast.
//! The native context batches arbitrary complete 128x128 blocks into a lazy
//! workspace smaller than 64 MiB, and returns only after output is complete.

use crate::codec::{EncodedBlock, QuantMeta};
use anyhow::{ensure, Context, Result};
use half::f16;
use std::ffi::c_int;

/// Runtime verification choice, separate from a manifest's encoding backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum DecodeBackend {
    #[default]
    Cpu,
    Hip,
}

impl DecodeBackend {
    /// Fail explicitly for unavailable HIP, even without an eligible tensor.
    pub fn preflight(self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Hip => {
                #[cfg(feature = "hip")]
                {
                    // SAFETY: no pointers; native context access is serialized.
                    let result = unsafe { hgn_hip_decode_preflight() };
                    ensure!(
                        result == 0,
                        "HIP decode preflight failed: {}",
                        hip_last_error()
                    );
                    Ok(())
                }
                #[cfg(not(feature = "hip"))]
                anyhow::bail!("HIP decode backend requires a build with --features hip");
            }
        }
    }
}

fn geometry(rows: usize, cols: usize) -> Result<(usize, c_int, c_int)> {
    ensure!(
        rows > 0 && cols > 0 && rows % 128 == 0 && cols % 128 == 0,
        "HIP decode requires nonempty complete 128-row/column blocks"
    );
    let count = rows
        .checked_mul(cols)
        .context("matrix element count overflow")?;
    ensure!(
        count <= isize::MAX as usize / size_of::<f32>(),
        "decoded block length exceeds address space"
    );
    let bands =
        c_int::try_from(rows / 128).context("HIP decode band count exceeds native range")?;
    let cols = c_int::try_from(cols).context("HIP decode column count exceeds native range")?;
    Ok((count, bands, cols))
}

/// Decode one or more I4R bands with one shared input rotation vector.
/// Scales are the existing contiguous little-endian half plane, without pad.
pub fn decode_i4r(
    block: &EncodedBlock,
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<Vec<f32>> {
    let (count, _bands, _native_cols) = geometry(rows, cols)?;
    ensure!(
        block.codes.len() == count / 2 && block.scales.len() == count / 64,
        "encoded block plane lengths do not match shape"
    );
    ensure!(
        meta.su.len() == cols && meta.sv.len() == rows,
        "rotation vector dimensions do not match band"
    );
    ensure!(
        meta.su
            .iter()
            .chain(meta.sv)
            .all(|&value| { value.is_finite() && f16::from_f32(value).to_f32() == value }),
        "rotation vectors must contain finite, exactly representable f16 values"
    );
    ensure!(
        block
            .scales
            .chunks_exact(2)
            .all(|bytes| { f16::from_le_bytes([bytes[0], bytes[1]]).is_finite() }),
        "non-finite quantization scale"
    );
    #[cfg(feature = "hip")]
    {
        native_output(
            count,
            "I4R",
            "decoded block contains non-finite values",
            |out| {
                // SAFETY: all input extents and output capacity match the checked
                // shape. Native treats scales as bytes on the host, so casting the
                // byte plane does not require host uint16_t alignment or a copy.
                // It synchronizes every transfer before returning, including errors.
                unsafe {
                    hgn_hip_decode_i4r(
                        block.codes.as_ptr(),
                        block.scales.as_ptr().cast(),
                        meta.su.as_ptr(),
                        meta.sv.as_ptr(),
                        out,
                        _bands,
                        _native_cols,
                    )
                }
            },
        )
    }
    #[cfg(not(feature = "hip"))]
    anyhow::bail!("HIP decode backend requires a build with --features hip")
}

/// Decode independent HT3/HT4 verification bands, including paired gate/up.
/// `su` contains `cols` entries per band, `sv` contains `rows` entries, and
/// `bytes` includes both projections when `row_tiles == 16`. Only tile offsets
/// 0 (ordinary/gate) and 8 (paired up) are legal. The supplied 65536-value
/// codebook is content-cached by native code, never replaced by an encoder one.
#[allow(clippy::too_many_arguments)]
pub fn decode_ht(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    bits: usize,
    su: &[f32],
    sv: &[f32],
    row_tiles: usize,
    tile_offset: usize,
    codebook: &[f32],
) -> Result<Vec<f32>> {
    ensure!(bits == 3 || bits == 4, "invalid HT verifier width");
    ensure!(
        (row_tiles == 8 && tile_offset == 0) || (row_tiles == 16 && matches!(tile_offset, 0 | 8)),
        "invalid HT verifier tile geometry"
    );
    let (_count, _bands, _native_cols) = geometry(rows, cols)?;
    let bands = rows / 128;
    let inputs = bands
        .checked_mul(cols)
        .context("HT verifier rotation size overflow")?;
    ensure!(
        su.len() == inputs && sv.len() == rows,
        "invalid HT verifier geometry"
    );
    let expected = inputs
        .checked_div(16)
        .and_then(|n| n.checked_mul(row_tiles))
        .and_then(|n| n.checked_mul(32 * bits))
        .context("HT verifier payload overflow")?;
    ensure!(bytes.len() == expected, "invalid HT verifier band size");
    ensure!(
        su.iter().chain(sv).all(|value| value.is_finite()),
        "nonfinite HT verifier rotation"
    );
    ensure!(
        codebook.len() == 65536,
        "invalid HT verifier codebook length"
    );
    ensure!(
        codebook.iter().all(|value| value.is_finite()),
        "nonfinite HT verifier codebook"
    );
    #[cfg(feature = "hip")]
    {
        native_output(_count, "HT", "nonfinite HT verifier value", |out| {
            // SAFETY: every extent and native integer is checked above. Output
            // is exclusively borrowed and native success means all values have
            // been written and all asynchronous work has completed.
            unsafe {
                hgn_hip_decode_ht(
                    bytes.as_ptr(),
                    codebook.as_ptr(),
                    su.as_ptr(),
                    sv.as_ptr(),
                    out,
                    _bands,
                    _native_cols,
                    bits as c_int,
                    row_tiles as c_int,
                    tile_offset as c_int,
                )
            }
        })
    }
    #[cfg(not(feature = "hip"))]
    anyhow::bail!("HIP decode backend requires a build with --features hip")
}

#[cfg(feature = "hip")]
unsafe extern "C" {
    fn hgn_hip_decode_i4r(
        codes: *const u8,
        scales: *const u16,
        su: *const f32,
        sv: *const f32,
        out: *mut f32,
        bands: c_int,
        cols: c_int,
    ) -> c_int;
    fn hgn_hip_decode_ht(
        bytes: *const u8,
        codebook: *const f32,
        su: *const f32,
        sv: *const f32,
        out: *mut f32,
        bands: c_int,
        cols: c_int,
        bits: c_int,
        row_tiles: c_int,
        tile_offset: c_int,
    ) -> c_int;
    fn hgn_hip_decode_preflight() -> c_int;
    fn ht_hip_last_error() -> *const std::ffi::c_char;
}

#[cfg(feature = "hip")]
fn hip_last_error() -> String {
    // SAFETY: the shared ABI returns thread-local NUL-terminated text, valid
    // until this thread's next native call. Copy it immediately.
    let error = unsafe { ht_hip_last_error() };
    if error.is_null() {
        return "native HIP decoder returned no error detail".to_owned();
    }
    unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(feature = "hip")]
fn native_output(
    count: usize,
    codec: &str,
    finite_error: &str,
    operation: impl FnOnce(*mut f32) -> c_int,
) -> Result<Vec<f32>> {
    let mut values = Vec::<f32>::new();
    values
        .try_reserve_exact(count)
        .context("allocate decoded block")?;
    let result = operation(values.as_mut_ptr());
    ensure!(
        result == 0,
        "HIP {codec} decode failed: {}",
        hip_last_error()
    );
    // SAFETY: both native decoders synchronously initialize exactly count
    // values on success. A failed partial output keeps length zero and is
    // discarded. Do not zero-fill a buffer that the device fully overwrites.
    unsafe { values.set_len(count) };
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{finite_error}"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rejects_shapes_planes_and_metadata_before_native_reads() {
        let mut block = EncodedBlock {
            codes: vec![0x88; 128 * 128 / 2],
            scales: vec![0; 128 * 128 / 64],
        };
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ..Default::default()
        };
        for (rows, cols) in [
            (0, 128),
            (128, 0),
            (127, 128),
            (128, 129),
            (usize::MAX & !127, 128),
        ] {
            assert!(decode_i4r(&block, rows, cols, &meta).is_err());
        }
        block.codes.pop();
        assert!(decode_i4r(&block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains("plane lengths"));
        block.codes.push(0x88);
        block.scales.extend_from_slice(&[0, 0]);
        assert!(decode_i4r(&block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains("plane lengths"));
        block.scales.truncate(256);
        block.scales[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(decode_i4r(&block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains("quantization scale"));
        block.scales[..2].fill(0);
        for value in [f32::NAN, f32::INFINITY, 0.1, f32::from_bits(1)] {
            let invalid = vec![value; 128];
            let meta = QuantMeta {
                su: &invalid,
                ..meta
            };
            assert!(decode_i4r(&block, 128, 128, &meta)
                .unwrap_err()
                .to_string()
                .contains("representable f16"));
        }
        let meta = QuantMeta {
            sv: &sv[..127],
            ..meta
        };
        assert!(decode_i4r(&block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains("vector dimensions"));
    }

    #[test]
    fn ht_decode_rejects_invalid_width_layout_and_nonfinite_metadata() {
        let bytes = vec![0; 128 * 128 / 2];
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let mut codebook = vec![0.0; 65536];
        for (bits, tiles, offset) in [(2, 8, 0), (5, 8, 0), (4, 8, 8), (4, 16, 1), (4, 24, 0)] {
            assert!(decode_ht(&bytes, 128, 128, bits, &su, &sv, tiles, offset, &codebook).is_err());
        }
        assert!(decode_ht(
            &bytes[..bytes.len() - 1],
            128,
            128,
            4,
            &su,
            &sv,
            8,
            0,
            &codebook
        )
        .unwrap_err()
        .to_string()
        .contains("band size"));
        assert!(
            decode_ht(&bytes, 128, 128, 4, &su[..127], &sv, 8, 0, &codebook)
                .unwrap_err()
                .to_string()
                .contains("geometry")
        );
        assert!(
            decode_ht(&bytes, 128, 128, 4, &su, &sv, 8, 0, &codebook[..65535])
                .unwrap_err()
                .to_string()
                .contains("codebook length")
        );
        codebook[65535] = f32::NAN;
        assert!(decode_ht(&bytes, 128, 128, 4, &su, &sv, 8, 0, &codebook)
            .unwrap_err()
            .to_string()
            .contains("nonfinite HT verifier codebook"));
        let invalid = vec![f32::NEG_INFINITY; 128];
        assert!(
            decode_ht(&bytes, 128, 128, 4, &su, &invalid, 8, 0, &codebook)
                .unwrap_err()
                .to_string()
                .contains("nonfinite HT verifier rotation")
        );
        assert!(decode_ht(&[], usize::MAX & !127, 128, 4, &[], &[], 8, 0, &codebook).is_err());
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_has_explicit_hip_unavailable_errors() {
        DecodeBackend::Cpu.preflight().unwrap();
        assert!(DecodeBackend::Hip
            .preflight()
            .unwrap_err()
            .to_string()
            .contains("--features hip"));
        let block = EncodedBlock {
            codes: vec![0x88; 8192],
            scales: vec![0; 256],
        };
        let signs = vec![1.0; 128];
        let meta = QuantMeta {
            su: &signs,
            sv: &signs,
            ..Default::default()
        };
        assert!(decode_i4r(&block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains("--features hip"));
        assert!(decode_ht(
            &block.codes,
            128,
            128,
            4,
            &signs,
            &signs,
            8,
            0,
            &vec![0.0; 65536]
        )
        .unwrap_err()
        .to_string()
        .contains("--features hip"));
    }

    #[cfg(feature = "hip")]
    mod device {
        use super::*;
        use crate::codec::{decode_rows, Encoding};
        use crate::verify::{decode_ht_independent, VERIFICATION_HT_CODEBOOK};

        unsafe extern "C" {
            fn hipSetDevice(device: c_int) -> c_int;
        }

        fn equal_bits(actual: &[f32], expected: &[f32]) {
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "f32 mismatch at element {index}: {actual:?} != {expected:?}"
                );
            }
        }

        fn i4r_input(rows: usize, cols: usize) -> (EncodedBlock, Vec<f32>, Vec<f32>) {
            let half_bits = [
                0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3555, 0xb555,
                0x3c00, 0xbc00, 0x7bff, 0xfbff,
            ];
            let block = EncodedBlock {
                codes: (0..rows * cols / 2)
                    .map(|i| ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 31) as u8)
                    .collect(),
                scales: (0..rows * cols / 128)
                    .flat_map(|i| u16::to_le_bytes(half_bits[i % half_bits.len()]))
                    .collect(),
            };
            let su = (0..cols)
                .map(|i| f16::from_bits(half_bits[(i * 5 + 1) % half_bits.len()]).to_f32())
                .collect();
            let sv = (0..rows)
                .map(|i| f16::from_bits(half_bits[(i * 3 + 2) % half_bits.len()]).to_f32())
                .collect();
            (block, su, sv)
        }

        fn cpu_i4r(
            block: &EncodedBlock,
            rows: usize,
            cols: usize,
            su: &[f32],
            sv: &[f32],
        ) -> Vec<f32> {
            let mut output = Vec::with_capacity(rows * cols);
            for band in 0..rows / 128 {
                let start = band * 128 * cols;
                let plane = EncodedBlock {
                    codes: block.codes[start / 2..(start + 128 * cols) / 2].to_vec(),
                    scales: block.scales[start / 64..(start + 128 * cols) / 64].to_vec(),
                };
                let meta = QuantMeta {
                    su,
                    sv: &sv[band * 128..(band + 1) * 128],
                    ..Default::default()
                };
                output.extend(decode_rows(Encoding::I4r, &plane, 128, cols, &meta).unwrap());
            }
            output
        }

        fn ht_input(
            rows: usize,
            cols: usize,
            bits: usize,
            tiles: usize,
        ) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
            let len = rows / 128 * (cols / 16) * tiles * 32 * bits;
            let bytes = (0..len / 4)
                .flat_map(|i| (i as u32 + 1).wrapping_mul(0x9e3779b9).to_le_bytes())
                .collect();
            // The independent HT verifier permits all finite f32 metadata,
            // unlike the f16-constrained I4R decoder. Exercise that contract.
            let signs = [
                1.0,
                -1.0,
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                0.3,
                1024.0,
            ];
            let scales = [
                1.0,
                -1.0,
                0.0,
                -0.0,
                f16::from_bits(1).to_f32(),
                0.1,
                -0.33,
                f32::MIN_POSITIVE,
                -f32::MIN_POSITIVE,
            ];
            let su = (0..rows / 128 * cols)
                .map(|i| signs[i % signs.len()])
                .collect();
            let sv = (0..rows).map(|i| scales[i % scales.len()]).collect();
            (bytes, su, sv)
        }

        fn cpu_ht(
            bytes: &[u8],
            rows: usize,
            cols: usize,
            bits: usize,
            tiles: usize,
            offset: usize,
            su: &[f32],
            sv: &[f32],
        ) -> Vec<f32> {
            let mut output = Vec::with_capacity(rows * cols);
            let band_bytes = cols / 16 * tiles * 32 * bits;
            for band in 0..rows / 128 {
                output.extend(
                    decode_ht_independent(
                        &bytes[band * band_bytes..(band + 1) * band_bytes],
                        bits,
                        cols,
                        if tiles == 8 { None } else { Some(offset == 8) },
                        &su[band * cols..(band + 1) * cols],
                        &sv[band * 128..(band + 1) * 128],
                    )
                    .unwrap(),
                );
            }
            output
        }

        #[test]
        #[ignore = "requires gfx1151 HIP; I4R levels, signed zeros, half subnormals and extreme finite scales"]
        fn hip_i4r_decode_matches_cpu_edge_corpus() {
            DecodeBackend::Hip.preflight().unwrap();
            for (rows, cols) in [(128, 128), (256, 640), (128, 2560)] {
                let (block, su, sv) = i4r_input(rows, cols);
                let expected = cpu_i4r(&block, rows, cols, &su, &sv);
                // SAFETY: a negative device ordinal is rejected without a
                // device switch. Its sticky HIP 7 error must not taint launch.
                assert_ne!(unsafe { hipSetDevice(-1) }, 0);
                let meta = QuantMeta {
                    su: &su,
                    sv: &sv,
                    ..Default::default()
                };
                equal_bits(&decode_i4r(&block, rows, cols, &meta).unwrap(), &expected);
            }
            let su: Vec<f32> = (0..128)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let sv = su.clone();
            for scale in [0u16, 0x8000, 1, 0x8001, 0x7bff, 0xfbff] {
                for code in 0..16u8 {
                    let block = EncodedBlock {
                        codes: vec![code | code << 4; 8192],
                        scales: (0..128).flat_map(|_| scale.to_le_bytes()).collect(),
                    };
                    let meta = QuantMeta {
                        su: &su,
                        sv: &sv,
                        ..Default::default()
                    };
                    equal_bits(
                        &decode_i4r(&block, 128, 128, &meta).unwrap(),
                        &cpu_i4r(&block, 128, 128, &su, &sv),
                    );
                }
            }
        }

        #[test]
        #[ignore = "requires gfx1151 HIP; independent HT3/HT4 circular states, paired tiles and f32 subnormals"]
        fn hip_ht_decode_matches_independent_verifier() {
            DecodeBackend::Hip.preflight().unwrap();
            let mut saw_subnormal = false;
            for bits in [3, 4] {
                for (tiles, offset) in [(8, 0), (16, 0), (16, 8)] {
                    for (rows, cols) in [(128, 128), (256, 640)] {
                        let (bytes, su, sv) = ht_input(rows, cols, bits, tiles);
                        let expected = cpu_ht(&bytes, rows, cols, bits, tiles, offset, &su, &sv);
                        saw_subnormal |= expected.iter().any(|value| value.is_subnormal());
                        // SAFETY: intentionally retain a previously reported
                        // runtime failure before a valid independent decode.
                        assert_ne!(unsafe { hipSetDevice(-1) }, 0);
                        equal_bits(
                            &decode_ht(
                                &bytes,
                                rows,
                                cols,
                                bits,
                                &su,
                                &sv,
                                tiles,
                                offset,
                                &VERIFICATION_HT_CODEBOOK,
                            )
                            .unwrap(),
                            &expected,
                        );
                    }
                    for fill in [0, 255] {
                        let (mut bytes, su, sv) = ht_input(128, 128, bits, tiles);
                        bytes.fill(fill);
                        equal_bits(
                            &decode_ht(
                                &bytes,
                                128,
                                128,
                                bits,
                                &su,
                                &sv,
                                tiles,
                                offset,
                                &VERIFICATION_HT_CODEBOOK,
                            )
                            .unwrap(),
                            &cpu_ht(&bytes, 128, 128, bits, tiles, offset, &su, &sv),
                        );
                    }
                }
            }
            assert!(
                saw_subnormal,
                "corpus must exercise final f32 gradual underflow"
            );
        }

        #[test]
        #[ignore = "requires gfx1151 HIP; exceed bounded workspace in both rectangle dimensions"]
        fn hip_decode_crosses_band_and_column_batch_boundaries() {
            DecodeBackend::Hip.preflight().unwrap();
            // Native holds 256 128x128 blocks. 257 forces a nonempty tail;
            // the wide case also exercises pitched code/scale/output copies.
            for (rows, cols) in [(128, 257 * 128), (257 * 128, 128)] {
                let (block, su, sv) = i4r_input(rows, cols);
                let meta = QuantMeta {
                    su: &su,
                    sv: &sv,
                    ..Default::default()
                };
                equal_bits(
                    &decode_i4r(&block, rows, cols, &meta).unwrap(),
                    &cpu_i4r(&block, rows, cols, &su, &sv),
                );
                for (bits, tiles, offset) in [(3, 16, 8), (4, 8, 0)] {
                    let (bytes, su, sv) = ht_input(rows, cols, bits, tiles);
                    equal_bits(
                        &decode_ht(
                            &bytes,
                            rows,
                            cols,
                            bits,
                            &su,
                            &sv,
                            tiles,
                            offset,
                            &VERIFICATION_HT_CODEBOOK,
                        )
                        .unwrap(),
                        &cpu_ht(&bytes, rows, cols, bits, tiles, offset, &su, &sv),
                    );
                }
            }
        }

        #[test]
        #[ignore = "requires gfx1151 HIP; content cache changes, exact signed zeros and smallest f32 values"]
        fn hip_ht_decode_caches_contents_not_addresses() {
            DecodeBackend::Hip.preflight().unwrap();
            let (rows, cols) = (256, 256);
            let mut codebook = VERIFICATION_HT_CODEBOOK.to_vec();
            for bits in [3, 4] {
                let (bytes, su, sv) = ht_input(rows, cols, bits, 16);
                codebook.copy_from_slice(&VERIFICATION_HT_CODEBOOK);
                equal_bits(
                    &decode_ht(&bytes, rows, cols, bits, &su, &sv, 16, 8, &codebook).unwrap(),
                    &cpu_ht(&bytes, rows, cols, bits, 16, 8, &su, &sv),
                );
                let su: Vec<f32> = (0..rows / 128 * cols)
                    .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                    .collect();
                let sv = vec![1.0 / 128.0; rows];
                for constant in [0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0] {
                    codebook.fill(constant);
                    // H * constant * H has only one DC value per 128x128
                    // block. With sv=1/128 that value is the original constant,
                    // even for the smallest subnormal and negative zero.
                    let expected: Vec<f32> = (0..rows * cols)
                        .map(|index| {
                            let row = index / cols;
                            let col = index % cols;
                            let transformed = if row % 128 == 0 && col % 128 == 0 {
                                f64::from(constant) * 128.0
                            } else {
                                0.0
                            };
                            (transformed
                                * f64::from(sv[row])
                                * f64::from(su[row / 128 * cols + col]))
                                as f32
                        })
                        .collect();
                    equal_bits(
                        &decode_ht(&bytes, rows, cols, bits, &su, &sv, 16, 8, &codebook).unwrap(),
                        &expected,
                    );
                }
            }
        }

        #[test]
        #[ignore = "requires gfx1151 HIP; malformed native ABI extents, output overflow and recovery"]
        fn hip_decode_errors_do_not_read_invalid_extents_or_poison_context() {
            DecodeBackend::Hip.preflight().unwrap();
            // SAFETY: each deliberately invalid ABI call must reject its shape,
            // null pointers, or overflowing address extent before any read.
            unsafe {
                assert_ne!(
                    hgn_hip_decode_i4r(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        1,
                        128
                    ),
                    0
                );
                assert_ne!(
                    hgn_hip_decode_i4r(
                        1usize as *const u8,
                        2usize as *const u16,
                        4usize as *const f32,
                        4usize as *const f32,
                        4usize as *mut f32,
                        c_int::MAX,
                        c_int::MAX & !127
                    ),
                    0
                );
                assert!(hip_last_error().contains("shape overflows"));
                assert_ne!(
                    hgn_hip_decode_i4r(
                        (usize::MAX - 3) as *const u8,
                        2usize as *const u16,
                        4usize as *const f32,
                        4usize as *const f32,
                        4usize as *mut f32,
                        1,
                        128
                    ),
                    0
                );
                assert!(hip_last_error().contains("buffer extent"));
                assert_ne!(
                    hgn_hip_decode_ht(
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null_mut(),
                        1,
                        128,
                        4,
                        8,
                        8
                    ),
                    0
                );
                assert!(hip_last_error().contains("tile geometry"));
                assert_ne!(
                    hgn_hip_decode_ht(
                        1usize as *const u8,
                        (usize::MAX - 3) as *const f32,
                        4usize as *const f32,
                        4usize as *const f32,
                        4usize as *mut f32,
                        1,
                        128,
                        3,
                        8,
                        0
                    ),
                    0
                );
                assert!(hip_last_error().contains("buffer extent"));
            }
            let bytes = vec![0; 8192];
            let signs = vec![1.0; 128];
            let codebook = vec![f32::MAX; 65536];
            assert!(
                decode_ht(&bytes, 128, 128, 4, &signs, &signs, 8, 0, &codebook)
                    .unwrap_err()
                    .to_string()
                    .contains("nonfinite HT verifier value")
            );
            let (bytes, su, sv) = ht_input(128, 128, 4, 8);
            equal_bits(
                &decode_ht(
                    &bytes,
                    128,
                    128,
                    4,
                    &su,
                    &sv,
                    8,
                    0,
                    &VERIFICATION_HT_CODEBOOK,
                )
                .unwrap(),
                &cpu_ht(&bytes, 128, 128, 4, 8, 0, &su, &sv),
            );
        }
    }
}
