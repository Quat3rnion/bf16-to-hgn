use anyhow::Result;
use bf16_to_hgn::codec::{
    self, decode_rows, encode_rows, EncodedBlock, Encoding, HtBackend, I4rBackend, QuantMeta,
};
use clap::ValueEnum;

const BAND: usize = 128;

fn mixed_values(rows: usize, cols: usize, salt: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let value = ((i * 37 + row * 19 + salt * 53) % 509) as f32 - 254.0;
            match (row + col + salt) % 11 {
                0 => 0.0,
                1 => -0.0,
                2 => value / 65536.0,
                _ => value / 128.0,
            }
        })
        .collect()
}

fn assert_planes(actual: &EncodedBlock, expected: &EncodedBlock) {
    for (plane, actual, expected) in [
        ("codes", &actual.codes, &expected.codes),
        (
            "little-endian half scales",
            &actual.scales,
            &expected.scales,
        ),
    ] {
        assert_eq!(actual.len(), expected.len(), "{plane} length");
        if let Some(index) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            assert_eq!(actual[index], expected[index], "{plane} byte {index}");
        }
    }
}

fn assert_float_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "decoded value {index}"
        );
    }
}

fn cpu_bands(input: &[f32], cols: usize, su: &[f32], sv: &[f32]) -> Result<EncodedBlock> {
    let mut output = EncodedBlock {
        codes: Vec::with_capacity(input.len() / 2),
        scales: Vec::with_capacity(input.len() / 64),
    };
    for (band, values) in input.chunks_exact(BAND * cols).enumerate() {
        let encoded = encode_rows(
            Encoding::I4r,
            values,
            BAND,
            cols,
            &QuantMeta {
                su,
                sv: &sv[band * BAND..(band + 1) * BAND],
                ..QuantMeta::default()
            },
        )?;
        output.codes.extend(encoded.codes);
        output.scales.extend(encoded.scales);
    }
    Ok(output)
}

fn assert_batch(
    input: &[f32],
    rows: usize,
    cols: usize,
    su: &[f32],
    sv: &[f32],
    backend: I4rBackend,
) -> Result<()> {
    assert_eq!(input.len(), rows * cols);
    let expected = cpu_bands(input, cols, su, sv)?;
    let meta = QuantMeta {
        su,
        sv,
        i4r_backend: backend,
        ..QuantMeta::default()
    };
    let actual = encode_rows(Encoding::I4r, input, rows, cols, &meta)?;
    assert_eq!(actual.codes.len(), rows * cols / 2);
    assert_eq!(actual.scales.len(), rows * cols / 64);
    assert_planes(&actual, &expected);
    let decoded = decode_rows(Encoding::I4r, &actual, rows, cols, &meta)?;
    for band in 0..rows / BAND {
        let code_start = band * BAND * cols / 2;
        let scale_start = band * BAND * cols / 64;
        let block = EncodedBlock {
            codes: expected.codes[code_start..code_start + BAND * cols / 2].to_vec(),
            scales: expected.scales[scale_start..scale_start + BAND * cols / 64].to_vec(),
        };
        let reference = decode_rows(
            Encoding::I4r,
            &block,
            BAND,
            cols,
            &QuantMeta {
                su,
                sv: &sv[band * BAND..(band + 1) * BAND],
                ..QuantMeta::default()
            },
        )?;
        assert_float_bits(
            &decoded[band * BAND * cols..(band + 1) * BAND * cols],
            &reference,
        );
    }
    Ok(())
}

// Construct a known rotated first row with all remaining rows zero. H is its
// own inverse up to 128, so repeating H*target/128 in every source row gives
// target after the full two-axis rotation. This is corpus construction only:
// the unmodified public CPU encoder remains the sole byte oracle.
fn source_for_rotated_row(target: &[f32; BAND], su: &[f32], sv: &[f32]) -> Vec<f32> {
    let columns: Vec<f32> = (0..BAND)
        .map(|col| {
            let sum: f64 = target
                .iter()
                .enumerate()
                .map(|(index, &value)| {
                    let sign = if (index & col).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    f64::from(value) * sign
                })
                .sum();
            (sum / BAND as f64) as f32
        })
        .collect();
    (0..BAND * BAND)
        .map(|i| columns[i % BAND] * su[i % BAND] * sv[i / BAND])
        .collect()
}

fn extreme_and_rne_row(sign: f32) -> [f32; BAND] {
    let mut target = [-8.0; BAND];
    target[1] = 8.0; // Equal absolute extreme must not replace index zero.
    for (index, value) in target[2..18].iter_mut().enumerate() {
        *value = index as f32 - 7.5;
    }
    target.map(|value| value * sign)
}

#[test]
fn i4r_backend_is_explicit_lowercase_and_independent_of_ht() -> Result<()> {
    assert_eq!(I4rBackend::default(), I4rBackend::Cpu);
    assert_eq!(QuantMeta::default().i4r_backend, I4rBackend::Cpu);
    assert_eq!(
        I4rBackend::value_variants(),
        &[I4rBackend::Cpu, I4rBackend::Hip, I4rBackend::Cuda]
    );
    for (name, backend) in [
        ("cpu", I4rBackend::Cpu),
        ("hip", I4rBackend::Hip),
        ("cuda", I4rBackend::Cuda),
    ] {
        assert_eq!(serde_json::to_value(backend)?, name);
        assert_eq!(I4rBackend::from_str(name, false).unwrap(), backend);
        assert_eq!(
            serde_json::from_value::<I4rBackend>(serde_json::json!(name))?,
            backend
        );
    }
    for name in ["gpu", "CPU", "HIP", "CUDA", "amd", "nvidia"] {
        assert!(I4rBackend::from_str(name, false).is_err());
        assert!(serde_json::from_value::<I4rBackend>(serde_json::json!(name)).is_err());
    }
    I4rBackend::Cpu.preflight()?;
    let signs = [1.0; BAND];
    let input = mixed_values(BAND, BAND, 0);
    let meta = QuantMeta {
        su: &signs,
        sv: &signs,
        ..QuantMeta::default()
    };
    let expected = encode_rows(Encoding::I4r, &input, BAND, BAND, &meta)?;
    for (ht_backend, ht_beam) in [
        (HtBackend::Cpu, 0),
        (HtBackend::Hip, 4097),
        (HtBackend::Cuda, usize::MAX),
    ] {
        let selected = QuantMeta {
            ht_backend,
            ht_beam,
            ..meta
        };
        assert_planes(
            &encode_rows(Encoding::I4r, &input, BAND, BAND, &selected)?,
            &expected,
        );
    }
    // Merely recording GPU encoding provenance never initializes a decoder.
    let decoded = decode_rows(Encoding::I4r, &expected, BAND, BAND, &meta)?;
    for i4r_backend in [I4rBackend::Hip, I4rBackend::Cuda] {
        let provenance = QuantMeta {
            i4r_backend,
            ..meta
        };
        assert_float_bits(
            &decode_rows(Encoding::I4r, &expected, BAND, BAND, &provenance)?,
            &decoded,
        );
    }
    for encoding in [Encoding::Bf16, Encoding::Ht, Encoding::Ht3] {
        let cpu = QuantMeta { ht_beam: 1, ..meta };
        let expected = encode_rows(encoding, &input, BAND, BAND, &cpu)?;
        for i4r_backend in [I4rBackend::Hip, I4rBackend::Cuda] {
            let selected = QuantMeta { i4r_backend, ..cpu };
            assert_planes(
                &encode_rows(encoding, &input, BAND, BAND, &selected)?,
                &expected,
            );
        }
    }
    Ok(())
}

#[test]
fn i4r_cpu_multiband_planes_and_decoding_equal_one_band_concatenation() -> Result<()> {
    for bands in [1, 3, 16, 32, 33] {
        let rows = bands * BAND;
        let cols = 2 * BAND;
        let su = codec::generate_signs(17, "expert.su", cols);
        let sv = codec::generate_signs(23, "expert.sv", rows);
        assert_batch(
            &mixed_values(rows, cols, bands),
            rows,
            cols,
            &su,
            &sv,
            I4rBackend::Cpu,
        )?;
    }
    Ok(())
}

#[test]
fn i4r_corner_setup_pins_extreme_rne_half_ties_and_signed_zero() -> Result<()> {
    let signs = [1.0; BAND];
    let meta = QuantMeta {
        su: &signs,
        sv: &signs,
        ..QuantMeta::default()
    };
    for sign in [1.0, -1.0] {
        let target = extreme_and_rne_row(sign);
        let input = source_for_rotated_row(&target, &signs, &signs);
        let cpu = encode_rows(Encoding::I4r, &input, BAND, BAND, &meta)?;
        assert_eq!(&cpu.scales[..2], &half::f16::from_f32(sign).to_le_bytes());
        for (index, value) in target.iter().enumerate() {
            let expected = ((value / sign).round_ties_even().clamp(-8.0, 7.0) as i32 + 8) as u8;
            assert_eq!((cpu.codes[index / 2] >> (4 * (index % 2))) & 15, expected);
        }
        // First extremes deliberately straddle lanes and warp-sized boundaries.
        for (first, second) in [(1, 127), (7, 8), (31, 32), (63, 64), (126, 127)] {
            let mut target = [0.0; BAND];
            target[first] = -8.0 * sign;
            target[second] = 8.0 * sign;
            let input = source_for_rotated_row(&target, &signs, &signs);
            let cpu = encode_rows(Encoding::I4r, &input, BAND, BAND, &meta)?;
            assert_eq!(&cpu.scales[..2], &half::f16::from_f32(sign).to_le_bytes());
        }
    }
    for lower in [0x0000, 0x0001, 0x03ff, 0x3c00, 0x3c01] {
        let midpoint =
            (half::f16::from_bits(lower).to_f32() + half::f16::from_bits(lower + 1).to_f32()) / 2.0;
        let mut target = [0.0; BAND];
        target[0] = -8.0 * midpoint;
        let input = source_for_rotated_row(&target, &signs, &signs);
        let cpu = encode_rows(Encoding::I4r, &input, BAND, BAND, &meta)?;
        assert_eq!(
            &cpu.scales[..2],
            &half::f16::from_f32(midpoint).to_le_bytes()
        );
    }
    for (zero, first_scale) in [(0.0, 0x8000u16), (-0.0, 0x0000u16)] {
        let cpu = encode_rows(Encoding::I4r, &vec![zero; BAND * BAND], BAND, BAND, &meta)?;
        assert!(cpu.codes.iter().all(|&code| code == 0x88));
        assert_eq!(&cpu.scales[..2], &first_scale.to_le_bytes());
    }
    Ok(())
}

#[test]
fn i4r_rejects_invalid_inputs_before_any_native_backend_call() -> Result<()> {
    let rows = 2 * BAND;
    let cols = 2 * BAND;
    let input = mixed_values(rows, cols, 1);
    let su = vec![1.0; cols];
    let sv = vec![1.0; rows];
    for i4r_backend in [I4rBackend::Cpu, I4rBackend::Hip, I4rBackend::Cuda] {
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            i4r_backend,
            ..QuantMeta::default()
        };
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut nonfinite = input.clone();
            *nonfinite.last_mut().unwrap() = invalid;
            let error = encode_rows(Encoding::I4r, &nonfinite, rows, cols, &meta).unwrap_err();
            assert!(
                error.to_string().contains("input contains non-finite"),
                "{error:#}"
            );
        }
        for (values, bad_rows, bad_cols) in [
            (&input[..input.len() - 1], rows, cols),
            (&[][..], 0, cols),
            (&[][..], rows, 0),
            (&input[..127 * cols], 127, cols),
            (&input[..rows * 127], rows, 127),
        ] {
            let error = encode_rows(Encoding::I4r, values, bad_rows, bad_cols, &meta).unwrap_err();
            assert!(!error.to_string().contains("--features"), "{error:#}");
            assert!(!error.to_string().contains("encoding failed"), "{error:#}");
        }
        let long = vec![1.0; rows + 1];
        for (bad_su, bad_sv) in [
            (&su[..cols - 1], &sv[..]),
            (&long[..], &sv[..]),
            (&su[..], &sv[..rows - 1]),
            (&su[..], &long[..]),
        ] {
            let bad = QuantMeta {
                su: bad_su,
                sv: bad_sv,
                ..meta
            };
            let error = encode_rows(Encoding::I4r, &input, rows, cols, &bad).unwrap_err();
            assert!(
                error.to_string().contains("rotation vector dimensions"),
                "{error:#}"
            );
        }
        for invalid in [
            0.0,
            -0.0,
            0.5,
            2.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut signs = su.clone();
            *signs.last_mut().unwrap() = invalid;
            for bad in [
                QuantMeta { su: &signs, ..meta },
                QuantMeta { sv: &signs, ..meta },
            ] {
                let error = encode_rows(Encoding::I4r, &input, rows, cols, &bad).unwrap_err();
                assert!(error.to_string().contains("vector"), "{error:#}");
                assert!(!error.to_string().contains("--features"), "{error:#}");
            }
        }
    }
    assert!(encode_rows(Encoding::I4r, &[], usize::MAX, 2, &QuantMeta::default()).is_err());
    let meta = QuantMeta {
        su: &su,
        sv: &sv,
        ..QuantMeta::default()
    };
    let encoded = encode_rows(Encoding::I4r, &input, rows, cols, &meta)?;
    for scale_plane in [false, true] {
        let mut truncated = encoded.clone();
        if scale_plane {
            truncated.scales.pop();
        } else {
            truncated.codes.pop();
        }
        assert!(decode_rows(Encoding::I4r, &truncated, rows, cols, &meta).is_err());
    }
    for bad in [
        QuantMeta {
            su: &su[..cols - 1],
            ..meta
        },
        QuantMeta {
            sv: &sv[..rows - 1],
            ..meta
        },
    ] {
        assert!(decode_rows(Encoding::I4r, &encoded, rows, cols, &bad).is_err());
    }
    let mut bad_scales = encoded;
    bad_scales.scales[..2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
    assert!(decode_rows(Encoding::I4r, &bad_scales, rows, cols, &meta).is_err());
    for encoding in [Encoding::Ht, Encoding::Ht3] {
        // Multi-band HT encoding is accepted (it equals concatenated bands;
        // see tests/ht_gpu_batch.rs); HT decoding still takes one band.
        assert!(encode_rows(encoding, &input, rows, cols, &meta).is_ok());
        let block = EncodedBlock {
            codes: vec![0; codec::payload_layout(encoding, rows, cols)?.len as usize],
            scales: Vec::new(),
        };
        assert!(decode_rows(encoding, &block, rows, cols, &meta).is_err());
    }
    Ok(())
}

#[cfg(target_pointer_width = "64")]
#[test]
fn i4r_gpu_rejects_unrepresentable_native_dimensions_without_allocation() {
    for i4r_backend in [I4rBackend::Hip, I4rBackend::Cuda] {
        let meta = QuantMeta {
            i4r_backend,
            ..QuantMeta::default()
        };
        for (rows, cols, message) in [
            (
                (i32::MAX as usize + 1) * BAND,
                BAND,
                "band count exceeds signed-int",
            ),
            (
                BAND,
                i32::MAX as usize + 1,
                "column count exceeds signed-int",
            ),
        ] {
            let error = encode_rows(Encoding::I4r, &[], rows, cols, &meta).unwrap_err();
            assert!(error.to_string().contains(message), "{error:#}");
            assert!(error.to_string().contains(i4r_backend.label()), "{error:#}");
        }
    }
}

#[cfg(not(feature = "hip"))]
#[test]
fn i4r_hip_requires_feature_without_falling_back_to_cpu() {
    requires_feature_without_falling_back_to_cpu(I4rBackend::Hip, "hip");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn i4r_cuda_requires_feature_without_falling_back_to_cpu() {
    requires_feature_without_falling_back_to_cpu(I4rBackend::Cuda, "cuda");
}

#[cfg(any(not(feature = "hip"), not(feature = "cuda")))]
fn requires_feature_without_falling_back_to_cpu(backend: I4rBackend, feature: &str) {
    let signs = [1.0; BAND];
    let input = vec![0.0; BAND * BAND];
    let meta = QuantMeta {
        su: &signs,
        sv: &signs,
        i4r_backend: backend,
        ..QuantMeta::default()
    };
    for error in [
        backend.preflight().unwrap_err(),
        encode_rows(Encoding::I4r, &input, BAND, BAND, &meta).unwrap_err(),
    ] {
        assert!(
            error
                .to_string()
                .contains(&format!("{} I4R", backend.label())),
            "{error:#}"
        );
        assert!(
            error.to_string().contains(&format!("--features {feature}")),
            "{error:#}"
        );
    }
}

#[cfg(any(feature = "hip", feature = "cuda"))]
mod gpu {
    use super::*;
    use std::ffi::{c_char, c_int, CStr};

    #[cfg(feature = "hip")]
    unsafe extern "C" {
        fn hgn_hip_encode_i4r(
            source: *const f32,
            su: *const f32,
            sv: *const f32,
            codes: *mut u8,
            scales: *mut u8,
            bands: c_int,
            cols: c_int,
        ) -> c_int;
        fn ht_hip_last_error() -> *const c_char;
        fn hipSetDevice(device: c_int) -> c_int;
    }

    #[cfg(feature = "cuda")]
    unsafe extern "C" {
        fn hgn_cuda_encode_i4r(
            source: *const f32,
            su: *const f32,
            sv: *const f32,
            codes: *mut u8,
            scales: *mut u8,
            bands: c_int,
            cols: c_int,
        ) -> c_int;
        fn ht_cuda_last_error() -> *const c_char;
        fn cudaSetDevice(device: c_int) -> c_int;
    }

    type NativeEncoder = unsafe extern "C" fn(
        *const f32,
        *const f32,
        *const f32,
        *mut u8,
        *mut u8,
        c_int,
        c_int,
    ) -> c_int;

    fn native_encoder(backend: I4rBackend) -> NativeEncoder {
        match backend {
            #[cfg(feature = "hip")]
            I4rBackend::Hip => hgn_hip_encode_i4r,
            #[cfg(feature = "cuda")]
            I4rBackend::Cuda => hgn_cuda_encode_i4r,
            _ => panic!("{} encoder feature is not enabled", backend.label()),
        }
    }

    fn last_error(backend: I4rBackend) -> String {
        let accessor = match backend {
            #[cfg(feature = "hip")]
            I4rBackend::Hip => ht_hip_last_error as unsafe extern "C" fn() -> *const c_char,
            #[cfg(feature = "cuda")]
            I4rBackend::Cuda => ht_cuda_last_error as unsafe extern "C" fn() -> *const c_char,
            _ => panic!("{} encoder feature is not enabled", backend.label()),
        };
        // SAFETY: the native thread-local NUL-terminated string is copied before
        // the next native call on this thread can invalidate it.
        unsafe { CStr::from_ptr(accessor()) }
            .to_string_lossy()
            .into_owned()
    }

    fn native_rejects_shapes_nulls_and_extents_without_a_device(backend: I4rBackend) {
        let encode = native_encoder(backend);
        let value = 1.0f32;
        let mut code = 0u8;
        let mut scale = 0u8;
        for (bands, cols, message) in [
            (0, 128, "positive bands"),
            (-1, 128, "positive bands"),
            (1, 0, "positive bands"),
            (1, 129, "multiple-of-128"),
            (i32::MAX, i32::MAX - 127, "shape overflows"),
        ] {
            // SAFETY: all shapes are rejected before any buffer access or GPU
            // initialization. The maximum dimensions require no large allocation.
            let status =
                unsafe { encode(&value, &value, &value, &mut code, &mut scale, bands, cols) };
            assert_eq!(status, -1);
            assert!(
                last_error(backend).contains(message),
                "{}",
                last_error(backend)
            );
        }
        for pointer in 0..5 {
            let mut source = &value as *const f32;
            let mut su = source;
            let mut sv = source;
            let mut codes = &mut code as *mut u8;
            let mut scales = &mut scale as *mut u8;
            match pointer {
                0 => source = std::ptr::null(),
                1 => su = std::ptr::null(),
                2 => sv = std::ptr::null(),
                3 => codes = std::ptr::null_mut(),
                4 => scales = std::ptr::null_mut(),
                _ => unreachable!(),
            }
            // SAFETY: a null required buffer is rejected before dereferencing
            // any of the other deliberately undersized buffers.
            let status = unsafe { encode(source, su, sv, codes, scales, 1, 128) };
            assert_eq!(status, -1);
            assert!(last_error(backend).contains("nonnull buffers"));
        }
        for pointer in 0..5 {
            let mut source = &value as *const f32;
            let mut su = source;
            let mut sv = source;
            let mut codes = &mut code as *mut u8;
            let mut scales = &mut scale as *mut u8;
            let near_end = (usize::MAX - 3) as *const f32;
            match pointer {
                0 => source = near_end,
                1 => su = near_end,
                2 => sv = near_end,
                3 => codes = near_end as *mut u8,
                4 => scales = near_end as *mut u8,
                _ => unreachable!(),
            }
            // SAFETY: each overflowing byte extent must be rejected by the ABI
            // before pointer access or device discovery; no allocation is made.
            let status = unsafe { encode(source, su, sv, codes, scales, 1, 128) };
            assert_eq!(status, -1);
            assert!(last_error(backend).contains("buffer extent overflows"));
        }
    }

    fn corner_corpus_matches_public_cpu_oracle(backend: I4rBackend) -> Result<()> {
        backend.preflight()?;
        for alternate in [false, true] {
            let su = if alternate {
                codec::generate_signs(17, "corner.su", BAND)
            } else {
                vec![1.0; BAND]
            };
            let sv = if alternate {
                codec::generate_signs(29, "corner.sv", BAND)
            } else {
                vec![1.0; BAND]
            };
            for sign in [1.0, -1.0] {
                let input = source_for_rotated_row(&extreme_and_rne_row(sign), &su, &sv);
                assert_batch(&input, BAND, BAND, &su, &sv, backend)?;
                for (first, second) in [(1, 127), (7, 8), (31, 32), (63, 64), (126, 127)] {
                    let mut target = [0.0; BAND];
                    target[first] = -8.0 * sign;
                    target[second] = 8.0 * sign;
                    let input = source_for_rotated_row(&target, &su, &sv);
                    assert_batch(&input, BAND, BAND, &su, &sv, backend)?;
                }
            }
            for bits in [0x0001, 0x03ff, 0x0400, 0x7bff] {
                for sign in [1.0, -1.0] {
                    let mut target = [0.0; BAND];
                    target[0] = -8.0 * half::f16::from_bits(bits).to_f32() * sign;
                    let input = source_for_rotated_row(&target, &su, &sv);
                    assert_batch(&input, BAND, BAND, &su, &sv, backend)?;
                }
            }
            for lower in [0x0000, 0x0001, 0x03ff, 0x3c00, 0x3c01] {
                let midpoint = (half::f16::from_bits(lower).to_f32()
                    + half::f16::from_bits(lower + 1).to_f32())
                    / 2.0;
                for sign in [1.0, -1.0] {
                    let mut target = [0.0; BAND];
                    target[0] = -8.0 * midpoint * sign;
                    let input = source_for_rotated_row(&target, &su, &sv);
                    assert_batch(&input, BAND, BAND, &su, &sv, backend)?;
                }
            }
            for value in [
                0.0,
                -0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                1.0e-20,
            ] {
                assert_batch(&vec![value; BAND * BAND], BAND, BAND, &su, &sv, backend)?;
            }
            let target = std::array::from_fn(|i| (((5 * i + 3) % 16) as f32 - 8.0) / 64.0);
            let input = source_for_rotated_row(&target, &su, &sv);
            assert_batch(&input, BAND, BAND, &su, &sv, backend)?;
            assert_batch(&mixed_values(BAND, BAND, 53), BAND, BAND, &su, &sv, backend)?;
        }
        Ok(())
    }

    fn multiband_tails_and_wide_rows_match_cpu_planes(backend: I4rBackend) -> Result<()> {
        backend.preflight()?;
        for (bands, cols) in [
            (1, 128),
            (3, 256),
            (16, 256),
            (32, 256),
            (33, 256),
            (16, 2560), // 12 + 4 bands in the native 256-block workspace.
            (257, 128), // 256 + 1 bands with a single block per row.
            (1, 32768), // Exactly the native 256-block workspace column limit.
            (1, 32896), // 32768 + 128 columns, preserving row-major plane strides.
        ] {
            let rows = bands * BAND;
            let su = codec::generate_signs(17, "batch.su", cols);
            let sv = codec::generate_signs(23, "batch.sv", rows);
            assert_batch(
                &mixed_values(rows, cols, bands),
                rows,
                cols,
                &su,
                &sv,
                backend,
            )?;
        }
        Ok(())
    }

    fn expert_portions_keep_their_signs_and_ignore_ht_beam(backend: I4rBackend) -> Result<()> {
        backend.preflight()?;
        let cols = 2 * BAND;
        let half_rows = 33 * BAND;
        let rows = 2 * half_rows;
        let sv = codec::generate_signs(19, "expert.local.sv", rows);
        let signs = [
            codec::generate_signs(7, "expert.0.su", cols),
            codec::generate_signs(7, "expert.1.su", cols),
        ];
        assert_ne!(signs[0], signs[1]);
        for (expert, su) in signs.iter().enumerate() {
            let input = mixed_values(rows, cols, expert);
            let expected = cpu_bands(&input, cols, su, &sv)?;
            let mut assembled = EncodedBlock {
                codes: Vec::new(),
                scales: Vec::new(),
            };
            for portion in 0..2 {
                let start = portion * half_rows;
                for local in (0..half_rows).step_by(32 * BAND) {
                    let row = start + local;
                    let take = (half_rows - local).min(32 * BAND);
                    let selected = QuantMeta {
                        su,
                        sv: &sv[row..row + take],
                        ht_backend: if expert == 0 {
                            HtBackend::Hip
                        } else {
                            HtBackend::Cuda
                        },
                        ht_beam: usize::MAX,
                        i4r_backend: backend,
                        ..QuantMeta::default()
                    };
                    let actual = encode_rows(
                        Encoding::I4r,
                        &input[row * cols..(row + take) * cols],
                        take,
                        cols,
                        &selected,
                    )?;
                    assembled.codes.extend(actual.codes);
                    assembled.scales.extend(actual.scales);
                }
            }
            assert_planes(&assembled, &expected);
        }
        Ok(())
    }

    fn reports_numerical_errors_and_recovers_without_stale_status(
        backend: I4rBackend,
    ) -> Result<()> {
        backend.preflight()?;
        let signs = [1.0; BAND];
        let cpu = QuantMeta {
            su: &signs,
            sv: &signs,
            ..QuantMeta::default()
        };
        let gpu = QuantMeta {
            i4r_backend: backend,
            ..cpu
        };
        let valid = mixed_values(BAND, BAND, 71);
        let expected = encode_rows(Encoding::I4r, &valid, BAND, BAND, &cpu)?;
        // Follow the existing HT regression: leave a reported runtime error
        // pending, without changing the selected device or injecting output.
        // SAFETY: a negative ordinal is rejected without changing the device.
        let status = unsafe {
            match backend {
                #[cfg(feature = "hip")]
                I4rBackend::Hip => hipSetDevice(-1),
                #[cfg(feature = "cuda")]
                I4rBackend::Cuda => cudaSetDevice(-1),
                _ => unreachable!(),
            }
        };
        assert_ne!(status, 0);
        assert_planes(
            &encode_rows(Encoding::I4r, &valid, BAND, BAND, &gpu)?,
            &expected,
        );
        assert!(last_error(backend).is_empty());
        for (value, detail) in [(f32::MAX, "rotation overflows f32"), (4096.0, "finite f16")] {
            let invalid = vec![value; BAND * BAND];
            let reference = encode_rows(Encoding::I4r, &invalid, BAND, BAND, &cpu).unwrap_err();
            let error = encode_rows(Encoding::I4r, &invalid, BAND, BAND, &gpu).unwrap_err();
            assert!(reference.to_string().contains(detail), "{reference:#}");
            assert!(
                error
                    .to_string()
                    .contains(&format!("{} I4R encoding failed", backend.label())),
                "{error:#}"
            );
            assert!(error.to_string().contains(detail), "{error:#}");
            assert!(last_error(backend).contains(detail));
            assert_planes(
                &encode_rows(Encoding::I4r, &valid, BAND, BAND, &gpu)?,
                &expected,
            );
            assert!(last_error(backend).is_empty());
        }
        // Eight lanes quantize each row group. On wave64, rows 4..7 and
        // 12..15 occupy the upper halves of the first and second waves.
        // The source row signs below place -8*scale at rotated[row][0] only.
        // +/-65520 is the f16 overflow midpoint, unlike finite +/-65504.
        let mut invalid = vec![0.0; BAND * BAND];
        for rotated_row in [4, 5, 6, 7, 12, 13, 14, 15] {
            for scale in [65520.0f32, -65520.0] {
                for (row, values) in invalid.chunks_exact_mut(BAND).enumerate() {
                    let sign = if (row & rotated_row).count_ones() % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    values.fill(-scale / 16.0 * sign);
                }
                let reference = encode_rows(Encoding::I4r, &invalid, BAND, BAND, &cpu).unwrap_err();
                let error = encode_rows(Encoding::I4r, &invalid, BAND, BAND, &gpu).unwrap_err();
                assert!(
                    reference.to_string().contains("finite f16"),
                    "{reference:#}"
                );
                assert!(
                    error
                        .to_string()
                        .contains(&format!("{} I4R encoding failed", backend.label())),
                    "row={rotated_row} scale={scale}: {error:#}"
                );
                assert!(error.to_string().contains("finite f16"), "{error:#}");
                assert!(last_error(backend).contains("finite f16"));
                assert_planes(
                    &encode_rows(Encoding::I4r, &valid, BAND, BAND, &gpu)?,
                    &expected,
                );
                assert!(last_error(backend).is_empty());
            }
        }
        // CPU rotates the entire request before quantization. A later rotation
        // overflow must win over a prior half-scale failure across native chunks.
        let rows = 33 * BAND;
        let cols = 9 * BAND;
        let mut invalid = vec![0.0; rows * cols];
        for row in 0..BAND {
            invalid[row * cols..row * cols + BAND].fill(4096.0);
        }
        for row in rows - BAND..rows {
            invalid[row * cols..row * cols + BAND].fill(f32::MAX);
        }
        let su = vec![1.0; cols];
        let sv = vec![1.0; rows];
        for i4r_backend in [I4rBackend::Cpu, backend] {
            let meta = QuantMeta {
                su: &su,
                sv: &sv,
                i4r_backend,
                ..QuantMeta::default()
            };
            let error = encode_rows(Encoding::I4r, &invalid, rows, cols, &meta).unwrap_err();
            assert!(
                error.to_string().contains("rotation overflows f32"),
                "{error:#}"
            );
        }
        assert_planes(
            &encode_rows(Encoding::I4r, &valid, BAND, BAND, &gpu)?,
            &expected,
        );
        assert!(last_error(backend).is_empty());
        backend.preflight()?;
        assert!(last_error(backend).is_empty());
        Ok(())
    }

    macro_rules! gpu_device_tests {
        ($feature:literal, $backend:expr,
         $invalid:ident, $corpus:ident, $batches:ident, $experts:ident, $recovery:ident) => {
            #[cfg(feature = $feature)]
            #[test]
            fn $invalid() {
                native_rejects_shapes_nulls_and_extents_without_a_device($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; compares exact I4R codes and half-scale bits"]
            fn $corpus() -> Result<()> {
                corner_corpus_matches_public_cpu_oracle($backend)
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; crosses bounded native row and column batches"]
            fn $batches() -> Result<()> {
                multiband_tails_and_wide_rows_match_cpu_planes($backend)
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; exercises independent expert signs and HT settings"]
            fn $experts() -> Result<()> {
                expert_portions_keep_their_signs_and_ignore_ht_beam($backend)
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; injects real numerical and prior runtime errors"]
            fn $recovery() -> Result<()> {
                reports_numerical_errors_and_recovers_without_stale_status($backend)
            }
        };
    }

    gpu_device_tests!(
        "hip",
        I4rBackend::Hip,
        hip_i4r_native_rejects_shapes_nulls_and_extents_without_a_device,
        hip_i4r_corner_corpus_matches_public_cpu_oracle,
        hip_i4r_multiband_tails_and_wide_rows_match_cpu_planes,
        hip_i4r_expert_portions_keep_their_signs_and_ignore_ht_beam,
        hip_i4r_reports_numerical_errors_and_recovers_without_stale_status
    );
    gpu_device_tests!(
        "cuda",
        I4rBackend::Cuda,
        cuda_i4r_native_rejects_shapes_nulls_and_extents_without_a_device,
        cuda_i4r_corner_corpus_matches_public_cpu_oracle,
        cuda_i4r_multiband_tails_and_wide_rows_match_cpu_planes,
        cuda_i4r_expert_portions_keep_their_signs_and_ignore_ht_beam,
        cuda_i4r_reports_numerical_errors_and_recovers_without_stale_status
    );
}
