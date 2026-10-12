use bf16_to_hgn::codec::{encode_rows, Encoding, Q4cpBackend, QuantMeta};
#[cfg(feature = "cuda")]
use bf16_to_hgn::codec::NF4_CODEBOOK;

#[test]
fn q4cp_default_backend_is_cpu() {
    assert_eq!(QuantMeta::default().q4cp_backend, Q4cpBackend::Cpu);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn q4cp_cuda_requires_feature_without_falling_back_to_cpu() {
    let input = vec![0.5; 64];
    let meta = QuantMeta {
        q4cp_backend: Q4cpBackend::Cuda,
        ..QuantMeta::default()
    };
    for error in [
        Q4cpBackend::Cuda.preflight().unwrap_err(),
        encode_rows(Encoding::Q4Cp, &input, 2, 32, &meta).unwrap_err(),
    ] {
        let text = error.to_string();
        assert!(text.contains("CUDA Q4C-P"), "{text}");
        assert!(text.contains("--features cuda"), "{text}");
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    /// Deterministic xorshift values with exact ties, signed zeros, half
    /// subnormal ranges, constant groups and wide dynamic range per group.
    fn values(count: usize, salt: u64) -> Vec<f32> {
        let mut state = 0x9e3779b97f4a7c15u64 ^ salt.wrapping_mul(0xbf58476d1ce4e5b9);
        let mids: Vec<f32> = (0..15)
            .map(|i| (NF4_CODEBOOK[i] as f64 * 0.5 + NF4_CODEBOOK[i + 1] as f64 * 0.5) as f32)
            .collect();
        (0..count)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                let group = i / 32;
                match group % 9 {
                    0 => 0.0,
                    1 => if i % 2 == 0 { -0.0 } else { 0.0 },
                    // Group absmax 1 with codebook-midpoint members: exact ties
                    // at scale 1 and near-ties for the reduced multipliers.
                    2 => if i % 32 == 0 { 1.0 } else { mids[(state % 15) as usize] },
                    3 => unit * 1.0e-6,
                    4 => unit * 6.0e-8,
                    5 => 0.25,
                    6 => unit * 60000.0,
                    7 => f32::from_bits(state as u32 & 0x807f_ffff),
                    // Up to 2^15: every candidate scale stays a finite half.
                    _ => unit * (1u32 << (state % 16)) as f32,
                }
            })
            .collect()
    }

    fn assert_same(rows: usize, cols: usize, input: &[f32], codebook: &[f32; 16]) {
        let cpu = QuantMeta {
            codebook,
            ..QuantMeta::default()
        };
        let gpu = QuantMeta {
            q4cp_backend: Q4cpBackend::Cuda,
            ..cpu
        };
        let expected = encode_rows(Encoding::Q4Cp, input, rows, cols, &cpu).unwrap();
        let actual = encode_rows(Encoding::Q4Cp, input, rows, cols, &gpu).unwrap();
        for (plane, a, e) in [
            ("codes", &actual.codes, &expected.codes),
            ("scales", &actual.scales, &expected.scales),
        ] {
            assert_eq!(a.len(), e.len(), "{plane} length {rows}x{cols}");
            if let Some(index) = a.iter().zip(e.iter()).position(|(x, y)| x != y) {
                panic!(
                    "{plane} byte {index} differs for {rows}x{cols}: CUDA {:#04x}, CPU {:#04x}",
                    a[index], e[index]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a visible CUDA device"]
    fn cuda_q4cp_matches_cpu_bytes_across_shapes() {
        Q4cpBackend::Cuda.preflight().unwrap();
        for (rows, cols, salt) in [
            (1, 32, 1),
            (1, 64, 2),
            (3, 96, 3),
            (48, 2560, 4),
            (128, 4096, 5),
            (7, 32 * 1000, 6),
            // More than one 262,144-group device batch, ending mid-batch.
            (1100, 8192, 7),
        ] {
            let input = values(rows * cols, salt);
            assert_same(rows, cols, &input, &NF4_CODEBOOK);
        }
    }

    #[test]
    #[ignore = "requires a visible CUDA device"]
    fn cuda_q4cp_matches_cpu_bytes_for_custom_codebooks() {
        let uniform: [f32; 16] = std::array::from_fn(|i| (i as f32 - 7.5) / 7.5);
        let skewed: [f32; 16] = std::array::from_fn(|i| (i as f32 - 3.0) * 0.37 + 1.0e-3);
        let tiny: [f32; 16] = std::array::from_fn(|i| (i as f32 + 1.0) * 1.0e-3);
        for (index, codebook) in [uniform, skewed, tiny].iter().enumerate() {
            // Scale by cmax so absmax/cmax stays within the finite half range.
            let cmax = codebook[0].abs().max(codebook[15].abs());
            let input: Vec<f32> = values(64 * 512, 100 + index as u64)
                .into_iter()
                .map(|value| value * cmax)
                .collect();
            assert_same(64, 512, &input, codebook);
        }
    }

    #[test]
    #[ignore = "requires a visible CUDA device"]
    fn cuda_q4cp_scale_overflow_reports_cpu_error() {
        let mut input = vec![0.25f32; 4 * 64];
        input[3 * 64 + 40] = 70000.0;
        input[2 * 64 + 1] = 80000.0;
        let cpu = QuantMeta::default();
        let gpu = QuantMeta {
            q4cp_backend: Q4cpBackend::Cuda,
            ..cpu
        };
        let expected = encode_rows(Encoding::Q4Cp, &input, 4, 64, &cpu).unwrap_err();
        let actual = encode_rows(Encoding::Q4Cp, &input, 4, 64, &gpu).unwrap_err();
        assert_eq!(format!("{actual:#}"), format!("{expected:#}"));
        // The device stays usable after the reported overflow.
        let input = values(4 * 64, 9);
        assert_same(4, 64, &input, &NF4_CODEBOOK);
    }

    #[test]
    #[ignore = "requires a visible CUDA device"]
    fn cuda_q4cp_concurrent_callers_match_cpu() {
        std::thread::scope(|scope| {
            for salt in 0..4 {
                scope.spawn(move || {
                    let input = values(256 * 1024, 200 + salt);
                    assert_same(256, 1024, &input, &NF4_CODEBOOK);
                });
            }
        });
    }
}
