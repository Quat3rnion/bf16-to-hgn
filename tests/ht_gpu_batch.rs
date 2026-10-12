use bf16_to_hgn::codec::{encode_rows, generate_signs, ht_row_scales, Encoding, QuantMeta};
#[cfg(any(feature = "cuda", feature = "hip"))]
use bf16_to_hgn::codec::HtBackend;

fn values(rows: usize, cols: usize, salt: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i * 37 + salt * 101) % 509) as f32 - 254.0) / 128.0)
        .collect()
}

/// Concatenated single-band CPU calls: the conversion reference for HT.
fn cpu_bands(encoding: Encoding, input: &[f32], rows: usize, cols: usize, su: &[f32], sv: &[f32]) -> Vec<u8> {
    let mut output = Vec::new();
    for band in 0..rows / 128 {
        let meta = QuantMeta {
            su,
            sv: &sv[band * 128..(band + 1) * 128],
            ht_beam: 8,
            ..QuantMeta::default()
        };
        let block = encode_rows(
            encoding,
            &input[band * 128 * cols..(band + 1) * 128 * cols],
            128,
            cols,
            &meta,
        )
        .unwrap();
        output.extend_from_slice(&block.codes);
    }
    output
}

#[test]
fn ht_cpu_accepts_multiple_bands_as_concatenation() {
    let (rows, cols) = (3 * 128, 256);
    let input = values(rows, cols, 1);
    let su = generate_signs(5, "su", cols);
    let signs = generate_signs(5, "sv", rows);
    let sv = ht_row_scales(&input, rows, cols, &signs).unwrap();
    for encoding in [Encoding::Ht, Encoding::Ht3] {
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ht_beam: 8,
            ..QuantMeta::default()
        };
        let batched = encode_rows(encoding, &input, rows, cols, &meta).unwrap();
        assert_eq!(batched.codes, cpu_bands(encoding, &input, rows, cols, &su, &sv));
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a visible CUDA device"]
fn cuda_ht_multiple_bands_match_concatenated_cpu_bands() {
    multiple_bands_match_concatenated_cpu_bands(HtBackend::Cuda);
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires a visible gfx1151 HIP device"]
fn hip_ht_multiple_bands_match_concatenated_cpu_bands() {
    multiple_bands_match_concatenated_cpu_bands(HtBackend::Hip);
}

#[cfg(any(feature = "cuda", feature = "hip"))]
fn multiple_bands_match_concatenated_cpu_bands(backend: HtBackend) {
    backend.preflight().unwrap();
    // 33x128x1024 is 16,896 tiles: crosses one 16,384-tile device launch.
    for (rows, cols, salt) in [(5 * 128, 640, 2), (2 * 128, 2560, 3), (33 * 128, 1024, 6)] {
        let input = values(rows, cols, salt);
        let su = generate_signs(9, "su", cols);
        let signs = generate_signs(9, "sv", rows);
        let sv = ht_row_scales(&input, rows, cols, &signs).unwrap();
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            let meta = QuantMeta {
                su: &su,
                sv: &sv,
                ht_beam: 8,
                ht_backend: backend,
                ..QuantMeta::default()
            };
            let batched = encode_rows(encoding, &input, rows, cols, &meta).unwrap();
            assert_eq!(
                batched.codes,
                cpu_bands(encoding, &input, rows, cols, &su, &sv),
                "{encoding:?} {rows}x{cols}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a visible CUDA device"]
fn cuda_ht_rejected_rotations_report_the_cpu_error() {
    rejected_rotations_report_the_cpu_error(HtBackend::Cuda);
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires a visible gfx1151 HIP device"]
fn hip_ht_rejected_rotations_report_the_cpu_error() {
    rejected_rotations_report_the_cpu_error(HtBackend::Hip);
}

#[cfg(any(feature = "cuda", feature = "hip"))]
fn rejected_rotations_report_the_cpu_error(backend: HtBackend) {
    let (rows, cols) = (2 * 128, 256);
    let su = generate_signs(4, "su", cols);
    // Zero row scale with a nonzero source value in that row.
    let mut input = values(rows, cols, 4);
    let signs = generate_signs(4, "sv", rows);
    let mut sv = ht_row_scales(&input, rows, cols, &signs).unwrap();
    sv[130] = 0.0;
    input[130 * cols + 7] = 0.25;
    // Huge values: rotation exceeds the 1e17 trellis cost range.
    let huge: Vec<f32> = (0..rows * cols).map(|i| if i % 3 == 0 { 3.0e38 } else { -3.0e38 }).collect();
    let ones = vec![1.0f32; rows];
    for (input, sv) in [(&input, &sv), (&huge, &ones)] {
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            let cpu = QuantMeta {
                su: &su,
                sv,
                ht_beam: 8,
                ..QuantMeta::default()
            };
            let gpu = QuantMeta {
                ht_backend: backend,
                ..cpu
            };
            let expected = encode_rows(encoding, input, rows, cols, &cpu).unwrap_err();
            let actual = encode_rows(encoding, input, rows, cols, &gpu).unwrap_err();
            assert_eq!(format!("{actual:#}"), format!("{expected:#}"));
        }
    }
    // The device remains usable after a rejected rotation.
    let input = values(rows, cols, 5);
    let sv = ht_row_scales(&input, rows, cols, &generate_signs(5, "sv", rows)).unwrap();
    let gpu = QuantMeta {
        su: &su,
        sv: &sv,
        ht_beam: 8,
        ht_backend: backend,
        ..QuantMeta::default()
    };
    assert_eq!(
        encode_rows(Encoding::Ht, &input, rows, cols, &gpu).unwrap().codes,
        cpu_bands(Encoding::Ht, &input, rows, cols, &su, &sv)
    );
}
