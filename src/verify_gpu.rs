//! Verification-only GPU decoding. Encoding provenance never selects this backend.
//!
//! I4R follows `codec::decode_rows`; HT receives the independent verifier's
//! codebook and uses its circular-state mapping, not the encoder's decoder.
//! Both native axis passes use ordered f64 butterflies and one final f32 cast.
//! The native context batches arbitrary complete 128x128 blocks into a lazy
//! workspace smaller than 64 MiB, and returns only after output is complete.
//! HIP and CUDA calls serialize their shared native context with a mutex;
//! concurrent callers retain separate thread-local errors and caller buffers.

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
    Cuda,
}

impl DecodeBackend {
    pub const fn is_gpu(self) -> bool {
        matches!(self, Self::Hip | Self::Cuda)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Hip => "HIP",
            Self::Cuda => "CUDA",
        }
    }

    /// Fail explicitly for an unavailable GPU, even without an eligible tensor.
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
                        native_last_error(self)
                    );
                    Ok(())
                }
                #[cfg(not(feature = "hip"))]
                anyhow::bail!("HIP decode backend requires a build with --features hip");
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // SAFETY: no pointers; native context access is serialized.
                    let result = unsafe { hgn_cuda_decode_preflight() };
                    ensure!(
                        result == 0,
                        "CUDA decode preflight failed: {}",
                        native_last_error(self)
                    );
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                anyhow::bail!("CUDA decode backend requires a build with --features cuda");
            }
        }
    }
}

fn geometry(rows: usize, cols: usize) -> Result<(usize, c_int, c_int)> {
    ensure!(
        rows > 0 && cols > 0 && rows % 128 == 0 && cols % 128 == 0,
        "GPU decode requires nonempty complete 128-row/column blocks"
    );
    let count = rows
        .checked_mul(cols)
        .context("matrix element count overflow")?;
    ensure!(
        count <= isize::MAX as usize / size_of::<f32>(),
        "decoded block length exceeds address space"
    );
    let bands =
        c_int::try_from(rows / 128).context("GPU decode band count exceeds native range")?;
    let cols = c_int::try_from(cols).context("GPU decode column count exceeds native range")?;
    Ok((count, bands, cols))
}

/// Decode one or more I4R bands with one shared input rotation vector.
/// Scales are the existing contiguous little-endian half plane, without pad.
pub fn decode_i4r(
    backend: DecodeBackend,
    block: &EncodedBlock,
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<Vec<f32>> {
    ensure!(
        backend.is_gpu(),
        "CPU backend cannot use GPU-only I4R decode"
    );
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
    match backend {
        DecodeBackend::Hip => {
            #[cfg(feature = "hip")]
            {
                native_output(
                    backend,
                    count,
                    "I4R",
                    "decoded block contains non-finite values",
                    |out| {
                        // SAFETY: all input extents and output capacity match the
                        // checked shape. Native treats scales as bytes on the host,
                        // so this cast needs neither host u16 alignment nor a copy.
                        // Every transfer completes before return, including errors.
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
        DecodeBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                native_output(
                    backend,
                    count,
                    "I4R",
                    "decoded block contains non-finite values",
                    |out| {
                        // SAFETY: extents match the checked shape. Native reads
                        // the scale plane as bytes, allowing this unaligned cast,
                        // and completes all transfers before returning.
                        unsafe {
                            hgn_cuda_decode_i4r(
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
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA decode backend requires a build with --features cuda")
        }
        DecodeBackend::Cpu => unreachable!("CPU backend rejected before GPU decode"),
    }
}

/// Decode independent HT3/HT4 verification bands, including paired gate/up.
/// `su` contains `cols` entries per band, `sv` contains `rows` entries, and
/// `bytes` includes both projections when `row_tiles == 16`. Only tile offsets
/// 0 (ordinary/gate) and 8 (paired up) are legal. The supplied 65536-value
/// codebook is content-cached by native code, never replaced by an encoder one.
#[allow(clippy::too_many_arguments)]
pub fn decode_ht(
    backend: DecodeBackend,
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
    ensure!(
        backend.is_gpu(),
        "CPU backend cannot use GPU-only HT decode"
    );
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
    match backend {
        DecodeBackend::Hip => {
            #[cfg(feature = "hip")]
            {
                native_output(
                    backend,
                    _count,
                    "HT",
                    "nonfinite HT verifier value",
                    |out| {
                        // SAFETY: every extent and native integer is checked above.
                        // Output is exclusively borrowed and native success means
                        // all values are written and asynchronous work is complete.
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
                    },
                )
            }
            #[cfg(not(feature = "hip"))]
            anyhow::bail!("HIP decode backend requires a build with --features hip")
        }
        DecodeBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                native_output(
                    backend,
                    _count,
                    "HT",
                    "nonfinite HT verifier value",
                    |out| {
                        // SAFETY: checked extents and native integers; the output
                        // buffer is exclusive and fully initialized synchronously
                        // on success, with no caller pointers retained.
                        unsafe {
                            hgn_cuda_decode_ht(
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
                    },
                )
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA decode backend requires a build with --features cuda")
        }
        DecodeBackend::Cpu => unreachable!("CPU backend rejected before GPU decode"),
    }
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

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn hgn_cuda_decode_i4r(
        codes: *const u8,
        scales: *const u16,
        su: *const f32,
        sv: *const f32,
        out: *mut f32,
        bands: c_int,
        cols: c_int,
    ) -> c_int;
    fn hgn_cuda_decode_ht(
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
    fn hgn_cuda_decode_preflight() -> c_int;
    fn ht_cuda_last_error() -> *const std::ffi::c_char;
}

#[cfg(any(feature = "hip", feature = "cuda"))]
fn native_last_error(backend: DecodeBackend) -> String {
    // SAFETY: the shared ABI returns thread-local NUL-terminated text, valid
    // until this thread's next native call. Copy it immediately.
    let error = match backend {
        #[cfg(feature = "hip")]
        DecodeBackend::Hip => unsafe { ht_hip_last_error() },
        #[cfg(feature = "cuda")]
        DecodeBackend::Cuda => unsafe { ht_cuda_last_error() },
        _ => unreachable!("native error requires an enabled GPU backend"),
    };
    if error.is_null() {
        return format!(
            "native {} decoder returned no error detail",
            backend.label()
        );
    }
    unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(any(feature = "hip", feature = "cuda"))]
fn native_output(
    backend: DecodeBackend,
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
        "{} {codec} decode failed: {}",
        backend.label(),
        native_last_error(backend)
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
    fn decode_backends_parse_and_label_without_feature_gates() {
        use clap::ValueEnum;

        assert_eq!(DecodeBackend::default(), DecodeBackend::Cpu);
        for (name, backend, label, is_gpu) in [
            ("cpu", DecodeBackend::Cpu, "CPU", false),
            ("hip", DecodeBackend::Hip, "HIP", true),
            ("cuda", DecodeBackend::Cuda, "CUDA", true),
        ] {
            assert_eq!(DecodeBackend::from_str(name, false).unwrap(), backend);
            assert_eq!(backend.label(), label);
            assert_eq!(backend.is_gpu(), is_gpu);
        }
        assert!(DecodeBackend::from_str("vulkan", false).is_err());
    }

    #[test]
    fn decode_rejects_shapes_planes_and_metadata_before_native_reads() {
        for backend in [DecodeBackend::Hip, DecodeBackend::Cuda] {
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
            for (rows, cols, error) in [
                (0, 128, "complete 128-row/column blocks"),
                (128, 0, "complete 128-row/column blocks"),
                (127, 128, "complete 128-row/column blocks"),
                (128, 129, "complete 128-row/column blocks"),
                (usize::MAX & !127, 128, "element count overflow"),
            ] {
                assert!(decode_i4r(backend, &block, rows, cols, &meta)
                    .unwrap_err()
                    .to_string()
                    .contains(error));
            }
            block.codes.pop();
            assert!(decode_i4r(backend, &block, 128, 128, &meta)
                .unwrap_err()
                .to_string()
                .contains("plane lengths"));
            block.codes.push(0x88);
            block.scales.extend_from_slice(&[0, 0]);
            assert!(decode_i4r(backend, &block, 128, 128, &meta)
                .unwrap_err()
                .to_string()
                .contains("plane lengths"));
            block.scales.truncate(256);
            block.scales[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
            assert!(decode_i4r(backend, &block, 128, 128, &meta)
                .unwrap_err()
                .to_string()
                .contains("quantization scale"));
            block.scales[..2].fill(0);
            for value in [f32::NAN, f32::INFINITY, 0.1, f32::from_bits(1)] {
                let invalid = vec![value; 128];
                for meta in [
                    QuantMeta {
                        su: &invalid,
                        ..meta
                    },
                    QuantMeta {
                        sv: &invalid,
                        ..meta
                    },
                ] {
                    assert!(decode_i4r(backend, &block, 128, 128, &meta)
                        .unwrap_err()
                        .to_string()
                        .contains("representable f16"));
                }
            }
            let meta = QuantMeta {
                sv: &sv[..127],
                ..meta
            };
            assert!(decode_i4r(backend, &block, 128, 128, &meta)
                .unwrap_err()
                .to_string()
                .contains("vector dimensions"));
        }
    }

    #[test]
    fn ht_decode_rejects_invalid_width_layout_and_nonfinite_metadata() {
        for backend in [DecodeBackend::Hip, DecodeBackend::Cuda] {
            let bytes = vec![0; 128 * 128 / 2];
            let su = vec![1.0; 128];
            let sv = vec![1.0; 128];
            let mut codebook = vec![0.0; 65536];
            for (bits, tiles, offset, error) in [
                (2, 8, 0, "verifier width"),
                (5, 8, 0, "verifier width"),
                (4, 8, 8, "tile geometry"),
                (4, 16, 1, "tile geometry"),
                (4, 24, 0, "tile geometry"),
            ] {
                assert!(decode_ht(
                    backend, &bytes, 128, 128, bits, &su, &sv, tiles, offset, &codebook
                )
                .unwrap_err()
                .to_string()
                .contains(error));
            }
            for (rows, cols, error) in [
                (0, 128, "complete 128-row/column blocks"),
                (128, 0, "complete 128-row/column blocks"),
                (127, 128, "complete 128-row/column blocks"),
                (128, 129, "complete 128-row/column blocks"),
                (usize::MAX & !127, 128, "element count overflow"),
            ] {
                assert!(
                    decode_ht(backend, &bytes, rows, cols, 4, &su, &sv, 8, 0, &codebook)
                        .unwrap_err()
                        .to_string()
                        .contains(error)
                );
            }
            assert!(decode_ht(
                backend,
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
            assert!(decode_ht(
                backend,
                &bytes,
                128,
                128,
                4,
                &su[..127],
                &sv,
                8,
                0,
                &codebook
            )
            .unwrap_err()
            .to_string()
            .contains("geometry"));
            assert!(decode_ht(
                backend,
                &bytes,
                128,
                128,
                4,
                &su,
                &sv,
                8,
                0,
                &codebook[..65535]
            )
            .unwrap_err()
            .to_string()
            .contains("codebook length"));
            codebook[65535] = f32::NAN;
            assert!(
                decode_ht(backend, &bytes, 128, 128, 4, &su, &sv, 8, 0, &codebook)
                    .unwrap_err()
                    .to_string()
                    .contains("nonfinite HT verifier codebook")
            );
            let invalid = vec![f32::NEG_INFINITY; 128];
            assert!(
                decode_ht(backend, &bytes, 128, 128, 4, &su, &invalid, 8, 0, &codebook)
                    .unwrap_err()
                    .to_string()
                    .contains("nonfinite HT verifier rotation")
            );
        }
    }

    fn assert_backend_rejected(backend: DecodeBackend, error: &str) {
        if backend.is_gpu() {
            assert!(backend.preflight().unwrap_err().to_string().contains(error));
        } else {
            backend.preflight().unwrap();
        }
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
        assert!(decode_i4r(backend, &block, 128, 128, &meta)
            .unwrap_err()
            .to_string()
            .contains(error));
        assert!(decode_ht(
            backend,
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
        .contains(error));
    }

    #[test]
    fn cpu_backend_cannot_use_gpu_only_helpers() {
        assert_backend_rejected(DecodeBackend::Cpu, "CPU backend cannot use GPU-only");
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn default_build_has_explicit_hip_unavailable_errors() {
        assert_backend_rejected(
            DecodeBackend::Hip,
            "HIP decode backend requires a build with --features hip",
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_backend_requires_feature() {
        assert_backend_rejected(
            DecodeBackend::Cuda,
            "CUDA decode backend requires a build with --features cuda",
        );
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    mod device {
        use super::*;
        use crate::codec::{decode_rows, Encoding};
        use crate::verify::{decode_ht_independent, VERIFICATION_HT_CODEBOOK};

        #[cfg(feature = "hip")]
        unsafe extern "C" {
            fn hipSetDevice(device: c_int) -> c_int;
        }

        #[cfg(feature = "cuda")]
        unsafe extern "C" {
            fn cudaSetDevice(device: c_int) -> c_int;
        }

        // Deliberately invalid raw ABI calls share the corpus without routing
        // through Rust validation. The caller must uphold each FFI safety contract.
        macro_rules! native_call {
            ($backend:expr, $hip:ident, $cuda:ident $(, $arg:expr)* $(,)?) => {
                match $backend {
                    #[cfg(feature = "hip")]
                    DecodeBackend::Hip => $hip($($arg),*),
                    #[cfg(feature = "cuda")]
                    DecodeBackend::Cuda => $cuda($($arg),*),
                    _ => panic!("native test requires an enabled GPU backend"),
                }
            };
        }

        fn inject_runtime_error(backend: DecodeBackend) {
            // SAFETY: a negative device ordinal is rejected without switching
            // devices. Its previously reported error must not taint a valid launch.
            assert_ne!(
                unsafe { native_call!(backend, hipSetDevice, cudaSetDevice, -1) },
                0
            );
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

        fn constant_ht_expected(
            rows: usize,
            cols: usize,
            constant: f32,
            su: &[f32],
            sv: &[f32],
        ) -> Vec<f32> {
            // H * constant * H has only one DC value per 128x128 block.
            // sv=1/128 preserves the original constant there, including the
            // smallest subnormal and negative zero.
            (0..rows * cols)
                .map(|index| {
                    let row = index / cols;
                    let col = index % cols;
                    let transformed = if row % 128 == 0 && col % 128 == 0 {
                        f64::from(constant) * 128.0
                    } else {
                        0.0
                    };
                    (transformed * f64::from(sv[row]) * f64::from(su[row / 128 * cols + col]))
                        as f32
                })
                .collect()
        }

        fn i4r_decode_matches_cpu_edge_corpus(backend: DecodeBackend) {
            backend.preflight().unwrap();
            for (rows, cols) in [(128, 128), (256, 640), (128, 2560)] {
                let (block, su, sv) = i4r_input(rows, cols);
                let expected = cpu_i4r(&block, rows, cols, &su, &sv);
                inject_runtime_error(backend);
                let meta = QuantMeta {
                    su: &su,
                    sv: &sv,
                    ..Default::default()
                };
                equal_bits(
                    &decode_i4r(backend, &block, rows, cols, &meta).unwrap(),
                    &expected,
                );
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
                        &decode_i4r(backend, &block, 128, 128, &meta).unwrap(),
                        &cpu_i4r(&block, 128, 128, &su, &sv),
                    );
                }
            }
        }

        fn ht_decode_matches_independent_verifier(backend: DecodeBackend) {
            backend.preflight().unwrap();
            let mut saw_subnormal = false;
            for bits in [3, 4] {
                for (tiles, offset) in [(8, 0), (16, 0), (16, 8)] {
                    for (rows, cols) in [(128, 128), (256, 640)] {
                        let (bytes, su, sv) = ht_input(rows, cols, bits, tiles);
                        let expected = cpu_ht(&bytes, rows, cols, bits, tiles, offset, &su, &sv);
                        saw_subnormal |= expected.iter().any(|value| value.is_subnormal());
                        inject_runtime_error(backend);
                        equal_bits(
                            &decode_ht(
                                backend,
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
                                backend,
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

        fn decode_crosses_band_and_column_batch_boundaries(backend: DecodeBackend) {
            backend.preflight().unwrap();
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
                    &decode_i4r(backend, &block, rows, cols, &meta).unwrap(),
                    &cpu_i4r(&block, rows, cols, &su, &sv),
                );
                for (bits, tiles, offset) in [(3, 16, 8), (4, 8, 0)] {
                    let (bytes, su, sv) = ht_input(rows, cols, bits, tiles);
                    equal_bits(
                        &decode_ht(
                            backend,
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

        fn ht_decode_caches_contents_not_addresses(backend: DecodeBackend) {
            backend.preflight().unwrap();
            let (rows, cols) = (256, 256);
            let mut codebook = VERIFICATION_HT_CODEBOOK.to_vec();
            for bits in [3, 4] {
                let (bytes, su, sv) = ht_input(rows, cols, bits, 16);
                codebook.copy_from_slice(&VERIFICATION_HT_CODEBOOK);
                equal_bits(
                    &decode_ht(
                        backend, &bytes, rows, cols, bits, &su, &sv, 16, 8, &codebook,
                    )
                    .unwrap(),
                    &cpu_ht(&bytes, rows, cols, bits, 16, 8, &su, &sv),
                );
                let su: Vec<f32> = (0..rows / 128 * cols)
                    .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                    .collect();
                let sv = vec![1.0 / 128.0; rows];
                for constant in [0.0, -0.0, f32::from_bits(1), -f32::from_bits(1), 1.0, -1.0] {
                    codebook.fill(constant);
                    let expected = constant_ht_expected(rows, cols, constant, &su, &sv);
                    equal_bits(
                        &decode_ht(
                            backend, &bytes, rows, cols, bits, &su, &sv, 16, 8, &codebook,
                        )
                        .unwrap(),
                        &expected,
                    );
                }
            }
        }

        fn decode_errors_do_not_read_invalid_extents_or_poison_context(backend: DecodeBackend) {
            backend.preflight().unwrap();
            // SAFETY: each deliberately invalid ABI call must reject its shape,
            // null pointers, or overflowing address extent before any read.
            unsafe {
                assert_ne!(
                    native_call!(
                        backend,
                        hgn_hip_decode_i4r,
                        hgn_cuda_decode_i4r,
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
                    native_call!(
                        backend,
                        hgn_hip_decode_i4r,
                        hgn_cuda_decode_i4r,
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
                assert!(native_last_error(backend).contains("shape overflows"));
                assert_ne!(
                    native_call!(
                        backend,
                        hgn_hip_decode_i4r,
                        hgn_cuda_decode_i4r,
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
                assert!(native_last_error(backend).contains("buffer extent"));
                assert_ne!(
                    native_call!(
                        backend,
                        hgn_hip_decode_ht,
                        hgn_cuda_decode_ht,
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
                assert!(native_last_error(backend).contains("tile geometry"));
                assert_ne!(
                    native_call!(
                        backend,
                        hgn_hip_decode_ht,
                        hgn_cuda_decode_ht,
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
                assert!(native_last_error(backend).contains("buffer extent"));
            }
            let bytes = vec![0; 8192];
            let signs = vec![1.0; 128];
            let codebook = vec![f32::MAX; 65536];
            assert!(
                decode_ht(backend, &bytes, 128, 128, 4, &signs, &signs, 8, 0, &codebook)
                    .unwrap_err()
                    .to_string()
                    .contains("nonfinite HT verifier value")
            );
            let (bytes, su, sv) = ht_input(128, 128, 4, 8);
            equal_bits(
                &decode_ht(
                    backend,
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

        fn decode_serializes_concurrent_callers(backend: DecodeBackend) {
            backend.preflight().unwrap();
            let signs = [1.0; 128];
            let scales = [1.0 / 128.0; 128];
            let constants = [1.0, -1.0, f32::from_bits(1), -f32::from_bits(1)];
            // Prepare every oracle before starting threads: an oracle failure
            // must not strand a caller at the start barrier.
            let cases: Vec<_> = (0..constants.len())
                .map(|caller| {
                    let bits = 3 + caller % 2;
                    let offset = 8 * (caller / 2);
                    let block = EncodedBlock {
                        codes: vec![0x11 * (caller as u8 + 1); 8192],
                        scales: (0..128)
                            .flat_map(|_| f16::from_f32(caller as f32 + 1.0).to_le_bytes())
                            .collect(),
                    };
                    let bytes = vec![0x55 * caller as u8; 8 * 16 * 32 * bits];
                    let expected_i4r = cpu_i4r(&block, 128, 128, &signs, &signs);
                    let expected_ht = cpu_ht(&bytes, 128, 128, bits, 16, offset, &signs, &scales);
                    let expected_constant =
                        constant_ht_expected(128, 128, constants[caller], &signs, &scales);
                    (
                        block,
                        bytes,
                        expected_i4r,
                        expected_ht,
                        expected_constant,
                        VERIFICATION_HT_CODEBOOK.to_vec(),
                    )
                })
                .collect();
            let start = std::sync::Barrier::new(cases.len());
            let (start, signs, scales) = (&start, &signs, &scales);
            std::thread::scope(|scope| {
                for (
                    caller,
                    (block, bytes, expected_i4r, expected_ht, expected_constant, mut codebook),
                ) in cases.into_iter().enumerate()
                {
                    scope.spawn(move || {
                        // Only the first call is synchronized. Later assertions
                        // may fail without trapping another worker at a barrier.
                        start.wait();
                        let bits = 3 + caller % 2;
                        let offset = 8 * (caller / 2);
                        let meta = QuantMeta {
                            su: signs,
                            sv: signs,
                            ..Default::default()
                        };
                        for _ in 0..4 {
                            inject_runtime_error(backend);
                            equal_bits(
                                &decode_i4r(backend, &block, 128, 128, &meta).unwrap(),
                                &expected_i4r,
                            );
                            codebook.copy_from_slice(&VERIFICATION_HT_CODEBOOK);
                            equal_bits(
                                &decode_ht(
                                    backend, &bytes, 128, 128, bits, signs, scales, 16, offset,
                                    &codebook,
                                )
                                .unwrap(),
                                &expected_ht,
                            );
                            // Different contents at stable, thread-owned addresses
                            // exercise the shared native cache under contention.
                            codebook.fill(constants[caller]);
                            equal_bits(
                                &decode_ht(
                                    backend, &bytes, 128, 128, bits, signs, scales, 16, offset,
                                    &codebook,
                                )
                                .unwrap(),
                                &expected_constant,
                            );
                        }
                    });
                }
            });
        }

        macro_rules! device_tests {
            ($backend:expr, $requirement:literal, $($name:ident => $corpus:ident),+ $(,)?) => {
                $(
                    #[test]
                    #[ignore = $requirement]
                    fn $name() {
                        $corpus($backend);
                    }
                )+
            };
        }

        #[cfg(feature = "hip")]
        device_tests!(
            DecodeBackend::Hip,
            "requires gfx1151 HIP; bit-exact decoding against CPU/independent oracles",
            hip_i4r_decode_matches_cpu_edge_corpus => i4r_decode_matches_cpu_edge_corpus,
            hip_ht_decode_matches_independent_verifier => ht_decode_matches_independent_verifier,
            hip_decode_crosses_band_and_column_batch_boundaries => decode_crosses_band_and_column_batch_boundaries,
            hip_ht_decode_caches_contents_not_addresses => ht_decode_caches_contents_not_addresses,
            hip_decode_errors_do_not_read_invalid_extents_or_poison_context => decode_errors_do_not_read_invalid_extents_or_poison_context,
            hip_decode_serializes_concurrent_callers => decode_serializes_concurrent_callers,
        );

        #[cfg(feature = "cuda")]
        device_tests!(
            DecodeBackend::Cuda,
            "requires CUDA device; bit-exact decoding against CPU/independent oracles",
            cuda_i4r_decode_matches_cpu_edge_corpus => i4r_decode_matches_cpu_edge_corpus,
            cuda_ht_decode_matches_independent_verifier => ht_decode_matches_independent_verifier,
            cuda_decode_crosses_band_and_column_batch_boundaries => decode_crosses_band_and_column_batch_boundaries,
            cuda_ht_decode_caches_contents_not_addresses => ht_decode_caches_contents_not_addresses,
            cuda_decode_errors_do_not_read_invalid_extents_or_poison_context => decode_errors_do_not_read_invalid_extents_or_poison_context,
            cuda_decode_serializes_concurrent_callers => decode_serializes_concurrent_callers,
        );
    }
}
