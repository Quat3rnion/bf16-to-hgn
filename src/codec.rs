//! Bounded codecs for the native Flash-Next v2 and HT43 storage profiles.
//!
//! Layouts and golden vectors: jtsylve/hgn-spec 1.1.0, commit
//! 79e75fd4ae7642478823525b0a3212951f584af5 (STORAGE.md, TEST-VECTORS.md).
//! This is an uncalibrated, deterministic encoder, not peonist's unpublished
//! recipe: fixed NF4 + group scale search, affine min/max, signed I4R RTN,
//! and two-pass cyclic HT beam search with signed row-RMS normalization.
//! The caller streams rows and places the returned planes at the offsets from
//! `payload_layout`; a block never includes a tensor header or inter-plane pad.

use anyhow::{bail, ensure, Context, Result};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use std::ffi::c_int;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Encoding {
    Bf16,
    F16,
    F32,
    I64,
    Q4Cp,
    Q8G64,
    Q6G64,
    Fp8G128,
    Ht,
    Ht3,
    I4r,
}

impl Encoding {
    pub fn code(self) -> u32 {
        match self {
            Self::Bf16 => 0,
            Self::F32 => 1,
            Self::F16 => 2,
            Self::I64 => 4,
            Self::Q4Cp => 5,
            Self::Q8G64 => 7,
            Self::Fp8G128 => 10,
            Self::Ht | Self::Ht3 => 16,
            Self::I4r => 23,
            Self::Q6G64 => 24,
        }
    }

    pub fn variant(self) -> u32 {
        match self {
            Self::Q4Cp => 2,
            Self::Q6G64 => 64,
            Self::Ht => 0x1208,
            Self::Ht3 => 0x1206,
            Self::I4r => 128,
            _ => 0,
        }
    }

    pub fn ht_bits(self) -> Option<usize> {
        match self {
            Self::Ht => Some(4),
            Self::Ht3 => Some(3),
            _ => None,
        }
    }
}

// The exact f32 bits are the NF4 codebook in TEST-VECTORS.md, not rounded
// decimal approximations. Variant 2 permits this codebook as well as learned ones.
pub const NF4_CODEBOOK: [f32; 16] = [
    f32::from_bits(0xbf800000),
    f32::from_bits(0xbf3239b1),
    f32::from_bits(0xbf066b30),
    f32::from_bits(0xbeca32a0),
    f32::from_bits(0xbe91a24d),
    f32::from_bits(0xbe3d353f),
    f32::from_bits(0xbdba7871),
    f32::from_bits(0x00000000),
    f32::from_bits(0x3da2faff),
    f32::from_bits(0x3e24cae3),
    f32::from_bits(0x3e7c04dd),
    f32::from_bits(0x3ead033a),
    f32::from_bits(0x3ee1a4b8),
    f32::from_bits(0x3f1007ab),
    f32::from_bits(0x3f3913b3),
    f32::from_bits(0x3f800000),
];

pub fn q4c_codebook() -> [f32; 16] {
    NF4_CODEBOOK
}
/// Encoding backend for HT3/HT4 trellis search only.
/// Encoding rotations remain on CPU. I4R chooses its independent backend through
/// [`I4rBackend`], and verification through [`crate::verify_gpu::DecodeBackend`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum HtBackend {
    #[default]
    Cpu,
    Hip,
    Cuda,
}

impl HtBackend {
    /// Device family used by HT trellis search only.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Hip => "HIP",
            Self::Cuda => "CUDA",
        }
    }

    pub const fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Validate real device availability without encoding a band.
    pub fn preflight(self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Hip => {
                #[cfg(feature = "hip")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { ht_hip_preflight() };
                    ensure!(result == 0, "HIP HT preflight failed: {}", hip_last_error());
                    Ok(())
                }
                #[cfg(not(feature = "hip"))]
                bail!("HIP HT backend requires a build with --features hip");
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { ht_cuda_preflight() };
                    ensure!(
                        result == 0,
                        "CUDA HT preflight failed: {}",
                        cuda_last_error()
                    );
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                bail!("CUDA HT backend requires a build with --features cuda");
            }
        }
    }
}

/// Encoding backend for the complete I4R rotation and quantization pipeline.
/// This selection is independent of HT trellis search and verification.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum I4rBackend {
    #[default]
    Cpu,
    Hip,
    Cuda,
}

impl I4rBackend {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Hip => "HIP",
            Self::Cuda => "CUDA",
        }
    }

    pub const fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Validate I4R device availability without allocating an HT workspace.
    pub fn preflight(self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Hip => {
                #[cfg(feature = "hip")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { hgn_hip_i4r_preflight() };
                    ensure!(
                        result == 0,
                        "HIP I4R preflight failed: {}",
                        hip_last_error()
                    );
                    Ok(())
                }
                #[cfg(not(feature = "hip"))]
                bail!("HIP I4R backend requires a build with --features hip");
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { hgn_cuda_i4r_preflight() };
                    ensure!(
                        result == 0,
                        "CUDA I4R preflight failed: {}",
                        cuda_last_error()
                    );
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                bail!("CUDA I4R backend requires a build with --features cuda");
            }
        }
    }
}

/// Encoding backend for Q4C-P group quantization, independent of HT and I4R.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Q4cpBackend {
    #[default]
    Cpu,
    Hip,
    Cuda,
}

impl Q4cpBackend {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Hip => "HIP",
            Self::Cuda => "CUDA",
        }
    }

    pub const fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Validate Q4C-P device availability without allocating other workspaces.
    pub fn preflight(self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Hip => {
                #[cfg(feature = "hip")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { hgn_hip_q4cp_preflight() };
                    ensure!(result == 0, "HIP Q4C-P preflight failed: {}", hip_last_error());
                    Ok(())
                }
                #[cfg(not(feature = "hip"))]
                bail!("HIP Q4C-P backend requires a build with --features hip");
            }
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // SAFETY: preflight takes no pointers and synchronizes native context access.
                    let result = unsafe { hgn_cuda_q4cp_preflight() };
                    ensure!(
                        result == 0,
                        "CUDA Q4C-P preflight failed: {}",
                        cuda_last_error()
                    );
                    Ok(())
                }
                #[cfg(not(feature = "cuda"))]
                bail!("CUDA Q4C-P backend requires a build with --features cuda");
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantMeta<'a> {
    /// Input signs for the current matrix/expert; exactly `cols` values.
    pub su: &'a [f32],
    /// Output signs/scales for the current band; exactly `rows` values.
    pub sv: &'a [f32],
    pub codebook: &'a [f32; 16],
    /// One positive f32 scale shared by the complete n-gram table.
    pub fp8_scale: f32,
    pub threads: usize,
    pub ht_beam: usize,
    pub ht_backend: HtBackend,
    pub i4r_backend: I4rBackend,
    pub q4cp_backend: Q4cpBackend,
}

impl Default for QuantMeta<'_> {
    fn default() -> Self {
        Self {
            su: &[],
            sv: &[],
            codebook: &NF4_CODEBOOK,
            fp8_scale: 1.0,
            threads: 1,
            ht_beam: 128,
            ht_backend: HtBackend::Cpu,
            i4r_backend: I4rBackend::Cpu,
            q4cp_backend: Q4cpBackend::Cpu,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedBlock {
    /// Code plane, or complete interleaved rows for non-planar formats.
    pub codes: Vec<u8>,
    /// Q4Cp: padded scale rows. I4r: contiguous half scales. Otherwise empty.
    pub scales: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayloadLayout {
    pub len: u64,
    pub codes_offset: u64,
    /// Zero for formats without a separate scale plane.
    pub scales_offset: u64,
    /// Zero except for Q4Cp and I4r; fp8g has one trailing global scale.
    pub scale_row_stride: u64,
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).context("payload size overflow")
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("payload size overflow")
}

fn pad(value: u64, alignment: u64) -> Result<u64> {
    Ok(add(value, alignment - 1)? & !(alignment - 1))
}

/// Full tensor geometry, including the Q4Cp codebook and fp8g global scale.
/// The architecture planner additionally validates the rank/expert shapes;
/// this function accepts matrix bands for independent codec fixtures.
pub fn payload_layout(encoding: Encoding, rows: usize, cols: usize) -> Result<PayloadLayout> {
    ensure!(rows > 0 && cols > 0, "matrix dimensions must be positive");
    let n = rows as u64;
    let k = cols as u64;
    let values = mul(n, k)?;
    let mut layout = PayloadLayout {
        len: 0,
        codes_offset: 0,
        scales_offset: 0,
        scale_row_stride: 0,
    };
    layout.len = match encoding {
        Encoding::Bf16 | Encoding::F16 => mul(values, 2)?,
        Encoding::F32 => mul(values, 4)?,
        Encoding::I64 => mul(values, 8)?,
        Encoding::Q4Cp => {
            ensure!(cols % 32 == 0, "Q4Cp columns must be divisible by 32");
            layout.codes_offset = 64;
            layout.scales_offset = add(64, pad(values / 2, 64)?)?;
            layout.scale_row_stride = pad(k / 16, 16)?;
            add(layout.scales_offset, mul(n, layout.scale_row_stride)?)?
        }
        Encoding::Q8G64 => {
            ensure!(cols % 64 == 0, "Q8G64 columns must be divisible by 64");
            mul(n, add(k, k / 16)?)?
        }
        Encoding::Q6G64 => {
            ensure!(cols % 64 == 0, "Q6G64 columns must be divisible by 64");
            mul(n, pad(mul(k / 16, 13)?, 16)?)?
        }
        Encoding::Fp8G128 => {
            ensure!(
                cols == 160,
                "fp8g is only defined for n-gram rows of 160 values"
            );
            layout.scales_offset = pad(values, 64)?;
            add(layout.scales_offset, 4)?
        }
        Encoding::Ht | Encoding::Ht3 | Encoding::I4r => {
            ensure!(
                rows % 128 == 0 && cols % 128 == 0,
                "rotated matrices require rows and columns divisible by 128"
            );
            if encoding == Encoding::I4r {
                layout.scales_offset = values / 2;
                layout.scale_row_stride = k / 64;
                add(values / 2, values / 64)?
            } else {
                mul(values, encoding.ht_bits().expect("HT width") as u64)? / 8
            }
        }
    };
    Ok(layout)
}

fn element_count(input: &[f32], rows: usize, cols: usize) -> Result<usize> {
    let count = rows
        .checked_mul(cols)
        .context("matrix element count overflow")?;
    ensure!(
        rows > 0 && cols > 0 && input.len() == count,
        "input length does not match matrix shape"
    );
    ensure!(
        input.iter().all(|x| x.is_finite()),
        "input contains non-finite values"
    );
    Ok(count)
}

fn checked_half(value: f32) -> Result<f16> {
    let result = f16::from_f32(value);
    ensure!(
        result.is_finite(),
        "value {value} cannot be represented as a finite f16"
    );
    Ok(result)
}

fn validate_codebook(codebook: &[f32; 16]) -> Result<()> {
    ensure!(
        codebook.iter().all(|x| x.is_finite()),
        "codebook contains non-finite values"
    );
    ensure!(
        codebook.windows(2).all(|p| p[0] < p[1]),
        "encoder needs a strictly increasing codebook"
    );
    Ok(())
}

/// FNV-1a over UTF-8 key bytes, XOR seed, then SplitMix64 per element.
/// Generate the vector once and slice it: restarting this function for a later
/// band would intentionally restart the vector, not advance its row index.
pub fn generate_signs(seed: u64, key: &str, len: usize) -> Vec<f32> {
    let mut state = 0xcbf29ce484222325u64;
    for byte in key.bytes() {
        state = (state ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    state ^= seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = state;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            if (x ^ (x >> 31)) >> 63 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Signed row RMS rounded directly to binary16. Nonzero rows below binary16's
/// range use its smallest subnormal, rather than being silently made dead rows.
pub fn ht_row_scales(input: &[f32], rows: usize, cols: usize, signs: &[f32]) -> Result<Vec<f32>> {
    element_count(input, rows, cols)?;
    ensure!(
        signs.len() == rows && signs.iter().all(|x| x.abs() == 1.0),
        "invalid output signs"
    );
    input
        .chunks_exact(cols)
        .zip(signs)
        .map(|(row, sign)| {
            let sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            if sum == 0.0 {
                return Ok(0.0);
            }
            let mut scale = f16::from_f64((sum / cols as f64).sqrt());
            ensure!(scale.is_finite(), "HT row RMS exceeds finite f16 range");
            if scale.to_bits() == 0 {
                scale = f16::from_bits(1);
            }
            Ok(scale.to_f32() * sign)
        })
        .collect()
}

fn validate_rotation(rows: usize, cols: usize, meta: &QuantMeta<'_>, i4r: bool) -> Result<()> {
    ensure!(
        rows % 128 == 0 && cols % 128 == 0,
        "rotation requires complete 128-by-128 blocks"
    );
    ensure!(
        meta.su.len() == cols && meta.sv.len() == rows,
        "rotation vector dimensions do not match band"
    );
    ensure!(
        meta.su.iter().all(|x| x.abs() == 1.0),
        "input rotation vector must contain only +/-1"
    );
    ensure!(
        meta.sv
            .iter()
            .all(|x| x.is_finite() && f16::from_f32(*x).to_f32() == *x),
        "output rotation vector must contain finite, exactly representable f16 values"
    );
    if i4r {
        ensure!(
            meta.sv.iter().all(|x| x.abs() == 1.0),
            "I4R output vector must contain only +/-1"
        );
    }
    Ok(())
}

fn validate_decode_rotation(rows: usize, cols: usize, meta: &QuantMeta<'_>) -> Result<()> {
    ensure!(
        rows % 128 == 0 && cols % 128 == 0 && meta.su.len() == cols && meta.sv.len() == rows,
        "rotation vector dimensions do not match band"
    );
    ensure!(
        meta.su
            .iter()
            .chain(meta.sv)
            .all(|&value| { value.is_finite() && f16::from_f32(value).to_f32() == value }),
        "rotation vectors must contain finite, exactly representable f16 values"
    );
    Ok(())
}

// Two unnormalized Sylvester transforms followed by one exact /128. f64
// scratch avoids double-rounding sqrt(128), and also reproduces the spec's
// exact-value f32 rotation golden. Scratch is always one 128-by-128 block.
fn hadamard_2d(block: &mut [f64]) {
    let mut half = 1;
    while half < 128 {
        for row in block.chunks_exact_mut(128) {
            for group in row.chunks_exact_mut(2 * half) {
                for i in 0..half {
                    let a = group[i];
                    let b = group[i + half];
                    group[i] = a + b;
                    group[i + half] = a - b;
                }
            }
        }
        half *= 2;
    }
    half = 1;
    while half < 128 {
        for row_base in (0..128).step_by(2 * half) {
            for i in 0..half {
                let a_base = (row_base + i) * 128;
                let b_base = (row_base + i + half) * 128;
                for col in 0..128 {
                    let a = block[a_base + col];
                    let b = block[b_base + col];
                    block[a_base + col] = a + b;
                    block[b_base + col] = a - b;
                }
            }
        }
        half *= 2;
    }
}

#[derive(Clone, Copy)]
enum RotationOrder {
    Rows,
    Tiles {
        row_tiles: usize,
        row_tile_offset: usize,
    },
}

fn tile_value_index(row: usize, col: usize, cols: usize, row_tiles: usize, offset: usize) -> usize {
    let tile = ((row / 128) * (cols / 16) + col / 16) * row_tiles + (row % 128) / 16 + offset;
    let value = (col % 16 / 8) * 128 + (row % 16) * 8 + col % 8;
    tile * 256 + value
}

fn rotate_into(
    input: &[f32],
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
    order: RotationOrder,
    out: &mut [f32],
) -> Result<()> {
    let mut block = vec![0.0f64; 128 * 128];
    for row_base in (0..rows).step_by(128) {
        for col_base in (0..cols).step_by(128) {
            for r in 0..128 {
                let sv = meta.sv[row_base + r] as f64;
                for c in 0..128 {
                    let value = input[(row_base + r) * cols + col_base + c];
                    if sv == 0.0 {
                        ensure!(
                            value == 0.0,
                            "a zero HT row scale cannot encode a nonzero source row"
                        );
                        block[r * 128 + c] = 0.0;
                    } else {
                        block[r * 128 + c] = (value as f64 / sv) * meta.su[col_base + c] as f64;
                    }
                }
            }
            hadamard_2d(&mut block);
            for r in 0..128 {
                for c in 0..128 {
                    let value = (block[r * 128 + c] / 128.0) as f32;
                    ensure!(value.is_finite(), "rotation overflows f32");
                    let index = match order {
                        RotationOrder::Rows => (row_base + r) * cols + col_base + c,
                        RotationOrder::Tiles {
                            row_tiles,
                            row_tile_offset,
                        } => {
                            // Prevent overflow of the native f32 path costs.
                            ensure!(
                                value.abs() <= 1.0e17,
                                "normalized HT value exceeds trellis cost range"
                            );
                            tile_value_index(
                                row_base + r,
                                col_base + c,
                                cols,
                                row_tiles,
                                row_tile_offset,
                            )
                        }
                    };
                    out[index] = value;
                }
            }
        }
    }
    Ok(())
}

fn unrotate_in_place(values: &mut [f32], rows: usize, cols: usize, meta: &QuantMeta<'_>) {
    let mut block = vec![0.0f64; 128 * 128];
    for row_base in (0..rows).step_by(128) {
        for col_base in (0..cols).step_by(128) {
            for r in 0..128 {
                for c in 0..128 {
                    block[r * 128 + c] = values[(row_base + r) * cols + col_base + c] as f64;
                }
            }
            hadamard_2d(&mut block);
            for r in 0..128 {
                for c in 0..128 {
                    values[(row_base + r) * cols + col_base + c] = (block[r * 128 + c] / 128.0
                        * meta.sv[row_base + r] as f64
                        * meta.su[col_base + c] as f64)
                        as f32;
                }
            }
        }
    }
}

// Callers clamp to integral code bounds inside [-8, 255]. Adding this exact
// bias puts every value in the f32 binade with unit spacing, so the addition
// performs RNE; conversion and bias removal are then exact integer operations.
// Clamping before rounding is equivalent here because both bounds are integers.
#[inline]
fn round_small_code(value: f32, lower: f32, upper: f32) -> i32 {
    (value.clamp(lower, upper) + 12_582_912.0) as i32 - 12_582_912
}

fn q4_midpoints(cb: &[f32; 16]) -> [f32; 15] {
    let mut mids = [0.0f32; 15];
    for (i, midpoint) in mids.iter_mut().enumerate() {
        *midpoint = (cb[i] as f64 * 0.5 + cb[i + 1] as f64 * 0.5) as f32;
    }
    mids
}

/// GPU Q4C-P (CUDA or HIP). There is no CPU fallback; a scale that is not a
/// finite half reruns the CPU reference only to report its exact error.
// Feature-free builds have no GPU arm: everything after the match is dead.
#[allow(unused_variables, unreachable_code)]
fn encode_q4_gpu(
    backend: Q4cpBackend,
    input: &[f32],
    rows: usize,
    cols: usize,
    cb: &[f32; 16],
) -> Result<EncodedBlock> {
    type Native = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut u8, *mut u16, usize) -> c_int;
    let (native, last_error): (Native, fn() -> String) = match backend {
        #[cfg(feature = "cuda")]
        Q4cpBackend::Cuda => (hgn_cuda_encode_q4cp, cuda_last_error),
        #[cfg(feature = "hip")]
        Q4cpBackend::Hip => (hgn_hip_encode_q4cp, hip_last_error),
        Q4cpBackend::Cpu => unreachable!("CPU Q4C-P does not use a native backend"),
        #[allow(unreachable_patterns)]
        _ => bail!(
            "{} Q4C-P backend requires a build with --features {}",
            backend.label(),
            backend.label().to_ascii_lowercase()
        ),
    };
    validate_codebook(cb)?;
    let stride = pad((cols / 16) as u64, 16)? as usize;
    let groups = input.len() / 32;
    let mids = q4_midpoints(cb);
    let mut codes = vec![0u8; input.len() / 2];
    let mut halves = vec![0u16; groups];
    // SAFETY: input holds groups*32 finite values (checked by the caller);
    // exclusive outputs hold exactly groups*16 bytes and groups half words.
    // The native backend synchronizes before return and retains no host pointers.
    let result = unsafe {
        native(
            input.as_ptr(),
            cb.as_ptr(),
            mids.as_ptr(),
            codes.as_mut_ptr(),
            halves.as_mut_ptr(),
            groups,
        )
    };
    if result == GPU_Q4CP_SCALE_OVERFLOW {
        // Reproduce the CPU reference's exact first-group error message.
        encode_q4(input, rows, cols, cb)?;
        bail!(
            "{} Q4C-P reported a non-finite f16 scale that the CPU reference accepted",
            backend.label()
        );
    }
    ensure!(
        result == 0,
        "{} Q4C-P encoding failed: {}",
        backend.label(),
        last_error()
    );
    let per_row = cols / 32;
    let mut scales = vec![0u8; rows * stride];
    for (out, row_halves) in scales
        .chunks_exact_mut(stride)
        .zip(halves.chunks_exact(per_row))
    {
        for (g, half) in row_halves.iter().enumerate() {
            out[2 * g..2 * g + 2].copy_from_slice(&half.to_le_bytes());
        }
    }
    Ok(EncodedBlock { codes, scales })
}

const GPU_Q4CP_SCALE_OVERFLOW: c_int = -6;

fn encode_q4(input: &[f32], rows: usize, cols: usize, cb: &[f32; 16]) -> Result<EncodedBlock> {
    validate_codebook(cb)?;
    let stride = pad((cols / 16) as u64, 16)? as usize;
    let mut codes = vec![0; input.len() / 2];
    let mut scales = vec![0; rows * stride];
    let mids = q4_midpoints(cb);
    let cmax = cb[0].abs().max(cb[15].abs());
    for (group_index, group) in input.chunks_exact(32).enumerate() {
        let absmax = group.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        let mut best_error = f64::INFINITY;
        let mut best_codes = [0u8; 32];
        let mut best_scale = f16::ZERO;
        for multiplier in [1.0f32, 0.95, 0.9, 0.85, 0.8] {
            let scale = checked_half(absmax / cmax * multiplier)?;
            let sf = scale.to_f32();
            let divisor = if sf == 0.0 { 1.0 } else { sf };
            let mut candidate = [0u8; 32];
            let mut error = 0.0f64;
            for (i, &value) in group.iter().enumerate() {
                let normalized = value / divisor;
                let code = mids.partition_point(|m| *m < normalized);
                candidate[i] = code as u8;
                let delta = cb[code] as f64 * sf as f64 - value as f64;
                error += delta * delta;
            }
            if error < best_error {
                best_error = error;
                best_codes = candidate;
                best_scale = scale;
            }
        }
        for i in 0..16 {
            codes[group_index * 16 + i] = best_codes[2 * i] | (best_codes[2 * i + 1] << 4);
        }
        let row = group_index / (cols / 32);
        let g = group_index % (cols / 32);
        scales[row * stride + 2 * g..row * stride + 2 * g + 2]
            .copy_from_slice(&best_scale.to_le_bytes());
    }
    Ok(EncodedBlock { codes, scales })
}

fn encode_affine(input: &[f32], rows: usize, cols: usize, bits: u32) -> Result<EncodedBlock> {
    let stride = if bits == 8 {
        cols + cols / 16
    } else {
        pad((cols / 16 * 13) as u64, 16)? as usize
    };
    let mut codes = vec![0; rows * stride];
    let max_code = ((1u32 << bits) - 1) as f32;
    for (row, input_row) in codes.chunks_exact_mut(stride).zip(input.chunks_exact(cols)) {
        for (g, group) in input_row.chunks_exact(64).enumerate() {
            let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = checked_half(((hi as f64 - lo as f64) / max_code as f64) as f32)?;
            let bias = checked_half(lo)?;
            let sf = scale.to_f32();
            let bf = bias.to_f32();
            for (i, value) in group.iter().enumerate() {
                let code = if sf == 0.0 {
                    0
                } else {
                    round_small_code((value - bf) / sf, 0.0, max_code) as u8
                };
                let k = 64 * g + i;
                if bits == 8 {
                    row[k] = code;
                } else {
                    row[k / 2] |= (code & 15) << (4 * (k % 2));
                    row[cols / 2 + k / 4] |= (code >> 4) << (2 * (k % 4));
                }
            }
            let record = if bits == 8 { cols } else { 3 * cols / 4 } + 4 * g;
            row[record..record + 2].copy_from_slice(&scale.to_le_bytes());
            row[record + 2..record + 4].copy_from_slice(&bias.to_le_bytes());
        }
    }
    Ok(EncodedBlock {
        codes,
        scales: Vec::new(),
    })
}

fn encode_i4r_rotated(input: &[f32]) -> Result<EncodedBlock> {
    let mut codes = vec![0; input.len() / 2];
    let mut scales = Vec::with_capacity(input.len() / 64);
    for (g, group) in input.chunks_exact(128).enumerate() {
        let mut extreme = group[0];
        for &v in &group[1..] {
            if v.abs() > extreme.abs() {
                extreme = v;
            }
        }
        let mut best_error = f64::INFINITY;
        let mut best_codes = [0u8; 128];
        let mut best_scale = f16::ZERO;
        for multiplier in [1.0f32, 0.97, 0.94, 0.91, 0.88, 0.85] {
            let scale = checked_half(-extreme / 8.0 * multiplier)?;
            let sf = scale.to_f32();
            let divisor = if sf == 0.0 { 1.0 } else { sf };
            let mut candidate = [0u8; 128];
            let mut error = 0.0f64;
            for (i, &value) in group.iter().enumerate() {
                let code = round_small_code(value / divisor, -8.0, 7.0);
                candidate[i] = (code + 8) as u8;
                let delta = code as f64 * sf as f64 - value as f64;
                error += delta * delta;
            }
            if error < best_error {
                best_error = error;
                best_codes = candidate;
                best_scale = scale;
            }
        }
        for i in 0..64 {
            codes[g * 64 + i] = best_codes[2 * i] | (best_codes[2 * i + 1] << 4);
        }
        scales.extend_from_slice(&best_scale.to_le_bytes());
    }
    Ok(EncodedBlock { codes, scales })
}

/// Decode the HGN fp8 encoding, including finite +/-480 (unlike E4M3FN).
pub fn fp8_decode(code: u8) -> f32 {
    let exponent = u32::from((code >> 3) & 15);
    let mantissa = u32::from(code & 7);
    let magnitude_bits = if exponent == 0 {
        (mantissa as f32 / 512.0).to_bits()
    } else {
        ((exponent + 120) << 23) | (mantissa << 20)
    };
    f32::from_bits(magnitude_bits | (u32::from(code & 0x80) << 24))
}

// RNE to three mantissa bits. Encoding deliberately caps magnitude at 448,
// reserving the finite 480 codes just as published fp8g writers do.
fn fp8_encode_normalized(value: f32) -> u8 {
    let sign = ((value.to_bits() >> 24) & 0x80) as u8;
    let magnitude = value.abs().min(448.0);
    let code = if magnitude < 0.015625 {
        (magnitude * 512.0).round_ties_even() as u8
    } else {
        let bits = magnitude.to_bits();
        let rounded = bits + 0x7ffff + ((bits >> 20) & 1);
        ((rounded >> 20) - 960).min(126) as u8
    };
    sign | code
}

pub fn ht_codebook_value(state: u16) -> f32 {
    let hash = u32::from(state).wrapping_mul(0x83dcd12d);
    let sum: i32 = hash.to_le_bytes().into_iter().map(i32::from).sum();
    f16::from_f32((1774 * sum - 905216) as f32 / 262144.0).to_f32()
}

static HT_CODEBOOK: LazyLock<Vec<f32>> =
    LazyLock::new(|| (0..=u16::MAX).map(ht_codebook_value).collect());

fn ht_codebook() -> &'static [f32] {
    &HT_CODEBOOK
}

unsafe extern "C" {
    fn ht_encode_bits(
        tiles: *const f32,
        codebook: *const f32,
        codes: *mut u8,
        count: c_int,
        threads: c_int,
        beam: c_int,
        bits: c_int,
    ) -> c_int;
    #[cfg(test)]
    fn ht_search_fixed_bits(
        values: *const f32,
        codebook: *const f32,
        count: c_int,
        boundary: c_int,
        codes: *mut u8,
        bits: c_int,
    ) -> f32;
}

#[cfg(feature = "hip")]
unsafe extern "C" {
    fn ht_encode_bits_hip(
        tiles: *const f32,
        codebook: *const f32,
        codes: *mut u8,
        count: c_int,
        beam: c_int,
        bits: c_int,
    ) -> c_int;
    fn ht_hip_last_error() -> *const std::ffi::c_char;
    fn ht_hip_preflight() -> c_int;
    fn hgn_hip_i4r_preflight() -> c_int;
    fn hgn_hip_q4cp_preflight() -> c_int;
    fn hgn_hip_encode_q4cp(
        src: *const f32,
        codebook: *const f32,
        midpoints: *const f32,
        codes: *mut u8,
        scales: *mut u16,
        groups: usize,
    ) -> c_int;
    fn hgn_hip_encode_ht(
        sources: *const *const f32,
        su: *const f32,
        svs: *const *const f32,
        parts: c_int,
        codebook: *const f32,
        symbols: *mut u8,
        bands: c_int,
        cols: c_int,
        beam: c_int,
        bits: c_int,
    ) -> c_int;
    fn hgn_hip_encode_i4r(
        src: *const f32,
        su: *const f32,
        sv: *const f32,
        codes: *mut u8,
        scales: *mut u8,
        bands: c_int,
        cols: c_int,
    ) -> c_int;
}

#[cfg(feature = "hip")]
fn hip_last_error() -> String {
    // SAFETY: the ABI returns a thread-local NUL-terminated error string,
    // valid until this thread's next native HIP call. Copy it immediately.
    let error = unsafe { ht_hip_last_error() };
    if error.is_null() {
        return "native HIP encoder returned no error detail".to_owned();
    }
    unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn ht_encode_bits_cuda(
        tiles: *const f32,
        codebook: *const f32,
        codes: *mut u8,
        count: c_int,
        beam: c_int,
        bits: c_int,
    ) -> c_int;
    fn ht_cuda_last_error() -> *const std::ffi::c_char;
    fn ht_cuda_preflight() -> c_int;
    fn hgn_cuda_encode_ht(
        sources: *const *const f32,
        su: *const f32,
        svs: *const *const f32,
        parts: c_int,
        codebook: *const f32,
        symbols: *mut u8,
        bands: c_int,
        cols: c_int,
        beam: c_int,
        bits: c_int,
    ) -> c_int;
    fn hgn_cuda_q4cp_preflight() -> c_int;
    fn hgn_cuda_encode_q4cp(
        src: *const f32,
        codebook: *const f32,
        midpoints: *const f32,
        codes: *mut u8,
        scales: *mut u16,
        groups: usize,
    ) -> c_int;
    fn hgn_cuda_i4r_preflight() -> c_int;
    fn hgn_cuda_encode_i4r(
        src: *const f32,
        su: *const f32,
        sv: *const f32,
        codes: *mut u8,
        scales: *mut u8,
        bands: c_int,
        cols: c_int,
    ) -> c_int;
}


#[cfg(feature = "cuda")]
fn cuda_last_error() -> String {
    // SAFETY: the ABI returns a thread-local NUL-terminated error string,
    // valid until this thread's next native CUDA call. Copy it immediately.
    let error = unsafe { ht_cuda_last_error() };
    if error.is_null() {
        return "native CUDA encoder returned no error detail".to_owned();
    }
    unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned()
}

fn native_i4r_shape(rows: usize, cols: usize, backend: I4rBackend) -> Result<(c_int, c_int)> {
    ensure!(
        rows > 0 && cols > 0 && rows % 128 == 0 && cols % 128 == 0,
        "I4R encoding requires complete 128-by-128 blocks"
    );
    let count = rows
        .checked_mul(cols)
        .with_context(|| format!("{} I4R matrix element count overflow", backend.label()))?;
    ensure!(
        count <= isize::MAX as usize / std::mem::size_of::<f32>(),
        "{} I4R block exceeds addressable buffer size",
        backend.label()
    );
    let bands = c_int::try_from(rows / 128).with_context(|| {
        format!(
            "{} I4R band count exceeds signed-int range",
            backend.label()
        )
    })?;
    let cols = c_int::try_from(cols).with_context(|| {
        format!(
            "{} I4R column count exceeds signed-int range",
            backend.label()
        )
    })?;
    Ok((bands, cols))
}

fn encode_i4r_hip(
    input: &[f32],
    meta: &QuantMeta<'_>,
    bands: c_int,
    cols: c_int,
) -> Result<EncodedBlock> {
    #[cfg(feature = "hip")]
    {
        let mut codes = vec![0; input.len() / 2];
        let mut scales = vec![0; input.len() / 64];
        // SAFETY: shape, finite source values, and complete +/-1 sign planes
        // were checked before this call. Exclusive output planes have exactly
        // values/2 and values/64 bytes. Native HIP synchronizes before return
        // and retains no host pointers; failed output is never exposed.
        let result = unsafe {
            hgn_hip_encode_i4r(
                input.as_ptr(),
                meta.su.as_ptr(),
                meta.sv.as_ptr(),
                codes.as_mut_ptr(),
                scales.as_mut_ptr(),
                bands,
                cols,
            )
        };
        ensure!(result == 0, "HIP I4R encoding failed: {}", hip_last_error());
        Ok(EncodedBlock { codes, scales })
    }
    #[cfg(not(feature = "hip"))]
    {
        let _ = (input, meta, bands, cols);
        bail!("HIP I4R backend requires a build with --features hip");
    }
}

fn encode_i4r_cuda(
    input: &[f32],
    meta: &QuantMeta<'_>,
    bands: c_int,
    cols: c_int,
) -> Result<EncodedBlock> {
    #[cfg(feature = "cuda")]
    {
        let mut codes = vec![0; input.len() / 2];
        let mut scales = vec![0; input.len() / 64];
        // SAFETY: shape, finite source values, and complete +/-1 sign planes
        // were checked before this call. Exclusive output planes have exactly
        // values/2 and values/64 bytes. Native CUDA synchronizes before return
        // and retains no host pointers; failed output is never exposed.
        let result = unsafe {
            hgn_cuda_encode_i4r(
                input.as_ptr(),
                meta.su.as_ptr(),
                meta.sv.as_ptr(),
                codes.as_mut_ptr(),
                scales.as_mut_ptr(),
                bands,
                cols,
            )
        };
        ensure!(
            result == 0,
            "CUDA I4R encoding failed: {}",
            cuda_last_error()
        );
        Ok(EncodedBlock { codes, scales })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (input, meta, bands, cols);
        bail!("CUDA I4R backend requires a build with --features cuda");
    }
}

fn validate_ht_backend(meta: &QuantMeta<'_>) -> Result<()> {
    match meta.ht_backend {
        HtBackend::Cpu => ensure!(
            meta.ht_beam <= 4096,
            "HT beam must be in 0..=4096 (0 is full-state search)"
        ),
        HtBackend::Hip => {
            ensure!(
                (1..=256).contains(&meta.ht_beam),
                "HIP HT beam must be in 1..=256; beam 0/full-state and larger beams require CPU"
            );
            ensure!(
                cfg!(feature = "hip"),
                "HIP HT backend requires a build with --features hip"
            );
        }
        HtBackend::Cuda => {
            ensure!(
                (1..=256).contains(&meta.ht_beam),
                "CUDA HT beam must be in 1..=256; beam 0/full-state and larger beams require CPU"
            );
            ensure!(
                cfg!(feature = "cuda"),
                "CUDA HT backend requires a build with --features cuda"
            );
        }
    }
    Ok(())
}

/// GPU HT (CUDA or HIP) with GPU rotation. `parts` are (source, row scales)
/// pairs sharing `su`: one part of complete bands, or a gate band and an up
/// band. Returns None for a CPU backend, or when the device rejects the
/// rotation; the caller then runs the CPU rotation, which reports the exact
/// error.
// Feature-free builds have no GPU arm: everything after the match is dead.
#[allow(unused_variables, unreachable_code)]
fn encode_ht_gpu_rotated(
    parts: &[(&[f32], &[f32])],
    su: &[f32],
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
    bits: usize,
) -> Result<Option<Vec<u8>>> {
    type Native = unsafe extern "C" fn(
        *const *const f32,
        *const f32,
        *const *const f32,
        c_int,
        *const f32,
        *mut u8,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int;
    let (native, last_error): (Native, fn() -> String) = match meta.ht_backend {
        #[cfg(feature = "cuda")]
        HtBackend::Cuda => (hgn_cuda_encode_ht, cuda_last_error),
        #[cfg(feature = "hip")]
        HtBackend::Hip => (hgn_hip_encode_ht, hip_last_error),
        _ => return Ok(None),
    };
    let count = parts.len() * rows * cols;
    native_ht_tile_count(count)?;
    let sources: Vec<*const f32> = parts.iter().map(|(values, _)| values.as_ptr()).collect();
    let scales: Vec<*const f32> = parts.iter().map(|(_, sv)| sv.as_ptr()).collect();
    let mut symbols = vec![0u8; count];
    // SAFETY: each part holds rows*cols finite values and rows scales, su holds
    // cols signs (validated by the caller); symbols holds every tile's 256
    // symbols. The native backend synchronizes before return and keeps no pointers.
    let result = unsafe {
        native(
            sources.as_ptr(),
            su.as_ptr(),
            scales.as_ptr(),
            c_int::try_from(parts.len())?,
            ht_codebook().as_ptr(),
            symbols.as_mut_ptr(),
            c_int::try_from(rows / 128)?,
            c_int::try_from(cols)?,
            meta.ht_beam as c_int,
            bits as c_int,
        )
    };
    if result == GPU_HT_ROTATION_REJECTED {
        return Ok(None);
    }
    ensure!(
        result == 0,
        "{} HT encoding failed: {}",
        meta.ht_backend.label(),
        last_error()
    );
    Ok(Some(pack_ht_symbols(&symbols, bits)))
}

const GPU_HT_ROTATION_REJECTED: c_int = -6;

fn pack_ht_symbols(symbols: &[u8], bits: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(symbols.len() / 8 * bits);
    let mut word = 0u64;
    let mut filled = 0;
    for &symbol in symbols {
        word = (word << bits) | u64::from(symbol);
        filled += bits;
        if filled >= 32 {
            filled -= 32;
            bytes.extend_from_slice(&((word >> filled) as u32).to_le_bytes());
            word &= (1u64 << filled) - 1;
        }
    }
    debug_assert_eq!(filled, 0);
    bytes
}

// The unchanged C++ helper computes i*256 in signed int before doing pointer
// arithmetic. Checking only the number of tiles fits in int is insufficient.
// Keep the entire element index range in int as well, including paired bands.
fn native_ht_tile_count(value_count: usize) -> Result<c_int> {
    ensure!(
        value_count > 0 && value_count % 256 == 0,
        "HT requires complete 16-by-16 tiles"
    );
    ensure!(
        value_count <= c_int::MAX as usize,
        "HT block exceeds the native signed-int element offset range"
    );
    Ok((value_count / 256) as c_int)
}

fn ht_band_value_count(rows: usize, cols: usize, paired: bool) -> Result<usize> {
    // Tiles are independent and stored band-major, so several complete
    // bands in one call equal the concatenation of single-band calls.
    ensure!(
        rows > 0 && rows % 128 == 0 && cols > 0 && cols % 128 == 0 && (!paired || rows == 128),
        "HT encoding requires complete 128-row bands (one band when paired) and columns divisible by 128"
    );
    let count = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(if paired { 2 } else { 1 }))
        .context("HT band element count overflow")?;
    native_ht_tile_count(count)?;
    let bytes = count
        .checked_mul(std::mem::size_of::<f32>())
        .context("HT band byte count overflow")?;
    ensure!(
        bytes <= isize::MAX as usize,
        "HT band exceeds addressable buffer size"
    );
    Ok(count)
}

fn encode_ht_tiles(tiles: &[f32], meta: &QuantMeta<'_>, bits: usize) -> Result<Vec<u8>> {
    let count = native_ht_tile_count(tiles.len())?;
    ensure!(bits == 3 || bits == 4, "unsupported HT code width");
    ensure!(
        tiles.iter().all(|x| x.is_finite() && x.abs() <= 1.0e17),
        "HT values exceed finite cost range"
    );
    validate_ht_backend(meta)?;
    ensure!(meta.threads > 0, "HT thread count must be positive");
    let threads = c_int::try_from(meta.threads).context("HT thread count exceeds C ABI range")?;
    let mut symbols = vec![0u8; tiles.len()];
    match meta.ht_backend {
        HtBackend::Cpu => {
            // SAFETY: inputs contain count*256 floats and 65536 codebook
            // values; exclusive output contains count*256 bytes. Signed-int
            // tile products fit and all buffers outlive the native workers.
            let result = unsafe {
                ht_encode_bits(
                    tiles.as_ptr(),
                    ht_codebook().as_ptr(),
                    symbols.as_mut_ptr(),
                    count,
                    threads,
                    meta.ht_beam as c_int,
                    bits as c_int,
                )
            };
            ensure!(result == 0, "native HT trellis encoder rejected parameters");
        }
        HtBackend::Hip => {
            #[cfg(feature = "hip")]
            {
                // SAFETY: the same validated host buffer sizes apply. Native
                // HIP copies and synchronizes before returning; it does not
                // retain the caller's tile/output pointers.
                let result = unsafe {
                    ht_encode_bits_hip(
                        tiles.as_ptr(),
                        ht_codebook().as_ptr(),
                        symbols.as_mut_ptr(),
                        count,
                        meta.ht_beam as c_int,
                        bits as c_int,
                    )
                };
                ensure!(result == 0, "HIP HT encoding failed: {}", hip_last_error());
            }
            #[cfg(not(feature = "hip"))]
            bail!("HIP HT backend requires a build with --features hip");
        }
        HtBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                // SAFETY: the same validated host buffer sizes apply. Native
                // CUDA copies and synchronizes before returning; it does not
                // retain the caller's tile/output pointers.
                let result = unsafe {
                    ht_encode_bits_cuda(
                        tiles.as_ptr(),
                        ht_codebook().as_ptr(),
                        symbols.as_mut_ptr(),
                        count,
                        meta.ht_beam as c_int,
                        bits as c_int,
                    )
                };
                ensure!(
                    result == 0,
                    "CUDA HT encoding failed: {}",
                    cuda_last_error()
                );
            }
            #[cfg(not(feature = "cuda"))]
            bail!("CUDA HT backend requires a build with --features cuda");
        }
    }
    ensure!(
        symbols
            .iter()
            .all(|&symbol| usize::from(symbol) < 1 << bits),
        "native HT encoder returned an invalid symbol"
    );
    Ok(pack_ht_symbols(&symbols, bits))
}

/// Encode a bounded block from ORIGINAL (not already rotated) source values.
/// HT requires exactly 128 rows; I4R accepts positive multiples of 128.
/// The caller passes the complete row-indexed sv slice and must not cross an
/// expert boundary, where the input signs may change.
/// I64 must use `encode_i64` or an exact little-endian source byte copy.
pub fn encode_rows(
    encoding: Encoding,
    input: &[f32],
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<EncodedBlock> {
    if encoding.ht_bits().is_some() {
        ht_band_value_count(rows, cols, false)?;
        validate_ht_backend(meta)?;
    }
    let i4r_gpu_shape = if encoding == Encoding::I4r && meta.i4r_backend.is_gpu() {
        Some(native_i4r_shape(rows, cols, meta.i4r_backend)?)
    } else {
        None
    };
    let count = element_count(input, rows, cols)?;
    payload_layout(encoding, rows, cols)?;
    match encoding {
        Encoding::Bf16 | Encoding::F16 | Encoding::F32 => {
            let width = if encoding == Encoding::F32 { 4 } else { 2 };
            let mut codes =
                Vec::with_capacity(count.checked_mul(width).context("block size overflow")?);
            for &value in input {
                match encoding {
                    Encoding::Bf16 => {
                        let v = bf16::from_f32(value);
                        ensure!(v.is_finite(), "value exceeds finite bf16 range");
                        codes.extend_from_slice(&v.to_le_bytes());
                    }
                    Encoding::F16 => codes.extend_from_slice(&checked_half(value)?.to_le_bytes()),
                    Encoding::F32 => codes.extend_from_slice(&value.to_le_bytes()),
                    _ => unreachable!(),
                }
            }
            Ok(EncodedBlock {
                codes,
                scales: Vec::new(),
            })
        }
        Encoding::I64 => bail!("I64 cannot be encoded through f32; preserve source integer bytes"),
        Encoding::Q4Cp => match meta.q4cp_backend {
            Q4cpBackend::Cpu => encode_q4(input, rows, cols, meta.codebook),
            backend => encode_q4_gpu(backend, input, rows, cols, meta.codebook),
        },
        Encoding::Q8G64 => encode_affine(input, rows, cols, 8),
        Encoding::Q6G64 => encode_affine(input, rows, cols, 6),
        Encoding::Fp8G128 => {
            ensure!(
                meta.fp8_scale.is_finite() && meta.fp8_scale > 0.0,
                "fp8g needs a positive finite global scale"
            );
            Ok(EncodedBlock {
                codes: input
                    .iter()
                    .map(|&v| fp8_encode_normalized(v / meta.fp8_scale))
                    .collect(),
                scales: Vec::new(),
            })
        }
        Encoding::Ht | Encoding::Ht3 => {
            validate_rotation(rows, cols, meta, false)?;
            {
                let bits = encoding.ht_bits().unwrap();
                if let Some(codes) =
                    encode_ht_gpu_rotated(&[(input, meta.sv)], meta.su, rows, cols, meta, bits)?
                {
                    return Ok(EncodedBlock {
                        codes,
                        scales: Vec::new(),
                    });
                }
            }
            let mut rotated = vec![0.0; count];
            rotate_into(
                input,
                rows,
                cols,
                meta,
                RotationOrder::Tiles {
                    row_tiles: 8,
                    row_tile_offset: 0,
                },
                &mut rotated,
            )?;
            Ok(EncodedBlock {
                codes: encode_ht_tiles(&rotated, meta, encoding.ht_bits().unwrap())?,
                scales: Vec::new(),
            })
        }
        Encoding::I4r => {
            validate_rotation(rows, cols, meta, true)?;
            if let Some((bands, cols)) = i4r_gpu_shape {
                match meta.i4r_backend {
                    I4rBackend::Hip => encode_i4r_hip(input, meta, bands, cols),
                    I4rBackend::Cuda => encode_i4r_cuda(input, meta, bands, cols),
                    I4rBackend::Cpu => unreachable!("CPU I4R does not use a native shape"),
                }
            } else {
                let mut rotated = vec![0.0; count];
                rotate_into(input, rows, cols, meta, RotationOrder::Rows, &mut rotated)?;
                encode_i4r_rotated(&rotated)
            }
        }
    }
}

/// Exact integer path, avoiding any loss of the PLE hash constants in f32.
pub fn encode_i64(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().checked_mul(8).expect("integer payload size"));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// HT expert gate/up requires paired bands: gate rows [128p,128p+128),
/// up rows [640+128p,768+128p). Append the five returned pairs for one expert,
/// then concatenate experts. Within each column tile, 8 gate tiles precede
/// 8 up tiles; neither concatenating complete projections nor row interleaving
/// is the on-disk layout. Down experts use ordinary encode_rows bands.
pub fn encode_ht_gate_up_bands(
    encoding: Encoding,
    gate: &[f32],
    up: &[f32],
    cols: usize,
    gate_meta: &QuantMeta<'_>,
    up_meta: &QuantMeta<'_>,
) -> Result<Vec<u8>> {
    let bits = encoding.ht_bits().context("gate/up bands require HT")?;
    let paired_count = ht_band_value_count(128, cols, true)?;
    element_count(gate, 128, cols)?;
    element_count(up, 128, cols)?;
    validate_rotation(128, cols, gate_meta, false)?;
    validate_rotation(128, cols, up_meta, false)?;
    ensure!(
        gate_meta.su == up_meta.su,
        "gate and up must share the same input signs"
    );
    ensure!(
        gate_meta.threads == up_meta.threads
            && gate_meta.ht_beam == up_meta.ht_beam
            && gate_meta.ht_backend == up_meta.ht_backend,
        "paired HT bands must use the same encoder settings"
    );
    validate_ht_backend(gate_meta)?;
    {
        if let Some(codes) = encode_ht_gpu_rotated(
            &[(gate, gate_meta.sv), (up, up_meta.sv)],
            gate_meta.su,
            128,
            cols,
            gate_meta,
            bits,
        )? {
            return Ok(codes);
        }
    }
    let mut tiles = vec![0.0; paired_count];
    rotate_into(
        gate,
        128,
        cols,
        gate_meta,
        RotationOrder::Tiles {
            row_tiles: 16,
            row_tile_offset: 0,
        },
        &mut tiles,
    )?;
    rotate_into(
        up,
        128,
        cols,
        up_meta,
        RotationOrder::Tiles {
            row_tiles: 16,
            row_tile_offset: 8,
        },
        &mut tiles,
    )?;
    encode_ht_tiles(&tiles, gate_meta, bits)
}

fn read_half(bytes: &[u8]) -> f32 {
    f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

fn decode_ht_tile(bytes: &[u8], bits: usize, out: &mut [f32; 256]) {
    let words = bytes.len() / 4;
    let codebook = ht_codebook();
    for (i, value) in out.iter_mut().enumerate() {
        let end = bits * (i + 1) - 1;
        let w = end / 32;
        let bit = end % 32;
        let word = u32::from_le_bytes(bytes[4 * w..4 * w + 4].try_into().unwrap());
        let mut state = word >> (31 - bit);
        if bit < 15 {
            let previous = (w + words - 1) % words;
            let previous =
                u32::from_le_bytes(bytes[4 * previous..4 * previous + 4].try_into().unwrap());
            state |= previous << (bit + 1);
        }
        *value = codebook[(state & 0xffff) as usize];
    }
}

fn decode_ht_rotated(
    bytes: &[u8],
    bits: usize,
    rows: usize,
    cols: usize,
    row_tiles: usize,
    offset: usize,
) -> Vec<f32> {
    let mut values = vec![0.0; rows * cols];
    let mut tile = [0.0; 256];
    for band in 0..rows / 128 {
        for col_tile in 0..cols / 16 {
            for row_tile in 0..8 {
                let t = (band * (cols / 16) + col_tile) * row_tiles + row_tile + offset;
                let tile_bytes = 32 * bits;
                decode_ht_tile(
                    &bytes[t * tile_bytes..(t + 1) * tile_bytes],
                    bits,
                    &mut tile,
                );
                for (i, &value) in tile.iter().enumerate() {
                    let row = band * 128 + row_tile * 16 + (i / 8) % 16;
                    let col = col_tile * 16 + 8 * (i / 128) + i % 8;
                    values[row * cols + col] = value;
                }
            }
        }
    }
    values
}

/// Decode one projection of a complete paired gate/up 128-row band. The bytes
/// contain 8 gate tiles then 8 up tiles for each column tile, not two matrices.
pub fn decode_ht_gate_up_band(
    encoding: Encoding,
    bytes: &[u8],
    cols: usize,
    up: bool,
    meta: &QuantMeta<'_>,
) -> Result<Vec<f32>> {
    let bits = encoding.ht_bits().context("gate/up bands require HT")?;
    let expected = payload_layout(encoding, 256, cols)?.len;
    ensure!(
        bytes.len() as u64 == expected,
        "paired HT band size mismatch"
    );
    validate_decode_rotation(128, cols, meta)?;
    let mut values = decode_ht_rotated(bytes, bits, 128, cols, 16, if up { 8 } else { 0 });
    unrotate_in_place(&mut values, 128, cols, meta);
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "decoded HT band is nonfinite"
    );
    Ok(values)
}

/// Decode the SAME separated block representation returned by encode_rows,
/// to original-basis f32 values. Tensor headers/padding are handled by the
/// container reader, not this method. This is also the bounded source oracle.
pub fn decode_rows(
    encoding: Encoding,
    block: &EncodedBlock,
    rows: usize,
    cols: usize,
    meta: &QuantMeta<'_>,
) -> Result<Vec<f32>> {
    let layout = payload_layout(encoding, rows, cols)?;
    let count = rows
        .checked_mul(cols)
        .context("matrix element count overflow")?;
    let (code_len, scale_len) = match encoding {
        Encoding::Q4Cp | Encoding::I4r => (count / 2, rows * layout.scale_row_stride as usize),
        Encoding::Fp8G128 => (count, 0),
        _ => (
            usize::try_from(layout.len).context("block length exceeds address space")?,
            0,
        ),
    };
    ensure!(
        block.codes.len() == code_len && block.scales.len() == scale_len,
        "encoded block plane lengths do not match shape"
    );
    if encoding == Encoding::I64 {
        bail!("I64 cannot be decoded losslessly to f32; compare source integer bytes");
    }
    if encoding.ht_bits().is_some() {
        ensure!(
            rows == 128,
            "HT decoding consumes one 128-row band at a time"
        );
    }
    if encoding.ht_bits().is_some() || encoding == Encoding::I4r {
        validate_decode_rotation(rows, cols, meta)?;
    }
    if encoding == Encoding::Q4Cp {
        ensure!(
            meta.codebook.iter().all(|x| x.is_finite()),
            "codebook contains non-finite values"
        );
    }
    if encoding == Encoding::Fp8G128 {
        ensure!(
            meta.fp8_scale.is_finite() && meta.fp8_scale > 0.0,
            "fp8g needs a positive finite global scale"
        );
    }
    let mut values = match encoding.ht_bits() {
        Some(bits) => decode_ht_rotated(&block.codes, bits, rows, cols, 8, 0),
        None => vec![0.0; count],
    };
    match encoding {
        Encoding::Bf16 | Encoding::F16 => {
            for (value, bytes) in values.iter_mut().zip(block.codes.chunks_exact(2)) {
                *value = if encoding == Encoding::Bf16 {
                    bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
                } else {
                    read_half(bytes)
                };
            }
        }
        Encoding::F32 => {
            for (value, bytes) in values.iter_mut().zip(block.codes.chunks_exact(4)) {
                *value = f32::from_le_bytes(bytes.try_into().expect("four-byte value"));
            }
        }
        Encoding::Q4Cp | Encoding::I4r => {
            let stride = layout.scale_row_stride as usize;
            let group = if encoding == Encoding::Q4Cp { 32 } else { 128 };
            for row in 0..rows {
                let scale_row = &block.scales[row * stride..(row + 1) * stride];
                ensure!(
                    scale_row[cols / group * 2..].iter().all(|&b| b == 0),
                    "nonzero scale-row padding"
                );
                for g in 0..cols / group {
                    let scale = read_half(&scale_row[2 * g..2 * g + 2]);
                    ensure!(scale.is_finite(), "non-finite quantization scale");
                    for k in g * group..(g + 1) * group {
                        let byte = block.codes[(row * cols + k) / 2];
                        let code = (byte >> (4 * (k % 2))) & 15;
                        let level = if encoding == Encoding::Q4Cp {
                            meta.codebook[code as usize]
                        } else {
                            f32::from(code) - 8.0
                        };
                        values[row * cols + k] = level * scale;
                    }
                }
            }
        }
        Encoding::Q8G64 | Encoding::Q6G64 => {
            let stride = code_len / rows;
            let records = if encoding == Encoding::Q8G64 {
                cols
            } else {
                cols * 3 / 4
            };
            for (r, row) in block.codes.chunks_exact(stride).enumerate() {
                ensure!(
                    row[records + cols / 16..].iter().all(|&b| b == 0),
                    "nonzero affine-row padding"
                );
                for g in 0..cols / 64 {
                    let rec = records + 4 * g;
                    let scale = read_half(&row[rec..rec + 2]);
                    let bias = read_half(&row[rec + 2..rec + 4]);
                    ensure!(
                        scale.is_finite() && bias.is_finite(),
                        "non-finite affine scale or bias"
                    );
                    for k in g * 64..(g + 1) * 64 {
                        let code = if encoding == Encoding::Q8G64 {
                            row[k]
                        } else {
                            let low = (row[k / 2] >> (4 * (k % 2))) & 15;
                            let high = (row[cols / 2 + k / 4] >> (2 * (k % 4))) & 3;
                            low | (high << 4)
                        };
                        values[r * cols + k] = f32::from(code).mul_add(scale, bias);
                    }
                }
            }
        }
        Encoding::Fp8G128 => {
            for (value, &code) in values.iter_mut().zip(&block.codes) {
                *value = fp8_decode(code) * meta.fp8_scale;
            }
        }
        Encoding::Ht | Encoding::Ht3 => {}
        Encoding::I64 => unreachable!(),
    }
    if encoding.ht_bits().is_some() || encoding == Encoding::I4r {
        unrotate_in_place(&mut values, rows, cols, meta);
    }
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "decoded block contains non-finite values"
    );
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn hex(text: &str) -> Vec<u8> {
        let text: String = text.split_whitespace().collect();
        text.as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect()
    }

    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn spec_words(count: usize) -> Vec<u8> {
        (1..=count)
            .flat_map(|w| 0x9e3779b9u32.wrapping_mul(w as u32).to_le_bytes())
            .collect()
    }

    #[test]
    fn layouts_match_published_sizes_and_padding() {
        for (encoding, rows, cols, expected) in [
            (Encoding::Q4Cp, 512 * 1280, 2560, 943_718_464),
            (Encoding::Q4Cp, 512 * 2560, 640, 482_345_024),
            (Encoding::I4r, 512 * 1280, 2560, 865_075_200),
            (Encoding::I4r, 512 * 2560, 640, 432_537_600),
            (Encoding::Ht, 248320, 2560, 317_849_600),
            (Encoding::Q6G64, 10240, 320, 2_785_280),
            (Encoding::Q8G64, 2560, 6144, 16_711_680),
            (Encoding::Fp8G128, 128 * 2_500_012, 160, 51_200_245_764),
        ] {
            assert_eq!(payload_layout(encoding, rows, cols).unwrap().len, expected);
        }
        let q4 = payload_layout(Encoding::Q4Cp, 1, 64).unwrap();
        assert_eq!(
            (
                q4.len,
                q4.codes_offset,
                q4.scales_offset,
                q4.scale_row_stride
            ),
            (144, 64, 128, 16)
        );
        assert_eq!(payload_layout(Encoding::Q6G64, 1, 64).unwrap().len, 64);
        assert!(payload_layout(Encoding::F32, usize::MAX, 2).is_err());
        assert!(payload_layout(Encoding::Ht, 127, 128).is_err());
        assert!(payload_layout(Encoding::Q4Cp, 1, 31).is_err());
        assert!(payload_layout(Encoding::Fp8G128, 1, 128).is_err());
    }

    #[test]
    fn bf16_spec_rounding_and_dense_little_endian() {
        let values = [
            1.0,
            1.00390625,
            1.005859375,
            1.009765625,
            -3.1415927410125732,
            65504.0,
        ];
        let encoded = encode_rows(Encoding::Bf16, &values, 1, 6, &QuantMeta::default()).unwrap();
        assert_eq!(encoded.codes, hex("803f803f813f813f49c08047"));
        let encoded = encode_rows(
            Encoding::F16,
            &[1.0, -2.0, 0.0, -0.0],
            1,
            4,
            &QuantMeta::default(),
        )
        .unwrap();
        assert_eq!(encoded.codes, hex("003c00c000000080"));
        let values = [1.0, -2.0, -0.0];
        let encoded = encode_rows(Encoding::F32, &values, 1, 3, &QuantMeta::default()).unwrap();
        assert_eq!(encoded.codes, hex("0000803f000000c000000080"));
        assert_eq!(
            decode_rows(Encoding::F32, &encoded, 1, 3, &QuantMeta::default()).unwrap(),
            values
        );
        assert_eq!(
            encode_i64(&[3, 5, 7]),
            hex("030000000000000005000000000000000700000000000000")
        );
        let ple = [23703573157769i64, 20109073645365, 8052911324071];
        for (chunk, expected) in encode_i64(&ple).chunks_exact(8).zip(ple) {
            assert_eq!(i64::from_le_bytes(chunk.try_into().unwrap()), expected);
        }
        assert!(encode_rows(Encoding::I64, &[16_777_216.0], 1, 1, &QuantMeta::default()).is_err());
        assert!(encode_rows(Encoding::F16, &[70_000.0], 1, 1, &QuantMeta::default()).is_err());
    }

    fn f32_neighbors(value: f32) -> [f32; 3] {
        let bits = value.to_bits();
        let (below, above) = if value == 0.0 {
            (0x80000001, 1)
        } else if value > 0.0 {
            (bits - 1, bits + 1)
        } else {
            (bits + 1, bits - 1)
        };
        [f32::from_bits(below), value, f32::from_bits(above)]
    }

    #[test]
    fn q4_partition_preserves_all_midpoint_ties_and_neighbors() {
        let codebooks = [
            NF4_CODEBOOK,
            std::array::from_fn(|i| i as f32 - 8.0),
            std::array::from_fn(|i| (i as f32 - 8.0) * f32::from_bits(1)),
            // Adjacent entries produce repeated, rounded midpoints.
            std::array::from_fn(|i| f32::from_bits(0x3f800000 + i as u32)),
            std::array::from_fn(|i| f32::from_bits(0x7f7ffff0 + i as u32)),
            std::array::from_fn(|i| f32::from_bits(0xff7fffff - i as u32)),
        ];
        for cb in codebooks {
            let mids: [f32; 15] =
                std::array::from_fn(|i| (cb[i] as f64 * 0.5 + cb[i + 1] as f64 * 0.5) as f32);
            for value in mids
                .iter()
                .flat_map(|&mid| f32_neighbors(mid))
                .chain(cb)
                .chain([
                    f32::NEG_INFINITY,
                    -f32::MAX,
                    -f32::from_bits(1),
                    -0.0,
                    0.0,
                    f32::from_bits(1),
                    f32::MAX,
                    f32::INFINITY,
                ])
            {
                // Check the encoder's strict partition against a linear lower
                // bound, including ties at repeated, rounded midpoints.
                let expected = mids.iter().filter(|&&mid| mid < value).count();
                assert_eq!(
                    mids.partition_point(|mid| *mid < value),
                    expected,
                    "value bits {:08x}, codebook {cb:?}",
                    value.to_bits()
                );
            }
        }
    }

    #[test]
    fn small_code_rounding_matches_rne_at_every_boundary() {
        for (lower, upper) in [(-8.0, 7.0), (0.0, 63.0), (0.0, 255.0)] {
            let check = |value: f32| {
                assert_eq!(
                    round_small_code(value, lower, upper),
                    value.round_ties_even().clamp(lower, upper) as i32,
                    "value bits {:08x}, bounds [{lower}, {upper}]",
                    value.to_bits()
                );
            };
            for code in lower as i32 - 1..=upper as i32 + 1 {
                for value in f32_neighbors(code as f32)
                    .into_iter()
                    .chain(f32_neighbors(code as f32 + 0.5))
                {
                    check(value);
                }
            }
            for value in [
                f32::NEG_INFINITY,
                -f32::MAX,
                -f32::MIN_POSITIVE,
                -f32::from_bits(1),
                -0.0,
                0.0,
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                f32::MAX,
                f32::INFINITY,
            ] {
                check(value);
            }
            let mut bits = 0x9e3779b9u32;
            for _ in 0..65_536 {
                bits ^= bits << 13;
                bits ^= bits >> 17;
                bits ^= bits << 5;
                let value = f32::from_bits(bits);
                if !value.is_nan() {
                    check(value);
                }
            }
        }
    }

    #[test]
    fn affine_halfway_neighbors_preserve_metadata_and_planes() {
        for (encoding, max_code) in [(Encoding::Q6G64, 63), (Encoding::Q8G64, 255)] {
            let mut input = Vec::new();
            let mut expected = Vec::new();
            for code in 0..max_code {
                for value in f32_neighbors(code as f32 + 0.5) {
                    // Endpoints fix the stored half scale/bias at exactly 1/0.
                    input.extend_from_slice(&[0.0, max_code as f32]);
                    input.extend([value; 62]);
                    expected.extend_from_slice(&[0.0, max_code as f32]);
                    expected.extend([value.round_ties_even(); 62]);
                }
            }
            let rows = input.len() / 64;
            let meta = QuantMeta::default();
            let encoded = encode_rows(encoding, &input, rows, 64, &meta).unwrap();
            assert!(encoded.scales.is_empty());
            let record = if encoding == Encoding::Q8G64 { 64 } else { 48 };
            for row in encoded.codes.chunks_exact(encoded.codes.len() / rows) {
                assert_eq!(&row[record..record + 4], &[0x00, 0x3c, 0x00, 0x00]);
                assert!(row[record + 4..].iter().all(|&byte| byte == 0));
            }
            // At scale 1/bias 0 the decoded integer uniquely identifies every
            // packed code, including all six-bit high planes.
            assert_eq!(
                decode_rows(encoding, &encoded, rows, 64, &meta).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn q4_nf4_golden_uses_common_nibble_order_and_scale_row_padding() {
        let cb_bytes: Vec<u8> = NF4_CODEBOOK.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            cb_bytes,
            hex(
                "000080bfb13932bf306b06bfa032cabe4da291be3f353dbe7178babd00000000
            fffaa23de3ca243edd047c3e3a03ad3eb8a4e13eab07103fb313393f0000803f"
            )
        );
        let input: Vec<f32> = (0..32)
            .map(|i| NF4_CODEBOOK[(5 * i) % 16] * 0.03125)
            .collect();
        let encoded = encode_rows(Encoding::Q4Cp, &input, 1, 32, &QuantMeta::default()).unwrap();
        // The spec variant-1 vector has the same codes and group scale as
        // variant 2; only the placement and tail padding differ.
        assert_eq!(encoded.codes, hex("50fa943ed8721cb650fa943ed8721cb6"));
        assert_eq!(encoded.scales, hex("00280000000000000000000000000000"));
        assert_eq!(
            decode_rows(Encoding::Q4Cp, &encoded, 1, 32, &QuantMeta::default()).unwrap(),
            input
        );
        let mut corrupt = encoded;
        corrupt.scales[15] = 1;
        assert!(decode_rows(Encoding::Q4Cp, &corrupt, 1, 32, &QuantMeta::default()).is_err());
    }

    #[test]
    fn q4_signed_scale_decoder_matches_spec_q4_0_vector() {
        let cb = std::array::from_fn(|i| i as f32 - 8.0);
        let meta = QuantMeta {
            codebook: &cb,
            ..QuantMeta::default()
        };
        let block = EncodedBlock {
            codes: hex("832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9"),
            scales: hex("003800b4000000000000000000000000"),
        };
        let decoded = decode_rows(Encoding::Q4Cp, &block, 1, 64, &meta).unwrap();
        assert_eq!(
            &decoded[..8],
            &[-2.5, 0.0, 2.5, -3.0, -0.5, 2.0, -3.5, -1.0]
        );
        assert_eq!(
            &decoded[32..40],
            &[1.25, -0.0, -1.25, 1.5, 0.25, -1.0, 1.75, 0.5]
        );
    }

    #[test]
    fn q6_spec_golden_is_an_exact_encoder_fixed_point() {
        let input: Vec<f32> = (0..64)
            .map(|k| ((37 * k + 11) % 64) as f32 * 0.03125 - 1.0)
            .collect();
        let encoded = encode_rows(Encoding::Q6G64, &input, 1, 64, &QuantMeta::default()).unwrap();
        assert_eq!(
            encoded.codes,
            hex(
                "0ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc761
            dc21b7d82176c82d768b1d728bdc6287002800bc000000000000000000000000"
            )
        );
        assert_eq!(
            decode_rows(Encoding::Q6G64, &encoded, 1, 64, &QuantMeta::default()).unwrap(),
            input
        );
        let input: Vec<f32> = (0..640).map(|i| (i % 64) as f32 - 32.0).collect();
        let encoded = encode_rows(Encoding::Q6G64, &input, 2, 320, &QuantMeta::default()).unwrap();
        assert_eq!(encoded.codes.len(), 544);
        assert!(encoded.codes[260..272].iter().all(|&v| v == 0));
        assert!(encoded.codes[532..544].iter().all(|&v| v == 0));
        assert_eq!(
            decode_rows(Encoding::Q6G64, &encoded, 2, 320, &QuantMeta::default()).unwrap(),
            input
        );
    }

    #[test]
    fn q8_affine_known_bytes_and_rne_ties() {
        let input: Vec<f32> = (0..64)
            .map(|k| match k {
                63 => 255.0,
                _ => k as f32,
            })
            .collect();
        let encoded = encode_rows(Encoding::Q8G64, &input, 1, 64, &QuantMeta::default()).unwrap();
        assert_eq!(&encoded.codes[..63], &(0..63u8).collect::<Vec<_>>());
        assert_eq!(&encoded.codes[63..], &[255, 0x00, 0x3c, 0, 0]);
        assert_eq!(
            decode_rows(Encoding::Q8G64, &encoded, 1, 64, &QuantMeta::default()).unwrap(),
            input
        );
        let mut ties = input;
        ties[1] = 0.5;
        ties[2] = 1.5;
        ties[3] = 2.5;
        let encoded = encode_rows(Encoding::Q8G64, &ties, 1, 64, &QuantMeta::default()).unwrap();
        assert_eq!(&encoded.codes[1..4], &[0, 2, 2]);
        let encoded =
            encode_rows(Encoding::Q8G64, &[2.0; 128], 2, 64, &QuantMeta::default()).unwrap();
        assert_eq!(&encoded.codes[64..68], &[0, 0, 0, 0x40]);
        assert_eq!(
            decode_rows(Encoding::Q8G64, &encoded, 2, 64, &QuantMeta::default()).unwrap(),
            vec![2.0; 128]
        );
    }

    #[test]
    fn fp8_spec_values_boundaries_and_global_scale() {
        for (code, expected) in [
            (0x00, 0.0),
            (0x01, 0.001953125),
            (0x07, 0.013671875),
            (0x08, 0.015625),
            (0x38, 1.0),
            (0x3f, 1.875),
            (0x40, 2.0),
            (0x77, 240.0),
            (0x7e, 448.0),
            (0x7f, 480.0),
            (0xb8, -1.0),
            (0xff, -480.0),
        ] {
            assert_eq!(fp8_decode(code), expected);
            assert_eq!(
                fp8_encode_normalized(expected),
                if code == 0x7f {
                    0x7e
                } else if code == 0xff {
                    0xfe
                } else {
                    code
                }
            );
        }
        assert_eq!(fp8_decode(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(fp8_encode_normalized(-0.0), 0x80);
        for (value, code) in [
            (1.0625, 0x38),
            (1.1875, 0x3a),
            (0.0009765625, 0),
            (0.0029296875, 2),
            (0.0146484375, 8),
            (464.0, 0x7e),
        ] {
            assert_eq!(fp8_encode_normalized(value), code);
        }
        // Independent exhaustive nearest-code reference, including every tie.
        for code in 0..126u8 {
            let a = fp8_decode(code);
            let b = fp8_decode(code + 1);
            let midpoint = (a + b) / 2.0;
            assert_eq!(
                fp8_encode_normalized(midpoint),
                if code % 2 == 0 { code } else { code + 1 }
            );
        }
        let mut input = vec![0.0; 160];
        input[..4].copy_from_slice(&[0.5, -0.5, 224.0, -0.0]);
        let meta = QuantMeta {
            fp8_scale: 0.5,
            ..QuantMeta::default()
        };
        let encoded = encode_rows(Encoding::Fp8G128, &input, 1, 160, &meta).unwrap();
        assert_eq!(&encoded.codes[..4], &[0x38, 0xb8, 0x7e, 0x80]);
        assert_eq!(
            decode_rows(Encoding::Fp8G128, &encoded, 1, 160, &meta).unwrap(),
            input
        );
        assert_eq!(
            payload_layout(Encoding::Fp8G128, 1, 160)
                .unwrap()
                .scales_offset,
            192
        );
    }

    #[test]
    fn i4r_spec_row_signed_scale_and_offset_binary() {
        let rotated: Vec<f32> = (0..128)
            .map(|k| (((5 * k + 3) % 16) as f32 - 8.0) * 0.015625)
            .collect();
        let encoded = encode_i4r_rotated(&rotated).unwrap();
        assert_eq!(
            encoded.codes,
            hex(
                "832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9
            832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9832dc7610ba54fe9"
            )
        );
        assert_eq!(encoded.scales, hex("0024"));
        let negative: Vec<f32> = rotated.iter().map(|v| -v).collect();
        let encoded_negative = encode_i4r_rotated(&negative).unwrap();
        assert_eq!(encoded_negative.codes, encoded.codes);
        assert_eq!(encoded_negative.scales, hex("00a4"));
    }

    #[test]
    fn ht_codebook_and_cyclic_tile_golden() {
        for (state, bits) in [
            (0x0000, 0xc2e8),
            (0x0001, 0x3921),
            (0x0002, 0xb72d),
            (0x1234, 0xb89e),
            (0x8000, 0xbaf2),
            (0xabcd, 0xbd42),
            (0xffff, 0xb936),
        ] {
            assert_eq!(f16::from_f32(ht_codebook_value(state)).to_bits(), bits);
        }
        let bytes = spec_words(32);
        assert_eq!(&bytes[..16], &hex("b979379e72f36e3c2b6da6dae4e6dd78"));
        let mut tile = [0.0; 256];
        decode_ht_tile(&bytes, 4, &mut tile);
        let decoded_bits: Vec<u8> = tile
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(
            digest(&decoded_bits),
            "32ae1812fa6a2dc63150621a31ec916757607f03eb9058d51923ab7b92a78d9b"
        );
        assert_eq!(tile[0], 1.48046875);
        assert_eq!(tile[128], -1.341796875);
        let symbols: Vec<u8> = (0..256).map(|i| (i % 16) as u8).collect();
        assert_eq!(&pack_ht_symbols(&symbols, 4)[..8], &hex("67452301efcdab89"));
    }

    #[test]
    fn ht_full_rotation_matches_exact_spec_hash() {
        let bytes = spec_words(2048);
        assert_eq!(
            digest(&bytes),
            "cfab929290971c665ca08cb3222f626e41bfb1b013837ced3b7178136b8f2ca1"
        );
        let su: Vec<f32> = (0..128)
            .map(|c| if (5 * c) % 7 < 3 { -1.0 } else { 1.0 })
            .collect();
        let sv: Vec<f32> = (0..128)
            .map(|r| (8 + r % 8) as f32 / 512.0 * if r % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ..QuantMeta::default()
        };
        let block = EncodedBlock {
            codes: bytes,
            scales: Vec::new(),
        };
        let values = decode_rows(Encoding::Ht, &block, 128, 128, &meta).unwrap();
        let decoded: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            digest(&decoded),
            "1a1f0f4bee42296506b55f20ae44b69a6b558ee16bf5c70ec9519fa52cbfc1b2"
        );
        assert_eq!(values[0], -0.0005001630634069443);
        assert_eq!(values[129], 0.006169011816382408);
    }

    #[test]
    fn independent_rotation_impulse_and_i4r_streaming_planes() {
        let su = generate_signs(17, "experts.su", 256);
        let sv = generate_signs(17, "experts.gate_up_proj.svh", 256);
        let mut input = vec![0.0f32; 128 * 256];
        input[3 * 256 + 5] = 128.0;
        let meta = QuantMeta {
            su: &su,
            sv: &sv[..128],
            ..QuantMeta::default()
        };
        let mut rotated = vec![0.0; input.len()];
        rotate_into(&input, 128, 256, &meta, RotationOrder::Rows, &mut rotated).unwrap();
        for r in 0..128 {
            for c in 0..256 {
                let expected = if c >= 128 {
                    0.0
                } else {
                    let parity = ((r & 3usize).count_ones() + (c & 5usize).count_ones()) % 2;
                    su[5] * sv[3] * if parity == 0 { 1.0 } else { -1.0 }
                };
                assert_eq!(rotated[r * 256 + c], expected);
            }
        }
        unrotate_in_place(&mut rotated, 128, 256, &meta);
        assert_eq!(rotated, input);
        let first = encode_rows(Encoding::I4r, &input, 128, 256, &meta).unwrap();
        let reconstructed = decode_rows(Encoding::I4r, &first, 128, 256, &meta).unwrap();
        let error: f64 = reconstructed
            .iter()
            .zip(&input)
            .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
            .sum();
        assert!(error > 0.0 && error / (128.0 * 128.0) < 0.02);
        let zero = vec![0.0; input.len()];
        let other_meta = QuantMeta {
            su: &su,
            sv: &sv[128..],
            ..QuantMeta::default()
        };
        let second = encode_rows(Encoding::I4r, &zero, 128, 256, &other_meta).unwrap();
        let layout = payload_layout(Encoding::I4r, 256, 256).unwrap();
        let mut payload = vec![0u8; layout.len as usize];
        payload[..first.codes.len()].copy_from_slice(&first.codes);
        payload[first.codes.len()..layout.scales_offset as usize].copy_from_slice(&second.codes);
        let split = layout.scales_offset as usize;
        payload[split..split + first.scales.len()].copy_from_slice(&first.scales);
        payload[split + first.scales.len()..].copy_from_slice(&second.scales);
        assert_eq!(payload[split..split + 2], first.scales[..2]);
        assert_eq!(second.codes, vec![0x88; 128 * 256 / 2]);
        let decoded = decode_rows(Encoding::I4r, &second, 128, 256, &other_meta).unwrap();
        assert!(decoded.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn gate_up_tiles_are_interleaved_by_column_not_projection() {
        let cols = 256;
        let su = vec![1.0; cols];
        let sv = vec![1.0; 128];
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ..QuantMeta::default()
        };
        let gate: Vec<f32> = (0..128 * cols).map(|i| (i % 29) as f32 / 16.0).collect();
        let up: Vec<f32> = (0..128 * cols).map(|i| -((i % 31) as f32) / 8.0).collect();
        let mut paired = vec![0.0; gate.len() * 2];
        rotate_into(
            &gate,
            128,
            cols,
            &meta,
            RotationOrder::Tiles {
                row_tiles: 16,
                row_tile_offset: 0,
            },
            &mut paired,
        )
        .unwrap();
        rotate_into(
            &up,
            128,
            cols,
            &meta,
            RotationOrder::Tiles {
                row_tiles: 16,
                row_tile_offset: 8,
            },
            &mut paired,
        )
        .unwrap();
        // Independent row-matrix oracle and the specification's explicit t/i
        // formulas catch both tile order and the two 8-column half streams.
        let mut gate_rows = vec![0.0; gate.len()];
        let mut up_rows = vec![0.0; up.len()];
        rotate_into(&gate, 128, cols, &meta, RotationOrder::Rows, &mut gate_rows).unwrap();
        rotate_into(&up, 128, cols, &meta, RotationOrder::Rows, &mut up_rows).unwrap();
        for r in 0..128 {
            for c in 0..cols {
                let tile = (c / 16) * 16 + r / 16;
                let i = (c % 16 / 8) * 128 + (r % 16) * 8 + c % 8;
                assert_eq!(paired[tile * 256 + i], gate_rows[r * cols + c]);
                assert_eq!(paired[(tile + 8) * 256 + i], up_rows[r * cols + c]);
            }
        }
        let bytes = spec_words(128 * cols / 4);
        let gate_decoded = decode_ht_rotated(&bytes, 4, 128, cols, 16, 0);
        let up_decoded = decode_ht_rotated(&bytes, 4, 128, cols, 16, 8);
        let mut first_gate = [0.0; 256];
        let mut first_up = [0.0; 256];
        decode_ht_tile(&bytes[..128], 4, &mut first_gate);
        decode_ht_tile(&bytes[1024..1152], 4, &mut first_up);
        assert_eq!(gate_decoded[0], first_gate[0]);
        assert_eq!(up_decoded[0], first_up[0]);
        assert_eq!(gate_decoded[8], first_gate[128]);
    }

    #[test]
    fn oversized_native_ht_geometry_is_rejected_without_allocating() {
        let max_native_values = c_int::MAX as usize / 256 * 256;
        assert_eq!(
            native_ht_tile_count(max_native_values).unwrap() as usize,
            max_native_values / 256
        );
        assert!(native_ht_tile_count(max_native_values + 256).is_err());
        assert!(native_ht_tile_count(usize::MAX).is_err());
        assert!(native_ht_tile_count(0).is_err());
        let max_band_values =
            (c_int::MAX as usize).min(isize::MAX as usize / std::mem::size_of::<f32>());
        let max_dense_cols = max_band_values / (128 * 128) * 128;
        let max_paired_cols = max_band_values / (256 * 128) * 128;
        assert!(ht_band_value_count(128, max_dense_cols, false).is_ok());
        assert!(ht_band_value_count(128, max_dense_cols + 128, false).is_err());
        assert!(ht_band_value_count(128, max_paired_cols, true).is_ok());
        assert!(ht_band_value_count(128, max_paired_cols + 128, true).is_err());
        assert!(ht_band_value_count(128, usize::MAX & !127, true).is_err());
        // Public calls reach the geometry rejection before inspecting source
        // slices or vectors, so the oversized case needs no real allocation.
        let error = encode_rows(Encoding::Ht, &[], 128, 1 << 24, &QuantMeta::default())
            .unwrap_err()
            .to_string();
        assert!(error.contains("signed-int element offset"));
        let error = encode_ht_gate_up_bands(
            Encoding::Ht,
            &[],
            &[],
            1 << 23,
            &QuantMeta::default(),
            &QuantMeta::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("signed-int element offset"));
    }

    #[test]
    fn public_ht_gate_up_encoder_matches_canonical_band_pair() {
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ht_beam: 1,
            ..QuantMeta::default()
        };
        let mut gate = vec![0.0; 128 * 128];
        gate[0] = -442.0; // Constant rotated -3.453125, exactly zero symbols.
        let up = vec![0.0; 128 * 128];
        let gate_block = encode_rows(Encoding::Ht, &gate, 128, 128, &meta).unwrap();
        let up_block = encode_rows(Encoding::Ht, &up, 128, 128, &meta).unwrap();
        assert_ne!(gate_block.codes, up_block.codes);
        let paired = encode_ht_gate_up_bands(Encoding::Ht, &gate, &up, 128, &meta, &meta).unwrap();
        assert_eq!(paired.len(), 16_384);
        for c in 0..8 {
            assert_eq!(
                &paired[c * 2048..c * 2048 + 1024],
                &gate_block.codes[c * 1024..(c + 1) * 1024]
            );
            assert_eq!(
                &paired[c * 2048 + 1024..(c + 1) * 2048],
                &up_block.codes[c * 1024..(c + 1) * 1024]
            );
        }
    }

    #[test]
    fn native_trellis_is_real_closed_and_thread_deterministic() {
        // An exactly representable cyclic path with a unique minimum-valued
        // state: the external native encoder must return 256 zero symbols.
        let source = [-3.453125f32; 256];
        let meta = QuantMeta {
            ht_beam: 128,
            threads: 1,
            ..QuantMeta::default()
        };
        let encoded = encode_ht_tiles(&source, &meta, 4).unwrap();
        assert_eq!(encoded, vec![0u8; 128]);
        let encoded_two = encode_ht_tiles(&source, &QuantMeta { threads: 2, ..meta }, 4).unwrap();
        assert_eq!(encoded, encoded_two);
        let mut decoded = [0.0; 256];
        decode_ht_tile(&encoded, 4, &mut decoded);
        assert_eq!(decoded, source);
    }

    #[test]
    fn scales_signs_zero_rows_and_invalid_inputs() {
        let a = generate_signs(123, "layers.0.test.suh", 128);
        assert_eq!(a, generate_signs(123, "layers.0.test.suh", 128));
        assert_ne!(a, generate_signs(124, "layers.0.test.suh", 128));
        assert_eq!(&a[..8], &generate_signs(123, "layers.0.test.suh", 8));
        assert!(a.contains(&1.0) && a.contains(&-1.0));
        let input = [0.0, 0.0, 3.0, 4.0, 1.0e-12, 1.0e-12];
        let scales = ht_row_scales(&input, 3, 2, &[1.0, -1.0, 1.0]).unwrap();
        assert_eq!(scales[0], 0.0);
        assert_eq!(scales[1], -f16::from_f64((12.5f64).sqrt()).to_f32());
        assert_eq!(scales[2], f16::from_bits(1).to_f32());
        assert!(encode_rows(Encoding::Bf16, &[f32::NAN], 1, 1, &QuantMeta::default()).is_err());
        let input = vec![0.0; 128 * 128];
        let su = vec![1.0; 128];
        let sv = vec![0.0; 128];
        let meta = QuantMeta {
            su: &su,
            sv: &sv,
            ..QuantMeta::default()
        };
        let mut out = vec![0.0; input.len()];
        rotate_into(&input, 128, 128, &meta, RotationOrder::Rows, &mut out).unwrap();
        assert!(out.iter().all(|&x| x == 0.0));
        let mut revived = input;
        revived[0] = 1.0;
        assert!(rotate_into(&revived, 128, 128, &meta, RotationOrder::Rows, &mut out).is_err());
        assert!(encode_rows(Encoding::I4r, &revived, 128, 128, &meta).is_err());
        assert!(encode_ht_tiles(&[f32::NAN; 256], &QuantMeta::default(), 4).is_err());
        assert!(encode_ht_tiles(
            &[0.0; 256],
            &QuantMeta {
                ht_beam: 4097,
                ..QuantMeta::default()
            },
            4
        )
        .is_err());
    }

    #[test]
    fn ht3_published_tile_golden_covers_word_crossings_and_wraparound() {
        let bytes = spec_words(24);
        let mut tile = [0.0; 256];
        decode_ht_tile(&bytes, 3, &mut tile);
        let halves: Vec<u8> = tile
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(
            digest(&halves),
            "415420d5d59d5f2b288f563473b597b207e94852290120291887286c5f00555b"
        );
        for (index, bits) in [
            (0, 0xafea),
            (1, 0x3f08),
            (2, 0xb8e3),
            (7, 0xb257),
            (8, 0xafea),
            (127, 0xb7ef),
            (128, 0x3cf9),
            (255, 0x39f1),
        ] {
            assert_eq!(f16::from_f32(tile[index]).to_bits(), bits);
        }
        let symbols: Vec<u8> = (0..512).map(|i| (i % 8) as u8).collect();
        let packed = pack_ht_symbols(&symbols, 3);
        assert_eq!(packed.len(), 192);
        assert_eq!(&packed[..12], &hex("057739053905773977390577"));
        // A bit-at-a-time, word-MSB-first oracle includes symbols straddling
        // every word boundary; it does not call the trellis decoder.
        for (i, &symbol) in symbols.iter().enumerate() {
            let mut actual = 0;
            for bit in 0..3 {
                let position = i * 3 + bit;
                let word = u32::from_le_bytes(
                    packed[position / 32 * 4..position / 32 * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                actual = (actual << 1) | ((word >> (31 - position % 32)) & 1) as u8;
            }
            assert_eq!(actual, symbol);
        }
        assert_eq!(
            payload_layout(Encoding::Ht3, 512 * 1280, 2560).unwrap().len,
            629_145_600
        );
        assert_eq!(
            payload_layout(Encoding::Ht3, 512 * 2560, 640).unwrap().len,
            314_572_800
        );
    }

    #[test]
    fn ht3_fixed_search_closes_thirteen_bits_not_five_whole_symbols() {
        let mut source = [0.0; 256];
        decode_ht_tile(&spec_words(24), 3, &mut source);
        let boundary = 0x6958 & 0x1fff;
        let mut symbols = [0u8; 256];
        // SAFETY: native search receives complete 256-value input/output and
        // its immutable 65536-state codebook, with a valid 13-bit boundary.
        let cost = unsafe {
            ht_search_fixed_bits(
                source.as_ptr(),
                ht_codebook().as_ptr(),
                256,
                boundary,
                symbols.as_mut_ptr(),
                3,
            )
        };
        assert_eq!(cost, 0.0);
        let tail = symbols[251..]
            .iter()
            .fold(0u16, |state, &symbol| (state << 3) | u16::from(symbol));
        assert_eq!(tail & 0x1fff, boundary as u16);
        let packed = pack_ht_symbols(&symbols, 3);
        let mut decoded = [0.0; 256];
        decode_ht_tile(&packed, 3, &mut decoded);
        assert_eq!(decoded, source);
        for beam in [1, 7, 128] {
            let meta = QuantMeta {
                ht_beam: beam,
                ..QuantMeta::default()
            };
            let encoded = encode_ht_tiles(&[-3.453125; 256], &meta, 3).unwrap();
            assert_eq!(encoded, vec![0; 96]);
            assert_eq!(
                encoded,
                encode_ht_tiles(&[-3.453125; 256], &QuantMeta { threads: 2, ..meta }, 3).unwrap()
            );
        }
    }

    #[test]
    fn ht3_pairing_has_distinct_gate_up_tiles_and_inverse_scales() {
        let cols = 256;
        let su = vec![1.0; cols];
        let gate_sv = vec![0.5; 128];
        let up_sv = vec![-0.25; 128];
        let gate_meta = QuantMeta {
            su: &su,
            sv: &gate_sv,
            ht_beam: 1,
            ..QuantMeta::default()
        };
        let up_meta = QuantMeta {
            sv: &up_sv,
            ..gate_meta
        };
        let mut gate = vec![0.0; 128 * cols];
        let mut up = vec![0.0; 128 * cols];
        gate[0] = -221.0;
        gate[128] = -221.0;
        up[0] = 110.5;
        up[128] = 110.5;
        let bytes =
            encode_ht_gate_up_bands(Encoding::Ht3, &gate, &up, cols, &gate_meta, &up_meta).unwrap();
        assert_eq!(bytes, vec![0; 256 * cols * 3 / 8]);
        assert_eq!(
            decode_ht_gate_up_band(Encoding::Ht3, &bytes, cols, false, &gate_meta).unwrap(),
            gate
        );
        assert_eq!(
            decode_ht_gate_up_band(Encoding::Ht3, &bytes, cols, true, &up_meta).unwrap(),
            up
        );
        // Constant stored gate codebook[0] and up codebook[65535] give known
        // original-basis impulses. Fill up tiles directly, without an encoder.
        let mut mixed = vec![0; bytes.len()];
        for column in 0..cols / 16 {
            mixed[column * 16 * 96 + 8 * 96..(column + 1) * 16 * 96].fill(255);
        }
        let decoded_gate =
            decode_ht_gate_up_band(Encoding::Ht3, &mixed, cols, false, &gate_meta).unwrap();
        let decoded_up =
            decode_ht_gate_up_band(Encoding::Ht3, &mixed, cols, true, &up_meta).unwrap();
        assert_eq!(decoded_gate, gate);
        assert_eq!(decoded_up[0], 20.84375);
        assert_eq!(decoded_up[128], 20.84375);
        assert_eq!(decoded_up.iter().filter(|&&value| value != 0.0).count(), 2);
        assert!(decode_ht_gate_up_band(
            Encoding::Ht3,
            &mixed[..mixed.len() - 1],
            cols,
            true,
            &up_meta
        )
        .is_err());
        let nonunit = vec![2.0; cols];
        let nonunit_meta = QuantMeta {
            su: &nonunit,
            ..gate_meta
        };
        let decoded =
            decode_ht_gate_up_band(Encoding::Ht3, &mixed, cols, false, &nonunit_meta).unwrap();
        assert_eq!(decoded[0], -442.0); // Decoder permits legal non-unit f16 vectors.
        assert!(encode_rows(Encoding::Ht3, &gate, 128, cols, &nonunit_meta).is_err());
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[cfg(feature = "hip")]
    unsafe extern "C" {
        fn hipSetDevice(device: c_int) -> c_int;
    }

    #[cfg(feature = "cuda")]
    unsafe extern "C" {
        fn cudaSetDevice(device: c_int) -> c_int;
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    type GpuEncoder =
        unsafe extern "C" fn(*const f32, *const f32, *mut u8, c_int, c_int, c_int) -> c_int;

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn gpu_encoder(backend: HtBackend) -> GpuEncoder {
        match backend {
            #[cfg(feature = "hip")]
            HtBackend::Hip => ht_encode_bits_hip,
            #[cfg(feature = "cuda")]
            HtBackend::Cuda => ht_encode_bits_cuda,
            _ => panic!("{} encoder feature is not enabled", backend.label()),
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn gpu_last_error(backend: HtBackend) -> String {
        match backend {
            HtBackend::Cpu => "native CPU encoder rejected parameters".to_owned(),
            #[cfg(feature = "hip")]
            HtBackend::Hip => hip_last_error(),
            #[cfg(feature = "cuda")]
            HtBackend::Cuda => cuda_last_error(),
            #[cfg(not(all(feature = "hip", feature = "cuda")))]
            _ => panic!("{} encoder feature is not enabled", backend.label()),
        }
    }

    #[test]
    fn ht_backend_defaults_and_beam_boundaries() {
        assert_eq!(HtBackend::default(), HtBackend::Cpu);
        assert_eq!(QuantMeta::default().ht_backend, HtBackend::Cpu);
        for beam in [0, 1, 256, 4096] {
            assert!(validate_ht_backend(&QuantMeta {
                ht_beam: beam,
                ..Default::default()
            })
            .is_ok());
        }
        assert!(validate_ht_backend(&QuantMeta {
            ht_beam: 4097,
            ..Default::default()
        })
        .is_err());
        for (backend, enabled) in [
            (HtBackend::Hip, cfg!(feature = "hip")),
            (HtBackend::Cuda, cfg!(feature = "cuda")),
        ] {
            assert!(backend.is_gpu());
            for beam in [0, 257, 4096, usize::MAX] {
                let meta = QuantMeta {
                    ht_backend: backend,
                    ht_beam: beam,
                    ..Default::default()
                };
                assert!(validate_ht_backend(&meta)
                    .unwrap_err()
                    .to_string()
                    .contains("1..=256"));
            }
            for beam in [1, 8, 128, 256] {
                let meta = QuantMeta {
                    ht_backend: backend,
                    ht_beam: beam,
                    ..Default::default()
                };
                assert_eq!(validate_ht_backend(&meta).is_ok(), enabled);
            }
        }
        assert!(!HtBackend::Cpu.is_gpu());
        assert_eq!(HtBackend::Cpu.label(), "CPU");
        assert_eq!(HtBackend::Hip.label(), "HIP");
        assert_eq!(HtBackend::Cuda.label(), "CUDA");
    }

    #[test]
    fn non_ht_quantizers_and_decoding_ignore_execution_backend() {
        let input: Vec<f32> = (0..128 * 160)
            .map(|i| (i % 37) as f32 / 32.0 - 0.5)
            .collect();
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let cpu = QuantMeta {
            su: &su,
            sv: &sv,
            ht_beam: 0,
            ..Default::default()
        };
        for encoding in [
            Encoding::Bf16,
            Encoding::F16,
            Encoding::F32,
            Encoding::Q4Cp,
            Encoding::Q6G64,
            Encoding::Q8G64,
            Encoding::Fp8G128,
            Encoding::I4r,
        ] {
            let cols = if encoding == Encoding::Fp8G128 {
                160
            } else {
                128
            };
            let values = &input[..128 * cols];
            let expected = encode_rows(encoding, values, 128, cols, &cpu).unwrap();
            for backend in [HtBackend::Hip, HtBackend::Cuda] {
                let gpu = QuantMeta {
                    ht_backend: backend,
                    ..cpu
                };
                assert_eq!(
                    encode_rows(encoding, values, 128, cols, &gpu).unwrap(),
                    expected
                );
                assert_eq!(
                    decode_rows(encoding, &expected, 128, cols, &gpu).unwrap(),
                    decode_rows(encoding, &expected, 128, cols, &cpu).unwrap()
                );
            }
        }
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            let bytes = payload_layout(encoding, 128, 128).unwrap().len as usize;
            let block = EncodedBlock {
                codes: vec![0; bytes],
                scales: Vec::new(),
            };
            for backend in [HtBackend::Hip, HtBackend::Cuda] {
                let gpu = QuantMeta {
                    ht_backend: backend,
                    ..cpu
                };
                assert_eq!(
                    decode_rows(encoding, &block, 128, 128, &gpu).unwrap(),
                    decode_rows(encoding, &block, 128, 128, &cpu).unwrap()
                );
            }
        }
    }

    #[test]
    fn paired_bands_reject_different_backends_before_execution() {
        let input = vec![0.0; 128 * 128];
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let cpu = QuantMeta {
            su: &su,
            sv: &sv,
            ..Default::default()
        };
        for (left, right) in [
            (HtBackend::Cpu, HtBackend::Hip),
            (HtBackend::Cpu, HtBackend::Cuda),
            (HtBackend::Hip, HtBackend::Cuda),
        ] {
            let left = QuantMeta {
                ht_backend: left,
                ..cpu
            };
            let right = QuantMeta {
                ht_backend: right,
                ..cpu
            };
            for encoding in [Encoding::Ht, Encoding::Ht3] {
                let error = encode_ht_gate_up_bands(encoding, &input, &input, 128, &left, &right)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("same encoder settings"));
            }
        }
    }

    #[cfg(any(not(feature = "hip"), not(feature = "cuda")))]
    fn dispatch_without_feature_is_an_explicit_error(backend: HtBackend, feature: &str) {
        assert!(backend
            .preflight()
            .unwrap_err()
            .to_string()
            .contains(feature));
        let gpu = QuantMeta {
            ht_backend: backend,
            ht_beam: 8,
            ..Default::default()
        };
        for bits in [3, 4] {
            let error = encode_ht_tiles(&[0.0; 256], &gpu, bits)
                .unwrap_err()
                .to_string();
            assert!(error.contains(feature));
        }
        let su = vec![1.0; 128];
        let sv = vec![1.0; 128];
        let gpu = QuantMeta {
            su: &su,
            sv: &sv,
            ..gpu
        };
        let input = vec![0.0; 128 * 128];
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            assert!(encode_rows(encoding, &input, 128, 128, &gpu)
                .unwrap_err()
                .to_string()
                .contains(feature));
            assert!(
                encode_ht_gate_up_bands(encoding, &input, &input, 128, &gpu, &gpu)
                    .unwrap_err()
                    .to_string()
                    .contains(feature)
            );
        }
    }

    #[cfg(not(feature = "hip"))]
    #[test]
    fn hip_dispatch_without_feature_is_an_explicit_error() {
        dispatch_without_feature_is_an_explicit_error(HtBackend::Hip, "--features hip");
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_dispatch_without_feature_is_an_explicit_error() {
        dispatch_without_feature_is_an_explicit_error(HtBackend::Cuda, "--features cuda");
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn native_rejects_invalid_parameters_without_a_device(backend: HtBackend) {
        let encode = gpu_encoder(backend);
        let input = [0.0; 256];
        let codebook = ht_codebook();
        let mut output = [0x55; 256];
        // A count whose complete values exceed the CPU-compatible index range
        // is rejected by Rust without passing undersized buffers to native GPU code.
        assert!(native_ht_tile_count((c_int::MAX as usize / 256 + 1) * 256).is_err());
        // SAFETY: an empty native call performs no accesses or device work.
        let result = unsafe {
            encode(
                input.as_ptr(),
                codebook.as_ptr(),
                output.as_mut_ptr(),
                0,
                1,
                4,
            )
        };
        assert_eq!(result, 0);
        assert_eq!(output, [0x55; 256]);
        for (count, beam, bits) in [(-1, 1, 4), (1, 0, 4), (1, 257, 4), (1, 1, 2), (1, 1, 5)] {
            // SAFETY: invalid scalar parameters must be rejected before any
            // buffer access; the valid-count cases have complete host buffers.
            let result = unsafe {
                encode(
                    input.as_ptr(),
                    codebook.as_ptr(),
                    output.as_mut_ptr(),
                    count,
                    beam,
                    bits,
                )
            };
            assert_ne!(result, 0, "count={count} beam={beam} bits={bits}");
            assert!(!gpu_last_error(backend).is_empty());
            assert_eq!(output, [0x55; 256]);
        }
        // SAFETY: null pointers are an explicitly rejected ABI input.
        let result = unsafe {
            encode(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                1,
                1,
                4,
            )
        };
        assert_ne!(result, 0);
        assert!(!gpu_last_error(backend).is_empty());
        // These nonnull addresses cannot contain even one tile/codebook.
        // The native ABI checks address-plus-extent overflow before device work.
        // Unlike a large valid count paired with a tiny slice, these addresses
        // are provably rejected.
        for (tiles, book, codes) in [
            (
                usize::MAX as *const f32,
                codebook.as_ptr(),
                output.as_mut_ptr(),
            ),
            (
                input.as_ptr(),
                usize::MAX as *const f32,
                output.as_mut_ptr(),
            ),
            (input.as_ptr(), codebook.as_ptr(), usize::MAX as *mut u8),
        ] {
            // SAFETY: one buffer address overflows even a single-tile extent;
            // native rejects it before accessing any pointer or creating a device.
            let result = unsafe { encode(tiles, book, codes, 1, 1, 4) };
            assert_ne!(result, 0);
            assert!(gpu_last_error(backend).contains("overflows the host address space"));
            assert_eq!(output, [0x55; 256]);
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_native_rejects_invalid_parameters_without_a_device() {
        native_rejects_invalid_parameters_without_a_device(HtBackend::Hip);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_native_rejects_invalid_parameters_without_a_device() {
        native_rejects_invalid_parameters_without_a_device(HtBackend::Cuda);
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn native_symbols(
        backend: HtBackend,
        tiles: &[f32],
        codebook: &[f32],
        beam: usize,
        bits: usize,
    ) -> Vec<u8> {
        let count = native_ht_tile_count(tiles.len()).unwrap();
        assert_eq!(codebook.len(), 65536);
        let mut symbols = vec![0x55; tiles.len()];
        // SAFETY: checked complete tile buffers, full state codebook, exclusive
        // output and bounded ABI scalars; both implementations synchronize.
        let result = unsafe {
            match backend {
                HtBackend::Cpu => ht_encode_bits(
                    tiles.as_ptr(),
                    codebook.as_ptr(),
                    symbols.as_mut_ptr(),
                    count,
                    1,
                    beam as c_int,
                    bits as c_int,
                ),
                gpu => gpu_encoder(gpu)(
                    tiles.as_ptr(),
                    codebook.as_ptr(),
                    symbols.as_mut_ptr(),
                    count,
                    beam as c_int,
                    bits as c_int,
                ),
            }
        };
        assert_eq!(
            result,
            0,
            "backend={backend:?} beam={beam} bits={bits}: {}",
            gpu_last_error(backend)
        );
        assert!(symbols
            .iter()
            .all(|&symbol| usize::from(symbol) < 1 << bits));
        symbols
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn tile_corpus(bits: usize) -> Vec<f32> {
        let mut tiles = Vec::new();
        tiles.extend((0..256).map(|i| ((i * 73 % 509) as f32 - 254.0) / 128.0));
        tiles.extend((0..256).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }));
        tiles.extend((0..256).map(|i| {
            [
                0.0,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
                1.0e-20,
                -1.0e-20,
                1.0e-19,
                -1.0e-19,
            ][i % 8]
        }));
        tiles.extend((0..256).map(|i| {
            [
                -0.5,
                f32::from_bits(0xbf000001),
                -0.25,
                0.25,
                0.5,
                f32::from_bits(0x3f000001),
            ][i % 6]
        }));
        tiles.extend([-3.453125; 256]);
        // Decode a closed nontrivial byte stream as an exact cyclic target.
        // HT3 exercises the partial-symbol 13-bit boundary closure.
        let packed: Vec<u8> = (0..8 * bits)
            .flat_map(|i| (0x6958e34bu32.wrapping_mul(i as u32 + 1)).to_le_bytes())
            .collect();
        let mut closed = [0.0; 256];
        decode_ht_tile(&packed, bits, &mut closed);
        tiles.extend(closed);
        tiles
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn native_corpus_matches_cpu_ht3_ht4_and_closure(backend: HtBackend) {
        backend.preflight().unwrap();
        let mut inject_prior_runtime_failure = true;
        for bits in [3, 4] {
            let tiles = tile_corpus(bits);
            for beam in [1, 7, 8, 128, 256] {
                let cpu = native_symbols(HtBackend::Cpu, &tiles, ht_codebook(), beam, bits);
                if inject_prior_runtime_failure {
                    // SAFETY: each runtime accepts a scalar device ordinal and rejects
                    // a negative ordinal without changing the current device.
                    // Leave that reported runtime error pending deliberately.
                    let result = unsafe {
                        match backend {
                            #[cfg(feature = "hip")]
                            HtBackend::Hip => hipSetDevice(-1),
                            #[cfg(feature = "cuda")]
                            HtBackend::Cuda => cudaSetDevice(-1),
                            _ => unreachable!("GPU test requires its feature"),
                        }
                    };
                    assert_ne!(result, 0);
                    inject_prior_runtime_failure = false;
                }
                let gpu = native_symbols(backend, &tiles, ht_codebook(), beam, bits);
                assert_eq!(gpu, cpu, "raw symbols: bits={bits} beam={beam}");
                assert_eq!(pack_ht_symbols(&gpu, bits), pack_ht_symbols(&cpu, bits));
            }
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn every_supported_beam_matches_cpu(backend: HtBackend) {
        backend.preflight().unwrap();
        for bits in [3, 4] {
            let tiles = tile_corpus(bits);
            for beam in 1..=256 {
                assert_eq!(
                    native_symbols(backend, &tiles, ht_codebook(), beam, bits),
                    native_symbols(HtBackend::Cpu, &tiles, ht_codebook(), beam, bits),
                    "bits={bits} beam={beam}"
                );
            }
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn native_batches_and_changed_codebooks_match_cpu(backend: HtBackend) {
        backend.preflight().unwrap();
        // Cross the 16,384-tile launch batch; this also crosses the
        // persistent-grid stride (fewer resident blocks than tiles).
        let tiles_per_call = 16385;
        let tiles: Vec<f32> = (0..tiles_per_call * 256)
            .map(|i| ((i * 17 % 97) as f32 - 48.0) / 32.0)
            .collect();
        let mut codebook = ht_codebook().to_vec();
        for bits in [3, 4] {
            assert_eq!(
                native_symbols(backend, &tiles, &codebook, 8, bits),
                native_symbols(HtBackend::Cpu, &tiles, &codebook, 8, bits)
            );
            // Modify the same allocation: pointer identity is not a cache key.
            codebook.iter_mut().for_each(|value| *value = -*value);
            assert_eq!(
                native_symbols(backend, &tiles, &codebook, 8, bits),
                native_symbols(HtBackend::Cpu, &tiles, &codebook, 8, bits)
            );
            let tiny_tiles = vec![1.0e-20; 256];
            codebook
                .iter_mut()
                .enumerate()
                .for_each(|(i, value)| *value = (i % 17) as f32 * 1.0e-21);
            assert_eq!(
                native_symbols(backend, &tiny_tiles, &codebook, 128, bits),
                native_symbols(HtBackend::Cpu, &tiny_tiles, &codebook, 128, bits)
            );
            codebook.copy_from_slice(ht_codebook());
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn public_bands_and_pairs_match_cpu(backend: HtBackend) {
        backend.preflight().unwrap();
        let cols = 128;
        let su = generate_signs(7, "hip-test-su", cols);
        let gate: Vec<f32> = (0..128 * cols)
            .map(|i| ((i * 13 % 61) as f32 - 30.0) / 32.0)
            .collect();
        let up: Vec<f32> = (0..128 * cols)
            .map(|i| ((i * 29 % 71) as f32 - 35.0) / 64.0)
            .collect();
        let gate_signs = generate_signs(11, "hip-test-gate", 128);
        let up_signs = generate_signs(19, "hip-test-up", 128);
        let gate_sv = ht_row_scales(&gate, 128, cols, &gate_signs).unwrap();
        let up_sv = ht_row_scales(&up, 128, cols, &up_signs).unwrap();
        for encoding in [Encoding::Ht, Encoding::Ht3] {
            for beam in [1, 8, 128, 256] {
                let cpu = QuantMeta {
                    su: &su,
                    sv: &gate_sv,
                    ht_beam: beam,
                    ..Default::default()
                };
                let cpu_up = QuantMeta { sv: &up_sv, ..cpu };
                let gpu = QuantMeta {
                    ht_backend: backend,
                    ..cpu
                };
                let gpu_up = QuantMeta { sv: &up_sv, ..gpu };
                assert_eq!(
                    encode_rows(encoding, &gate, 128, cols, &gpu).unwrap(),
                    encode_rows(encoding, &gate, 128, cols, &cpu).unwrap()
                );
                assert_eq!(
                    encode_ht_gate_up_bands(encoding, &gate, &up, cols, &gpu, &gpu_up).unwrap(),
                    encode_ht_gate_up_bands(encoding, &gate, &up, cols, &cpu, &cpu_up).unwrap()
                );
            }
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn native_error_recovery_matches_cpu(backend: HtBackend) {
        backend.preflight().unwrap();
        let encode = gpu_encoder(backend);
        let tiles = tile_corpus(3);
        let bad_tile = [f32::NAN; 256];
        let mut output = [0x55; 256];
        for bits in [3, 4] {
            // SAFETY: one complete tile, a full codebook and exclusive output.
            // Native reports an empty finite survivor set rather than retaining
            // any caller pointer; this is not a fatal runtime/device fault.
            let result = unsafe {
                encode(
                    bad_tile.as_ptr(),
                    ht_codebook().as_ptr(),
                    output.as_mut_ptr(),
                    1,
                    8,
                    bits as c_int,
                )
            };
            assert_eq!(result, -4);
            assert!(!gpu_last_error(backend).is_empty());
            assert_eq!(
                native_symbols(backend, &tiles, ht_codebook(), 8, bits),
                native_symbols(HtBackend::Cpu, &tiles, ht_codebook(), 8, bits)
            );
        }
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn concurrent_callers_and_codebook_changes_match_cpu(backend: HtBackend) {
        backend.preflight().unwrap();
        let barrier = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut codebook = ht_codebook().to_vec();
                    let beam = [1, 7, 128, 256][worker];
                    for bits in [3, 4] {
                        let tiles = tile_corpus(bits);
                        let sign = if worker % 2 == 0 { 1.0 } else { -1.0 };
                        let scale = (worker + 1) as f32 / 4.0;
                        for (value, &original) in codebook.iter_mut().zip(ht_codebook()) {
                            *value = sign * scale * original;
                        }
                        let expected =
                            native_symbols(HtBackend::Cpu, &tiles, &codebook, beam, bits);
                        // Synchronize the first call only: a failing worker
                        // must not strand its peers at a later barrier.
                        if bits == 3 {
                            barrier.wait();
                        }
                        assert_eq!(
                            native_symbols(backend, &tiles, &codebook, beam, bits),
                            expected,
                            "worker={worker} bits={bits} beam={beam}"
                        );
                    }
                });
            }
        });
    }

    macro_rules! gpu_device_tests {
        ($feature:literal, $backend:expr,
         $corpus:ident, $beams:ident, $batches:ident, $bands:ident,
         $recovery:ident, $concurrent:ident) => {
            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; exact CPU/GPU corpus comparison"]
            fn $corpus() {
                native_corpus_matches_cpu_ht3_ht4_and_closure($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; full corpus at every beam 1..=256"]
            fn $beams() {
                every_supported_beam_matches_cpu($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; content caches and bounded batch boundary"]
            fn $batches() {
                native_batches_and_changed_codebooks_match_cpu($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; public rotations and paired-band path"]
            fn $bands() {
                public_bands_and_pairs_match_cpu($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; recover after no finite survivors"]
            fn $recovery() {
                native_error_recovery_matches_cpu($backend);
            }

            #[cfg(feature = $feature)]
            #[test]
            #[ignore = "requires the selected GPU; simultaneous callers use distinct codebooks"]
            fn $concurrent() {
                concurrent_callers_and_codebook_changes_match_cpu($backend);
            }
        };
    }

    gpu_device_tests!(
        "hip",
        HtBackend::Hip,
        hip_native_corpus_matches_cpu_ht3_ht4_and_closure,
        hip_every_supported_beam_matches_cpu,
        hip_native_batches_and_changed_codebooks_match_cpu,
        hip_public_bands_and_pairs_match_cpu,
        hip_native_error_recovery_matches_cpu,
        hip_concurrent_callers_and_codebook_changes_match_cpu
    );
    gpu_device_tests!(
        "cuda",
        HtBackend::Cuda,
        cuda_native_corpus_matches_cpu_ht3_ht4_and_closure,
        cuda_every_supported_beam_matches_cpu,
        cuda_native_batches_and_changed_codebooks_match_cpu,
        cuda_public_bands_and_pairs_match_cpu,
        cuda_native_error_recovery_matches_cpu,
        cuda_concurrent_callers_and_codebook_changes_match_cpu
    );
}
