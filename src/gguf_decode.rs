//! GGML little-endian block decoding into the canonical F32 source stream.
//!
//! Layouts: hgn-spec 1.1.0 GGUF-IMPORT.md, commit
//! 79e75fd4ae7642478823525b0a3212951f584af5. Unlike Halogen's direct-store
//! importer, this decoder does not round IQ scales or affine values to F16/BF16.

use anyhow::{bail, ensure, Result};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq4Xs = 23,
    Bf16 = 30,
}

impl GgmlType {
    pub(crate) fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            23 => Self::Iq4Xs,
            30 => Self::Bf16,
            _ => bail!("unsupported GGUF tensor type {code}; supported: F32 F16 BF16 Q4_0 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K IQ4_NL IQ3_S IQ4_XS"),
        })
    }

    /// (elements, bytes), with no padding between consecutive blocks.
    pub(crate) fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Iq3S => (256, 110),
            Self::Iq4Xs => (256, 136),
        }
    }
}

const IQ4_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
include!("gguf_iq3_grid.rs");

fn half(bytes: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}

pub(crate) fn decode(ty: GgmlType, bytes: &[u8], output: &mut [f32]) -> Result<()> {
    let (elements, size) = ty.block();
    ensure!(
        bytes.len() % size == 0
            && output.len() % elements == 0
            && bytes.len() / size == output.len() / elements,
        "invalid GGUF decode block length"
    );
    for (block, values) in bytes
        .chunks_exact(size)
        .zip(output.chunks_exact_mut(elements))
    {
        match ty {
            GgmlType::F32 => values[0] = f32::from_le_bytes(block.try_into().unwrap()),
            GgmlType::F16 => values[0] = half(block, 0),
            GgmlType::Bf16 => {
                values[0] =
                    f32::from_bits(u32::from(u16::from_le_bytes(block.try_into().unwrap())) << 16)
            }
            GgmlType::Q4_0 | GgmlType::Iq4Nl => {
                let d = half(block, 0);
                for j in 0..32 {
                    let q = (block[2 + j % 16] >> (4 * (j / 16))) & 15;
                    let level = if ty == GgmlType::Q4_0 {
                        i32::from(q) - 8
                    } else {
                        i32::from(IQ4_LEVELS[q as usize])
                    };
                    values[j] = d * level as f32;
                }
            }
            GgmlType::Q5_0 | GgmlType::Q5_1 => {
                let d = half(block, 0);
                let affine = ty == GgmlType::Q5_1;
                let high_at = if affine { 4 } else { 2 };
                let high = u32::from_le_bytes(block[high_at..high_at + 4].try_into().unwrap());
                for j in 0..32 {
                    let low = (block[high_at + 4 + j % 16] >> (4 * (j / 16))) & 15;
                    let q = u32::from(low) | (((high >> j) & 1) << 4);
                    values[j] = if affine {
                        d * q as f32 + half(block, 2)
                    } else {
                        d * (q as i32 - 16) as f32
                    };
                }
            }
            GgmlType::Q8_0 => {
                let d = half(block, 0);
                for j in 0..32 {
                    values[j] = d * (block[2 + j] as i8) as f32;
                }
            }
            GgmlType::Q4K | GgmlType::Q5K => {
                let d = half(block, 0);
                let dmin = half(block, 2);
                let scales = &block[4..16];
                let five = ty == GgmlType::Q5K;
                let qs = &block[if five { 48 } else { 16 }..];
                for group in 0..8 {
                    let (sc, min) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    let scale = d * f32::from(sc);
                    let minimum = dmin * f32::from(min);
                    for j in 0..32 {
                        let mut q = (qs[32 * (group / 2) + j] >> (4 * (group % 2))) & 15;
                        if five {
                            q |= ((block[16 + j] >> group) & 1) << 4;
                        }
                        values[group * 32 + j] = scale * f32::from(q) - minimum;
                    }
                }
            }
            GgmlType::Q6K => {
                let d = half(block, 208);
                for j in 0..256 {
                    let half_block = j / 128;
                    let lane = j % 32;
                    let quarter = (j % 128) / 32;
                    let low = (block[half_block * 64 + (quarter % 2) * 32 + lane]
                        >> (4 * (quarter / 2)))
                        & 15;
                    let high = (block[128 + half_block * 32 + lane] >> (2 * quarter)) & 3;
                    let q = i32::from(low | (high << 4)) - 32;
                    let scale = (block[192 + j / 16] as i8) as f32;
                    values[j] = (d * scale) * q as f32;
                }
            }
            GgmlType::Iq4Xs => {
                let d = half(block, 0);
                let high = u16::from_le_bytes([block[2], block[3]]);
                for group in 0..8 {
                    let ls = u16::from((block[4 + group / 2] >> (4 * (group % 2))) & 15)
                        | (((high >> (2 * group)) & 3) << 4);
                    let scale = d * (i32::from(ls) - 32) as f32;
                    for j in 0..32 {
                        let q = (block[8 + group * 16 + j % 16] >> (4 * (j / 16))) & 15;
                        values[group * 32 + j] = scale * f32::from(IQ4_LEVELS[q as usize]);
                    }
                }
            }
            GgmlType::Iq3S => {
                let d = half(block, 0);
                for group in 0..8 {
                    let ls = (block[106 + group / 2] >> (4 * (group % 2))) & 15;
                    let scale = d * f32::from(2 * ls + 1);
                    for word in 0..8 {
                        let index = usize::from(block[2 + 8 * group + word])
                            | (usize::from((block[66 + group] >> word) & 1) << 8);
                        let grid = IQ3_GRID[index];
                        let signs = block[74 + group * 4 + word / 2];
                        for byte in 0..4 {
                            let level = ((grid >> (8 * byte)) & 255) as f32;
                            let value = scale * level;
                            values[32 * group + 4 * word + byte] =
                                if signs & (1 << (4 * (word % 2) + byte)) != 0 {
                                    -value
                                } else {
                                    value
                                };
                        }
                    }
                }
            }
        }
    }
    ensure!(
        output.iter().all(|v| v.is_finite()),
        "non-finite GGUF decoded value"
    );
    Ok(())
}

#[cfg(test)]
#[path = "gguf_decode_tests.rs"]
mod tests;
