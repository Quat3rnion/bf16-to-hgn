//! GGML little-endian block decoding into the canonical F32 source stream.
//!
//! Layouts: hgn-spec 1.1.0 GGUF-IMPORT.md, commit
//! 79e75fd4ae7642478823525b0a3212951f584af5, plus UD/I/K types aligned with
//! ggml-org/llama.cpp ggml-quants.c. Unlike Halogen's direct-store importer,
//! this decoder does not round IQ scales or affine values to F16/BF16.

use anyhow::{bail, ensure, Result};
use half::f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    Iq1M = 29,
    Bf16 = 30,
    Q2_0 = 42,
}

impl GgmlType {
    pub(crate) fn from_code(code: u32) -> Result<Self> {
        Ok(match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            42 => Self::Q2_0,
            _ => bail!(
                "unsupported GGUF tensor type {code}; supported: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ1_S IQ4_NL IQ3_S IQ2_S IQ4_XS IQ1_M Q2_0"
            ),
        })
    }

    /// (elements, bytes), with no padding between consecutive blocks.
    pub(crate) fn block(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::Bf16 => (1, 2),
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q2_0 => (64, 18),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq2S => (256, 82),
            Self::Iq3S => (256, 110),
            Self::Iq4Xs => (256, 136),
            Self::Iq1M => (256, 56),
        }
    }
}

const IQ4_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const IQ1_DELTA: f32 = 0.125;

include!("gguf_iq3_grid.rs");
include!("gguf_iq_tables.rs");

fn half(bytes: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
}

fn grid_u64_bytes(entry: u64) -> [u8; 8] {
    entry.to_le_bytes()
}

fn grid_u32_bytes(entry: u32) -> [u8; 4] {
    entry.to_le_bytes()
}

fn iq_sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
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
            GgmlType::Q4_1 => {
                let d = half(block, 0);
                let m = half(block, 2);
                for j in 0..16 {
                    let x0 = block[4 + j] & 15;
                    let x1 = block[4 + j] >> 4;
                    values[j] = f32::from(x0) * d + m;
                    values[j + 16] = f32::from(x1) * d + m;
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
            GgmlType::Q2_0 => {
                let d = half(block, 0);
                for j in 0..64 {
                    let q = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
                    values[j] = d * (i32::from(q) - 1) as f32;
                }
            }
            GgmlType::Q2K => {
                // scales[16] + qs[64] + d + dmin
                let d = half(block, 80);
                let dmin = half(block, 82);
                let scales = &block[..16];
                let qs = &block[16..80];
                let mut out = 0usize;
                let mut is = 0usize;
                for n in (0..256).step_by(128) {
                    let q = &qs[n / 4..n / 4 + 32];
                    let mut shift = 0u8;
                    for _ in 0..4 {
                        let sc = scales[is];
                        is += 1;
                        let dl = d * f32::from(sc & 15);
                        let ml = dmin * f32::from(sc >> 4);
                        for l in 0..16 {
                            values[out] = dl * f32::from((q[l] >> shift) & 3) - ml;
                            out += 1;
                        }
                        let sc = scales[is];
                        is += 1;
                        let dl = d * f32::from(sc & 15);
                        let ml = dmin * f32::from(sc >> 4);
                        for l in 0..16 {
                            values[out] = dl * f32::from((q[l + 16] >> shift) & 3) - ml;
                            out += 1;
                        }
                        shift += 2;
                    }
                }
            }
            GgmlType::Q3K => {
                // hmask[32] + qs[64] + scales[12] + d
                let d_all = half(block, 108);
                let hmask = &block[..32];
                let qs = &block[32..96];
                let mut aux = [0u32; 4];
                for (i, chunk) in block[96..108].chunks_exact(4).enumerate() {
                    aux[i] = u32::from_le_bytes(chunk.try_into().unwrap());
                }
                let kmask1 = 0x0303_0303u32;
                let kmask2 = 0x0f0f_0f0fu32;
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                let scales = bytemuck_scales(&aux);
                let mut out = 0usize;
                let mut is = 0usize;
                let mut m = 1u8;
                for n in (0..256).step_by(128) {
                    let q = &qs[n / 4..n / 4 + 32];
                    let mut shift = 0u8;
                    for _ in 0..4 {
                        let dl = d_all * f32::from(i16::from(scales[is]) - 32);
                        is += 1;
                        for l in 0..16 {
                            let qv = i32::from((q[l] >> shift) & 3)
                                - if hmask[l] & m != 0 { 0 } else { 4 };
                            values[out] = dl * qv as f32;
                            out += 1;
                        }
                        let dl = d_all * f32::from(i16::from(scales[is]) - 32);
                        is += 1;
                        for l in 0..16 {
                            let qv = i32::from((q[l + 16] >> shift) & 3)
                                - if hmask[l + 16] & m != 0 { 0 } else { 4 };
                            values[out] = dl * qv as f32;
                            out += 1;
                        }
                        shift += 2;
                        m <<= 1;
                    }
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
            GgmlType::Iq3Xxs => {
                let d = half(block, 0);
                let qs = &block[2..66];
                let scales_and_signs = &block[66..98];
                let mut out = 0usize;
                for ib32 in 0..8 {
                    let aux32 = u32::from_le_bytes(
                        scales_and_signs[4 * ib32..4 * ib32 + 4]
                            .try_into()
                            .unwrap(),
                    );
                    let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                    let q = &qs[8 * ib32..8 * ib32 + 8];
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        let g1 = grid_u32_bytes(IQ3_XXS_GRID[usize::from(q[2 * l])]);
                        let g2 = grid_u32_bytes(IQ3_XXS_GRID[usize::from(q[2 * l + 1])]);
                        for j in 0..4 {
                            values[out + j] = db * f32::from(g1[j]) * iq_sign(signs, j);
                            values[out + j + 4] = db * f32::from(g2[j]) * iq_sign(signs, j + 4);
                        }
                        out += 8;
                    }
                }
            }
            GgmlType::Iq2Xxs => {
                let d = half(block, 0);
                let qs = &block[2..66];
                let mut out = 0usize;
                for ib32 in 0..8 {
                    let aux0 = u32::from_le_bytes(qs[8 * ib32..8 * ib32 + 4].try_into().unwrap());
                    let aux1 =
                        u32::from_le_bytes(qs[8 * ib32 + 4..8 * ib32 + 8].try_into().unwrap());
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    let aux8 = aux0.to_le_bytes();
                    for l in 0..4 {
                        let grid = grid_u64_bytes(IQ2_XXS_GRID[usize::from(aux8[l])]);
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            values[out + j] = db * f32::from(grid[j]) * iq_sign(signs, j);
                        }
                        out += 8;
                    }
                }
            }
            GgmlType::Iq2Xs => {
                let d = half(block, 0);
                let qs = &block[2..66];
                let scales = &block[66..74];
                let mut out = 0usize;
                for ib32 in 0..8 {
                    let db0 = d * (0.5 + f32::from(scales[ib32] & 15)) * 0.25;
                    let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
                    for l in 0..4 {
                        let q = u16::from_le_bytes(
                            qs[8 * ib32 + 2 * l..8 * ib32 + 2 * l + 2]
                                .try_into()
                                .unwrap(),
                        );
                        let grid = grid_u64_bytes(IQ2_XS_GRID[usize::from(q & 511)]);
                        let signs = KSIGNS_IQ2XS[usize::from(q >> 9)];
                        let db = if l < 2 { db0 } else { db1 };
                        for j in 0..8 {
                            values[out + j] = db * f32::from(grid[j]) * iq_sign(signs, j);
                        }
                        out += 8;
                    }
                }
            }
            GgmlType::Iq2S => {
                let d = half(block, 0);
                let qs = &block[2..34];
                let signs = &block[34..66];
                let qh = &block[66..74];
                let scales = &block[74..82];
                let mut out = 0usize;
                for ib32 in 0..8 {
                    let db0 = d * (0.5 + f32::from(scales[ib32] & 15)) * 0.25;
                    let db1 = d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25;
                    for l in 0..4 {
                        let index = usize::from(qs[4 * ib32 + l])
                            | ((((usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300) as usize));
                        let grid = grid_u64_bytes(IQ2_S_GRID[index]);
                        let db = if l < 2 { db0 } else { db1 };
                        let s = signs[4 * ib32 + l];
                        for j in 0..8 {
                            values[out + j] = db * f32::from(grid[j]) * iq_sign(s, j);
                        }
                        out += 8;
                    }
                }
            }
            GgmlType::Iq1S => {
                let d = half(block, 0);
                let qs = &block[2..34];
                let mut out = 0usize;
                for ib in 0..8 {
                    let qh = u16::from_le_bytes(block[34 + 2 * ib..36 + 2 * ib].try_into().unwrap());
                    let dl = d * (2.0 * f32::from((qh >> 12) & 7) + 1.0);
                    let delta = if qh & 0x8000 != 0 {
                        -IQ1_DELTA
                    } else {
                        IQ1_DELTA
                    };
                    for l in 0..4 {
                        let index = usize::from(qs[4 * ib + l])
                            | ((usize::from((qh >> (3 * l)) & 7)) << 8);
                        let grid = grid_u64_bytes(IQ1_S_GRID[index]);
                        for j in 0..8 {
                            // iq1s_grid stores int8 levels in each byte
                            let level = grid[j] as i8 as f32;
                            values[out + j] = dl * (level + delta);
                        }
                        out += 8;
                    }
                }
            }
            GgmlType::Iq1M => {
                let qs = &block[..32];
                let qh = &block[32..48];
                let scales = &block[48..56];
                let sc = [
                    u16::from_le_bytes(scales[0..2].try_into().unwrap()),
                    u16::from_le_bytes(scales[2..4].try_into().unwrap()),
                    u16::from_le_bytes(scales[4..6].try_into().unwrap()),
                    u16::from_le_bytes(scales[6..8].try_into().unwrap()),
                ];
                let scale_bits =
                    (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                let d = f16::from_bits(scale_bits).to_f32();
                let mut out = 0usize;
                for ib in 0..8 {
                    let sc_word = sc[ib / 2];
                    let dl1 = d * (2.0 * f32::from((sc_word >> (6 * (ib % 2))) & 7) + 1.0);
                    let dl2 = d * (2.0 * f32::from((sc_word >> (6 * (ib % 2) + 3)) & 7) + 1.0);
                    let qh0 = qh[2 * ib];
                    let qh1 = qh[2 * ib + 1];
                    let idx = [
                        u16::from(qs[4 * ib]) | (u16::from(qh0) << 8 & 0x700),
                        u16::from(qs[4 * ib + 1]) | (u16::from(qh0) << 4 & 0x700),
                        u16::from(qs[4 * ib + 2]) | (u16::from(qh1) << 8 & 0x700),
                        u16::from(qs[4 * ib + 3]) | (u16::from(qh1) << 4 & 0x700),
                    ];
                    let deltas = [
                        if qh0 & 0x08 != 0 {
                            -IQ1_DELTA
                        } else {
                            IQ1_DELTA
                        },
                        if qh0 & 0x80 != 0 {
                            -IQ1_DELTA
                        } else {
                            IQ1_DELTA
                        },
                        if qh1 & 0x08 != 0 {
                            -IQ1_DELTA
                        } else {
                            IQ1_DELTA
                        },
                        if qh1 & 0x80 != 0 {
                            -IQ1_DELTA
                        } else {
                            IQ1_DELTA
                        },
                    ];
                    for l in 0..2 {
                        let grid = grid_u64_bytes(IQ1_S_GRID[usize::from(idx[l])]);
                        for j in 0..8 {
                            values[out + j] = dl1 * (grid[j] as i8 as f32 + deltas[l]);
                        }
                        out += 8;
                    }
                    for l in 2..4 {
                        let grid = grid_u64_bytes(IQ1_S_GRID[usize::from(idx[l])]);
                        for j in 0..8 {
                            values[out + j] = dl2 * (grid[j] as i8 as f32 + deltas[l]);
                        }
                        out += 8;
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

fn bytemuck_scales(aux: &[u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, word) in aux.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
#[path = "gguf_decode_tests.rs"]
mod tests;
