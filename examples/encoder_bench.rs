//! Identical-source baseline/candidate benchmark; no private codec APIs.
//!
//! Build into a separate CARGO_TARGET_DIR, then run the executable directly.
//! Example arguments: --cases ht4,ht3 --cols 2560 --threads 1,2
//! --iterations 1 --repetitions 3. Use --suite corpus for finite edge cases.
//! Real HF example: --hf /snapshot --tensor layers.0.mlp.down_proj.weight
//! --cases ht4,ht3,i4r,nf4 --row-start 0. Paired real inputs additionally
//! require --up-tensor and optionally --up-row-start (640 for fused gate/up).
//!
//! JSONL sample elapsed_ns covers only public encoding calls and insertion
//! into a preallocated result vector. It includes validation, codec allocation,
//! and rotation. Source reads/generation, sign/RMS preparation, warmup, hashing,
//! printing, result-vector allocation, and output destruction are excluded.
//! All returned code/scale bytes, padding, codebook/global scale, and rotation
//! side planes are fingerprinted; every iteration, thread count and backend must agree.
//! GPU preflight durations are recorded separately per selected backend.
//! Only HT3/HT4 use the selected GPU; every other encoding remains CPU.

use anyhow::{bail, ensure, Context, Result};
use bf16_to_hgn::codec::{
    encode_ht_gate_up_bands, encode_rows, fp8_decode, generate_signs, ht_row_scales,
    payload_layout, EncodedBlock, Encoding, HtBackend, QuantMeta, NF4_CODEBOOK,
};
use bf16_to_hgn::hf::{Dtype, Snapshot};
use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

const ROWS: usize = 128;
const MAX_BAND_VALUES: usize = 8 * 1024 * 1024;
const MAX_RETAINED_BYTES: u64 = 128 * 1024 * 1024;
const CONTRACT: &str = "encoder-bench-v2";
const SCHEMA: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Case {
    All,
    Ht4,
    Ht3,
    Ht4Pair,
    Ht3Pair,
    I4r,
    #[value(alias = "q4cp")]
    Nf4,
    #[value(alias = "affine6")]
    Q6g64,
    #[value(alias = "affine8")]
    Q8g64,
    #[value(alias = "fp8")]
    Fp8g128,
}

impl Case {
    const ALL: [Self; 9] = [
        Self::Ht4,
        Self::Ht3,
        Self::Ht4Pair,
        Self::Ht3Pair,
        Self::I4r,
        Self::Nf4,
        Self::Q6g64,
        Self::Q8g64,
        Self::Fp8g128,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Ht4 => "ht4",
            Self::Ht3 => "ht3",
            Self::Ht4Pair => "ht4-pair",
            Self::Ht3Pair => "ht3-pair",
            Self::I4r => "i4r",
            Self::Nf4 => "nf4",
            Self::Q6g64 => "q6g64",
            Self::Q8g64 => "q8g64",
            Self::Fp8g128 => "fp8g128",
        }
    }

    fn encoding(self) -> Encoding {
        match self {
            Self::Ht4 | Self::Ht4Pair => Encoding::Ht,
            Self::Ht3 | Self::Ht3Pair => Encoding::Ht3,
            Self::I4r => Encoding::I4r,
            Self::Nf4 => Encoding::Q4Cp,
            Self::Q6g64 => Encoding::Q6G64,
            Self::Q8g64 => Encoding::Q8G64,
            Self::Fp8g128 => Encoding::Fp8G128,
            Self::All => unreachable!("all is expanded before preparing a case"),
        }
    }

    fn paired(self) -> bool {
        matches!(self, Self::Ht4Pair | Self::Ht3Pair)
    }

    fn rotated(self) -> bool {
        self.encoding().ht_bits().is_some() || self == Self::I4r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Suite {
    Timing,
    Corpus,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Fixture {
    All,
    Ordinary,
    Zeros,
    Constant,
    Ties,
    Signed,
    Tiny,
    Trellis,
}

impl Fixture {
    const ALL: [Self; 7] = [
        Self::Ordinary,
        Self::Zeros,
        Self::Constant,
        Self::Ties,
        Self::Signed,
        Self::Tiny,
        Self::Trellis,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Ordinary => "ordinary-bf16",
            Self::Zeros => "signed-zeros",
            Self::Constant => "constant-groups",
            Self::Ties => "ties-and-neighbors",
            Self::Signed => "signed-extremes",
            Self::Tiny => "finite-subnormals",
            Self::Trellis => "constant-rotated-tiles",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Deterministic encoder timing and complete byte fingerprints (JSONL)")]
struct Args {
    #[arg(long, value_enum, default_value = "timing")]
    suite: Suite,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "all")]
    cases: Vec<Case>,
    /// Comma-separated list of 1 and/or 2; runs sequentially and checks equality.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    threads: Vec<usize>,
    /// HT backend axis; sequential runs must produce identical payload/side-plane hashes.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "cpu")]
    backends: Vec<HtBackend>,
    /// Timed calls per sample; retained outputs are limited to 128 MiB.
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    #[arg(long, alias = "samples", default_value_t = 3)]
    repetitions: usize,
    /// Untimed calls per case/thread/beam before timing samples.
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Synthetic timing geometry is one 128-row matrix band.
    #[arg(long, default_value_t = 2560)]
    cols: usize,
    /// FP8G128 timing uses n-gram geometry rather than --cols.
    #[arg(long, default_value_t = 8192)]
    fp8_rows: usize,
    #[arg(long, default_value_t = 123)]
    seed: u64,
    /// Corpus always uses one call/sample, no warmup and bounded small shapes.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "all")]
    fixtures: Vec<Fixture>,
    #[arg(long, value_delimiter = ',', default_value = "1,8,128")]
    corpus_beams: Vec<usize>,
    /// Timed HT beam width. CPU permits up to 4096; GPUs support 1..=256.
    #[arg(long, default_value_t = 128)]
    timing_beam: usize,
    /// Indexed HF snapshot; only metadata and the selected BF16 bands are read.
    #[arg(long)]
    hf: Option<PathBuf>,
    #[arg(long, requires = "hf")]
    tensor: Option<String>,
    #[arg(long, default_value_t = 0)]
    row_start: usize,
    #[arg(long, requires = "hf")]
    up_tensor: Option<String>,
    #[arg(long, default_value_t = 0)]
    up_row_start: usize,
    /// Caller-supplied baseline/candidate/run identifier, never in output hashes.
    #[arg(long)]
    label: Option<String>,
}

struct RealBands {
    cols: usize,
    input: Vec<f32>,
    up: Option<Vec<f32>>,
    source: Value,
    read_elapsed_ns: u64,
}

struct Prepared<'a> {
    case: Case,
    fixture: Fixture,
    rows: usize,
    cols: usize,
    input: Cow<'a, [f32]>,
    up: Option<Cow<'a, [f32]>>,
    su: Vec<f32>,
    sv: Vec<f32>,
    up_sv: Vec<f32>,
    fp8_scale: f32,
    source: Value,
    preparation: &'static str,
    setup_elapsed_ns: u64,
    preparation_elapsed_ns: u64,
}

fn ns(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).context("elapsed nanoseconds exceed u64")
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn float_digest(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn bounded_values(rows: usize, cols: usize) -> Result<usize> {
    let count = rows
        .checked_mul(cols)
        .context("band element count overflow")?;
    ensure!(
        rows > 0 && cols > 0 && count <= MAX_BAND_VALUES,
        "each input band must contain 1..={MAX_BAND_VALUES} values"
    );
    Ok(count)
}

fn read_band(
    snapshot: &Snapshot,
    name: &str,
    row_start: usize,
) -> Result<(usize, Vec<f32>, Value)> {
    let info = snapshot
        .tensors
        .get(name)
        .with_context(|| format!("unknown tensor {name}"))?;
    ensure!(
        info.dtype == Dtype::Bf16,
        "real band tensor {name} must be BF16"
    );
    ensure!(
        info.shape.len() >= 2,
        "real band tensor {name} must have rank >= 2"
    );
    let (total_rows, cols) = bf16_to_hgn::rows_cols(&info.shape)?;
    let matrix_rows = usize::try_from(info.shape[info.shape.len() - 2])?;
    let end = row_start.checked_add(ROWS).context("row range overflow")?;
    ensure!(
        row_start % ROWS == 0 && end <= total_rows,
        "real input needs an aligned, complete 128-row band within {name}"
    );
    ensure!(
        matrix_rows > 0 && row_start % matrix_rows + ROWS <= matrix_rows,
        "real band must not cross a matrix/expert boundary"
    );
    let count = bounded_values(ROWS, cols)?;
    let offset = row_start
        .checked_mul(cols)
        .context("element offset overflow")?;
    let values = snapshot.read_f32(name, u64::try_from(offset)?, count)?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "real BF16 band contains non-finite values"
    );
    let source = json!({
        "tensor": name, "tensor_shape": info.shape, "row_start": row_start,
        "rows_read": ROWS, "cols": cols, "dtype": "BF16",
        "payload_bytes_read": count * 2,
        "decoded_f32_sha256": float_digest(&values),
    });
    Ok((cols, values, source))
}

fn load_real(args: &Args, cases: &[Case]) -> Result<Option<RealBands>> {
    let Some(root) = &args.hf else {
        return Ok(None);
    };
    ensure!(
        args.suite != Suite::Corpus,
        "--hf applies to timing, not the synthetic correctness corpus"
    );
    ensure!(
        cases.iter().any(|case| *case != Case::Fp8g128),
        "--hf requires at least one matrix-band case; FP8 uses synthetic n-gram rows"
    );
    let name = args.tensor.as_deref().context("--hf requires --tensor")?;
    let start = Instant::now();
    let snapshot = Snapshot::open(root)?;
    let (cols, input, primary_source) = read_band(&snapshot, name, args.row_start)?;
    let (up, up_source) = if cases.iter().any(|case| case.paired()) {
        let up_name = args
            .up_tensor
            .as_deref()
            .context("paired real cases require --up-tensor")?;
        let (up_cols, up, source) = read_band(&snapshot, up_name, args.up_row_start)?;
        ensure!(
            cols == up_cols,
            "paired real bands must have the same column count"
        );
        (Some(up), Some(source))
    } else {
        ensure!(
            args.up_tensor.is_none(),
            "--up-tensor requires a paired case"
        );
        (None, None)
    };
    snapshot.check_unchanged()?;
    let read_elapsed_ns = ns(start)?;
    Ok(Some(RealBands {
        cols,
        input,
        up,
        source: json!({"kind": "hf-bf16-band", "root": root, "primary": primary_source, "up": up_source}),
        read_elapsed_ns,
    }))
}

fn adjacent(value: f32, up: bool) -> f32 {
    if value == 0.0 {
        return f32::from_bits(if up { 1 } else { 0x8000_0001 });
    }
    let bits = value.to_bits();
    f32::from_bits(if up == (value > 0.0) {
        bits + 1
    } else {
        bits - 1
    })
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn i4r_rounding_fixture(fixture: Fixture) -> Vec<f32> {
    // Specify equal-magnitude signed extrema and RNE ties in the rotated basis.
    // H is symmetric and H*H=128*I: H*target*H/128 is the original source.
    // This independent f64 fixture construction is entirely outside encoding;
    // the final original-basis values are rounded to f32, just like real input.
    let mut values = Vec::with_capacity(ROWS * ROWS);
    for row in 0..ROWS {
        for col in 0..ROWS {
            let value = match col {
                0 => {
                    if row % 2 == 0 {
                        -8.0
                    } else {
                        8.0
                    }
                }
                1 => {
                    if row % 2 == 0 {
                        8.0
                    } else {
                        -8.0
                    }
                }
                _ if fixture == Fixture::Ties => {
                    let middle = ((col - 2) % 16) as f32 - 7.5;
                    match (row / 2) % 3 {
                        0 => adjacent(middle, false),
                        1 => middle,
                        _ => adjacent(middle, true),
                    }
                }
                _ => ((5 * col + 3) % 16) as f32 - 8.0,
            };
            values.push((value * 0.03125) as f64);
        }
    }
    let mut width = 1;
    while width < ROWS {
        for row in 0..ROWS {
            for start in (0..ROWS).step_by(2 * width) {
                for offset in 0..width {
                    let first = row * ROWS + start + offset;
                    let second = first + width;
                    let a = values[first];
                    let b = values[second];
                    values[first] = a + b;
                    values[second] = a - b;
                }
            }
        }
        for col in 0..ROWS {
            for start in (0..ROWS).step_by(2 * width) {
                for offset in 0..width {
                    let first = (start + offset) * ROWS + col;
                    let second = first + width * ROWS;
                    let a = values[first];
                    let b = values[second];
                    values[first] = a + b;
                    values[second] = a - b;
                }
            }
        }
        width *= 2;
    }
    values
        .into_iter()
        .map(|value| (value / ROWS as f64) as f32)
        .collect()
}

fn synthetic(
    case: Case,
    fixture: Fixture,
    rows: usize,
    cols: usize,
    seed: u64,
    up: bool,
) -> Result<Vec<f32>> {
    let count = bounded_values(rows, cols)?;
    if case == Case::I4r && matches!(fixture, Fixture::Ties | Fixture::Signed) {
        ensure!(
            rows == ROWS && cols == ROWS,
            "I4R rounding corpus requires one 128x128 tile"
        );
        return Ok(i4r_rounding_fixture(fixture));
    }
    let mut state = seed
        ^ if up {
            0xd1b54a32d192ed03
        } else {
            0x94d049bb133111eb
        };
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let row = index / cols;
        let col = index % cols;
        let value = match fixture {
            Fixture::Ordinary => {
                // Integer-to-float conversion and dyadic scaling are exact; no libm/RNG dependency.
                let word = splitmix(&mut state);
                let centered = ((word >> 48) as i32) - 32768;
                let amplitude = match row % 4 {
                    0 => 0.015625,
                    1 => 0.03125,
                    2 => 0.0625,
                    _ => 0.125,
                };
                bf16::from_f32(centered as f32 * (1.0 / 32768.0) * amplitude).to_f32()
            }
            Fixture::Zeros => {
                if (index + usize::from(up)) % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
            Fixture::Constant => {
                if (row + usize::from(up)) % 2 == 0 {
                    0.03125
                } else {
                    -0.03125
                }
            }
            Fixture::Ties => match case {
                Case::Nf4 => {
                    let slot = col % 32;
                    if slot == 0 {
                        -1.0
                    } else if slot == 1 {
                        1.0
                    } else {
                        let middle = (NF4_CODEBOOK[(slot - 2) % 15] as f64 * 0.5
                            + NF4_CODEBOOK[(slot - 2) % 15 + 1] as f64 * 0.5)
                            as f32;
                        match (row + slot / 15) % 3 {
                            0 => adjacent(middle, false),
                            1 => middle,
                            _ => adjacent(middle, true),
                        }
                    }
                }
                Case::Q6g64 | Case::Q8g64 => {
                    let max_code = if case == Case::Q6g64 { 63.0 } else { 255.0 };
                    let slot = col % 64;
                    if slot == 0 {
                        0.0
                    } else if slot == 63 {
                        max_code
                    } else {
                        let middle =
                            ((slot - 1 + (col / 64) * 62) % max_code as usize) as f32 + 0.5;
                        match row % 3 {
                            0 => adjacent(middle, false),
                            1 => middle,
                            _ => adjacent(middle, true),
                        }
                    }
                }
                Case::Fp8g128 => {
                    let code = ((index / 6) % 126) as u8;
                    let middle = (fp8_decode(code) + fp8_decode(code + 1)) * 0.5;
                    let magnitude = match index % 3 {
                        0 => adjacent(middle, false),
                        1 => middle,
                        _ => adjacent(middle, true),
                    };
                    if index % 6 < 3 {
                        magnitude
                    } else {
                        -magnitude
                    }
                }
                _ => match col % 8 {
                    0 | 1 => -0.125,
                    2 | 3 => 0.125,
                    4 => 0.0,
                    5 => -0.0,
                    6 => -0.0625,
                    _ => 0.0625,
                },
            },
            Fixture::Signed => {
                if case == Case::Fp8g128 {
                    const SPECIAL: [f32; 12] = [
                        0.0,
                        -0.0,
                        448.0,
                        -448.0,
                        480.0,
                        -480.0,
                        f32::MAX,
                        -f32::MAX,
                        0.015625,
                        -0.015625,
                        1.0625,
                        -1.0625,
                    ];
                    SPECIAL[index % SPECIAL.len()]
                } else {
                    let magnitude = [0.0, 0.0625, 1.0, 2048.0][(col / 2) % 4];
                    if (row + col + usize::from(up)) % 2 == 0 {
                        magnitude
                    } else {
                        -magnitude
                    }
                }
            }
            Fixture::Tiny => {
                const BITS: [u32; 8] = [
                    1, 2, 0x007fffff, 0x00800000, 0x33000000, 0x33800000, 0, 0x80000000,
                ];
                let value = f32::from_bits(BITS[index % BITS.len()]);
                if (index / BITS.len() + usize::from(up)) % 2 == 0 {
                    value
                } else {
                    -value
                }
            }
            Fixture::Trellis => {
                // Unit-sign/scale preparation makes every 128x128 rotated block constant.
                if row == 0 && col % 128 == 0 {
                    let impulse = if case.encoding().ht_bits().is_some() {
                        442.0
                    } else {
                        128.0
                    };
                    if up {
                        impulse
                    } else {
                        -impulse
                    }
                } else {
                    0.0
                }
            }
            Fixture::All => unreachable!("all is expanded before input generation"),
        };
        values.push(value);
    }
    Ok(values)
}

fn prepare<'a>(
    args: &Args,
    case: Case,
    fixture: Fixture,
    timing: bool,
    real: Option<&'a RealBands>,
) -> Result<Prepared<'a>> {
    let setup_start = Instant::now();
    let real = if timing && case != Case::Fp8g128 {
        real
    } else {
        None
    };
    let rows = if case == Case::Fp8g128 && timing {
        args.fp8_rows
    } else if !timing && !case.rotated() {
        7
    } else {
        ROWS
    };
    let cols = if let Some(real) = real {
        real.cols
    } else if case == Case::Fp8g128 {
        160
    } else if timing {
        args.cols
    } else if case.rotated() {
        128
    } else {
        320
    };
    bounded_values(rows, cols)?;
    let payload_rows = if case.paired() { rows * 2 } else { rows };
    payload_layout(case.encoding(), payload_rows, cols)?;
    let input = if let Some(real) = real {
        Cow::Borrowed(real.input.as_slice())
    } else {
        Cow::Owned(synthetic(case, fixture, rows, cols, args.seed, false)?)
    };
    let up = if case.paired() {
        Some(if let Some(real) = real {
            Cow::Borrowed(real.up.as_deref().context("missing paired real band")?)
        } else {
            Cow::Owned(synthetic(case, fixture, rows, cols, args.seed, true)?)
        })
    } else {
        None
    };
    let source = if let Some(real) = real {
        real.source.clone()
    } else {
        json!({"kind": "synthetic", "recipe": fixture.name(),
            "generator": if fixture == Fixture::Ordinary { "splitmix64-dyadic-bf16-v1" } else { "finite-corpus-v1" }})
    };
    let setup_elapsed_ns = ns(setup_start)?;
    let prep_start = Instant::now();
    let mut su = Vec::new();
    let mut sv = Vec::new();
    let mut up_sv = Vec::new();
    let preparation = if case.rotated() {
        let mut signs = generate_signs(args.seed, "encoder-bench.output", rows);
        let up_signs = generate_signs(args.seed, "encoder-bench.up-output", rows);
        su = generate_signs(args.seed, "encoder-bench.shared-input", cols);
        if fixture == Fixture::Signed {
            for (index, sign) in signs.iter_mut().enumerate() {
                *sign = if index % 2 == 0 { -1.0 } else { 1.0 };
            }
            for (index, sign) in su.iter_mut().enumerate() {
                *sign = if index % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        if fixture == Fixture::Trellis
            || (case == Case::I4r && matches!(fixture, Fixture::Ties | Fixture::Signed))
        {
            su.fill(1.0);
            sv = vec![1.0; rows];
            if case.paired() {
                up_sv = vec![1.0; rows];
            }
            if fixture == Fixture::Trellis {
                "unit-signs-and-scales-for-constant-rotated-fixture"
            } else {
                "unit-signs-for-inverse-rotated-i4r-rounding-fixture"
            }
        } else if case.encoding().ht_bits().is_some() {
            sv = ht_row_scales(&input, rows, cols, &signs)?;
            if let Some(up) = &up {
                up_sv = ht_row_scales(up, rows, cols, &up_signs)?;
            }
            "seeded-signs-and-signed-f16-row-rms-outside-encoding"
        } else {
            sv = signs;
            "seeded-unit-signs-outside-encoding"
        }
    } else {
        "no-rotation-preparation"
    };
    let fp8_scale = if fixture == Fixture::Ordinary {
        0.015625
    } else {
        1.0
    };
    let preparation_elapsed_ns = ns(prep_start)?;
    Ok(Prepared {
        case,
        fixture,
        rows,
        cols,
        input,
        up,
        su,
        sv,
        up_sv,
        fp8_scale,
        source,
        preparation,
        setup_elapsed_ns,
        preparation_elapsed_ns,
    })
}

fn encode(
    work: &Prepared<'_>,
    threads: usize,
    beam: usize,
    backend: HtBackend,
) -> Result<EncodedBlock> {
    let meta = QuantMeta {
        su: &work.su,
        sv: &work.sv,
        fp8_scale: work.fp8_scale,
        threads,
        ht_beam: beam,
        ht_backend: backend,
        ..QuantMeta::default()
    };
    let encoding = work.case.encoding();
    if let Some(up) = &work.up {
        let up_meta = QuantMeta {
            sv: &work.up_sv,
            ..meta
        };
        Ok(EncodedBlock {
            codes: encode_ht_gate_up_bands(
                encoding,
                black_box(&work.input),
                black_box(up),
                work.cols,
                &meta,
                &up_meta,
            )?,
            scales: Vec::new(),
        })
    } else {
        encode_rows(
            encoding,
            black_box(&work.input),
            work.rows,
            work.cols,
            &meta,
        )
    }
}

fn half_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_le_bytes())
        .collect()
}

fn component(bytes: &[u8]) -> Value {
    json!({"bytes": bytes.len(), "sha256": digest(bytes)})
}

fn tagged(hash: &mut Sha256, name: &str, bytes: &[u8]) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn hash_zeros(hash: &mut Sha256, mut bytes: u64) {
    const ZEROS: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let count = bytes.min(ZEROS.len() as u64) as usize;
        hash.update(&ZEROS[..count]);
        bytes -= count as u64;
    }
}

fn fingerprint(work: &Prepared<'_>, block: &EncodedBlock) -> Result<Value> {
    let encoding = work.case.encoding();
    let payload_rows = work.rows * if work.case.paired() { 2 } else { 1 };
    let layout = payload_layout(encoding, payload_rows, work.cols)?;
    let su = half_bytes(&work.su);
    let sv = half_bytes(&work.sv);
    let up_sv = half_bytes(&work.up_sv);
    let codebook: Vec<u8> = if encoding == Encoding::Q4Cp {
        NF4_CODEBOOK
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    } else {
        Vec::new()
    };
    let fp8_scale = if encoding == Encoding::Fp8G128 {
        work.fp8_scale.to_le_bytes().to_vec()
    } else {
        Vec::new()
    };
    let mut metadata = Vec::with_capacity(64);
    metadata.extend_from_slice(&encoding.code().to_le_bytes());
    metadata.extend_from_slice(&encoding.variant().to_le_bytes());
    for value in [
        payload_rows as u64,
        work.cols as u64,
        layout.len,
        layout.codes_offset,
        layout.scales_offset,
        layout.scale_row_stride,
    ] {
        metadata.extend_from_slice(&value.to_le_bytes());
    }
    let mut payload = Sha256::new();
    ensure!(
        codebook.len() as u64 <= layout.codes_offset,
        "codebook exceeds code offset"
    );
    payload.update(&codebook);
    hash_zeros(&mut payload, layout.codes_offset - codebook.len() as u64);
    payload.update(&block.codes);
    let position = layout.codes_offset + block.codes.len() as u64;
    let tail: &[u8] = if encoding == Encoding::Fp8G128 {
        &fp8_scale
    } else {
        &block.scales
    };
    if tail.is_empty() {
        ensure!(position <= layout.len, "code plane exceeds payload length");
        hash_zeros(&mut payload, layout.len - position);
    } else {
        ensure!(
            position <= layout.scales_offset,
            "code plane overlaps scale plane"
        );
        ensure!(
            layout.scales_offset + tail.len() as u64 <= layout.len,
            "scale plane exceeds payload length"
        );
        hash_zeros(&mut payload, layout.scales_offset - position);
        payload.update(tail);
        hash_zeros(
            &mut payload,
            layout.len - layout.scales_offset - tail.len() as u64,
        );
    }
    let payload_sha256 = format!("{:x}", payload.finalize());
    let mut complete = Sha256::new();
    tagged(&mut complete, "contract", CONTRACT.as_bytes());
    for (name, bytes) in [
        ("layout", metadata.as_slice()),
        ("codes", block.codes.as_slice()),
        ("scales", block.scales.as_slice()),
        ("rotation_su_f16", su.as_slice()),
        ("rotation_sv_f16", sv.as_slice()),
        ("rotation_up_sv_f16", up_sv.as_slice()),
        ("codebook_f32", codebook.as_slice()),
        ("fp8_global_scale_f32", fp8_scale.as_slice()),
    ] {
        tagged(&mut complete, name, bytes);
    }
    // This also covers deterministic inter-plane and row padding, not just active values.
    tagged(&mut complete, "payload_sha256", payload_sha256.as_bytes());
    Ok(json!({
        "complete_sha256": format!("{:x}", complete.finalize()),
        "payload": {"bytes": layout.len, "sha256": payload_sha256},
        "codes": component(&block.codes), "scales": component(&block.scales),
        "rotation_su_f16": component(&su), "rotation_sv_f16": component(&sv),
        "rotation_up_sv_f16": component(&up_sv), "codebook_f32": component(&codebook),
        "fp8_global_scale_f32": component(&fp8_scale), "layout_metadata": component(&metadata),
    }))
}

fn check_output(expected: &mut BTreeMap<String, Value>, key: &str, actual: Value) -> Result<Value> {
    if let Some(previous) = expected.get(key) {
        ensure!(
            previous == &actual,
            "nondeterministic output for {key} (iteration/sample/thread/backend mismatch)"
        );
    } else {
        expected.insert(key.to_owned(), actual.clone());
    }
    Ok(actual)
}

fn emit(out: &mut impl Write, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    writeln!(out)?;
    out.flush()?;
    Ok(())
}

fn run_workload(
    args: &Args,
    work: &Prepared<'_>,
    timing: bool,
    expected: &mut BTreeMap<String, Value>,
    out: &mut impl Write,
) -> Result<()> {
    let suite = if timing { "timing" } else { "corpus" };
    let ht = work.case.encoding().ht_bits().is_some();
    let beams: &[usize] = if ht && !timing {
        &args.corpus_beams
    } else {
        &[args.timing_beam]
    };
    let iterations = if timing { args.iterations } else { 1 };
    let repetitions = if timing { args.repetitions } else { 1 };
    let warmup = if timing { args.warmup } else { 0 };
    let payload_rows = work.rows * if work.case.paired() { 2 } else { 1 };
    let layout = payload_layout(work.case.encoding(), payload_rows, work.cols)?;
    let retained = layout
        .len
        .checked_mul(u64::try_from(iterations)?)
        .context("retained byte count overflow")?;
    ensure!(
        retained <= MAX_RETAINED_BYTES,
        "iterations retain more than 128 MiB; reduce --iterations"
    );
    let input_fingerprint = json!({
        "primary_f32_sha256": float_digest(&work.input),
        "up_f32_sha256": work.up.as_deref().map(float_digest),
    });
    for &beam in beams {
        let key = format!(
            "{suite}:{}:{}:{}:{}:{beam}",
            work.case.name(),
            work.fixture.name(),
            work.rows,
            work.cols
        );
        // Neither backend nor threads is part of the equality key.
        for &backend in &args.backends {
            let effective_backend = if ht { backend } else { HtBackend::Cpu };
            let timed_region = json!({
                "public_encoding": true, "validation": true, "codec_allocations": true,
                "rotation": work.case.rotated(), "input_setup": false, "sign_generation": false,
                "row_rms": false, "hashing": false, "output_destruction": false,
                "gpu_host_device_copies_and_synchronization": effective_backend.is_gpu(),
                "gpu_device_preflight": false, "preallocated_result_push": true,
            });
            for &threads in &args.threads {
                for _ in 0..warmup {
                    let block = encode(work, threads, beam, effective_backend)?;
                    check_output(expected, &key, fingerprint(work, &block)?)?;
                }
                for sample in 0..repetitions {
                    let mut outputs = Vec::with_capacity(iterations);
                    let start = Instant::now();
                    for _ in 0..iterations {
                        outputs.push(black_box(encode(work, threads, beam, effective_backend)?));
                    }
                    let elapsed_ns = ns(start)?;
                    let mut output_fingerprint = None;
                    for block in &outputs {
                        output_fingerprint =
                            Some(check_output(expected, &key, fingerprint(work, block)?)?);
                    }
                    let values_per_iteration =
                        work.rows * work.cols * if work.case.paired() { 2 } else { 1 };
                    emit(
                        out,
                        &json!({
                            "schema": SCHEMA, "kind": "sample", "contract": CONTRACT,
                            "label": args.label, "suite": suite, "case": work.case.name(),
                            "encoding_code": work.case.encoding().code(), "encoding_variant": work.case.encoding().variant(),
                            "fixture": work.fixture.name(), "source": work.source, "seed": args.seed,
                            "rows": work.rows, "cols": work.cols, "projection_count": if work.case.paired() { 2 } else { 1 },
                            "payload_rows": payload_rows, "values_per_iteration": values_per_iteration,
                            "beam": if ht { Some(beam) } else { None }, "threads": threads,
                            "selected_backend": backend, "effective_backend": effective_backend,
                            "sample": sample, "iterations": iterations, "repetitions": repetitions, "warmup": warmup,
                            "elapsed_ns": elapsed_ns, "ns_per_iteration": elapsed_ns as f64 / iterations as f64,
                            "values_per_second": values_per_iteration as f64 * iterations as f64 * 1.0e9 / elapsed_ns.max(1) as f64,
                            "setup_elapsed_ns": work.setup_elapsed_ns, "preparation_elapsed_ns": work.preparation_elapsed_ns,
                            "preparation": work.preparation, "input_fingerprint": input_fingerprint,
                            "timed_region": timed_region,
                            "verified_identical_outputs": iterations,
                            "output_fingerprint": output_fingerprint.context("empty sample")?,
                        }),
                    )?;
                    // Retained outputs are destroyed after timing and fingerprinting.
                }
            }
        }
    }
    Ok(())
}

fn unique_expanded<T: Copy + PartialEq>(selected: &[T], all_marker: T, all: &[T]) -> Vec<T> {
    let selected = if selected.contains(&all_marker) {
        all
    } else {
        selected
    };
    let mut result = Vec::with_capacity(selected.len());
    for &value in selected {
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.threads.is_empty()
            && args
                .threads
                .iter()
                .all(|threads| *threads == 1 || *threads == 2),
        "--threads accepts only 1 and/or 2"
    );
    ensure!(
        args.iterations > 0 && args.repetitions > 0,
        "iterations and repetitions must be positive"
    );
    ensure!(
        !args.corpus_beams.is_empty()
            && args
                .corpus_beams
                .iter()
                .all(|beam| *beam > 0 && *beam <= 4096),
        "corpus beams must be in 1..=4096 (full-state search is intentionally excluded)"
    );
    ensure!(
        (1..=4096).contains(&args.timing_beam),
        "timing beam must be in 1..=4096"
    );
    ensure!(!args.backends.is_empty(), "select at least one backend");
    let mut gpu_preflight = Vec::new();
    for backend in [HtBackend::Hip, HtBackend::Cuda] {
        if !args.backends.contains(&backend) {
            continue;
        }
        ensure!(
            args.timing_beam <= 256 && args.corpus_beams.iter().all(|beam| *beam <= 256),
            "{} beams must be in 1..=256",
            backend.label(),
        );
        let start = Instant::now();
        backend.preflight()?;
        gpu_preflight.push(json!({"backend": backend, "elapsed_ns": ns(start)?}));
    }
    if args.hf.is_none()
        && (args.tensor.is_some()
            || args.up_tensor.is_some()
            || args.row_start != 0
            || args.up_row_start != 0)
    {
        bail!("tensor and row-selection options require --hf");
    }
    ensure!(
        args.up_tensor.is_some() || args.up_row_start == 0,
        "--up-row-start requires --up-tensor"
    );
    let cases = unique_expanded(&args.cases, Case::All, &Case::ALL);
    let fixtures = unique_expanded(&args.fixtures, Fixture::All, &Fixture::ALL);
    ensure!(
        !cases.is_empty() && !fixtures.is_empty(),
        "select at least one case and fixture"
    );
    let real = load_real(&args, &cases)?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    emit(
        &mut out,
        &json!({
            "schema": SCHEMA, "kind": "run", "contract": CONTRACT, "label": args.label,
            "encoder_sha256": bf16_to_hgn::ENCODER_SHA256, "package_version": env!("CARGO_PKG_VERSION"),
            "debug_assertions": cfg!(debug_assertions), "cases": cases.iter().map(|case| case.name()).collect::<Vec<_>>(),
            "threads": args.threads, "backends": args.backends,
            "timing_beam": args.timing_beam, "corpus_beams": args.corpus_beams,
            "iterations": args.iterations, "repetitions": args.repetitions, "warmup": args.warmup,
            "synthetic_cols": args.cols, "fp8_rows": args.fp8_rows, "seed": args.seed,
            "real_band_read_elapsed_ns": real.as_ref().map(|real| real.read_elapsed_ns),
            "gpu_preflight_before_samples": !gpu_preflight.is_empty(),
            "gpu_preflight": gpu_preflight,
            "gpu_timing": "Synchronous HT encoding includes GPU copies/kernels/synchronization only when effective_backend is a GPU. Preflight is separate; a fresh process with one GPU selected and --warmup 0 measures its first encoding. Subsequent samples reuse native workspace/codebook. Non-HT encodings remain CPU regardless of selected_backend.",
            "input_band_value_limit": MAX_BAND_VALUES, "retained_output_byte_limit": MAX_RETAINED_BYTES,
            "fingerprint": "SHA-256; little-endian raw floats/side planes; tagged component lengths; excludes timing, threads, backend and label",
        }),
    )?;
    let mut expected = BTreeMap::new();
    if args.suite != Suite::Corpus {
        for &case in &cases {
            let work = prepare(&args, case, Fixture::Ordinary, true, real.as_ref())?;
            run_workload(&args, &work, true, &mut expected, &mut out)?;
        }
    }
    if args.suite != Suite::Timing {
        for &case in &cases {
            for &fixture in &fixtures {
                let work = prepare(&args, case, fixture, false, None)?;
                run_workload(&args, &work, false, &mut expected, &mut out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_key_rejects_backend_byte_or_side_plane_differences() {
        let mut expected = BTreeMap::new();
        let key = "corpus:ht3:ordinary-bf16:128:128:256";
        let cpu =
            json!({"codes": {"sha256": "cpu-bytes"}, "rotation_sv_f16": {"sha256": "scale-bytes"}});
        check_output(&mut expected, key, cpu.clone()).unwrap();
        check_output(&mut expected, key, cpu.clone()).unwrap();
        let mut changed_codes = cpu.clone();
        changed_codes["codes"]["sha256"] = json!("different-gpu-bytes");
        assert!(check_output(&mut expected, key, changed_codes).is_err());
        let mut changed_plane = cpu;
        changed_plane["rotation_sv_f16"]["sha256"] = json!("different-scale-bytes");
        assert!(check_output(&mut expected, key, changed_plane).is_err());
    }

    #[test]
    fn non_ht_samples_report_cpu_work_for_every_selected_backend() {
        let args = Args::parse_from(["encoder_bench", "--backends", "cpu,hip,cuda"]);
        for case in [
            Case::I4r,
            Case::Nf4,
            Case::Q6g64,
            Case::Q8g64,
            Case::Fp8g128,
        ] {
            let work = prepare(&args, case, Fixture::Ordinary, false, None).unwrap();
            let mut expected = BTreeMap::new();
            let mut output = Vec::new();
            run_workload(&args, &work, false, &mut expected, &mut output).unwrap();
            let output = String::from_utf8(output).unwrap();
            let samples: Vec<Value> = output
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            assert_eq!(samples.len(), 3);
            for (sample, selected) in samples.iter().zip(["cpu", "hip", "cuda"]) {
                assert_eq!(sample["schema"], 2);
                assert_eq!(sample["contract"], "encoder-bench-v2");
                assert_eq!(sample["selected_backend"], selected);
                assert_eq!(sample["effective_backend"], "cpu");
                assert_eq!(
                    sample["timed_region"]["gpu_host_device_copies_and_synchronization"],
                    false
                );
                assert_eq!(sample["timed_region"]["gpu_device_preflight"], false);
                assert_eq!(
                    sample["output_fingerprint"],
                    samples[0]["output_fingerprint"]
                );
                assert!(sample.get("backend").is_none());
                assert!(sample.get("encoding_backend").is_none());
                assert!(sample["timed_region"]
                    .get("hip_host_device_copies_and_synchronization")
                    .is_none());
                assert!(sample["timed_region"].get("hip_device_preflight").is_none());
            }
        }
    }
}
