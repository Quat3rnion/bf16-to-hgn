use anyhow::{Context, Result};
use bf16_to_hgn::{
    architecture,
    codec::{HtBackend, I4rBackend},
    container, conversion,
    manifest::{Options, OutputProfile},
    progress::Progress,
    source::{self, SourceKind, SourceOptions, TensorSource},
    verify,
    verify_gpu::DecodeBackend,
    ConversionPlan, SPEC_COMMIT, SPEC_VERSION,
};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    version,
    about = "Convert complete Qwen3.8 Flash-Next HF, HGN, or GGUF sources to native HGN",
    long_about = "Deterministic native v2 (default) or HT43 HGN conversion for complete Flash-Next weights. Imported HGN/GGUF sources are requantized from their decoded values, not treated as original BF16. Matching config/tokenizer and all required sidecars must be supplied; no stock weights are substituted."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read and validate source metadata and the requested output profile.
    Inspect {
        #[command(flatten)]
        input: InputOptions,
        #[arg(long, value_enum, default_value_t = OutputProfile::V2)]
        profile: OutputProfile,
        /// HT execution backend to describe; inspection does not load a device.
        #[arg(long, value_enum, default_value_t = HtBackend::Cpu)]
        ht_backend: HtBackend,
        /// I4R execution backend to describe; inspection does not load a device.
        #[arg(long, value_enum, default_value_t = I4rBackend::Cpu)]
        i4r_backend: I4rBackend,
    },
    /// Print the complete tensor/store/layout plan without writing anything.
    Plan {
        #[command(flatten)]
        input: InputOptions,
        #[command(flatten)]
        encoding: EncodingOptions,
    },
    /// Stream a complete new model directory, then publish atomically without clobbering.
    Convert {
        #[command(flatten)]
        input: InputOptions,
        #[arg(long)]
        output: PathBuf,
        /// Validate and print the plan; do not create output files or read tensor payloads.
        #[arg(long)]
        dry_run: bool,
        #[command(flatten)]
        encoding: EncodingOptions,
    },
    /// Check actual source hashes, every output payload/layout, and decoded values.
    Verify {
        #[command(flatten)]
        input: InputOptions,
        #[arg(long)]
        output: PathBuf,
        /// Runtime decoder for HT3/HT4 and I4R; other codecs remain on CPU.
        #[arg(long, value_enum, default_value_t = DecodeBackend::Cpu)]
        decode_backend: DecodeBackend,
    },
}

#[derive(Args)]
struct InputOptions {
    /// HF snapshot directory, HGN model/output directory, or GGUF model file.
    #[arg(long)]
    source: PathBuf,
    /// Physical input format; otherwise detect an HF index or HGN/GGUF magic.
    #[arg(long, value_enum)]
    source_format: Option<SourceKind>,
    /// Matching config/tokenizer asset directory for HGN/GGUF inputs.
    #[arg(long)]
    assets: Option<PathBuf>,
    /// Matching n-gram HGN sidecar when not embedded in the input.
    #[arg(long)]
    ngram: Option<PathBuf>,
    /// Matching vision HGN sidecar when not embedded in the input.
    #[arg(long)]
    vision: Option<PathBuf>,
    /// Matching MTP HGN sidecar when not embedded in the input.
    #[arg(long)]
    mtp: Option<PathBuf>,
}
impl InputOptions {
    fn open(self) -> Result<Box<dyn TensorSource>> {
        source::open(
            &self.source,
            self.source_format,
            &SourceOptions {
                assets: self.assets,
                ngram: self.ngram,
                vision: self.vision,
                mtp: self.mtp,
            },
        )
    }
}

#[derive(Args)]
struct EncodingOptions {
    /// Native output profile. Existing v2 is the default.
    #[arg(long, value_enum, default_value_t = OutputProfile::V2)]
    profile: OutputProfile,
    /// Stable seed for generated Hadamard sign vectors.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Native HT workers (1..64). Rust streams one tensor/band at a time.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Deterministic approximate HT trellis beam width (1..256).
    #[arg(long, default_value_t = 128)]
    ht_beam: usize,
    /// HT3/HT4 trellis backend; real GPU execution requires --features hip or cuda.
    #[arg(long, value_enum, default_value_t = HtBackend::Cpu)]
    ht_backend: HtBackend,
    /// I4R rotation and quantization backend, independent of HT; CUDA requires --features cuda.
    #[arg(long, value_enum, default_value_t = I4rBackend::Cpu)]
    i4r_backend: I4rBackend,
}
impl EncodingOptions {
    fn options(&self) -> Result<Options> {
        let options = Options {
            profile: self.profile,
            seed: self.seed,
            threads: self.threads,
            ht_beam: self.ht_beam,
            ht_backend: self.ht_backend,
            i4r_backend: self.i4r_backend,
        };
        options.validate()?;
        Ok(options)
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bf16-to-hgn: {error:#}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect {
            input,
            profile,
            ht_backend,
            i4r_backend,
        } => {
            let options = Options {
                profile,
                ht_backend,
                i4r_backend,
                ..Options::default()
            };
            let snapshot = input.open()?;
            let plan = architecture::plan(snapshot.as_ref(), &options)?;
            print_json(&summary(snapshot.as_ref(), &plan, &options)?)?;
        }
        Command::Plan { input, encoding } => {
            let options = encoding.options()?;
            let snapshot = input.open()?;
            let plan = architecture::plan(snapshot.as_ref(), &options)?;
            print_json(&plan_report(snapshot.as_ref(), &plan, &options)?)?;
        }
        Command::Convert {
            input,
            output,
            dry_run,
            encoding,
        } => {
            let options = encoding.options()?;
            if !dry_run {
                options.validate_for_execution()?;
            }
            let snapshot = input.open()?;
            let plan = architecture::plan(snapshot.as_ref(), &options)?;
            conversion::destination_path(snapshot.as_ref(), &output)?;
            if dry_run {
                print_json(&plan_report(snapshot.as_ref(), &plan, &options)?)?;
            } else {
                let manifest = conversion::convert(
                    snapshot.as_ref(),
                    &plan,
                    &output,
                    &options,
                    &Progress::stderr(),
                )?;
                print_json(
                    &json!({ "published": true, "source_tensors": manifest.sources.len(), "output_tensors": manifest.containers.iter().map(|c| c.tensors.len()).sum::<usize>(), "manifest": "conversion-manifest.json", "source_format": manifest.source_kind, "profile": manifest.options.profile, "ht_backend": manifest.options.ht_backend, "i4r_backend": manifest.options.i4r_backend, "policy": manifest.policy }),
                )?;
            }
        }
        Command::Verify {
            input,
            output,
            decode_backend,
        } => {
            decode_backend.preflight()?;
            let options = verify::manifest_options(&output)?;
            let snapshot = input.open()?;
            let plan = architecture::plan(snapshot.as_ref(), &options)?;
            let report = verify::verify(
                snapshot.as_ref(),
                &plan,
                &output,
                decode_backend,
                &Progress::stderr(),
            )?;
            print_json(&report)?;
        }
    }
    Ok(())
}

fn summary(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    options: &Options,
) -> Result<serde_json::Value> {
    let mut dtypes = BTreeMap::<String, usize>::new();
    let mut shards = BTreeSet::new();
    let mut source_bytes = 0u64;
    for tensor in snapshot.tensors().values() {
        *dtypes.entry(format!("{:?}", tensor.dtype)).or_default() += 1;
        shards.insert(&tensor.shard);
        source_bytes = source_bytes
            .checked_add(tensor.size)
            .context("source total size overflow")?;
    }
    let mut outputs = Vec::new();
    for file in &plan.files {
        let mut stores = BTreeMap::<String, usize>::new();
        for tensor in &file.tensors {
            *stores.entry(format!("{:?}", tensor.encoding)).or_default() += 1;
        }
        outputs.push(json!({ "filename": file.filename, "identity": file.identity, "tensors": file.tensors.len(), "bytes": container::layout(file)?.file_size, "stores": stores }));
    }
    Ok(
        json!({ "architecture": plan.architecture, "source_format": snapshot.kind(), "source_tensors": snapshot.tensors().len(), "source_shards": shards.len(), "canonical_source_payload_bytes": source_bytes, "source_dtypes": dtypes, "outputs": outputs, "assets": plan.assets, "output_hgn_bytes": conversion::planned_bytes(plan)?, "spec_version": SPEC_VERSION, "spec_commit": SPEC_COMMIT, "profile": options.profile, "ht_backend": options.ht_backend, "i4r_backend": options.i4r_backend, "policy": options.profile.policy(), "tensor_payloads_read": false }),
    )
}

fn plan_report(
    snapshot: &dyn TensorSource,
    plan: &ConversionPlan,
    options: &Options,
) -> Result<serde_json::Value> {
    let layouts = plan
        .files
        .iter()
        .map(container::layout)
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({ "summary": summary(snapshot, plan, options)?, "options": options, "plan": plan, "layouts": layouts, "source_config_preserved": true, "quantization": "deterministic conversion from canonical source values; no calibration/upstream recipe or BF16-equivalent requantization claim", "working_set": "bounded source windows, 128-row matrix bands, 8192-row table chunks, <=8MiB raw-copy buffers; native workers additionally allocate per-thread scratch", "disk_reserve": "HGN bytes +5% +64MiB; tokenizer assets copied separately" }),
    )
}

fn print_json(value: &impl serde::Serialize) -> Result<()> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer_pretty(&mut output, value)?;
    use std::io::Write;
    output.write_all(b"\n")?;
    Ok(())
}
