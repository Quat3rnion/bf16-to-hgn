use anyhow::{Context, Result};
use bf16_to_hgn::{
    architecture, container, conversion, hf::Snapshot, manifest::Options, verify, ConversionPlan,
    POLICY, SPEC_COMMIT, SPEC_VERSION,
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
    about = "Convert complete Qwen3.8 Flash-Next BF16 safetensors snapshots to native v2 HGN",
    long_about = "From-scratch deterministic native v2 HGN conversion. Only complete Qwen4ExpForConditionalGeneration snapshots with the Flash-Next tensor/config contract are accepted. No reference HGN or GPU is used."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Read and validate source metadata only; no tensor payloads are loaded.
    Inspect {
        #[arg(long)]
        source: PathBuf,
    },
    /// Print the complete tensor/store/layout plan without writing anything.
    Plan {
        #[arg(long)]
        source: PathBuf,
        #[command(flatten)]
        encoding: EncodingOptions,
    },
    /// Stream a complete new model directory, then publish atomically without clobbering.
    Convert {
        #[arg(long)]
        source: PathBuf,
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
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Args)]
struct EncodingOptions {
    /// Stable seed for generated Hadamard sign vectors.
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Native HT workers (1..64). Rust streams one tensor/band at a time.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Deterministic approximate HT trellis beam width (1..256).
    #[arg(long, default_value_t = 128)]
    ht_beam: usize,
}
impl EncodingOptions {
    fn options(&self) -> Result<Options> {
        let options = Options {
            seed: self.seed,
            threads: self.threads,
            ht_beam: self.ht_beam,
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
        Command::Inspect { source } => {
            let snapshot = Snapshot::open(&source)?;
            let plan = architecture::plan(&snapshot)?;
            print_json(&summary(&snapshot, &plan)?)?;
        }
        Command::Plan { source, encoding } => {
            let options = encoding.options()?;
            let snapshot = Snapshot::open(&source)?;
            let plan = architecture::plan(&snapshot)?;
            print_json(&plan_report(&snapshot, &plan, &options)?)?;
        }
        Command::Convert {
            source,
            output,
            dry_run,
            encoding,
        } => {
            let options = encoding.options()?;
            let snapshot = Snapshot::open(&source)?;
            let plan = architecture::plan(&snapshot)?;
            conversion::destination_path(&snapshot, &output)?;
            if dry_run {
                print_json(&plan_report(&snapshot, &plan, &options)?)?;
            } else {
                let manifest = conversion::convert(&snapshot, &plan, &output, &options)?;
                print_json(
                    &json!({ "published": true, "source_tensors": manifest.sources.len(), "output_tensors": manifest.containers.iter().map(|c| c.tensors.len()).sum::<usize>(), "manifest": "conversion-manifest.json", "policy": manifest.policy }),
                )?;
            }
        }
        Command::Verify { source, output } => {
            let snapshot = Snapshot::open(&source)?;
            let plan = architecture::plan(&snapshot)?;
            let report = verify::verify(&snapshot, &plan, &output)?;
            print_json(&report)?;
        }
    }
    Ok(())
}

fn summary(snapshot: &Snapshot, plan: &ConversionPlan) -> Result<serde_json::Value> {
    let mut dtypes = BTreeMap::<String, usize>::new();
    let mut shards = BTreeSet::new();
    let mut source_bytes = 0u64;
    for tensor in snapshot.tensors.values() {
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
        json!({ "architecture": plan.architecture, "source_tensors": snapshot.tensors.len(), "source_shards": shards.len(), "source_payload_bytes": source_bytes, "source_dtypes": dtypes, "outputs": outputs, "assets": plan.assets, "output_hgn_bytes": conversion::planned_bytes(plan)?, "spec_version": SPEC_VERSION, "spec_commit": SPEC_COMMIT, "policy": POLICY, "tensor_payloads_read": false }),
    )
}

fn plan_report(
    snapshot: &Snapshot,
    plan: &ConversionPlan,
    options: &Options,
) -> Result<serde_json::Value> {
    let layouts = plan
        .files
        .iter()
        .map(container::layout)
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({ "summary": summary(snapshot, plan)?, "options": options, "plan": plan, "layouts": layouts, "source_config_preserved": true, "reference_hgn_required": false, "quantization": "deterministic NF4 + RTN I4R + affine min/max + signed row-RMS HT trellis; no calibration/upstream recipe claim", "working_set": "128-row matrix bands, 8192-row table chunks, <=8MiB raw-copy buffers; native workers additionally allocate per-thread scratch", "disk_reserve": "HGN bytes +5% +64MiB; tokenizer assets copied separately" }),
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
