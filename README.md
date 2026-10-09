# BF16 safetensors → native v2 HGN

A standalone Rust CLI with a vendored C++ trellis encoder. It builds every
payload from the input HF snapshot; it does **not** copy stock HGN weights,
rotation vectors, codebooks, an MTP head, or a reference checkpoint.

## Support boundary

Only a complete **Qwen3.8 Flash-Next** HF safetensors directory is accepted:
`Qwen4ExpForConditionalGeneration`, `model_type=qwen4_exp`, and the fixed
Flash-Next configuration and tensor shapes. Other architectures, partial
snapshots, GGUF, quantized safetensors, unexpected tensors, and incompatible
configuration values are rejected before payload conversion.

The contract is independently enumerated, not inferred from tensor counts:
1,658 source tensors → 1,197 trunk/MTP entries, 128 n-gram shards, and 333
vision entries. The converter generates all 818 rotation planes. Weights
must be BF16; the three PLE hash vectors must be I64. F32 is also accepted
and preserved **only** for DeltaNet `A_log` and `dt_bias`. The observed base
HF metadata uses BF16 for those vectors.

The binary format is pinned to [hgn-spec 1.1.0](https://github.com/jtsylve/hgn-spec/tree/79e75fd4ae7642478823525b0a3212951f584af5)
(`79e75fd4ae7642478823525b0a3212951f584af5`). Its canonical v2 store map,
HF training conventions, and names are used. HF norm gains are already
zero-centred and DeltaNet heads already grouped: no GGUF-style offset or
head permutation is applied. The rank-five vision patch kernel is flattened
row-major, without transposing values.

**Format compatibility is not inference qualification.** No full-model
conversion, inference validation, quality comparison, or performance
benchmark is claimed here. The publisher's calibration, learned codebooks,
and rotation policy are not public; this tool uses its own explicit policy.

## Requirements and build

Linux, Rust 1.85 or newer, a C++17 compiler with OpenMP (for example GCC and
`libgomp`), and sufficient disk space. There are no host/model paths compiled
into the tool, no GPU dependency, and no Python runtime dependency.
Linux publication uses `renameat2(RENAME_NOREPLACE)`; the output filesystem
must support it. A macOS no-replace publication implementation is present,
but its compiler/OpenMP toolchain is not qualified. Other OS publication is
rejected rather than weakened to a non-atomic overwrite-prone operation.

```sh
cargo build --release --locked --manifest-path deployments/frame/bf16-to-hgn/Cargo.toml
BIN=deployments/frame/bf16-to-hgn/target/release/bf16-to-hgn
```

`build.rs` compiles `native/ht_trellis.cpp` with C++17, `-O3`, OpenMP and
`-ffp-contract=off`, without `-march=native`. The native encoder is an exact
vendored copy of the existing proven trellis implementation; its notices
and license are under `native/`. The conversion manifest records an
aggregate SHA-256 of the compiled tool's source files, lockfile and helper.

## Commands

```sh
# Metadata only: headers, config, index, shapes/dtypes, required assets.
$BIN inspect --source /path/to/hf-bf16

# Complete deterministic store/shape/offset/disk plan; writes nothing.
$BIN plan --source /path/to/hf-bf16 --seed 0 --threads 1 --ht-beam 128
$BIN convert --source /path/to/hf-bf16 --output /path/to/new-output --dry-run

# Real conversion: the parent directory must exist; output must not exist.
$BIN convert --source /path/to/hf-bf16 --output /path/to/new-output \
  --seed 0 --threads 1 --ht-beam 128

# Full file/source verification and streaming decoded-error report.
$BIN verify --source /path/to/hf-bf16 --output /path/to/new-output
```

`inspect`, `plan`, and `convert --dry-run` never load tensor payloads or
write output files. `verify` is intentionally a full source/output read,
not a fast metadata check. It checks real headers, tables, sizes, all
padding, store variants, SHA-256 and XOR checksums, exact source coverage,
assets, generated vectors, raw-copy equality, finite quantized scales, and
decodes every quantized value against the corresponding source. Its F64
squared-error totals measure quantization error; they are not model quality
or an acceptance threshold.

The input must remain immutable during conversion and verification. Header,
index, file and asset signatures are checked; per-tensor source SHA-256 is
recorded from the actual consumed bytes. Concurrent external truncation of
an mmap-backed input cannot be made safe by metadata checks alone.

## Output directory

```text
new-output/
  model.hgn                         # qwen3.8-flash-next-v2, trunk + MTP + rotations
  qwen38-flash-next-ngram.hgn        # qwen3.8-flash-next-ngram, stacked FP8G table
  qwen38-flash-next-vision.hgn       # qwen3.8-flash-next-vision, 333 BF16 tensors
  config.json                       # original source config, unchanged
  tokenizer/                        # flat HF tokenizer files + unchanged source config
  conversion-manifest.json
```

Required assets are `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `merges.txt`, and `generation_config.json`. Present
`vocab.json`, `special_tokens_map.json`, `added_tokens.json`, and processor
configuration files are also copied from an explicit allowlist. Processor
configs stay at the output root; tokenizer assets are flat under
`tokenizer/`. The source config is retained verbatim at the root and in
`tokenizer/config.json`; no fake model metadata or reconstructed vocabulary
is invented. HF cache files, credentials and arbitrary unrelated assets
are never copied.

The manifest records the exact plan, policy/seed/worker/beam settings, format
pin, encoder source digest, config/index digests, each input tensor's
relative shard/shape/dtype/bounds/hash, copied asset hashes, and every output
entry's bounds/type/variant/checksum/SHA-256. It contains no absolute source
paths, timestamps, credentials, or assumption that weights came from a
particular upstream repository. The canonical HGN identity denotes the
engine's format; provenance is the source-byte inventory, not the identity.

A staging directory is created beside the requested output. Payloads are
written at preplanned aligned offsets; header/table are finalized only
once all checksums are known. Files and directories are synced before
atomic no-clobber publication. An ordinary pre-publication failure removes
the staging directory; a failed parent-directory sync removes the newly
published output. Existing destinations and any overlap with the
source directory are refused. The tool never modifies the input, services,
or a live checkpoint. A process killed without cleanup may leave an
unpublished `.bf16-to-hgn-*` staging directory; it is never treated as a
finished output or resumed automatically.

For serving, point the engine at `model.hgn`, its n-gram file and tokenizer,
and enable its normal native-v2 trunk pinning. The MTP weights are inside
`model.hgn`; no stock sidecar is required. Engine startup and inference
must be qualified separately on a safe host/outage before deployment.

## Deterministic quantization policy

Same input bytes, compiled tool and options produce the same payload bytes.
Cross-toolchain byte identity is **not** promised: C++ library tie traversal
in the approximate beam search can differ. The manifest pins the source
implementation; preserve the executable/toolchain as well when that level
of reproducibility matters.

- **BF16/F32/I64:** exact little-endian copies. Integer values never pass
  through F32. All floating input values must be finite. PLE hash vectors
  are checked against the stacked table's row bounds.
- **Q4C-P:** one fixed standard NF4 codebook per tensor, rather than an
  unpublished learned codebook. Groups of 32 search scale multipliers
  `[1, .95, .9, .85, .8]`, round scales to F16, and choose the lowest
  reconstruction error. Sorted-codebook midpoint ties choose the lower
  level. Nibbles and padded scale planes follow the specification.
- **Q8G64/Q6G64:** affine min/max quantization in groups of 64, F16 scale
  and bias, round-to-nearest ties-to-even codes, with exact row padding.
- **Rotations:** seed + canonical plane name, FNV-1a followed by SplitMix64,
  yields ±1 signs. Both axes use the documented normalized blockwise
  128-point Sylvester Hadamard transform.
- **HT4:** input signs and signed per-row RMS scales rounded to F16; a
  nonzero RMS below F16 range uses the smallest F16 subnormal, and an
  actual zero row has scale zero. The native fixed-codebook two-pass
  boundary-fixed trellis beam search encodes 128-row bands. The default
  beam is 128. This is approximate, not a global optimality claim.
- **I4R:** all routed experts of a layer share its four sign vectors.
  Independently rotated 128-row bands use signed extreme/8 scales and
  multipliers `[1, .97, .94, .91, .88, .85]`, rounded to F16 before
  nearest-even RTN. Full fused tensors hold the complete code plane,
  then the complete scale plane; no per-expert plane interleaving occurs.
- **N-gram FP8G:** two streaming passes, numeric shard order, one global
  F32 `absmax/448` scale (1 for an all-zero table), nearest-even E4M3
  codes capped at magnitude 448, zero padding then the global scale.

There are no Hessians, calibration, LDLQ, learned stock codebooks, or
upstream-quality equivalence promises. Unsupported values that cannot be
represented by the selected finite layouts fail instead of silently
switching storage formats.

## Memory and CPU safeguards

Rust processes one tensor and 128 matrix rows at a time, including each
expert separately in 128-row bands. The table uses 8,192-row chunks and two
passes; raw copies/hashes use at most 8 MiB chunks. Only small generated
rotation vectors are retained across tensors. Inputs are mmap-read in
bounded windows with page-cache eviction hints. Output dirty data is
synced/evicted every 64 MiB of writes and final hashing is streamed.
These are working-set bounds, not an exact RSS guarantee for every allocator
or operating system; native workers each add trellis scratch memory.

Default `--threads 1` is deliberate. `--threads` only controls native HT
workers (1–64); there is no parallel whole-tensor/expert queue. Start with
one or two workers on a memory-constrained machine. The tool does not
consult or modify production service state. Do not run full conversion on
a host whose live model already consumes its memory/disk headroom.

The real BF16 inputs are approximately 360 GB; native-v2 output plus the
FP8G table and vision tower is approximately 119 GB. The plan gives exact
HGN sizes. Conversion reserves an additional 5% plus 64 MiB of free disk
space; copied assets require their small additional space. No full-model
conversion timing or throughput measurement is available.

## Bounded structural checks

The public CLI has no fixture mode or architecture bypass. The library
entry point accepts an explicit structural plan, allowing tiny genuine
safetensors fixtures that exercise every codec, generated planes, an MTP
matrix, the stacked table, vision reshape, container writer, assets,
atomic publication, manifest and full verifier. These fixtures deliberately
use a non-engine identity/config and cannot be served as Flash-Next.

```sh
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 \
  cargo test --locked --manifest-path deployments/frame/bf16-to-hgn/Cargo.toml \
  -- --test-threads=1

CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 \
  cargo run --locked --manifest-path deployments/frame/bf16-to-hgn/Cargo.toml \
  --example tiny_conversion
```

The example uses one native worker and beam 8, performs real quantization
and conversion, verifies all payloads/source coverage, prints its report,
and removes its temporary inputs/output. Golden tests include the pinned
complete-container SHA-256 and format codec vectors. These checks do not
load a full model or establish inference quality.

Verified on 2026-10-09 with bounded CPU-only checks:

- **61 tests passed:** 50 unit, 2 container golden, and 9 integration tests.
- The real tiny conversion example consumed **13 source tensors**, wrote
  **18 output tensors / 63,604 payload bytes**, and passed full source/hash,
  layout and decoded-value verification.
- Metadata-only CLI planning passed against the complete base BF16 snapshot:
  **1,658 tensors in 131 shards**, **359,999,963,128 source payload bytes**,
  and output inventories **2,015 + 1 + 333**. The planned trunk size is
  **66,687,678,432 bytes**. No source tensor payloads were read by that plan.

These results do not include a full-model conversion, GPU/engine loading,
inference-quality validation, or a conversion-performance benchmark.
