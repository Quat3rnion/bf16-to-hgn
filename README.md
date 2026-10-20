# HF / HGN / GGUF → native v2 or HT43 HGN

A standalone Rust CLI with a vendored C++ trellis encoder. The default
output profile remains **v2**; `--profile ht43` selects the native HT43
store/layout contract. All weights come from the supplied inputs: no stock
checkpoint, MTP head, vision tower, n-gram table, learned codebook, or hidden
calibration data is substituted.

## Support boundary

Only complete **Qwen3.8 Flash-Next** weights are accepted:
`Qwen4ExpForConditionalGeneration`, `model_type=qwen4_exp`, and the fixed
Flash-Next configuration, canonical tensor inventory and shapes. Other
architectures, missing or unexpected tensors, incompatible configuration,
and unsupported encodings fail before output publication.

- **HF:** an indexed safetensors snapshot directory, including all split
  shards and required assets. Weights must be BF16; the three PLE hash
  vectors must be exact I64. F32 is accepted only for DeltaNet `A_log` and
  `dt_bias`. Quantized safetensors are not supported. HF gains and DeltaNet
  heads are already in canonical training order; no GGUF-style transform
  is applied. The rank-five vision patch kernel is flattened row-major.
- **HGN:** a model file or directory containing one model, with identity
  `qwen3.8-flash-next-v2` or `qwen3.8-flash-next-ht43`, container version
  1 or 2. Native append-only r2 files are supported: live payloads may be
  out of table order, with unreachable old payloads/gaps. Live spans must
  remain valid, aligned, disjoint and within the declared actual file size.
  Separate overlay application is not supported.
- **GGUF:** little-endian v2/v3 files with `general.architecture=qwen4exp`
  and compatible fixed Flash-Next metadata. Split filenames
  `model-00001-of-00003.gguf` discover all numbered siblings; any shard may
  be selected, but every shard must exist. Duplicate tensors and inconsistent
  metadata are rejected. Arbitrary shard naming and other architectures are
  not supported.

Supported **GGUF tensor types, exhaustively**:
`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q5_1`, `Q8_0`, `Q4_K`, `Q5_K`,
`Q6_K`, `IQ4_NL`, `IQ3_S`, `IQ4_XS`. All other types fail.

Supported **native HGN store/variant pairs, exhaustively**:
BF16 `0/0`, F32 `1/0`, F16 `2/0`, I64 `4/0`, Q4C-P `5/2`,
Q8G64 `7/0`, FP8G `10/0` (stacked n-gram table only), HT3 `16/0x1206`,
HT4 `16/0x1208`, I4R `23/128`, Q6G64 `24/64`. Other pairs are rejected,
including I32, Q8S32, Q6_K, IQ4_NL, Q4A/Q5A and GGUF-native n-gram stores
inside HGN. GGUF type support does **not** imply support for the corresponding
direct-import HGN store.

HGN/GGUF floats are decoded on demand to canonical F32 without an intermediate
BF16 rounding; I64 remains exact. HGN decoding undoes native rotations and
paired expert layouts. GGUF decoding applies norm-minus-one, DeltaNet
tiled-to-grouped head reorders, strictly negative F32 `ssm_a` → `ln(-A)`,
indexer Q/K and expert gate/up assembly, and exact PLE integer metadata.
An embedded GGUF n-gram table is exposed as canonical shards. These decoded
values, not an unavailable original BF16 checkpoint, are the conversion and
verification reference.

### Matching assets and auxiliary weights

HF contains its own assets and auxiliary weights; `--assets`, `--mtp`,
`--vision`, and `--ngram` are rejected for HF input.

For **GGUF**, `--assets`, `--mtp` and `--vision` are mandatory. MTP and vision
must be matching native HGN sidecars containing only their respective tensor
categories, not an entire trunk. `--ngram` is required if
`per_layer_token_embd.weight` is absent, and rejected if it is already embedded.

For **HGN**, `--assets` defaults to the model's parent directory. Categories
already embedded need no sidecar. Missing categories require explicit
`--mtp`, `--vision`, `--ngram`, or existing standard-named siblings
`qwen38-flash-next-{mtp,vision,ngram}.hgn`; duplicate sources fail. No search
for stock weights or fallback to another checkpoint occurs.

The assets directory must contain matching config/tokenizer files listed
below. A converter output can be reimported with
`--assets /path/to/previous-output/tokenizer`; its root is not a flat tokenizer
assets directory. To retain optional processor configs as well, supply a
matching flat assets directory containing them. Structure checks cannot prove
that independently supplied sidecars/assets came from the same checkpoint;
that provenance is the caller's responsibility.

The binary format is pinned to [hgn-spec 1.1.0](https://github.com/jtsylve/hgn-spec/tree/79e75fd4ae7642478823525b0a3212951f584af5)
(`79e75fd4ae7642478823525b0a3212951f584af5`).
**Format compatibility is not inference qualification.** Full-model HT43
conversion, inference quality, engine loading and performance have not been
validated here. The publisher's calibration, learned codebooks and rotation
policy are not public; this tool uses the explicit policy below.

## Requirements and build

Linux, Rust 1.85 or newer, a C++17 compiler with OpenMP (for example GCC and
`libgomp`), and sufficient disk space. The default build has no GPU dependency.
No model paths or Python runtime are required.
Linux publication uses `renameat2(RENAME_NOREPLACE)`; the output filesystem
must support it. A macOS no-replace publication implementation is present,
but its compiler/OpenMP toolchain is not qualified. Other OS publication is
rejected rather than weakened to a non-atomic overwrite-prone operation.

```sh
cargo build --release --locked --manifest-path deployments/frame/bf16-to-hgn/Cargo.toml
BIN=deployments/frame/bf16-to-hgn/target/release/bf16-to-hgn
```

`build.rs` compiles `native/ht_trellis.cpp` with C++17, `-O3`, OpenMP and
`-ffp-contract=off`, without `-march=native`. The vendored encoder has local
byte-preserving optimizations: runtime-dispatched AVX2 transition expansion
and a guarded packed cost/state comparator. Portable arithmetic/comparison
fallbacks remain. Notices and the original source attribution are under
`native/`. The conversion manifest records an aggregate SHA-256 of the
compiled tool's source files, lockfile and helper.

For optional AMD GPU encoding and verification, build with ROCm/HIP
and ordinary **libstdc++ 16 for both C++ translation units**:

```sh
# From this crate directory; separate output keeps existing binaries intact.
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo build --release --locked --features hip
HIP_BIN=/tmp/bf16-hgn-hip-target/release/bf16-to-hgn
$HIP_BIN convert --source /path/to/hf-bf16 --output /path/to/new-output \
  --profile ht43 --ht-backend hip --ht-beam 128
$HIP_BIN verify --source /path/to/hf-bf16 --output /path/to/new-output \
  --decode-backend hip
# v2 dense HT and routed I4R experts use independent backend axes.
$HIP_BIN convert --source /path/to/hf-bf16 --output /path/to/new-dual-output \
  --profile v2 --ht-backend hip --i4r-backend hip --ht-beam 128
$HIP_BIN verify --source /path/to/hf-bf16 --output /path/to/new-dual-output \
  --decode-backend hip
# I4R-only HIP encoding: HT stays on CPU; use another new output.
$HIP_BIN convert --source /path/to/hf-bf16 --output /path/to/new-i4r-output \
  --profile v2 --ht-backend cpu --i4r-backend hip
$HIP_BIN verify --source /path/to/hf-bf16 --output /path/to/new-i4r-output \
  --decode-backend cpu
```

`ROCM_PATH` defaults to `/opt/rocm`; `HIPCC` may override its compiler.
The HIP build targets `gfx1151`, preserves FP32 denormals, disables fast math
and FMA contraction, explicitly requests compiler-supported correctly rounded
binary32 division/square root (`-fhip-fp32-correctly-rounded-divide-sqrt`),
and links the ROCm HIP runtime. Its CPU and GPU searches
depend on the same libstdc++ 16 selection permutation; incompatible standard
libraries or parallel algorithms are rejected at build time.
The I4R HIP source handles AMD wave32 and wave64 execution; this source
portability is not device/toolchain qualification beyond gfx1151, and does
not change the build's default target.

For optional NVIDIA CUDA encoding and verification, build explicitly with
`--features cuda`. With the workhorse CUDA 12.0 toolchain, select GCC 12
rather than the system's newer compiler:

```sh
# From this crate directory; default builds remain CPU-only.
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo build --release --locked --features cuda
CUDA_BIN=/tmp/bf16-hgn-cuda-target/release/bf16-to-hgn
# v2 routed experts use I4R; dense HT and I4R are selected independently.
$CUDA_BIN convert --source /path/to/hf-bf16 --output /path/to/new-output \
  --profile v2 --ht-backend cuda --i4r-backend cuda --q4cp-backend cuda --ht-beam 128
$CUDA_BIN verify --source /path/to/hf-bf16 --output /path/to/new-output \
  --decode-backend cuda
# I4R-only CUDA encoding: HT stays on CPU; use a different new output.
$CUDA_BIN convert --source /path/to/hf-bf16 --output /path/to/new-i4r-output \
  --profile v2 --ht-backend cpu --i4r-backend cuda
$CUDA_BIN verify --source /path/to/hf-bf16 --output /path/to/new-i4r-output \
  --decode-backend cpu
```

`NVCC` selects the CUDA compiler; it must support the selected `CXX` host
compiler without an unsupported-compiler bypass. CPU and CUDA translation
units require the same ordinary **libstdc++ 12 or 16** selection permutation;
HIP still requires libstdc++ 16. `HGN_CUDA_ARCH` is a numeric compute
capability, default `89`, producing matching `sm_89` SASS and `compute_89`
PTX. Set it explicitly for a different target. Device preflight checks the
compiled kernels' compatibility with the visible device without launching
them; an incompatible image/device or unavailable runtime is an error.
CUDA disables FMA contraction and fast math, preserves FP32 denormals, and
requests precise division and square root. The historical HT encoding and
GPU decoding qualification below used CUDA 12.0/GCC 12, including cross-host
fidelity against Frame GCC 16; it is not qualification of the new I4R encoder.
A build alone is not hardware, byte-fidelity or performance qualification;
qualify different toolchains/devices separately.

## Commands

`--source-format hf|hgn|gguf` is optional when the index/file magic is
unambiguous. `inspect`, `plan` and `convert` accept `--profile v2|ht43`;
omitting it selects v2. `verify` derives the profile and encoding options
from `conversion-manifest.json`, not a CLI profile override.

`convert --i4r-backend cpu|hip|cuda` selects I4R forward rotation and quantization
independently of `--ht-backend cpu|hip|cuda`. Both encoder defaults are CPU,
even in GPU-feature builds. `--profile v2` uses I4R routed trunk experts;
`--profile ht43` retains its HT3/HT4 routed-expert policy. Backend choices
do not change codecs, profile or HT beam width. `plan` and `inspect` can
describe these choices without loading a GPU.

`verify --decode-backend cpu|hip|cuda` selects the verifier's runtime decoder
independently of the encoder backend recorded in the manifest. CPU is the
default. HIP and CUDA accelerate I4R and independent HT3/HT4 decoding and
inverse rotations, including paired HT bands. Other codecs, source import
decoding/I/O, SHA-256/XOR checks, layout/padding/metadata validation and
sequential F64 error accumulation stay on CPU. GPU verification does not
re-encode the weights or skip checks.

Conversion and verification draw a compact indicatif block on terminal
stderr, ordered from overall to local work and laid out again for the
terminal's current width at every redraw (every 250 ms):

```text
⠙ Convert passes  37% [===========>                  ] 512.00G/1.37T values
  elapsed 04:12:34 · ETA ~7h · 33.79M values/s
  encode HT3 on HIP · layers.12.mlp.experts.gate_up_proj.weight
  expert 37 of 512 · gate+up 384/640 row pairs (60%)
```

The first line is the stage and its counted work: a rounded-down percentage
of values (work units, not time), a gauge that never fills while the stage
runs, and counts. The second holds elapsed time, the ETA and the recent
aggregate rate. The third names the operation, encoding (`HT4`, `HT3`,
`I4R`, etc.), actual CPU/HIP/CUDA backend and full output tensor, whose
`layers.N`, `mtp.layers.N` or `visual.blocks.N` prefix distinguishes trunk,
MTP and vision layers. The fourth locates the work inside that tensor: a
zero-based expert index with the expert count, the gate/up portion, and rows,
row pairs (paired HT encoding) or bytes, updated at streaming chunk/band
boundaries. Nonexpert or flattened shapes do not invent an expert index.
Hashing and finalization replace that context with a single step line.

Lines shrink to the terminal: they drop optional fields (rate, gauge, counts,
gate/up label, item percentage) as needed and shorten the tensor name in the
middle, keeping its layer scope and as much of its end as fits, so this
tool's lines stay unwrapped down to 60 columns. ETA wording is never
shortened. The stage label is bold unless `NO_COLOR`, `CLICOLOR=0` or a
non-color terminal disables styling; no information depends on it.

A separate **Prepare rotations** stage counts generated signs and HT source
scale scans, with CPU row-scan updates before encoding/verification begins.
GPU selection does not imply GPU execution of this preparation stage.
JSON stdout is unchanged; redirected stderr remains quiet. Library callers
can explicitly supply disabled progress with no progress-only allocations.
“Convert passes” counts each output value once, plus n-gram values once more
for the mandatory global-scale scan; it is work across passes, not the
model's unique parameter count. Verification counts each output value once.
Conversion ETA sums the remaining value-work budget of each operation/pass,
encoding and actual backend at that class's observed mean chunk rate. HT4,
HT3, I4R, direct copies, rotation writes, and the n-gram scale scan/encode
passes learn independent rates; completed tensors' measurements remain
available when a class is revisited. Timed chunks include their source reads,
hashing, rotations, encoding and payload writes, not unrelated work or gaps
between chunks. No rate is borrowed from another codec or a previous run.

Before all remaining classes have samples, ETA says **learning (N classes)**
or **partial ~duration + N unmeasured**: the duration covers measured classes
only, not a full conversion estimate. File creation/hashing, asset copying,
provenance and publication are outside this value-work ETA and display
**finalizing (outside ETA)**, even between files that still have values left.
Preparation and verification keep their aggregate recent-rate ETA (`?` before
any measured progress) and likewise show **finalizing (outside ETA)** while
closing checks run after all counted values. A finished stage collapses to
one “done” line with its average rate; an error leaves a “stopped” line with
the last position and context. All ETAs are measurements-based estimates,
not completion promises.

A genuine temporary mixed HT4/HT3/I4R/n-gram conversion and verification smoke
is available with `cargo run --release --example tiny_conversion`. Run it in
a terminal with a non-dumb `TERM` to see stderr progress; stdout remains the
JSON verification report. Its structural fixture is not an engine-compatible
checkpoint and is removed when the example exits.

```sh
# HF: metadata inspection and deterministic plan, without tensor payload reads.
$BIN inspect --source-format hf --source /path/to/hf-bf16 --profile ht43
$BIN plan --source /path/to/hf-bf16 --profile ht43 \
  --seed 0 --threads 1 --ht-beam 128
$BIN convert --source /path/to/hf-bf16 --output /path/to/new-hf-v2 --dry-run

# HF real conversion: v2 is the default.
$BIN convert --source /path/to/hf-bf16 --output /path/to/new-hf-v2 \
  --seed 0 --threads 1 --ht-beam 128
$BIN verify --source /path/to/hf-bf16 --output /path/to/new-hf-v2

# Native HGN with embedded MTP and explicitly matching assets/ngram/vision.
$BIN convert --source-format hgn --source /path/to/native/model.hgn \
  --assets /path/to/matching-assets \
  --ngram /path/to/matching-ngram.hgn --vision /path/to/matching-vision.hgn \
  --profile ht43 --output /path/to/new-hgn-ht43
$BIN verify --source-format hgn --source /path/to/native/model.hgn \
  --assets /path/to/matching-assets \
  --ngram /path/to/matching-ngram.hgn --vision /path/to/matching-vision.hgn \
  --output /path/to/new-hgn-ht43
# Add --mtp /path/to/matching-mtp.hgn only if MTP is not embedded.

# Split GGUF with embedded ngram: all numbered sibling shards must be present.
$BIN convert --source-format gguf --source /path/to/model-00001-of-00003.gguf \
  --assets /path/to/matching-assets --mtp /path/to/matching-mtp.hgn \
  --vision /path/to/matching-vision.hgn \
  --profile ht43 --output /path/to/new-gguf-ht43
$BIN verify --source-format gguf --source /path/to/model-00001-of-00003.gguf \
  --assets /path/to/matching-assets --mtp /path/to/matching-mtp.hgn \
  --vision /path/to/matching-vision.hgn --output /path/to/new-gguf-ht43
# Add --ngram /path/to/matching-ngram.hgn only when the GGUF table is absent.
```

For HGN/GGUF, reuse the same source and auxiliary input flags for inspection,
planning, conversion and verification. Real conversion requires an existing
parent directory and a destination that does not exist.

`inspect`, `plan`, and `convert --dry-run` do not decode tensor payloads or
write files. They validate metadata, configuration, inventory, shapes, stores
and required assets; payload values/integrity are checked during real
conversion/verification. `verify` is a full source/output read, not a fast
metadata check. It checks headers, tables, sizes, output padding, variants,
SHA-256/XOR checksums, source coverage, assets, generated vectors, raw-copy
equality where applicable, finite quantized scales, and every decoded
quantized value against canonical source values. F64 squared-error totals
measure conversion error, not model quality or an acceptance threshold.

Strix Halo verification acceptance (2026-10-10): the rebuilt HIP verifier
checked Blackfrost's schema-1 conversion: 1,658 source tensors, 2,349 output
tensors and 118,785,459,868 payload bytes. All 1,531 numerical error records
matched the original CPU verifier bit-for-bit. The GPU run took 19m36s
versus the original CPU run's 22m19s, both with a 4 GiB memory cap, no swap
and a two-CPU quota; these shared-host runs are not an isolated benchmark.
Eleven hardware regressions passed, including exact CPU/GPU decode and
complete-report comparisons. A verification-only ROCm trace confirmed
HT3/HT4 row decoding and column unrotation kernels, not encoding kernels;
profiler timestamp warnings mean those trace durations are not benchmarks.
Terminal conversion/verification, quiet redirected JSON and a corrupted
payload's nonzero exit with “stopped” progress were exercised.

The same unchanged Blackfrost files passed Halogen 0.17.5 native verification
and isolated arithmetic, instruction, tool-call and vision smokes with no
NPU models loaded. Its thinking-forcing template required the explicitly
approved `HALOGEN_TEMPLATE_UNCHECKED=1` override. This is functional smoke
coverage, not a model-quality or full-context qualification.

All source files, sidecars and assets must remain immutable during conversion
and verification. Signatures and consumed-byte hashes are checked. Concurrent
external truncation of an mmap-backed HF input cannot be made safe by metadata
checks alone; HGN/GGUF use bounded positional reads.

## Output directory

```text
new-output/
  model.hgn                         # native v2 or HT43: trunk + MTP + rotations
  qwen38-flash-next-ngram.hgn        # qwen3.8-flash-next-ngram, stacked FP8G table
  qwen38-flash-next-vision.hgn       # qwen3.8-flash-next-vision, BF16 vision tensors
  config.json                       # supplied source config, unchanged
  tokenizer/                        # flat tokenizer files + unchanged source config
  conversion-manifest.json           # schema 2, profile and source provenance
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

The schema-2 manifest records the exact profile/plan, policy, seed/workers/beam,
format pin, encoder source digest, config/assets, canonical source tensors and
every output entry's bounds/type/variant/checksum/SHA-256. The HF index digest
is present only for HF inputs. `sources[].sha256` hashes canonical
little-endian dtype bytes: HF's actual BF16/F32/I64 bytes, or decoded/transformed
F32 and exact I64 for imports. An imported tensor's canonical dtype/size/bounds
describe its logical stream, not its physical encoded storage.

`source_provenance` separately records physical files, headers/tables,
encoded payload hashes and HGN rotation dependencies; canonical hashes are
never presented as raw HGN/GGUF payload hashes. There are no absolute source
paths, timestamps, credentials or assumed upstream repository. The HGN
identity denotes the engine format, not the checkpoint's provenance.

The writer produces schema 2. The verifier also reads **strict schema-1 HF
manifests** without modifying or migrating them. Schema 1 attests raw HF
safetensors tensor bytes, the required index, assets and output payloads;
it cannot attest HGN/GGUF sources or schema-2 physical provenance such as
shard-header hashes. Its profile is v2 and its encoder backend is CPU.
The verification report exposes `manifest_schema` and `provenance_attested`;
schema-1 provenance is explicitly unattested, never synthesized from the
current source. Missing provenance in a schema-2 manifest remains an error.

A staging directory is created beside the requested output. Payloads are
written at preplanned aligned offsets; header/table are finalized only
once all checksums are known. Files and directories are synced before
atomic no-clobber publication. An ordinary pre-publication failure removes
the staging directory; a failed parent-directory sync removes the newly
published output. Existing destinations and any overlap with the
source inputs/assets/sidecars are refused. The tool never modifies the input,
services, or a live checkpoint. A process killed without cleanup may leave an
unpublished `.bf16-to-hgn-*` staging directory; it is never treated as a
finished output or resumed automatically.

For serving, select the engine's matching native-v2 or native-HT43 loader for
`model.hgn`, its n-gram file and tokenizer. MTP weights are inside the output
`model.hgn`; no stock sidecar is required. Engine startup and inference must
be qualified separately on a safe host/outage before deployment.

## Output profiles and preservation

| Contract | v2 (default) | HT43 (`--profile ht43`) |
| --- | --- | --- |
| Main identity | `qwen3.8-flash-next-v2` | `qwen3.8-flash-next-ht43` |
| Routed trunk experts | I4R | HT4 in zero-based even layers, HT3 in odd layers; both gate/up and down |
| Trunk hyper-connection mixers | Q6G64 | Q8G64 |
| Other trunk codecs | Existing v2 policy | Same as v2 |
| MTP, n-gram, vision | Existing v2 policy | Same codec/layout policy as v2 |

Both profiles retain Q4C-P embeddings/small projections, HT4 trunk dense
matrices and LM head, BF16 gates/norms/vision, exact PLE I64, and source F32
`A_log`/`dt_bias`. MTP uses Q8G64 dense/mixer matrices and Q4C-P routed
experts. The output layout is selected by the profile, never by reusing a
source encoding.

HT43 experts use per-expert signed row-RMS scales. Gate/up share one input
sign vector for the layer; down input signs are per expert. Fused gate/up
payloads interleave corresponding 128-row gate and up trellis bands within
each expert, while canonical tensor/hash order remains gate then up per
expert. Down bands are ordinary expert-major bands. The output includes
freshly generated sign/scale planes.

“Preserved” means unchanged config/tokenizer bytes, exact integer values,
HF raw copies where dtype/layout match, and unchanged profile policy for
MTP/vision/n-gram—not preservation of imported compressed payloads.
**Every HGN/GGUF input is decoded and encoded anew**, even when input and
output profiles match. Imported raw BF16 output is rounded from canonical
F32; quantized output is requantized. Source rotations/codebooks are used
only to decode, not copied into the new model. Requantization adds error and
does not recover BF16 originals or promise stock-checkpoint equivalence.

## Deterministic quantization policy

Same input bytes, compiled tool and options produce the same payload bytes.
Cross-toolchain byte identity is **not** promised: C++ library tie traversal
in the approximate beam search can differ. The manifest pins the source
implementation; preserve the executable/toolchain as well when that level
of reproducibility matters.

- **BF16/F32/I64:** exact little-endian HF copies when output dtype matches.
  Imported floats round only as required by the selected output dtype.
  Integer values never pass through F32. Floating values must be finite;
  PLE hash vectors are checked against the stacked table's row bounds.
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
- **HT4/HT3:** input signs and signed per-row RMS scales rounded to F16; a
  nonzero RMS below F16 range uses the smallest F16 subnormal, and an
  actual zero row has scale zero. The native fixed-codebook two-pass
  boundary-fixed trellis beam search encodes 128-row bands with four or
  three bits respectively. The default beam is 128. This is approximate,
  not a global optimality claim. HT3 is used only for HT43 odd-layer experts.
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

Rust normally streams one tensor and 128 matrix rows at a time, with experts
processed individually and paired HT gate/up bands handled separately.
HIP and CUDA I4R conversion read at most **32 × 128-row bands per host batch**,
never crossing an expert boundary or a fused gate/up portion boundary;
tails remain complete bands. Both native GPU paths further tile each batch
into bounded rectangles of at most 256 independent 128×128 blocks. Neither
layer queues a whole tensor or all experts. The n-gram table uses 8,192-row
chunks and two passes; raw copies/hashes use at most 8 MiB chunks.
HGN/GGUF decoding uses bounded source windows, not an intermediate decoded
HF checkpoint or temporary full model. HF uses windowed mmap reads with
eviction hints; output dirty data is synced/evicted every 64 MiB and final
hashing is streamed.

Generated rotations are retained as F32 vectors across conversion/verification.
They are not all “small”: HT43's per-expert signs/scales require roughly
**440 MB**, and the full prepared rotation map approximately **450 MB
(429 MiB)**, once per conversion/verifier, not once per worker. Row buffers,
import decoder scratch and native per-worker trellis scratch are additional.
These are working-set bounds, not an exact RSS guarantee for every allocator/OS.

Default `--ht-backend cpu --i4r-backend cpu --threads 1` is deliberate.
`--threads` controls only CPU HT workers (1–64), not HIP or CUDA. HIP/CUDA
HT encoding accelerates HT3/HT4 trellis search; HT row-scale preparation and
forward rotation application remain on CPU. **HIP/CUDA I4R are the non-HT encoder
exception:** both axes of the full forward rotation and I4R quantization
run on GPU, without a CPU rotation pass. Sign generation, all other encoders,
source import decoding/I/O and sidecar resolution remain on CPU.
GPU selection does not change matching-sidecar rules or permit
stock-checkpoint fallback. GPU HT encoding supports beams 1–256; exact
beam 0/full-state search and larger library-only beams require CPU.
Requesting either GPU encoder without its feature, a compatible device or
working runtime is an error, never a silent CPU fallback; HIP/CUDA I4R failures
also propagate rather than retrying on CPU. Real conversion preflights both
selected encoders before creating staging output; inspect/plan do not
initialize a GPU.

HIP uses device 0 (respecting `HIP_VISIBLE_DEVICES`) and a least-priority
stream. HT encoding uses 64 tiles per batch and a reusable **27,607,808-byte
device allocation**. Independent lazy decode storage uses **55,115,776 bytes**,
and lazy I4R encoding storage uses **52,756,488 bytes**. All three together total
**135,480,072 bytes**, excluding HIP context/runtime/driver memory and host
buffers, not a total device-memory or RSS guarantee. I4R-only encoding does
not allocate the HT workspace. Calls share a serialized persistent
context and content-checked codebook cache. A low-priority stream is not
isolation: HIP still competes with inference for GPU compute and shared UMA.
There is no parallel whole-tensor/expert queue or automatic service management.
Do not run full conversion on a host whose live model consumes its headroom.

CUDA uses visible device 0 (`CUDA_VISIBLE_DEVICES` applies), restoring the
caller's current device after each call. Calls are synchronous and
mutex-serialized within persistent encoder/decoder contexts; concurrent
callers do not create parallel GPU queues. Codebooks are cached by content,
not pointer identity. HT encoding batches adapt to beam/bits, capped at 512
tiles, using **135,141,376 bytes** of reusable device storage. Independent
lazy decode storage uses **55,115,776 bytes**, and lazy I4R encoding storage
uses **52,756,488 bytes**. All three together total **243,013,640 bytes**;
these are native workspace bytes, excluding CUDA context/runtime/driver
memory and host buffers, not a total device-memory or RSS guarantee.
I4R-only encoding does not allocate the HT workspace. GPU selection does
not manage serving processes or reserve device headroom; run qualification
and conversion only where contention and memory use are safe.

The manifest records per-encoder `options.ht_backend` and
`options.i4r_backend`; either omitted field in an older schema-2 manifest
means CPU. These provenance fields are separate from verification-time
`--decode-backend`, which does not rewrite them. CPU-only builds can verify
HIP- or CUDA-produced output without a GPU. Container bytes must match the
same-toolchain CPU reference; the manifest legitimately differs in backend
metadata. Changing backends does not resume an unfinished conversion.

The BF16 source is approximately 360 GB. Including n-gram and vision, outputs
are approximately **119 GB for v2** or **110 GB for HT43** (decimal, excluding
assets). HT43's main file is about 57.618 GB, n-gram 51.200 GB and vision
0.898 GB. Imported F32 `A_log`/`dt_bias` add a few kilobytes. Logical decoded
F32 source sizes reported for imports are not temporary disk requirements.
`plan` gives exact per-file sizes for the chosen profile/source dtypes.
Conversion reserves an additional 5% plus 64 MiB of free disk space; copied
assets need their small additional space. No full-model HT43 timing or
throughput measurement is available.

## Encoder performance and byte fidelity

The optimized paths retain the same beam, scales, codebooks, divisions,
floating-point operation order and strict tie rules. HT transition expansion
uses runtime AVX2 without FMA; candidate selection keeps `nth_element` and
its original ordering through an exact packed comparator. I4R/Q6/Q8 use
bounded binary32 bias rounding instead of general round-to-even calls.
This is not a precision/profile change. NF4 and FP8 algorithms are unchanged;
a slower NF4 lookup rewrite was measured and rejected.

`examples/encoder_bench.rs` emits JSONL contract `encoder-bench-v2`, schema 2,
with input fingerprints, complete output/metadata/rotation/padding
fingerprints, and per-sample timings.
Timing includes public encoding, validation, codec allocations and rotation,
but excludes input loading, sign/row-RMS preparation, hashing and destruction.
It caps each input at 8,388,608 values and retained outputs at 128 MiB,
reads only selected HF BF16 bands, and checks equality across iterations,
samples, worker counts and backends. I4R additionally compares code and scale
bytes to concatenated single-band CPU encoding, outside the timed region.

`--backends cpu|hip|cuda` is the HT-only axis; I4R uses the independent
`--i4r-backends cpu|hip|cuda` axis (both default to CPU). `--i4r-bands 1..=32`
defaults to 1 and sets the number of contiguous 128-row bands encoded in
**one public call**, not the number of calls or a whole-model batch.
The input-value cap still applies: 32 bands fit at 256 columns, but not at
2560 columns. For real expert data, select rows wholly within one expert
and gate/up portion. I4R has no HT beam/thread search axis.

Samples retain HT `selected_backend`, add I4R `selected_i4r_backend` and
`i4r_bands_per_call`, and report the actual `effective_backend`.
Non-HT codecs other than explicitly selected HIP/CUDA I4R stay on CPU even
on a GPU-selected HT axis. Their
`timed_region.gpu_host_device_copies_and_synchronization` is false; it is
true for effective HIP/CUDA HT or HIP/CUDA I4R encoding.
`timed_region.gpu_device_preflight` is always false. The run record's
`gpu_preflight` array reports `{backend, elapsed_ns}` separately for each
selected HT GPU, including a non-HT-only run; independent HIP/CUDA I4R entries
also have `encoding: "i4r"`. `gpu_preflight_before_samples` and `gpu_timing`
describe startup cost and measured scope. Schema 2 replaces the schema-1
HIP-specific timing fields.

Rates below are **million input values/second (Mvalues/s)**, calculated as
`values_per_call / (median_ms × 1000)`, not encoded bytes/second.
Paired HT calls count both projections (655,360 values at 128×2560 each);
single bands count 327,680 values. Historical rates are approximate,
derived from their displayed rounded median times. Multi-iteration samples
report average elapsed time per call before taking the sample median.

```sh
CARGO_BUILD_JOBS=1 cargo build --release --locked --example encoder_bench
BENCH=target/release/examples/encoder_bench
$BENCH --suite corpus --threads 1,2
$BENCH --cases ht4,ht3,ht4-pair,ht3-pair --cols 256 \
  --threads 1,2 --iterations 1 --repetitions 3 --warmup 1
$BENCH --cases i4r,nf4,q6g64,q8g64,fp8g128 --cols 2560 \
  --threads 1 --iterations 32 --repetitions 3 --warmup 1

# One real expert: gate rows 0..128, up rows 640..768.
$BENCH --hf /path/to/hf-bf16 \
  --tensor model.language_model.layers.0.mlp.experts.gate_up_proj \
  --up-tensor model.language_model.layers.0.mlp.experts.gate_up_proj \
  --up-row-start 640 --cases ht4,ht3,ht4-pair,ht3-pair \
  --threads 2 --iterations 1 --repetitions 2 --warmup 0
```

For A/B comparisons, build the **same harness source** against both revisions
with identical compiler/options, use fixed physical cores with idle SMT
siblings, and alternate baseline/candidate/candidate/baseline. Compare all
input and output fingerprints before interpreting medians and ranges.
The harness itself does not compare separate processes/revisions.

Measured against `a0157bf` on Frame's Ryzen AI MAX+ 395, Rust 1.98.1,
GCC 16.2.1, normal release flags, CPUs 10/11, beam 128:

| Case | Baseline median ms (range) | Optimized median ms (range) | Baseline Mvalues/s | Optimized Mvalues/s | Throughput ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real Blackfrost HT4 | 6790 (5948–7682) | 4529 (3936–4928) | 0.0483 | 0.0724 | 1.50× |
| Real Blackfrost HT3 | 3800 (3399–5286) | 2802 (2625–3561) | 0.0862 | 0.1169 | 1.36× |
| Real Blackfrost HT4 gate/up | 12773 (11439–13919) | 9333 (8676–9695) | 0.0513 | 0.0702 | 1.37× |
| Real Blackfrost HT3 gate/up | 7121 (6728–7708) | 5728 (5098–6448) | 0.0920 | 0.1144 | 1.24× |
| Synthetic I4R | 15.681 (13.378–17.842) | 6.754 (5.994–7.653) | 20.897 | 48.516 | 2.32× |
| Synthetic Q6G64 | 2.426 (2.340–2.782) | 2.320 (2.040–2.636) | 135.070 | 141.241 | 1.05× |
| Synthetic Q8G64 | 2.053 (1.935–3.983) | 1.820 (1.696–2.420) | 159.610 | 180.044 | 1.13× |

Real bands are 128×2560 per projection, two HT workers, four samples per
revision, from Blackfrost revision `3e30d25ddb0e802389d62e55e9ff31ad05f0c96e`.
Synthetic quantizers use 128×2560, one worker, six samples per revision,
32 calls/sample. All measured output fingerprints matched. Independent native
checks also matched 216 baseline/AVX2/forced-scalar cases across HT3/4,
beams 0/1/8/64/128/256, one/two workers, repeated ties and subnormals.
The scalar override tests arithmetic, not the packed comparator's fallback.
The final rebuilt candidate matched all 238 baseline corpus samples across
nine codec cases. The integrated suite passed 124 tests (one optional stock
probe ignored), and all four tiny conversion/publication/verifier executables
passed after the optimizations.

This was a **shared, non-idle host**, not an isolated performance lab.
Earlier samples suffered severe contention; even unchanged NF4/FP8 controls
varied, and the small Q6/Q8 differences overlap noise. Do not extrapolate
these encoding-only results into whole-model throughput or an ETA.
Baseline profiling attributed about 66% of HT4 cycles to candidate selection
and 26% of I4R cycles to general rounding; selection, rotation, input decoding
and output I/O remain costs. Full-model conversion/inference was not run,
and no active conversion or serving process was replaced.

### Strix Halo HIP backend

HIP preserves the CPU selection permutation, tie handling, closure and
separate binary32 subtraction/multiplication/addition; it does not reduce
beam width or precision. The tested reference is GCC/libstdc++ 16.2.1 and
ROCm 7.2 on gfx1151. Native differential coverage checks every beam 1–256
for HT3/HT4, batch boundaries, ties, subnormal squares, changing codebooks
and concurrent callers. The GPU selection regression passed 5,896 full
candidate-permutation and forced-heap-fallback cases against libstdc++.
All five opt-in device/conversion tests passed. The executable tiny smoke
published byte-identical CPU/HIP containers; a separate CPU-only executable
also verified them with GPU visibility disabled.

```sh
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo build --release --locked --features hip --example encoder_bench
HIP_BENCH=/tmp/bf16-hgn-hip-target/release/examples/encoder_bench
$HIP_BENCH --suite corpus --backends cpu,hip --threads 1,2 \
  --cases ht4,ht3,ht4-pair,ht3-pair --corpus-beams 1,8,128,256
# Add --backends cpu,hip --timing-beam 128 to the real-band command above.
```

HIP timings include host/device copies and synchronization. Device preflight
is reported separately; warm samples reuse workspace and codebook. Use a
fresh process with `--backends hip --warmup 0` for first-encoding cost, and
include the reported preflight for startup-inclusive cost. Small batches or
small beams may be slower on GPU. Host-side preparation, other codecs and
I/O are not accelerated; band results are not a whole-model ETA.

Measured on Frame before unloading production Halogen: real Blackfrost
128×2560 bands, beam 128, CPUs 10/11, two CPU workers, four warm samples per
backend/case across CPU→HIP and HIP→CPU process orders:

| Case | CPU median ms (range) | HIP median ms (range) | CPU Mvalues/s | HIP Mvalues/s | Throughput ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| HT4 | 3128 (3085–3476) | 862 (423–1327) | 0.1048 | 0.3801 | 3.63× |
| HT3 | 2120 (2024–2243) | 450 (301–666) | 0.1546 | 0.7282 | 4.71× |
| HT4 gate/up | 6460 (6119–6966) | 845 (844–847) | 0.1014 | 0.7756 | 7.64× |
| HT3 gate/up | 4336 (4035–5146) | 914 (601–1273) | 0.1511 | 0.7170 | 4.74× |

All 64 timing samples (including the separate one-worker axis) matched
complete input/output fingerprints across processes. The public corpus
matched 448 samples across both backends, one/two workers, seven fixtures
and beams 1/8/128/256. Native comparisons matched 640 cases covering every
supported beam. Device preflight took 200–328 ms, excluded from this table.
The shared GPU was not idle: wide HIP ranges reflect contention, and the
unusually tight paired-HT4 subset must not be generalized. No full-model
GPU conversion or inference qualification was performed.

### Strix Halo HIP I4R encoder

HIP I4R performs the complete forward Hadamard rotation on GPU with ordered
FP64 butterflies, followed by the CPU-equivalent signed-extreme scale search,
F16 rounding, nearest-even codes, sequential FP64 error sums and strict
first-wins ties. It is independent of the HT backend and remains opt-in.
Wave32/wave64 source support is not qualification on other AMD devices.
The HT table above measures a different encoder; the integrated HIP I4R
qualification and bounded measurements below are separate observations.

Use the HIP benchmark executable built above:

```sh
# Independent I4R correctness corpus; HT remains on CPU.
$HIP_BENCH --suite corpus --fixtures all --cases i4r --backends cpu \
  --i4r-backends cpu,hip --i4r-bands 1 --threads 1,2
# Full public-call rotation/quantization timing, one 128x2560 band.
$HIP_BENCH --suite timing --cases i4r --cols 2560 --backends cpu \
  --i4r-backends cpu,hip --i4r-bands 1 --threads 1,2 \
  --iterations 1 --repetitions 5 --warmup 2
# Sixteen contiguous bands (2048x2560) in one public call.
$HIP_BENCH --suite timing --cases i4r --cols 2560 --backends cpu \
  --i4r-backends cpu,hip --i4r-bands 16 --threads 1,2 \
  --iterations 1 --repetitions 5 --warmup 2
# The 32-band bound at 256 columns stays below the input-value cap.
$HIP_BENCH --suite timing --cases i4r --cols 256 --backends cpu \
  --i4r-backends cpu,hip --i4r-bands 32 --threads 1 \
  --iterations 1 --repetitions 3 --warmup 1
# Real BF16 gate portion: five bands within one expert/portion.
$HIP_BENCH --hf /path/to/hf-bf16 \
  --tensor model.language_model.layers.0.mlp.experts.gate_up_proj \
  --row-start 0 --cases i4r --backends cpu --i4r-backends cpu,hip \
  --i4r-bands 5 --threads 1 --iterations 1 --repetitions 3 --warmup 1
```

The one-/16-band timing recipes match the measured synthetic geometries below;
the 32-band and five-band real-input recipes were not timed in that table.
Timings include public validation, codec allocation, full forward rotation, quantization and
synchronous GPU H2D/kernel/D2H copies. Input loading/I/O, sign preparation,
converter setup/output writes/hashing, output destruction, GPU preflight
and the single-band CPU byte oracle are outside timing. Warm calls reuse
native workspace. Use a fresh process with `--i4r-backends hip --warmup 0`
for first-encoding cost, adding its separately reported I4R preflight for
startup-inclusive cost. `--threads` and HT beam width do not parallelize
CPU I4R or change I4R's search. Band geometry and timing boundaries must
accompany any speed claim; these are not whole-model conversion/ETA recipes.

Bounded integrated qualification on 2026-10-12 used Frame's AMD Radeon 8060S
(`gfx1151`), ROCm HIP 7.2.53211, GCC/libstdc++ 16.2.1 and Rust 1.98.1.
Encoder source SHA-256 was
`892c2e6a9e7e2cf9bb9f56ca6b5e1deefc7537c9dff9f58cf1e2684c8e46e9c4`.
Commands used a two-CPU quota, 2 GiB memory-high/3 GiB memory-max, zero
per-command swap and nice 10. The host was shared with active Halogen;
the existing HT43 job remained frozen. Neither service was replaced.

Each timing row is the median of **five single-call samples after two
warmups**, with observed min–max in parentheses. Both `--threads 1` and `2`
axes are shown: CPU I4R rotation/quantization is serial, so this is **not**
a one-/two-thread scaling benchmark. Synthetic samples were grouped CPU
threads 1/2 then HIP threads 1/2, not an alternating isolated A/B comparison.
All 220 timing samples and 56 correctness-corpus samples matched CPU byte
oracles. Corpus timings are correctness evidence, not warm speed benchmarks.

| Synthetic geometry per call | Threads axis | CPU median ms (range) | HIP median ms (range) | CPU Mvalues/s | HIP Mvalues/s | CPU/HIP ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 band, 128×2560 | 1 | 4.113592 (4.088465–4.330941) | 0.330150 (0.318689–0.335831) | 79.658 | 992.519 | 12.460× |
| 1 band, 128×2560 | 2 | 4.174035 (4.093765–4.314629) | 0.328317 (0.316244–0.356900) | 78.504 | 998.060 | 12.713× |
| 16 bands, 2048×2560 | 1 | 68.855964 (68.489305–69.867625) | 4.424977 (4.421271–4.577804) | 76.143 | 1184.838 | 15.561× |
| 16 bands, 2048×2560 | 2 | 68.963396 (68.313354–69.116493) | 4.611578 (4.526288–5.180677) | 76.024 | 1136.895 | 14.954× |

Separate synthetic I4R preflights were **57.536497 ms** (one band) and
**82.591472 ms** (16 bands), excluded from these warm medians. Correctness
corpus preflights were 223.588083 ms and 63.169887 ms respectively.

The original Orca HF snapshot was unavailable as a benchmark source because
it lacked `config.json`. Nine small **outside-repository, benchmark-only HF
fixtures** were reconstructed losslessly from the retained real BF16 band
exports, not from synthetic values and not by modifying the original snapshot.
Retained BF16 SHA-256, decoded F32, sign and complete CPU output fingerprints
were checked. These use the retained seed-123 benchmark signs, restarted per
band, **not production full-tensor signs**. Original upstream shard provenance
was retained rather than rereading complete upstream shards. The temporary
fixtures are not full-architecture checkpoints or a public CLI bypass.

All nine real bands and both thread axes are shown below; row starts refer
to the original flattened tensor, while the temporary fixture starts at row 0.
Gate/up calls count 327,680 input values; down calls count 81,920.

| Real Orca band (layer / expert / projection / original row start) | Threads axis | Geometry per call | CPU median ms (range) | HIP median ms (range) | CPU Mvalues/s | HIP Mvalues/s | CPU/HIP ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 / gate / 0 | 1 | 128×2560 | 6.845055 (5.942518–7.438710) | 1.340189 (0.487896–4.159437) | 47.871 | 244.503 | 5.108× |
| 0 / 0 / gate / 0 | 2 | 128×2560 | 5.608861 (4.957066–6.185946) | 0.462219 (0.443273–0.469803) | 58.422 | 708.928 | 12.135× |
| 0 / 0 / up / 640 | 1 | 128×2560 | 7.333192 (6.491870–7.790652) | 2.360176 (0.522402–10.966550) | 44.684 | 138.837 | 3.107× |
| 0 / 0 / up / 640 | 2 | 128×2560 | 7.030373 (6.115222–8.007669) | 0.476035 (0.452901–2.172493) | 46.609 | 688.353 | 14.769× |
| 0 / 0 / down / 0 | 1 | 128×640 | 2.711976 (2.709683–2.755018) | 0.243848 (0.237616–0.272291) | 30.207 | 335.947 | 11.122× |
| 0 / 0 / down / 0 | 2 | 128×640 | 2.702269 (2.699974–2.710754) | 0.251974 (0.227388–0.262894) | 30.315 | 325.113 | 10.724× |
| 23 / 255 / gate / 326400 | 1 | 128×2560 | 8.848951 (6.964449–9.286814) | 0.377259 (0.365427–0.556376) | 37.030 | 868.581 | 23.456× |
| 23 / 255 / gate / 326400 | 2 | 128×2560 | 8.530392 (7.633706–9.824464) | 0.635905 (0.365056–14.468793) | 38.413 | 515.297 | 13.415× |
| 23 / 255 / up / 327040 | 1 | 128×2560 | 11.052301 (11.025691–11.508379) | 1.210054 (0.630084–4.168033) | 29.648 | 270.798 | 9.134× |
| 23 / 255 / up / 327040 | 2 | 128×2560 | 10.851935 (10.813553–10.884937) | 0.460586 (0.449304–0.474091) | 30.196 | 711.442 | 23.561× |
| 23 / 255 / down / 654080 | 1 | 128×640 | 1.118381 (1.101841–1.128080) | 0.189787 (0.187222–0.192932) | 73.249 | 431.642 | 5.893× |
| 23 / 255 / down / 654080 | 2 | 128×640 | 1.242144 (1.192181–1.257753) | 0.189416 (0.187402–0.236926) | 65.950 | 432.487 | 6.558× |
| 47 / 511 / gate / 654080 | 1 | 128×2560 | 10.390457 (8.339523–10.832258) | 1.727888 (0.564351–8.729035) | 31.537 | 189.642 | 6.013× |
| 47 / 511 / gate / 654080 | 2 | 128×2560 | 10.813452 (9.252509–10.835855) | 0.582806 (0.552568–2.951808) | 30.303 | 562.245 | 18.554× |
| 47 / 511 / up / 654720 | 1 | 128×2560 | 10.837437 (10.149565–11.088179) | 0.564180 (0.545515–0.579609) | 30.236 | 580.808 | 19.209× |
| 47 / 511 / up / 654720 | 2 | 128×2560 | 10.960008 (10.800207–11.175183) | 0.592584 (0.562688–0.635946) | 29.898 | 552.968 | 18.495× |
| 47 / 511 / down / 1310592 | 1 | 128×640 | 1.077345 (1.064952–1.130966) | 0.767964 (0.274757–5.564067) | 76.039 | 106.672 | 1.403× |
| 47 / 511 / down / 1310592 | 2 | 128×640 | 1.130154 (1.114715–1.146054) | 0.208983 (0.188123–2.127659) | 72.486 | 391.994 | 5.408× |

Real-band I4R preflights, also untimed, were gate/up/down respectively:
layer 0 **199.775665 / 58.357577 / 63.022403 ms**;
layer 23 **82.619373 / 90.950190 / 81.902767 ms**;
layer 47 **85.462507 / 160.284686 / 90.304657 ms**.
Real median ratios span **1.403–23.561×** on this visibly variable shared host.
No best-case cherry-pick or controlled CUDA-versus-HIP ranking follows from
these separately collected runs. No whole-model conversion throughput, ETA,
deployment, inference-quality or inference-throughput claim follows either.

The HIP all-target suite passed **187 tests (20 ignored)**. Five focused
I4R device cases, 12 existing library regressions and three conversion cases
passed; one conversion case overlaps the focused run (**19 unique opt-in
cases**, not 20). Both I4R-only and dual HIP tiny conversions produced three
byte-identical CPU/HIP containers, identical CPU/HIP numerical verification
reports, and matching feature-free CPU source verification. Main's separate
dual-HIP smoke also matched all three containers and CPU/HIP reports.
The final CPU all-target suite passed **190 tests (1 ignored)**; post-HIP CUDA
checks passed **187 tests (20 ignored)** and seven focused cases, plus dual
container parity and feature-free CPU verification. These are bounded
structural fixtures, not engine-loadable full-model qualification.

Actual `encode I4R on CPU` and `encode I4R on HIP` progress labels were
observed during real tiny conversion/verification in a 100-column PTY.
A disclosed 60 ms `strace` delay on `lseek` exposed short stages; this is
visual proof, not timing or broader terminal-width qualification.
Local receipts are retained at
`/home/quaternion/models/bf16-to-hgn-hip-i4r-qualification-20261012T084946Z/`
(`qualification-result.json`, `summary.json`, `tiny-parity.json`,
`retained-real-fixtures.json` and `pty-label-proof.json`).

### NVIDIA CUDA HT backend — historical qualification

HT CUDA uses the same beam, CPU selection permutation, strict ties and
separate binary32 operations; it does not trade precision or search width
for speed. The historical bounded HT encoding/GPU decoding qualification
on 2026-10-12 used CUDA 12.0.140, GCC 12.4.0, Rust 1.98.1 and driver
595.91.07 on the workhorse RTX 4090 Laptop GPU (`sm_89`). Its source encoder
SHA-256 was
`905faf22ddf6d22bc227132c2b07f54d2bfd2d85266909539192a31234ef1788`,
matching Frame at that revision. These results predate the integrated CUDA
I4R encoder and do not qualify it. Different toolchains/devices still
require qualification; HIP measurements above are separate observations.

```sh
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo build --release --locked --features cuda --example encoder_bench
CUDA_BENCH=/tmp/bf16-hgn-cuda-target/release/examples/encoder_bench
$CUDA_BENCH --suite corpus --backends cpu,cuda --threads 1,2 \
  --cases ht4,ht3,ht4-pair,ht3-pair --corpus-beams 1,8,128,256
# Add --backends cpu,cuda --timing-beam 128 to the real-band command above.
```

Observed median public-encoding times for synthetic 128 × 2560 bands,
beam 128, one CPU thread, three samples after one warmup:

| Case | CPU median (ms) | CUDA median (ms) | CPU Mvalues/s | CUDA Mvalues/s | CPU/CUDA ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| HT3 | 8977.939 | 120.974 | 0.0365 | 2.709 | 74.2× |
| HT4 | 10942.228 | 197.998 | 0.0299 | 1.655 | 55.3× |
| HT3 pair | 14237.961 | 222.159 | 0.0460 | 2.950 | 64.1× |
| HT4 pair | 27443.945 | 409.626 | 0.0239 | 1.600 | 67.0× |

Paired cases encode two such bands. All timed CPU/CUDA samples were
bit-identical. The timed region includes public validation, codec allocation,
rotation and synchronous GPU copies/kernels; it excludes source I/O, input
setup, device preflight, hashing and output destruction. This run's CUDA
preflight took **163.134074 ms**, separately from the table. Use a fresh
process with `--backends cuda --warmup 0` for first-encoding cost and add its
reported preflight for startup-inclusive cost. Warm samples reuse native
storage/codebooks.

These observations came from a **shared Ubuntu VM and GPU**, not an isolated
throughput run. An unrelated `render_raytracing` process was observed after
the run, when GPU utilization was 99%; this is not a measurement of its
activity during the samples. The ratios describe only these synthetic
128-row bands, not whole-model conversion, verification or inference
speedups. All services were untouched. Matching driver userspace was
isolated for qualification, with no global driver/package or `ldconfig`
changes. Local evidence (not a durable publication) is retained at
`/tmp/bf16-hgn-cuda-crosshost/workhorse/qualification-summary.json`.

### NVIDIA CUDA I4R encoder

The integrated I4R path performs the complete forward Hadamard rotation
on GPU with ordered FP64 butterflies, then the same signed-extreme scale
search, F16 rounding, nearest-even codes, sequential FP64 error sums and
strict first-wins ties as CPU. CPU remains the default; selecting CUDA
does not reduce precision or fall back to CPU.

Use the CUDA benchmark build above for these recipes. The one- and 16-band
synthetic timing cases were exercised in the qualification reported below:

```sh
# Synthetic correctness corpus, independent of the HT backend axis.
$CUDA_BENCH --suite corpus --cases i4r --backends cpu \
  --i4r-backends cpu,cuda --i4r-bands 1 --threads 1
# Synthetic public-call timings; includes full rotation and synchronous copies.
$CUDA_BENCH --cases i4r --cols 2560 --backends cpu \
  --i4r-backends cpu,cuda --i4r-bands 1 --threads 1 \
  --iterations 1 --repetitions 3 --warmup 1
# Synthetic 16-band public-call timing, matching the measured batch geometry.
$CUDA_BENCH --cases i4r --cols 2560 --backends cpu \
  --i4r-backends cpu,cuda --i4r-bands 16 --threads 1 \
  --iterations 1 --repetitions 3 --warmup 1
# 32-band synthetic batching case; narrower columns keep input below the cap.
$CUDA_BENCH --cases i4r --cols 256 --backends cpu \
  --i4r-backends cpu,cuda --i4r-bands 32 --threads 1 \
  --iterations 1 --repetitions 3 --warmup 1
# Real BF16 gate portion: five 128-row bands of one expert, not synthetic input.
$CUDA_BENCH --hf /path/to/hf-bf16 \
  --tensor model.language_model.layers.0.mlp.experts.gate_up_proj \
  --row-start 0 --cases i4r --backends cpu --i4r-backends cpu,cuda \
  --i4r-bands 5 --threads 1 --iterations 1 --repetitions 3 --warmup 1
```

Preflight and the single-band CPU byte oracle are untimed; warm samples
reuse native workspace. Use a fresh process with
`--i4r-backends cuda --warmup 0` for first-encode cost and add reported I4R
preflight for startup-inclusive cost. The HT beam setting does not affect
I4R. The 32-band and contiguous five-band real-input commands above are
recipes, not the geometries measured in the following table.

Bounded **integrated production-code** qualification on the workhorse CUDA
host on 2026-10-12 used the RTX 4090 Laptop GPU (`sm_89`), CUDA 12.0.140,
GCC 12.4.0, Rust 1.98.1 and driver 595.91.07, under a two-CPU quota,
4 GiB memory cap and no swap. Encoder source SHA-256 was
`a59f3218829380ad2f96b9ccc10fa6f1a9785aab4e74b6c157746c3a7c1b1100`.
These observations use the production public encoder, not the earlier
standalone prototype:

| Synthetic geometry per call | CPU median (ms) | CUDA median (ms) | CPU Mvalues/s | CUDA Mvalues/s | CPU/CUDA ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 band, 128×2560 | 4.571640 | 0.422171 | 71.677 | 776.178 | 10.83× |
| 16 bands, 2048×2560 | 109.438385 | 5.552646 | 47.907 | 944.213 | 19.71× |

Both cases used one CPU thread, three repetitions after one warmup, and
byte-identical CPU/CUDA fingerprints. Timings include public validation,
allocation, the full forward rotation, quantization and synchronous GPU
H2D/kernel/D2H copies; they exclude input loading/I/O, input/sign preparation,
converter setup/output writes/hashing, GPU preflight and the CPU byte oracle.
Their separate CUDA preflights took **94.482573 ms** (one band) and
**91.161707 ms** (16 bands); neither is included in the medians.
The host/GPU were shared; these are bounded warm public-call measurements,
not isolated whole-model conversion or inference speedups.

Nine retained real Orca gate/up/down bands also matched both CUDA and the
retained Frame CPU code/scale hashes. Their signs were the benchmark's
seed-123 signs, restarted per band, **not production full-tensor signs**.
The retained extracted inputs/signs were hash-checked; upstream model-shard
provenance was retained, not reverified. This is real-data encoder evidence,
not full-model fidelity or inference-quality qualification.

The same integrated encoder also measured these real inputs, one public
call per sample, **six repetitions after two CPU/CUDA warmup pairs**.
Timed order alternated CPU→CUDA and CUDA→CPU (three pairs each way).
The harness passed `threads=2`, but production CPU I4R rotation/quantization
is serial; this is not a two-thread speedup. I4R has no beam search.
All rows use the same complete-call timing boundary above, with output
destruction/hash comparison and batch construction outside timing; CUDA
preflight was separately **125.294505 ms**.

| Real Orca band (layer / expert / projection / flattened row start) | Geometry per call | CPU median (ms) | CUDA median (ms) | CPU Mvalues/s | CUDA Mvalues/s | CPU/CUDA ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 0 / gate / 0 | 128×2560 | 8.1091755 | 0.6013525 | 40.409 | 544.905 | 13.48× |
| 0 / 0 / up / 640 | 128×2560 | 13.8168715 | 1.0813485 | 23.716 | 303.029 | 12.78× |
| 0 / 0 / down / 0 | 128×640 | 2.3486050 | 0.2503545 | 34.880 | 327.216 | 9.38× |
| 23 / 255 / gate / 326400 | 128×2560 | 14.9540965 | 1.2237065 | 21.912 | 267.777 | 12.22× |
| 23 / 255 / up / 327040 | 128×2560 | 14.2310365 | 1.0858435 | 23.026 | 301.775 | 13.11× |
| 23 / 255 / down / 654080 | 128×640 | 3.4466350 | 0.3005510 | 23.768 | 272.566 | 11.47× |
| 47 / 511 / gate / 654080 | 128×2560 | 13.3696520 | 0.9136140 | 24.509 | 358.664 | 14.63× |
| 47 / 511 / up / 654720 | 128×2560 | 11.9377735 | 0.6846490 | 27.449 | 478.610 | 17.44× |
| 47 / 511 / down / 1310592 | 128×640 | 2.1410865 | 0.2128100 | 38.261 | 384.944 | 10.06× |
| Gate band 0 / 0 / 0 repeated 16 times | 2048×2560 | 119.5972420 | 7.4305140 | 43.838 | 705.588 | 16.10× |

The last row is **16 copies of one retained real band in one call**
(5,242,880 values), not 16 distinct bands or a contiguous real expert
slice. Every measured code/scale plane matched CPU; the nine original bands
also matched their retained Frame oracles.

For the nine original bands only (2,211,840 values), summing corresponding
repetition-index timings across independently warmed cases and then taking
each backend's median gives **85.9036895 ms CPU / 6.8872210 ms CUDA**,
or **25.748 / 321.151 Mvalues/s**, a **12.47×** ratio.
This is a **median of six index-wise sums, not a continuously timed
nine-band pass**, and excludes the repeated 16-band case. The real-band
times vary substantially on the shared, uncontrolled host/GPU; they neither
establish isolated performance nor a comparison with the earlier prototype,
and are not full-converter, full-model or inference speed measurements.

The earlier CUDA all-target suite passed 182 tests (20 ignored); all 19
selected opt-in device tests passed. The final Frame CPU all-target suite
passed **186 tests (1 ignored)**, including the fixture alignment regression.
The earlier HIP-feature suite passed 182 (15 ignored; HIP device tests were
not run); the corrected fixture's rebuilt GPU smokes also passed.
I4R-only and dual HT/I4R CUDA tiny conversions each produced three
byte-identical CPU/CUDA containers. Complete CPU/CUDA numerical verification
reports matched, and feature-free CPU executables verified both outputs.
The public CLI retained architecture rejection; these genuine structural
fixtures are not architecture-valid full models or engine-loadable weights.

Actual `encode I4R on CPU` and `encode I4R on CUDA` progress labels were
observed in 120-column PTYs using a disclosed 60 ms `strace` delay on `lseek`
to expose short stages. Delayed stdout matched undelayed stdout; those
delayed runs are display checks, not performance measurements.

The qualification archive is retained locally at
`/home/quaternion/models/i4r-cuda-production-qualification-20261012T073944Z/qualification-evidence.tar.gz`
(`evidence/qualification-summary.json` and `evidence/real-band-summary.json`).
Prototype figures and the historical HT table do not establish I4R production
performance. No entire-model conversion, deployment, inference-quality or
inference-throughput claim follows from these bounded checks.

### NVIDIA CUDA Q4C-P encoder and HT search rework (2026-10-12)

`--q4cp-backend cuda` or `--q4cp-backend hip` encodes Q4C-P (NF4) groups on
the GPU. It is
independent of `--ht-backend` and `--i4r-backend`. CPU stays the default.
The manifest records `q4cp_backend`. Older schema-2 manifests without the
field verify as CPU. There is no CPU fallback: a CUDA error stops the
conversion. One thread encodes one 32-value group. The kernel keeps the CPU
order of operations: binary32 scale and division, sequential FP64 error sums,
strict first-wins candidate choice and lower-level midpoint ties. If a scale
is not a finite F16, the encoder runs the CPU reference again to get the
exact CPU error text.

The CUDA HT3/HT4 search has these changes. Each change keeps CPU byte
identity:

1. **Persistent grid.** The kernel starts only the number of blocks that can
   be resident at the same time. Each block encodes a series of tiles. The old
   launch used one block per tile with 512 tiles per launch. That gave a
   partly empty second wave in each launch.
2. **Coalesced access.** Each warp owns one contiguous segment of positions.
   Its lanes read the segment interleaved. Ballot ranks keep the compaction
   in ascending history order.
3. **Set selection with exact tie replay.** Candidate keys (cost, history) are
   unique. Thus the beam set is unique and does not depend on libstdc++
   `nth_element` order. A radix select finds this set and keeps it in history
   order. The libstdc++ order has an effect only on equal-cost decisions:
   which parent wins a history, and which final state is first among equal
   minimum costs. The permutation of a step depends only on the candidate keys
   of that step. Thus, when such a tie occurs, the kernel runs the existing
   exact introselect again on the saved candidates of the previous step. It
   then resolves the tie by exact position. Ties are rare in real data. The
   tie, constant and zero corpus fixtures use the replay path.
4. **Parent lists.** Each touched group keeps a list of its actual parents.
   Expansion visits only those parents, not all 16 (HT4) or 8 (HT3) possible
   parent histories. The minimum and the tie flag do not depend on list
   order.
5. **Computed codebook.** If the supplied table is bit-identical to the
   standard HT codebook, the kernel computes each entry with the same integer
   and F16 rounding steps instead of a global load. The host checks the full
   table. Any other table uses the uploaded copy.
6. **No candidate compaction.** Selection reads the expanded array directly
   and skips histories that no parent reached. This array is in ascending
   history order, so the selected set and its order do not change. The rare
   replay path compacts the previous step only when it needs it. Keys of the
   first 8 rounds of each lane stay in registers from expansion to selection.
7. **Early radix stop.** When the bin that holds rank k must be fully
   selected, the radix search stops. Lower digits cannot change the set.
8. **Multi-band GPU calls.** With a GPU HT backend, conversion sends up to
   2^23 values of complete 128-row bands in one call. A batch does not cross
   an expert, because down experts have their own input signs. HT tiles are
   independent and band-major, so the result equals the concatenated
   single-band output. A single 128×640 down-expert band is only 320 tiles,
   about one device wave. CPU HT still uses single bands.
9. **GPU rotation.** CUDA HT does the forward rotation on the GPU with the
   CPU order of operations: FP64 `(value / sv) * su`, the row axis then the
   column axis, one exact /128 and f32 narrowing. The kernel writes trellis
   tile order directly, for single bands and for paired gate/up bands. A
   zero row scale with a nonzero value, a nonfinite value or a value above
   1e17 sets a flag. Then the encoder runs the CPU path again to report the
   exact CPU error. The trellis kernel reads the rotated tiles in device
   memory, in launches of up to 16,384 tiles.

Measured on the workhorse RTX 4090 Laptop GPU (CUDA 12.0, GCC 12.4,
`sm_89`), beam 128, one 128×2560 band, median of 3 warm samples. The
"Before" column is the 2026-10-12 build before these changes, in the same
session:

| Case | Before (ms) | After changes 1–3 (ms) | After all (ms) | After all Mvalues/s | Ratio to before |
| --- | ---: | ---: | ---: | ---: | ---: |
| HT4 synthetic | 181–186 | 60.5 | 31.9 | 10.3 | ~5.7× |
| HT3 synthetic | about 105 | 43.8 | 26.6 | 12.3 | ~3.9× |
| HT4 real Orca layer 0 gate | — | 60.9 | 32.6 | 10.0 | — |
| HT3 real Orca layer 0 gate | — | 44.0 | 27.1 | 12.1 | — |
| HT4 pair real Orca layer 0 | — | 113.7 | 60.6 | 10.8 | — |
| Q4C-P 128×4096 (16 bands) | CPU 58.9 Mvalues/s | — | — | 1180 | ~20× vs CPU |

For down-expert batches of 35 bands (128×640 each), changes 8 and 9 give
about 11.2 Mvalues/s for HT4 and 14 Mvalues/s for HT3. One band of this
shape was 6.8 and 8.2 Mvalues/s before change 8. A 35-band HT4 call takes
244 ms instead of 264 ms without the GPU rotation.

Interleaved A/B rounds in one session gave the steps for changes 4–7:
HT4 60.5 → 44 → 40 → 35 → 31 → 30 ms; HT3 43.8 → 34 → 32.6 → 30.5 → 26 →
25 ms. A tested one-barrier triple-buffered radix variant was slower and was
not kept.

Evidence for byte identity:

- All CUDA opt-in tests passed, including every HT beam 1–256 against CPU and
  a 4,097-tile call that crosses the batch and grid stride.
- The 224-sample HT corpus matched CPU at beams 1/8/128/256.
- Real Orca expert bands from layers 0 and 23 matched CPU fingerprints.
- New Q4C-P tests matched CPU bytes for shapes, custom codebooks, more than
  one device batch and concurrent callers. The overflow error text is also
  identical.
- The dual HT and Q4C-P CUDA tiny conversion gave three containers that are
  byte-identical to CPU. A CPU-only executable verified them.
- Fault injection: if you remove the step replay or the end-of-pass replay,
  the corpus and the native CPU comparison tests fail. Thus the tests cover
  both replay paths. This check was done again after changes 4–7.

The build uses `__launch_bounds__(256, 4)`. On this GPU, 4 blocks per SM was
the fastest setting. 3, 5 and 6 blocks were slower. Different GPUs need their
own measurement. These are band timings on a shared GPU. They are not
whole-model conversion times.

```sh
# Q4C-P CPU/CUDA timing; every sample is byte-checked against CPU.
$CUDA_BENCH --suite timing --cases nf4 --cols 4096 --backends cpu \
  --q4cp-backends cpu,cuda --q4cp-bands 16 --iterations 1 --repetitions 5 --warmup 2
# Q4C-P device tests (use --features hip on gfx1151).
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo test --release --locked --features cuda --test q4cp_gpu -- --ignored
```

### HIP HT search port (2026-10-13)

The HIP HT3/HT4 encoder now uses the same search as CUDA: persistent grid,
coalesced wave segments, radix set selection with exact libstdc++ tie
replay, parent lists, no candidate compaction, early radix stop, GPU forward
rotation, multi-band calls and in-place launches of up to 16,384 tiles.
Wave operations use the HIP 64-bit ballot and the device `warpSize`, so the
code is correct for wave32 and wave64. gfx1151 runs wave32.

Two settings differ from CUDA because they measured faster on gfx1151:

- HT4 uses the uploaded codebook table and `amdgpu_waves_per_eu(16)`.
- HT3 uses the computed standard codebook and the default occupancy.
- HT3 keeps 4 register-cached keys per lane instead of 8. Its steps expand
  at most 1,024 positions, and the smaller cache gives 12 waves per SIMD
  (HT3 46.0 ms instead of 47.5 ms). HT4 still needs all 8.

A profile pass on gfx1151 (per-phase clock counters) put selection at about
half of the kernel time and expansion at 30–40%. These variants were
measured and rejected because they were slower: a barrier-free redundant
radix scan, per-wave radix histograms, 64/128/512-thread blocks, OR/AND key
range folded into the expansion reduction, other HT4 occupancy targets
(12/14; 16 is the compiler maximum, and 18/20 fall back to 12), fewer HT4
cached keys, and `-mcumode`.

Measured on Frame (Radeon 8060S, ROCm HIP 7.2, GCC 16.2.1) with Halogen
stopped, beam 128, one 128×2560 band, interleaved with the previous build:

| Case | Previous HIP (ms) | New HIP (ms) | New Mvalues/s | Ratio |
| --- | ---: | ---: | ---: | ---: |
| HT4 synthetic | 419.7 | 57.9 | 5.66 | ~7.2× |
| HT3 synthetic | 296.9 | 47.5 | 6.90 | ~6.3× |
| HT4 pair synthetic | — | 112.2 | 5.84 | — |
| HT4 real Blackfrost layer 0 gate | — | 63.4 | 5.17 | — |
| HT3 real Blackfrost layer 0 gate | — | 48.9 | 6.70 | — |

Evidence for byte identity on gfx1151:

- All 12 HIP library device tests, the HIP conversion tests and the new HIP
  multi-band and rotation-error tests passed.
- The 224-sample corpus matched CPU at beams 1/8/128/256.
- Real Blackfrost expert bands from layers 0 and 23 matched CPU fingerprints.
- The HIP tiny conversion gave three containers that are byte-identical to
  CPU, and a CPU-only executable verified them.
- Fault injection of the step replay and the end-of-pass replay made the
  corpus fail.

### HIP Q4C-P encoder (2026-10-13)

`--q4cp-backend hip` uses the same kernel as CUDA: one thread per 32-value
group, sequential FP64 error sums, strict first-wins candidates and
lower-level midpoint ties. Half scales use the integer RNE helper of the
HIP I4R encoder. A scale that is not a finite F16 reruns the CPU reference
for its exact error. There is no CPU fallback.

On Frame (gfx1151, ROCm 7.2), 128×4096 bands, median of 5 samples, every
sample byte-checked against CPU: 1 band 1559 Mvalues/s, 16 bands 1619
Mvalues/s; CPU 71 Mvalues/s (about 22×). The four Q4C-P device tests (shapes,
custom codebooks, scale-overflow error parity and concurrent callers), the
NF4 correctness corpus and a tiny conversion with HT and Q4C-P on HIP passed.
A CPU-only executable verified the containers. Q4C-P is about 2% of HT43
values, so this saves only minutes in a full conversion.

## Bounded structural checks

The public CLI has no fixture mode or architecture bypass. Library examples
use tiny genuine safetensors/HGN/GGUF inputs and explicit structural plans
to perform real encoding, atomic publication and full verification. Their
non-engine identities/configurations cannot be served as Flash-Next.

From this crate directory:

```sh
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 \
  cargo test --locked --all-targets -- --test-threads=1

# Existing v2 fixture and each new HT43 input path.
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 cargo run --locked --example tiny_conversion
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 cargo run --locked --example tiny_hf_to_ht43
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 cargo run --locked --example tiny_hgn_to_ht43
CARGO_BUILD_JOBS=1 OMP_NUM_THREADS=1 cargo run --locked --example tiny_gguf_to_ht43
```

Examples use one native worker and beam 8, print their conversion/verifier
reports, and remove temporary inputs/output. Tests include pinned container
goldens, codec vectors, independent HT3/HT4 paired decoding, source provenance,
split GGUFs, append-only HGN, input immutability and publication boundaries.
The optional real-stock HGN probe is ignored by default and requires its
local fixture; it is not a portable test prerequisite.

Optional HIP acceptance on gfx1151:

```sh
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo test --release --locked --features hip hip_ -- --ignored --test-threads=1
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features hip --example tiny_hip_conversion \
  -- --root /tmp/tiny-hip-conversion
# Separate default-feature executable: no GPU or HIP runtime needed to verify.
HIP_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --example tiny_hip_conversion \
  -- --root /tmp/tiny-hip-conversion --verify-only

hipcc -std=c++17 -O3 --offload-arch=gfx1151 -ffp-contract=off \
  -fno-fast-math -fno-gpu-flush-denormals-to-zero -pthread \
  tests/hip_selection.hip -o /tmp/hip-selection
/tmp/hip-selection
```

For the new HIP I4R path, use separate new roots for I4R-only and dual
HT/I4R conversion. Both paths retain CPU oracle and HIP outputs; the
independent CPU-only verification commands need neither a GPU nor HIP runtime:

```sh
# I4R-only HIP encoding; HT and verification remain on CPU.
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features hip --example tiny_hip_conversion \
  -- --backend hip --ht-backend cpu --i4r-backend hip --decode-backend cpu \
  --root /tmp/tiny-i4r-only-hip
HIP_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --no-default-features --example tiny_hip_conversion \
  -- --backend hip --ht-backend cpu --i4r-backend hip --decode-backend cpu \
  --root /tmp/tiny-i4r-only-hip --verify-only

# Dual HIP HT/I4R encoding and HIP verification; independent CPU verification.
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features hip --example tiny_hip_conversion \
  -- --backend hip --i4r-backend hip --decode-backend hip \
  --root /tmp/tiny-dual-hip
HIP_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --no-default-features --example tiny_hip_conversion \
  -- --backend hip --i4r-backend hip --decode-backend cpu \
  --root /tmp/tiny-dual-hip --verify-only
```

The example defaults are `--backend hip`, `--i4r-backend cpu` and
`--decode-backend cpu`; omitting `--ht-backend` uses the selected output
backend for HT. Selecting HIP or CUDA I4R enables an actual expert I4R
matrix in the fixture. `--verify-only` preserves the retained encoder
provenance while allowing an independently selected decoder. I4R-only and
dual HIP acceptance is recorded in the bounded qualification above.

The retained smoke root must be new. It compares every byte of all three
published CPU/HIP HGN containers and independently verifies both conversions.
These structural fixtures are deliberately not engine-loadable models.

The existing example name is retained; `--backend hip` is its default and
`--backend cuda` selects the retained CUDA output and default HT encoder.
`--ht-backend cpu --i4r-backend cuda` selects I4R-only CUDA; omitting that
HT override selects dual HT/I4R CUDA. Adding `--i4r-backend cuda` enables
an actual I4R matrix in the fixture. The I4R-only and dual paths were
exercised in the bounded qualification above; use a compatible device:

```sh
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo test --release --locked --features cuda cuda_ -- --ignored --test-threads=1
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features cuda --example tiny_hip_conversion \
  -- --backend cuda --decode-backend cuda --root /tmp/tiny-cuda-conversion
# Retained cpu/ and cuda/ outputs; independent CPU-only verification.
CUDA_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --example tiny_hip_conversion \
  -- --backend cuda --root /tmp/tiny-cuda-conversion --verify-only

# I4R-only CUDA encoding and CUDA decoding, in a separate new fixture root.
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features cuda --example tiny_hip_conversion \
  -- --backend cuda --ht-backend cpu --i4r-backend cuda \
  --decode-backend cuda --root /tmp/tiny-i4r-only-cuda
CUDA_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --example tiny_hip_conversion \
  -- --backend cuda --ht-backend cpu --i4r-backend cuda \
  --root /tmp/tiny-i4r-only-cuda --verify-only

# Dual HT/I4R CUDA encoding; independently verify using a CPU-only executable.
CXX=g++-12 NVCC=nvcc CARGO_TARGET_DIR=/tmp/bf16-hgn-cuda-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --features cuda --example tiny_hip_conversion \
  -- --backend cuda --i4r-backend cuda --decode-backend cuda \
  --root /tmp/tiny-dual-cuda
CUDA_VISIBLE_DEVICES=-1 CARGO_TARGET_DIR=/tmp/bf16-hgn-cpu-target CARGO_BUILD_JOBS=1 \
  cargo run --release --locked --example tiny_hip_conversion \
  -- --backend cuda --i4r-backend cuda --root /tmp/tiny-dual-cuda --verify-only

# Full device candidate permutations and forced heap fallback; arch 89.
nvcc -ccbin g++-12 -std=c++17 -O3 \
  -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_89,code=compute_89 \
  --fmad=false --ftz=false --prec-div=true --prec-sqrt=true \
  -Xcompiler=-ffp-contract=off -Xcompiler=-fno-fast-math -Xcompiler=-pthread \
  tests/cuda_selection.cu -o /tmp/bf16-hgn-cuda-selection
/tmp/bf16-hgn-cuda-selection
```

Use a new root for conversion. The CUDA path compares all three published
HGN containers against CPU and can verify the complete numerical reports
with either decoder. `--backend` identifies the retained output, while
`--decode-backend` selects verification execution; CPU is the decoder default.
Ignored CUDA tests include
`cuda_tiny_conversion_matches_cpu_all_containers_and_verifies` and
`cuda_full_verification_reports_match_cpu_bitwise`, alongside native encoding/
decoding fidelity cases. In the historical HT qualification on 2026-10-12,
workhorse passed 167 nonignored all-target tests and all 14 then-existing
CUDA device tests. Frame passed 170 CPU tests (1 ignored) and 167 HIP-feature
tests (15 ignored; HIP GPU tests were not run). These counts predate the
integrated I4R encoder; the independent full-model stock oracle remained
outside that CUDA qualification.

The CUDA selection regression passed 5,896 cases with permutation fingerprint
`1e1a514d2974fd0f`, matching Frame GCC 16. All 196 CPU/CUDA corpus samples
across 98 distinct fixture keys matched Frame. The tiny conversion produced
three byte-identical CPU/CUDA HGN containers; complete CPU/CUDA numerical
verification reports were also identical, including independent CPU-only
verification of the retained CUDA output. These are bounded fixture results,
not full-model conversion or inference qualification.

Encoding regressions cover every beam 1–256 for HT3/HT4, closure, paired
bands, a 513-tile batch boundary, subnormal costs and content-changed
codebooks. `cuda_concurrent_callers_and_codebook_changes_match_cpu` exercises
four concurrent callers through the serialized native context; recovery
cases distinguish a prior runtime error from a current encode failure.

For a CPU-only cross-host selection fingerprint (not GPU qualification):

```sh
g++ -x c++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math \
  -DHGN_SELECTION_HOST_ONLY tests/cuda_selection.cu \
  -o /tmp/bf16-hgn-host-selection
/tmp/bf16-hgn-host-selection
```

Compare its `permutation-fingerprint` against the device regression on the
selected ordinary libstdc++ 12/16 toolchain. The host-only mode is labeled
explicitly; it does not exercise CUDA kernels. Adjust both device `-gencode`
arguments when qualifying a compute capability other than 89.

Bounded CPU-only checks on 2026-10-10 passed the integrated all-target test
suite, the separately enabled real-stock HT43 probe, and all four executable
examples (real conversion plus verification). Metadata-only public CLI
inspection with `--profile ht43` also passed a complete BF16 snapshot,
appended native r2 HGN with sidecars, and BF16 GGUF with MTP/vision sidecars.
The HGN/GGUF metadata smokes used structurally compatible config/assets;
they do not endorse checkpoint provenance or quality.

These checks do **not** include full-model HT43 conversion, engine model
loading, full-model inference, inference-quality comparisons or full-model
throughput. A full v2 conversion is not evidence for HT43 qualification.

## Changes

- Added opt-in `--q4cp-backend hip`, with the same byte-exact Q4C-P kernel as
  CUDA and shared GPU dispatch in Rust.

- Ported the CUDA HT search work to HIP, with gfx1151-specific occupancy and
  codebook choices. HIP HT4 is about 7.2× and HT3 about 6.3× faster per
  band; output is byte-identical to CPU.

- Added opt-in `--q4cp-backend cuda`, with manifest provenance, a separate
  progress class, value-bounded batches and no CPU fallback. Reworked CUDA
  HT3/HT4 search: persistent grid, coalesced warp segments, radix set
  selection with exact libstdc++ tie replay, per-group parent lists, a
  computed standard codebook, no candidate compaction, an early radix
  stop, multi-band GPU HT calls within an expert and GPU forward rotation. Output is byte-identical to CPU; HT4 is about 5.7× and HT3 about 3.9×
  faster per band. See the CUDA Q4C-P section above.

- Added opt-in `--i4r-backend hip`, independent of HT and verification backends,
  with CPU defaults, explicit per-encoder provenance and no CPU fallback.
  HIP I4R performs the full forward rotation and quantization on GPU in bounded
  batches; its workspace is allocated lazily. Added HIP-only/dual smoke and
  independent benchmark recipes. Recorded integrated fidelity and complete
  bounded synthetic/real-band timing axes, separately from historical HT/CUDA
  measurements and without whole-model speed claims. AMD wave32/wave64 source
  support does not establish qualification beyond gfx1151.
- Added opt-in `--i4r-backend cuda`, independent of HT encoding and runtime
  verification selection, with per-encoder provenance and CPU defaults.
  The full I4R forward rotation and quantizer run on GPU without CPU fallback;
  conversion batches at most 32 bands within expert/portion boundaries.
  Added independent I4R benchmark/backend/batch axes and tiny I4R-only/dual
  CUDA acceptance coverage; recorded integrated encoder fidelity and bounded
  timings separately from historical HT and prototype results.
- Added opt-in CUDA HT3/HT4 encoding and I4R/HT3/HT4 verification with explicit
  feature/device errors, preserved CPU defaults and separate encoder
  provenance versus verification-backend selection.
- Generalized the retained tiny GPU conversion example with `--backend cuda`
  and added CUDA fidelity/selection coverage and GPU-neutral benchmark
  schema 2; recorded 2026-10-12 cross-host fidelity and measured bounded
  CUDA encoder ratios above, not isolated or whole-model speedup claims.

- Reorganized terminal progress into a compact block laid out for the current
  width: stage percentage/gauge/counts, elapsed/ETA/rate, operation/codec/
  backend with the full tensor name, then expert/part/row position. Narrow
  terminals drop optional fields and shorten names in the middle instead of
  wrapping. ETA math and its learning/partial/finalizing wording are
  unchanged; preparation and verification now also say when they are
  finalizing outside their ETA.

- Added explicit HIP verification of I4R and independent HT3/HT4 decoding,
  retaining CPU integrity checks and exact sequential error accumulation.
- Added strict read-only schema-1 HF verification alongside schema 2,
  with explicit provenance-attestation status and no artifact migration.
- Added indicatif terminal progress for conversion and verification,
  including updates within large tensors and clean redirected JSON output.
- Added visible layer/expert/codec/backend and within-tensor context, plus a
  separately counted rotation-preparation stage. CPU conversion/verification
  smoke runs exercised 80/100/120-column terminals (file-seek latency exposed
  sub-frame fixture tensors), quiet redirected JSON and checksum-failure output.

- Added opt-in gfx1151 HIP HT3/HT4 encoding, bounded reusable device storage,
  explicit backend provenance and CPU-only verification of HIP output.
- Added CPU/GPU differential benchmarks, native selection regressions and
  real tiny GPU conversion/publication/verification coverage. CPU remains
  the default; HIP has no non-HT quantizer, while CUDA now also supports I4R.

- Added byte-preserving AVX2 HT transitions, exact packed candidate ordering
  and bounded I4R/Q6/Q8 rounding, with scalar/portable fallbacks.
- Added bounded encoder benchmarks, complete byte fingerprints and numerical
  boundary coverage; measured gains and shared-host limitations are above.

- Added bounded native HGN and GGUF inputs with explicit auxiliary weights
  and matching assets; HF BF16 input and default v2 quantization remain.
- Added the exact HT43 expert/mixer store and paired-band layout profile.
- Added schema-2 canonical-source hashes plus separate physical provenance;
  schema-1 raw HF hashes remain explicitly distinguished during verification.
- Added executable tiny conversions for HF/HGN/GGUF → HT43 and bounded
  regression coverage; no hidden calibration or stock-weight substitution.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

The vendored trellis encoder in `native/` is licensed under Apache-2.0 only. See `native/LICENSE` and `native/NOTICE`.
