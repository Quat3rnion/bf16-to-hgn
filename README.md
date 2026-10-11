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

For optional Strix Halo GPU encoding, build with ROCm/HIP and ordinary
**libstdc++ 16 for both C++ translation units**:

```sh
# From this crate directory; separate output keeps existing binaries intact.
CARGO_TARGET_DIR=/tmp/bf16-hgn-hip-target CARGO_BUILD_JOBS=1 \
  cargo build --release --locked --features hip
HIP_BIN=/tmp/bf16-hgn-hip-target/release/bf16-to-hgn
$HIP_BIN convert --source /path/to/hf-bf16 --output /path/to/new-output \
  --profile ht43 --ht-backend hip --ht-beam 128
```

`ROCM_PATH` defaults to `/opt/rocm`; `HIPCC` may override its compiler.
The HIP build targets `gfx1151`, preserves FP32 denormals, disables fast math
and FMA contraction, and links the ROCm HIP runtime. Its CPU and GPU searches
depend on the same libstdc++ 16 selection permutation; incompatible standard
libraries or parallel algorithms are rejected at build time.
CUDA and other GPU architectures are not implemented.

## Commands

`--source-format hf|hgn|gguf` is optional when the index/file magic is
unambiguous. `inspect`, `plan` and `convert` accept `--profile v2|ht43`;
omitting it selects v2. `verify` derives the profile and encoding options
from `conversion-manifest.json`, not a CLI profile override.

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

**Schema 2 is a clean manifest cutover, not schema-1 compatible.** The new
verifier rejects schema-1 outputs; use the original preserved schema-1
executable to verify old/running conversions. It does not migrate old
manifests or reinterpret their source hashes.

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

Rust streams one tensor and 128 matrix rows at a time, with experts processed
individually and paired HT gate/up bands handled separately. The n-gram table
uses 8,192-row chunks and two passes; raw copies/hashes use at most 8 MiB
chunks. HGN/GGUF decoding uses bounded source windows, not an intermediate
decoded HF checkpoint or temporary full model. HF uses windowed mmap reads
with eviction hints; output dirty data is synced/evicted every 64 MiB and
final hashing is streamed.

Generated rotations are retained as F32 vectors across conversion/verification.
They are not all “small”: HT43's per-expert signs/scales require roughly
**440 MB**, and the full prepared rotation map approximately **450 MB
(429 MiB)**, once per conversion/verifier, not once per worker. Row buffers,
import decoder scratch and native per-worker trellis scratch are additional.
These are working-set bounds, not an exact RSS guarantee for every allocator/OS.

Default `--ht-backend cpu --threads 1` is deliberate. `--threads` controls
only CPU HT workers (1–64), not HIP. HIP accelerates HT3/HT4 trellis search;
rotations and all other codecs remain on CPU. It supports every CLI beam
1–256; exact beam 0 and larger library-only beams require the CPU backend.
Requesting HIP without the feature, a compatible device, or a working runtime
is an error, never a silent CPU fallback. Real conversion preflights HIP before
creating staging output; metadata-only inspect/plan do not initialize a GPU.

HIP uses device 0 (respecting `HIP_VISIBLE_DEVICES`), a least-priority stream,
64 tiles per batch and one reusable **27,607,808-byte device allocation**.
Runtime/driver memory is additional. Calls share a serialized persistent
context and content-checked codebook cache. A low-priority stream is not
isolation: HIP still competes with inference for GPU compute and shared UMA.
There is no parallel whole-tensor/expert queue or automatic service management.
Do not run full conversion on a host whose live model consumes its headroom.

The manifest records `options.ht_backend`; an omitted field in an older
schema-2 manifest means CPU. CPU-only builds can verify HIP-produced output
without a GPU. Container bytes must match the same-toolchain CPU reference;
the manifest legitimately differs in backend metadata. Changing backends
does not resume an unfinished conversion.

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

`examples/encoder_bench.rs` emits JSONL with input fingerprints, complete
output/metadata/rotation/padding fingerprints, and per-sample timings.
Timing includes public encoding, validation, codec allocations and rotation,
but excludes input loading, sign/row-RMS preparation, hashing and destruction.
It caps input bands and retained outputs, reads only selected HF BF16 bands,
and checks equality across iterations, samples, worker counts and backends.

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

| Case | Baseline median ms (range) | Optimized median ms (range) | Throughput ratio |
| --- | ---: | ---: | ---: |
| Real Blackfrost HT4 | 6790 (5948–7682) | 4529 (3936–4928) | 1.50× |
| Real Blackfrost HT3 | 3800 (3399–5286) | 2802 (2625–3561) | 1.36× |
| Real Blackfrost HT4 gate/up | 12773 (11439–13919) | 9333 (8676–9695) | 1.37× |
| Real Blackfrost HT3 gate/up | 7121 (6728–7708) | 5728 (5098–6448) | 1.24× |
| Synthetic I4R | 15.681 (13.378–17.842) | 6.754 (5.994–7.653) | 2.32× |
| Synthetic Q6G64 | 2.426 (2.340–2.782) | 2.320 (2.040–2.636) | 1.05× |
| Synthetic Q8G64 | 2.053 (1.935–3.983) | 1.820 (1.696–2.420) | 1.13× |

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

| Case | CPU median ms (range) | HIP median ms (range) | Throughput ratio |
| --- | ---: | ---: | ---: |
| HT4 | 3128 (3085–3476) | 862 (423–1327) | 3.63× |
| HT3 | 2120 (2024–2243) | 450 (301–666) | 4.71× |
| HT4 gate/up | 6460 (6119–6966) | 845 (844–847) | 7.64× |
| HT3 gate/up | 4336 (4035–5146) | 914 (601–1273) | 4.74× |

All 64 timing samples (including the separate one-worker axis) matched
complete input/output fingerprints across processes. The public corpus
matched 448 samples across both backends, one/two workers, seven fixtures
and beams 1/8/128/256. Native comparisons matched 640 cases covering every
supported beam. Device preflight took 200–328 ms, excluded from this table.
The shared GPU was not idle: wide HIP ranges reflect contention, and the
unusually tight paired-HT4 subset must not be generalized. No full-model
GPU conversion or inference qualification was performed.

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

Optional GPU acceptance on gfx1151:

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

The retained smoke root must be new. It compares every byte of all three
published CPU/HIP HGN containers and independently verifies both conversions.
These structural fixtures are deliberately not engine-loadable models.

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

- Added opt-in gfx1151 HIP HT3/HT4 encoding, bounded reusable device storage,
  explicit backend provenance and CPU-only verification of HIP output.
- Added CPU/GPU differential benchmarks, native selection regressions and
  real tiny GPU conversion/publication/verification coverage. CPU remains
  the default; CUDA and non-HT GPU quantizers are not implemented.

- Added byte-preserving AVX2 HT transitions, exact packed candidate ordering
  and bounded I4R/Q6/Q8 rounding, with scalar/portable fallbacks.
- Added bounded encoder benchmarks, complete byte fingerprints and numerical
  boundary coverage; measured gains and shared-host limitations are above.

- Added bounded native HGN and GGUF inputs with explicit auxiliary weights
  and matching assets; HF BF16 input and default v2 quantization remain.
- Added the exact HT43 expert/mixer store and paired-band layout profile.
- Added schema-2 canonical-source hashes plus separate physical provenance;
  old schema-1 outputs require their original verifier.
- Added executable tiny conversions for HF/HGN/GGUF → HT43 and bounded
  regression coverage; no hidden calibration or stock-weight substitution.
