// HIP HT/I4R encoders and independent verification decode/unrotation.
// The CPU implementations remain the references; this file never calls them.
//
// Selection below is an independent implementation of the observable
// libstdc++ 16 nth_element permutation, not merely its selected set. Changing
// the CPU standard library/selection algorithm requires a new equivalence
// qualification. In particular, libc++ and parallel-mode libstdc++ are not
// interchangeable. Build both encoders without fast math or FP contraction;
// HIP additionally needs -fno-gpu-flush-denormals-to-zero.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <array>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>

#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#if !defined(__HIP_PLATFORM_AMD__)
#error "The HT HIP encoder requires the AMD HIP platform and gfx1151"
#endif
#if !defined(_GLIBCXX_RELEASE) || _GLIBCXX_RELEASE != 16 || defined(_GLIBCXX_PARALLEL)
#error "The HT HIP encoder requires ordinary libstdc++ 16 selection semantics"
#endif
#if defined(__FAST_MATH__)
#error "The HT HIP encoder cannot preserve CPU bytes with fast math enabled"
#endif

namespace {
constexpr int kStates = 65536;
constexpr int kLength = 256;
constexpr int kMaxBeam = 256;
constexpr int kThreads = 256;
// Tiles per launch. Blocks are persistent: the grid is bounded by co-resident
// blocks and workspace slots, not by this batch, so a band is one launch.
constexpr int kBatch = 16384;
constexpr size_t kWorkspaceBytes = 128u * 1024u * 1024u;
constexpr uint16_t kMissing = 0xffff;

struct Path {
    uint16_t parent;
    uint16_t state;
    float cost;
};
struct Trace {
    uint16_t parent;
    uint16_t symbol;
};
struct TileStatus {
    int code;
    int pass;
    int step;
};
struct TileWorkspace {
    Path *expanded;
    Path *candidates;
    Path *previous; // The prior step's candidates, kept for exact tie replay.
    uint16_t *wrong_left;
    uint16_t *wrong_right;
    Trace *trace;
};
__host__ __device__ constexpr size_t tile_workspace_bytes(int bits, int beam) {
    return size_t(kStates >> bits) * (3 * sizeof(Path) + 2 * sizeof(uint16_t))
        + size_t(kLength) * size_t(beam) * sizeof(Trace);
}
__device__ __forceinline__ TileWorkspace tile_workspace(uint8_t *bytes, int histories) {
    Path *expanded = reinterpret_cast<Path *>(bytes);
    Path *candidates = expanded + histories;
    Path *previous = candidates + histories;
    uint16_t *wrong_left = reinterpret_cast<uint16_t *>(previous + histories);
    uint16_t *wrong_right = wrong_left + histories;
    Trace *trace = reinterpret_cast<Trace *>(wrong_right + histories);
    return {expanded, candidates, previous, wrong_left, wrong_right, trace};
}
struct DeviceStorage {
    float codebook[kStates];
    float tiles[kBatch * kLength];
    uint8_t codes[kBatch * kLength];
    TileStatus status[kBatch];
    alignas(Path) uint8_t workspace[kWorkspaceBytes];
};
static_assert(sizeof(Path) == 8 && sizeof(Trace) == 4);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
// One persistent allocation: 128 MiB arena plus codebook/I/O for kBatch tiles.
// Per-slot layout uses 20*histories + 1024*beam bytes, all naturally aligned.
// Slots: HT3 beam256: 315; HT3 beam128: 455; HT4 beam256: 390; HT4 beam128: 615.
// Changing bits/beam only reinterprets this arena after the preceding sync.
static_assert(sizeof(DeviceStorage) == kStates * 4 + size_t(kBatch) * kLength * 5
              + size_t(kBatch) * sizeof(TileStatus) + kWorkspaceBytes);
static_assert(kWorkspaceBytes / tile_workspace_bytes(3, kMaxBeam) >= 256);
static_assert(sizeof(DeviceStorage) <= 150u * 1024u * 1024u);

// Decode uses two FP64 axis passes, with only 16 independent 128-point
// transforms in LDS at once. The complete 128x128 block would require 128 KiB.
// Batch rectangles contain at most 256 such blocks, even for a single very
// wide band. Encoder and decoder allocations are independent and lazy.
constexpr int kDecodeBlocks = 256;
constexpr int kDecodeValues = kDecodeBlocks * 128 * 128;
struct DecodeStorage {
    float codebook[kStates];
    uint8_t payload[kDecodeValues]; // HT4 paired is the largest input: 1 B/value.
    uint16_t scales[kDecodeValues / 128];
    float su[kDecodeBlocks * 128];
    float sv[kDecodeBlocks * 128];
    double scratch[kDecodeValues];
    float output[kDecodeValues];
};
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
static_assert(sizeof(DecodeStorage) <= 64u * 1024u * 1024u);

// Independent, lazy I4R storage: at most 256 complete 128x128 blocks. Whole
// bands stay contiguous unless the source is wider than 32768 columns. The
// source plane is reused for the rotated values after both FP64 axis passes.
constexpr int kI4rBlocks = 256;
constexpr int kI4rValues = kI4rBlocks * 128 * 128;
constexpr int kI4rThreads = 128;
constexpr int kI4rGroups = kI4rThreads / 8;
struct I4rStatus {
    // Quantizer blocks read the completed rotation status while other
    // quantizer blocks may atomically report a bad half scale.
    unsigned rotation_overflow;
    unsigned scale_overflow;
};
struct I4rStorage {
    float values[kI4rValues];
    double scratch[kI4rValues];
    float su[kI4rBlocks * 128];
    float sv[kI4rBlocks * 128];
    uint8_t codes[kI4rValues / 2];
    uint8_t scales[kI4rValues / 64];
    I4rStatus status;
};
static_assert(kI4rThreads % 64 == 0); // Complete physical waves in wave32 and wave64.
static_assert(sizeof(I4rStorage) == 52756488);
static_assert(sizeof(DeviceStorage) + sizeof(DecodeStorage) + sizeof(I4rStorage)
              <= 272u * 1024u * 1024u);

__device__ __forceinline__ float decode_half(uint16_t bits) {
    const unsigned sign = unsigned(bits & 0x8000) << 16;
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    if (exponent == 0) {
        // Integer conversion and this power-of-two multiplication are exact,
        // including half subnormals. Restore the sign even when fraction=0.
        return __uint_as_float(sign | __float_as_uint(float(fraction) * 0x1p-24f));
    }
    return __uint_as_float(sign | ((exponent == 31 ? 255 : exponent + 112) << 23)
                           | (fraction << 13));
}

__device__ __forceinline__ uint32_t decode_word(const uint8_t *bytes) {
    return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8)
        | (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}

// Each pair has exactly one owner. Complete all pairs at a level before
// reading the next: both CPU references use this same FP64 butterfly order.
template<int Stride>
__device__ __forceinline__ void decode_axis(double *values) {
    __syncthreads();
    for (int step = 1; step < 128; step *= 2) {
        for (int pair = int(threadIdx.x); pair < 16 * 64; pair += kThreads) {
            const int lane = pair % 64;
            const int first = (pair / 64) * Stride + (lane / step) * (2 * step) + lane % step;
            const double a = values[first];
            const double b = values[first + step];
            values[first] = a + b;
            values[first + step] = a - b;
        }
        __syncthreads();
    }
}

template<int Bits>
__global__ __launch_bounds__(kThreads) void decode_row_axis(
    DecodeStorage *storage, int cols, int row_tiles, int tile_offset) {
    __shared__ double values[16 * 128];
    const int row_base = int(blockIdx.y) * 16;
    const int col_base = int(blockIdx.x) * 128;
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        if constexpr (Bits == 0) {
            const uint8_t packed = storage->payload[(row * cols + col) / 2];
            const unsigned code = (packed >> (4 * (col % 2))) & 15;
            const float scale = decode_half(storage->scales[row * (cols / 128) + col / 128]);
            // Preserve the CPU's f32 multiply, notably the sign of zero.
            values[index] = double((float(code) - 8.0f) * scale);
        } else {
            // Independent verifier mapping: address the circular 16-bit state
            // directly, rather than using the encoder's symbol decoder.
            const int tile = ((row / 128) * (cols / 16) + col / 16) * row_tiles
                + (row % 128) / 16 + tile_offset;
            const int value = (col % 16 / 8) * 128 + (row % 16) * 8 + col % 8;
            const int start = (Bits * (value + 1) + 256 * Bits - 16) % (256 * Bits);
            const int word = start / 32;
            const int next = (word + 1) % (8 * Bits);
            const uint8_t *bytes = storage->payload + tile * (32 * Bits);
            const uint32_t first = decode_word(bytes + 4 * word);
            const uint32_t second = decode_word(bytes + 4 * next);
            const uint16_t state = uint16_t(((uint64_t(first) << 32) | second)
                                            >> (48 - start % 32));
            values[index] = double(storage->codebook[state]);
        }
    }
    decode_axis<128>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        storage->scratch[row * cols + col] = values[index];
    }
}

__global__ __launch_bounds__(kThreads) void decode_column_axis(
    DecodeStorage *storage, int cols, int per_band_su) {
    // Padding keeps the transpose from mapping all 16 columns to one LDS bank.
    __shared__ double values[16 * 129];
    const int row_base = int(blockIdx.y) * 128;
    const int col_base = int(blockIdx.x) * 16;
    // Coalesced global accesses, transposed LDS: the independent transforms
    // now run down rows, only after the complete column-coordinate pass.
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 16;
        const int col = col_base + index % 16;
        values[(index % 16) * 129 + index / 16] = storage->scratch[row * cols + col];
    }
    decode_axis<129>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 16;
        const int col = col_base + index % 16;
        const double value = values[(index % 16) * 129 + index / 16];
        const int su_index = (per_band_su ? (row / 128) * cols : 0) + col;
        // /128 is an exact power-of-two scaling. Keep the two following
        // multiplies separate and ordered; build.rs disables FP contraction
        // and preserves f32 denormals, including the final narrowing cast.
        storage->output[row * cols + col] = float(value * 0x1p-7
            * double(storage->sv[row]) * double(storage->su[su_index]));
    }
}

__global__ __launch_bounds__(kThreads) void i4r_forward_rows(I4rStorage *storage, int cols) {
    __shared__ double values[16 * 128];
    const int row_base = int(blockIdx.y) * 16;
    const int col_base = int(blockIdx.x) * 128;
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        // Rust validates finite source values and +/-1 signs. Preserve the
        // CPU's divide-then-multiply order, including signed zero.
        values[index] = __dmul_rn(__ddiv_rn(double(storage->values[row * cols + col]),
                                          double(storage->sv[row])), double(storage->su[col]));
    }
    decode_axis<128>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        storage->scratch[row * cols + col] = values[index];
    }
}

__global__ __launch_bounds__(kThreads) void i4r_forward_columns(I4rStorage *storage, int cols) {
    __shared__ double values[16 * 129];
    const int row_base = int(blockIdx.y) * 128;
    const int col_base = int(blockIdx.x) * 16;
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 16;
        const int col = col_base + index % 16;
        values[(index % 16) * 129 + index / 16] = storage->scratch[row * cols + col];
    }
    decode_axis<129>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 16;
        const int col = col_base + index % 16;
        // Normalize only after both ordered FP64 axes, then narrow once.
        const float value = __double2float_rn(
            __dmul_rn(values[(index % 16) * 129 + index / 16], 0x1p-7));
        if ((__float_as_uint(value) & 0x7fffffffU) >= 0x7f800000U)
            atomicExch(&storage->status.rotation_overflow, 1u);
        storage->values[row * cols + col] = value;
    }
}

__device__ __forceinline__ uint16_t i4r_half_rne(float value) {
    // Rotation has rejected nonfinite values, and scale construction cannot
    // overflow f32. Integer RNE avoids depending on the device's FP16
    // denormal mode while preserving signed zero and half subnormals.
    const unsigned bits = __float_as_uint(value);
    const unsigned sign = (bits >> 16) & 0x8000U;
    const unsigned magnitude = bits & 0x7fffffffU;
    if (magnitude >= 0x477ff000U) return uint16_t(sign | 0x7c00U); // >= 65520.
    if (magnitude <= 0x33000000U) return uint16_t(sign); // <= 2^-25; zero wins ties.
    if (magnitude < 0x38800000U) { // Below the minimum normal half, 2^-14.
        const unsigned shift = 126 - (magnitude >> 23);
        const unsigned significand = (magnitude & 0x007fffffU) | 0x00800000U;
        const unsigned truncated = significand >> shift;
        const unsigned remainder = significand & ((1u << shift) - 1u);
        const unsigned midpoint = 1u << (shift - 1);
        const unsigned rounded = truncated
            + (remainder > midpoint || (remainder == midpoint && (truncated & 1u)));
        return uint16_t(sign | rounded);
    }
    const unsigned rounded = magnitude + 0x00000fffU + ((magnitude >> 13) & 1u);
    return uint16_t(sign | ((rounded - 0x38000000U) >> 13));
}

// HT forward rotation, the CPU rotate_into order: (value / sv) * su in FP64,
// the two ordered butterfly axes, one exact /128 and f32 narrowing. A zero
// row scale yields +0 and requires a zero source; any violation, nonfinite
// result or |value| > 1e17 only raises a flag. The host then reruns the CPU
// path, which reports the exact first error.
__global__ __launch_bounds__(kThreads) void ht_forward_rows(I4rStorage *storage, int cols) {
    __shared__ double values[16 * 128];
    const int row_base = int(blockIdx.y) * 16;
    const int col_base = int(blockIdx.x) * 128;
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        const float value = storage->values[row * cols + col];
        const float scale = storage->sv[row];
        double rotated = 0.0;
        if (scale == 0.0f) {
            if (value != 0.0f) atomicExch(&storage->status.rotation_overflow, 1u);
        } else {
            rotated = __dmul_rn(__ddiv_rn(double(value), double(scale)), double(storage->su[col]));
        }
        values[index] = rotated;
    }
    decode_axis<128>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 128;
        const int col = col_base + index % 128;
        storage->scratch[row * cols + col] = values[index];
    }
}

// Writes the rectangle's rotated values straight into trellis tile order.
// row_offset/col_offset place the rectangle within the full call.
__global__ __launch_bounds__(kThreads) void ht_forward_columns(
    I4rStorage *storage, float *tiles, int cols, int row_offset, int col_offset, int full_cols,
    int row_tiles, int tile_offset) {
    __shared__ double values[16 * 129];
    const int row_base = int(blockIdx.y) * 128;
    const int col_base = int(blockIdx.x) * 16;
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_base + index / 16;
        const int col = col_base + index % 16;
        values[(index % 16) * 129 + index / 16] = storage->scratch[row * cols + col];
    }
    decode_axis<129>(values);
    for (int index = int(threadIdx.x); index < 16 * 128; index += kThreads) {
        const int row = row_offset + row_base + index / 16;
        const int col = col_offset + col_base + index % 16;
        const float value = __double2float_rn(
            __dmul_rn(values[(index % 16) * 129 + index / 16], 0x1p-7));
        const unsigned magnitude = __float_as_uint(value) & 0x7fffffffU;
        if (magnitude >= 0x7f800000U || __uint_as_float(magnitude) > 1.0e17f)
            atomicExch(&storage->status.rotation_overflow, 1u);
        const int tile = ((row / 128) * (full_cols / 16) + col / 16) * row_tiles
            + (row % 128) / 16 + tile_offset;
        tiles[size_t(tile) * kLength + (col % 16 / 8) * 128 + (row % 16) * 8 + col % 8] = value;
    }
}

__device__ __forceinline__ int i4r_round_code(float value, float divisor) {
    // build.rs explicitly enables correctly rounded HIP binary32 division.
    // A reciprocal approximation or FP64 quotient can change half-way codes;
    // use the supported compiler path, not optional OCML rounding symbols.
    const float quotient = value / divisor;
    const float clamped = fminf(fmaxf(quotient, -8.0f), 7.0f);
    return __float2int_rz(__fadd_rn(clamped, 12582912.0f)) - 12582912;
}

__global__ __launch_bounds__(kI4rThreads) void i4r_quantize(I4rStorage *storage) {
    // The column kernel has finished on this stream. Every thread observes
    // the same status, so this return never strands a thread at a barrier.
    if (storage->status.rotation_overflow) return;
    __shared__ float values[kI4rGroups * 128];
    __shared__ double errors[kI4rGroups][8];
    __shared__ uint16_t halves[kI4rGroups][8], selected[kI4rGroups];
    const int thread = int(threadIdx.x);
    const int local = thread / 8;
    const int lane = thread % 8;
    const int group = int(blockIdx.x) * kI4rGroups + local;
    // Complete 128x128 rectangles contain a multiple of 128 groups: no
    // partially active physical wave or logical eight-lane group is possible.
    for (int index = thread; index < kI4rGroups * 128; index += kI4rThreads)
        values[index] = storage->values[int(blockIdx.x) * kI4rGroups * 128 + index];
    __syncthreads();

    float extreme = values[local * 128 + lane];
    int position = lane;
    for (int index = lane + 8; index < 128; index += 8) {
        const float value = values[local * 128 + index];
        if (fabsf(value) > fabsf(extreme)) {
            extreme = value;
            position = index;
        }
    }
    for (int delta = 4; delta != 0; delta /= 2) {
        // HIP width=8 partitions either wave32 or wave64 into aligned groups.
        // All lanes execute the shuffle, not only its eventual consumers.
        const float value = __shfl_down(extreme, delta, 8);
        const int index = __shfl_down(position, delta, 8);
        if (lane + delta < 8 && (fabsf(value) > fabsf(extreme)
            || (fabsf(value) == fabsf(extreme) && index < position))) {
            extreme = value;
            position = index;
        }
    }
    // First source index wins equal magnitudes, including the sign of zero.
    extreme = __shfl(extreme, 0, 8);
    uint16_t half_bits = 0;
    if (lane < 6) {
        const float multipliers[6] = {1.0f, 0.97f, 0.94f, 0.91f, 0.88f, 0.85f};
        const float scale = __fmul_rn(__fdiv_rn(-extreme, 8.0f), multipliers[lane]);
        half_bits = i4r_half_rne(scale);
    }
    // HIP ballots are 64-bit even on wave32. Use the actual physical lane
    // index so groups in the upper half of wave64 are not silently ignored.
    const unsigned long long invalid = __ballot((half_bits & 0x7c00) == 0x7c00);
    const unsigned long long group_mask = 0xffull << ((thread % warpSize / 8) * 8);
    const bool invalid_group = (invalid & group_mask) != 0;
    if (invalid_group && lane == 0) atomicExch(&storage->status.scale_overflow, 1u);

    double error = 0.0;
    if (lane < 6 && !invalid_group) {
        const float scale = decode_half(half_bits);
        const float divisor = scale == 0.0f ? 1.0f : scale;
        // Parallel candidates/groups, never elements within one candidate:
        // sequential FP64 rounding controls the CPU's strict first-win ties.
        #pragma unroll 1
        for (int index = 0; index < 128; ++index) {
            const float value = values[local * 128 + index];
            const int code = i4r_round_code(value, divisor);
            const double delta = __dsub_rn(__dmul_rn(double(code), double(scale)), double(value));
            error = __dadd_rn(error, __dmul_rn(delta, delta));
        }
    }
    errors[local][lane] = error;
    halves[local][lane] = half_bits;
    __syncthreads();
    if (lane == 0 && !invalid_group) {
        int best = 0;
        for (int candidate = 1; candidate < 6; ++candidate)
            if (errors[local][candidate] < errors[local][best]) best = candidate;
        selected[local] = halves[local][best];
        storage->scales[group * 2] = uint8_t(selected[local]);
        storage->scales[group * 2 + 1] = uint8_t(selected[local] >> 8);
    }
    __syncthreads();
    if (!invalid_group) {
        const float scale = decode_half(selected[local]);
        const float divisor = scale == 0.0f ? 1.0f : scale;
        for (int index = lane; index < 64; index += 8) {
            const int first = i4r_round_code(values[local * 128 + 2 * index], divisor) + 8;
            const int second = i4r_round_code(values[local * 128 + 2 * index + 1], divisor) + 8;
            storage->codes[group * 64 + index] = uint8_t(first | (second << 4));
        }
    }
}

// Introselect ranges at most this long finish on one warp in shared memory.
constexpr int kWarpSelect = 256;
struct WarpSelect {
    Path values[kWarpSelect];
    uint16_t wrong_left[kWarpSelect];
    uint16_t wrong_right[kWarpSelect];
};

template<int Bits> struct BlockState {
    static constexpr int histories = kStates >> Bits;
    static constexpr int groups = histories >> Bits;
    Path active[kMaxBeam];
    // Per-group parent lists: head[group] -> active index -> next[index].
    // List order is arbitrary; expansion's result does not depend on it.
    unsigned head[groups]; // 32-bit: HIP has no 16-bit atomicExch.
    uint16_t next[kMaxBeam];
    uint16_t ordered_groups[groups];
    unsigned touched[(groups + 31) / 32];
    unsigned warp_sums[kThreads / 32];
    WarpSelect select;
    unsigned histogram[256];
    uint16_t exact_rank[kMaxBeam];
    unsigned long long key_or;
    unsigned long long key_and;
    int digit;
    int rank;
    int stop;
    int end;
    int count;
    int boundary;
};
static_assert(sizeof(BlockState<3>) <= 14u * 1024u);
static_assert(sizeof(BlockState<4>) <= 10u * 1024u);

__device__ __forceinline__ uint64_t key(const Path &p) {
    // Finite squared-error costs are nonnegative. Masking the sign treats
    // both signed zeros alike, just like the CPU comparator.
    return (uint64_t(__float_as_uint(p.cost) & 0x7fffffffU) << 16) | p.state;
}

__device__ __forceinline__ bool before(const Path &a, const Path &b) {
    return key(a) < key(b);
}

__device__ __forceinline__ bool finite_cost(float value) {
    return (__float_as_uint(value) & 0x7fffffffU) < 0x7f800000U;
}

__device__ __forceinline__ void exchange(Path *a, int left, int right) {
    const Path temporary = a[left];
    a[left] = a[right];
    a[right] = temporary;
}

struct Prefix {
    unsigned offset;
    unsigned total;
};

// Every lane participates. A third barrier protects the last reads of the
// shared sums from the next invocation, including callers with empty spans.
__device__ __forceinline__ Prefix prefix_sum(unsigned value, unsigned *sums) {
    const int lane = int(threadIdx.x) % warpSize;
    const int wave = int(threadIdx.x) / warpSize;
    const int waves = kThreads / warpSize;
    unsigned inclusive = value;
    for (int distance = 1; distance < warpSize; distance <<= 1) {
        const unsigned earlier = __shfl_up(inclusive, distance, warpSize);
        if (lane >= distance) inclusive += earlier;
    }
    if (lane == warpSize - 1) sums[wave] = inclusive;
    __syncthreads();
    if (wave == 0) {
        unsigned wave_sum = lane < waves ? sums[lane] : 0;
        for (int distance = 1; distance < warpSize; distance <<= 1) {
            const unsigned earlier = __shfl_up(wave_sum, distance, warpSize);
            if (lane >= distance) wave_sum += earlier;
        }
        if (lane < waves) sums[lane] = wave_sum;
    }
    __syncthreads();
    const Prefix result = {
        inclusive - value + (wave == 0 ? 0 : sums[wave - 1]),
        sums[waves - 1]
    };
    __syncthreads();
    return result;
}

// Leaf-first repair is intentional: an early-exit sift-down can leave a
// different heap permutation. Larger children move up to the leaf, then
// the saved value climbs only as far as this repair's original root.
__device__ void repair_heap(Path *a, int root, int size, Path saved) {
    int hole = root;
    while (2 * hole + 1 < size) {
        const int left = 2 * hole + 1;
        const int right = left + 1;
        int child = left;
        if (right < size && !before(a[right], a[left])) child = right;
        a[hole] = a[child];
        hole = child;
    }
    while (hole > root) {
        const int parent = (hole - 1) / 2;
        if (!before(a[parent], saved)) break;
        a[hole] = a[parent];
        hole = parent;
    }
    a[hole] = saved;
}

// Used only when introselect exhausts its 2*floor(log2(n)) partitions.
// Include nth itself in the heap, visit the remainder in increasing order,
// put each displaced root at that visit's position, and finally swap nth.
__device__ void heap_finish(Path *a, int first, int nth, int last) {
    Path *heap = a + first;
    const int size = nth - first + 1;
    if (size >= 2) {
        for (int root = (size - 2) / 2; root >= 0; --root) {
            const Path saved = heap[root];
            repair_heap(heap, root, size, saved);
        }
    }
    for (int position = nth + 1; position < last; ++position) {
        if (before(a[position], heap[0])) {
            const Path saved = a[position];
            a[position] = heap[0];
            repair_heap(heap, 0, size, saved);
        }
    }
    exchange(a, first, nth);
}

__device__ void insertion_finish(Path *a, int first, int last) {
    for (int position = first + 1; position < last; ++position) {
        const Path saved = a[position];
        int hole = position;
        while (hole > first && before(saved, a[hole - 1])) {
            a[hole] = a[hole - 1];
            --hole;
        }
        a[hole] = saved;
    }
}

__device__ __forceinline__ unsigned warp_inclusive_sum(unsigned value) {
    const int lane = int(threadIdx.x) % warpSize;
    for (int distance = 1; distance < warpSize; distance <<= 1) {
        const unsigned earlier = __shfl_up(value, distance, warpSize);
        if (lane >= distance) value += earlier;
    }
    return value;
}

// The median-of-three pivot step shared by the block and warp phases.
// Indices are relative to the array passed in, so the same arithmetic gives
// the same choice on a shared copy of [first, last) rebased to zero.
__device__ __forceinline__ void place_median(Path *a, int first, int last) {
    int lower = first + 1;
    int middle = first + (last - first) / 2;
    int upper = last - 1;
    if (before(a[middle], a[lower])) {
        const int swap = lower; lower = middle; middle = swap;
    }
    if (before(a[upper], a[middle])) {
        const int swap = middle; middle = upper; upper = swap;
    }
    if (before(a[middle], a[lower])) {
        const int swap = lower; lower = middle; middle = swap;
    }
    exchange(a, first, middle);
}

// Warp 0 continues the identical introselect (same depth budget, pivots,
// Hoare pair swaps, heap fallback and insertion finish) on a shared-memory
// copy of a short range. Every index rule is translation invariant, so the
// rebased copy receives exactly the permutation the block phase would apply.
// Only __syncwarp is needed: no other warp touches the copy.
__device__ void warp_select(Path *a, int first, int last, int nth, int depth, WarpSelect &scratch) {
    const int lane = int(threadIdx.x) % warpSize;
    const int size = last - first;
    Path *values = scratch.values;
    for (int index = lane; index < size; index += warpSize) values[index] = a[first + index];
    __syncwarp();
    int low = 0;
    int high = size;
    const int target = nth - first;
    bool heap = false;
    while (high - low > 3) {
        if (depth == 0) {
            heap = true;
            break;
        }
        --depth;
        if (lane == 0) place_median(values, low, high);
        __syncwarp();
        const uint64_t pivot = key(values[low]);
        const int span = high - low - 1;
        const int begin = low + 1 + span * lane / warpSize;
        const int end = low + 1 + span * (lane + 1) / warpSize;
        unsigned less = 0;
        for (int position = begin; position < end; ++position)
            less += key(values[position]) < pivot;
        const int cut = low + 1 + int(__shfl(warp_inclusive_sum(less), warpSize - 1));
        unsigned left_count = 0;
        unsigned right_count = 0;
        for (int position = begin; position < end; ++position) {
            const uint64_t candidate = key(values[position]);
            left_count += position < cut && candidate > pivot;
            right_count += position >= cut && candidate < pivot;
        }
        const unsigned packed = left_count | (right_count << 16);
        const unsigned inclusive = warp_inclusive_sum(packed);
        const unsigned total = __shfl(inclusive, warpSize - 1);
        unsigned left_rank = (inclusive - packed) & 0xffffU;
        unsigned right_rank = (inclusive - packed) >> 16;
        const unsigned pairs = total & 0xffffU;
        const unsigned right_total = total >> 16;
        for (int position = begin; position < end; ++position) {
            const uint64_t candidate = key(values[position]);
            if (position < cut && candidate > pivot)
                scratch.wrong_left[left_rank++] = uint16_t(position);
            if (position >= cut && candidate < pivot)
                scratch.wrong_right[right_total - 1 - right_rank++] = uint16_t(position);
        }
        __syncwarp();
        for (unsigned pair = unsigned(lane); pair < pairs; pair += warpSize)
            exchange(values, scratch.wrong_left[pair], scratch.wrong_right[pair]);
        __syncwarp();
        if (cut <= target) low = cut;
        else high = cut;
    }
    if (lane == 0) {
        if (heap) heap_finish(values, low, target, high);
        else insertion_finish(values, low, high);
    }
    __syncwarp();
    for (int index = lane; index < size; index += warpSize) a[first + index] = values[index];
}

// Candidates have unique histories, hence unique (cost, history) keys.
// With the median moved to first, there is no pivot-equal key in the scanned
// range. Its Hoare cut is first+1+count(key<pivot). The serial scanner pairs
// precisely the misplaced positions below that cut in increasing order
// with misplaced positions above it in decreasing order. Those pairs are
// disjoint: collecting and swapping them in parallel preserves *every*
// serial swap and the entire output permutation, not just the top-k set.
__device__ __noinline__ void select_prefix(Path *a, int nth, int size,
                              uint16_t *wrong_left, uint16_t *wrong_right,
                              unsigned *sums, WarpSelect &scratch) {
    const int thread = int(threadIdx.x);
    int first = 0;
    int last = size;
    int depth = 0;
    for (int n = size; n > 1; n >>= 1) depth += 2;
    while (last - first > 3) {
        if (depth == 0) {
            if (thread == 0) heap_finish(a, first, nth, last);
            __syncthreads();
            return;
        }
        if (last - first <= kWarpSelect) {
            // Short ranges are barrier-bound on the block: finish on warp 0.
            if (thread < warpSize) warp_select(a, first, last, nth, depth, scratch);
            __syncthreads();
            return;
        }
        --depth;
        if (thread == 0) {
            // Find the median of these three *distinct* keys by sorting
            // their indices only; only the eventual pivot swap is applied.
            place_median(a, first, last);
        }
        __syncthreads();
        const uint64_t pivot = key(a[first]);
        const int span = last - first - 1;
        const int begin = first + 1 + span * thread / kThreads;
        const int end = first + 1 + span * (thread + 1) / kThreads;
        unsigned less = 0;
        for (int position = begin; position < end; ++position)
            less += key(a[position]) < pivot;
        const int cut = first + 1 + int(prefix_sum(less, sums).total);

        unsigned left_count = 0;
        unsigned right_count = 0;
        for (int position = begin; position < end; ++position) {
            const uint64_t candidate = key(a[position]);
            left_count += position < cut && candidate > pivot;
            right_count += position >= cut && candidate < pivot;
        }
        // Each total is below 8192, so the low 16-bit sum cannot carry into
        // the high one. One scan supplies both independent ordered lists.
        const Prefix ranks = prefix_sum(left_count | (right_count << 16), sums);
        unsigned left_rank = ranks.offset & 0xffffU;
        unsigned right_rank = ranks.offset >> 16;
        const unsigned pairs = ranks.total & 0xffffU;
        const unsigned right_total = ranks.total >> 16;
        for (int position = begin; position < end; ++position) {
            const uint64_t candidate = key(a[position]);
            if (position < cut && candidate > pivot)
                wrong_left[left_rank++] = uint16_t(position);
            if (position >= cut && candidate < pivot)
                wrong_right[right_total - 1 - right_rank++] = uint16_t(position);
        }
        __syncthreads();
        for (unsigned pair = unsigned(thread); pair < pairs; pair += kThreads)
            exchange(a, wrong_left[pair], wrong_right[pair]);
        __syncthreads();
        if (cut <= nth) first = cut;
        else last = cut;
    }
    if (thread == 0) insertion_finish(a, first, last);
    __syncthreads();
}

// Stable top-k: candidates have unique (cost, history) keys, so the k smallest
// form a unique set regardless of any permutation. A most-significant-digit
// radix search finds the k-th key; the set is then compacted in history
// order. This is NOT libstdc++'s nth_element order: the caller replays that
// exact permutation whenever an order-dependent equal-cost decision arises.
// Each warp owns a contiguous segment walked by interleaved lanes (coalesced),
// and Cached > 0 keeps that many keys per lane in registers.
// Register-cached keys per lane. HT3 steps expand at most 1024 positions
// (4 per lane); HT4 needs 8. Fewer keys keep HT3 at 12 waves/SIMD.
template<int Bits> constexpr int kCachedKeys = Bits == 3 ? 4 : 8;
template<int Bits, int Cached>
__device__ __forceinline__ void select_set(const Path *candidates, int size, int count,
                                           BlockState<Bits> &shared, Trace *trace, bool record,
                                           const uint64_t (&provided)[kCachedKeys<Bits>]) {
    static_assert(Cached == 0 || Cached == kCachedKeys<Bits>, "cached keys come from expansion");
    constexpr int symbols = 1 << Bits;
    const int warps = kThreads / warpSize;
    constexpr int Slots = Cached > 0 ? Cached : 1;
    const int thread = int(threadIdx.x);
    const int lane = thread % warpSize;
    const int warp = thread / warpSize;
    const int segment_begin = size * warp / warps;
    const int segment_end = size * (warp + 1) / warps;
    const int rounds = (segment_end - segment_begin + warpSize - 1) / warpSize;
    auto load = [&](int index) -> uint64_t {
        const int position = segment_begin + index * warpSize + lane;
        if (position >= segment_end) return ~uint64_t(0);
        // Expanded arrays keep non-finite (unreached) histories in place.
        const Path candidate = candidates[position];
        return finite_cost(candidate.cost) ? key(candidate) : ~uint64_t(0);
    };
    uint64_t keys[Slots];
    if (Cached > 0) {
        #pragma unroll
        for (int index = 0; index < Slots; ++index) keys[index] = provided[index];
    }
    // OR and AND over valid keys: their XOR marks every bit that differs
    // between some pair, and all bits above the highest one are common.
    uint64_t any_bits = 0;
    uint64_t all_bits = ~uint64_t(0);
    const int scans = Cached > 0 ? Slots : rounds;
    #pragma unroll
    for (int index = 0; index < scans; ++index) {
        const uint64_t value = Cached > 0 ? keys[index] : load(index);
        if (value != ~uint64_t(0)) {
            any_bits |= value;
            all_bits &= value;
        }
    }
    for (int distance = warpSize / 2; distance > 0; distance /= 2) {
        any_bits |= __shfl_xor(any_bits, distance);
        all_bits &= __shfl_xor(all_bits, distance);
    }
    if (thread == 0) {
        shared.key_or = 0;
        shared.key_and = ~0ULL;
    }
    __syncthreads();
    if (lane == 0) {
        atomicOr(&shared.key_or, (unsigned long long)any_bits);
        atomicAnd(&shared.key_and, (unsigned long long)all_bits);
    }
    __syncthreads();
    // At least two unique valid keys guarantee a differing bit. Keys use 47
    // bits, so the all-ones invalid marker never matches a selected prefix.
    int shift = 64 - __clzll((long long)(shared.key_or ^ shared.key_and));
    uint64_t high = shared.key_or >> shift;
    int rank = count - 1;
    while (shift > 0) {
        const int width = shift < 8 ? shift : 8;
        shift -= width;
        const unsigned mask = (1U << width) - 1U;
        shared.histogram[thread] = 0;
        __syncthreads();
        const int passes = Cached > 0 ? Slots : rounds;
        #pragma unroll
        for (int index = 0; index < passes; ++index) {
            const uint64_t value = Cached > 0 ? keys[index] : load(index);
            const bool valid = value != ~uint64_t(0) && (value >> (shift + width)) == high;
            const unsigned digit = unsigned(value >> shift) & mask;
            // Plain shared atomics: replaying same-bin conflicts is far
            // cheaper than __match_any_sync aggregation (measured 2x per band).
            if (valid) atomicAdd(&shared.histogram[digit], 1U);
        }
        __syncthreads();
        if (thread < warpSize) {
            // Eight bins per lane in ascending order; find the bin holding rank.
            unsigned local = 0;
            for (int bin = lane * 8; bin < lane * 8 + 8; ++bin) local += shared.histogram[bin];
            const unsigned inclusive = warp_inclusive_sum(local);
            const unsigned exclusive = inclusive - local;
            if (unsigned(rank) >= exclusive && unsigned(rank) < inclusive) {
                unsigned before_bin = exclusive;
                for (int bin = lane * 8; bin < lane * 8 + 8; ++bin) {
                    const unsigned here = shared.histogram[bin];
                    if (unsigned(rank) < before_bin + here) {
                        shared.digit = bin;
                        shared.rank = rank - int(before_bin);
                        // The rank'th key is this bin's largest: the whole
                        // bin is selected, so lower digits cannot matter.
                        shared.stop = unsigned(rank) - before_bin + 1U == here;
                        break;
                    }
                    before_bin += here;
                }
            }
        }
        __syncthreads();
        high = (high << width) | uint64_t(shared.digit);
        rank = shared.rank;
        const bool stop = shared.stop;
        __syncthreads();
        if (stop) break;
    }
    // Selected: every valid key whose prefix above `shift` is <= high. The
    // invalid marker's prefix exceeds any 47-bit key prefix.
    auto chosen = [&](uint64_t value) { return (value >> shift) <= high; };
    // Stable compaction: warp offsets from per-warp totals, ballot ranks inside.
    unsigned selected = 0;
    const int passes = Cached > 0 ? Slots : rounds;
    #pragma unroll
    for (int index = 0; index < passes; ++index)
        selected += chosen(Cached > 0 ? keys[index] : load(index));
    for (int distance = warpSize / 2; distance > 0; distance /= 2)
        selected += __shfl_xor(selected, distance);
    if (lane == 0) shared.warp_sums[warp] = selected;
    __syncthreads();
    unsigned slot = 0;
    for (int other = 0; other < warp; ++other) slot += shared.warp_sums[other];
    const unsigned long long earlier_lanes = (1ULL << lane) - 1ULL;
    #pragma unroll
    for (int index = 0; index < passes; ++index) {
        const uint64_t value = Cached > 0 ? keys[index] : load(index);
        const bool keep = chosen(value);
        const unsigned long long kept = __ballot(keep);
        if (keep) {
            const Path candidate = candidates[segment_begin + index * warpSize + lane];
            const unsigned target = slot + __popcll(kept & earlier_lanes);
            shared.active[target] = candidate;
            if (record) trace[target] = {candidate.parent, uint16_t(candidate.state & (symbols - 1))};
        }
        slot += __popcll(kept);
    }
    __syncthreads();
}

// Stable block-wide compaction of finite paths (ascending history order is
// kept). Optionally records each kept path's trace entry at the same slot.
// Must be called by the whole block; returns the number kept.
template<int Bits>
__device__ int compact_finite(const Path *source, int size, Path *destination, Trace *trace,
                              BlockState<Bits> &shared) {
    constexpr int symbols = 1 << Bits;
    const int warps = kThreads / warpSize;
    const int thread = int(threadIdx.x);
    const int lane = thread % warpSize;
    const int warp = thread / warpSize;
    const int segment_begin = size * warp / warps;
    const int segment_end = size * (warp + 1) / warps;
    unsigned finite = 0;
    for (int position = segment_begin + lane; position < segment_end; position += warpSize)
        finite += finite_cost(source[position].cost);
    for (int distance = warpSize / 2; distance > 0; distance /= 2)
        finite += __shfl_xor(finite, distance);
    if (lane == 0) shared.warp_sums[warp] = finite;
    __syncthreads();
    unsigned slot = 0;
    unsigned total = 0;
    for (int other = 0; other < warps; ++other) {
        const unsigned here = shared.warp_sums[other];
        slot += other < warp ? here : 0;
        total += here;
    }
    const unsigned long long earlier_lanes = (1ULL << lane) - 1ULL;
    for (int base = segment_begin; base < segment_end; base += warpSize) {
        const int position = base + lane;
        Path candidate{};
        bool keep = false;
        if (position < segment_end) {
            candidate = source[position];
            keep = finite_cost(candidate.cost);
        }
        const unsigned long long kept = __ballot(keep);
        if (keep) {
            const unsigned target = slot + __popcll(kept & earlier_lanes);
            destination[target] = candidate;
            if (trace) trace[target] = {candidate.parent, uint16_t(candidate.state & (symbols - 1))};
        }
        slot += __popcll(kept);
    }
    __syncthreads();
    return int(total);
}

// Exact libstdc++ positions of the previous step's selection, indexed by the
// fast (history-ordered) active position. The step's candidate keys alone
// determine nth_element's permutation, so replaying it on the saved array
// reproduces the CPU order. This path is rare, so a linear state search over
// the (unique-state) active array maps each exact position to its fast index.
template<int Bits>
__device__ void exact_ranks(const TileWorkspace &workspace, const Path *saved, int saved_size,
                            Path *scratch, int available, int count, BlockState<Bits> &shared) {
    const int thread = int(threadIdx.x);
    if (count < available) {
        // The saved step is an expanded array with unreached gaps; the CPU's
        // nth_element saw its finite entries in the same ascending order.
        compact_finite<Bits>(saved, saved_size, scratch, nullptr, shared);
        select_prefix(scratch, count, available, workspace.wrong_left, workspace.wrong_right,
                      shared.warp_sums, shared.select);
        for (int index = thread; index < count; index += kThreads) {
            const uint16_t state = scratch[index].state;
            int fast = 0;
            while (shared.active[fast].state != state) ++fast;
            shared.exact_rank[fast] = uint16_t(index);
        }
    } else {
        // No nth_element call: the CPU kept ascending history order too.
        for (int index = thread; index < count; index += kThreads)
            shared.exact_rank[index] = uint16_t(index);
    }
    __syncthreads();
}

// The standard HT codebook entry, as ht_codebook_value computes it: byte sum
// of state*0x83dcd12d, an exact int->f32 conversion (|v| < 2^24), an exact
// power-of-two scale, then binary16 RNE. Used only after the host has checked
// that the supplied table equals this function bit for bit.
__host__ __device__ __forceinline__ float standard_codebook(unsigned state) {
    const unsigned hash = state * 0x83dcd12dU;
    const int sum = int((hash & 255) + ((hash >> 8) & 255) + ((hash >> 16) & 255) + (hash >> 24));
    const float value = float(1774 * sum - 905216) * 0x1p-18f;
    return __half2float(__float2half_rn(value));
}

// One complete two-pass tile search. Every early return is block-uniform.
// The workspace slot belongs to the calling block, not to the tile: a
// persistent block reuses it serially for each tile it owns.
template<int Bits, bool Standard>
__device__ void encode_tile(DeviceStorage *storage, const float *tiles, int width, int tile, int slot,
                            BlockState<Bits> &shared) {
    constexpr int symbols = 1 << Bits;
    constexpr int histories = kStates >> Bits;
    constexpr int groups = histories >> Bits;
    constexpr int closing_symbols = (16 - Bits + Bits - 1) / Bits;
    constexpr int group_words = (groups + 31) / 32;
    const int thread = int(threadIdx.x);
    const float *x = tiles + size_t(tile) * kLength;
    const float *cb = storage->codebook;
    const TileWorkspace workspace = tile_workspace(
        storage->workspace + size_t(slot) * tile_workspace_bytes(Bits, width), histories);
    for (int group = thread; group < groups; group += kThreads) shared.head[group] = kMissing;
    if (thread == 0) {
        storage->status[tile] = {0, 0, 0};
        shared.boundary = 0;
    }
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
        Path *current = workspace.candidates;
        Path *previous = workspace.previous;
        // Pass one starts from one fixed parent: no selection preceded it.
        int previous_available = 1;
        int previous_size = 1;
        if (thread == 0) {
            // Pass one's initial H paths are implicit (state=position,
            // cost=0); later active arrays never exceed the requested beam.
            shared.count = pass == 0 ? 0 : 1;
            if (pass == 1) shared.active[0] = {0, uint16_t(shared.boundary), 0.f};
        }
        __syncthreads();
        for (int t = 0; t < kLength; ++t) {
            const bool initial = pass == 0 && t == 0;
            for (int word = thread; word < group_words; word += kThreads)
                shared.touched[word] = initial ? 0xffffffffU : 0;
            __syncthreads();
            if (!initial && thread < shared.count) {
                const int state = shared.active[thread].state;
                const int group = state & (groups - 1);
                shared.next[thread] = uint16_t(atomicExch(&shared.head[group], unsigned(thread)));
                atomicOr(&shared.touched[group >> 5], 1U << (group & 31));
            }
            __syncthreads();

            // Compact touched groups in ascending order before expansion.
            // Contiguous per-lane spans plus the prefix scan preserve the
            // CPU's ascending history gather, including sparse groups.
            const int group_begin = groups * thread / kThreads;
            const int group_end = groups * (thread + 1) / kThreads;
            unsigned touched = 0;
            for (int group = group_begin; group < group_end; ++group)
                touched += (shared.touched[group >> 5] >> (group & 31)) & 1U;
            const Prefix group_ranks = prefix_sum(touched, shared.warp_sums);
            unsigned group_slot = group_ranks.offset;
            for (int group = group_begin; group < group_end; ++group)
                if ((shared.touched[group >> 5] >> (group & 31)) & 1U)
                    shared.ordered_groups[group_slot++] = uint16_t(group);
            __syncthreads();

            int closure_mask = 0;
            int closure_value = 0;
            if (pass == 1 && t >= kLength - closing_symbols) {
                const int shift = Bits * (kLength - 1 - t);
                closure_mask = (histories - 1) >> shift;
                if (closure_mask > symbols - 1) closure_mask = symbols - 1;
                closure_value = (shared.boundary >> shift) & closure_mask;
            }
            const int expanded_count = int(group_ranks.total) * symbols;
            // Each warp owns one contiguous segment; its lanes walk it
            // interleaved. Adjacent lanes then share a group (broadcast
            // parent reads) and touch adjacent codebook/expanded entries,
            // while ballot ranks keep the compaction in history order.
            const int lane = thread % warpSize;
            const int warp = thread / warpSize;
            const int warps = kThreads / warpSize;
            const int segment_begin = expanded_count * warp / warps;
            const int segment_end = expanded_count * (warp + 1) / warps;
            const float value = x[t];
            unsigned finite = 0;
            uint64_t keys[kCachedKeys<Bits>];
            // Fast pass: break equal costs by fast position and only flag
            // them. Without a minimal-cost tie, the winner is independent of
            // active order. With one, replay the exact order and redo it.
            for (int exact = 0; exact < 2; ++exact) {
                int tied = 0;
                finite = 0;
                // Same warp-segment/lane mapping as select_set: round i of
                // this lane is select_set's cached key i, kept in registers.
                auto expand = [&](int position) -> Path {
                    const int group = shared.ordered_groups[position >> Bits];
                    const int symbol = position & (symbols - 1);
                    const int history = (group << Bits) | symbol;
                    Path best = {kMissing, uint16_t(history), __int_as_float(0x7f800000)};
                    unsigned best_order = 0xffffffffU;
                    bool best_tied = false;
                    // HT3's first closing symbol constrains only its low bit;
                    // its other two bits remain free. Subsequent symbols are
                    // fully fixed, just as d += mask+1 in the CPU loop.
                    // Initial: all histories of the group are parents with zero
                    // cost; the parent index is the history itself. Later:
                    // walk only this group's actual parents.
                    const bool open = (symbol & closure_mask) == closure_value;
                    int high = 0;
                    int parent = open ? (initial ? group : int(shared.head[group])) : int(kMissing);
                    while (parent != kMissing) {
                        {
                            const int previous = initial ? parent : int(shared.active[parent].state);
                            const float prior = initial ? 0.f : shared.active[parent].cost;
                            const int state = (previous << Bits) | symbol;
                            const float entry = Standard ? standard_codebook(unsigned(state)) : cb[state];
                            const float delta = __fsub_rn(value, entry);
                            const float squared = __fmul_rn(delta, delta);
                            const float cost = __fadd_rn(prior, squared);
                            // The CPU visits parents in active order with a
                            // strict update: minimum cost, then lowest order.
                            const unsigned order = exact ? shared.exact_rank[parent] : unsigned(parent);
                            if (cost < best.cost) {
                                best.cost = cost;
                                best.parent = uint16_t(parent);
                                best_order = order;
                                best_tied = false;
                            } else if (cost == best.cost) {
                                best_tied = true;
                                if (order < best_order) {
                                    best.parent = uint16_t(parent);
                                    best_order = order;
                                }
                            }
                        }
                        if (initial) {
                            ++high;
                            parent = high < symbols ? (group | (high << (16 - 2 * Bits))) : int(kMissing);
                        } else {
                            parent = shared.next[parent];
                        }
                    }
                    current[position] = best;
                    finite += finite_cost(best.cost);
                    // Initial parents are in exact (identity) order already.
                    tied |= best_tied && !initial && finite_cost(best.cost);
                    return best;
                };
                #pragma unroll
                for (int round = 0; round < kCachedKeys<Bits>; ++round) {
                    const int position = segment_begin + lane + round * warpSize;
                    keys[round] = ~uint64_t(0);
                    if (position < segment_end) {
                        const Path best = expand(position);
                        if (finite_cost(best.cost)) keys[round] = key(best);
                    }
                }
                for (int position = segment_begin + lane + kCachedKeys<Bits> * warpSize; position < segment_end;
                     position += warpSize)
                    expand(position);
                if (exact || !__syncthreads_or(tied)) break;
                exact_ranks<Bits>(workspace, previous, previous_size, workspace.expanded,
                                  previous_available, shared.count, shared);
            }
            for (int distance = warpSize / 2; distance > 0; distance /= 2)
                finite += __shfl_xor(finite, distance);
            if (lane == 0) shared.warp_sums[warp] = finite;
            __syncthreads();
            unsigned available_total = 0;
            for (int other = 0; other < warps; ++other) available_total += shared.warp_sums[other];
            // No compaction: selection reads the expanded array directly and
            // skips unreached histories, which keeps ascending history order.
            // Remove the old position map before overwriting active. Only
            // touched entries need clearing, not all H entries every step.
            if (!initial && thread < shared.count)
                shared.head[shared.active[thread].state & (groups - 1)] = kMissing;
            __syncthreads();
            const int available = int(available_total);
            if (available == 0) {
                if (thread == 0) storage->status[tile] = {1, pass, t};
                return;
            }
            const int count = width < available ? width : available;
            Trace *trace = workspace.trace + t * width;
            if (count < available) {
                // Register-cached keys cover every steady-state beam<=128 step.
                if (expanded_count <= kCachedKeys<Bits> * kThreads)
                    select_set<Bits, kCachedKeys<Bits>>(current, expanded_count, count, shared, trace, pass == 1, keys);
                else
                    select_set<Bits, 0>(current, expanded_count, count, shared, trace, pass == 1, keys);
            } else {
                compact_finite<Bits>(current, expanded_count, shared.active, pass == 1 ? trace : nullptr, shared);
            }
            if (thread == 0) shared.count = count;
            previous_available = available;
            previous_size = expanded_count;
            Path *const swap = previous;
            previous = current;
            current = swap;
            __syncthreads();
        }
        // First minimal cost in exact active order. Without an equal-cost
        // competitor, any order finds the same unique minimum.
        if (thread == 0) {
            int end = 0;
            bool tied = false;
            for (int position = 1; position < shared.count; ++position) {
                if (shared.active[position].cost < shared.active[end].cost) {
                    end = position;
                    tied = false;
                } else if (shared.active[position].cost == shared.active[end].cost) {
                    tied = true;
                }
            }
            shared.end = tied ? -1 - end : end;
        }
        __syncthreads();
        if (shared.end < 0 && shared.count < previous_available) {
            // Replay the last step's exact permutation on scratch storage.
            Path *scratch = workspace.expanded;
            compact_finite<Bits>(previous, previous_size, scratch, nullptr, shared);
            select_prefix(scratch, shared.count, previous_available, workspace.wrong_left,
                          workspace.wrong_right, shared.warp_sums, shared.select);
            if (thread == 0) {
                const float minimum = shared.active[-1 - shared.end].cost;
                int exact_state = -1;
                for (int position = 0; position < shared.count && exact_state < 0; ++position)
                    if (scratch[position].cost == minimum) exact_state = scratch[position].state;
                int end = 0;
                while (shared.active[end].state != exact_state) ++end;
                shared.end = end;
            }
            __syncthreads();
        }
        if (thread == 0) {
            // A tie with no nth_element call keeps history (= fast) order.
            int end = shared.end < 0 ? -1 - shared.end : shared.end;
            if (pass == 0) {
                shared.boundary = shared.active[end].state;
            } else {
                for (int t = kLength - 1; t >= 0; --t) {
                    const Trace back = workspace.trace[t * width + end];
                    storage->codes[tile * kLength + t] = uint8_t(back.symbol);
                    end = back.parent;
                }
            }
        }
        __syncthreads();
    }
}

// Persistent grid: launch only as many blocks as are co-resident (and have
// workspace slots); each strides over the batch. Every tile still runs the
// identical search, so this changes scheduling, never encoded bytes. It
// removes the partially filled second wave of a one-block-per-tile launch.
// Four blocks per SM (<= 64 registers) measured fastest on the RTX 4090
// Laptop for HT3 and HT4 at beam 128; fewer registers spill into latency.
template<int Bits, bool Standard>
__global__ __launch_bounds__(kThreads)
// gfx1151 measurements: HT4 is fastest at 16 waves/SIMD with the uploaded
// table; HT3 at the compiler's default with the computed codebook.
__attribute__((amdgpu_waves_per_eu(Bits == 4 ? 16 : 1))) void encode_tiles(DeviceStorage *storage, const float *tiles, int width, int count) {
    __shared__ BlockState<Bits> shared;
    for (int tile = int(blockIdx.x); tile < count; tile += int(gridDim.x)) {
        encode_tile<Bits, Standard>(storage, tiles, width, tile, int(blockIdx.x), shared);
        __syncthreads();
    }
}

thread_local char last_error[768] = {};

int fail(int code, const char *message) {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
    return code;
}

int hip_failure(hipError_t error, const char *operation) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s (%s, %d)", operation,
                  hipGetErrorString(error), hipGetErrorName(error), int(error));
    return -3;
}

void append_hip_failure(hipError_t error, const char *operation) {
    const size_t used = std::strlen(last_error);
    if (used < sizeof(last_error) - 1)
        std::snprintf(last_error + used, sizeof(last_error) - used, "; %s: %s (%d)",
                      operation, hipGetErrorString(error), int(error));
}

struct Context {
    std::mutex mutex;
    hipStream_t stream = nullptr;
    DeviceStorage *storage = nullptr;
    DecodeStorage *decode_storage = nullptr;
    I4rStorage *i4r_storage = nullptr;
    float *ht_tiles = nullptr; // Rotated tiles of one GPU-rotated HT call (grow-only).
    size_t ht_tiles_capacity = 0;
    std::array<float, kStates> cached_codebook;
    std::array<float, kStates> decode_cached_codebook;
    // Remains alive while an error path drains a queued D2H transfer.
    std::array<TileStatus, kBatch> tile_status;
    I4rStatus i4r_status{};
    bool codebook_valid = false;
    bool decode_codebook_valid = false;
    int multiprocessors = 0;
    int resident_ht[2][2] = {}; // Co-resident blocks: [bits-3][standard codebook].
    bool standard_codebook_cached = false; // cached_codebook is the standard table.

    int initialize() {
        if (stream) return 0;
        hipDeviceProp_t properties{};
        hipError_t error = hipGetDeviceProperties(&properties, 0);
        if (error != hipSuccess) return hip_failure(error, "query HIP device 0");
        const char *arch = properties.gcnArchName;
        if (std::strncmp(arch, "gfx1151", 7) != 0 || (arch[7] != '\0' && arch[7] != ':')) {
            std::snprintf(last_error, sizeof(last_error),
                          "HIP encoding/verification supports only gfx1151; visible device 0 is %s (%s)",
                          properties.name, arch);
            return -2;
        }
        multiprocessors = properties.multiProcessorCount;
        int least_priority = 0;
        int greatest_priority = 0;
        error = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (error != hipSuccess) return hip_failure(error, "query HIP stream priorities");
        error = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least_priority);
        if (error != hipSuccess) return hip_failure(error, "create low-priority HIP stream");
        return 0;
    }

    int ensure_encoder() {
        if (storage) return 0;
        const hipError_t error = hipMalloc(reinterpret_cast<void **>(&storage), sizeof(DeviceStorage));
        if (error != hipSuccess) {
            storage = nullptr;
            return hip_failure(error, "allocate bounded HIP HT workspace");
        }
        return 0;
    }

    int ensure_decoder() {
        if (decode_storage) return 0;
        const hipError_t error = hipMalloc(reinterpret_cast<void **>(&decode_storage), sizeof(DecodeStorage));
        if (error != hipSuccess) {
            decode_storage = nullptr;
            return hip_failure(error, "allocate bounded HIP verification workspace");
        }
        return 0;
    }

    int ensure_i4r_encoder() {
        if (i4r_storage) return 0;
        const hipError_t error = hipMalloc(reinterpret_cast<void **>(&i4r_storage), sizeof(I4rStorage));
        if (error != hipSuccess) {
            i4r_storage = nullptr;
            return hip_failure(error, "allocate bounded 52,756,488-byte HIP I4R workspace");
        }
        return 0;
    }

    ~Context() {
        // Process-lifetime resources only; never touch another device's
        // allocations or reset a device shared with an inference server.
        if (!stream && !storage && !decode_storage && !i4r_storage && !ht_tiles) return;
        int previous = 0;
        hipError_t error = hipGetDevice(&previous);
        if (error == hipSuccess) error = hipSetDevice(0);
        if (error != hipSuccess) {
            std::fprintf(stderr, "HIP HT cleanup: select device 0: %s\n", hipGetErrorString(error));
            return;
        }
        if (stream) {
            error = hipStreamSynchronize(stream);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP HT cleanup: synchronize: %s\n", hipGetErrorString(error));
        }
        if (storage) {
            error = hipFree(storage);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP HT cleanup: free workspace: %s\n", hipGetErrorString(error));
        }
        if (decode_storage) {
            error = hipFree(decode_storage);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP cleanup: free verification workspace: %s\n", hipGetErrorString(error));
        }
        if (ht_tiles) {
            error = hipFree(ht_tiles);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP cleanup: free HT tiles: %s\n", hipGetErrorString(error));
        }
        if (i4r_storage) {
            error = hipFree(i4r_storage);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP cleanup: free I4R workspace: %s\n", hipGetErrorString(error));
        }
        if (stream) {
            error = hipStreamDestroy(stream);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP HT cleanup: destroy stream: %s\n", hipGetErrorString(error));
        }
        if (previous != 0) {
            error = hipSetDevice(previous);
            if (error != hipSuccess)
                std::fprintf(stderr, "HIP HT cleanup: restore device: %s\n", hipGetErrorString(error));
        }
    }
};

Context &context() {
    static Context value;
    return value;
}

int check_host_rounding() {
    if (std::fegetround() != FE_TONEAREST)
        return fail(-2, "HIP byte equivalence requires CPU round-to-nearest floating-point mode");
#if defined(__x86_64__) || defined(__i386__)
    // AVX/SSE honor MXCSR rather than just the C floating-point environment.
    constexpr unsigned rounding_or_flush = (3u << 13) | (1u << 15) | (1u << 6);
    if ((_mm_getcsr() & rounding_or_flush) != 0)
        return fail(-2, "HIP byte equivalence requires CPU MXCSR round-to-nearest with FTZ/DAZ disabled");
#endif
    return 0;
}

template<class Operation> int with_context(Operation operation) {
    try {
        const int rounding = check_host_rounding();
        if (rounding != 0) return rounding;
        Context &state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!state.stream) {
            int devices = 0;
            const hipError_t error = hipGetDeviceCount(&devices);
            if (error != hipSuccess) return hip_failure(error, "enumerate HIP devices");
            if (devices == 0) return fail(-2, "no visible HIP device; gfx1151 is required");
        }
        int previous = 0;
        hipError_t error = hipGetDevice(&previous);
        if (error != hipSuccess) return hip_failure(error, "read caller's HIP device");
        if (previous != 0) {
            error = hipSetDevice(0);
            if (error != hipSuccess) return hip_failure(error, "select HIP device 0");
        }
        int result = state.initialize();
        if (result == 0) result = operation(state);
        if (result != 0 && state.stream) {
            // Do not return while an enqueued transfer could still access a
            // caller-owned buffer. Never trust a partially uploaded cache.
            state.codebook_valid = false;
            state.decode_codebook_valid = false;
            error = hipStreamSynchronize(state.stream);
            if (error != hipSuccess) append_hip_failure(error, "drain failed HIP operation");
        }
        if (previous != 0) {
            error = hipSetDevice(previous);
            if (error != hipSuccess) {
                if (result == 0) result = hip_failure(error, "restore caller's HIP device");
                else append_hip_failure(error, "restore caller's HIP device");
            }
        }
        return result;
    } catch (const std::exception &error) {
        std::snprintf(last_error, sizeof(last_error), "HIP HT host exception: %s", error.what());
        return -5;
    } catch (...) {
        return fail(-5, "unknown HIP HT host exception");
    }
}

bool address_overflows(const void *pointer, size_t bytes) {
    return reinterpret_cast<uintptr_t>(pointer) > std::numeric_limits<uintptr_t>::max() - bytes;
}

// device_tiles: `tiles` is device memory (GPU-rotated) read in place;
// otherwise host tiles are uploaded per batch.
int encode(Context &state, const float *tiles, const float *codebook,
           uint8_t *codes, int count, int beam, int bits, bool device_tiles = false) {
    const int allocation = state.ensure_encoder();
    if (allocation != 0) return allocation;
    const size_t codebook_bytes = sizeof(state.cached_codebook);
    if (!state.codebook_valid || std::memcmp(codebook, state.cached_codebook.data(), codebook_bytes) != 0) {
        // Cache contents, not the caller's address: an in-place changed
        // codebook must take effect on the very next call.
        std::memcpy(state.cached_codebook.data(), codebook, codebook_bytes);
        const hipError_t error = hipMemcpyAsync(state.storage->codebook, state.cached_codebook.data(),
                                               codebook_bytes, hipMemcpyHostToDevice, state.stream);
        if (error != hipSuccess) return hip_failure(error, "upload HIP codebook");
        state.codebook_valid = true;
        bool standard = true;
        for (unsigned entry = 0; entry < unsigned(kStates) && standard; ++entry) {
            const float expected = standard_codebook(entry);
            standard = std::memcmp(&expected, &state.cached_codebook[entry], sizeof(float)) == 0;
        }
        state.standard_codebook_cached = standard;
    }
    auto &status = state.tile_status;
    const size_t total = size_t(count);
    // A bit-identical standard table lets the kernel compute entries instead
    // of loading them; any other table uses the uploaded copy.
    const bool standard = bits == 3 && state.standard_codebook_cached;
    const void *kernel = bits == 3
        ? (standard ? reinterpret_cast<const void *>(encode_tiles<3, true>)
                    : reinterpret_cast<const void *>(encode_tiles<3, false>))
        : (standard ? reinterpret_cast<const void *>(encode_tiles<4, true>)
                    : reinterpret_cast<const void *>(encode_tiles<4, false>));
    int &resident = state.resident_ht[bits - 3][standard];
    if (resident == 0) {
        hipError_t error = hipSuccess;
        int per_multiprocessor = 0;
        error = hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor, kernel, kThreads, 0);
        if (error != hipSuccess) return hip_failure(error, "query HIP HT occupancy");
        resident = per_multiprocessor * state.multiprocessors;
        if (resident < 1) return fail(-2, "HIP HT kernel cannot be resident on device 0");
    }
    // Grid: one persistent block per workspace slot, at most one full wave.
    const size_t slots = kWorkspaceBytes / tile_workspace_bytes(bits, beam);
    const size_t grid_limit = slots < size_t(resident) ? slots : size_t(resident);
    const size_t batch_limit = size_t(kBatch);
    for (size_t base = 0; base < total; base += batch_limit) {
        int batch = int(total - base < batch_limit ? total - base : batch_limit);
        const unsigned grid = unsigned(size_t(batch) < grid_limit ? size_t(batch) : grid_limit);
        const size_t elements = size_t(batch) * kLength;
        const float *batch_tiles = device_tiles ? tiles + base * kLength : state.storage->tiles;
        void *arguments[] = {&state.storage, &batch_tiles, &beam, &batch};
        hipError_t error = hipSuccess;
        if (!device_tiles) {
            error = hipMemcpyAsync(state.storage->tiles, tiles + base * kLength,
                                    elements * sizeof(float), hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP tile batch");
        }
        // Check this launch's direct status: an earlier recoverable runtime
        // error must not become a false launch failure via hipGetLastError.
        // Fatal asynchronous device errors still fail; never reset the GPU.
        error = hipLaunchKernel(kernel, dim3(grid), dim3(kThreads), arguments, 0, state.stream);
        if (error != hipSuccess) return hip_failure(error, "launch HIP beam-search batch");
        error = hipMemcpyAsync(status.data(), state.storage->status, size_t(batch) * sizeof(TileStatus),
                               hipMemcpyDeviceToHost, state.stream);
        if (error != hipSuccess) return hip_failure(error, "download HIP tile status");
        error = hipMemcpyAsync(codes + base * kLength, state.storage->codes, elements,
                               hipMemcpyDeviceToHost, state.stream);
        if (error != hipSuccess) return hip_failure(error, "download HIP symbols");
        error = hipStreamSynchronize(state.stream);
        if (error != hipSuccess) return hip_failure(error, "complete HIP beam-search batch");
        for (int tile = 0; tile < batch; ++tile) {
            if (status[tile].code != 0) {
                std::snprintf(last_error, sizeof(last_error),
                              "HIP HT%d tile %zu has no finite survivor at pass %d, step %d; "
                              "check nonfinite inputs/codebook or squared-error overflow",
                              bits, base + size_t(tile), status[tile].pass + 1, status[tile].step);
                return -4;
            }
        }
    }
    return 0;
}

hipError_t copy_i4r_rectangle(void *destination, size_t destination_pitch,
                            const void *source, size_t source_pitch,
                            size_t width, size_t rows, hipMemcpyKind kind,
                            hipStream_t stream) {
    if (destination_pitch == width && source_pitch == width)
        return hipMemcpyAsync(destination, source, width * rows, kind, stream);
    // Only widths above 32768 columns need strided copies. Row copies avoid
    // a device-dependent 2D pitch ceiling for an otherwise valid wide host
    // matrix; ordinary whole bands use the single contiguous transfer above.
    for (size_t row = 0; row < rows; ++row) {
        const hipError_t error = hipMemcpyAsync(
            static_cast<uint8_t *>(destination) + row * destination_pitch,
            static_cast<const uint8_t *>(source) + row * source_pitch, width, kind, stream);
        if (error != hipSuccess) return error;
    }
    return hipSuccess;
}

int encode_i4r(Context &state, const float *source, const float *su, const float *sv,
               uint8_t *codes, uint8_t *scales, int bands, int cols) {
    const int allocation = state.ensure_i4r_encoder();
    if (allocation != 0) return allocation;
    I4rStorage *storage = state.i4r_storage;
    const size_t full_cols = size_t(cols);
    const size_t width_limit = full_cols < size_t(kI4rBlocks * 128)
        ? full_cols : size_t(kI4rBlocks * 128);
    const size_t band_limit = kI4rBlocks / (width_limit / 128);
    bool scale_failed = false;
    size_t scale_band = 0;
    size_t scale_col = 0;
    for (size_t band_base = 0; band_base < size_t(bands);) {
        const size_t remaining = size_t(bands) - band_base;
        const size_t batch_bands = remaining < band_limit ? remaining : band_limit;
        const size_t rows = batch_bands * 128;
        for (size_t col_base = 0; col_base < full_cols; col_base += width_limit) {
            const size_t width = full_cols - col_base < width_limit
                ? full_cols - col_base : width_limit;
            int batch_cols = int(width);
            hipError_t error = hipMemsetAsync(&storage->status, 0, sizeof(I4rStatus), state.stream);
            if (error != hipSuccess) return hip_failure(error, "clear HIP I4R batch status");
            error = hipMemcpyAsync(storage->su, su + col_base, width * sizeof(float),
                                   hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP I4R input signs");
            error = hipMemcpyAsync(storage->sv, sv + band_base * 128, rows * sizeof(float),
                                   hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP I4R output signs");
            error = copy_i4r_rectangle(storage->values, width * sizeof(float),
                source + band_base * 128 * full_cols + col_base, full_cols * sizeof(float),
                width * sizeof(float), rows, hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP I4R source rectangle");
            void *axis_arguments[] = {&storage, &batch_cols};
            // Direct launch status preserves recovery from a caller's stale
            // HIP error; hipGetLastError would misattribute that older error.
            error = hipLaunchKernel(reinterpret_cast<const void *>(i4r_forward_rows),
                dim3(unsigned(width / 128), unsigned(rows / 16)), dim3(kThreads),
                axis_arguments, 0, state.stream);
            if (error != hipSuccess) return hip_failure(error, "launch HIP I4R row butterflies");
            error = hipLaunchKernel(reinterpret_cast<const void *>(i4r_forward_columns),
                dim3(unsigned(width / 16), unsigned(batch_bands)), dim3(kThreads),
                axis_arguments, 0, state.stream);
            if (error != hipSuccess) return hip_failure(error, "launch HIP I4R column butterflies");
            if (!scale_failed) {
                void *quantize_arguments[] = {&storage};
                error = hipLaunchKernel(reinterpret_cast<const void *>(i4r_quantize),
                    dim3(unsigned(rows * width / 128 / kI4rGroups)), dim3(kI4rThreads),
                    quantize_arguments, 0, state.stream);
                if (error != hipSuccess) return hip_failure(error, "launch HIP I4R exact quantizer");
                error = copy_i4r_rectangle(codes + band_base * 128 * (full_cols / 2) + col_base / 2,
                    full_cols / 2, storage->codes, width / 2, width / 2, rows,
                    hipMemcpyDeviceToHost, state.stream);
                if (error != hipSuccess) return hip_failure(error, "download HIP I4R codes");
                error = copy_i4r_rectangle(scales + band_base * 128 * (full_cols / 64) + col_base / 64,
                    full_cols / 64, storage->scales, width / 64, width / 64, rows,
                    hipMemcpyDeviceToHost, state.stream);
                if (error != hipSuccess) return hip_failure(error, "download HIP I4R half scales");
            }
            // Context-owned status survives failed asynchronous copies until
            // with_context drains the stream before releasing caller buffers.
            error = hipMemcpyAsync(&state.i4r_status, &storage->status, sizeof(I4rStatus),
                                   hipMemcpyDeviceToHost, state.stream);
            if (error != hipSuccess) return hip_failure(error, "download HIP I4R batch status");
            error = hipStreamSynchronize(state.stream);
            if (error != hipSuccess) return hip_failure(error, "complete HIP I4R batch");
            if (state.i4r_status.rotation_overflow) {
                std::snprintf(last_error, sizeof(last_error),
                    "HIP I4R rotation overflows f32 in band rectangle starting at band %zu, column %zu; "
                    "reduce source magnitude before encoding", band_base, col_base);
                return -4;
            }
            if (!scale_failed && state.i4r_status.scale_overflow) {
                scale_failed = true;
                scale_band = band_base;
                scale_col = col_base;
                // CPU rotation validates the entire request before checking
                // half scales. Keep rotating later rectangles so their f32
                // overflow retains priority over this earlier half overflow.
            }
        }
        band_base += batch_bands;
    }
    if (scale_failed) {
        std::snprintf(last_error, sizeof(last_error),
            "HIP I4R quantization scale exceeds finite f16 range in band rectangle starting at "
            "band %zu, column %zu; value cannot be represented as a finite f16", scale_band, scale_col);
        return -4;
    }
    return 0;
}

constexpr int kHtRotationOverflow = -6;

// Rotate one or two (gate/up) source parts on the GPU into trellis tile
// order, then run the unchanged trellis search on those device tiles.
int encode_ht_rows(Context &state, const float *const *sources, const float *su,
                   const float *const *svs, int parts, const float *codebook, uint8_t *symbols,
                   int bands, int cols, int beam, int bits) {
    int allocation = state.ensure_i4r_encoder();
    if (allocation == 0) allocation = state.ensure_encoder();
    if (allocation != 0) return allocation;
    const size_t full_cols = size_t(cols);
    const size_t values = size_t(parts) * size_t(bands) * 128 * full_cols;
    if (state.ht_tiles_capacity < values) {
        if (state.ht_tiles) {
            const hipError_t error = hipStreamSynchronize(state.stream);
            if (error != hipSuccess) return hip_failure(error, "drain before growing HIP HT tiles");
            (void)hipFree(state.ht_tiles);
            state.ht_tiles = nullptr;
            state.ht_tiles_capacity = 0;
        }
        const hipError_t error = hipMalloc(reinterpret_cast<void **>(&state.ht_tiles), values * sizeof(float));
        if (error != hipSuccess) {
            state.ht_tiles = nullptr;
            return hip_failure(error, "allocate HIP HT rotated tiles");
        }
        state.ht_tiles_capacity = values;
    }
    I4rStorage *storage = state.i4r_storage;
    const size_t width_limit = full_cols < size_t(kI4rBlocks * 128) ? full_cols : size_t(kI4rBlocks * 128);
    const size_t band_limit = kI4rBlocks / (width_limit / 128);
    const int row_tiles = parts == 2 ? 16 : 8;
    hipError_t error = hipMemsetAsync(&storage->status, 0, sizeof(I4rStatus), state.stream);
    if (error != hipSuccess) return hip_failure(error, "clear HIP HT rotation status");
    for (int part = 0; part < parts; ++part) {
        int tile_offset = part * 8;
        for (size_t band_base = 0; band_base < size_t(bands);) {
            const size_t remaining = size_t(bands) - band_base;
            const size_t batch_bands = remaining < band_limit ? remaining : band_limit;
            const size_t rows = batch_bands * 128;
            for (size_t col_base = 0; col_base < full_cols; col_base += width_limit) {
                const size_t width = full_cols - col_base < width_limit ? full_cols - col_base : width_limit;
                int batch_cols = int(width);
                int row_offset = int(band_base * 128);
                int col_offset = int(col_base);
                int all_cols = cols;
                int tiles_per_row = row_tiles;
                error = hipMemcpyAsync(storage->su, su + col_base, width * sizeof(float),
                                        hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP HT input signs");
                error = hipMemcpyAsync(storage->sv, svs[part] + band_base * 128, rows * sizeof(float),
                                        hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP HT row scales");
                error = copy_i4r_rectangle(storage->values, width * sizeof(float),
                    sources[part] + band_base * 128 * full_cols + col_base, full_cols * sizeof(float),
                    width * sizeof(float), rows, hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP HT source rectangle");
                void *row_arguments[] = {&storage, &batch_cols};
                error = hipLaunchKernel(reinterpret_cast<const void *>(ht_forward_rows),
                    dim3(unsigned(width / 128), unsigned(rows / 16)), dim3(kThreads),
                    row_arguments, 0, state.stream);
                if (error != hipSuccess) return hip_failure(error, "launch HIP HT row butterflies");
                void *column_arguments[] = {&storage, &state.ht_tiles, &batch_cols, &row_offset,
                                            &col_offset, &all_cols, &tiles_per_row, &tile_offset};
                error = hipLaunchKernel(reinterpret_cast<const void *>(ht_forward_columns),
                    dim3(unsigned(width / 16), unsigned(batch_bands)), dim3(kThreads),
                    column_arguments, 0, state.stream);
                if (error != hipSuccess) return hip_failure(error, "launch HIP HT column butterflies");
            }
            band_base += batch_bands;
        }
    }
    error = hipMemcpyAsync(&state.i4r_status, &storage->status, sizeof(I4rStatus),
                            hipMemcpyDeviceToHost, state.stream);
    if (error != hipSuccess) return hip_failure(error, "download HIP HT rotation status");
    error = hipStreamSynchronize(state.stream);
    if (error != hipSuccess) return hip_failure(error, "complete HIP HT rotation");
    if (state.i4r_status.rotation_overflow)
        return fail(kHtRotationOverflow, "HIP HT rotation rejected the input; rerun on CPU for the exact error");
    return encode(state, state.ht_tiles, codebook, symbols, int(values / kLength), beam, bits, true);
}

hipError_t copy_decode_rectangle(void *destination, size_t destination_pitch,
                                const void *source, size_t source_pitch,
                                size_t width, size_t rows, hipMemcpyKind kind,
                                hipStream_t stream) {
    if (destination_pitch == width && source_pitch == width)
        return hipMemcpyAsync(destination, source, width * rows, kind, stream);
    return hipMemcpy2DAsync(destination, destination_pitch, source, source_pitch,
                            width, rows, kind, stream);
}

int decode(Context &state, const uint8_t *bytes, const uint16_t *scales,
           const float *codebook, const float *su, const float *sv, float *output,
           int bands, int cols, int bits, int row_tiles, int tile_offset) {
    const int allocation = state.ensure_decoder();
    if (allocation != 0) return allocation;
    DecodeStorage *storage = state.decode_storage;
    if (bits != 0) {
        const size_t bytes_count = sizeof(state.decode_cached_codebook);
        if (!state.decode_codebook_valid
            || std::memcmp(codebook, state.decode_cached_codebook.data(), bytes_count) != 0) {
            // Verification supplies its own independent codebook. Never use
            // the encoder's cache, nor assume stable pointer means stable data.
            std::memcpy(state.decode_cached_codebook.data(), codebook, bytes_count);
            const hipError_t error = hipMemcpyAsync(storage->codebook,
                state.decode_cached_codebook.data(), bytes_count, hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP verifier codebook");
            state.decode_codebook_valid = true;
        }
    }
    const size_t full_cols = size_t(cols);
    const size_t width_limit = full_cols < size_t(kDecodeBlocks * 128)
        ? full_cols : size_t(kDecodeBlocks * 128);
    const size_t band_limit = kDecodeBlocks / (width_limit / 128);
    int per_band_su = bits != 0;
    const void *row_kernel = bits == 0 ? reinterpret_cast<const void *>(decode_row_axis<0>)
        : bits == 3 ? reinterpret_cast<const void *>(decode_row_axis<3>)
        : reinterpret_cast<const void *>(decode_row_axis<4>);
    for (size_t band_base = 0; band_base < size_t(bands);) {
        const size_t remaining = size_t(bands) - band_base;
        const size_t batch_bands = remaining < band_limit ? remaining : band_limit;
        const size_t rows = batch_bands * 128;
        for (size_t col_base = 0; col_base < full_cols; col_base += width_limit) {
            const size_t width = full_cols - col_base < width_limit
                ? full_cols - col_base : width_limit;
            int batch_cols = int(width);
            // A horizontally split rectangle always contains one band.
            // Otherwise whole bands and their per-band metadata are contiguous.
            const float *input_su = su + (per_band_su ? band_base * full_cols : 0) + col_base;
            hipError_t error = hipMemcpyAsync(storage->su, input_su,
                width * (per_band_su ? batch_bands : 1) * sizeof(float),
                hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP verifier input rotation");
            error = hipMemcpyAsync(storage->sv, sv + band_base * 128, rows * sizeof(float),
                                   hipMemcpyHostToDevice, state.stream);
            if (error != hipSuccess) return hip_failure(error, "upload HIP verifier output rotation");
            if (bits == 0) {
                error = copy_decode_rectangle(storage->payload, width / 2,
                    bytes + band_base * 128 * (full_cols / 2) + col_base / 2, full_cols / 2,
                    width / 2, rows, hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP I4R codes");
                // The ABI's half words are little-endian bytes; the host may
                // supply an unaligned byte plane. No host uint16_t dereference.
                const uint8_t *scale_bytes = reinterpret_cast<const uint8_t *>(scales);
                error = copy_decode_rectangle(storage->scales, width / 64,
                    scale_bytes + band_base * 128 * (full_cols / 64) + col_base / 64,
                    full_cols / 64, width / 64, rows, hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP I4R scales");
            } else {
                const size_t column_bytes = size_t(row_tiles) * size_t(32 * bits);
                const size_t offset = (band_base * (full_cols / 16) + col_base / 16) * column_bytes;
                error = hipMemcpyAsync(storage->payload, bytes + offset,
                    batch_bands * (width / 16) * column_bytes, hipMemcpyHostToDevice, state.stream);
                if (error != hipSuccess) return hip_failure(error, "upload HIP HT verifier tiles");
            }
            void *row_arguments[] = {&storage, &batch_cols, &row_tiles, &tile_offset};
            error = hipLaunchKernel(row_kernel, dim3(unsigned(width / 128), unsigned(rows / 16)),
                                    dim3(kThreads), row_arguments, 0, state.stream);
            if (error != hipSuccess) return hip_failure(error, "launch HIP verifier row butterflies");
            void *column_arguments[] = {&storage, &batch_cols, &per_band_su};
            error = hipLaunchKernel(reinterpret_cast<const void *>(decode_column_axis),
                dim3(unsigned(width / 16), unsigned(rows / 128)), dim3(kThreads),
                column_arguments, 0, state.stream);
            if (error != hipSuccess) return hip_failure(error, "launch HIP verifier column butterflies");
            error = copy_decode_rectangle(output + band_base * 128 * full_cols + col_base,
                full_cols * sizeof(float), storage->output, width * sizeof(float),
                width * sizeof(float), rows, hipMemcpyDeviceToHost, state.stream);
            if (error != hipSuccess) return hip_failure(error, "download HIP verifier values");
            error = hipStreamSynchronize(state.stream);
            if (error != hipSuccess) return hip_failure(error, "complete HIP verifier batch");
        }
        band_base += batch_bands;
    }
    return 0;
}

int decode_extent(int bands, int cols, size_t &values) {
    if (bands <= 0 || cols <= 0 || cols % 128 != 0)
        return fail(-1, "HIP decode requires positive bands and a positive multiple-of-128 column count");
    if (size_t(bands) > std::numeric_limits<size_t>::max() / 128 / size_t(cols) / sizeof(float))
        return fail(-1, "HIP decode shape overflows the host address space");
    values = size_t(bands) * 128 * size_t(cols);
    return 0;
}
} // namespace

// Error text is per calling thread, valid until its next encode/decode/preflight.
// Failures may leave a partial output buffer; callers must discard it.
extern "C" const char *ht_hip_last_error() {
    return last_error;
}

extern "C" int ht_hip_preflight() {
    last_error[0] = '\0';
    // Initializes device, low-priority stream and the fixed workspace only;
    // no encode or large kernel is run by preflight.
    return with_context([](Context &state) { return state.ensure_encoder(); });
}

extern "C" int ht_encode_bits_hip(const float *tiles, const float *codebook,
                                   uint8_t *codes, int count, int beam, int bits) {
    last_error[0] = '\0';
    if (bits != 3 && bits != 4) return fail(-1, "HIP HT bits must be 3 or 4");
    if (beam < 1 || beam > kMaxBeam)
        return fail(-1, "HIP HT beam must be 1..256; exact beam 0 and larger beams are CPU-only");
    if (count < 0) return fail(-1, "HIP HT tile count must be nonnegative");
    if (count == 0) return 0;
    if (!tiles || !codebook || !codes)
        return fail(-1, "HIP HT nonempty input requires nonnull tiles, codebook and codes");
    if (size_t(count) > std::numeric_limits<size_t>::max() / kLength / sizeof(float))
        return fail(-1, "HIP HT tile count overflows the host address space");
    const size_t elements = size_t(count) * kLength;
    if (address_overflows(tiles, elements * sizeof(float)) ||
        address_overflows(codes, elements) || address_overflows(codebook, kStates * sizeof(float)))
        return fail(-1, "HIP HT buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return encode(state, tiles, codebook, codes, count, beam, bits);
    });
}

// parts == 1: one or more complete 128-row bands (row_tiles 8).
// parts == 2: one gate band and one up band sharing su (row_tiles 16, up at 8).
// symbols receives parts*bands*128*cols trellis symbols in tile order.
// Returns -6 when the rotation is invalid; callers rerun the CPU reference.
extern "C" int hgn_hip_encode_ht(const float *const *sources, const float *su,
                                  const float *const *svs, int parts, const float *codebook,
                                  uint8_t *symbols, int bands, int cols, int beam, int bits) {
    last_error[0] = '\0';
    if (bits != 3 && bits != 4) return fail(-1, "HIP HT bits must be 3 or 4");
    if (beam < 1 || beam > kMaxBeam)
        return fail(-1, "HIP HT beam must be 1..256; exact beam 0 and larger beams are CPU-only");
    if ((parts != 1 && parts != 2) || bands <= 0 || (parts == 2 && bands != 1) || cols <= 0 || cols % 128 != 0)
        return fail(-1, "HIP HT encode requires complete bands (one band per part when paired)");
    if (!sources || !su || !svs || !codebook || !symbols)
        return fail(-1, "HIP HT encode requires nonnull buffers");
    if (size_t(bands) * size_t(parts) > size_t(std::numeric_limits<int>::max()) / 128 / size_t(cols))
        return fail(-1, "HIP HT encode shape exceeds the native index range");
    for (int part = 0; part < parts; ++part)
        if (!sources[part] || !svs[part]) return fail(-1, "HIP HT encode requires nonnull part buffers");
    return with_context([&](Context &state) {
        return encode_ht_rows(state, sources, su, svs, parts, codebook, symbols, bands, cols, beam, bits);
    });
}

extern "C" int hgn_hip_i4r_preflight() {
    last_error[0] = '\0';
    // Common device/low-priority stream guards, but only the independent I4R
    // allocation: no HT beam storage, decoder storage, or kernel launches.
    return with_context([](Context &state) { return state.ensure_i4r_encoder(); });
}

extern "C" int hgn_hip_encode_i4r(const float *source, const float *su, const float *sv,
                                uint8_t *codes, uint8_t *scales, int bands, int cols) {
    last_error[0] = '\0';
    if (bands <= 0 || cols <= 0 || cols % 128 != 0)
        return fail(-1, "HIP I4R encode requires positive bands and a positive multiple-of-128 column count");
    if (size_t(bands) > std::numeric_limits<size_t>::max() / 128 / size_t(cols) / sizeof(float))
        return fail(-1, "HIP I4R encode shape overflows the host address space");
    if (!source || !su || !sv || !codes || !scales)
        return fail(-1, "HIP I4R encode requires nonnull buffers");
    const size_t values = size_t(bands) * 128 * size_t(cols);
    if (address_overflows(source, values * sizeof(float))
        || address_overflows(su, size_t(cols) * sizeof(float))
        || address_overflows(sv, size_t(bands) * 128 * sizeof(float))
        || address_overflows(codes, values / 2) || address_overflows(scales, values / 64))
        return fail(-1, "HIP I4R encode buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return encode_i4r(state, source, su, sv, codes, scales, bands, cols);
    });
}

extern "C" int hgn_hip_decode_preflight() {
    last_error[0] = '\0';
    // Allocate only the bounded decoder workspace; never the encoder's beam
    // storage, and never launch a kernel merely to discover availability.
    return with_context([](Context &state) { return state.ensure_decoder(); });
}

extern "C" int hgn_hip_decode_i4r(const uint8_t *codes, const uint16_t *scales,
                                  const float *su, const float *sv, float *out,
                                  int bands, int cols) {
    last_error[0] = '\0';
    size_t values = 0;
    const int shape = decode_extent(bands, cols, values);
    if (shape != 0) return shape;
    if (!codes || !scales || !su || !sv || !out)
        return fail(-1, "HIP I4R decode requires nonnull buffers");
    if (address_overflows(codes, values / 2) || address_overflows(scales, values / 64)
        || address_overflows(su, size_t(cols) * sizeof(float))
        || address_overflows(sv, size_t(bands) * 128 * sizeof(float))
        || address_overflows(out, values * sizeof(float)))
        return fail(-1, "HIP I4R decode buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return decode(state, codes, scales, nullptr, su, sv, out, bands, cols, 0, 8, 0);
    });
}

extern "C" int hgn_hip_decode_ht(const uint8_t *bytes, const float *codebook,
                                 const float *su, const float *sv, float *out,
                                 int bands, int cols, int bits, int row_tiles, int tile_offset) {
    last_error[0] = '\0';
    if (bits != 3 && bits != 4) return fail(-1, "invalid HT verifier width");
    if (!((row_tiles == 8 && tile_offset == 0)
          || (row_tiles == 16 && (tile_offset == 0 || tile_offset == 8))))
        return fail(-1, "invalid HT verifier tile geometry");
    size_t values = 0;
    const int shape = decode_extent(bands, cols, values);
    if (shape != 0) return shape;
    if (!bytes || !codebook || !su || !sv || !out)
        return fail(-1, "HIP HT decode requires nonnull buffers");
    const size_t payload_bytes = size_t(bands) * (size_t(cols) / 16) * size_t(row_tiles) * size_t(32 * bits);
    if (address_overflows(bytes, payload_bytes)
        || address_overflows(codebook, kStates * sizeof(float))
        || address_overflows(su, size_t(bands) * size_t(cols) * sizeof(float))
        || address_overflows(sv, size_t(bands) * 128 * sizeof(float))
        || address_overflows(out, values * sizeof(float)))
        return fail(-1, "HIP HT decode buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return decode(state, bytes, nullptr, codebook, su, sv, out, bands, cols, bits, row_tiles, tile_offset);
    });
}
