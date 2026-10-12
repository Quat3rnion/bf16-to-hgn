// CUDA trellis encoder and independent verification decode/unrotation.
// The CPU implementations remain the references; this file never calls them.
//
// Selection below is an independent implementation of the observable
// libstdc++ nth_element permutation, not merely its selected set. libstdc++
// 12 and 16 are the build candidates; tests/cuda_selection.cu qualifies the
// installed version against complete device permutations. Changing either
// selection algorithm requires renewed byte-equivalence qualification.
// libc++ and parallel-mode libstdc++ are not interchangeable. Both encoders
// require separate IEEE operations: --fmad=false --ftz=false for CUDA, and
// -ffp-contract=off without fast math for the host compiler.
#include <cuda_runtime.h>

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

#if !defined(__CUDACC__)
#error "The HT CUDA encoder requires nvcc"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#error "The HT CUDA encoder requires compute capability 6.0 or newer"
#endif
#if !defined(_GLIBCXX_RELEASE) || (_GLIBCXX_RELEASE != 12 && _GLIBCXX_RELEASE != 16) || defined(_GLIBCXX_PARALLEL)
#error "The HT CUDA encoder requires ordinary libstdc++ 12 or 16 selection semantics"
#endif
#if defined(__FAST_MATH__) || defined(__USE_FAST_MATH__)
#error "The HT CUDA encoder cannot preserve CPU bytes with fast math enabled"
#endif
#if defined(__CUDA_FTZ) && __CUDA_FTZ
#error "The HT CUDA encoder requires --ftz=false"
#endif
#if defined(__CUDA_FMAD) && __CUDA_FMAD
#error "The HT CUDA encoder requires --fmad=false"
#endif

namespace {
constexpr int kStates = 65536;
constexpr int kLength = 256;
constexpr int kMaxBeam = 256;
constexpr int kThreads = 256;
// Several blocks per SM on a 76-SM RTX 4090 Laptop, without unbounded VRAM.
// Workspace capacity, not the input length, determines each request's batch.
constexpr int kBatch = 512;
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
    uint16_t *wrong_left;
    uint16_t *wrong_right;
    Trace *trace;
};
__host__ __device__ constexpr size_t tile_workspace_bytes(int bits, int beam) {
    return size_t(kStates >> bits) * (2 * sizeof(Path) + 2 * sizeof(uint16_t))
        + size_t(kLength) * size_t(beam) * sizeof(Trace);
}
__device__ __forceinline__ TileWorkspace tile_workspace(uint8_t *bytes, int histories) {
    Path *expanded = reinterpret_cast<Path *>(bytes);
    Path *candidates = expanded + histories;
    uint16_t *wrong_left = reinterpret_cast<uint16_t *>(candidates + histories);
    uint16_t *wrong_right = wrong_left + histories;
    Trace *trace = reinterpret_cast<Trace *>(wrong_right + histories);
    return {expanded, candidates, wrong_left, wrong_right, trace};
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
// One persistent allocation: 128 MiB arena plus 923,648 B for codebook/I/O.
// Per-tile layout uses 20*histories + 1024*beam bytes, all naturally aligned.
// HT3 beam256: 315 tiles; HT3 beam128: 455; HT4 beam256: 390; HT4 beam128: 512.
// Changing bits/beam only reinterprets this arena after the preceding sync.
static_assert(sizeof(DeviceStorage) == 135141376);
static_assert(kWorkspaceBytes / tile_workspace_bytes(3, kMaxBeam) >= 256);
static_assert(sizeof(DeviceStorage) <= 130u * 1024u * 1024u);

// Decode uses two FP64 axis passes, with only 16 independent 128-point
// transforms in shared memory at once. The complete 128x128 block would require 128 KiB.
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
static_assert(sizeof(DecodeStorage) == 55115776);
static_assert(sizeof(DeviceStorage) + sizeof(DecodeStorage) <= 192u * 1024u * 1024u);

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
    // Padding keeps the transpose from mapping all 16 columns to one shared memory bank.
    __shared__ double values[16 * 129];
    const int row_base = int(blockIdx.y) * 128;
    const int col_base = int(blockIdx.x) * 16;
    // Coalesced global accesses, transposed shared memory: the independent transforms
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

template<int Bits> struct BlockState {
    static constexpr int histories = kStates >> Bits;
    static constexpr int groups = histories >> Bits;
    Path active[kMaxBeam];
    uint16_t position[histories];
    uint16_t ordered_groups[groups];
    unsigned touched[(groups + 31) / 32];
    unsigned warp_sums[kThreads / 32];
    int count;
    int boundary;
};
static_assert(sizeof(BlockState<3>) <= 24u * 1024u);
static_assert(sizeof(BlockState<4>) <= 24u * 1024u);

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
        const unsigned earlier = __shfl_up_sync(0xffffffffU, inclusive, distance, warpSize);
        if (lane >= distance) inclusive += earlier;
    }
    if (lane == warpSize - 1) sums[wave] = inclusive;
    __syncthreads();
    if (wave == 0) {
        unsigned wave_sum = lane < waves ? sums[lane] : 0;
        for (int distance = 1; distance < warpSize; distance <<= 1) {
            const unsigned earlier = __shfl_up_sync(0xffffffffU, wave_sum, distance, warpSize);
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

// Candidates have unique histories, hence unique (cost, history) keys.
// With the median moved to first, there is no pivot-equal key in the scanned
// range. Its Hoare cut is first+1+count(key<pivot). The serial scanner pairs
// precisely the misplaced positions below that cut in increasing order
// with misplaced positions above it in decreasing order. Those pairs are
// disjoint: collecting and swapping them in parallel preserves *every*
// serial swap and the entire output permutation, not just the top-k set.
__device__ void select_prefix(Path *a, int nth, int size,
                              uint16_t *wrong_left, uint16_t *wrong_right,
                              unsigned *sums) {
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
        --depth;
        if (thread == 0) {
            // Find the median of these three *distinct* keys by sorting
            // their indices only; only the eventual pivot swap is applied.
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

template<int Bits>
__global__ __launch_bounds__(kThreads) void encode_tiles(DeviceStorage *storage, int width) {
    constexpr int symbols = 1 << Bits;
    constexpr int histories = kStates >> Bits;
    constexpr int groups = histories >> Bits;
    constexpr int closing_symbols = (16 - Bits + Bits - 1) / Bits;
    constexpr int group_words = (groups + 31) / 32;
    __shared__ BlockState<Bits> shared;
    const int thread = int(threadIdx.x);
    const int tile = int(blockIdx.x);
    const float *x = storage->tiles + tile * kLength;
    const float *cb = storage->codebook;
    const TileWorkspace workspace = tile_workspace(
        storage->workspace + size_t(tile) * tile_workspace_bytes(Bits, width), histories);
    for (int h = thread; h < histories; h += kThreads) shared.position[h] = kMissing;
    if (thread == 0) {
        storage->status[tile] = {0, 0, 0};
        shared.boundary = 0;
    }
    __syncthreads();
    for (int pass = 0; pass < 2; ++pass) {
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
                shared.position[state] = uint16_t(thread);
                const int group = state & (groups - 1);
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
            const int begin = expanded_count * thread / kThreads;
            const int end = expanded_count * (thread + 1) / kThreads;
            const float value = x[t];
            unsigned finite = 0;
            for (int position = begin; position < end; ++position) {
                const int group = shared.ordered_groups[position >> Bits];
                const int symbol = position & (symbols - 1);
                const int history = (group << Bits) | symbol;
                Path best = {kMissing, uint16_t(history), __int_as_float(0x7f800000)};
                // HT3's first closing symbol constrains only its low bit;
                // its other two bits remain free. Subsequent symbols are
                // fully fixed, just as d += mask+1 in the CPU loop.
                if ((symbol & closure_mask) == closure_value) {
                    for (int high = 0; high < symbols; ++high) {
                        const int previous = group | (high << (16 - 2 * Bits));
                        const int parent = initial ? previous : shared.position[previous];
                        if (parent == kMissing) continue;
                        const float prior = initial ? 0.f : shared.active[parent].cost;
                        const int state = (previous << Bits) | symbol;
                        const float delta = __fsub_rn(value, cb[state]);
                        const float squared = __fmul_rn(delta, delta);
                        const float cost = __fadd_rn(prior, squared);
                        // The scan is by state, NOT by active-array position.
                        // Resolve equal costs explicitly to the lower parent
                        // position to reproduce the CPU's strict update.
                        if (cost < best.cost || (cost == best.cost && parent < best.parent)) {
                            best.cost = cost;
                            best.parent = uint16_t(parent);
                        }
                    }
                }
                workspace.expanded[position] = best;
                finite += finite_cost(best.cost);
            }
            const Prefix candidate_ranks = prefix_sum(finite, shared.warp_sums);
            unsigned candidate_slot = candidate_ranks.offset;
            for (int position = begin; position < end; ++position) {
                const Path candidate = workspace.expanded[position];
                if (finite_cost(candidate.cost)) workspace.candidates[candidate_slot++] = candidate;
            }
            // Remove the old position map before overwriting active. Only
            // touched entries need clearing, not all H entries every step.
            if (!initial && thread < shared.count)
                shared.position[shared.active[thread].state] = kMissing;
            __syncthreads();
            const int available = int(candidate_ranks.total);
            if (available == 0) {
                if (thread == 0) storage->status[tile] = {1, pass, t};
                return;
            }
            const int count = width < available ? width : available;
            if (count < available)
                select_prefix(workspace.candidates, count, available,
                              workspace.wrong_left, workspace.wrong_right, shared.warp_sums);
            if (thread < count) {
                const Path selected = workspace.candidates[thread];
                shared.active[thread] = selected;
                if (pass == 1)
                    workspace.trace[t * width + thread] = {
                        selected.parent, uint16_t(selected.state & (symbols - 1))
                    };
            }
            if (thread == 0) shared.count = count;
            __syncthreads();
        }
        if (thread == 0) {
            int end = 0;
            for (int position = 1; position < shared.count; ++position)
                if (shared.active[position].cost < shared.active[end].cost) end = position;
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

thread_local char last_error[768] = {};

int fail(int code, const char *message) {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
    return code;
}

int cuda_failure(cudaError_t error, const char *operation) {
    std::snprintf(last_error, sizeof(last_error), "%s: %s (%s, %d)", operation,
                  cudaGetErrorString(error), cudaGetErrorName(error), int(error));
    return -3;
}

void append_cuda_failure(cudaError_t error, const char *operation) {
    const size_t used = std::strlen(last_error);
    if (used < sizeof(last_error) - 1)
        std::snprintf(last_error + used, sizeof(last_error) - used, "; %s: %s (%d)",
                      operation, cudaGetErrorString(error), int(error));
}

struct Context {
    std::mutex mutex;
    cudaStream_t stream = nullptr;
    DeviceStorage *storage = nullptr;
    DecodeStorage *decode_storage = nullptr;
    std::array<float, kStates> cached_codebook;
    std::array<float, kStates> decode_cached_codebook;
    // Remains alive while an error path drains a queued D2H transfer.
    std::array<TileStatus, kBatch> tile_status;
    bool codebook_valid = false;
    bool decode_codebook_valid = false;

    int initialize() {
        if (stream) return 0;
        cudaDeviceProp properties{};
        cudaError_t error = cudaGetDeviceProperties(&properties, 0);
        if (error != cudaSuccess) return cuda_failure(error, "query CUDA device 0");
        if (properties.major < 6 || properties.maxThreadsPerBlock < kThreads) {
            std::snprintf(last_error, sizeof(last_error),
                          "CUDA encoding/verification requires compute capability >= 6.0 and %d-thread blocks; "
                          "visible device 0 is %s (sm_%d%d)",
                          kThreads, properties.name, properties.major, properties.minor);
            return -2;
        }
        // Resolve every compiled kernel, including under CUDA_MODULE_LOADING=LAZY.
        // A device query alone cannot detect a missing SASS/PTX image. Attribute
        // queries load/JIT code when needed but never launch inference or kernels.
        const void *kernels[] = {
            reinterpret_cast<const void *>(encode_tiles<3>),
            reinterpret_cast<const void *>(encode_tiles<4>),
            reinterpret_cast<const void *>(decode_row_axis<0>),
            reinterpret_cast<const void *>(decode_row_axis<3>),
            reinterpret_cast<const void *>(decode_row_axis<4>),
            reinterpret_cast<const void *>(decode_column_axis)
        };
        for (const void *kernel : kernels) {
            cudaFuncAttributes attributes{};
            error = cudaFuncGetAttributes(&attributes, kernel);
            if (error != cudaSuccess) {
                cuda_failure(error, "resolve compiled CUDA kernel on device 0; check HGN_CUDA_ARCH/driver");
                return error == cudaErrorNoKernelImageForDevice || error == cudaErrorInvalidDeviceFunction
                    || error == cudaErrorUnsupportedPtxVersion ? -2 : -3;
            }
            if (attributes.maxThreadsPerBlock < kThreads
                || attributes.sharedSizeBytes > properties.sharedMemPerBlock)
                return fail(-2, "compiled CUDA kernel exceeds device 0's thread/shared-memory limits");
        }
        int least_priority = 0;
        int greatest_priority = 0;
        error = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (error != cudaSuccess) return cuda_failure(error, "query CUDA stream priorities");
        error = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least_priority);
        if (error != cudaSuccess) return cuda_failure(error, "create low-priority CUDA stream");
        return 0;
    }

    int ensure_encoder() {
        if (storage) return 0;
        const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&storage), sizeof(DeviceStorage));
        if (error != cudaSuccess) {
            storage = nullptr;
            return cuda_failure(error, "allocate bounded 135,141,376-byte CUDA workspace");
        }
        return 0;
    }

    int ensure_decoder() {
        if (decode_storage) return 0;
        const cudaError_t error = cudaMalloc(reinterpret_cast<void **>(&decode_storage), sizeof(DecodeStorage));
        if (error != cudaSuccess) {
            decode_storage = nullptr;
            return cuda_failure(error, "allocate bounded CUDA verification workspace");
        }
        return 0;
    }

    ~Context() {
        // The CUDA runtime can already be unloading at static destruction.
        // Best-effort silent cleanup; never reset a device shared by services.
        if (!stream && !storage && !decode_storage) return;
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != 0 && cudaSetDevice(0) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        if (storage) cudaFree(storage);
        if (decode_storage) cudaFree(decode_storage);
        if (stream) cudaStreamDestroy(stream);
        if (previous != 0) cudaSetDevice(previous);
    }
};

Context &context() {
    static Context value;
    return value;
}

int check_host_rounding() {
    if (std::fegetround() != FE_TONEAREST)
        return fail(-2, "CUDA byte equivalence requires CPU round-to-nearest floating-point mode");
#if defined(__x86_64__) || defined(__i386__)
    // AVX/SSE honor MXCSR rather than just the C floating-point environment.
    constexpr unsigned rounding_or_flush = (3u << 13) | (1u << 15) | (1u << 6);
    if ((_mm_getcsr() & rounding_or_flush) != 0)
        return fail(-2, "CUDA byte equivalence requires CPU MXCSR round-to-nearest with FTZ/DAZ disabled");
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
            const cudaError_t error = cudaGetDeviceCount(&devices);
            if (error == cudaErrorNoDevice || (error == cudaSuccess && devices == 0))
                return fail(-2, "no visible CUDA device; check CUDA_VISIBLE_DEVICES");
            if (error != cudaSuccess) {
                cuda_failure(error, "enumerate CUDA devices");
                return error == cudaErrorInsufficientDriver ? -2 : -3;
            }
        }
        int previous = 0;
        cudaError_t error = cudaGetDevice(&previous);
        if (error != cudaSuccess) return cuda_failure(error, "read caller's CUDA device");
        if (previous != 0) {
            error = cudaSetDevice(0);
            if (error != cudaSuccess) return cuda_failure(error, "select CUDA device 0");
        }
        int result = state.initialize();
        if (result == 0) result = operation(state);
        if (result != 0 && state.stream) {
            // Do not return while an enqueued transfer could still access a
            // caller-owned buffer. Never trust a partially uploaded cache.
            state.codebook_valid = false;
            state.decode_codebook_valid = false;
            error = cudaStreamSynchronize(state.stream);
            if (error != cudaSuccess) append_cuda_failure(error, "drain failed CUDA operation");
        }
        if (previous != 0) {
            error = cudaSetDevice(previous);
            if (error != cudaSuccess) {
                if (result == 0) result = cuda_failure(error, "restore caller's CUDA device");
                else append_cuda_failure(error, "restore caller's CUDA device");
            }
        }
        return result;
    } catch (const std::exception &error) {
        std::snprintf(last_error, sizeof(last_error), "CUDA HT host exception: %s", error.what());
        return -5;
    } catch (...) {
        return fail(-5, "unknown CUDA HT host exception");
    }
}

bool address_overflows(const void *pointer, size_t bytes) {
    return reinterpret_cast<uintptr_t>(pointer) > std::numeric_limits<uintptr_t>::max() - bytes;
}

int encode(Context &state, const float *tiles, const float *codebook,
           uint8_t *codes, int count, int beam, int bits) {
    const int allocation = state.ensure_encoder();
    if (allocation != 0) return allocation;
    const size_t codebook_bytes = sizeof(state.cached_codebook);
    if (!state.codebook_valid || std::memcmp(codebook, state.cached_codebook.data(), codebook_bytes) != 0) {
        // Cache contents, not the caller's address: an in-place changed
        // codebook must take effect on the very next call.
        std::memcpy(state.cached_codebook.data(), codebook, codebook_bytes);
        const cudaError_t error = cudaMemcpyAsync(state.storage->codebook, state.cached_codebook.data(),
                                               codebook_bytes, cudaMemcpyHostToDevice, state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "upload CUDA codebook");
        state.codebook_valid = true;
    }
    auto &status = state.tile_status;
    const size_t total = size_t(count);
    void *arguments[] = {&state.storage, &beam};
    const void *kernel = bits == 3
        ? reinterpret_cast<const void *>(encode_tiles<3>)
        : reinterpret_cast<const void *>(encode_tiles<4>);
    const size_t capacity = kWorkspaceBytes / tile_workspace_bytes(bits, beam);
    const size_t batch_limit = capacity < size_t(kBatch) ? capacity : size_t(kBatch);
    for (size_t base = 0; base < total; base += batch_limit) {
        const int batch = int(total - base < batch_limit ? total - base : batch_limit);
        const size_t elements = size_t(batch) * kLength;
        cudaError_t error = cudaMemcpyAsync(state.storage->tiles, tiles + base * kLength,
                                         elements * sizeof(float), cudaMemcpyHostToDevice, state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "upload CUDA tile batch");
        // Check this launch's direct status: an earlier recoverable runtime
        // error must not become a false launch failure via cudaGetLastError.
        // Fatal asynchronous device errors still fail; never reset the GPU.
        error = cudaLaunchKernel(kernel, dim3(batch), dim3(kThreads), arguments, 0, state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "launch CUDA beam-search batch");
        error = cudaMemcpyAsync(status.data(), state.storage->status, size_t(batch) * sizeof(TileStatus),
                               cudaMemcpyDeviceToHost, state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "download CUDA tile status");
        error = cudaMemcpyAsync(codes + base * kLength, state.storage->codes, elements,
                               cudaMemcpyDeviceToHost, state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "download CUDA symbols");
        error = cudaStreamSynchronize(state.stream);
        if (error != cudaSuccess) return cuda_failure(error, "complete CUDA beam-search batch");
        for (int tile = 0; tile < batch; ++tile) {
            if (status[tile].code != 0) {
                std::snprintf(last_error, sizeof(last_error),
                              "CUDA HT%d tile %zu has no finite survivor at pass %d, step %d; "
                              "check nonfinite inputs/codebook or squared-error overflow",
                              bits, base + size_t(tile), status[tile].pass + 1, status[tile].step);
                return -4;
            }
        }
    }
    return 0;
}

cudaError_t copy_decode_rectangle(void *destination, size_t destination_pitch,
                                const void *source, size_t source_pitch,
                                size_t width, size_t rows, cudaMemcpyKind kind,
                                cudaStream_t stream) {
    if (destination_pitch == width && source_pitch == width)
        return cudaMemcpyAsync(destination, source, width * rows, kind, stream);
    return cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch,
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
            const cudaError_t error = cudaMemcpyAsync(storage->codebook,
                state.decode_cached_codebook.data(), bytes_count, cudaMemcpyHostToDevice, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "upload CUDA verifier codebook");
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
            cudaError_t error = cudaMemcpyAsync(storage->su, input_su,
                width * (per_band_su ? batch_bands : 1) * sizeof(float),
                cudaMemcpyHostToDevice, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "upload CUDA verifier input rotation");
            error = cudaMemcpyAsync(storage->sv, sv + band_base * 128, rows * sizeof(float),
                                   cudaMemcpyHostToDevice, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "upload CUDA verifier output rotation");
            if (bits == 0) {
                error = copy_decode_rectangle(storage->payload, width / 2,
                    bytes + band_base * 128 * (full_cols / 2) + col_base / 2, full_cols / 2,
                    width / 2, rows, cudaMemcpyHostToDevice, state.stream);
                if (error != cudaSuccess) return cuda_failure(error, "upload CUDA I4R codes");
                // The ABI's half words are little-endian bytes; the host may
                // supply an unaligned byte plane. No host uint16_t dereference.
                const uint8_t *scale_bytes = reinterpret_cast<const uint8_t *>(scales);
                error = copy_decode_rectangle(storage->scales, width / 64,
                    scale_bytes + band_base * 128 * (full_cols / 64) + col_base / 64,
                    full_cols / 64, width / 64, rows, cudaMemcpyHostToDevice, state.stream);
                if (error != cudaSuccess) return cuda_failure(error, "upload CUDA I4R scales");
            } else {
                const size_t column_bytes = size_t(row_tiles) * size_t(32 * bits);
                const size_t offset = (band_base * (full_cols / 16) + col_base / 16) * column_bytes;
                error = cudaMemcpyAsync(storage->payload, bytes + offset,
                    batch_bands * (width / 16) * column_bytes, cudaMemcpyHostToDevice, state.stream);
                if (error != cudaSuccess) return cuda_failure(error, "upload CUDA HT verifier tiles");
            }
            void *row_arguments[] = {&storage, &batch_cols, &row_tiles, &tile_offset};
            error = cudaLaunchKernel(row_kernel, dim3(unsigned(width / 128), unsigned(rows / 16)),
                                    dim3(kThreads), row_arguments, 0, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "launch CUDA verifier row butterflies");
            void *column_arguments[] = {&storage, &batch_cols, &per_band_su};
            error = cudaLaunchKernel(reinterpret_cast<const void *>(decode_column_axis),
                dim3(unsigned(width / 16), unsigned(rows / 128)), dim3(kThreads),
                column_arguments, 0, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "launch CUDA verifier column butterflies");
            error = copy_decode_rectangle(output + band_base * 128 * full_cols + col_base,
                full_cols * sizeof(float), storage->output, width * sizeof(float),
                width * sizeof(float), rows, cudaMemcpyDeviceToHost, state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "download CUDA verifier values");
            error = cudaStreamSynchronize(state.stream);
            if (error != cudaSuccess) return cuda_failure(error, "complete CUDA verifier batch");
        }
        band_base += batch_bands;
    }
    return 0;
}

int decode_extent(int bands, int cols, size_t &values) {
    if (bands <= 0 || cols <= 0 || cols % 128 != 0)
        return fail(-1, "CUDA decode requires positive bands and a positive multiple-of-128 column count");
    if (size_t(bands) > std::numeric_limits<size_t>::max() / 128 / size_t(cols) / sizeof(float))
        return fail(-1, "CUDA decode shape overflows the host address space");
    values = size_t(bands) * 128 * size_t(cols);
    return 0;
}
} // namespace

// Error text is per calling thread, valid until its next encode/decode/preflight.
// Failures may leave a partial output buffer; callers must discard it.
extern "C" const char *ht_cuda_last_error() {
    return last_error;
}

extern "C" int ht_cuda_preflight() {
    last_error[0] = '\0';
    // Initializes device, low-priority stream and the fixed workspace only;
    // no encode or large kernel is run by preflight.
    return with_context([](Context &state) { return state.ensure_encoder(); });
}

extern "C" int ht_encode_bits_cuda(const float *tiles, const float *codebook,
                                   uint8_t *codes, int count, int beam, int bits) {
    last_error[0] = '\0';
    if (bits != 3 && bits != 4) return fail(-1, "CUDA HT bits must be 3 or 4");
    if (beam < 1 || beam > kMaxBeam)
        return fail(-1, "CUDA HT beam must be 1..256; exact beam 0 and larger beams are CPU-only");
    if (count < 0) return fail(-1, "CUDA HT tile count must be nonnegative");
    if (count == 0) return 0;
    if (!tiles || !codebook || !codes)
        return fail(-1, "CUDA HT nonempty input requires nonnull tiles, codebook and codes");
    if (size_t(count) > std::numeric_limits<size_t>::max() / kLength / sizeof(float))
        return fail(-1, "CUDA HT tile count overflows the host address space");
    const size_t elements = size_t(count) * kLength;
    if (address_overflows(tiles, elements * sizeof(float)) ||
        address_overflows(codes, elements) || address_overflows(codebook, kStates * sizeof(float)))
        return fail(-1, "CUDA HT buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return encode(state, tiles, codebook, codes, count, beam, bits);
    });
}

extern "C" int hgn_cuda_decode_preflight() {
    last_error[0] = '\0';
    // Allocate only the bounded decoder workspace; never the encoder's beam
    // storage, and never launch a kernel merely to discover availability.
    return with_context([](Context &state) { return state.ensure_decoder(); });
}

extern "C" int hgn_cuda_decode_i4r(const uint8_t *codes, const uint16_t *scales,
                                  const float *su, const float *sv, float *out,
                                  int bands, int cols) {
    last_error[0] = '\0';
    size_t values = 0;
    const int shape = decode_extent(bands, cols, values);
    if (shape != 0) return shape;
    if (!codes || !scales || !su || !sv || !out)
        return fail(-1, "CUDA I4R decode requires nonnull buffers");
    if (address_overflows(codes, values / 2) || address_overflows(scales, values / 64)
        || address_overflows(su, size_t(cols) * sizeof(float))
        || address_overflows(sv, size_t(bands) * 128 * sizeof(float))
        || address_overflows(out, values * sizeof(float)))
        return fail(-1, "CUDA I4R decode buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return decode(state, codes, scales, nullptr, su, sv, out, bands, cols, 0, 8, 0);
    });
}

extern "C" int hgn_cuda_decode_ht(const uint8_t *bytes, const float *codebook,
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
        return fail(-1, "CUDA HT decode requires nonnull buffers");
    const size_t payload_bytes = size_t(bands) * (size_t(cols) / 16) * size_t(row_tiles) * size_t(32 * bits);
    if (address_overflows(bytes, payload_bytes)
        || address_overflows(codebook, kStates * sizeof(float))
        || address_overflows(su, size_t(bands) * size_t(cols) * sizeof(float))
        || address_overflows(sv, size_t(bands) * 128 * sizeof(float))
        || address_overflows(out, values * sizeof(float)))
        return fail(-1, "CUDA HT decode buffer extent overflows the host address space");
    return with_context([&](Context &state) {
        return decode(state, bytes, nullptr, codebook, su, sv, out, bands, cols, bits, row_tiles, tile_offset);
    });
}
