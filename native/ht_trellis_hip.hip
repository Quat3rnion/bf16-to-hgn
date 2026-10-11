// HIP beam encoder for Halogen's scalar 16-bit / three- or four-bit trellis.
// The CPU implementation remains the reference; this file never calls it.
//
// Selection below is an independent implementation of the observable
// libstdc++ 16 nth_element permutation, not merely its selected set. Changing
// the CPU standard library/selection algorithm requires a new equivalence
// qualification. In particular, libc++ and parallel-mode libstdc++ are not
// interchangeable. Build both encoders without fast math or FP contraction;
// HIP additionally needs -fno-gpu-flush-denormals-to-zero.
#include <hip/hip_runtime.h>

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
constexpr int kMaxHistories = kStates >> 3;
constexpr int kThreads = 256;
constexpr int kBatch = 64;
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
    Path expanded[kMaxHistories];
    Path candidates[kMaxHistories];
    uint16_t wrong_left[kMaxHistories];
    uint16_t wrong_right[kMaxHistories];
    Trace trace[kLength * kMaxBeam];
};
struct DeviceStorage {
    float codebook[kStates];
    float tiles[kBatch * kLength];
    uint8_t codes[kBatch * kLength];
    TileStatus status[kBatch];
    TileWorkspace workspace[kBatch];
};
static_assert(sizeof(Path) == 8 && sizeof(Trace) == 4);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
// One allocation, independent of the caller's band size, bits and beam:
// 64 * (2*8192*8 + 2*8192*2 + 256*256*4) + 65536*4
//     + 64*256*(4+1) + 64*12 = 27,607,808 bytes (< 27 MiB).
static_assert(sizeof(DeviceStorage) == 27607808);
static_assert(sizeof(DeviceStorage) <= 256u * 1024u * 1024u);

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
    TileWorkspace &workspace = storage->workspace[tile];
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
    std::array<float, kStates> cached_codebook;
    // Remains alive while an error path drains a queued D2H transfer.
    std::array<TileStatus, kBatch> tile_status;
    bool codebook_valid = false;

    int initialize() {
        if (storage) return 0;
        hipDeviceProp_t properties{};
        hipError_t error = hipGetDeviceProperties(&properties, 0);
        if (error != hipSuccess) return hip_failure(error, "query HIP device 0");
        const char *arch = properties.gcnArchName;
        if (std::strncmp(arch, "gfx1151", 7) != 0 || (arch[7] != '\0' && arch[7] != ':')) {
            std::snprintf(last_error, sizeof(last_error),
                          "HIP HT encoding supports only gfx1151; visible device 0 is %s (%s)",
                          properties.name, arch);
            return -2;
        }
        int least_priority = 0;
        int greatest_priority = 0;
        error = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        if (error != hipSuccess) return hip_failure(error, "query HIP stream priorities");
        if (!stream) {
            error = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least_priority);
            if (error != hipSuccess) return hip_failure(error, "create low-priority HIP stream");
        }
        error = hipMalloc(reinterpret_cast<void **>(&storage), sizeof(DeviceStorage));
        if (error != hipSuccess) {
            storage = nullptr;
            return hip_failure(error, "allocate bounded 27,607,808-byte HIP workspace");
        }
        return 0;
    }

    ~Context() {
        // Process-lifetime resources only; never touch another device's
        // allocations or reset a device shared with an inference server.
        if (!stream && !storage) return;
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
        if (!state.storage) {
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

int encode(Context &state, const float *tiles, const float *codebook,
           uint8_t *codes, int count, int beam, int bits) {
    const size_t codebook_bytes = sizeof(state.cached_codebook);
    if (!state.codebook_valid || std::memcmp(codebook, state.cached_codebook.data(), codebook_bytes) != 0) {
        // Cache contents, not the caller's address: an in-place changed
        // codebook must take effect on the very next call.
        std::memcpy(state.cached_codebook.data(), codebook, codebook_bytes);
        const hipError_t error = hipMemcpyAsync(state.storage->codebook, state.cached_codebook.data(),
                                               codebook_bytes, hipMemcpyHostToDevice, state.stream);
        if (error != hipSuccess) return hip_failure(error, "upload HIP codebook");
        state.codebook_valid = true;
    }
    auto &status = state.tile_status;
    const size_t total = size_t(count);
    void *arguments[] = {&state.storage, &beam};
    const void *kernel = bits == 3
        ? reinterpret_cast<const void *>(encode_tiles<3>)
        : reinterpret_cast<const void *>(encode_tiles<4>);
    for (size_t base = 0; base < total; base += kBatch) {
        const int batch = int(total - base < size_t(kBatch) ? total - base : size_t(kBatch));
        const size_t elements = size_t(batch) * kLength;
        hipError_t error = hipMemcpyAsync(state.storage->tiles, tiles + base * kLength,
                                         elements * sizeof(float), hipMemcpyHostToDevice, state.stream);
        if (error != hipSuccess) return hip_failure(error, "upload HIP tile batch");
        // HIP 7 retains earlier errors across successful calls. Check this
        // launch's return directly, without consuming a caller's last error.
        error = hipLaunchKernel(kernel, dim3(batch), dim3(kThreads), arguments, 0, state.stream);
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
} // namespace

// Error text is per calling thread, valid until its next encode/preflight.
// Failures may leave a partial output buffer; callers must discard it.
extern "C" const char *ht_hip_last_error() {
    return last_error;
}

extern "C" int ht_hip_preflight() {
    last_error[0] = '\0';
    // Initializes device, low-priority stream and the fixed workspace only;
    // no encode or large kernel is run by preflight.
    return with_context([](Context &) { return 0; });
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
