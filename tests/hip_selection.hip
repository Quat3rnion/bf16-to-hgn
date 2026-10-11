// Standalone GPU selection regression; Cargo does not compile/run this file.
// From the repository root (requires gfx1151 and libstdc++16):
// /opt/rocm/bin/hipcc -std=c++17 -O3 --offload-arch=gfx1151 \
//   -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero \
//   -pthread tests/hip_selection.hip -o /tmp/bf16-hgn-hip-selection
// /tmp/bf16-hgn-hip-selection
// Includes actual device helpers; adds no production ABI or test hooks.
#include "../native/ht_trellis_hip.hip"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

namespace selector_probe {
constexpr int kProbeBatch = 32;
constexpr int kCapacity = 8192;
constexpr int kGuard = 8;
constexpr int kStride = kCapacity + 2 * kGuard;

enum class Pattern {
    Random,
    EqualCost,
    Sorted,
    Reversed,
    OrganPipe,
    MedianThreeKiller,
    AlternatingExtremes
};

const char *pattern_name(Pattern pattern) {
    switch (pattern) {
    case Pattern::Random: return "random";
    case Pattern::EqualCost: return "equal-cost-signed-zero";
    case Pattern::Sorted: return "sorted";
    case Pattern::Reversed: return "reversed";
    case Pattern::OrganPipe: return "organ-pipe";
    case Pattern::MedianThreeKiller: return "median-three-killer";
    case Pattern::AlternatingExtremes: return "alternating-extremes";
    }
    return "invalid";
}

struct Invocation {
    int size;
    int nth;
    int first;
    int last;
    int heap_only;
};
struct TestCase {
    Invocation invocation;
    Pattern pattern;
    uint64_t seed;
};
struct ProbeStorage {
    Path values[kProbeBatch * kStride];
    uint16_t wrong_left[kProbeBatch * kCapacity];
    uint16_t wrong_right[kProbeBatch * kCapacity];
    Invocation invocation[kProbeBatch];
};
static_assert(sizeof(ProbeStorage) < 4u * 1024u * 1024u);

__global__ __launch_bounds__(kThreads) void run_helpers(ProbeStorage *storage) {
    __shared__ unsigned sums[kThreads / 32];
    const int slot = int(blockIdx.x);
    const Invocation invocation = storage->invocation[slot];
    Path *values = storage->values + slot * kStride + kGuard;
    if (invocation.heap_only) {
        if (threadIdx.x == 0)
            heap_finish(values, invocation.first, invocation.nth, invocation.last);
    } else {
        select_prefix(values, invocation.nth, invocation.size,
                      storage->wrong_left + slot * kCapacity,
                      storage->wrong_right + slot * kCapacity, sums);
    }
}

float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool host_order(const Path &a, const Path &b) {
    // Independent numeric comparator, not the production device key helper.
    return a.cost < b.cost || (a.cost == b.cost && a.state < b.state);
}

void make_input(Path *destination, const TestCase &test, std::vector<Path> &ordered) {
    const int size = test.invocation.size;
    std::mt19937_64 random(test.seed);
    // Repeated nonnegative finite costs cover both signed zeros, subnormals,
    // normal magnitudes and the largest finite binary32 value. Histories
    // remain unique, so the comparator's complete keys are always unique.
    constexpr uint32_t costs[] = {
        0x00000000U, 0x80000000U, 0x00000001U, 0x00000002U,
        0x000116c2U, 0x00800000U, 0x3f000000U, 0x3f800000U,
        0x40000000U, 0x4b000000U, 0x7f7fffffU
    };
    for (int state = 0; state < size; ++state) {
        const uint32_t cost = test.pattern == Pattern::EqualCost
            ? (state & 1 ? 0x80000000U : 0U)
            : costs[random() % (sizeof(costs) / sizeof(costs[0]))];
        ordered[state] = {uint16_t(random()), uint16_t(state), float_from_bits(cost)};
    }
    if (test.pattern == Pattern::Random || test.pattern == Pattern::EqualCost) {
        std::shuffle(ordered.begin(), ordered.begin() + size, random);
        std::copy_n(ordered.begin(), size, destination);
        return;
    }
    std::sort(ordered.begin(), ordered.begin() + size, host_order);
    int out = 0;
    const auto emit = [&](int rank) { destination[out++] = ordered[rank]; };
    switch (test.pattern) {
    case Pattern::Sorted:
        for (int rank = 0; rank < size; ++rank) emit(rank);
        break;
    case Pattern::Reversed:
        for (int rank = size - 1; rank >= 0; --rank) emit(rank);
        break;
    case Pattern::MedianThreeKiller:
        if (size % 4 == 0) {
            // Zero-based version of interleaved low/high odd ranks followed
            // by even ranks: 0,n/2,2,n/2+2,...,1,3,5,... .
            for (int rank = 0; rank < size / 2; rank += 2) {
                emit(rank);
                emit(rank + size / 2);
            }
            for (int rank = 1; rank < size; rank += 2) emit(rank);
            break;
        }
        [[fallthrough]]; // Nonmultiples of four use a distinct organ pipe.
    case Pattern::OrganPipe:
        for (int rank = 0; rank < size; rank += 2) emit(rank);
        for (int rank = size - (size % 2 == 0 ? 1 : 2); rank >= 1; rank -= 2) emit(rank);
        break;
    case Pattern::AlternatingExtremes:
        for (int low = 0, high = size - 1; low <= high; ++low, --high) {
            emit(low);
            if (low != high) emit(high);
        }
        break;
    default:
        std::fprintf(stderr, "invalid generation pattern\n");
        std::exit(2);
    }
    if (out != size) {
        std::fprintf(stderr, "generation length mismatch: %d != %d\n", out, size);
        std::exit(2);
    }
}

std::vector<TestCase> make_cases() {
    constexpr int sizes[] = {
        2, 3, 4, 5, 8, 17, 33, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 1024, 2048, 4096, 8192
    };
    constexpr int nths[] = {
        1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256
    };
    constexpr int heap_sizes[] = {
        1, 2, 3, 4, 7, 8, 9, 16, 17, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257
    };
    constexpr Pattern patterns[] = {
        Pattern::Random, Pattern::EqualCost, Pattern::Sorted, Pattern::Reversed,
        Pattern::OrganPipe, Pattern::MedianThreeKiller, Pattern::AlternatingExtremes
    };
    std::vector<TestCase> tests;
    uint64_t seed = 0x48545045524d5554ULL;
    for (int size : sizes) {
        for (Pattern pattern : patterns) {
            for (int nth : nths) {
                if (nth >= size) continue;
                tests.push_back({{size, nth, 0, size, 0}, pattern, seed++});
            }
            for (int first : {0, 3}) {
                const int last = size >= 8 ? size - 2 : size;
                for (int heap_size : heap_sizes) {
                    const int nth = first + heap_size - 1;
                    if (nth < 1 || nth > 256 || nth >= last) continue;
                    tests.push_back({{size, nth, first, last, 1}, pattern, seed++});
                }
            }
        }
    }
    // Exercise EVERY supported nth, not just powers-of-two boundaries.
    // Four different sizes/seeds each stress the actual introselect helper.
    constexpr int random_sizes[] = {257, 513, 2048, 8192};
    for (int nth = 1; nth <= 256; ++nth)
        for (int size : random_sizes)
            tests.push_back({{size, nth, 0, size, 0}, Pattern::Random, seed++});
    return tests;
}

void check(hipError_t status, const char *operation) {
    if (status == hipSuccess) return;
    std::fprintf(stderr, "%s: %s (%s, %d)\n", operation,
                  hipGetErrorString(status), hipGetErrorName(status), int(status));
    std::exit(2);
}

void print_path(const char *label, const Path &path) {
    std::fprintf(stderr, "%s parent=%u state=%u cost=0x%08x\n", label,
                  unsigned(path.parent), unsigned(path.state), unsigned(float_bits(path.cost)));
}

int run() {
    check(hipSetDevice(0), "select device 0");
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0), "query device");
    if (std::strncmp(properties.gcnArchName, "gfx1151", 7) != 0 ||
        (properties.gcnArchName[7] != '\0' && properties.gcnArchName[7] != ':')) {
        std::fprintf(stderr, "expected gfx1151, found %s\n", properties.gcnArchName);
        return 2;
    }
    if (check_host_rounding() != 0) {
        std::fprintf(stderr, "%s\n", ht_hip_last_error());
        return 2;
    }
    int least = 0;
    int greatest = 0;
    check(hipDeviceGetStreamPriorityRange(&least, &greatest), "query stream priorities");
    hipStream_t stream = nullptr;
    check(hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, least), "create low-priority stream");
    ProbeStorage *device = nullptr;
    check(hipMalloc(reinterpret_cast<void **>(&device), sizeof(ProbeStorage)), "allocate bounded probe storage");

    const std::vector<TestCase> tests = make_cases();
    std::vector<Path> actual(kProbeBatch * kStride);
    std::vector<Path> expected(actual.size());
    std::vector<Path> ordered(kCapacity);
    std::array<Invocation, kProbeBatch> invocation;
    size_t full_cases = 0;
    size_t heap_cases = 0;
    int result = 0;
    for (size_t base = 0; base < tests.size(); base += kProbeBatch) {
        const size_t batch = std::min(size_t(kProbeBatch), tests.size() - base);
        const size_t path_count = batch * kStride;
        for (size_t position = 0; position < path_count; ++position)
            actual[position] = {
                uint16_t(0xa500U ^ unsigned(position)), uint16_t(0xf000U ^ unsigned(position)),
                float_from_bits(0x4f012345U)
            };
        for (size_t slot = 0; slot < batch; ++slot) {
            invocation[slot] = tests[base + slot].invocation;
            make_input(actual.data() + slot * kStride + kGuard, tests[base + slot], ordered);
        }
        std::copy_n(actual.data(), path_count, expected.data());
        for (size_t slot = 0; slot < batch; ++slot) {
            const Invocation call = invocation[slot];
            Path *values = expected.data() + slot * kStride + kGuard;
            if (call.heap_only) {
                // Deliberately use the installed reference implementation
                // for the rare fallback, not a second handwritten oracle.
                std::__heap_select(values + call.first, values + call.nth + 1,
                                   values + call.last, host_order);
                std::iter_swap(values + call.first, values + call.nth);
                ++heap_cases;
            } else {
                std::nth_element(values, values + call.nth, values + call.size, host_order);
                ++full_cases;
            }
        }
        check(hipMemcpyAsync(device->values, actual.data(), path_count * sizeof(Path),
                             hipMemcpyHostToDevice, stream), "upload guarded paths");
        check(hipMemcpyAsync(device->invocation, invocation.data(), batch * sizeof(Invocation),
                             hipMemcpyHostToDevice, stream), "upload invocations");
        hipLaunchKernelGGL(run_helpers, dim3(unsigned(batch)), dim3(kThreads), 0, stream, device);
        check(hipGetLastError(), "launch actual selection helpers");
        check(hipMemcpyAsync(actual.data(), device->values, path_count * sizeof(Path),
                             hipMemcpyDeviceToHost, stream), "download guarded paths");
        check(hipStreamSynchronize(stream), "complete selection batch");
        for (size_t slot = 0; slot < batch; ++slot) {
            const Path *got = actual.data() + slot * kStride;
            const Path *want = expected.data() + slot * kStride;
            if (std::memcmp(got, want, kStride * sizeof(Path)) == 0) continue;
            int position = 0;
            while (position < kStride && std::memcmp(got + position, want + position, sizeof(Path)) == 0)
                ++position;
            const TestCase &test = tests[base + slot];
            const Invocation call = test.invocation;
            std::fprintf(stderr,
                          "MISMATCH case=%zu mode=%s size=%d nth=%d first=%d last=%d "
                          "pattern=%s seed=0x%016llx position=%d (outside [0,size) means guard corruption)\n",
                          base + slot, call.heap_only ? "heap-finish" : "introselect", call.size,
                          call.nth, call.first, call.last, pattern_name(test.pattern),
                          static_cast<unsigned long long>(test.seed), position - kGuard);
            print_path("device", got[position]);
            print_path("libstdc++16", want[position]);
            result = 1;
            break;
        }
        if (result != 0) break;
    }
    check(hipFree(device), "free probe storage");
    check(hipStreamDestroy(stream), "destroy probe stream");
    if (result == 0)
        std::printf("PASS full-permutation=%zu forced-heap-finish=%zu total=%zu "
                    "device-bytes=%zu max-size=%d nth=1..256 guards=all-unused-slots "
                    "device=%s arch=%s priority=%d\n",
                    full_cases, heap_cases, tests.size(), sizeof(ProbeStorage), kCapacity,
                    properties.name, properties.gcnArchName, least);
    return result;
}
} // namespace selector_probe

int main() {
    return selector_probe::run();
}
