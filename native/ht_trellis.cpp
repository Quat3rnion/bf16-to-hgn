// CPU Viterbi for Halogen's scalar 16-bit / three- or four-bit-shift trellis.
// Build: c++ -O3 -ffp-contract=off -fopenmp -shared -fPIC ht_trellis.cpp -o ht_trellis.so
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// The build-only override keeps the production API unchanged while allowing
// equivalence checks to exercise the portable scalar fallback on AVX2 hosts.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(HT_TRELLIS_FORCE_SCALAR)
#define HT_TRELLIS_RUNTIME_AVX2 1
#include <immintrin.h>
#endif

namespace {
constexpr int states = 65536, length = 256;
template<int bits> struct Workspace {
    static constexpr int histories = states >> bits;
    std::vector<float> a, b, minima;
    std::vector<uint8_t> trace;
    explicit Workspace(bool enabled = true)
        : a(enabled ? states : 0), b(a.size()), minima(enabled ? histories : 0),
          trace(enabled ? length * histories : 0) {}
};

// Fixed boundary holds the final 16-bits history constant. This includes a
// partial fifth symbol for HT3: 13 history bits, not five complete symbols.
template<int bits>
float search(const float *x, const float *cb, int n, int boundary,
             uint8_t *codes, Workspace<bits> &w) {
    constexpr int symbols = 1 << bits, histories = states >> bits;
    std::fill(w.a.begin(), w.a.end(), boundary < 0 ? 0.f : INFINITY);
    if (boundary >= 0)
        for (int d = 0; d < symbols; ++d) w.a[d * histories + boundary] = 0.f;
    for (int t = 0; t < n; ++t) {
        float *prev = w.a.data(), *next = w.b.data();
        uint8_t *back = w.trace.data() + t * histories;
        for (int h = 0; h < histories; ++h) {
            float best = prev[h];
            uint8_t arg = 0;
            for (int d = 1; d < symbols; ++d) {
                float candidate = prev[d * histories + h];
                if (candidate < best) { best = candidate; arg = d; }
            }
            w.minima[h] = best;
            back[h] = arg;
        }
        for (int s = 0; s < states; ++s) {
            float error = x[t] - cb[s];
            next[s] = w.minima[s >> bits] + error * error;
        }
        w.a.swap(w.b);
    }
    int end = 0;
    if (boundary < 0) {
        end = std::min_element(w.a.begin(), w.a.end()) - w.a.begin();
    } else {
        end = boundary;
        for (int d = 1; d < symbols; ++d) {
            int s = d * histories + boundary;
            if (w.a[s] < w.a[end]) end = s;
        }
    }
    float cost = w.a[end];
    for (int t = n - 1; t >= 0; --t) {
        codes[t] = end & (symbols - 1);
        end = (end >> bits) | (int(w.trace[t * histories + (end >> bits)]) << (16 - bits));
    }
    return cost;
}

// Keep paths eight bytes on the fast comparator's supported layout: the
// little-endian high 48 bits then contain cost followed by the state tie-break.
struct Path { uint16_t parent, state; float cost; };
template<int bits> struct BeamWorkspace {
    static constexpr int histories = states >> bits;
    std::vector<Path> active, candidates;
    std::vector<float> best;
    std::vector<uint16_t> parent, trace, codes;
    explicit BeamWorkspace(int width)
        : active(width ? histories : 0), candidates(active.size()), best(active.size()),
          parent(active.size()), trace(length * width), codes(trace.size()) {}
};

template<int bits>
using ExpandGroups = void (*)(float, const float *, int, BeamWorkspace<bits> &, uint64_t *);

#ifdef HT_TRELLIS_RUNTIME_AVX2
template<int bits>
__attribute__((target("avx2")))
void expand_groups_avx2(float value, const float *cb, int count,
                       BeamWorkspace<bits> &w, uint64_t *touched) {
    constexpr int symbols = 1 << bits, histories = states >> bits;
    const __m256 values = _mm256_set1_ps(value);
    const __m256 infinity = _mm256_set1_ps(INFINITY);
    // Batch a whole step so target dispatch does not add a call per parent.
    // Parent visitation and sibling lanes keep the scalar strict-tie order.
    for (int j = 0; j < count; ++j) {
        const Path p = w.active[j];
        const int group = int(p.state) & (histories / symbols - 1);
        const int offset = group * symbols;
        const uint64_t mark = uint64_t(1) << (group & 63);
        uint64_t &word = touched[group >> 6];
        const bool initialize = !(word & mark);
        word |= mark;
        const __m256 previous_cost = _mm256_set1_ps(p.cost);
        const __m128i parent = _mm_set1_epi16(j);
        for (int d = 0; d < symbols; d += 8) {
            const int h = offset + d;
            const __m256 previous = initialize ? infinity : _mm256_loadu_ps(w.best.data() + h);
            const __m256 delta = _mm256_sub_ps(values, _mm256_loadu_ps(cb + (int(p.state) << bits) + d));
            // Keep separate float rounding, matching the scalar build's
            // -ffp-contract=off; explicit FMA would change encoded bytes.
            const __m256 squared = _mm256_mul_ps(delta, delta);
            const __m256 cost = _mm256_add_ps(previous_cost, squared);
            const __m256 better = _mm256_cmp_ps(cost, previous, _CMP_LT_OQ);
            // Store every lane: an untouched non-winning lane must become
            // +Inf, not retain a finite cost from an earlier step.
            _mm256_storeu_ps(w.best.data() + h, _mm256_blendv_ps(previous, cost, better));
            const __m256i mask = _mm256_castps_si256(better);
            const __m128i parent_mask = _mm_packs_epi32(_mm256_castsi256_si128(mask),
                                                     _mm256_extracti128_si256(mask, 1));
            const __m128i previous_parents = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(w.parent.data() + h));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(w.parent.data() + h),
                            _mm_blendv_epi8(previous_parents, parent, parent_mask));
        }
    }
}
#endif

template<int bits>
int beam_search(const float *x, const float *cb, int boundary, int width,
                 uint8_t *codes, BeamWorkspace<bits> &w, ExpandGroups<bits> expand) {
    constexpr int symbols = 1 << bits, histories = states >> bits;
    constexpr int closing_symbols = (16 - bits + bits - 1) / bits;
    constexpr int group_words = (histories / symbols + 63) / 64;
    int count = boundary < 0 ? histories : 1;
    for (int h = 0; h < count; ++h)
        w.active[h] = {0, uint16_t(boundary < 0 ? h : boundary), 0.f};
    for (int t = 0; t < length; ++t) {
        uint64_t touched[group_words] = {};
        int first = 0, last = symbols, step = 1;
        if (boundary >= 0 && t >= length - closing_symbols) {
            const int shift = bits * (length - 1 - t);
            const int mask = std::min(symbols - 1, (histories - 1) >> shift);
            first = (boundary >> shift) & mask;
            step = mask + 1;
        }
        if (step == 1 && expand) {
            expand(x[t], cb, count, w, touched);
        } else {
            for (int j = 0; j < count; ++j) {
                const Path p = w.active[j];
                const int group = int(p.state) & (histories / symbols - 1);
                const int offset = group * symbols;
                const uint64_t mark = uint64_t(1) << (group & 63);
                uint64_t &word = touched[group >> 6];
                if (!(word & mark)) {
                    std::fill_n(w.best.data() + offset, symbols, INFINITY);
                    word |= mark;
                }
                for (int d = first; d < last; d += step) {
                    int state = (int(p.state) << bits) | d;
                    float delta = x[t] - cb[state];
                    float cost = p.cost + delta * delta;
                    int h = state & (histories - 1);
                    if (cost < w.best[h]) {
                        w.best[h] = cost;
                        w.parent[h] = j;
                    }
                }
            }
        }
        int available = 0;
        // Ascending history order is part of the nth_element/tie contract.
        for (int word = 0; word < group_words; ++word) {
            uint64_t remaining = touched[word];
            while (remaining) {
                const int group = word * 64 + __builtin_ctzll(remaining);
                remaining &= remaining - 1;
                for (int h = group * symbols; h < (group + 1) * symbols; ++h)
                    if (std::isfinite(w.best[h]))
                        w.candidates[available++] = {w.parent[h], uint16_t(h), w.best[h]};
            }
        }
        count = std::min(width, available);
        auto order = [](const Path &a, const Path &b) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            if constexpr (sizeof(Path) == sizeof(uint64_t) &&
                          offsetof(Path, parent) == 0 && offsetof(Path, state) == 2 &&
                          offsetof(Path, cost) == 4 && sizeof(float) == 4 &&
                          std::numeric_limits<float>::is_iec559 &&
                          std::numeric_limits<float>::radix == 2 &&
                          std::numeric_limits<float>::digits == 24) {
                // Candidates have finite nonnegative squared-error costs.
                // Their binary32 magnitude bits have the same numeric order;
                // clearing the sign also keeps +0 and -0 equal. Parent bits
                // do not participate. memcpy avoids aliasing/alignment UB.
                uint64_t a_key, b_key;
                std::memcpy(&a_key, &a, sizeof(a_key));
                std::memcpy(&b_key, &b, sizeof(b_key));
                constexpr uint64_t key_mask = (uint64_t(1) << 47) - 1;
                return ((a_key >> 16) & key_mask) < ((b_key >> 16) & key_mask);
            }
#endif
            return a.cost < b.cost || (a.cost == b.cost && a.state < b.state);
        };
        if (count < available)
            std::nth_element(w.candidates.begin(), w.candidates.begin() + count,
                             w.candidates.begin() + available, order);
        w.active.swap(w.candidates);
        if (codes)
            for (int j = 0; j < count; ++j) {
                w.trace[t * width + j] = w.active[j].parent;
                w.codes[t * width + j] = w.active[j].state & (symbols - 1);
            }
    }
    int end = 0;
    for (int j = 1; j < count; ++j)
        if (w.active[j].cost < w.active[end].cost) end = j;
    const int final_history = w.active[end].state;
    if (codes)
        for (int t = length - 1; t >= 0; --t) {
            codes[t] = w.codes[t * width + end];
            end = w.trace[t * width + end];
        }
    return final_history;
}

template<int bits>
int encode(const float *tiles, const float *codebook,
           uint8_t *codes, int count, int threads, int beam) {
    constexpr int histories = states >> bits;
    if (count < 0 || threads < 1 || beam < 0 || beam > histories) return -1;
    ExpandGroups<bits> expand = nullptr;
#ifdef HT_TRELLIS_RUNTIME_AVX2
    if (beam && __builtin_cpu_supports("avx2"))
        expand = expand_groups_avx2<bits>;
#endif
    #pragma omp parallel num_threads(threads)
    {
        Workspace<bits> w(beam == 0);
        BeamWorkspace<bits> bw(beam);
        #pragma omp for schedule(static)
        for (int i = 0; i < count; ++i) {
            const float *x = tiles + i * length;
            uint8_t *q = codes + i * length;
            if (beam) {
                // The first pass needs only its final history, not a traceback.
                const int boundary = beam_search(x, codebook, -1, beam, nullptr, bw, expand);
                beam_search(x, codebook, boundary, beam, q, bw, expand);
            } else {
                search(x, codebook, length, -1, q, w);
                int boundary = 0;
                for (int t = length - (16 - bits + bits - 1) / bits; t < length; ++t)
                    boundary = ((boundary << bits) | q[t]) & (histories - 1);
                search(x, codebook, length, boundary, q, w);
            }
        }
    }
    return 0;
}
} // namespace

extern "C" int ht_encode_bits(const float *tiles, const float *codebook,
                               uint8_t *codes, int count, int threads, int beam, int bits) {
    if (bits == 4) return encode<4>(tiles, codebook, codes, count, threads, beam);
    if (bits == 3) return encode<3>(tiles, codebook, codes, count, threads, beam);
    return -1;
}

extern "C" float ht_search_fixed_bits(const float *x, const float *codebook,
                                      int n, int boundary, uint8_t *codes, int bits) {
    if ((bits != 3 && bits != 4) || n < 1 || n > length ||
        boundary < 0 || boundary >= (states >> bits))
        return std::numeric_limits<float>::quiet_NaN();
    if (bits == 4) {
        Workspace<4> w;
        return search(x, codebook, n, boundary, codes, w);
    }
    Workspace<3> w;
    return search(x, codebook, n, boundary, codes, w);
}
