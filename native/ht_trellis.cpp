// CPU Viterbi for Halogen's scalar 16-bit / three- or four-bit-shift trellis.
// Build: c++ -O3 -march=native -fopenmp -shared -fPIC ht_trellis.cpp -o ht_trellis.so
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__FMA__)
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

struct Path { float cost; uint16_t state, parent; };
template<int bits> struct BeamWorkspace {
    static constexpr int histories = states >> bits;
    std::vector<Path> active, candidates;
    std::vector<float> best;
    std::vector<uint16_t> parent, trace, codes;
    explicit BeamWorkspace(int width)
        : active(width ? histories : 0), candidates(active.size()), best(active.size()),
          parent(active.size()), trace(length * width), codes(trace.size()) {}
};

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__FMA__)
template<int bits>
void expand_symbols(float value, const float *cb, Path path, int parent,
                    float *best, uint16_t *parents) {
    cb += int(path.state) << bits;
    // The native scalar build contracts cost + delta * delta to one FMA.
    // Only siblings are parallel: parent visitation and strict ties stay intact.
    if (bits == 4) {
        const __m512 delta = _mm512_sub_ps(_mm512_set1_ps(value), _mm512_loadu_ps(cb));
        const __m512 cost = _mm512_fmadd_ps(delta, delta, _mm512_set1_ps(path.cost));
        const __mmask16 better = _mm512_cmp_ps_mask(cost, _mm512_loadu_ps(best), _CMP_LT_OQ);
        _mm512_mask_storeu_ps(best, better, cost);
        _mm256_mask_storeu_epi16(parents, better, _mm256_set1_epi16(parent));
    } else {
        const __m256 delta = _mm256_sub_ps(_mm256_set1_ps(value), _mm256_loadu_ps(cb));
        const __m256 cost = _mm256_fmadd_ps(delta, delta, _mm256_set1_ps(path.cost));
        const __mmask8 better = _mm256_cmp_ps_mask(cost, _mm256_loadu_ps(best), _CMP_LT_OQ);
        _mm256_mask_storeu_ps(best, better, cost);
        _mm_mask_storeu_epi16(parents, better, _mm_set1_epi16(parent));
    }
}
#endif

template<int bits>
int beam_search(const float *x, const float *cb, int boundary, int width,
                 uint8_t *codes, BeamWorkspace<bits> &w) {
    constexpr int symbols = 1 << bits, histories = states >> bits;
    constexpr int closing_symbols = (16 - bits + bits - 1) / bits;
    constexpr int group_words = (histories / symbols + 63) / 64;
    int count = boundary < 0 ? histories : 1;
    for (int h = 0; h < count; ++h)
        w.active[h] = {0.f, uint16_t(boundary < 0 ? h : boundary), 0};
    for (int t = 0; t < length; ++t) {
        uint64_t touched[group_words] = {};
        int first = 0, last = symbols, step = 1;
        if (boundary >= 0 && t >= length - closing_symbols) {
            const int shift = bits * (length - 1 - t);
            const int mask = std::min(symbols - 1, (histories - 1) >> shift);
            first = (boundary >> shift) & mask;
            step = mask + 1;
        }
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
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__FMA__)
            if (step == 1) {
                expand_symbols<bits>(x[t], cb, p, j, w.best.data() + offset, w.parent.data() + offset);
                continue;
            }
#endif
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
        int available = 0;
        // Ascending history order is part of the nth_element/tie contract.
        for (int word = 0; word < group_words; ++word) {
            uint64_t remaining = touched[word];
            while (remaining) {
                const int group = word * 64 + __builtin_ctzll(remaining);
                remaining &= remaining - 1;
                for (int h = group * symbols; h < (group + 1) * symbols; ++h)
                    if (std::isfinite(w.best[h]))
                        w.candidates[available++] = {w.best[h], uint16_t(h), w.parent[h]};
            }
        }
        count = std::min(width, available);
        auto order = [](const Path &a, const Path &b) {
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
                const int boundary = beam_search(x, codebook, -1, beam, nullptr, bw);
                beam_search(x, codebook, boundary, beam, q, bw);
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
