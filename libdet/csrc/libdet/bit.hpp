#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>
#include <utility>

namespace libdet {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

[[nodiscard]] inline i32 to_i32(std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<i32>::max())) {
        throw std::overflow_error("index overflow");
    }
    return static_cast<i32>(n);
}

namespace bits {

[[nodiscard]] inline constexpr u32 words_for(int norb) noexcept {
    return norb <= 0 ? 0u : static_cast<u32>((norb + 63) >> 6);
}

[[nodiscard]] inline bool equal(std::span<const u64> a, std::span<const u64> b) noexcept {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

[[nodiscard]] inline bool test(std::span<const u64> x, int p) noexcept {
    return (x[static_cast<std::size_t>(p >> 6)] >> static_cast<unsigned>(p & 63)) & u64{1};
}

inline void set(std::span<u64> x, int p) noexcept {
    x[static_cast<std::size_t>(p >> 6)] |= (u64{1} << static_cast<unsigned>(p & 63));
}

inline void clear(std::span<u64> x, int p) noexcept {
    x[static_cast<std::size_t>(p >> 6)] &= ~(u64{1} << static_cast<unsigned>(p & 63));
}

[[nodiscard]] inline int popcount(std::span<const u64> x) noexcept {
    int out = 0;
    for (u64 w : x) out += static_cast<int>(std::popcount(w));
    return out;
}

[[nodiscard]] inline int popcount_xor(std::span<const u64> a, std::span<const u64> b) noexcept {
    int out = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out += static_cast<int>(std::popcount(a[i] ^ b[i]));
    }
    return out;
}

/*
 * Count occupied orbitals strictly between i and a.
 *
 * This parity controls the fermionic phase of a single excitation.
 * The word-specialized path is important because most small molecules
 * fit in one 64-bit word.
 */
[[nodiscard]] inline int popcount_between_word(u64 occ, int i, int a) noexcept {
    if (i == a) return 0;

    const int lo = std::min(i, a) + 1;
    const int hi = std::max(i, a);
    if (lo >= hi) return 0;

    const u64 lo_mask = ~u64{0} << static_cast<unsigned>(lo);
    const u64 hi_mask = hi >= 64
        ? ~u64{0}
        : ((u64{1} << static_cast<unsigned>(hi)) - 1u);

    return static_cast<int>(std::popcount(occ & lo_mask & hi_mask));
}

[[nodiscard]] inline int popcount_between(std::span<const u64> occ, int i, int a) noexcept {
    if (occ.size() == 1) return popcount_between_word(occ[0], i, a);
    if (i == a) return 0;

    const int lo = std::min(i, a) + 1;
    const int hi = std::max(i, a);
    if (lo >= hi) return 0;

    const int w0 = lo >> 6;
    const int w1 = (hi - 1) >> 6;
    const unsigned b0 = static_cast<unsigned>(lo & 63);
    const unsigned b1 = static_cast<unsigned>((hi - 1) & 63);

    if (w0 == w1) {
        const u64 lo_mask = ~u64{0} << b0;
        const u64 hi_mask = b1 == 63 ? ~u64{0} : ((u64{1} << (b1 + 1u)) - 1u);
        return static_cast<int>(
            std::popcount(occ[static_cast<std::size_t>(w0)] & lo_mask & hi_mask)
        );
    }

    int out = 0;
    out += static_cast<int>(
        std::popcount(occ[static_cast<std::size_t>(w0)] & (~u64{0} << b0))
    );

    for (int w = w0 + 1; w < w1; ++w) {
        out += static_cast<int>(std::popcount(occ[static_cast<std::size_t>(w)]));
    }

    const u64 hi_mask = b1 == 63 ? ~u64{0} : ((u64{1} << (b1 + 1u)) - 1u);
    out += static_cast<int>(
        std::popcount(occ[static_cast<std::size_t>(w1)] & hi_mask)
    );

    return out;
}

[[nodiscard]] inline bool parity_between(std::span<const u64> occ, int i, int a) noexcept {
    return (popcount_between(occ, i, a) & 1) != 0;
}

/*
 * Prefix count convention:
 *
 *   prefix[p] = number of occupied orbitals q with q < p.
 *
 * Then the number of occupied orbitals strictly between i and a is
 *
 *   prefix[max(i,a)] - prefix[min(i,a)+1].
 *
 * Row-local prefix arrays make excitation signs O(1).
 */
inline void fill_prefix(std::span<const u64> occ, int norb, std::vector<int>& prefix) {
    prefix.assign(static_cast<std::size_t>(norb + 1), 0);
    int acc = 0;
    for (int p = 0; p < norb; ++p) {
        prefix[static_cast<std::size_t>(p)] = acc;
        if (test(occ, p)) ++acc;
    }
    prefix[static_cast<std::size_t>(norb)] = acc;
}

[[nodiscard]] inline int count_between(std::span<const int> prefix, int i, int a) noexcept {
    if (i == a) return 0;

    const int lo = std::min(i, a) + 1;
    const int hi = std::max(i, a);
    if (lo >= hi) return 0;

    return prefix[static_cast<std::size_t>(hi)] - prefix[static_cast<std::size_t>(lo)];
}

[[nodiscard]] inline bool parity_between(std::span<const int> prefix, int i, int a) noexcept {
    return (count_between(prefix, i, a) & 1) != 0;
}

template <class F>
inline void each_set_word(u64 word, F&& f) {
    while (word != 0u) {
        const unsigned b = std::countr_zero(word);
        f(static_cast<int>(b));
        word &= (word - 1u);
    }
}

template <class F>
inline void each_set(std::span<const u64> x, F&& f) {
    if (x.size() == 1) {
        each_set_word(x[0], std::forward<F>(f));
        return;
    }

    for (std::size_t w = 0; w < x.size(); ++w) {
        u64 word = x[w];
        while (word != 0u) {
            const unsigned b = std::countr_zero(word);
            f(static_cast<int>((w << 6) + b));
            word &= (word - 1u);
        }
    }
}

template <class F>
inline void each_clear(std::span<const u64> occ, int norb, F&& f) {
    if (occ.size() == 1) {
        const u64 valid = norb >= 64
            ? ~u64{0}
            : ((u64{1} << static_cast<unsigned>(norb)) - 1u);

        u64 word = (~occ[0]) & valid;
        while (word != 0u) {
            const unsigned b = std::countr_zero(word);
            f(static_cast<int>(b));
            word &= (word - 1u);
        }
        return;
    }

    for (std::size_t w = 0; w < occ.size(); ++w) {
        const int base = static_cast<int>(w << 6);
        const int rem = norb - base;
        if (rem <= 0) break;

        const u64 valid = rem >= 64
            ? ~u64{0}
            : ((u64{1} << static_cast<unsigned>(rem)) - 1u);

        u64 word = (~occ[w]) & valid;
        while (word != 0u) {
            const unsigned b = std::countr_zero(word);
            f(base + static_cast<int>(b));
            word &= (word - 1u);
        }
    }
}

inline void set_list(std::span<const u64> x, std::vector<int>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(popcount(x)));
    each_set(x, [&](int p) { out.push_back(p); });
}

[[nodiscard]] inline std::vector<int> set_list(std::span<const u64> x) {
    std::vector<int> out;
    set_list(x, out);
    return out;
}

inline void clear_list(std::span<const u64> x, int norb, std::vector<int>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(norb - popcount(x)));
    each_clear(x, norb, [&](int p) { out.push_back(p); });
}

} // namespace bits
} // namespace libdet