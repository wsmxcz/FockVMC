#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

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

[[nodiscard]] inline bool equal(
    std::span<const u64> a,
    std::span<const u64> b
) noexcept {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

[[nodiscard]] inline bool test(std::span<const u64> x, int p) noexcept {
    return (
        x[static_cast<std::size_t>(p >> 6)]
        >> static_cast<unsigned>(p & 63)
    ) & u64{1};
}

inline void set(std::span<u64> x, int p) noexcept {
    x[static_cast<std::size_t>(p >> 6)] |= (
        u64{1} << static_cast<unsigned>(p & 63)
    );
}

inline void clear(std::span<u64> x, int p) noexcept {
    x[static_cast<std::size_t>(p >> 6)] &= ~(
        u64{1} << static_cast<unsigned>(p & 63)
    );
}

[[nodiscard]] inline int popcount(std::span<const u64> x) noexcept {
    int out = 0;
    for (u64 word : x) out += static_cast<int>(std::popcount(word));
    return out;
}

[[nodiscard]] inline int popcount_xor(
    std::span<const u64> a,
    std::span<const u64> b
) noexcept {
    int out = 0;

    for (std::size_t i = 0; i < a.size(); ++i) {
        out += static_cast<int>(std::popcount(a[i] ^ b[i]));
    }

    return out;
}

[[nodiscard]] inline int popcount_between_word(
    u64 occ,
    int i,
    int a
) noexcept {
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

[[nodiscard]] inline int popcount_between(
    std::span<const u64> occ,
    int i,
    int a
) noexcept {
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
        const u64 hi_mask = b1 == 63
            ? ~u64{0}
            : ((u64{1} << (b1 + 1u)) - 1u);
        return static_cast<int>(
            std::popcount(
                occ[static_cast<std::size_t>(w0)] & lo_mask & hi_mask
            )
        );
    }

    int out = static_cast<int>(
        std::popcount(
            occ[static_cast<std::size_t>(w0)] & (~u64{0} << b0)
        )
    );

    for (int w = w0 + 1; w < w1; ++w) {
        out += static_cast<int>(
            std::popcount(occ[static_cast<std::size_t>(w)])
        );
    }

    const u64 hi_mask = b1 == 63
        ? ~u64{0}
        : ((u64{1} << (b1 + 1u)) - 1u);
    out += static_cast<int>(
        std::popcount(
            occ[static_cast<std::size_t>(w1)] & hi_mask
        )
    );

    return out;
}

[[nodiscard]] inline bool parity_between(
    std::span<const u64> occ,
    int i,
    int a
) noexcept {
    return (popcount_between(occ, i, a) & 1) != 0;
}

inline void fill_prefix(
    std::span<const u64> occ,
    int norb,
    std::vector<int>& prefix
) {
    prefix.assign(static_cast<std::size_t>(norb + 1), 0);
    int count = 0;

    for (int p = 0; p < norb; ++p) {
        prefix[static_cast<std::size_t>(p)] = count;
        if (test(occ, p)) ++count;
    }

    prefix[static_cast<std::size_t>(norb)] = count;
}

[[nodiscard]] inline int count_between(
    std::span<const int> prefix,
    int i,
    int a
) noexcept {
    if (i == a) return 0;

    const int lo = std::min(i, a) + 1;
    const int hi = std::max(i, a);
    if (lo >= hi) return 0;

    return prefix[static_cast<std::size_t>(hi)]
        - prefix[static_cast<std::size_t>(lo)];
}

[[nodiscard]] inline bool parity_between(
    std::span<const int> prefix,
    int i,
    int a
) noexcept {
    return (count_between(prefix, i, a) & 1) != 0;
}

template <class F>
inline void each_set_word(u64 word, F&& visit) {
    while (word != 0u) {
        const unsigned bit = std::countr_zero(word);
        visit(static_cast<int>(bit));
        word &= word - 1u;
    }
}

template <class F>
inline void each_set(std::span<const u64> words, F&& visit) {
    if (words.size() == 1) {
        each_set_word(words[0], std::forward<F>(visit));
        return;
    }

    for (std::size_t w = 0; w < words.size(); ++w) {
        u64 word = words[w];

        while (word != 0u) {
            const unsigned bit = std::countr_zero(word);
            visit(static_cast<int>((w << 6) + bit));
            word &= word - 1u;
        }
    }
}

template <class F>
inline void each_clear(
    std::span<const u64> occ,
    int norb,
    F&& visit
) {
    if (occ.size() == 1) {
        const u64 valid = norb >= 64
            ? ~u64{0}
            : ((u64{1} << static_cast<unsigned>(norb)) - 1u);
        u64 word = (~occ[0]) & valid;

        while (word != 0u) {
            const unsigned bit = std::countr_zero(word);
            visit(static_cast<int>(bit));
            word &= word - 1u;
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
            const unsigned bit = std::countr_zero(word);
            visit(base + static_cast<int>(bit));
            word &= word - 1u;
        }
    }
}

inline void set_list(std::span<const u64> words, std::vector<int>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(popcount(words)));
    each_set(words, [&](int p) { out.push_back(p); });
}

[[nodiscard]] inline std::vector<int> set_list(
    std::span<const u64> words
) {
    std::vector<int> out;
    set_list(words, out);
    return out;
}

inline void clear_list(
    std::span<const u64> words,
    int norb,
    std::vector<int>& out
) {
    out.clear();
    out.reserve(static_cast<std::size_t>(norb - popcount(words)));
    each_clear(words, norb, [&](int p) { out.push_back(p); });
}

} // namespace bits

[[nodiscard]] inline constexpr std::size_t det_size(u32 nword) noexcept {
    return 2u * static_cast<std::size_t>(nword);
}

/*
 * A determinant is stored as two packed spin bit strings:
 *
 *   alpha words, then beta words.
 *
 * DetRef is a non-owning view. It does not manage lifetime.
 */
class DetRef {
public:
    constexpr DetRef() noexcept = default;

    constexpr DetRef(const u64* alpha, const u64* beta, u32 nword) noexcept
        : alpha_(alpha), beta_(beta), nword_(nword) {}

    [[nodiscard]] constexpr std::span<const u64> alpha() const noexcept {
        return {alpha_, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] constexpr std::span<const u64> beta() const noexcept {
        return {beta_, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] constexpr u32 nword() const noexcept { return nword_; }

private:
    const u64* alpha_ = nullptr;
    const u64* beta_ = nullptr;
    u32 nword_ = 0;
};

class DetScratch {
public:
    explicit DetScratch(u32 nword) : words_(det_size(nword), 0u), nword_(nword) {}

    void load(DetRef det) {
        std::copy(det.alpha().begin(), det.alpha().end(), words_.begin());
        std::copy(
            det.beta().begin(),
            det.beta().end(),
            words_.begin() + static_cast<std::ptrdiff_t>(nword_)
        );
    }

    [[nodiscard]] std::span<u64> alpha() noexcept {
        return {words_.data(), static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] std::span<u64> beta() noexcept {
        return {
            words_.data() + static_cast<std::size_t>(nword_),
            static_cast<std::size_t>(nword_)
        };
    }

    [[nodiscard]] DetRef view() const noexcept {
        return DetRef(
            words_.data(),
            words_.data() + static_cast<std::size_t>(nword_),
            nword_
        );
    }

private:
    std::vector<u64> words_;
    u32 nword_ = 0;
};

enum class ExcitationKind : unsigned char {
    alpha1,
    beta1,
    alpha2,
    beta2,
    mixed2,
};

struct Excitation {
    ExcitationKind kind = ExcitationKind::alpha1;
    int i = 0;
    int j = 0;
    int a = 0;
    int b = 0;
};

[[nodiscard]] inline bool excitation_less(
    Excitation lhs,
    Excitation rhs
) noexcept {
    if (lhs.kind != rhs.kind) {
        return static_cast<unsigned char>(lhs.kind)
            < static_cast<unsigned char>(rhs.kind);
    }
    if (lhs.i != rhs.i) return lhs.i < rhs.i;
    if (lhs.j != rhs.j) return lhs.j < rhs.j;
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    return lhs.b < rhs.b;
}

[[nodiscard]] inline Excitation alpha1(int i, int a) noexcept {
    return Excitation{ExcitationKind::alpha1, i, 0, a, 0};
}

[[nodiscard]] inline Excitation beta1(int i, int a) noexcept {
    return Excitation{ExcitationKind::beta1, i, 0, a, 0};
}

[[nodiscard]] inline Excitation alpha2(
    int i,
    int j,
    int a,
    int b
) noexcept {
    return Excitation{ExcitationKind::alpha2, i, j, a, b};
}

[[nodiscard]] inline Excitation beta2(
    int i,
    int j,
    int a,
    int b
) noexcept {
    return Excitation{ExcitationKind::beta2, i, j, a, b};
}

[[nodiscard]] inline Excitation mixed2(
    int i,
    int j,
    int a,
    int b
) noexcept {
    return Excitation{ExcitationKind::mixed2, i, j, a, b};
}

inline DetRef apply(
    DetRef ket,
    Excitation excitation,
    DetScratch& scratch
) {
    scratch.load(ket);
    auto alpha = scratch.alpha();
    auto beta = scratch.beta();

    switch (excitation.kind) {
    case ExcitationKind::alpha1:
        bits::clear(alpha, excitation.i);
        bits::set(alpha, excitation.a);
        break;
    case ExcitationKind::beta1:
        bits::clear(beta, excitation.i);
        bits::set(beta, excitation.a);
        break;
    case ExcitationKind::alpha2:
        bits::clear(alpha, excitation.i);
        bits::clear(alpha, excitation.j);
        bits::set(alpha, excitation.a);
        bits::set(alpha, excitation.b);
        break;
    case ExcitationKind::beta2:
        bits::clear(beta, excitation.i);
        bits::clear(beta, excitation.j);
        bits::set(beta, excitation.a);
        bits::set(beta, excitation.b);
        break;
    case ExcitationKind::mixed2:
        bits::clear(alpha, excitation.i);
        bits::clear(beta, excitation.j);
        bits::set(alpha, excitation.a);
        bits::set(beta, excitation.b);
        break;
    }

    return scratch.view();
}

struct DetBatchView {
    const u64* data = nullptr;
    std::size_t n_dets = 0;
    u32 nword = 0;

    [[nodiscard]] DetRef operator[](std::size_t idx) const noexcept {
        const std::size_t stride = det_size(nword);
        const u64* ptr = data + idx * stride;
        return DetRef(ptr, ptr + static_cast<std::size_t>(nword), nword);
    }
};

[[nodiscard]] inline DetRef packed_det(const u64* data, u32 nword) noexcept {
    return DetRef(data, data + static_cast<std::size_t>(nword), nword);
}

[[nodiscard]] inline DetRef det_at(
    const std::vector<u64>& data,
    u32 nword,
    std::size_t idx
) noexcept {
    const std::size_t stride = det_size(nword);
    const u64* ptr = data.data() + idx * stride;
    return DetRef(ptr, ptr + static_cast<std::size_t>(nword), nword);
}

inline void append_det(std::vector<u64>& out, DetRef det) {
    out.insert(out.end(), det.alpha().begin(), det.alpha().end());
    out.insert(out.end(), det.beta().begin(), det.beta().end());
}

inline void copy_batch(std::vector<u64>& words, DetBatchView dets) {
    words.resize(dets.n_dets * det_size(dets.nword));
    if (!words.empty()) std::copy_n(dets.data, words.size(), words.data());
}

[[nodiscard]] inline bool det_equal(DetRef a, DetRef b) noexcept {
    return a.nword() == b.nword()
        && bits::equal(a.alpha(), b.alpha())
        && bits::equal(a.beta(), b.beta());
}

struct DetLess {
    [[nodiscard]] bool operator()(DetRef lhs, DetRef rhs) const noexcept {
        const auto la = lhs.alpha();
        const auto ra = rhs.alpha();

        if (la.size() == 1 && ra.size() == 1) {
            if (la[0] != ra[0]) return la[0] < ra[0];
            return lhs.beta()[0] < rhs.beta()[0];
        }

        for (std::size_t w = 0; w < la.size(); ++w) {
            if (la[w] != ra[w]) return la[w] < ra[w];
        }

        const auto lb = lhs.beta();
        const auto rb = rhs.beta();

        for (std::size_t w = 0; w < lb.size(); ++w) {
            if (lb[w] != rb[w]) return lb[w] < rb[w];
        }

        return false;
    }
};

[[nodiscard]] inline u64 splitmix64(u64 x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

[[nodiscard]] inline u64 hash_words(u64 seed, std::span<const u64> words) noexcept {
    u64 h = splitmix64(seed ^ (0x9e3779b97f4a7c15ULL + static_cast<u64>(words.size())));

    for (u64 x : words) {
        h = splitmix64(h ^ splitmix64(x + 0x517cc1b727220a95ULL));
    }

    return h;
}

struct DetHash {
    [[nodiscard]] std::size_t operator()(DetRef det) const noexcept {
        if (det.nword() == 1) {
            const u64 h =
                det.alpha()[0]
                ^ splitmix64(det.beta()[0] + 0x517cc1b727220a95ULL);

            return static_cast<std::size_t>(splitmix64(h));
        }

        u64 h = 0x9e3779b97f4a7c15ULL ^ static_cast<u64>(det.nword());

        for (u64 x : det.alpha()) h = splitmix64(h ^ x);
        for (u64 x : det.beta()) h = splitmix64(h ^ (x + 0x517cc1b727220a95ULL));

        return static_cast<std::size_t>(h);
    }
};

[[nodiscard]] inline u64 det_fingerprint(DetRef det, u64 seed = 0) noexcept {
    u64 h = splitmix64(seed ^ 0x6465746b65747331ULL);

    if (det.nword() == 1) {
        h = splitmix64(h ^ det.alpha()[0]);
        return splitmix64(h ^ (det.beta()[0] + 0x517cc1b727220a95ULL));
    }

    for (u64 x : det.alpha()) h = splitmix64(h ^ x);
    for (u64 x : det.beta()) h = splitmix64(h ^ (x + 0x517cc1b727220a95ULL));

    return h;
}

/*
 * Sort packed determinants lexicographically and remove duplicates.
 *
 * This is intentionally simple and deterministic. High-throughput connection
 * construction should avoid feeding this routine one determinant per
 * connection; it is best used on already compact determinant pools.
 */
inline void sort_unique_dets(std::vector<u64>& packed, u32 nword) {
    const std::size_t stride = det_size(nword);
    if (stride == 0 || packed.empty()) return;

    const std::size_t n_dets = packed.size() / stride;

    std::vector<std::size_t> order(n_dets);
    for (std::size_t idet = 0; idet < n_dets; ++idet) order[idet] = idet;

    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return DetLess{}(det_at(packed, nword, lhs), det_at(packed, nword, rhs));
    });

    std::vector<u64> out;
    out.reserve(packed.size());

    std::size_t prev = static_cast<std::size_t>(-1);

    for (std::size_t idet : order) {
        if (
            prev != static_cast<std::size_t>(-1)
            && det_equal(det_at(packed, nword, idet), det_at(packed, nword, prev))
        ) {
            continue;
        }

        append_det(out, det_at(packed, nword, idet));
        prev = idet;
    }

    packed.swap(out);
}

namespace detail {

[[nodiscard]] inline double sign_single(std::span<const u64> occ, int i, int a) noexcept {
    return bits::parity_between(occ, i, a) ? -1.0 : 1.0;
}

[[nodiscard]] inline double sign_single(std::span<const int> prefix, int i, int a) noexcept {
    return bits::parity_between(prefix, i, a) ? -1.0 : 1.0;
}

/*
 * Fermionic sign for two same-spin replacements i,j -> a,b.
 *
 * The crossing correction accounts for the fact that two single-excitation
 * strings may interleave on the orbital line.
 */
[[nodiscard]] inline double sign_double(
    std::span<const u64> occ,
    int i,
    int j,
    int a,
    int b
) noexcept {
    int p = 0;

    p ^= bits::popcount_between(occ, i, a);
    p ^= bits::popcount_between(occ, j, b);

    const int x0 = std::min(i, a);
    const int x1 = std::max(i, a);
    const int y0 = std::min(j, b);
    const int y1 = std::max(j, b);

    const bool cross =
        (y0 > x0 && y0 < x1 && y1 > x1)
        || (x0 > y0 && x0 < y1 && x1 > y1);

    if (cross) ++p;

    return (p & 1) ? -1.0 : 1.0;
}

[[nodiscard]] inline double sign_double(
    std::span<const int> prefix,
    int i,
    int j,
    int a,
    int b
) noexcept {
    int p = 0;

    p ^= bits::count_between(prefix, i, a);
    p ^= bits::count_between(prefix, j, b);

    const int x0 = std::min(i, a);
    const int x1 = std::max(i, a);
    const int y0 = std::min(j, b);
    const int y1 = std::max(j, b);

    const bool cross =
        (y0 > x0 && y0 < x1 && y1 > x1)
        || (x0 > y0 && x0 < y1 && x1 > y1);

    if (cross) ++p;

    return (p & 1) ? -1.0 : 1.0;
}

} // namespace detail

struct DetDiff {
    int deg = 0;
    int na = 0;
    int nb = 0;

    std::array<int, 2> occ_a{0, 0};
    std::array<int, 2> vir_a{0, 0};
    std::array<int, 2> occ_b{0, 0};
    std::array<int, 2> vir_b{0, 0};

    double sign = 1.0;
};

struct SpinDiff {
    int deg = 0;
    std::array<int, 2> occ{0, 0};
    std::array<int, 2> vir{0, 0};
    double sign = 1.0;
};

struct DetOcc {
    explicit DetOcc(int norb = 0) {
        resize(norb);
    }

    void resize(int norb) {
        occ_a.clear();
        occ_b.clear();
        vir_a.clear();
        vir_b.clear();
        pref_a.assign(static_cast<std::size_t>(norb + 1), 0);
        pref_b.assign(static_cast<std::size_t>(norb + 1), 0);
    }

    std::vector<int> occ_a;
    std::vector<int> occ_b;
    std::vector<int> vir_a;
    std::vector<int> vir_b;

    std::vector<int> pref_a;
    std::vector<int> pref_b;
};

inline void fill_occ(DetRef det, int norb, DetOcc& work) {
    bits::set_list(det.alpha(), work.occ_a);
    bits::set_list(det.beta(), work.occ_b);

    bits::clear_list(det.alpha(), norb, work.vir_a);
    bits::clear_list(det.beta(), norb, work.vir_b);

    bits::fill_prefix(det.alpha(), norb, work.pref_a);
    bits::fill_prefix(det.beta(), norb, work.pref_b);
}

[[nodiscard]] inline DetDiff det_diff(DetRef bra, DetRef ket) noexcept {
    DetDiff ex;

    const int dx_a = bits::popcount_xor(bra.alpha(), ket.alpha());
    const int dx_b = bits::popcount_xor(bra.beta(), ket.beta());

    if ((dx_a & 1) || (dx_b & 1)) {
        ex.deg = 3;
        return ex;
    }

    ex.na = dx_a >> 1;
    ex.nb = dx_b >> 1;
    ex.deg = ex.na + ex.nb;

    if (ex.deg > 2) return ex;

    auto fill_one = [](
        std::span<const u64> bra_bits,
        std::span<const u64> ket_bits,
        std::array<int, 2>& occ,
        std::array<int, 2>& vir,
        int n
    ) {
        int io = 0;
        int iv = 0;

        for (std::size_t w = 0; w < bra_bits.size(); ++w) {
            u64 gone = bra_bits[w] & ~ket_bits[w];
            u64 come = ket_bits[w] & ~bra_bits[w];

            while (gone != 0u && io < n) {
                const unsigned b = std::countr_zero(gone);
                occ[io++] = static_cast<int>((w << 6) + b);
                gone &= (gone - 1u);
            }

            while (come != 0u && iv < n) {
                const unsigned b = std::countr_zero(come);
                vir[iv++] = static_cast<int>((w << 6) + b);
                come &= (come - 1u);
            }
        }
    };

    fill_one(bra.alpha(), ket.alpha(), ex.occ_a, ex.vir_a, ex.na);
    fill_one(bra.beta(), ket.beta(), ex.occ_b, ex.vir_b, ex.nb);

    if (ex.deg == 1) {
        ex.sign = ex.na == 1
            ? detail::sign_single(bra.alpha(), ex.occ_a[0], ex.vir_a[0])
            : detail::sign_single(bra.beta(), ex.occ_b[0], ex.vir_b[0]);
    } else if (ex.na == 2) {
        ex.sign = detail::sign_double(
            bra.alpha(),
            ex.occ_a[0],
            ex.occ_a[1],
            ex.vir_a[0],
            ex.vir_a[1]
        );
    } else if (ex.nb == 2) {
        ex.sign = detail::sign_double(
            bra.beta(),
            ex.occ_b[0],
            ex.occ_b[1],
            ex.vir_b[0],
            ex.vir_b[1]
        );
    } else if (ex.deg == 2) {
        ex.sign =
            detail::sign_single(bra.alpha(), ex.occ_a[0], ex.vir_a[0])
            * detail::sign_single(bra.beta(), ex.occ_b[0], ex.vir_b[0]);
    }

    return ex;
}

[[nodiscard]] inline SpinDiff spin_diff(
    std::span<const u64> bra_bits,
    std::span<const u64> ket_bits
) noexcept {
    SpinDiff ex;

    if (bra_bits.size() == 1) {
        const u64 gone0 = bra_bits[0] & ~ket_bits[0];
        const u64 come0 = ket_bits[0] & ~bra_bits[0];

        const int dx =
            static_cast<int>(std::popcount(gone0))
            + static_cast<int>(std::popcount(come0));

        if (dx & 1) {
            ex.deg = 3;
            return ex;
        }

        ex.deg = dx >> 1;
        if (ex.deg > 2) return ex;

        u64 gone = gone0;
        u64 come = come0;

        for (int k = 0; k < ex.deg; ++k) {
            ex.occ[k] = static_cast<int>(std::countr_zero(gone));
            gone &= (gone - 1u);

            ex.vir[k] = static_cast<int>(std::countr_zero(come));
            come &= (come - 1u);
        }

        if (ex.deg == 1) {
            ex.sign = detail::sign_single(bra_bits, ex.occ[0], ex.vir[0]);
        } else if (ex.deg == 2) {
            ex.sign = detail::sign_double(
                bra_bits,
                ex.occ[0],
                ex.occ[1],
                ex.vir[0],
                ex.vir[1]
            );
        }

        return ex;
    }

    const int dx = bits::popcount_xor(bra_bits, ket_bits);

    if (dx & 1) {
        ex.deg = 3;
        return ex;
    }

    ex.deg = dx >> 1;
    if (ex.deg > 2) return ex;

    int io = 0;
    int iv = 0;

    for (std::size_t w = 0; w < bra_bits.size(); ++w) {
        u64 gone = bra_bits[w] & ~ket_bits[w];
        u64 come = ket_bits[w] & ~bra_bits[w];

        while (gone != 0u && io < ex.deg) {
            const unsigned b = std::countr_zero(gone);
            ex.occ[io++] = static_cast<int>((w << 6) + b);
            gone &= (gone - 1u);
        }

        while (come != 0u && iv < ex.deg) {
            const unsigned b = std::countr_zero(come);
            ex.vir[iv++] = static_cast<int>((w << 6) + b);
            come &= (come - 1u);
        }
    }

    if (ex.deg == 1) {
        ex.sign = detail::sign_single(bra_bits, ex.occ[0], ex.vir[0]);
    } else if (ex.deg == 2) {
        ex.sign = detail::sign_double(
            bra_bits,
            ex.occ[0],
            ex.occ[1],
            ex.vir[0],
            ex.vir[1]
        );
    }

    return ex;
}

} // namespace libdet
