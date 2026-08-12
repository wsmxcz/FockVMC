#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <libdet/bit.hpp>

namespace libdet {

[[nodiscard]] inline constexpr std::size_t det_size(u32 nword) noexcept {
    return 2u * static_cast<std::size_t>(nword);
}

// Non-owning view of packed alpha words followed by beta words.
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

    bool operator==(const Excitation&) const noexcept = default;
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

struct DetHash {
    [[nodiscard]] std::size_t operator()(DetRef det) const noexcept {
        if (det.nword() == 1) {
            const u64 h =
                det.alpha()[0]
                ^ mix64(det.beta()[0] + 0x517cc1b727220a95ULL);

            return static_cast<std::size_t>(mix64(h));
        }

        u64 h = 0x9e3779b97f4a7c15ULL ^ static_cast<u64>(det.nword());

        for (u64 x : det.alpha()) h = mix64(h ^ x);
        for (u64 x : det.beta()) h = mix64(h ^ (x + 0x517cc1b727220a95ULL));

        return static_cast<std::size_t>(h);
    }
};

[[nodiscard]] inline u64 det_fingerprint(DetRef det, u64 seed = 0) noexcept {
    u64 h = mix64(seed ^ 0x6465746b65747331ULL);

    if (det.nword() == 1) {
        h = mix64(h ^ det.alpha()[0]);
        return mix64(h ^ (det.beta()[0] + 0x517cc1b727220a95ULL));
    }

    for (u64 x : det.alpha()) h = mix64(h ^ x);
    for (u64 x : det.beta()) h = mix64(h ^ (x + 0x517cc1b727220a95ULL));

    return h;
}

// Deterministic lexicographic deduplication for compact pools.
inline void sort_unique(std::vector<u64>& packed, u32 nword) {
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

[[nodiscard]] inline double sign_single(std::span<const u64> occ, int i, int a) noexcept {
    return bits::parity_between(occ, i, a) ? -1.0 : 1.0;
}

[[nodiscard]] inline double sign_single(std::span<const int> prefix, int i, int a) noexcept {
    return bits::parity_between(prefix, i, a) ? -1.0 : 1.0;
}

// Same-spin double sign includes crossing excitations.
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

struct DetExcitation {
    int degree = 0;
    int na = 0;
    int nb = 0;
    Excitation excitation{};
    double sign = 1.0;
};

struct SpinDiff {
    int deg = 0;
    std::array<int, 2> occ{0, 0};
    std::array<int, 2> vir{0, 0};
    double sign = 1.0;
};

struct DetOcc {
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

[[nodiscard]] inline DetExcitation excitation(DetRef ket, DetRef bra) noexcept {
    DetExcitation ex;

    const int dx_a = bits::popcount_xor(ket.alpha(), bra.alpha());
    const int dx_b = bits::popcount_xor(ket.beta(), bra.beta());

    if ((dx_a & 1) || (dx_b & 1)) {
        ex.degree = 3;
        return ex;
    }

    ex.na = dx_a >> 1;
    ex.nb = dx_b >> 1;
    ex.degree = ex.na + ex.nb;

    if (ex.degree > 2) return ex;

    std::array<int, 2> rem_a{0, 0};
    std::array<int, 2> add_a{0, 0};
    std::array<int, 2> rem_b{0, 0};
    std::array<int, 2> add_b{0, 0};

    auto fill_one = [](
        std::span<const u64> ket_bits,
        std::span<const u64> bra_bits,
        std::array<int, 2>& rem,
        std::array<int, 2>& add,
        int n
    ) {
        int ir = 0;
        int ia = 0;

        for (std::size_t w = 0; w < ket_bits.size(); ++w) {
            u64 gone = ket_bits[w] & ~bra_bits[w];
            u64 come = bra_bits[w] & ~ket_bits[w];

            while (gone != 0u && ir < n) {
                const unsigned b = std::countr_zero(gone);
                rem[ir++] = static_cast<int>((w << 6) + b);
                gone &= (gone - 1u);
            }

            while (come != 0u && ia < n) {
                const unsigned b = std::countr_zero(come);
                add[ia++] = static_cast<int>((w << 6) + b);
                come &= (come - 1u);
            }
        }
    };

    fill_one(ket.alpha(), bra.alpha(), rem_a, add_a, ex.na);
    fill_one(ket.beta(), bra.beta(), rem_b, add_b, ex.nb);

    if (ex.degree == 1) {
        if (ex.na == 1) {
            ex.excitation = alpha1(rem_a[0], add_a[0]);
            ex.sign = sign_single(ket.alpha(), rem_a[0], add_a[0]);
        } else {
            ex.excitation = beta1(rem_b[0], add_b[0]);
            ex.sign = sign_single(ket.beta(), rem_b[0], add_b[0]);
        }
    } else if (ex.na == 2) {
        ex.excitation = alpha2(rem_a[0], rem_a[1], add_a[0], add_a[1]);
        ex.sign = sign_double(
            ket.alpha(),
            rem_a[0],
            rem_a[1],
            add_a[0],
            add_a[1]
        );
    } else if (ex.nb == 2) {
        ex.excitation = beta2(rem_b[0], rem_b[1], add_b[0], add_b[1]);
        ex.sign = sign_double(
            ket.beta(),
            rem_b[0],
            rem_b[1],
            add_b[0],
            add_b[1]
        );
    } else if (ex.degree == 2) {
        ex.excitation = mixed2(rem_a[0], rem_b[0], add_a[0], add_b[0]);
        ex.sign =
            sign_single(ket.alpha(), rem_a[0], add_a[0])
            * sign_single(ket.beta(), rem_b[0], add_b[0]);
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
            ex.sign = sign_single(bra_bits, ex.occ[0], ex.vir[0]);
        } else if (ex.deg == 2) {
            ex.sign = sign_double(
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
        ex.sign = sign_single(bra_bits, ex.occ[0], ex.vir[0]);
    } else if (ex.deg == 2) {
        ex.sign = sign_double(
            bra_bits,
            ex.occ[0],
            ex.occ[1],
            ex.vir[0],
            ex.vir[1]
        );
    }

    return ex;
}


class DetIndex {
public:
    explicit DetIndex(DetBatchView dets) : dets_(dets) {
        index_.reserve(dets.n_dets);
        for (std::size_t i = 0; i < dets.n_dets; ++i) {
            index_[det_fingerprint(dets_[i])].push_back(static_cast<i32>(i));
        }
    }

    [[nodiscard]] i32 find(DetRef det) const noexcept {
        const auto it = index_.find(det_fingerprint(det));
        if (it == index_.end()) return -1;

        for (i32 idx : it->second) {
            if (det_equal(dets_[static_cast<std::size_t>(idx)], det)) {
                return idx;
            }
        }

        return -1;
    }

private:
    DetBatchView dets_;
    ankerl::unordered_dense::map<u64, std::vector<i32>> index_;
};

class DetPool {
public:
    explicit DetPool(u32 nword = 0) : nword_(nword) {}

    explicit DetPool(DetBatchView dets) : nword_(dets.nword) {
        copy_batch(words_, dets);
        index_.reserve(size());
        for (std::size_t i = 0; i < size(); ++i) {
            index_[det_fingerprint(get(i))].push_back(static_cast<i32>(i));
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return nword_ == 0 ? 0u : words_.size() / det_size(nword_);
    }

    [[nodiscard]] DetRef get(std::size_t idx) const noexcept {
        return det_at(words_, nword_, idx);
    }

    [[nodiscard]] std::vector<u64>& words() noexcept {
        return words_;
    }

    [[nodiscard]] const std::vector<u64>& words() const noexcept {
        return words_;
    }

    [[nodiscard]] i32 find_add(DetRef det) {
        const u64 fingerprint = det_fingerprint(det);
        const auto it = index_.find(fingerprint);
        if (it != index_.end()) {
            for (i32 idx : it->second) {
                if (det_equal(get(static_cast<std::size_t>(idx)), det)) {
                    return idx;
                }
            }
        }

        const i32 fresh = to_i32(size());
        append_det(words_, det);
        index_[fingerprint].push_back(fresh);
        return fresh;
    }

private:
    u32 nword_ = 0;
    std::vector<u64> words_;
    ankerl::unordered_dense::map<u64, std::vector<i32>> index_;
};

// Determinant-driven search within a known ket space.

[[nodiscard]] inline u64 orb_fp(int p) noexcept {
    return mix64(0xd1b54a32d192ed03ULL ^ static_cast<u64>(p + 1));
}

[[nodiscard]] inline u64 spin_fp(std::span<const u64> words) noexcept {
    u64 h = 0;
    bits::each_set(words, [&](int p) { h ^= orb_fp(p); });
    return h;
}

class SpinSet {
public:
    explicit SpinSet(u32 nword = 0, u64 tag = 0)
        : nword_(nword), tag_(tag) {}

    [[nodiscard]] i32 find(std::span<const u64> words) const noexcept {
        if (nword_ == 1) {
            const auto it = one_.find(words[0]);
            return it == one_.end() ? -1 : it->second;
        }

        const auto it = many_.find(hash_words(tag_, words));
        if (it == many_.end()) return -1;

        for (i32 idx : it->second) {
            if (bits::equal(get(idx), words)) return idx;
        }

        return -1;
    }

    [[nodiscard]] i32 find_add(std::span<const u64> words) {
        const i32 old = find(words);
        if (old >= 0) return old;

        const i32 idx = to_i32(size());

        data_.insert(data_.end(), words.begin(), words.end());
        fp_.push_back(spin_fp(words));

        if (nword_ == 1) {
            one_.emplace(words[0], idx);
        } else {
            many_[hash_words(tag_, words)].push_back(idx);
        }

        return idx;
    }

    [[nodiscard]] std::span<const u64> get(i32 idx) const noexcept {
        const std::size_t off =
            static_cast<std::size_t>(idx) * static_cast<std::size_t>(nword_);

        return {data_.data() + off, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] u64 fp(i32 idx) const noexcept {
        return fp_[static_cast<std::size_t>(idx)];
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return nword_ == 0 ? 0u : data_.size() / static_cast<std::size_t>(nword_);
    }

private:
    u32 nword_ = 0;
    u64 tag_ = 0;

    std::vector<u64> data_;
    std::vector<u64> fp_;

    ankerl::unordered_dense::map<u64, i32> one_;
    ankerl::unordered_dense::map<u64, std::vector<i32>> many_;
};

struct KeyId {
    u64 key = 0;
    i32 id = -1;
};

class Residues {
public:
    void build(std::vector<KeyId>&& items) {
        items_ = std::move(items);

        std::sort(items_.begin(), items_.end(), [](const KeyId& a, const KeyId& b) {
            if (a.key != b.key) return a.key < b.key;
            return a.id < b.id;
        });
    }

    [[nodiscard]] std::span<const KeyId> equal(u64 key) const noexcept {
        if (items_.empty()) return {};

        const auto lo = std::lower_bound(
            items_.begin(),
            items_.end(),
            key,
            [](const KeyId& a, u64 b) { return a.key < b; }
        );

        const auto hi = std::upper_bound(
            items_.begin(),
            items_.end(),
            key,
            [](u64 a, const KeyId& b) { return a < b.key; }
        );

        return {
            items_.data() + static_cast<std::size_t>(lo - items_.begin()),
            static_cast<std::size_t>(hi - lo)
        };
    }

private:
    std::vector<KeyId> items_;
};

struct SpinExcitation {
    i32 spin = -1;
    int i = 0;
    int j = 0;
    int a = 0;
    int b = 0;
    double sign = 1.0;
};

struct SpinMate {
    i32 spin = -1;
    i32 ket = -1;
};

class DetSpace {
public:
    explicit DetSpace(DetBatchView kets)
        : nword(kets.nword),
          alpha(nword, 0x0f1234ab5678cdefULL),
          beta(nword, 0x1a2b3c4d5e6f7081ULL) {
        copy_batch(ket_words, kets);

        alpha_id.resize(kets.n_dets);
        beta_id.resize(kets.n_dets);

        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            const DetRef ket = kets[iket];
            alpha_id[iket] = alpha.find_add(ket.alpha());
            beta_id[iket] = beta.find_add(ket.beta());
        }

        build_mates();
        sort_mates();
        build_residues();
    }

    [[nodiscard]] std::span<const SpinMate> alpha_mates(i32 alpha_spin) const noexcept {
        const std::size_t lo = alpha_off[static_cast<std::size_t>(alpha_spin)];
        const std::size_t hi = alpha_off[static_cast<std::size_t>(alpha_spin) + 1u];

        return {alpha_mate.data() + lo, hi - lo};
    }

    [[nodiscard]] std::span<const SpinMate> beta_mates(i32 beta_spin) const noexcept {
        const std::size_t lo = beta_off[static_cast<std::size_t>(beta_spin)];
        const std::size_t hi = beta_off[static_cast<std::size_t>(beta_spin) + 1u];

        return {beta_mate.data() + lo, hi - lo};
    }

    [[nodiscard]] i32 find_alpha(i32 alpha_spin, i32 beta_spin) const noexcept {
        const auto mates = alpha_mates(alpha_spin);

        const auto it = std::lower_bound(
            mates.begin(),
            mates.end(),
            beta_spin,
            [](const SpinMate& e, i32 x) { return e.spin < x; }
        );

        return it == mates.end() || it->spin != beta_spin ? -1 : it->ket;
    }

    [[nodiscard]] i32 find_beta(i32 beta_spin, i32 alpha_spin) const noexcept {
        const auto mates = beta_mates(beta_spin);

        const auto it = std::lower_bound(
            mates.begin(),
            mates.end(),
            alpha_spin,
            [](const SpinMate& e, i32 x) { return e.spin < x; }
        );

        return it == mates.end() || it->spin != alpha_spin ? -1 : it->ket;
    }

    u32 nword = 0;

    std::vector<u64> ket_words;

    SpinSet alpha;
    SpinSet beta;

    std::vector<i32> alpha_id;
    std::vector<i32> beta_id;

    std::vector<std::size_t> alpha_off;
    std::vector<SpinMate> alpha_mate;

    std::vector<std::size_t> beta_off;
    std::vector<SpinMate> beta_mate;

    Residues alpha1;
    Residues beta1;
    Residues alpha2;
    Residues beta2;

private:
    void build_mates() {
        alpha_off.assign(alpha.size() + 1u, 0u);
        beta_off.assign(beta.size() + 1u, 0u);

        for (std::size_t iket = 0; iket < alpha_id.size(); ++iket) {
            ++alpha_off[static_cast<std::size_t>(alpha_id[iket]) + 1u];
            ++beta_off[static_cast<std::size_t>(beta_id[iket]) + 1u];
        }

        std::partial_sum(alpha_off.begin(), alpha_off.end(), alpha_off.begin());
        std::partial_sum(beta_off.begin(), beta_off.end(), beta_off.begin());

        alpha_mate.resize(alpha_id.size());
        beta_mate.resize(beta_id.size());

        std::vector<std::size_t> ap = alpha_off;
        std::vector<std::size_t> bp = beta_off;

        for (std::size_t iket = 0; iket < alpha_id.size(); ++iket) {
            alpha_mate[ap[static_cast<std::size_t>(alpha_id[iket])]++] =
                SpinMate{beta_id[iket], static_cast<i32>(iket)};

            beta_mate[bp[static_cast<std::size_t>(beta_id[iket])]++] =
                SpinMate{alpha_id[iket], static_cast<i32>(iket)};
        }
    }

    void sort_mates() {
        for (std::size_t id = 0; id + 1u < alpha_off.size(); ++id) {
            const auto lo = static_cast<std::ptrdiff_t>(alpha_off[id]);
            const auto hi = static_cast<std::ptrdiff_t>(alpha_off[id + 1u]);

            std::sort(
                alpha_mate.begin() + lo,
                alpha_mate.begin() + hi,
                [](const SpinMate& x, const SpinMate& y) { return x.spin < y.spin; }
            );
        }

        for (std::size_t id = 0; id + 1u < beta_off.size(); ++id) {
            const auto lo = static_cast<std::ptrdiff_t>(beta_off[id]);
            const auto hi = static_cast<std::ptrdiff_t>(beta_off[id + 1u]);

            std::sort(
                beta_mate.begin() + lo,
                beta_mate.begin() + hi,
                [](const SpinMate& x, const SpinMate& y) { return x.spin < y.spin; }
            );
        }
    }

    static void add_residues(
        const SpinSet& set,
        std::vector<KeyId>& one,
        std::vector<KeyId>& two,
        std::vector<int>& occ
    ) {
        for (i32 id = 0; id < static_cast<i32>(set.size()); ++id) {
            const auto words = set.get(id);

            bits::set_list(words, occ);
            const u64 fp = set.fp(id);

            for (int i : occ) {
                one.push_back(KeyId{fp ^ orb_fp(i), id});
            }

            for (std::size_t p = 0; p < occ.size(); ++p) {
                for (std::size_t q = p + 1u; q < occ.size(); ++q) {
                    two.push_back(KeyId{fp ^ orb_fp(occ[p]) ^ orb_fp(occ[q]), id});
                }
            }
        }
    }

    void build_residues() {
        std::vector<KeyId> a1;
        std::vector<KeyId> a2;
        std::vector<KeyId> b1;
        std::vector<KeyId> b2;

        std::vector<int> occ;

        add_residues(alpha, a1, a2, occ);
        add_residues(beta, b1, b2, occ);

        alpha1.build(std::move(a1));
        alpha2.build(std::move(a2));
        beta1.build(std::move(b1));
        beta2.build(std::move(b2));
    }
};

struct VisitScratch {
    void ensure_seen(std::size_t na, std::size_t nb) {
        if (seen_a.size() < na) {
            seen_a.assign(na, 0u);
            stamp_a = 1u;
        }

        if (seen_b.size() < nb) {
            seen_b.assign(nb, 0u);
            stamp_b = 1u;
        }
    }

    void next_a() {
        if (++stamp_a == 0u) {
            std::fill(seen_a.begin(), seen_a.end(), 0u);
            stamp_a = 1u;
        }
    }

    void next_b() {
        if (++stamp_b == 0u) {
            std::fill(seen_b.begin(), seen_b.end(), 0u);
            stamp_b = 1u;
        }
    }

    void ensure_cross(std::size_t nb) {
        if (cross_b.size() < nb) {
            cross_b.assign(nb, 0u);
            cross_i.assign(nb, 0);
            cross_a.assign(nb, 0);
            cross_sign.assign(nb, 1.0);
            cross_stamp = 1u;
        }
    }

    void next_cross() {
        if (++cross_stamp == 0u) {
            std::fill(cross_b.begin(), cross_b.end(), 0u);
            cross_stamp = 1u;
        }
    }

    std::vector<int> tmp_occ;

    std::vector<u32> seen_a;
    std::vector<u32> seen_b;
    u32 stamp_a = 1u;
    u32 stamp_b = 1u;

    std::vector<u32> cross_b;
    std::vector<int> cross_i;
    std::vector<int> cross_a;
    std::vector<double> cross_sign;
    u32 cross_stamp = 1u;

    std::vector<SpinExcitation> alpha_single;
    std::vector<SpinExcitation> beta_single;
    std::vector<SpinExcitation> alpha_double;
    std::vector<SpinExcitation> beta_double;
};

inline void find_single(
    const SpinSet& set,
    const Residues& res,
    std::span<const u64> bra_spin,
    std::vector<int>& occ,
    std::vector<u32>& seen,
    u32 stamp,
    std::vector<SpinExcitation>& out
) {
    out.clear();

    bits::set_list(bra_spin, occ);
    const u64 fp = spin_fp(bra_spin);

    for (int i : occ) {
        const u64 key = fp ^ orb_fp(i);

        for (const KeyId& item : res.equal(key)) {
            const std::size_t pos = static_cast<std::size_t>(item.id);

            if (seen[pos] == stamp) continue;

            const SpinDiff ex = spin_diff(bra_spin, set.get(item.id));

            if (ex.deg == 1) {
                seen[pos] = stamp;
                out.push_back(SpinExcitation{
                    item.id,
                    ex.occ[0],
                    0,
                    ex.vir[0],
                    0,
                    ex.sign
                });
            }
        }
    }
}

inline void find_double(
    const SpinSet& set,
    const Residues& res,
    std::span<const u64> bra_spin,
    std::vector<int>& occ,
    std::vector<u32>& seen,
    u32 stamp,
    std::vector<SpinExcitation>& out
) {
    out.clear();

    bits::set_list(bra_spin, occ);
    const u64 fp = spin_fp(bra_spin);

    for (std::size_t p = 0; p < occ.size(); ++p) {
        for (std::size_t q = p + 1u; q < occ.size(); ++q) {
            const u64 key = fp ^ orb_fp(occ[p]) ^ orb_fp(occ[q]);

            for (const KeyId& item : res.equal(key)) {
                const std::size_t pos = static_cast<std::size_t>(item.id);

                if (seen[pos] == stamp) continue;

                const SpinDiff ex = spin_diff(bra_spin, set.get(item.id));

                if (ex.deg == 2) {
                    seen[pos] = stamp;
                    out.push_back(SpinExcitation{
                        item.id,
                        ex.occ[0],
                        ex.occ[1],
                        ex.vir[0],
                        ex.vir[1],
                        ex.sign
                    });
                }
            }
        }
    }
}

[[nodiscard]] inline i32 find_det(const DetSpace& kets, DetRef det) noexcept {
    const i32 alpha_id = kets.alpha.find(det.alpha());
    if (alpha_id < 0) return -1;

    const i32 beta_id = kets.beta.find(det.beta());
    if (beta_id < 0) return -1;

    return kets.find_alpha(alpha_id, beta_id);
}


} // namespace libdet
