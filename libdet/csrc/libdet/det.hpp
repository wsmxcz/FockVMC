#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <span>
#include <vector>

#include <libdet/bit.hpp>

namespace libdet {

[[nodiscard]] inline constexpr std::size_t det_size(u32 nword) noexcept {
    return 2u * static_cast<std::size_t>(nword);
}

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
    DetScratch() = default;
    explicit DetScratch(u32 nword) : words_(det_size(nword), 0u), nword_(nword) {}

    void resize(u32 nword) {
        nword_ = nword;
        words_.assign(det_size(nword), 0u);
    }

    void load(DetRef det) {
        std::copy(det.alpha().begin(), det.alpha().end(), words_.begin());
        std::copy(det.beta().begin(), det.beta().end(), words_.begin() + static_cast<std::ptrdiff_t>(nword_));
    }

    [[nodiscard]] std::span<u64> alpha() noexcept {
        return {words_.data(), static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] std::span<u64> beta() noexcept {
        return {words_.data() + static_cast<std::size_t>(nword_), static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] DetRef view() const noexcept {
        return DetRef(words_.data(), words_.data() + static_cast<std::size_t>(nword_), nword_);
    }

private:
    std::vector<u64> words_;
    u32 nword_ = 0;
};

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

[[nodiscard]] inline DetRef det_at(const std::vector<u64>& data, u32 nword, std::size_t idx) noexcept {
    const std::size_t stride = det_size(nword);
    const u64* ptr = data.data() + idx * stride;
    return DetRef(ptr, ptr + static_cast<std::size_t>(nword), nword);
}

inline void append_det(std::vector<u64>& out, DetRef det) {
    out.insert(out.end(), det.alpha().begin(), det.alpha().end());
    out.insert(out.end(), det.beta().begin(), det.beta().end());
}

[[nodiscard]] inline std::size_t append_det_index(std::vector<u64>& out, u32 nword, DetRef det) {
    const std::size_t idx = out.size() / det_size(nword);
    append_det(out, det);
    return idx;
}

inline void copy_batch(std::vector<u64>& dst, DetBatchView src) {
    dst.resize(src.n_dets * det_size(src.nword));
    if (!dst.empty()) std::copy_n(src.data, dst.size(), dst.data());
}

[[nodiscard]] inline bool det_equal(DetRef a, DetRef b) noexcept {
    return a.nword() == b.nword() && bits::equal(a.alpha(), b.alpha()) && bits::equal(a.beta(), b.beta());
}

struct DetLess {
    [[nodiscard]] bool operator()(DetRef lhs, DetRef rhs) const noexcept {
        const auto la = lhs.alpha();
        const auto ra = rhs.alpha();
        if (la.size() == 1 && ra.size() == 1) {
            if (la[0] != ra[0]) return la[0] < ra[0];
            return lhs.beta()[0] < rhs.beta()[0];
        }
        for (std::size_t i = 0; i < la.size(); ++i) {
            if (la[i] != ra[i]) return la[i] < ra[i];
        }
        const auto lb = lhs.beta();
        const auto rb = rhs.beta();
        for (std::size_t i = 0; i < lb.size(); ++i) {
            if (lb[i] != rb[i]) return lb[i] < rb[i];
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
    for (u64 x : words) h = splitmix64(h ^ splitmix64(x + 0x517cc1b727220a95ULL));
    return h;
}

struct DetHash {
    [[nodiscard]] std::size_t operator()(DetRef det) const noexcept {
        if (det.nword() == 1) {
            const u64 h = det.alpha()[0] ^ splitmix64(det.beta()[0] + 0x517cc1b727220a95ULL);
            return static_cast<std::size_t>(splitmix64(h));
        }
        u64 h = 0x9e3779b97f4a7c15ULL ^ static_cast<u64>(det.nword());
        for (u64 x : det.alpha()) h = splitmix64(h ^ x);
        for (u64 x : det.beta()) h = splitmix64(h ^ (x + 0x517cc1b727220a95ULL));
        return static_cast<std::size_t>(h);
    }
};

[[nodiscard]] inline u64 det_fingerprint(DetRef det, u64 seed = 0) noexcept {
    u64 h = splitmix64(seed ^ 0x726f776c6f63616cULL);
    if (det.nword() == 1) {
        h = splitmix64(h ^ det.alpha()[0]);
        return splitmix64(h ^ (det.beta()[0] + 0x517cc1b727220a95ULL));
    }
    for (u64 x : det.alpha()) h = splitmix64(h ^ x);
    for (u64 x : det.beta()) h = splitmix64(h ^ (x + 0x517cc1b727220a95ULL));
    return h;
}

inline void sort_unique_dets(std::vector<u64>& packed, u32 nword) {
    const std::size_t stride = det_size(nword);
    if (stride == 0 || packed.empty()) return;
    const std::size_t ndet = packed.size() / stride;

    std::vector<std::size_t> order(ndet);
    for (std::size_t i = 0; i < ndet; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return DetLess{}(det_at(packed, nword, a), det_at(packed, nword, b));
    });

    std::vector<u64> out;
    out.reserve(packed.size());
    std::size_t prev = static_cast<std::size_t>(-1);
    for (std::size_t idx : order) {
        if (prev != static_cast<std::size_t>(-1) && det_equal(det_at(packed, nword, idx), det_at(packed, nword, prev))) {
            continue;
        }
        append_det(out, det_at(packed, nword, idx));
        prev = idx;
    }
    packed.swap(out);
}

namespace detail {

[[nodiscard]] inline double sign_single(std::span<const u64> occ, int i, int a) noexcept {
    return bits::parity_between(occ, i, a) ? -1.0 : 1.0;
}

[[nodiscard]] inline double sign_double(std::span<const u64> occ, int i, int j, int a, int b) noexcept {
    int p = 0;
    p ^= bits::popcount_between(occ, i, a);
    p ^= bits::popcount_between(occ, j, b);

    const int x0 = std::min(i, a);
    const int x1 = std::max(i, a);
    const int y0 = std::min(j, b);
    const int y1 = std::max(j, b);
    const bool cross = (y0 > x0 && y0 < x1 && y1 > x1) || (x0 > y0 && x0 < y1 && x1 > y1);
    if (cross) ++p;
    return (p & 1) ? -1.0 : 1.0;
}

} // namespace detail

struct Excitation {
    int deg = 0;
    int na = 0;
    int nb = 0;
    std::array<int, 2> occ_a{0, 0};
    std::array<int, 2> vir_a{0, 0};
    std::array<int, 2> occ_b{0, 0};
    std::array<int, 2> vir_b{0, 0};
    double sign = 1.0;
};

struct RowOcc {
    explicit RowOcc(u32 nword = 0) : det(nword) {}

    void resize(u32 nword, int) { det.resize(nword); }

    DetScratch det;
    std::vector<int> occ_a;
    std::vector<int> occ_b;
};

inline void fill_occ(DetRef det, RowOcc& work) {
    work.det.load(det);
    bits::set_list(det.alpha(), work.occ_a);
    bits::set_list(det.beta(), work.occ_b);
}

[[nodiscard]] inline Excitation diff(DetRef bra, DetRef ket) noexcept {
    Excitation ex;
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

    auto fill_one = [](std::span<const u64> src, std::span<const u64> dst,
                       std::array<int, 2>& occ, std::array<int, 2>& vir, int n) {
        int io = 0;
        int iv = 0;
        for (std::size_t w = 0; w < src.size(); ++w) {
            u64 gone = src[w] & ~dst[w];
            u64 come = dst[w] & ~src[w];
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
        ex.sign = ex.na == 1 ? detail::sign_single(bra.alpha(), ex.occ_a[0], ex.vir_a[0])
                             : detail::sign_single(bra.beta(), ex.occ_b[0], ex.vir_b[0]);
    } else if (ex.na == 2) {
        ex.sign = detail::sign_double(bra.alpha(), ex.occ_a[0], ex.occ_a[1], ex.vir_a[0], ex.vir_a[1]);
    } else if (ex.nb == 2) {
        ex.sign = detail::sign_double(bra.beta(), ex.occ_b[0], ex.occ_b[1], ex.vir_b[0], ex.vir_b[1]);
    } else if (ex.deg == 2) {
        ex.sign = detail::sign_single(bra.alpha(), ex.occ_a[0], ex.vir_a[0]) *
                  detail::sign_single(bra.beta(), ex.occ_b[0], ex.vir_b[0]);
    }
    return ex;
}

} // namespace libdet
