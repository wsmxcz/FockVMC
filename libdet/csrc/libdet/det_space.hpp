#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <libdet/det.hpp>

namespace libdet {

class DetIndex {
public:
    DetIndex() = default;
    explicit DetIndex(DetBatchView dets) { build(dets); }

    void build(DetBatchView dets) {
        dets_ = dets;
        std::size_t cap = 8;
        while (cap < dets.n_dets * 2u + 1u) cap <<= 1u;
        slot_.assign(cap, -1);
        mask_ = cap - 1u;
        for (std::size_t i = 0; i < dets.n_dets; ++i) insert_existing(static_cast<i32>(i));
    }

    [[nodiscard]] i32 find(DetRef det) const noexcept {
        if (slot_.empty()) return -1;
        std::size_t s = DetHash{}(det) & mask_;
        for (;;) {
            const i32 idx = slot_[s];
            if (idx < 0) return -1;
            if (det_equal(dets_[static_cast<std::size_t>(idx)], det)) return idx;
            s = (s + 1u) & mask_;
        }
    }

private:
    DetBatchView dets_{};
    std::vector<i32> slot_;
    std::size_t mask_ = 0;

    void insert_existing(i32 idx) {
        const DetRef det = dets_[static_cast<std::size_t>(idx)];
        std::size_t s = DetHash{}(det) & mask_;
        while (slot_[s] >= 0) s = (s + 1u) & mask_;
        slot_[s] = idx;
    }
};

[[nodiscard]] inline u64 orb_fp(int p) noexcept {
    return splitmix64(0xd1b54a32d192ed03ULL ^ static_cast<u64>(p + 1));
}

[[nodiscard]] inline u64 spin_fp(std::span<const u64> words) noexcept {
    u64 h = 0;
    bits::each_set(words, [&](int p) { h ^= orb_fp(p); });
    return h;
}

class SpinSet {
public:
    explicit SpinSet(u32 nword = 0, u64 tag = 0) : nword_(nword), tag_(tag) {}

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

    [[nodiscard]] i32 find_or_add(std::span<const u64> words) {
        const i32 old = find(words);
        if (old >= 0) return old;
        const i32 idx = to_i32(size());
        data_.insert(data_.end(), words.begin(), words.end());
        fp_.push_back(spin_fp(words));
        if (nword_ == 1) one_.emplace(words[0], idx);
        else many_[hash_words(tag_, words)].push_back(idx);
        return idx;
    }

    [[nodiscard]] std::span<const u64> get(i32 idx) const noexcept {
        const std::size_t off = static_cast<std::size_t>(idx) * static_cast<std::size_t>(nword_);
        return {data_.data() + off, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] u64 fp(i32 idx) const noexcept { return fp_[static_cast<std::size_t>(idx)]; }

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
        const auto lo = std::lower_bound(items_.begin(), items_.end(), key,
            [](const KeyId& a, u64 b) { return a.key < b; });
        const auto hi = std::upper_bound(items_.begin(), items_.end(), key,
            [](u64 a, const KeyId& b) { return a < b.key; });
        return {items_.data() + static_cast<std::size_t>(lo - items_.begin()), static_cast<std::size_t>(hi - lo)};
    }

private:
    std::vector<KeyId> items_;
};

struct HalfExcitation {
    int deg = 0;
    std::array<int, 2> occ{0, 0};
    std::array<int, 2> vir{0, 0};
    double sign = 1.0;
};

[[nodiscard]] inline HalfExcitation diff_half(std::span<const u64> src, std::span<const u64> dst) noexcept {
    HalfExcitation ex;
    if (src.size() == 1) {
        const u64 gone0 = src[0] & ~dst[0];
        const u64 come0 = dst[0] & ~src[0];
        const int dx = static_cast<int>(std::popcount(gone0) + std::popcount(come0));
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
        if (ex.deg == 1) ex.sign = detail::sign_single(src, ex.occ[0], ex.vir[0]);
        else if (ex.deg == 2) ex.sign = detail::sign_double(src, ex.occ[0], ex.occ[1], ex.vir[0], ex.vir[1]);
        return ex;
    }

    const int dx = bits::popcount_xor(src, dst);
    if (dx & 1) {
        ex.deg = 3;
        return ex;
    }
    ex.deg = dx >> 1;
    if (ex.deg > 2) return ex;

    int io = 0;
    int iv = 0;
    for (std::size_t w = 0; w < src.size(); ++w) {
        u64 gone = src[w] & ~dst[w];
        u64 come = dst[w] & ~src[w];
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
    if (ex.deg == 1) ex.sign = detail::sign_single(src, ex.occ[0], ex.vir[0]);
    else if (ex.deg == 2) ex.sign = detail::sign_double(src, ex.occ[0], ex.occ[1], ex.vir[0], ex.vir[1]);
    return ex;
}

struct SelSingle {
    i32 id = -1;
    int occ = 0;
    int vir = 0;
    double sign = 1.0;
};

struct SelDouble {
    i32 id = -1;
    int occ_i = 0;
    int occ_j = 0;
    int vir_a = 0;
    int vir_b = 0;
    double sign = 1.0;
};

struct Mate {
    i32 other = -1;
    i32 det = -1;
};

class DetSpace {
public:
    explicit DetSpace(DetBatchView dets)
        : nword(dets.nword), alpha(nword, 0x0f1234ab5678cdefULL), beta(nword, 0x1a2b3c4d5e6f7081ULL) {
        copy_batch(det_words, dets);
        aid.resize(dets.n_dets);
        bid.resize(dets.n_dets);
        for (std::size_t idx = 0; idx < dets.n_dets; ++idx) {
            const DetRef d = dets[idx];
            aid[idx] = alpha.find_or_add(d.alpha());
            bid[idx] = beta.find_or_add(d.beta());
        }
        build_mates();
        sort_mates();
        build_residues();
    }

    [[nodiscard]] std::span<const Mate> alpha_mates(i32 alpha_id) const noexcept {
        const std::size_t lo = alpha_off[static_cast<std::size_t>(alpha_id)];
        const std::size_t hi = alpha_off[static_cast<std::size_t>(alpha_id) + 1u];
        return {alpha_pair.data() + lo, hi - lo};
    }

    [[nodiscard]] std::span<const Mate> beta_mates(i32 beta_id) const noexcept {
        const std::size_t lo = beta_off[static_cast<std::size_t>(beta_id)];
        const std::size_t hi = beta_off[static_cast<std::size_t>(beta_id) + 1u];
        return {beta_pair.data() + lo, hi - lo};
    }

    [[nodiscard]] i32 find_with_alpha(i32 alpha_id, i32 beta_id) const noexcept {
        const auto s = alpha_mates(alpha_id);
        auto it = std::lower_bound(s.begin(), s.end(), beta_id, [](const Mate& e, i32 x) { return e.other < x; });
        return it == s.end() || it->other != beta_id ? -1 : it->det;
    }

    [[nodiscard]] i32 find_with_beta(i32 beta_id, i32 alpha_id) const noexcept {
        const auto s = beta_mates(beta_id);
        auto it = std::lower_bound(s.begin(), s.end(), alpha_id, [](const Mate& e, i32 x) { return e.other < x; });
        return it == s.end() || it->other != alpha_id ? -1 : it->det;
    }

    u32 nword = 0;
    std::vector<u64> det_words;
    SpinSet alpha;
    SpinSet beta;
    std::vector<i32> aid;
    std::vector<i32> bid;
    std::vector<std::size_t> alpha_off;
    std::vector<Mate> alpha_pair;
    std::vector<std::size_t> beta_off;
    std::vector<Mate> beta_pair;
    Residues alpha1;
    Residues beta1;
    Residues alpha2;
    Residues beta2;

private:
    void build_mates() {
        alpha_off.assign(alpha.size() + 1u, 0u);
        beta_off.assign(beta.size() + 1u, 0u);
        for (std::size_t idx = 0; idx < aid.size(); ++idx) {
            ++alpha_off[static_cast<std::size_t>(aid[idx]) + 1u];
            ++beta_off[static_cast<std::size_t>(bid[idx]) + 1u];
        }
        std::partial_sum(alpha_off.begin(), alpha_off.end(), alpha_off.begin());
        std::partial_sum(beta_off.begin(), beta_off.end(), beta_off.begin());

        alpha_pair.resize(aid.size());
        beta_pair.resize(bid.size());
        std::vector<std::size_t> ap = alpha_off;
        std::vector<std::size_t> bp = beta_off;
        for (std::size_t idx = 0; idx < aid.size(); ++idx) {
            alpha_pair[ap[static_cast<std::size_t>(aid[idx])]++] = Mate{bid[idx], static_cast<i32>(idx)};
            beta_pair[bp[static_cast<std::size_t>(bid[idx])]++] = Mate{aid[idx], static_cast<i32>(idx)};
        }
    }

    void sort_mates() {
        for (std::size_t id = 0; id + 1u < alpha_off.size(); ++id) {
            const auto lo = static_cast<std::ptrdiff_t>(alpha_off[id]);
            const auto hi = static_cast<std::ptrdiff_t>(alpha_off[id + 1u]);
            std::sort(alpha_pair.begin() + lo, alpha_pair.begin() + hi, [](const Mate& x, const Mate& y) { return x.other < y.other; });
        }
        for (std::size_t id = 0; id + 1u < beta_off.size(); ++id) {
            const auto lo = static_cast<std::ptrdiff_t>(beta_off[id]);
            const auto hi = static_cast<std::ptrdiff_t>(beta_off[id + 1u]);
            std::sort(beta_pair.begin() + lo, beta_pair.begin() + hi, [](const Mate& x, const Mate& y) { return x.other < y.other; });
        }
    }

    static void add_residues(const SpinSet& set, std::vector<KeyId>& one, std::vector<KeyId>& two, std::vector<int>& occ) {
        for (i32 id = 0; id < static_cast<i32>(set.size()); ++id) {
            const auto w = set.get(id);
            bits::set_list(w, occ);
            const u64 fp = set.fp(id);
            for (int i : occ) one.push_back(KeyId{fp ^ orb_fp(i), id});
            for (std::size_t p = 0; p < occ.size(); ++p) {
                for (std::size_t q = p + 1u; q < occ.size(); ++q) {
                    two.push_back(KeyId{fp ^ orb_fp(occ[p]) ^ orb_fp(occ[q]), id});
                }
            }
        }
    }

    void build_residues() {
        std::vector<KeyId> a1, a2, b1, b2;
        std::vector<int> occ;
        add_residues(alpha, a1, a2, occ);
        add_residues(beta, b1, b2, occ);
        alpha1.build(std::move(a1));
        alpha2.build(std::move(a2));
        beta1.build(std::move(b1));
        beta2.build(std::move(b2));
    }
};

inline void find_single(const SpinSet& set, const Residues& res, std::span<const u64> src,
                        std::vector<int>& occ, std::vector<u32>& seen, u32 stamp,
                        std::vector<SelSingle>& out) {
    out.clear();
    bits::set_list(src, occ);
    const u64 fp = spin_fp(src);
    for (int i : occ) {
        const u64 key = fp ^ orb_fp(i);
        for (const KeyId& item : res.equal(key)) {
            const std::size_t pos = static_cast<std::size_t>(item.id);
            if (seen[pos] == stamp) continue;
            const HalfExcitation ex = diff_half(src, set.get(item.id));
            if (ex.deg == 1) {
                seen[pos] = stamp;
                out.push_back(SelSingle{item.id, ex.occ[0], ex.vir[0], ex.sign});
            }
        }
    }
}

inline void find_double(const SpinSet& set, const Residues& res, std::span<const u64> src,
                        std::vector<int>& occ, std::vector<u32>& seen, u32 stamp,
                        std::vector<SelDouble>& out) {
    out.clear();
    bits::set_list(src, occ);
    const u64 fp = spin_fp(src);
    for (std::size_t p = 0; p < occ.size(); ++p) {
        for (std::size_t q = p + 1u; q < occ.size(); ++q) {
            const u64 key = fp ^ orb_fp(occ[p]) ^ orb_fp(occ[q]);
            for (const KeyId& item : res.equal(key)) {
                const std::size_t pos = static_cast<std::size_t>(item.id);
                if (seen[pos] == stamp) continue;
                const HalfExcitation ex = diff_half(src, set.get(item.id));
                if (ex.deg == 2) {
                    seen[pos] = stamp;
                    out.push_back(SelDouble{item.id, ex.occ[0], ex.occ[1], ex.vir[0], ex.vir[1], ex.sign});
                }
            }
        }
    }
}

[[nodiscard]] inline i32 find_det(const DetSpace& space, DetRef det) noexcept {
    const i32 a = space.alpha.find(det.alpha());
    if (a < 0) return -1;
    const i32 b = space.beta.find(det.beta());
    if (b < 0) return -1;
    return space.find_with_alpha(a, b);
}

class SmallRng {
public:
    explicit SmallRng(u64 seed) : state_(seed) {}

    [[nodiscard]] u64 next_u64() noexcept {
        state_ = splitmix64(state_);
        return state_;
    }

    [[nodiscard]] double uniform01() noexcept {
        return static_cast<double>((next_u64() >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

} // namespace libdet
