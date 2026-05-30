#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <libdet/det.hpp>
#include <libdet/integrals.hpp>
#include <libdet/slater.hpp>

namespace libdet {

/*
 * Determinant-driven finite-space search.
 *
 * Given a bra and a finite ket space, find connected kets in that space.
 *
 * Used by matrix, matvec, matmat, and project. This path does not generate
 * unrestricted external bras.
 */

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

    [[nodiscard]] i32 find_or_add(std::span<const u64> words) {
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

struct HalfSingle {
    i32 spin = -1;
    int i = 0;
    int a = 0;
    double sign = 1.0;
};

struct HalfDouble {
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

class KetSpace {
public:
    explicit KetSpace(DetBatchView kets)
        : nword(kets.nword),
          alpha(nword, 0x0f1234ab5678cdefULL),
          beta(nword, 0x1a2b3c4d5e6f7081ULL) {
        copy_batch(ket_words, kets);

        alpha_id.resize(kets.n_dets);
        beta_id.resize(kets.n_dets);

        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            const DetRef ket = kets[iket];
            alpha_id[iket] = alpha.find_or_add(ket.alpha());
            beta_id[iket] = beta.find_or_add(ket.beta());
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

    [[nodiscard]] i32 find_with_alpha(i32 alpha_spin, i32 beta_spin) const noexcept {
        const auto mates = alpha_mates(alpha_spin);

        const auto it = std::lower_bound(
            mates.begin(),
            mates.end(),
            beta_spin,
            [](const SpinMate& e, i32 x) { return e.spin < x; }
        );

        return it == mates.end() || it->spin != beta_spin ? -1 : it->ket;
    }

    [[nodiscard]] i32 find_with_beta(i32 beta_spin, i32 alpha_spin) const noexcept {
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

struct BraWork {
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

    std::vector<HalfSingle> alpha_single;
    std::vector<HalfSingle> beta_single;
    std::vector<HalfDouble> alpha_double;
    std::vector<HalfDouble> beta_double;
};

inline void find_single(
    const SpinSet& set,
    const Residues& res,
    std::span<const u64> bra_spin,
    std::vector<int>& occ,
    std::vector<u32>& seen,
    u32 stamp,
    std::vector<HalfSingle>& out
) {
    out.clear();

    bits::set_list(bra_spin, occ);
    const u64 fp = spin_fp(bra_spin);

    for (int i : occ) {
        const u64 key = fp ^ orb_fp(i);

        for (const KeyId& item : res.equal(key)) {
            const std::size_t pos = static_cast<std::size_t>(item.id);

            if (seen[pos] == stamp) continue;

            const HalfExcitation ex = diff_half(bra_spin, set.get(item.id));

            if (ex.deg == 1) {
                seen[pos] = stamp;
                out.push_back(HalfSingle{
                    item.id,
                    ex.occ[0],
                    ex.vir[0],
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
    std::vector<HalfDouble>& out
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

                const HalfExcitation ex = diff_half(bra_spin, set.get(item.id));

                if (ex.deg == 2) {
                    seen[pos] = stamp;
                    out.push_back(HalfDouble{
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

[[nodiscard]] inline i32 find_ket(const KetSpace& kets, DetRef det) noexcept {
    const i32 alpha_id = kets.alpha.find(det.alpha());
    if (alpha_id < 0) return -1;

    const i32 beta_id = kets.beta.find(det.beta());
    if (beta_id < 0) return -1;

    return kets.find_with_alpha(alpha_id, beta_id);
}

template <class Emit>
inline void emit_nonzero(double h, i32 iket, Emit&& emit) {
    if (h != 0.0) emit(iket, h);
}

/*
 * Scan off-diagonal kets in a finite ket space connected to bra.
 *
 * The diagonal is intentionally not emitted. Callers handle it separately
 * because diagonal terms require ket-index lookup and may be treated
 * differently in matrix and projection kernels.
 */
template <class Emit>
inline void scan_kets(
    const RHFIntegrals& ints,
    const KetSpace& kets,
    DetRef bra,
    BraWork& work,
    Emit&& emit
) {
    work.ensure_seen(kets.alpha.size(), kets.beta.size());

    work.next_a();
    find_single(
        kets.alpha,
        kets.alpha1,
        bra.alpha(),
        work.tmp_occ,
        work.seen_a,
        work.stamp_a,
        work.alpha_single
    );

    work.next_a();
    find_double(
        kets.alpha,
        kets.alpha2,
        bra.alpha(),
        work.tmp_occ,
        work.seen_a,
        work.stamp_a,
        work.alpha_double
    );

    work.next_b();
    find_single(
        kets.beta,
        kets.beta1,
        bra.beta(),
        work.tmp_occ,
        work.seen_b,
        work.stamp_b,
        work.beta_single
    );

    work.next_b();
    find_double(
        kets.beta,
        kets.beta2,
        bra.beta(),
        work.tmp_occ,
        work.seen_b,
        work.stamp_b,
        work.beta_double
    );

    const i32 bra_alpha = kets.alpha.find(bra.alpha());
    const i32 bra_beta = kets.beta.find(bra.beta());

    if (bra_beta >= 0) {
        for (const auto& ex : work.alpha_single) {
            const i32 iket = kets.find_with_beta(bra_beta, ex.spin);

            if (iket >= 0) {
                const double h =
                    ex.sign * Slater::single_a(ints, bra, ex.i, ex.a);

                emit_nonzero(h, iket, emit);
            }
        }

        for (const auto& ex : work.alpha_double) {
            const i32 iket = kets.find_with_beta(bra_beta, ex.spin);

            if (iket >= 0) {
                const double h =
                    ex.sign
                    * Slater::double_aa(ints, ex.i, ex.j, ex.a, ex.b);

                emit_nonzero(h, iket, emit);
            }
        }
    }

    if (bra_alpha >= 0) {
        for (const auto& ex : work.beta_single) {
            const i32 iket = kets.find_with_alpha(bra_alpha, ex.spin);

            if (iket >= 0) {
                const double h =
                    ex.sign * Slater::single_b(ints, bra, ex.i, ex.a);

                emit_nonzero(h, iket, emit);
            }
        }

        for (const auto& ex : work.beta_double) {
            const i32 iket = kets.find_with_alpha(bra_alpha, ex.spin);

            if (iket >= 0) {
                const double h =
                    ex.sign
                    * Slater::double_bb(ints, ex.i, ex.j, ex.a, ex.b);

                emit_nonzero(h, iket, emit);
            }
        }
    }

    /*
     * Opposite-spin double excitation is the intersection of one alpha
     * half-excitation and one beta half-excitation.
     */
    work.ensure_cross(kets.beta.size());
    work.next_cross();

    for (const auto& ex : work.beta_single) {
        const std::size_t beta_id = static_cast<std::size_t>(ex.spin);

        work.cross_b[beta_id] = work.cross_stamp;
        work.cross_i[beta_id] = ex.i;
        work.cross_a[beta_id] = ex.a;
        work.cross_sign[beta_id] = ex.sign;
    }

    for (const auto& ax : work.alpha_single) {
        for (const SpinMate& mate : kets.alpha_mates(ax.spin)) {
            const std::size_t beta_id = static_cast<std::size_t>(mate.spin);

            if (work.cross_b[beta_id] != work.cross_stamp) continue;

            const double h =
                ax.sign
                * work.cross_sign[beta_id]
                * Slater::double_ab(
                    ints,
                    ax.i,
                    work.cross_i[beta_id],
                    ax.a,
                    work.cross_a[beta_id]
                );

            emit_nonzero(h, mate.ket, emit);
        }
    }
}

} // namespace libdet