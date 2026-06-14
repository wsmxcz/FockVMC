#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/integrals.hpp>
#include <libdet/slater.hpp>

namespace libdet {

struct Cand {
    int a = 0;
    int b = 0;
    double h = 0.0;
};

/*
 * Screened Hamiltonian Grpah.
 *
 * A Screen owns all same-spin and opposite-spin double candidates whose
 * integral factor satisfies abs(h) >= cutoff. Candidate lists are sorted by
 * descending abs(h), so row scans can cheaply tighten the per-ket threshold.
 */
class Screen {
public:
    Screen(const RHFIntegrals& ints, double cutoff)
        : norb_(ints.norb()), cutoff_(cutoff) {
        if (norb_ < 0) {
            throw std::invalid_argument("Screen: norb must be nonnegative");
        }

        if (std::isnan(cutoff_) || cutoff_ <= 0.0) {
            throw std::invalid_argument("Screen: cutoff must be positive");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        const std::size_t np = n * (n + 1u) / 2u;

        aa_off_.assign(np + 1u, 0u);
        ab_off_.assign(n * n + 1u, 0u);

        build_aa(ints);
        build_ab(ints);
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double cutoff() const noexcept { return cutoff_; }

    [[nodiscard]] std::span<const Cand> aa(
        int i,
        int j,
        double min_abs = 0.0
    ) const noexcept {
        return prefix_ge(aa_candidates(i, j), min_abs);
    }

    [[nodiscard]] std::span<const Cand> bb(
        int i,
        int j,
        double min_abs = 0.0
    ) const noexcept {
        return prefix_ge(aa_candidates(i, j), min_abs);
    }

    [[nodiscard]] std::span<const Cand> ab(
        int ia,
        int ib,
        double min_abs = 0.0
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);

        const std::size_t lo = ab_off_[k];
        const std::size_t hi = ab_off_[k + 1u];

        return prefix_ge({ab_data_.data() + lo, hi - lo}, min_abs);
    }

private:
    int norb_ = 0;
    double cutoff_ = 0.0;

    std::vector<std::size_t> aa_off_;
    std::vector<Cand> aa_data_;

    std::vector<std::size_t> ab_off_;
    std::vector<Cand> ab_data_;

    [[nodiscard]] std::span<const Cand> aa_candidates(int i, int j) const noexcept {
        const std::size_t k = RHFIntegrals::pair_index(i, j);
        const std::size_t lo = aa_off_[k];
        const std::size_t hi = aa_off_[k + 1u];

        return {aa_data_.data() + lo, hi - lo};
    }

    [[nodiscard]] bool keep(double h) const noexcept {
        return std::abs(h) >= cutoff_;
    }

    [[nodiscard]] static double abs_h(const Cand& c) noexcept {
        return std::abs(c.h);
    }

    [[nodiscard]] static bool cand_before(const Cand& x, const Cand& y) noexcept {
        const double ax = abs_h(x);
        const double ay = abs_h(y);

        if (ax != ay) return ax > ay;
        if (x.a != y.a) return x.a < y.a;
        if (x.b != y.b) return x.b < y.b;
        return x.h < y.h;
    }

    static void sort_candidates(std::vector<Cand>& cands) {
        std::sort(cands.begin(), cands.end(), cand_before);
    }

    [[nodiscard]] static std::size_t first_lt(
        std::span<const Cand> cands,
        double min_abs
    ) noexcept {
        if (min_abs <= 0.0) return cands.size();

        std::size_t lo = 0;
        std::size_t hi = cands.size();

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;

            if (abs_h(cands[mid]) < min_abs) {
                hi = mid;
            } else {
                lo = mid + 1u;
            }
        }

        return lo;
    }

    [[nodiscard]] static std::span<const Cand> prefix_ge(
        std::span<const Cand> cands,
        double min_abs
    ) noexcept {
        if (cands.empty()) return {};

        const std::size_t end = first_lt(cands, min_abs);
        return {cands.data(), end};
    }

    void build_aa(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;

        std::vector<Cand> cands;

        for (int hi = 0; hi < norb_; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) {
                const std::size_t k = RHFIntegrals::pair_index(hi, lo);
                aa_off_[k] = aa_data_.size();

                if (hi == lo) continue;

                cands.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = a + 1; b < norb_; ++b) {
                        const double h = Slater::double_aa(ints, lo, hi, a, b);

                        if (keep(h)) {
                            cands.push_back(Cand{a, b, h});
                        }
                    }
                }

                sort_candidates(cands);
                aa_data_.insert(aa_data_.end(), cands.begin(), cands.end());
            }
        }

        aa_off_.back() = aa_data_.size();
    }

    void build_ab(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;

        std::vector<Cand> cands;

        for (int ia = 0; ia < norb_; ++ia) {
            for (int ib = 0; ib < norb_; ++ib) {
                const std::size_t k =
                    static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
                    + static_cast<std::size_t>(ib);

                ab_off_[k] = ab_data_.size();
                cands.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = 0; b < norb_; ++b) {
                        const double h = Slater::double_ab(ints, ia, ib, a, b);

                        if (keep(h)) {
                            cands.push_back(Cand{a, b, h});
                        }
                    }
                }

                sort_candidates(cands);
                ab_data_.insert(ab_data_.end(), cands.begin(), cands.end());
            }
        }

        ab_off_.back() = ab_data_.size();
    }
};

} // namespace libdet
