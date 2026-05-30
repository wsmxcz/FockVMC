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

/*
 * Fixed cutoff for precomputed heat-bath double-excitation candidates.
 *
 * The table is an integral-level acceleration structure. Callers that need
 * exact enumeration below eps_hb must use the exact excitation path.
 */
inline constexpr double eps_hb = 1.0e-12;

struct DoubleCand {
    int a = 0;
    int b = 0;
    double h = 0.0;
    double abs_h = 0.0;
};

/*
 * HeatBathTable stores double-excitation integral factors sorted by |h|.
 *
 * Same-spin candidates:
 *
 *   occupied pair i,j -> target pair a,b, a < b
 *
 * Opposite-spin candidates:
 *
 *   occupied pair ia,ib -> target pair a,b
 *
 * The determinant sign is ket-dependent and is not stored here.
 */
class HeatBathTable {
public:
    explicit HeatBathTable(const RHFIntegrals& ints)
        : norb_(ints.norb()) {
        if (norb_ < 0) {
            throw std::invalid_argument("HeatBathTable: norb must be nonnegative");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        const std::size_t np = n * (n + 1u) / 2u;

        aa_off_.assign(np + 1u, 0u);
        ab_off_.assign(n * n + 1u, 0u);

        build_aa(ints);
        build_ab(ints);
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double cutoff() const noexcept { return eps_hb; }

    [[nodiscard]] std::span<const DoubleCand> aa(int i, int j) const noexcept {
        return aa_candidates(i, j);
    }

    [[nodiscard]] std::span<const DoubleCand> bb(int i, int j) const noexcept {
        return aa_candidates(i, j);
    }

    [[nodiscard]] std::span<const DoubleCand> ab(int ia, int ib) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);

        const std::size_t lo = ab_off_[k];
        const std::size_t hi = ab_off_[k + 1u];

        return {ab_data_.data() + lo, hi - lo};
    }

    [[nodiscard]] std::span<const DoubleCand> aa_ge(int i, int j, double eps) const noexcept {
        return prefix_ge(aa(i, j), eps);
    }

    [[nodiscard]] std::span<const DoubleCand> bb_ge(int i, int j, double eps) const noexcept {
        return prefix_ge(bb(i, j), eps);
    }

    [[nodiscard]] std::span<const DoubleCand> ab_ge(int ia, int ib, double eps) const noexcept {
        return prefix_ge(ab(ia, ib), eps);
    }

    [[nodiscard]] std::span<const DoubleCand> aa_window(
        int i,
        int j,
        double eps2,
        double eps1
    ) const noexcept {
        return window(aa(i, j), eps2, eps1);
    }

    [[nodiscard]] std::span<const DoubleCand> bb_window(
        int i,
        int j,
        double eps2,
        double eps1
    ) const noexcept {
        return window(bb(i, j), eps2, eps1);
    }

    [[nodiscard]] std::span<const DoubleCand> ab_window(
        int ia,
        int ib,
        double eps2,
        double eps1
    ) const noexcept {
        return window(ab(ia, ib), eps2, eps1);
    }

private:
    int norb_ = 0;

    std::vector<std::size_t> aa_off_;
    std::vector<DoubleCand> aa_data_;

    std::vector<std::size_t> ab_off_;
    std::vector<DoubleCand> ab_data_;

    [[nodiscard]] std::span<const DoubleCand> aa_candidates(int i, int j) const noexcept {
        const std::size_t k = RHFIntegrals::pair_index(i, j);
        const std::size_t lo = aa_off_[k];
        const std::size_t hi = aa_off_[k + 1u];

        return {aa_data_.data() + lo, hi - lo};
    }

    /*
     * Candidate lists are sorted by descending abs_h.
     *
     * first_lt(cands, eps) returns the first index k such that
     *
     *   cands[k].abs_h < eps.
     */
    [[nodiscard]] static std::size_t first_lt(
        std::span<const DoubleCand> cands,
        double eps
    ) noexcept {
        std::size_t lo = 0;
        std::size_t hi = cands.size();

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;

            if (cands[mid].abs_h < eps) {
                hi = mid;
            } else {
                lo = mid + 1u;
            }
        }

        return lo;
    }

    [[nodiscard]] static std::span<const DoubleCand> prefix_ge(
        std::span<const DoubleCand> cands,
        double eps
    ) noexcept {
        if (cands.empty()) return {};

        const std::size_t end = first_lt(cands, eps);
        return {cands.data(), end};
    }

    /*
     * Return candidates satisfying:
     *
     *   eps2 <= abs_h < eps1.
     */
    [[nodiscard]] static std::span<const DoubleCand> window(
        std::span<const DoubleCand> cands,
        double eps2,
        double eps1
    ) noexcept {
        if (cands.empty() || eps1 <= eps2 || eps1 <= 0.0) return {};

        const std::size_t begin = first_lt(cands, eps1);
        const std::size_t end = first_lt(cands, eps2);

        if (end <= begin) return {};
        return {cands.data() + begin, end - begin};
    }

    static void sort_candidates(std::vector<DoubleCand>& cands) {
        std::sort(cands.begin(), cands.end(), [](const DoubleCand& x, const DoubleCand& y) {
            return x.abs_h > y.abs_h;
        });
    }

    void build_aa(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;

        std::vector<DoubleCand> cands;

        for (int i = 0; i < norb_; ++i) {
            for (int j = 0; j <= i; ++j) {
                const std::size_t k = RHFIntegrals::pair_index(i, j);
                aa_off_[k] = aa_data_.size();

                if (i == j) continue;

                cands.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = a + 1; b < norb_; ++b) {
                        const double h = ints.chem(i, b, j, a) - ints.chem(i, a, j, b);
                        const double abs_h = std::abs(h);

                        if (abs_h >= eps_hb) {
                            cands.push_back(DoubleCand{a, b, h, abs_h});
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

        std::vector<DoubleCand> cands;

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
                        const double abs_h = std::abs(h);

                        if (abs_h >= eps_hb) {
                            cands.push_back(DoubleCand{a, b, h, abs_h});
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