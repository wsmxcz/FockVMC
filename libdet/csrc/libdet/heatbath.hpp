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
 * Fixed heat-bath cutoff for precomputed double-excitation tables.
 *
 * The table is an integral-level acceleration structure. Callers that need
 * exact enumeration below eps_hb must use the exact excitation path.
 */
inline constexpr double eps_hb = 1.0e-12;

struct DoubleCand {
    int p = 0;
    int q = 0;
    double h = 0.0;
    double abs_h = 0.0;
};

/*
 * HeatBathTable stores double-excitation integral factors sorted by |h|.
 *
 * Same-spin rows:
 *
 *   occupied pair (i,j) -> virtual pair (p,q), p < q
 *   h = <ij||pq>
 *
 * Opposite-spin rows:
 *
 *   occupied pair (ia,ib) -> virtual pair (pa,pb)
 *   h = (ia pa | ib pb)
 *
 * The determinant sign is not stored here; it is row-dependent and must be
 * computed from the source determinant occupation.
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
        return aa_row(i, j);
    }

    [[nodiscard]] std::span<const DoubleCand> bb(int i, int j) const noexcept {
        return aa_row(i, j);
    }

    [[nodiscard]] std::span<const DoubleCand> ab(int i, int j) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(j);

        const std::size_t lo = ab_off_[k];
        const std::size_t hi = ab_off_[k + 1u];

        return {ab_data_.data() + lo, hi - lo};
    }

    [[nodiscard]] std::span<const DoubleCand> aa_ge(int i, int j, double cut) const noexcept {
        return prefix_ge(aa(i, j), cut);
    }

    [[nodiscard]] std::span<const DoubleCand> bb_ge(int i, int j, double cut) const noexcept {
        return prefix_ge(bb(i, j), cut);
    }

    [[nodiscard]] std::span<const DoubleCand> ab_ge(int i, int j, double cut) const noexcept {
        return prefix_ge(ab(i, j), cut);
    }

    [[nodiscard]] std::span<const DoubleCand> aa_window(
        int i,
        int j,
        double lo,
        double hi
    ) const noexcept {
        return window(aa(i, j), lo, hi);
    }

    [[nodiscard]] std::span<const DoubleCand> bb_window(
        int i,
        int j,
        double lo,
        double hi
    ) const noexcept {
        return window(bb(i, j), lo, hi);
    }

    [[nodiscard]] std::span<const DoubleCand> ab_window(
        int i,
        int j,
        double lo,
        double hi
    ) const noexcept {
        return window(ab(i, j), lo, hi);
    }

private:
    int norb_ = 0;

    std::vector<std::size_t> aa_off_;
    std::vector<DoubleCand> aa_data_;

    std::vector<std::size_t> ab_off_;
    std::vector<DoubleCand> ab_data_;

    [[nodiscard]] std::span<const DoubleCand> aa_row(int i, int j) const noexcept {
        const std::size_t k = RHFIntegrals::pair_index(i, j);
        const std::size_t lo = aa_off_[k];
        const std::size_t hi = aa_off_[k + 1u];

        return {aa_data_.data() + lo, hi - lo};
    }

    /*
     * Rows are sorted in descending abs_h.
     *
     * first_lt(row, x) returns the first index k such that
     *
     *   row[k].abs_h < x.
     */
    [[nodiscard]] static std::size_t first_lt(
        std::span<const DoubleCand> row,
        double x
    ) noexcept {
        std::size_t lo = 0;
        std::size_t hi = row.size();

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;

            if (row[mid].abs_h < x) {
                hi = mid;
            } else {
                lo = mid + 1u;
            }
        }

        return lo;
    }

    [[nodiscard]] static std::span<const DoubleCand> prefix_ge(
        std::span<const DoubleCand> row,
        double cut
    ) noexcept {
        if (row.empty()) return {};

        const std::size_t end = first_lt(row, cut);
        return {row.data(), end};
    }

    /*
     * Return candidates satisfying:
     *
     *   lo <= abs_h < hi.
     *
     * If hi is infinity, begin becomes zero.
     */
    [[nodiscard]] static std::span<const DoubleCand> window(
        std::span<const DoubleCand> row,
        double lo,
        double hi
    ) noexcept {
        if (row.empty() || hi <= lo || hi <= 0.0) return {};

        const std::size_t begin = first_lt(row, hi);
        const std::size_t end = first_lt(row, lo);

        if (end <= begin) return {};
        return {row.data() + begin, end - begin};
    }

    static void sort_row(std::vector<DoubleCand>& row) {
        std::sort(row.begin(), row.end(), [](const DoubleCand& x, const DoubleCand& y) {
            return x.abs_h > y.abs_h;
        });
    }

    void build_aa(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;

        std::vector<DoubleCand> row;

        for (int i = 0; i < norb_; ++i) {
            for (int j = 0; j <= i; ++j) {
                const std::size_t k = RHFIntegrals::pair_index(i, j);
                aa_off_[k] = aa_data_.size();

                if (i == j) continue;

                row.clear();

                for (int p = 0; p < norb_; ++p) {
                    for (int q = p + 1; q < norb_; ++q) {
                        const double h = ints.chem(i, q, j, p) - ints.chem(i, p, j, q);
                        const double ah = std::abs(h);

                        if (ah >= eps_hb) {
                            row.push_back(DoubleCand{p, q, h, ah});
                        }
                    }
                }

                sort_row(row);
                aa_data_.insert(aa_data_.end(), row.begin(), row.end());
            }
        }

        aa_off_.back() = aa_data_.size();
    }

    void build_ab(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;

        std::vector<DoubleCand> row;

        for (int i = 0; i < norb_; ++i) {
            for (int j = 0; j < norb_; ++j) {
                const std::size_t k =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_)
                    + static_cast<std::size_t>(j);

                ab_off_[k] = ab_data_.size();

                row.clear();

                for (int p = 0; p < norb_; ++p) {
                    for (int q = 0; q < norb_; ++q) {
                        const double h = Slater::double_ab(ints, i, j, p, q);
                        const double ah = std::abs(h);

                        if (ah >= eps_hb) {
                            row.push_back(DoubleCand{p, q, h, ah});
                        }
                    }
                }

                sort_row(row);
                ab_data_.insert(ab_data_.end(), row.begin(), row.end());
            }
        }

        ab_off_.back() = ab_data_.size();
    }
};

} // namespace libdet