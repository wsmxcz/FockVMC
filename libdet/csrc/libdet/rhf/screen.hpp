#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/rhf/element.hpp>

namespace libdet::rhf {

struct BraPair {
    int a = 0;
    int b = 0;
    double h = 0.0;
};

class Screen {
public:
    Screen(const Integral& ints, double cutoff)
        : norb_(ints.norb()), cutoff_(cutoff) {
        if (std::isnan(cutoff_) || cutoff_ <= 0.0) {
            throw std::invalid_argument("Screen: cutoff must be positive");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        const std::size_t np = n * (n + 1u) / 2u;

        same_off_.assign(np + 1u, 0u);
        opposite_off_.assign(n * n + 1u, 0u);

        build_same(ints);
        build_opposite(ints);
    }

    [[nodiscard]] double cutoff() const noexcept {
        return cutoff_;
    }

    [[nodiscard]] std::span<const BraPair> same(
        int i,
        int j,
        double lo = 0.0,
        double hi = std::numeric_limits<double>::infinity()
    ) const noexcept {
        return window(same_pairs(i, j), lo, hi);
    }

    [[nodiscard]] std::span<const BraPair> opposite(
        int ia,
        int ib,
        double lo = 0.0,
        double hi = std::numeric_limits<double>::infinity()
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);
        const std::size_t begin = opposite_off_[k];
        const std::size_t end = opposite_off_[k + 1u];

        return window({opposite_data_.data() + begin, end - begin}, lo, hi);
    }

private:
    int norb_ = 0;
    double cutoff_ = 0.0;
    std::vector<std::size_t> same_off_;
    std::vector<BraPair> same_data_;
    std::vector<std::size_t> opposite_off_;
    std::vector<BraPair> opposite_data_;

    [[nodiscard]] std::span<const BraPair> same_pairs(
        int i,
        int j
    ) const noexcept {
        const std::size_t k = Integral::pair_index(i, j);
        const std::size_t lo = same_off_[k];
        const std::size_t hi = same_off_[k + 1u];
        return {same_data_.data() + lo, hi - lo};
    }

    [[nodiscard]] bool keep(double h) const noexcept {
        return std::abs(h) >= cutoff_;
    }

    [[nodiscard]] static double abs_h(const BraPair& pair) noexcept {
        return std::abs(pair.h);
    }

    [[nodiscard]] static bool before(
        const BraPair& lhs,
        const BraPair& rhs
    ) noexcept {
        const double a = abs_h(lhs);
        const double b = abs_h(rhs);

        if (a != b) return a > b;
        if (lhs.a != rhs.a) return lhs.a < rhs.a;
        if (lhs.b != rhs.b) return lhs.b < rhs.b;
        return lhs.h < rhs.h;
    }

    [[nodiscard]] static std::size_t first_lt(
        std::span<const BraPair> pairs,
        double cutoff
    ) noexcept {
        if (cutoff <= 0.0) return pairs.size();

        std::size_t lo = 0;
        std::size_t hi = pairs.size();

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;
            if (abs_h(pairs[mid]) < cutoff) hi = mid;
            else lo = mid + 1u;
        }

        return lo;
    }

    [[nodiscard]] static std::span<const BraPair> window(
        std::span<const BraPair> pairs,
        double lo,
        double hi
    ) noexcept {
        if (pairs.empty() || hi <= lo) return {};

        const std::size_t begin = std::isfinite(hi) ? first_lt(pairs, hi) : 0u;
        const std::size_t end = first_lt(pairs, lo);

        if (end <= begin) return {};
        return {pairs.data() + begin, end - begin};
    }

    void build_same(const Integral& ints) {
        std::vector<BraPair> pairs;

        for (int hi = 0; hi < norb_; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) {
                const std::size_t pair = Integral::pair_index(hi, lo);
                same_off_[pair] = same_data_.size();
                if (hi == lo) continue;

                pairs.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = a + 1; b < norb_; ++b) {
                        const double h = double_alpha(ints, lo, hi, a, b);
                        if (keep(h)) pairs.push_back({a, b, h});
                    }
                }

                std::sort(pairs.begin(), pairs.end(), before);
                same_data_.insert(same_data_.end(), pairs.begin(), pairs.end());
            }
        }

        same_off_.back() = same_data_.size();
    }

    void build_opposite(const Integral& ints) {
        std::vector<BraPair> pairs;

        for (int ia = 0; ia < norb_; ++ia) {
            for (int ib = 0; ib < norb_; ++ib) {
                const std::size_t pair =
                    static_cast<std::size_t>(ia)
                    * static_cast<std::size_t>(norb_)
                    + static_cast<std::size_t>(ib);
                opposite_off_[pair] = opposite_data_.size();
                pairs.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = 0; b < norb_; ++b) {
                        const double h = double_mixed(ints, ia, ib, a, b);
                        if (keep(h)) pairs.push_back({a, b, h});
                    }
                }

                std::sort(pairs.begin(), pairs.end(), before);
                opposite_data_.insert(
                    opposite_data_.end(),
                    pairs.begin(),
                    pairs.end()
                );
            }
        }

        opposite_off_.back() = opposite_data_.size();
    }
};

} // namespace libdet::rhf
