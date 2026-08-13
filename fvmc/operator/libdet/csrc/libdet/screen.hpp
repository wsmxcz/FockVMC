#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/integral.hpp>

namespace libdet {

struct ScreenPair {
    int a = 0;
    int b = 0;
    double h = 0.0;
};

class ScreenTable {
public:
    ScreenTable(const Integral& ints, double base_eps)
        : norb_(ints.norb()), base_eps_(base_eps) {
        if (std::isnan(base_eps_) || base_eps_ <= 0.0) {
            throw std::invalid_argument("ScreenTable: base_eps must be positive");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        const std::size_t np = n * (n + 1u) / 2u;
        same_off_.assign(np + 1u, 0u);
        mixed_off_.assign(n * n + 1u, 0u);

        build_same(ints);
        build_mixed(ints);
    }

    [[nodiscard]] double base_eps() const noexcept {
        return base_eps_;
    }

    [[nodiscard]] std::span<const ScreenPair> same_spin(
        int i,
        int j,
        double eps
    ) const noexcept {
        const std::size_t k = Integral::pair_index(i, j);
        const std::size_t begin = same_off_[k];
        const std::size_t end = same_off_[k + 1u];
        return take_prefix(
            std::span<const ScreenPair>{same_data_}.subspan(
                begin,
                end - begin
            ),
            eps
        );
    }

    [[nodiscard]] std::span<const ScreenPair> mixed_spin(
        int ia,
        int ib,
        double eps
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);
        const std::size_t begin = mixed_off_[k];
        const std::size_t end = mixed_off_[k + 1u];
        return take_prefix(
            std::span<const ScreenPair>{mixed_data_}.subspan(
                begin,
                end - begin
            ),
            eps
        );
    }

    [[nodiscard]] std::span<const ScreenPair> same_window(
        int i,
        int j,
        double eps1,
        double eps2
    ) const noexcept {
        const std::size_t k = Integral::pair_index(i, j);
        return take_window(
            same_data_,
            same_off_[k],
            same_off_[k + 1u],
            eps1,
            eps2
        );
    }

    [[nodiscard]] std::span<const ScreenPair> mixed_window(
        int ia,
        int ib,
        double eps1,
        double eps2
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);
        return take_window(
            mixed_data_,
            mixed_off_[k],
            mixed_off_[k + 1u],
            eps1,
            eps2
        );
    }

private:
    int norb_ = 0;
    double base_eps_ = 0.0;
    std::vector<std::size_t> same_off_;
    std::vector<ScreenPair> same_data_;
    std::vector<std::size_t> mixed_off_;
    std::vector<ScreenPair> mixed_data_;

    [[nodiscard]] static bool before(
        const ScreenPair& lhs,
        const ScreenPair& rhs
    ) noexcept {
        const double a = std::abs(lhs.h);
        const double b = std::abs(rhs.h);
        if (a != b) return a > b;
        if (lhs.a != rhs.a) return lhs.a < rhs.a;
        if (lhs.b != rhs.b) return lhs.b < rhs.b;
        return lhs.h < rhs.h;
    }

    [[nodiscard]] static std::size_t first_lt(
        std::span<const ScreenPair> pairs,
        double eps
    ) noexcept {
        if (eps <= 0.0) return pairs.size();
        std::size_t lo = 0;
        std::size_t hi = pairs.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;
            if (std::abs(pairs[mid].h) < eps) hi = mid;
            else lo = mid + 1u;
        }
        return lo;
    }

    [[nodiscard]] static std::span<const ScreenPair> take_prefix(
        std::span<const ScreenPair> pairs,
        double eps
    ) noexcept {
        if (pairs.empty()) return {};
        return pairs.first(first_lt(pairs, eps));
    }

    [[nodiscard]] static std::span<const ScreenPair> take_window(
        const std::vector<ScreenPair>& data,
        std::size_t first,
        std::size_t last,
        double eps1,
        double eps2
    ) noexcept {
        const auto block = std::span<const ScreenPair>{data}.subspan(
            first,
            last - first
        );
        const std::size_t begin = first_lt(block, eps1);
        const std::size_t end = first_lt(block, eps2);
        if (begin >= end) return {};
        return block.subspan(begin, end - begin);
    }

    void build_same(const Integral& ints) {
        std::vector<ScreenPair> pairs;

        for (int hi = 0; hi < norb_; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) {
                const std::size_t idx = Integral::pair_index(hi, lo);
                same_off_[idx] = same_data_.size();
                if (lo == hi) continue;

                pairs.clear();
                for (int a = 0; a < norb_; ++a) {
                    if (a == lo || a == hi) continue;
                    for (int b = a + 1; b < norb_; ++b) {
                        if (b == lo || b == hi) continue;
                        const double h = double_same(ints, lo, hi, a, b);
                        if (std::abs(h) >= base_eps_) pairs.push_back({a, b, h});
                    }
                }
                std::sort(pairs.begin(), pairs.end(), before);
                same_data_.insert(
                    same_data_.end(),
                    pairs.begin(),
                    pairs.end()
                );
            }
        }
        same_off_.back() = same_data_.size();
    }

    void build_mixed(const Integral& ints) {
        const std::size_t n = static_cast<std::size_t>(norb_);
        std::vector<ScreenPair> pairs;

        for (std::size_t idx = 0; idx < n * n; ++idx) {
            const int ia = static_cast<int>(idx / n);
            const int ib = static_cast<int>(idx % n);
            mixed_off_[idx] = mixed_data_.size();

            pairs.clear();
            for (int a = 0; a < norb_; ++a) {
                if (a == ia) continue;
                for (int b = 0; b < norb_; ++b) {
                    if (b == ib) continue;
                    const double h = double_mixed(ints, ia, ib, a, b);
                    if (std::abs(h) >= base_eps_) pairs.push_back({a, b, h});
                }
            }
            std::sort(pairs.begin(), pairs.end(), before);
            mixed_data_.insert(
                mixed_data_.end(),
                pairs.begin(),
                pairs.end()
            );
        }
        mixed_off_.back() = mixed_data_.size();
    }
};

} // namespace libdet
