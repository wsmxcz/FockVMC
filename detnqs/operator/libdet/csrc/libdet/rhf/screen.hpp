#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/rhf/element.hpp>

namespace libdet::rhf {

struct ScreenPair {
    int a = 0;
    int b = 0;
    double h = 0.0;
};

struct ScreenWindow {
    std::span<const ScreenPair> pairs;
    std::span<const double> prefix_abs;
    double base = 0.0;

    [[nodiscard]] double weight() const noexcept {
        return prefix_abs.empty() ? 0.0 : prefix_abs.back() - base;
    }

    [[nodiscard]] const ScreenPair& draw(double target) const noexcept {
        const double value = base + target;
        auto it = std::upper_bound(prefix_abs.begin(), prefix_abs.end(), value);
        if (it == prefix_abs.end()) --it;
        return pairs[static_cast<std::size_t>(it - prefix_abs.begin())];
    }
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
        return prefix(same_pairs(i, j), eps);
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
        return prefix({mixed_data_.data() + begin, end - begin}, eps);
    }

    [[nodiscard]] ScreenWindow same_window(
        int i,
        int j,
        double eps1,
        double eps2
    ) const noexcept {
        const std::size_t k = Integral::pair_index(i, j);
        return window(
            same_data_,
            same_prefix_,
            same_off_[k],
            same_off_[k + 1u],
            eps1,
            eps2
        );
    }

    [[nodiscard]] ScreenWindow mixed_window(
        int ia,
        int ib,
        double eps1,
        double eps2
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);
        return window(
            mixed_data_,
            mixed_prefix_,
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
    std::vector<double> same_prefix_;
    std::vector<std::size_t> mixed_off_;
    std::vector<ScreenPair> mixed_data_;
    std::vector<double> mixed_prefix_;

    [[nodiscard]] std::span<const ScreenPair> same_pairs(
        int i,
        int j
    ) const noexcept {
        const std::size_t k = Integral::pair_index(i, j);
        const std::size_t begin = same_off_[k];
        const std::size_t end = same_off_[k + 1u];
        return {same_data_.data() + begin, end - begin};
    }

    [[nodiscard]] bool keep(double h) const noexcept {
        return std::abs(h) >= base_eps_;
    }

    [[nodiscard]] static double abs_h(const ScreenPair& pair) noexcept {
        return std::abs(pair.h);
    }

    [[nodiscard]] static bool before(
        const ScreenPair& lhs,
        const ScreenPair& rhs
    ) noexcept {
        const double a = abs_h(lhs);
        const double b = abs_h(rhs);
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
            if (abs_h(pairs[mid]) < eps) hi = mid;
            else lo = mid + 1u;
        }
        return lo;
    }

    [[nodiscard]] static std::span<const ScreenPair> prefix(
        std::span<const ScreenPair> pairs,
        double eps
    ) noexcept {
        if (pairs.empty()) return {};
        const std::size_t end = first_lt(pairs, eps);
        return {pairs.data(), end};
    }

    [[nodiscard]] static ScreenWindow window(
        const std::vector<ScreenPair>& data,
        const std::vector<double>& prefix_abs,
        std::size_t first,
        std::size_t last,
        double eps1,
        double eps2
    ) noexcept {
        const std::span<const ScreenPair> block{
            data.data() + first,
            last - first,
        };
        const std::size_t begin = first + first_lt(block, eps1);
        const std::size_t end = first + first_lt(block, eps2);
        if (begin >= end) return {};
        return {
            {data.data() + begin, end - begin},
            {prefix_abs.data() + begin, end - begin},
            begin == first ? 0.0 : prefix_abs[begin - 1u],
        };
    }

    void build_same(const Integral& ints) {
        const std::size_t np = same_off_.size() - 1u;
        std::vector<std::vector<ScreenPair>> blocks(np);

        for (int hi = 0; hi < norb_; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) {
                if (lo == hi) continue;
                auto& pairs = blocks[Integral::pair_index(hi, lo)];
                for (int a = 0; a < norb_; ++a) {
                    for (int b = a + 1; b < norb_; ++b) {
                        const double h = double_same(ints, lo, hi, a, b);
                        if (keep(h)) pairs.push_back({a, b, h});
                    }
                }
                std::sort(pairs.begin(), pairs.end(), before);
            }
        }

        std::size_t size = 0;
        for (const auto& block : blocks) size += block.size();
        same_data_.clear();
        same_data_.reserve(size);
        same_prefix_.clear();
        same_prefix_.reserve(size);
        for (std::size_t k = 0; k < np; ++k) {
            same_off_[k] = same_data_.size();
            double total = 0.0;
            for (const ScreenPair& pair : blocks[k]) {
                same_data_.push_back(pair);
                total += std::abs(pair.h);
                same_prefix_.push_back(total);
            }
        }
        same_off_.back() = same_data_.size();
    }

    void build_mixed(const Integral& ints) {
        const std::size_t n = static_cast<std::size_t>(norb_);
        std::vector<std::vector<ScreenPair>> blocks(n * n);

        for (std::size_t idx = 0; idx < n * n; ++idx) {
            const int ia = static_cast<int>(idx / n);
            const int ib = static_cast<int>(idx % n);
            auto& pairs = blocks[idx];
            for (int a = 0; a < norb_; ++a) {
                for (int b = 0; b < norb_; ++b) {
                    const double h = double_mixed(ints, ia, ib, a, b);
                    if (keep(h)) pairs.push_back({a, b, h});
                }
            }
            std::sort(pairs.begin(), pairs.end(), before);
        }

        std::size_t size = 0;
        for (const auto& block : blocks) size += block.size();
        mixed_data_.clear();
        mixed_data_.reserve(size);
        mixed_prefix_.clear();
        mixed_prefix_.reserve(size);
        for (std::size_t k = 0; k < blocks.size(); ++k) {
            mixed_off_[k] = mixed_data_.size();
            double total = 0.0;
            for (const ScreenPair& pair : blocks[k]) {
                mixed_data_.push_back(pair);
                total += std::abs(pair.h);
                mixed_prefix_.push_back(total);
            }
        }
        mixed_off_.back() = mixed_data_.size();
    }
};

} // namespace libdet::rhf
