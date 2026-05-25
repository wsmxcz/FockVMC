#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/det.hpp>

namespace libdet {

class RHFIntegrals {
public:
    enum class Layout : std::uint8_t { full, pair_square, pair_tri };

    RHFIntegrals() = default;

    RHFIntegrals(int norb, std::span<const double> h1, std::span<const double> eri, double ecore)
        : norb_(norb), h1_(h1.begin(), h1.end()), eri_(eri.begin(), eri.end()), ecore_(ecore) {
        if (norb_ < 0) throw std::invalid_argument("RHFIntegrals: norb must be nonnegative");
        const std::size_t n = static_cast<std::size_t>(norb_);
        if (h1_.size() != n * n) throw std::invalid_argument("RHFIntegrals: h1 size mismatch");

        const std::size_t np = n * (n + 1u) / 2u;
        if (eri_.size() == n * n * n * n) layout_ = Layout::full;
        else if (eri_.size() == np * np) layout_ = Layout::pair_square;
        else if (eri_.size() == np * (np + 1u) / 2u) layout_ = Layout::pair_tri;
        else throw std::invalid_argument("RHFIntegrals: unsupported ERI layout");
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double ecore() const noexcept { return ecore_; }

    [[nodiscard]] double h1(int i, int j) const noexcept {
        return h1_[static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_) + static_cast<std::size_t>(j)];
    }

    [[nodiscard]] static constexpr std::size_t pair_index(int i, int j) noexcept {
        const int a = i >= j ? i : j;
        const int b = i >= j ? j : i;
        return static_cast<std::size_t>(a * (a + 1) / 2 + b);
    }

    // Chemist notation: (ij|kl).
    [[nodiscard]] double chem(int i, int j, int k, int l) const noexcept {
        switch (layout_) {
            case Layout::full: {
                const std::size_t n = static_cast<std::size_t>(norb_);
                return eri_[((static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)) * n + static_cast<std::size_t>(k)) * n + static_cast<std::size_t>(l)];
            }
            case Layout::pair_square: {
                const std::size_t p = pair_index(i, j);
                const std::size_t q = pair_index(k, l);
                const std::size_t np = static_cast<std::size_t>(norb_) * static_cast<std::size_t>(norb_ + 1) / 2u;
                return eri_[p * np + q];
            }
            case Layout::pair_tri: {
                const std::size_t p = pair_index(i, j);
                const std::size_t q = pair_index(k, l);
                const std::size_t hi = p >= q ? p : q;
                const std::size_t lo = p >= q ? q : p;
                return eri_[hi * (hi + 1u) / 2u + lo];
            }
        }
        return 0.0;
    }

private:
    int norb_ = 0;
    std::vector<double> h1_;
    std::vector<double> eri_;
    double ecore_ = 0.0;
    Layout layout_ = Layout::full;
};

struct DoubleCandidate {
    int p = 0;
    int q = 0;
    double h = 0.0;
};

class HeatBathTable {
public:
    explicit HeatBathTable(const RHFIntegrals& ints, double eps_table)
        : norb_(ints.norb()), eps_table_(eps_table) {
        if (eps_table_ < 0.0) throw std::invalid_argument("HeatBathTable: eps_table must be nonnegative");
        const std::size_t n = static_cast<std::size_t>(std::max(0, norb_));
        const std::size_t np = n * (n + 1u) / 2u;
        aa_off_.assign(np + 1u, 0u);
        ab_off_.assign(n * n + 1u, 0u);
        build_aa(ints);
        build_ab(ints);
    }

    [[nodiscard]] double eps_table() const noexcept { return eps_table_; }

    [[nodiscard]] std::span<const DoubleCandidate> aa(int i, int j) const noexcept { return aa_row(i, j); }
    [[nodiscard]] std::span<const DoubleCandidate> bb(int i, int j) const noexcept { return aa_row(i, j); }

    [[nodiscard]] std::span<const DoubleCandidate> ab(int i, int j) const noexcept {
        const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_) + static_cast<std::size_t>(j);
        const std::size_t lo = ab_off_[k];
        const std::size_t hi = ab_off_[k + 1u];
        return {ab_data_.data() + lo, hi - lo};
    }

private:
    int norb_ = 0;
    double eps_table_ = 0.0;
    std::vector<std::size_t> aa_off_;
    std::vector<DoubleCandidate> aa_data_;
    std::vector<std::size_t> ab_off_;
    std::vector<DoubleCandidate> ab_data_;

    [[nodiscard]] std::span<const DoubleCandidate> aa_row(int i, int j) const noexcept {
        const std::size_t k = RHFIntegrals::pair_index(i, j);
        const std::size_t lo = aa_off_[k];
        const std::size_t hi = aa_off_[k + 1u];
        return {aa_data_.data() + lo, hi - lo};
    }

    void build_aa(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;
        for (int i = 0; i < norb_; ++i) {
            for (int j = 0; j <= i; ++j) {
                const std::size_t k = RHFIntegrals::pair_index(i, j);
                aa_off_[k] = aa_data_.size();
                if (i == j) continue;

                std::vector<DoubleCandidate> row;
                for (int a = 0; a < norb_; ++a) {
                    for (int b = 0; b < a; ++b) {
                        const double h = ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
                        if (std::abs(h) >= eps_table_) row.push_back(DoubleCandidate{b, a, h});
                    }
                }
                std::sort(row.begin(), row.end(), [](const DoubleCandidate& x, const DoubleCandidate& y) {
                    return std::abs(x.h) > std::abs(y.h);
                });
                aa_data_.insert(aa_data_.end(), row.begin(), row.end());
            }
        }
        aa_off_.back() = aa_data_.size();
    }

    void build_ab(const RHFIntegrals& ints) {
        if (norb_ <= 0) return;
        for (int i = 0; i < norb_; ++i) {
            for (int j = 0; j < norb_; ++j) {
                const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_) + static_cast<std::size_t>(j);
                ab_off_[k] = ab_data_.size();

                std::vector<DoubleCandidate> row;
                for (int a = 0; a < norb_; ++a) {
                    for (int b = 0; b < norb_; ++b) {
                        const double h = ints.chem(i, a, j, b);
                        if (std::abs(h) >= eps_table_) row.push_back(DoubleCandidate{a, b, h});
                    }
                }
                std::sort(row.begin(), row.end(), [](const DoubleCandidate& x, const DoubleCandidate& y) {
                    return std::abs(x.h) > std::abs(y.h);
                });
                ab_data_.insert(ab_data_.end(), row.begin(), row.end());
            }
        }
        ab_off_.back() = ab_data_.size();
    }
};

} // namespace libdet
