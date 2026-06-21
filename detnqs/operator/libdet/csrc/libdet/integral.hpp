#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace libdet {

// Spatial-orbital integrals in PySCF chemist s8 notation.
class Integral {
public:
    Integral(
        int norb,
        std::span<const double> h1,
        std::span<const double> eri,
        double ecore
    )
        : norb_(norb),
          h1_(h1.begin(), h1.end()),
          eri_(eri.begin(), eri.end()),
          ecore_(ecore) {
        if (norb_ < 0) {
            throw std::invalid_argument("Integral: norb must be nonnegative");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        if (h1_.size() != n * n) {
            throw std::invalid_argument("Integral: h1 size mismatch");
        }

        const std::size_t np = n * (n + 1u) / 2u;
        if (eri_.size() != np * (np + 1u) / 2u) {
            throw std::invalid_argument("Integral: eri must be PySCF s8 packed");
        }
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double ecore() const noexcept { return ecore_; }

    [[nodiscard]] double h1(int p, int q) const noexcept {
        return h1_[
            static_cast<std::size_t>(p) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(q)
        ];
    }

    [[nodiscard]] static constexpr std::size_t pair_index(
        int p,
        int q
    ) noexcept {
        const int hi = p >= q ? p : q;
        const int lo = p >= q ? q : p;
        return static_cast<std::size_t>(hi * (hi + 1) / 2 + lo);
    }

    [[nodiscard]] double chem(int p, int q, int r, int s) const noexcept {
        const std::size_t pq = pair_index(p, q);
        const std::size_t rs = pair_index(r, s);
        const std::size_t hi = pq >= rs ? pq : rs;
        const std::size_t lo = pq >= rs ? rs : pq;
        return eri_[hi * (hi + 1u) / 2u + lo];
    }

private:
    int norb_ = 0;
    std::vector<double> h1_;
    std::vector<double> eri_;
    double ecore_ = 0.0;
};

} // namespace libdet
