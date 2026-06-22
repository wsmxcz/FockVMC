#pragma once

#include <cstdint>
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

        build_tables();
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double ecore() const noexcept { return ecore_; }

    [[nodiscard]] double h1(int p, int q) const noexcept {
        return h1_[ij(p, q)];
    }

    [[nodiscard]] double hdiag(int p) const noexcept {
        return hdiag_[static_cast<std::size_t>(p)];
    }

    [[nodiscard]] double chem(int p, int q, int r, int s) const noexcept {
        const std::size_t pq = pair_[ij(p, q)];
        const std::size_t rs = pair_[ij(r, s)];
        const std::size_t hi = pq >= rs ? pq : rs;
        const std::size_t lo = pq >= rs ? rs : pq;
        return eri_[hi * (hi + 1u) / 2u + lo];
    }

    [[nodiscard]] double coulomb(int p, int q) const noexcept {
        return coulomb_[ij(p, q)];
    }

    [[nodiscard]] double exchange(int p, int q) const noexcept {
        return exchange_[ij(p, q)];
    }

    [[nodiscard]] double coulomb(int p, int q, int k) const noexcept {
        return coulomb_pair_[ijk(p, q, k)];
    }

    [[nodiscard]] double exchange(int p, int q, int k) const noexcept {
        return exchange_pair_[ijk(p, q, k)];
    }

    [[nodiscard]] static constexpr std::size_t pair_index(int p, int q) noexcept {
        const int hi = p >= q ? p : q;
        const int lo = p >= q ? q : p;
        return static_cast<std::size_t>(hi * (hi + 1) / 2 + lo);
    }

private:
    int norb_ = 0;
    std::vector<double> h1_;
    std::vector<double> eri_;
    double ecore_ = 0.0;

    std::vector<std::uint32_t> pair_;
    std::vector<double> hdiag_;
    std::vector<double> coulomb_;
    std::vector<double> exchange_;
    std::vector<double> coulomb_pair_;
    std::vector<double> exchange_pair_;

    [[nodiscard]] std::size_t ij(int p, int q) const noexcept {
        return static_cast<std::size_t>(p) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(q);
    }

    [[nodiscard]] std::size_t ijk(int p, int q, int k) const noexcept {
        const std::size_t n = static_cast<std::size_t>(norb_);
        return (static_cast<std::size_t>(p) * n + static_cast<std::size_t>(q)) * n
            + static_cast<std::size_t>(k);
    }

    [[nodiscard]] double chem_raw(int p, int q, int r, int s) const noexcept {
        const std::size_t pq = pair_index(p, q);
        const std::size_t rs = pair_index(r, s);
        const std::size_t hi = pq >= rs ? pq : rs;
        const std::size_t lo = pq >= rs ? rs : pq;
        return eri_[hi * (hi + 1u) / 2u + lo];
    }

    void build_tables() {
        const std::size_t n = static_cast<std::size_t>(norb_);

        pair_.resize(n * n);
        hdiag_.resize(n);
        coulomb_.resize(n * n);
        exchange_.resize(n * n);
        coulomb_pair_.resize(n * n * n);
        exchange_pair_.resize(n * n * n);

        for (int p = 0; p < norb_; ++p) {
            hdiag_[static_cast<std::size_t>(p)] = h1(p, p);
            for (int q = 0; q < norb_; ++q) {
                pair_[ij(p, q)] = static_cast<std::uint32_t>(pair_index(p, q));
                coulomb_[ij(p, q)] = chem_raw(p, p, q, q);
                exchange_[ij(p, q)] = chem_raw(p, q, q, p);
                for (int k = 0; k < norb_; ++k) {
                    coulomb_pair_[ijk(p, q, k)] = chem_raw(p, q, k, k);
                    exchange_pair_[ijk(p, q, k)] = chem_raw(p, k, k, q);
                }
            }
        }
    }
};

} // namespace libdet
