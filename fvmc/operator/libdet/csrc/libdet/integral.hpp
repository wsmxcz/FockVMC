#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/det.hpp>

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

        pair_.resize(n * n);
        hdiag_.resize(n);
        coulomb_.resize(n * n);
        exchange_.resize(n * n);
        coulomb_pair_.resize(n * n * n);
        exchange_pair_.resize(n * n * n);

        for (int p = 0; p < norb_; ++p) {
            hdiag_[static_cast<std::size_t>(p)] = h1_[ij(p, p)];
            for (int q = 0; q < norb_; ++q) {
                pair_[ij(p, q)] = static_cast<std::uint32_t>(pair_index(p, q));
            }
        }

        for (int p = 0; p < norb_; ++p) {
            for (int q = 0; q < norb_; ++q) {
                coulomb_[ij(p, q)] = chem(p, p, q, q);
                exchange_[ij(p, q)] = chem(p, q, q, p);
                for (int k = 0; k < norb_; ++k) {
                    coulomb_pair_[ijk(p, q, k)] = chem(p, q, k, k);
                    exchange_pair_[ijk(p, q, k)] = chem(p, k, k, q);
                }
            }
        }
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
};

[[nodiscard]] inline double single_alpha(
    const Integral& ints,
    std::span<const int> occ_a,
    std::span<const int> occ_b,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    for (int j : occ_a) value += ints.coulomb(i, a, j) - ints.exchange(i, a, j);
    for (int j : occ_b) value += ints.coulomb(i, a, j);

    return value;
}

[[nodiscard]] inline double single_alpha(
    const Integral& ints,
    const DetOcc& occ,
    int i,
    int a
) noexcept {
    return single_alpha(ints, occ.occ_a, occ.occ_b, i, a);
}

[[nodiscard]] inline double single_beta(
    const Integral& ints,
    std::span<const int> occ_a,
    std::span<const int> occ_b,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    for (int j : occ_b) value += ints.coulomb(i, a, j) - ints.exchange(i, a, j);
    for (int j : occ_a) value += ints.coulomb(i, a, j);

    return value;
}

[[nodiscard]] inline double single_beta(
    const Integral& ints,
    const DetOcc& occ,
    int i,
    int a
) noexcept {
    return single_beta(ints, occ.occ_a, occ.occ_b, i, a);
}

[[nodiscard]] inline double double_same(
    const Integral& ints,
    int i,
    int j,
    int a,
    int b
) noexcept {
    return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
}

[[nodiscard]] inline double double_mixed(
    const Integral& ints,
    int ia,
    int ib,
    int a,
    int b
) noexcept {
    return ints.chem(ia, a, ib, b);
}

[[nodiscard]] inline double diag(
    const Integral& ints,
    std::span<const int> occ_a,
    std::span<const int> occ_b
) noexcept {
    double value = ints.ecore();

    for (int i : occ_a) value += ints.hdiag(i);
    for (int i : occ_b) value += ints.hdiag(i);

    for (std::size_t x = 0; x < occ_a.size(); ++x) {
        const int i = occ_a[x];
        for (std::size_t y = x + 1u; y < occ_a.size(); ++y) {
            const int j = occ_a[y];
            value += ints.coulomb(i, j) - ints.exchange(i, j);
        }
    }

    for (std::size_t x = 0; x < occ_b.size(); ++x) {
        const int i = occ_b[x];
        for (std::size_t y = x + 1u; y < occ_b.size(); ++y) {
            const int j = occ_b[y];
            value += ints.coulomb(i, j) - ints.exchange(i, j);
        }
    }

    for (int i : occ_a) {
        for (int j : occ_b) value += ints.coulomb(i, j);
    }

    return value;
}

[[nodiscard]] inline double diag(
    const Integral& ints,
    const DetOcc& occ
) noexcept {
    return diag(ints, occ.occ_a, occ.occ_b);
}

class ElementScratch {
public:
    explicit ElementScratch(int norb)
        : norb_(norb) {
        const std::size_t n2 = static_cast<std::size_t>(norb_) * static_cast<std::size_t>(norb_);
        alpha_.resize(n2, 0.0);
        beta_.resize(n2, 0.0);
    }

    void load(const Integral& ints, DetRef ket) {
        fill_occ(ket, norb_, occ);
        diag_ = libdet::diag(ints, occ);

        for (int p = 0; p < norb_; ++p) {
            for (int q = 0; q < norb_; ++q) {
                double alpha = ints.h1(p, q);
                double beta = alpha;

                for (int k : occ.occ_a) {
                    const double coulomb = ints.coulomb(p, q, k);
                    alpha += coulomb - ints.exchange(p, q, k);
                    beta += coulomb;
                }

                for (int k : occ.occ_b) {
                    const double coulomb = ints.coulomb(p, q, k);
                    alpha += coulomb;
                    beta += coulomb - ints.exchange(p, q, k);
                }

                const std::size_t idx =
                    static_cast<std::size_t>(p)
                    * static_cast<std::size_t>(norb_)
                    + static_cast<std::size_t>(q);
                alpha_[idx] = alpha;
                beta_[idx] = beta;
            }
        }
    }

    [[nodiscard]] double single_alpha(int i, int a) const noexcept {
        return alpha_[
            static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(a)
        ];
    }

    [[nodiscard]] double single_beta(int i, int a) const noexcept {
        return beta_[
            static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(a)
        ];
    }

    [[nodiscard]] double diag() const noexcept { return diag_; }

    DetOcc occ;

private:
    int norb_ = 0;
    double diag_ = 0.0;
    std::vector<double> alpha_;
    std::vector<double> beta_;
};

} // namespace libdet
