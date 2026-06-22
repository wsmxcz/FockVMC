#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <libdet/integral.hpp>
#include <libdet/rhf/det.hpp>

namespace libdet::rhf {

// Slater-Condon elements for an RHF spatial-orbital Hamiltonian.
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
        : norb_(norb), occ(norb) {
        const std::size_t n2 = static_cast<std::size_t>(norb_) * static_cast<std::size_t>(norb_);
        alpha_single.resize(n2, 0.0);
        beta_single.resize(n2, 0.0);
    }

    void load(const Integral& ints, DetRef ket) {
        fill_occ(ket, norb_, occ);
        diag_value = rhf::diag(ints, occ);
        build_single(ints);
    }

    [[nodiscard]] double single_alpha(int i, int a) const noexcept {
        return alpha_single[index(i, a)];
    }

    [[nodiscard]] double single_beta(int i, int a) const noexcept {
        return beta_single[index(i, a)];
    }

    [[nodiscard]] double diag() const noexcept { return diag_value; }

    DetOcc occ;

private:
    int norb_ = 0;
    double diag_value = 0.0;
    std::vector<double> alpha_single;
    std::vector<double> beta_single;

    [[nodiscard]] std::size_t index(int p, int q) const noexcept {
        return static_cast<std::size_t>(p) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(q);
    }

    void build_single(const Integral& ints) noexcept {
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

                const std::size_t idx = index(p, q);
                alpha_single[idx] = alpha;
                beta_single[idx] = beta;
            }
        }
    }
};

} // namespace libdet::rhf
