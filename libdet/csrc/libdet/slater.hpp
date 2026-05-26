#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <libdet/det.hpp>
#include <libdet/integrals.hpp>

namespace libdet {

/*
 * Slater-Condon matrix elements for an RHF spatial-orbital Hamiltonian.
 *
 * The sign is supplied by determinant bit-string ordering. The integral
 * factors use the same convention in both exact and heat-bath paths.
 */
struct Slater {
    [[nodiscard]] static double sign_single(
        std::span<const u64> occ,
        int i,
        int a
    ) noexcept {
        return detail::sign_single(occ, i, a);
    }

    [[nodiscard]] static double sign_single(
        std::span<const int> prefix,
        int i,
        int a
    ) noexcept {
        return detail::sign_single(prefix, i, a);
    }

    [[nodiscard]] static double sign_double(
        std::span<const u64> occ,
        int i,
        int j,
        int a,
        int b
    ) noexcept {
        return detail::sign_double(occ, i, j, a, b);
    }

    [[nodiscard]] static double sign_double(
        std::span<const int> prefix,
        int i,
        int j,
        int a,
        int b
    ) noexcept {
        return detail::sign_double(prefix, i, j, a, b);
    }

    /*
     * Alpha single excitation i -> a.
     *
     * Matrix element:
     *
     *   h_ia + sum_j^alpha [(ia|jj) - (ij|ja)]
     *        + sum_j^beta  [(ia|jj)]
     */
    [[nodiscard]] static double single_a(
        const RHFIntegrals& ints,
        std::span<const int> occ_a,
        std::span<const int> occ_b,
        int i,
        int a
    ) noexcept {
        double v = ints.h1(i, a);

        for (int j : occ_a) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        }

        for (int j : occ_b) {
            v += ints.chem(i, a, j, j);
        }

        return v;
    }

    [[nodiscard]] static double single_a(
        const RHFIntegrals& ints,
        DetRef ket,
        int i,
        int a
    ) noexcept {
        double v = ints.h1(i, a);

        bits::each_set(ket.alpha(), [&](int j) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        });

        bits::each_set(ket.beta(), [&](int j) {
            v += ints.chem(i, a, j, j);
        });

        return v;
    }

    [[nodiscard]] static double single_b(
        const RHFIntegrals& ints,
        std::span<const int> occ_a,
        std::span<const int> occ_b,
        int i,
        int a
    ) noexcept {
        double v = ints.h1(i, a);

        for (int j : occ_b) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        }

        for (int j : occ_a) {
            v += ints.chem(i, a, j, j);
        }

        return v;
    }

    [[nodiscard]] static double single_b(
        const RHFIntegrals& ints,
        DetRef ket,
        int i,
        int a
    ) noexcept {
        double v = ints.h1(i, a);

        bits::each_set(ket.beta(), [&](int j) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        });

        bits::each_set(ket.alpha(), [&](int j) {
            v += ints.chem(i, a, j, j);
        });

        return v;
    }

    [[nodiscard]] static double double_aa(
        const RHFIntegrals& ints,
        int i,
        int j,
        int a,
        int b
    ) noexcept {
        return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
    }

    [[nodiscard]] static double double_bb(
        const RHFIntegrals& ints,
        int i,
        int j,
        int a,
        int b
    ) noexcept {
        return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
    }

    [[nodiscard]] static double double_ab(
        const RHFIntegrals& ints,
        int ia,
        int ib,
        int aa,
        int ab
    ) noexcept {
        return ints.chem(ia, aa, ib, ab);
    }

    [[nodiscard]] static double diag(
        const RHFIntegrals& ints,
        std::span<const int> occ_a,
        std::span<const int> occ_b
    ) noexcept {
        double e = ints.ecore();

        for (int i : occ_a) e += ints.h1(i, i);
        for (int i : occ_b) e += ints.h1(i, i);

        for (std::size_t x = 0; x < occ_a.size(); ++x) {
            const int i = occ_a[x];

            for (std::size_t y = x + 1u; y < occ_a.size(); ++y) {
                const int j = occ_a[y];
                e += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
            }
        }

        for (std::size_t x = 0; x < occ_b.size(); ++x) {
            const int i = occ_b[x];

            for (std::size_t y = x + 1u; y < occ_b.size(); ++y) {
                const int j = occ_b[y];
                e += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
            }
        }

        for (int i : occ_a) {
            for (int j : occ_b) {
                e += ints.chem(i, i, j, j);
            }
        }

        return e;
    }

    [[nodiscard]] static double diag(const RHFIntegrals& ints, const DetOcc& occ) noexcept {
        return diag(ints, occ.occ_a, occ.occ_b);
    }

    [[nodiscard]] static double diag(const RHFIntegrals& ints, DetRef det) {
        const std::vector<int> occ_a = bits::set_list(det.alpha());
        const std::vector<int> occ_b = bits::set_list(det.beta());
        return diag(ints, occ_a, occ_b);
    }
};

} // namespace libdet