#pragma once

#include <span>
#include <vector>

#include <libdet/integral.hpp>
#include <libdet/rhf/det.hpp>

namespace libdet::rhf {

// Slater-Condon elements for an RHF spatial-orbital Hamiltonian.
[[nodiscard]] inline double sign_single(
    std::span<const u64> occ,
    int i,
    int a
) noexcept {
    return detail::sign_single(occ, i, a);
}

[[nodiscard]] inline double sign_single(
    std::span<const int> prefix,
    int i,
    int a
) noexcept {
    return detail::sign_single(prefix, i, a);
}

[[nodiscard]] inline double sign_double(
    std::span<const u64> occ,
    int i,
    int j,
    int a,
    int b
) noexcept {
    return detail::sign_double(occ, i, j, a, b);
}

[[nodiscard]] inline double sign_double(
    std::span<const int> prefix,
    int i,
    int j,
    int a,
    int b
) noexcept {
    return detail::sign_double(prefix, i, j, a, b);
}

[[nodiscard]] inline double single_alpha(
    const Integral& ints,
    std::span<const int> occ_a,
    std::span<const int> occ_b,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    for (int j : occ_a) {
        if (j != i) value += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
    }
    for (int j : occ_b) value += ints.chem(i, a, j, j);

    return value;
}

[[nodiscard]] inline double single_alpha(
    const Integral& ints,
    DetRef ket,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    bits::each_set(ket.alpha(), [&](int j) {
        if (j != i) value += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
    });
    bits::each_set(ket.beta(), [&](int j) {
        value += ints.chem(i, a, j, j);
    });

    return value;
}

[[nodiscard]] inline double single_beta(
    const Integral& ints,
    std::span<const int> occ_a,
    std::span<const int> occ_b,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    for (int j : occ_b) {
        if (j != i) value += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
    }
    for (int j : occ_a) value += ints.chem(i, a, j, j);

    return value;
}

[[nodiscard]] inline double single_beta(
    const Integral& ints,
    DetRef ket,
    int i,
    int a
) noexcept {
    double value = ints.h1(i, a);

    bits::each_set(ket.beta(), [&](int j) {
        if (j != i) value += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
    });
    bits::each_set(ket.alpha(), [&](int j) {
        value += ints.chem(i, a, j, j);
    });

    return value;
}

[[nodiscard]] inline double double_alpha(
    const Integral& ints,
    int i,
    int j,
    int a,
    int b
) noexcept {
    return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
}

[[nodiscard]] inline double double_beta(
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

    for (int i : occ_a) value += ints.h1(i, i);
    for (int i : occ_b) value += ints.h1(i, i);

    for (std::size_t x = 0; x < occ_a.size(); ++x) {
        const int i = occ_a[x];
        for (std::size_t y = x + 1u; y < occ_a.size(); ++y) {
            const int j = occ_a[y];
            value += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
        }
    }

    for (std::size_t x = 0; x < occ_b.size(); ++x) {
        const int i = occ_b[x];
        for (std::size_t y = x + 1u; y < occ_b.size(); ++y) {
            const int j = occ_b[y];
            value += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
        }
    }

    for (int i : occ_a) {
        for (int j : occ_b) value += ints.chem(i, i, j, j);
    }

    return value;
}

[[nodiscard]] inline double diag(
    const Integral& ints,
    const DetOcc& occ
) noexcept {
    return diag(ints, occ.occ_a, occ.occ_b);
}

[[nodiscard]] inline double diag(const Integral& ints, DetRef det) {
    const std::vector<int> occ_a = bits::set_list(det.alpha());
    const std::vector<int> occ_b = bits::set_list(det.beta());
    return diag(ints, occ_a, occ_b);
}

} // namespace libdet::rhf
