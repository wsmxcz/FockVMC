#pragma once

#include <cstddef>
#include <span>

#include <libdet/det.hpp>
#include <libdet/integrals.hpp>

namespace libdet {

struct SlaterCondon {
    [[nodiscard]] static double sign_single(std::span<const u64> occ, int i, int a) noexcept {
        return detail::sign_single(occ, i, a);
    }

    [[nodiscard]] static double sign_double(std::span<const u64> occ, int i, int j, int a, int b) noexcept {
        return detail::sign_double(occ, i, j, a, b);
    }

    [[nodiscard]] static double single_a(const RHFIntegrals& ints, DetRef ket, int i, int a) noexcept {
        double v = ints.h1(i, a);
        bits::each_set(ket.alpha(), [&](int j) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        });
        bits::each_set(ket.beta(), [&](int j) { v += ints.chem(i, a, j, j); });
        return v;
    }

    [[nodiscard]] static double single_b(const RHFIntegrals& ints, DetRef ket, int i, int a) noexcept {
        double v = ints.h1(i, a);
        bits::each_set(ket.beta(), [&](int j) {
            if (j != i) v += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        });
        bits::each_set(ket.alpha(), [&](int j) { v += ints.chem(i, a, j, j); });
        return v;
    }

    [[nodiscard]] static double double_aa(const RHFIntegrals& ints, int i, int j, int a, int b) noexcept {
        return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
    }

    [[nodiscard]] static double double_bb(const RHFIntegrals& ints, int i, int j, int a, int b) noexcept {
        return ints.chem(i, a, j, b) - ints.chem(i, b, j, a);
    }

    [[nodiscard]] static double double_ab(const RHFIntegrals& ints, int ia, int ib, int aa, int ab) noexcept {
        return ints.chem(ia, aa, ib, ab);
    }

    [[nodiscard]] static double diagonal(const RHFIntegrals& ints, std::span<const int> oa, std::span<const int> ob) noexcept {
        double e = ints.ecore();
        for (int i : oa) e += ints.h1(i, i);
        for (int i : ob) e += ints.h1(i, i);

        for (std::size_t x = 0; x < oa.size(); ++x) {
            const int i = oa[x];
            for (std::size_t y = x + 1; y < oa.size(); ++y) {
                const int j = oa[y];
                e += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
            }
        }
        for (std::size_t x = 0; x < ob.size(); ++x) {
            const int i = ob[x];
            for (std::size_t y = x + 1; y < ob.size(); ++y) {
                const int j = ob[y];
                e += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
            }
        }
        for (int i : oa) {
            for (int j : ob) e += ints.chem(i, i, j, j);
        }
        return e;
    }

    [[nodiscard]] static double diagonal(const RHFIntegrals& ints, DetRef det) noexcept {
        const auto oa = bits::set_list(det.alpha());
        const auto ob = bits::set_list(det.beta());
        return diagonal(ints, oa, ob);
    }
};

} // namespace libdet
