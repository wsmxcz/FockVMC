#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/det.hpp>

namespace libdet {

class RHFIntegrals {
public:
    enum class Layout : std::uint8_t {
        full,
        pair_square,
        pair_tri
    };

    RHFIntegrals(
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
            throw std::invalid_argument(
                "RHFIntegrals: norb must be nonnegative"
            );
        }

        const std::size_t n = static_cast<std::size_t>(norb_);

        if (h1_.size() != n * n) {
            throw std::invalid_argument("RHFIntegrals: h1 size mismatch");
        }

        const std::size_t np = n * (n + 1u) / 2u;

        if (eri_.size() == n * n * n * n) {
            layout_ = Layout::full;
        } else if (eri_.size() == np * np) {
            layout_ = Layout::pair_square;
        } else if (eri_.size() == np * (np + 1u) / 2u) {
            layout_ = Layout::pair_tri;
        } else {
            throw std::invalid_argument(
                "RHFIntegrals: unsupported ERI layout"
            );
        }
    }

    [[nodiscard]] int norb() const noexcept {
        return norb_;
    }

    [[nodiscard]] double ecore() const noexcept {
        return ecore_;
    }

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
        switch (layout_) {
        case Layout::full: {
            const std::size_t n = static_cast<std::size_t>(norb_);

            return eri_[
                (
                    (
                        static_cast<std::size_t>(p) * n
                        + static_cast<std::size_t>(q)
                    ) * n
                    + static_cast<std::size_t>(r)
                ) * n
                + static_cast<std::size_t>(s)
            ];
        }
        case Layout::pair_square: {
            const std::size_t pq = pair_index(p, q);
            const std::size_t rs = pair_index(r, s);
            const std::size_t np =
                static_cast<std::size_t>(norb_)
                * static_cast<std::size_t>(norb_ + 1)
                / 2u;

            return eri_[pq * np + rs];
        }
        case Layout::pair_tri: {
            const std::size_t pq = pair_index(p, q);
            const std::size_t rs = pair_index(r, s);
            const std::size_t hi = pq >= rs ? pq : rs;
            const std::size_t lo = pq >= rs ? rs : pq;

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

/*
 * Slater-Condon matrix elements for an RHF spatial-orbital Hamiltonian.
 *
 * Excitation signs are supplied by determinant bit-string ordering. Integral
 * factors use the same convention in exact and heat-bath paths.
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

    /*
     * Beta single excitation i -> a.
     */
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

    /*
     * Same-spin double excitation i,j -> a,b.
     */
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

    /*
     * Opposite-spin double excitation ia,ib -> a,b.
     */
    [[nodiscard]] static double double_ab(
        const RHFIntegrals& ints,
        int ia,
        int ib,
        int a,
        int b
    ) noexcept {
        return ints.chem(ia, a, ib, b);
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
