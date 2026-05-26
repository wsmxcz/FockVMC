#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace libdet {

/*
 * Restricted-Hartree-Fock spatial-orbital integrals.
 *
 * Chemist notation:
 *
 *   chem(i,j,k,l) = (ij|kl)
 *
 * The Hamiltonian layer uses spatial orbital determinants represented by
 * separate alpha and beta occupation strings.
 */
class RHFIntegrals {
public:
    enum class Layout : std::uint8_t {
        full,
        pair_square,
        pair_tri
    };

    RHFIntegrals() = default;

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
            throw std::invalid_argument("RHFIntegrals: norb must be nonnegative");
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
            throw std::invalid_argument("RHFIntegrals: unsupported ERI layout");
        }
    }

    [[nodiscard]] int norb() const noexcept { return norb_; }
    [[nodiscard]] double ecore() const noexcept { return ecore_; }

    [[nodiscard]] double h1(int i, int j) const noexcept {
        return h1_[
            static_cast<std::size_t>(i) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(j)
        ];
    }

    [[nodiscard]] static constexpr std::size_t pair_index(int i, int j) noexcept {
        const int a = i >= j ? i : j;
        const int b = i >= j ? j : i;
        return static_cast<std::size_t>(a * (a + 1) / 2 + b);
    }

    [[nodiscard]] double chem(int i, int j, int k, int l) const noexcept {
        switch (layout_) {
            case Layout::full: {
                const std::size_t n = static_cast<std::size_t>(norb_);

                return eri_[
                    (
                        (
                            static_cast<std::size_t>(i) * n
                            + static_cast<std::size_t>(j)
                        ) * n
                        + static_cast<std::size_t>(k)
                    ) * n
                    + static_cast<std::size_t>(l)
                ];
            }

            case Layout::pair_square: {
                const std::size_t p = pair_index(i, j);
                const std::size_t q = pair_index(k, l);
                const std::size_t np =
                    static_cast<std::size_t>(norb_)
                    * static_cast<std::size_t>(norb_ + 1)
                    / 2u;

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

} // namespace libdet