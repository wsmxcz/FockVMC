#pragma once

#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>
#include <libdet/rhf/hamiltonian.hpp>

namespace libdet {

class Hamiltonian {
private:
    using Backend = std::variant<rhf::Hamiltonian, guga::Hamiltonian>;

    explicit Hamiltonian(Backend backend) : backend_(std::move(backend)) {}

    template <class F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), backend_);
    }

    Backend backend_;

public:
    [[nodiscard]] static Hamiltonian det(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    ) {
        return Hamiltonian(rhf::Hamiltonian::make(h1, norb, eri, ecore));
    }

    [[nodiscard]] static Hamiltonian spin(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        int n_alpha,
        int n_beta,
        double ecore = 0.0
    ) {
        return Hamiltonian(
            guga::Hamiltonian::make(
                h1,
                norb,
                eri,
                n_alpha,
                n_beta,
                ecore
            )
        );
    }

    [[nodiscard]] int norb() const {
        return visit([](const auto& ham) { return ham.norb(); });
    }

    [[nodiscard]] u32 nword() const {
        return visit([](const auto& ham) { return ham.nword(); });
    }

    [[nodiscard]] double hij(DetRef bra, DetRef ket) const {
        return visit([&](const auto& ham) {
            return ham.hij(bra, ket);
        });
    }

    [[nodiscard]] std::vector<double> diags(DetBatchView dets) const {
        return visit([&](const auto& ham) {
            return ham.diags(dets);
        });
    }

    [[nodiscard]] std::vector<u64> expand(
        DetBatchView kets,
        double eps,
        std::span<const double> scale = {},
        const DetBatchView* exclude = nullptr
    ) const {
        return visit([&](const auto& ham) {
            return ham.expand(kets, eps, scale, exclude);
        });
    }

    [[nodiscard]] Projection project(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const {
        return visit([&](const auto& ham) {
            return ham.project(bras, kets, scale, eps);
        });
    }

    [[nodiscard]] Projection project(
        DetBatchView kets,
        std::span<const double> scale,
        double eps,
        const DetBatchView* exclude
    ) const {
        return visit([&](const auto& ham) {
            return ham.project(kets, scale, eps, exclude);
        });
    }

    [[nodiscard]] Conns conn(
        DetBatchView kets,
        double eps = 0.0,
        const DetBatchView* include = nullptr
    ) const {
        return visit([&](const auto& ham) {
            return ham.conn(kets, eps, include);
        });
    }

    [[nodiscard]] Conns sample_conn(
        DetBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0,
        bool bra_weight = false,
        const DetBatchView* include = nullptr
    ) const {
        return visit([&](const auto& ham) {
            return ham.sample_conn(
                kets,
                counts,
                n_streams,
                eps1,
                eps2,
                seed,
                bra_weight,
                include
            );
        });
    }

    [[nodiscard]] Projections sample_project(
        DetBatchView kets,
        std::span<const double> scale,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        const DetBatchView* exclude = nullptr,
        u64 seed = 0
    ) const {
        return visit([&](const auto& ham) {
            return ham.sample_project(
                kets,
                scale,
                counts,
                n_streams,
                eps1,
                eps2,
                exclude,
                seed
            );
        });
    }

    [[nodiscard]] Matrix matrix(DetBatchView bras, DetBatchView kets) const {
        return visit([&](const auto& ham) {
            return ham.matrix(bras, kets);
        });
    }

    [[nodiscard]] std::vector<double> matvec(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x
    ) const {
        return visit([&](const auto& ham) {
            return ham.matvec(bras, kets, x);
        });
    }

    [[nodiscard]] std::vector<double> matmat(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const {
        return visit([&](const auto& ham) {
            return ham.matmat(bras, kets, x, nrhs);
        });
    }
};

} // namespace libdet
