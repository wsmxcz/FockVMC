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
    using Backend = std::variant<rhf::Hamiltonian, libdet::guga::Hamiltonian>;

    explicit Hamiltonian(Backend backend) : backend_(std::move(backend)) {}

    template <class F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), backend_);
    }

    Backend backend_;

public:
    [[nodiscard]] static Hamiltonian make(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    ) {
        return Hamiltonian(rhf::Hamiltonian::make(h1, norb, eri, ecore));
    }

    [[nodiscard]] static Hamiltonian guga(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        int n_alpha,
        int n_beta,
        double ecore = 0.0
    ) {
        return Hamiltonian(
            libdet::guga::Hamiltonian::make(
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
        return visit([&](const auto& ham) { return ham.hij(bra, ket); });
    }

    [[nodiscard]] std::vector<double> diags(DetBatchView dets) const {
        return visit([&](const auto& ham) { return ham.diags(dets); });
    }

    [[nodiscard]] std::vector<u64> expand(
        DetBatchView kets,
        double eps,
        std::span<const double> coeffs = {},
        const DetBatchView* exclude = nullptr
    ) const {
        return visit([&](const auto& ham) {
            return ham.expand(kets, eps, coeffs, exclude);
        });
    }

    [[nodiscard]] Projection project(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps = 0.0
    ) const {
        return visit([&](const auto& ham) {
            return ham.project(bras, kets, coeffs, eps);
        });
    }

    [[nodiscard]] Projection project(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps,
        const DetBatchView* exclude
    ) const {
        return visit([&](const auto& ham) {
            return ham.project(kets, coeffs, eps, exclude);
        });
    }

    [[nodiscard]] Conns conns(
        DetBatchView kets,
        double eps,
        i64 sample = 0,
        double sample_eps = 0.0,
        u64 seed = 0
    ) const {
        return visit([&](const auto& ham) {
            return ham.conns(kets, eps, sample, sample_eps, seed);
        });
    }

    [[nodiscard]] std::pair<std::vector<double>, std::vector<i64>> degrees(
        DetBatchView kets,
        double eps
    ) const {
        return visit([&](const auto& ham) { return ham.degrees(kets, eps); });
    }

    [[nodiscard]] Matrix matrix(DetBatchView bras, DetBatchView kets) const {
        return visit([&](const auto& ham) { return ham.matrix(bras, kets); });
    }

    [[nodiscard]] std::vector<double> matvec(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x
    ) const {
        return visit([&](const auto& ham) { return ham.matvec(bras, kets, x); });
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

    [[nodiscard]] ConnSamples sample_conns(
        DetBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const {
        return visit([&](const auto& ham) {
            return ham.sample_conns(kets, counts, n_streams, eps1, eps2, seed);
        });
    }

    [[nodiscard]] ProjectSamples sample_project(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        const DetBatchView* exclude = nullptr,
        i64 n_rep = 1,
        u64 seed = 0
    ) const {
        return visit([&](const auto& ham) {
            return ham.sample_project(
                kets,
                coeffs,
                eps1,
                eps2,
                counts,
                exclude,
                n_rep,
                seed
            );
        });
    }

};

} // namespace libdet
