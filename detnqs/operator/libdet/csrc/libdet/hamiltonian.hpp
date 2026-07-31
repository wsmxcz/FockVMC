#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <libdet/bit.hpp>
#include <libdet/results.hpp>

namespace libdet {

// Public packed-word view used at the facade boundary.
class StateRef {
public:
    constexpr StateRef() noexcept = default;
    constexpr StateRef(const u64* words, u32 nword) noexcept
        : words_(words), nword_(nword) {}

    [[nodiscard]] constexpr const u64* data() const noexcept { return words_; }
    [[nodiscard]] constexpr u32 nword() const noexcept { return nword_; }

private:
    const u64* words_ = nullptr;
    u32 nword_ = 0;
};

struct StateBatchView {
    const u64* data = nullptr;
    std::size_t n_states = 0;
    u32 nword = 0;

    [[nodiscard]] StateRef operator[](std::size_t idx) const noexcept {
        return StateRef(data + idx * word_pair_size(nword), nword);
    }
};

} // namespace libdet

#include <libdet/rhf/hamiltonian.hpp>

namespace libdet {

namespace detail {

[[nodiscard]] inline rhf::DetRef as_det(StateRef state) noexcept {
    return rhf::DetRef(
        state.data(),
        state.data() + static_cast<std::size_t>(state.nword()),
        state.nword()
    );
}

[[nodiscard]] inline rhf::DetBatchView as_dets(StateBatchView states) noexcept {
    return rhf::DetBatchView{states.data, states.n_states, states.nword};
}

} // namespace detail

class Hamiltonian {
private:
    explicit Hamiltonian(rhf::Hamiltonian backend) : backend_(std::move(backend)) {}

    rhf::Hamiltonian backend_;

public:
    [[nodiscard]] static Hamiltonian det(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    ) {
        return Hamiltonian(rhf::Hamiltonian::make(h1, norb, eri, ecore));
    }

    [[nodiscard]] int norb() const {
        return backend_.norb();
    }

    [[nodiscard]] u32 nword() const {
        return backend_.nword();
    }

    [[nodiscard]] double hij(StateRef bra, StateRef ket) const {
        return backend_.hij(detail::as_det(bra), detail::as_det(ket));
    }

    [[nodiscard]] std::vector<double> diags(StateBatchView states) const {
        return backend_.diags(detail::as_dets(states));
    }

    [[nodiscard]] std::vector<u64> expand(
        StateBatchView kets,
        double eps,
        std::span<const double> scale = {},
        const StateBatchView* exclude = nullptr
    ) const {
        const auto ex = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
        return backend_.expand(
            detail::as_dets(kets),
            eps,
            scale,
            exclude ? &ex : nullptr
        );
    }

    [[nodiscard]] Projection project(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const {
        return backend_.project(
            detail::as_dets(bras),
            detail::as_dets(kets),
            scale,
            eps
        );
    }

    [[nodiscard]] Projection project(
        StateBatchView kets,
        std::span<const double> scale,
        double eps,
        const StateBatchView* exclude
    ) const {
        const auto ex = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
        return backend_.project(
            detail::as_dets(kets),
            scale,
            eps,
            exclude ? &ex : nullptr
        );
    }

    [[nodiscard]] Conns conn(
        StateBatchView kets,
        double eps = 0.0
    ) const {
        return backend_.conn(detail::as_dets(kets), eps);
    }

    [[nodiscard]] Conns sample_conn(
        StateBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const {
        return backend_.sample_conn(
            detail::as_dets(kets),
            counts,
            n_streams,
            eps1,
            eps2,
            seed
        );
    }

    [[nodiscard]] LocalConn local_conn(
        StateBatchView kets,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        u64 seed = 0
    ) const {
        return backend_.local_conn(
            detail::as_dets(kets),
            eps1,
            eps2,
            counts,
            seed
        );
    }

    [[nodiscard]] Projections sample_project(
        StateBatchView kets,
        std::span<const double> scale,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        const StateBatchView* exclude = nullptr,
        u64 seed = 0
    ) const {
        const auto ex = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
        return backend_.sample_project(
            detail::as_dets(kets),
            scale,
            counts,
            n_streams,
            eps1,
            eps2,
            exclude ? &ex : nullptr,
            seed
        );
    }

    [[nodiscard]] Matrix matrix(StateBatchView bras, StateBatchView kets) const {
        return backend_.matrix(detail::as_dets(bras), detail::as_dets(kets));
    }

    [[nodiscard]] std::vector<double> matvec(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> x
    ) const {
        return backend_.matvec(
            detail::as_dets(bras),
            detail::as_dets(kets),
            x
        );
    }

    [[nodiscard]] std::vector<double> matmat(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const {
        return backend_.matmat(
            detail::as_dets(bras),
            detail::as_dets(kets),
            x,
            nrhs
        );
    }
};

} // namespace libdet
