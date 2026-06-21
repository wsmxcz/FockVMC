#pragma once

#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <libdet/bit.hpp>

namespace libdet {

// Public packed-word view used only at the facade boundary.
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

struct Matrix {
    std::size_t n_bra = 0;
    std::size_t n_ket = 0;
    std::vector<i32> indptr;
    std::vector<i32> indices;
    std::vector<double> data;
};

struct Conns {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> bra_words;
    std::vector<double> diag;
    std::vector<i32> ptr;
    std::vector<i32> bra;
    std::vector<double> h;
    std::vector<double> weight;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

struct Projections {
    u32 nword = 0;
    std::size_t n_streams = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

} // namespace libdet

#include <libdet/guga/hamiltonian.hpp>
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

[[nodiscard]] inline guga::PathRef as_path(StateRef state) noexcept {
    return guga::PathRef(
        state.data(),
        state.data() + static_cast<std::size_t>(state.nword()),
        state.nword()
    );
}

[[nodiscard]] inline guga::PathBatchView as_paths(StateBatchView states) noexcept {
    return guga::PathBatchView{states.data, states.n_states, states.nword};
}

} // namespace detail

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

    [[nodiscard]] double hij(StateRef bra, StateRef ket) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.hij(detail::as_det(bra), detail::as_det(ket));
            } else {
                return ham.hij(detail::as_path(bra), detail::as_path(ket));
            }
        });
    }

    [[nodiscard]] std::vector<double> diags(StateBatchView states) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.diags(detail::as_dets(states));
            } else {
                return ham.diags(detail::as_paths(states));
            }
        });
    }

    [[nodiscard]] std::vector<u64> expand(
        StateBatchView kets,
        double eps,
        std::span<const double> scale = {},
        const StateBatchView* exclude = nullptr
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                const auto ek = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
                return ham.expand(detail::as_dets(kets), eps, scale, exclude ? &ek : nullptr);
            } else {
                const auto ep = exclude ? detail::as_paths(*exclude) : guga::PathBatchView{};
                return ham.expand(detail::as_paths(kets), eps, scale, exclude ? &ep : nullptr);
            }
        });
    }

    [[nodiscard]] Projection project(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.project(detail::as_dets(bras), detail::as_dets(kets), scale, eps);
            } else {
                return ham.project(detail::as_paths(bras), detail::as_paths(kets), scale, eps);
            }
        });
    }

    [[nodiscard]] Projection project(
        StateBatchView kets,
        std::span<const double> scale,
        double eps,
        const StateBatchView* exclude
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                const auto ex = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
                return ham.project(detail::as_dets(kets), scale, eps, exclude ? &ex : nullptr);
            } else {
                const auto ex = exclude ? detail::as_paths(*exclude) : guga::PathBatchView{};
                return ham.project(detail::as_paths(kets), scale, eps, exclude ? &ex : nullptr);
            }
        });
    }

    [[nodiscard]] Conns conn(
        StateBatchView kets,
        double eps = 0.0,
        const StateBatchView* include = nullptr
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                const auto inc = include ? detail::as_dets(*include) : rhf::DetBatchView{};
                return ham.conn(detail::as_dets(kets), eps, include ? &inc : nullptr);
            } else {
                const auto inc = include ? detail::as_paths(*include) : guga::PathBatchView{};
                return ham.conn(detail::as_paths(kets), eps, include ? &inc : nullptr);
            }
        });
    }

    [[nodiscard]] Conns sample_conn(
        StateBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0,
        bool bra_weight = false,
        const StateBatchView* include = nullptr
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                const auto inc = include ? detail::as_dets(*include) : rhf::DetBatchView{};
                return ham.sample_conn(detail::as_dets(kets), counts, n_streams, eps1, eps2, seed, bra_weight, include ? &inc : nullptr);
            } else {
                const auto inc = include ? detail::as_paths(*include) : guga::PathBatchView{};
                return ham.sample_conn(detail::as_paths(kets), counts, n_streams, eps1, eps2, seed, bra_weight, include ? &inc : nullptr);
            }
        });
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
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                const auto ex = exclude ? detail::as_dets(*exclude) : rhf::DetBatchView{};
                return ham.sample_project(detail::as_dets(kets), scale, counts, n_streams, eps1, eps2, exclude ? &ex : nullptr, seed);
            } else {
                const auto ex = exclude ? detail::as_paths(*exclude) : guga::PathBatchView{};
                return ham.sample_project(detail::as_paths(kets), scale, counts, n_streams, eps1, eps2, exclude ? &ex : nullptr, seed);
            }
        });
    }

    [[nodiscard]] Matrix matrix(StateBatchView bras, StateBatchView kets) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.matrix(detail::as_dets(bras), detail::as_dets(kets));
            } else {
                return ham.matrix(detail::as_paths(bras), detail::as_paths(kets));
            }
        });
    }

    [[nodiscard]] std::vector<double> matvec(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> x
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.matvec(detail::as_dets(bras), detail::as_dets(kets), x);
            } else {
                return ham.matvec(detail::as_paths(bras), detail::as_paths(kets), x);
            }
        });
    }

    [[nodiscard]] std::vector<double> matmat(
        StateBatchView bras,
        StateBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const {
        return visit([&](const auto& ham) {
            using H = std::decay_t<decltype(ham)>;
            if constexpr (std::is_same_v<H, rhf::Hamiltonian>) {
                return ham.matmat(detail::as_dets(bras), detail::as_dets(kets), x, nrhs);
            } else {
                return ham.matmat(detail::as_paths(bras), detail::as_paths(kets), x, nrhs);
            }
        });
    }
};

} // namespace libdet
