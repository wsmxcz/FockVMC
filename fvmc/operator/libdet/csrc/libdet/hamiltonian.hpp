#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <libdet/cache.hpp>
#include <libdet/screen.hpp>

namespace libdet {

struct Matrix {
    std::size_t n_bra = 0;
    std::size_t n_ket = 0;
    std::vector<i64> indptr;
    std::vector<i32> indices;
    std::vector<double> data;
};

struct Conns {
    // bra = [kets, connection records]; record r is bra[n_kets + r].
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i64> ptr;
    std::vector<double> h;
    std::vector<double> degree;
};

struct LocalConn {
    // bra = [kets, strong records, weak records].
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i64> strong_ptr;
    std::vector<double> strong_h;
    std::vector<double> strong_degree;
    std::vector<i64> weak_ptr;
    std::vector<double> weak_coeff;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra;
    std::vector<double> hpsi;
    std::vector<double> diag;
};

struct Projections {
    u32 nword = 0;
    std::size_t n_streams = 0;
    std::vector<u64> bra;
    std::vector<double> hpsi;
    std::vector<double> diag;
};

class Hamiltonian {
public:
    Hamiltonian() = delete;
    Hamiltonian(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    );

    [[nodiscard]] double hij(DetRef bra, DetRef ket) const;
    [[nodiscard]] std::vector<double> diag(DetBatch dets) const;

    [[nodiscard]] std::vector<u64> expand(
        DetBatch kets,
        double eps,
        std::span<const double> scale = {},
        const DetBatch* exclude = nullptr
    ) const;

    [[nodiscard]] Projection project(
        DetBatch bras,
        DetBatch kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const;

    [[nodiscard]] Projection project(
        DetBatch kets,
        std::span<const double> scale,
        double eps,
        const DetBatch* exclude
    ) const;

    [[nodiscard]] Conns conn(
        DetBatch kets,
        double eps = 0.0
    ) const;

    [[nodiscard]] LocalConn local_conn(
        DetBatch kets,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        u64 seed = 0
    ) const;

    [[nodiscard]] Conns sample_conn(
        DetBatch kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const;

    [[nodiscard]] Projections sample_project(
        DetBatch kets,
        std::span<const double> scale,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        const DetBatch* exclude = nullptr,
        u64 seed = 0
    ) const;

    [[nodiscard]] Matrix matrix(DetBatch bras, DetBatch kets) const;

    [[nodiscard]] std::vector<double> matvec(
        DetBatch bras,
        DetBatch kets,
        std::span<const double> x
    ) const;

    [[nodiscard]] std::vector<double> matmat(
        DetBatch bras,
        DetBatch kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const;

private:
    Integral ints_;
    u32 nword_ = 0;

    mutable std::mutex screen_mutex_;
    mutable std::shared_ptr<const ScreenTable> screen_table_;
    mutable std::mutex conn_mutex_;
    mutable ConnCache conn_cache_;
    mutable std::mutex space_mutex_;
    mutable SpaceCache space_cache_;

    void check_dets(DetBatch dets, const char* where) const;
    static void check_eps(double eps);
    static void check_window(double eps1, double eps2);

    [[nodiscard]] std::shared_ptr<const ScreenTable> screen_table(double cutoff) const;
    [[nodiscard]] static double max_abs(std::span<const double> values) noexcept;
    [[nodiscard]] static double screen_cutoff(
        double eps,
        double max_scale
    ) noexcept;
    [[nodiscard]] std::shared_ptr<const DetSpace> cached_space(
        DetBatch kets
    ) const;

    [[nodiscard]] std::shared_ptr<const ConnSet> make_conns(
        DetRef ket,
        double eps,
        const ScreenTable* screen,
        ElementScratch& element
    ) const;

    [[nodiscard]] std::vector<std::shared_ptr<const ConnSet>> cached_conns(
        DetBatch kets,
        double eps
    ) const;

};

} // namespace libdet

#include <libdet/external.hpp>
#include <libdet/sample.hpp>
#include <libdet/internal.hpp>

namespace libdet {

inline Hamiltonian::Hamiltonian(
    std::span<const double> h1,
    int norb,
    std::span<const double> eri,
    double ecore
) : ints_(norb, h1, eri, ecore),
    nword_(bits::words_for(norb)),
    conn_cache_(nword_) {}

inline void Hamiltonian::check_dets(
    DetBatch dets,
    const char* where
) const {
    if (dets.nword != nword_) {
        throw std::invalid_argument(
            std::string(where) + ": determinant nword mismatch"
        );
    }
}

inline void Hamiltonian::check_eps(double eps) {
    if (std::isnan(eps)) throw std::invalid_argument("eps must not be NaN");
    if (eps < 0.0) throw std::invalid_argument("eps must be nonnegative");
}

inline void Hamiltonian::check_window(double eps1, double eps2) {
    check_eps(eps1);
    check_eps(eps2);
    if (eps2 > eps1) {
        throw std::invalid_argument("eps2 must be <= eps1");
    }
}

inline std::shared_ptr<const ScreenTable> Hamiltonian::screen_table(
    double cutoff
) const {
    if (cutoff <= 0.0 || !std::isfinite(cutoff)) return {};

    std::lock_guard<std::mutex> lock(screen_mutex_);
    if (!screen_table_ || cutoff < screen_table_->base_eps()) {
        screen_table_ = std::make_shared<ScreenTable>(ints_, cutoff);
    }
    return screen_table_;
}

inline double Hamiltonian::max_abs(
    std::span<const double> values
) noexcept {
    double out = 0.0;
    for (double value : values) out = std::max(out, std::abs(value));
    return out;
}

inline double Hamiltonian::screen_cutoff(
    double eps,
    double max_scale
) noexcept {
    if (eps <= 0.0) return 0.0;
    if (max_scale <= 0.0) return std::numeric_limits<double>::infinity();
    return eps / max_scale;
}

inline std::shared_ptr<const DetSpace> Hamiltonian::cached_space(
    DetBatch kets
) const {
    {
        std::lock_guard<std::mutex> lock(space_mutex_);
        if (auto space = space_cache_.find(kets)) return space;
    }

    auto fresh = std::make_shared<DetSpace>(kets);
    std::lock_guard<std::mutex> lock(space_mutex_);
    if (auto space = space_cache_.find(kets)) return space;
    space_cache_.insert(kets, fresh);
    return fresh;
}

inline double Hamiltonian::hij(DetRef bra, DetRef ket) const {
    if (bra.nword() != nword_) {
        throw std::invalid_argument("hij(bra): determinant nword mismatch");
    }
    if (ket.nword() != nword_) {
        throw std::invalid_argument("hij(ket): determinant nword mismatch");
    }

    const DetExcitation ex = excitation(ket, bra);
    if (ex.degree > 2) return 0.0;

    DetOcc occ;
    fill_occ(ket, ints_.norb(), occ);

    if (ex.degree == 0) return libdet::diag(ints_, occ);

    const Excitation& e = ex.excitation;
    switch (e.kind) {
    case ExcitationKind::alpha1:
        return ex.sign * single_alpha(ints_, occ, e.i, e.a);
    case ExcitationKind::beta1:
        return ex.sign * single_beta(ints_, occ, e.i, e.a);
    case ExcitationKind::alpha2:
    case ExcitationKind::beta2:
        return ex.sign * double_same(ints_, e.i, e.j, e.a, e.b);
    case ExcitationKind::mixed2:
        return ex.sign * double_mixed(ints_, e.i, e.j, e.a, e.b);
    }

    return 0.0;
}

} // namespace libdet
