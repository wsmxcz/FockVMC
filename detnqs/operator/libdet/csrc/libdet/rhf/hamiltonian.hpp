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
#include <utility>
#include <vector>

#include <libdet/rhf/cache.hpp>
#include <libdet/rhf/screen.hpp>

namespace libdet::rhf {

struct KetScratch {
    explicit KetScratch(int norb) : occ(norb) {}

    DetOcc occ;
};

class Hamiltonian {
public:
    Hamiltonian() = delete;
    Hamiltonian(const Hamiltonian& other);
    Hamiltonian& operator=(const Hamiltonian& other);
    Hamiltonian(Hamiltonian&& other) noexcept;
    Hamiltonian& operator=(Hamiltonian&& other) noexcept;

    [[nodiscard]] static Hamiltonian make(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    );

    [[nodiscard]] int norb() const noexcept;
    [[nodiscard]] u32 nword() const noexcept;

    [[nodiscard]] double hij(DetRef bra, DetRef ket) const;
    [[nodiscard]] std::vector<double> diags(DetBatchView dets) const;

    [[nodiscard]] std::vector<u64> expand(
        DetBatchView kets,
        double eps,
        std::span<const double> scale = {},
        const DetBatchView* exclude = nullptr
    ) const;

    [[nodiscard]] Projection project(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const;

    [[nodiscard]] Projection project(
        DetBatchView kets,
        std::span<const double> scale,
        double eps,
        const DetBatchView* exclude
    ) const;

    [[nodiscard]] ::libdet::Conns conn(
        DetBatchView kets,
        double eps = 0.0,
        const DetBatchView* include = nullptr
    ) const;

    [[nodiscard]] ::libdet::Conns sample_conn(
        DetBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0,
        bool bra_weight = false,
        const DetBatchView* include = nullptr
    ) const;

    [[nodiscard]] Projections sample_project(
        DetBatchView kets,
        std::span<const double> scale,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        const DetBatchView* exclude = nullptr,
        u64 seed = 0
    ) const;

    [[nodiscard]] Matrix matrix(DetBatchView bras, DetBatchView kets) const;

    [[nodiscard]] std::vector<double> matvec(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x
    ) const;

    [[nodiscard]] std::vector<double> matmat(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const;

private:
    explicit Hamiltonian(Integral ints);

    Integral ints_;
    u32 nword_ = 0;

    mutable std::mutex screen_mutex_;
    mutable std::shared_ptr<const Screen> screen_;
    mutable std::mutex conn_cache_mutex_;
    mutable ConnCache conn_cache_;
    mutable std::mutex space_cache_mutex_;
    mutable SpaceCache space_cache_;

    void check_one(DetRef det, const char* where) const;
    void check_dets(DetBatchView dets, const char* where) const;
    static void check_eps(double eps);
    static void check_window_eps(double eps1, double eps2);

    [[nodiscard]] std::shared_ptr<const Screen> screen(double cutoff) const;
    [[nodiscard]] static double max_abs(std::span<const double> values) noexcept;
    [[nodiscard]] static double screen_cutoff(
        double eps,
        double max_scale
    ) noexcept;
    [[nodiscard]] std::shared_ptr<const DetSpace> cached_space(
        DetBatchView kets
    ) const;

    [[nodiscard]] std::shared_ptr<const Conns> build_conns(
        DetRef ket,
        double eps,
        const Screen* screen,
        KetScratch& scratch
    ) const;

    [[nodiscard]] std::vector<std::shared_ptr<const Conns>> ket_conns(
        DetBatchView kets,
        double eps
    ) const;

    [[nodiscard]] Projection project_impl(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> scale,
        double eps
    ) const;
};

} // namespace libdet::rhf

#include <libdet/rhf/external.hpp>
#include <libdet/rhf/internal.hpp>

namespace libdet::rhf {

inline Hamiltonian::Hamiltonian(Integral ints)
    : ints_(std::move(ints)),
      nword_(bits::words_for(ints_.norb())),
      conn_cache_(nword_) {}

inline Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : ints_(other.ints_),
      nword_(other.nword_),
      conn_cache_(other.nword_) {}

inline Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
    if (this != &other) {
        ints_ = other.ints_;
        nword_ = other.nword_;
        screen_.reset();
        conn_cache_ = ConnCache(nword_);
        space_cache_ = SpaceCache();
    }
    return *this;
}

inline Hamiltonian::Hamiltonian(Hamiltonian&& other) noexcept
    : ints_(std::move(other.ints_)),
      nword_(other.nword_),
      screen_(std::move(other.screen_)),
      conn_cache_(other.nword_) {}

inline Hamiltonian& Hamiltonian::operator=(Hamiltonian&& other) noexcept {
    if (this != &other) {
        ints_ = std::move(other.ints_);
        nword_ = other.nword_;
        screen_ = std::move(other.screen_);
        conn_cache_ = ConnCache(nword_);
        space_cache_ = SpaceCache();
    }
    return *this;
}

inline Hamiltonian Hamiltonian::make(
    std::span<const double> h1,
    int norb,
    std::span<const double> eri,
    double ecore
) {
    return Hamiltonian(Integral(norb, h1, eri, ecore));
}

inline int Hamiltonian::norb() const noexcept {
    return ints_.norb();
}

inline u32 Hamiltonian::nword() const noexcept {
    return nword_;
}

inline void Hamiltonian::check_one(DetRef det, const char* where) const {
    if (det.nword() != nword_) {
        throw std::invalid_argument(
            std::string(where) + ": determinant nword mismatch"
        );
    }
}

inline void Hamiltonian::check_dets(
    DetBatchView dets,
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

inline void Hamiltonian::check_window_eps(double eps1, double eps2) {
    check_eps(eps1);
    check_eps(eps2);
    if (eps2 > eps1) {
        throw std::invalid_argument("eps2 must be <= eps1");
    }
}

inline std::shared_ptr<const Screen> Hamiltonian::screen(
    double cutoff
) const {
    if (cutoff <= 0.0 || !std::isfinite(cutoff)) return {};

    std::lock_guard<std::mutex> lock(screen_mutex_);
    if (!screen_ || cutoff < screen_->cutoff()) {
        screen_ = std::make_shared<Screen>(ints_, cutoff);
    }
    return screen_;
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
    DetBatchView kets
) const {
    {
        std::lock_guard<std::mutex> lock(space_cache_mutex_);
        if (auto space = space_cache_.find(kets)) return space;
    }

    auto fresh = std::make_shared<DetSpace>(kets);
    std::lock_guard<std::mutex> lock(space_cache_mutex_);
    if (auto space = space_cache_.find(kets)) return space;
    space_cache_.insert(kets, fresh);
    return fresh;
}

inline double Hamiltonian::hij(DetRef bra, DetRef ket) const {
    check_one(bra, "hij(bra)");
    check_one(ket, "hij(ket)");

    const DetDiff ex = det_diff(bra, ket);
    if (ex.deg > 2) return 0.0;
    if (ex.deg == 0) return diag(ints_, bra);

    if (ex.deg == 1) {
        return ex.na == 1
            ? ex.sign * single_alpha(
                ints_,
                bra,
                ex.occ_a[0],
                ex.vir_a[0]
            )
            : ex.sign * single_beta(
                ints_,
                bra,
                ex.occ_b[0],
                ex.vir_b[0]
            );
    }

    if (ex.na == 2) {
        return ex.sign * double_alpha(
            ints_,
            ex.occ_a[0],
            ex.occ_a[1],
            ex.vir_a[0],
            ex.vir_a[1]
        );
    }
    if (ex.nb == 2) {
        return ex.sign * double_beta(
            ints_,
            ex.occ_b[0],
            ex.occ_b[1],
            ex.vir_b[0],
            ex.vir_b[1]
        );
    }

    return ex.sign * double_mixed(
        ints_,
        ex.occ_a[0],
        ex.occ_b[0],
        ex.vir_a[0],
        ex.vir_b[0]
    );
}

} // namespace libdet::rhf

