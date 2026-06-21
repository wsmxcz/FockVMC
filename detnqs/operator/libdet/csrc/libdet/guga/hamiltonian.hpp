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

#include <libdet/guga/cache.hpp>
#include <libdet/guga/element.hpp>
#include <libdet/guga/screen.hpp>

namespace libdet::guga {

struct KetScratch;

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
        int n_alpha,
        int n_beta,
        double ecore = 0.0
    );

    [[nodiscard]] int norb() const noexcept;
    [[nodiscard]] u32 nword() const noexcept;

    [[nodiscard]] double hij(PathRef bra, PathRef ket) const;
    [[nodiscard]] std::vector<double> diags(PathBatchView paths) const;

    [[nodiscard]] std::vector<u64> expand(
        PathBatchView kets,
        double eps,
        std::span<const double> scale = {},
        const PathBatchView* exclude = nullptr
    ) const;

    [[nodiscard]] Projection project(
        PathBatchView bras,
        PathBatchView kets,
        std::span<const double> scale,
        double eps = 0.0
    ) const;

    [[nodiscard]] Projection project(
        PathBatchView kets,
        std::span<const double> scale,
        double eps,
        const PathBatchView* exclude
    ) const;

    [[nodiscard]] ::libdet::Conns conn(
        PathBatchView kets,
        double eps = 0.0,
        const PathBatchView* include = nullptr
    ) const;

    [[nodiscard]] ::libdet::Conns sample_conn(
        PathBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0,
        bool bra_weight = false,
        const PathBatchView* include = nullptr
    ) const;

    [[nodiscard]] Projections sample_project(
        PathBatchView kets,
        std::span<const double> scale,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        const PathBatchView* exclude = nullptr,
        u64 seed = 0
    ) const;

    [[nodiscard]] Matrix matrix(PathBatchView bras, PathBatchView kets) const;

    [[nodiscard]] std::vector<double> matvec(
        PathBatchView bras,
        PathBatchView kets,
        std::span<const double> x
    ) const;

    [[nodiscard]] std::vector<double> matmat(
        PathBatchView bras,
        PathBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const;

private:
    Hamiltonian(Integral ints, int n_alpha, int n_beta);

    Integral ints_;
    int n_alpha_ = 0;
    int n_beta_ = 0;
    Sector sector_;

    mutable std::mutex screen_mutex_;
    mutable std::shared_ptr<const Screen> screen_;
    mutable std::mutex conn_cache_mutex_;
    mutable ConnCache conn_cache_;
    mutable std::mutex space_cache_mutex_;
    mutable SpaceCache space_cache_;

    void check_one(PathRef path_ref, const char* where) const;
    void check_paths(PathBatchView paths, const char* where) const;
    static void check_eps(double eps);
    static void check_window_eps(double eps1, double eps2);

    [[nodiscard]] static double max_abs(std::span<const double> values) noexcept;
    [[nodiscard]] static double screen_cutoff(
        double eps,
        double max_scale
    ) noexcept;

    [[nodiscard]] std::shared_ptr<const Screen> screen(double cutoff) const;

    [[nodiscard]] std::shared_ptr<const PathSpace> cached_space(
        PathBatchView paths
    ) const;

    [[nodiscard]] std::shared_ptr<const Conns> build_conns(
        PathRef ket,
        double eps,
        const Screen& screen,
        KetScratch& scratch
    ) const;

    [[nodiscard]] std::shared_ptr<const Conns> ket_conns(
        PathRef ket,
        double eps
    ) const;

    [[nodiscard]] std::vector<std::shared_ptr<const Conns>> ket_conns(
        PathBatchView kets,
        double eps
    ) const;

    [[nodiscard]] Projection project_impl(
        PathBatchView bras,
        PathBatchView kets,
        std::span<const double> scale,
        double eps
    ) const;
};

inline Hamiltonian::Hamiltonian(Integral ints, int n_alpha, int n_beta)
    : ints_(std::move(ints)),
      n_alpha_(n_alpha),
      n_beta_(n_beta),
      sector_{
          ints_.norb(),
          n_alpha + n_beta,
          n_alpha - n_beta,
          bits::words_for(ints_.norb())
      },
      conn_cache_(sector_.nword) {}

inline Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : ints_(other.ints_),
      n_alpha_(other.n_alpha_),
      n_beta_(other.n_beta_),
      sector_(other.sector_),
      conn_cache_(sector_.nword) {}

inline Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
    if (this != &other) {
        ints_ = other.ints_;
        n_alpha_ = other.n_alpha_;
        n_beta_ = other.n_beta_;
        sector_ = other.sector_;
        screen_.reset();
        conn_cache_ = ConnCache(sector_.nword);
        space_cache_ = SpaceCache();
    }
    return *this;
}

inline Hamiltonian::Hamiltonian(Hamiltonian&& other) noexcept
    : ints_(std::move(other.ints_)),
      n_alpha_(other.n_alpha_),
      n_beta_(other.n_beta_),
      sector_(other.sector_),
      screen_(std::move(other.screen_)),
      conn_cache_(sector_.nword) {}

inline Hamiltonian& Hamiltonian::operator=(Hamiltonian&& other) noexcept {
    if (this != &other) {
        ints_ = std::move(other.ints_);
        n_alpha_ = other.n_alpha_;
        n_beta_ = other.n_beta_;
        sector_ = other.sector_;
        screen_ = std::move(other.screen_);
        conn_cache_ = ConnCache(sector_.nword);
        space_cache_ = SpaceCache();
    }
    return *this;
}

inline Hamiltonian Hamiltonian::make(
    std::span<const double> h1,
    int norb,
    std::span<const double> eri,
    int n_alpha,
    int n_beta,
    double ecore
) {
    if (n_alpha < n_beta) {
        throw std::invalid_argument("spin Hamiltonian requires n_alpha >= n_beta");
    }
    if (n_alpha < 0 || n_beta < 0 || n_alpha > norb || n_beta > norb) {
        throw std::invalid_argument("spin Hamiltonian: invalid electron counts");
    }

    return Hamiltonian(Integral(norb, h1, eri, ecore), n_alpha, n_beta);
}

inline int Hamiltonian::norb() const noexcept {
    return sector_.norb;
}

inline u32 Hamiltonian::nword() const noexcept {
    return sector_.nword;
}

inline void Hamiltonian::check_one(PathRef path_ref, const char* where) const {
    check_path(path_ref, sector_, where);
}

inline void Hamiltonian::check_paths(PathBatchView paths, const char* where) const {
    if (paths.nword != sector_.nword) {
        throw std::invalid_argument(std::string(where) + ": path nword mismatch");
    }

    for (std::size_t ipath = 0; ipath < paths.n_paths; ++ipath) {
        check_path(paths[ipath], sector_, where);
    }
}

inline void Hamiltonian::check_eps(double eps) {
    if (std::isnan(eps)) throw std::invalid_argument("eps must not be NaN");
    if (eps < 0.0) throw std::invalid_argument("eps must be nonnegative");
}

inline void Hamiltonian::check_window_eps(double eps1, double eps2) {
    check_eps(eps1);
    check_eps(eps2);
    if (eps2 > eps1) throw std::invalid_argument("eps2 must be <= eps1");
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

inline std::shared_ptr<const Screen> Hamiltonian::screen(double cutoff) const {
    if (cutoff < 0.0) cutoff = 0.0;

    std::lock_guard<std::mutex> lock(screen_mutex_);
    if (!screen_ || cutoff < screen_->cutoff()) {
        screen_ = std::make_shared<Screen>(ints_, cutoff);
    }
    return screen_;
}

inline std::shared_ptr<const PathSpace> Hamiltonian::cached_space(
    PathBatchView paths
) const {
    {
        std::lock_guard<std::mutex> lock(space_cache_mutex_);
        if (auto space = space_cache_.find(paths)) return space;
    }

    auto fresh = std::make_shared<PathSpace>(paths, sector_);
    std::lock_guard<std::mutex> lock(space_cache_mutex_);
    if (auto space = space_cache_.find(paths)) return space;
    space_cache_.insert(paths, fresh);
    return fresh;
}

} // namespace libdet::guga

#include <libdet/guga/sample.hpp>
#include <libdet/guga/external.hpp>
#include <libdet/guga/internal.hpp>
