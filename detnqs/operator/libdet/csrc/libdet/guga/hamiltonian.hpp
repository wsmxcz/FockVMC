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
#include <libdet/rhf/hamiltonian.hpp>

namespace libdet::guga {

using Matrix = rhf::Matrix;
using Conns = rhf::Conns;
using Projection = rhf::Projection;
using Projections = rhf::Projections;

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

    [[nodiscard]] Conns conn(
        DetBatchView kets,
        double eps = 0.0,
        const DetBatchView* include = nullptr
    ) const;

    [[nodiscard]] Conns sample_conn(
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
    Hamiltonian(Integral ints, int n_alpha, int n_beta);

    Integral ints_;
    int n_alpha_ = 0;
    int n_beta_ = 0;
    Sector sector_;
    Screen screen_;

    mutable std::mutex ket_cache_mutex_;
    mutable KetCache ket_cache_;
    mutable std::mutex csf_space_cache_mutex_;
    mutable CsfSpaceCache csf_space_cache_;

    void check_one(DetRef det, const char* where) const;
    void check_dets(DetBatchView dets, const char* where) const;
    static void check_eps(double eps);
    static void check_window_eps(double eps1, double eps2);

    [[nodiscard]] Csf csf(DetRef det, const char* where) const;
    [[nodiscard]] double element(const Csf& bra, const Csf& ket) const;
    [[nodiscard]] double element(DetRef bra, DetRef ket) const;

    [[nodiscard]] static double scaled_eps(double eps, double scale) noexcept;

    [[nodiscard]] std::shared_ptr<const CsfSpace> cached_csf_space(
        DetBatchView dets
    ) const;

    [[nodiscard]] std::shared_ptr<const KetConns> build_ket_conns(
        DetRef ket,
        double eps
    ) const;

    [[nodiscard]] std::shared_ptr<const KetConns> ket_conns(
        DetRef ket,
        double eps
    ) const;

    [[nodiscard]] std::vector<std::shared_ptr<const KetConns>> ket_conns(
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
      screen_(ints_),
      ket_cache_(sector_.nword) {}

inline Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : ints_(other.ints_),
      n_alpha_(other.n_alpha_),
      n_beta_(other.n_beta_),
      sector_(other.sector_),
      screen_(ints_),
      ket_cache_(sector_.nword) {}

inline Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
    if (this != &other) {
        ints_ = other.ints_;
        n_alpha_ = other.n_alpha_;
        n_beta_ = other.n_beta_;
        sector_ = other.sector_;
        screen_ = Screen(ints_);
        ket_cache_ = KetCache(sector_.nword);
        csf_space_cache_ = CsfSpaceCache();
    }
    return *this;
}

inline Hamiltonian::Hamiltonian(Hamiltonian&& other) noexcept
    : ints_(std::move(other.ints_)),
      n_alpha_(other.n_alpha_),
      n_beta_(other.n_beta_),
      sector_(other.sector_),
      screen_(ints_),
      ket_cache_(sector_.nword) {}

inline Hamiltonian& Hamiltonian::operator=(Hamiltonian&& other) noexcept {
    if (this != &other) {
        ints_ = std::move(other.ints_);
        n_alpha_ = other.n_alpha_;
        n_beta_ = other.n_beta_;
        sector_ = other.sector_;
        screen_ = Screen(ints_);
        ket_cache_ = KetCache(sector_.nword);
        csf_space_cache_ = CsfSpaceCache();
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

inline void Hamiltonian::check_one(DetRef det, const char* where) const {
    check_csf(det, sector_, where);
}

inline void Hamiltonian::check_dets(DetBatchView dets, const char* where) const {
    if (dets.nword != sector_.nword) {
        throw std::invalid_argument(std::string(where) + ": det nword mismatch");
    }

    for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
        check_csf(dets[idet], sector_, where);
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

inline Csf Hamiltonian::csf(DetRef det, const char* where) const {
    return decode_csf(det, sector_, where);
}

inline double Hamiltonian::element(
    const Csf& bra,
    const Csf& ket
) const {
    return guga::element(ints_, bra, ket, sector_.nword);
}

inline double Hamiltonian::element(DetRef bra, DetRef ket) const {
    return element(
        csf(bra, "element(bra)"),
        csf(ket, "element(ket)")
    );
}

inline double Hamiltonian::scaled_eps(double eps, double scale) noexcept {
    if (eps <= 0.0) return 0.0;
    if (scale <= 0.0) return std::numeric_limits<double>::infinity();
    return eps / scale;
}

inline std::shared_ptr<const CsfSpace> Hamiltonian::cached_csf_space(
    DetBatchView dets
) const {
    {
        std::lock_guard<std::mutex> lock(csf_space_cache_mutex_);
        if (auto hit = csf_space_cache_.find(dets)) return hit;
    }

    auto fresh = std::make_shared<CsfSpace>(dets, sector_);
    std::lock_guard<std::mutex> lock(csf_space_cache_mutex_);
    if (auto hit = csf_space_cache_.find(dets)) return hit;
    csf_space_cache_.insert(dets, fresh);
    return fresh;
}

} // namespace libdet::guga

#include <libdet/guga/sample.hpp>
#include <libdet/guga/external.hpp>
#include <libdet/guga/internal.hpp>
