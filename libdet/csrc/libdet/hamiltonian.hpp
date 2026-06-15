#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include <libdet/cache.hpp>

namespace libdet {

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
    std::vector<u64> det_words;
    std::vector<double> diag;
    std::vector<i32> ket_ptr;
    std::vector<i32> bra_idx;
    std::vector<double> h;
    std::vector<double> weight;
    std::vector<i32> sample_ket_ptr;
    std::vector<i32> sample_bra_idx;
    std::vector<double> sample_h;
    std::vector<i64> sample_count;
    std::vector<double> sample_weight;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

struct ConnSamples {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> det_words;
    std::vector<i32> ket_ptr;
    std::vector<i32> bra_idx;
    std::vector<double> h;
    std::vector<i64> count;
    std::vector<double> weight;
};

struct ProjectSamples {
    u32 nword = 0;
    std::vector<i32> rep_ptr;
    std::vector<u64> bra_words;
    std::vector<double> diags;
    std::vector<double> hpsi_strong;
    std::vector<double> hpsi_a;
    std::vector<double> hpsi_b;
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
        std::span<const double> coeffs = {},
        const DetBatchView* exclude = nullptr
    ) const;

    [[nodiscard]] Projection project(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps = 0.0
    ) const;

    [[nodiscard]] Projection project(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps,
        const DetBatchView* exclude
    ) const;

    [[nodiscard]] Conns conns(
        DetBatchView kets,
        double eps,
        i64 sample = 0,
        double sample_eps = 0.0,
        u64 seed = 0
    ) const;

    [[nodiscard]] std::pair<std::vector<double>, std::vector<i64>> degrees(
        DetBatchView kets,
        double eps
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

    [[nodiscard]] ConnSamples sample_conns(
        DetBatchView kets,
        std::span<const i64> counts,
        std::size_t n_streams,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const;

    [[nodiscard]] ProjectSamples sample_project(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        const DetBatchView* exclude = nullptr,
        i64 n_rep = 1,
        u64 seed = 0
    ) const;

private:
    explicit Hamiltonian(RHFIntegrals ints);

    RHFIntegrals ints_;
    u32 nword_ = 0;

    mutable std::mutex screen_mutex_;
    mutable std::shared_ptr<const Screen> screen_;
    mutable std::mutex ket_cache_mutex_;
    mutable KetCache ket_cache_;
    mutable std::mutex ket_space_cache_mutex_;
    mutable KetSpaceCache ket_space_cache_;

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
    [[nodiscard]] static AbsWindow abs_window(
        double lo,
        double hi,
        double scale
    ) noexcept;

    [[nodiscard]] std::shared_ptr<const KetSpace> cached_ket_space(
        DetBatchView kets
    ) const;

    [[nodiscard]] std::vector<u64> merge_det_parts(
        std::vector<std::vector<u64>>& parts
    ) const;

    [[nodiscard]] std::shared_ptr<const KetConns> build_ket_conns(
        DetRef ket,
        double eps,
        KetScratch& scratch,
        const Screen* screen
    ) const;

    [[nodiscard]] std::vector<std::shared_ptr<const KetConns>> ket_conns(
        DetBatchView kets,
        double eps
    ) const;

    [[nodiscard]] Projection project_impl(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps
    ) const;

    void build_matrix(Matrix& out, DetBatchView bras, DetBatchView kets) const;
};

} // namespace libdet
