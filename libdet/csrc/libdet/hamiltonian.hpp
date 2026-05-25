#pragma once

#include <libdet/enumerate.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet {

class Determinants {
public:
    [[nodiscard]] std::size_t n_dets() const noexcept {
        return nword_ == 0 ? 0u : det_words_.size() / det_size(nword_);
    }
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::span<const u64> det_words() const noexcept { return det_words_; }

    void set_nword(u32 x) noexcept { nword_ = x; }
    [[nodiscard]] std::vector<u64>& det_words_mut() noexcept { return det_words_; }

private:
    u32 nword_ = 0;
    std::vector<u64> det_words_;
};

class Matrix {
public:
    [[nodiscard]] std::size_t n_bra() const noexcept { return n_bra_; }
    [[nodiscard]] std::size_t n_ket() const noexcept { return n_ket_; }
    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }
    [[nodiscard]] std::span<const i32> row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] std::span<const i32> col() const noexcept { return col_; }
    [[nodiscard]] std::span<const double> h() const noexcept { return h_; }

    void resize(std::size_t n_bra, std::size_t n_ket) noexcept { n_bra_ = n_bra; n_ket_ = n_ket; }
    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<i32>& row_ptr_mut() noexcept { return row_ptr_; }
    [[nodiscard]] std::vector<i32>& col_mut() noexcept { return col_; }
    [[nodiscard]] std::vector<double>& h_mut() noexcept { return h_; }

private:
    std::size_t n_bra_ = 0;
    std::size_t n_ket_ = 0;
    std::vector<double> diags_;
    std::vector<i32> row_ptr_;
    std::vector<i32> col_;
    std::vector<double> h_;
};

class Edges {
public:
    [[nodiscard]] std::size_t n_rows() const noexcept { return n_rows_; }
    [[nodiscard]] std::size_t n_cols() const noexcept {
        return nword_ == 0 ? 0u : col_words_.size() / det_size(nword_);
    }
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::span<const u64> row_words() const noexcept { return row_words_; }
    [[nodiscard]] std::span<const u64> col_words() const noexcept { return col_words_; }
    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }
    [[nodiscard]] std::span<const i32> row_ptr() const noexcept { return row_ptr_; }
    [[nodiscard]] std::span<const i32> col() const noexcept { return col_; }
    [[nodiscard]] std::span<const double> h() const noexcept { return h_; }
    [[nodiscard]] std::span<const double> row_weight() const noexcept { return row_weight_; }
    [[nodiscard]] std::span<const i64> row_nnz() const noexcept { return row_nnz_; }

    void set_nword(u32 x) noexcept { nword_ = x; }
    void set_n_rows(std::size_t x) noexcept { n_rows_ = x; }
    [[nodiscard]] std::vector<u64>& row_words_mut() noexcept { return row_words_; }
    [[nodiscard]] std::vector<u64>& col_words_mut() noexcept { return col_words_; }
    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<i32>& row_ptr_mut() noexcept { return row_ptr_; }
    [[nodiscard]] std::vector<i32>& col_mut() noexcept { return col_; }
    [[nodiscard]] std::vector<double>& h_mut() noexcept { return h_; }
    [[nodiscard]] std::vector<double>& row_weight_mut() noexcept { return row_weight_; }
    [[nodiscard]] std::vector<i64>& row_nnz_mut() noexcept { return row_nnz_; }

private:
    u32 nword_ = 0;
    std::size_t n_rows_ = 0;
    std::vector<u64> row_words_;
    std::vector<u64> col_words_;
    std::vector<double> diags_;
    std::vector<i32> row_ptr_;
    std::vector<i32> col_;
    std::vector<double> h_;
    std::vector<double> row_weight_;
    std::vector<i64> row_nnz_;
};

class Degrees {
public:
    [[nodiscard]] std::size_t n_rows() const noexcept { return row_nnz_.size(); }
    [[nodiscard]] std::span<const i64> row_nnz() const noexcept { return row_nnz_; }
    [[nodiscard]] std::span<const double> row_weight() const noexcept { return row_weight_; }

    [[nodiscard]] std::vector<i64>& row_nnz_mut() noexcept { return row_nnz_; }
    [[nodiscard]] std::vector<double>& row_weight_mut() noexcept { return row_weight_; }

private:
    std::vector<i64> row_nnz_;
    std::vector<double> row_weight_;
};

class Projection {
public:
    [[nodiscard]] std::size_t n_bras() const noexcept { return hpsi_.size(); }
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::span<const u64> bra_words() const noexcept { return bra_words_; }
    [[nodiscard]] std::span<const double> hpsi() const noexcept { return hpsi_; }
    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }

    void set_nword(u32 x) noexcept { nword_ = x; }
    [[nodiscard]] std::vector<u64>& bra_words_mut() noexcept { return bra_words_; }
    [[nodiscard]] std::vector<double>& hpsi_mut() noexcept { return hpsi_; }
    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }

private:
    u32 nword_ = 0;
    std::vector<u64> bra_words_;
    std::vector<double> hpsi_;
    std::vector<double> diags_;
};

class EdgeSamples {
public:
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::size_t n_samples() const noexcept { return pgen_.size(); }

    void set_nword(u32 x) noexcept { nword_ = x; }
    [[nodiscard]] std::span<const i64> row_nnz() const noexcept { return row_nnz_; }
    [[nodiscard]] std::span<const double> row_weight() const noexcept { return row_weight_; }
    [[nodiscard]] std::span<const i32> rows() const noexcept { return rows_; }
    [[nodiscard]] std::span<const u64> det_words() const noexcept { return det_words_; }
    [[nodiscard]] std::span<const double> h() const noexcept { return h_; }
    [[nodiscard]] std::span<const double> pgen() const noexcept { return pgen_; }
    [[nodiscard]] std::span<const i64> counts() const noexcept { return counts_; }

    [[nodiscard]] std::vector<i64>& row_nnz_mut() noexcept { return row_nnz_; }
    [[nodiscard]] std::vector<double>& row_weight_mut() noexcept { return row_weight_; }
    [[nodiscard]] std::vector<i32>& rows_mut() noexcept { return rows_; }
    [[nodiscard]] std::vector<u64>& det_words_mut() noexcept { return det_words_; }
    [[nodiscard]] std::vector<double>& h_mut() noexcept { return h_; }
    [[nodiscard]] std::vector<double>& pgen_mut() noexcept { return pgen_; }
    [[nodiscard]] std::vector<i64>& counts_mut() noexcept { return counts_; }

private:
    u32 nword_ = 0;
    std::vector<i64> row_nnz_;
    std::vector<double> row_weight_;
    std::vector<i32> rows_;
    std::vector<u64> det_words_;
    std::vector<double> h_;
    std::vector<double> pgen_;
    std::vector<i64> counts_;
};

class ShellSamples {
public:
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::size_t n_samples() const noexcept { return hpsi_a_.size(); }

    void set_nword(u32 x) noexcept { nword_ = x; }
    [[nodiscard]] std::span<const i32> rep_ptr() const noexcept { return rep_ptr_; }
    [[nodiscard]] std::span<const u64> det_words() const noexcept { return det_words_; }
    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }
    [[nodiscard]] std::span<const double> hpsi_strong() const noexcept { return hpsi_strong_; }
    [[nodiscard]] std::span<const double> hpsi_a() const noexcept { return hpsi_a_; }
    [[nodiscard]] std::span<const double> hpsi_b() const noexcept { return hpsi_b_; }

    [[nodiscard]] std::vector<i32>& rep_ptr_mut() noexcept { return rep_ptr_; }
    [[nodiscard]] std::vector<u64>& det_words_mut() noexcept { return det_words_; }
    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<double>& hpsi_strong_mut() noexcept { return hpsi_strong_; }
    [[nodiscard]] std::vector<double>& hpsi_a_mut() noexcept { return hpsi_a_; }
    [[nodiscard]] std::vector<double>& hpsi_b_mut() noexcept { return hpsi_b_; }

private:
    u32 nword_ = 0;
    std::vector<i32> rep_ptr_;
    std::vector<u64> det_words_;
    std::vector<double> diags_;
    std::vector<double> hpsi_strong_;
    std::vector<double> hpsi_a_;
    std::vector<double> hpsi_b_;
};

class Hamiltonian {
public:
    Hamiltonian() = delete;

    Hamiltonian(const Hamiltonian& other) : ints_(other.ints_), nword_(other.nword_) {}

    Hamiltonian& operator=(const Hamiltonian& other) {
        if (this != &other) {
            ints_ = other.ints_;
            nword_ = other.nword_;
            hb_.reset();
            hb_cut_ = 0.0;
        }
        return *this;
    }

    Hamiltonian(Hamiltonian&& other) noexcept
        : ints_(std::move(other.ints_)), nword_(other.nword_), hb_(std::move(other.hb_)), hb_cut_(other.hb_cut_) {}

    Hamiltonian& operator=(Hamiltonian&& other) noexcept {
        if (this != &other) {
            ints_ = std::move(other.ints_);
            nword_ = other.nword_;
            hb_ = std::move(other.hb_);
            hb_cut_ = other.hb_cut_;
        }
        return *this;
    }

    [[nodiscard]] static Hamiltonian make(std::span<const double> h1, int norb, std::span<const double> eri, double ecore = 0.0) {
        return Hamiltonian(RHFIntegrals(norb, h1, eri, ecore));
    }

    [[nodiscard]] int norb() const noexcept { return ints_.norb(); }
    [[nodiscard]] u32 nword() const noexcept { return nword_; }

    [[nodiscard]] double hij(DetRef bra, DetRef ket) const {
        if (bra.nword() != nword_ || ket.nword() != nword_) throw std::invalid_argument("hij: determinant nword mismatch");
        const Excitation ex = diff(bra, ket);
        if (ex.deg > 2) return 0.0;
        if (ex.deg == 0) return SlaterCondon::diagonal(ints_, bra);
        if (ex.deg == 1) return ex.na == 1 ? ex.sign * SlaterCondon::single_a(ints_, bra, ex.occ_a[0], ex.vir_a[0])
                                           : ex.sign * SlaterCondon::single_b(ints_, bra, ex.occ_b[0], ex.vir_b[0]);
        if (ex.na == 2) return ex.sign * SlaterCondon::double_aa(ints_, ex.occ_a[0], ex.occ_a[1], ex.vir_a[0], ex.vir_a[1]);
        if (ex.nb == 2) return ex.sign * SlaterCondon::double_bb(ints_, ex.occ_b[0], ex.occ_b[1], ex.vir_b[0], ex.vir_b[1]);
        return ex.sign * SlaterCondon::double_ab(ints_, ex.occ_a[0], ex.occ_b[0], ex.vir_a[0], ex.vir_b[0]);
    }

    [[nodiscard]] std::vector<double> diags(DetBatchView dets) const {
        check_dets(dets, "diags");
        std::vector<double> out(dets.n_dets, 0.0);
        for_static(dets.n_dets, [&](std::size_t r, RowWork& work) {
            bits::set_list(dets[r].alpha(), work.occ.occ_a);
            bits::set_list(dets[r].beta(), work.occ.occ_b);
            out[r] = SlaterCondon::diagonal(ints_, work.occ.occ_a, work.occ.occ_b);
        });
        return out;
    }

    [[nodiscard]] Determinants expand(DetBatchView kets, double eps, std::span<const double> coeffs = {}, const DetBatchView* exclude = nullptr) const {
        check_dets(kets, "expand(kets)");
        check_eps(eps);
        if (!coeffs.empty() && coeffs.size() != kets.n_dets) throw std::invalid_argument("coeffs.shape must match kets.shape[0]");
        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "expand(exclude)");
        return expand_impl(kets, eps, coeffs, base);
    }

    [[nodiscard]] Projection project(DetBatchView bras, DetBatchView kets, std::span<const double> coeffs, double eps = 0.0) const {
        check_dets(bras, "project(bras)");
        check_dets(kets, "project(kets)");
        if (coeffs.size() != kets.n_dets) throw std::invalid_argument("coeffs.shape must match kets.shape[0]");
        check_eps(eps);
        return project_impl(bras, kets, coeffs, eps);
    }

    [[nodiscard]] Edges edges(DetBatchView dets, double eps) const {
        check_dets(dets, "edges(dets)");
        check_eps(eps);
        return edges_impl(dets, eps);
    }

    [[nodiscard]] Degrees degrees(DetBatchView dets, double eps) const {
        check_dets(dets, "degrees(dets)");
        check_eps(eps);
        return degrees_impl(dets, eps);
    }

    [[nodiscard]] Matrix matrix(DetBatchView bras, DetBatchView kets) const {
        check_dets(bras, "matrix(bras)");
        check_dets(kets, "matrix(kets)");
        Matrix out;
        out.resize(bras.n_dets, kets.n_dets);
        build_matrix(out, bras, kets);
        return out;
    }

    [[nodiscard]] std::vector<double> matvec(DetBatchView bras, DetBatchView kets, std::span<const double> x) const {
        check_dets(bras, "matvec(bras)");
        check_dets(kets, "matvec(kets)");
        if (x.size() != kets.n_dets) throw std::invalid_argument("x.shape must match kets.shape[0]");
        const DetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets, 0.0);
        for_guided(bras.n_dets, [&](std::size_t r, RowWork& work) {
            const DetRef bra = bras[r];
            double acc = 0.0;
            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) acc += SlaterCondon::diagonal(ints_, bra) * x[static_cast<std::size_t>(diag_idx)];
            enumerate_internal(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                acc += h * x[static_cast<std::size_t>(idx)];
            });
            out[r] = acc;
        });
        return out;
    }

    [[nodiscard]] std::vector<double> matmat(DetBatchView bras, DetBatchView kets, std::span<const double> x, std::size_t nrhs) const {
        check_dets(bras, "matmat(bras)");
        check_dets(kets, "matmat(kets)");
        if (x.size() != kets.n_dets * nrhs) throw std::invalid_argument("X.shape must be (n_ket, n_rhs)");
        const DetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets * nrhs, 0.0);
        for_guided(bras.n_dets, [&](std::size_t r, RowWork& work) {
            const DetRef bra = bras[r];
            double* yr = out.data() + r * nrhs;
            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) {
                const double h = SlaterCondon::diagonal(ints_, bra);
                const double* xr = x.data() + static_cast<std::size_t>(diag_idx) * nrhs;
                for (std::size_t j = 0; j < nrhs; ++j) yr[j] += h * xr[j];
            }
            enumerate_internal(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                const double* xr = x.data() + static_cast<std::size_t>(idx) * nrhs;
                for (std::size_t j = 0; j < nrhs; ++j) yr[j] += h * xr[j];
            });
        });
        return out;
    }

    [[nodiscard]] EdgeSamples sample_edges(DetBatchView dets, std::span<const i64> counts, double eps1, double eps2, u64 seed = 0) const {
        check_dets(dets, "sample_edges");
        check_window_eps(eps1, eps2);
        if (!counts.empty() && counts.size() != dets.n_dets) throw std::invalid_argument("counts.shape must match dets.shape[0]");
        return sample_edges_impl(dets, counts, eps1, eps2, seed);
    }

    [[nodiscard]] ShellSamples sample_shell(DetBatchView kets, std::span<const double> coeffs,
                                            double eps1, double eps2, std::span<const i64> counts,
                                            const DetBatchView* exclude = nullptr, i64 n_rep = 1, u64 seed = 0) const {
        check_dets(kets, "sample_shell(kets)");
        if (coeffs.size() != kets.n_dets) throw std::invalid_argument("coeffs.shape must match kets.shape[0]");
        if (counts.size() != kets.n_dets) throw std::invalid_argument("counts.shape must match kets.shape[0]");
        if (n_rep <= 0) throw std::invalid_argument("n_rep must be positive");
        check_window_eps(eps1, eps2);
        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "sample_shell(exclude)");
        return sample_shell_impl(kets, coeffs, eps1, eps2, counts, base, n_rep, seed);
    }

private:
    explicit Hamiltonian(RHFIntegrals ints) : ints_(std::move(ints)), nword_(bits::words_for(ints_.norb())) {}

    RHFIntegrals ints_;
    u32 nword_ = 0;
    mutable std::mutex hb_mutex_;
    mutable std::shared_ptr<const HeatBathTable> hb_;
    mutable double hb_cut_ = 0.0;

    void check_dets(DetBatchView dets, const char* where) const {
        if (dets.nword != nword_) throw std::invalid_argument(std::string(where) + ": determinant nword mismatch");
    }

    static void check_eps(double eps) {
        if (eps < 0.0) throw std::invalid_argument("eps must be nonnegative");
    }

    static void check_window_eps(double eps1, double eps2) {
        if (std::isnan(eps1)) throw std::invalid_argument("eps1 must not be NaN");
        if (std::isnan(eps2)) throw std::invalid_argument("eps2 must not be NaN");
        if (eps1 < 0.0) throw std::invalid_argument("eps1 must be nonnegative");
        if (eps2 < 0.0) throw std::invalid_argument("eps2 must be nonnegative");
        if (eps2 > eps1) throw std::invalid_argument("eps2 must be <= eps1");
    }

    [[nodiscard]] const HeatBathTable* heatbath(double eps) const {
        if (eps <= 1.0e-12) return nullptr;
        const double table_cut = std::max(1.0e-12, eps * 1.0e-3);
        std::lock_guard<std::mutex> lock(hb_mutex_);
        if (!hb_ || hb_cut_ > table_cut) {
            hb_ = std::make_shared<HeatBathTable>(ints_, table_cut);
            hb_cut_ = table_cut;
        }
        return hb_.get();
    }

    [[nodiscard]] static double row_cut(double eps, double scale) noexcept {
        if (eps <= 0.0) return 0.0;
        if (scale <= 0.0) return std::numeric_limits<double>::infinity();
        return eps / scale;
    }

    [[nodiscard]] static u64 sample_seed(u64 seed, DetRef det, i64 rep = 0, int pair = 0) noexcept {
        u64 x = splitmix64(seed ^ 0x243f6a8885a308d3ULL);
        x = splitmix64(x ^ det_fingerprint(det));
        x = splitmix64(x ^ static_cast<u64>(rep + 1));
        x = splitmix64(x ^ (pair == 0 ? 0x13198a2e03707344ULL : 0xa4093822299f31d0ULL));
        return x;
    }

    template <class F>
    void for_static(std::size_t nrow, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            RowWork work(nword_, ints_.norb());
#pragma omp for schedule(static)
            for (i64 r = 0; r < static_cast<i64>(nrow); ++r) f(static_cast<std::size_t>(r), work);
        }
#else
        RowWork work(nword_, ints_.norb());
        for (std::size_t r = 0; r < nrow; ++r) f(r, work);
#endif
    }

    template <class F>
    void for_guided(std::size_t nrow, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            RowWork work(nword_, ints_.norb());
#pragma omp for schedule(guided)
            for (i64 r = 0; r < static_cast<i64>(nrow); ++r) f(static_cast<std::size_t>(r), work);
        }
#else
        RowWork work(nword_, ints_.norb());
        for (std::size_t r = 0; r < nrow; ++r) f(r, work);
#endif
    }

    [[nodiscard]] Determinants merge_det_parts(std::vector<std::vector<u64>>& parts) const {
        std::size_t total = 0;
        for (auto& p : parts) {
            sort_unique_dets(p, nword_);
            total += p.size();
        }
        Determinants out;
        out.set_nword(nword_);
        out.det_words_mut().reserve(total);
        for (auto& p : parts) out.det_words_mut().insert(out.det_words_mut().end(), p.begin(), p.end());
        sort_unique_dets(out.det_words_mut(), nword_);
        return out;
    }

    [[nodiscard]] Determinants expand_impl(DetBatchView kets, double eps, std::span<const double> coeffs, DetBatchView exclude) const {
        const DetIndex exclude_index(exclude);
        const HeatBathTable* hb = heatbath(eps);
#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif
        std::vector<std::vector<u64>> local(static_cast<std::size_t>(nthread));

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
            auto& words = local[static_cast<std::size_t>(tid)];
#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(kets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);
#else
        {
            RowWork work(nword_, ints_.norb());
            auto& words = local[0];
            for (std::size_t r = 0; r < kets.n_dets; ++r) {
#endif
                const double scale = coeffs.empty() ? 1.0 : std::abs(coeffs[r]);
                if (scale <= 0.0) continue;
                const double cut = row_cut(eps, scale);
                if (!std::isfinite(cut)) continue;
                enumerate_screened(ints_, hb, kets[r], work, cut, [&](DetRef det, double h) {
                    if (exclude_index.find(det) >= 0) return;
                    if (std::abs(h) * scale >= eps) append_det(words, det);
                });
            }
        }
        return merge_det_parts(local);
    }

    struct ProjEntry {
        i32 idx = -1;
        double v = 0.0;
    };

    [[nodiscard]] Projection project_impl(DetBatchView bras, DetBatchView kets, std::span<const double> coeffs, double eps) const {
        const DetIndex bra_index(bras);
        const HeatBathTable* hb = heatbath(eps);
        Projection out;
        out.set_nword(nword_);
        copy_batch(out.bra_words_mut(), bras);
        out.hpsi_mut().assign(bras.n_dets, 0.0);
        out.diags_mut() = diags(bras);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif
        std::vector<std::vector<ProjEntry>> local(static_cast<std::size_t>(nthread));

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
            auto& acc = local[static_cast<std::size_t>(tid)];
#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(kets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);
#else
        {
            RowWork work(nword_, ints_.norb());
            auto& acc = local[0];
            for (std::size_t r = 0; r < kets.n_dets; ++r) {
#endif
                const double c = coeffs[r];
                const double scale = std::abs(c);
                if (scale <= 0.0) continue;
                const DetRef ket = kets[r];
                const i32 diag_idx = bra_index.find(ket);
                if (diag_idx >= 0) {
                    const double v = c * SlaterCondon::diagonal(ints_, ket);
                    if (std::abs(v) >= eps) acc.push_back(ProjEntry{diag_idx, v});
                }
                const double cut = row_cut(eps, scale);
                enumerate_screened(ints_, hb, ket, work, cut, [&](DetRef bra, double h) {
                    const i32 idx = bra_index.find(bra);
                    if (idx < 0) return;
                    const double v = c * h;
                    if (std::abs(v) >= eps) acc.push_back(ProjEntry{idx, v});
                });
            }
        }
        for (const auto& part : local) {
            for (const ProjEntry& e : part) out.hpsi_mut()[static_cast<std::size_t>(e.idx)] += e.v;
        }
        return out;
    }

    struct EdgeRec {
        i32 row = 0;
        std::size_t det = 0;
        double h = 0.0;
    };

    struct EdgeLocal {
        std::vector<EdgeRec> edge;
        std::vector<u64> words;
    };

    [[nodiscard]] Edges edges_impl(DetBatchView dets, double eps) const {
        const HeatBathTable* hb = heatbath(eps);
#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif
        std::vector<EdgeLocal> local(static_cast<std::size_t>(nthread));

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
            auto& loc = local[static_cast<std::size_t>(tid)];
#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(dets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);
#else
        {
            RowWork work(nword_, ints_.norb());
            auto& loc = local[0];
            for (std::size_t r = 0; r < dets.n_dets; ++r) {
#endif
                enumerate_screened(ints_, hb, dets[r], work, eps, [&](DetRef det, double h) {
                    const std::size_t pos = append_det_index(loc.words, nword_, det);
                    loc.edge.push_back(EdgeRec{static_cast<i32>(r), pos, h});
                });
            }
        }

        std::vector<EdgeRec> edges;
        std::vector<u64> edge_words;
        std::size_t nedge = 0;
        std::size_t nword_total = 0;
        for (const auto& loc : local) {
            nedge += loc.edge.size();
            nword_total += loc.words.size();
        }
        edges.reserve(nedge);
        edge_words.reserve(nword_total);
        for (auto& loc : local) {
            const std::size_t det_offset = edge_words.size() / det_size(nword_);
            edge_words.insert(edge_words.end(), loc.words.begin(), loc.words.end());
            for (EdgeRec e : loc.edge) {
                e.det += det_offset;
                edges.push_back(e);
            }
        }

        std::vector<u64> pool;
        pool.reserve(edges.size() * det_size(nword_));
        for (const EdgeRec& e : edges) append_det(pool, det_at(edge_words, nword_, e.det));
        sort_unique_dets(pool, nword_);

        DetIndex row_index(dets);
        Edges out;
        out.set_nword(nword_);
        out.set_n_rows(dets.n_dets);
        copy_batch(out.row_words_mut(), dets);
        copy_batch(out.col_words_mut(), dets);

        const std::size_t pool_n = nword_ == 0 ? 0u : pool.size() / det_size(nword_);
        for (std::size_t i = 0; i < pool_n; ++i) {
            const DetRef det = det_at(pool, nword_, i);
            if (row_index.find(det) < 0) append_det(out.col_words_mut(), det);
        }

        const DetBatchView col_space{out.col_words().data(), out.n_cols(), nword_};
        DetIndex col_index(col_space);

        struct CsrEdge { i32 row; i32 col; double h; };
        std::vector<CsrEdge> csr;
        csr.reserve(edges.size());
        for (const EdgeRec& e : edges) {
            const i32 col = col_index.find(det_at(edge_words, nword_, e.det));
            if (col >= 0) csr.push_back(CsrEdge{e.row, col, e.h});
        }
        std::sort(csr.begin(), csr.end(), [](const CsrEdge& a, const CsrEdge& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });

        out.diags_mut() = diags(dets);
        out.row_ptr_mut().assign(dets.n_dets + 1u, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);
        out.row_nnz_mut().assign(dets.n_dets, 0);
        for (const CsrEdge& e : csr) ++out.row_ptr_mut()[static_cast<std::size_t>(e.row) + 1u];
        std::size_t acc = 0;
        for (std::size_t r = 0; r < dets.n_dets; ++r) {
            acc += static_cast<std::size_t>(out.row_ptr_mut()[r + 1u]);
            out.row_ptr_mut()[r + 1u] = to_i32(acc);
        }
        out.col_mut().resize(csr.size());
        out.h_mut().resize(csr.size());
        std::vector<i32> pos = out.row_ptr_mut();
        for (const CsrEdge& e : csr) {
            const std::size_t p = static_cast<std::size_t>(pos[static_cast<std::size_t>(e.row)]++);
            out.col_mut()[p] = e.col;
            out.h_mut()[p] = e.h;
            out.row_weight_mut()[static_cast<std::size_t>(e.row)] += std::abs(e.h);
            ++out.row_nnz_mut()[static_cast<std::size_t>(e.row)];
        }
        return out;
    }

    [[nodiscard]] Degrees degrees_impl(DetBatchView dets, double eps) const {
        const HeatBathTable* hb = heatbath(eps);

        Degrees out;
        out.row_nnz_mut().assign(dets.n_dets, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);

        for_guided(dets.n_dets, [&](std::size_t r, RowWork& work) {
            i64 nnz = 0;
            double weight = 0.0;

            // Count screened off-diagonal edges without materializing columns.
            enumerate_screened(ints_, hb, dets[r], work, eps, [&](DetRef, double h) {
                ++nnz;
                weight += std::abs(h);
            });

            out.row_nnz_mut()[r] = nnz;
            out.row_weight_mut()[r] = weight;
        });

        return out;
    }

    void build_matrix(Matrix& out, DetBatchView bras, DetBatchView kets) const {
        const DetSpace ket_space(kets);
        const std::size_t nrow = bras.n_dets;
        out.diags_mut().assign(nrow, 0.0);
        out.row_ptr_mut().assign(nrow + 1u, 0);

        for_guided(nrow, [&](std::size_t r, RowWork& work) {
            const DetRef bra = bras[r];
            i32 nnz = 0;
            out.diags_mut()[r] = SlaterCondon::diagonal(ints_, bra);
            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0 && out.diags_mut()[r] != 0.0) ++nnz;
            enumerate_internal(ints_, ket_space, bra, work, [&](i32, double) { ++nnz; });
            out.row_ptr_mut()[r + 1u] = nnz;
        });

        std::size_t nnz = 0;
        for (std::size_t r = 0; r < nrow; ++r) {
            nnz += static_cast<std::size_t>(out.row_ptr_mut()[r + 1u]);
            out.row_ptr_mut()[r + 1u] = to_i32(nnz);
        }
        out.col_mut().resize(nnz);
        out.h_mut().resize(nnz);

        for_guided(nrow, [&](std::size_t r, RowWork& work) {
            const DetRef bra = bras[r];
            std::size_t pos = static_cast<std::size_t>(out.row_ptr_mut()[r]);
            const double diag = out.diags_mut()[r];
            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0 && diag != 0.0) {
                out.col_mut()[pos] = diag_idx;
                out.h_mut()[pos] = diag;
                ++pos;
            }
            enumerate_internal(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                out.col_mut()[pos] = idx;
                out.h_mut()[pos] = h;
                ++pos;
            });
        });
    }

    [[nodiscard]] EdgeSamples sample_edges_impl(DetBatchView dets, std::span<const i64> counts, double eps1, double eps2, u64 seed) const {
        EdgeSamples out;
        out.set_nword(nword_);
        out.row_nnz_mut().assign(dets.n_dets, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);
        const bool draw = !counts.empty();
        const HeatBathTable* hb = heatbath(eps2);
#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif
        std::vector<std::vector<i32>> local_row(static_cast<std::size_t>(nthread));
        std::vector<std::vector<u64>> local_det(static_cast<std::size_t>(nthread));
        std::vector<std::vector<double>> local_h(static_cast<std::size_t>(nthread));
        std::vector<std::vector<double>> local_pgen(static_cast<std::size_t>(nthread));
        std::vector<std::vector<i64>> local_counts(static_cast<std::size_t>(nthread));

        auto process_one = [&](std::size_t r, RowWork& work, std::vector<i32>& rows, std::vector<u64>& words,
                               std::vector<double>& hs, std::vector<double>& pgens, std::vector<i64>& cts) {
            work.weak_words.clear();
            work.weak_h.clear();
            work.weak_cdf.clear();
            double norm = 0.0;
            i64 nnz = 0;
            enumerate_window(ints_, hb, dets[r], work, eps2, eps1, [&](DetRef det, double h) {
                append_det(work.weak_words, det);
                work.weak_h.push_back(h);
                norm += std::abs(h);
                work.weak_cdf.push_back(norm);
                ++nnz;
            });
            out.row_nnz_mut()[r] = nnz;
            out.row_weight_mut()[r] = norm;
            if (!draw || counts[r] <= 0 || work.weak_h.empty() || norm <= 0.0) return;

            SmallRng rng(sample_seed(seed, dets[r]));
            work.sample_freq.assign(work.weak_h.size(), 0);
            work.sample_touched.clear();
            for (i64 k = 0; k < counts[r]; ++k) {
                const double u = rng.uniform01() * norm;
                auto it = std::lower_bound(work.weak_cdf.begin(), work.weak_cdf.end(), u);
                if (it == work.weak_cdf.end()) it = work.weak_cdf.end() - 1;
                const i32 idx = static_cast<i32>(std::distance(work.weak_cdf.begin(), it));
                if (work.sample_freq[static_cast<std::size_t>(idx)] == 0) work.sample_touched.push_back(idx);
                ++work.sample_freq[static_cast<std::size_t>(idx)];
            }
            for (i32 idx : work.sample_touched) {
                const std::size_t pos = static_cast<std::size_t>(idx);
                rows.push_back(static_cast<i32>(r));
                append_det(words, det_at(work.weak_words, nword_, pos));
                hs.push_back(work.weak_h[pos]);
                pgens.push_back(std::abs(work.weak_h[pos]) / norm);
                cts.push_back(work.sample_freq[pos]);
            }
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(dets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);
                process_one(r, work, local_row[static_cast<std::size_t>(tid)], local_det[static_cast<std::size_t>(tid)],
                            local_h[static_cast<std::size_t>(tid)], local_pgen[static_cast<std::size_t>(tid)], local_counts[static_cast<std::size_t>(tid)]);
            }
        }
#else
        {
            RowWork work(nword_, ints_.norb());
            for (std::size_t r = 0; r < dets.n_dets; ++r) process_one(r, work, local_row[0], local_det[0], local_h[0], local_pgen[0], local_counts[0]);
        }
#endif
        for (std::size_t t = 0; t < local_row.size(); ++t) {
            out.rows_mut().insert(out.rows_mut().end(), local_row[t].begin(), local_row[t].end());
            out.det_words_mut().insert(out.det_words_mut().end(), local_det[t].begin(), local_det[t].end());
            out.h_mut().insert(out.h_mut().end(), local_h[t].begin(), local_h[t].end());
            out.pgen_mut().insert(out.pgen_mut().end(), local_pgen[t].begin(), local_pgen[t].end());
            out.counts_mut().insert(out.counts_mut().end(), local_counts[t].begin(), local_counts[t].end());
        }
        return out;
    }

    struct ShellRec {
        i32 rep = 0;
        std::size_t det = 0;
        double a = 0.0;
        double b = 0.0;
    };

    struct ShellLocal {
        std::vector<ShellRec> rec;
        std::vector<u64> words;
    };

    [[nodiscard]] ShellSamples sample_shell_impl(DetBatchView kets, std::span<const double> coeffs,
                                                 double eps1, double eps2, std::span<const i64> counts,
                                                 DetBatchView exclude, i64 n_rep, u64 seed) const {
        const DetIndex exclude_index(exclude);
        const HeatBathTable* hb = heatbath(eps2);
#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif
        std::vector<ShellLocal> local(static_cast<std::size_t>(nthread));

        auto process_row = [&](std::size_t r, RowWork& work, ShellLocal& loc) {
            const i64 n_draw = counts[r];
            const double c = coeffs[r];
            const double scale = std::abs(c);
            if (n_draw <= 0 || scale <= 0.0) return;

            const double lo = row_cut(eps2, scale);
            const double hi = row_cut(eps1, scale);
            work.weak_words.clear();
            work.weak_h.clear();
            work.weak_cdf.clear();
            double norm = 0.0;
            enumerate_window(ints_, hb, kets[r], work, lo, hi, [&](DetRef det, double h) {
                if (exclude_index.find(det) >= 0) return;
                append_det(work.weak_words, det);
                work.weak_h.push_back(h);
                norm += std::abs(h);
                work.weak_cdf.push_back(norm);
            });
            if (work.weak_h.empty() || norm <= 0.0) return;

            for (i64 rep = 0; rep < n_rep; ++rep) {
                for (int pair = 0; pair < 2; ++pair) {
                    SmallRng rng(sample_seed(seed, kets[r], rep, pair));
                    work.sample_freq.assign(work.weak_h.size(), 0);
                    work.sample_touched.clear();
                    for (i64 k = 0; k < n_draw; ++k) {
                        const double u = rng.uniform01() * norm;
                        auto it = std::lower_bound(work.weak_cdf.begin(), work.weak_cdf.end(), u);
                        if (it == work.weak_cdf.end()) it = work.weak_cdf.end() - 1;
                        const i32 idx = static_cast<i32>(std::distance(work.weak_cdf.begin(), it));
                        if (work.sample_freq[static_cast<std::size_t>(idx)] == 0) work.sample_touched.push_back(idx);
                        ++work.sample_freq[static_cast<std::size_t>(idx)];
                    }
                    for (i32 idx : work.sample_touched) {
                        const std::size_t pos = static_cast<std::size_t>(idx);
                        const double h = work.weak_h[pos];
                        const double delta = static_cast<double>(work.sample_freq[pos]) * c * h * norm /
                                             (static_cast<double>(n_draw) * std::abs(h));
                        ShellRec rec;
                        rec.rep = static_cast<i32>(rep);
                        rec.det = append_det_index(loc.words, nword_, det_at(work.weak_words, nword_, pos));
                        if (pair == 0) rec.a = delta;
                        else rec.b = delta;
                        loc.rec.push_back(rec);
                    }
                }
            }
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(kets.n_dets); ++rr) {
                process_row(static_cast<std::size_t>(rr), work, local[static_cast<std::size_t>(tid)]);
            }
        }
#else
        {
            RowWork work(nword_, ints_.norb());
            for (std::size_t r = 0; r < kets.n_dets; ++r) process_row(r, work, local[0]);
        }
#endif

        std::vector<ShellRec> records;
        std::vector<u64> record_words;
        std::size_t nrec = 0;
        std::size_t nwords = 0;
        for (const auto& loc : local) {
            nrec += loc.rec.size();
            nwords += loc.words.size();
        }
        records.reserve(nrec);
        record_words.reserve(nwords);
        for (auto& loc : local) {
            const std::size_t det_offset = record_words.size() / det_size(nword_);
            record_words.insert(record_words.end(), loc.words.begin(), loc.words.end());
            for (ShellRec rec : loc.rec) {
                rec.det += det_offset;
                records.push_back(rec);
            }
        }

        std::sort(records.begin(), records.end(), [&](const ShellRec& x, const ShellRec& y) {
            if (x.rep != y.rep) return x.rep < y.rep;
            return DetLess{}(det_at(record_words, nword_, x.det), det_at(record_words, nword_, y.det));
        });

        ShellSamples out;
        out.set_nword(nword_);
        out.rep_ptr_mut().assign(static_cast<std::size_t>(n_rep + 1), 0);
        std::vector<u64> global_pool;
        global_pool.reserve(records.size() * det_size(nword_));

        std::size_t i = 0;
        std::size_t pos = 0;
        i64 current_rep = 0;
        while (i < records.size()) {
            const i32 rep = records[i].rep;
            while (current_rep <= rep && current_rep <= n_rep) {
                out.rep_ptr_mut()[static_cast<std::size_t>(current_rep)] = to_i32(pos);
                ++current_rep;
            }
            const DetRef det = det_at(record_words, nword_, records[i].det);
            double ha = 0.0;
            double hbv = 0.0;
            std::size_t j = i;
            while (j < records.size() && records[j].rep == rep && det_equal(det_at(record_words, nword_, records[j].det), det)) {
                ha += records[j].a;
                hbv += records[j].b;
                ++j;
            }
            append_det(out.det_words_mut(), det);
            append_det(global_pool, det);
            out.hpsi_a_mut().push_back(ha);
            out.hpsi_b_mut().push_back(hbv);
            ++pos;
            i = j;
        }
        while (current_rep <= n_rep) {
            out.rep_ptr_mut()[static_cast<std::size_t>(current_rep)] = to_i32(pos);
            ++current_rep;
        }

        out.diags_mut().assign(pos, 0.0);
        out.hpsi_strong_mut().assign(pos, 0.0);
        if (pos == 0) return out;

        sort_unique_dets(global_pool, nword_);
        const std::size_t target_n = global_pool.size() / det_size(nword_);
        const DetBatchView target{global_pool.data(), target_n, nword_};
        Projection strong = project_impl(target, kets, coeffs, eps1);
        const DetBatchView strong_view{strong.bra_words().data(), strong.n_bras(), nword_};
        const DetIndex strong_index(strong_view);
        const DetBatchView out_view{out.det_words().data(), out.n_samples(), nword_};
        for (std::size_t k = 0; k < out.n_samples(); ++k) {
            const i32 idx = strong_index.find(out_view[k]);
            if (idx >= 0) {
                out.diags_mut()[k] = strong.diags()[static_cast<std::size_t>(idx)];
                out.hpsi_strong_mut()[k] = strong.hpsi()[static_cast<std::size_t>(idx)];
            }
        }
        return out;
    }
};

} // namespace libdet
