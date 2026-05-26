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

#include <libdet/excite.hpp>
#include <libdet/heatbath.hpp>
#include <libdet/sample.hpp>
#include <libdet/space.hpp>

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

    void resize(std::size_t n_bra, std::size_t n_ket) noexcept {
        n_bra_ = n_bra;
        n_ket_ = n_ket;
    }

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

class DetIndex {
public:
    DetIndex() = default;
    explicit DetIndex(DetBatchView dets) { build(dets); }

    void build(DetBatchView dets) {
        dets_ = dets;

        std::size_t cap = 8;
        while (cap < dets.n_dets * 2u + 1u) cap <<= 1u;

        slot_.assign(cap, -1);
        mask_ = cap - 1u;

        for (std::size_t i = 0; i < dets.n_dets; ++i) {
            insert_existing(static_cast<i32>(i));
        }
    }

    [[nodiscard]] i32 find(DetRef det) const noexcept {
        if (slot_.empty()) return -1;

        std::size_t s = DetHash{}(det) & mask_;

        for (;;) {
            const i32 idx = slot_[s];

            if (idx < 0) return -1;
            if (det_equal(dets_[static_cast<std::size_t>(idx)], det)) return idx;

            s = (s + 1u) & mask_;
        }
    }

private:
    DetBatchView dets_{};
    std::vector<i32> slot_;
    std::size_t mask_ = 0;

    void insert_existing(i32 idx) {
        const DetRef det = dets_[static_cast<std::size_t>(idx)];

        std::size_t s = DetHash{}(det) & mask_;

        while (slot_[s] >= 0) {
            s = (s + 1u) & mask_;
        }

        slot_[s] = idx;
    }
};

namespace detail_ham {

class DetPool {
public:
    explicit DetPool(u32 nword = 0)
        : nword_(nword) {
        rehash(8);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return nword_ == 0 ? 0u : words_.size() / det_size(nword_);
    }

    [[nodiscard]] DetRef get(std::size_t idx) const noexcept {
        return det_at(words_, nword_, idx);
    }

    [[nodiscard]] std::vector<u64>& words() noexcept { return words_; }
    [[nodiscard]] const std::vector<u64>& words() const noexcept { return words_; }

    [[nodiscard]] i32 find_or_add(DetRef det) {
        if ((size() + 1u) * 2u >= slot_.size()) {
            rehash(slot_.size() * 2u);
        }

        std::size_t s = DetHash{}(det) & mask_;

        for (;;) {
            const i32 idx = slot_[s];

            if (idx < 0) {
                const i32 fresh = to_i32(size());
                append_det(words_, det);
                slot_[s] = fresh;
                return fresh;
            }

            if (det_equal(get(static_cast<std::size_t>(idx)), det)) {
                return idx;
            }

            s = (s + 1u) & mask_;
        }
    }

private:
    u32 nword_ = 0;
    std::vector<u64> words_;

    std::vector<i32> slot_;
    std::size_t mask_ = 0;

    void rehash(std::size_t cap) {
        std::size_t n = 8;
        while (n < cap) n <<= 1u;

        slot_.assign(n, -1);
        mask_ = n - 1u;

        const std::size_t ndet = size();

        for (std::size_t idx = 0; idx < ndet; ++idx) {
            const DetRef det = get(idx);
            std::size_t s = DetHash{}(det) & mask_;

            while (slot_[s] >= 0) {
                s = (s + 1u) & mask_;
            }

            slot_[s] = static_cast<i32>(idx);
        }
    }
};

struct EdgeRec {
    i32 row = 0;
    i32 col = 0;
    double h = 0.0;
};

struct ThreadEdges {
    explicit ThreadEdges(u32 nword = 0)
        : pool(nword) {}

    DetPool pool;
    std::vector<EdgeRec> edges;
};

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

} // namespace detail_ham

class Hamiltonian {
public:
    Hamiltonian() = delete;

    Hamiltonian(const Hamiltonian& other)
        : ints_(other.ints_),
          nword_(other.nword_) {}

    Hamiltonian& operator=(const Hamiltonian& other) {
        if (this != &other) {
            ints_ = other.ints_;
            nword_ = other.nword_;
            hb_.reset();
        }

        return *this;
    }

    Hamiltonian(Hamiltonian&& other) noexcept
        : ints_(std::move(other.ints_)),
          nword_(other.nword_),
          hb_(std::move(other.hb_)) {}

    Hamiltonian& operator=(Hamiltonian&& other) noexcept {
        if (this != &other) {
            ints_ = std::move(other.ints_);
            nword_ = other.nword_;
            hb_ = std::move(other.hb_);
        }

        return *this;
    }

    [[nodiscard]] static Hamiltonian make(
        std::span<const double> h1,
        int norb,
        std::span<const double> eri,
        double ecore = 0.0
    ) {
        return Hamiltonian(RHFIntegrals(norb, h1, eri, ecore));
    }

    [[nodiscard]] int norb() const noexcept { return ints_.norb(); }
    [[nodiscard]] u32 nword() const noexcept { return nword_; }

    [[nodiscard]] double hij(DetRef bra, DetRef ket) const {
        check_one(bra, "hij(bra)");
        check_one(ket, "hij(ket)");

        const Excitation ex = diff(bra, ket);

        if (ex.deg > 2) return 0.0;

        if (ex.deg == 0) {
            return Slater::diag(ints_, bra);
        }

        if (ex.deg == 1) {
            return ex.na == 1
                ? ex.sign * Slater::single_a(ints_, bra, ex.occ_a[0], ex.vir_a[0])
                : ex.sign * Slater::single_b(ints_, bra, ex.occ_b[0], ex.vir_b[0]);
        }

        if (ex.na == 2) {
            return ex.sign
                * Slater::double_aa(ints_, ex.occ_a[0], ex.occ_a[1], ex.vir_a[0], ex.vir_a[1]);
        }

        if (ex.nb == 2) {
            return ex.sign
                * Slater::double_bb(ints_, ex.occ_b[0], ex.occ_b[1], ex.vir_b[0], ex.vir_b[1]);
        }

        return ex.sign
            * Slater::double_ab(ints_, ex.occ_a[0], ex.occ_b[0], ex.vir_a[0], ex.vir_b[0]);
    }

    [[nodiscard]] std::vector<double> diags(DetBatchView dets) const {
        check_dets(dets, "diags");

        std::vector<double> out(dets.n_dets, 0.0);

        for_static(dets.n_dets, [&](std::size_t r, RowWork& work) {
            fill_occ(dets[r], ints_.norb(), work.occ);
            out[r] = Slater::diag(ints_, work.occ);
        });

        return out;
    }

    [[nodiscard]] Determinants expand(
        DetBatchView kets,
        double eps,
        std::span<const double> coeffs = {},
        const DetBatchView* exclude = nullptr
    ) const {
        check_dets(kets, "expand(kets)");
        check_eps(eps);

        if (!coeffs.empty() && coeffs.size() != kets.n_dets) {
            throw std::invalid_argument("expand: coeffs size must match kets");
        }

        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "expand(exclude)");

        return expand_impl(kets, eps, coeffs, base);
    }

    [[nodiscard]] Projection project(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps = 0.0
    ) const {
        check_dets(bras, "project(bras)");
        check_dets(kets, "project(kets)");

        if (coeffs.size() != kets.n_dets) {
            throw std::invalid_argument("project: coeffs size must match kets");
        }

        check_eps(eps);

        return project_impl(bras, kets, coeffs, eps);
    }

    [[nodiscard]] Edges edges(DetBatchView dets, double eps) const {
        check_dets(dets, "edges");
        check_eps(eps);

        return edges_impl(dets, eps);
    }

    [[nodiscard]] Degrees degrees(DetBatchView dets, double eps) const {
        check_dets(dets, "degrees");
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

    [[nodiscard]] std::vector<double> matvec(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x
    ) const {
        check_dets(bras, "matvec(bras)");
        check_dets(kets, "matvec(kets)");

        if (x.size() != kets.n_dets) {
            throw std::invalid_argument("matvec: x size must match kets");
        }

        const DetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets, 0.0);

        for_space(bras.n_dets, [&](std::size_t r, SpaceWork& work) {
            const DetRef bra = bras[r];

            double acc = 0.0;

            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) {
                acc += Slater::diag(ints_, bra) * x[static_cast<std::size_t>(diag_idx)];
            }

            scan_space(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                acc += h * x[static_cast<std::size_t>(idx)];
            });

            out[r] = acc;
        });

        return out;
    }

    [[nodiscard]] std::vector<double> matmat(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> x,
        std::size_t nrhs
    ) const {
        check_dets(bras, "matmat(bras)");
        check_dets(kets, "matmat(kets)");

        if (x.size() != kets.n_dets * nrhs) {
            throw std::invalid_argument("matmat: X size must be n_ket * n_rhs");
        }

        const DetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets * nrhs, 0.0);

        for_space(bras.n_dets, [&](std::size_t r, SpaceWork& work) {
            const DetRef bra = bras[r];
            double* yr = out.data() + r * nrhs;

            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) {
                const double h = Slater::diag(ints_, bra);
                const double* xr = x.data() + static_cast<std::size_t>(diag_idx) * nrhs;

                for (std::size_t j = 0; j < nrhs; ++j) {
                    yr[j] += h * xr[j];
                }
            }

            scan_space(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                const double* xr = x.data() + static_cast<std::size_t>(idx) * nrhs;

                for (std::size_t j = 0; j < nrhs; ++j) {
                    yr[j] += h * xr[j];
                }
            });
        });

        return out;
    }

    [[nodiscard]] EdgeSamples sample_edges(
        DetBatchView dets,
        std::span<const i64> counts,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const {
        check_dets(dets, "sample_edges");
        check_window_eps(eps1, eps2);

        if (!counts.empty() && counts.size() != dets.n_dets) {
            throw std::invalid_argument("sample_edges: counts size must match dets");
        }

        return sample_edges_impl(dets, counts, eps1, eps2, seed);
    }

    [[nodiscard]] ShellSamples sample_shell(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        const DetBatchView* exclude = nullptr,
        i64 n_rep = 1,
        u64 seed = 0
    ) const {
        check_dets(kets, "sample_shell(kets)");

        if (coeffs.size() != kets.n_dets) {
            throw std::invalid_argument("sample_shell: coeffs size must match kets");
        }

        if (counts.size() != kets.n_dets) {
            throw std::invalid_argument("sample_shell: counts size must match kets");
        }

        if (n_rep <= 0) {
            throw std::invalid_argument("sample_shell: n_rep must be positive");
        }

        check_window_eps(eps1, eps2);

        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "sample_shell(exclude)");

        return sample_shell_impl(kets, coeffs, eps1, eps2, counts, base, n_rep, seed);
    }

private:
    explicit Hamiltonian(RHFIntegrals ints)
        : ints_(std::move(ints)),
          nword_(bits::words_for(ints_.norb())) {}

    RHFIntegrals ints_;
    u32 nword_ = 0;

    mutable std::mutex hb_mutex_;
    mutable std::shared_ptr<const HeatBathTable> hb_;

    void check_one(DetRef det, const char* where) const {
        if (det.nword() != nword_) {
            throw std::invalid_argument(std::string(where) + ": determinant nword mismatch");
        }
    }

    void check_dets(DetBatchView dets, const char* where) const {
        if (dets.nword != nword_) {
            throw std::invalid_argument(std::string(where) + ": determinant nword mismatch");
        }
    }

    static void check_eps(double eps) {
        if (std::isnan(eps)) throw std::invalid_argument("eps must not be NaN");
        if (eps < 0.0) throw std::invalid_argument("eps must be nonnegative");
    }

    static void check_window_eps(double eps1, double eps2) {
        if (std::isnan(eps1)) throw std::invalid_argument("eps1 must not be NaN");
        if (std::isnan(eps2)) throw std::invalid_argument("eps2 must not be NaN");
        if (eps1 < 0.0) throw std::invalid_argument("eps1 must be nonnegative");
        if (eps2 < 0.0) throw std::invalid_argument("eps2 must be nonnegative");
        if (eps2 > eps1) throw std::invalid_argument("eps2 must be <= eps1");
    }

    [[nodiscard]] const HeatBathTable& heatbath() const {
        std::lock_guard<std::mutex> lock(hb_mutex_);

        if (!hb_) {
            hb_ = std::make_shared<HeatBathTable>(ints_);
        }

        return *hb_;
    }

    [[nodiscard]] static double row_cut(double eps, double scale) noexcept {
        if (eps <= 0.0) return 0.0;
        if (scale <= 0.0) return std::numeric_limits<double>::infinity();
        return eps / scale;
    }

    template <class F>
    void for_static(std::size_t nrow, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            RowWork work(nword_, ints_.norb());

#pragma omp for schedule(static)
            for (i64 r = 0; r < static_cast<i64>(nrow); ++r) {
                f(static_cast<std::size_t>(r), work);
            }
        }
#else
        RowWork work(nword_, ints_.norb());

        for (std::size_t r = 0; r < nrow; ++r) {
            f(r, work);
        }
#endif
    }

    template <class F>
    void for_guided(std::size_t nrow, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            RowWork work(nword_, ints_.norb());

#pragma omp for schedule(guided)
            for (i64 r = 0; r < static_cast<i64>(nrow); ++r) {
                f(static_cast<std::size_t>(r), work);
            }
        }
#else
        RowWork work(nword_, ints_.norb());

        for (std::size_t r = 0; r < nrow; ++r) {
            f(r, work);
        }
#endif
    }

    template <class F>
    void for_space(std::size_t nrow, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            SpaceWork work;

#pragma omp for schedule(guided)
            for (i64 r = 0; r < static_cast<i64>(nrow); ++r) {
                f(static_cast<std::size_t>(r), work);
            }
        }
#else
        SpaceWork work;

        for (std::size_t r = 0; r < nrow; ++r) {
            f(r, work);
        }
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

        for (auto& p : parts) {
            out.det_words_mut().insert(out.det_words_mut().end(), p.begin(), p.end());
        }

        sort_unique_dets(out.det_words_mut(), nword_);

        return out;
    }

    [[nodiscard]] Determinants expand_impl(
        DetBatchView kets,
        double eps,
        std::span<const double> coeffs,
        DetBatchView exclude
    ) const {
        const HeatBathTable& hb = heatbath();
        const DetIndex exclude_index(exclude);

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

                scan_edges(ints_, hb, kets[r], work, cut, [&](DetRef det, double h) {
                    if (exclude_index.find(det) >= 0) return;
                    if (std::abs(h) * scale >= eps) append_det(words, det);
                });
            }
        }

        return merge_det_parts(local);
    }

    [[nodiscard]] Projection project_impl(
        DetBatchView bras,
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps
    ) const {
        const DetSpace ket_space(kets);

        Projection out;
        out.set_nword(nword_);
        copy_batch(out.bra_words_mut(), bras);
        out.hpsi_mut().assign(bras.n_dets, 0.0);
        out.diags_mut().assign(bras.n_dets, 0.0);

        for_space(bras.n_dets, [&](std::size_t r, SpaceWork& work) {
            const DetRef bra = bras[r];

            const double diag = Slater::diag(ints_, bra);

            out.diags_mut()[r] = diag;

            double acc = 0.0;

            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) {
                const double v = diag * coeffs[static_cast<std::size_t>(diag_idx)];

                if (std::abs(v) >= eps) acc += v;
            }

            scan_space(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                const double v = h * coeffs[static_cast<std::size_t>(idx)];

                if (std::abs(v) >= eps) acc += v;
            });

            out.hpsi_mut()[r] = acc;
        });

        return out;
    }

    [[nodiscard]] Edges edges_impl(DetBatchView dets, double eps) const {
        const HeatBathTable& hb = heatbath();

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::ThreadEdges> local;
        local.reserve(static_cast<std::size_t>(nthread));

        for (int t = 0; t < nthread; ++t) {
            local.emplace_back(nword_);
        }

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& loc = local[static_cast<std::size_t>(tid)];
            RowWork work(nword_, ints_.norb());

#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(dets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);
#else
        {
            auto& loc = local[0];
            RowWork work(nword_, ints_.norb());

            for (std::size_t r = 0; r < dets.n_dets; ++r) {
#endif
                scan_edges(ints_, hb, dets[r], work, eps, [&](DetRef det, double h) {
                    const i32 col = loc.pool.find_or_add(det);
                    loc.edges.push_back(detail_ham::EdgeRec{static_cast<i32>(r), col, h});
                });
            }
        }

        detail_ham::DetPool global(nword_);

        for (std::size_t r = 0; r < dets.n_dets; ++r) {
            global.find_or_add(dets[r]);
        }

        std::vector<std::vector<i32>> remap(local.size());

        for (std::size_t t = 0; t < local.size(); ++t) {
            remap[t].resize(local[t].pool.size());

            for (std::size_t c = 0; c < local[t].pool.size(); ++c) {
                remap[t][c] = global.find_or_add(local[t].pool.get(c));
            }
        }

        Edges out;
        out.set_nword(nword_);
        out.set_n_rows(dets.n_dets);

        copy_batch(out.row_words_mut(), dets);
        out.col_words_mut() = global.words();

        out.diags_mut() = diags(dets);
        out.row_ptr_mut().assign(dets.n_dets + 1u, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);
        out.row_nnz_mut().assign(dets.n_dets, 0);

        std::size_t nedge = 0;

        for (std::size_t t = 0; t < local.size(); ++t) {
            nedge += local[t].edges.size();

            for (const auto& e : local[t].edges) {
                ++out.row_ptr_mut()[static_cast<std::size_t>(e.row) + 1u];
                out.row_weight_mut()[static_cast<std::size_t>(e.row)] += std::abs(e.h);
                ++out.row_nnz_mut()[static_cast<std::size_t>(e.row)];
            }
        }

        std::size_t acc = 0;
        for (std::size_t r = 0; r < dets.n_dets; ++r) {
            acc += static_cast<std::size_t>(out.row_ptr_mut()[r + 1u]);
            out.row_ptr_mut()[r + 1u] = to_i32(acc);
        }

        out.col_mut().resize(nedge);
        out.h_mut().resize(nedge);

        std::vector<i32> pos = out.row_ptr_mut();

        for (std::size_t t = 0; t < local.size(); ++t) {
            for (const auto& e : local[t].edges) {
                const std::size_t row = static_cast<std::size_t>(e.row);
                const std::size_t p = static_cast<std::size_t>(pos[row]++);

                out.col_mut()[p] =
                    remap[t][static_cast<std::size_t>(e.col)];

                out.h_mut()[p] = e.h;
            }
        }

        return out;
    }

    [[nodiscard]] Degrees degrees_impl(DetBatchView dets, double eps) const {
        const HeatBathTable& hb = heatbath();

        Degrees out;
        out.row_nnz_mut().assign(dets.n_dets, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);

        for_guided(dets.n_dets, [&](std::size_t r, RowWork& work) {
            i64 nnz = 0;
            double weight = 0.0;

            scan_values(ints_, hb, dets[r], work, eps, [&](double h) {
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

        for_space(nrow, [&](std::size_t r, SpaceWork& work) {
            const DetRef bra = bras[r];

            i32 nnz = 0;

            const double diag = Slater::diag(ints_, bra);
            out.diags_mut()[r] = diag;

            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0 && diag != 0.0) ++nnz;

            scan_space(ints_, ket_space, bra, work, [&](i32, double) {
                ++nnz;
            });

            out.row_ptr_mut()[r + 1u] = nnz;
        });

        std::size_t nnz = 0;

        for (std::size_t r = 0; r < nrow; ++r) {
            nnz += static_cast<std::size_t>(out.row_ptr_mut()[r + 1u]);
            out.row_ptr_mut()[r + 1u] = to_i32(nnz);
        }

        out.col_mut().resize(nnz);
        out.h_mut().resize(nnz);

        for_space(nrow, [&](std::size_t r, SpaceWork& work) {
            const DetRef bra = bras[r];

            std::size_t pos = static_cast<std::size_t>(out.row_ptr_mut()[r]);

            const double diag = out.diags_mut()[r];
            const i32 diag_idx = find_det(ket_space, bra);

            if (diag_idx >= 0 && diag != 0.0) {
                out.col_mut()[pos] = diag_idx;
                out.h_mut()[pos] = diag;
                ++pos;
            }

            scan_space(ints_, ket_space, bra, work, [&](i32 idx, double h) {
                out.col_mut()[pos] = idx;
                out.h_mut()[pos] = h;
                ++pos;
            });
        });
    }

    [[nodiscard]] EdgeSamples sample_edges_impl(
        DetBatchView dets,
        std::span<const i64> counts,
        double eps1,
        double eps2,
        u64 seed
    ) const {
        const HeatBathTable& hb = heatbath();
        const bool draw = !counts.empty();

        EdgeSamples out;
        out.set_nword(nword_);
        out.row_nnz_mut().assign(dets.n_dets, 0);
        out.row_weight_mut().assign(dets.n_dets, 0.0);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<std::vector<i32>> local_rows(static_cast<std::size_t>(nthread));
        std::vector<std::vector<u64>> local_words(static_cast<std::size_t>(nthread));
        std::vector<std::vector<double>> local_h(static_cast<std::size_t>(nthread));
        std::vector<std::vector<double>> local_pgen(static_cast<std::size_t>(nthread));
        std::vector<std::vector<i64>> local_counts(static_cast<std::size_t>(nthread));

        auto process_one = [&](
            std::size_t r,
            RowWork& work,
            std::vector<double>& targets,
            std::vector<i32>& rows,
            std::vector<u64>& words,
            std::vector<double>& hs,
            std::vector<double>& pgens,
            std::vector<i64>& cts
        ) {
            i64 nnz = 0;
            double norm = 0.0;

            scan_window_values(ints_, hb, dets[r], work, eps2, eps1, [&](double h) {
                ++nnz;
                norm += std::abs(h);
            });

            out.row_nnz_mut()[r] = nnz;
            out.row_weight_mut()[r] = norm;

            if (!draw || counts[r] <= 0 || nnz == 0 || !(norm > 0.0)) {
                return;
            }

            SmallRng rng(sample_seed(seed, dets[r]));
            make_targets(rng, counts[r], norm, targets);

            if (targets.empty()) return;

            std::size_t target_pos = 0;
            double cdf = 0.0;

            scan_window_edges(ints_, hb, dets[r], work, eps2, eps1, [&](DetRef det, double h) {
                const double ah = std::abs(h);
                cdf += ah;

                i64 hit = 0;

                while (target_pos < targets.size() && targets[target_pos] <= cdf) {
                    ++hit;
                    ++target_pos;
                }

                if (hit <= 0) return;

                rows.push_back(static_cast<i32>(r));
                append_det(words, det);
                hs.push_back(h);
                pgens.push_back(ah / norm);
                cts.push_back(hit);
            });
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(dets.n_dets); ++rr) {
                const std::size_t r = static_cast<std::size_t>(rr);

                process_one(
                    r,
                    work,
                    targets,
                    local_rows[static_cast<std::size_t>(tid)],
                    local_words[static_cast<std::size_t>(tid)],
                    local_h[static_cast<std::size_t>(tid)],
                    local_pgen[static_cast<std::size_t>(tid)],
                    local_counts[static_cast<std::size_t>(tid)]
                );
            }
        }
#else
        {
            RowWork work(nword_, ints_.norb());
            std::vector<double> targets;

            for (std::size_t r = 0; r < dets.n_dets; ++r) {
                process_one(
                    r,
                    work,
                    targets,
                    local_rows[0],
                    local_words[0],
                    local_h[0],
                    local_pgen[0],
                    local_counts[0]
                );
            }
        }
#endif

        for (std::size_t t = 0; t < local_rows.size(); ++t) {
            out.rows_mut().insert(out.rows_mut().end(), local_rows[t].begin(), local_rows[t].end());
            out.det_words_mut().insert(out.det_words_mut().end(), local_words[t].begin(), local_words[t].end());
            out.h_mut().insert(out.h_mut().end(), local_h[t].begin(), local_h[t].end());
            out.pgen_mut().insert(out.pgen_mut().end(), local_pgen[t].begin(), local_pgen[t].end());
            out.counts_mut().insert(out.counts_mut().end(), local_counts[t].begin(), local_counts[t].end());
        }

        return out;
    }

    [[nodiscard]] ShellSamples sample_shell_impl(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        DetBatchView exclude,
        i64 n_rep,
        u64 seed
    ) const {
        const HeatBathTable& hb = heatbath();
        const DetIndex exclude_index(exclude);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::ShellLocal> local(static_cast<std::size_t>(nthread));

        auto process_row = [&](
            std::size_t r,
            RowWork& work,
            std::vector<double>& targets,
            detail_ham::ShellLocal& loc
        ) {
            const i64 n_draw = counts[r];
            const double c = coeffs[r];
            const double scale = std::abs(c);

            if (n_draw <= 0 || scale <= 0.0) return;

            const double lo = row_cut(eps2, scale);
            const double hi = row_cut(eps1, scale);

            if (!std::isfinite(lo) || hi <= lo) return;

            double norm = 0.0;

            scan_window_edges(ints_, hb, kets[r], work, lo, hi, [&](DetRef det, double h) {
                if (exclude_index.find(det) >= 0) return;
                norm += std::abs(h);
            });

            if (!(norm > 0.0)) return;

            for (i64 rep = 0; rep < n_rep; ++rep) {
                for (int pair = 0; pair < 2; ++pair) {
                    SmallRng rng(sample_seed(seed, kets[r], rep, pair));
                    make_targets(rng, n_draw, norm, targets);

                    if (targets.empty()) continue;

                    std::size_t target_pos = 0;
                    double cdf = 0.0;

                    scan_window_edges(ints_, hb, kets[r], work, lo, hi, [&](DetRef det, double h) {
                        if (exclude_index.find(det) >= 0) return;

                        const double ah = std::abs(h);
                        cdf += ah;

                        i64 hit = 0;

                        while (target_pos < targets.size() && targets[target_pos] <= cdf) {
                            ++hit;
                            ++target_pos;
                        }

                        if (hit <= 0) return;

                        const double delta =
                            static_cast<double>(hit) * c * h * norm
                            / (static_cast<double>(n_draw) * ah);

                        detail_ham::ShellRec rec;
                        rec.rep = static_cast<i32>(rep);
                        rec.det = append_det_index(loc.words, nword_, det);

                        if (pair == 0) rec.a = delta;
                        else rec.b = delta;

                        loc.rec.push_back(rec);
                    });
                }
            }
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            RowWork work(nword_, ints_.norb());
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 rr = 0; rr < static_cast<i64>(kets.n_dets); ++rr) {
                process_row(
                    static_cast<std::size_t>(rr),
                    work,
                    targets,
                    local[static_cast<std::size_t>(tid)]
                );
            }
        }
#else
        {
            RowWork work(nword_, ints_.norb());
            std::vector<double> targets;

            for (std::size_t r = 0; r < kets.n_dets; ++r) {
                process_row(r, work, targets, local[0]);
            }
        }
#endif

        std::vector<detail_ham::ShellRec> records;
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
            const std::size_t offset = record_words.size() / det_size(nword_);

            record_words.insert(record_words.end(), loc.words.begin(), loc.words.end());

            for (auto rec : loc.rec) {
                rec.det += offset;
                records.push_back(rec);
            }
        }

        std::sort(records.begin(), records.end(), [&](const auto& x, const auto& y) {
            if (x.rep != y.rep) return x.rep < y.rep;

            return DetLess{}(
                det_at(record_words, nword_, x.det),
                det_at(record_words, nword_, y.det)
            );
        });

        ShellSamples out;
        out.set_nword(nword_);
        out.rep_ptr_mut().assign(static_cast<std::size_t>(n_rep + 1), 0);

        std::vector<u64> global_pool;

        std::size_t i = 0;
        std::size_t pos = 0;

        for (i64 rep = 0; rep < n_rep; ++rep) {
            out.rep_ptr_mut()[static_cast<std::size_t>(rep)] = to_i32(pos);

            while (i < records.size() && records[i].rep < rep) {
                ++i;
            }

            while (i < records.size() && records[i].rep == rep) {
                const DetRef det = det_at(record_words, nword_, records[i].det);

                double ha = 0.0;
                double hbv = 0.0;

                std::size_t j = i;

                while (
                    j < records.size()
                    && records[j].rep == rep
                    && det_equal(det_at(record_words, nword_, records[j].det), det)
                ) {
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
        }

        out.rep_ptr_mut()[static_cast<std::size_t>(n_rep)] = to_i32(pos);

        out.diags_mut().assign(pos, 0.0);
        out.hpsi_strong_mut().assign(pos, 0.0);

        if (pos == 0) return out;

        sort_unique_dets(global_pool, nword_);

        const std::size_t target_n = global_pool.size() / det_size(nword_);
        const DetBatchView target{
            global_pool.data(),
            target_n,
            nword_
        };

        const Projection strong = project_impl(target, kets, coeffs, eps1);

        const DetBatchView strong_view{
            strong.bra_words().data(),
            strong.n_bras(),
            nword_
        };

        const DetIndex strong_index(strong_view);

        const DetBatchView out_view{
            out.det_words().data(),
            out.n_samples(),
            nword_
        };

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