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
#include <libdet/screen.hpp>
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
    [[nodiscard]] std::span<const i32> indptr() const noexcept { return indptr_; }
    [[nodiscard]] std::span<const i32> indices() const noexcept { return indices_; }
    [[nodiscard]] std::span<const double> data() const noexcept { return data_; }

    void resize(std::size_t n_bra, std::size_t n_ket) noexcept {
        n_bra_ = n_bra;
        n_ket_ = n_ket;
    }

    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<i32>& indptr_mut() noexcept { return indptr_; }
    [[nodiscard]] std::vector<i32>& indices_mut() noexcept { return indices_; }
    [[nodiscard]] std::vector<double>& data_mut() noexcept { return data_; }

private:
    std::size_t n_bra_ = 0;
    std::size_t n_ket_ = 0;

    std::vector<double> diags_;
    std::vector<i32> indptr_;
    std::vector<i32> indices_;
    std::vector<double> data_;
};

class Conns {
public:
    [[nodiscard]] std::size_t n_kets() const noexcept { return n_kets_; }

    [[nodiscard]] std::size_t n_bras() const noexcept {
        return nword_ == 0 ? 0u : bra_words_.size() / det_size(nword_);
    }

    [[nodiscard]] u32 nword() const noexcept { return nword_; }

    [[nodiscard]] std::span<const u64> ket_words() const noexcept { return ket_words_; }
    [[nodiscard]] std::span<const u64> bra_words() const noexcept { return bra_words_; }

    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }
    [[nodiscard]] std::span<const i32> ket_ptr() const noexcept { return ket_ptr_; }
    [[nodiscard]] std::span<const i32> bra() const noexcept { return bra_; }
    [[nodiscard]] std::span<const double> h() const noexcept { return h_; }
    [[nodiscard]] std::span<const double> ket_weight() const noexcept { return ket_weight_; }
    [[nodiscard]] std::span<const i64> ket_nconn() const noexcept { return ket_nconn_; }

    void set_nword(u32 x) noexcept { nword_ = x; }
    void set_n_kets(std::size_t x) noexcept { n_kets_ = x; }

    [[nodiscard]] std::vector<u64>& ket_words_mut() noexcept { return ket_words_; }
    [[nodiscard]] std::vector<u64>& bra_words_mut() noexcept { return bra_words_; }

    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<i32>& ket_ptr_mut() noexcept { return ket_ptr_; }
    [[nodiscard]] std::vector<i32>& bra_mut() noexcept { return bra_; }
    [[nodiscard]] std::vector<double>& h_mut() noexcept { return h_; }
    [[nodiscard]] std::vector<double>& ket_weight_mut() noexcept { return ket_weight_; }
    [[nodiscard]] std::vector<i64>& ket_nconn_mut() noexcept { return ket_nconn_; }

private:
    u32 nword_ = 0;
    std::size_t n_kets_ = 0;

    std::vector<u64> ket_words_;
    std::vector<u64> bra_words_;

    std::vector<double> diags_;
    std::vector<i32> ket_ptr_;
    std::vector<i32> bra_;
    std::vector<double> h_;

    std::vector<double> ket_weight_;
    std::vector<i64> ket_nconn_;
};

class Degrees {
public:
    [[nodiscard]] std::size_t n_kets() const noexcept { return ket_nconn_.size(); }

    [[nodiscard]] std::span<const i64> ket_nconn() const noexcept { return ket_nconn_; }
    [[nodiscard]] std::span<const double> ket_weight() const noexcept { return ket_weight_; }

    [[nodiscard]] std::vector<i64>& ket_nconn_mut() noexcept { return ket_nconn_; }
    [[nodiscard]] std::vector<double>& ket_weight_mut() noexcept { return ket_weight_; }

private:
    std::vector<i64> ket_nconn_;
    std::vector<double> ket_weight_;
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

class ConnSamples {
public:
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::size_t n_samples() const noexcept { return pgen_.size(); }

    void set_nword(u32 x) noexcept { nword_ = x; }

    [[nodiscard]] std::span<const i64> ket_nconn() const noexcept { return ket_nconn_; }
    [[nodiscard]] std::span<const double> ket_weight() const noexcept { return ket_weight_; }
    [[nodiscard]] std::span<const i32> ket() const noexcept { return ket_; }
    [[nodiscard]] std::span<const u64> bra_words() const noexcept { return bra_words_; }
    [[nodiscard]] std::span<const double> h() const noexcept { return h_; }
    [[nodiscard]] std::span<const double> pgen() const noexcept { return pgen_; }
    [[nodiscard]] std::span<const i64> counts() const noexcept { return counts_; }

    [[nodiscard]] std::vector<i64>& ket_nconn_mut() noexcept { return ket_nconn_; }
    [[nodiscard]] std::vector<double>& ket_weight_mut() noexcept { return ket_weight_; }
    [[nodiscard]] std::vector<i32>& ket_mut() noexcept { return ket_; }
    [[nodiscard]] std::vector<u64>& bra_words_mut() noexcept { return bra_words_; }
    [[nodiscard]] std::vector<double>& h_mut() noexcept { return h_; }
    [[nodiscard]] std::vector<double>& pgen_mut() noexcept { return pgen_; }
    [[nodiscard]] std::vector<i64>& counts_mut() noexcept { return counts_; }

private:
    u32 nword_ = 0;

    std::vector<i64> ket_nconn_;
    std::vector<double> ket_weight_;

    std::vector<i32> ket_;
    std::vector<u64> bra_words_;
    std::vector<double> h_;
    std::vector<double> pgen_;
    std::vector<i64> counts_;
};

class ProjectSamples {
public:
    [[nodiscard]] u32 nword() const noexcept { return nword_; }
    [[nodiscard]] std::size_t n_samples() const noexcept { return hpsi_a_.size(); }

    void set_nword(u32 x) noexcept { nword_ = x; }

    [[nodiscard]] std::span<const i32> rep_ptr() const noexcept { return rep_ptr_; }
    [[nodiscard]] std::span<const u64> bra_words() const noexcept { return bra_words_; }
    [[nodiscard]] std::span<const double> diags() const noexcept { return diags_; }
    [[nodiscard]] std::span<const double> hpsi_strong() const noexcept { return hpsi_strong_; }
    [[nodiscard]] std::span<const double> hpsi_a() const noexcept { return hpsi_a_; }
    [[nodiscard]] std::span<const double> hpsi_b() const noexcept { return hpsi_b_; }

    [[nodiscard]] std::vector<i32>& rep_ptr_mut() noexcept { return rep_ptr_; }
    [[nodiscard]] std::vector<u64>& bra_words_mut() noexcept { return bra_words_; }
    [[nodiscard]] std::vector<double>& diags_mut() noexcept { return diags_; }
    [[nodiscard]] std::vector<double>& hpsi_strong_mut() noexcept { return hpsi_strong_; }
    [[nodiscard]] std::vector<double>& hpsi_a_mut() noexcept { return hpsi_a_; }
    [[nodiscard]] std::vector<double>& hpsi_b_mut() noexcept { return hpsi_b_; }

private:
    u32 nword_ = 0;

    std::vector<i32> rep_ptr_;
    std::vector<u64> bra_words_;

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

        for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
            insert_existing(static_cast<i32>(idet));
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

        const std::size_t n_dets = size();

        for (std::size_t idet = 0; idet < n_dets; ++idet) {
            const DetRef det = get(idet);
            std::size_t s = DetHash{}(det) & mask_;

            while (slot_[s] >= 0) {
                s = (s + 1u) & mask_;
            }

            slot_[s] = static_cast<i32>(idet);
        }
    }
};

struct ConnRec {
    i32 iket = 0;
    i32 ibra = 0;
    double h = 0.0;
};

struct ConnPart {
    explicit ConnPart(u32 nword = 0)
        : pool(nword) {}

    DetPool pool;
    std::vector<ConnRec> conns;
};

struct ProjTerm {
    i32 ibra = 0;
    double hpsi = 0.0;
};

struct ProjPart {
    explicit ProjPart(u32 nword = 0)
        : pool(nword) {}

    DetPool pool;
    std::vector<ProjTerm> terms;
};

struct ConnSampleRec {
    i32 iket = 0;
    std::size_t ibra = 0;
    double h = 0.0;
    double pgen = 0.0;
    i64 count = 0;
};

struct ConnSamplePart {
    std::vector<ConnSampleRec> recs;
    std::vector<u64> words;
};

struct SampleRec {
    i32 rep = 0;
    std::size_t ibra = 0;
    double hpsi_a = 0.0;
    double hpsi_b = 0.0;
};

struct SamplePart {
    std::vector<SampleRec> recs;
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
            screen_.reset();
        }

        return *this;
    }

    Hamiltonian(Hamiltonian&& other) noexcept
        : ints_(std::move(other.ints_)),
          nword_(other.nword_),
          screen_(std::move(other.screen_)) {}

    Hamiltonian& operator=(Hamiltonian&& other) noexcept {
        if (this != &other) {
            ints_ = std::move(other.ints_);
            nword_ = other.nword_;
            screen_ = std::move(other.screen_);
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

        for_static(dets.n_dets, [&](std::size_t idet, KetWork& work) {
            fill_occ(dets[idet], ints_.norb(), work.occ);
            out[idet] = Slater::diag(ints_, work.occ);
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

    [[nodiscard]] Projection project(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps,
        const DetBatchView* exclude
    ) const {
        check_dets(kets, "project(kets)");

        if (coeffs.size() != kets.n_dets) {
            throw std::invalid_argument("project: coeffs size must match kets");
        }

        check_eps(eps);

        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "project(exclude)");

        return project_generated_impl(kets, coeffs, eps, base);
    }

    [[nodiscard]] Conns conns(DetBatchView kets, double eps) const {
        check_dets(kets, "conns(kets)");
        check_eps(eps);

        return conns_impl(kets, eps);
    }

    [[nodiscard]] Degrees degrees(DetBatchView kets, double eps) const {
        check_dets(kets, "degrees(kets)");
        check_eps(eps);

        return degrees_impl(kets, eps);
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

        const KetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets, 0.0);

        for_bras(bras.n_dets, [&](std::size_t ibra, BraWork& work) {
            const DetRef bra = bras[ibra];

            double acc = 0.0;

            const i32 diag_idx = find_ket(ket_space, bra);
            if (diag_idx >= 0) {
                acc += Slater::diag(ints_, bra) * x[static_cast<std::size_t>(diag_idx)];
            }

            scan_kets(ints_, ket_space, bra, work, [&](i32 iket, double h) {
                acc += h * x[static_cast<std::size_t>(iket)];
            });

            out[ibra] = acc;
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

        const KetSpace ket_space(kets);
        std::vector<double> out(bras.n_dets * nrhs, 0.0);

        for_bras(bras.n_dets, [&](std::size_t ibra, BraWork& work) {
            const DetRef bra = bras[ibra];
            double* y = out.data() + ibra * nrhs;

            const i32 diag_idx = find_ket(ket_space, bra);
            if (diag_idx >= 0) {
                const double h = Slater::diag(ints_, bra);
                const double* xrow = x.data() + static_cast<std::size_t>(diag_idx) * nrhs;

                for (std::size_t j = 0; j < nrhs; ++j) {
                    y[j] += h * xrow[j];
                }
            }

            scan_kets(ints_, ket_space, bra, work, [&](i32 iket, double h) {
                const double* xrow = x.data() + static_cast<std::size_t>(iket) * nrhs;

                for (std::size_t j = 0; j < nrhs; ++j) {
                    y[j] += h * xrow[j];
                }
            });
        });

        return out;
    }

    [[nodiscard]] ConnSamples sample_conns(
        DetBatchView kets,
        std::span<const i64> counts,
        double eps1,
        double eps2,
        u64 seed = 0
    ) const {
        check_dets(kets, "sample_conns(kets)");
        check_window_eps(eps1, eps2);

        if (!counts.empty() && counts.size() != kets.n_dets) {
            throw std::invalid_argument("sample_conns: counts size must match kets");
        }

        return sample_conns_impl(kets, counts, eps1, eps2, seed);
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
        check_dets(kets, "sample_project(kets)");

        if (coeffs.size() != kets.n_dets) {
            throw std::invalid_argument("sample_project: coeffs size must match kets");
        }

        if (counts.size() != kets.n_dets) {
            throw std::invalid_argument("sample_project: counts size must match kets");
        }

        if (n_rep <= 0) {
            throw std::invalid_argument("sample_project: n_rep must be positive");
        }

        check_window_eps(eps1, eps2);

        const DetBatchView base = exclude == nullptr ? kets : *exclude;
        check_dets(base, "sample_project(exclude)");

        return sample_project_impl(kets, coeffs, eps1, eps2, counts, base, n_rep, seed);
    }

private:
    explicit Hamiltonian(RHFIntegrals ints)
        : ints_(std::move(ints)),
          nword_(bits::words_for(ints_.norb())) {}

    RHFIntegrals ints_;
    u32 nword_ = 0;

    mutable std::mutex screen_mutex_;
    mutable std::shared_ptr<const Screen> screen_;

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

    [[nodiscard]] std::shared_ptr<const Screen> screen(double cutoff) const {
        if (cutoff <= 0.0 || !std::isfinite(cutoff)) return {};

        std::lock_guard<std::mutex> lock(screen_mutex_);

        if (!screen_ || cutoff < screen_->cutoff()) {
            screen_ = std::make_shared<Screen>(ints_, cutoff);
        }

        return screen_;
    }

    [[nodiscard]] static double max_abs(std::span<const double> values) noexcept {
        double out = 0.0;

        for (double x : values) {
            out = std::max(out, std::abs(x));
        }

        return out;
    }

    [[nodiscard]] static double screen_cutoff(double eps, double max_scale) noexcept {
        if (eps <= 0.0) return 0.0;
        if (max_scale <= 0.0) return std::numeric_limits<double>::infinity();
        return eps / max_scale;
    }

    [[nodiscard]] static EdgeWindow edge_window(
        double lo,
        double hi,
        double scale
    ) noexcept {
        return EdgeWindow{lo, hi, scale};
    }

    template <class F>
    void for_static(std::size_t n, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            KetWork work(nword_, ints_.norb());

#pragma omp for schedule(static)
            for (i64 k = 0; k < static_cast<i64>(n); ++k) {
                f(static_cast<std::size_t>(k), work);
            }
        }
#else
        KetWork work(nword_, ints_.norb());

        for (std::size_t k = 0; k < n; ++k) {
            f(k, work);
        }
#endif
    }

    template <class F>
    void for_guided(std::size_t n, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            KetWork work(nword_, ints_.norb());

#pragma omp for schedule(guided)
            for (i64 k = 0; k < static_cast<i64>(n); ++k) {
                f(static_cast<std::size_t>(k), work);
            }
        }
#else
        KetWork work(nword_, ints_.norb());

        for (std::size_t k = 0; k < n; ++k) {
            f(k, work);
        }
#endif
    }

    template <class F>
    void for_bras(std::size_t n, F&& f) const {
#if defined(_OPENMP)
#pragma omp parallel
        {
            BraWork work;

#pragma omp for schedule(guided)
            for (i64 k = 0; k < static_cast<i64>(n); ++k) {
                f(static_cast<std::size_t>(k), work);
            }
        }
#else
        BraWork work;

        for (std::size_t k = 0; k < n; ++k) {
            f(k, work);
        }
#endif
    }

    [[nodiscard]] Determinants merge_det_parts(std::vector<std::vector<u64>>& parts) const {
        std::size_t total = 0;

        for (auto& part : parts) {
            sort_unique_dets(part, nword_);
            total += part.size();
        }

        Determinants out;
        out.set_nword(nword_);
        out.det_words_mut().reserve(total);

        for (auto& part : parts) {
            out.det_words_mut().insert(out.det_words_mut().end(), part.begin(), part.end());
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
        const double max_scale = coeffs.empty() ? 1.0 : max_abs(coeffs);
        auto screen_ptr = screen(screen_cutoff(eps, max_scale));
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
            KetWork work(nword_, ints_.norb());
            auto& words = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
#else
        {
            KetWork work(nword_, ints_.norb());
            auto& words = local[0];

            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
                const double scale = coeffs.empty() ? 1.0 : std::abs(coeffs[iket]);
                const EdgeWindow win = edge_window(eps, std::numeric_limits<double>::infinity(), scale);

                scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double) {
                    if (exclude_index.find(bra) < 0) append_det(words, bra);
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
        const KetSpace ket_space(kets);

        Projection out;
        out.set_nword(nword_);
        copy_batch(out.bra_words_mut(), bras);
        out.hpsi_mut().assign(bras.n_dets, 0.0);
        out.diags_mut().assign(bras.n_dets, 0.0);

        for_bras(bras.n_dets, [&](std::size_t ibra, BraWork& work) {
            const DetRef bra = bras[ibra];

            const double diag = Slater::diag(ints_, bra);

            out.diags_mut()[ibra] = diag;

            double acc = 0.0;

            const i32 diag_idx = find_ket(ket_space, bra);
            if (diag_idx >= 0) {
                const double v = diag * coeffs[static_cast<std::size_t>(diag_idx)];

                if (std::abs(v) >= eps) acc += v;
            }

            scan_kets(ints_, ket_space, bra, work, [&](i32 iket, double h) {
                const double v = h * coeffs[static_cast<std::size_t>(iket)];

                if (std::abs(v) >= eps) acc += v;
            });

            out.hpsi_mut()[ibra] = acc;
        });

        return out;
    }

    [[nodiscard]] Projection project_generated_impl(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps,
        DetBatchView exclude
    ) const {
        const double max_scale = max_abs(coeffs);
        auto screen_ptr = screen(screen_cutoff(eps, max_scale));
        const DetIndex exclude_index(exclude);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::ProjPart> local;
        local.reserve(static_cast<std::size_t>(nthread));

        for (int t = 0; t < nthread; ++t) {
            local.emplace_back(nword_);
        }

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& part = local[static_cast<std::size_t>(tid)];
            KetWork work(nword_, ints_.norb());

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
#else
        {
            auto& part = local[0];
            KetWork work(nword_, ints_.norb());

            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
                const double coeff = coeffs[iket];
                const double scale = std::abs(coeff);
                const EdgeWindow win = edge_window(eps, std::numeric_limits<double>::infinity(), scale);

                scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double h) {
                    if (exclude_index.find(bra) >= 0) return;

                    const i32 ibra = part.pool.find_or_add(bra);
                    part.terms.push_back(detail_ham::ProjTerm{ibra, h * coeff});
                });
            }
        }

        std::vector<u64> global_words;

        for (const auto& part : local) {
            global_words.insert(global_words.end(), part.pool.words().begin(), part.pool.words().end());
        }

        sort_unique_dets(global_words, nword_);

        const DetBatchView global_view{
            global_words.data(),
            global_words.size() / det_size(nword_),
            nword_
        };

        const DetIndex global_index(global_view);
        std::vector<double> hpsi(global_view.n_dets, 0.0);

        for (std::size_t t = 0; t < local.size(); ++t) {
            for (const auto& term : local[t].terms) {
                const DetRef bra = local[t].pool.get(static_cast<std::size_t>(term.ibra));
                const i32 ibra = global_index.find(bra);

                if (ibra >= 0) {
                    hpsi[static_cast<std::size_t>(ibra)] += term.hpsi;
                }
            }
        }

        Projection out;
        out.set_nword(nword_);
        out.bra_words_mut() = std::move(global_words);
        out.hpsi_mut() = std::move(hpsi);

        const DetBatchView bras{
            out.bra_words().data(),
            out.hpsi().size(),
            nword_
        };

        out.diags_mut() = diags(bras);

        return out;
    }

    [[nodiscard]] Conns conns_impl(DetBatchView kets, double eps) const {
        auto screen_ptr = screen(eps);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::ConnPart> local;
        local.reserve(static_cast<std::size_t>(nthread));

        for (int t = 0; t < nthread; ++t) {
            local.emplace_back(nword_);
        }

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& part = local[static_cast<std::size_t>(tid)];
            KetWork work(nword_, ints_.norb());

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
#else
        {
            auto& part = local[0];
            KetWork work(nword_, ints_.norb());

            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
                const EdgeWindow win = edge_window(eps, std::numeric_limits<double>::infinity(), 1.0);

                scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double h) {
                    const i32 ibra = part.pool.find_or_add(bra);
                    part.conns.push_back(detail_ham::ConnRec{
                        static_cast<i32>(iket),
                        ibra,
                        h
                    });
                });
            }
        }

        std::vector<u64> global_words;

        for (const auto& part : local) {
            global_words.insert(global_words.end(), part.pool.words().begin(), part.pool.words().end());
        }

        sort_unique_dets(global_words, nword_);

        const DetBatchView global_view{
            global_words.data(),
            global_words.size() / det_size(nword_),
            nword_
        };

        const DetIndex global_index(global_view);

        Conns out;
        out.set_nword(nword_);
        out.set_n_kets(kets.n_dets);

        copy_batch(out.ket_words_mut(), kets);
        out.bra_words_mut() = std::move(global_words);

        out.diags_mut() = diags(kets);
        out.ket_ptr_mut().assign(kets.n_dets + 1u, 0);
        out.ket_weight_mut().assign(kets.n_dets, 0.0);
        out.ket_nconn_mut().assign(kets.n_dets, 0);

        std::size_t nconn = 0;

        for (std::size_t t = 0; t < local.size(); ++t) {
            nconn += local[t].conns.size();

            for (const auto& c : local[t].conns) {
                ++out.ket_ptr_mut()[static_cast<std::size_t>(c.iket) + 1u];
                out.ket_weight_mut()[static_cast<std::size_t>(c.iket)] += std::abs(c.h);
                ++out.ket_nconn_mut()[static_cast<std::size_t>(c.iket)];
            }
        }

        std::size_t acc = 0;
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            acc += static_cast<std::size_t>(out.ket_ptr_mut()[iket + 1u]);
            out.ket_ptr_mut()[iket + 1u] = to_i32(acc);
        }

        out.bra_mut().resize(nconn);
        out.h_mut().resize(nconn);

        std::vector<i32> pos = out.ket_ptr_mut();

        for (std::size_t t = 0; t < local.size(); ++t) {
            for (const auto& c : local[t].conns) {
                const std::size_t iket = static_cast<std::size_t>(c.iket);
                const std::size_t p = static_cast<std::size_t>(pos[iket]++);
                const DetRef bra = local[t].pool.get(static_cast<std::size_t>(c.ibra));

                out.bra_mut()[p] = global_index.find(bra);
                out.h_mut()[p] = c.h;
            }
        }

        return out;
    }

    [[nodiscard]] Degrees degrees_impl(DetBatchView kets, double eps) const {
        auto screen_ptr = screen(eps);

        Degrees out;
        out.ket_nconn_mut().assign(kets.n_dets, 0);
        out.ket_weight_mut().assign(kets.n_dets, 0.0);

        for_guided(kets.n_dets, [&](std::size_t iket, KetWork& work) {
            i64 nconn = 0;
            double weight = 0.0;
            const EdgeWindow win = edge_window(eps, std::numeric_limits<double>::infinity(), 1.0);

            scan_values(ints_, screen_ptr.get(), kets[iket], work, win, [&](double h) {
                ++nconn;
                weight += std::abs(h);
            });

            out.ket_nconn_mut()[iket] = nconn;
            out.ket_weight_mut()[iket] = weight;
        });

        return out;
    }

    void build_matrix(Matrix& out, DetBatchView bras, DetBatchView kets) const {
        const KetSpace ket_space(kets);
        const std::size_t nbras = bras.n_dets;

        out.diags_mut().assign(nbras, 0.0);
        out.indptr_mut().assign(nbras + 1u, 0);

        for_bras(nbras, [&](std::size_t ibra, BraWork& work) {
            const DetRef bra = bras[ibra];

            i32 nnz = 0;

            const double diag = Slater::diag(ints_, bra);
            out.diags_mut()[ibra] = diag;

            const i32 diag_idx = find_ket(ket_space, bra);
            if (diag_idx >= 0 && diag != 0.0) ++nnz;

            scan_kets(ints_, ket_space, bra, work, [&](i32, double) {
                ++nnz;
            });

            out.indptr_mut()[ibra + 1u] = nnz;
        });

        std::size_t nnz = 0;

        for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
            nnz += static_cast<std::size_t>(out.indptr_mut()[ibra + 1u]);
            out.indptr_mut()[ibra + 1u] = to_i32(nnz);
        }

        out.indices_mut().resize(nnz);
        out.data_mut().resize(nnz);

        for_bras(nbras, [&](std::size_t ibra, BraWork& work) {
            const DetRef bra = bras[ibra];

            std::size_t pos = static_cast<std::size_t>(out.indptr_mut()[ibra]);

            const double diag = out.diags_mut()[ibra];
            const i32 diag_idx = find_ket(ket_space, bra);

            if (diag_idx >= 0 && diag != 0.0) {
                out.indices_mut()[pos] = diag_idx;
                out.data_mut()[pos] = diag;
                ++pos;
            }

            scan_kets(ints_, ket_space, bra, work, [&](i32 iket, double h) {
                out.indices_mut()[pos] = iket;
                out.data_mut()[pos] = h;
                ++pos;
            });
        });
    }

    [[nodiscard]] ConnSamples sample_conns_impl(
        DetBatchView kets,
        std::span<const i64> counts,
        double eps1,
        double eps2,
        u64 seed
    ) const {
        auto screen_ptr = screen(eps2);
        const bool draw = !counts.empty();

        ConnSamples out;
        out.set_nword(nword_);
        out.ket_nconn_mut().assign(kets.n_dets, 0);
        out.ket_weight_mut().assign(kets.n_dets, 0.0);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::ConnSamplePart> local(static_cast<std::size_t>(nthread));

        auto process_one = [&](
            std::size_t iket,
            KetWork& work,
            std::vector<double>& targets,
            detail_ham::ConnSamplePart& part
        ) {
            i64 nconn = 0;
            double weight = 0.0;
            const EdgeWindow win = edge_window(eps2, eps1, 1.0);

            scan_values(ints_, screen_ptr.get(), kets[iket], work, win, [&](double h) {
                ++nconn;
                weight += std::abs(h);
            });

            out.ket_nconn_mut()[iket] = nconn;
            out.ket_weight_mut()[iket] = weight;

            if (!draw || counts[iket] <= 0 || nconn == 0 || !(weight > 0.0)) {
                return;
            }

            SmallRng rng(sample_seed(seed, kets[iket]));
            make_targets(rng, counts[iket], weight, targets);

            if (targets.empty()) return;

            std::size_t target_pos = 0;
            double cdf = 0.0;

            scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double h) {
                const double abs_h = std::abs(h);
                cdf += abs_h;

                i64 hit = 0;

                while (target_pos < targets.size() && targets[target_pos] <= cdf) {
                    ++hit;
                    ++target_pos;
                }

                if (hit <= 0) return;

                detail_ham::ConnSampleRec rec;
                rec.iket = static_cast<i32>(iket);
                rec.ibra = append_det_index(part.words, nword_, bra);
                rec.h = h;
                rec.pgen = abs_h / weight;
                rec.count = hit;

                part.recs.push_back(rec);
            });
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            KetWork work(nword_, ints_.norb());
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                process_one(
                    static_cast<std::size_t>(ii),
                    work,
                    targets,
                    local[static_cast<std::size_t>(tid)]
                );
            }
        }
#else
        {
            KetWork work(nword_, ints_.norb());
            std::vector<double> targets;

            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
                process_one(iket, work, targets, local[0]);
            }
        }
#endif

        std::vector<detail_ham::ConnSampleRec> records;
        std::vector<u64> record_words;

        std::size_t nrec = 0;
        std::size_t nwords = 0;

        for (const auto& part : local) {
            nrec += part.recs.size();
            nwords += part.words.size();
        }

        records.reserve(nrec);
        record_words.reserve(nwords);

        for (auto& part : local) {
            const std::size_t offset = record_words.size() / det_size(nword_);

            record_words.insert(record_words.end(), part.words.begin(), part.words.end());

            for (auto rec : part.recs) {
                rec.ibra += offset;
                records.push_back(rec);
            }
        }

        std::sort(records.begin(), records.end(), [&](const auto& x, const auto& y) {
            if (x.iket != y.iket) return x.iket < y.iket;

            return DetLess{}(
                det_at(record_words, nword_, x.ibra),
                det_at(record_words, nword_, y.ibra)
            );
        });

        for (const auto& rec : records) {
            out.ket_mut().push_back(rec.iket);
            append_det(out.bra_words_mut(), det_at(record_words, nword_, rec.ibra));
            out.h_mut().push_back(rec.h);
            out.pgen_mut().push_back(rec.pgen);
            out.counts_mut().push_back(rec.count);
        }

        return out;
    }

    [[nodiscard]] ProjectSamples sample_project_impl(
        DetBatchView kets,
        std::span<const double> coeffs,
        double eps1,
        double eps2,
        std::span<const i64> counts,
        DetBatchView exclude,
        i64 n_rep,
        u64 seed
    ) const {
        const double max_scale = max_abs(coeffs);
        auto screen_ptr = screen(screen_cutoff(eps2, max_scale));
        const DetIndex exclude_index(exclude);

#if defined(_OPENMP)
        const int nthread = std::max(1, omp_get_max_threads());
#else
        const int nthread = 1;
#endif

        std::vector<detail_ham::SamplePart> local(static_cast<std::size_t>(nthread));

        auto process_ket = [&](
            std::size_t iket,
            KetWork& work,
            std::vector<double>& targets,
            detail_ham::SamplePart& part
        ) {
            const i64 n_draw = counts[iket];
            const double coeff = coeffs[iket];
            const double scale = std::abs(coeff);

            if (n_draw <= 0 || scale <= 0.0) return;

            const EdgeWindow win = edge_window(eps2, eps1, scale);
            double weight = 0.0;

            scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double h) {
                if (exclude_index.find(bra) >= 0) return;
                weight += std::abs(h);
            });

            if (!(weight > 0.0)) return;

            for (i64 rep = 0; rep < n_rep; ++rep) {
                for (int stream = 0; stream < 2; ++stream) {
                    SmallRng rng(sample_seed(seed, kets[iket], rep, stream));
                    make_targets(rng, n_draw, weight, targets);

                    if (targets.empty()) continue;

                    std::size_t target_pos = 0;
                    double cdf = 0.0;

                    scan_conns(ints_, screen_ptr.get(), kets[iket], work, win, [&](DetRef bra, double h) {
                        if (exclude_index.find(bra) >= 0) return;

                        const double abs_h = std::abs(h);
                        cdf += abs_h;

                        i64 hit = 0;

                        while (target_pos < targets.size() && targets[target_pos] <= cdf) {
                            ++hit;
                            ++target_pos;
                        }

                        if (hit <= 0) return;

                        const double delta =
                            static_cast<double>(hit) * coeff * h * weight
                            / (static_cast<double>(n_draw) * abs_h);

                        detail_ham::SampleRec rec;
                        rec.rep = static_cast<i32>(rep);
                        rec.ibra = append_det_index(part.words, nword_, bra);

                        if (stream == 0) rec.hpsi_a = delta;
                        else rec.hpsi_b = delta;

                        part.recs.push_back(rec);
                    });
                }
            }
        };

#if defined(_OPENMP)
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            KetWork work(nword_, ints_.norb());
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                process_ket(
                    static_cast<std::size_t>(ii),
                    work,
                    targets,
                    local[static_cast<std::size_t>(tid)]
                );
            }
        }
#else
        {
            KetWork work(nword_, ints_.norb());
            std::vector<double> targets;

            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
                process_ket(iket, work, targets, local[0]);
            }
        }
#endif

        std::vector<detail_ham::SampleRec> records;
        std::vector<u64> record_words;

        std::size_t nrec = 0;
        std::size_t nwords = 0;

        for (const auto& part : local) {
            nrec += part.recs.size();
            nwords += part.words.size();
        }

        records.reserve(nrec);
        record_words.reserve(nwords);

        for (auto& part : local) {
            const std::size_t offset = record_words.size() / det_size(nword_);

            record_words.insert(record_words.end(), part.words.begin(), part.words.end());

            for (auto rec : part.recs) {
                rec.ibra += offset;
                records.push_back(rec);
            }
        }

        std::sort(records.begin(), records.end(), [&](const auto& x, const auto& y) {
            if (x.rep != y.rep) return x.rep < y.rep;

            return DetLess{}(
                det_at(record_words, nword_, x.ibra),
                det_at(record_words, nword_, y.ibra)
            );
        });

        ProjectSamples out;
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
                const DetRef bra = det_at(record_words, nword_, records[i].ibra);

                double hpsi_a = 0.0;
                double hpsi_b = 0.0;

                std::size_t j = i;

                while (
                    j < records.size()
                    && records[j].rep == rep
                    && det_equal(det_at(record_words, nword_, records[j].ibra), bra)
                ) {
                    hpsi_a += records[j].hpsi_a;
                    hpsi_b += records[j].hpsi_b;
                    ++j;
                }

                append_det(out.bra_words_mut(), bra);
                append_det(global_pool, bra);

                out.hpsi_a_mut().push_back(hpsi_a);
                out.hpsi_b_mut().push_back(hpsi_b);

                ++pos;
                i = j;
            }
        }

        out.rep_ptr_mut()[static_cast<std::size_t>(n_rep)] = to_i32(pos);

        out.diags_mut().assign(pos, 0.0);
        out.hpsi_strong_mut().assign(pos, 0.0);

        if (pos == 0) return out;

        sort_unique_dets(global_pool, nword_);

        const std::size_t n_bras = global_pool.size() / det_size(nword_);
        const DetBatchView bras{
            global_pool.data(),
            n_bras,
            nword_
        };

        const Projection strong = project_impl(bras, kets, coeffs, eps1);

        const DetBatchView strong_view{
            strong.bra_words().data(),
            strong.n_bras(),
            nword_
        };

        const DetIndex strong_index(strong_view);

        const DetBatchView out_view{
            out.bra_words().data(),
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