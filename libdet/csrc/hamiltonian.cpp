#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <libdet/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet {

Hamiltonian::Hamiltonian(RHFIntegrals ints)
    : ints_(std::move(ints)),
      nword_(bits::words_for(ints_.norb())),
      ket_cache_(nword_) {}

Hamiltonian::Hamiltonian(const Hamiltonian& other)
    : ints_(other.ints_),
      nword_(other.nword_),
      ket_cache_(other.nword_) {}

Hamiltonian& Hamiltonian::operator=(const Hamiltonian& other) {
    if (this != &other) {
        ints_ = other.ints_;
        nword_ = other.nword_;
        screen_.reset();
        ket_cache_ = KetCache(nword_);
        ket_space_cache_ = KetSpaceCache();
    }

    return *this;
}

Hamiltonian::Hamiltonian(Hamiltonian&& other) noexcept
    : ints_(std::move(other.ints_)),
      nword_(other.nword_),
      screen_(std::move(other.screen_)),
      ket_cache_(other.nword_) {}

Hamiltonian& Hamiltonian::operator=(Hamiltonian&& other) noexcept {
    if (this != &other) {
        ints_ = std::move(other.ints_);
        nword_ = other.nword_;
        screen_ = std::move(other.screen_);
        ket_cache_ = KetCache(nword_);
        ket_space_cache_ = KetSpaceCache();
    }

    return *this;
}

Hamiltonian Hamiltonian::make(
    std::span<const double> h1,
    int norb,
    std::span<const double> eri,
    double ecore
) {
    return Hamiltonian(RHFIntegrals(norb, h1, eri, ecore));
}

int Hamiltonian::norb() const noexcept {
    return ints_.norb();
}

u32 Hamiltonian::nword() const noexcept {
    return nword_;
}

void Hamiltonian::check_one(DetRef det, const char* where) const {
    if (det.nword() != nword_) {
        throw std::invalid_argument(
            std::string(where) + ": determinant nword mismatch"
        );
    }
}

void Hamiltonian::check_dets(DetBatchView dets, const char* where) const {
    if (dets.nword != nword_) {
        throw std::invalid_argument(
            std::string(where) + ": determinant nword mismatch"
        );
    }
}

void Hamiltonian::check_eps(double eps) {
    if (std::isnan(eps)) throw std::invalid_argument("eps must not be NaN");
    if (eps < 0.0) throw std::invalid_argument("eps must be nonnegative");
}

void Hamiltonian::check_window_eps(double eps1, double eps2) {
    check_eps(eps1);
    check_eps(eps2);
    if (eps2 > eps1) throw std::invalid_argument("eps2 must be <= eps1");
}

std::shared_ptr<const Screen> Hamiltonian::screen(double cutoff) const {
    if (cutoff <= 0.0 || !std::isfinite(cutoff)) return {};

    std::lock_guard<std::mutex> lock(screen_mutex_);

    if (!screen_ || cutoff < screen_->cutoff()) {
        screen_ = std::make_shared<Screen>(ints_, cutoff);
    }

    return screen_;
}

double Hamiltonian::max_abs(std::span<const double> values) noexcept {
    double out = 0.0;
    for (double value : values) out = std::max(out, std::abs(value));
    return out;
}

double Hamiltonian::screen_cutoff(double eps, double max_scale) noexcept {
    if (eps <= 0.0) return 0.0;
    if (max_scale <= 0.0) return std::numeric_limits<double>::infinity();
    return eps / max_scale;
}

AbsWindow Hamiltonian::abs_window(
    double lo,
    double hi,
    double scale
) noexcept {
    if (scale <= 0.0) {
        return {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
        };
    }

    return {
        lo <= 0.0 ? 0.0 : lo / scale,
        std::isfinite(hi)
            ? hi / scale
            : std::numeric_limits<double>::infinity()
    };
}

std::shared_ptr<const KetSpace> Hamiltonian::cached_ket_space(
    DetBatchView kets
) const {
    {
        std::lock_guard<std::mutex> lock(ket_space_cache_mutex_);
        if (auto space = ket_space_cache_.find(kets)) return space;
    }

    auto fresh = std::make_shared<KetSpace>(kets);

    std::lock_guard<std::mutex> lock(ket_space_cache_mutex_);
    if (auto space = ket_space_cache_.find(kets)) return space;
    ket_space_cache_.insert(kets, fresh);
    return fresh;
}

std::vector<u64> Hamiltonian::merge_det_parts(
    std::vector<std::vector<u64>>& parts
) const {
    std::size_t total = 0;

    for (auto& part : parts) {
        sort_unique_dets(part, nword_);
        total += part.size();
    }

    std::vector<u64> out;
    out.reserve(total);

    for (auto& part : parts) {
        out.insert(out.end(), part.begin(), part.end());
    }

    sort_unique_dets(out, nword_);
    return out;
}

double Hamiltonian::hij(DetRef bra, DetRef ket) const {
    check_one(bra, "hij(bra)");
    check_one(ket, "hij(ket)");

    const DetDiff ex = det_diff(bra, ket);

    if (ex.deg > 2) return 0.0;
    if (ex.deg == 0) return Slater::diag(ints_, bra);

    if (ex.deg == 1) {
        return ex.na == 1
            ? ex.sign * Slater::single_a(
                ints_,
                bra,
                ex.occ_a[0],
                ex.vir_a[0]
            )
            : ex.sign * Slater::single_b(
                ints_,
                bra,
                ex.occ_b[0],
                ex.vir_b[0]
            );
    }

    if (ex.na == 2) {
        return ex.sign * Slater::double_aa(
            ints_,
            ex.occ_a[0],
            ex.occ_a[1],
            ex.vir_a[0],
            ex.vir_a[1]
        );
    }

    if (ex.nb == 2) {
        return ex.sign * Slater::double_bb(
            ints_,
            ex.occ_b[0],
            ex.occ_b[1],
            ex.vir_b[0],
            ex.vir_b[1]
        );
    }

    return ex.sign * Slater::double_ab(
        ints_,
        ex.occ_a[0],
        ex.occ_b[0],
        ex.vir_a[0],
        ex.vir_b[0]
    );
}

std::vector<double> Hamiltonian::diags(DetBatchView dets) const {
    check_dets(dets, "diags");
    std::vector<double> out(dets.n_dets, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        KetScratch scratch(ints_.norb());

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(dets.n_dets); ++ii) {
            const std::size_t idet = static_cast<std::size_t>(ii);
            fill_occ(dets[idet], ints_.norb(), scratch.occ);
            out[idet] = Slater::diag(ints_, scratch.occ);
        }
    }
#else
    KetScratch scratch(ints_.norb());
    for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
        fill_occ(dets[idet], ints_.norb(), scratch.occ);
        out[idet] = Slater::diag(ints_, scratch.occ);
    }
#endif

    return out;
}

Projection Hamiltonian::project(
    DetBatchView bras,
    DetBatchView kets,
    std::span<const double> coeffs,
    double eps
) const {
    check_dets(bras, "project(bras)");
    check_dets(kets, "project(kets)");

    if (coeffs.size() != kets.n_dets) {
        throw std::invalid_argument("project: coeffs size must match kets");
    }

    check_eps(eps);
    return project_impl(bras, kets, coeffs, eps);
}

Matrix Hamiltonian::matrix(DetBatchView bras, DetBatchView kets) const {
    check_dets(bras, "matrix(bras)");
    check_dets(kets, "matrix(kets)");

    Matrix out;
    out.n_bra = bras.n_dets;
    out.n_ket = kets.n_dets;
    build_matrix(out, bras, kets);
    return out;
}

void Hamiltonian::build_matrix(
    Matrix& out,
    DetBatchView bras,
    DetBatchView kets
) const {
    const auto ket_space = cached_ket_space(kets);
    const std::size_t nbras = bras.n_dets;
    std::vector<double> hdiag(nbras, 0.0);

    out.indptr.assign(nbras + 1u, 0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        BraScratch scratch;

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(nbras); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            const double diag = Slater::diag(ints_, bra);
            i32 nnz = find_ket(*ket_space, bra) >= 0 && diag != 0.0 ? 1 : 0;

            visit_kets(ints_, *ket_space, bra, scratch, [&](i32, double) {
                ++nnz;
            });

            hdiag[ibra] = diag;
            out.indptr[ibra + 1u] = nnz;
        }

#pragma omp single
        {
            std::size_t nnz = 0;

            for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
                nnz += static_cast<std::size_t>(
                    out.indptr[ibra + 1u]
                );
                out.indptr[ibra + 1u] = to_i32(nnz);
            }

            out.indices.resize(nnz);
            out.data.resize(nnz);
        }

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(nbras); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            std::size_t pos = static_cast<std::size_t>(
                out.indptr[ibra]
            );
            const double diag = hdiag[ibra];
            const i32 diag_idx = find_ket(*ket_space, bra);

            if (diag_idx >= 0 && diag != 0.0) {
                out.indices[pos] = diag_idx;
                out.data[pos] = diag;
                ++pos;
            }

            visit_kets(
                ints_,
                *ket_space,
                bra,
                scratch,
                [&](i32 iket, double h) {
                    out.indices[pos] = iket;
                    out.data[pos] = h;
                    ++pos;
                }
            );
        }
    }
#else
    BraScratch scratch;

    for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
        const DetRef bra = bras[ibra];
        const double diag = Slater::diag(ints_, bra);
        i32 nnz = find_ket(*ket_space, bra) >= 0 && diag != 0.0 ? 1 : 0;

        visit_kets(ints_, *ket_space, bra, scratch, [&](i32, double) {
            ++nnz;
        });

        hdiag[ibra] = diag;
        out.indptr[ibra + 1u] = nnz;
    }

    std::size_t nnz = 0;
    for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
        nnz += static_cast<std::size_t>(out.indptr[ibra + 1u]);
        out.indptr[ibra + 1u] = to_i32(nnz);
    }

    out.indices.resize(nnz);
    out.data.resize(nnz);

    for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
        const DetRef bra = bras[ibra];
        std::size_t pos = static_cast<std::size_t>(out.indptr[ibra]);
        const double diag = hdiag[ibra];
        const i32 diag_idx = find_ket(*ket_space, bra);

        if (diag_idx >= 0 && diag != 0.0) {
            out.indices[pos] = diag_idx;
            out.data[pos] = diag;
            ++pos;
        }

        visit_kets(
            ints_,
            *ket_space,
            bra,
            scratch,
            [&](i32 iket, double h) {
                out.indices[pos] = iket;
                out.data[pos] = h;
                ++pos;
            }
        );
    }
#endif
}

std::vector<double> Hamiltonian::matvec(
    DetBatchView bras,
    DetBatchView kets,
    std::span<const double> x
) const {
    check_dets(bras, "matvec(bras)");
    check_dets(kets, "matvec(kets)");

    if (x.size() != kets.n_dets) {
        throw std::invalid_argument("matvec: x size must match kets");
    }

    const auto ket_space = cached_ket_space(kets);
    std::vector<double> out(bras.n_dets, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        BraScratch scratch;

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
#else
    {
        BraScratch scratch;
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
#endif
            const DetRef bra = bras[ibra];
            double value = 0.0;

            const i32 diag_idx = find_ket(*ket_space, bra);
            if (diag_idx >= 0) {
                value += Slater::diag(ints_, bra)
                    * x[static_cast<std::size_t>(diag_idx)];
            }

            visit_kets(
                ints_,
                *ket_space,
                bra,
                scratch,
                [&](i32 iket, double h) {
                    value += h * x[static_cast<std::size_t>(iket)];
                }
            );

            out[ibra] = value;
        }
    }

    return out;
}

std::vector<double> Hamiltonian::matmat(
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

    const auto ket_space = cached_ket_space(kets);
    std::vector<double> out(bras.n_dets * nrhs, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        BraScratch scratch;

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
#else
    {
        BraScratch scratch;
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
#endif
            const DetRef bra = bras[ibra];
            double* y = out.data() + ibra * nrhs;

            const i32 diag_idx = find_ket(*ket_space, bra);
            if (diag_idx >= 0) {
                const double h = Slater::diag(ints_, bra);
                const double* xrow =
                    x.data() + static_cast<std::size_t>(diag_idx) * nrhs;

                for (std::size_t j = 0; j < nrhs; ++j) y[j] += h * xrow[j];
            }

            visit_kets(
                ints_,
                *ket_space,
                bra,
                scratch,
                [&](i32 iket, double h) {
                    const double* xrow =
                        x.data() + static_cast<std::size_t>(iket) * nrhs;

                    for (std::size_t j = 0; j < nrhs; ++j) {
                        y[j] += h * xrow[j];
                    }
                }
            );
        }
    }

    return out;
}

Projection Hamiltonian::project_impl(
    DetBatchView bras,
    DetBatchView kets,
    std::span<const double> coeffs,
    double eps
) const {
    const auto ket_space = cached_ket_space(kets);

    Projection out;
    out.nword = nword_;
    copy_batch(out.bra_words, bras);
    out.hpsi.assign(bras.n_dets, 0.0);
    out.diags.assign(bras.n_dets, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        BraScratch scratch;

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
#else
    {
        BraScratch scratch;
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
#endif
            const DetRef bra = bras[ibra];
            const double diag = Slater::diag(ints_, bra);
            double value = 0.0;

            out.diags[ibra] = diag;

            const i32 diag_idx = find_ket(*ket_space, bra);
            if (diag_idx >= 0) {
                const double term =
                    diag * coeffs[static_cast<std::size_t>(diag_idx)];
                if (std::abs(term) >= eps) value += term;
            }

            visit_kets(
                ints_,
                *ket_space,
                bra,
                scratch,
                [&](i32 iket, double h) {
                    const double term =
                        h * coeffs[static_cast<std::size_t>(iket)];
                    if (std::abs(term) >= eps) value += term;
                }
            );

            out.hpsi[ibra] = value;
        }
    }

    return out;
}

} // namespace libdet
