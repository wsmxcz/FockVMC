#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/rhf/element.hpp>
#include <libdet/spatial/space.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::rhf {

// Internal implementation header included by hamiltonian.hpp.
struct Matrix {
    std::size_t n_bra = 0;
    std::size_t n_ket = 0;
    std::vector<i32> indptr;
    std::vector<i32> indices;
    std::vector<double> data;
};

template <class Emit>
inline void emit_nonzero(double h, i32 iket, Emit&& emit) {
    if (h != 0.0) emit(iket, h);
}

// Visit off-diagonal RHF couplings from a bra into a known ket space.
template <class Emit>
inline void visit_kets(
    const Integral& ints,
    const KetSpace& kets,
    DetRef bra,
    BraScratch& work,
    Emit&& emit
) {
    work.ensure_seen(kets.alpha.size(), kets.beta.size());

    work.next_a();
    find_single(
        kets.alpha,
        kets.alpha1,
        bra.alpha(),
        work.tmp_occ,
        work.seen_a,
        work.stamp_a,
        work.alpha_single
    );

    work.next_a();
    find_double(
        kets.alpha,
        kets.alpha2,
        bra.alpha(),
        work.tmp_occ,
        work.seen_a,
        work.stamp_a,
        work.alpha_double
    );

    work.next_b();
    find_single(
        kets.beta,
        kets.beta1,
        bra.beta(),
        work.tmp_occ,
        work.seen_b,
        work.stamp_b,
        work.beta_single
    );

    work.next_b();
    find_double(
        kets.beta,
        kets.beta2,
        bra.beta(),
        work.tmp_occ,
        work.seen_b,
        work.stamp_b,
        work.beta_double
    );

    const i32 bra_alpha = kets.alpha.find(bra.alpha());
    const i32 bra_beta = kets.beta.find(bra.beta());

    if (bra_beta >= 0) {
        for (const auto& ex : work.alpha_single) {
            const i32 iket = kets.find_with_beta(bra_beta, ex.spin);
            if (iket >= 0) {
                emit_nonzero(
                    ex.sign * single_alpha(ints, bra, ex.i, ex.a),
                    iket,
                    emit
                );
            }
        }

        for (const auto& ex : work.alpha_double) {
            const i32 iket = kets.find_with_beta(bra_beta, ex.spin);
            if (iket >= 0) {
                emit_nonzero(
                    ex.sign
                        * double_alpha(
                            ints,
                            ex.i,
                            ex.j,
                            ex.a,
                            ex.b
                        ),
                    iket,
                    emit
                );
            }
        }
    }

    if (bra_alpha >= 0) {
        for (const auto& ex : work.beta_single) {
            const i32 iket = kets.find_with_alpha(bra_alpha, ex.spin);
            if (iket >= 0) {
                emit_nonzero(
                    ex.sign * single_beta(ints, bra, ex.i, ex.a),
                    iket,
                    emit
                );
            }
        }

        for (const auto& ex : work.beta_double) {
            const i32 iket = kets.find_with_alpha(bra_alpha, ex.spin);
            if (iket >= 0) {
                emit_nonzero(
                    ex.sign
                        * double_beta(
                            ints,
                            ex.i,
                            ex.j,
                            ex.a,
                            ex.b
                        ),
                    iket,
                    emit
                );
            }
        }
    }

    // Mixed-spin doubles are intersections of alpha and beta singles.
    work.ensure_cross(kets.beta.size());
    work.next_cross();

    for (const auto& ex : work.beta_single) {
        const std::size_t beta_id = static_cast<std::size_t>(ex.spin);
        work.cross_b[beta_id] = work.cross_stamp;
        work.cross_i[beta_id] = ex.i;
        work.cross_a[beta_id] = ex.a;
        work.cross_sign[beta_id] = ex.sign;
    }

    for (const auto& ax : work.alpha_single) {
        for (const SpinMate& mate : kets.alpha_mates(ax.spin)) {
            const std::size_t beta_id = static_cast<std::size_t>(mate.spin);
            if (work.cross_b[beta_id] != work.cross_stamp) continue;

            const double h =
                ax.sign
                * work.cross_sign[beta_id]
                * double_mixed(
                    ints,
                    ax.i,
                    work.cross_i[beta_id],
                    ax.a,
                    work.cross_a[beta_id]
                );
            emit_nonzero(h, mate.ket, emit);
        }
    }
}

inline std::vector<double> Hamiltonian::diags(DetBatchView dets) const {
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
            out[idet] = diag(ints_, scratch.occ);
        }
    }
#else
    KetScratch scratch(ints_.norb());
    for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
        fill_occ(dets[idet], ints_.norb(), scratch.occ);
        out[idet] = diag(ints_, scratch.occ);
    }
#endif

    return out;
}

inline Projection Hamiltonian::project(
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

inline Matrix Hamiltonian::matrix(
    DetBatchView bras,
    DetBatchView kets
) const {
    check_dets(bras, "matrix(bras)");
    check_dets(kets, "matrix(kets)");

    Matrix out;
    out.n_bra = bras.n_dets;
    out.n_ket = kets.n_dets;
    build_matrix(out, bras, kets);
    return out;
}

inline void Hamiltonian::build_matrix(
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
            const double h_bra_bra = diag(ints_, bra);
            i32 nnz = find_ket(*ket_space, bra) >= 0 && h_bra_bra != 0.0 ? 1 : 0;

            visit_kets(ints_, *ket_space, bra, scratch, [&](i32, double) {
                ++nnz;
            });
            hdiag[ibra] = h_bra_bra;
            out.indptr[ibra + 1u] = nnz;
        }

#pragma omp single
        {
            std::size_t nnz = 0;
            for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
                nnz += static_cast<std::size_t>(out.indptr[ibra + 1u]);
                out.indptr[ibra + 1u] = to_i32(nnz);
            }
            out.indices.resize(nnz);
            out.data.resize(nnz);
        }

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(nbras); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            std::size_t pos = static_cast<std::size_t>(out.indptr[ibra]);
            const double diag = hdiag[ibra];
            const i32 diag_idx = find_ket(*ket_space, bra);

            if (diag_idx >= 0 && diag != 0.0) {
                out.indices[pos] = diag_idx;
                out.data[pos++] = diag;
            }

            visit_kets(
                ints_,
                *ket_space,
                bra,
                scratch,
                [&](i32 iket, double h) {
                    out.indices[pos] = iket;
                    out.data[pos++] = h;
                }
            );
        }
    }
#else
    BraScratch scratch;
    for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
        const DetRef bra = bras[ibra];
        const double h_bra_bra = diag(ints_, bra);
        i32 nnz = find_ket(*ket_space, bra) >= 0 && h_bra_bra != 0.0 ? 1 : 0;
        visit_kets(ints_, *ket_space, bra, scratch, [&](i32, double) {
            ++nnz;
        });
        hdiag[ibra] = h_bra_bra;
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
            out.data[pos++] = diag;
        }

        visit_kets(
            ints_,
            *ket_space,
            bra,
            scratch,
            [&](i32 iket, double h) {
                out.indices[pos] = iket;
                out.data[pos++] = h;
            }
        );
    }
#endif
}

inline std::vector<double> Hamiltonian::matvec(
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
                value += diag(ints_, bra)
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

inline std::vector<double> Hamiltonian::matmat(
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
                const double h = diag(ints_, bra);
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

inline Projection Hamiltonian::project_impl(
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
            const double h_bra_bra = diag(ints_, bra);
            double value = 0.0;
            out.diags[ibra] = h_bra_bra;

            const i32 diag_idx = find_ket(*ket_space, bra);
            if (diag_idx >= 0) {
                const double term =
                    h_bra_bra * coeffs[static_cast<std::size_t>(diag_idx)];
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

} // namespace libdet::rhf
