#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/integral.hpp>

#include <omp.h>

namespace libdet {

// Visit off-diagonal terms from a bra into a known ket space.
template <class Emit>
inline void visit_internal(
    const Integral& ints,
    const DetSpace& kets,
    DetRef bra,
    const ElementScratch& element,
    VisitScratch& work,
    Emit&& emit
) {
    const auto output = [&](double h, i32 iket) {
        if (h != 0.0) emit(iket, h);
    };
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
            const i32 iket = kets.find_beta(bra_beta, ex.spin);
            if (iket >= 0) {
                output(
                    ex.sign * element.single_alpha(ex.i, ex.a),
                    iket
                );
            }
        }

        for (const auto& ex : work.alpha_double) {
            const i32 iket = kets.find_beta(bra_beta, ex.spin);
            if (iket >= 0) {
                output(
                    ex.sign
                        * double_same(
                            ints,
                            ex.i,
                            ex.j,
                            ex.a,
                            ex.b
                        ),
                    iket
                );
            }
        }
    }

    if (bra_alpha >= 0) {
        for (const auto& ex : work.beta_single) {
            const i32 iket = kets.find_alpha(bra_alpha, ex.spin);
            if (iket >= 0) {
                output(
                    ex.sign * element.single_beta(ex.i, ex.a),
                    iket
                );
            }
        }

        for (const auto& ex : work.beta_double) {
            const i32 iket = kets.find_alpha(bra_alpha, ex.spin);
            if (iket >= 0) {
                output(
                    ex.sign
                        * double_same(
                            ints,
                            ex.i,
                            ex.j,
                            ex.a,
                            ex.b
                        ),
                    iket
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
            output(h, mate.ket);
        }
    }
}

inline std::vector<double> Hamiltonian::diag(DetBatchView dets) const {
    check_dets(dets, "diag");
    std::vector<double> out(dets.n_dets, 0.0);

#pragma omp parallel
    {
        DetOcc occ;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(dets.n_dets); ++ii) {
            const std::size_t idet = static_cast<std::size_t>(ii);
            fill_occ(dets[idet], ints_.norb(), occ);
            out[idet] = libdet::diag(ints_, occ);
        }
    }

    return out;
}

inline Projection Hamiltonian::project(
    DetBatchView bras,
    DetBatchView kets,
    std::span<const double> scale,
    double eps
) const {
    check_dets(bras, "project(bras)");
    check_dets(kets, "project(kets)");
    if (scale.size() != kets.n_dets) {
        throw std::invalid_argument("project: scale size must match kets");
    }

    check_eps(eps);
    const DetSpace ket_space(kets);

    Projection out;
    out.nword = nword_;
    copy_batch(out.bra, bras);
    out.hpsi.assign(bras.n_dets, 0.0);
    out.diag.assign(bras.n_dets, 0.0);

#pragma omp parallel
    {
        VisitScratch scratch;
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            element.load(ints_, bra);
            const double h_diag = element.diag();
            double value = 0.0;
            out.diag[ibra] = h_diag;

            const i32 diag_idx = find_det(ket_space, bra);
            if (diag_idx >= 0) {
                const double term =
                    h_diag * scale[static_cast<std::size_t>(diag_idx)];
                if (std::abs(term) >= eps) value += term;
            }

            visit_internal(
                ints_,
                ket_space,
                bra,
                element,
                scratch,
                [&](i32 iket, double h) {
                    const double term =
                        h * scale[static_cast<std::size_t>(iket)];
                    if (std::abs(term) >= eps) value += term;
                }
            );
            out.hpsi[ibra] = value;
        }
    }
    return out;
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

    const DetSpace ket_space(kets);
    const std::size_t nbras = bras.n_dets;
    std::vector<double> hdiag(nbras, 0.0);
    out.indptr.assign(nbras + 1u, 0);

#pragma omp parallel
    {
        VisitScratch scratch;
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(nbras); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            element.load(ints_, bra);
            const double h_diag = element.diag();
            std::size_t nnz =
                find_det(ket_space, bra) >= 0 && h_diag != 0.0 ? 1u : 0u;

            visit_internal(ints_, ket_space, bra, element, scratch, [&](i32, double) {
                ++nnz;
            });
            hdiag[ibra] = h_diag;
            out.indptr[ibra + 1u] = to_i64(nnz);
        }

#pragma omp single
        {
            std::size_t nnz = 0;
            for (std::size_t ibra = 0; ibra < nbras; ++ibra) {
                nnz += static_cast<std::size_t>(out.indptr[ibra + 1u]);
                out.indptr[ibra + 1u] = to_i64(nnz);
            }
            out.indices.resize(nnz);
            out.data.resize(nnz);
        }

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(nbras); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            element.load(ints_, bra);
            std::size_t pos = static_cast<std::size_t>(out.indptr[ibra]);
            const double hdiag_i = hdiag[ibra];
            const i32 diag_idx = find_det(ket_space, bra);

            if (diag_idx >= 0 && hdiag_i != 0.0) {
                out.indices[pos] = diag_idx;
                out.data[pos++] = hdiag_i;
            }

            visit_internal(
                ints_,
                ket_space,
                bra,
                element,
                scratch,
                [&](i32 iket, double h) {
                    out.indices[pos] = iket;
                    out.data[pos++] = h;
                }
            );
        }
    }

    return out;
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

    const auto ket_space = cached_space(kets);
    std::vector<double> out(bras.n_dets, 0.0);

#pragma omp parallel
    {
        VisitScratch scratch;
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            element.load(ints_, bra);
            double value = 0.0;
            const i32 diag_idx = find_det(*ket_space, bra);
            if (diag_idx >= 0) {
                value += element.diag()
                    * x[static_cast<std::size_t>(diag_idx)];
            }

            visit_internal(
                ints_,
                *ket_space,
                bra,
                element,
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

    const auto ket_space = cached_space(kets);
    std::vector<double> out(bras.n_dets * nrhs, 0.0);

#pragma omp parallel
    {
        VisitScratch scratch;
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            const DetRef bra = bras[ibra];
            element.load(ints_, bra);
            double* y = out.data() + ibra * nrhs;
            const i32 diag_idx = find_det(*ket_space, bra);

            if (diag_idx >= 0) {
                const double h = element.diag();
                const double* xrow =
                    x.data() + static_cast<std::size_t>(diag_idx) * nrhs;
                for (std::size_t j = 0; j < nrhs; ++j) y[j] += h * xrow[j];
            }

            visit_internal(
                ints_,
                *ket_space,
                bra,
                element,
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

} // namespace libdet
