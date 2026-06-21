#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::guga {

namespace detail {

struct MatTerm {
    i32 ibra = 0;
    i32 iket = 0;
    double h = 0.0;
};

} // namespace detail

inline Projection Hamiltonian::project(
    PathBatchView bras,
    PathBatchView kets,
    std::span<const double> scale,
    double eps
) const {
    check_paths(bras, "project(bras)");
    check_paths(kets, "project(kets)");
    if (scale.size() != kets.n_paths) {
        throw std::invalid_argument("project: scale size must match kets");
    }
    check_eps(eps);
    return project_impl(bras, kets, scale, eps);
}

inline Projection Hamiltonian::project_impl(
    PathBatchView bras,
    PathBatchView kets,
    std::span<const double> scale,
    double eps
) const {
    const auto path_space = cached_space(bras);

    Projection out;
    out.nword = sector_.nword;
    copy_paths(out.bra_words, bras);
    out.diags = diags(bras);

    const double max_scale = eps == 0.0 ? 0.0 : max_abs(scale);
    const auto screen_ptr = (eps > 0.0 && max_scale > 0.0)
        ? screen(screen_cutoff(eps, max_scale))
        : std::shared_ptr<const Screen>{};

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    const std::size_t stride = bras.n_paths;
    std::vector<double> local(static_cast<std::size_t>(nthread) * stride, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* hpsi = local.data() + static_cast<std::size_t>(tid) * stride;
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        double* hpsi = local.data();
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double coeff = scale[iket];
            if (coeff == 0.0) continue;

            const double lo = eps == 0.0 ? 0.0 : eps / std::abs(coeff);
            decode_path(kets[iket], sector_, "project(ket)", ket_scratch, ket);
            path_space->visit(ket, visit_scratch, [&](i32 ibra, const OccMove& move) {
                const PathState& bra = path_space->state(ibra);
                if (screen_ptr) {
                    if (move.degree == 0) {
                        if (!same_path(bra, ket) && screen_ptr->same(ket.occ) < lo) return;
                    } else if (screen_ptr->bound(ket.occ, Move::from(move)) < lo) {
                        return;
                    }
                }

                const double h = guga::hij(elem_scratch, ints_, bra, ket, move);
                const double term = h * coeff;
                if (term != 0.0 && std::abs(term) >= eps) {
                    hpsi[static_cast<std::size_t>(ibra)] += term;
                }
            });
        }
    }

    out.hpsi.assign(bras.n_paths, 0.0);
    for (int t = 0; t < nthread; ++t) {
        const double* part = local.data() + static_cast<std::size_t>(t) * stride;
        for (std::size_t i = 0; i < stride; ++i) out.hpsi[i] += part[i];
    }
    return out;
}

inline Matrix Hamiltonian::matrix(PathBatchView bras, PathBatchView kets) const {
    check_paths(bras, "matrix(bras)");
    check_paths(kets, "matrix(kets)");

    const auto path_space = cached_space(bras);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    std::vector<std::vector<detail::MatTerm>> local(static_cast<std::size_t>(nthread));

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& terms = local[static_cast<std::size_t>(tid)];
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& terms = local[0];
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            decode_path(kets[iket], sector_, "matrix(ket)", ket_scratch, ket);
            path_space->visit(ket, visit_scratch, [&](i32 ibra, const OccMove& move) {
                const double h = guga::hij(elem_scratch, ints_, path_space->state(ibra), ket, move);
                if (h != 0.0) terms.push_back({ibra, to_i32(iket), h});
            });
        }
    }

    std::size_t nnz = 0;
    for (const auto& part : local) nnz += part.size();

    std::vector<detail::MatTerm> terms;
    terms.reserve(nnz);
    for (auto& part : local) {
        terms.insert(terms.end(), part.begin(), part.end());
        std::vector<detail::MatTerm>().swap(part);
    }
    std::sort(terms.begin(), terms.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.ibra != rhs.ibra) return lhs.ibra < rhs.ibra;
        return lhs.iket < rhs.iket;
    });

    Matrix out;
    out.n_bra = bras.n_paths;
    out.n_ket = kets.n_paths;
    out.indptr.assign(bras.n_paths + 1u, 0);
    for (const auto& term : terms) ++out.indptr[static_cast<std::size_t>(term.ibra) + 1u];
    for (std::size_t i = 0; i < bras.n_paths; ++i) out.indptr[i + 1u] += out.indptr[i];

    out.indices.reserve(terms.size());
    out.data.reserve(terms.size());
    for (const auto& term : terms) {
        out.indices.push_back(term.iket);
        out.data.push_back(term.h);
    }
    return out;
}

inline std::vector<double> Hamiltonian::matvec(
    PathBatchView bras,
    PathBatchView kets,
    std::span<const double> x
) const {
    check_paths(bras, "matvec(bras)");
    check_paths(kets, "matvec(kets)");
    if (x.size() != kets.n_paths) {
        throw std::invalid_argument("matvec: x size must match kets");
    }

    const auto path_space = cached_space(bras);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    const std::size_t stride = bras.n_paths;
    std::vector<double> local(static_cast<std::size_t>(nthread) * stride, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* out = local.data() + static_cast<std::size_t>(tid) * stride;
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        double* out = local.data();
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double coeff = x[iket];
            if (coeff == 0.0) continue;

            decode_path(kets[iket], sector_, "matvec(ket)", ket_scratch, ket);
            path_space->visit(ket, visit_scratch, [&](i32 ibra, const OccMove& move) {
                const double h = guga::hij(elem_scratch, ints_, path_space->state(ibra), ket, move);
                if (h != 0.0) out[static_cast<std::size_t>(ibra)] += h * coeff;
            });
        }
    }

    std::vector<double> out(bras.n_paths, 0.0);
    for (int t = 0; t < nthread; ++t) {
        const double* part = local.data() + static_cast<std::size_t>(t) * stride;
        for (std::size_t i = 0; i < stride; ++i) out[i] += part[i];
    }
    return out;
}

inline std::vector<double> Hamiltonian::matmat(
    PathBatchView bras,
    PathBatchView kets,
    std::span<const double> x,
    std::size_t nrhs
) const {
    check_paths(bras, "matmat(bras)");
    check_paths(kets, "matmat(kets)");
    if (x.size() != kets.n_paths * nrhs) {
        throw std::invalid_argument("matmat: X size must be n_ket * n_rhs");
    }

    const auto path_space = cached_space(bras);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    const std::size_t stride = bras.n_paths * nrhs;
    std::vector<double> local(static_cast<std::size_t>(nthread) * stride, 0.0);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double* out = local.data() + static_cast<std::size_t>(tid) * stride;
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        double* out = local.data();
        PathScratch ket_scratch;
        ElementScratch elem_scratch(sector_.norb);
        PathState ket;
        VisitScratch visit_scratch;
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double* xket = x.data() + iket * nrhs;
            bool any = false;
            for (std::size_t j = 0; j < nrhs; ++j) {
                if (xket[j] != 0.0) {
                    any = true;
                    break;
                }
            }
            if (!any) continue;

            decode_path(kets[iket], sector_, "matmat(ket)", ket_scratch, ket);
            path_space->visit(ket, visit_scratch, [&](i32 ibra, const OccMove& move) {
                const double h = guga::hij(elem_scratch, ints_, path_space->state(ibra), ket, move);
                if (h == 0.0) return;

                double* y = out + static_cast<std::size_t>(ibra) * nrhs;
                for (std::size_t j = 0; j < nrhs; ++j) y[j] += h * xket[j];
            });
        }
    }

    std::vector<double> out(bras.n_paths * nrhs, 0.0);
    for (int t = 0; t < nthread; ++t) {
        const double* part = local.data() + static_cast<std::size_t>(t) * stride;
        for (std::size_t i = 0; i < stride; ++i) out[i] += part[i];
    }
    return out;
}

} // namespace libdet::guga
