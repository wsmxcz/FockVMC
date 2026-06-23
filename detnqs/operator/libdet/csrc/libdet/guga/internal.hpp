#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>

#include <omp.h>

namespace libdet::guga {

namespace detail {

[[nodiscard]] inline double path_h(
    ElementScratch& elem,
    const Seg2Table& seg2,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const OccMove& move
) {
    const PathDiff diff = path_diff(ket, bra, move);
    if (move.degree == 0) {
        return diff.same_path()
            ? diag(elem, seg2, ints, ket)
            : same_ocfg(elem, seg2, ints, bra, ket, diff);
    }
    if (move.degree == 1) {
        const int p = move.add[0];
        const int q = move.remove[0];
        return single_move(elem, seg2, ints, bra, ket, diff, move, elem.single_coulomb(p, q));
    }
    return double_move(elem, seg2, ints, bra, ket, diff, move);
}

[[nodiscard]] inline bool pass_bound(
    const ScreenTable* table,
    const PathState& ket,
    const PathState& bra,
    const OccMove& move,
    double eps
) {
    if (table == nullptr || eps <= 0.0) return true;
    if (move.degree == 0) {
        return same_path(bra, ket) || table->same_bound(ket) >= eps;
    }
    return table->bound(ket, move) >= eps;
}

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
    return project_internal(bras, kets, scale, eps);
}

inline Projection Hamiltonian::project_internal(
    PathBatchView bras,
    PathBatchView kets,
    std::span<const double> scale,
    double eps
) const {
    const auto ket_space = cached_space(kets);
    const double max_scale = eps == 0.0 ? 0.0 : max_abs(scale);
    const auto screen = (eps > 0.0 && max_scale > 0.0)
        ? screen_table(screen_table_cutoff(eps, max_scale))
        : std::shared_ptr<const ScreenTable>{};

    Projection out;
    out.nword = sector_.nword;
    copy_paths(out.bra, bras);
    out.diag = diags(bras);
    out.hpsi.assign(bras.n_paths, 0.0);

#pragma omp parallel
    {
        PathScratch bra_work;
        PathState bra;
        ElementScratch elem(sector_.norb);
        VisitScratch visit;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_paths); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            decode_path(bras[ibra], sector_, "project(bra)", bra_work, bra);
            elem.load_single(ints_, bra);

            double sum = 0.0;
            ket_space->visit(bra, visit, [&](i32 iket, const OccMove& move) {
                const std::size_t k = static_cast<std::size_t>(iket);
                const double s = scale[k];
                if (s == 0.0) return;

                const double h_eps = eps == 0.0 ? 0.0 : eps / std::abs(s);
                const PathState& ket = ket_space->state(iket);
                if (!detail::pass_bound(screen.get(), bra, ket, move, h_eps)) return;

                const double h = detail::path_h(elem, seg2_, ints_, ket, bra, move);
                const double term = h * s;
                if (term != 0.0 && std::abs(term) >= eps) sum += term;
            });
            out.hpsi[ibra] = sum;
        }
    }
    return out;
}

inline Matrix Hamiltonian::matrix(PathBatchView bras, PathBatchView kets) const {
    check_paths(bras, "matrix(bras)");
    check_paths(kets, "matrix(kets)");

    const auto ket_space = cached_space(kets);
    std::vector<std::vector<i32>> ket_id(bras.n_paths);
    std::vector<std::vector<double>> hval(bras.n_paths);

#pragma omp parallel
    {
        PathScratch bra_work;
        PathState bra;
        ElementScratch elem(sector_.norb);
        VisitScratch visit;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_paths); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            auto& ids = ket_id[ibra];
            auto& values = hval[ibra];
            decode_path(bras[ibra], sector_, "matrix(bra)", bra_work, bra);
            elem.load_single(ints_, bra);

            ket_space->visit(bra, visit, [&](i32 iket, const OccMove& move) {
                const PathState& ket = ket_space->state(iket);
                const double h = detail::path_h(elem, seg2_, ints_, ket, bra, move);
                if (h != 0.0) {
                    ids.push_back(iket);
                    values.push_back(h);
                }
            });
        }
    }

    Matrix out;
    out.n_bra = bras.n_paths;
    out.n_ket = kets.n_paths;
    out.indptr.assign(bras.n_paths + 1u, 0);
    for (std::size_t ibra = 0; ibra < bras.n_paths; ++ibra) {
        out.indptr[ibra + 1u] = out.indptr[ibra] + to_i32(ket_id[ibra].size());
    }

    const std::size_t nnz = static_cast<std::size_t>(out.indptr.back());
    out.indices.resize(nnz);
    out.data.resize(nnz);

#pragma omp parallel for schedule(guided)
    for (i64 ii = 0; ii < static_cast<i64>(bras.n_paths); ++ii) {
        const std::size_t ibra = static_cast<std::size_t>(ii);
        const std::size_t begin = static_cast<std::size_t>(out.indptr[ibra]);
        std::copy(ket_id[ibra].begin(), ket_id[ibra].end(), out.indices.begin() + static_cast<std::ptrdiff_t>(begin));
        std::copy(hval[ibra].begin(), hval[ibra].end(), out.data.begin() + static_cast<std::ptrdiff_t>(begin));
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

    const auto ket_space = cached_space(kets);
    std::vector<double> out(bras.n_paths, 0.0);

#pragma omp parallel
    {
        PathScratch bra_work;
        PathState bra;
        ElementScratch elem(sector_.norb);
        VisitScratch visit;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_paths); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            decode_path(bras[ibra], sector_, "matvec(bra)", bra_work, bra);
            elem.load_single(ints_, bra);

            double sum = 0.0;
            ket_space->visit(bra, visit, [&](i32 iket, const OccMove& move) {
                const double s = x[static_cast<std::size_t>(iket)];
                if (s == 0.0) return;
                const PathState& ket = ket_space->state(iket);
                const double h = detail::path_h(elem, seg2_, ints_, ket, bra, move);
                if (h != 0.0) sum += h * s;
            });
            out[ibra] = sum;
        }
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

    const auto ket_space = cached_space(kets);
    std::vector<double> out(bras.n_paths * nrhs, 0.0);

#pragma omp parallel
    {
        PathScratch bra_work;
        PathState bra;
        ElementScratch elem(sector_.norb);
        VisitScratch visit;
        std::vector<double> sum(nrhs, 0.0);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(bras.n_paths); ++ii) {
            const std::size_t ibra = static_cast<std::size_t>(ii);
            std::fill(sum.begin(), sum.end(), 0.0);
            decode_path(bras[ibra], sector_, "matmat(bra)", bra_work, bra);
            elem.load_single(ints_, bra);

            ket_space->visit(bra, visit, [&](i32 iket, const OccMove& move) {
                const std::size_t k = static_cast<std::size_t>(iket);
                const double* xket = x.data() + k * nrhs;
                bool any = false;
                for (std::size_t j = 0; j < nrhs; ++j) {
                    if (xket[j] != 0.0) {
                        any = true;
                        break;
                    }
                }
                if (!any) return;

                const PathState& ket = ket_space->state(iket);
                const double h = detail::path_h(elem, seg2_, ints_, ket, bra, move);
                if (h == 0.0) return;
                for (std::size_t j = 0; j < nrhs; ++j) sum[j] += h * xket[j];
            });

            double* y = out.data() + ibra * nrhs;
            std::copy(sum.begin(), sum.end(), y);
        }
    }
    return out;
}

} // namespace libdet::guga
