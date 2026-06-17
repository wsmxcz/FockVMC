#pragma once

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

inline Projection Hamiltonian::project_impl(
    DetBatchView bras,
    DetBatchView kets,
    std::span<const double> coeffs,
    double eps
) const {
    const auto bra_space = cached_csf_space(bras);
    const auto ket_space = cached_csf_space(kets);

    Projection out;
    out.nword = sector_.nword;
    copy_batch(out.bra_words, bras);
    out.hpsi.assign(bras.n_dets, 0.0);
    out.diags.assign(bras.n_dets, 0.0);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (i64 ii = 0; ii < static_cast<i64>(bras.n_dets); ++ii) {
        const std::size_t ibra = static_cast<std::size_t>(ii);
#else
    for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
#endif
        const Csf& bra_csf = bra_space->csf(ibra);
        out.diags[ibra] = element(bra_csf, bra_csf);
    }

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<std::vector<double>> local(
        static_cast<std::size_t>(nthread),
        std::vector<double>(bras.n_dets, 0.0)
    );

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& hpsi = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& hpsi = local[0];
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const double coeff = coeffs[iket];
        if (coeff == 0.0) continue;

        const double cutoff = scaled_eps(eps, std::abs(coeff));
        if (!std::isfinite(cutoff)) continue;

        const Csf& ket_csf = ket_space->csf(iket);
        screen_.visit_bra_cfgs(ket_csf, cutoff, [&](const BraCfg& bra_cfg) {
            for (const CsfSpaceItem& item : bra_space->with_cfg(bra_cfg.cfg)) {
                const std::size_t ibra = static_cast<std::size_t>(item.det);
                const double h = element(bra_space->csf(ibra), ket_csf);
                const double term = h * coeff;
                if (term != 0.0 && std::abs(term) >= eps) {
                    hpsi[ibra] += term;
                }
            }
        });
    }
    }

    for (const auto& hpsi : local) {
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
            out.hpsi[ibra] += hpsi[ibra];
        }
    }

    return out;
}

inline Matrix Hamiltonian::matrix(DetBatchView bras, DetBatchView kets) const {
    check_dets(bras, "matrix(bras)");
    check_dets(kets, "matrix(kets)");

    const auto bra_space = cached_csf_space(bras);
    const auto ket_space = cached_csf_space(kets);
    std::vector<std::vector<std::pair<i32, double>>> bra_terms(bras.n_dets);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
    std::vector<std::vector<std::vector<std::pair<i32, double>>>> local(
        static_cast<std::size_t>(nthread),
        std::vector<std::vector<std::pair<i32, double>>>(bras.n_dets)
    );

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& thread_terms = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    auto& thread_terms = bra_terms;
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const Csf& ket_csf = ket_space->csf(iket);

        screen_.visit_bra_cfgs(ket_csf, 0.0, [&](const BraCfg& bra_cfg) {
            for (const CsfSpaceItem& item : bra_space->with_cfg(bra_cfg.cfg)) {
                const std::size_t ibra = static_cast<std::size_t>(item.det);
                const double h = element(bra_space->csf(ibra), ket_csf);
                if (h != 0.0) thread_terms[ibra].push_back({to_i32(iket), h});
            }
        });
    }
#if defined(_OPENMP)
    }

    for (auto& thread_terms : local) {
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
            bra_terms[ibra].insert(
                bra_terms[ibra].end(),
                thread_terms[ibra].begin(),
                thread_terms[ibra].end()
            );
        }
    }
#endif

    Matrix out;
    out.n_bra = bras.n_dets;
    out.n_ket = kets.n_dets;
    out.indptr.assign(bras.n_dets + 1u, 0);

    for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
        out.indptr[ibra + 1u] =
            out.indptr[ibra] + to_i32(bra_terms[ibra].size());
    }

    out.indices.reserve(static_cast<std::size_t>(out.indptr.back()));
    out.data.reserve(static_cast<std::size_t>(out.indptr.back()));
    for (const auto& terms : bra_terms) {
        for (const auto& [iket, h] : terms) {
            out.indices.push_back(iket);
            out.data.push_back(h);
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

    const auto bra_space = cached_csf_space(bras);
    const auto ket_space = cached_csf_space(kets);
    std::vector<double> out(bras.n_dets, 0.0);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
    std::vector<std::vector<double>> local(
        static_cast<std::size_t>(nthread),
        std::vector<double>(bras.n_dets, 0.0)
    );

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& y = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& y = out;
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const double coeff = x[iket];
        if (coeff == 0.0) continue;

        const Csf& ket_csf = ket_space->csf(iket);
        screen_.visit_bra_cfgs(ket_csf, 0.0, [&](const BraCfg& bra_cfg) {
            for (const CsfSpaceItem& item : bra_space->with_cfg(bra_cfg.cfg)) {
                const std::size_t ibra = static_cast<std::size_t>(item.det);
                const double h = element(bra_space->csf(ibra), ket_csf);
                if (h != 0.0) y[ibra] += h * coeff;
            }
        });
    }
    }

#if defined(_OPENMP)
    for (const auto& y : local) {
        for (std::size_t ibra = 0; ibra < bras.n_dets; ++ibra) {
            out[ibra] += y[ibra];
        }
    }
#endif

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

    const auto bra_space = cached_csf_space(bras);
    const auto ket_space = cached_csf_space(kets);
    std::vector<double> out(bras.n_dets * nrhs, 0.0);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
    std::vector<std::vector<double>> local(
        static_cast<std::size_t>(nthread),
        std::vector<double>(bras.n_dets * nrhs, 0.0)
    );

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& yout = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& yout = out;
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const double* xrow = x.data() + iket * nrhs;

        bool any = false;
        for (std::size_t j = 0; j < nrhs; ++j) {
            if (xrow[j] != 0.0) {
                any = true;
                break;
            }
        }
        if (!any) continue;

        const Csf& ket_csf = ket_space->csf(iket);
        screen_.visit_bra_cfgs(ket_csf, 0.0, [&](const BraCfg& bra_cfg) {
            for (const CsfSpaceItem& item : bra_space->with_cfg(bra_cfg.cfg)) {
                const std::size_t ibra = static_cast<std::size_t>(item.det);
                const double h = element(bra_space->csf(ibra), ket_csf);
                if (h == 0.0) continue;

                double* y = yout.data() + ibra * nrhs;
                for (std::size_t j = 0; j < nrhs; ++j) {
                    y[j] += h * xrow[j];
                }
            }
        });
    }
    }

#if defined(_OPENMP)
    for (const auto& yout : local) {
        for (std::size_t i = 0; i < out.size(); ++i) out[i] += yout[i];
    }
#endif

    return out;
}

} // namespace libdet::guga
