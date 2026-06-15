#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <libdet/hamiltonian.hpp>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet {
namespace {

struct ProjectPart {
    explicit ProjectPart(u32 nword)
        : bras(nword) {}

    DetPool bras;
    std::vector<double> hpsi;

    void add(DetRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        if (static_cast<std::size_t>(idx) == hpsi.size()) {
            hpsi.push_back(value);
        } else {
            hpsi[static_cast<std::size_t>(idx)] += value;
        }
    }
};

} // namespace

std::vector<u64> Hamiltonian::expand(
    DetBatchView kets,
    double eps,
    std::span<const double> coeffs,
    const DetBatchView* exclude
) const {
    check_dets(kets, "expand(kets)");
    check_eps(eps);

    if (!coeffs.empty() && coeffs.size() != kets.n_dets) {
        throw std::invalid_argument("expand: coeffs size must match kets");
    }

    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "expand(exclude)");

    const double scale_max = coeffs.empty() ? 1.0 : max_abs(coeffs);
    auto screen_ptr = screen(screen_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

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
        KetScratch scratch(ints_.norb());
        DetScratch bra_scratch(nword_);
        auto& words = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        KetScratch scratch(ints_.norb());
        DetScratch bra_scratch(nword_);
        auto& words = local[0];

        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
            const double scale = coeffs.empty()
                ? 1.0
                : std::abs(coeffs[iket]);
            const AbsWindow window = abs_window(
                eps,
                std::numeric_limits<double>::infinity(),
                scale
            );

            visit_bras(
                ints_,
                screen_ptr.get(),
                kets[iket],
                scratch,
                window,
                [&](Excitation excitation, double) {
                    const DetRef bra = apply(
                        kets[iket],
                        excitation,
                        bra_scratch
                    );
                    if (exclude_index.find(bra) < 0) append_det(words, bra);
                }
            );
        }
    }

    return merge_det_parts(local);
}

Projection Hamiltonian::project(
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

    const double scale_max = max_abs(coeffs);
    auto screen_ptr = screen(screen_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<ProjectPart> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(nword_);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        ProjectPart& part = local[static_cast<std::size_t>(tid)];
        KetScratch scratch(ints_.norb());
        DetScratch bra_scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        ProjectPart& part = local[0];
        KetScratch scratch(ints_.norb());
        DetScratch bra_scratch(nword_);

        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
            const double coeff = coeffs[iket];
            const AbsWindow window = abs_window(
                eps,
                std::numeric_limits<double>::infinity(),
                std::abs(coeff)
            );

            visit_bras(
                ints_,
                screen_ptr.get(),
                kets[iket],
                scratch,
                window,
                [&](Excitation excitation, double h) {
                    const DetRef bra = apply(
                        kets[iket],
                        excitation,
                        bra_scratch
                    );
                    if (exclude_index.find(bra) >= 0) return;

                    part.add(bra, h * coeff);
                }
            );
        }
    }

    std::vector<u64> bra_words;
    for (const ProjectPart& part : local) {
        bra_words.insert(
            bra_words.end(),
            part.bras.words().begin(),
            part.bras.words().end()
        );
    }
    sort_unique_dets(bra_words, nword_);

    const DetBatchView bras{
        bra_words.data(),
        bra_words.size() / det_size(nword_),
        nword_
    };
    const DetIndex bra_index(bras);
    std::vector<double> hpsi(bras.n_dets, 0.0);

    for (const ProjectPart& part : local) {
        for (std::size_t local_bra = 0; local_bra < part.hpsi.size(); ++local_bra) {
            const i32 ibra = bra_index.find(
                part.bras.get(local_bra)
            );
            if (ibra >= 0) {
                hpsi[static_cast<std::size_t>(ibra)] += part.hpsi[local_bra];
            }
        }
    }

    Projection out;
    out.nword = nword_;
    out.bra_words = std::move(bra_words);
    out.hpsi = std::move(hpsi);

    const DetBatchView out_bras{
        out.bra_words.data(),
        out.hpsi.size(),
        nword_
    };
    out.diags = diags(out_bras);
    return out;
}

} // namespace libdet
