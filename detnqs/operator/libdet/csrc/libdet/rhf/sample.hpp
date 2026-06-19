#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/spatial/space.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::rhf {

class SmallRng {
public:
    explicit SmallRng(u64 seed) : state_(seed) {}

    [[nodiscard]] double uniform01() noexcept {
        state_ = splitmix64(state_);
        return static_cast<double>((state_ >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

[[nodiscard]] inline u64 sample_seed(
    u64 seed,
    DetRef ket,
    i64 rep = 0,
    int stream = 0
) noexcept {
    u64 value = splitmix64(seed ^ 0x243f6a8885a308d3ULL);
    value = splitmix64(value ^ det_fingerprint(ket));
    value = splitmix64(value ^ static_cast<u64>(rep + 1));
    value = splitmix64(
        value ^ (
            stream == 0
                ? 0x13198a2e03707344ULL
                : 0xa4093822299f31d0ULL
        )
    );
    return value;
}

inline void make_targets(
    SmallRng& rng,
    i64 n_draw,
    double norm,
    std::vector<double>& out
) {
    out.clear();
    if (n_draw <= 0 || !(norm > 0.0) || !std::isfinite(norm)) return;

    out.reserve(static_cast<std::size_t>(n_draw));
    for (i64 k = 0; k < n_draw; ++k) {
        out.push_back(rng.uniform01() * norm);
    }
    std::sort(out.begin(), out.end());
}

namespace detail {

struct SampleHit {
    Coupling coupling;
    i64 count = 0;
};

struct SampleProjectPart {
    SampleProjectPart(u32 nword, std::size_t streams)
        : n_streams(streams), bras(nword) {}

    std::size_t n_streams = 0;
    DetPool bras;
    std::vector<double> hpsi;

    void add(std::size_t stream, DetRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        const std::size_t pos = static_cast<std::size_t>(idx);
        if ((pos + 1u) * n_streams > hpsi.size()) {
            hpsi.resize((pos + 1u) * n_streams, 0.0);
        }
        hpsi[pos * n_streams + stream] += value;
    }
};

} // namespace detail

inline Conns Hamiltonian::sample_conn(
    DetBatchView kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed,
    bool bra_weight,
    const DetBatchView* include
) const {
    check_dets(kets, "sample_conn(kets)");
    check_window_eps(eps1, eps2);
    if (n_streams == 0) {
        throw std::invalid_argument("sample_conn: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_dets) {
        throw std::invalid_argument(
            "sample_conn: counts shape must be (n_streams, n_kets)"
        );
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) {
        return n < 0;
    })) {
        throw std::invalid_argument("sample_conn: counts must be nonnegative");
    }
    if (include != nullptr) {
        check_dets(*include, "sample_conn(include)");
        if (include->n_dets < kets.n_dets) {
            throw std::invalid_argument("sample_conn: include must start with kets");
        }
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            if (!det_equal((*include)[iket], kets[iket])) {
                throw std::invalid_argument("sample_conn: include must start with kets");
            }
        }
    }

    const auto all = ket_conns(kets, eps2);
    std::vector<double> ket_weight(kets.n_dets, 0.0);
    std::vector<std::vector<detail::SampleHit>> hits(n_streams * kets.n_dets);

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> pos(n_streams);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#else
    {
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> pos(n_streams);
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#endif
            const std::size_t iket = static_cast<std::size_t>(ii);
            const KetConns& src = *all[iket];
            const std::size_t begin = std::isfinite(eps1) ? src.count(eps1) : 0u;
            const std::size_t end = src.count(eps2);
            const double weight = src.prefix_abs[end] - src.prefix_abs[begin];
            ket_weight[iket] = weight;
            if (!(weight > 0.0)) continue;

            std::fill(pos.begin(), pos.end(), 0u);
            bool any = false;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                SmallRng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                make_targets(
                    rng,
                    counts[stream * kets.n_dets + iket],
                    weight,
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            double cdf = 0.0;
            for (std::size_t k = begin; k < end; ++k) {
                const Coupling& coupling = src.couplings[k];
                cdf += std::abs(coupling.h);
                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    i64 count = 0;
                    while (
                        pos[stream] < targets[stream].size()
                        && targets[stream][pos[stream]] <= cdf
                    ) {
                        ++count;
                        ++pos[stream];
                    }
                    if (count > 0) {
                        hits[stream * kets.n_dets + iket].push_back(
                            {coupling, count}
                        );
                    }
                }
            }
        }
    }

    Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.n_streams = n_streams;
    out.ptr.assign(1, 0);

    DetPool pool(include == nullptr ? kets : *include);
    DetScratch bra_scratch(nword_);
    for (std::size_t stream = 0; stream < n_streams; ++stream) {
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            const DetRef ket = kets[iket];
            for (const auto& hit : hits[stream * kets.n_dets + iket]) {
                const DetRef bra = apply(ket, hit.coupling.excitation, bra_scratch);
                const i32 idx = pool.find_or_add(bra);
                for (i64 n = 0; n < hit.count; ++n) {
                    out.bra.push_back(idx);
                    out.h.push_back(hit.coupling.h);
                }
            }
            out.ptr.push_back(to_i32(out.bra.size()));
        }
    }

    const std::size_t pool_size = pool.size();
    out.x_words = std::move(pool.words());
    out.weight.assign(bra_weight ? pool_size : kets.n_dets, 0.0);
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        out.weight[iket] = ket_weight[iket];
    }

    if (bra_weight) {
        const DetBatchView pool_view{
            out.x_words.data(),
            pool_size,
            nword_
        };
        const auto pool_conns = ket_conns(pool_view, eps2);
        for (std::size_t i = kets.n_dets; i < pool_size; ++i) {
            const KetConns& src = *pool_conns[i];
            const std::size_t begin = std::isfinite(eps1) ? src.count(eps1) : 0u;
            const std::size_t end = src.count(eps2);
            out.weight[i] = src.prefix_abs[end] - src.prefix_abs[begin];
        }
    }
    return out;
}

inline Projections Hamiltonian::sample_project(
    DetBatchView kets,
    std::span<const double> scale,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    const DetBatchView* exclude,
    u64 seed
) const {
    check_dets(kets, "sample_project(kets)");
    check_window_eps(eps1, eps2);
    if (scale.size() != kets.n_dets) {
        throw std::invalid_argument("sample_project: scale size must match kets");
    }
    if (n_streams == 0) {
        throw std::invalid_argument("sample_project: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_dets) {
        throw std::invalid_argument(
            "sample_project: counts shape must be (n_streams, n_kets)"
        );
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) {
        return n < 0;
    })) {
        throw std::invalid_argument("sample_project: counts must be nonnegative");
    }
    for (double value : scale) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sample_project: scale must be finite");
        }
    }

    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "sample_project(exclude)");
    const DetIndex exclude_index(base);

    const double scale_max = max_abs(scale);
    const auto all = ket_conns(kets, screen_cutoff(eps2, scale_max));

    Projections out;
    out.nword = nword_;
    out.n_streams = n_streams;
#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    std::vector<detail::SampleProjectPart> parts;
    parts.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) {
        parts.emplace_back(nword_, n_streams);
    }

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = parts[static_cast<std::size_t>(tid)];
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<Coupling> candidates;
        DetScratch bra_scratch(nword_);

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& part = parts[0];
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<Coupling> candidates;
        DetScratch bra_scratch(nword_);
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
            const double coeff = scale[iket];
            const double coeff_abs = std::abs(coeff);
            if (coeff_abs <= 0.0) continue;

            const KetConns& src = *all[iket];
            candidates.clear();
            double weight = 0.0;

            for (const Coupling& coupling : src.couplings) {
                const double term_abs = std::abs(coupling.h) * coeff_abs;
                if (term_abs < eps2 || term_abs >= eps1) continue;

                const DetRef bra =
                    apply(kets[iket], coupling.excitation, bra_scratch);
                if (exclude_index.find(bra) >= 0) continue;

                candidates.push_back(coupling);
                weight += std::abs(coupling.h);
            }
            if (!(weight > 0.0)) continue;

            bool any = false;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                SmallRng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                make_targets(
                    rng,
                    counts[stream * kets.n_dets + iket],
                    weight,
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            std::vector<std::size_t> pos(n_streams, 0u);
            double cdf = 0.0;
            for (const Coupling& coupling : candidates) {
                const double abs_h = std::abs(coupling.h);
                cdf += abs_h;
                if (!(abs_h > 0.0)) continue;

                const DetRef bra =
                    apply(kets[iket], coupling.excitation, bra_scratch);
                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    i64 count = 0;
                    while (
                        pos[stream] < targets[stream].size()
                        && targets[stream][pos[stream]] <= cdf
                    ) {
                        ++count;
                        ++pos[stream];
                    }
                    if (count <= 0) continue;

                    const i64 draws = counts[stream * kets.n_dets + iket];
                    const double value =
                        static_cast<double>(count) * coeff * coupling.h * weight
                        / (static_cast<double>(draws) * abs_h);
                    part.add(stream, bra, value);
                }
            }
        }
    }

    std::vector<u64> bra_words;
    for (const auto& part : parts) {
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
    out.bra_words = std::move(bra_words);
    out.hpsi.assign(n_streams * bras.n_dets, 0.0);

    for (const auto& part : parts) {
        for (std::size_t i = 0; i < part.bras.size(); ++i) {
            const i32 ibra = bra_index.find(part.bras.get(i));
            if (ibra < 0) continue;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                out.hpsi[stream * bras.n_dets + static_cast<std::size_t>(ibra)] +=
                    part.hpsi[i * n_streams + stream];
            }
        }
    }

    if (bras.n_dets > 0) out.diags = diags(bras);
    return out;
}

} // namespace libdet::rhf
