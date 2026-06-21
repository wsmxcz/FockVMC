#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::guga {

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
    PathRef ket,
    i64 rep = 0,
    int stream = 0
) noexcept {
    u64 value = splitmix64(seed ^ 0x9e3779b97f4a7c15ULL);
    value = splitmix64(value ^ path_fingerprint(ket));
    value = splitmix64(value ^ static_cast<u64>(rep + 1));
    value = splitmix64(value ^ static_cast<u64>(stream + 17));
    return value;
}

namespace detail {

struct SampleHit {
    std::size_t conn = 0;
    i64 count = 0;
};

struct SampleBuffer {
    SampleBuffer(u32 nword, std::size_t streams)
        : n_streams(streams), bras(nword) {}

    std::size_t n_streams = 0;
    PathPool bras;
    std::vector<double> hpsi;

    void add(std::size_t stream, PathRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        const std::size_t pos = static_cast<std::size_t>(idx);
        if ((pos + 1u) * n_streams > hpsi.size()) {
            hpsi.resize((pos + 1u) * n_streams, 0.0);
        }
        hpsi[pos * n_streams + stream] += value;
    }
};

inline void make_targets(
    SmallRng& rng,
    i64 n_draw,
    double norm,
    std::vector<double>& targets
) {
    targets.clear();
    if (n_draw <= 0 || !(norm > 0.0) || !std::isfinite(norm)) return;

    targets.reserve(static_cast<std::size_t>(n_draw));
    for (i64 k = 0; k < n_draw; ++k) targets.push_back(rng.uniform01() * norm);
    std::sort(targets.begin(), targets.end());
}

inline void draw_window_search(
    SmallRng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<SampleHit>& hits
) {
    if (n_draw <= 0 || begin >= end || !(weight > 0.0)) return;

    const double base = conns.prefix_abs[begin];
    const auto first = conns.prefix_abs.begin() + static_cast<std::ptrdiff_t>(begin + 1u);
    const auto last = conns.prefix_abs.begin() + static_cast<std::ptrdiff_t>(end + 1u);

    for (i64 draw = 0; draw < n_draw; ++draw) {
        const double target = base + rng.uniform01() * weight;
        auto it = std::upper_bound(first, last, target);
        if (it == last) --it;
        const std::size_t idx = static_cast<std::size_t>(it - conns.prefix_abs.begin() - 1);
        if (idx >= begin && idx < end) hits.push_back({idx, 1});
    }
}

inline void draw_window_scan(
    SmallRng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<SampleHit>& hits
) {
    make_targets(rng, n_draw, weight, targets);
    if (targets.empty()) return;

    std::size_t pos = 0;
    double cdf = 0.0;
    for (std::size_t k = begin; k < end; ++k) {
        cdf += std::abs(conns.h[k]);
        i64 count = 0;
        while (pos < targets.size() && targets[pos] <= cdf) {
            ++count;
            ++pos;
        }
        if (count > 0) hits.push_back({k, count});
    }
}

inline void draw_window(
    SmallRng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<SampleHit>& hits
) {
    if (n_draw <= 0 || begin >= end || !(weight > 0.0)) return;
    const std::size_t n_conn = end - begin;
    if (static_cast<std::size_t>(n_draw) * 16u < n_conn) {
        draw_window_search(rng, conns, begin, end, n_draw, weight, hits);
    } else {
        draw_window_scan(rng, conns, begin, end, n_draw, weight, targets, hits);
    }
}

} // namespace detail

inline ::libdet::Conns Hamiltonian::sample_conn(
    PathBatchView kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed,
    bool bra_weight,
    const PathBatchView* include
) const {
    check_paths(kets, "sample_conn(kets)");
    check_window_eps(eps1, eps2);
    if (n_streams == 0) {
        throw std::invalid_argument("sample_conn: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_paths) {
        throw std::invalid_argument("sample_conn: counts shape must be (n_streams, n_kets)");
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("sample_conn: counts must be nonnegative");
    }
    if (include != nullptr) {
        check_paths(*include, "sample_conn(include)");
        if (include->n_paths < kets.n_paths) {
            throw std::invalid_argument("sample_conn: include must start with kets");
        }
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            if (!path_equal((*include)[iket], kets[iket])) {
                throw std::invalid_argument("sample_conn: include must start with kets");
            }
        }
    }

    const auto all = ket_conns(kets, eps2);
    std::vector<double> ket_weight(kets.n_paths, 0.0);
    std::vector<std::vector<detail::SampleHit>> hits(n_streams * kets.n_paths);

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<double> targets;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
#else
    {
        std::vector<double> targets;
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
#endif
            const std::size_t iket = static_cast<std::size_t>(ii);
            const Conns& conns = *all[iket];
            const ConnWindow win = conns.window(AbsWindow{eps2, eps1});
            ket_weight[iket] = win.weight;
            if (!(win.weight > 0.0)) continue;

            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                const i64 n_draw = counts[stream * kets.n_paths + iket];
                SmallRng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                detail::draw_window(
                    rng,
                    conns,
                    win.begin,
                    win.end,
                    n_draw,
                    win.weight,
                    targets,
                    hits[stream * kets.n_paths + iket]
                );
            }
        }
    }

    ::libdet::Conns out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_paths;
    out.n_streams = n_streams;
    out.ptr.assign(1, 0);

    PathPool pool(include == nullptr ? kets : *include);
    for (std::size_t stream = 0; stream < n_streams; ++stream) {
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            const Conns& conns = *all[iket];
            for (const auto& hit : hits[stream * kets.n_paths + iket]) {
                const PathRef bra = conns.bra(hit.conn, sector_.nword);
                const i32 idx = pool.find_or_add(bra);
                for (i64 n = 0; n < hit.count; ++n) {
                    out.bra.push_back(idx);
                    out.h.push_back(conns.h[hit.conn]);
                }
            }
            out.ptr.push_back(to_i32(out.bra.size()));
        }
    }

    const std::size_t pool_size = pool.size();
    out.bra_words = std::move(pool.words());
    out.weight.assign(bra_weight ? pool_size : kets.n_paths, 0.0);
    for (std::size_t iket = 0; iket < kets.n_paths; ++iket) out.weight[iket] = ket_weight[iket];

    if (bra_weight) {
        const PathBatchView pool_view{out.bra_words.data(), pool_size, sector_.nword};
        const auto pool_conns = ket_conns(pool_view, eps2);
        for (std::size_t i = kets.n_paths; i < pool_size; ++i) {
            const Conns& conns = *pool_conns[i];
            out.weight[i] = conns.window(AbsWindow{eps2, eps1}).weight;
        }
    }
    return out;
}

inline Projections Hamiltonian::sample_project(
    PathBatchView kets,
    std::span<const double> scale,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    const PathBatchView* exclude,
    u64 seed
) const {
    check_paths(kets, "sample_project(kets)");
    check_window_eps(eps1, eps2);
    if (scale.size() != kets.n_paths) {
        throw std::invalid_argument("sample_project: scale size must match kets");
    }
    if (n_streams == 0) {
        throw std::invalid_argument("sample_project: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_paths) {
        throw std::invalid_argument("sample_project: counts shape must be (n_streams, n_kets)");
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("sample_project: counts must be nonnegative");
    }
    for (double value : scale) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("sample_project: scale must be finite");
        }
    }

    const PathBatchView base = exclude == nullptr ? kets : *exclude;
    check_paths(base, "sample_project(exclude)");
    const PathIndex exclude_index(base);

    const auto all = ket_conns(kets, screen_cutoff(eps2, max_abs(scale)));

    Projections out;
    out.nword = sector_.nword;
    out.n_streams = n_streams;
#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif
    std::vector<detail::SampleBuffer> parts;
    parts.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) parts.emplace_back(sector_.nword, n_streams);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = parts[static_cast<std::size_t>(tid)];
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> candidates;
        std::vector<std::size_t> pos(n_streams, 0u);

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& part = parts[0];
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> candidates;
        std::vector<std::size_t> pos(n_streams, 0u);
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double coeff = scale[iket];
            const double coeff_abs = std::abs(coeff);
            if (coeff_abs <= 0.0) continue;

            const double lo = eps2 <= 0.0 ? 0.0 : eps2 / coeff_abs;
            const double hi = std::isfinite(eps1) ? eps1 / coeff_abs : eps1;
            const Conns& conns = *all[iket];
            const ConnWindow win = conns.window(AbsWindow{lo, hi});
            if (win.begin >= win.end) continue;

            candidates.clear();
            double weight = 0.0;
            for (std::size_t k = win.begin; k < win.end; ++k) {
                const PathRef bra = conns.bra(k, sector_.nword);
                if (exclude_index.find(bra) >= 0) continue;

                candidates.push_back(k);
                weight += std::abs(conns.h[k]);
            }
            if (!(weight > 0.0)) continue;

            bool any = false;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                SmallRng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                detail::make_targets(
                    rng,
                    counts[stream * kets.n_paths + iket],
                    weight,
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            std::fill(pos.begin(), pos.end(), 0u);
            double cdf = 0.0;
            for (std::size_t k : candidates) {
                const double h = conns.h[k];
                const double abs_h = std::abs(h);
                if (!(abs_h > 0.0)) continue;
                cdf += abs_h;

                const PathRef bra = conns.bra(k, sector_.nword);
                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    i64 count = 0;
                    while (pos[stream] < targets[stream].size() && targets[stream][pos[stream]] <= cdf) {
                        ++count;
                        ++pos[stream];
                    }
                    if (count <= 0) continue;

                    const i64 draws = counts[stream * kets.n_paths + iket];
                    const double value =
                        static_cast<double>(count) * coeff * h * weight
                        / (static_cast<double>(draws) * abs_h);
                    part.add(stream, bra, value);
                }
            }
        }
    }

    std::vector<u64> bra_words;
    for (const auto& part : parts) {
        bra_words.insert(bra_words.end(), part.bras.words().begin(), part.bras.words().end());
    }
    sort_unique_paths(bra_words, sector_.nword);

    const PathBatchView bras{
        bra_words.data(),
        bra_words.size() / path_size(sector_.nword),
        sector_.nword
    };
    const PathIndex bra_index(bras);
    out.bra_words = std::move(bra_words);
    out.hpsi.assign(n_streams * bras.n_paths, 0.0);

    for (const auto& part : parts) {
        for (std::size_t i = 0; i < part.bras.size(); ++i) {
            const i32 ibra = bra_index.find(part.bras.get(i));
            if (ibra < 0) continue;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                out.hpsi[stream * bras.n_paths + static_cast<std::size_t>(ibra)] +=
                    part.hpsi[i * n_streams + stream];
            }
        }
    }

    if (bras.n_paths > 0) out.diags = diags(bras);
    return out;
}

} // namespace libdet::guga
