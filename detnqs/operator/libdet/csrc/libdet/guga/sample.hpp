#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <limits>
#include <stdexcept>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>
#include <libdet/sample.hpp>

#include <omp.h>

namespace libdet::guga {


[[nodiscard]] inline u64 sample_seed(
    u64 seed,
    PathRef ket,
    i64 rep = 0,
    int stream = 0
) noexcept {
    u64 value = mix64(seed ^ 0x9e3779b97f4a7c15ULL);
    value = mix64(value ^ path_fingerprint(ket));
    value = mix64(value ^ static_cast<u64>(rep + 1));
    value = mix64(value ^ static_cast<u64>(stream + 17));
    return value;
}

namespace detail {

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

} // namespace detail

inline ::libdet::Conns Hamiltonian::sample_conn(
    PathBatchView kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed
) const {
    check_paths(kets, "sample_conn(kets)");
    check_sample_eps(eps1, eps2);
    if (n_streams == 0) {
        throw std::invalid_argument("sample_conn: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_paths) {
        throw std::invalid_argument("sample_conn: counts shape must be (n_streams, n_kets)");
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("sample_conn: counts must be nonnegative");
    }
    std::vector<double> ket_degree(kets.n_paths, 0.0);
    std::vector<std::vector<::libdet::sample::Hit>> hits(n_streams * kets.n_paths);

    struct DirectConns {
        std::vector<u64> words;
        std::vector<double> h;
    };

    std::vector<DirectConns> direct;
    std::vector<std::shared_ptr<const Conns>> all;
    const bool use_cache = eps2 > 0.0;

    if (use_cache) {
        all = cached_conns(kets, eps2);
#pragma omp parallel
        {
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                const Conns& conns = *all[iket];
                const ConnSpan win = conns.span(eps1, eps2);
                ket_degree[iket] = win.degree;
                if (!(win.degree > 0.0)) continue;

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    const i64 n_draw = counts[stream * kets.n_paths + iket];
                    ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    ::libdet::sample::draw_span(
                        rng,
                        conns,
                        win.begin,
                        win.end,
                        n_draw,
                        win.degree,
                        targets,
                        hits[stream * kets.n_paths + iket],
                        [&](std::size_t k) noexcept { return conns.h[k]; }
                    );
                }
            }
        }
    } else {
        direct.resize(kets.n_paths);
        const auto screen_table_ptr = screen_table(eps2);
#pragma omp parallel
        {
            KetScratch scratch(sector_.norb);
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                auto& conns = direct[iket];
                double degree = 0.0;
                visit_external(
                    scratch,
                    ints_,
                    seg2_,
                    sector_,
                    screen_table_ptr.get(),
                    kets[iket],
                    eps2,
                    false,
                    [&](PathRef bra, double h) {
                        const double abs_h = std::abs(h);
                        if (!(abs_h > 0.0) || abs_h >= eps1) return;
                        append_path(conns.words, bra);
                        conns.h.push_back(h);
                        degree += abs_h;
                    }
                );
                ket_degree[iket] = degree;
                if (!(degree > 0.0)) continue;

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    const i64 n_draw = counts[stream * kets.n_paths + iket];
                    ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    ::libdet::sample::draw_scan(
                        rng,
                        0u,
                        conns.h.size(),
                        n_draw,
                        degree,
                        targets,
                        hits[stream * kets.n_paths + iket],
                        [&](std::size_t k) noexcept { return conns.h[k]; }
                    );
                }
            }
        }
    }

    ::libdet::Conns out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_paths;
    out.n_streams = n_streams;
    out.ptr.assign(1, 0);

    PathPool pool(kets);
    for (std::size_t stream = 0; stream < n_streams; ++stream) {
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            for (const auto& hit : hits[stream * kets.n_paths + iket]) {
                PathRef bra;
                double h = 0.0;
                if (use_cache) {
                    const Conns& conns = *all[iket];
                    bra = conns.bra(hit.conn, sector_.nword);
                    h = conns.h[hit.conn];
                } else {
                    const auto& conns = direct[iket];
                    bra = path_at(conns.words, sector_.nword, hit.conn);
                    h = conns.h[hit.conn];
                }
                const i32 idx = pool.find_or_add(bra);
                for (i64 n = 0; n < hit.count; ++n) {
                    out.idx.push_back(idx);
                    out.h.push_back(h);
                }
            }
            out.ptr.push_back(to_i32(out.idx.size()));
        }
    }

    out.bra_words = std::move(pool.words());
    out.degree.assign(kets.n_paths, 0.0);
    for (std::size_t iket = 0; iket < kets.n_paths; ++iket) out.degree[iket] = ket_degree[iket];
    return out;
}


inline ::libdet::LocalConns Hamiltonian::local_conn(
    PathBatchView kets,
    double eps1,
    double eps2,
    std::span<const i64> counts,
    u64 seed
) const {
    check_paths(kets, "local_conn(kets)");
    check_sample_eps(eps1, eps2);
    if (counts.size() != kets.n_paths) {
        throw std::invalid_argument("local_conn: counts size must match kets");
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("local_conn: counts must be nonnegative");
    }

    struct WeakHit {
        std::size_t pos = 0;
        i64 count = 0;
    };

    struct Item {
        double diag = 0.0;
        double strong_degree = 0.0;
        double weak_degree = 0.0;
        std::vector<u64> strong_words;
        std::vector<double> strong_h;
        std::vector<u64> weak_words;
        std::vector<double> weak_h;
        std::vector<i64> weak_count;
    };

    auto write_strong = [&](const std::vector<std::shared_ptr<const Conns>>& all) {
        ::libdet::LocalConns out;
        out.nword = sector_.nword;
        out.n_kets = kets.n_paths;
        out.diag.reserve(kets.n_paths);
        out.strong_degree.reserve(kets.n_paths);
        out.weak_degree.assign(kets.n_paths, 0.0);
        out.strong_ptr.assign(1, 0);
        out.weak_ptr.assign(1, 0);

        PathPool pool(kets);
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            const Conns& conns = *all[iket];
            const ConnSpan span = conns.span(
                std::numeric_limits<double>::infinity(),
                eps1
            );

            out.diag.push_back(conns.diag);
            out.strong_degree.push_back(span.degree);
            for (std::size_t k = span.begin; k < span.end; ++k) {
                out.strong_idx.push_back(
                    pool.find_or_add(conns.bra(k, sector_.nword))
                );
                out.strong_h.push_back(conns.h[k]);
            }
            out.strong_ptr.push_back(to_i32(out.strong_idx.size()));
            out.weak_ptr.push_back(to_i32(out.weak_idx.size()));
        }
        out.bra_words = std::move(pool.words());
        return out;
    };

    const bool all_zero = std::all_of(
        counts.begin(),
        counts.end(),
        [](i64 n) { return n == 0; }
    );
    if (all_zero) return write_strong(cached_conns(kets, eps1));

    std::vector<Item> items(kets.n_paths);
    std::vector<u64> cached_words;
    std::vector<std::size_t> cached_map;
    cached_map.reserve(kets.n_paths);

    for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
        if (counts[iket] == 0) {
            append_path(cached_words, kets[iket]);
            cached_map.push_back(iket);
        }
    }

    if (!cached_map.empty()) {
        const PathBatchView cached{
            cached_words.data(),
            cached_map.size(),
            sector_.nword
        };
        const auto all = cached_conns(cached, eps1);
        for (std::size_t pos = 0; pos < cached_map.size(); ++pos) {
            const std::size_t iket = cached_map[pos];
            const Conns& conns = *all[pos];
            Item& item = items[iket];
            item.diag = conns.diag;
            const ConnSpan span = conns.span(
                std::numeric_limits<double>::infinity(),
                eps1
            );
            item.strong_degree = span.degree;
            item.strong_words.reserve((span.end - span.begin) * path_size(sector_.nword));
            item.strong_h.reserve(span.end - span.begin);
            for (std::size_t k = span.begin; k < span.end; ++k) {
                append_path(item.strong_words, conns.bra(k, sector_.nword));
                item.strong_h.push_back(conns.h[k]);
            }
        }
    }

    const auto screen_table_ptr = screen_table(eps2);

#pragma omp parallel
    {
        KetScratch scratch(sector_.norb);
        std::vector<u64> weak_words;
        std::vector<double> weak_h;
        std::vector<double> targets;
        std::vector<::libdet::sample::Hit> hits;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const i64 n_draw = counts[iket];
            if (n_draw <= 0) continue;

            Item& item = items[iket];
            item.strong_words.clear();
            item.strong_h.clear();
            item.weak_words.clear();
            item.weak_h.clear();
            item.weak_count.clear();
            weak_words.clear();
            weak_h.clear();
            hits.clear();

            visit_external(
                scratch,
                ints_,
                seg2_,
                sector_,
                screen_table_ptr.get(),
                kets[iket],
                eps2,
                false,
                [&](PathRef bra, double h) {
                    const double abs_h = std::abs(h);
                    if (!(abs_h > 0.0)) return;
                    if (abs_h >= eps1) {
                        append_path(item.strong_words, bra);
                        item.strong_h.push_back(h);
                        item.strong_degree += abs_h;
                    } else {
                        append_path(weak_words, bra);
                        weak_h.push_back(h);
                        item.weak_degree += abs_h;
                    }
                }
            );
            item.diag = guga::diag(scratch.elem, seg2_, ints_, scratch.ket);

            if (!(item.weak_degree > 0.0)) continue;
            ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, 0));
            ::libdet::sample::draw_scan(
                rng,
                0u,
                weak_h.size(),
                n_draw,
                item.weak_degree,
                targets,
                hits,
                [&](std::size_t k) noexcept { return weak_h[k]; }
            );

            item.weak_words.reserve(hits.size() * path_size(sector_.nword));
            item.weak_h.reserve(hits.size());
            item.weak_count.reserve(hits.size());
            for (const auto& hit : hits) {
                append_path(item.weak_words, path_at(weak_words, sector_.nword, hit.conn));
                item.weak_h.push_back(weak_h[hit.conn]);
                item.weak_count.push_back(hit.count);
            }
        }
    }

    ::libdet::LocalConns out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_paths;
    out.diag.reserve(kets.n_paths);
    out.strong_degree.reserve(kets.n_paths);
    out.weak_degree.reserve(kets.n_paths);
    out.strong_ptr.assign(1, 0);
    out.weak_ptr.assign(1, 0);

    PathPool pool(kets);
    for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
        const Item& item = items[iket];
        out.diag.push_back(item.diag);
        out.strong_degree.push_back(item.strong_degree);
        out.weak_degree.push_back(item.weak_degree);

        for (std::size_t k = 0; k < item.strong_h.size(); ++k) {
            const PathRef bra = path_at(item.strong_words, sector_.nword, k);
            out.strong_idx.push_back(pool.find_or_add(bra));
            out.strong_h.push_back(item.strong_h[k]);
        }
        out.strong_ptr.push_back(to_i32(out.strong_idx.size()));

        for (std::size_t k = 0; k < item.weak_h.size(); ++k) {
            const PathRef bra = path_at(item.weak_words, sector_.nword, k);
            out.weak_idx.push_back(pool.find_or_add(bra));
            out.weak_h.push_back(item.weak_h[k]);
            out.weak_count.push_back(item.weak_count[k]);
        }
        out.weak_ptr.push_back(to_i32(out.weak_idx.size()));
    }

    out.bra_words = std::move(pool.words());
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
    check_sample_eps(eps1, eps2);
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

    const double scale_max = max_abs(scale);
    const auto screen_table_ptr = screen_table(screen_table_cutoff(eps2, scale_max));

    Projections out;
    out.nword = sector_.nword;
    out.n_streams = n_streams;
    const int nthread = std::max(1, omp_get_max_threads());
    std::vector<detail::SampleBuffer> parts;
    parts.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) parts.emplace_back(sector_.nword, n_streams);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = parts[static_cast<std::size_t>(tid)];
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<u64> cand_words;
        std::vector<double> cand_h;
        std::vector<std::size_t> pos(n_streams, 0u);
        KetScratch scratch(sector_.norb);

#pragma omp for schedule(static)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double s = scale[iket];
            const double abs_s = std::abs(s);
            if (abs_s <= 0.0) continue;

            const double h_eps2 = eps2 <= 0.0 ? 0.0 : eps2 / abs_s;
            const double h_eps1 = std::isfinite(eps1) ? eps1 / abs_s : eps1;

            cand_words.clear();
            cand_h.clear();
            double degree = 0.0;
            const PathRef ket = kets[iket];
            visit_external(
                scratch,
                ints_,
                seg2_,
                sector_,
                screen_table_ptr.get(),
                ket,
                h_eps2,
                false,
                [&](PathRef bra, double h) {
                    const double abs_h = std::abs(h);
                    if (!(abs_h > 0.0) || abs_h >= h_eps1) return;
                    if (exclude_index.find(bra) >= 0) return;
                    append_path(cand_words, bra);
                    cand_h.push_back(h);
                    degree += abs_h;
                }
            );
            if (!(degree > 0.0)) continue;

            bool any = false;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                ::libdet::sample::make_targets(
                    rng,
                    counts[stream * kets.n_paths + iket],
                    degree,
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            std::fill(pos.begin(), pos.end(), 0u);
            double cdf = 0.0;
            for (std::size_t k = 0; k < cand_h.size(); ++k) {
                const double h = cand_h[k];
                const double abs_h = std::abs(h);
                if (!(abs_h > 0.0)) continue;
                cdf += abs_h;

                const PathRef bra = path_at(cand_words, sector_.nword, k);
                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    i64 count = 0;
                    while (pos[stream] < targets[stream].size() && targets[stream][pos[stream]] <= cdf) {
                        ++count;
                        ++pos[stream];
                    }
                    if (count <= 0) continue;

                    const i64 draws = counts[stream * kets.n_paths + iket];
                    const double value =
                        static_cast<double>(count) * s * h * degree
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
