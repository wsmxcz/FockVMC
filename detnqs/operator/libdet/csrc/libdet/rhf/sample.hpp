#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <span>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/rhf/det.hpp>
#include <libdet/sample.hpp>

#include <omp.h>

namespace libdet::rhf {


[[nodiscard]] inline u64 sample_seed(
    u64 seed,
    DetRef ket,
    i64 rep = 0,
    int stream = 0
) noexcept {
    u64 value = mix64(seed ^ 0x243f6a8885a308d3ULL);
    value = mix64(value ^ det_fingerprint(ket));
    value = mix64(value ^ static_cast<u64>(rep + 1));
    value = mix64(
        value ^ (
            stream == 0
                ? 0x13198a2e03707344ULL
                : 0xa4093822299f31d0ULL
        )
    );
    return value;
}


namespace detail {

struct Term {
    Excitation excitation;
    double h = 0.0;
};

struct Item {
    std::vector<Term> term;
};


inline void copy_det_to(std::vector<u64>& words, std::size_t idet, DetRef det) {
    const std::size_t stride = det_size(det.nword());
    u64* dst = words.data() + idet * stride;
    std::copy(det.alpha().begin(), det.alpha().end(), dst);
    std::copy(det.beta().begin(), det.beta().end(), dst + det.nword());
}

[[nodiscard]] inline ::libdet::Conns assemble_conn(
    DetBatchView kets,
    std::span<const Item> items,
    std::span<const double> diag,
    std::span<const double> degree,
    std::size_t n_streams
) {
    const u32 nword = kets.nword;
    const std::size_t n_kets = kets.n_dets;

    ::libdet::Conns out;
    out.nword = nword;
    out.n_kets = n_kets;
    out.n_streams = n_streams;
    out.diag.assign(diag.begin(), diag.end());
    out.degree.assign(degree.begin(), degree.end());
    out.ptr.resize(items.size() + 1u, 0);

    std::size_t n_term = 0;
    for (std::size_t row = 0; row < items.size(); ++row) {
        out.ptr[row] = to_i32(n_term);
        n_term += items[row].term.size();
    }
    out.ptr[items.size()] = to_i32(n_term);
    out.h.resize(n_term);

    const std::size_t stride = det_size(nword);
    copy_batch(out.bra, kets);
    out.bra.resize((n_kets + n_term) * stride);

#pragma omp parallel
    {
        DetScratch scratch(nword);

#pragma omp for schedule(guided)
        for (i64 rr = 0; rr < static_cast<i64>(items.size()); ++rr) {
            const std::size_t row = static_cast<std::size_t>(rr);
            const DetRef ket = kets[row % n_kets];
            const Item& item = items[row];

            for (std::size_t k = 0; k < item.term.size(); ++k) {
                const std::size_t slot = static_cast<std::size_t>(out.ptr[row]) + k;
                const Term& term = item.term[k];
                copy_det_to(out.bra, n_kets + slot, apply(ket, term.excitation, scratch));
                out.h[slot] = term.h;
            }
        }
    }

    return out;
}

} // namespace detail

inline ::libdet::Conns Hamiltonian::sample_conn(
    DetBatchView kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed
) const {
    check_dets(kets, "sample_conn(kets)");
    check_sample_eps(eps1, eps2);
    if (n_streams == 0) {
        throw std::invalid_argument("sample_conn: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_dets) {
        throw std::invalid_argument(
            "sample_conn: counts shape must be (n_streams, n_kets)"
        );
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("sample_conn: counts must be nonnegative");
    }

    std::vector<double> ket_diag(kets.n_dets, 0.0);
    std::vector<double> ket_degree(kets.n_dets, 0.0);
    std::vector<std::vector<::libdet::sample::Hit>> hits(n_streams * kets.n_dets);

    struct Candidate {
        Excitation excitation;
        double h = 0.0;
    };

    std::vector<std::vector<Candidate>> direct;
    std::vector<std::shared_ptr<const Conns>> all;
    const bool use_cache = eps2 > 0.0;

    if (use_cache) {
        all = cached_conns(kets, eps2);
#pragma omp parallel
        {
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                const Conns& conns = *all[iket];
                const ConnSpan win = conns.span(eps1, eps2);
                ket_diag[iket] = conns.diag;
                ket_degree[iket] = win.degree;
                if (!(win.degree > 0.0)) continue;

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    const i64 n_draw = counts[stream * kets.n_dets + iket];
                    ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    ::libdet::sample::draw_span(
                        rng,
                        conns,
                        win.begin,
                        win.end,
                        n_draw,
                        win.degree,
                        targets,
                        hits[stream * kets.n_dets + iket],
                        [&](std::size_t k) noexcept { return conns.terms[k].h; }
                    );
                }
            }
        }
    } else {
        direct.resize(kets.n_dets);
        const auto screen_table_ptr = screen_table(eps2);
#pragma omp parallel
        {
            ElementScratch element(ints_.norb());
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                auto& conns = direct[iket];
                double degree = 0.0;
                visit_external(
                    ints_,
                    screen_table_ptr.get(),
                    kets[iket],
                    element,
                    eps2,
                    [&](Excitation excitation, double h) {
                        const double abs_h = std::abs(h);
                        if (!(abs_h > 0.0) || abs_h >= eps1) return;
                        conns.push_back(Candidate{excitation, h});
                        degree += abs_h;
                    }
                );
                ket_diag[iket] = element.diag();
                ket_degree[iket] = degree;
                if (!(degree > 0.0)) continue;

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    const i64 n_draw = counts[stream * kets.n_dets + iket];
                    ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    ::libdet::sample::draw_scan(
                        rng,
                        0u,
                        conns.size(),
                        n_draw,
                        degree,
                        targets,
                        hits[stream * kets.n_dets + iket],
                        [&](std::size_t k) noexcept { return conns[k].h; }
                    );
                }
            }
        }
    }

    std::vector<detail::Item> items(n_streams * kets.n_dets);
#pragma omp parallel for schedule(guided)
    for (i64 rr = 0; rr < static_cast<i64>(items.size()); ++rr) {
        const std::size_t row = static_cast<std::size_t>(rr);
        const std::size_t iket = row % kets.n_dets;
        const auto& row_hits = hits[row];
        detail::Item& item = items[row];
        std::size_t n_term = 0;
        for (const auto& hit : row_hits) n_term += static_cast<std::size_t>(hit.count);
        item.term.reserve(n_term);

        for (const auto& hit : row_hits) {
            Excitation excitation;
            double h = 0.0;
            if (use_cache) {
                const Conn& term = all[iket]->terms[hit.conn];
                excitation = term.excitation;
                h = term.h;
            } else {
                const Candidate& term = direct[iket][hit.conn];
                excitation = term.excitation;
                h = term.h;
            }
            for (i64 n = 0; n < hit.count; ++n) {
                item.term.push_back(detail::Term{excitation, h});
            }
        }
    }

    return detail::assemble_conn(
        kets,
        items,
        ket_diag,
        ket_degree,
        n_streams
    );
}

inline ::libdet::LocalConn Hamiltonian::local_conn(
    DetBatchView kets,
    double eps1,
    double eps2,
    std::span<const i64> counts,
    u64 seed
) const {
    check_dets(kets, "local_conn(kets)");
    check_sample_eps(eps1, eps2);
    if (counts.size() != kets.n_dets) {
        throw std::invalid_argument("local_conn: counts size must match kets");
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) { return n < 0; })) {
        throw std::invalid_argument("local_conn: counts must be nonnegative");
    }

    struct Term {
        Excitation excitation;
        double h = 0.0;
    };

    struct WeakTerm {
        Excitation excitation;
        double h = 0.0;
        i64 count = 0;
    };

    struct Item {
        double diag = 0.0;
        double strong_degree = 0.0;
        double weak_degree = 0.0;
        std::vector<Term> strong;
        std::vector<WeakTerm> weak;
    };

    std::vector<Item> items(kets.n_dets);
    std::vector<u64> cached_words;
    std::vector<std::size_t> cached_map;
    cached_map.reserve(kets.n_dets);

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        if (counts[iket] == 0) {
            append_det(cached_words, kets[iket]);
            cached_map.push_back(iket);
        }
    }

    if (!cached_map.empty()) {
        const DetBatchView cached{cached_words.data(), cached_map.size(), nword_};
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
            item.strong.reserve(span.end - span.begin);
            for (std::size_t k = span.begin; k < span.end; ++k) {
                const Conn& term = conns.terms[k];
                item.strong.push_back(Term{term.excitation, term.h});
            }
        }
    }

    {
        const auto screen_table_ptr = screen_table(eps2);
#pragma omp parallel
        {
            ElementScratch element(ints_.norb());
            std::vector<Term> weak;
            std::vector<double> targets;
            std::vector<::libdet::sample::Hit> hits;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                const i64 n_draw = counts[iket];
                if (n_draw <= 0) continue;

                Item& item = items[iket];
                item.strong.clear();
                item.weak.clear();
                item.strong_degree = 0.0;
                item.weak_degree = 0.0;
                weak.clear();
                hits.clear();

                visit_external(
                    ints_,
                    screen_table_ptr.get(),
                    kets[iket],
                    element,
                    eps2,
                    [&](Excitation excitation, double h) {
                        const double abs_h = std::abs(h);
                        if (!(abs_h > 0.0)) return;
                        if (abs_h >= eps1) {
                            item.strong.push_back(Term{excitation, h});
                            item.strong_degree += abs_h;
                        } else {
                            weak.push_back(Term{excitation, h});
                            item.weak_degree += abs_h;
                        }
                    }
                );
                item.diag = element.diag();

                if (!(item.weak_degree > 0.0)) continue;
                ::libdet::sample::Rng rng(sample_seed(seed, kets[iket], 0, 0));
                ::libdet::sample::draw_scan(
                    rng,
                    0u,
                    weak.size(),
                    n_draw,
                    item.weak_degree,
                    targets,
                    hits,
                    [&](std::size_t k) noexcept { return weak[k].h; }
                );

                item.weak.reserve(hits.size());
                for (const auto& hit : hits) {
                    const Term& term = weak[hit.conn];
                    item.weak.push_back(WeakTerm{term.excitation, term.h, hit.count});
                }
            }
        }
    }

    ::libdet::LocalConn out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.resize(kets.n_dets);
    out.strong_degree.resize(kets.n_dets);
    out.weak_degree.resize(kets.n_dets);
    out.strong_ptr.resize(kets.n_dets + 1u, 0);
    out.weak_ptr.resize(kets.n_dets + 1u, 0);

    std::size_t n_strong = 0;
    std::size_t n_weak = 0;
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const Item& item = items[iket];
        out.diag[iket] = item.diag;
        out.strong_degree[iket] = item.strong_degree;
        out.weak_degree[iket] = item.weak_degree;
        out.strong_ptr[iket] = to_i32(n_strong);
        out.weak_ptr[iket] = to_i32(n_weak);
        n_strong += item.strong.size();
        n_weak += item.weak.size();
    }
    out.strong_ptr[kets.n_dets] = to_i32(n_strong);
    out.weak_ptr[kets.n_dets] = to_i32(n_weak);

    out.strong_h.resize(n_strong);
    out.weak_h.resize(n_weak);
    out.weak_count.resize(n_weak);

    const std::size_t stride = det_size(nword_);
    copy_batch(out.bra, kets);
    out.bra.resize((kets.n_dets + n_strong + n_weak) * stride);

#pragma omp parallel
    {
        DetScratch scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const DetRef ket = kets[iket];
            const Item& item = items[iket];

            for (std::size_t k = 0; k < item.strong.size(); ++k) {
                const std::size_t slot = static_cast<std::size_t>(out.strong_ptr[iket]) + k;
                const Term& term = item.strong[k];
                detail::copy_det_to(
                    out.bra,
                    kets.n_dets + slot,
                    apply(ket, term.excitation, scratch)
                );
                out.strong_h[slot] = term.h;
            }

            for (std::size_t k = 0; k < item.weak.size(); ++k) {
                const std::size_t slot = static_cast<std::size_t>(out.weak_ptr[iket]) + k;
                const WeakTerm& term = item.weak[k];
                detail::copy_det_to(
                    out.bra,
                    kets.n_dets + n_strong + slot,
                    apply(ket, term.excitation, scratch)
                );
                out.weak_h[slot] = term.h;
                out.weak_count[slot] = term.count;
            }
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
    check_sample_eps(eps1, eps2);
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
    const auto screen_table_ptr = screen_table(screen_table_cutoff(eps2, scale_max));

    struct Bin {
        std::vector<u64> words;
        std::vector<i32> stream;
        std::vector<double> value;
    };

    struct Part {
        explicit Part(std::size_t n) : bin(n) {}
        std::vector<Bin> bin;
    };

    struct Shard {
        Shard(u32 nw, std::size_t ns) : nword(nw), n_streams(ns) {}

        u32 nword = 0;
        std::size_t n_streams = 0;
        std::vector<u64> words;
        std::vector<double> hpsi;
        ankerl::unordered_dense::map<u64, std::vector<i32>> map;

        [[nodiscard]] std::size_t size() const noexcept {
            return words.size() / det_size(nword);
        }

        [[nodiscard]] i32 find_add(DetRef det) {
            const u64 fingerprint = det_fingerprint(det);
            auto& hits = map[fingerprint];
            for (i32 idx : hits) {
                if (det_equal(det_at(words, nword, static_cast<std::size_t>(idx)), det)) {
                    return idx;
                }
            }

            const i32 idx = to_i32(size());
            append_det(words, det);
            hpsi.resize((static_cast<std::size_t>(idx) + 1u) * n_streams, 0.0);
            hits.push_back(idx);
            return idx;
        }
    };

    const int nthread = std::max(1, omp_get_max_threads());
    const std::size_t n_shard = ceil_pow2(2u * static_cast<std::size_t>(nthread));
    const std::size_t mask = n_shard - 1u;

    std::vector<Part> parts;
    parts.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) parts.emplace_back(n_shard);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        Part& part = parts[static_cast<std::size_t>(tid)];

        struct Candidate {
            Excitation excitation;
            double h = 0.0;
        };

        std::vector<std::vector<double>> targets(n_streams);
        std::vector<Candidate> conns;
        std::vector<std::size_t> pos(n_streams, 0u);
        DetScratch scratch(nword_);
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double s = scale[iket];
            const double abs_s = std::abs(s);
            if (abs_s <= 0.0) continue;

            const double h_eps2 = eps2 <= 0.0 ? 0.0 : eps2 / abs_s;
            const double h_eps1 = std::isfinite(eps1) ? eps1 / abs_s : eps1;
            const DetRef ket = kets[iket];

            conns.clear();
            double degree = 0.0;
            visit_external(
                ints_,
                screen_table_ptr.get(),
                ket,
                element,
                h_eps2,
                [&](Excitation excitation, double h) {
                    const double abs_h = std::abs(h);
                    if (!(abs_h > 0.0) || abs_h >= h_eps1) return;
                    const DetRef bra = apply(ket, excitation, scratch);
                    if (exclude_index.find(bra) >= 0) return;
                    conns.push_back(Candidate{excitation, h});
                    degree += abs_h;
                }
            );
            if (!(degree > 0.0)) continue;

            bool any = false;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                ::libdet::sample::Rng rng(sample_seed(seed, ket, 0, static_cast<int>(stream)));
                ::libdet::sample::make_targets(
                    rng,
                    counts[stream * kets.n_dets + iket],
                    degree,
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            std::fill(pos.begin(), pos.end(), 0u);
            double cdf = 0.0;
            for (const Candidate& term : conns) {
                const double abs_h = std::abs(term.h);
                cdf += abs_h;
                if (!(abs_h > 0.0)) continue;

                const DetRef bra = apply(ket, term.excitation, scratch);
                Bin& bin = part.bin[det_fingerprint(bra) & mask];

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
                        static_cast<double>(count) * s * term.h * degree
                        / (static_cast<double>(draws) * abs_h);
                    append_det(bin.words, bra);
                    bin.stream.push_back(to_i32(stream));
                    bin.value.push_back(value);
                }
            }
        }
    }

    std::vector<Shard> shard;
    shard.reserve(n_shard);
    for (std::size_t s = 0; s < n_shard; ++s) shard.emplace_back(nword_, n_streams);

#pragma omp parallel for schedule(static)
    for (i64 ss = 0; ss < static_cast<i64>(n_shard); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        Shard& acc = shard[s];
        std::size_t n_route = 0;
        for (const Part& part : parts) n_route += part.bin[s].value.size();
        acc.map.reserve(n_route);
        acc.words.reserve(n_route * det_size(nword_));

        for (const Part& part : parts) {
            const Bin& bin = part.bin[s];
            for (std::size_t k = 0; k < bin.value.size(); ++k) {
                const i32 ibra = acc.find_add(det_at(bin.words, nword_, k));
                const std::size_t pos =
                    static_cast<std::size_t>(ibra) * n_streams
                    + static_cast<std::size_t>(bin.stream[k]);
                acc.hpsi[pos] += bin.value[k];
            }
        }
    }

    std::vector<std::size_t> start(n_shard + 1u, 0u);
    for (std::size_t s = 0; s < n_shard; ++s) start[s + 1u] = start[s] + shard[s].size();

    Projections out;
    out.nword = nword_;
    out.n_streams = n_streams;
    out.bra.resize(start.back() * det_size(nword_));
    out.hpsi.assign(n_streams * start.back(), 0.0);

#pragma omp parallel for schedule(static)
    for (i64 ss = 0; ss < static_cast<i64>(n_shard); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        const std::size_t stride = det_size(nword_);
        std::copy(
            shard[s].words.begin(),
            shard[s].words.end(),
            out.bra.begin() + static_cast<std::ptrdiff_t>(start[s] * stride)
        );

        for (std::size_t ibra = 0; ibra < shard[s].size(); ++ibra) {
            const std::size_t dst = start[s] + ibra;
            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                out.hpsi[stream * start.back() + dst] =
                    shard[s].hpsi[ibra * n_streams + stream];
            }
        }
    }

    if (start.back() > 0) {
        const DetBatchView bras{out.bra.data(), start.back(), nword_};
        out.diag = diags(bras);
    }
    return out;
}

} // namespace libdet::rhf
