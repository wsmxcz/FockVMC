#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
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
    i64 n_draw,
    u64 seed
) const {
    check_dets(kets, "local_conn(kets)");
    check_sample_eps(eps1, eps2);
    if (n_draw < 0) {
        throw std::invalid_argument("local_conn: n_draw must be nonnegative");
    }
    const bool sample_weak = eps2 < eps1;
    if (sample_weak) {
        if (eps2 == 0.0) {
            throw std::invalid_argument("local_conn: weak sampling requires eps2 > 0");
        }
        if (n_draw == 0) {
            throw std::invalid_argument("local_conn: weak sampling requires n_draw > 0");
        }
        if (n_draw > std::numeric_limits<i32>::max()) {
            throw std::invalid_argument("local_conn: n_draw exceeds output capacity");
        }
    }

    struct WeakTerm {
        Excitation excitation;
        double coeff;
    };

    struct Item {
        double diag = 0.0;
        double strong_degree = 0.0;
        ConnSpan strong;
        std::size_t target_begin = 0;
        std::size_t target_end = 0;
        std::vector<WeakTerm> weak;
    };

    std::vector<Item> items(kets.n_dets);
    std::vector<std::shared_ptr<const Conns>> strong_conns;

    if (std::isfinite(eps1)) {
        strong_conns = cached_conns(kets, eps1);
#pragma omp parallel for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const Conns& conns = *strong_conns[iket];
            Item& item = items[iket];
            item.diag = conns.diag;
            item.strong = conns.span(
                std::numeric_limits<double>::infinity(),
                eps1
            );
            item.strong_degree = item.strong.degree;
        }
    } else {
        const auto ket_diag = diag(kets);
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            items[iket].diag = ket_diag[iket];
        }
    }

    if (sample_weak) {
        const auto screen_table_ptr = screen_table(eps2);
        auto scan_weak = [&, screen_table_ptr](
            DetRef ket,
            ElementScratch& element,
            auto&& single,
            auto&& block
        ) {
            element.load(ints_, ket);
            const DetOcc& occ = element.occ;

            for (int spin = 0; spin < 2; ++spin) {
                const auto& occupied = spin == 0 ? occ.occ_a : occ.occ_b;
                const auto& virtuals = spin == 0 ? occ.vir_a : occ.vir_b;
                const auto& sign_prefix = spin == 0 ? occ.pref_a : occ.pref_b;

                for (int i : occupied) {
                    for (int a : virtuals) {
                        const double value = spin == 0
                            ? element.single_alpha(i, a)
                            : element.single_beta(i, a);
                        const double h = sign_single(sign_prefix, i, a) * value;
                        const double abs_h = std::abs(h);
                        if (abs_h >= eps2 && abs_h < eps1) {
                            single(spin, i, a, h);
                        }
                    }
                }

                const ExcitationKind kind = spin == 0
                    ? ExcitationKind::alpha2
                    : ExcitationKind::beta2;
                for (std::size_t x = 0; x < occupied.size(); ++x) {
                    const int i = occupied[x];
                    for (std::size_t y = x + 1u; y < occupied.size(); ++y) {
                        const int j = occupied[y];
                        const ScreenWindow window =
                            screen_table_ptr->same_window(i, j, eps1, eps2);
                        if (window.weight() > 0.0) block(kind, i, j, window);
                    }
                }
            }

            for (int i : occ.occ_a) {
                for (int j : occ.occ_b) {
                    const ScreenWindow window =
                        screen_table_ptr->mixed_window(i, j, eps1, eps2);
                    if (window.weight() > 0.0) {
                        block(ExcitationKind::mixed2, i, j, window);
                    }
                }
            }
        };

        std::vector<double> weak_prefix(kets.n_dets + 1u, 0.0);
#pragma omp parallel
        {
            ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                double mass = 0.0;
                scan_weak(
                    kets[iket],
                    element,
                    [&](int, int, int, double h) { mass += std::abs(h); },
                    [&](ExcitationKind, int, int, const ScreenWindow& window) {
                        mass += window.weight();
                    }
                );
                weak_prefix[iket + 1u] = mass;
            }
        }

        double total_mass = 0.0;
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            const double mass = weak_prefix[iket + 1u];
            if (!(mass >= 0.0) || !std::isfinite(mass)) {
                throw std::runtime_error("local_conn: invalid weak proposal mass");
            }
            total_mass += mass;
            weak_prefix[iket + 1u] = total_mass;
        }
        if (!std::isfinite(total_mass)) {
            throw std::runtime_error("local_conn: invalid total weak proposal mass");
        }

        if (total_mass > 0.0) {
            const double delta = total_mass / static_cast<double>(n_draw);
            if (!(delta > 0.0) || !std::isfinite(delta)) {
                throw std::runtime_error("local_conn: invalid weak stratum width");
            }
            std::vector<double> targets(static_cast<std::size_t>(n_draw));
            ::libdet::sample::Rng rng(seed);
            for (i64 draw = 0; draw < n_draw; ++draw) {
                double target = (
                    static_cast<double>(draw) + rng.uniform01()
                ) * delta;
                if (!(target < total_mass)) {
                    target = std::nextafter(total_mass, 0.0);
                }
                targets[static_cast<std::size_t>(draw)] = target;
            }

            std::size_t target_pos = 0;
            for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
                Item& item = items[iket];
                item.target_begin = target_pos;
                const double end = weak_prefix[iket + 1u];
                while (
                    target_pos < targets.size()
                    && targets[target_pos] < end
                ) {
                    ++target_pos;
                }
                item.target_end = target_pos;
            }

#pragma omp parallel
            {
                ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
                for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                    const std::size_t iket = static_cast<std::size_t>(ii);
                    Item& item = items[iket];
                    std::size_t pos = item.target_begin;
                    if (pos == item.target_end) continue;

                    item.weak.reserve(item.target_end - pos);
                    const DetRef ket = kets[iket];
                    double cdf = weak_prefix[iket];

                    auto add = [&](Excitation excitation, double coeff) {
                        if (
                            !item.weak.empty()
                            && item.weak.back().excitation == excitation
                        ) {
                            item.weak.back().coeff += coeff;
                        } else {
                            item.weak.push_back({excitation, coeff});
                        }
                    };

                    scan_weak(
                        ket,
                        element,
                        [&](int spin, int i, int a, double h) {
                            const double next = cdf + std::abs(h);
                            const std::size_t first = pos;
                            while (
                                pos < item.target_end
                                && targets[pos] < next
                            ) {
                                ++pos;
                            }
                            if (pos > first) {
                                const Excitation excitation = spin == 0
                                    ? alpha1(i, a)
                                    : beta1(i, a);
                                add(
                                    excitation,
                                    std::copysign(
                                        delta * static_cast<double>(pos - first),
                                        h
                                    )
                                );
                            }
                            cdf = next;
                        },
                        [&](
                            ExcitationKind kind,
                            int i,
                            int j,
                            const ScreenWindow& window
                        ) {
                            const double next = cdf + window.weight();
                            while (
                                pos < item.target_end
                                && targets[pos] < next
                            ) {
                                const ScreenPair& pair = window.draw(
                                    targets[pos] - cdf
                                );
                                ++pos;

                                Excitation excitation;
                                double h = 0.0;
                                if (kind == ExcitationKind::mixed2) {
                                    if (
                                        bits::test(ket.alpha(), pair.a)
                                        || bits::test(ket.beta(), pair.b)
                                    ) {
                                        continue;
                                    }
                                    excitation = mixed2(i, j, pair.a, pair.b);
                                    h =
                                        sign_single(element.occ.pref_a, i, pair.a)
                                        * sign_single(element.occ.pref_b, j, pair.b)
                                        * pair.h;
                                } else {
                                    const bool alpha = kind == ExcitationKind::alpha2;
                                    const auto spin_bits =
                                        alpha ? ket.alpha() : ket.beta();
                                    if (
                                        bits::test(spin_bits, pair.a)
                                        || bits::test(spin_bits, pair.b)
                                    ) {
                                        continue;
                                    }
                                    const auto& sign_prefix = alpha
                                        ? element.occ.pref_a
                                        : element.occ.pref_b;
                                    excitation = alpha
                                        ? alpha2(i, j, pair.a, pair.b)
                                        : beta2(i, j, pair.a, pair.b);
                                    h = sign_double(
                                        sign_prefix,
                                        i,
                                        j,
                                        pair.a,
                                        pair.b
                                    ) * pair.h;
                                }
                                add(excitation, std::copysign(delta, h));
                            }
                            cdf = next;
                        }
                    );
                }
            }
        }
    }

    ::libdet::LocalConn out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.resize(kets.n_dets);
    out.strong_degree.resize(kets.n_dets);
    out.strong_ptr.resize(kets.n_dets + 1u, 0);
    out.weak_ptr.resize(kets.n_dets + 1u, 0);

    std::size_t n_strong = 0;
    std::size_t n_weak = 0;
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const Item& item = items[iket];
        out.diag[iket] = item.diag;
        out.strong_degree[iket] = item.strong_degree;
        out.strong_ptr[iket] = to_i32(n_strong);
        out.weak_ptr[iket] = to_i32(n_weak);
        n_strong += item.strong.end - item.strong.begin;
        n_weak += item.weak.size();
    }
    out.strong_ptr[kets.n_dets] = to_i32(n_strong);
    out.weak_ptr[kets.n_dets] = to_i32(n_weak);

    out.strong_h.resize(n_strong);
    out.weak_coeff.resize(n_weak);

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

            for (std::size_t k = item.strong.begin; k < item.strong.end; ++k) {
                const std::size_t slot =
                    static_cast<std::size_t>(out.strong_ptr[iket])
                    + k - item.strong.begin;
                const Conn& term = strong_conns[iket]->terms[k];
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
                out.weak_coeff[slot] = term.coeff;
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
        out.diag = diag(bras);
    }
    return out;
}

} // namespace libdet::rhf
