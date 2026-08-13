#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/det.hpp>

#include <omp.h>

namespace libdet::sample {

class Rng {
public:
    explicit Rng(u64 seed) : state_(seed) {}

    [[nodiscard]] double uniform01() noexcept {
        state_ = mix64(state_);
        return static_cast<double>((state_ >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

struct Hit {
    std::size_t conn = 0;
    i64 count = 0;
};

inline void make_targets(
    Rng& rng,
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

template <class Conns>
inline void draw_search(
    Rng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<Hit>& hits
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

template <class HAt>
inline void draw_scan(
    Rng& rng,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<Hit>& hits,
    HAt&& h_at
) {
    make_targets(rng, n_draw, weight, targets);
    if (targets.empty()) return;

    std::size_t pos = 0;
    double cdf = 0.0;
    for (std::size_t k = begin; k < end; ++k) {
        cdf += std::abs(h_at(k));
        i64 count = 0;
        while (pos < targets.size() && targets[pos] <= cdf) {
            ++count;
            ++pos;
        }
        if (count > 0) hits.push_back({k, count});
    }
}

template <class Conns, class HAt>
inline void draw_span(
    Rng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<Hit>& hits,
    HAt&& h_at
) {
    if (n_draw <= 0 || begin >= end || !(weight > 0.0)) return;

    const std::size_t n_conn = end - begin;
    if (static_cast<std::size_t>(n_draw) * 16u < n_conn) {
        draw_search(rng, conns, begin, end, n_draw, weight, hits);
    } else {
        draw_scan(
            rng,
            begin,
            end,
            n_draw,
            weight,
            targets,
            hits,
            std::forward<HAt>(h_at)
        );
    }
}

} // namespace libdet::sample

namespace libdet {

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

inline Conns Hamiltonian::sample_conn(
    DetBatch kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed
) const {
    check_dets(kets, "sample_conn(kets)");
    check_window(eps1, eps2);
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
    std::vector<std::vector<sample::Hit>> hits(n_streams * kets.n_dets);

    struct Candidate {
        Excitation excitation;
        double h = 0.0;
    };

    std::vector<std::vector<Candidate>> direct;
    std::vector<std::shared_ptr<const ConnSet>> all;
    const bool use_cache = eps2 > 0.0;

    if (use_cache) {
        all = cached_conns(kets, eps2);
#pragma omp parallel
        {
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                const ConnSet& conns = *all[iket];
                const ConnSpan win = conns.span(eps1, eps2);
                ket_diag[iket] = conns.diag;
                ket_degree[iket] = win.degree;
                if (!(win.degree > 0.0)) continue;

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    const i64 n_draw = counts[stream * kets.n_dets + iket];
                    sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    sample::draw_span(
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
        const auto screen_ptr = screen_table(eps2);
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
                    screen_ptr.get(),
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
                    sample::Rng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                    sample::draw_scan(
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

    Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.n_streams = n_streams;
    out.diag = std::move(ket_diag);
    out.degree = std::move(ket_degree);
    out.ptr.resize(hits.size() + 1u, 0);

    std::size_t n_term = 0;
    for (std::size_t row = 0; row < hits.size(); ++row) {
        out.ptr[row] = to_i64(n_term);
        for (const sample::Hit& hit : hits[row]) {
            n_term += static_cast<std::size_t>(hit.count);
        }
    }
    out.ptr.back() = to_i64(n_term);
    out.h.resize(n_term);

    const std::size_t stride = det_size(nword_);
    copy_batch(out.bra, kets);
    out.bra.resize((kets.n_dets + n_term) * stride);

#pragma omp parallel
    {
        DetScratch scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 rr = 0; rr < static_cast<i64>(hits.size()); ++rr) {
            const std::size_t row = static_cast<std::size_t>(rr);
            const std::size_t iket = row % kets.n_dets;
            const DetRef ket = kets[iket];
            std::size_t pos = static_cast<std::size_t>(out.ptr[row]);

            for (const sample::Hit& hit : hits[row]) {
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

                const DetRef bra = apply(ket, excitation, scratch);
                for (i64 n = 0; n < hit.count; ++n) {
                    copy_det(out.bra, kets.n_dets + pos, bra);
                    out.h[pos++] = h;
                }
            }
        }
    }

    return out;
}

inline LocalConn Hamiltonian::local_conn(
    DetBatch kets,
    double eps1,
    double eps2,
    std::span<const i64> counts,
    u64 seed
) const {
    check_dets(kets, "local_conn(kets)");
    check_window(eps1, eps2);
    if (counts.size() != kets.n_dets) {
        throw std::invalid_argument("local_conn: counts size must match kets");
    }
    for (i64 count : counts) {
        if (count < 0) {
            throw std::invalid_argument("local_conn: counts must be nonnegative");
        }
    }
    const bool sample_weak = eps2 < eps1;
    if (sample_weak) {
        if (eps2 == 0.0) {
            throw std::invalid_argument("local_conn: weak sampling requires eps2 > 0");
        }
        if (std::any_of(
            counts.begin(),
            counts.end(),
            [](i64 n) { return n == 0; }
        )) {
            throw std::invalid_argument(
                "local_conn: weak sampling requires positive counts"
            );
        }
    }

    struct WeakTerm {
        Excitation excitation;
        double coeff;
    };

    struct DoubleBlock {
        ExcitationKind kind;
        int i;
        int j;
        std::span<const ScreenPair> pairs;
    };

    struct Item {
        double diag = 0.0;
        double strong_degree = 0.0;
        ConnSpan strong;
        std::vector<WeakTerm> weak;
    };

    std::vector<Item> items(kets.n_dets);
    std::vector<std::shared_ptr<const ConnSet>> strong_conns;

    if (std::isfinite(eps1)) {
        strong_conns = cached_conns(kets, eps1);
#pragma omp parallel for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const ConnSet& conns = *strong_conns[iket];
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
        const auto screen_ptr = screen_table(eps2);
#pragma omp parallel
        {
            DetOcc occ;
            std::vector<Conn> singles;
            std::vector<double> single_prefix;
            std::vector<DoubleBlock> doubles;
            std::vector<double> double_prefix;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                const i64 n_sample = counts[iket];

                Item& item = items[iket];
                singles.clear();
                single_prefix.assign(1u, 0.0);
                doubles.clear();
                double_prefix.assign(1u, 0.0);

                const DetRef ket = kets[iket];
                fill_occ(ket, ints_.norb(), occ);
                const auto alpha_bits = ket.alpha();
                const auto beta_bits = ket.beta();

                for (int spin = 0; spin < 2; ++spin) {
                    const auto& occupied = spin == 0 ? occ.occ_a : occ.occ_b;
                    const auto& virtuals = spin == 0 ? occ.vir_a : occ.vir_b;
                    const auto& sign_prefix = spin == 0 ? occ.pref_a : occ.pref_b;
                    const auto spin_bits = spin == 0 ? alpha_bits : beta_bits;

                    for (int i : occupied) {
                        for (int a : virtuals) {
                            const double value = spin == 0
                                ? single_alpha(ints_, occ, i, a)
                                : single_beta(ints_, occ, i, a);
                            const double h = sign_single(sign_prefix, i, a) * value;
                            const double abs_h = std::abs(h);
                            if (abs_h >= eps2 && abs_h < eps1) {
                                const Excitation excitation = spin == 0
                                    ? alpha1(i, a)
                                    : beta1(i, a);
                                singles.push_back({excitation, h});
                                single_prefix.push_back(single_prefix.back() + abs_h);
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
                            const auto pairs =
                                screen_ptr->same_window(i, j, eps1, eps2);
                            double weight = 0.0;
                            for (const ScreenPair& pair : pairs) {
                                if (
                                    !bits::test(spin_bits, pair.a)
                                    && !bits::test(spin_bits, pair.b)
                                ) {
                                    weight += std::abs(pair.h);
                                }
                            }
                            if (!(weight > 0.0)) continue;
                            doubles.push_back({kind, i, j, pairs});
                            double_prefix.push_back(double_prefix.back() + weight);
                        }
                    }
                }

                for (int i : occ.occ_a) {
                    for (int j : occ.occ_b) {
                        const auto pairs =
                            screen_ptr->mixed_window(i, j, eps1, eps2);
                        double weight = 0.0;
                        for (const ScreenPair& pair : pairs) {
                            if (
                                !bits::test(alpha_bits, pair.a)
                                && !bits::test(beta_bits, pair.b)
                            ) {
                                weight += std::abs(pair.h);
                            }
                        }
                        if (!(weight > 0.0)) continue;
                        doubles.push_back({ExcitationKind::mixed2, i, j, pairs});
                        double_prefix.push_back(double_prefix.back() + weight);
                    }
                }

                const double single_norm = single_prefix.back();
                const double proposal_norm = single_norm + double_prefix.back();
                if (!(proposal_norm > 0.0) || !std::isfinite(proposal_norm)) {
                    continue;
                }

                sample::Rng rng(sample_seed(seed, kets[iket], 0, 0));
                const double scale =
                    proposal_norm / static_cast<double>(n_sample);
                std::size_t single_pos = 0;
                std::size_t double_pos = 0;
                std::size_t pair_block = doubles.size();
                std::size_t pair_pos = 0;
                double pair_cdf = 0.0;
                const ScreenPair* pair_current = nullptr;

                // Stratified targets are ordered.
                for (i64 draw = 0; draw < n_sample; ++draw) {
                    double target = (
                        static_cast<double>(draw) + rng.uniform01()
                    ) * scale;

                    Excitation excitation;
                    double h = 0.0;
                    if (doubles.empty() || target < single_norm) {
                        while (
                            single_pos + 1u < singles.size()
                            && single_prefix[single_pos + 1u] <= target
                        ) {
                            ++single_pos;
                        }
                        const Conn& term = singles[single_pos];
                        excitation = term.excitation;
                        h = term.h;
                    } else {
                        target -= single_norm;
                        while (
                            double_pos + 1u < doubles.size()
                            && double_prefix[double_pos + 1u] <= target
                        ) {
                            ++double_pos;
                        }
                        const DoubleBlock& block = doubles[double_pos];
                        if (pair_block != double_pos) {
                            pair_block = double_pos;
                            pair_pos = 0;
                            pair_cdf = 0.0;
                            pair_current = nullptr;
                        }

                        const double pair_target =
                            target - double_prefix[double_pos];
                        const ScreenPair* selected =
                            pair_target < pair_cdf ? pair_current : nullptr;
                        const bool mixed =
                            block.kind == ExcitationKind::mixed2;
                        const auto spin_bits =
                            block.kind == ExcitationKind::alpha2
                                ? alpha_bits
                                : beta_bits;

                        while (
                            selected == nullptr
                            && pair_pos < block.pairs.size()
                        ) {
                            const ScreenPair& candidate =
                                block.pairs[pair_pos++];
                            const bool valid = mixed
                                ? !bits::test(alpha_bits, candidate.a)
                                    && !bits::test(beta_bits, candidate.b)
                                : !bits::test(spin_bits, candidate.a)
                                    && !bits::test(spin_bits, candidate.b);
                            if (!valid) continue;

                            pair_cdf += std::abs(candidate.h);
                            pair_current = &candidate;
                            if (pair_target < pair_cdf) selected = &candidate;
                        }

                        // Keep the final valid pair at the floating-point edge.
                        if (selected == nullptr) selected = pair_current;
                        const ScreenPair& pair = *selected;

                        if (block.kind == ExcitationKind::mixed2) {
                            excitation = mixed2(block.i, block.j, pair.a, pair.b);
                            h =
                                sign_single(occ.pref_a, block.i, pair.a)
                                * sign_single(occ.pref_b, block.j, pair.b)
                                * pair.h;
                        } else {
                            const bool alpha =
                                block.kind == ExcitationKind::alpha2;
                            const auto& sign_prefix =
                                alpha ? occ.pref_a : occ.pref_b;
                            excitation = alpha
                                ? alpha2(block.i, block.j, pair.a, pair.b)
                                : beta2(block.i, block.j, pair.a, pair.b);
                            h = sign_double(
                                sign_prefix,
                                block.i,
                                block.j,
                                pair.a,
                                pair.b
                            ) * pair.h;
                        }
                    }

                    const double coeff = std::copysign(scale, h);
                    if (
                        !item.weak.empty()
                        && item.weak.back().excitation == excitation
                    ) {
                        item.weak.back().coeff += coeff;
                    } else {
                        item.weak.push_back({excitation, coeff});
                    }
                }
            }
        }
    }

    LocalConn out;
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
        out.strong_ptr[iket] = to_i64(n_strong);
        out.weak_ptr[iket] = to_i64(n_weak);
        n_strong += item.strong.end - item.strong.begin;
        n_weak += item.weak.size();
    }
    out.strong_ptr[kets.n_dets] = to_i64(n_strong);
    out.weak_ptr[kets.n_dets] = to_i64(n_weak);

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
                copy_det(
                    out.bra,
                    kets.n_dets + slot,
                    apply(ket, term.excitation, scratch)
                );
                out.strong_h[slot] = term.h;
            }

            for (std::size_t k = 0; k < item.weak.size(); ++k) {
                const std::size_t slot = static_cast<std::size_t>(out.weak_ptr[iket]) + k;
                const WeakTerm& term = item.weak[k];
                copy_det(
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
    DetBatch kets,
    std::span<const double> scale,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    const DetBatch* exclude,
    u64 seed
) const {
    check_dets(kets, "sample_project(kets)");
    check_window(eps1, eps2);
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

    const DetBatch base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "sample_project(exclude)");
    const DetIndex exclude_index(base);

    const double scale_max = max_abs(scale);
    const auto screen_ptr = screen_table(screen_cutoff(eps2, scale_max));

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
                screen_ptr.get(),
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
                sample::Rng rng(sample_seed(seed, ket, 0, static_cast<int>(stream)));
                sample::make_targets(
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
        const DetBatch bras{out.bra.data(), start.back(), nword_};
        out.diag = diag(bras);
    }
    return out;
}

} // namespace libdet
