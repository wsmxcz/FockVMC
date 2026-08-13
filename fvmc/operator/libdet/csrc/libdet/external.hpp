#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/screen.hpp>
#include <libdet/det.hpp>

#include <omp.h>

namespace libdet {

template <class Visit>
inline void visit_external(
    const Integral& ints,
    const ScreenTable* screen,
    DetRef ket,
    ElementScratch& element,
    double eps,
    Visit&& visit
) {
    element.load(ints, ket);
    const DetOcc& occ = element.occ;
    const auto keep = [eps](double h) {
        const double value = std::abs(h);
        return value > 0.0 && value >= eps;
    };

    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                sign_single(occ.pref_a, i, a)
                * element.single_alpha(i, a);
            if (keep(h)) visit(alpha1(i, a), h);
        }
    }

    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                sign_single(occ.pref_b, i, a)
                * element.single_beta(i, a);
            if (keep(h)) visit(beta1(i, a), h);
        }
    }

    if (screen == nullptr) {
        for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
            const int i = occ.occ_a[x];

            for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
                const int j = occ.occ_a[y];

                for (std::size_t pa = 0; pa < occ.vir_a.size(); ++pa) {
                    const int a = occ.vir_a[pa];

                    for (std::size_t pb = pa + 1u; pb < occ.vir_a.size(); ++pb) {
                        const int b = occ.vir_a[pb];
                        const double h =
                            sign_double(occ.pref_a, i, j, a, b)
                            * double_same(ints, i, j, a, b);
                        if (keep(h)) visit(alpha2(i, j, a, b), h);
                    }
                }
            }
        }

        for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
            const int i = occ.occ_b[x];

            for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
                const int j = occ.occ_b[y];

                for (std::size_t pa = 0; pa < occ.vir_b.size(); ++pa) {
                    const int a = occ.vir_b[pa];

                    for (std::size_t pb = pa + 1u; pb < occ.vir_b.size(); ++pb) {
                        const int b = occ.vir_b[pb];
                        const double h =
                            sign_double(occ.pref_b, i, j, a, b)
                            * double_same(ints, i, j, a, b);
                        if (keep(h)) visit(beta2(i, j, a, b), h);
                    }
                }
            }
        }

        for (int ia : occ.occ_a) {
            for (int a : occ.vir_a) {
                const double sign_a = sign_single(occ.pref_a, ia, a);

                for (int ib : occ.occ_b) {
                    for (int b : occ.vir_b) {
                        const double h =
                            sign_a
                            * sign_single(occ.pref_b, ib, b)
                            * double_mixed(ints, ia, ib, a, b);
                        if (keep(h)) visit(mixed2(ia, ib, a, b), h);
                    }
                }
            }
        }

        return;
    }

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const ScreenPair& c : screen->same_spin(i, j, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h =
                    sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (keep(h)) visit(alpha2(i, j, c.a, c.b), h);
            }
        }
    }

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const ScreenPair& c : screen->same_spin(i, j, eps)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (keep(h)) visit(beta2(i, j, c.a, c.b), h);
            }
        }
    }

    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const ScreenPair& c : screen->mixed_spin(ia, ib, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    sign_single(occ.pref_a, ia, c.a)
                    * sign_single(occ.pref_b, ib, c.b)
                    * c.h;
                if (keep(h)) visit(mixed2(ia, ib, c.a, c.b), h);
            }
        }
    }
}

inline void copy_det(std::vector<u64>& words, std::size_t idet, DetRef det) {
    const std::size_t stride = det_size(det.nword());
    u64* dst = words.data() + idet * stride;
    std::copy(det.alpha().begin(), det.alpha().end(), dst);
    std::copy(det.beta().begin(), det.beta().end(), dst + det.nword());
}

inline std::vector<u64> Hamiltonian::expand(
    DetBatch kets,
    double eps,
    std::span<const double> scale,
    const DetBatch* exclude
) const {
    check_dets(kets, "expand(kets)");
    check_eps(eps);
    if (!scale.empty() && scale.size() != kets.n_dets) {
        throw std::invalid_argument("expand: scale size must match kets");
    }

    const DetBatch base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "expand(exclude)");

    const double scale_max = scale.empty() ? 1.0 : max_abs(scale);
    auto screen_ptr = screen_table(screen_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

    const int nthread = std::max(1, omp_get_max_threads());

    std::vector<std::vector<u64>> local(static_cast<std::size_t>(nthread));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        ElementScratch element(ints_.norb());
        DetScratch bra_scratch(nword_);
        auto& words = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double scale_i = scale.empty()
                ? 1.0
                : std::abs(scale[iket]);
            if (scale_i <= 0.0) continue;
            const double h_eps = eps <= 0.0 ? 0.0 : eps / scale_i;

            visit_external(
                ints_,
                screen_ptr.get(),
                kets[iket],
                element,
                h_eps,
                [&](Excitation excitation, double) {
                    const DetRef bra =
                        apply(kets[iket], excitation, bra_scratch);
                    if (exclude_index.find(bra) < 0) append_det(words, bra);
                }
            );
        }
    }

    std::size_t total = 0;
    for (auto& part : local) {
        sort_unique(part, nword_);
        total += part.size();
    }

    std::vector<u64> words;
    words.reserve(total);
    for (auto& part : local) words.insert(words.end(), part.begin(), part.end());
    sort_unique(words, nword_);
    return words;
}

inline Projection Hamiltonian::project(
    DetBatch kets,
    std::span<const double> scale,
    double eps,
    const DetBatch* exclude
) const {
    check_dets(kets, "project(kets)");
    if (scale.size() != kets.n_dets) {
        throw std::invalid_argument("project: scale size must match kets");
    }

    check_eps(eps);
    const DetBatch base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "project(exclude)");

    const double scale_max = max_abs(scale);
    auto screen_ptr = screen_table(screen_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

    struct Bin {
        std::vector<u64> words;
        std::vector<double> value;
    };

    struct Part {
        explicit Part(std::size_t n) : bin(n) {}
        std::vector<Bin> bin;
    };

    struct Shard {
        explicit Shard(u32 nw) : nword(nw) {}

        u32 nword = 0;
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
            hpsi.push_back(0.0);
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
        ElementScratch element(ints_.norb());
        DetScratch scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double scale_i = scale[iket];
            const double abs_scale = std::abs(scale_i);
            if (abs_scale <= 0.0) continue;
            const double h_eps = eps <= 0.0 ? 0.0 : eps / abs_scale;
            const DetRef ket = kets[iket];

            visit_external(
                ints_,
                screen_ptr.get(),
                ket,
                element,
                h_eps,
                [&](Excitation excitation, double h) {
                    const DetRef bra = apply(ket, excitation, scratch);
                    if (exclude_index.find(bra) >= 0) return;
                    Bin& bin = part.bin[det_fingerprint(bra) & mask];
                    append_det(bin.words, bra);
                    bin.value.push_back(h * scale_i);
                }
            );
        }
    }

    std::vector<Shard> shard;
    shard.reserve(n_shard);
    for (std::size_t s = 0; s < n_shard; ++s) shard.emplace_back(nword_);

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
                acc.hpsi[static_cast<std::size_t>(ibra)] += bin.value[k];
            }
        }
    }

    std::vector<std::size_t> start(n_shard + 1u, 0u);
    for (std::size_t s = 0; s < n_shard; ++s) start[s + 1u] = start[s] + shard[s].size();

    Projection out;
    out.nword = nword_;
    out.bra.resize(start.back() * det_size(nword_));
    out.hpsi.assign(start.back(), 0.0);

#pragma omp parallel for schedule(static)
    for (i64 ss = 0; ss < static_cast<i64>(n_shard); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        const std::size_t stride = det_size(nword_);
        std::copy(
            shard[s].words.begin(),
            shard[s].words.end(),
            out.bra.begin() + static_cast<std::ptrdiff_t>(start[s] * stride)
        );
        std::copy(
            shard[s].hpsi.begin(),
            shard[s].hpsi.end(),
            out.hpsi.begin() + static_cast<std::ptrdiff_t>(start[s])
        );
    }

    if (start.back() > 0) {
        const DetBatch bras{out.bra.data(), start.back(), nword_};
        out.diag = diag(bras);
    }
    return out;
}

inline std::shared_ptr<const ConnSet> Hamiltonian::make_conns(
    DetRef ket,
    double eps,
    const ScreenTable* screen_ptr,
    ElementScratch& element
) const {
    auto conns = std::make_shared<ConnSet>();
    conns->cutoff = eps;
    visit_external(
        ints_,
        screen_ptr,
        ket,
        element,
        eps,
        [&](Excitation excitation, double h) {
            conns->add(excitation, h);
        }
    );
    conns->diag = element.diag();
    conns->finish();
    return conns;
}

inline std::vector<std::shared_ptr<const ConnSet>>
Hamiltonian::cached_conns(DetBatch kets, double eps) const {
    auto screen_ptr = screen_table(eps);
    std::vector<std::shared_ptr<const ConnSet>> out(kets.n_dets);

    if (eps <= 0.0) {
#pragma omp parallel
        {
            ElementScratch element(ints_.norb());
#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                out[iket] = make_conns(kets[iket], eps, screen_ptr.get(), element);
            }
        }
        return out;
    }

    std::vector<std::size_t> misses;
    misses.reserve(kets.n_dets);

    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            out[iket] = conn_cache_.find(kets[iket], eps);
            if (!out[iket]) misses.push_back(iket);
        }
    }
    if (misses.empty()) return out;

#pragma omp parallel
    {
        ElementScratch element(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
            const std::size_t iket = misses[static_cast<std::size_t>(ii)];
            out[iket] = make_conns(kets[iket], eps, screen_ptr.get(), element);
        }
    }

    {
        std::lock_guard<std::mutex> lock(conn_mutex_);
        for (std::size_t iket : misses) conn_cache_.insert(kets[iket], out[iket]);
    }
    return out;
}

inline Conns Hamiltonian::conn(
    DetBatch kets,
    double eps
) const {
    check_dets(kets, "conn(kets)");
    check_eps(eps);

    const auto all = cached_conns(kets, eps);
    std::vector<ConnSpan> spans(kets.n_dets);

    Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.n_streams = 1u;
    out.diag.resize(kets.n_dets);
    out.degree.resize(kets.n_dets);
    out.ptr.resize(kets.n_dets + 1u, 0);

#pragma omp parallel for schedule(guided)
    for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
        const std::size_t iket = static_cast<std::size_t>(ii);
        const ConnSet& conns = *all[iket];
        spans[iket] = conns.span(
            std::numeric_limits<double>::infinity(),
            eps
        );
        out.diag[iket] = conns.diag;
        out.degree[iket] = spans[iket].degree;
    }

    std::size_t n_term = 0;
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        out.ptr[iket] = to_i64(n_term);
        n_term += spans[iket].end - spans[iket].begin;
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
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const DetRef ket = kets[iket];
            const ConnSpan span = spans[iket];

            for (std::size_t k = span.begin; k < span.end; ++k) {
                const std::size_t slot =
                    static_cast<std::size_t>(out.ptr[iket])
                    + k - span.begin;
                const Conn& term = all[iket]->terms[k];
                copy_det(
                    out.bra,
                    kets.n_dets + slot,
                    apply(ket, term.excitation, scratch)
                );
                out.h[slot] = term.h;
            }
        }
    }

    return out;
}

} // namespace libdet
