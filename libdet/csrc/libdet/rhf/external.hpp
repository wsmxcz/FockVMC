#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/rhf/screen.hpp>
#include <libdet/spatial/space.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::rhf {

[[nodiscard]] inline bool in_window(double h, AbsWindow win) noexcept {
    const double value = std::abs(h);
    return value > 0.0 && value >= win.lo && value < win.hi;
}

template <class Visit>
inline void visit_bras_prepared(
    const Integral& ints,
    const Screen* screen,
    DetRef ket,
    DetOcc& occ,
    AbsWindow win,
    Visit&& visit
) {
    if (win.hi <= win.lo) return;

    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                sign_single(occ.pref_a, i, a)
                * single_alpha(ints, occ.occ_a, occ.occ_b, i, a);
            if (in_window(h, win)) visit(alpha1(i, a), h);
        }
    }

    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                sign_single(occ.pref_b, i, a)
                * single_beta(ints, occ.occ_a, occ.occ_b, i, a);
            if (in_window(h, win)) visit(beta1(i, a), h);
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
                            * double_alpha(ints, i, j, a, b);
                        if (in_window(h, win)) visit(alpha2(i, j, a, b), h);
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
                            * double_beta(ints, i, j, a, b);
                        if (in_window(h, win)) visit(beta2(i, j, a, b), h);
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
                        if (in_window(h, win)) visit(mixed2(ia, ib, a, b), h);
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

            for (const BraPair& c : screen->same(i, j, win.lo, win.hi)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h =
                    sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (in_window(h, win)) visit(alpha2(i, j, c.a, c.b), h);
            }
        }
    }

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const BraPair& c : screen->same(i, j, win.lo, win.hi)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (in_window(h, win)) visit(beta2(i, j, c.a, c.b), h);
            }
        }
    }

    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const BraPair& c : screen->opposite(ia, ib, win.lo, win.hi)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    sign_single(occ.pref_a, ia, c.a)
                    * sign_single(occ.pref_b, ib, c.b)
                    * c.h;
                if (in_window(h, win)) visit(mixed2(ia, ib, c.a, c.b), h);
            }
        }
    }
}

template <class Visit>
inline void visit_bras(
    const Integral& ints,
    const Screen* screen,
    DetRef ket,
    KetScratch& scratch,
    AbsWindow win,
    Visit&& visit
) {
    fill_occ(ket, ints.norb(), scratch.occ);
    visit_bras_prepared(
        ints,
        screen,
        ket,
        scratch.occ,
        win,
        std::forward<Visit>(visit)
    );
}

// Excitation-driven connections generated from kets.
struct Conns {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::vector<u64> det_words;
    std::vector<double> diag;
    std::vector<i32> ket_ptr;
    std::vector<i32> bra_idx;
    std::vector<double> h;
    std::vector<double> weight;
    std::vector<i32> sample_ket_ptr;
    std::vector<i32> sample_bra_idx;
    std::vector<double> sample_h;
    std::vector<i64> sample_count;
    std::vector<double> sample_weight;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

namespace detail {

struct SampleHit {
    Coupling coupling;
    i64 count = 0;
};

} // namespace detail

} // namespace libdet::rhf

#include <libdet/rhf/sample.hpp>

namespace libdet::rhf {

namespace detail {

struct ProjectPart {
    explicit ProjectPart(u32 nword) : bras(nword) {}

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

} // namespace detail

inline std::vector<u64> Hamiltonian::merge_det_parts(
    std::vector<std::vector<u64>>& parts
) const {
    std::size_t total = 0;
    for (auto& part : parts) {
        sort_unique_dets(part, nword_);
        total += part.size();
    }

    std::vector<u64> out;
    out.reserve(total);
    for (auto& part : parts) {
        out.insert(out.end(), part.begin(), part.end());
    }
    sort_unique_dets(out, nword_);
    return out;
}

inline std::vector<u64> Hamiltonian::expand(
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
                    const DetRef bra =
                        apply(kets[iket], excitation, bra_scratch);
                    if (exclude_index.find(bra) < 0) append_det(words, bra);
                }
            );
        }
    }

    return merge_det_parts(local);
}

inline Projection Hamiltonian::project(
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

    std::vector<detail::ProjectPart> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(nword_);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = local[static_cast<std::size_t>(tid)];
        KetScratch scratch(ints_.norb());
        DetScratch bra_scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& part = local[0];
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
                    const DetRef bra =
                        apply(kets[iket], excitation, bra_scratch);
                    if (exclude_index.find(bra) < 0) {
                        part.add(bra, h * coeff);
                    }
                }
            );
        }
    }

    std::vector<u64> bra_words;
    for (const auto& part : local) {
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

    for (const auto& part : local) {
        for (std::size_t i = 0; i < part.hpsi.size(); ++i) {
            const i32 ibra = bra_index.find(part.bras.get(i));
            if (ibra >= 0) {
                hpsi[static_cast<std::size_t>(ibra)] += part.hpsi[i];
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

inline std::shared_ptr<const KetConns> Hamiltonian::build_ket_conns(
    DetRef ket,
    double eps,
    KetScratch& scratch,
    const Screen* screen_ptr
) const {
    auto conns = std::make_shared<KetConns>();
    conns->cutoff = eps;
    fill_occ(ket, ints_.norb(), scratch.occ);
    conns->diag = diag(ints_, scratch.occ);

    visit_bras_prepared(
        ints_,
        screen_ptr,
        ket,
        scratch.occ,
        abs_window(eps, std::numeric_limits<double>::infinity(), 1.0),
        [&](Excitation excitation, double h) {
            conns->couplings.push_back({excitation, h});
        }
    );
    conns->finish();
    return conns;
}

inline std::vector<std::shared_ptr<const KetConns>>
Hamiltonian::ket_conns(DetBatchView kets, double eps) const {
    auto screen_ptr = screen(eps);
    std::vector<std::shared_ptr<const KetConns>> out(kets.n_dets);
    std::vector<std::size_t> misses;
    misses.reserve(kets.n_dets);

    {
        std::lock_guard<std::mutex> lock(ket_cache_mutex_);
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            out[iket] = ket_cache_.find(kets[iket], eps);
            if (!out[iket]) misses.push_back(iket);
        }
    }
    if (misses.empty()) return out;

#if defined(_OPENMP)
#pragma omp parallel
    {
        KetScratch scratch(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
            const std::size_t iket = misses[static_cast<std::size_t>(ii)];
            out[iket] = build_ket_conns(
                kets[iket],
                eps,
                scratch,
                screen_ptr.get()
            );
        }
    }
#else
    KetScratch scratch(ints_.norb());
    for (std::size_t iket : misses) {
        out[iket] =
            build_ket_conns(kets[iket], eps, scratch, screen_ptr.get());
    }
#endif

    {
        std::lock_guard<std::mutex> lock(ket_cache_mutex_);
        for (std::size_t iket : misses) {
            ket_cache_.insert(kets[iket], out[iket]);
        }
    }
    return out;
}

inline Conns Hamiltonian::conns(
    DetBatchView kets,
    double eps,
    i64 n_sample,
    double sample_eps,
    u64 seed
) const {
    check_dets(kets, "conns(kets)");
    check_eps(eps);
    check_eps(sample_eps);
    if (n_sample < 0) {
        throw std::invalid_argument("conns: sample must be nonnegative");
    }
    if (n_sample > 0 && sample_eps > eps) {
        throw std::invalid_argument("conns: sample_eps must be <= eps");
    }

    const bool sample_weak = n_sample > 0 && sample_eps < eps;
    const auto all = ket_conns(kets, sample_weak ? sample_eps : eps);
    std::vector<double> weak_weight(kets.n_dets, 0.0);
    std::vector<std::vector<detail::SampleHit>> weak_hits(kets.n_dets);

    if (sample_weak) {
#if defined(_OPENMP)
#pragma omp parallel
        {
            std::vector<double> targets;

#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#else
        {
            std::vector<double> targets;
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#endif
                const std::size_t iket = static_cast<std::size_t>(ii);
                const DetRef ket = kets[iket];
                const KetConns& ket_conn = *all[iket];
                const std::size_t begin = ket_conn.count(eps);
                const std::size_t end = ket_conn.count(sample_eps);
                const double weight =
                    ket_conn.prefix_abs[end] - ket_conn.prefix_abs[begin];
                weak_weight[iket] = weight;
                if (!(weight > 0.0)) continue;

                SmallRng rng(sample_seed(seed, ket, 0));
                make_targets(rng, n_sample, weight, targets);
                std::size_t target_pos = 0;
                double cdf = 0.0;

                for (std::size_t k = begin; k < end; ++k) {
                    const Coupling& coupling = ket_conn.couplings[k];
                    cdf += std::abs(coupling.h);
                    i64 count = 0;
                    while (
                        target_pos < targets.size()
                        && targets[target_pos] <= cdf
                    ) {
                        ++count;
                        ++target_pos;
                    }
                    if (count > 0) {
                        weak_hits[iket].push_back({coupling, count});
                    }
                }
            }
        }
    }

    Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.reserve(kets.n_dets);
    out.weight.reserve(kets.n_dets);
    out.ket_ptr.assign(1, 0);
    out.sample_ket_ptr.assign(1, 0);

    DetPool pool(kets);
    DetScratch bra_scratch(nword_);
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const DetRef ket = kets[iket];
        const KetConns& ket_conn = *all[iket];
        const std::size_t nconn = ket_conn.count(eps);
        out.diag.push_back(ket_conn.diag);
        out.weight.push_back(ket_conn.prefix_abs[nconn]);

        for (std::size_t k = 0; k < nconn; ++k) {
            const Coupling& coupling = ket_conn.couplings[k];
            const DetRef bra = apply(ket, coupling.excitation, bra_scratch);
            out.bra_idx.push_back(pool.find_or_add(bra));
            out.h.push_back(coupling.h);
        }
        out.ket_ptr.push_back(to_i32(out.bra_idx.size()));
        out.sample_weight.push_back(sample_weak ? weak_weight[iket] : 0.0);

        for (const auto& hit : weak_hits[iket]) {
            const DetRef bra =
                apply(ket, hit.coupling.excitation, bra_scratch);
            out.sample_bra_idx.push_back(pool.find_or_add(bra));
            out.sample_h.push_back(hit.coupling.h);
            out.sample_count.push_back(hit.count);
        }
        out.sample_ket_ptr.push_back(to_i32(out.sample_bra_idx.size()));
    }

    out.det_words = std::move(pool.words());
    return out;
}

inline std::pair<std::vector<double>, std::vector<i64>>
Hamiltonian::degrees(DetBatchView kets, double eps) const {
    check_dets(kets, "degrees(kets)");
    check_eps(eps);

    const auto all = ket_conns(kets, eps);
    std::vector<double> weight(kets.n_dets, 0.0);
    std::vector<i64> nconn(kets.n_dets, 0);
    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const std::size_t n = all[iket]->count(eps);
        weight[iket] = all[iket]->prefix_abs[n];
        nconn[iket] = static_cast<i64>(n);
    }
    return {std::move(weight), std::move(nconn)};
}

} // namespace libdet::rhf
