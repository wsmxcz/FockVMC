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
#include <libdet/rhf/det.hpp>

#include <omp.h>

namespace libdet::rhf {

[[nodiscard]] inline bool keep_h(double h, double eps) noexcept {
    const double value = std::abs(h);
    return value > 0.0 && value >= eps;
}

template <class Visit>
inline void visit_external_state(
    const Integral& ints,
    const ScreenTable* screen,
    DetRef ket,
    ElementScratch& element,
    double eps,
    Visit&& visit
) {
    const DetOcc& occ = element.occ;

    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                sign_single(occ.pref_a, i, a)
                * element.single_alpha(i, a);
            if (keep_h(h, eps)) visit(alpha1(i, a), h);
        }
    }

    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                sign_single(occ.pref_b, i, a)
                * element.single_beta(i, a);
            if (keep_h(h, eps)) visit(beta1(i, a), h);
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
                        if (keep_h(h, eps)) visit(alpha2(i, j, a, b), h);
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
                        if (keep_h(h, eps)) visit(beta2(i, j, a, b), h);
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
                        if (keep_h(h, eps)) visit(mixed2(ia, ib, a, b), h);
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
                if (keep_h(h, eps)) visit(alpha2(i, j, c.a, c.b), h);
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
                if (keep_h(h, eps)) visit(beta2(i, j, c.a, c.b), h);
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
                if (keep_h(h, eps)) visit(mixed2(ia, ib, c.a, c.b), h);
            }
        }
    }
}

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
    visit_external_state(
        ints,
        screen,
        ket,
        element,
        eps,
        std::forward<Visit>(visit)
    );
}

} // namespace libdet::rhf

#include <libdet/rhf/sample.hpp>

namespace libdet::rhf {

namespace detail {

struct ProjectBuffer {
    explicit ProjectBuffer(u32 nword) : bras(nword) {}

    DetPool bras;
    std::vector<double> hpsi;

    void add(DetRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        const std::size_t pos = static_cast<std::size_t>(idx);
        if (pos == hpsi.size()) {
            hpsi.push_back(value);
        } else {
            hpsi[pos] += value;
        }
    }
};

} // namespace detail

inline std::vector<u64> Hamiltonian::expand(
    DetBatchView kets,
    double eps,
    std::span<const double> scale,
    const DetBatchView* exclude
) const {
    check_dets(kets, "expand(kets)");
    check_eps(eps);
    if (!scale.empty() && scale.size() != kets.n_dets) {
        throw std::invalid_argument("expand: scale size must match kets");
    }

    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "expand(exclude)");

    const double scale_max = scale.empty() ? 1.0 : max_abs(scale);
    auto screen_table_ptr = screen_table(screen_table_cutoff(eps, scale_max));
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
                screen_table_ptr.get(),
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
        sort_unique_dets(part, nword_);
        total += part.size();
    }

    std::vector<u64> words;
    words.reserve(total);
    for (auto& part : local) words.insert(words.end(), part.begin(), part.end());
    sort_unique_dets(words, nword_);
    return words;
}

inline Projection Hamiltonian::project(
    DetBatchView kets,
    std::span<const double> scale,
    double eps,
    const DetBatchView* exclude
) const {
    check_dets(kets, "project(kets)");
    if (scale.size() != kets.n_dets) {
        throw std::invalid_argument("project: scale size must match kets");
    }

    check_eps(eps);
    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "project(exclude)");

    const double scale_max = max_abs(scale);
    auto screen_table_ptr = screen_table(screen_table_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

    const int nthread = std::max(1, omp_get_max_threads());

    std::vector<detail::ProjectBuffer> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(nword_);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = local[static_cast<std::size_t>(tid)];
        ElementScratch element(ints_.norb());
        DetScratch bra_scratch(nword_);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double scale_i = scale[iket];
            const double abs_scale = std::abs(scale_i);
            if (abs_scale <= 0.0) continue;
            const double h_eps = eps <= 0.0 ? 0.0 : eps / abs_scale;

            visit_external(
                ints_,
                screen_table_ptr.get(),
                kets[iket],
                element,
                h_eps,
                [&](Excitation excitation, double h) {
                    const DetRef bra =
                        apply(kets[iket], excitation, bra_scratch);
                    if (exclude_index.find(bra) < 0) {
                        part.add(bra, h * scale_i);
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

inline std::shared_ptr<const Conns> Hamiltonian::make_conns(
    DetRef ket,
    double eps,
    const ScreenTable* screen_table_ptr,
    ElementScratch& element
) const {
    auto conns = std::make_shared<Conns>();
    conns->cutoff = eps;
    element.load(ints_, ket);
    conns->diag = element.diag();

    visit_external_state(
        ints_,
        screen_table_ptr,
        ket,
        element,
        eps,
        [&](Excitation excitation, double h) {
            conns->add(excitation, h);
        }
    );
    conns->finish();
    return conns;
}

inline std::vector<std::shared_ptr<const Conns>>
Hamiltonian::cached_conns(DetBatchView kets, double eps) const {
    auto screen_table_ptr = screen_table(eps);
    std::vector<std::shared_ptr<const Conns>> out(kets.n_dets);

    if (eps <= 0.0) {
#pragma omp parallel
        {
            ElementScratch element(ints_.norb());
#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                out[iket] = make_conns(kets[iket], eps, screen_table_ptr.get(), element);
            }
        }
        return out;
    }

    std::vector<std::size_t> misses;
    misses.reserve(kets.n_dets);

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
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
            out[iket] = make_conns(kets[iket], eps, screen_table_ptr.get(), element);
        }
    }

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        for (std::size_t iket : misses) conn_cache_.insert(kets[iket], out[iket]);
    }
    return out;
}

inline ::libdet::Conns Hamiltonian::conn(
    DetBatchView kets,
    double eps
) const {
    check_dets(kets, "conn(kets)");
    check_eps(eps);

    const auto all = cached_conns(kets, eps);

    ::libdet::Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.reserve(kets.n_dets);
    out.degree.reserve(kets.n_dets);
    out.ptr.assign(1, 0);

    DetPool pool(kets);
    DetScratch bra_scratch(nword_);

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const DetRef ket = kets[iket];
        const Conns& ket_conn = *all[iket];
        out.diag.push_back(ket_conn.diag);

        const ConnSpan win = ket_conn.span(std::numeric_limits<double>::infinity(), eps);
        const double degree = win.degree;
        for (std::size_t k = win.begin; k < win.end; ++k) {
            const Conn& term = ket_conn.terms[k];
            const DetRef bra = apply(ket, term.excitation, bra_scratch);
            out.idx.push_back(pool.find_or_add(bra));
            out.h.push_back(term.h);
        }

        out.degree.push_back(degree);
        out.ptr.push_back(to_i32(out.idx.size()));
    }

    out.bra_words = std::move(pool.words());
    return out;
}

} // namespace libdet::rhf
