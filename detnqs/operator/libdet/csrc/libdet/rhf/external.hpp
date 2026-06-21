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

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::rhf {

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

            for (const Pair& c : screen->same(i, j, win)) {
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

            for (const Pair& c : screen->same(i, j, win)) {
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
            for (const Pair& c : screen->opposite(ia, ib, win)) {
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
            const double scale_i = scale.empty()
                ? 1.0
                : std::abs(scale[iket]);
            const AbsWindow window{
                scale_i <= 0.0 || eps <= 0.0 ? 0.0 : eps / scale_i,
                scale_i <= 0.0 ? 0.0 : std::numeric_limits<double>::infinity()
            };

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
    auto screen_ptr = screen(screen_cutoff(eps, scale_max));
    const DetIndex exclude_index(base);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<detail::ProjectBuffer> local;
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
            const double scale_i = scale[iket];
            const double abs_scale = std::abs(scale_i);
            const AbsWindow window{
                abs_scale <= 0.0 || eps <= 0.0 ? 0.0 : eps / abs_scale,
                abs_scale <= 0.0 ? 0.0 : std::numeric_limits<double>::infinity()
            };

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

inline std::shared_ptr<const Conns> Hamiltonian::build_conns(
    DetRef ket,
    double eps,
    const Screen* screen_ptr,
    KetScratch& scratch
) const {
    auto conns = std::make_shared<Conns>();
    conns->cutoff = eps;
    fill_occ(ket, ints_.norb(), scratch.occ);
    conns->diag = diag(ints_, scratch.occ);

    visit_bras_prepared(
        ints_,
        screen_ptr,
        ket,
        scratch.occ,
        AbsWindow{eps, std::numeric_limits<double>::infinity()},
        [&](Excitation excitation, double h) {
            conns->add(excitation, h);
        }
    );
    conns->finish();
    return conns;
}

inline std::vector<std::shared_ptr<const Conns>>
Hamiltonian::ket_conns(DetBatchView kets, double eps) const {
    auto screen_ptr = screen(eps);
    std::vector<std::shared_ptr<const Conns>> out(kets.n_dets);
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

#if defined(_OPENMP)
#pragma omp parallel
    {
        KetScratch scratch(ints_.norb());

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
            const std::size_t iket = misses[static_cast<std::size_t>(ii)];
            out[iket] = build_conns(
                kets[iket],
                eps,
                screen_ptr.get(),
                scratch
            );
        }
    }
#else
    KetScratch scratch(ints_.norb());
    for (std::size_t iket : misses) {
        out[iket] =
            build_conns(kets[iket], eps, screen_ptr.get(), scratch);
    }
#endif

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        for (std::size_t iket : misses) {
            conn_cache_.insert(kets[iket], out[iket]);
        }
    }
    return out;
}

inline ::libdet::Conns Hamiltonian::conn(
    DetBatchView kets,
    double eps,
    const DetBatchView* include
) const {
    check_dets(kets, "conn(kets)");
    check_eps(eps);

    if (include != nullptr) {
        check_dets(*include, "conn(include)");
        if (include->n_dets < kets.n_dets) {
            throw std::invalid_argument("conn: include must start with kets");
        }
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            if (!det_equal((*include)[iket], kets[iket])) {
                throw std::invalid_argument("conn: include must start with kets");
            }
        }
    }

    const auto all = ket_conns(kets, eps);

    ::libdet::Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.reserve(kets.n_dets);
    out.weight.reserve(kets.n_dets);
    out.ptr.assign(1, 0);

    DetPool pool(include == nullptr ? kets : *include);
    DetScratch bra_scratch(nword_);

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const DetRef ket = kets[iket];
        const Conns& ket_conn = *all[iket];
        double weight = 0.0;

        out.diag.push_back(ket_conn.diag);

        const ConnWindow win = ket_conn.window(AbsWindow{eps, std::numeric_limits<double>::infinity()});
        weight = win.weight;
        for (std::size_t k = win.begin; k < win.end; ++k) {
            const Conn& term = ket_conn.terms[k];
            const DetRef bra = apply(ket, term.excitation, bra_scratch);
            out.bra.push_back(pool.find_or_add(bra));
            out.h.push_back(term.h);
        }

        out.weight.push_back(weight);
        out.ptr.push_back(to_i32(out.bra.size()));
    }

    out.bra_words = std::move(pool.words());
    return out;
}

} // namespace libdet::rhf
