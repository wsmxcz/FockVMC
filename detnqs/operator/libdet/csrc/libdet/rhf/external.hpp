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

struct Conns {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> x_words;
    std::vector<double> diag;
    std::vector<i32> ptr;
    std::vector<i32> bra;
    std::vector<double> h;
    std::vector<double> weight;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

struct Projections {
    u32 nword = 0;
    std::size_t n_streams = 0;
    std::vector<u64> bra_words;
    std::vector<double> hpsi;
    std::vector<double> diags;
};

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
        const std::size_t pos = static_cast<std::size_t>(idx);
        if (pos == hpsi.size()) {
            hpsi.push_back(value);
        } else {
            hpsi[pos] += value;
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
            const AbsWindow window = abs_window(
                eps,
                std::numeric_limits<double>::infinity(),
                scale_i
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
            const double scale_i = scale[iket];
            const AbsWindow window = abs_window(
                eps,
                std::numeric_limits<double>::infinity(),
                std::abs(scale_i)
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

inline Conns Hamiltonian::conn(
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

    Conns out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.diag.reserve(kets.n_dets);
    out.weight.reserve(kets.n_dets);
    out.ptr.assign(1, 0);

    DetPool pool(include == nullptr ? kets : *include);
    DetScratch bra_scratch(nword_);

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const DetRef ket = kets[iket];
        const KetConns& ket_conn = *all[iket];
        double weight = 0.0;

        out.diag.push_back(ket_conn.diag);

        for (const Coupling& coupling : ket_conn.couplings) {
            if (std::abs(coupling.h) < eps) continue;

            const DetRef bra = apply(ket, coupling.excitation, bra_scratch);
            out.bra.push_back(pool.find_or_add(bra));
            out.h.push_back(coupling.h);
            weight += std::abs(coupling.h);
        }

        out.weight.push_back(weight);
        out.ptr.push_back(to_i32(out.bra.size()));
    }

    out.x_words = std::move(pool.words());
    return out;
}

} // namespace libdet::rhf