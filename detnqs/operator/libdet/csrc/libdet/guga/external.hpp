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

#include <libdet/guga/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::guga {

namespace detail {

struct ProjectBuffer {
    explicit ProjectBuffer(u32 nword) : bras(nword) {}

    PathPool bras;
    std::vector<double> hpsi;

    void add(PathRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        const std::size_t pos = static_cast<std::size_t>(idx);
        if (pos == hpsi.size()) hpsi.push_back(value);
        else hpsi[pos] += value;
    }
};

} // namespace detail

struct KetScratch {
    explicit KetScratch(int norb) : elem(norb) {}

    ElementScratch elem;
    PathScratch ket_scratch;
    PathScratch bra_scratch;
    PathState ket;
    PathState bra;
    std::vector<Step> steps;
    std::vector<u64> words;
    std::vector<unsigned char> occ;
    std::vector<unsigned char> removed;

    [[nodiscard]] PathRef encode(std::span<const Step> step, u32 nword) {
        encode_path(step, nword, words);
        return PathRef(words.data(), words.data() + static_cast<std::size_t>(nword), nword);
    }
};

namespace detail {

[[nodiscard]] inline bool apply_move(
    Occ ket,
    const Move& move,
    std::vector<unsigned char>& removed,
    std::vector<unsigned char>& occ
) {
    if (move.degree == 1u) {
        return move_one_occ(
            ket,
            static_cast<int>(move.remove[0]),
            static_cast<int>(move.add[0]),
            occ
        );
    }
    if (move.degree == 2u) {
        return remove_pair_occ(
            ket,
            static_cast<int>(move.remove[0]),
            static_cast<int>(move.remove[1]),
            removed
        ) && add_pair_occ(
            removed,
            static_cast<int>(move.add[0]),
            static_cast<int>(move.add[1]),
            occ
        );
    }
    return false;
}


} // namespace detail

template <class Visit>
inline void occ_paths_rec(
    std::span<const unsigned char> occ,
    int p,
    int norb,
    int spin,
    int target_spin,
    std::vector<Step>& steps,
    Visit&& visit
) {
    if (p == norb) {
        if (spin == target_spin) visit(steps);
        return;
    }

    const unsigned char n = occ[static_cast<std::size_t>(p)];
    if (n == 0u || n == 2u) {
        steps[static_cast<std::size_t>(p)] = n == 0u ? Step::empty : Step::doubly;
        occ_paths_rec(
            occ, p + 1, norb, spin, target_spin, steps, std::forward<Visit>(visit)
        );
        return;
    }

    steps[static_cast<std::size_t>(p)] = Step::up;
    occ_paths_rec(
        occ, p + 1, norb, spin + 1, target_spin, steps, std::forward<Visit>(visit)
    );

    if (spin > 0) {
        steps[static_cast<std::size_t>(p)] = Step::down;
        occ_paths_rec(
            occ, p + 1, norb, spin - 1, target_spin, steps, std::forward<Visit>(visit)
        );
    }
}

template <class Visit>
inline void visit_occ_paths(
    std::span<const unsigned char> occ,
    Sector sector,
    std::vector<Step>& steps,
    Visit&& visit
) {
    steps.resize(static_cast<std::size_t>(sector.norb));
    occ_paths_rec(
        occ, 0, sector.norb, 0, sector.spin_twice, steps, std::forward<Visit>(visit)
    );
}

template <class Emit>
inline void visit_bras_state(
    KetScratch& scratch,
    const Integral& ints,
    Sector sector,
    const Screen& screen,
    PathRef ket_words,
    const PathState& ket,
    AbsWindow win,
    bool include_diag,
    Emit&& emit
) {
    if (win.hi <= win.lo) return;

    if (include_diag) {
        const double h = hij(scratch.elem, ints, ket, ket, OccMove{});
        if (in_window(h, win)) emit(ket_words, h);
    }

    auto add_bra = [&](std::span<const Step> steps, const OccMove& move) {
        const PathRef bra_ref = scratch.encode(steps, sector.nword);
        if (path_equal(bra_ref, ket_words)) return;

        load_path(steps, sector, "visit_bras(bra)", scratch.bra_scratch, scratch.bra);
        const double h = hij(scratch.elem, ints, scratch.bra, ket, move);
        if (in_window(h, win)) emit(bra_ref, h);
    };

    auto add_occ = [&](Occ occ, const OccMove& move) {
        visit_occ_paths(occ, sector, scratch.steps, [&](const std::vector<Step>& steps) {
            add_bra(std::span<const Step>(steps.data(), steps.size()), move);
        });
    };

    if (win.lo == 0.0 || screen.same(ket.occ) >= win.lo) {
        add_occ(ket.occ, OccMove{});
    }

    for (const Move& move : screen.one(win.lo)) {
        if (!detail::apply_move(ket.occ, move, scratch.removed, scratch.occ)) continue;
        if (win.lo > 0.0 && screen.bound(ket.occ, move) < win.lo) continue;
        add_occ(scratch.occ, move.occ_move());
    }

    for (const Move& move : screen.two(win.lo)) {
        if (!detail::apply_move(ket.occ, move, scratch.removed, scratch.occ)) continue;
        if (win.lo > 0.0 && screen.bound(ket.occ, move) < win.lo) continue;
        add_occ(scratch.occ, move.occ_move());
    }
}

template <class Emit>
inline void visit_bras(
    KetScratch& scratch,
    const Integral& ints,
    Sector sector,
    const Screen& screen,
    PathRef ket_words,
    AbsWindow win,
    bool include_diag,
    Emit&& emit
) {
    decode_path(ket_words, sector, "visit_bras(ket)", scratch.ket_scratch, scratch.ket);
    visit_bras_state(
        scratch,
        ints,
        sector,
        screen,
        ket_words,
        scratch.ket,
        win,
        include_diag,
        std::forward<Emit>(emit)
    );
}

inline double Hamiltonian::hij(PathRef bra, PathRef ket) const {
    check_one(bra, "hij(bra)");
    check_one(ket, "hij(ket)");
    PathScratch bra_scratch;
    PathScratch ket_scratch;
    PathState bra_state;
    PathState ket_state;
    decode_path(bra, sector_, "hij(bra)", bra_scratch, bra_state);
    decode_path(ket, sector_, "hij(ket)", ket_scratch, ket_state);
    ElementScratch elem_scratch(sector_.norb);
    return guga::hij(elem_scratch, ints_, bra_state, ket_state);
}

inline std::vector<double> Hamiltonian::diags(PathBatchView paths) const {
    check_paths(paths, "diags");

    std::vector<double> out(paths.n_paths, 0.0);
    PathScratch scratch;
    ElementScratch elem_scratch(sector_.norb);
    PathState item;
    for (std::size_t ipath = 0; ipath < paths.n_paths; ++ipath) {
        decode_path(paths[ipath], sector_, "diags", scratch, item);
        out[ipath] = guga::hij(elem_scratch, ints_, item, item, OccMove{});
    }
    return out;
}

inline std::shared_ptr<const Conns> Hamiltonian::build_conns(
    PathRef ket,
    double eps,
    const Screen& screen,
    KetScratch& scratch
) const {
    auto out = std::make_shared<Conns>();
    out->cutoff = eps;

    decode_path(ket, sector_, "ket_conns(ket)", scratch.ket_scratch, scratch.ket);
    out->diag = guga::hij(scratch.elem, ints_, scratch.ket, scratch.ket, OccMove{});

    visit_bras_state(
        scratch,
        ints_,
        sector_,
        screen,
        ket,
        scratch.ket,
        AbsWindow{eps, std::numeric_limits<double>::infinity()},
        false,
        [&](PathRef bra, double h) {
            out->add(bra, h);
        }
    );

    out->finish(sector_.nword);
    return out;
}

inline std::shared_ptr<const Conns> Hamiltonian::ket_conns(
    PathRef ket,
    double eps
) const {
    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        if (auto hit = conn_cache_.find(ket, eps)) return hit;
    }

    const auto screen_ptr = screen(eps);
    KetScratch scratch(sector_.norb);
    auto fresh = build_conns(ket, eps, *screen_ptr, scratch);

    std::lock_guard<std::mutex> lock(conn_cache_mutex_);
    if (auto hit = conn_cache_.find(ket, eps)) return hit;
    conn_cache_.insert(ket, fresh);
    return fresh;
}

inline std::vector<std::shared_ptr<const Conns>>
Hamiltonian::ket_conns(PathBatchView kets, double eps) const {
    check_paths(kets, "ket_conns(kets)");
    check_eps(eps);

    std::vector<std::shared_ptr<const Conns>> out(kets.n_paths);
    std::vector<std::size_t> misses;
    misses.reserve(kets.n_paths);

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            out[iket] = conn_cache_.find(kets[iket], eps);
            if (!out[iket]) misses.push_back(iket);
        }
    }
    if (misses.empty()) return out;

    const auto screen_ptr = screen(eps);

#if defined(_OPENMP)
#pragma omp parallel
    {
        KetScratch scratch(sector_.norb);
#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
            const std::size_t iket = misses[static_cast<std::size_t>(ii)];
            out[iket] = build_conns(kets[iket], eps, *screen_ptr, scratch);
        }
    }
#else
    KetScratch scratch(sector_.norb);
    for (std::size_t iket : misses) {
        out[iket] = build_conns(kets[iket], eps, *screen_ptr, scratch);
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


inline std::vector<u64> Hamiltonian::expand(
    PathBatchView kets,
    double eps,
    std::span<const double> scale,
    const PathBatchView* exclude
) const {
    check_paths(kets, "expand(kets)");
    check_eps(eps);
    if (!scale.empty() && scale.size() != kets.n_paths) {
        throw std::invalid_argument("expand: scale size must match kets");
    }

    const PathBatchView base = exclude == nullptr ? kets : *exclude;
    check_paths(base, "expand(exclude)");
    const PathIndex exclude_index(base);
    const double max_scale = scale.empty() ? 1.0 : max_abs(scale);
    const auto screen_ptr = screen(screen_cutoff(eps, max_scale));

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
        auto& words = local[static_cast<std::size_t>(tid)];
        KetScratch scratch(sector_.norb);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& words = local[0];
        KetScratch scratch(sector_.norb);
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double scale_i = scale.empty() ? 1.0 : std::abs(scale[iket]);
            if (scale_i == 0.0) continue;

            const double lo = eps == 0.0 ? 0.0 : eps / scale_i;
            visit_bras(
                scratch,
                ints_,
                sector_,
                *screen_ptr,
                kets[iket],
                AbsWindow{lo, std::numeric_limits<double>::infinity()},
                false,
                [&](PathRef bra, double) {
                    if (exclude_index.find(bra) < 0) append_path(words, bra);
                }
            );
        }
    }

    std::vector<u64> words;
    for (auto& part : local) words.insert(words.end(), part.begin(), part.end());
    sort_unique_paths(words, sector_.nword);
    return words;
}

inline Projection Hamiltonian::project(
    PathBatchView kets,
    std::span<const double> scale,
    double eps,
    const PathBatchView* exclude
) const {
    check_paths(kets, "project(kets)");
    if (scale.size() != kets.n_paths) {
        throw std::invalid_argument("project: scale size must match kets");
    }
    check_eps(eps);

    const PathBatchView base = exclude == nullptr ? kets : *exclude;
    check_paths(base, "project(exclude)");
    const PathIndex exclude_index(base);
    const double max_scale = scale.empty() ? 1.0 : max_abs(scale);
    const auto screen_ptr = screen(screen_cutoff(eps, max_scale));

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<detail::ProjectBuffer> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(sector_.nword);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = local[static_cast<std::size_t>(tid)];
        KetScratch scratch(sector_.norb);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& part = local[0];
        KetScratch scratch(sector_.norb);
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
#endif
            const double scale_i = scale[iket];
            const double abs_scale = std::abs(scale_i);
            if (abs_scale == 0.0) continue;

            const double lo = eps == 0.0 ? 0.0 : eps / abs_scale;
            visit_bras(
                scratch,
                ints_,
                sector_,
                *screen_ptr,
                kets[iket],
                AbsWindow{lo, std::numeric_limits<double>::infinity()},
                false,
                [&](PathRef bra, double h) {
                    if (exclude_index.find(bra) >= 0) return;
                    const double term = h * scale_i;
                    if (std::abs(term) >= eps) part.add(bra, term);
                }
            );
        }
    }

    std::vector<u64> bra_words;
    for (const auto& part : local) {
        bra_words.insert(bra_words.end(), part.bras.words().begin(), part.bras.words().end());
    }
    sort_unique_paths(bra_words, sector_.nword);

    PathBatchView merged_bras{
        bra_words.data(),
        bra_words.size() / path_size(sector_.nword),
        sector_.nword
    };
    PathIndex bra_index(merged_bras);
    std::vector<double> hpsi(merged_bras.n_paths, 0.0);

    for (const auto& part : local) {
        for (std::size_t i = 0; i < part.hpsi.size(); ++i) {
            const i32 ibra = bra_index.find(part.bras.get(i));
            if (ibra >= 0) hpsi[static_cast<std::size_t>(ibra)] += part.hpsi[i];
        }
    }

    std::vector<u64> filtered_words;
    std::vector<double> filtered_hpsi;
    filtered_words.reserve(bra_words.size());
    filtered_hpsi.reserve(hpsi.size());
    for (std::size_t i = 0; i < hpsi.size(); ++i) {
        if (hpsi[i] == 0.0) continue;
        append_path(filtered_words, merged_bras[i]);
        filtered_hpsi.push_back(hpsi[i]);
    }

    Projection out;
    out.nword = sector_.nword;
    out.bra_words = std::move(filtered_words);
    out.hpsi = std::move(filtered_hpsi);
    const PathBatchView bras{out.bra_words.data(), out.hpsi.size(), sector_.nword};
    out.diags = diags(bras);
    return out;
}

inline ::libdet::Conns Hamiltonian::conn(
    PathBatchView kets,
    double eps,
    const PathBatchView* include
) const {
    check_paths(kets, "conn(kets)");
    check_eps(eps);

    if (include != nullptr) {
        check_paths(*include, "conn(include)");
        if (include->n_paths < kets.n_paths) {
            throw std::invalid_argument("conn: include must start with kets");
        }
        for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
            if (!path_equal((*include)[iket], kets[iket])) {
                throw std::invalid_argument("conn: include must start with kets");
            }
        }
    }

    const auto all = ket_conns(kets, eps);

    ::libdet::Conns out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_paths;
    out.diag.reserve(kets.n_paths);
    out.weight.reserve(kets.n_paths);
    out.ptr.assign(1, 0);

    PathPool pool(include == nullptr ? kets : *include);

    for (std::size_t iket = 0; iket < kets.n_paths; ++iket) {
        const Conns& ket_conn = *all[iket];
        double weight = 0.0;

        out.diag.push_back(ket_conn.diag);

        const ConnWindow win = ket_conn.window(AbsWindow{eps, std::numeric_limits<double>::infinity()});
        weight = win.weight;
        for (std::size_t k = win.begin; k < win.end; ++k) {
            const PathRef bra = ket_conn.bra(k, sector_.nword);
            out.bra.push_back(pool.find_or_add(bra));
            out.h.push_back(ket_conn.h[k]);
        }

        out.weight.push_back(weight);
        out.ptr.push_back(to_i32(out.bra.size()));
    }

    out.bra_words = std::move(pool.words());
    return out;
}

} // namespace libdet::guga
