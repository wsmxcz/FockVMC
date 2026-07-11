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

#include <omp.h>

namespace libdet::guga {

[[nodiscard]] inline bool keep_h(double h, double eps) noexcept {
    const double value = std::abs(h);
    return value > 0.0 && value >= eps;
}

namespace detail {

[[nodiscard]] inline bool apply_move(
    Occ ket,
    const OccMove& move,
    std::vector<unsigned char>& work,
    std::vector<unsigned char>& occ
) {
    if (move.degree == 1) {
        return move_one_occ(ket, move.remove[0], move.add[0], occ);
    }
    if (move.degree == 2) {
        return remove_pair_occ(ket, move.remove[0], move.remove[1], work)
            && add_pair_occ(work, move.add[0], move.add[1], occ);
    }
    return false;
}


} // namespace detail

template <class Visit>
inline void occ_paths_rec(
    std::span<const unsigned char> occ,
    std::span<const int> suffix,
    int p,
    int norb,
    int spin,
    int target_spin,
    std::vector<Step>& steps,
    Visit&& visit
) {
    const int remain = suffix[static_cast<std::size_t>(p)];
    const int need = target_spin - spin;
    if (std::abs(need) > remain || ((remain - std::abs(need)) & 1) != 0) return;

    if (p == norb) {
        visit(steps);
        return;
    }

    const unsigned char n = occ[static_cast<std::size_t>(p)];
    if (n == 0u || n == 2u) {
        steps[static_cast<std::size_t>(p)] = n == 0u ? Step::empty : Step::doubly;
        occ_paths_rec(
            occ, suffix, p + 1, norb, spin, target_spin, steps, std::forward<Visit>(visit)
        );
        return;
    }

    steps[static_cast<std::size_t>(p)] = Step::up;
    occ_paths_rec(
        occ, suffix, p + 1, norb, spin + 1, target_spin, steps, std::forward<Visit>(visit)
    );

    if (spin > 0) {
        steps[static_cast<std::size_t>(p)] = Step::down;
        occ_paths_rec(
            occ, suffix, p + 1, norb, spin - 1, target_spin, steps, std::forward<Visit>(visit)
        );
    }
}

template <class Visit>
inline void visit_occ_paths(
    std::span<const unsigned char> occ,
    Sector sector,
    std::vector<Step>& steps,
    std::vector<int>& suffix,
    Visit&& visit
) {
    steps.resize(static_cast<std::size_t>(sector.norb));
    suffix.assign(static_cast<std::size_t>(sector.norb + 1), 0);
    for (int p = sector.norb - 1; p >= 0; --p) {
        suffix[static_cast<std::size_t>(p)] = suffix[static_cast<std::size_t>(p + 1)]
            + (occ[static_cast<std::size_t>(p)] == 1u ? 1 : 0);
    }
    occ_paths_rec(
        occ, suffix, 0, sector.norb, 0, sector.spin_twice, steps, std::forward<Visit>(visit)
    );
}

template <class Emit>
inline void visit_external_state(
    KetScratch& scratch,
    const Integral& ints,
    const Seg2Table& seg2,
    Sector sector,
    const ScreenTable* table,
    PathRef ket_words,
    const PathState& ket,
    double eps,
    bool include_diag,
    Emit&& emit
) {
    scratch.elem.load_single(ints, ket);

    if (include_diag) {
        const double h = diag(scratch.elem, seg2, ints, ket);
        if (keep_h(h, eps)) emit(ket_words, h);
    }

    auto add_bra = [&](std::span<const Step> steps, const OccMove& move) {
        const PathRef bra_ref = scratch.encode(steps, sector.nword);
        if (path_equal(bra_ref, ket_words)) return;

        load_path(steps, sector, "visit_external(bra)", scratch.bra_work, scratch.bra);
        const PathDiff diff = path_diff(ket, scratch.bra, move);
        double h = 0.0;
        if (move.degree == 0) {
            h = same_ocfg(scratch.elem, seg2, ints, scratch.bra, ket, diff);
        } else if (move.degree == 1) {
            const int p = move.add[0];
            const int q = move.remove[0];
            h = single_move(scratch.elem, seg2, ints, scratch.bra, ket, diff, move, scratch.elem.single_coulomb(p, q));
        } else {
            h = double_move(scratch.elem, seg2, ints, scratch.bra, ket, diff, move);
        }
        if (keep_h(h, eps)) emit(bra_ref, h);
    };

    auto add_occ = [&](Occ occ, const OccMove& move) {
        visit_occ_paths(occ, sector, scratch.step, scratch.suffix, [&](const std::vector<Step>& steps) {
            add_bra(std::span<const Step>(steps.data(), steps.size()), move);
        });
    };

    if (table == nullptr || eps == 0.0 || table->same_bound(ket) >= eps) {
        add_occ(ket.occ, OccMove{});
    }

    auto add_move = [&](const OccMove& move) {
        if (!detail::apply_move(ket.occ, move, scratch.work, scratch.occ)) return;
        if (table != nullptr && eps > 0.0 && table->bound(ket, move) < eps) return;
        add_occ(scratch.occ, move);
    };

    if (table != nullptr) {
        for (const ScreenMove& cand : table->singles(eps)) add_move(cand.move);
        for (const ScreenMove& cand : table->doubles(eps)) add_move(cand.move);
        return;
    }

    for (int q = 0; q < sector.norb; ++q) {
        for (int p = 0; p < sector.norb; ++p) {
            if (p == q) continue;
            OccMove move;
            move.degree = 1;
            move.remove = {q, -1};
            move.add = {p, -1};
            add_move(move);
        }
    }

    for (int q = 0; q < sector.norb; ++q) {
        for (int s = q; s < sector.norb; ++s) {
            for (int p = 0; p < sector.norb; ++p) {
                for (int r = p; r < sector.norb; ++r) {
                    if (p == q || p == s || r == q || r == s) continue;
                    OccMove move;
                    move.degree = 2;
                    move.remove = {q, s};
                    move.add = {p, r};
                    add_move(move);
                }
            }
        }
    }
}

template <class Emit>
inline void visit_external(
    KetScratch& scratch,
    const Integral& ints,
    const Seg2Table& seg2,
    Sector sector,
    const ScreenTable* table,
    PathRef ket_words,
    double eps,
    bool include_diag,
    Emit&& emit
) {
    decode_path(ket_words, sector, "visit_external(ket)", scratch.ket_work, scratch.ket);
    visit_external_state(
        scratch,
        ints,
        seg2,
        sector,
        table,
        ket_words,
        scratch.ket,
        eps,
        include_diag,
        std::forward<Emit>(emit)
    );
}

} // namespace libdet::guga

#include <libdet/guga/sample.hpp>

namespace libdet::guga {

inline double Hamiltonian::hij(PathRef bra, PathRef ket) const {
    check_one(bra, "hij(bra)");
    check_one(ket, "hij(ket)");
    PathScratch bra_work;
    PathScratch ket_work;
    PathState bra_state;
    PathState ket_state;
    decode_path(bra, sector_, "hij(bra)", bra_work, bra_state);
    decode_path(ket, sector_, "hij(ket)", ket_work, ket_state);
    ElementScratch elem_scratch(sector_.norb);
    return guga::hij(elem_scratch, seg2_, ints_, bra_state, ket_state);
}

inline std::vector<double> Hamiltonian::diags(PathBatchView paths) const {
    check_paths(paths, "diags");

    std::vector<double> out(paths.n_paths, 0.0);
    PathScratch scratch;
    ElementScratch elem_scratch(sector_.norb);
    PathState item;
    for (std::size_t ipath = 0; ipath < paths.n_paths; ++ipath) {
        decode_path(paths[ipath], sector_, "diags", scratch, item);
        out[ipath] = guga::diag(elem_scratch, seg2_, ints_, item);
    }
    return out;
}

inline std::shared_ptr<const Conns> Hamiltonian::make_conns(
    PathRef ket,
    double eps,
    const ScreenTable* table,
    KetScratch& scratch
) const {
    auto out = std::make_shared<Conns>();
    out->cutoff = eps;

    decode_path(ket, sector_, "cached_conns(ket)", scratch.ket_work, scratch.ket);
    out->diag = guga::diag(scratch.elem, seg2_, ints_, scratch.ket);

    visit_external_state(
        scratch,
        ints_,
        seg2_,
        sector_,
        table,
        ket,
        scratch.ket,
        eps,
        false,
        [&](PathRef bra, double h) {
            out->add(bra, h);
        }
    );

    out->finish(sector_.nword);
    return out;
}

inline std::shared_ptr<const Conns> Hamiltonian::cached_conns(
    PathRef ket,
    double eps
) const {
    const auto screen_table_ptr = screen_table(eps);
    if (eps <= 0.0) {
        KetScratch scratch(sector_.norb);
        return make_conns(ket, eps, screen_table_ptr.get(), scratch);
    }

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        if (auto hit = conn_cache_.find(ket, eps)) return hit;
    }

    KetScratch scratch(sector_.norb);
    auto fresh = make_conns(ket, eps, screen_table_ptr.get(), scratch);

    std::lock_guard<std::mutex> lock(conn_cache_mutex_);
    if (auto hit = conn_cache_.find(ket, eps)) return hit;
    conn_cache_.insert(ket, fresh);
    return fresh;
}

inline std::vector<std::shared_ptr<const Conns>>
Hamiltonian::cached_conns(PathBatchView kets, double eps) const {
    check_paths(kets, "cached_conns(kets)");
    check_eps(eps);

    const auto screen_table_ptr = screen_table(eps);
    std::vector<std::shared_ptr<const Conns>> out(kets.n_paths);

    if (eps <= 0.0) {
#pragma omp parallel
        {
            KetScratch scratch(sector_.norb);
#pragma omp for schedule(guided)
            for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
                const std::size_t iket = static_cast<std::size_t>(ii);
                out[iket] = make_conns(kets[iket], eps, screen_table_ptr.get(), scratch);
            }
        }
        return out;
    }

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

#pragma omp parallel
    {
        KetScratch scratch(sector_.norb);
#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
            const std::size_t iket = misses[static_cast<std::size_t>(ii)];
            out[iket] = make_conns(kets[iket], eps, screen_table_ptr.get(), scratch);
        }
    }

    {
        std::lock_guard<std::mutex> lock(conn_cache_mutex_);
        for (std::size_t iket : misses) conn_cache_.insert(kets[iket], out[iket]);
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
    const auto screen_table_ptr = screen_table(screen_table_cutoff(eps, max_scale));

    const int nthread = std::max(1, omp_get_max_threads());

    std::vector<std::vector<u64>> local(static_cast<std::size_t>(nthread));

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& words = local[static_cast<std::size_t>(tid)];
        KetScratch scratch(sector_.norb);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double scale_i = scale.empty() ? 1.0 : std::abs(scale[iket]);
            if (scale_i == 0.0) continue;

            const double h_eps = eps == 0.0 ? 0.0 : eps / scale_i;
            visit_external(
                scratch,
                ints_,
                seg2_,
                sector_,
                screen_table_ptr.get(),
                kets[iket],
                h_eps,
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
    const double max_scale = max_abs(scale);
    const auto screen_table_ptr = screen_table(screen_table_cutoff(eps, max_scale));

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
            return words.size() / path_size(nword);
        }

        [[nodiscard]] i32 find_add(PathRef path) {
            const u64 fingerprint = path_fingerprint(path);
            auto& hits = map[fingerprint];
            for (i32 idx : hits) {
                if (path_equal(path_at(words, nword, static_cast<std::size_t>(idx)), path)) {
                    return idx;
                }
            }

            const i32 idx = to_i32(size());
            append_path(words, path);
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
        KetScratch scratch(sector_.norb);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
            const double scale_i = scale[iket];
            const double abs_scale = std::abs(scale_i);
            if (abs_scale <= 0.0) continue;

            const double h_eps = eps <= 0.0 ? 0.0 : eps / abs_scale;
            visit_external(
                scratch,
                ints_,
                seg2_,
                sector_,
                screen_table_ptr.get(),
                kets[iket],
                h_eps,
                false,
                [&](PathRef bra, double h) {
                    if (exclude_index.find(bra) >= 0) return;
                    Bin& bin = part.bin[path_fingerprint(bra) & mask];
                    append_path(bin.words, bra);
                    bin.value.push_back(h * scale_i);
                }
            );
        }
    }

    std::vector<Shard> shard;
    shard.reserve(n_shard);
    for (std::size_t s = 0; s < n_shard; ++s) shard.emplace_back(sector_.nword);

#pragma omp parallel for schedule(static)
    for (i64 ss = 0; ss < static_cast<i64>(n_shard); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        Shard& acc = shard[s];
        std::size_t n_route = 0;
        for (const Part& part : parts) n_route += part.bin[s].value.size();
        acc.map.reserve(n_route);
        acc.words.reserve(n_route * path_size(sector_.nword));

        for (const Part& part : parts) {
            const Bin& bin = part.bin[s];
            for (std::size_t k = 0; k < bin.value.size(); ++k) {
                const i32 ibra = acc.find_add(path_at(bin.words, sector_.nword, k));
                acc.hpsi[static_cast<std::size_t>(ibra)] += bin.value[k];
            }
        }
    }

    std::vector<std::size_t> start(n_shard + 1u, 0u);
    for (std::size_t s = 0; s < n_shard; ++s) start[s + 1u] = start[s] + shard[s].size();

    Projection out;
    out.nword = sector_.nword;
    out.bra.resize(start.back() * path_size(sector_.nword));
    out.hpsi.assign(start.back(), 0.0);

#pragma omp parallel for schedule(static)
    for (i64 ss = 0; ss < static_cast<i64>(n_shard); ++ss) {
        const std::size_t s = static_cast<std::size_t>(ss);
        const std::size_t stride = path_size(sector_.nword);
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
        const PathBatchView bras{out.bra.data(), start.back(), sector_.nword};
        out.diag = diags(bras);
    }
    return out;
}

inline ::libdet::Conns Hamiltonian::conn(
    PathBatchView kets,
    double eps
) const {
    check_paths(kets, "conn(kets)");
    check_eps(eps);

    const auto all = cached_conns(kets, eps);
    std::vector<detail::Item> items(kets.n_paths);
    std::vector<double> diag(kets.n_paths, 0.0);
    std::vector<double> degree(kets.n_paths, 0.0);

#pragma omp parallel for schedule(guided)
    for (i64 ii = 0; ii < static_cast<i64>(kets.n_paths); ++ii) {
        const std::size_t iket = static_cast<std::size_t>(ii);
        const Conns& ket_conn = *all[iket];
        const ConnSpan win = ket_conn.span(std::numeric_limits<double>::infinity(), eps);
        detail::Item& item = items[iket];
        item.words.reserve((win.end - win.begin) * path_size(sector_.nword));
        item.h.reserve(win.end - win.begin);
        diag[iket] = ket_conn.diag;
        degree[iket] = win.degree;

        for (std::size_t k = win.begin; k < win.end; ++k) {
            append_path(item.words, ket_conn.bra(k, sector_.nword));
            item.h.push_back(ket_conn.h[k]);
        }
    }

    return detail::assemble_conn(kets, items, diag, degree, 1u);
}

} // namespace libdet::guga
