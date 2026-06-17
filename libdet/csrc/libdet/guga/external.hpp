#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet::guga {

namespace detail {

struct ProjectPart {
    explicit ProjectPart(u32 nword) : bras(nword) {}

    DetPool bras;
    std::vector<double> hpsi;

    void add(DetRef bra, double value) {
        const i32 idx = bras.find_or_add(bra);
        const std::size_t pos = static_cast<std::size_t>(idx);
        if (pos == hpsi.size()) hpsi.push_back(value);
        else hpsi[pos] += value;
    }
};

} // namespace detail

inline double Hamiltonian::hij(DetRef bra, DetRef ket) const {
    check_one(bra, "hij(bra)");
    check_one(ket, "hij(ket)");
    return element(bra, ket);
}

inline std::vector<double> Hamiltonian::diags(DetBatchView dets) const {
    check_dets(dets, "diags");

    const auto space = cached_csf_space(dets);
    std::vector<double> out(dets.n_dets, 0.0);
    for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
        const Csf& item = space->csf(idet);
        out[idet] = element(item, item);
    }
    return out;
}

inline std::shared_ptr<const KetConns> Hamiltonian::build_ket_conns(
    DetRef ket,
    double eps
) const {
    auto out = std::make_shared<KetConns>();
    out->cutoff = eps;

    const Csf ket_csf = csf(ket, "ket_conns(ket)");
    out->diag = element(ket_csf, ket_csf);

    DetPool pool(sector_.nword);
    std::vector<double> hpool;
    std::vector<u64> bra_words;

    screen_.visit_bra_cfgs(ket_csf, eps, [&](const BraCfg& bra_cfg) {
        visit_csfs(bra_cfg.cfg, sector_, [&](const std::vector<Step>& steps) {
            const std::span<const Step> step_view(steps.data(), steps.size());
            encode_csf(step_view, sector_.nword, bra_words);

            const DetRef bra = packed_det(bra_words.data(), sector_.nword);
            if (det_equal(bra, ket)) return;

            const Csf bra_csf = make_csf(step_view, sector_, "ket_conns(bra)");
            const double h = element(bra_csf, ket_csf);
            if (h == 0.0 || std::abs(h) < eps) return;

            const i32 idx = pool.find_or_add(bra);
            const std::size_t pos = static_cast<std::size_t>(idx);
            if (pos == hpool.size()) hpool.push_back(h);
            else hpool[pos] += h;
        });
    });

    out->bra_words.reserve(hpool.size() * det_size(sector_.nword));
    out->h.reserve(hpool.size());
    for (std::size_t i = 0; i < hpool.size(); ++i) {
        const double h = hpool[i];
        if (h == 0.0 || std::abs(h) < eps) continue;

        append_det(out->bra_words, pool.get(i));
        out->h.push_back(h);
    }

    out->finish(sector_.nword);
    return out;
}

inline std::shared_ptr<const KetConns> Hamiltonian::ket_conns(
    DetRef ket,
    double eps
) const {
    {
        std::lock_guard<std::mutex> lock(ket_cache_mutex_);
        if (auto hit = ket_cache_.find(ket, eps)) return hit;
    }

    auto fresh = build_ket_conns(ket, eps);

    std::lock_guard<std::mutex> lock(ket_cache_mutex_);
    if (auto hit = ket_cache_.find(ket, eps)) return hit;
    ket_cache_.insert(ket, fresh);
    return fresh;
}

inline std::vector<std::shared_ptr<const KetConns>>
Hamiltonian::ket_conns(DetBatchView kets, double eps) const {
    check_dets(kets, "ket_conns(kets)");
    check_eps(eps);

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
#pragma omp parallel for schedule(guided)
    for (i64 ii = 0; ii < static_cast<i64>(misses.size()); ++ii) {
        const std::size_t iket = misses[static_cast<std::size_t>(ii)];
        out[iket] = build_ket_conns(kets[iket], eps);
    }
#else
    for (std::size_t iket : misses) {
        out[iket] = build_ket_conns(kets[iket], eps);
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
        auto& words = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& words = local[0];
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const double scale = coeffs.empty() ? 1.0 : std::abs(coeffs[iket]);
        const double cutoff = scaled_eps(eps, scale);
        if (!std::isfinite(cutoff)) continue;

        const auto ket_conn = ket_conns(kets[iket], cutoff);
        for (std::size_t k = 0; k < ket_conn->size(); ++k) {
            const DetRef bra = ket_conn->bra(k, sector_.nword);
            if (exclude_index.find(bra) < 0) append_det(words, bra);
        }
    }
    }

    std::vector<u64> words;
    for (auto& part : local) {
        words.insert(words.end(), part.begin(), part.end());
    }
    sort_unique_dets(words, sector_.nword);
    return words;
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
    const DetIndex exclude_index(base);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<detail::ProjectPart> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(sector_.nword);

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& part = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            const std::size_t iket = static_cast<std::size_t>(ii);
#else
    {
        auto& part = local[0];
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
#endif
        const double coeff = coeffs[iket];
        const double cutoff = scaled_eps(eps, std::abs(coeff));
        if (!std::isfinite(cutoff)) continue;

        const auto ket_conn = ket_conns(kets[iket], cutoff);
        for (std::size_t k = 0; k < ket_conn->size(); ++k) {
            const DetRef bra = ket_conn->bra(k, sector_.nword);
            if (exclude_index.find(bra) >= 0) continue;

            const double term = ket_conn->h[k] * coeff;
            if (std::abs(term) < eps) continue;

            part.add(bra, term);
        }
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
    sort_unique_dets(bra_words, sector_.nword);

    const DetBatchView merged_bras{
        bra_words.data(),
        bra_words.size() / det_size(sector_.nword),
        sector_.nword
    };
    const DetIndex bra_index(merged_bras);
    std::vector<double> hpsi(merged_bras.n_dets, 0.0);

    for (const auto& part : local) {
        for (std::size_t i = 0; i < part.hpsi.size(); ++i) {
            const i32 ibra = bra_index.find(part.bras.get(i));
            if (ibra >= 0) hpsi[static_cast<std::size_t>(ibra)] += part.hpsi[i];
        }
    }

    Projection out;
    out.nword = sector_.nword;
    out.bra_words = std::move(bra_words);
    out.hpsi = std::move(hpsi);
    const DetBatchView bras{
        out.bra_words.data(),
        out.hpsi.size(),
        sector_.nword
    };
    out.diags = diags(bras);
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
    if (n_sample < 0) throw std::invalid_argument("conns: sample must be nonnegative");
    if (n_sample > 0 && sample_eps > eps) {
        throw std::invalid_argument("conns: sample_eps must be <= eps");
    }

    const bool sample_weak = n_sample > 0 && sample_eps < eps;
    const auto all = ket_conns(kets, sample_weak ? sample_eps : eps);

    Conns out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_dets;
    out.ket_ptr.assign(1, 0);
    out.sample_ket_ptr.assign(1, 0);
    DetPool pool(kets);

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const KetConns& ket_conn = *all[iket];
        const std::size_t nstrong = ket_conn.count(eps);
        out.diag.push_back(ket_conn.diag);
        out.weight.push_back(ket_conn.prefix_abs[nstrong]);

        for (std::size_t k = 0; k < nstrong; ++k) {
            const DetRef bra = ket_conn.bra(k, sector_.nword);
            out.bra_idx.push_back(pool.find_or_add(bra));
            out.h.push_back(ket_conn.h[k]);
        }
        out.ket_ptr.push_back(to_i32(out.bra_idx.size()));

        const std::size_t begin = ket_conn.count(eps);
        const std::size_t end = ket_conn.count(sample_eps);
        const double weak_weight =
            ket_conn.prefix_abs[end] - ket_conn.prefix_abs[begin];
        out.sample_weight.push_back(sample_weak ? weak_weight : 0.0);

        if (sample_weak && weak_weight > 0.0) {
            SmallRng rng(sample_seed(seed, kets[iket]));
            std::vector<double> targets;
            make_targets(rng, n_sample, weak_weight, targets);

            std::size_t pos = 0;
            double cdf = 0.0;
            for (std::size_t k = begin; k < end; ++k) {
                cdf += std::abs(ket_conn.h[k]);
                i64 count = 0;
                while (pos < targets.size() && targets[pos] <= cdf) {
                    ++count;
                    ++pos;
                }
                if (count <= 0) continue;

                const DetRef bra = ket_conn.bra(k, sector_.nword);
                out.sample_bra_idx.push_back(pool.find_or_add(bra));
                out.sample_h.push_back(ket_conn.h[k]);
                out.sample_count.push_back(count);
            }
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

} // namespace libdet::guga
