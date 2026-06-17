#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/guga/hamiltonian.hpp>

namespace libdet::guga {

class SmallRng {
public:
    explicit SmallRng(u64 seed) : state_(seed) {}

    [[nodiscard]] double uniform01() noexcept {
        state_ = splitmix64(state_);
        return static_cast<double>((state_ >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

[[nodiscard]] inline u64 sample_seed(
    u64 seed,
    DetRef ket,
    i64 rep = 0,
    int stream = 0
) noexcept {
    u64 value = splitmix64(seed ^ 0x9e3779b97f4a7c15ULL);
    value = splitmix64(value ^ det_fingerprint(ket));
    value = splitmix64(value ^ static_cast<u64>(rep + 1));
    value = splitmix64(value ^ static_cast<u64>(stream + 17));
    return value;
}

inline void make_targets(
    SmallRng& rng,
    i64 n_draw,
    double norm,
    std::vector<double>& targets
) {
    targets.clear();
    if (n_draw <= 0 || !(norm > 0.0) || !std::isfinite(norm)) return;

    targets.reserve(static_cast<std::size_t>(n_draw));
    for (i64 k = 0; k < n_draw; ++k) {
        targets.push_back(rng.uniform01() * norm);
    }
    std::sort(targets.begin(), targets.end());
}

inline ConnSamples Hamiltonian::sample_conns(
    DetBatchView kets,
    std::span<const i64> counts,
    std::size_t n_streams,
    double eps1,
    double eps2,
    u64 seed
) const {
    check_dets(kets, "sample_conns(kets)");
    check_window_eps(eps1, eps2);
    if (n_streams == 0) {
        throw std::invalid_argument("sample_conns: n_streams must be positive");
    }
    if (counts.size() != n_streams * kets.n_dets) {
        throw std::invalid_argument("sample_conns: counts shape mismatch");
    }

    const auto all = ket_conns(kets, eps2);
    ConnSamples out;
    out.nword = sector_.nword;
    out.n_kets = kets.n_dets;
    out.n_streams = n_streams;
    out.weight.assign(kets.n_dets, 0.0);
    out.ket_ptr.assign(1, 0);
    DetPool pool(kets);

    for (std::size_t stream = 0; stream < n_streams; ++stream) {
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            const KetConns& ket_conn = *all[iket];
            const std::size_t begin =
                std::isfinite(eps1) ? ket_conn.count(eps1) : 0u;
            const std::size_t end = ket_conn.count(eps2);
            const double weight =
                ket_conn.prefix_abs[end] - ket_conn.prefix_abs[begin];
            out.weight[iket] = weight;

            if (weight > 0.0) {
                SmallRng rng(sample_seed(seed, kets[iket], 0, static_cast<int>(stream)));
                std::vector<double> targets;
                make_targets(rng, counts[stream * kets.n_dets + iket], weight, targets);

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
                    out.bra_idx.push_back(pool.find_or_add(bra));
                    out.h.push_back(ket_conn.h[k]);
                    out.count.push_back(count);
                }
            }
            out.ket_ptr.push_back(to_i32(out.bra_idx.size()));
        }
    }

    out.det_words = std::move(pool.words());
    return out;
}

inline ProjectSamples Hamiltonian::sample_project(
    DetBatchView kets,
    std::span<const double> coeffs,
    double eps1,
    double eps2,
    std::span<const i64> counts,
    const DetBatchView* exclude,
    i64 n_rep,
    u64 seed
) const {
    check_dets(kets, "sample_project(kets)");
    if (coeffs.size() != kets.n_dets) {
        throw std::invalid_argument("sample_project: coeffs size must match kets");
    }
    if (counts.size() != kets.n_dets) {
        throw std::invalid_argument("sample_project: counts size must match kets");
    }
    if (n_rep <= 0) throw std::invalid_argument("sample_project: n_rep must be positive");
    check_window_eps(eps1, eps2);

    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "sample_project(exclude)");
    const DetIndex exclude_index(base);

    struct Sample {
        i32 rep = 0;
        i32 bra = 0;
        double hpsi_a = 0.0;
        double hpsi_b = 0.0;
    };

    DetPool pool(sector_.nword);
    std::vector<Sample> samples;

    for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
        const double coeff = coeffs[iket];
        const double scale = std::abs(coeff);
        const i64 n_draw = counts[iket];
        if (n_draw <= 0 || scale <= 0.0) continue;

        const double cutoff = scaled_eps(eps2, scale);
        if (!std::isfinite(cutoff)) continue;
        const auto ket_conn = ket_conns(kets[iket], cutoff);

        std::vector<std::size_t> weak;
        double weight = 0.0;
        for (std::size_t k = 0; k < ket_conn->size(); ++k) {
            const DetRef bra = ket_conn->bra(k, sector_.nword);
            if (exclude_index.find(bra) >= 0) continue;

            const double term_abs = std::abs(ket_conn->h[k] * coeff);
            if (term_abs >= eps2 && term_abs < eps1) {
                weak.push_back(k);
                weight += std::abs(ket_conn->h[k]);
            }
        }
        if (!(weight > 0.0)) continue;

        for (i64 rep = 0; rep < n_rep; ++rep) {
            for (int stream = 0; stream < 2; ++stream) {
                SmallRng rng(sample_seed(seed, kets[iket], rep, stream));
                std::vector<double> targets;
                make_targets(rng, n_draw, weight, targets);
                if (targets.empty()) continue;

                std::size_t pos = 0;
                double cdf = 0.0;
                for (std::size_t k : weak) {
                    const double h = ket_conn->h[k];
                    const double abs_h = std::abs(h);
                    cdf += abs_h;

                    i64 hit = 0;
                    while (pos < targets.size() && targets[pos] <= cdf) {
                        ++hit;
                        ++pos;
                    }
                    if (hit <= 0) continue;

                    const DetRef bra = ket_conn->bra(k, sector_.nword);
                    const double delta =
                        static_cast<double>(hit)
                        * coeff * h * weight
                        / (static_cast<double>(n_draw) * abs_h);

                    Sample sample;
                    sample.rep = static_cast<i32>(rep);
                    sample.bra = pool.find_or_add(bra);
                    if (stream == 0) sample.hpsi_a = delta;
                    else sample.hpsi_b = delta;
                    samples.push_back(sample);
                }
            }
        }
    }

    std::sort(samples.begin(), samples.end(), [&](const Sample& lhs, const Sample& rhs) {
        if (lhs.rep != rhs.rep) return lhs.rep < rhs.rep;
        const DetRef l = pool.get(static_cast<std::size_t>(lhs.bra));
        const DetRef r = pool.get(static_cast<std::size_t>(rhs.bra));
        return DetLess{}(l, r);
    });

    ProjectSamples out;
    out.nword = sector_.nword;
    out.rep_ptr.assign(static_cast<std::size_t>(n_rep + 1), 0);

    std::vector<u64> unique_bras;
    std::size_t i = 0;
    std::size_t pos = 0;
    for (i64 rep = 0; rep < n_rep; ++rep) {
        out.rep_ptr[static_cast<std::size_t>(rep)] = to_i32(pos);
        while (i < samples.size() && samples[i].rep < rep) ++i;
        while (i < samples.size() && samples[i].rep == rep) {
            const DetRef bra = pool.get(static_cast<std::size_t>(samples[i].bra));
            double hpsi_a = 0.0;
            double hpsi_b = 0.0;

            std::size_t j = i;
            while (
                j < samples.size()
                && samples[j].rep == rep
                && det_equal(pool.get(static_cast<std::size_t>(samples[j].bra)), bra)
            ) {
                hpsi_a += samples[j].hpsi_a;
                hpsi_b += samples[j].hpsi_b;
                ++j;
            }

            append_det(out.bra_words, bra);
            append_det(unique_bras, bra);
            out.hpsi_a.push_back(hpsi_a);
            out.hpsi_b.push_back(hpsi_b);
            ++pos;
            i = j;
        }
    }
    out.rep_ptr[static_cast<std::size_t>(n_rep)] = to_i32(pos);
    out.diags.assign(pos, 0.0);
    out.hpsi_strong.assign(pos, 0.0);

    if (pos == 0) return out;

    sort_unique_dets(unique_bras, sector_.nword);
    const DetBatchView bras{
        unique_bras.data(),
        unique_bras.size() / det_size(sector_.nword),
        sector_.nword
    };
    const Projection strong = project_impl(bras, kets, coeffs, eps1);
    const DetBatchView strong_bras{
        strong.bra_words.data(),
        strong.hpsi.size(),
        sector_.nword
    };
    const DetIndex strong_index(strong_bras);
    const DetBatchView sampled_bras{
        out.bra_words.data(),
        out.hpsi_a.size(),
        sector_.nword
    };

    for (std::size_t k = 0; k < out.hpsi_a.size(); ++k) {
        const i32 idx = strong_index.find(sampled_bras[k]);
        if (idx < 0) continue;
        out.diags[k] = strong.diags[static_cast<std::size_t>(idx)];
        out.hpsi_strong[k] = strong.hpsi[static_cast<std::size_t>(idx)];
    }
    return out;
}

} // namespace libdet::guga
