#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <libdet/hamiltonian.hpp>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace libdet {

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
    u64 value = splitmix64(seed ^ 0x243f6a8885a308d3ULL);
    value = splitmix64(value ^ det_fingerprint(ket));
    value = splitmix64(value ^ static_cast<u64>(rep + 1));
    value = splitmix64(
        value ^ (
            stream == 0
                ? 0x13198a2e03707344ULL
                : 0xa4093822299f31d0ULL
        )
    );
    return value;
}

inline void make_targets(
    SmallRng& rng,
    i64 n_draw,
    double norm,
    std::vector<double>& out
) {
    out.clear();
    if (n_draw <= 0 || !(norm > 0.0) || !std::isfinite(norm)) return;

    out.reserve(static_cast<std::size_t>(n_draw));

    for (i64 k = 0; k < n_draw; ++k) {
        out.push_back(rng.uniform01() * norm);
    }

    std::sort(out.begin(), out.end());
}

namespace {

struct SampleHit {
    Coupling coupling;
    i64 count = 0;
};

struct ProjectSample {
    i32 rep = 0;
    i32 bra = 0;
    i32 ket = 0;
    unsigned char stream = 0;
    double hpsi_a = 0.0;
    double hpsi_b = 0.0;
};

struct SamplePart {
    explicit SamplePart(u32 nword)
        : bras(nword) {}

    DetPool bras;
    std::vector<ProjectSample> samples;
};

} // namespace

Conns Hamiltonian::conns(
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
    std::vector<std::vector<SampleHit>> weak_hits(kets.n_dets);

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
                const KetConns& src = *all[iket];
                const std::size_t begin = src.count(eps);
                const std::size_t end = src.count(sample_eps);
                const double weight =
                    src.prefix_abs[end] - src.prefix_abs[begin];
                weak_weight[iket] = weight;

                if (!(weight > 0.0)) continue;

                SmallRng rng(sample_seed(seed, ket, 0));
                make_targets(rng, n_sample, weight, targets);

                std::size_t target_pos = 0;
                double cdf = 0.0;

                for (std::size_t k = begin; k < end; ++k) {
                    const Coupling& coupling = src.couplings[k];
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
        const KetConns& src = *all[iket];
        const std::size_t nconn = src.count(eps);

        out.diag.push_back(src.diag);
        out.weight.push_back(src.prefix_abs[nconn]);

        for (std::size_t k = 0; k < nconn; ++k) {
            const Coupling& coupling = src.couplings[k];
            const DetRef bra = apply(ket, coupling.excitation, bra_scratch);
            out.bra_idx.push_back(pool.find_or_add(bra));
            out.h.push_back(coupling.h);
        }

        out.ket_ptr.push_back(to_i32(out.bra_idx.size()));
        out.sample_weight.push_back(
            sample_weak ? weak_weight[iket] : 0.0
        );

        for (const SampleHit& hit : weak_hits[iket]) {
            const DetRef bra = apply(
                ket,
                hit.coupling.excitation,
                bra_scratch
            );
            out.sample_bra_idx.push_back(pool.find_or_add(bra));
            out.sample_h.push_back(hit.coupling.h);
            out.sample_count.push_back(hit.count);
        }

        out.sample_ket_ptr.push_back(
            to_i32(out.sample_bra_idx.size())
        );
    }

    out.det_words = std::move(pool.words());
    return out;
}

std::pair<std::vector<double>, std::vector<i64>> Hamiltonian::degrees(
    DetBatchView kets,
    double eps
) const {
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

ConnSamples Hamiltonian::sample_conns(
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
        throw std::invalid_argument(
            "sample_conns: n_streams must be positive"
        );
    }
    if (counts.size() != n_streams * kets.n_dets) {
        throw std::invalid_argument(
            "sample_conns: counts shape must be (n_streams, n_kets)"
        );
    }
    if (std::any_of(counts.begin(), counts.end(), [](i64 n) {
        return n < 0;
    })) {
        throw std::invalid_argument(
            "sample_conns: counts must be nonnegative"
        );
    }

    const auto all = ket_conns(kets, eps2);
    std::vector<double> weight(kets.n_dets, 0.0);
    std::vector<std::vector<SampleHit>> hits(n_streams * kets.n_dets);

#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> pos(n_streams);

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#else
    {
        std::vector<std::vector<double>> targets(n_streams);
        std::vector<std::size_t> pos(n_streams);

        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
#endif
            const std::size_t iket = static_cast<std::size_t>(ii);
            const DetRef ket = kets[iket];
            const KetConns& src = *all[iket];
            const std::size_t begin = std::isfinite(eps1)
                ? src.count(eps1)
                : 0u;
            const std::size_t end = src.count(eps2);

            weight[iket] = src.prefix_abs[end] - src.prefix_abs[begin];
            if (!(weight[iket] > 0.0)) continue;

            std::fill(pos.begin(), pos.end(), 0u);
            bool any = false;

            for (std::size_t stream = 0; stream < n_streams; ++stream) {
                SmallRng rng(
                    sample_seed(seed, ket, static_cast<i64>(stream))
                );
                make_targets(
                    rng,
                    counts[stream * kets.n_dets + iket],
                    weight[iket],
                    targets[stream]
                );
                any = any || !targets[stream].empty();
            }
            if (!any) continue;

            double cdf = 0.0;

            for (std::size_t k = begin; k < end; ++k) {
                const Coupling& coupling = src.couplings[k];
                cdf += std::abs(coupling.h);

                for (std::size_t stream = 0; stream < n_streams; ++stream) {
                    i64 count = 0;

                    while (
                        pos[stream] < targets[stream].size()
                        && targets[stream][pos[stream]] <= cdf
                    ) {
                        ++count;
                        ++pos[stream];
                    }

                    if (count > 0) {
                        hits[stream * kets.n_dets + iket].push_back(
                            {coupling, count}
                        );
                    }
                }
            }
        }
    }

    ConnSamples out;
    out.nword = nword_;
    out.n_kets = kets.n_dets;
    out.n_streams = n_streams;
    out.weight = std::move(weight);
    out.ket_ptr.assign(1, 0);

    DetPool pool(kets);
    DetScratch bra_scratch(nword_);

    for (std::size_t stream = 0; stream < n_streams; ++stream) {
        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            for (const SampleHit& hit : hits[stream * kets.n_dets + iket]) {
                const DetRef bra = apply(
                    kets[iket],
                    hit.coupling.excitation,
                    bra_scratch
                );
                out.bra_idx.push_back(pool.find_or_add(bra));
                out.h.push_back(hit.coupling.h);
                out.count.push_back(hit.count);
            }
            out.ket_ptr.push_back(to_i32(out.bra_idx.size()));
        }
    }

    out.det_words = std::move(pool.words());
    return out;
}

std::shared_ptr<const KetConns> Hamiltonian::build_ket_conns(
    DetRef ket,
    double eps,
    KetScratch& scratch,
    const Screen* screen_ptr
) const {
    auto conns = std::make_shared<KetConns>();
    conns->cutoff = eps;

    fill_occ(ket, ints_.norb(), scratch.occ);
    conns->diag = Slater::diag(ints_, scratch.occ);

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

std::vector<std::shared_ptr<const KetConns>> Hamiltonian::ket_conns(
    DetBatchView kets,
    double eps
) const {
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
        out[iket] = build_ket_conns(
            kets[iket],
            eps,
            scratch,
            screen_ptr.get()
        );
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

ProjectSamples Hamiltonian::sample_project(
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
        throw std::invalid_argument(
            "sample_project: coeffs size must match kets"
        );
    }
    if (counts.size() != kets.n_dets) {
        throw std::invalid_argument(
            "sample_project: counts size must match kets"
        );
    }
    if (n_rep <= 0) {
        throw std::invalid_argument(
            "sample_project: n_rep must be positive"
        );
    }

    check_window_eps(eps1, eps2);
    const DetBatchView base = exclude == nullptr ? kets : *exclude;
    check_dets(base, "sample_project(exclude)");

    const double scale_max = max_abs(coeffs);
    auto screen_ptr = screen(screen_cutoff(eps2, scale_max));
    const DetIndex exclude_index(base);

#if defined(_OPENMP)
    const int nthread = std::max(1, omp_get_max_threads());
#else
    const int nthread = 1;
#endif

    std::vector<SamplePart> local;
    local.reserve(static_cast<std::size_t>(nthread));
    for (int t = 0; t < nthread; ++t) local.emplace_back(nword_);

    auto sample_ket = [&, screen_ptr](
        std::size_t iket,
        KetScratch& scratch,
        std::vector<double>& targets,
        std::vector<Coupling>& candidates,
        SamplePart& part
    ) {
        const i64 n_draw = counts[iket];
        const double coeff = coeffs[iket];
        const double scale = std::abs(coeff);

        if (n_draw <= 0 || scale <= 0.0) return;

        const DetRef ket = kets[iket];
        const AbsWindow window = abs_window(eps2, eps1, scale);
        DetScratch bra_scratch(nword_);
        double weight = 0.0;
        candidates.clear();

        visit_bras(
            ints_,
            screen_ptr.get(),
            ket,
            scratch,
            window,
            [&](Excitation excitation, double h) {
                const DetRef bra = apply(ket, excitation, bra_scratch);
                if (exclude_index.find(bra) >= 0) return;

                candidates.push_back({excitation, h});
                weight += std::abs(h);
            }
        );

        if (!(weight > 0.0)) return;

        for (i64 rep = 0; rep < n_rep; ++rep) {
            for (int stream = 0; stream < 2; ++stream) {
                SmallRng rng(sample_seed(seed, ket, rep, stream));
                make_targets(rng, n_draw, weight, targets);
                if (targets.empty()) continue;

                std::size_t target_pos = 0;
                double cdf = 0.0;

                for (const Coupling& coupling : candidates) {
                    const double abs_h = std::abs(coupling.h);
                    cdf += abs_h;
                    i64 hit = 0;

                    while (
                        target_pos < targets.size()
                        && targets[target_pos] <= cdf
                    ) {
                        ++hit;
                        ++target_pos;
                    }
                    if (hit <= 0) continue;

                    const DetRef bra = apply(
                        ket,
                        coupling.excitation,
                        bra_scratch
                    );
                    const double delta =
                        static_cast<double>(hit)
                        * coeff * coupling.h * weight
                        / (static_cast<double>(n_draw) * abs_h);

                    ProjectSample sample;
                    sample.rep = static_cast<i32>(rep);
                    sample.bra = part.bras.find_or_add(bra);
                    sample.ket = static_cast<i32>(iket);
                    sample.stream = static_cast<unsigned char>(stream);
                    if (stream == 0) sample.hpsi_a = delta;
                    else sample.hpsi_b = delta;
                    part.samples.push_back(sample);
                }
            }
        }
    };

#if defined(_OPENMP)
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        KetScratch scratch(ints_.norb());
        std::vector<double> targets;
        std::vector<Coupling> candidates;

#pragma omp for schedule(guided)
        for (i64 ii = 0; ii < static_cast<i64>(kets.n_dets); ++ii) {
            sample_ket(
                static_cast<std::size_t>(ii),
                scratch,
                targets,
                candidates,
                local[static_cast<std::size_t>(tid)]
            );
        }
    }
#else
    {
        KetScratch scratch(ints_.norb());
        std::vector<double> targets;
        std::vector<Coupling> candidates;

        for (std::size_t iket = 0; iket < kets.n_dets; ++iket) {
            sample_ket(iket, scratch, targets, candidates, local[0]);
        }
    }
#endif

    std::vector<ProjectSample> samples;
    std::vector<u64> sample_words;
    std::size_t n_samples = 0;
    std::size_t n_words = 0;

    for (const SamplePart& part : local) {
        n_samples += part.samples.size();
        n_words += part.bras.words().size();
    }

    samples.reserve(n_samples);
    sample_words.reserve(n_words);

    for (SamplePart& part : local) {
        const std::size_t offset = sample_words.size() / det_size(nword_);
        sample_words.insert(
            sample_words.end(),
            part.bras.words().begin(),
            part.bras.words().end()
        );

        for (ProjectSample sample : part.samples) {
            sample.bra += to_i32(offset);
            samples.push_back(sample);
        }
    }

    std::sort(
        samples.begin(),
        samples.end(),
        [&](const ProjectSample& lhs, const ProjectSample& rhs) {
            if (lhs.rep != rhs.rep) return lhs.rep < rhs.rep;
            const DetRef lhs_bra = det_at(sample_words, nword_, lhs.bra);
            const DetRef rhs_bra = det_at(sample_words, nword_, rhs.bra);

            if (DetLess{}(lhs_bra, rhs_bra)) return true;
            if (DetLess{}(rhs_bra, lhs_bra)) return false;
            if (lhs.ket != rhs.ket) return lhs.ket < rhs.ket;
            return lhs.stream < rhs.stream;
        }
    );

    ProjectSamples out;
    out.nword = nword_;
    out.rep_ptr.assign(static_cast<std::size_t>(n_rep + 1), 0);

    std::vector<u64> unique_bras;
    std::size_t i = 0;
    std::size_t pos = 0;

    for (i64 rep = 0; rep < n_rep; ++rep) {
        out.rep_ptr[static_cast<std::size_t>(rep)] = to_i32(pos);
        while (i < samples.size() && samples[i].rep < rep) ++i;

        while (i < samples.size() && samples[i].rep == rep) {
            const DetRef bra = det_at(sample_words, nword_, samples[i].bra);
            double hpsi_a = 0.0;
            double hpsi_b = 0.0;
            std::size_t j = i;

            while (
                j < samples.size()
                && samples[j].rep == rep
                && det_equal(
                    det_at(sample_words, nword_, samples[j].bra),
                    bra
                )
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

    sort_unique_dets(unique_bras, nword_);
    const DetBatchView bras{
        unique_bras.data(),
        unique_bras.size() / det_size(nword_),
        nword_
    };
    const Projection strong = project_impl(bras, kets, coeffs, eps1);
    const DetBatchView strong_bras{
        strong.bra_words.data(),
        strong.hpsi.size(),
        nword_
    };
    const DetIndex strong_index(strong_bras);
    const DetBatchView sampled_bras{
        out.bra_words.data(),
        out.hpsi_a.size(),
        nword_
    };

    for (std::size_t k = 0; k < out.hpsi_a.size(); ++k) {
        const i32 idx = strong_index.find(sampled_bras[k]);
        if (idx < 0) continue;

        out.diags[k] = strong.diags[static_cast<std::size_t>(idx)];
        out.hpsi_strong[k] = strong.hpsi[static_cast<std::size_t>(idx)];
    }

    return out;
}

} // namespace libdet
