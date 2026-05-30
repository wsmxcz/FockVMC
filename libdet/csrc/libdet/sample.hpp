#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <libdet/det.hpp>

namespace libdet {

/*
 * Small deterministic RNG for ket-local stochastic estimators.
 *
 * The seed is mixed with a ket fingerprint so sampling is reproducible and
 * independent of thread scheduling.
 */
class SmallRng {
public:
    explicit SmallRng(u64 seed) : state_(seed) {}

    [[nodiscard]] u64 next_u64() noexcept {
        state_ = splitmix64(state_);
        return state_;
    }

    [[nodiscard]] double uniform01() noexcept {
        return static_cast<double>((next_u64() >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

[[nodiscard]] inline u64 sample_seed(
    u64 seed,
    DetRef ket,
    i64 rep = 0,
    int pair = 0
) noexcept {
    u64 x = splitmix64(seed ^ 0x243f6a8885a308d3ULL);
    x = splitmix64(x ^ det_fingerprint(ket));
    x = splitmix64(x ^ static_cast<u64>(rep + 1));
    x = splitmix64(x ^ (pair == 0 ? 0x13198a2e03707344ULL : 0xa4093822299f31d0ULL));
    return x;
}

/*
 * Generate sorted inverse-CDF targets in [0, norm).
 *
 * For exact two-pass streaming sampling:
 *
 *   pass 1: compute norm = sum_a |H_ai|
 *   pass 2: scan connected bras in the same deterministic order and hit
 *           sorted targets
 *
 * This samples connected bras with probability p(a|i) = |H_ai| / norm
 * without storing the full ket-local CDF.
 */
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

} // namespace libdet