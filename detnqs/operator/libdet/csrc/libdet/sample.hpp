#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <libdet/bit.hpp>
#include <libdet/hash.hpp>

namespace libdet::sample {

class Rng {
public:
    explicit Rng(u64 seed) : state_(seed) {}

    [[nodiscard]] double uniform01() noexcept {
        state_ = mix64(state_);
        return static_cast<double>((state_ >> 11) * 0x1.0p-53);
    }

private:
    u64 state_ = 0;
};

struct Hit {
    std::size_t conn = 0;
    i64 count = 0;
};

inline void make_targets(
    Rng& rng,
    i64 n_draw,
    double norm,
    std::vector<double>& targets
) {
    targets.clear();
    if (n_draw <= 0 || !(norm > 0.0) || !std::isfinite(norm)) return;

    targets.reserve(static_cast<std::size_t>(n_draw));
    for (i64 k = 0; k < n_draw; ++k) targets.push_back(rng.uniform01() * norm);
    std::sort(targets.begin(), targets.end());
}

template <class Conns>
inline void draw_search(
    Rng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<Hit>& hits
) {
    if (n_draw <= 0 || begin >= end || !(weight > 0.0)) return;

    const double base = conns.prefix_abs[begin];
    const auto first = conns.prefix_abs.begin() + static_cast<std::ptrdiff_t>(begin + 1u);
    const auto last = conns.prefix_abs.begin() + static_cast<std::ptrdiff_t>(end + 1u);

    for (i64 draw = 0; draw < n_draw; ++draw) {
        const double target = base + rng.uniform01() * weight;
        auto it = std::upper_bound(first, last, target);
        if (it == last) --it;
        const std::size_t idx = static_cast<std::size_t>(it - conns.prefix_abs.begin() - 1);
        if (idx >= begin && idx < end) hits.push_back({idx, 1});
    }
}

template <class HAt>
inline void draw_scan(
    Rng& rng,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<Hit>& hits,
    HAt&& h_at
) {
    make_targets(rng, n_draw, weight, targets);
    if (targets.empty()) return;

    std::size_t pos = 0;
    double cdf = 0.0;
    for (std::size_t k = begin; k < end; ++k) {
        cdf += std::abs(h_at(k));
        i64 count = 0;
        while (pos < targets.size() && targets[pos] <= cdf) {
            ++count;
            ++pos;
        }
        if (count > 0) hits.push_back({k, count});
    }
}

template <class Conns, class HAt>
inline void draw_span(
    Rng& rng,
    const Conns& conns,
    std::size_t begin,
    std::size_t end,
    i64 n_draw,
    double weight,
    std::vector<double>& targets,
    std::vector<Hit>& hits,
    HAt&& h_at
) {
    if (n_draw <= 0 || begin >= end || !(weight > 0.0)) return;

    const std::size_t n_conn = end - begin;
    if (static_cast<std::size_t>(n_draw) * 16u < n_conn) {
        draw_search(rng, conns, begin, end, n_draw, weight, hits);
    } else {
        draw_scan(
            rng,
            begin,
            end,
            n_draw,
            weight,
            targets,
            hits,
            std::forward<HAt>(h_at)
        );
    }
}

} // namespace libdet::sample
