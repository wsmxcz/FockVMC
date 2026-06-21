#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace libdet {

struct AbsWindow {
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
};

struct ConnWindow {
    std::size_t begin = 0;
    std::size_t end = 0;
    double weight = 0.0;
};

[[nodiscard]] inline bool in_window(double h, AbsWindow win) noexcept {
    const double value = std::abs(h);
    return value > 0.0 && value >= win.lo && value < win.hi;
}

} // namespace libdet
