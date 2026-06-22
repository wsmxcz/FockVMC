#pragma once

#include <cstddef>
#include <span>

#include <libdet/bit.hpp>

namespace libdet {

[[nodiscard]] inline u64 mix64(u64 x) noexcept {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

[[nodiscard]] inline u64 hash_words(u64 seed, std::span<const u64> words) noexcept {
    u64 h = mix64(seed ^ (0x9e3779b97f4a7c15ULL + static_cast<u64>(words.size())));
    for (u64 x : words) h = mix64(h ^ mix64(x + 0x517cc1b727220a95ULL));
    return h;
}

[[nodiscard]] inline u64 hash_pair(
    u64 seed,
    std::span<const u64> a,
    std::span<const u64> b
) noexcept {
    u64 h = hash_words(seed ^ 0xa0761d6478bd642fULL, a);
    return hash_words(h ^ 0xe7037ed1a0b428dbULL, b);
}

[[nodiscard]] inline std::size_t hash_size(std::size_t n) noexcept {
    std::size_t out = 1;
    const std::size_t target = n == 0 ? 1 : 2 * n;
    while (out < target) out <<= 1u;
    return out;
}

} // namespace libdet
