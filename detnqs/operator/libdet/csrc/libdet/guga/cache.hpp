#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <libdet/hash.hpp>

#include <libdet/guga/path.hpp>

namespace libdet::guga {

struct ConnSpan {
    std::size_t begin = 0;
    std::size_t end = 0;
    double degree = 0.0;
};

struct Conns {
    double cutoff = 0.0;
    double diag = 0.0;
    std::vector<u64> bra_words;
    std::vector<double> h;
    std::vector<double> prefix_abs;

    [[nodiscard]] std::size_t size() const noexcept {
        return h.size();
    }

    [[nodiscard]] PathRef bra(std::size_t idx, u32 nword) const noexcept {
        return path_at(bra_words, nword, idx);
    }

    void add(PathRef bra, double value) {
        append_path(bra_words, bra);
        h.push_back(value);
    }

    void finish(u32 nword) {
        if (h.empty()) {
            bra_words.clear();
            prefix_abs.assign(1u, 0.0);
            return;
        }

        if (h.size() == 1u) {
            prefix_abs = {0.0, std::abs(h[0])};
            return;
        }

        std::vector<std::size_t> order(h.size());
        for (std::size_t k = 0; k < order.size(); ++k) {
            order[k] = k;
        }

        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                const double a = std::abs(h[lhs]);
                const double b = std::abs(h[rhs]);

                if (a != b) return a > b;
                return PathLess{}(
                    path_at(bra_words, nword, lhs),
                    path_at(bra_words, nword, rhs)
                );
            }
        );

        std::vector<u64> sorted_words;
        std::vector<double> sorted_h;

        sorted_words.reserve(bra_words.size());
        sorted_h.reserve(h.size());

        for (std::size_t old : order) {
            append_path(sorted_words, path_at(bra_words, nword, old));
            sorted_h.push_back(h[old]);
        }

        bra_words.swap(sorted_words);
        h.swap(sorted_h);

        prefix_abs.resize(h.size() + 1u);
        prefix_abs[0] = 0.0;

        for (std::size_t k = 0; k < h.size(); ++k) {
            prefix_abs[k + 1u] = prefix_abs[k] + std::abs(h[k]);
        }
    }

    [[nodiscard]] std::size_t count(double eps) const noexcept {
        const auto it = std::partition_point(
            h.begin(),
            h.end(),
            [eps](double value) {
                return std::abs(value) >= eps;
            }
        );

        return static_cast<std::size_t>(it - h.begin());
    }

    [[nodiscard]] double abs_sum(
        std::size_t begin,
        std::size_t end
    ) const noexcept {
        if (begin >= end || end >= prefix_abs.size()) return 0.0;
        return prefix_abs[end] - prefix_abs[begin];
    }

    [[nodiscard]] ConnSpan span(double eps1, double eps2) const noexcept {
        const std::size_t begin = count(eps1);
        const std::size_t end = count(eps2);

        if (end <= begin) return ConnSpan{begin, begin, 0.0};
        return ConnSpan{begin, end, abs_sum(begin, end)};
    }
};

class ConnCache {
public:
    static constexpr std::size_t way = 4;
    static constexpr std::size_t capacity = 4096;
    static_assert((capacity & (capacity - 1u)) == 0u);
    static_assert(capacity % way == 0u);

    static constexpr std::size_t nset = capacity / way;
    static_assert((nset & (nset - 1u)) == 0u);

    explicit ConnCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * path_size(nword), 0u),
          entries_(capacity) {}

    [[nodiscard]] std::shared_ptr<const Conns> find(PathRef ket, double eps) {
        if (eps <= 0.0) return {};

        const u64 fingerprint = path_fingerprint(ket);
        const std::size_t begin = set_begin(ket, fingerprint);
        if (begin >= entries_.size()) return {};

        for (std::size_t k = 0; k < way; ++k) {
            const std::size_t slot = begin + k;
            Entry& entry = entries_[slot];

            if (!entry.conns) continue;
            if (entry.fingerprint != fingerprint) continue;
            if (!path_equal(ket_at(slot), ket)) continue;
            if (entry.conns->cutoff > eps) return {};

            touch(entry);
            return entry.conns;
        }

        return {};
    }

    void insert(PathRef ket, std::shared_ptr<const Conns> conns) {
        if (
            nword_ == 0
            || ket.nword() != nword_
            || !conns
            || conns->cutoff <= 0.0
        ) {
            return;
        }

        const u64 fingerprint = path_fingerprint(ket);
        const std::size_t begin = set_begin(ket, fingerprint);
        if (begin >= entries_.size()) return;

        std::size_t slot = begin;
        bool found = false;

        for (std::size_t k = 0; k < way; ++k) {
            const std::size_t item = begin + k;
            Entry& entry = entries_[item];

            if (!entry.conns) {
                slot = item;
                break;
            }

            if (
                entry.fingerprint == fingerprint
                && path_equal(ket_at(item), ket)
            ) {
                if (entry.conns->cutoff <= conns->cutoff) {
                    touch(entry);
                    return;
                }

                slot = item;
                found = true;
                break;
            }

            if (victim_less(entry, entries_[slot])) {
                slot = item;
            }
        }

        const std::size_t stride = path_size(nword_);
        u64* ptr = words_.data() + slot * stride;

        std::copy(ket.up().begin(), ket.up().end(), ptr);
        std::copy(ket.down().begin(), ket.down().end(), ptr + nword_);

        Entry& entry = entries_[slot];
        entry.conns = std::move(conns);
        entry.fingerprint = fingerprint;
        entry.stamp = ++clock_;
        entry.hit = found ? bump(entry.hit) : 1u;
    }

private:
    struct Entry {
        std::shared_ptr<const Conns> conns;
        u64 fingerprint = 0;
        u64 stamp = 0;
        unsigned char hit = 0;
    };

    u32 nword_ = 0;
    u64 clock_ = 0;
    std::vector<u64> words_;
    std::vector<Entry> entries_;

    [[nodiscard]] PathRef ket_at(std::size_t slot) const noexcept {
        const std::size_t stride = path_size(nword_);
        const u64* ptr = words_.data() + slot * stride;
        return PathRef(ptr, ptr + nword_, nword_);
    }

    [[nodiscard]] std::size_t set_begin(
        PathRef ket,
        u64 fingerprint
    ) const noexcept {
        if (
            nword_ == 0
            || ket.nword() != nword_
            || entries_.empty()
        ) {
            return entries_.size();
        }

        return set_begin(fingerprint);
    }

    [[nodiscard]] static std::size_t set_begin(u64 fingerprint) noexcept {
        return (
            static_cast<std::size_t>(mix64(fingerprint))
            & (nset - 1u)
        ) * way;
    }

    [[nodiscard]] static unsigned char bump(unsigned char hit) noexcept {
        return hit < 3u ? static_cast<unsigned char>(hit + 1u) : hit;
    }

    void touch(Entry& entry) noexcept {
        entry.stamp = ++clock_;
        entry.hit = bump(entry.hit);
    }

    [[nodiscard]] static bool victim_less(
        const Entry& lhs,
        const Entry& rhs
    ) noexcept {
        if (!rhs.conns) return false;
        if (!lhs.conns) return true;
        if (lhs.hit != rhs.hit) return lhs.hit < rhs.hit;
        return lhs.stamp < rhs.stamp;
    }
};

class SpaceCache {
public:
    [[nodiscard]] std::shared_ptr<const PathSpace> find(
        PathBatchView paths
    ) const {
        const std::size_t size = paths.n_paths * path_size(paths.nword);

        if (
            !space_
            || paths.nword != nword_
            || paths.n_paths != n_paths_
            || size != words_.size()
            || fingerprint(paths) != fingerprint_
        ) {
            return {};
        }

        if (!std::equal(words_.begin(), words_.end(), paths.data)) return {};
        return space_;
    }

    void insert(
        PathBatchView paths,
        std::shared_ptr<const PathSpace> space
    ) {
        nword_ = paths.nword;
        n_paths_ = paths.n_paths;
        fingerprint_ = fingerprint(paths);
        copy_paths(words_, paths);
        space_ = std::move(space);
    }

private:
    u32 nword_ = 0;
    std::size_t n_paths_ = 0;
    u64 fingerprint_ = 0;
    std::vector<u64> words_;
    std::shared_ptr<const PathSpace> space_;

    [[nodiscard]] static u64 fingerprint(PathBatchView paths) noexcept {
        const std::size_t size = paths.n_paths * path_size(paths.nword);
        const u64 seed = mix64(
            static_cast<u64>(paths.n_paths)
            ^ (static_cast<u64>(paths.nword) << 32)
        );

        return hash_words(seed, {paths.data, size});
    }
};

} // namespace libdet::guga