#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <list>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <libdet/guga/path.hpp>
#include <libdet/window.hpp>

namespace libdet::guga {

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
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;

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
            [eps](double value) { return std::abs(value) >= eps; }
        );
        return static_cast<std::size_t>(it - h.begin());
    }

    [[nodiscard]] double abs_sum(std::size_t begin, std::size_t end) const noexcept {
        if (begin >= end || end >= prefix_abs.size()) return 0.0;
        return prefix_abs[end] - prefix_abs[begin];
    }

    [[nodiscard]] ::libdet::ConnWindow window(::libdet::AbsWindow win) const noexcept {
        const std::size_t begin = std::isfinite(win.hi) ? count(win.hi) : 0u;
        const std::size_t end = count(win.lo);
        if (end <= begin) return ::libdet::ConnWindow{begin, begin, 0.0};
        return ::libdet::ConnWindow{begin, end, abs_sum(begin, end)};
    }
};

class ConnCache {
public:
    static constexpr std::size_t capacity = 4096;

    explicit ConnCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * path_size(nword), 0u),
          entries_(capacity) {
        index_.reserve(capacity);
    }

    [[nodiscard]] std::shared_ptr<const Conns> find(PathRef ket, double eps) {
        const std::size_t slot = find_slot(ket);
        if (slot == npos) return {};

        Entry& entry = entries_[slot];
        if (entry.conns->cutoff > eps) return {};

        touch(slot);
        return entry.conns;
    }

    void insert(PathRef ket, std::shared_ptr<const Conns> conns) {
        if (nword_ == 0 || ket.nword() != nword_ || !conns) return;

        const std::size_t old = find_slot(ket);
        const u64 fingerprint = path_fingerprint(ket);

        if (
            old != npos
            && entries_[old].conns
            && entries_[old].conns->cutoff <= conns->cutoff
        ) {
            touch(old);
            return;
        }

        std::size_t slot = old;
        if (old == npos) {
            if (size_ < capacity) {
                slot = size_++;
                lru_.push_front(slot);
                entries_[slot].position = lru_.begin();
            } else {
                slot = lru_.back();
                erase_index(slot);
                touch(slot);
            }
        } else {
            touch(slot);
        }

        const std::size_t stride = path_size(nword_);
        u64* ptr = words_.data() + slot * stride;
        std::copy(ket.up().begin(), ket.up().end(), ptr);
        std::copy(ket.down().begin(), ket.down().end(), ptr + nword_);

        entries_[slot].conns = std::move(conns);
        entries_[slot].fingerprint = fingerprint;
        if (old == npos) index_.emplace(fingerprint, slot);
    }

private:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    struct Entry {
        std::shared_ptr<const Conns> conns;
        u64 fingerprint = 0;
        std::list<std::size_t>::iterator position;
    };

    u32 nword_ = 0;
    std::vector<u64> words_;
    std::vector<Entry> entries_;
    std::unordered_multimap<u64, std::size_t> index_;
    std::list<std::size_t> lru_;
    std::size_t size_ = 0;

    [[nodiscard]] PathRef ket_at(std::size_t slot) const noexcept {
        const std::size_t stride = path_size(nword_);
        const u64* ptr = words_.data() + slot * stride;
        return PathRef(ptr, ptr + nword_, nword_);
    }

    [[nodiscard]] std::size_t find_slot(PathRef ket) const noexcept {
        if (nword_ == 0 || ket.nword() != nword_) return npos;

        const u64 fingerprint = path_fingerprint(ket);
        const auto range = index_.equal_range(fingerprint);
        for (auto it = range.first; it != range.second; ++it) {
            const std::size_t slot = it->second;
            if (path_equal(ket_at(slot), ket)) return slot;
        }
        return npos;
    }

    void erase_index(std::size_t slot) {
        const u64 fingerprint = entries_[slot].fingerprint;
        const auto range = index_.equal_range(fingerprint);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == slot) {
                index_.erase(it);
                return;
            }
        }
    }

    void touch(std::size_t slot) {
        lru_.splice(lru_.begin(), lru_, entries_[slot].position);
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
        const u64 seed = splitmix64(
            static_cast<u64>(paths.n_paths)
            ^ (static_cast<u64>(paths.nword) << 32)
        );
        return hash_words(seed, {paths.data, size});
    }
};

} // namespace libdet::guga
