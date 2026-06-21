#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include <libdet/rhf/det.hpp>
#include <libdet/window.hpp>

namespace libdet::rhf {

struct Conn {
    Excitation excitation;
    double h = 0.0;
};

struct Conns {
    double cutoff = 0.0;
    double diag = 0.0;
    std::vector<Conn> terms;
    std::vector<double> prefix_abs;

    [[nodiscard]] std::size_t size() const noexcept {
        return terms.size();
    }

    void add(Excitation excitation, double value) {
        terms.push_back(Conn{excitation, value});
    }

    void finish() {
        if (terms.empty()) {
            prefix_abs.assign(1u, 0.0);
            return;
        }
        if (terms.size() > 1u) {
            std::sort(
                terms.begin(),
                terms.end(),
                [](const Conn& lhs, const Conn& rhs) {
                    const double a = std::abs(lhs.h);
                    const double b = std::abs(rhs.h);
                    if (a != b) return a > b;
                    return excitation_less(lhs.excitation, rhs.excitation);
                }
            );
        }

        prefix_abs.resize(terms.size() + 1u);
        prefix_abs[0] = 0.0;
        for (std::size_t k = 0; k < terms.size(); ++k) {
            prefix_abs[k + 1u] = prefix_abs[k] + std::abs(terms[k].h);
        }
    }

    [[nodiscard]] std::size_t count(double eps) const noexcept {
        const auto it = std::partition_point(
            terms.begin(),
            terms.end(),
            [eps](const Conn& term) {
                return std::abs(term.h) >= eps;
            }
        );

        return static_cast<std::size_t>(it - terms.begin());
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
    static constexpr std::size_t capacity = 8192;

    explicit ConnCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * det_size(nword), 0u),
          entries_(capacity) {
        index_.reserve(capacity);
    }

    [[nodiscard]] std::shared_ptr<const Conns> find(
        DetRef ket,
        double eps
    ) {
        const std::size_t slot = find_slot(ket);
        if (slot == npos) return {};

        Entry& entry = entries_[slot];
        if (entry.conns->cutoff > eps) return {};

        touch(slot);
        return entry.conns;
    }

    void insert(DetRef ket, std::shared_ptr<const Conns> conns) {
        if (nword_ == 0 || ket.nword() != nword_ || !conns) {
            return;
        }

        const std::size_t old = find_slot(ket);
        const u64 fingerprint = det_fingerprint(ket);

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

        const std::size_t stride = det_size(nword_);
        u64* slot_words = words_.data() + slot * stride;

        std::copy(ket.alpha().begin(), ket.alpha().end(), slot_words);
        std::copy(ket.beta().begin(), ket.beta().end(), slot_words + nword_);

        entries_[slot].conns = std::move(conns);
        entries_[slot].fingerprint = fingerprint;
        if (old == npos) {
            index_.emplace(fingerprint, slot);
        }
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

    [[nodiscard]] DetRef ket_at(std::size_t slot) const noexcept {
        const std::size_t stride = det_size(nword_);
        const u64* ptr = words_.data() + slot * stride;
        return DetRef(ptr, ptr + nword_, nword_);
    }

    [[nodiscard]] std::size_t find_slot(DetRef ket) const noexcept {
        if (nword_ == 0 || ket.nword() != nword_) return npos;

        const u64 fingerprint = det_fingerprint(ket);
        const auto range = index_.equal_range(fingerprint);
        for (auto it = range.first; it != range.second; ++it) {
            const std::size_t slot = it->second;
            if (det_equal(ket_at(slot), ket)) return slot;
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
    [[nodiscard]] std::shared_ptr<const DetSpace> find(
        DetBatchView kets
    ) const {
        const std::size_t size = kets.n_dets * det_size(kets.nword);

        if (
            !space_
            || kets.nword != nword_
            || size != words_.size()
            || fingerprint(kets) != fingerprint_
        ) {
            return {};
        }

        if (!std::equal(words_.begin(), words_.end(), kets.data)) return {};
        return space_;
    }

    void insert(
        DetBatchView kets,
        std::shared_ptr<const DetSpace> space
    ) {
        nword_ = kets.nword;
        fingerprint_ = fingerprint(kets);
        copy_batch(words_, kets);
        space_ = std::move(space);
    }

private:
    u32 nword_ = 0;
    u64 fingerprint_ = 0;
    std::vector<u64> words_;
    std::shared_ptr<const DetSpace> space_;

    [[nodiscard]] static u64 fingerprint(DetBatchView kets) noexcept {
        const std::size_t size = kets.n_dets * det_size(kets.nword);
        const u64 seed = splitmix64(
            static_cast<u64>(kets.n_dets)
            ^ (static_cast<u64>(kets.nword) << 32)
        );
        return hash_words(seed, {kets.data, size});
    }
};

} // namespace libdet::rhf
