#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <list>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include <libdet/guga/element.hpp>
#include <libdet/spatial/space.hpp>

namespace libdet::guga {

struct KetConns {
    double cutoff = 0.0;
    double diag = 0.0;
    std::vector<u64> bra_words;
    std::vector<double> h;
    std::vector<double> prefix_abs;

    [[nodiscard]] std::size_t size() const noexcept {
        return h.size();
    }

    [[nodiscard]] DetRef bra(std::size_t idx, u32 nword) const noexcept {
        return det_at(bra_words, nword, idx);
    }

    void finish(u32 nword) {
        std::vector<std::size_t> order(h.size());
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;

        std::sort(
            order.begin(),
            order.end(),
            [&](std::size_t lhs, std::size_t rhs) {
                const double a = std::abs(h[lhs]);
                const double b = std::abs(h[rhs]);
                if (a != b) return a > b;
                return DetLess{}(
                    det_at(bra_words, nword, lhs),
                    det_at(bra_words, nword, rhs)
                );
            }
        );

        std::vector<u64> sorted_words;
        std::vector<double> sorted_h;
        sorted_words.reserve(bra_words.size());
        sorted_h.reserve(h.size());

        for (std::size_t old : order) {
            append_det(sorted_words, det_at(bra_words, nword, old));
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
};

class KetCache {
public:
    static constexpr std::size_t capacity = 4096;

    explicit KetCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * det_size(nword), 0u),
          entries_(capacity) {
        index_.reserve(capacity);
    }

    [[nodiscard]] std::shared_ptr<const KetConns> find(DetRef ket, double eps) {
        const std::size_t slot = find_slot(ket);
        if (slot == npos) return {};

        Entry& entry = entries_[slot];
        if (entry.conns->cutoff > eps) return {};

        touch(slot);
        return entry.conns;
    }

    void insert(DetRef ket, std::shared_ptr<const KetConns> conns) {
        if (nword_ == 0 || ket.nword() != nword_ || !conns) return;

        const std::size_t old = find_slot(ket);
        const u64 fingerprint = det_fingerprint(ket);

        const auto collision = index_.find(fingerprint);
        if (
            old == npos
            && collision != index_.end()
            && !det_equal(ket_at(collision->second), ket)
        ) {
            return;
        }

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
        u64* ptr = words_.data() + slot * stride;
        std::copy(ket.alpha().begin(), ket.alpha().end(), ptr);
        std::copy(ket.beta().begin(), ket.beta().end(), ptr + nword_);

        entries_[slot].conns = std::move(conns);
        entries_[slot].fingerprint = fingerprint;
        if (old == npos) index_.emplace(fingerprint, slot);
    }

private:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    struct Entry {
        std::shared_ptr<const KetConns> conns;
        u64 fingerprint = 0;
        std::list<std::size_t>::iterator position;
    };

    u32 nword_ = 0;
    std::vector<u64> words_;
    std::vector<Entry> entries_;
    std::unordered_map<u64, std::size_t> index_;
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
        const auto it = index_.find(fingerprint);
        if (it == index_.end()) return npos;

        const std::size_t slot = it->second;
        return det_equal(ket_at(slot), ket) ? slot : npos;
    }

    void erase_index(std::size_t slot) {
        const u64 fingerprint = entries_[slot].fingerprint;
        const auto it = index_.find(fingerprint);
        if (it != index_.end() && it->second == slot) index_.erase(it);
    }

    void touch(std::size_t slot) {
        lru_.splice(lru_.begin(), lru_, entries_[slot].position);
    }
};

struct CsfSpaceItem {
    Cfg cfg;
    i32 det = -1;
};

class CsfSpace {
public:
    CsfSpace(DetBatchView dets, Sector sector)
        : nword(dets.nword) {
        copy_batch(det_words, dets);
        csfs.reserve(dets.n_dets);
        items.reserve(dets.n_dets);

        for (std::size_t idet = 0; idet < dets.n_dets; ++idet) {
            csfs.push_back(decode_csf(dets[idet], sector, "CsfSpace"));
            items.push_back({csfs.back().cfg, to_i32(idet)});
        }

        std::sort(
            items.begin(),
            items.end(),
            [](const CsfSpaceItem& lhs, const CsfSpaceItem& rhs) {
                if (lhs.cfg != rhs.cfg) return lhs.cfg < rhs.cfg;
                return lhs.det < rhs.det;
            }
        );
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return nword == 0 ? 0u : det_words.size() / det_size(nword);
    }

    [[nodiscard]] DetRef det(std::size_t idet) const noexcept {
        return det_at(det_words, nword, idet);
    }

    [[nodiscard]] const Csf& csf(std::size_t idet) const noexcept {
        return csfs[idet];
    }

    [[nodiscard]] std::span<const CsfSpaceItem> with_cfg(
        const Cfg& cfg
    ) const noexcept {
        if (items.empty()) return {};

        const auto lo = std::lower_bound(
            items.begin(),
            items.end(),
            cfg,
            [](const CsfSpaceItem& item, const Cfg& value) {
                return item.cfg < value;
            }
        );

        const auto hi = std::upper_bound(
            items.begin(),
            items.end(),
            cfg,
            [](const Cfg& value, const CsfSpaceItem& item) {
                return value < item.cfg;
            }
        );

        return {
            items.data() + static_cast<std::size_t>(lo - items.begin()),
            static_cast<std::size_t>(hi - lo)
        };
    }

    u32 nword = 0;
    std::vector<u64> det_words;
    std::vector<Csf> csfs;
    std::vector<CsfSpaceItem> items;
};

class CsfSpaceCache {
public:
    [[nodiscard]] std::shared_ptr<const CsfSpace> find(DetBatchView dets) const {
        const std::size_t size = dets.n_dets * det_size(dets.nword);

        if (
            !space_
            || dets.nword != nword_
            || size != words_.size()
            || fingerprint(dets) != fingerprint_
        ) {
            return {};
        }

        if (!std::equal(words_.begin(), words_.end(), dets.data)) return {};
        return space_;
    }

    void insert(DetBatchView dets, std::shared_ptr<const CsfSpace> space) {
        nword_ = dets.nword;
        fingerprint_ = fingerprint(dets);
        copy_batch(words_, dets);
        space_ = std::move(space);
    }

private:
    u32 nword_ = 0;
    u64 fingerprint_ = 0;
    std::vector<u64> words_;
    std::shared_ptr<const CsfSpace> space_;

    [[nodiscard]] static u64 fingerprint(DetBatchView dets) noexcept {
        const std::size_t size = dets.n_dets * det_size(dets.nword);
        const u64 seed = splitmix64(
            static_cast<u64>(dets.n_dets)
            ^ (static_cast<u64>(dets.nword) << 32)
        );
        return hash_words(seed, {dets.data, size});
    }
};

} // namespace libdet::guga
