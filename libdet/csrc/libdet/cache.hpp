#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include <libdet/bras.hpp>
#include <libdet/kets.hpp>

namespace libdet {

struct KetConns {
    double cutoff = 0.0;
    double diag = 0.0;
    std::vector<Coupling> couplings;
    std::vector<double> prefix_abs;

    void finish() {
        std::sort(
            couplings.begin(),
            couplings.end(),
            [](const Coupling& lhs, const Coupling& rhs) {
                const double a = std::abs(lhs.h);
                const double b = std::abs(rhs.h);
                if (a != b) return a > b;
                return excitation_less(lhs.excitation, rhs.excitation);
            }
        );

        prefix_abs.resize(couplings.size() + 1u);
        prefix_abs[0] = 0.0;

        for (std::size_t k = 0; k < couplings.size(); ++k) {
            prefix_abs[k + 1u] = prefix_abs[k] + std::abs(couplings[k].h);
        }
    }

    [[nodiscard]] std::size_t count(double eps) const noexcept {
        const auto it = std::partition_point(
            couplings.begin(),
            couplings.end(),
            [eps](const Coupling& coupling) {
                return std::abs(coupling.h) >= eps;
            }
        );

        return static_cast<std::size_t>(it - couplings.begin());
    }

};

class KetCache {
public:
    static constexpr std::size_t capacity = 8192;

    explicit KetCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * det_size(nword), 0u),
          entries_(capacity) {}

    [[nodiscard]] std::shared_ptr<const KetConns> find(
        DetRef ket,
        double eps
    ) {
        const std::size_t slot = find_slot(ket);
        if (slot == npos) return {};

        Entry& entry = entries_[slot];
        if (entry.conns->cutoff > eps) return {};

        entry.stamp = ++clock_;
        return entry.conns;
    }

    void insert(DetRef ket, std::shared_ptr<const KetConns> conns) {
        if (nword_ == 0 || ket.nword() != nword_ || !conns) {
            return;
        }

        const std::size_t old = find_slot(ket);

        if (
            old != npos
            && entries_[old].conns
            && entries_[old].conns->cutoff <= conns->cutoff
        ) {
            entries_[old].stamp = ++clock_;
            return;
        }

        const std::size_t slot = old == npos ? victim_slot() : old;
        const std::size_t stride = det_size(nword_);
        u64* dst = words_.data() + slot * stride;

        std::copy(ket.alpha().begin(), ket.alpha().end(), dst);
        std::copy(ket.beta().begin(), ket.beta().end(), dst + nword_);

        entries_[slot].conns = std::move(conns);
        entries_[slot].stamp = ++clock_;
    }

private:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    struct Entry {
        std::shared_ptr<const KetConns> conns;
        u64 stamp = 0;
    };

    u32 nword_ = 0;
    std::vector<u64> words_;
    std::vector<Entry> entries_;
    std::size_t size_ = 0;
    u64 clock_ = 0;

    [[nodiscard]] DetRef ket_at(std::size_t slot) const noexcept {
        const std::size_t stride = det_size(nword_);
        const u64* ptr = words_.data() + slot * stride;
        return DetRef(ptr, ptr + nword_, nword_);
    }

    [[nodiscard]] std::size_t find_slot(DetRef ket) const noexcept {
        if (nword_ == 0 || ket.nword() != nword_) return npos;

        for (std::size_t slot = 0; slot < size_; ++slot) {
            if (det_equal(ket_at(slot), ket)) return slot;
        }

        return npos;
    }

    [[nodiscard]] std::size_t victim_slot() noexcept {
        if (size_ < capacity) return size_++;

        std::size_t victim = 0;
        u64 oldest = std::numeric_limits<u64>::max();

        for (std::size_t slot = 0; slot < size_; ++slot) {
            if (entries_[slot].stamp < oldest) {
                oldest = entries_[slot].stamp;
                victim = slot;
            }
        }

        return victim;
    }
};

class KetSpaceCache {
public:
    [[nodiscard]] std::shared_ptr<const KetSpace> find(
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
        std::shared_ptr<const KetSpace> space
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
    std::shared_ptr<const KetSpace> space_;

    [[nodiscard]] static u64 fingerprint(DetBatchView kets) noexcept {
        const std::size_t size = kets.n_dets * det_size(kets.nword);
        const u64 seed = splitmix64(
            static_cast<u64>(kets.n_dets)
            ^ (static_cast<u64>(kets.nword) << 32)
        );
        return hash_words(seed, {kets.data, size});
    }
};

} // namespace libdet
