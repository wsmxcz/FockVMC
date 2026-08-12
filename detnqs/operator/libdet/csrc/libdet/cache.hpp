#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <libdet/det.hpp>

namespace libdet {

struct ConnSpan {
    std::size_t begin = 0;
    std::size_t end = 0;
    double degree = 0.0;
};

struct Conn {
    Excitation excitation;
    double h = 0.0;
};

struct ConnSet {
    double cutoff = 0.0;
    double diag = 0.0;
    std::vector<Conn> terms;
    std::vector<double> prefix_abs;

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

    [[nodiscard]] ConnSpan span(double eps1, double eps2) const noexcept {
        const auto count = [&](double eps) {
            const auto it = std::partition_point(
                terms.begin(),
                terms.end(),
                [eps](const Conn& term) {
                    return std::abs(term.h) >= eps;
                }
            );
            return static_cast<std::size_t>(it - terms.begin());
        };
        const std::size_t begin = count(eps1);
        const std::size_t end = count(eps2);

        if (end <= begin) return ConnSpan{begin, begin, 0.0};
        return ConnSpan{begin, end, prefix_abs[end] - prefix_abs[begin]};
    }
};

class ConnCache {
public:
    static constexpr std::size_t way = 4;
    static constexpr std::size_t capacity = 8192;
    static_assert((capacity & (capacity - 1u)) == 0u);
    static_assert(capacity % way == 0u);

    static constexpr std::size_t nset = capacity / way;
    static_assert((nset & (nset - 1u)) == 0u);

    explicit ConnCache(u32 nword = 0)
        : nword_(nword),
          words_(capacity * det_size(nword), 0u),
          entries_(capacity) {}

    [[nodiscard]] std::shared_ptr<const ConnSet> find(DetRef ket, double eps) {
        if (eps <= 0.0 || nword_ == 0 || ket.nword() != nword_) return {};

        const u64 fingerprint = det_fingerprint(ket);
        const std::size_t begin = set_begin(fingerprint);

        for (std::size_t k = 0; k < way; ++k) {
            const std::size_t slot = begin + k;
            Entry& entry = entries_[slot];

            if (!entry.conns) continue;
            if (entry.fingerprint != fingerprint) continue;
            if (!det_equal(ket_at(slot), ket)) continue;
            if (entry.conns->cutoff > eps) return {};

            touch(entry);
            return entry.conns;
        }

        return {};
    }

    void insert(DetRef ket, std::shared_ptr<const ConnSet> conns) {
        if (
            nword_ == 0
            || ket.nword() != nword_
            || !conns
            || conns->cutoff <= 0.0
        ) {
            return;
        }

        const u64 fingerprint = det_fingerprint(ket);
        const std::size_t begin = set_begin(fingerprint);

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
                && det_equal(ket_at(item), ket)
            ) {
                if (entry.conns->cutoff <= conns->cutoff) {
                    touch(entry);
                    return;
                }

                slot = item;
                found = true;
                break;
            }

            const Entry& victim = entries_[slot];
            if (
                !entry.conns
                || (
                    victim.conns
                    && (
                        entry.hit < victim.hit
                        || (entry.hit == victim.hit && entry.stamp < victim.stamp)
                    )
                )
            ) {
                slot = item;
            }
        }

        const std::size_t stride = det_size(nword_);
        u64* ptr = words_.data() + slot * stride;

        std::copy(ket.alpha().begin(), ket.alpha().end(), ptr);
        std::copy(ket.beta().begin(), ket.beta().end(), ptr + nword_);

        Entry& entry = entries_[slot];
        entry.conns = std::move(conns);
        entry.fingerprint = fingerprint;
        entry.stamp = ++clock_;
        if (!found) entry.hit = 1u;
        else if (entry.hit < 3u) ++entry.hit;
    }

private:
    struct Entry {
        std::shared_ptr<const ConnSet> conns;
        u64 fingerprint = 0;
        u64 stamp = 0;
        unsigned char hit = 0;
    };

    u32 nword_ = 0;
    u64 clock_ = 0;
    std::vector<u64> words_;
    std::vector<Entry> entries_;

    [[nodiscard]] DetRef ket_at(std::size_t slot) const noexcept {
        const std::size_t stride = det_size(nword_);
        const u64* ptr = words_.data() + slot * stride;
        return DetRef(ptr, ptr + nword_, nword_);
    }

    [[nodiscard]] static std::size_t set_begin(u64 fingerprint) noexcept {
        return (
            static_cast<std::size_t>(mix64(fingerprint))
            & (nset - 1u)
        ) * way;
    }

    void touch(Entry& entry) noexcept {
        entry.stamp = ++clock_;
        if (entry.hit < 3u) ++entry.hit;
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
        const u64 seed = mix64(
            static_cast<u64>(kets.n_dets)
            ^ (static_cast<u64>(kets.nword) << 32)
        );

        return hash_words(seed, {kets.data, size});
    }
};

} // namespace libdet
