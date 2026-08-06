#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <libdet/rhf/hamiltonian.hpp>

using libdet::i64;
using libdet::u32;
using libdet::u64;
using libdet::rhf::DetBatchView;
using libdet::rhf::DetRef;
using libdet::rhf::Hamiltonian;

struct Basis {
    u32 nword = 0;
    std::vector<u64> words;

    [[nodiscard]] std::size_t size() const noexcept {
        return words.size() / libdet::word_pair_size(nword);
    }

    [[nodiscard]] DetBatchView view() const noexcept {
        return DetBatchView{words.data(), size(), nword};
    }

    [[nodiscard]] DetRef get(std::size_t i) const noexcept {
        return view()[i];
    }
};

inline std::size_t pair_id(int p, int q) {
    const int hi = std::max(p, q);
    const int lo = std::min(p, q);
    return static_cast<std::size_t>(hi * (hi + 1) / 2 + lo);
}

inline std::size_t eri_id(int p, int q, int r, int s) {
    return pair_id(
        static_cast<int>(pair_id(p, q)),
        static_cast<int>(pair_id(r, s))
    );
}

inline std::vector<double> make_h1(int n) {
    std::vector<double> h(static_cast<std::size_t>(n * n), 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q <= p; ++q) {
            const double v = 0.11 + 0.07 * (p + 1) - 0.03 * (q + 2)
                + 0.015 * ((p + 2 * q) % 5);
            h[static_cast<std::size_t>(p * n + q)] = v;
            h[static_cast<std::size_t>(q * n + p)] = v;
        }
    }
    return h;
}

inline std::vector<double> make_eri(int n) {
    const int npair = n * (n + 1) / 2;
    std::vector<double> eri(static_cast<std::size_t>(npair * (npair + 1) / 2), 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    eri[eri_id(p, q, r, s)] =
                        0.012 * (1 + ((3 * p + 5 * q + 7 * r + 11 * s) % 23));
                }
            }
        }
    }
    return eri;
}

template <class F>
void choose(int n, int k, int first, u64 bits, F&& visit) {
    if (k == 0) {
        visit(bits);
        return;
    }
    for (int p = first; p <= n - k; ++p) {
        choose(n, k - 1, p + 1, bits | (u64{1} << p), visit);
    }
}

inline Basis det_basis(int norb, int na, int nb) {
    const u32 nword = libdet::bits::words_for(norb);
    Basis basis{nword, {}};
    choose(norb, na, 0, u64{0}, [&](u64 a) {
        choose(norb, nb, 0, u64{0}, [&](u64 b) {
            basis.words.push_back(a);
            for (u32 w = 1; w < nword; ++w) basis.words.push_back(0);
            basis.words.push_back(b);
            for (u32 w = 1; w < nword; ++w) basis.words.push_back(0);
        });
    });
    return basis;
}

inline bool same_state(DetRef a, DetRef b) noexcept {
    if (a.nword() != b.nword()) return false;
    return std::equal(a.alpha().begin(), a.alpha().end(), b.alpha().begin())
        && std::equal(a.beta().begin(), a.beta().end(), b.beta().begin());
}

inline int find_state(DetBatchView x, DetRef y) {
    for (std::size_t i = 0; i < x.n_dets; ++i) {
        if (same_state(x[i], y)) return static_cast<int>(i);
    }
    return -1;
}

inline Basis take_basis(const Basis& basis, std::size_t n) {
    const std::size_t stride = libdet::word_pair_size(basis.nword);
    if (n > basis.size()) n = basis.size();
    Basis out{basis.nword, {}};
    out.words.insert(
        out.words.end(),
        basis.words.begin(),
        basis.words.begin() + static_cast<std::ptrdiff_t>(n * stride)
    );
    return out;
}
