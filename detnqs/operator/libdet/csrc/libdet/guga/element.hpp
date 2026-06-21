#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <libdet/integral.hpp>
#include <libdet/guga/path.hpp>
#include <libdet/guga/segment.hpp>

namespace libdet::guga {

struct ElementScratch {
    // Scratch for the reduced intermediate-spin walk.
    explicit ElementScratch(int n = 0) { reserve(n); }

    void reserve(int n) {
        const std::size_t m = static_cast<std::size_t>(std::max(0, 2 * n + 1));
        curr.reserve(m);
        next.reserve(m);
        act.reserve(m);
        tmp.reserve(m);
    }

    std::vector<double> curr;
    std::vector<double> next;
    std::vector<std::size_t> act;
    std::vector<std::size_t> tmp;
    Seg2Table seg2;
};


[[nodiscard]] inline bool same_outer(
    const PathState& bra,
    const PathState& ket,
    int lo,
    int hi
) noexcept {
    const int n = bra.norb();
    for (int k = 0; k < lo; ++k) {
        if (bra.step[static_cast<std::size_t>(k)] != ket.step[static_cast<std::size_t>(k)]) return false;
    }
    for (int k = hi + 1; k < n; ++k) {
        if (bra.step[static_cast<std::size_t>(k)] != ket.step[static_cast<std::size_t>(k)]) return false;
    }
    return true;
}


[[nodiscard]] inline Seg1Kind seg1_kind(int k, int lo, int hi, bool right) noexcept {
    if (lo == hi) return Seg1Kind::diag;
    if (k == lo) return right ? Seg1Kind::r_tail : Seg1Kind::l_head;
    if (k == hi) return right ? Seg1Kind::r_head : Seg1Kind::l_tail;
    return right ? Seg1Kind::r_mid : Seg1Kind::l_mid;
}

[[nodiscard]] inline double coeff1(
    ElementScratch&,
    const PathState& bra,
    const PathState& ket,
    int p,
    int q
) {
    if (bra.norb() != ket.norb()) return 0.0;
    const int n = bra.norb();
    if (p < 0 || p >= n || q < 0 || q >= n) return 0.0;

    if (p == q) {
        return same_path(bra, ket)
            ? static_cast<double>(ket.occ[static_cast<std::size_t>(p)])
            : 0.0;
    }

    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    if (!same_outer(bra, ket, lo, hi)) return 0.0;

    for (int k = lo; k <= hi; ++k) {
        int diff = static_cast<int>(bra.occ[static_cast<std::size_t>(k)])
            - static_cast<int>(ket.occ[static_cast<std::size_t>(k)]);
        if (k == p) --diff;
        if (k == q) ++diff;
        if (diff != 0) return 0.0;
    }

    const bool right = p < q;
    double value = 1.0;
    for (int k = lo; k <= hi; ++k) {
        const double w = seg1(
            seg1_kind(k, lo, hi, right),
            bra.step[static_cast<std::size_t>(k)],
            ket.step[static_cast<std::size_t>(k)],
            ket.spin[static_cast<std::size_t>(k + 1)] - bra.spin[static_cast<std::size_t>(k + 1)],
            ket.spin[static_cast<std::size_t>(k + 1)]
        );
        if (w == 0.0) return 0.0;
        value *= w;
    }
    return value;
}


[[nodiscard]] inline bool coeff2_occ(
    const PathState& bra,
    const PathState& ket,
    int p,
    int q,
    int r,
    int s,
    int lo,
    int hi
) noexcept {
    for (int k = lo; k <= hi; ++k) {
        int diff = static_cast<int>(bra.occ[static_cast<std::size_t>(k)])
            - static_cast<int>(ket.occ[static_cast<std::size_t>(k)]);
        if (k == p) --diff;
        if (k == r) --diff;
        if (k == q) ++diff;
        if (k == s) ++diff;
        if (diff != 0) return false;
    }
    return true;
}

[[nodiscard]] inline double coeff2_prod(
    ElementScratch& scratch,
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    int p,
    int q,
    int r,
    int s,
    int lo,
    int hi
) {
    if (p == q && r == s) {
        if (!same_path(bra, ket)) return 0.0;
        const double np = static_cast<double>(ket.occ[static_cast<std::size_t>(p)]);
        const double nr = static_cast<double>(ket.occ[static_cast<std::size_t>(r)]);
        return np * nr;
    }
    if (p == q) {
        return static_cast<double>(bra.occ[static_cast<std::size_t>(p)])
            * coeff1(scratch, bra, ket, r, s);
    }
    if (r == s) {
        return static_cast<double>(ket.occ[static_cast<std::size_t>(r)])
            * coeff1(scratch, bra, ket, p, q);
    }

    const int n = ket.norb();
    const int width = 2 * n + 1;
    const int shift = n;
    if (scratch.curr.size() != static_cast<std::size_t>(width)) {
        scratch.curr.assign(static_cast<std::size_t>(width), 0.0);
        scratch.next.assign(static_cast<std::size_t>(width), 0.0);
    }
    scratch.act.clear();
    scratch.tmp.clear();

    if (bra.spin[static_cast<std::size_t>(lo)] != ket.spin[static_cast<std::size_t>(lo)]) return 0.0;
    const int z0 = ket.spin[static_cast<std::size_t>(lo)];
    scratch.curr[static_cast<std::size_t>(z0 + shift)] = 1.0;
    scratch.act.push_back(static_cast<std::size_t>(z0 + shift));

    for (int k = lo; k <= hi; ++k) {
        scratch.tmp.clear();
        const Shape2 sh = shape2(k, p, q, r, s);
        const Step dx = bra.step[static_cast<std::size_t>(k)];
        const Step dy = ket.step[static_cast<std::size_t>(k)];
        const int db = ket.spin[static_cast<std::size_t>(k + 1)]
            - bra.spin[static_cast<std::size_t>(k + 1)];
        const int b = ket.spin[static_cast<std::size_t>(k + 1)];

        for (std::size_t id : scratch.act) {
            const double base = scratch.curr[id];
            scratch.curr[id] = 0.0;
            if (base == 0.0) continue;

            const int z = static_cast<int>(id) - shift;
            for (const Seg2Entry& e : table.get(sh, dx, dy, db, b, z)) {
                const std::size_t pos = static_cast<std::size_t>(static_cast<int>(e.z) + shift);
                if (scratch.next[pos] == 0.0) scratch.tmp.push_back(pos);
                scratch.next[pos] += base * e.w;
            }
        }

        scratch.act.swap(scratch.tmp);
        for (std::size_t id : scratch.act) {
            scratch.curr[id] = scratch.next[id];
            scratch.next[id] = 0.0;
        }
    }

    const int edge = ket.spin[static_cast<std::size_t>(hi + 1)];
    if (edge != bra.spin[static_cast<std::size_t>(hi + 1)]) return 0.0;
    const std::size_t pos = static_cast<std::size_t>(edge + shift);
    double value = 0.0;
    for (std::size_t id : scratch.act) {
        if (id == pos) value = scratch.curr[id];
        scratch.curr[id] = 0.0;
    }
    scratch.act.clear();
    return value;
}

[[nodiscard]] inline double coeff2(
    ElementScratch& scratch,
    const PathState& bra,
    const PathState& ket,
    int p,
    int q,
    int r,
    int s
) {
    const int n = bra.norb();
    if (ket.norb() != n) return 0.0;
    if (p < 0 || p >= n || q < 0 || q >= n || r < 0 || r >= n || s < 0 || s >= n) return 0.0;

    const auto h = hull2(p, q, r, s);
    const int lo = h[0];
    const int hi = h[1];
    if (!same_outer(bra, ket, lo, hi)) return 0.0;
    if (!coeff2_occ(bra, ket, p, q, r, s, lo, hi)) return 0.0;
    if (scratch.seg2.max_b < n) build_seg2(scratch.seg2, n);

    double value = coeff2_prod(scratch, scratch.seg2, bra, ket, p, q, r, s, lo, hi);
    if (q == r) value -= coeff1(scratch, bra, ket, p, s);
    return value;
}


[[nodiscard]] inline double diag(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& ket
) {
    const int n = ket.norb();
    double value = ints.ecore();

    for (int p = 0; p < n; ++p) {
        const int np = static_cast<int>(ket.occ[static_cast<std::size_t>(p)]);
        value += ints.h1(p, p) * static_cast<double>(np);
        value += 0.5 * ints.chem(p, p, p, p) * static_cast<double>(np * (np - 1));
    }

    for (int p = 0; p < n; ++p) {
        const int np = static_cast<int>(ket.occ[static_cast<std::size_t>(p)]);
        if (np == 0) continue;
        for (int q = p + 1; q < n; ++q) {
            const int nq = static_cast<int>(ket.occ[static_cast<std::size_t>(q)]);
            if (nq == 0) continue;
            value += ints.chem(p, p, q, q) * static_cast<double>(np * nq);
            value += ints.chem(p, q, q, p) * coeff2(scratch, ket, ket, p, q, q, p);
        }
    }

    return value;
}

[[nodiscard]] inline double same_ocfg(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket
) {
    const int n = ket.norb();
    int first = n;
    int last = -1;
    for (int p = 0; p < n; ++p) {
        if (bra.step[static_cast<std::size_t>(p)] != ket.step[static_cast<std::size_t>(p)]) {
            if (first == n) first = p;
            last = p;
        }
    }
    if (last < 0) return 0.0;

    double value = 0.0;
    for (int p = 0; p <= first; ++p) {
        if (ket.occ[static_cast<std::size_t>(p)] != 1u) continue;
        for (int q = std::max(p + 1, last); q < n; ++q) {
            if (ket.occ[static_cast<std::size_t>(q)] != 1u) continue;
            value += ints.chem(p, q, q, p) * coeff2(scratch, bra, ket, p, q, q, p);
        }
    }
    return value;
}

[[nodiscard]] inline double single_move(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const OccMove& move
) {
    const int n = ket.norb();
    const int p = move.add[0];
    const int q = move.remove[0];

    double value = ints.h1(p, q) * coeff1(scratch, bra, ket, p, q);
    value += ints.chem(p, p, p, q) * coeff2(scratch, bra, ket, p, p, p, q);
    value += ints.chem(p, q, q, q) * coeff2(scratch, bra, ket, p, q, q, q);

    for (int k = 0; k < n; ++k) {
        if (k == p || k == q) continue;
        value += ints.chem(p, q, k, k) * coeff2(scratch, bra, ket, p, q, k, k);
        value += ints.chem(p, k, k, q) * coeff2(scratch, bra, ket, p, k, k, q);
    }

    return value;
}

template <class Visit>
inline void visit_pair(int a, int b, Visit&& visit) {
    visit(a, b);
    if (a != b) visit(b, a);
}

[[nodiscard]] inline double double_move(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const OccMove& move
) {
    double value = 0.0;
    visit_pair(move.add[0], move.add[1], [&](int p, int r) {
        visit_pair(move.remove[0], move.remove[1], [&](int q, int s) {
            value += 0.5 * ints.chem(p, q, r, s) * coeff2(scratch, bra, ket, p, q, r, s);
        });
    });
    return value;
}

[[nodiscard]] inline double hij(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const OccMove& move
) {
    if (bra.norb() != ket.norb() || move.degree > 2) return 0.0;

    if (move.degree == 0) {
        return same_path(bra, ket) ? diag(scratch, ints, ket) : same_ocfg(scratch, ints, bra, ket);
    }
    if (move.degree == 1) return single_move(scratch, ints, bra, ket, move);
    return double_move(scratch, ints, bra, ket, move);
}

[[nodiscard]] inline double hij(
    ElementScratch& scratch,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket
) {
    return hij(scratch, ints, bra, ket, occ_move(bra.occ, ket.occ));
}


} // namespace libdet::guga
