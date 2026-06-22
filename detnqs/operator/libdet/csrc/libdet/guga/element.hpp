#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include <libdet/integral.hpp>
#include <libdet/guga/path.hpp>
#include <libdet/guga/segment.hpp>

namespace libdet::guga {

struct PathDiff {
    OccMove move;
    int first = 0;
    int last = -1;

    [[nodiscard]] bool same_path() const noexcept { return last < first; }

    [[nodiscard]] bool contains(int lo, int hi) const noexcept {
        return same_path() || (lo <= first && last <= hi);
    }
};

[[nodiscard]] inline PathDiff path_diff(
    const PathState& ket,
    const PathState& bra,
    OccMove move
) noexcept {
    PathDiff out;
    out.move = move;
    const int n = ket.norb();
    out.first = n;
    out.last = -1;
    if (bra.norb() != n) {
        out.move.degree = 3;
        out.first = 0;
        out.last = 0;
        return out;
    }
    for (int k = 0; k < n; ++k) {
        if (bra.step[static_cast<std::size_t>(k)] != ket.step[static_cast<std::size_t>(k)]) {
            if (out.first == n) out.first = k;
            out.last = k;
        }
    }
    return out;
}

[[nodiscard]] inline PathDiff path_diff(
    const PathState& ket,
    const PathState& bra
) noexcept {
    return path_diff(ket, bra, occ_move(ket.occ, bra.occ));
}

struct ElementScratch {
    explicit ElementScratch(int n = 0) { reserve(n); }

    void reserve(int n) {
        norb = std::max(0, n);
        const std::size_t width = static_cast<std::size_t>(2 * norb + 1);
        curr.reserve(width);
        next.reserve(width);
        act.reserve(width);
        tmp.reserve(width);
        single.assign(static_cast<std::size_t>(norb) * static_cast<std::size_t>(norb), 0.0);
    }

    void load_single(const Integral& ints, const PathState& ket) {
        const int n = ket.norb();
        if (n != norb) reserve(n);
        for (int p = 0; p < n; ++p) {
            for (int q = 0; q < n; ++q) {
                double value = ints.h1(p, q) - ints.coulomb(p, q, q);
                for (int k : ket.donor) {
                    value += static_cast<double>(ket.occ[static_cast<std::size_t>(k)])
                        * ints.coulomb(p, q, k);
                }
                single[slot(p, q)] = value;
            }
        }
    }

    [[nodiscard]] double single_coulomb(int p, int q) const noexcept {
        return single[slot(p, q)];
    }

    int norb = 0;
    std::vector<double> curr;
    std::vector<double> next;
    std::vector<std::size_t> act;
    std::vector<std::size_t> tmp;
    std::vector<double> single;

private:
    [[nodiscard]] std::size_t slot(int p, int q) const noexcept {
        return static_cast<std::size_t>(p) * static_cast<std::size_t>(norb)
            + static_cast<std::size_t>(q);
    }
};

[[nodiscard]] inline OccMove invalid_move() noexcept {
    OccMove out;
    out.degree = 3;
    out.remove = {-1, -1};
    out.add = {-1, -1};
    return out;
}

[[nodiscard]] inline bool same_move(const OccMove& lhs, const OccMove& rhs) noexcept {
    if (lhs.degree != rhs.degree) return false;
    if (lhs.degree < 0 || lhs.degree > 2) return false;
    for (int i = 0; i < lhs.degree; ++i) {
        const std::size_t j = static_cast<std::size_t>(i);
        if (lhs.remove[j] != rhs.remove[j] || lhs.add[j] != rhs.add[j]) return false;
    }
    return true;
}

namespace detail {

struct MoveBuilder {
    std::array<int, 4> idx{-1, -1, -1, -1};
    std::array<int, 4> val{0, 0, 0, 0};
    int size = 0;

    void add(int p, int delta) noexcept {
        for (int i = 0; i < size; ++i) {
            if (idx[static_cast<std::size_t>(i)] == p) {
                val[static_cast<std::size_t>(i)] += delta;
                return;
            }
        }
        idx[static_cast<std::size_t>(size)] = p;
        val[static_cast<std::size_t>(size)] = delta;
        ++size;
    }

    [[nodiscard]] OccMove move() const noexcept {
        OccMove out;
        int nr = 0;
        int na = 0;
        for (int p = 0; p < size; ++p) {
            const int v = val[static_cast<std::size_t>(p)];
            const int id = idx[static_cast<std::size_t>(p)];
            if (v == 0) continue;
            if (v < -2 || v > 2) return invalid_move();
            if (v < 0) {
                for (int k = 0; k < -v; ++k) {
                    if (nr >= 2) return invalid_move();
                    out.remove[static_cast<std::size_t>(nr++)] = id;
                }
            } else {
                for (int k = 0; k < v; ++k) {
                    if (na >= 2) return invalid_move();
                    out.add[static_cast<std::size_t>(na++)] = id;
                }
            }
        }
        if (nr != na) return invalid_move();
        out.degree = nr;
        if (out.degree == 2) {
            if (out.remove[1] < out.remove[0]) std::swap(out.remove[0], out.remove[1]);
            if (out.add[1] < out.add[0]) std::swap(out.add[0], out.add[1]);
        }
        return out;
    }
};

} // namespace detail

[[nodiscard]] inline OccMove one_move(int p, int q) noexcept {
    detail::MoveBuilder move;
    move.add(q, -1);
    move.add(p, 1);
    return move.move();
}

[[nodiscard]] inline OccMove two_move(int p, int q, int r, int s) noexcept {
    detail::MoveBuilder move;
    move.add(q, -1);
    move.add(s, -1);
    move.add(p, 1);
    move.add(r, 1);
    return move.move();
}

[[nodiscard]] inline Seg1Kind seg1_kind(int k, int lo, int hi, bool right) noexcept {
    if (lo == hi) return Seg1Kind::diag;
    if (k == lo) return right ? Seg1Kind::r_tail : Seg1Kind::l_head;
    if (k == hi) return right ? Seg1Kind::r_head : Seg1Kind::l_tail;
    return right ? Seg1Kind::r_mid : Seg1Kind::l_mid;
}

[[nodiscard]] inline double coeff1_move(
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q
) {
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    if (!diff.contains(lo, hi)) return 0.0;

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

[[nodiscard]] inline double coeff1(
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q
) {
    if (bra.norb() != ket.norb()) return 0.0;
    const int n = bra.norb();
    if (p < 0 || p >= n || q < 0 || q >= n) return 0.0;

    if (p == q) {
        return diff.same_path()
            ? static_cast<double>(ket.occ[static_cast<std::size_t>(p)])
            : 0.0;
    }

    if (!same_move(diff.move, one_move(p, q))) return 0.0;
    return coeff1_move(bra, ket, diff, p, q);
}

inline void clear_active(ElementScratch& scratch) noexcept {
    for (std::size_t id : scratch.act) scratch.curr[id] = 0.0;
    scratch.act.clear();
    for (std::size_t id : scratch.tmp) scratch.next[id] = 0.0;
    scratch.tmp.clear();
}

[[nodiscard]] inline double prod2(
    ElementScratch& scratch,
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q,
    int r,
    int s,
    int lo,
    int hi
) {
    if (p == q && r == s) {
        if (!diff.same_path()) return 0.0;
        const double np = static_cast<double>(ket.occ[static_cast<std::size_t>(p)]);
        const double nr = static_cast<double>(ket.occ[static_cast<std::size_t>(r)]);
        return np * nr;
    }
    if (p == q) {
        return static_cast<double>(bra.occ[static_cast<std::size_t>(p)])
            * coeff1(bra, ket, diff, r, s);
    }
    if (r == s) {
        return static_cast<double>(ket.occ[static_cast<std::size_t>(r)])
            * coeff1(bra, ket, diff, p, q);
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

    const int spin = ket.spin[static_cast<std::size_t>(hi + 1)];
    if (spin != bra.spin[static_cast<std::size_t>(hi + 1)]) {
        clear_active(scratch);
        return 0.0;
    }

    const std::size_t pos = static_cast<std::size_t>(spin + shift);
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
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
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
    if (!diff.contains(lo, hi)) return 0.0;
    if (!same_move(diff.move, two_move(p, q, r, s))) return 0.0;

    double value = prod2(scratch, table, bra, ket, diff, p, q, r, s, lo, hi);
    if (q == r) value -= coeff1(bra, ket, diff, p, s);
    return value;
}

[[nodiscard]] inline double exchange_coeff(
    ElementScratch& scratch,
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q,
    double sub
) {
    if (diff.move.degree != 0) return 0.0;
    const int lo = std::min(p, q);
    const int hi = std::max(p, q);
    if (!diff.contains(lo, hi)) return 0.0;
    return prod2(scratch, table, bra, ket, diff, p, q, q, p, lo, hi) - sub;
}

[[nodiscard]] inline double single_exchange_coeff(
    ElementScratch& scratch,
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q,
    int k,
    double c1
) {
    const auto h = hull2(p, k, k, q);
    const int lo = h[0];
    const int hi = h[1];
    if (!diff.contains(lo, hi)) return 0.0;
    return prod2(scratch, table, bra, ket, diff, p, k, k, q, lo, hi) - c1;
}

[[nodiscard]] inline double double_coeff(
    ElementScratch& scratch,
    const Seg2Table& table,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    int p,
    int q,
    int r,
    int s
) {
    const auto h = hull2(p, q, r, s);
    const int lo = h[0];
    const int hi = h[1];
    if (!diff.contains(lo, hi)) return 0.0;
    return prod2(scratch, table, bra, ket, diff, p, q, r, s, lo, hi);
}

[[nodiscard]] inline double single_coulomb(
    const Integral& ints,
    const PathState& ket,
    int p,
    int q
) {
    double value = ints.h1(p, q) - ints.coulomb(p, q, q);
    for (int k : ket.donor) {
        value += static_cast<double>(ket.occ[static_cast<std::size_t>(k)])
            * ints.coulomb(p, q, k);
    }
    return value;
}

[[nodiscard]] inline double diag(
    ElementScratch& scratch,
    const Seg2Table& table,
    const Integral& ints,
    const PathState& ket
) {
    const PathDiff diff = path_diff(ket, ket, OccMove{});
    double value = ints.ecore();

    for (int p : ket.donor) {
        const int np = static_cast<int>(ket.occ[static_cast<std::size_t>(p)]);
        value += ints.hdiag(p) * static_cast<double>(np);
        value += 0.5 * ints.coulomb(p, p) * static_cast<double>(np * (np - 1));
    }

    for (std::size_t ip = 0; ip < ket.donor.size(); ++ip) {
        const int p = ket.donor[ip];
        const int np = static_cast<int>(ket.occ[static_cast<std::size_t>(p)]);
        for (std::size_t iq = ip + 1; iq < ket.donor.size(); ++iq) {
            const int q = ket.donor[iq];
            const int nq = static_cast<int>(ket.occ[static_cast<std::size_t>(q)]);
            value += ints.coulomb(p, q) * static_cast<double>(np * nq);
            value += ints.exchange(p, q)
                * exchange_coeff(scratch, table, ket, ket, diff, p, q, static_cast<double>(np));
        }
    }

    return value;
}

[[nodiscard]] inline double same_ocfg(
    ElementScratch& scratch,
    const Seg2Table& table,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff
) {
    if (diff.same_path()) return 0.0;

    double value = 0.0;
    for (std::size_t ip = 0; ip < ket.open.size(); ++ip) {
        const int p = ket.open[ip];
        if (p > diff.first) break;
        const int qmin = std::max(diff.last, p + 1);
        auto iq = std::lower_bound(ket.open.begin() + static_cast<std::ptrdiff_t>(ip + 1),
                                   ket.open.end(),
                                   qmin);
        for (; iq != ket.open.end(); ++iq) {
            const int q = *iq;
            value += ints.exchange(p, q)
                * exchange_coeff(scratch, table, bra, ket, diff, p, q, 0.0);
        }
    }
    return value;
}

[[nodiscard]] inline double single_move(
    ElementScratch& scratch,
    const Seg2Table& table,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    const OccMove& move,
    double gpq
) {
    const int n = ket.norb();
    const int p = move.add[0];
    const int q = move.remove[0];

    const double c1 = coeff1_move(bra, ket, diff, p, q);
    double value = c1 * gpq;
    for (int k = 0; k < n; ++k) {
        if (k == p || k == q) continue;
        value += ints.exchange(p, q, k)
            * single_exchange_coeff(scratch, table, bra, ket, diff, p, q, k, c1);
    }
    return value;
}

template <class Visit>
inline void visit_ordered_pair(int a, int b, Visit&& visit) {
    visit(a, b);
    if (a != b) visit(b, a);
}

[[nodiscard]] inline double double_move(
    ElementScratch& scratch,
    const Seg2Table& table,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket,
    const PathDiff& diff,
    const OccMove& move
) {
    double value = 0.0;
    visit_ordered_pair(move.add[0], move.add[1], [&](int p, int r) {
        visit_ordered_pair(move.remove[0], move.remove[1], [&](int q, int s) {
            value += 0.5 * ints.chem(p, q, r, s)
                * double_coeff(scratch, table, bra, ket, diff, p, q, r, s);
        });
    });
    return value;
}

[[nodiscard]] inline double hij(
    ElementScratch& scratch,
    const Seg2Table& table,
    const Integral& ints,
    const PathState& bra,
    const PathState& ket
) {
    const PathDiff diff = path_diff(ket, bra);
    if (bra.norb() != ket.norb() || diff.move.degree > 2) return 0.0;
    if (diff.move.degree == 0) {
        return diff.same_path()
            ? diag(scratch, table, ints, ket)
            : same_ocfg(scratch, table, ints, bra, ket, diff);
    }
    if (diff.move.degree == 1) {
        const int p = diff.move.add[0];
        const int q = diff.move.remove[0];
        return single_move(scratch, table, ints, bra, ket, diff, diff.move, single_coulomb(ints, ket, p, q));
    }
    return double_move(scratch, table, ints, bra, ket, diff, diff.move);
}

} // namespace libdet::guga
