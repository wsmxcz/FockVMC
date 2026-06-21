#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <libdet/guga/path.hpp>
#include <libdet/integral.hpp>

namespace libdet::guga {

struct Move {
    unsigned char degree = 0;
    std::array<unsigned char, 2> remove{0u, 0u};
    std::array<unsigned char, 2> add{0u, 0u};
    double ub = 0.0;

    [[nodiscard]] static Move from(const OccMove& src) noexcept {
        Move out;
        out.degree = static_cast<unsigned char>(src.degree);
        if (src.degree <= 0) return out;
        out.remove[0] = static_cast<unsigned char>(src.remove[0]);
        out.add[0] = static_cast<unsigned char>(src.add[0]);
        if (src.degree == 2) {
            out.remove[1] = static_cast<unsigned char>(src.remove[1]);
            out.add[1] = static_cast<unsigned char>(src.add[1]);
        } else {
            out.remove[1] = out.remove[0];
            out.add[1] = out.add[0];
        }
        return out;
    }

    [[nodiscard]] OccMove occ_move() const noexcept {
        OccMove out;
        out.degree = static_cast<int>(degree);
        out.remove[0] = static_cast<int>(remove[0]);
        out.remove[1] = static_cast<int>(remove[1]);
        out.add[0] = static_cast<int>(add[0]);
        out.add[1] = static_cast<int>(add[1]);
        if (degree == 1u) {
            out.remove[1] = -1;
            out.add[1] = -1;
        }
        return out;
    }
};

// Occupation-level screen.  It gives safe bounds and cutoff-specific moves.
class Screen {
public:
    Screen(const Integral& ints, double cutoff)
        : ints_(&ints), norb_(ints.norb()), cutoff_(cutoff) {
        if (std::isnan(cutoff_) || cutoff_ < 0.0) {
            throw std::invalid_argument("guga::Screen: cutoff must be nonnegative");
        }
        build_moves();
    }

    [[nodiscard]] double cutoff() const noexcept { return cutoff_; }

    [[nodiscard]] std::span<const Move> one(double lo) const noexcept {
        return prefix(one_, std::max(lo, cutoff_));
    }

    [[nodiscard]] std::span<const Move> two(double lo) const noexcept {
        return prefix(two_, std::max(lo, cutoff_));
    }

    [[nodiscard]] double same(Occ ket) const {
        if (static_cast<int>(ket.size()) != norb_) return 0.0;

        double out = 0.0;
        for (int p = 0; p < norb_; ++p) {
            if (nocc(ket, p) != 1) continue;
            for (int q = p + 1; q < norb_; ++q) {
                if (nocc(ket, q) != 1) continue;
                out += std::abs(ints_->chem(p, q, q, p)) * two_bound(ket, p, q, q, p);
            }
        }
        return pad(out);
    }

    [[nodiscard]] bool keep(Occ ket, Occ bra) const {
        return bound(ket, bra) >= cutoff_;
    }

    [[nodiscard]] double bound(Occ ket, Occ bra) const {
        if (ket.size() != bra.size() || static_cast<int>(ket.size()) != norb_) return 0.0;

        const OccMove move = occ_move(bra, ket);
        if (move.degree > 2) return 0.0;
        if (move.degree == 0) return pad(diag_bound(ket) + same(ket));
        return bound(ket, Move::from(move));
    }

    [[nodiscard]] double bound(Occ ket, const Move& move) const {
        if (static_cast<int>(ket.size()) != norb_) return 0.0;
        if (move.degree == 1u) return pad(one_bound(ket, move));
        if (move.degree == 2u) return pad(two_move_bound(ket, move));
        return 0.0;
    }

private:
    const Integral* ints_ = nullptr;
    int norb_ = 0;
    double cutoff_ = 0.0;
    std::vector<Move> one_;
    std::vector<Move> two_;

    [[nodiscard]] static int nocc(Occ occ, int p) noexcept {
        return static_cast<int>(occ[static_cast<std::size_t>(p)]);
    }

    [[nodiscard]] static double pad(double x) noexcept {
        return x * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
    }

    [[nodiscard]] static std::span<const Move> prefix(
        const std::vector<Move>& moves,
        double lo
    ) noexcept {
        const auto it = std::partition_point(
            moves.begin(),
            moves.end(),
            [lo](const Move& move) noexcept { return move.ub >= lo; }
        );
        return {moves.data(), static_cast<std::size_t>(it - moves.begin())};
    }

    [[nodiscard]] static double one_coeff_bound(Occ occ, int p, int q) noexcept {
        if (p == q) return static_cast<double>(nocc(occ, p));
        return (nocc(occ, q) > 0 && nocc(occ, p) < 2) ? 2.0 : 0.0;
    }

    [[nodiscard]] static bool first_ok(Occ occ, int r, int s, int& nr, int& ns) noexcept {
        nr = nocc(occ, r);
        ns = nocc(occ, s);
        if (r == s) return ns > 0;
        if (ns <= 0 || nr >= 2) return false;
        ++nr;
        --ns;
        return true;
    }

    [[nodiscard]] static int after_first(Occ occ, int x, int r, int s, int nr, int ns) noexcept {
        if (x == r) return nr;
        if (x == s) return ns;
        return nocc(occ, x);
    }

    [[nodiscard]] static double two_bound(Occ occ, int p, int q, int r, int s) noexcept {
        int nr = 0;
        int ns = 0;
        const double first = one_coeff_bound(occ, r, s);
        double prod = 0.0;
        if (first > 0.0 && first_ok(occ, r, s, nr, ns)) {
            const int np = after_first(occ, p, r, s, nr, ns);
            const int nq = after_first(occ, q, r, s, nr, ns);
            prod = first
                * (p == q ? static_cast<double>(nq) : (nq > 0 && np < 2 ? 2.0 : 0.0));
        }

        const double sub = (q == r) ? one_coeff_bound(occ, p, s) : 0.0;
        return prod + sub;
    }

    [[nodiscard]] double diag_bound(Occ ket) const {
        double out = std::abs(ints_->ecore());
        for (int p = 0; p < norb_; ++p) {
            const int np = nocc(ket, p);
            out += std::abs(ints_->h1(p, p)) * static_cast<double>(np);
            out += 0.5 * std::abs(ints_->chem(p, p, p, p))
                * static_cast<double>(np * (np - 1));
        }

        for (int p = 0; p < norb_; ++p) {
            const int np = nocc(ket, p);
            if (np == 0) continue;
            for (int q = p + 1; q < norb_; ++q) {
                const int nq = nocc(ket, q);
                if (nq == 0) continue;
                out += std::abs(ints_->chem(p, p, q, q)) * static_cast<double>(np * nq);
                out += std::abs(ints_->chem(p, q, q, p)) * two_bound(ket, p, q, q, p);
            }
        }
        return out;
    }

    [[nodiscard]] static bool move_one_ok(Occ occ, int q, int p) noexcept {
        if (p == q) return false;
        return nocc(occ, q) > 0 && nocc(occ, p) < 2;
    }

    [[nodiscard]] static bool move_two_ok(Occ occ, const Move& move) noexcept {
        int idx[4];
        int val[4];
        int n = 0;

        auto pos = [&](int x) noexcept {
            for (int i = 0; i < n; ++i) {
                if (idx[i] == x) return i;
            }
            idx[n] = x;
            val[n] = nocc(occ, x);
            return n++;
        };
        auto remove = [&](int x) noexcept {
            const int i = pos(x);
            if (val[i] <= 0) return false;
            --val[i];
            return true;
        };
        auto add = [&](int x) noexcept {
            const int i = pos(x);
            if (val[i] >= 2) return false;
            ++val[i];
            return true;
        };

        return remove(static_cast<int>(move.remove[0]))
            && remove(static_cast<int>(move.remove[1]))
            && add(static_cast<int>(move.add[0]))
            && add(static_cast<int>(move.add[1]));
    }

    [[nodiscard]] double one_bound(Occ ket, const Move& move) const {
        const int p = static_cast<int>(move.add[0]);
        const int q = static_cast<int>(move.remove[0]);
        if (!move_one_ok(ket, q, p)) return 0.0;

        double out = std::abs(ints_->h1(p, q)) * one_coeff_bound(ket, p, q);
        out += std::abs(ints_->chem(p, p, p, q)) * two_bound(ket, p, p, p, q);
        out += std::abs(ints_->chem(p, q, q, q)) * two_bound(ket, p, q, q, q);

        for (int k = 0; k < norb_; ++k) {
            if (k == p || k == q) continue;
            out += std::abs(ints_->chem(p, q, k, k)) * two_bound(ket, p, q, k, k);
            out += std::abs(ints_->chem(p, k, k, q)) * two_bound(ket, p, k, k, q);
        }
        return out;
    }

    template <class Visit>
    static void visit_pair(int a, int b, Visit&& visit) {
        visit(a, b);
        if (a != b) visit(b, a);
    }

    [[nodiscard]] double two_move_bound(Occ ket, const Move& move) const {
        if (!move_two_ok(ket, move)) return 0.0;

        double out = 0.0;
        visit_pair(static_cast<int>(move.add[0]), static_cast<int>(move.add[1]), [&](int p, int r) {
            visit_pair(static_cast<int>(move.remove[0]), static_cast<int>(move.remove[1]), [&](int q, int s) {
                out += 0.5 * std::abs(ints_->chem(p, q, r, s)) * two_bound(ket, p, q, r, s);
            });
        });
        return out;
    }

    [[nodiscard]] static double c2_static(int q, int r) noexcept {
        return q == r ? 6.0 : 4.0;
    }

    [[nodiscard]] double one_static(int p, int q) const {
        double out = 2.0 * std::abs(ints_->h1(p, q));
        out += c2_static(p, p) * std::abs(ints_->chem(p, p, p, q));
        out += c2_static(q, q) * std::abs(ints_->chem(p, q, q, q));
        for (int k = 0; k < norb_; ++k) {
            if (k == p || k == q) continue;
            out += c2_static(q, k) * std::abs(ints_->chem(p, q, k, k));
            out += c2_static(k, k) * std::abs(ints_->chem(p, k, k, q));
        }
        return pad(out);
    }

    [[nodiscard]] double two_static(const Move& move) const {
        double out = 0.0;
        visit_pair(static_cast<int>(move.add[0]), static_cast<int>(move.add[1]), [&](int p, int r) {
            visit_pair(static_cast<int>(move.remove[0]), static_cast<int>(move.remove[1]), [&](int q, int s) {
                out += 0.5 * c2_static(q, r) * std::abs(ints_->chem(p, q, r, s));
            });
        });
        return pad(out);
    }

    static void sort_moves(std::vector<Move>& moves) {
        std::sort(moves.begin(), moves.end(), [](const Move& lhs, const Move& rhs) {
            if (lhs.ub != rhs.ub) return lhs.ub > rhs.ub;
            if (lhs.degree != rhs.degree) return lhs.degree < rhs.degree;
            if (lhs.remove != rhs.remove) return lhs.remove < rhs.remove;
            return lhs.add < rhs.add;
        });
    }

    void build_moves() {
        one_.clear();
        two_.clear();

        for (int q = 0; q < norb_; ++q) {
            for (int p = 0; p < norb_; ++p) {
                if (p == q) continue;
                Move move;
                move.degree = 1u;
                move.remove = {static_cast<unsigned char>(q), static_cast<unsigned char>(q)};
                move.add = {static_cast<unsigned char>(p), static_cast<unsigned char>(p)};
                move.ub = one_static(p, q);
                if (move.ub >= cutoff_) one_.push_back(move);
            }
        }

        for (int q = 0; q < norb_; ++q) {
            for (int s = q; s < norb_; ++s) {
                for (int p = 0; p < norb_; ++p) {
                    for (int r = p; r < norb_; ++r) {
                        if (p == q || p == s || r == q || r == s) continue;

                        Move move;
                        move.degree = 2u;
                        move.remove = {
                            static_cast<unsigned char>(q),
                            static_cast<unsigned char>(s)
                        };
                        move.add = {
                            static_cast<unsigned char>(p),
                            static_cast<unsigned char>(r)
                        };
                        move.ub = two_static(move);
                        if (move.ub >= cutoff_) two_.push_back(move);
                    }
                }
            }
        }

        sort_moves(one_);
        sort_moves(two_);
    }
};

} // namespace libdet::guga
