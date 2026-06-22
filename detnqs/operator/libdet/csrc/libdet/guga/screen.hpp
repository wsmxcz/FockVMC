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

struct ScreenMove {
    OccMove move;
    double bound = 0.0;
};

class ScreenTable {
public:
    ScreenTable(const Integral& ints, double base_eps)
        : ints_(&ints), norb_(ints.norb()), base_eps_(base_eps) {
        if (std::isnan(base_eps_) || base_eps_ < 0.0) {
            throw std::invalid_argument("guga::ScreenTable: base_eps must be nonnegative");
        }
        build_moves();
    }

    [[nodiscard]] double base_eps() const noexcept { return base_eps_; }

    [[nodiscard]] std::span<const ScreenMove> singles(double eps) const noexcept {
        return prefix(singles_, std::max(eps, base_eps_));
    }

    [[nodiscard]] std::span<const ScreenMove> doubles(double eps) const noexcept {
        return prefix(doubles_, std::max(eps, base_eps_));
    }

    [[nodiscard]] double same_bound(Occ ket) const {
        if (static_cast<int>(ket.size()) != norb_) return 0.0;

        double out = 0.0;
        for (int p = 0; p < norb_; ++p) {
            if (nocc(ket, p) != 1) continue;
            for (int q = p + 1; q < norb_; ++q) {
                if (nocc(ket, q) != 1) continue;
                out += std::abs(ints_->exchange(p, q)) * coeff2_bound(ket, p, q, q, p);
            }
        }
        return pad(out);
    }

    [[nodiscard]] double same_bound(const PathState& ket) const {
        if (static_cast<int>(ket.occ.size()) != norb_) return 0.0;
        double out = 0.0;
        for (std::size_t ip = 0; ip < ket.open.size(); ++ip) {
            const int p = ket.open[ip];
            for (std::size_t iq = ip + 1u; iq < ket.open.size(); ++iq) {
                const int q = ket.open[iq];
                out += std::abs(ints_->exchange(p, q)) * coeff2_bound(ket.occ, p, q, q, p);
            }
        }
        return pad(out);
    }

    [[nodiscard]] bool keep(Occ ket, Occ bra) const {
        return bound(ket, bra) >= base_eps_;
    }

    [[nodiscard]] double bound(Occ ket, Occ bra) const {
        if (ket.size() != bra.size() || static_cast<int>(ket.size()) != norb_) return 0.0;

        const OccMove move = occ_move(ket, bra);
        if (move.degree > 2) return 0.0;
        if (move.degree == 0) return pad(diag_bound(ket) + same_bound(ket));
        return bound(ket, move);
    }

    [[nodiscard]] double bound(Occ ket, const OccMove& move) const {
        if (static_cast<int>(ket.size()) != norb_) return 0.0;
        if (move.degree == 1) return pad(single_bound(ket, move));
        if (move.degree == 2) return pad(double_bound(ket, move));
        return 0.0;
    }

    [[nodiscard]] double bound(const PathState& ket, const OccMove& move) const {
        return bound(ket.occ, move);
    }

private:
    const Integral* ints_ = nullptr;
    int norb_ = 0;
    double base_eps_ = 0.0;
    std::vector<ScreenMove> singles_;
    std::vector<ScreenMove> doubles_;

    [[nodiscard]] static int nocc(Occ occ, int p) noexcept {
        return static_cast<int>(occ[static_cast<std::size_t>(p)]);
    }

    [[nodiscard]] static double pad(double x) noexcept {
        return x * (1.0 + 128.0 * std::numeric_limits<double>::epsilon());
    }

    [[nodiscard]] static std::span<const ScreenMove> prefix(
        const std::vector<ScreenMove>& moves,
        double eps
    ) noexcept {
        const auto it = std::partition_point(
            moves.begin(),
            moves.end(),
            [eps](const ScreenMove& move) noexcept { return move.bound >= eps; }
        );
        return {moves.data(), static_cast<std::size_t>(it - moves.begin())};
    }

    [[nodiscard]] static double coeff1_bound(Occ occ, int p, int q) noexcept {
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

    [[nodiscard]] static double coeff2_bound(Occ occ, int p, int q, int r, int s) noexcept {
        int nr = 0;
        int ns = 0;
        const double first = coeff1_bound(occ, r, s);
        double prod = 0.0;
        if (first > 0.0 && first_ok(occ, r, s, nr, ns)) {
            const int np = after_first(occ, p, r, s, nr, ns);
            const int nq = after_first(occ, q, r, s, nr, ns);
            prod = first
                * (p == q ? static_cast<double>(nq) : (nq > 0 && np < 2 ? 2.0 : 0.0));
        }

        const double sub = (q == r) ? coeff1_bound(occ, p, s) : 0.0;
        return prod + sub;
    }

    [[nodiscard]] double diag_bound(Occ ket) const {
        double out = std::abs(ints_->ecore());
        for (int p = 0; p < norb_; ++p) {
            const int np = nocc(ket, p);
            out += std::abs(ints_->hdiag(p)) * static_cast<double>(np);
            out += 0.5 * std::abs(ints_->coulomb(p, p))
                * static_cast<double>(np * (np - 1));
        }

        for (int p = 0; p < norb_; ++p) {
            const int np = nocc(ket, p);
            if (np == 0) continue;
            for (int q = p + 1; q < norb_; ++q) {
                const int nq = nocc(ket, q);
                if (nq == 0) continue;
                out += std::abs(ints_->coulomb(p, q)) * static_cast<double>(np * nq);
                out += std::abs(ints_->exchange(p, q)) * coeff2_bound(ket, p, q, q, p);
            }
        }
        return out;
    }

    [[nodiscard]] static bool single_move_ok(Occ occ, int q, int p) noexcept {
        if (p == q) return false;
        return nocc(occ, q) > 0 && nocc(occ, p) < 2;
    }

    [[nodiscard]] static bool double_move_ok(Occ occ, const OccMove& move) noexcept {
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

        return remove(move.remove[0])
            && remove(move.remove[1])
            && add(move.add[0])
            && add(move.add[1]);
    }

    [[nodiscard]] double single_bound(Occ ket, const OccMove& move) const {
        const int p = move.add[0];
        const int q = move.remove[0];
        if (!single_move_ok(ket, q, p)) return 0.0;

        double out = std::abs(ints_->h1(p, q)) * coeff1_bound(ket, p, q);
        out += std::abs(ints_->coulomb(p, q, p)) * coeff2_bound(ket, p, p, p, q);
        out += std::abs(ints_->coulomb(p, q, q)) * coeff2_bound(ket, p, q, q, q);

        for (int k = 0; k < norb_; ++k) {
            if (k == p || k == q) continue;
            out += std::abs(ints_->coulomb(p, q, k)) * coeff2_bound(ket, p, q, k, k);
            out += std::abs(ints_->exchange(p, q, k)) * coeff2_bound(ket, p, k, k, q);
        }
        return out;
    }

    template <class Visit>
    static void visit_pair(int a, int b, Visit&& visit) {
        visit(a, b);
        if (a != b) visit(b, a);
    }

    [[nodiscard]] double double_bound(Occ ket, const OccMove& move) const {
        if (!double_move_ok(ket, move)) return 0.0;

        double out = 0.0;
        visit_pair(move.add[0], move.add[1], [&](int p, int r) {
            visit_pair(move.remove[0], move.remove[1], [&](int q, int s) {
                out += 0.5 * std::abs(ints_->chem(p, q, r, s)) * coeff2_bound(ket, p, q, r, s);
            });
        });
        return out;
    }

    [[nodiscard]] static double c2_static(int q, int r) noexcept {
        return q == r ? 6.0 : 4.0;
    }

    [[nodiscard]] double single_static_bound(int p, int q) const {
        double out = 2.0 * std::abs(ints_->h1(p, q));
        out += c2_static(p, p) * std::abs(ints_->coulomb(p, q, p));
        out += c2_static(q, q) * std::abs(ints_->coulomb(p, q, q));
        for (int k = 0; k < norb_; ++k) {
            if (k == p || k == q) continue;
            out += c2_static(q, k) * std::abs(ints_->coulomb(p, q, k));
            out += c2_static(k, k) * std::abs(ints_->exchange(p, q, k));
        }
        return pad(out);
    }

    [[nodiscard]] double double_static_bound(const OccMove& move) const {
        double out = 0.0;
        visit_pair(move.add[0], move.add[1], [&](int p, int r) {
            visit_pair(move.remove[0], move.remove[1], [&](int q, int s) {
                out += 0.5 * c2_static(q, r) * std::abs(ints_->chem(p, q, r, s));
            });
        });
        return pad(out);
    }

    static void sort_moves(std::vector<ScreenMove>& moves) {
        std::sort(moves.begin(), moves.end(), [](const ScreenMove& lhs, const ScreenMove& rhs) {
            if (lhs.bound != rhs.bound) return lhs.bound > rhs.bound;
            if (lhs.move.remove != rhs.move.remove) return lhs.move.remove < rhs.move.remove;
            return lhs.move.add < rhs.move.add;
        });
    }

    void build_moves() {
        singles_.clear();
        doubles_.clear();

        for (int q = 0; q < norb_; ++q) {
            for (int p = 0; p < norb_; ++p) {
                if (p == q) continue;
                ScreenMove item;
                item.move.degree = 1;
                item.move.remove = {q, -1};
                item.move.add = {p, -1};
                item.bound = single_static_bound(p, q);
                if (item.bound >= base_eps_) singles_.push_back(item);
            }
        }

        for (int q = 0; q < norb_; ++q) {
            for (int s = q; s < norb_; ++s) {
                for (int p = 0; p < norb_; ++p) {
                    for (int r = p; r < norb_; ++r) {
                        if (p == q || p == s || r == q || r == s) continue;
                        ScreenMove item;
                        item.move.degree = 2;
                        item.move.remove = {q, s};
                        item.move.add = {p, r};
                        item.bound = double_static_bound(item.move);
                        if (item.bound >= base_eps_) doubles_.push_back(item);
                    }
                }
            }
        }

        sort_moves(singles_);
        sort_moves(doubles_);
    }

};
} // namespace libdet::guga
