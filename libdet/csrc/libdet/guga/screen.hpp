#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include <libdet/guga/csf.hpp>
#include <libdet/guga/integral.hpp>

namespace libdet::guga {

struct CfgChange {
    std::vector<int> ket_orbs;
    std::vector<int> bra_orbs;
};

struct BraCfg {
    Cfg cfg;
    CfgChange change;
    double bound = 0.0;
};

class Screen {
public:
    explicit Screen(const Integral& ints)
        : norb_(ints.norb()),
          global_bound_(build_global_bound(ints)),
          one_bound_(static_cast<std::size_t>(norb_ * norb_), 0.0),
          two_bound_(
              static_cast<std::size_t>(norb_ * norb_ * norb_ * norb_),
              0.0
          ) {
        build_one_bounds(ints);
        build_two_bounds(ints);
    }

    template <class Visit>
    void visit_bra_cfgs(const Csf& ket, double cutoff, Visit&& visit) const {
        std::vector<BraCfg> bras;
        build_bra_cfgs(ket.cfg, bras);

        for (const BraCfg& bra : bras) {
            if (cutoff <= 0.0 || bra.bound >= cutoff) visit(bra);
        }
    }

private:
    int norb_ = 0;
    double global_bound_ = 0.0;
    std::vector<double> one_bound_;
    std::vector<double> two_bound_;

    [[nodiscard]] std::size_t one_index(int ket_orb, int bra_orb) const noexcept {
        return static_cast<std::size_t>(ket_orb)
            * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(bra_orb);
    }

    [[nodiscard]] std::size_t two_index(
        int ket_a,
        int ket_b,
        int bra_a,
        int bra_b
    ) const noexcept {
        return (
            (
                static_cast<std::size_t>(ket_a)
                * static_cast<std::size_t>(norb_)
                + static_cast<std::size_t>(ket_b)
            ) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(bra_a)
        ) * static_cast<std::size_t>(norb_)
        + static_cast<std::size_t>(bra_b);
    }

    [[nodiscard]] static double build_global_bound(const Integral& ints) {
        double bound = std::abs(ints.ecore());
        const int norb = ints.norb();

        for (int p = 0; p < norb; ++p) {
            for (int q = 0; q < norb; ++q) bound += std::abs(ints.h1(p, q));
        }

        for (int p = 0; p < norb; ++p) {
            for (int q = 0; q < norb; ++q) {
                for (int r = 0; r < norb; ++r) {
                    for (int s = 0; s < norb; ++s) {
                        bound += std::abs(ints.chem(p, q, r, s));
                    }
                }
            }
        }

        return bound;
    }

    void build_one_bounds(const Integral& ints) {
        for (int q = 0; q < norb_; ++q) {
            for (int p = 0; p < norb_; ++p) {
                double bound = std::abs(ints.h1(q, p)) + std::abs(ints.h1(p, q));

                for (int t = 0; t < norb_; ++t) {
                    bound += std::abs(ints.chem(q, p, t, t));
                    bound += std::abs(ints.chem(q, t, t, p));
                    bound += std::abs(ints.chem(p, q, t, t));
                    bound += std::abs(ints.chem(p, t, t, q));
                    bound += std::abs(ints.chem(t, t, q, p));
                    bound += std::abs(ints.chem(t, q, p, t));
                }

                one_bound_[one_index(q, p)] = bound;
            }
        }
    }

    void build_two_bounds(const Integral& ints) {
        for (int q = 0; q < norb_; ++q) {
            for (int s = 0; s < norb_; ++s) {
                for (int p = 0; p < norb_; ++p) {
                    for (int r = 0; r < norb_; ++r) {
                        const int ket_orbs[2] = {q, s};
                        const int bra_orbs[2] = {p, r};
                        double bound = 0.0;

                        for (int i : ket_orbs) {
                            for (int j : ket_orbs) {
                                for (int a : bra_orbs) {
                                    for (int b : bra_orbs) {
                                        bound += std::abs(ints.chem(i, a, j, b));
                                        bound += std::abs(ints.chem(i, b, j, a));
                                        bound += std::abs(ints.chem(a, i, b, j));
                                        bound += std::abs(ints.chem(a, j, b, i));
                                    }
                                }
                            }
                        }

                        two_bound_[two_index(q, s, p, r)] = bound;
                    }
                }
            }
        }
    }

    [[nodiscard]] static CfgChange change(const Cfg& bra_cfg, const Cfg& ket_cfg) {
        CfgChange out;
        for (std::size_t p = 0; p < ket_cfg.size(); ++p) {
            const int diff =
                static_cast<int>(bra_cfg[p]) - static_cast<int>(ket_cfg[p]);
            for (int k = 0; k < -diff; ++k) {
                out.ket_orbs.push_back(static_cast<int>(p));
            }
            for (int k = 0; k < diff; ++k) {
                out.bra_orbs.push_back(static_cast<int>(p));
            }
        }
        return out;
    }

    [[nodiscard]] static double l1_bound(const Cfg& cfg) noexcept {
        int singly = 0;
        for (unsigned char occ : cfg) {
            if (occ == 1) ++singly;
        }
        return std::pow(2.0, 0.5 * static_cast<double>(singly));
    }

    [[nodiscard]] double bound(
        const Cfg& ket_cfg,
        const Cfg& bra_cfg,
        const CfgChange& change
    ) const noexcept {
        const double coeff_bound = l1_bound(ket_cfg) * l1_bound(bra_cfg);

        if (change.bra_orbs.empty()) return global_bound_;

        if (change.bra_orbs.size() == 1u) {
            const int q = change.ket_orbs[0];
            const int p = change.bra_orbs[0];
            return 64.0 * coeff_bound * one_bound_[one_index(q, p)];
        }

        if (change.bra_orbs.size() == 2u) {
            const int q = change.ket_orbs[0];
            const int s = change.ket_orbs[1];
            const int p = change.bra_orbs[0];
            const int r = change.bra_orbs[1];
            return 64.0 * coeff_bound * two_bound_[two_index(q, s, p, r)];
        }

        return 0.0;
    }

    [[nodiscard]] static bool move_one(
        Cfg& cfg,
        int ket_orb,
        int bra_orb
    ) noexcept {
        if (ket_orb == bra_orb) return true;

        auto& ket_occ = cfg[static_cast<std::size_t>(ket_orb)];
        auto& bra_occ = cfg[static_cast<std::size_t>(bra_orb)];
        if (ket_occ == 0 || bra_occ == 2) return false;

        --ket_occ;
        ++bra_occ;
        return true;
    }

    void add_bra_cfg(
        const Cfg& ket_cfg,
        Cfg bra_cfg,
        std::vector<BraCfg>& bras
    ) const {
        BraCfg bra;
        bra.change = change(bra_cfg, ket_cfg);
        bra.bound = bound(ket_cfg, bra_cfg, bra.change);
        bra.cfg = std::move(bra_cfg);
        bras.push_back(std::move(bra));
    }

    void build_bra_cfgs(const Cfg& ket_cfg, std::vector<BraCfg>& bras) const {
        bras.clear();
        bras.reserve(1u + static_cast<std::size_t>(norb_ * norb_));

        add_bra_cfg(ket_cfg, ket_cfg, bras);

        for (int q = 0; q < norb_; ++q) {
            if (ket_cfg[static_cast<std::size_t>(q)] == 0) continue;

            for (int p = 0; p < norb_; ++p) {
                if (p == q) continue;

                Cfg bra_cfg = ket_cfg;
                if (move_one(bra_cfg, q, p)) {
                    add_bra_cfg(ket_cfg, std::move(bra_cfg), bras);
                }
            }
        }

        for (int q = 0; q < norb_; ++q) {
            if (ket_cfg[static_cast<std::size_t>(q)] == 0) continue;

            for (int p = 0; p < norb_; ++p) {
                Cfg first = ket_cfg;
                if (!move_one(first, q, p)) continue;

                for (int s = 0; s < norb_; ++s) {
                    if (first[static_cast<std::size_t>(s)] == 0) continue;

                    for (int r = 0; r < norb_; ++r) {
                        if (r == s) continue;

                        Cfg bra_cfg = first;
                        if (move_one(bra_cfg, s, r)) {
                            add_bra_cfg(ket_cfg, std::move(bra_cfg), bras);
                        }
                    }
                }
            }
        }

        std::sort(
            bras.begin(),
            bras.end(),
            [](const BraCfg& lhs, const BraCfg& rhs) {
                return lhs.cfg < rhs.cfg;
            }
        );

        const auto last = std::unique(
            bras.begin(),
            bras.end(),
            [](const BraCfg& lhs, const BraCfg& rhs) {
                return lhs.cfg == rhs.cfg;
            }
        );
        bras.erase(last, bras.end());
    }
};

} // namespace libdet::guga
