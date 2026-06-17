#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <libdet/guga/csf.hpp>
#include <libdet/guga/integral.hpp>

namespace libdet::guga {

struct SpinTerm {
    std::vector<Step> occ;
    int m = 0;
    double coeff = 1.0;
};

[[nodiscard]] inline bool has_alpha(Step step) noexcept {
    return step == Step::upper || step == Step::doubly;
}

[[nodiscard]] inline bool has_beta(Step step) noexcept {
    return step == Step::lower || step == Step::doubly;
}

[[nodiscard]] inline double spin_phase(const SpinTerm& term) noexcept {
    int parity = 0;
    const int norb = static_cast<int>(term.occ.size());
    for (int p = 0; p < norb; ++p) {
        if (!has_beta(term.occ[static_cast<std::size_t>(p)])) continue;
        for (int q = p + 1; q < norb; ++q) {
            if (has_alpha(term.occ[static_cast<std::size_t>(q)])) parity ^= 1;
        }
    }
    return parity ? -1.0 : 1.0;
}

inline void fill_spin_prefix(
    const SpinTerm& term,
    bool alpha,
    std::vector<int>& out
) {
    const int norb = static_cast<int>(term.occ.size());
    out.assign(static_cast<std::size_t>(norb + 1), 0);
    for (int p = 0; p < norb; ++p) {
        const Step step = term.occ[static_cast<std::size_t>(p)];
        const bool occupied = alpha ? has_alpha(step) : has_beta(step);
        out[static_cast<std::size_t>(p + 1)] =
            out[static_cast<std::size_t>(p)] + (occupied ? 1 : 0);
    }
}

inline void fill_spin_occ(
    const SpinTerm& term,
    bool alpha,
    std::vector<int>& out
) {
    out.clear();
    const int norb = static_cast<int>(term.occ.size());
    for (int p = 0; p < norb; ++p) {
        const Step step = term.occ[static_cast<std::size_t>(p)];
        if (alpha ? has_alpha(step) : has_beta(step)) out.push_back(p);
    }
}

struct TermDiff {
    int deg = 0;
    int na = 0;
    int nb = 0;

    std::array<int, 2> occ_a{0, 0};
    std::array<int, 2> vir_a{0, 0};
    std::array<int, 2> occ_b{0, 0};
    std::array<int, 2> vir_b{0, 0};
};

[[nodiscard]] inline TermDiff term_diff(
    const SpinTerm& bra,
    const SpinTerm& ket
) noexcept {
    TermDiff ex;
    const int norb = static_cast<int>(bra.occ.size());

    for (int p = 0; p < norb; ++p) {
        const Step b = bra.occ[static_cast<std::size_t>(p)];
        const Step k = ket.occ[static_cast<std::size_t>(p)];

        const bool ba = has_alpha(b);
        const bool ka = has_alpha(k);
        if (ba && !ka) {
            if (ex.na >= 2) {
                ex.deg = 3;
                return ex;
            }
            ex.occ_a[ex.na++] = p;
        }

        const bool bb = has_beta(b);
        const bool kb = has_beta(k);
        if (bb && !kb) {
            if (ex.nb >= 2) {
                ex.deg = 3;
                return ex;
            }
            ex.occ_b[ex.nb++] = p;
        }
    }

    int va = 0;
    int vb = 0;
    for (int p = 0; p < norb; ++p) {
        const Step b = bra.occ[static_cast<std::size_t>(p)];
        const Step k = ket.occ[static_cast<std::size_t>(p)];

        const bool ba = has_alpha(b);
        const bool ka = has_alpha(k);
        if (!ba && ka) {
            if (va >= 2) {
                ex.deg = 3;
                return ex;
            }
            ex.vir_a[va++] = p;
        }

        const bool bb = has_beta(b);
        const bool kb = has_beta(k);
        if (!bb && kb) {
            if (vb >= 2) {
                ex.deg = 3;
                return ex;
            }
            ex.vir_b[vb++] = p;
        }
    }

    if (va != ex.na || vb != ex.nb) {
        ex.deg = 3;
        return ex;
    }

    ex.deg = ex.na + ex.nb;
    return ex;
}

[[nodiscard]] inline double diag_value(
    const Integral& ints,
    const SpinTerm& term
) {
    std::vector<int> occ_a;
    std::vector<int> occ_b;
    fill_spin_occ(term, true, occ_a);
    fill_spin_occ(term, false, occ_b);

    double energy = ints.ecore();
    for (int i : occ_a) energy += ints.h1(i, i);
    for (int i : occ_b) energy += ints.h1(i, i);

    for (std::size_t x = 0; x < occ_a.size(); ++x) {
        const int i = occ_a[x];
        for (std::size_t y = x + 1u; y < occ_a.size(); ++y) {
            const int j = occ_a[y];
            energy += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
        }
    }

    for (std::size_t x = 0; x < occ_b.size(); ++x) {
        const int i = occ_b[x];
        for (std::size_t y = x + 1u; y < occ_b.size(); ++y) {
            const int j = occ_b[y];
            energy += ints.chem(i, i, j, j) - ints.chem(i, j, j, i);
        }
    }

    for (int i : occ_a) {
        for (int j : occ_b) energy += ints.chem(i, i, j, j);
    }

    return energy;
}

[[nodiscard]] inline double single_value(
    const Integral& ints,
    const SpinTerm& bra,
    bool alpha,
    int i,
    int a
) {
    std::vector<int> occ_same;
    std::vector<int> occ_other;
    fill_spin_occ(bra, alpha, occ_same);
    fill_spin_occ(bra, !alpha, occ_other);

    double value = ints.h1(i, a);
    for (int j : occ_same) {
        if (j != i) {
            value += ints.chem(i, a, j, j) - ints.chem(i, j, j, a);
        }
    }
    for (int j : occ_other) value += ints.chem(i, a, j, j);
    return value;
}

[[nodiscard]] inline double term_element(
    const Integral& ints,
    const SpinTerm& bra,
    const SpinTerm& ket
) {
    const TermDiff ex = term_diff(bra, ket);
    if (ex.deg > 2) return 0.0;
    if (ex.deg == 0) return diag_value(ints, bra);

    std::vector<int> pref_a;
    std::vector<int> pref_b;
    fill_spin_prefix(bra, true, pref_a);
    fill_spin_prefix(bra, false, pref_b);

    if (ex.deg == 1) {
        if (ex.na == 1) {
            return libdet::detail::sign_single(pref_a, ex.occ_a[0], ex.vir_a[0])
                * single_value(ints, bra, true, ex.occ_a[0], ex.vir_a[0]);
        }

        return libdet::detail::sign_single(pref_b, ex.occ_b[0], ex.vir_b[0])
            * single_value(ints, bra, false, ex.occ_b[0], ex.vir_b[0]);
    }

    if (ex.na == 2) {
        return libdet::detail::sign_double(
            pref_a,
            ex.occ_a[0],
            ex.occ_a[1],
            ex.vir_a[0],
            ex.vir_a[1]
        ) * (ints.chem(ex.occ_a[0], ex.vir_a[0], ex.occ_a[1], ex.vir_a[1])
             - ints.chem(ex.occ_a[0], ex.vir_a[1], ex.occ_a[1], ex.vir_a[0]));
    }

    if (ex.nb == 2) {
        return libdet::detail::sign_double(
            pref_b,
            ex.occ_b[0],
            ex.occ_b[1],
            ex.vir_b[0],
            ex.vir_b[1]
        ) * (ints.chem(ex.occ_b[0], ex.vir_b[0], ex.occ_b[1], ex.vir_b[1])
             - ints.chem(ex.occ_b[0], ex.vir_b[1], ex.occ_b[1], ex.vir_b[0]));
    }

    return libdet::detail::sign_single(pref_a, ex.occ_a[0], ex.vir_a[0])
        * libdet::detail::sign_single(pref_b, ex.occ_b[0], ex.vir_b[0])
        * ints.chem(ex.occ_a[0], ex.vir_a[0], ex.occ_b[0], ex.vir_b[0]);
}

[[nodiscard]] inline double cg_coeff(
    int spin_before,
    int m_before,
    int electron_spin,
    int spin_after
) noexcept {
    const int b = spin_before;
    const int m = m_before;
    const double denom = static_cast<double>(2 * (b + 1));

    if (spin_after == b + 1) {
        const int num = electron_spin > 0 ? b + m + 2 : b - m + 2;
        return num <= 0 ? 0.0 : std::sqrt(static_cast<double>(num) / denom);
    }

    if (spin_after == b - 1) {
        const int num = electron_spin > 0 ? b - m : b + m;
        if (num <= 0) return 0.0;

        const double value = std::sqrt(static_cast<double>(num) / denom);
        return electron_spin > 0 ? -value : value;
    }

    return 0.0;
}

template <class Visit>
inline void visit_coupled_terms(const Csf& csf, Visit&& visit) {
    std::vector<SpinTerm> current;
    current.push_back({
        std::vector<Step>(static_cast<std::size_t>(csf.norb()), Step::empty),
        0,
        1.0
    });

    const int norb = csf.norb();
    for (int p = 0; p < norb; ++p) {
        const Step step = csf.step[static_cast<std::size_t>(p)];
        const int spin_before = csf.spin[static_cast<std::size_t>(p)];
        const int spin_after = csf.spin[static_cast<std::size_t>(p + 1)];

        if (step == Step::empty || step == Step::doubly) {
            for (auto& item : current) {
                item.occ[static_cast<std::size_t>(p)] = step;
            }
            continue;
        }

        std::vector<SpinTerm> next;
        next.reserve(current.size() * 2u);
        for (const SpinTerm& item : current) {
            for (int sigma : {1, -1}) {
                const double coeff =
                    cg_coeff(spin_before, item.m, sigma, spin_after);
                if (coeff == 0.0) continue;

                SpinTerm out = item;
                out.occ[static_cast<std::size_t>(p)] =
                    sigma > 0 ? Step::upper : Step::lower;
                out.coeff *= coeff;
                out.m += sigma;
                next.push_back(std::move(out));
            }
        }
        current.swap(next);
    }

    const int target_m = csf.spin_twice();
    for (SpinTerm& item : current) {
        if (item.m != target_m || item.coeff == 0.0) continue;
        item.coeff *= spin_phase(item);
        visit(item);
    }
}

[[nodiscard]] inline double element(
    const Integral& ints,
    const Csf& bra,
    const Csf& ket,
    u32
) {
    if (cfg_degree(bra.cfg, ket.cfg) > 2) return 0.0;

    double value = 0.0;
    visit_coupled_terms(bra, [&](const SpinTerm& bra_term) {
        visit_coupled_terms(ket, [&](const SpinTerm& ket_term) {
            value += bra_term.coeff
                * ket_term.coeff
                * term_element(ints, bra_term, ket_term);
        });
    });
    return value;
}

} // namespace libdet::guga
