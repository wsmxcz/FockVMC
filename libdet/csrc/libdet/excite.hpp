#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <libdet/det.hpp>
#include <libdet/heatbath.hpp>
#include <libdet/slater.hpp>

namespace libdet {

/*
 * Excitation-driven free-neighborhood enumeration.
 *
 * This file answers the question:
 *
 *   Given a source determinant ket, which unrestricted determinants are
 *   connected by single or double excitations?
 *
 * It is used by edges, degrees, expand, and stochastic edge sampling.
 */

struct RowWork {
    explicit RowWork(u32 nword, int norb)
        : occ(nword, norb), norb(norb) {}

    void resize(u32 nword, int n) {
        norb = n;
        occ.resize(nword, n);
    }

    DetOcc occ;
    int norb = 0;
};

[[nodiscard]] inline bool keep_h(double h, double cut) noexcept {
    const double a = std::abs(h);
    return a > 0.0 && a >= cut;
}

[[nodiscard]] inline bool in_window(double h, double lo, double hi) noexcept {
    const double a = std::abs(h);
    return a > 0.0 && a >= lo && a < hi;
}

/* ---------- value-only scans ---------- */

template <class Emit>
inline void singles_alpha_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, cut)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_beta_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, cut)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_alpha_window_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double lo,
    double hi,
    Emit&& emit
) {
    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, lo, hi)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_beta_window_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double lo,
    double hi,
    Emit&& emit
) {
    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, lo, hi)) emit(h);
        }
    }
}

template <class Emit>
inline void doubles_exact_aa_values(
    const RHFIntegrals& ints,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (std::size_t p = 0; p < occ.vir_a.size(); ++p) {
                const int a = occ.vir_a[p];

                for (std::size_t q = p + 1u; q < occ.vir_a.size(); ++q) {
                    const int b = occ.vir_a[q];

                    const double h =
                        Slater::sign_double(occ.pref_a, i, j, a, b)
                        * Slater::double_aa(ints, i, j, a, b);

                    if (keep_h(h, cut)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_bb_values(
    const RHFIntegrals& ints,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (std::size_t p = 0; p < occ.vir_b.size(); ++p) {
                const int a = occ.vir_b[p];

                for (std::size_t q = p + 1u; q < occ.vir_b.size(); ++q) {
                    const int b = occ.vir_b[q];

                    const double h =
                        Slater::sign_double(occ.pref_b, i, j, a, b)
                        * Slater::double_bb(ints, i, j, a, b);

                    if (keep_h(h, cut)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_ab_values(
    const RHFIntegrals& ints,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (int ia : occ.occ_a) {
        for (int aa : occ.vir_a) {
            const double sa = Slater::sign_single(occ.pref_a, ia, aa);

            for (int ib : occ.occ_b) {
                for (int ab : occ.vir_b) {
                    const double h =
                        sa
                        * Slater::sign_single(occ.pref_b, ib, ab)
                        * Slater::double_ab(ints, ia, ib, aa, ab);

                    if (keep_h(h, cut)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_hb_aa_values(
    const HeatBathTable& hb,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const auto& c : hb.aa_ge(i, j, cut)) {
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.p, c.q) * c.h;

                if (keep_h(h, cut)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_hb_bb_values(
    const HeatBathTable& hb,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const auto& c : hb.bb_ge(i, j, cut)) {
                if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.p, c.q) * c.h;

                if (keep_h(h, cut)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_hb_ab_values(
    const HeatBathTable& hb,
    DetRef ket,
    const DetOcc& occ,
    double cut,
    Emit&& emit
) {
    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const auto& c : hb.ab_ge(ia, ib, cut)) {
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) {
                    continue;
                }

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.p)
                    * Slater::sign_single(occ.pref_b, ib, c.q)
                    * c.h;

                if (keep_h(h, cut)) emit(h);
            }
        }
    }
}

/* ---------- determinant-emitting scans ---------- */

template <class Emit>
inline void singles_alpha_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (int i : occ.occ_a) {
        bits::clear(alpha, i);

        for (int a : occ.vir_a) {
            bits::set(alpha, a);

            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, cut)) emit(occ.det.view(), h);

            bits::clear(alpha, a);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void singles_beta_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (int i : occ.occ_b) {
        bits::clear(beta, i);

        for (int a : occ.vir_b) {
            bits::set(beta, a);

            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, cut)) emit(occ.det.view(), h);

            bits::clear(beta, a);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void singles_alpha_window_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double lo,
    double hi,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (int i : occ.occ_a) {
        bits::clear(alpha, i);

        for (int a : occ.vir_a) {
            bits::set(alpha, a);

            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, lo, hi)) emit(occ.det.view(), h);

            bits::clear(alpha, a);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void singles_beta_window_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double lo,
    double hi,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (int i : occ.occ_b) {
        bits::clear(beta, i);

        for (int a : occ.vir_b) {
            bits::set(beta, a);

            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, lo, hi)) emit(occ.det.view(), h);

            bits::clear(beta, a);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_exact_aa_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];
        bits::clear(alpha, i);

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];
            bits::clear(alpha, j);

            for (std::size_t p = 0; p < occ.vir_a.size(); ++p) {
                const int a = occ.vir_a[p];
                bits::set(alpha, a);

                for (std::size_t q = p + 1u; q < occ.vir_a.size(); ++q) {
                    const int b = occ.vir_a[q];
                    bits::set(alpha, b);

                    const double h =
                        Slater::sign_double(occ.pref_a, i, j, a, b)
                        * Slater::double_aa(ints, i, j, a, b);

                    if (keep_h(h, cut)) emit(occ.det.view(), h);

                    bits::clear(alpha, b);
                }

                bits::clear(alpha, a);
            }

            bits::set(alpha, j);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void doubles_exact_bb_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];
        bits::clear(beta, i);

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];
            bits::clear(beta, j);

            for (std::size_t p = 0; p < occ.vir_b.size(); ++p) {
                const int a = occ.vir_b[p];
                bits::set(beta, a);

                for (std::size_t q = p + 1u; q < occ.vir_b.size(); ++q) {
                    const int b = occ.vir_b[q];
                    bits::set(beta, b);

                    const double h =
                        Slater::sign_double(occ.pref_b, i, j, a, b)
                        * Slater::double_bb(ints, i, j, a, b);

                    if (keep_h(h, cut)) emit(occ.det.view(), h);

                    bits::clear(beta, b);
                }

                bits::clear(beta, a);
            }

            bits::set(beta, j);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_exact_ab_edges(
    const RHFIntegrals& ints,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int ia : occ.occ_a) {
        bits::clear(alpha, ia);

        for (int aa : occ.vir_a) {
            bits::set(alpha, aa);

            const double sa = Slater::sign_single(occ.pref_a, ia, aa);

            for (int ib : occ.occ_b) {
                bits::clear(beta, ib);

                for (int ab : occ.vir_b) {
                    bits::set(beta, ab);

                    const double h =
                        sa
                        * Slater::sign_single(occ.pref_b, ib, ab)
                        * Slater::double_ab(ints, ia, ib, aa, ab);

                    if (keep_h(h, cut)) emit(occ.det.view(), h);

                    bits::clear(beta, ab);
                }

                bits::set(beta, ib);
            }

            bits::clear(alpha, aa);
        }

        bits::set(alpha, ia);
    }
}

template <class Emit>
inline void doubles_hb_aa_edges(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];
        bits::clear(alpha, i);

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];
            bits::clear(alpha, j);

            for (const auto& c : hb.aa_ge(i, j, cut)) {
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) {
                    continue;
                }

                bits::set(alpha, c.p);
                bits::set(alpha, c.q);

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.p, c.q) * c.h;

                if (keep_h(h, cut)) emit(occ.det.view(), h);

                bits::clear(alpha, c.q);
                bits::clear(alpha, c.p);
            }

            bits::set(alpha, j);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void doubles_hb_bb_edges(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];
        bits::clear(beta, i);

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];
            bits::clear(beta, j);

            for (const auto& c : hb.bb_ge(i, j, cut)) {
                if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) {
                    continue;
                }

                bits::set(beta, c.p);
                bits::set(beta, c.q);

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.p, c.q) * c.h;

                if (keep_h(h, cut)) emit(occ.det.view(), h);

                bits::clear(beta, c.q);
                bits::clear(beta, c.p);
            }

            bits::set(beta, j);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_hb_ab_edges(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double cut,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int ia : occ.occ_a) {
        bits::clear(alpha, ia);

        for (int ib : occ.occ_b) {
            bits::clear(beta, ib);

            for (const auto& c : hb.ab_ge(ia, ib, cut)) {
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) {
                    continue;
                }

                bits::set(alpha, c.p);
                bits::set(beta, c.q);

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.p)
                    * Slater::sign_single(occ.pref_b, ib, c.q)
                    * c.h;

                if (keep_h(h, cut)) emit(occ.det.view(), h);

                bits::clear(beta, c.q);
                bits::clear(alpha, c.p);
            }

            bits::set(beta, ib);
        }

        bits::set(alpha, ia);
    }
}

/* ---------- public scan functions ---------- */

template <class Emit>
inline void scan_values(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    RowWork& work,
    double cut,
    Emit&& emit
) {
    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_values(ints, work.occ, cut, emit);
    singles_beta_values(ints, work.occ, cut, emit);

    if (cut >= eps_hb) {
        doubles_hb_aa_values(hb, ket, work.occ, cut, emit);
        doubles_hb_bb_values(hb, ket, work.occ, cut, emit);
        doubles_hb_ab_values(hb, ket, work.occ, cut, emit);
    } else {
        doubles_exact_aa_values(ints, ket, work.occ, cut, emit);
        doubles_exact_bb_values(ints, ket, work.occ, cut, emit);
        doubles_exact_ab_values(ints, ket, work.occ, cut, emit);
    }
}

template <class Emit>
inline void scan_edges(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    RowWork& work,
    double cut,
    Emit&& emit
) {
    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_edges(ints, ket, work.occ, cut, emit);
    singles_beta_edges(ints, ket, work.occ, cut, emit);

    if (cut >= eps_hb) {
        doubles_hb_aa_edges(hb, ket, work.occ, cut, emit);
        doubles_hb_bb_edges(hb, ket, work.occ, cut, emit);
        doubles_hb_ab_edges(hb, ket, work.occ, cut, emit);
    } else {
        doubles_exact_aa_edges(ints, ket, work.occ, cut, emit);
        doubles_exact_bb_edges(ints, ket, work.occ, cut, emit);
        doubles_exact_ab_edges(ints, ket, work.occ, cut, emit);
    }
}

template <class Emit>
inline void scan_window_values(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    RowWork& work,
    double lo,
    double hi,
    Emit&& emit
) {
    if (hi <= lo || hi <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_window_values(ints, work.occ, lo, hi, emit);
    singles_beta_window_values(ints, work.occ, lo, hi, emit);

    if (lo >= eps_hb) {
        for (std::size_t x = 0; x < work.occ.occ_a.size(); ++x) {
            const int i = work.occ.occ_a[x];

            for (std::size_t y = x + 1u; y < work.occ.occ_a.size(); ++y) {
                const int j = work.occ.occ_a[y];

                for (const auto& c : hb.aa_window(i, j, lo, hi)) {
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) continue;

                    const double h =
                        Slater::sign_double(work.occ.pref_a, i, j, c.p, c.q) * c.h;

                    emit(h);
                }
            }
        }

        for (std::size_t x = 0; x < work.occ.occ_b.size(); ++x) {
            const int i = work.occ.occ_b[x];

            for (std::size_t y = x + 1u; y < work.occ.occ_b.size(); ++y) {
                const int j = work.occ.occ_b[y];

                for (const auto& c : hb.bb_window(i, j, lo, hi)) {
                    if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) continue;

                    const double h =
                        Slater::sign_double(work.occ.pref_b, i, j, c.p, c.q) * c.h;

                    emit(h);
                }
            }
        }

        for (int ia : work.occ.occ_a) {
            for (int ib : work.occ.occ_b) {
                for (const auto& c : hb.ab_window(ia, ib, lo, hi)) {
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) continue;

                    const double h =
                        Slater::sign_single(work.occ.pref_a, ia, c.p)
                        * Slater::sign_single(work.occ.pref_b, ib, c.q)
                        * c.h;

                    emit(h);
                }
            }
        }
    } else {
        doubles_exact_aa_values(ints, ket, work.occ, 0.0, [&](double h) {
            if (in_window(h, lo, hi)) emit(h);
        });

        doubles_exact_bb_values(ints, ket, work.occ, 0.0, [&](double h) {
            if (in_window(h, lo, hi)) emit(h);
        });

        doubles_exact_ab_values(ints, ket, work.occ, 0.0, [&](double h) {
            if (in_window(h, lo, hi)) emit(h);
        });
    }
}

template <class Emit>
inline void scan_window_edges(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    RowWork& work,
    double lo,
    double hi,
    Emit&& emit
) {
    if (hi <= lo || hi <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_window_edges(ints, ket, work.occ, lo, hi, emit);
    singles_beta_window_edges(ints, ket, work.occ, lo, hi, emit);

    if (lo >= eps_hb) {
        auto alpha = work.occ.det.alpha();
        auto beta = work.occ.det.beta();

        for (std::size_t x = 0; x < work.occ.occ_a.size(); ++x) {
            const int i = work.occ.occ_a[x];
            bits::clear(alpha, i);

            for (std::size_t y = x + 1u; y < work.occ.occ_a.size(); ++y) {
                const int j = work.occ.occ_a[y];
                bits::clear(alpha, j);

                for (const auto& c : hb.aa_window(i, j, lo, hi)) {
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) continue;

                    bits::set(alpha, c.p);
                    bits::set(alpha, c.q);

                    const double h =
                        Slater::sign_double(work.occ.pref_a, i, j, c.p, c.q) * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(alpha, c.q);
                    bits::clear(alpha, c.p);
                }

                bits::set(alpha, j);
            }

            bits::set(alpha, i);
        }

        for (std::size_t x = 0; x < work.occ.occ_b.size(); ++x) {
            const int i = work.occ.occ_b[x];
            bits::clear(beta, i);

            for (std::size_t y = x + 1u; y < work.occ.occ_b.size(); ++y) {
                const int j = work.occ.occ_b[y];
                bits::clear(beta, j);

                for (const auto& c : hb.bb_window(i, j, lo, hi)) {
                    if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) continue;

                    bits::set(beta, c.p);
                    bits::set(beta, c.q);

                    const double h =
                        Slater::sign_double(work.occ.pref_b, i, j, c.p, c.q) * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(beta, c.q);
                    bits::clear(beta, c.p);
                }

                bits::set(beta, j);
            }

            bits::set(beta, i);
        }

        for (int ia : work.occ.occ_a) {
            bits::clear(alpha, ia);

            for (int ib : work.occ.occ_b) {
                bits::clear(beta, ib);

                for (const auto& c : hb.ab_window(ia, ib, lo, hi)) {
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) continue;

                    bits::set(alpha, c.p);
                    bits::set(beta, c.q);

                    const double h =
                        Slater::sign_single(work.occ.pref_a, ia, c.p)
                        * Slater::sign_single(work.occ.pref_b, ib, c.q)
                        * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(beta, c.q);
                    bits::clear(alpha, c.p);
                }

                bits::set(beta, ib);
            }

            bits::set(alpha, ia);
        }
    } else {
        doubles_exact_aa_edges(ints, ket, work.occ, 0.0, [&](DetRef det, double h) {
            if (in_window(h, lo, hi)) emit(det, h);
        });

        doubles_exact_bb_edges(ints, ket, work.occ, 0.0, [&](DetRef det, double h) {
            if (in_window(h, lo, hi)) emit(det, h);
        });

        doubles_exact_ab_edges(ints, ket, work.occ, 0.0, [&](DetRef det, double h) {
            if (in_window(h, lo, hi)) emit(det, h);
        });
    }
}

} // namespace libdet