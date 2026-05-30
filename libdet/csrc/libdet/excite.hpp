#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include <libdet/det.hpp>
#include <libdet/heatbath.hpp>
#include <libdet/slater.hpp>

namespace libdet {

/*
 * Excitation-driven free-bra enumeration.
 *
 * Given a source ket, generate unrestricted connected bras by single and
 * double excitations.
 *
 * Used by conns, degrees, expand, sample_conns, and sample_project.
 */

struct KetWork {
    explicit KetWork(u32 nword, int norb)
        : occ(nword, norb), norb(norb) {}

    void resize(u32 nword, int n) {
        norb = n;
        occ.resize(nword, n);
    }

    DetOcc occ;
    int norb = 0;
};

[[nodiscard]] inline bool keep_h(double h, double eps) noexcept {
    const double abs_h = std::abs(h);
    return abs_h > 0.0 && abs_h >= eps;
}

[[nodiscard]] inline bool in_window(double h, double eps2, double eps1) noexcept {
    const double abs_h = std::abs(h);
    return abs_h > 0.0 && abs_h >= eps2 && abs_h < eps1;
}

/* ---------- value-only scans ---------- */

template <class Emit>
inline void singles_alpha_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, eps)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_beta_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_h(h, eps)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_alpha_window_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps2,
    double eps1,
    Emit&& emit
) {
    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, eps2, eps1)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_beta_window_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps2,
    double eps1,
    Emit&& emit
) {
    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (in_window(h, eps2, eps1)) emit(h);
        }
    }
}

template <class Emit>
inline void doubles_exact_aa_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (std::size_t pa = 0; pa < occ.vir_a.size(); ++pa) {
                const int a = occ.vir_a[pa];

                for (std::size_t pb = pa + 1u; pb < occ.vir_a.size(); ++pb) {
                    const int b = occ.vir_a[pb];

                    const double h =
                        Slater::sign_double(occ.pref_a, i, j, a, b)
                        * Slater::double_aa(ints, i, j, a, b);

                    if (keep_h(h, eps)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_bb_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (std::size_t pa = 0; pa < occ.vir_b.size(); ++pa) {
                const int a = occ.vir_b[pa];

                for (std::size_t pb = pa + 1u; pb < occ.vir_b.size(); ++pb) {
                    const int b = occ.vir_b[pb];

                    const double h =
                        Slater::sign_double(occ.pref_b, i, j, a, b)
                        * Slater::double_bb(ints, i, j, a, b);

                    if (keep_h(h, eps)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_ab_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (int ia : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double sign_a = Slater::sign_single(occ.pref_a, ia, a);

            for (int ib : occ.occ_b) {
                for (int b : occ.vir_b) {
                    const double h =
                        sign_a
                        * Slater::sign_single(occ.pref_b, ib, b)
                        * Slater::double_ab(ints, ia, ib, a, b);

                    if (keep_h(h, eps)) emit(h);
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
    double eps,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const auto& c : hb.aa_ge(i, j, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;

                if (keep_h(h, eps)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_hb_bb_values(
    const HeatBathTable& hb,
    DetRef ket,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const auto& c : hb.bb_ge(i, j, eps)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;

                if (keep_h(h, eps)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_hb_ab_values(
    const HeatBathTable& hb,
    DetRef ket,
    const DetOcc& occ,
    double eps,
    Emit&& emit
) {
    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const auto& c : hb.ab_ge(ia, ib, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;

                if (keep_h(h, eps)) emit(h);
            }
        }
    }
}

/* ---------- bra-emitting scans ---------- */

template <class Emit>
inline void singles_alpha_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps,
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

            if (keep_h(h, eps)) emit(occ.det.view(), h);

            bits::clear(alpha, a);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void singles_beta_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps,
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

            if (keep_h(h, eps)) emit(occ.det.view(), h);

            bits::clear(beta, a);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void singles_alpha_window_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps2,
    double eps1,
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

            if (in_window(h, eps2, eps1)) emit(occ.det.view(), h);

            bits::clear(alpha, a);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void singles_beta_window_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps2,
    double eps1,
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

            if (in_window(h, eps2, eps1)) emit(occ.det.view(), h);

            bits::clear(beta, a);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_exact_aa_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];
        bits::clear(alpha, i);

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];
            bits::clear(alpha, j);

            for (std::size_t pa = 0; pa < occ.vir_a.size(); ++pa) {
                const int a = occ.vir_a[pa];
                bits::set(alpha, a);

                for (std::size_t pb = pa + 1u; pb < occ.vir_a.size(); ++pb) {
                    const int b = occ.vir_a[pb];
                    bits::set(alpha, b);

                    const double h =
                        Slater::sign_double(occ.pref_a, i, j, a, b)
                        * Slater::double_aa(ints, i, j, a, b);

                    if (keep_h(h, eps)) emit(occ.det.view(), h);

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
inline void doubles_exact_bb_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];
        bits::clear(beta, i);

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];
            bits::clear(beta, j);

            for (std::size_t pa = 0; pa < occ.vir_b.size(); ++pa) {
                const int a = occ.vir_b[pa];
                bits::set(beta, a);

                for (std::size_t pb = pa + 1u; pb < occ.vir_b.size(); ++pb) {
                    const int b = occ.vir_b[pb];
                    bits::set(beta, b);

                    const double h =
                        Slater::sign_double(occ.pref_b, i, j, a, b)
                        * Slater::double_bb(ints, i, j, a, b);

                    if (keep_h(h, eps)) emit(occ.det.view(), h);

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
inline void doubles_exact_ab_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int ia : occ.occ_a) {
        bits::clear(alpha, ia);

        for (int a : occ.vir_a) {
            bits::set(alpha, a);

            const double sign_a = Slater::sign_single(occ.pref_a, ia, a);

            for (int ib : occ.occ_b) {
                bits::clear(beta, ib);

                for (int b : occ.vir_b) {
                    bits::set(beta, b);

                    const double h =
                        sign_a
                        * Slater::sign_single(occ.pref_b, ib, b)
                        * Slater::double_ab(ints, ia, ib, a, b);

                    if (keep_h(h, eps)) emit(occ.det.view(), h);

                    bits::clear(beta, b);
                }

                bits::set(beta, ib);
            }

            bits::clear(alpha, a);
        }

        bits::set(alpha, ia);
    }
}

template <class Emit>
inline void doubles_hb_aa_conns(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];
        bits::clear(alpha, i);

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];
            bits::clear(alpha, j);

            for (const auto& c : hb.aa_ge(i, j, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                bits::set(alpha, c.a);
                bits::set(alpha, c.b);

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;

                if (keep_h(h, eps)) emit(occ.det.view(), h);

                bits::clear(alpha, c.b);
                bits::clear(alpha, c.a);
            }

            bits::set(alpha, j);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void doubles_hb_bb_conns(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto beta = occ.det.beta();

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];
        bits::clear(beta, i);

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];
            bits::clear(beta, j);

            for (const auto& c : hb.bb_ge(i, j, eps)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                bits::set(beta, c.a);
                bits::set(beta, c.b);

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;

                if (keep_h(h, eps)) emit(occ.det.view(), h);

                bits::clear(beta, c.b);
                bits::clear(beta, c.a);
            }

            bits::set(beta, j);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_hb_ab_conns(
    const HeatBathTable& hb,
    DetRef ket,
    DetOcc& occ,
    double eps,
    Emit&& emit
) {
    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int ia : occ.occ_a) {
        bits::clear(alpha, ia);

        for (int ib : occ.occ_b) {
            bits::clear(beta, ib);

            for (const auto& c : hb.ab_ge(ia, ib, eps)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                bits::set(alpha, c.a);
                bits::set(beta, c.b);

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;

                if (keep_h(h, eps)) emit(occ.det.view(), h);

                bits::clear(beta, c.b);
                bits::clear(alpha, c.a);
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
    KetWork& work,
    double eps,
    Emit&& emit
) {
    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_values(ints, work.occ, eps, emit);
    singles_beta_values(ints, work.occ, eps, emit);

    if (eps >= eps_hb) {
        doubles_hb_aa_values(hb, ket, work.occ, eps, emit);
        doubles_hb_bb_values(hb, ket, work.occ, eps, emit);
        doubles_hb_ab_values(hb, ket, work.occ, eps, emit);
    } else {
        doubles_exact_aa_values(ints, work.occ, eps, emit);
        doubles_exact_bb_values(ints, work.occ, eps, emit);
        doubles_exact_ab_values(ints, work.occ, eps, emit);
    }
}

template <class Emit>
inline void scan_conns(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    KetWork& work,
    double eps,
    Emit&& emit
) {
    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_conns(ints, work.occ, eps, emit);
    singles_beta_conns(ints, work.occ, eps, emit);

    if (eps >= eps_hb) {
        doubles_hb_aa_conns(hb, ket, work.occ, eps, emit);
        doubles_hb_bb_conns(hb, ket, work.occ, eps, emit);
        doubles_hb_ab_conns(hb, ket, work.occ, eps, emit);
    } else {
        doubles_exact_aa_conns(ints, work.occ, eps, emit);
        doubles_exact_bb_conns(ints, work.occ, eps, emit);
        doubles_exact_ab_conns(ints, work.occ, eps, emit);
    }
}

template <class Emit>
inline void scan_window_values(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    KetWork& work,
    double eps2,
    double eps1,
    Emit&& emit
) {
    if (eps1 <= eps2 || eps1 <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_window_values(ints, work.occ, eps2, eps1, emit);
    singles_beta_window_values(ints, work.occ, eps2, eps1, emit);

    if (eps2 >= eps_hb) {
        for (std::size_t x = 0; x < work.occ.occ_a.size(); ++x) {
            const int i = work.occ.occ_a[x];

            for (std::size_t y = x + 1u; y < work.occ.occ_a.size(); ++y) {
                const int j = work.occ.occ_a[y];

                for (const auto& c : hb.aa_window(i, j, eps2, eps1)) {
                    if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) continue;

                    const double h =
                        Slater::sign_double(work.occ.pref_a, i, j, c.a, c.b) * c.h;

                    emit(h);
                }
            }
        }

        for (std::size_t x = 0; x < work.occ.occ_b.size(); ++x) {
            const int i = work.occ.occ_b[x];

            for (std::size_t y = x + 1u; y < work.occ.occ_b.size(); ++y) {
                const int j = work.occ.occ_b[y];

                for (const auto& c : hb.bb_window(i, j, eps2, eps1)) {
                    if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) continue;

                    const double h =
                        Slater::sign_double(work.occ.pref_b, i, j, c.a, c.b) * c.h;

                    emit(h);
                }
            }
        }

        for (int ia : work.occ.occ_a) {
            for (int ib : work.occ.occ_b) {
                for (const auto& c : hb.ab_window(ia, ib, eps2, eps1)) {
                    if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) continue;

                    const double h =
                        Slater::sign_single(work.occ.pref_a, ia, c.a)
                        * Slater::sign_single(work.occ.pref_b, ib, c.b)
                        * c.h;

                    emit(h);
                }
            }
        }
    } else {
        doubles_exact_aa_values(ints, work.occ, 0.0, [&](double h) {
            if (in_window(h, eps2, eps1)) emit(h);
        });

        doubles_exact_bb_values(ints, work.occ, 0.0, [&](double h) {
            if (in_window(h, eps2, eps1)) emit(h);
        });

        doubles_exact_ab_values(ints, work.occ, 0.0, [&](double h) {
            if (in_window(h, eps2, eps1)) emit(h);
        });
    }
}

template <class Emit>
inline void scan_window_conns(
    const RHFIntegrals& ints,
    const HeatBathTable& hb,
    DetRef ket,
    KetWork& work,
    double eps2,
    double eps1,
    Emit&& emit
) {
    if (eps1 <= eps2 || eps1 <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_window_conns(ints, work.occ, eps2, eps1, emit);
    singles_beta_window_conns(ints, work.occ, eps2, eps1, emit);

    if (eps2 >= eps_hb) {
        auto alpha = work.occ.det.alpha();
        auto beta = work.occ.det.beta();

        for (std::size_t x = 0; x < work.occ.occ_a.size(); ++x) {
            const int i = work.occ.occ_a[x];
            bits::clear(alpha, i);

            for (std::size_t y = x + 1u; y < work.occ.occ_a.size(); ++y) {
                const int j = work.occ.occ_a[y];
                bits::clear(alpha, j);

                for (const auto& c : hb.aa_window(i, j, eps2, eps1)) {
                    if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) continue;

                    bits::set(alpha, c.a);
                    bits::set(alpha, c.b);

                    const double h =
                        Slater::sign_double(work.occ.pref_a, i, j, c.a, c.b) * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(alpha, c.b);
                    bits::clear(alpha, c.a);
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

                for (const auto& c : hb.bb_window(i, j, eps2, eps1)) {
                    if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) continue;

                    bits::set(beta, c.a);
                    bits::set(beta, c.b);

                    const double h =
                        Slater::sign_double(work.occ.pref_b, i, j, c.a, c.b) * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(beta, c.b);
                    bits::clear(beta, c.a);
                }

                bits::set(beta, j);
            }

            bits::set(beta, i);
        }

        for (int ia : work.occ.occ_a) {
            bits::clear(alpha, ia);

            for (int ib : work.occ.occ_b) {
                bits::clear(beta, ib);

                for (const auto& c : hb.ab_window(ia, ib, eps2, eps1)) {
                    if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) continue;

                    bits::set(alpha, c.a);
                    bits::set(beta, c.b);

                    const double h =
                        Slater::sign_single(work.occ.pref_a, ia, c.a)
                        * Slater::sign_single(work.occ.pref_b, ib, c.b)
                        * c.h;

                    emit(work.occ.det.view(), h);

                    bits::clear(beta, c.b);
                    bits::clear(alpha, c.a);
                }

                bits::set(beta, ib);
            }

            bits::set(alpha, ia);
        }
    } else {
        doubles_exact_aa_conns(ints, work.occ, 0.0, [&](DetRef bra, double h) {
            if (in_window(h, eps2, eps1)) emit(bra, h);
        });

        doubles_exact_bb_conns(ints, work.occ, 0.0, [&](DetRef bra, double h) {
            if (in_window(h, eps2, eps1)) emit(bra, h);
        });

        doubles_exact_ab_conns(ints, work.occ, 0.0, [&](DetRef bra, double h) {
            if (in_window(h, eps2, eps1)) emit(bra, h);
        });
    }
}

} // namespace libdet