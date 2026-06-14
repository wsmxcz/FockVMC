#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <libdet/det.hpp>
#include <libdet/screen.hpp>
#include <libdet/slater.hpp>

namespace libdet {

/*
 * Excitation-driven row scans for free-bra generation.
 *
 * Singles are evaluated directly. Doubles use an optional Screen when the
 * requested lower bound is positive; otherwise they are enumerated exactly.
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

struct EdgeWindow {
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double scale = 1.0;
};

[[nodiscard]] inline bool keep_edge(double h, EdgeWindow win) noexcept {
    const double v = std::abs(h) * win.scale;
    return v > 0.0 && v >= win.lo && v < win.hi;
}

[[nodiscard]] inline double screen_min_abs(EdgeWindow win) noexcept {
    if (win.scale <= 0.0) return std::numeric_limits<double>::infinity();
    if (win.lo <= 0.0) return 0.0;
    return win.lo / win.scale;
}

/* ---------- singles ---------- */

template <class Emit>
inline void singles_alpha_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_edge(h, win)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_beta_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);

            if (keep_edge(h, win)) emit(h);
        }
    }
}

template <class Emit>
inline void singles_alpha_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    EdgeWindow win,
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

            if (keep_edge(h, win)) emit(occ.det.view(), h);

            bits::clear(alpha, a);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void singles_beta_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    EdgeWindow win,
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

            if (keep_edge(h, win)) emit(occ.det.view(), h);

            bits::clear(beta, a);
        }

        bits::set(beta, i);
    }
}

/* ---------- exact double scans ---------- */

template <class Emit>
inline void doubles_exact_aa_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    EdgeWindow win,
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

                    if (keep_edge(h, win)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_bb_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    EdgeWindow win,
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

                    if (keep_edge(h, win)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_ab_values(
    const RHFIntegrals& ints,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    for (int ia : occ.occ_a) {
        const double sign_a_base = 1.0;

        for (int a : occ.vir_a) {
            const double sign_a = sign_a_base * Slater::sign_single(occ.pref_a, ia, a);

            for (int ib : occ.occ_b) {
                for (int b : occ.vir_b) {
                    const double h =
                        sign_a
                        * Slater::sign_single(occ.pref_b, ib, b)
                        * Slater::double_ab(ints, ia, ib, a, b);

                    if (keep_edge(h, win)) emit(h);
                }
            }
        }
    }
}

template <class Emit>
inline void doubles_exact_aa_conns(
    const RHFIntegrals& ints,
    DetOcc& occ,
    EdgeWindow win,
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

                    if (keep_edge(h, win)) emit(occ.det.view(), h);

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
    EdgeWindow win,
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

                    if (keep_edge(h, win)) emit(occ.det.view(), h);

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
    EdgeWindow win,
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

                    if (keep_edge(h, win)) emit(occ.det.view(), h);

                    bits::clear(beta, b);
                }

                bits::set(beta, ib);
            }

            bits::clear(alpha, a);
        }

        bits::set(alpha, ia);
    }
}

/* ---------- screened double scans ---------- */

template <class Emit>
inline void doubles_screen_aa_values(
    const Screen& screen,
    DetRef ket,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const auto& c : screen.aa(i, j, min_abs)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h = Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (keep_edge(h, win)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_screen_bb_values(
    const Screen& screen,
    DetRef ket,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const auto& c : screen.bb(i, j, min_abs)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h = Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (keep_edge(h, win)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_screen_ab_values(
    const Screen& screen,
    DetRef ket,
    const DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);

    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const auto& c : screen.ab(ia, ib, min_abs)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;

                if (keep_edge(h, win)) emit(h);
            }
        }
    }
}

template <class Emit>
inline void doubles_screen_aa_conns(
    const Screen& screen,
    DetRef ket,
    DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);
    auto alpha = occ.det.alpha();

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];
        bits::clear(alpha, i);

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];
            bits::clear(alpha, j);

            for (const auto& c : screen.aa(i, j, min_abs)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                bits::set(alpha, c.a);
                bits::set(alpha, c.b);

                const double h = Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (keep_edge(h, win)) emit(occ.det.view(), h);

                bits::clear(alpha, c.b);
                bits::clear(alpha, c.a);
            }

            bits::set(alpha, j);
        }

        bits::set(alpha, i);
    }
}

template <class Emit>
inline void doubles_screen_bb_conns(
    const Screen& screen,
    DetRef ket,
    DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);
    auto beta = occ.det.beta();

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];
        bits::clear(beta, i);

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];
            bits::clear(beta, j);

            for (const auto& c : screen.bb(i, j, min_abs)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                bits::set(beta, c.a);
                bits::set(beta, c.b);

                const double h = Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (keep_edge(h, win)) emit(occ.det.view(), h);

                bits::clear(beta, c.b);
                bits::clear(beta, c.a);
            }

            bits::set(beta, j);
        }

        bits::set(beta, i);
    }
}

template <class Emit>
inline void doubles_screen_ab_conns(
    const Screen& screen,
    DetRef ket,
    DetOcc& occ,
    EdgeWindow win,
    Emit&& emit
) {
    const double min_abs = screen_min_abs(win);
    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int ia : occ.occ_a) {
        bits::clear(alpha, ia);

        for (int ib : occ.occ_b) {
            bits::clear(beta, ib);

            for (const auto& c : screen.ab(ia, ib, min_abs)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                bits::set(alpha, c.a);
                bits::set(beta, c.b);

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;

                if (keep_edge(h, win)) emit(occ.det.view(), h);

                bits::clear(beta, c.b);
                bits::clear(alpha, c.a);
            }

            bits::set(beta, ib);
        }

        bits::set(alpha, ia);
    }
}

/* ---------- public row scans ---------- */

template <class Emit>
inline void scan_values(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    KetWork& work,
    EdgeWindow win,
    Emit&& emit
) {
    if (win.hi <= win.lo || win.scale <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_values(ints, work.occ, win, emit);
    singles_beta_values(ints, work.occ, win, emit);

    if (screen == nullptr) {
        doubles_exact_aa_values(ints, work.occ, win, emit);
        doubles_exact_bb_values(ints, work.occ, win, emit);
        doubles_exact_ab_values(ints, work.occ, win, emit);
    } else {
        doubles_screen_aa_values(*screen, ket, work.occ, win, emit);
        doubles_screen_bb_values(*screen, ket, work.occ, win, emit);
        doubles_screen_ab_values(*screen, ket, work.occ, win, emit);
    }
}

template <class Emit>
inline void scan_conns(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    KetWork& work,
    EdgeWindow win,
    Emit&& emit
) {
    if (win.hi <= win.lo || win.scale <= 0.0) return;

    fill_occ(ket, ints.norb(), work.occ);

    singles_alpha_conns(ints, work.occ, win, emit);
    singles_beta_conns(ints, work.occ, win, emit);

    if (screen == nullptr) {
        doubles_exact_aa_conns(ints, work.occ, win, emit);
        doubles_exact_bb_conns(ints, work.occ, win, emit);
        doubles_exact_ab_conns(ints, work.occ, win, emit);
    } else {
        doubles_screen_aa_conns(*screen, ket, work.occ, win, emit);
        doubles_screen_bb_conns(*screen, ket, work.occ, win, emit);
        doubles_screen_ab_conns(*screen, ket, work.occ, win, emit);
    }
}

} // namespace libdet
