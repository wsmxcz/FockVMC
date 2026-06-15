#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <libdet/det.hpp>
#include <libdet/screen.hpp>
#include <libdet/slater.hpp>

namespace libdet {

struct KetWork {
    explicit KetWork(u32 nword, int norb)
        : occ(nword, norb) {}

    void resize(u32 nword, int norb) {
        occ.resize(nword, norb);
    }

    DetOcc occ;
};

struct EdgeWindow {
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
    double scale = 1.0;
};

struct EmitRequest {
    bool exact = false;
    i64 count = 0;

    [[nodiscard]] explicit operator bool() const noexcept {
        return exact || count > 0;
    }
};

[[nodiscard]] inline bool in_window(double h, EdgeWindow win) noexcept {
    const double value = std::abs(h) * win.scale;
    return value > 0.0 && value >= win.lo && value < win.hi;
}

[[nodiscard]] inline double window_lo(EdgeWindow win) noexcept {
    if (win.scale <= 0.0) return std::numeric_limits<double>::infinity();
    return win.lo <= 0.0 ? 0.0 : win.lo / win.scale;
}

[[nodiscard]] inline double window_hi(EdgeWindow win) noexcept {
    if (win.scale <= 0.0) return 0.0;
    return std::isfinite(win.hi)
        ? win.hi / win.scale
        : std::numeric_limits<double>::infinity();
}

template <class Decide, class Emit>
inline void scan_prepared(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    DetOcc& occ,
    EdgeWindow win,
    Decide&& decide,
    Emit&& emit
) {
    if (win.hi <= win.lo || win.scale <= 0.0) return;

    auto alpha = occ.det.alpha();
    auto beta = occ.det.beta();

    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);
            if (!in_window(h, win)) continue;

            const EmitRequest req = decide(h);
            if (!req) continue;

            bits::clear(alpha, i);
            bits::set(alpha, a);
            emit(occ.det.view(), h, req);
            bits::clear(alpha, a);
            bits::set(alpha, i);
        }
    }

    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);
            if (!in_window(h, win)) continue;

            const EmitRequest req = decide(h);
            if (!req) continue;

            bits::clear(beta, i);
            bits::set(beta, a);
            emit(occ.det.view(), h, req);
            bits::clear(beta, a);
            bits::set(beta, i);
        }
    }

    if (screen == nullptr) {
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
                        if (!in_window(h, win)) continue;

                        const EmitRequest req = decide(h);
                        if (!req) continue;

                        bits::clear(alpha, i);
                        bits::clear(alpha, j);
                        bits::set(alpha, a);
                        bits::set(alpha, b);
                        emit(occ.det.view(), h, req);
                        bits::clear(alpha, b);
                        bits::clear(alpha, a);
                        bits::set(alpha, j);
                        bits::set(alpha, i);
                    }
                }
            }
        }

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
                        if (!in_window(h, win)) continue;

                        const EmitRequest req = decide(h);
                        if (!req) continue;

                        bits::clear(beta, i);
                        bits::clear(beta, j);
                        bits::set(beta, a);
                        bits::set(beta, b);
                        emit(occ.det.view(), h, req);
                        bits::clear(beta, b);
                        bits::clear(beta, a);
                        bits::set(beta, j);
                        bits::set(beta, i);
                    }
                }
            }
        }

        for (int ia : occ.occ_a) {
            for (int a : occ.vir_a) {
                const double sign_a = Slater::sign_single(occ.pref_a, ia, a);

                for (int ib : occ.occ_b) {
                    for (int b : occ.vir_b) {
                        const double h =
                            sign_a
                            * Slater::sign_single(occ.pref_b, ib, b)
                            * Slater::double_ab(ints, ia, ib, a, b);
                        if (!in_window(h, win)) continue;

                        const EmitRequest req = decide(h);
                        if (!req) continue;

                        bits::clear(alpha, ia);
                        bits::clear(beta, ib);
                        bits::set(alpha, a);
                        bits::set(beta, b);
                        emit(occ.det.view(), h, req);
                        bits::clear(beta, b);
                        bits::clear(alpha, a);
                        bits::set(beta, ib);
                        bits::set(alpha, ia);
                    }
                }
            }
        }

        return;
    }

    const double lo = window_lo(win);
    const double hi = window_hi(win);

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const Cand& c : screen->aa(i, j, lo, hi)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (!in_window(h, win)) continue;

                const EmitRequest req = decide(h);
                if (!req) continue;

                bits::clear(alpha, i);
                bits::clear(alpha, j);
                bits::set(alpha, c.a);
                bits::set(alpha, c.b);
                emit(occ.det.view(), h, req);
                bits::clear(alpha, c.b);
                bits::clear(alpha, c.a);
                bits::set(alpha, j);
                bits::set(alpha, i);
            }
        }
    }

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const Cand& c : screen->bb(i, j, lo, hi)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (!in_window(h, win)) continue;

                const EmitRequest req = decide(h);
                if (!req) continue;

                bits::clear(beta, i);
                bits::clear(beta, j);
                bits::set(beta, c.a);
                bits::set(beta, c.b);
                emit(occ.det.view(), h, req);
                bits::clear(beta, c.b);
                bits::clear(beta, c.a);
                bits::set(beta, j);
                bits::set(beta, i);
            }
        }
    }

    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (const Cand& c : screen->ab(ia, ib, lo, hi)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;
                if (!in_window(h, win)) continue;

                const EmitRequest req = decide(h);
                if (!req) continue;

                bits::clear(alpha, ia);
                bits::clear(beta, ib);
                bits::set(alpha, c.a);
                bits::set(beta, c.b);
                emit(occ.det.view(), h, req);
                bits::clear(beta, c.b);
                bits::clear(alpha, c.a);
                bits::set(beta, ib);
                bits::set(alpha, ia);
            }
        }
    }
}

template <class Decide, class Emit>
inline void scan(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    KetWork& work,
    EdgeWindow win,
    Decide&& decide,
    Emit&& emit
) {
    fill_occ(ket, ints.norb(), work.occ);
    scan_prepared(
        ints,
        screen,
        ket,
        work.occ,
        win,
        std::forward<Decide>(decide),
        std::forward<Emit>(emit)
    );
}

} // namespace libdet
