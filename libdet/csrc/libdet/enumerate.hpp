#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

#include <libdet/det_space.hpp>
#include <libdet/integrals.hpp>
#include <libdet/slater_condon.hpp>

namespace libdet {

struct RowWork {
    explicit RowWork(u32 nword, int norb) : occ(nword) {
        occ.resize(nword, norb);
    }

    void ensure_seen(std::size_t na, std::size_t nb) {
        if (seen_a.size() < na) {
            seen_a.assign(na, 0u);
            stamp_a = 1u;
        }
        if (seen_b.size() < nb) {
            seen_b.assign(nb, 0u);
            stamp_b = 1u;
        }
    }

    void next_a() {
        if (++stamp_a == 0u) {
            std::fill(seen_a.begin(), seen_a.end(), 0u);
            stamp_a = 1u;
        }
    }

    void next_b() {
        if (++stamp_b == 0u) {
            std::fill(seen_b.begin(), seen_b.end(), 0u);
            stamp_b = 1u;
        }
    }

    void ensure_cross(std::size_t nb) {
        if (cross_b.size() < nb) {
            cross_b.assign(nb, 0u);
            cross_occ.assign(nb, 0);
            cross_vir.assign(nb, 0);
            cross_sign.assign(nb, 1.0);
            cross_stamp = 1u;
        }
    }

    void next_cross() {
        if (++cross_stamp == 0u) {
            std::fill(cross_b.begin(), cross_b.end(), 0u);
            cross_stamp = 1u;
        }
    }

    RowOcc occ;
    std::vector<int> tmp_occ;
    std::vector<u32> seen_a;
    std::vector<u32> seen_b;
    u32 stamp_a = 1u;
    u32 stamp_b = 1u;
    std::vector<u32> cross_b;
    std::vector<int> cross_occ;
    std::vector<int> cross_vir;
    std::vector<double> cross_sign;
    u32 cross_stamp = 1u;
    std::vector<SelSingle> alpha_single;
    std::vector<SelSingle> beta_single;
    std::vector<SelDouble> alpha_double;
    std::vector<SelDouble> beta_double;
    std::vector<u64> weak_words;
    std::vector<double> weak_h;
    std::vector<double> weak_cdf;
    std::vector<i64> sample_freq;
    std::vector<i32> sample_touched;
};

[[nodiscard]] inline bool keep_h(double h, double cut) noexcept {
    const double a = std::abs(h);
    return a > 0.0 && a >= cut;
}

[[nodiscard]] inline bool in_window(double h, double lo, double hi) noexcept {
    const double a = std::abs(h);
    return a > 0.0 && a >= lo && a < hi;
}

template <class Emit>
inline void singles_alpha(const RHFIntegrals& ints, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto a = work.occ.det.alpha();
    for (int i : work.occ.occ_a) {
        bits::clear(a, i);
        bits::each_clear(ket.alpha(), ints.norb(), [&](int av) {
            bits::set(a, av);
            const double h = SlaterCondon::sign_single(ket.alpha(), i, av) * SlaterCondon::single_a(ints, ket, i, av);
            if (keep_h(h, cut)) emit(work.occ.det.view(), h);
            bits::clear(a, av);
        });
        bits::set(a, i);
    }
}

template <class Emit>
inline void singles_beta(const RHFIntegrals& ints, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto b = work.occ.det.beta();
    for (int i : work.occ.occ_b) {
        bits::clear(b, i);
        bits::each_clear(ket.beta(), ints.norb(), [&](int av) {
            bits::set(b, av);
            const double h = SlaterCondon::sign_single(ket.beta(), i, av) * SlaterCondon::single_b(ints, ket, i, av);
            if (keep_h(h, cut)) emit(work.occ.det.view(), h);
            bits::clear(b, av);
        });
        bits::set(b, i);
    }
}

template <class Emit>
inline void singles_window_alpha(const RHFIntegrals& ints, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    auto a = work.occ.det.alpha();
    for (int i : work.occ.occ_a) {
        bits::clear(a, i);
        bits::each_clear(ket.alpha(), ints.norb(), [&](int av) {
            bits::set(a, av);
            const double h = SlaterCondon::sign_single(ket.alpha(), i, av) * SlaterCondon::single_a(ints, ket, i, av);
            if (in_window(h, lo, hi)) emit(work.occ.det.view(), h);
            bits::clear(a, av);
        });
        bits::set(a, i);
    }
}

template <class Emit>
inline void singles_window_beta(const RHFIntegrals& ints, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    auto b = work.occ.det.beta();
    for (int i : work.occ.occ_b) {
        bits::clear(b, i);
        bits::each_clear(ket.beta(), ints.norb(), [&](int av) {
            bits::set(b, av);
            const double h = SlaterCondon::sign_single(ket.beta(), i, av) * SlaterCondon::single_b(ints, ket, i, av);
            if (in_window(h, lo, hi)) emit(work.occ.det.view(), h);
            bits::clear(b, av);
        });
        bits::set(b, i);
    }
}

template <class Emit>
inline void doubles_exact_aa(const RHFIntegrals& ints, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto a = work.occ.det.alpha();
    const int norb = ints.norb();
    for (std::size_t x = 0; x < work.occ.occ_a.size(); ++x) {
        const int i = work.occ.occ_a[x];
        bits::clear(a, i);
        for (std::size_t y = x + 1; y < work.occ.occ_a.size(); ++y) {
            const int j = work.occ.occ_a[y];
            bits::clear(a, j);
            for (int av = 0; av < norb; ++av) {
                if (bits::test(ket.alpha(), av)) continue;
                bits::set(a, av);
                for (int bv = av + 1; bv < norb; ++bv) {
                    if (bits::test(ket.alpha(), bv)) continue;
                    bits::set(a, bv);
                    const double h = SlaterCondon::sign_double(ket.alpha(), i, j, av, bv) * SlaterCondon::double_aa(ints, i, j, av, bv);
                    if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                    bits::clear(a, bv);
                }
                bits::clear(a, av);
            }
            bits::set(a, j);
        }
        bits::set(a, i);
    }
}

template <class Emit>
inline void doubles_exact_bb(const RHFIntegrals& ints, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto b = work.occ.det.beta();
    const int norb = ints.norb();
    for (std::size_t x = 0; x < work.occ.occ_b.size(); ++x) {
        const int i = work.occ.occ_b[x];
        bits::clear(b, i);
        for (std::size_t y = x + 1; y < work.occ.occ_b.size(); ++y) {
            const int j = work.occ.occ_b[y];
            bits::clear(b, j);
            for (int av = 0; av < norb; ++av) {
                if (bits::test(ket.beta(), av)) continue;
                bits::set(b, av);
                for (int bv = av + 1; bv < norb; ++bv) {
                    if (bits::test(ket.beta(), bv)) continue;
                    bits::set(b, bv);
                    const double h = SlaterCondon::sign_double(ket.beta(), i, j, av, bv) * SlaterCondon::double_bb(ints, i, j, av, bv);
                    if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                    bits::clear(b, bv);
                }
                bits::clear(b, av);
            }
            bits::set(b, j);
        }
        bits::set(b, i);
    }
}

template <class Emit>
inline void doubles_exact_ab(const RHFIntegrals& ints, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto a = work.occ.det.alpha();
    auto b = work.occ.det.beta();
    for (int ia : work.occ.occ_a) {
        bits::clear(a, ia);
        bits::each_clear(ket.alpha(), ints.norb(), [&](int aa) {
            bits::set(a, aa);
            const double sa = SlaterCondon::sign_single(ket.alpha(), ia, aa);
            for (int ib : work.occ.occ_b) {
                bits::clear(b, ib);
                bits::each_clear(ket.beta(), ints.norb(), [&](int ab) {
                    bits::set(b, ab);
                    const double h = sa * SlaterCondon::sign_single(ket.beta(), ib, ab) * SlaterCondon::double_ab(ints, ia, ib, aa, ab);
                    if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                    bits::clear(b, ab);
                });
                bits::set(b, ib);
            }
            bits::clear(a, aa);
        });
        bits::set(a, ia);
    }
}

template <class Emit>
inline void doubles_hb_aa(const HeatBathTable& hb, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto a = work.occ.det.alpha();
    bits::each_set(ket.alpha(), [&](int i) {
        bits::clear(a, i);
        bits::each_set(ket.alpha(), [&](int j) {
            if (j <= i) return;
            bits::clear(a, j);
            for (const auto& c : hb.aa(i, j)) {
                if (std::abs(c.h) < cut) break;
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) continue;
                bits::set(a, c.p);
                bits::set(a, c.q);
                const double h = SlaterCondon::sign_double(ket.alpha(), i, j, c.p, c.q) * c.h;
                if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                bits::clear(a, c.q);
                bits::clear(a, c.p);
            }
            bits::set(a, j);
        });
        bits::set(a, i);
    });
}

template <class Emit>
inline void doubles_hb_bb(const HeatBathTable& hb, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto b = work.occ.det.beta();
    bits::each_set(ket.beta(), [&](int i) {
        bits::clear(b, i);
        bits::each_set(ket.beta(), [&](int j) {
            if (j <= i) return;
            bits::clear(b, j);
            for (const auto& c : hb.bb(i, j)) {
                if (std::abs(c.h) < cut) break;
                if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) continue;
                bits::set(b, c.p);
                bits::set(b, c.q);
                const double h = SlaterCondon::sign_double(ket.beta(), i, j, c.p, c.q) * c.h;
                if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                bits::clear(b, c.q);
                bits::clear(b, c.p);
            }
            bits::set(b, j);
        });
        bits::set(b, i);
    });
}

template <class Emit>
inline void doubles_hb_ab(const HeatBathTable& hb, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    auto a = work.occ.det.alpha();
    auto b = work.occ.det.beta();
    bits::each_set(ket.alpha(), [&](int ia) {
        bits::clear(a, ia);
        bits::each_set(ket.beta(), [&](int ib) {
            bits::clear(b, ib);
            for (const auto& c : hb.ab(ia, ib)) {
                if (std::abs(c.h) < cut) break;
                if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) continue;
                bits::set(a, c.p);
                bits::set(b, c.q);
                const double h = SlaterCondon::sign_single(ket.alpha(), ia, c.p) * SlaterCondon::sign_single(ket.beta(), ib, c.q) * c.h;
                if (keep_h(h, cut)) emit(work.occ.det.view(), h);
                bits::clear(b, c.q);
                bits::clear(a, c.p);
            }
            bits::set(b, ib);
        });
        bits::set(a, ia);
    });
}

template <class Emit>
inline void doubles_window_aa(const RHFIntegrals& ints, const HeatBathTable* hb, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    if (hb != nullptr && lo >= hb->eps_table()) {
        auto a = work.occ.det.alpha();
        bits::each_set(ket.alpha(), [&](int i) {
            bits::clear(a, i);
            bits::each_set(ket.alpha(), [&](int j) {
                if (j <= i) return;
                bits::clear(a, j);
                for (const auto& c : hb->aa(i, j)) {
                    const double ah = std::abs(c.h);
                    if (ah >= hi) continue;
                    if (ah < lo) break;
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.alpha(), c.q)) continue;
                    bits::set(a, c.p);
                    bits::set(a, c.q);
                    const double h = SlaterCondon::sign_double(ket.alpha(), i, j, c.p, c.q) * c.h;
                    if (in_window(h, lo, hi)) emit(work.occ.det.view(), h);
                    bits::clear(a, c.q);
                    bits::clear(a, c.p);
                }
                bits::set(a, j);
            });
            bits::set(a, i);
        });
    } else {
        doubles_exact_aa(ints, ket, work, 0.0, [&](DetRef e, double h) { if (in_window(h, lo, hi)) emit(e, h); });
    }
}

template <class Emit>
inline void doubles_window_bb(const RHFIntegrals& ints, const HeatBathTable* hb, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    if (hb != nullptr && lo >= hb->eps_table()) {
        auto b = work.occ.det.beta();
        bits::each_set(ket.beta(), [&](int i) {
            bits::clear(b, i);
            bits::each_set(ket.beta(), [&](int j) {
                if (j <= i) return;
                bits::clear(b, j);
                for (const auto& c : hb->bb(i, j)) {
                    const double ah = std::abs(c.h);
                    if (ah >= hi) continue;
                    if (ah < lo) break;
                    if (bits::test(ket.beta(), c.p) || bits::test(ket.beta(), c.q)) continue;
                    bits::set(b, c.p);
                    bits::set(b, c.q);
                    const double h = SlaterCondon::sign_double(ket.beta(), i, j, c.p, c.q) * c.h;
                    if (in_window(h, lo, hi)) emit(work.occ.det.view(), h);
                    bits::clear(b, c.q);
                    bits::clear(b, c.p);
                }
                bits::set(b, j);
            });
            bits::set(b, i);
        });
    } else {
        doubles_exact_bb(ints, ket, work, 0.0, [&](DetRef e, double h) { if (in_window(h, lo, hi)) emit(e, h); });
    }
}

template <class Emit>
inline void doubles_window_ab(const RHFIntegrals& ints, const HeatBathTable* hb, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    if (hb != nullptr && lo >= hb->eps_table()) {
        auto a = work.occ.det.alpha();
        auto b = work.occ.det.beta();
        bits::each_set(ket.alpha(), [&](int ia) {
            bits::clear(a, ia);
            bits::each_set(ket.beta(), [&](int ib) {
                bits::clear(b, ib);
                for (const auto& c : hb->ab(ia, ib)) {
                    const double ah = std::abs(c.h);
                    if (ah >= hi) continue;
                    if (ah < lo) break;
                    if (bits::test(ket.alpha(), c.p) || bits::test(ket.beta(), c.q)) continue;
                    bits::set(a, c.p);
                    bits::set(b, c.q);
                    const double h = SlaterCondon::sign_single(ket.alpha(), ia, c.p) * SlaterCondon::sign_single(ket.beta(), ib, c.q) * c.h;
                    if (in_window(h, lo, hi)) emit(work.occ.det.view(), h);
                    bits::clear(b, c.q);
                    bits::clear(a, c.p);
                }
                bits::set(b, ib);
            });
            bits::set(a, ia);
        });
    } else {
        doubles_exact_ab(ints, ket, work, 0.0, [&](DetRef e, double h) { if (in_window(h, lo, hi)) emit(e, h); });
    }
}

template <class Emit>
inline void enumerate_screened(const RHFIntegrals& ints, const HeatBathTable* hb, DetRef ket, RowWork& work, double cut, Emit&& emit) {
    fill_occ(ket, work.occ);
    singles_alpha(ints, ket, work, cut, emit);
    singles_beta(ints, ket, work, cut, emit);
    if (hb != nullptr && cut >= hb->eps_table()) {
        doubles_hb_aa(*hb, ket, work, cut, emit);
        doubles_hb_bb(*hb, ket, work, cut, emit);
        doubles_hb_ab(*hb, ket, work, cut, emit);
    } else {
        doubles_exact_aa(ints, ket, work, cut, emit);
        doubles_exact_bb(ints, ket, work, cut, emit);
        doubles_exact_ab(ints, ket, work, cut, emit);
    }
}

template <class Emit>
inline void enumerate_window(const RHFIntegrals& ints, const HeatBathTable* hb, DetRef ket, RowWork& work, double lo, double hi, Emit&& emit) {
    if (hi <= lo || hi <= 0.0) return;
    fill_occ(ket, work.occ);
    singles_window_alpha(ints, ket, work, lo, hi, emit);
    singles_window_beta(ints, ket, work, lo, hi, emit);
    doubles_window_aa(ints, hb, ket, work, lo, hi, emit);
    doubles_window_bb(ints, hb, ket, work, lo, hi, emit);
    doubles_window_ab(ints, hb, ket, work, lo, hi, emit);
}

template <class Emit>
inline void emit_nonzero(double h, i32 idx, Emit&& emit) {
    if (h != 0.0) emit(idx, h);
}

template <class Emit>
inline void enumerate_internal(const RHFIntegrals& ints, const DetSpace& ket_space, DetRef bra, RowWork& work, Emit&& emit) {
    work.ensure_seen(ket_space.alpha.size(), ket_space.beta.size());
    work.next_a();
    work.next_b();

    const i32 bra_alpha = ket_space.alpha.find(bra.alpha());
    const i32 bra_beta = ket_space.beta.find(bra.beta());

    find_single(ket_space.alpha, ket_space.alpha1, bra.alpha(), work.tmp_occ, work.seen_a, work.stamp_a, work.alpha_single);
    work.next_a();
    find_double(ket_space.alpha, ket_space.alpha2, bra.alpha(), work.tmp_occ, work.seen_a, work.stamp_a, work.alpha_double);

    find_single(ket_space.beta, ket_space.beta1, bra.beta(), work.tmp_occ, work.seen_b, work.stamp_b, work.beta_single);
    work.next_b();
    find_double(ket_space.beta, ket_space.beta2, bra.beta(), work.tmp_occ, work.seen_b, work.stamp_b, work.beta_double);

    if (bra_beta >= 0) {
        for (const auto& ex : work.alpha_single) {
            const i32 idx = ket_space.find_with_beta(bra_beta, ex.id);
            if (idx >= 0) emit_nonzero(ex.sign * SlaterCondon::single_a(ints, bra, ex.occ, ex.vir), idx, emit);
        }
        for (const auto& ex : work.alpha_double) {
            const i32 idx = ket_space.find_with_beta(bra_beta, ex.id);
            if (idx >= 0) emit_nonzero(ex.sign * SlaterCondon::double_aa(ints, ex.occ_i, ex.occ_j, ex.vir_a, ex.vir_b), idx, emit);
        }
    }
    if (bra_alpha >= 0) {
        for (const auto& ex : work.beta_single) {
            const i32 idx = ket_space.find_with_alpha(bra_alpha, ex.id);
            if (idx >= 0) emit_nonzero(ex.sign * SlaterCondon::single_b(ints, bra, ex.occ, ex.vir), idx, emit);
        }
        for (const auto& ex : work.beta_double) {
            const i32 idx = ket_space.find_with_alpha(bra_alpha, ex.id);
            if (idx >= 0) emit_nonzero(ex.sign * SlaterCondon::double_bb(ints, ex.occ_i, ex.occ_j, ex.vir_a, ex.vir_b), idx, emit);
        }
    }

    work.ensure_cross(ket_space.beta.size());
    work.next_cross();
    for (const auto& ex : work.beta_single) {
        const std::size_t pos = static_cast<std::size_t>(ex.id);
        work.cross_b[pos] = work.cross_stamp;
        work.cross_occ[pos] = ex.occ;
        work.cross_vir[pos] = ex.vir;
        work.cross_sign[pos] = ex.sign;
    }
    for (const auto& ax : work.alpha_single) {
        for (const Mate& e : ket_space.alpha_mates(ax.id)) {
            const std::size_t beta_id = static_cast<std::size_t>(e.other);
            if (work.cross_b[beta_id] != work.cross_stamp) continue;
            const double h = ax.sign * work.cross_sign[beta_id] * SlaterCondon::double_ab(ints, ax.occ, work.cross_occ[beta_id], ax.vir, work.cross_vir[beta_id]);
            emit_nonzero(h, e.det, emit);
        }
    }
}

} // namespace libdet
