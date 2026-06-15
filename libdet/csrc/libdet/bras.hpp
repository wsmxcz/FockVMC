#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libdet/det.hpp>
#include <libdet/slater.hpp>

namespace libdet {

struct Candidate {
    int a = 0;
    int b = 0;
    double h = 0.0;
};

class Screen {
public:
    Screen(const RHFIntegrals& ints, double cutoff)
        : norb_(ints.norb()), cutoff_(cutoff) {
        if (std::isnan(cutoff_) || cutoff_ <= 0.0) {
            throw std::invalid_argument("Screen: cutoff must be positive");
        }

        const std::size_t n = static_cast<std::size_t>(norb_);
        const std::size_t np = n * (n + 1u) / 2u;

        same_off_.assign(np + 1u, 0u);
        opposite_off_.assign(n * n + 1u, 0u);

        build_same(ints);
        build_opposite(ints);
    }

    [[nodiscard]] double cutoff() const noexcept {
        return cutoff_;
    }

    [[nodiscard]] std::span<const Candidate> same(
        int i,
        int j,
        double lo = 0.0,
        double hi = std::numeric_limits<double>::infinity()
    ) const noexcept {
        return window(same_candidates(i, j), lo, hi);
    }

    [[nodiscard]] std::span<const Candidate> opposite(
        int ia,
        int ib,
        double lo = 0.0,
        double hi = std::numeric_limits<double>::infinity()
    ) const noexcept {
        const std::size_t k =
            static_cast<std::size_t>(ia) * static_cast<std::size_t>(norb_)
            + static_cast<std::size_t>(ib);
        const std::size_t begin = opposite_off_[k];
        const std::size_t end = opposite_off_[k + 1u];

        return window(
            {opposite_data_.data() + begin, end - begin},
            lo,
            hi
        );
    }

private:
    int norb_ = 0;
    double cutoff_ = 0.0;
    std::vector<std::size_t> same_off_;
    std::vector<Candidate> same_data_;
    std::vector<std::size_t> opposite_off_;
    std::vector<Candidate> opposite_data_;

    [[nodiscard]] std::span<const Candidate> same_candidates(
        int i,
        int j
    ) const noexcept {
        const std::size_t k = RHFIntegrals::pair_index(i, j);
        const std::size_t lo = same_off_[k];
        const std::size_t hi = same_off_[k + 1u];
        return {same_data_.data() + lo, hi - lo};
    }

    [[nodiscard]] bool keep(double h) const noexcept {
        return std::abs(h) >= cutoff_;
    }

    [[nodiscard]] static double abs_h(const Candidate& candidate) noexcept {
        return std::abs(candidate.h);
    }

    [[nodiscard]] static bool before(
        const Candidate& lhs,
        const Candidate& rhs
    ) noexcept {
        const double a = abs_h(lhs);
        const double b = abs_h(rhs);

        if (a != b) return a > b;
        if (lhs.a != rhs.a) return lhs.a < rhs.a;
        if (lhs.b != rhs.b) return lhs.b < rhs.b;
        return lhs.h < rhs.h;
    }

    [[nodiscard]] static std::size_t first_lt(
        std::span<const Candidate> candidates,
        double cutoff
    ) noexcept {
        if (cutoff <= 0.0) return candidates.size();

        std::size_t lo = 0;
        std::size_t hi = candidates.size();

        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2u;
            if (abs_h(candidates[mid]) < cutoff) hi = mid;
            else lo = mid + 1u;
        }

        return lo;
    }

    [[nodiscard]] static std::span<const Candidate> window(
        std::span<const Candidate> candidates,
        double lo,
        double hi
    ) noexcept {
        if (candidates.empty() || hi <= lo) return {};

        const std::size_t begin =
            std::isfinite(hi) ? first_lt(candidates, hi) : 0u;
        const std::size_t end = first_lt(candidates, lo);

        if (end <= begin) return {};
        return {candidates.data() + begin, end - begin};
    }

    void build_same(const RHFIntegrals& ints) {
        std::vector<Candidate> candidates;

        for (int hi = 0; hi < norb_; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) {
                const std::size_t pair = RHFIntegrals::pair_index(hi, lo);
                same_off_[pair] = same_data_.size();
                if (hi == lo) continue;

                candidates.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = a + 1; b < norb_; ++b) {
                        const double h = Slater::double_aa(
                            ints,
                            lo,
                            hi,
                            a,
                            b
                        );
                        if (keep(h)) candidates.push_back({a, b, h});
                    }
                }

                std::sort(candidates.begin(), candidates.end(), before);
                same_data_.insert(
                    same_data_.end(),
                    candidates.begin(),
                    candidates.end()
                );
            }
        }

        same_off_.back() = same_data_.size();
    }

    void build_opposite(const RHFIntegrals& ints) {
        std::vector<Candidate> candidates;

        for (int ia = 0; ia < norb_; ++ia) {
            for (int ib = 0; ib < norb_; ++ib) {
                const std::size_t pair =
                    static_cast<std::size_t>(ia)
                    * static_cast<std::size_t>(norb_)
                    + static_cast<std::size_t>(ib);
                opposite_off_[pair] = opposite_data_.size();
                candidates.clear();

                for (int a = 0; a < norb_; ++a) {
                    for (int b = 0; b < norb_; ++b) {
                        const double h = Slater::double_ab(
                            ints,
                            ia,
                            ib,
                            a,
                            b
                        );
                        if (keep(h)) candidates.push_back({a, b, h});
                    }
                }

                std::sort(candidates.begin(), candidates.end(), before);
                opposite_data_.insert(
                    opposite_data_.end(),
                    candidates.begin(),
                    candidates.end()
                );
            }
        }

        opposite_off_.back() = opposite_data_.size();
    }
};

struct KetScratch {
    explicit KetScratch(int norb)
        : occ(norb) {}

    DetOcc occ;
};

struct AbsWindow {
    double lo = 0.0;
    double hi = std::numeric_limits<double>::infinity();
};

enum class ExcitationKind : unsigned char {
    alpha1,
    beta1,
    alpha2,
    beta2,
    mixed2,
};

struct Excitation {
    ExcitationKind kind = ExcitationKind::alpha1;
    int i = 0;
    int j = 0;
    int a = 0;
    int b = 0;
};

struct Coupling {
    Excitation excitation;
    double h = 0.0;
};

[[nodiscard]] inline bool excitation_less(
    Excitation lhs,
    Excitation rhs
) noexcept {
    if (lhs.kind != rhs.kind) {
        return static_cast<unsigned char>(lhs.kind)
            < static_cast<unsigned char>(rhs.kind);
    }
    if (lhs.i != rhs.i) return lhs.i < rhs.i;
    if (lhs.j != rhs.j) return lhs.j < rhs.j;
    if (lhs.a != rhs.a) return lhs.a < rhs.a;
    return lhs.b < rhs.b;
}

[[nodiscard]] inline Excitation alpha1(int i, int a) noexcept {
    return Excitation{ExcitationKind::alpha1, i, 0, a, 0};
}

[[nodiscard]] inline Excitation beta1(int i, int a) noexcept {
    return Excitation{ExcitationKind::beta1, i, 0, a, 0};
}

[[nodiscard]] inline Excitation alpha2(int i, int j, int a, int b) noexcept {
    return Excitation{ExcitationKind::alpha2, i, j, a, b};
}

[[nodiscard]] inline Excitation beta2(int i, int j, int a, int b) noexcept {
    return Excitation{ExcitationKind::beta2, i, j, a, b};
}

[[nodiscard]] inline Excitation mixed2(int i, int j, int a, int b) noexcept {
    return Excitation{ExcitationKind::mixed2, i, j, a, b};
}

[[nodiscard]] inline bool in_window(double h, AbsWindow win) noexcept {
    const double value = std::abs(h);
    return value > 0.0 && value >= win.lo && value < win.hi;
}

inline DetRef apply(DetRef ket, Excitation excitation, DetScratch& scratch) {
    scratch.load(ket);

    auto alpha = scratch.alpha();
    auto beta = scratch.beta();

    switch (excitation.kind) {
    case ExcitationKind::alpha1:
        bits::clear(alpha, excitation.i);
        bits::set(alpha, excitation.a);
        break;
    case ExcitationKind::beta1:
        bits::clear(beta, excitation.i);
        bits::set(beta, excitation.a);
        break;
    case ExcitationKind::alpha2:
        bits::clear(alpha, excitation.i);
        bits::clear(alpha, excitation.j);
        bits::set(alpha, excitation.a);
        bits::set(alpha, excitation.b);
        break;
    case ExcitationKind::beta2:
        bits::clear(beta, excitation.i);
        bits::clear(beta, excitation.j);
        bits::set(beta, excitation.a);
        bits::set(beta, excitation.b);
        break;
    case ExcitationKind::mixed2:
        bits::clear(alpha, excitation.i);
        bits::clear(beta, excitation.j);
        bits::set(alpha, excitation.a);
        bits::set(beta, excitation.b);
        break;
    }

    return scratch.view();
}

template <class Visit>
inline void visit_bras_prepared(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    DetOcc& occ,
    AbsWindow win,
    Visit&& visit
) {
    if (win.hi <= win.lo) return;

    for (int i : occ.occ_a) {
        for (int a : occ.vir_a) {
            const double h =
                Slater::sign_single(occ.pref_a, i, a)
                * Slater::single_a(ints, occ.occ_a, occ.occ_b, i, a);
            if (in_window(h, win)) visit(alpha1(i, a), h);
        }
    }

    for (int i : occ.occ_b) {
        for (int a : occ.vir_b) {
            const double h =
                Slater::sign_single(occ.pref_b, i, a)
                * Slater::single_b(ints, occ.occ_a, occ.occ_b, i, a);
            if (in_window(h, win)) visit(beta1(i, a), h);
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
                        if (in_window(h, win)) visit(alpha2(i, j, a, b), h);
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
                        if (in_window(h, win)) visit(beta2(i, j, a, b), h);
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
                        if (in_window(h, win)) visit(mixed2(ia, ib, a, b), h);
                    }
                }
            }
        }

        return;
    }

    for (std::size_t x = 0; x < occ.occ_a.size(); ++x) {
        const int i = occ.occ_a[x];

        for (std::size_t y = x + 1u; y < occ.occ_a.size(); ++y) {
            const int j = occ.occ_a[y];

            for (const Candidate& c : screen->same(i, j, win.lo, win.hi)) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.alpha(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_a, i, j, c.a, c.b) * c.h;
                if (in_window(h, win)) visit(alpha2(i, j, c.a, c.b), h);
            }
        }
    }

    for (std::size_t x = 0; x < occ.occ_b.size(); ++x) {
        const int i = occ.occ_b[x];

        for (std::size_t y = x + 1u; y < occ.occ_b.size(); ++y) {
            const int j = occ.occ_b[y];

            for (const Candidate& c : screen->same(i, j, win.lo, win.hi)) {
                if (bits::test(ket.beta(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_double(occ.pref_b, i, j, c.a, c.b) * c.h;
                if (in_window(h, win)) visit(beta2(i, j, c.a, c.b), h);
            }
        }
    }

    for (int ia : occ.occ_a) {
        for (int ib : occ.occ_b) {
            for (
                const Candidate& c
                : screen->opposite(ia, ib, win.lo, win.hi)
            ) {
                if (bits::test(ket.alpha(), c.a) || bits::test(ket.beta(), c.b)) {
                    continue;
                }

                const double h =
                    Slater::sign_single(occ.pref_a, ia, c.a)
                    * Slater::sign_single(occ.pref_b, ib, c.b)
                    * c.h;
                if (in_window(h, win)) visit(mixed2(ia, ib, c.a, c.b), h);
            }
        }
    }
}

template <class Visit>
inline void visit_bras(
    const RHFIntegrals& ints,
    const Screen* screen,
    DetRef ket,
    KetScratch& scratch,
    AbsWindow win,
    Visit&& visit
) {
    fill_occ(ket, ints.norb(), scratch.occ);
    visit_bras_prepared(
        ints,
        screen,
        ket,
        scratch.occ,
        win,
        std::forward<Visit>(visit)
    );
}

} // namespace libdet
