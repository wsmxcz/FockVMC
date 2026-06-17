#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <libdet/spatial/determinant.hpp>

namespace libdet::guga {

using Cfg = std::vector<unsigned char>;

enum class Step : unsigned char {
    empty = 0,
    lower = 1,
    upper = 2,
    doubly = 3,
};

struct Sector {
    int norb = 0;
    int nelec = 0;
    int spin_twice = 0;
    u32 nword = 0;
};

struct Csf {
    std::vector<Step> step;
    Cfg cfg;
    std::vector<int> spin;

    [[nodiscard]] int norb() const noexcept {
        return static_cast<int>(step.size());
    }

    [[nodiscard]] int spin_twice() const noexcept {
        return spin.empty() ? 0 : spin.back();
    }
};

[[nodiscard]] inline Step step_at(DetRef det, int p) noexcept {
    const bool upper = bits::test(det.alpha(), p);
    const bool lower = bits::test(det.beta(), p);
    return static_cast<Step>((upper ? 2u : 0u) + (lower ? 1u : 0u));
}

[[nodiscard]] inline int step_occ(Step step) noexcept {
    switch (step) {
    case Step::empty:
        return 0;
    case Step::lower:
    case Step::upper:
        return 1;
    case Step::doubly:
        return 2;
    }
    return 0;
}

[[nodiscard]] inline int step_spin_delta(Step step) noexcept {
    switch (step) {
    case Step::upper:
        return 1;
    case Step::lower:
        return -1;
    case Step::empty:
    case Step::doubly:
        return 0;
    }
    return 0;
}

inline void check_csf(DetRef det, Sector sector, const char* where) {
    if (det.nword() != sector.nword) {
        throw std::invalid_argument(std::string(where) + ": det nword mismatch");
    }

    int nelec = 0;
    int spin = 0;
    for (int p = 0; p < sector.norb; ++p) {
        const Step step = step_at(det, p);
        nelec += step_occ(step);
        spin += step_spin_delta(step);
        if (spin < 0) {
            throw std::invalid_argument(std::string(where) + ": invalid GUGA CSF");
        }
    }

    if (nelec != sector.nelec || spin != sector.spin_twice) {
        throw std::invalid_argument(std::string(where) + ": wrong GUGA sector");
    }
}

[[nodiscard]] inline Csf make_csf(
    std::span<const Step> steps,
    Sector sector,
    const char* where
) {
    if (static_cast<int>(steps.size()) != sector.norb) {
        throw std::invalid_argument(std::string(where) + ": CSF size mismatch");
    }

    Csf csf;
    csf.step.assign(steps.begin(), steps.end());
    csf.cfg.resize(static_cast<std::size_t>(sector.norb));
    csf.spin.assign(static_cast<std::size_t>(sector.norb + 1), 0);

    int nelec = 0;
    for (int p = 0; p < sector.norb; ++p) {
        const Step step = steps[static_cast<std::size_t>(p)];
        nelec += step_occ(step);
        csf.cfg[static_cast<std::size_t>(p)] =
            static_cast<unsigned char>(step_occ(step));
        csf.spin[static_cast<std::size_t>(p + 1)] =
            csf.spin[static_cast<std::size_t>(p)] + step_spin_delta(step);

        if (csf.spin[static_cast<std::size_t>(p + 1)] < 0) {
            throw std::invalid_argument(std::string(where) + ": invalid GUGA CSF");
        }
    }

    if (nelec != sector.nelec || csf.spin_twice() != sector.spin_twice) {
        throw std::invalid_argument(std::string(where) + ": wrong GUGA sector");
    }

    return csf;
}

[[nodiscard]] inline Csf decode_csf(
    DetRef det,
    Sector sector,
    const char* where
) {
    check_csf(det, sector, where);

    std::vector<Step> steps(static_cast<std::size_t>(sector.norb));
    for (int p = 0; p < sector.norb; ++p) {
        steps[static_cast<std::size_t>(p)] = step_at(det, p);
    }
    return make_csf(steps, sector, where);
}

inline void encode_csf(
    std::span<const Step> steps,
    u32 nword,
    std::vector<u64>& out
) {
    out.assign(det_size(nword), 0u);

    std::span<u64> upper(out.data(), static_cast<std::size_t>(nword));
    std::span<u64> lower(
        out.data() + static_cast<std::size_t>(nword),
        static_cast<std::size_t>(nword)
    );

    for (int p = 0; p < static_cast<int>(steps.size()); ++p) {
        switch (steps[static_cast<std::size_t>(p)]) {
        case Step::empty:
            break;
        case Step::lower:
            bits::set(lower, p);
            break;
        case Step::upper:
            bits::set(upper, p);
            break;
        case Step::doubly:
            bits::set(upper, p);
            bits::set(lower, p);
            break;
        }
    }
}

[[nodiscard]] inline int cfg_degree(const Cfg& bra_cfg, const Cfg& ket_cfg) {
    int moved = 0;
    for (std::size_t p = 0; p < ket_cfg.size(); ++p) {
        const int diff =
            static_cast<int>(bra_cfg[p]) - static_cast<int>(ket_cfg[p]);
        if (diff > 0) moved += diff;
    }
    return moved;
}

template <class Visit>
inline void visit_csfs_impl(
    const Cfg& cfg,
    Sector sector,
    int p,
    int spin,
    std::vector<Step>& steps,
    Visit&& visit
) {
    if (p == sector.norb) {
        if (spin == sector.spin_twice) visit(steps);
        return;
    }

    const unsigned char occ = cfg[static_cast<std::size_t>(p)];
    if (occ == 0 || occ == 2) {
        steps[static_cast<std::size_t>(p)] =
            occ == 0 ? Step::empty : Step::doubly;
        visit_csfs_impl(
            cfg,
            sector,
            p + 1,
            spin,
            steps,
            std::forward<Visit>(visit)
        );
        return;
    }

    steps[static_cast<std::size_t>(p)] = Step::upper;
    visit_csfs_impl(
        cfg,
        sector,
        p + 1,
        spin + 1,
        steps,
        std::forward<Visit>(visit)
    );

    if (spin > 0) {
        steps[static_cast<std::size_t>(p)] = Step::lower;
        visit_csfs_impl(
            cfg,
            sector,
            p + 1,
            spin - 1,
            steps,
            std::forward<Visit>(visit)
        );
    }
}

template <class Visit>
inline void visit_csfs(const Cfg& cfg, Sector sector, Visit&& visit) {
    std::vector<Step> steps(static_cast<std::size_t>(sector.norb), Step::empty);
    visit_csfs_impl(
        cfg,
        sector,
        0,
        0,
        steps,
        std::forward<Visit>(visit)
    );
}

} // namespace libdet::guga
