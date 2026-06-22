#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ankerl/unordered_dense.h>

#include <libdet/bit.hpp>
#include <libdet/hash.hpp>

namespace libdet::guga {

using Occ = std::span<const unsigned char>;
using MutableOcc = std::span<unsigned char>;


enum class Step : unsigned char {
    empty = 0,
    down = 1,
    up = 2,
    doubly = 3,
};

struct Sector {
    int norb = 0;
    int nelec = 0;
    int spin_twice = 0;
    u32 nword = 0;
};

struct OccMove {
    int degree = 0;
    std::array<int, 2> remove{-1, -1};
    std::array<int, 2> add{-1, -1};
};

[[nodiscard]] inline constexpr std::size_t path_size(u32 nword) noexcept {
    return word_pair_size(nword);
}

// A GUGA path is stored as two packed Shavitt step bits: up then down.
class PathRef {
public:
    constexpr PathRef() noexcept = default;

    constexpr PathRef(const u64* up, const u64* down, u32 nword) noexcept
        : up_(up), down_(down), nword_(nword) {}

    [[nodiscard]] constexpr std::span<const u64> up() const noexcept {
        return {up_, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] constexpr std::span<const u64> down() const noexcept {
        return {down_, static_cast<std::size_t>(nword_)};
    }

    [[nodiscard]] constexpr u32 nword() const noexcept { return nword_; }

private:
    const u64* up_ = nullptr;
    const u64* down_ = nullptr;
    u32 nword_ = 0;
};

struct PathBatchView {
    const u64* data = nullptr;
    std::size_t n_paths = 0;
    u32 nword = 0;

    [[nodiscard]] PathRef operator[](std::size_t idx) const noexcept {
        const std::size_t stride = path_size(nword);
        const u64* ptr = data + idx * stride;
        return PathRef(ptr, ptr + static_cast<std::size_t>(nword), nword);
    }
};

[[nodiscard]] inline PathRef path_at(
    const std::vector<u64>& data,
    u32 nword,
    std::size_t idx
) noexcept {
    const std::size_t stride = path_size(nword);
    const u64* ptr = data.data() + idx * stride;
    return PathRef(ptr, ptr + static_cast<std::size_t>(nword), nword);
}

inline void append_path(std::vector<u64>& out, PathRef path) {
    out.insert(out.end(), path.up().begin(), path.up().end());
    out.insert(out.end(), path.down().begin(), path.down().end());
}

inline void copy_paths(std::vector<u64>& words, PathBatchView paths) {
    words.resize(paths.n_paths * path_size(paths.nword));
    if (!words.empty()) std::copy_n(paths.data, words.size(), words.data());
}

[[nodiscard]] inline bool path_equal(PathRef a, PathRef b) noexcept {
    return a.nword() == b.nword()
        && bits::equal(a.up(), b.up())
        && bits::equal(a.down(), b.down());
}

struct PathLess {
    [[nodiscard]] bool operator()(PathRef lhs, PathRef rhs) const noexcept {
        const auto lu = lhs.up();
        const auto ru = rhs.up();

        for (std::size_t w = 0; w < lu.size(); ++w) {
            if (lu[w] != ru[w]) return lu[w] < ru[w];
        }

        const auto ld = lhs.down();
        const auto rd = rhs.down();
        for (std::size_t w = 0; w < ld.size(); ++w) {
            if (ld[w] != rd[w]) return ld[w] < rd[w];
        }

        return false;
    }
};

struct PathHash {
    [[nodiscard]] std::size_t operator()(PathRef path) const noexcept {
        u64 h = 0x706174686b657973ULL ^ static_cast<u64>(path.nword());
        for (u64 x : path.up()) h = mix64(h ^ x);
        for (u64 x : path.down()) h = mix64(h ^ (x + 0x517cc1b727220a95ULL));
        return static_cast<std::size_t>(h);
    }
};

[[nodiscard]] inline u64 path_fingerprint(PathRef path, u64 seed = 0) noexcept {
    u64 h = mix64(seed ^ 0x706174686b657973ULL);
    for (u64 x : path.up()) h = mix64(h ^ x);
    for (u64 x : path.down()) h = mix64(h ^ (x + 0x517cc1b727220a95ULL));
    return h;
}

inline void sort_unique_paths(std::vector<u64>& packed, u32 nword) {
    const std::size_t stride = path_size(nword);
    if (stride == 0 || packed.empty()) return;

    const std::size_t n_paths = packed.size() / stride;
    std::vector<std::size_t> order(n_paths);
    for (std::size_t ipath = 0; ipath < n_paths; ++ipath) order[ipath] = ipath;

    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return PathLess{}(path_at(packed, nword, lhs), path_at(packed, nword, rhs));
    });

    std::vector<u64> out;
    out.reserve(packed.size());

    std::size_t prev = static_cast<std::size_t>(-1);
    for (std::size_t ipath : order) {
        if (
            prev != static_cast<std::size_t>(-1)
            && path_equal(path_at(packed, nword, ipath), path_at(packed, nword, prev))
        ) {
            continue;
        }

        append_path(out, path_at(packed, nword, ipath));
        prev = ipath;
    }

    packed.swap(out);
}

class PathIndex {
public:
    explicit PathIndex(PathBatchView paths) : paths_(paths) {
        index_.reserve(paths.n_paths);
        for (std::size_t i = 0; i < paths.n_paths; ++i) {
            insert(static_cast<i32>(i));
        }
    }

    [[nodiscard]] i32 find(PathRef path) const noexcept {
        const auto it = index_.find(path_fingerprint(path));
        if (it == index_.end()) return -1;

        for (i32 idx : it->second) {
            if (path_equal(paths_[static_cast<std::size_t>(idx)], path)) {
                return idx;
            }
        }

        return -1;
    }

private:
    PathBatchView paths_;
    ankerl::unordered_dense::map<u64, std::vector<i32>> index_;

    void insert(i32 idx) {
        index_[path_fingerprint(paths_[static_cast<std::size_t>(idx)])].push_back(idx);
    }
};

class PathPool {
public:
    explicit PathPool(u32 nword = 0) : nword_(nword) {}

    explicit PathPool(PathBatchView paths) : nword_(paths.nword) {
        copy_paths(words_, paths);
        build_index();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return nword_ == 0 ? 0u : words_.size() / path_size(nword_);
    }

    [[nodiscard]] PathRef get(std::size_t idx) const noexcept {
        return path_at(words_, nword_, idx);
    }

    [[nodiscard]] std::vector<u64>& words() noexcept {
        return words_;
    }

    [[nodiscard]] const std::vector<u64>& words() const noexcept {
        return words_;
    }

    [[nodiscard]] i32 find_or_add(PathRef path) {
        const u64 fingerprint = path_fingerprint(path);
        const auto it = index_.find(fingerprint);
        if (it != index_.end()) {
            for (i32 idx : it->second) {
                if (path_equal(get(static_cast<std::size_t>(idx)), path)) {
                    return idx;
                }
            }
        }

        const i32 fresh = to_i32(size());
        append_path(words_, path);
        index_[fingerprint].push_back(fresh);
        return fresh;
    }

private:
    u32 nword_ = 0;
    std::vector<u64> words_;
    ankerl::unordered_dense::map<u64, std::vector<i32>> index_;

    void build_index() {
        index_.clear();
        index_.reserve(size());
        for (std::size_t i = 0; i < size(); ++i) {
            index_[path_fingerprint(get(i))].push_back(static_cast<i32>(i));
        }
    }
};

struct PathState {
    std::span<Step> step;
    MutableOcc occ;
    std::span<int> spin;
    std::span<int> donor;
    std::span<int> hole;
    std::span<int> open;

    [[nodiscard]] int norb() const noexcept {
        return static_cast<int>(step.size());
    }

    [[nodiscard]] int spin_twice() const noexcept {
        return spin.empty() ? 0 : spin.back();
    }
};

struct PathScratch {
    std::vector<Step> step;
    std::vector<unsigned char> occ;
    std::vector<int> spin;
    std::vector<int> donor;
    std::vector<int> hole;
    std::vector<int> open;
    std::vector<u64> words;
};

inline void resize_path(PathScratch& scratch, Sector sector) {
    scratch.step.resize(static_cast<std::size_t>(sector.norb));
    scratch.occ.resize(static_cast<std::size_t>(sector.norb));
    scratch.spin.resize(static_cast<std::size_t>(sector.norb + 1));
    scratch.donor.resize(static_cast<std::size_t>(sector.norb));
    scratch.hole.resize(static_cast<std::size_t>(sector.norb));
    scratch.open.resize(static_cast<std::size_t>(sector.norb));
}

inline void bind_path(PathScratch& scratch, int ndonor, int nhole, int nopen, PathState& path) noexcept {
    path.step = std::span<Step>(scratch.step.data(), scratch.step.size());
    path.occ = MutableOcc(scratch.occ.data(), scratch.occ.size());
    path.spin = std::span<int>(scratch.spin.data(), scratch.spin.size());
    path.donor = std::span<int>(scratch.donor.data(), static_cast<std::size_t>(ndonor));
    path.hole = std::span<int>(scratch.hole.data(), static_cast<std::size_t>(nhole));
    path.open = std::span<int>(scratch.open.data(), static_cast<std::size_t>(nopen));
}

[[nodiscard]] inline Step step_at(PathRef path, int p) noexcept {
    const bool up = bits::test(path.up(), p);
    const bool down = bits::test(path.down(), p);
    return static_cast<Step>((up ? 2u : 0u) + (down ? 1u : 0u));
}

[[nodiscard]] inline int step_occ(Step step) noexcept {
    switch (step) {
    case Step::empty:
        return 0;
    case Step::down:
    case Step::up:
        return 1;
    case Step::doubly:
        return 2;
    }
    return 0;
}

[[nodiscard]] inline int step_spin_delta(Step step) noexcept {
    switch (step) {
    case Step::up:
        return 1;
    case Step::down:
        return -1;
    case Step::empty:
    case Step::doubly:
        return 0;
    }
    return 0;
}


[[nodiscard]] inline std::size_t occ_key_nword(int norb) noexcept {
    return static_cast<std::size_t>((2 * norb + 63) >> 6);
}

inline void pack_occ_key(std::span<const unsigned char> occ, std::vector<u64>& out) {
    out.assign(occ_key_nword(static_cast<int>(occ.size())), 0u);
    for (std::size_t p = 0; p < occ.size(); ++p) {
        const unsigned bit = static_cast<unsigned>((2u * p) & 63u);
        const std::size_t word = (2u * p) >> 6u;
        out[word] |= (static_cast<u64>(occ[p]) & 3u) << bit;
    }
}

[[nodiscard]] inline bool same_occ_key(
    const std::vector<u64>& lhs,
    const std::vector<u64>& rhs
) noexcept {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

[[nodiscard]] inline bool occ_key_less(
    const std::vector<u64>& lhs,
    const std::vector<u64>& rhs
) noexcept {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

inline void check_path(PathRef path, Sector sector, const char* where) {
    if (path.nword() != sector.nword) {
        throw std::invalid_argument(std::string(where) + ": path nword mismatch");
    }

    int nelec = 0;
    int spin = 0;
    for (int p = 0; p < sector.norb; ++p) {
        const Step step = step_at(path, p);
        nelec += step_occ(step);
        spin += step_spin_delta(step);
        if (spin < 0) {
            throw std::invalid_argument(std::string(where) + ": invalid GUGA path");
        }
    }

    if (nelec != sector.nelec || spin != sector.spin_twice) {
        throw std::invalid_argument(std::string(where) + ": wrong GUGA sector");
    }
}

inline void finish_path(Sector sector, const char* where, PathScratch& scratch, PathState& path) {
    int ndonor = 0;
    int nhole = 0;
    int nopen = 0;
    int nelec = 0;
    std::fill(scratch.spin.begin(), scratch.spin.end(), 0);

    for (int p = 0; p < sector.norb; ++p) {
        const Step step = scratch.step[static_cast<std::size_t>(p)];
        const int occ = step_occ(step);
        nelec += occ;
        scratch.occ[static_cast<std::size_t>(p)] = static_cast<unsigned char>(occ);
        scratch.spin[static_cast<std::size_t>(p + 1)] =
            scratch.spin[static_cast<std::size_t>(p)] + step_spin_delta(step);

        if (scratch.spin[static_cast<std::size_t>(p + 1)] < 0) {
            throw std::invalid_argument(std::string(where) + ": invalid GUGA path");
        }
        if (occ > 0) scratch.donor[static_cast<std::size_t>(ndonor++)] = p;
        if (occ < 2) scratch.hole[static_cast<std::size_t>(nhole++)] = p;
        if (occ == 1) scratch.open[static_cast<std::size_t>(nopen++)] = p;
    }

    if (nelec != sector.nelec || scratch.spin[static_cast<std::size_t>(sector.norb)] != sector.spin_twice) {
        throw std::invalid_argument(std::string(where) + ": wrong GUGA sector");
    }

    bind_path(scratch, ndonor, nhole, nopen, path);
}

inline void load_path(
    std::span<const Step> steps,
    Sector sector,
    const char* where,
    PathScratch& scratch,
    PathState& path
) {
    if (static_cast<int>(steps.size()) != sector.norb) {
        throw std::invalid_argument(std::string(where) + ": path size mismatch");
    }

    resize_path(scratch, sector);
    std::copy(steps.begin(), steps.end(), scratch.step.begin());
    finish_path(sector, where, scratch, path);
}

inline void decode_path(
    PathRef ref,
    Sector sector,
    const char* where,
    PathScratch& scratch,
    PathState& path
) {
    if (ref.nword() != sector.nword) {
        throw std::invalid_argument(std::string(where) + ": path nword mismatch");
    }

    resize_path(scratch, sector);
    int ndonor = 0;
    int nhole = 0;
    int nopen = 0;
    int nelec = 0;
    scratch.spin[0] = 0;

    for (int p = 0; p < sector.norb; ++p) {
        const Step step = step_at(ref, p);
        const int occ = step_occ(step);
        const int next_spin = scratch.spin[static_cast<std::size_t>(p)] + step_spin_delta(step);

        if (next_spin < 0) {
            throw std::invalid_argument(std::string(where) + ": invalid GUGA path");
        }

        scratch.step[static_cast<std::size_t>(p)] = step;
        scratch.occ[static_cast<std::size_t>(p)] = static_cast<unsigned char>(occ);
        scratch.spin[static_cast<std::size_t>(p + 1)] = next_spin;
        nelec += occ;

        if (occ > 0) scratch.donor[static_cast<std::size_t>(ndonor++)] = p;
        if (occ < 2) scratch.hole[static_cast<std::size_t>(nhole++)] = p;
        if (occ == 1) scratch.open[static_cast<std::size_t>(nopen++)] = p;
    }

    if (nelec != sector.nelec || scratch.spin[static_cast<std::size_t>(sector.norb)] != sector.spin_twice) {
        throw std::invalid_argument(std::string(where) + ": wrong GUGA sector");
    }

    bind_path(scratch, ndonor, nhole, nopen, path);
}

[[nodiscard]] inline bool move_one_occ(
    std::span<const unsigned char> occ,
    int from,
    int to,
    std::vector<unsigned char>& out
) {
    out.assign(occ.begin(), occ.end());
    if (from == to) return true;
    if (out[static_cast<std::size_t>(from)] == 0u) return false;
    if (out[static_cast<std::size_t>(to)] >= 2u) return false;
    --out[static_cast<std::size_t>(from)];
    ++out[static_cast<std::size_t>(to)];
    return true;
}

[[nodiscard]] inline bool remove_pair_occ(
    std::span<const unsigned char> occ,
    int q,
    int s,
    std::vector<unsigned char>& out
) {
    out.assign(occ.begin(), occ.end());
    if (q == s) {
        if (out[static_cast<std::size_t>(q)] < 2u) return false;
        out[static_cast<std::size_t>(q)] =
            static_cast<unsigned char>(out[static_cast<std::size_t>(q)] - 2u);
        return true;
    }
    if (out[static_cast<std::size_t>(q)] == 0u) return false;
    if (out[static_cast<std::size_t>(s)] == 0u) return false;
    --out[static_cast<std::size_t>(q)];
    --out[static_cast<std::size_t>(s)];
    return true;
}

[[nodiscard]] inline bool add_pair_occ(
    std::span<const unsigned char> occ,
    int p,
    int r,
    std::vector<unsigned char>& out
) {
    out.assign(occ.begin(), occ.end());
    if (p == r) {
        if (out[static_cast<std::size_t>(p)] != 0u) return false;
        out[static_cast<std::size_t>(p)] =
            static_cast<unsigned char>(out[static_cast<std::size_t>(p)] + 2u);
        return true;
    }
    if (out[static_cast<std::size_t>(p)] >= 2u) return false;
    if (out[static_cast<std::size_t>(r)] >= 2u) return false;
    ++out[static_cast<std::size_t>(p)];
    ++out[static_cast<std::size_t>(r)];
    return true;
}

[[nodiscard]] inline OccMove occ_move(
    std::span<const unsigned char> ket,
    std::span<const unsigned char> bra
) noexcept {
    OccMove move;
    int nr = 0;
    int na = 0;

    auto invalid = [&]() noexcept {
        move.degree = 3;
        move.remove = {-1, -1};
        move.add = {-1, -1};
        return move;
    };

    if (ket.size() != bra.size()) return invalid();

    for (std::size_t i = 0; i < ket.size(); ++i) {
        const int k = static_cast<int>(ket[i]);
        const int b = static_cast<int>(bra[i]);
        if (b > 2 || k > 2) return invalid();

        const int p = static_cast<int>(i);
        switch (b - k) {
        case -2:
            if (nr != 0) return invalid();
            move.remove[0] = p;
            move.remove[1] = p;
            nr = 2;
            break;
        case -1:
            if (nr >= 2) return invalid();
            move.remove[static_cast<std::size_t>(nr++)] = p;
            break;
        case 0:
            break;
        case 1:
            if (na >= 2) return invalid();
            move.add[static_cast<std::size_t>(na++)] = p;
            break;
        case 2:
            if (na != 0) return invalid();
            move.add[0] = p;
            move.add[1] = p;
            na = 2;
            break;
        default:
            return invalid();
        }
    }

    if (nr != na) return invalid();
    move.degree = nr;
    return move;
}

[[nodiscard]] inline bool same_path(
    const PathState& bra,
    const PathState& ket
) noexcept {
    return bra.step.size() == ket.step.size()
        && std::equal(bra.step.begin(), bra.step.end(), ket.step.begin());
}

inline void encode_path(
    std::span<const Step> steps,
    u32 nword,
    std::vector<u64>& out
) {
    out.assign(path_size(nword), 0u);

    std::span<u64> up(out.data(), static_cast<std::size_t>(nword));
    std::span<u64> down(
        out.data() + static_cast<std::size_t>(nword),
        static_cast<std::size_t>(nword)
    );

    for (int p = 0; p < static_cast<int>(steps.size()); ++p) {
        switch (steps[static_cast<std::size_t>(p)]) {
        case Step::empty:
            break;
        case Step::down:
            bits::set(down, p);
            break;
        case Step::up:
            bits::set(up, p);
            break;
        case Step::doubly:
            bits::set(up, p);
            bits::set(down, p);
            break;
        }
    }
}


struct PathGroup {
    Occ occ;
    std::span<const i32> ids;

    [[nodiscard]] bool empty() const noexcept { return ids.empty(); }
};

[[nodiscard]] inline u64 occ_tag(int p, unsigned char n) noexcept {
    if (n == 0u) return 0u;
    return mix64(
        0x6f63636667746167ULL
        ^ (static_cast<u64>(p) * 0x9e3779b97f4a7c15ULL)
        ^ (static_cast<u64>(n) * 0xbf58476d1ce4e5b9ULL)
    );
}

[[nodiscard]] inline u64 occ_fingerprint(Occ occ) noexcept {
    u64 fp = mix64(0x6f6363666770726fULL ^ static_cast<u64>(occ.size()));
    for (int p = 0; p < static_cast<int>(occ.size()); ++p) {
        fp ^= occ_tag(p, occ[static_cast<std::size_t>(p)]);
    }
    return fp;
}

[[nodiscard]] inline u64 remove_one_fingerprint(
    u64 fp,
    int p,
    unsigned char n
) noexcept {
    return fp ^ occ_tag(p, n) ^ occ_tag(p, static_cast<unsigned char>(n - 1u));
}

[[nodiscard]] inline bool can_remove_pair(Occ occ, int p, int q) noexcept {
    if (p == q) return occ[static_cast<std::size_t>(p)] >= 2u;
    return occ[static_cast<std::size_t>(p)] > 0u
        && occ[static_cast<std::size_t>(q)] > 0u;
}

[[nodiscard]] inline u64 remove_pair_fingerprint(
    u64 fp,
    Occ occ,
    int p,
    int q
) noexcept {
    const unsigned char np = occ[static_cast<std::size_t>(p)];
    if (p == q) {
        return fp ^ occ_tag(p, np) ^ occ_tag(p, static_cast<unsigned char>(np - 2u));
    }
    fp = remove_one_fingerprint(fp, p, np);
    return remove_one_fingerprint(fp, q, occ[static_cast<std::size_t>(q)]);
}

struct VisitScratch {
    std::vector<u32> seen;
    u32 stamp = 0;

    void start(std::size_t n_group) {
        if (seen.size() < n_group) seen.assign(n_group, 0u);
        if (++stamp == 0u) {
            std::fill(seen.begin(), seen.end(), 0u);
            stamp = 1u;
        }
    }

    [[nodiscard]] bool marked(i32 group) const noexcept {
        return seen[static_cast<std::size_t>(group)] == stamp;
    }

    void mark(i32 group) noexcept {
        seen[static_cast<std::size_t>(group)] = stamp;
    }
};

// Decoded path space with oCFG residue tables for known-space search.
class PathSpace {
public:
    PathSpace(PathBatchView paths, Sector sector)
        : words_(copy_words(paths)),
          paths_{words_.data(), paths.n_paths, paths.nword},
          sector_(sector),
          index_(paths_) {
        build();
    }

    [[nodiscard]] PathBatchView paths() const noexcept { return paths_; }

    [[nodiscard]] i32 find(PathRef path) const noexcept {
        return index_.find(path);
    }

    [[nodiscard]] const PathState& state(i32 idx) const noexcept {
        return states_[static_cast<std::size_t>(idx)];
    }

    [[nodiscard]] PathGroup occ_group(Occ occ, std::vector<u64>& key) const {
        pack_occ_key(occ, key);
        return occ_group(std::span<const u64>(key.data(), key.size()));
    }

    [[nodiscard]] PathGroup occ_group(std::span<const u64> key) const noexcept {
        const auto it = std::lower_bound(
            groups_.begin(),
            groups_.end(),
            key,
            [](const OccGroup& lhs, std::span<const u64> rhs) {
                return std::lexicographical_compare(
                    lhs.key.begin(), lhs.key.end(), rhs.begin(), rhs.end()
                );
            }
        );
        if (it == groups_.end() || !same_key(it->key, key)) return {};
        return group_view(static_cast<i32>(it - groups_.begin()));
    }

    template <class Visit>
    void visit(const PathState& ket, VisitScratch& scratch, Visit&& visit) const {
        scratch.start(groups_.size());
        const u64 ket_fp = occ_fingerprint(ket.occ);

        auto emit_group = [&](i32 group, int degree) {
            if (scratch.marked(group)) return;
            const OccGroup& occ_group = groups_[static_cast<std::size_t>(group)];
            const OccMove move = occ_move(ket.occ, occ_group.occ);
            if (move.degree != degree) return;
            scratch.mark(group);
            for (i32 idx : group_ids(occ_group)) visit(idx, move);
        };

        for (const OccRes& res : equal_res(full_, ket_fp)) {
            const OccGroup& group = groups_[static_cast<std::size_t>(res.group)];
            if (same_occ(group.occ, ket.occ)) emit_group(res.group, 0);
        }

        for (int q : ket.donor) {
            const auto nq = ket.occ[static_cast<std::size_t>(q)];
            const u64 fp = remove_one_fingerprint(ket_fp, q, nq);
            for (const OccRes& res : equal_res(one_, fp)) emit_group(res.group, 1);
        }

        for (std::size_t iq = 0; iq < ket.donor.size(); ++iq) {
            const int q = ket.donor[iq];
            for (std::size_t is = iq; is < ket.donor.size(); ++is) {
                const int s = ket.donor[is];
                if (!can_remove_pair(ket.occ, q, s)) continue;
                const u64 fp = remove_pair_fingerprint(ket_fp, ket.occ, q, s);
                for (const OccRes& res : equal_res(two_, fp)) emit_group(res.group, 2);
            }
        }
    }

private:
    struct OccRecord {
        std::vector<u64> key;
        i32 idx = 0;
    };

    struct OccGroup {
        std::vector<u64> key;
        std::vector<unsigned char> occ;
        u64 fp = 0;
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    struct OccRes {
        u64 fp = 0;
        i32 group = 0;
    };

    std::vector<u64> words_;
    PathBatchView paths_;
    Sector sector_;
    PathIndex index_;
    std::vector<i32> ids_;
    std::vector<OccGroup> groups_;
    std::vector<OccRes> full_;
    std::vector<OccRes> one_;
    std::vector<OccRes> two_;
    std::vector<PathScratch> scratches_;
    std::vector<PathState> states_;

    [[nodiscard]] static std::vector<u64> copy_words(PathBatchView paths) {
        std::vector<u64> out;
        copy_paths(out, paths);
        return out;
    }

    [[nodiscard]] static bool same_key(
        const std::vector<u64>& lhs,
        std::span<const u64> rhs
    ) noexcept {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    [[nodiscard]] static bool same_occ(
        Occ lhs,
        Occ rhs
    ) noexcept {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    [[nodiscard]] std::span<const i32> group_ids(const OccGroup& group) const noexcept {
        return std::span<const i32>(ids_.data() + group.begin, group.end - group.begin);
    }

    [[nodiscard]] PathGroup group_view(i32 group) const noexcept {
        const OccGroup& item = groups_[static_cast<std::size_t>(group)];
        return PathGroup{
            Occ(item.occ.data(), item.occ.size()),
            group_ids(item)
        };
    }

    [[nodiscard]] static std::span<const OccRes> equal_res(
        const std::vector<OccRes>& items,
        u64 fp
    ) noexcept {
        const auto first = std::lower_bound(
            items.begin(),
            items.end(),
            fp,
            [](const OccRes& lhs, u64 rhs) { return lhs.fp < rhs; }
        );
        const auto last = std::upper_bound(
            first,
            items.end(),
            fp,
            [](u64 lhs, const OccRes& rhs) { return lhs < rhs.fp; }
        );
        const std::size_t begin = static_cast<std::size_t>(first - items.begin());
        const std::size_t end = static_cast<std::size_t>(last - items.begin());
        const std::size_t size = end - begin;
        return std::span<const OccRes>(size == 0u ? nullptr : items.data() + begin, size);
    }

    static void sort_res(std::vector<OccRes>& items) {
        std::sort(items.begin(), items.end(), [](const OccRes& lhs, const OccRes& rhs) {
            if (lhs.fp != rhs.fp) return lhs.fp < rhs.fp;
            return lhs.group < rhs.group;
        });
        items.erase(
            std::unique(items.begin(), items.end(), [](const OccRes& lhs, const OccRes& rhs) {
                return lhs.fp == rhs.fp && lhs.group == rhs.group;
            }),
            items.end()
        );
    }

    void build() {
        std::vector<OccRecord> records;
        records.reserve(paths_.n_paths);
        scratches_.resize(paths_.n_paths);
        states_.resize(paths_.n_paths);

        for (std::size_t i = 0; i < paths_.n_paths; ++i) {
            decode_path(paths_[i], sector_, "PathSpace", scratches_[i], states_[i]);
            OccRecord rec;
            pack_occ_key(states_[i].occ, rec.key);
            rec.idx = to_i32(i);
            records.push_back(std::move(rec));
        }

        std::sort(records.begin(), records.end(), [](const OccRecord& lhs, const OccRecord& rhs) {
            if (lhs.key != rhs.key) return lhs.key < rhs.key;
            return lhs.idx < rhs.idx;
        });

        ids_.reserve(records.size());
        groups_.reserve(records.size());

        for (std::size_t i = 0; i < records.size();) {
            const std::size_t begin = ids_.size();
            std::vector<u64> key = records[i].key;
            std::vector<unsigned char> occ(
                states_[static_cast<std::size_t>(records[i].idx)].occ.begin(),
                states_[static_cast<std::size_t>(records[i].idx)].occ.end()
            );
            const u64 fp = occ_fingerprint(occ);

            do {
                ids_.push_back(records[i].idx);
                ++i;
            } while (i < records.size() && records[i].key == key);

            groups_.push_back(OccGroup{
                std::move(key),
                std::move(occ),
                fp,
                begin,
                ids_.size()
            });
        }

        build_residues();
    }

    void build_residues() {
        full_.reserve(groups_.size());
        one_.reserve(groups_.size() * static_cast<std::size_t>(sector_.nelec));
        two_.reserve(groups_.size() * static_cast<std::size_t>(sector_.nelec + 1));

        for (std::size_t igroup = 0; igroup < groups_.size(); ++igroup) {
            const auto group = static_cast<i32>(igroup);
            const OccGroup& item = groups_[igroup];
            const Occ occ(item.occ.data(), item.occ.size());

            full_.push_back(OccRes{item.fp, group});

            for (int p = 0; p < sector_.norb; ++p) {
                const unsigned char np = occ[static_cast<std::size_t>(p)];
                if (np == 0u) continue;
                one_.push_back(OccRes{remove_one_fingerprint(item.fp, p, np), group});
            }

            for (int p = 0; p < sector_.norb; ++p) {
                if (occ[static_cast<std::size_t>(p)] == 0u) continue;
                for (int q = p; q < sector_.norb; ++q) {
                    if (!can_remove_pair(occ, p, q)) continue;
                    two_.push_back(OccRes{remove_pair_fingerprint(item.fp, occ, p, q), group});
                }
            }
        }

        sort_res(full_);
        sort_res(one_);
        sort_res(two_);
    }
};


} // namespace libdet::guga
