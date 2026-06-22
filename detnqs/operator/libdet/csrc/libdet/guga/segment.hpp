#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <span>
#include <utility>
#include <vector>

#include <libdet/guga/path.hpp>

namespace libdet::guga {

// Local shape and reduced transitions for E_pq E_rs.
enum class Dir2 : unsigned char {
    diag = 0,
    r = 1,
    l = 2,
};

enum class Op2 : unsigned char {
    a_cre = 1u << 0u,
    a_ann = 1u << 1u,
    b_cre = 1u << 2u,
    b_ann = 1u << 3u,
};

enum class Line2 : unsigned char {
    a_open = 1u << 0u,
    a_close = 1u << 1u,
    b_open = 1u << 2u,
    b_close = 1u << 3u,
};

enum class Act2 : unsigned char {
    a = 1u << 0u,
    b = 1u << 1u,
};

[[nodiscard]] inline constexpr unsigned char bit(Op2 x) noexcept {
    return static_cast<unsigned char>(x);
}

[[nodiscard]] inline constexpr unsigned char bit(Line2 x) noexcept {
    return static_cast<unsigned char>(x);
}

[[nodiscard]] inline constexpr unsigned char bit(Act2 x) noexcept {
    return static_cast<unsigned char>(x);
}

struct Shape2 {
    unsigned char op = 0;
    unsigned char line = 0;
    unsigned char act = 0;
    Dir2 a = Dir2::diag;
    Dir2 b = Dir2::diag;

    [[nodiscard]] constexpr bool has(Op2 x) const noexcept { return (op & bit(x)) != 0u; }
    [[nodiscard]] constexpr bool has(Line2 x) const noexcept { return (line & bit(x)) != 0u; }
    [[nodiscard]] constexpr bool has(Act2 x) const noexcept { return (act & bit(x)) != 0u; }
    [[nodiscard]] constexpr bool event() const noexcept { return op != 0u || line != 0u || act != 0u; }
};

[[nodiscard]] inline constexpr bool operator==(Shape2 x, Shape2 y) noexcept {
    return x.op == y.op && x.line == y.line && x.act == y.act && x.a == y.a && x.b == y.b;
}

[[nodiscard]] inline constexpr Dir2 dir2(int make, int kill) noexcept {
    if (make == kill) return Dir2::diag;
    return make < kill ? Dir2::r : Dir2::l;
}

inline constexpr void add_bit(unsigned char& bits, Op2 x) noexcept {
    bits = static_cast<unsigned char>(bits | bit(x));
}

inline constexpr void add_bit(unsigned char& bits, Line2 x) noexcept {
    bits = static_cast<unsigned char>(bits | bit(x));
}

inline constexpr void add_bit(unsigned char& bits, Act2 x) noexcept {
    bits = static_cast<unsigned char>(bits | bit(x));
}

[[nodiscard]] inline constexpr Shape2 shape2(int k, int p, int q, int r, int s) noexcept {
    Shape2 out;
    out.a = dir2(p, q);
    out.b = dir2(r, s);

    if (k == p) add_bit(out.op, Op2::a_cre);
    if (k == q) add_bit(out.op, Op2::a_ann);
    if (k == r) add_bit(out.op, Op2::b_cre);
    if (k == s) add_bit(out.op, Op2::b_ann);

    const int alo = p < q ? p : q;
    const int ahi = p < q ? q : p;
    const int blo = r < s ? r : s;
    const int bhi = r < s ? s : r;

    if (alo <= k && k <= ahi) add_bit(out.act, Act2::a);
    if (blo <= k && k <= bhi) add_bit(out.act, Act2::b);

    if (k == alo) add_bit(out.line, Line2::a_open);
    if (k == ahi) add_bit(out.line, Line2::a_close);
    if (k == blo) add_bit(out.line, Line2::b_open);
    if (k == bhi) add_bit(out.line, Line2::b_close);
    return out;
}

[[nodiscard]] inline constexpr std::array<int, 2> hull2(int p, int q, int r, int s) noexcept {
    int lo = p < q ? p : q;
    lo = lo < r ? lo : r;
    lo = lo < s ? lo : s;
    int hi = p > q ? p : q;
    hi = hi > r ? hi : r;
    hi = hi > s ? hi : s;
    return {lo, hi};
}

// One-body Shavitt segment table.
enum class Seg1Kind : unsigned char {
    diag,
    r_head,
    r_tail,
    r_mid,
    l_head,
    l_tail,
    l_mid,
};

[[nodiscard]] inline double guga_a(int b, int x, int y) noexcept {
    const int num = b + x;
    const int den = b + y;
    if (num <= 0 || den <= 0) return 0.0;
    return std::sqrt(static_cast<double>(num) / static_cast<double>(den));
}

[[nodiscard]] inline double guga_c(int b, int x) noexcept {
    const int den = b + x;
    const int a = b + x - 1;
    const int c = b + x + 1;
    if (den == 0 || a < 0 || c < 0) return 0.0;
    return std::sqrt(static_cast<double>(a) * static_cast<double>(c)) / static_cast<double>(den);
}

[[nodiscard]] inline double seg1_diag(Step dx, Step dy) noexcept {
    if (dx != dy) return 0.0;
    if (dx == Step::down || dx == Step::up) return 1.0;
    if (dx == Step::doubly) return 2.0;
    return 0.0;
}

[[nodiscard]] inline double seg1_r_head(Step dx, Step dy, int b) noexcept {
    if (dx == Step::empty && (dy == Step::down || dy == Step::up)) return 1.0;
    if (dx == Step::down && dy == Step::doubly) return guga_a(b, 0, 1);
    if (dx == Step::up && dy == Step::doubly) return guga_a(b, 2, 1);
    return 0.0;
}

[[nodiscard]] inline double seg1_l_head(Step dx, Step dy, int b) noexcept {
    if (dx == Step::empty && (dy == Step::down || dy == Step::up)) return 1.0;
    if (dx == Step::down && dy == Step::doubly) return guga_a(b, 2, 1);
    if (dx == Step::up && dy == Step::doubly) return guga_a(b, 0, 1);
    return 0.0;
}

[[nodiscard]] inline double seg1_r_tail(Step dx, Step dy, int b) noexcept {
    if ((dx == Step::down || dx == Step::up) && dy == Step::empty) return 1.0;
    if (dx == Step::doubly && dy == Step::down) return guga_a(b, 1, 0);
    if (dx == Step::doubly && dy == Step::up) return guga_a(b, 1, 2);
    return 0.0;
}

[[nodiscard]] inline double seg1_l_tail(Step dx, Step dy, int b) noexcept {
    if ((dx == Step::down || dx == Step::up) && dy == Step::empty) return 1.0;
    if (dx == Step::doubly && dy == Step::down) return guga_a(b, 0, 1);
    if (dx == Step::doubly && dy == Step::up) return guga_a(b, 2, 1);
    return 0.0;
}

[[nodiscard]] inline double seg1_r_mid(Step dx, Step dy, int db, int b) noexcept {
    if (db == -1) {
        if (dx == Step::empty && dy == Step::empty) return 1.0;
        if (dx == Step::down && dy == Step::down) return -1.0;
        if (dx == Step::down && dy == Step::up) return b == -2 ? 0.0 : -1.0 / static_cast<double>(b + 2);
        if (dx == Step::up && dy == Step::up) return guga_c(b, 2);
        if (dx == Step::doubly && dy == Step::doubly) return -1.0;
        return 0.0;
    }
    if (db == 1) {
        if (dx == Step::empty && dy == Step::empty) return 1.0;
        if (dx == Step::down && dy == Step::down) return guga_c(b, 0);
        if (dx == Step::up && dy == Step::down) return b == 0 ? 0.0 : 1.0 / static_cast<double>(b);
        if (dx == Step::up && dy == Step::up) return -1.0;
        if (dx == Step::doubly && dy == Step::doubly) return -1.0;
    }
    return 0.0;
}

[[nodiscard]] inline double seg1_l_mid(Step dx, Step dy, int db, int b) noexcept {
    if (db == -1) {
        if (dx == Step::empty && dy == Step::empty) return 1.0;
        if (dx == Step::down && dy == Step::down) return guga_c(b, 1);
        if (dx == Step::down && dy == Step::up) return b == -1 ? 0.0 : 1.0 / static_cast<double>(b + 1);
        if (dx == Step::up && dy == Step::up) return -1.0;
        if (dx == Step::doubly && dy == Step::doubly) return -1.0;
        return 0.0;
    }
    if (db == 1) {
        if (dx == Step::empty && dy == Step::empty) return 1.0;
        if (dx == Step::down && dy == Step::down) return -1.0;
        if (dx == Step::up && dy == Step::down) return b == -1 ? 0.0 : -1.0 / static_cast<double>(b + 1);
        if (dx == Step::up && dy == Step::up) return guga_c(b, 1);
        if (dx == Step::doubly && dy == Step::doubly) return -1.0;
    }
    return 0.0;
}

[[nodiscard]] inline Step seg_step(Step step) noexcept {
    if (step == Step::down) return Step::up;
    if (step == Step::up) return Step::down;
    return step;
}

[[nodiscard]] inline double seg1(
    Seg1Kind kind,
    Step dx,
    Step dy,
    int db,
    int b
) noexcept {
    dx = seg_step(dx);
    dy = seg_step(dy);
    switch (kind) {
    case Seg1Kind::diag:
        return seg1_diag(dx, dy);
    case Seg1Kind::r_head:
        return seg1_r_head(dx, dy, b);
    case Seg1Kind::r_tail:
        return seg1_r_tail(dx, dy, b);
    case Seg1Kind::r_mid:
        return seg1_r_mid(dx, dy, db, b);
    case Seg1Kind::l_head:
        return seg1_l_head(dx, dy, b);
    case Seg1Kind::l_tail:
        return seg1_l_tail(dx, dy, b);
    case Seg1Kind::l_mid:
        return seg1_l_mid(dx, dy, db, b);
    }
    return 0.0;
}

[[nodiscard]] inline int seg_delta(Step step) noexcept {
    if (step == Step::up) return 1;
    if (step == Step::down) return -1;
    return 0;
}

[[nodiscard]] inline Step occ_step(int occ, Step pref) noexcept {
    if (occ == 0) return Step::empty;
    if (occ == 2) return Step::doubly;
    return pref;
}

[[nodiscard]] inline bool active2(Shape2 shape, int line) noexcept {
    return line == 0 ? shape.has(Act2::a) : shape.has(Act2::b);
}

[[nodiscard]] inline bool open2(Shape2 shape, int line) noexcept {
    return line == 0 ? shape.has(Line2::a_open) : shape.has(Line2::b_open);
}

[[nodiscard]] inline bool close2(Shape2 shape, int line) noexcept {
    return line == 0 ? shape.has(Line2::a_close) : shape.has(Line2::b_close);
}

[[nodiscard]] inline Dir2 dir2(Shape2 shape, int line) noexcept {
    return line == 0 ? shape.a : shape.b;
}

[[nodiscard]] inline Seg1Kind line_kind(Shape2 shape, int line) noexcept {
    if (!active2(shape, line)) return Seg1Kind::diag;
    const bool open = open2(shape, line);
    const bool close = close2(shape, line);
    const Dir2 dir = dir2(shape, line);
    if (dir == Dir2::diag || (open && close)) return Seg1Kind::diag;
    if (dir == Dir2::r) {
        if (open) return Seg1Kind::r_tail;
        if (close) return Seg1Kind::r_head;
        return Seg1Kind::r_mid;
    }
    if (open) return Seg1Kind::l_head;
    if (close) return Seg1Kind::l_tail;
    return Seg1Kind::l_mid;
}

struct Seg2Entry {
    std::uint16_t z = 0;
    double w = 0.0;
};

struct Seg2Key {
    unsigned code = 0;
    Step dx = Step::empty;
    Step dy = Step::empty;
    std::int16_t db = 0;
    std::uint16_t b = 0;
    std::uint16_t z = 0;
};

[[nodiscard]] inline constexpr unsigned shape_code(Shape2 x) noexcept {
    return static_cast<unsigned>(x.op)
        | (static_cast<unsigned>(x.line) << 4u)
        | (static_cast<unsigned>(x.act) << 8u)
        | (static_cast<unsigned>(x.a) << 10u)
        | (static_cast<unsigned>(x.b) << 12u);
}

[[nodiscard]] inline constexpr bool operator==(Seg2Key x, Seg2Key y) noexcept {
    return x.code == y.code && x.dx == y.dx && x.dy == y.dy && x.db == y.db && x.b == y.b && x.z == y.z;
}

[[nodiscard]] inline constexpr bool operator<(Seg2Key x, Seg2Key y) noexcept {
    if (x.code != y.code) return x.code < y.code;
    if (x.dx != y.dx) return x.dx < y.dx;
    if (x.dy != y.dy) return x.dy < y.dy;
    if (x.db != y.db) return x.db < y.db;
    if (x.b != y.b) return x.b < y.b;
    return x.z < y.z;
}

[[nodiscard]] inline std::uint64_t seg2_hash(Seg2Key key) noexcept {
    std::uint64_t x = static_cast<std::uint64_t>(key.code);
    x ^= static_cast<std::uint64_t>(key.dx) << 16u;
    x ^= static_cast<std::uint64_t>(key.dy) << 18u;
    x ^= static_cast<std::uint64_t>(static_cast<std::uint16_t>(key.db)) << 20u;
    x ^= static_cast<std::uint64_t>(key.b) << 36u;
    x ^= static_cast<std::uint64_t>(key.z) << 52u;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27u)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31u);
}

struct Seg2Table {
    std::vector<Seg2Key> keys;
    std::vector<std::size_t> off;
    std::vector<Seg2Entry> val;
    std::vector<int> slot;
    std::size_t mask = 0;
    int max_b = -1;

    void clear() {
        keys.clear();
        off.clear();
        val.clear();
        slot.clear();
        mask = 0;
        max_b = -1;
    }

    void index() {
        std::size_t size = 8;
        while (size < keys.size() * 2u + 1u) size <<= 1u;
        slot.assign(size, -1);
        mask = size - 1u;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            std::size_t pos = static_cast<std::size_t>(seg2_hash(keys[i])) & mask;
            while (slot[pos] >= 0) pos = (pos + 1u) & mask;
            slot[pos] = static_cast<int>(i);
        }
    }

    [[nodiscard]] std::span<const Seg2Entry> get(
        Shape2 shape,
        Step dx,
        Step dy,
        int db,
        int b,
        int z
    ) const noexcept {
        if (b < 0 || z < 0 || b > max_b || z > max_b || slot.empty()) return {};
        const Seg2Key q{
            shape_code(shape),
            dx,
            dy,
            static_cast<std::int16_t>(db),
            static_cast<std::uint16_t>(b),
            static_cast<std::uint16_t>(z)
        };
        std::size_t pos = static_cast<std::size_t>(seg2_hash(q)) & mask;
        for (;;) {
            const int idx = slot[pos];
            if (idx < 0) return {};
            const std::size_t i = static_cast<std::size_t>(idx);
            if (keys[i] == q) {
                return std::span<const Seg2Entry>(val.data() + off[i], off[i + 1] - off[i]);
            }
            pos = (pos + 1u) & mask;
        }
    }
};

[[nodiscard]] inline std::vector<Shape2> shapes2() {
    std::vector<Shape2> out;
    for (int p = 0; p < 5; ++p) {
        for (int q = 0; q < 5; ++q) {
            for (int r = 0; r < 5; ++r) {
                for (int s = 0; s < 5; ++s) {
                    const auto h = hull2(p, q, r, s);
                    for (int k = h[0]; k <= h[1]; ++k) {
                        const Shape2 sh = shape2(k, p, q, r, s);
                        bool found = false;
                        for (Shape2 x : out) found = found || x == sh;
                        if (!found) out.push_back(sh);
                    }
                }
            }
        }
    }
    return out;
}

inline void add_local(
    std::vector<std::pair<Seg2Key, Seg2Entry>>& out,
    Shape2 shape,
    Step dx,
    Step dy,
    int db,
    int b,
    int z,
    int max_b
) {
    const int xb1 = b - db;
    const int xb0 = xb1 - seg_delta(dx);
    const int yb0 = b - seg_delta(dy);
    if (xb0 < 0 || xb1 < 0 || yb0 < 0) return;

    int occ = step_occ(dy);
    if (shape.has(Op2::b_cre)) ++occ;
    if (shape.has(Op2::b_ann)) --occ;
    if (occ < 0 || occ > 2) return;

    const std::array<Step, 2> zs{occ_step(occ, Step::down), occ_step(occ, Step::up)};
    const int nz = occ == 1 ? 2 : 1;
    for (int iz = 0; iz < nz; ++iz) {
        const Step zst = zs[static_cast<std::size_t>(iz)];
        const int z1 = z + seg_delta(zst);
        if (z1 < 0 || z1 > max_b) continue;

        double w = 1.0;
        if (active2(shape, 0)) {
            w *= seg1(line_kind(shape, 0), dx, zst, z1 - xb1, z1);
            if (w == 0.0) continue;
        } else if (z != xb0 || zst != dx) {
            continue;
        }

        if (active2(shape, 1)) {
            w *= seg1(line_kind(shape, 1), zst, dy, b - z1, b);
            if (w == 0.0) continue;
        } else if (z != yb0 || zst != dy) {
            continue;
        }

        const Seg2Key key{
            shape_code(shape),
            dx,
            dy,
            static_cast<std::int16_t>(db),
            static_cast<std::uint16_t>(b),
            static_cast<std::uint16_t>(z)
        };
        out.push_back({key, Seg2Entry{static_cast<std::uint16_t>(z1), w}});
    }
}

// Precompute local z-spin transitions for all segment shapes.
inline void build_seg2(Seg2Table& table, int max_b) {
    table.clear();
    table.max_b = max_b;
    std::vector<std::pair<Seg2Key, Seg2Entry>> entries;
    const std::vector<Shape2> shapes = shapes2();

    for (Shape2 shape : shapes) {
        for (int dx = 0; dx < 4; ++dx) {
            for (int dy = 0; dy < 4; ++dy) {
                for (int db = -2; db <= 2; ++db) {
                    for (int b = 0; b <= max_b; ++b) {
                        for (int z = 0; z <= max_b; ++z) {
                            add_local(
                                entries,
                                shape,
                                static_cast<Step>(dx),
                                static_cast<Step>(dy),
                                db,
                                b,
                                z,
                                max_b
                            );
                        }
                    }
                }
            }
        }
    }

    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first == rhs.first) return lhs.second.z < rhs.second.z;
        return lhs.first < rhs.first;
    });

    for (std::size_t i = 0; i < entries.size();) {
        const Seg2Key key = entries[i].first;
        table.keys.push_back(key);
        table.off.push_back(table.val.size());
        while (i < entries.size() && entries[i].first == key) {
            if (entries[i].second.w != 0.0) table.val.push_back(entries[i].second);
            ++i;
        }
    }
    table.off.push_back(table.val.size());

    table.index();
}

} // namespace libdet::guga
