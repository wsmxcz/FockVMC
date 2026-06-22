#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <libdet/hamiltonian.hpp>

using libdet::Hamiltonian;
using libdet::i64;
using libdet::StateBatchView;
using libdet::StateRef;
using libdet::u32;
using libdet::u64;

struct Basis {
    u32 nword = 0;
    std::vector<u64> words;

    [[nodiscard]] std::size_t size() const noexcept {
        return words.size() / libdet::word_pair_size(nword);
    }

    [[nodiscard]] StateBatchView view() const noexcept {
        return StateBatchView{words.data(), size(), nword};
    }

    [[nodiscard]] StateRef get(std::size_t i) const noexcept {
        return view()[i];
    }
};

inline void near(double x, double y, const char* msg) {
    const double tol = 4.0e-12 * (1.0 + std::abs(y));
    if (std::abs(x - y) > tol) {
        std::cerr << msg << ": " << x << " " << y << "\n";
        throw std::runtime_error(msg);
    }
}

inline std::size_t pair_id(int p, int q) {
    const int hi = std::max(p, q);
    const int lo = std::min(p, q);
    return static_cast<std::size_t>(hi * (hi + 1) / 2 + lo);
}

inline std::size_t eri_id(int p, int q, int r, int s) {
    return pair_id(
        static_cast<int>(pair_id(p, q)),
        static_cast<int>(pair_id(r, s))
    );
}

inline std::vector<double> make_h1(int n) {
    std::vector<double> h(static_cast<std::size_t>(n * n), 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q <= p; ++q) {
            const double v = 0.11 + 0.07 * (p + 1) - 0.03 * (q + 2)
                + 0.015 * ((p + 2 * q) % 5);
            h[static_cast<std::size_t>(p * n + q)] = v;
            h[static_cast<std::size_t>(q * n + p)] = v;
        }
    }
    return h;
}

inline std::vector<double> make_eri(int n) {
    const int npair = n * (n + 1) / 2;
    std::vector<double> eri(static_cast<std::size_t>(npair * (npair + 1) / 2), 0.0);
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < n; ++s) {
                    eri[eri_id(p, q, r, s)] =
                        0.012 * (1 + ((3 * p + 5 * q + 7 * r + 11 * s) % 23));
                }
            }
        }
    }
    return eri;
}

template <class F>
void choose(int n, int k, int first, u64 bits, F&& visit) {
    if (k == 0) {
        visit(bits);
        return;
    }
    for (int p = first; p <= n - k; ++p) {
        choose(n, k - 1, p + 1, bits | (u64{1} << p), visit);
    }
}

inline Basis det_basis(int norb, int na, int nb, u32 nword) {
    Basis basis{nword, {}};
    choose(norb, na, 0, u64{0}, [&](u64 a) {
        choose(norb, nb, 0, u64{0}, [&](u64 b) {
            basis.words.push_back(a);
            for (u32 w = 1; w < nword; ++w) basis.words.push_back(0);
            basis.words.push_back(b);
            for (u32 w = 1; w < nword; ++w) basis.words.push_back(0);
        });
    });
    return basis;
}

inline void put_path(Basis& basis, std::span<const int> step) {
    std::vector<u64> word(libdet::word_pair_size(basis.nword), 0u);
    for (int p = 0; p < static_cast<int>(step.size()); ++p) {
        const std::size_t w = static_cast<std::size_t>(p >> 6);
        const u64 bit = u64{1} << static_cast<unsigned>(p & 63);
        if (step[static_cast<std::size_t>(p)] == 2 || step[static_cast<std::size_t>(p)] == 3) {
            word[w] |= bit;
        }
        if (step[static_cast<std::size_t>(p)] == 1 || step[static_cast<std::size_t>(p)] == 3) {
            word[static_cast<std::size_t>(basis.nword) + w] |= bit;
        }
    }
    basis.words.insert(basis.words.end(), word.begin(), word.end());
}

inline void path_rec(
    Basis& basis,
    std::vector<int>& step,
    int p,
    int nelec,
    int spin,
    int target_e,
    int target_s
) {
    const int norb = static_cast<int>(step.size());
    if (p == norb) {
        if (nelec == target_e && spin == target_s) put_path(basis, step);
        return;
    }

    const int rem = norb - p - 1;
    for (int st : {0, 1, 2, 3}) {
        const int occ = st == 3 ? 2 : (st == 0 ? 0 : 1);
        const int ds = st == 2 ? 1 : (st == 1 ? -1 : 0);
        const int e1 = nelec + occ;
        const int s1 = spin + ds;
        if (s1 < 0) continue;
        if (e1 > target_e) continue;
        if (e1 + 2 * rem < target_e) continue;
        step[static_cast<std::size_t>(p)] = st;
        path_rec(basis, step, p + 1, e1, s1, target_e, target_s);
    }
}

inline Basis spin_basis(int norb, int na, int nb, u32 nword) {
    Basis basis{nword, {}};
    std::vector<int> step(static_cast<std::size_t>(norb), 0);
    path_rec(basis, step, 0, 0, 0, na + nb, na - nb);
    return basis;
}

inline bool same_state(StateRef a, StateRef b) noexcept {
    if (a.nword() != b.nword()) return false;
    const std::size_t n = libdet::word_pair_size(a.nword());
    return std::equal(a.data(), a.data() + n, b.data());
}

inline int find_state(StateBatchView x, StateRef y) {
    for (std::size_t i = 0; i < x.n_states; ++i) {
        if (same_state(x[i], y)) return static_cast<int>(i);
    }
    return -1;
}

inline StateBatchView pool_view(u32 nword, const std::vector<u64>& words) {
    return StateBatchView{
        words.data(),
        words.size() / libdet::word_pair_size(nword),
        nword
    };
}

inline std::vector<double> dense(const Hamiltonian& ham, const Basis& basis) {
    const std::size_t n = basis.size();
    std::vector<double> out(n * n, 0.0);
    for (std::size_t ib = 0; ib < n; ++ib) {
        for (std::size_t ik = 0; ik < n; ++ik) {
            out[ib * n + ik] = ham.hij(basis.get(ib), basis.get(ik));
        }
    }
    return out;
}

inline double exact_degree(
    const std::vector<double>& mat,
    std::size_t n,
    std::size_t ik,
    double eps1,
    double eps2
) {
    double out = 0.0;
    for (std::size_t ib = 0; ib < n; ++ib) {
        if (ib == ik) continue;
        const double a = std::abs(mat[ib * n + ik]);
        if (a > 0.0 && a < eps1 && a >= eps2) out += a;
    }
    return out;
}

inline void check_diag(const Hamiltonian& ham, const Basis& basis) {
    const auto d = ham.diags(basis.view());
    if (d.size() != basis.size()) throw std::runtime_error("diag size");
    for (std::size_t i = 0; i < basis.size(); ++i) {
        near(d[i], ham.hij(basis.get(i), basis.get(i)), "diag");
    }
}

inline void check_matrix(const Hamiltonian& ham, const Basis& basis) {
    const std::size_t n = basis.size();
    const auto mat = dense(ham, basis);
    const auto csr = ham.matrix(basis.view(), basis.view());
    if (csr.n_bra != n || csr.n_ket != n || csr.indptr.size() != n + 1u) {
        throw std::runtime_error("matrix shape");
    }

    for (std::size_t ib = 0; ib < n; ++ib) {
        std::vector<double> row(n, 0.0);
        for (int t = csr.indptr[ib]; t < csr.indptr[ib + 1u]; ++t) {
            row[static_cast<std::size_t>(csr.indices[static_cast<std::size_t>(t)])] =
                csr.data[static_cast<std::size_t>(t)];
        }
        for (std::size_t ik = 0; ik < n; ++ik) {
            near(row[ik], mat[ib * n + ik], "matrix");
        }
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) x[i] = 0.17 - 0.025 * static_cast<double>(i % 7);

    const auto y = ham.matvec(basis.view(), basis.view(), x);
    if (y.size() != n) throw std::runtime_error("matvec size");
    for (std::size_t ib = 0; ib < n; ++ib) {
        double ref = 0.0;
        for (std::size_t ik = 0; ik < n; ++ik) ref += mat[ib * n + ik] * x[ik];
        near(y[ib], ref, "matvec");
    }

    const std::size_t nrhs = 2;
    std::vector<double> xx(n * nrhs, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        xx[i * nrhs] = x[i];
        xx[i * nrhs + 1u] = -0.4 * x[i] + 0.03;
    }

    const auto yy = ham.matmat(basis.view(), basis.view(), xx, nrhs);
    if (yy.size() != n * nrhs) throw std::runtime_error("matmat size");
    for (std::size_t ib = 0; ib < n; ++ib) {
        for (std::size_t j = 0; j < nrhs; ++j) {
            double ref = 0.0;
            for (std::size_t ik = 0; ik < n; ++ik) {
                ref += mat[ib * n + ik] * xx[ik * nrhs + j];
            }
            near(yy[ib * nrhs + j], ref, "matmat");
        }
    }
}

inline void check_conn(const Hamiltonian& ham, const Basis& basis, double eps) {
    const std::size_t n = basis.size();
    const auto mat = dense(ham, basis);
    const auto con = ham.conn(basis.view(), eps);
    const auto pool = pool_view(con.nword, con.bra_words);

    if (con.nword != basis.nword || con.n_kets != n || con.ptr.size() != n + 1u) {
        throw std::runtime_error("conn shape");
    }
    if (con.diag.size() != n || con.degree.size() != n) throw std::runtime_error("conn meta");
    if (pool.n_states < n) throw std::runtime_error("conn pool");
    for (std::size_t i = 0; i < n; ++i) {
        if (!same_state(pool[i], basis.get(i))) throw std::runtime_error("conn prefix");
    }

    for (std::size_t ik = 0; ik < n; ++ik) {
        near(con.diag[ik], mat[ik * n + ik], "conn diag");
        near(con.degree[ik], exact_degree(mat, n, ik, std::numeric_limits<double>::infinity(), eps), "conn degree");

        std::vector<double> got(n, 0.0);
        double last = std::numeric_limits<double>::infinity();
        for (int p = con.ptr[ik]; p < con.ptr[ik + 1u]; ++p) {
            const std::size_t t = static_cast<std::size_t>(p);
            const int ib = find_state(basis.view(), pool[static_cast<std::size_t>(con.idx[t])]);
            if (ib < 0) throw std::runtime_error("conn bra");
            if (got[static_cast<std::size_t>(ib)] != 0.0) throw std::runtime_error("conn dup");
            const double h = con.h[t];
            const double a = std::abs(h);
            if (a > last + 1.0e-14) throw std::runtime_error("conn order");
            last = a;
            if (!(a >= eps)) throw std::runtime_error("conn eps");
            near(h, mat[static_cast<std::size_t>(ib) * n + ik], "conn h");
            got[static_cast<std::size_t>(ib)] = h;
        }

        for (std::size_t ib = 0; ib < n; ++ib) {
            const double ref = (ib != ik && std::abs(mat[ib * n + ik]) >= eps)
                ? mat[ib * n + ik]
                : 0.0;
            near(got[ib], ref, "conn exact");
        }
    }
}

inline void check_sample(
    const Hamiltonian& ham,
    const Basis& basis,
    double eps1,
    double eps2
) {
    const std::size_t n = basis.size();
    const std::size_t ns = 2;
    const auto mat = dense(ham, basis);
    std::vector<i64> count(ns * n, 0);
    for (std::size_t s = 0; s < ns; ++s) {
        for (std::size_t i = 0; i < n; ++i) count[s * n + i] = static_cast<i64>(2 + s);
    }

    const auto con = ham.sample_conn(basis.view(), count, ns, eps1, eps2, 17);
    const auto pool = pool_view(con.nword, con.bra_words);
    if (con.nword != basis.nword || con.n_kets != n || con.n_streams != ns) {
        throw std::runtime_error("sample shape");
    }
    if (con.ptr.size() != ns * n + 1u || con.degree.size() != n) {
        throw std::runtime_error("sample meta");
    }

    for (std::size_t ik = 0; ik < n; ++ik) {
        const double deg = exact_degree(mat, n, ik, eps1, eps2);
        near(con.degree[ik], deg, "sample degree");
    }

    for (std::size_t s = 0; s < ns; ++s) {
        for (std::size_t ik = 0; ik < n; ++ik) {
            const std::size_t row = s * n + ik;
            const i64 want = con.degree[ik] > 0.0 ? count[row] : 0;
            const i64 have = static_cast<i64>(con.ptr[row + 1u] - con.ptr[row]);
            if (have != want) throw std::runtime_error("sample count");

            for (int p = con.ptr[row]; p < con.ptr[row + 1u]; ++p) {
                const std::size_t t = static_cast<std::size_t>(p);
                const int ib = find_state(basis.view(), pool[static_cast<std::size_t>(con.idx[t])]);
                if (ib < 0) throw std::runtime_error("sample bra");
                const double h = con.h[t];
                const double a = std::abs(h);
                if (!(a >= eps2 && a < eps1 && a > 0.0)) throw std::runtime_error("sample eps");
                near(h, mat[static_cast<std::size_t>(ib) * n + ik], "sample h");
            }
        }
    }
}

inline void check_local(
    const Hamiltonian& ham,
    const Basis& basis,
    double eps1,
    double eps2,
    i64 n_draw
) {
    const std::size_t n = basis.size();
    const auto mat = dense(ham, basis);
    std::vector<i64> count(n, n_draw);
    const auto con = ham.local_conn(basis.view(), eps1, eps2, count, 31);
    const auto pool = pool_view(con.nword, con.bra_words);

    if (con.nword != basis.nword || con.n_kets != n) throw std::runtime_error("local shape");
    if (con.diag.size() != n || con.strong_degree.size() != n || con.weak_degree.size() != n) {
        throw std::runtime_error("local meta");
    }
    if (con.strong_ptr.size() != n + 1u || con.weak_ptr.size() != n + 1u) {
        throw std::runtime_error("local ptr");
    }
    if (con.weak_idx.size() != con.weak_h.size() || con.weak_idx.size() != con.weak_count.size()) {
        throw std::runtime_error("weak size");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!same_state(pool[i], basis.get(i))) throw std::runtime_error("local prefix");
    }

    for (std::size_t ik = 0; ik < n; ++ik) {
        near(con.diag[ik], mat[ik * n + ik], "local diag");
        near(con.strong_degree[ik], exact_degree(mat, n, ik, std::numeric_limits<double>::infinity(), eps1), "strong degree");
        const double ref_weak = n_draw > 0 ? exact_degree(mat, n, ik, eps1, eps2) : 0.0;
        near(con.weak_degree[ik], ref_weak, "weak degree");

        std::vector<double> got(n, 0.0);
        for (int p = con.strong_ptr[ik]; p < con.strong_ptr[ik + 1u]; ++p) {
            const std::size_t t = static_cast<std::size_t>(p);
            const int ib = find_state(basis.view(), pool[static_cast<std::size_t>(con.strong_idx[t])]);
            if (ib < 0) throw std::runtime_error("strong bra");
            const double h = con.strong_h[t];
            if (!(std::abs(h) >= eps1)) throw std::runtime_error("strong eps");
            near(h, mat[static_cast<std::size_t>(ib) * n + ik], "strong h");
            got[static_cast<std::size_t>(ib)] = h;
        }
        for (std::size_t ib = 0; ib < n; ++ib) {
            const double ref = (ib != ik && std::abs(mat[ib * n + ik]) >= eps1)
                ? mat[ib * n + ik]
                : 0.0;
            near(got[ib], ref, "strong exact");
        }

        i64 seen = 0;
        for (int p = con.weak_ptr[ik]; p < con.weak_ptr[ik + 1u]; ++p) {
            const std::size_t t = static_cast<std::size_t>(p);
            const int ib = find_state(basis.view(), pool[static_cast<std::size_t>(con.weak_idx[t])]);
            if (ib < 0) throw std::runtime_error("weak bra");
            const double h = con.weak_h[t];
            const double a = std::abs(h);
            if (!(a >= eps2 && a < eps1 && a > 0.0)) throw std::runtime_error("weak eps");
            if (con.weak_count[t] <= 0) throw std::runtime_error("weak count");
            seen += con.weak_count[t];
            near(h, mat[static_cast<std::size_t>(ib) * n + ik], "weak h");
        }
        const i64 want = con.weak_degree[ik] > 0.0 ? n_draw : 0;
        if (seen != want) throw std::runtime_error("weak draw");
    }
}
