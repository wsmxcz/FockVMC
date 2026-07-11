#pragma once

#include "basis.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

inline void near(double x, double y, const char* msg) {
    const double tol = 4.0e-12 * (1.0 + std::abs(y));
    if (std::abs(x - y) > tol) {
        std::cerr << msg << ": " << x << " " << y << "\n";
        throw std::runtime_error(msg);
    }
}

inline StateBatchView batch_view(u32 nword, const std::vector<u64>& words) {
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

inline void check_action(const Hamiltonian& ham, const Basis& basis) {
    const std::size_t n = basis.size();
    const auto mat = dense(ham, basis);
    const auto csr = ham.matrix(basis.view(), basis.view());
    if (csr.n_bra != n || csr.n_ket != n || csr.indptr.size() != n + 1u) {
        throw std::runtime_error("matrix shape");
    }

    for (std::size_t ib = 0; ib < n; ++ib) {
        std::vector<double> values(n, 0.0);
        for (int t = csr.indptr[ib]; t < csr.indptr[ib + 1u]; ++t) {
            values[static_cast<std::size_t>(csr.indices[static_cast<std::size_t>(t)])] =
                csr.data[static_cast<std::size_t>(t)];
        }
        for (std::size_t ik = 0; ik < n; ++ik) {
            near(values[ik], mat[ib * n + ik], "matrix");
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

    const Basis bras = take_basis(basis, std::min<std::size_t>(4u, n));
    const Basis kets = take_basis(basis, std::min<std::size_t>(5u, n));
    const std::size_t nb = bras.size();
    const std::size_t nk = kets.size();

    const auto csr2 = ham.matrix(bras.view(), kets.view());
    if (csr2.n_bra != nb || csr2.n_ket != nk || csr2.indptr.size() != nb + 1u) {
        throw std::runtime_error("rect matrix shape");
    }
    for (std::size_t ib = 0; ib < nb; ++ib) {
        std::vector<double> values(nk, 0.0);
        for (int t = csr2.indptr[ib]; t < csr2.indptr[ib + 1u]; ++t) {
            values[static_cast<std::size_t>(csr2.indices[static_cast<std::size_t>(t)])] =
                csr2.data[static_cast<std::size_t>(t)];
        }
        for (std::size_t ik = 0; ik < nk; ++ik) {
            near(values[ik], ham.hij(bras.get(ib), kets.get(ik)), "rect matrix");
        }
    }

    std::vector<double> x2(nk, 0.0);
    for (std::size_t ik = 0; ik < nk; ++ik) x2[ik] = 0.19 - 0.031 * static_cast<double>(ik % 5);
    const auto y2 = ham.matvec(bras.view(), kets.view(), x2);
    const auto p2 = ham.project(bras.view(), kets.view(), x2, 0.0);
    if (y2.size() != nb || p2.hpsi.size() != nb || p2.diag.size() != nb) {
        throw std::runtime_error("rect action size");
    }
    const auto pbras = batch_view(p2.nword, p2.bra);
    for (std::size_t ib = 0; ib < nb; ++ib) {
        if (!same_state(pbras[ib], bras.get(ib))) throw std::runtime_error("project bra");
        double ref = 0.0;
        for (std::size_t ik = 0; ik < nk; ++ik) ref += ham.hij(bras.get(ib), kets.get(ik)) * x2[ik];
        near(y2[ib], ref, "rect matvec");
        near(p2.hpsi[ib], ref, "rect project");
        near(p2.diag[ib], ham.hij(bras.get(ib), bras.get(ib)), "rect project diag");
    }
}

inline void check_conn(
    const Hamiltonian& ham,
    const Basis& basis,
    double eps
) {
    const std::size_t n = basis.size();
    const auto mat = dense(ham, basis);
    const auto con = ham.conn(basis.view(), eps);
    const auto batch = batch_view(con.nword, con.bra);

    if (con.nword != basis.nword || con.n_kets != n || con.ptr.size() != n + 1u) {
        throw std::runtime_error("conn shape");
    }
    if (con.diag.size() != n || con.degree.size() != n) throw std::runtime_error("conn meta");
    if (batch.n_states != n + con.h.size()) throw std::runtime_error("conn batch");
    for (std::size_t i = 0; i < n; ++i) {
        if (!same_state(batch[i], basis.get(i))) throw std::runtime_error("conn prefix");
    }

    for (std::size_t ik = 0; ik < n; ++ik) {
        near(con.diag[ik], mat[ik * n + ik], "conn diag");
        near(
            con.degree[ik],
            exact_degree(mat, n, ik, std::numeric_limits<double>::infinity(), eps),
            "conn degree"
        );

        std::vector<double> got(n, 0.0);
        double last = std::numeric_limits<double>::infinity();
        for (int p = con.ptr[ik]; p < con.ptr[ik + 1u]; ++p) {
            const std::size_t t = static_cast<std::size_t>(p);
            const int ib = find_state(
                basis.view(),
                batch[n + t]
            );
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
    const auto batch = batch_view(con.nword, con.bra);
    if (con.nword != basis.nword || con.n_kets != n || con.n_streams != ns) {
        throw std::runtime_error("sample shape");
    }
    if (con.ptr.size() != ns * n + 1u || con.degree.size() != n) {
        throw std::runtime_error("sample meta");
    }
    if (batch.n_states != n + con.h.size()) throw std::runtime_error("sample batch");

    for (std::size_t ik = 0; ik < n; ++ik) {
        const double deg = exact_degree(mat, n, ik, eps1, eps2);
        near(con.degree[ik], deg, "sample degree");
    }

    for (std::size_t s = 0; s < ns; ++s) {
        for (std::size_t ik = 0; ik < n; ++ik) {
            const std::size_t pos = s * n + ik;
            const i64 want = con.degree[ik] > 0.0 ? count[pos] : 0;
            const i64 have = static_cast<i64>(con.ptr[pos + 1u] - con.ptr[pos]);
            if (have != want) throw std::runtime_error("sample count");

            for (int p = con.ptr[pos]; p < con.ptr[pos + 1u]; ++p) {
                const std::size_t t = static_cast<std::size_t>(p);
                const int ib = find_state(
                    basis.view(),
                    batch[n + t]
                );
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
    const auto batch = batch_view(con.nword, con.bra);

    if (con.nword != basis.nword || con.n_kets != n) throw std::runtime_error("local shape");
    if (con.diag.size() != n || con.strong_degree.size() != n || con.weak_degree.size() != n) {
        throw std::runtime_error("local meta");
    }
    if (con.strong_ptr.size() != n + 1u || con.weak_ptr.size() != n + 1u) {
        throw std::runtime_error("local ptr");
    }
    if (con.weak_h.size() != con.weak_count.size()) {
        throw std::runtime_error("weak size");
    }
    if (batch.n_states != n + con.strong_h.size() + con.weak_h.size()) {
        throw std::runtime_error("local batch");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (!same_state(batch[i], basis.get(i))) throw std::runtime_error("local prefix");
    }

    for (std::size_t ik = 0; ik < n; ++ik) {
        near(con.diag[ik], mat[ik * n + ik], "local diag");
        near(
            con.strong_degree[ik],
            exact_degree(mat, n, ik, std::numeric_limits<double>::infinity(), eps1),
            "strong degree"
        );
        const double ref_weak = n_draw > 0 ? exact_degree(mat, n, ik, eps1, eps2) : 0.0;
        near(con.weak_degree[ik], ref_weak, "weak degree");

        std::vector<double> got(n, 0.0);
        for (int p = con.strong_ptr[ik]; p < con.strong_ptr[ik + 1u]; ++p) {
            const std::size_t t = static_cast<std::size_t>(p);
            const int ib = find_state(
                basis.view(),
                batch[n + t]
            );
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
            const int ib = find_state(
                basis.view(),
                batch[n + con.strong_h.size() + t]
            );
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

inline void check_project(const Hamiltonian& ham, const Basis& basis, double eps) {
    const std::size_t n = basis.size();
    const Basis kets = take_basis(basis, std::min<std::size_t>(3u, n));
    const std::size_t nk = kets.size();
    std::vector<double> scale(nk, 0.0);
    for (std::size_t i = 0; i < nk; ++i) scale[i] = 0.23 - 0.04 * static_cast<double>(i % 5);

    const StateBatchView exclude = kets.view();
    const auto out = ham.project(kets.view(), scale, eps, &exclude);
    const auto batch = batch_view(out.nword, out.bra);
    if (out.nword != basis.nword || out.hpsi.size() != batch.n_states) {
        throw std::runtime_error("project shape");
    }
    if (!out.diag.empty() && out.diag.size() != batch.n_states) {
        throw std::runtime_error("project diag shape");
    }

    std::vector<double> ref(n, 0.0);
    for (std::size_t ib = 0; ib < n; ++ib) {
        if (find_state(exclude, basis.get(ib)) >= 0) continue;
        for (std::size_t ik = 0; ik < nk; ++ik) {
            const double term = ham.hij(basis.get(ib), kets.get(ik)) * scale[ik];
            if (std::abs(term) >= eps) ref[ib] += term;
        }
    }

    std::vector<double> got(n, 0.0);
    for (std::size_t ibra = 0; ibra < batch.n_states; ++ibra) {
        const int ib = find_state(basis.view(), batch[ibra]);
        if (ib < 0) throw std::runtime_error("project bra");
        if (find_state(exclude, batch[ibra]) >= 0) throw std::runtime_error("project exclude");
        if (got[static_cast<std::size_t>(ib)] != 0.0) throw std::runtime_error("project dup");
        got[static_cast<std::size_t>(ib)] = out.hpsi[ibra];
        near(out.hpsi[ibra], ref[static_cast<std::size_t>(ib)], "project hpsi");
    }
    for (std::size_t ib = 0; ib < n; ++ib) near(got[ib], ref[ib], "project exact");
}

inline void check_sample_project(const Hamiltonian& ham, const Basis& basis) {
    const std::size_t n = basis.size();
    const Basis kets = take_basis(basis, std::min<std::size_t>(3u, n));
    const std::size_t nk = kets.size();
    const std::size_t ns = 2;
    std::vector<double> scale(nk, 0.0);
    for (std::size_t i = 0; i < nk; ++i) scale[i] = 0.21 - 0.03 * static_cast<double>(i % 3);
    std::vector<i64> counts(ns * nk, 3);

    const StateBatchView exclude = kets.view();
    const auto out = ham.sample_project(kets.view(), scale, counts, ns, 0.30, 0.02, &exclude, 19);
    const auto batch = batch_view(out.nword, out.bra);
    if (out.nword != basis.nword || out.n_streams != ns) {
        throw std::runtime_error("sample project shape");
    }
    if (out.hpsi.size() != ns * batch.n_states) throw std::runtime_error("sample project hpsi");
    if (!out.diag.empty() && out.diag.size() != batch.n_states) {
        throw std::runtime_error("sample project diag");
    }

    for (std::size_t ibra = 0; ibra < batch.n_states; ++ibra) {
        if (find_state(basis.view(), batch[ibra]) < 0) {
            throw std::runtime_error("sample project bra");
        }
        if (find_state(exclude, batch[ibra]) >= 0) {
            throw std::runtime_error("sample project exclude");
        }
        for (std::size_t jb = ibra + 1u; jb < batch.n_states; ++jb) {
            if (same_state(batch[ibra], batch[jb])) throw std::runtime_error("sample project dup");
        }
        for (std::size_t s = 0; s < ns; ++s) {
            const double value = out.hpsi[s * batch.n_states + ibra];
            if (!std::isfinite(value)) throw std::runtime_error("sample project finite");
        }
    }
}
