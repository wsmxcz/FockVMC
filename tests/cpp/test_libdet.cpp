#include <libdet/hamiltonian.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

using libdet::DetBatchView;
using libdet::DetRef;
using libdet::Hamiltonian;
using libdet::i64;
using libdet::u64;

void near(double actual, double expected, const char* message) {
    const double tol = 4.0e-12 * (1.0 + std::abs(expected));
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(message);
    }
}

std::size_t pair_id(int p, int q) {
    const int hi = std::max(p, q);
    const int lo = std::min(p, q);
    return static_cast<std::size_t>(hi * (hi + 1) / 2 + lo);
}

std::size_t eri_id(int p, int q, int r, int s) {
    return pair_id(
        static_cast<int>(pair_id(p, q)),
        static_cast<int>(pair_id(r, s))
    );
}

std::vector<double> make_h1(int norb) {
    std::vector<double> out(static_cast<std::size_t>(norb * norb), 0.0);
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q <= p; ++q) {
            const double h =
                0.11 + 0.07 * (p + 1) - 0.03 * (q + 2)
                + 0.015 * ((p + 2 * q) % 5);
            out[static_cast<std::size_t>(p * norb + q)] = h;
            out[static_cast<std::size_t>(q * norb + p)] = h;
        }
    }
    return out;
}

std::vector<double> make_eri(int norb) {
    const int npair = norb * (norb + 1) / 2;
    std::vector<double> out(
        static_cast<std::size_t>(npair * (npair + 1) / 2),
        0.0
    );
    for (int p = 0; p < norb; ++p) {
        for (int q = 0; q < norb; ++q) {
            for (int r = 0; r < norb; ++r) {
                for (int s = 0; s < norb; ++s) {
                    out[eri_id(p, q, r, s)] =
                        0.012
                        * (1 + ((3 * p + 5 * q + 7 * r + 11 * s) % 23));
                }
            }
        }
    }
    return out;
}

std::vector<u64> make_basis(int norb) {
    std::vector<u64> out;
    for (int i = 0; i < norb; ++i) {
        for (int j = i + 1; j < norb; ++j) {
            const u64 alpha = (u64{1} << i) | (u64{1} << j);
            for (int k = 0; k < norb; ++k) {
                out.push_back(alpha);
                out.push_back(u64{1} << k);
            }
        }
    }
    return out;
}

DetBatchView view(const std::vector<u64>& words) {
    return {words.data(), words.size() / 2u, 1u};
}

int find_state(DetBatchView states, DetRef target) {
    for (std::size_t i = 0; i < states.n_dets; ++i) {
        const DetRef state = states[i];
        if (
            state.alpha()[0] == target.alpha()[0]
            && state.beta()[0] == target.beta()[0]
        ) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<double> dense(
    const Hamiltonian& ham,
    DetBatchView basis
) {
    const std::size_t n = basis.n_dets;
    std::vector<double> out(n * n, 0.0);
    for (std::size_t bra = 0; bra < n; ++bra) {
        for (std::size_t ket = 0; ket < n; ++ket) {
            out[bra * n + ket] = ham.hij(basis[bra], basis[ket]);
        }
    }
    return out;
}

double sum_degree(
    const std::vector<double>& matrix,
    std::size_t n,
    std::size_t ket,
    double eps1,
    double eps2 = std::numeric_limits<double>::infinity()
) {
    double out = 0.0;
    for (std::size_t bra = 0; bra < n; ++bra) {
        if (bra == ket) continue;
        const double h = std::abs(matrix[bra * n + ket]);
        if (h >= eps1 && h < eps2) out += h;
    }
    return out;
}

void check_matrix(
    const Hamiltonian& ham,
    DetBatchView basis,
    const std::vector<double>& expected
) {
    const std::size_t n = basis.n_dets;
    const auto diag = ham.diag(basis);
    const auto matrix = ham.matrix(basis, basis);
    std::vector<double> vec(n * 2u, 0.0);
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec[i] = 0.1 * static_cast<double>(i + 1u);
    }
    const auto product = ham.matvec(basis, basis, {vec.data(), n});
    const auto products = ham.matmat(basis, basis, vec, 2u);

    for (std::size_t bra = 0; bra < n; ++bra) {
        near(diag[bra], expected[bra * n + bra], "diag");

        std::vector<double> row(n, 0.0);
        for (i64 p = matrix.indptr[bra]; p < matrix.indptr[bra + 1u]; ++p) {
            const std::size_t pos = static_cast<std::size_t>(p);
            row[static_cast<std::size_t>(matrix.indices[pos])] = matrix.data[pos];
        }

        double value = 0.0;
        double value0 = 0.0;
        double value1 = 0.0;
        for (std::size_t ket = 0; ket < n; ++ket) {
            const double h = expected[bra * n + ket];
            near(row[ket], h, "matrix");
            value += h * vec[ket];
            value0 += h * vec[2u * ket];
            value1 += h * vec[2u * ket + 1u];
        }
        near(product[bra], value, "matvec");
        near(products[2u * bra], value0, "matmat");
        near(products[2u * bra + 1u], value1, "matmat");
    }
}

void check_conn(
    const Hamiltonian& ham,
    DetBatchView basis,
    const std::vector<double>& matrix,
    double eps
) {
    const std::size_t n = basis.n_dets;
    const auto conn = ham.conn(basis, eps);
    const auto bras = view(conn.bra);

    for (std::size_t ket = 0; ket < n; ++ket) {
        near(conn.diag[ket], matrix[ket * n + ket], "conn diag");
        near(conn.degree[ket], sum_degree(matrix, n, ket, eps), "degree");

        double previous = std::numeric_limits<double>::infinity();
        std::vector<double> column(n, 0.0);
        for (i64 p = conn.ptr[ket]; p < conn.ptr[ket + 1u]; ++p) {
            const std::size_t pos = static_cast<std::size_t>(p);
            const int bra = find_state(basis, bras[n + pos]);
            if (bra < 0 || std::abs(conn.h[pos]) > previous) {
                throw std::runtime_error("conn order");
            }
            previous = std::abs(conn.h[pos]);
            column[static_cast<std::size_t>(bra)] = conn.h[pos];
        }

        for (std::size_t bra = 0; bra < n; ++bra) {
            const double h = matrix[bra * n + ket];
            const double value = bra != ket && std::abs(h) >= eps ? h : 0.0;
            near(column[bra], value, "conn value");
        }
    }
}

void check_project(
    const Hamiltonian& ham,
    DetBatchView basis,
    const std::vector<double>& matrix
) {
    const std::size_t n = basis.n_dets;
    const DetBatchView kets{basis.data, 4u, basis.nword};
    const std::vector<double> scale{0.4, -0.2, 0.7, 0.3};
    const auto fixed = ham.project(basis, kets, scale, 0.0);

    for (std::size_t bra = 0; bra < n; ++bra) {
        double value = 0.0;
        for (std::size_t ket = 0; ket < kets.n_dets; ++ket) {
            value += matrix[bra * n + ket] * scale[ket];
        }
        near(fixed.hpsi[bra], value, "project");
    }

    const double eps = 0.08;
    const auto words = ham.expand(kets, eps, scale, &kets);
    const auto expanded = view(words);
    std::vector<bool> expected(n, false);
    for (std::size_t bra = kets.n_dets; bra < n; ++bra) {
        for (std::size_t ket = 0; ket < kets.n_dets; ++ket) {
            if (std::abs(matrix[bra * n + ket] * scale[ket]) >= eps) {
                expected[bra] = true;
            }
        }
    }

    std::size_t count = 0;
    for (bool value : expected) count += value ? 1u : 0u;
    if (expanded.n_dets != count) throw std::runtime_error("expand size");
    for (std::size_t i = 0; i < expanded.n_dets; ++i) {
        const int bra = find_state(basis, expanded[i]);
        if (bra < 0 || !expected[static_cast<std::size_t>(bra)]) {
            throw std::runtime_error("expand value");
        }
    }
}

void check_sample(
    const Hamiltonian& ham,
    DetBatchView basis,
    const std::vector<double>& matrix,
    double eps1,
    double eps2
) {
    const std::size_t n = basis.n_dets;
    const std::vector<i64> counts(2u * n, 3);
    const auto sample_a = ham.sample_conn(
        basis,
        counts,
        2u,
        eps1,
        eps2,
        11
    );
    const auto sample_b = ham.sample_conn(
        basis,
        counts,
        2u,
        eps1,
        eps2,
        11
    );
    if (
        sample_a.bra != sample_b.bra
        || sample_a.ptr != sample_b.ptr
        || sample_a.h != sample_b.h
        || sample_a.degree != sample_b.degree
    ) {
        throw std::runtime_error("sample seed");
    }

    const auto sample_bra = view(sample_a.bra);
    for (std::size_t row = 0; row < 2u * n; ++row) {
        const std::size_t ket = row % n;
        near(
            sample_a.degree[ket],
            sum_degree(matrix, n, ket, eps2, eps1),
            "sample degree"
        );
        for (i64 p = sample_a.ptr[row]; p < sample_a.ptr[row + 1u]; ++p) {
            const std::size_t pos = static_cast<std::size_t>(p);
            const int bra = find_state(basis, sample_bra[n + pos]);
            if (bra < 0) throw std::runtime_error("sample bra");
            const double h = matrix[static_cast<std::size_t>(bra) * n + ket];
            if (!(std::abs(h) >= eps2 && std::abs(h) < eps1)) {
                throw std::runtime_error("sample window");
            }
            near(sample_a.h[pos], h, "sample value");
        }
    }

    const std::vector<i64> draws(n, 3);
    const auto local_a = ham.local_conn(basis, eps1, eps2, draws, 17);
    const auto local_b = ham.local_conn(basis, eps1, eps2, draws, 17);
    if (
        local_a.bra != local_b.bra
        || local_a.strong_ptr != local_b.strong_ptr
        || local_a.strong_h != local_b.strong_h
        || local_a.weak_ptr != local_b.weak_ptr
        || local_a.weak_coeff != local_b.weak_coeff
    ) {
        throw std::runtime_error("local seed");
    }

    const auto local_bra = view(local_a.bra);
    const std::size_t strong_n =
        static_cast<std::size_t>(local_a.strong_ptr.back());
    for (std::size_t ket = 0; ket < n; ++ket) {
        near(
            local_a.strong_degree[ket],
            sum_degree(matrix, n, ket, eps1),
            "strong degree"
        );
        for (
            i64 p = local_a.strong_ptr[ket];
            p < local_a.strong_ptr[ket + 1u];
            ++p
        ) {
            const std::size_t pos = static_cast<std::size_t>(p);
            const int bra = find_state(basis, local_bra[n + pos]);
            if (bra < 0) throw std::runtime_error("strong bra");
            const double h = matrix[static_cast<std::size_t>(bra) * n + ket];
            if (std::abs(h) < eps1) throw std::runtime_error("strong window");
            near(local_a.strong_h[pos], h, "strong value");
        }
        for (
            i64 p = local_a.weak_ptr[ket];
            p < local_a.weak_ptr[ket + 1u];
            ++p
        ) {
            const std::size_t pos = static_cast<std::size_t>(p);
            const int bra = find_state(
                basis,
                local_bra[n + strong_n + pos]
            );
            if (bra < 0) throw std::runtime_error("weak bra");
            const double h = matrix[static_cast<std::size_t>(bra) * n + ket];
            if (
                !(std::abs(h) >= eps2 && std::abs(h) < eps1)
                || local_a.weak_coeff[pos] * h <= 0.0
            ) {
                throw std::runtime_error("weak window");
            }
        }
    }
}

int main() {
    const int norb = 5;
    const auto h1 = make_h1(norb);
    const auto eri = make_eri(norb);
    const auto words = make_basis(norb);
    const auto basis = view(words);
    const Hamiltonian ham(h1, norb, eri, -0.13);
    const auto matrix = dense(ham, basis);

    check_matrix(ham, basis, matrix);
    check_conn(ham, basis, matrix, 0.10);
    check_project(ham, basis, matrix);
    check_sample(ham, basis, matrix, 0.10, 0.025);
    return 0;
}
