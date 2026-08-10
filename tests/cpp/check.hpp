#pragma once

#include "basis.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

inline void near(double actual, double expected, const char* message) {
    const double tolerance = 4.0e-12 * (1.0 + std::abs(expected));
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

inline DetBatchView batch_view(u32 nword, const std::vector<u64>& words) {
    return DetBatchView{
        words.data(),
        words.size() / libdet::word_pair_size(nword),
        nword
    };
}

inline std::vector<double> dense(const Hamiltonian& hamiltonian, const Basis& basis) {
    const std::size_t n = basis.size();
    std::vector<double> matrix(n * n, 0.0);
    for (std::size_t bra = 0; bra < n; ++bra) {
        for (std::size_t ket = 0; ket < n; ++ket) {
            matrix[bra * n + ket] = hamiltonian.hij(
                basis.get(bra),
                basis.get(ket)
            );
        }
    }
    return matrix;
}

inline double degree(
    const std::vector<double>& matrix,
    std::size_t n,
    std::size_t ket,
    double eps
) {
    double value = 0.0;
    for (std::size_t bra = 0; bra < n; ++bra) {
        if (bra != ket && std::abs(matrix[bra * n + ket]) >= eps) {
            value += std::abs(matrix[bra * n + ket]);
        }
    }
    return value;
}

inline void check_matrix(const Hamiltonian& hamiltonian, const Basis& basis) {
    const std::size_t n = basis.size();
    const auto expected = dense(hamiltonian, basis);
    const auto diagonal = hamiltonian.diag(basis.view());
    const auto matrix = hamiltonian.matrix(basis.view(), basis.view());

    if (diagonal.size() != n || matrix.indptr.size() != n + 1u) {
        throw std::runtime_error("matrix shape");
    }

    for (std::size_t bra = 0; bra < n; ++bra) {
        near(diagonal[bra], expected[bra * n + bra], "diagonal");

        std::vector<double> row(n, 0.0);
        for (int p = matrix.indptr[bra]; p < matrix.indptr[bra + 1u]; ++p) {
            const std::size_t record = static_cast<std::size_t>(p);
            row[static_cast<std::size_t>(matrix.indices[record])] = matrix.data[record];
        }
        for (std::size_t ket = 0; ket < n; ++ket) {
            near(row[ket], expected[bra * n + ket], "matrix element");
        }
    }
}

inline void check_conn(
    const Hamiltonian& hamiltonian,
    const Basis& basis,
    double eps
) {
    const std::size_t n = basis.size();
    const auto matrix = dense(hamiltonian, basis);
    const auto connections = hamiltonian.conn(basis.view(), eps);
    const auto configurations = batch_view(connections.nword, connections.bra);

    if (connections.ptr.size() != n + 1u || connections.degree.size() != n) {
        throw std::runtime_error("connection shape");
    }

    for (std::size_t ket = 0; ket < n; ++ket) {
        near(connections.diag[ket], matrix[ket * n + ket], "connection diagonal");
        near(connections.degree[ket], degree(matrix, n, ket, eps), "degree");

        std::vector<double> column(n, 0.0);
        double previous = std::numeric_limits<double>::infinity();
        for (int p = connections.ptr[ket]; p < connections.ptr[ket + 1u]; ++p) {
            const std::size_t record = static_cast<std::size_t>(p);
            const int bra = find_state(basis.view(), configurations[n + record]);
            if (bra < 0) throw std::runtime_error("unknown connection bra");

            const double value = connections.h[record];
            if (std::abs(value) > previous) {
                throw std::runtime_error("connection order");
            }
            previous = std::abs(value);
            column[static_cast<std::size_t>(bra)] = value;
        }

        for (std::size_t bra = 0; bra < n; ++bra) {
            const double value = matrix[bra * n + ket];
            const double expected = bra != ket && std::abs(value) >= eps ? value : 0.0;
            near(column[bra], expected, "connection element");
        }
    }
}

inline void check_local(
    const Hamiltonian& hamiltonian,
    const Basis& basis,
    double eps
) {
    const std::size_t n = basis.size();
    const auto matrix = dense(hamiltonian, basis);
    const auto local = hamiltonian.local_conn(basis.view(), eps, eps, 0, 7);
    const auto configurations = batch_view(local.nword, local.bra);

    if (!local.weak_coeff.empty() || local.strong_ptr.size() != n + 1u) {
        throw std::runtime_error("local connection shape");
    }

    for (std::size_t ket = 0; ket < n; ++ket) {
        near(local.diag[ket], matrix[ket * n + ket], "local diagonal");
        near(local.strong_degree[ket], degree(matrix, n, ket, eps), "local degree");

        for (int p = local.strong_ptr[ket]; p < local.strong_ptr[ket + 1u]; ++p) {
            const std::size_t record = static_cast<std::size_t>(p);
            const int bra = find_state(basis.view(), configurations[n + record]);
            if (bra < 0) throw std::runtime_error("unknown local bra");
            near(
                local.strong_h[record],
                matrix[static_cast<std::size_t>(bra) * n + ket],
                "local element"
            );
        }
    }

    bool rejected = false;
    try {
        static_cast<void>(hamiltonian.local_conn(
            basis.view(),
            eps,
            0.0,
            1,
            7
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("weak cutoff");

    rejected = false;
    try {
        static_cast<void>(hamiltonian.local_conn(
            basis.view(),
            eps,
            0.5 * eps,
            0,
            7
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("weak draws");
}

inline void check_weak(
    const Hamiltonian& hamiltonian,
    const Basis& basis,
    double eps1,
    double eps2
) {
    const std::size_t n = basis.size();
    const i64 n_draw = 32768;
    const auto matrix = dense(hamiltonian, basis);
    const auto local = hamiltonian.local_conn(
        basis.view(),
        eps1,
        eps2,
        n_draw,
        11
    );
    const auto configurations = batch_view(local.nword, local.bra);
    const std::size_t n_strong = static_cast<std::size_t>(local.strong_ptr.back());

    if (
        local.weak_ptr.size() != n + 1u
        || local.weak_coeff.size()
            != static_cast<std::size_t>(local.weak_ptr.back())
        || local.weak_coeff.size() > static_cast<std::size_t>(n_draw)
    ) {
        throw std::runtime_error("weak connection shape");
    }

    double estimate = 0.0;
    double expected = 0.0;
    for (std::size_t ket = 0; ket < n; ++ket) {
        near(local.diag[ket], matrix[ket * n + ket], "weak diagonal");
        near(local.strong_degree[ket], degree(matrix, n, ket, eps1), "strong degree");

        for (std::size_t bra = 0; bra < n; ++bra) {
            if (bra == ket) continue;
            const double h = matrix[bra * n + ket];
            const double abs_h = std::abs(h);
            if (abs_h >= eps2 && abs_h < eps1) {
                expected += h;
            }
        }

        for (int p = local.weak_ptr[ket]; p < local.weak_ptr[ket + 1u]; ++p) {
            const std::size_t record = static_cast<std::size_t>(p);
            const int bra = find_state(
                basis.view(),
                configurations[n + n_strong + record]
            );
            if (bra < 0) throw std::runtime_error("unknown weak bra");

            const double h = matrix[static_cast<std::size_t>(bra) * n + ket];
            if (!(std::abs(h) >= eps2 && std::abs(h) < eps1)) {
                throw std::runtime_error("weak window");
            }
            const double coeff = local.weak_coeff[record];
            if (!(coeff * h > 0.0)) throw std::runtime_error("weak coefficient");
            estimate += coeff;
        }
    }

    const double tolerance = 2.0e-2 * (1.0 + std::abs(expected));
    if (std::abs(estimate - expected) > tolerance) {
        throw std::runtime_error("weak estimator");
    }
}
