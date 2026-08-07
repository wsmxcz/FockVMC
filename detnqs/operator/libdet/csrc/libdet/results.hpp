#pragma once

#include <cstddef>
#include <vector>

#include <libdet/bit.hpp>

namespace libdet {

struct Matrix {
    std::size_t n_bra = 0;
    std::size_t n_ket = 0;
    std::vector<i32> indptr;
    std::vector<i32> indices;
    std::vector<double> data;
};

struct Conns {
    // bra = [kets, connection records]; record r is bra[n_kets + r].
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i32> ptr;
    std::vector<double> h;
    std::vector<double> degree;
};

struct LocalConn {
    // bra = [kets, strong records, weak records].
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i32> strong_ptr;
    std::vector<double> strong_h;
    std::vector<double> strong_degree;
    std::vector<i32> weak_ptr;
    std::vector<double> weak_coeff;
};

struct Projection {
    u32 nword = 0;
    std::vector<u64> bra;
    std::vector<double> hpsi;
    std::vector<double> diag;
};

struct Projections {
    u32 nword = 0;
    std::size_t n_streams = 0;
    std::vector<u64> bra;
    std::vector<double> hpsi;
    std::vector<double> diag;
};

} // namespace libdet
