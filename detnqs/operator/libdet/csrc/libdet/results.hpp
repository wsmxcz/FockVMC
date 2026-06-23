#pragma once

#include <cstddef>
#include <vector>

#include <libdet/bit.hpp>

namespace libdet {

enum class AssembleMode : unsigned char {
    unique,
    flat,
};

struct Matrix {
    std::size_t n_bra = 0;
    std::size_t n_ket = 0;
    std::vector<i32> indptr;
    std::vector<i32> indices;
    std::vector<double> data;
};

struct Conns {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::size_t n_streams = 1;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i32> ptr;
    std::vector<i32> idx;
    std::vector<double> h;
    std::vector<double> degree;
};

struct LocalConn {
    u32 nword = 0;
    std::size_t n_kets = 0;
    std::vector<u64> bra;
    std::vector<double> diag;
    std::vector<i32> strong_ptr;
    std::vector<i32> strong_bra;
    std::vector<double> strong_h;
    std::vector<double> strong_degree;
    std::vector<i32> weak_ptr;
    std::vector<i32> weak_bra;
    std::vector<double> weak_h;
    std::vector<i64> weak_count;
    std::vector<double> weak_degree;
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
