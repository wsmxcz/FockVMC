#include "check.hpp"

#include <limits>

int main() {
    const int norb = 5;
    const int na = 2;
    const int nb = 1;
    const double ecore = -0.13;
    const auto h1 = make_h1(norb);
    const auto eri = make_eri(norb);
    const Hamiltonian ham(h1, norb, eri, ecore);
    const auto basis = det_basis(norb, na, nb);

    check_matrix(ham, basis);
    check_conn(ham, basis, 0.10);
    check_local(ham, basis, 0.10);
    check_weak(ham, basis, 0.10, 0.025);
    check_weak(ham, basis, std::numeric_limits<double>::infinity(), 0.025);

    return 0;
}
