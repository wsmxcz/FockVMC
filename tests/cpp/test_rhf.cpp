#include "common.hpp"

int main() {
    const int norb = 5;
    const int na = 2;
    const int nb = 1;
    const double ecore = -0.13;
    const auto h1 = make_h1(norb);
    const auto eri = make_eri(norb);
    const auto ham = Hamiltonian::det(h1, norb, eri, ecore);
    const auto basis = det_basis(norb, na, nb, ham.nword());

    check_diag(ham, basis);
    check_matrix(ham, basis);
    check_conn(ham, basis, 0.10);
    check_sample(ham, basis, 0.20, 0.04);
    check_local(ham, basis, 0.20, 0.04, 0, LocalMode::unique);
    check_local(ham, basis, 0.20, 0.04, 5, LocalMode::unique);
    check_local(ham, basis, 0.20, 0.04, 0, LocalMode::flat);
    check_local(ham, basis, 0.20, 0.04, 5, LocalMode::flat);

    return 0;
}
