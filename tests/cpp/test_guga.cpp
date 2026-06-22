#include "common.hpp"

int main() {
    const int norb = 5;
    const int na = 2;
    const int nb = 1;
    const double ecore = -0.11;
    const auto h1 = make_h1(norb);
    const auto eri = make_eri(norb);
    const auto ham = Hamiltonian::spin(h1, norb, eri, na, nb, ecore);
    const auto basis = spin_basis(norb, na, nb, ham.nword());

    check_diag(ham, basis);
    check_matrix(ham, basis);
    check_conn(ham, basis, 0.08);
    check_sample(ham, basis, 0.18, 0.03);
    check_local(ham, basis, 0.18, 0.03, 0);
    check_local(ham, basis, 0.18, 0.03, 5);

    return 0;
}
