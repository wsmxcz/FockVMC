#include "check.hpp"

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
    check_action(ham, basis);
    check_conn(ham, basis, 0.08, AssembleMode::unique);
    check_conn(ham, basis, 0.08, AssembleMode::flat);
    check_sample(ham, basis, 0.18, 0.03, AssembleMode::unique);
    check_sample(ham, basis, 0.18, 0.03, AssembleMode::flat);
    check_local(ham, basis, 0.18, 0.03, 0, AssembleMode::unique);
    check_local(ham, basis, 0.18, 0.03, 5, AssembleMode::unique);
    check_local(ham, basis, 0.18, 0.03, 0, AssembleMode::flat);
    check_local(ham, basis, 0.18, 0.03, 5, AssembleMode::flat);
    check_project(ham, basis, 0.04);
    check_sample_project(ham, basis);

    return 0;
}
