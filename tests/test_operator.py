import numpy as np

from detnqs import Hamiltonian
from detnqs.hilbert import DetSector
from detnqs.operator import S2, annihilate, create, number


def test_hamiltonian() -> None:
    sector = DetSector(norb=3, nelec=2, spin=0)
    h1 = np.array(
        [[0.2, 0.4, -0.1], [0.4, 0.7, 0.3], [-0.1, 0.3, 1.1]],
        dtype=np.float64,
    )
    npair = sector.norb * (sector.norb + 1) // 2
    eri = np.zeros(npair * (npair + 1) // 2)
    hamiltonian = Hamiltonian(sector, h1, eri, ecore=0.5)
    basis = sector.enumerate()
    matrix = np.array(
        [
            [hamiltonian.hij(bra[None], ket[None]) for ket in basis]
            for bra in basis
        ]
    )
    vector = np.linspace(-0.7, 0.9, len(basis))

    np.testing.assert_allclose(hamiltonian.matrix(basis).toarray(), matrix)
    np.testing.assert_allclose(hamiltonian.diag(basis), np.diag(matrix))
    np.testing.assert_allclose(hamiltonian.matvec(basis, vector), matrix @ vector)

    connections = hamiltonian.conn(basis[:4], eps=0.15)
    bra = connections.bra
    ptr = connections.ptr
    value = connections.h

    for j, ket in enumerate(basis[:4]):
        actual = {
            bra[4 + p].tobytes(): value[p]
            for p in range(ptr[j], ptr[j + 1])
        }
        expected = {
            x.tobytes(): matrix[i, j]
            for i, x in enumerate(basis)
            if not np.array_equal(x, ket) and abs(matrix[i, j]) >= 0.15
        }
        assert actual == expected


def test_fermion() -> None:
    sector = DetSector(norb=4, nelec=3, spin=1)
    x = sector.reference(1)

    bra, sign, active = annihilate(x, 0, 1)
    np.testing.assert_array_equal(active, [True])
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_allclose(number(bra), [2.0])

    ket, sign, active = create(bra, 0, 1)
    np.testing.assert_array_equal(active, [True])
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_array_equal(ket, x)
    np.testing.assert_allclose(S2(sector).diag(x), [0.75])
