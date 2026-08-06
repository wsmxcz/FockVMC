from __future__ import annotations

import numpy as np

from detnqs import Hamiltonian
from detnqs.hilbert import DetSector
from detnqs.operator import S2, annihilate, create, number


def _packed_eri(norb: int) -> np.ndarray:
    npair = norb * (norb + 1) // 2
    return np.zeros(npair * (npair + 1) // 2, dtype=np.float64)


def _dense(hamiltonian: Hamiltonian, basis: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [
                hamiltonian.hij(bra[None], ket[None])
                for ket in basis
            ]
            for bra in basis
        ]
    )


def _system() -> tuple[Hamiltonian, np.ndarray]:
    sector = DetSector(norb=3, nelec=2, spin=0)
    h1 = np.array(
        [[0.2, 0.4, -0.1], [0.4, 0.7, 0.3], [-0.1, 0.3, 1.1]],
        dtype=np.float64,
    )
    hamiltonian = Hamiltonian(sector, h1, _packed_eri(sector.norb), ecore=0.5)
    basis = sector.enumerate()
    return hamiltonian, basis


def test_action() -> None:
    hamiltonian, basis = _system()
    matrix = _dense(hamiltonian, basis)
    vector = np.linspace(-0.7, 0.9, len(basis))

    np.testing.assert_allclose(hamiltonian.matrix(basis).toarray(), matrix)
    np.testing.assert_allclose(hamiltonian.diag(basis), np.diag(matrix))
    np.testing.assert_allclose(hamiltonian.matvec(basis, vector), matrix @ vector)


def test_connections() -> None:
    hamiltonian, basis = _system()
    matrix = _dense(hamiltonian, basis)
    connections = hamiltonian.conn(basis[:4], eps=0.15)
    bra = np.asarray(connections.bra)
    ptr = np.asarray(connections.ptr)
    value = np.asarray(connections.h)

    for j, ket in enumerate(basis[:4]):
        actual = {
            np.ascontiguousarray(bra[4 + p]).tobytes(): value[p]
            for p in range(ptr[j], ptr[j + 1])
        }
        expected = {
            np.ascontiguousarray(x).tobytes(): matrix[i, j]
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
