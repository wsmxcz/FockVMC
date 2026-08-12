import numpy as np

from fvmc import Hamiltonian
from fvmc.hilbert import DetSector
from fvmc.operator import annihilate, create, number


def test_hamiltonian() -> None:
    sector = DetSector(norb=3, nelec=2, spin=0)
    h1 = np.array(
        [[0.2, 0.4, -0.1], [0.4, 0.7, 0.3], [-0.1, 0.3, 1.1]],
        dtype=np.float64,
    )
    eri = np.linspace(-0.2, 0.3, 21, dtype=np.float64)
    ham = Hamiltonian(sector, h1, eri, ecore=0.5)
    basis = sector.enumerate()
    dense = np.array(
        [[ham.hij(bra[None], ket[None]) for ket in basis] for bra in basis]
    )

    np.testing.assert_allclose(ham.matrix(basis).toarray(), dense)
    np.testing.assert_allclose(ham.diag(basis), np.diag(dense))

    vec = np.arange(2 * len(basis), dtype=np.float64).reshape(len(basis), 2)
    np.testing.assert_allclose(ham.matvec(basis, vec[:, 0]), dense @ vec[:, 0])
    np.testing.assert_allclose(ham.matvec(basis, vec), dense @ vec)

    kets = basis[:3]
    conn = ham.conn(kets, eps=0.25)
    for ket, begin, end in zip(kets, conn.ptr[:-1], conn.ptr[1:], strict=True):
        values = {
            conn.bra[len(kets) + p].tobytes(): conn.h[p]
            for p in range(begin, end)
        }
        expected = {}
        for bra in basis:
            value = ham.hij(bra[None], ket[None])
            if not np.array_equal(bra, ket) and abs(value) >= 0.25:
                expected[bra.tobytes()] = value
        assert values == expected


def test_fermion() -> None:
    sector = DetSector(norb=65, nelec=3, spin=1)
    ket = sector.zeros(1)
    ket[0, 0] = (1, 1)
    ket[0, 1, 0] = 2

    bra, sign, active = annihilate(ket, 0, 64)
    np.testing.assert_array_equal(active, True)
    np.testing.assert_array_equal(sign, -1.0)
    np.testing.assert_array_equal(number(bra), 2.0)

    restored, sign, active = create(bra, 0, 64)
    np.testing.assert_array_equal(active, True)
    np.testing.assert_array_equal(sign, -1.0)
    np.testing.assert_array_equal(restored, ket)
