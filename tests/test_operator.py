from __future__ import annotations

import numpy as np

from detnqs import hilbert, operator


def dense(H: operator.Hamiltonian, basis: np.ndarray) -> np.ndarray:
    return np.array(
        [[H.hij(bra[None], ket[None]) for ket in basis] for bra in basis],
        dtype=np.float64,
    )


def eri(norb: int) -> np.ndarray:
    n2 = norb * (norb + 1) // 2
    return np.zeros(n2 * (n2 + 1) // 2, dtype=np.float64)


def test_det_core() -> None:
    sec = hilbert.DetSector(norb=3, nelec=2, spin=0)
    h1 = np.array(
        [
            [0.2, 0.4, -0.1],
            [0.4, 0.7, 0.3],
            [-0.1, 0.3, 1.1],
        ],
        dtype=np.float64,
    )

    H = operator.Hamiltonian(sec, h1, eri(sec.norb), ecore=0.5)
    basis = sec.enumerate()
    mat = dense(H, basis)
    vec = np.linspace(-0.7, 0.9, len(basis))

    np.testing.assert_allclose(H.matrix(basis).toarray(), mat)
    np.testing.assert_allclose(H.diag(basis), np.diag(mat))
    np.testing.assert_allclose(H.matvec(basis, vec), mat @ vec)

    con = H.conn(basis[:4], eps=0.15)
    pool = np.asarray(con.bra)
    ptr = np.asarray(con.ptr)
    val = np.asarray(con.h)
    n_ket = 4

    for j, ket in enumerate(basis[:4]):
        got = {
            np.ascontiguousarray(pool[n_ket + p]).tobytes(): val[p]
            for p in range(ptr[j], ptr[j + 1])
        }
        exp = {
            np.ascontiguousarray(bra).tobytes(): mat[i, j]
            for i, bra in enumerate(basis)
            if not np.array_equal(bra, ket) and abs(mat[i, j]) >= 0.15
        }
        assert got == exp


def test_fermion() -> None:
    sec = hilbert.DetSector(norb=4, nelec=3, spin=1)
    x = sec.reference(1)

    bra, sgn, active = operator.annihilate(x, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sgn, [-1.0])
    np.testing.assert_allclose(operator.number(bra), [2.0])

    ket, sgn, active = operator.create(bra, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sgn, [-1.0])
    np.testing.assert_array_equal(ket, x)

    np.testing.assert_allclose(operator.Sz(sec).diag(x), [0.5])
    np.testing.assert_allclose(operator.S2(sec).diag(x), [0.75])
