from __future__ import annotations

import numpy as np

from detnqs import hilbert
from detnqs import operator


def test_libdet_module() -> None:
    import detnqs.operator.libdet as libdet

    assert libdet.__name__ == "detnqs.operator.libdet"
    assert hasattr(libdet.Hamiltonian, "det")
    assert hasattr(libdet.Hamiltonian, "spin")
    assert not hasattr(libdet.Hamiltonian, "rhf")
    assert not hasattr(libdet.Hamiltonian, "guga")


def test_det_action() -> None:
    sector = hilbert.DetSector(norb=3, n_alpha=1, n_beta=1)
    h1 = np.array(
        [
            [0.2, 0.4, -0.1],
            [0.4, 0.7, 0.3],
            [-0.1, 0.3, 1.1],
        ],
        dtype=np.float64,
    )
    eri = np.zeros((3, 3, 3, 3), dtype=np.float64)
    H = operator.Hamiltonian(sector, h1, eri, ecore=0.5)

    basis = sector.enumerate()
    dense = np.array(
        [[H.hij(bra[None], ket[None]) for ket in basis] for bra in basis],
        dtype=np.float64,
    )
    coeff = np.linspace(-0.7, 0.9, len(basis))

    np.testing.assert_allclose(H.matrix(basis).toarray(), dense)
    np.testing.assert_allclose(H.diag(basis), np.diag(dense))
    np.testing.assert_allclose(H.matvec(basis, coeff), dense @ coeff)

    bra = basis[::3]
    ket = basis[:4]
    scale = np.array([0.8, -0.5, 0.0, 0.2])
    known = H.project(bra, ket, scale)
    np.testing.assert_array_equal(known.bra, bra)
    np.testing.assert_allclose(known.hpsi, dense[::3, :4] @ scale)

    eps = 0.15
    terms = dense[:, :4] * scale
    hpsi = np.where(np.abs(terms) >= eps, terms, 0.0).sum(axis=1)
    key = lambda x: np.ascontiguousarray(x).tobytes()
    outside = {key(x) for x in basis[4:]}
    expected = {
        key(x): float(v)
        for x, v in zip(basis, hpsi, strict=True)
        if key(x) in outside and v != 0.0
    }
    generated = H.project(None, ket, scale, eps=eps, exclude=ket)
    actual = {
        key(x): float(v)
        for x, v in zip(generated.bra, generated.hpsi, strict=True)
    }
    assert actual == expected

    conn = H.conn(ket, eps)
    pool = np.asarray(conn.x)
    ptr = np.asarray(conn.ptr)
    idx = np.asarray(conn.bra)
    val = np.asarray(conn.h)
    for j in range(len(ket)):
        actual = {key(pool[idx[p]]): val[p] for p in range(ptr[j], ptr[j + 1])}
        expected = {
            key(basis[i]): dense[i, j]
            for i in range(len(basis))
            if i != j and abs(dense[i, j]) >= eps
        }
        assert actual == expected

    counts = np.array([3, 2, 1, 4], dtype=np.int64)
    draw = H.sample_conn(ket, counts, eps1=np.inf, eps2=0.0, seed=11)
    again = H.sample_conn(ket, counts, eps1=np.inf, eps2=0.0, seed=11)
    np.testing.assert_array_equal(draw.x, again.x)
    np.testing.assert_array_equal(draw.ptr, again.ptr)
    np.testing.assert_array_equal(draw.bra, again.bra)
    np.testing.assert_array_equal(draw.h, again.h)


def test_spin_action() -> None:
    sector = hilbert.SpinSector(norb=2, nelec=2, spin=0)
    h1 = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    H = operator.Hamiltonian(sector, h1, eri)

    basis = sector.enumerate()
    expected = np.array(
        [
            [2.0 * h1[1, 1], np.sqrt(2.0) * h1[0, 1], 0.0],
            [
                np.sqrt(2.0) * h1[0, 1],
                h1[0, 0] + h1[1, 1],
                np.sqrt(2.0) * h1[0, 1],
            ],
            [0.0, np.sqrt(2.0) * h1[0, 1], 2.0 * h1[0, 0]],
        ],
        dtype=np.float64,
    )
    coeff = np.array([0.4, -0.2, 0.8], dtype=np.float64)

    np.testing.assert_allclose(H.matrix(basis).toarray(), expected)
    np.testing.assert_allclose(H.diag(basis), np.diag(expected))
    np.testing.assert_allclose(H.matvec(basis, coeff), expected @ coeff)
    np.testing.assert_allclose(operator.S2(sector).diag(basis), 0.0)


def test_fermion_ops() -> None:
    sector = hilbert.DetSector(norb=4, n_alpha=2, n_beta=1)
    x = sector.reference(1)

    bra, sign, active = operator.annihilate(x, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_allclose(operator.number(bra), [2.0])

    ket, sign, active = operator.create(bra, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_array_equal(ket, x)
    np.testing.assert_allclose(operator.Sz(sector).diag(x), [0.5])
    np.testing.assert_allclose(operator.S2(sector).diag(x), [0.75])
