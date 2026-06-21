from __future__ import annotations

import math

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


def ints(norb: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    h1 = rng.normal(size=(norb, norb))
    h1 = np.ascontiguousarray(0.5 * (h1 + h1.T))

    n2 = norb * (norb + 1) // 2
    h2 = rng.normal(scale=0.2, size=n2 * (n2 + 1) // 2)

    return h1, np.ascontiguousarray(h2)


def word(a: int, b: int, nword: int) -> np.ndarray:
    x = np.zeros((2, nword), dtype=np.uint64)
    x[0, 0] = np.uint64(a)
    x[1, 0] = np.uint64(b)
    return x


def step(x: np.ndarray, p: int) -> int:
    up = int((int(x[0, p >> 6]) >> (p & 63)) & 1)
    dn = int((int(x[1, p >> 6]) >> (p & 63)) & 1)
    return 2 * up + dn


def gauge(x: np.ndarray, norb: int) -> float:
    single = 0
    parity = 0

    for p in range(norb - 1, -1, -1):
        st = step(x, p)
        if st in (1, 3):
            parity ^= single
        if st in (1, 2):
            single ^= 1

    return -1.0 if parity else 1.0


def phase(a: int, b: int, norb: int) -> float:
    n = 0
    for p in range(norb):
        if (b >> p) & 1:
            n += (a >> (p + 1)).bit_count()
    return -1.0 if n & 1 else 1.0


def cg(b: int, m: int, sig: int, a: int) -> float:
    den = 2.0 * (b + 1)

    if a == b + 1:
        num = b + m + 2 if sig > 0 else b - m + 2
        return 0.0 if num <= 0 else math.sqrt(num / den)

    if a == b - 1:
        num = b - m if sig > 0 else b + m
        if num <= 0:
            return 0.0
        val = math.sqrt(num / den)
        return -val if sig > 0 else val

    return 0.0


def csfs(x: np.ndarray, norb: int, spin: int):
    steps = [step(x, p) for p in range(norb)]
    bs = [0]

    for st in steps:
        bs.append(bs[-1] + (1 if st == 2 else -1 if st == 1 else 0))

    out = []

    def rec(p: int, m: int, c: float, a: int, b: int) -> None:
        if p == norb:
            if m == spin:
                out.append((a, b, c * phase(a, b, norb)))
            return

        st = steps[p]

        if st == 0:
            rec(p + 1, m, c, a, b)
            return

        if st == 3:
            rec(p + 1, m, c, a | (1 << p), b | (1 << p))
            return

        ca = cg(bs[p], m, 1, bs[p + 1])
        cb = cg(bs[p], m, -1, bs[p + 1])

        if ca:
            rec(p + 1, m + 1, c * ca, a | (1 << p), b)
        if cb:
            rec(p + 1, m - 1, c * cb, a, b | (1 << p))

    rec(0, 0, 1.0, 0, 0)
    return out


def ref_hij(
    h1: np.ndarray,
    h2: np.ndarray,
    ecore: float,
    bra: np.ndarray,
    ket: np.ndarray,
    spin: int,
) -> float:
    norb = h1.shape[0]
    nelec = int(sum(step(ket, p).bit_count() for p in range(norb)))
    na = (nelec + spin) // 2
    nb = (nelec - spin) // 2
    nword = ket.shape[-1]

    sec = hilbert.DetSector(norb=norb, n_alpha=na, n_beta=nb)
    H = operator.Hamiltonian(sec, h1, h2, ecore=ecore)

    val = 0.0
    for ba, bb, bc in csfs(bra, norb, spin):
        bdet = word(ba, bb, nword)
        for ka, kb, kc in csfs(ket, norb, spin):
            kdet = word(ka, kb, nword)
            val += bc * kc * H.hij(bdet[None], kdet[None])

    return gauge(bra, norb) * gauge(ket, norb) * val


def test_det_core() -> None:
    sec = hilbert.DetSector(norb=3, n_alpha=1, n_beta=1)
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
    pool = np.asarray(con.x)
    ptr = np.asarray(con.ptr)
    idx = np.asarray(con.bra)
    val = np.asarray(con.h)

    for j, ket in enumerate(basis[:4]):
        got = {
            np.ascontiguousarray(pool[idx[p]]).tobytes(): val[p]
            for p in range(ptr[j], ptr[j + 1])
        }
        exp = {
            np.ascontiguousarray(bra).tobytes(): mat[i, j]
            for i, bra in enumerate(basis)
            if not np.array_equal(bra, ket) and abs(mat[i, j]) >= 0.15
        }
        assert got == exp


def test_spin_core() -> None:
    sec = hilbert.SpinSector(norb=2, nelec=2, spin=0)
    h1 = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=np.float64)
    H = operator.Hamiltonian(sec, h1, eri(sec.norb))

    basis = sec.enumerate()
    mat = np.array(
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

    np.testing.assert_allclose(H.matrix(basis).toarray(), mat)
    np.testing.assert_allclose(H.diag(basis), np.diag(mat))
    np.testing.assert_allclose(operator.S2(sec).diag(basis), 0.0)


def test_fermion() -> None:
    sec = hilbert.DetSector(norb=4, n_alpha=2, n_beta=1)
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


def test_guga_dense() -> None:
    for norb, nelec, spin, seed in [(3, 2, 0, 3), (4, 3, 1, 5)]:
        sec = hilbert.SpinSector(norb=norb, nelec=nelec, spin=spin)
        h1, h2 = ints(norb, seed)
        ecore = -0.37

        H = operator.Hamiltonian(sec, h1, h2, ecore=ecore)
        basis = sec.enumerate()

        mat = dense(H, basis)
        ref = np.array(
            [[ref_hij(h1, h2, ecore, bra, ket, spin) for ket in basis] for bra in basis],
            dtype=np.float64,
        )

        np.testing.assert_allclose(mat, ref, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(H.matrix(basis).toarray(), ref, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(H.diag(basis), np.diag(ref), atol=2e-12, rtol=2e-12)


def test_guga_conn() -> None:
    sec = hilbert.SpinSector(norb=4, nelec=4, spin=0)
    h1, h2 = ints(sec.norb, seed=11)
    H = operator.Hamiltonian(sec, h1, h2, ecore=0.2)

    basis = sec.enumerate()
    mat = dense(H, basis)
    con = H.conn(basis[:3], eps=0.05)

    pool = np.asarray(con.x)
    ptr = np.asarray(con.ptr)
    idx = np.asarray(con.bra)
    val = np.asarray(con.h)
    ids = {np.ascontiguousarray(x).tobytes(): i for i, x in enumerate(basis)}

    for j in range(3):
        got = {
            ids[np.ascontiguousarray(pool[idx[p]]).tobytes()]: val[p]
            for p in range(ptr[j], ptr[j + 1])
        }
        exp = {
            i: mat[i, j]
            for i in range(len(basis))
            if i != j and abs(mat[i, j]) >= 0.05
        }

        assert got.keys() == exp.keys()
        np.testing.assert_allclose(
            [got[i] for i in got],
            [exp[i] for i in got],
            atol=2e-12,
            rtol=2e-12,
        )