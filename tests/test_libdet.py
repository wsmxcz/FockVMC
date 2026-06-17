from __future__ import annotations

import numpy as np

import libdet


EMPTY = 0
LOWER = 1
UPPER = 2
DOUBLY = 3


def _key(det: np.ndarray) -> bytes:
    return np.ascontiguousarray(det).tobytes()


def _values(dets: np.ndarray, values: np.ndarray) -> dict[bytes, float]:
    return {
        _key(det): float(value)
        for det, value in zip(dets, values, strict=True)
    }


def _as_det(alpha: int, beta: int) -> np.ndarray:
    det = np.zeros((1, 2, 1), dtype=np.uint64)
    det[0, 0, 0] = np.uint64(alpha)
    det[0, 1, 0] = np.uint64(beta)
    return det


def _encode_steps(steps: list[int]) -> np.ndarray:
    alpha = 0
    beta = 0
    for p, step in enumerate(steps):
        if step in (UPPER, DOUBLY):
            alpha |= 1 << p
        if step in (LOWER, DOUBLY):
            beta |= 1 << p
    return _as_det(alpha, beta)[0]


def _guga_paths(norb: int, nelec: int, spin_twice: int) -> np.ndarray:
    steps: list[int] = [EMPTY] * norb
    dets = []

    def rec(p: int, n: int, spin: int) -> None:
        if p == norb:
            if n == nelec and spin == spin_twice:
                dets.append(_encode_steps(steps))
            return

        for step, occ, delta in (
            (EMPTY, 0, 0),
            (LOWER, 1, -1),
            (UPPER, 1, 1),
            (DOUBLY, 2, 0),
        ):
            if n + occ > nelec or spin + delta < 0:
                continue
            steps[p] = step
            rec(p + 1, n + occ, spin + delta)

    rec(0, 0, 0)
    return np.ascontiguousarray(dets, dtype=np.uint64)


def _cg_coeff(
    spin_before: int,
    m_before: int,
    electron_spin: int,
    spin_after: int,
) -> float:
    b = spin_before
    m = m_before
    denom = 2.0 * (b + 1)

    if spin_after == b + 1:
        num = b + m + 2 if electron_spin > 0 else b - m + 2
        return 0.0 if num <= 0 else float(np.sqrt(num / denom))

    if spin_after == b - 1:
        num = b - m if electron_spin > 0 else b + m
        if num <= 0:
            return 0.0
        value = float(np.sqrt(num / denom))
        return -value if electron_spin > 0 else value

    return 0.0


def _decode_steps(det: np.ndarray, norb: int) -> tuple[list[int], list[int]]:
    steps = []
    spin = [0]
    alpha = int(det[0, 0])
    beta = int(det[1, 0])
    for p in range(norb):
        step = ((alpha >> p) & 1) * 2 + ((beta >> p) & 1)
        steps.append(step)
        delta = 1 if step == UPPER else -1 if step == LOWER else 0
        spin.append(spin[-1] + delta)
    return steps, spin


def _canonical_sign(alpha: int, beta: int, norb: int) -> float:
    parity = 0
    for p in range(norb):
        if not ((beta >> p) & 1):
            continue
        for q in range(p + 1, norb):
            parity ^= (alpha >> q) & 1
    return -1.0 if parity else 1.0


def _csf_terms(det: np.ndarray, norb: int) -> dict[tuple[int, int], float]:
    steps, spin = _decode_steps(det, norb)
    current = [(0, 0, 0, 1.0)]

    for p, step in enumerate(steps):
        if step == EMPTY:
            continue
        if step == DOUBLY:
            current = [
                (alpha | (1 << p), beta | (1 << p), m, coeff)
                for alpha, beta, m, coeff in current
            ]
            continue

        next_terms = []
        for alpha, beta, m, coeff in current:
            for sigma in (1, -1):
                c = _cg_coeff(spin[p], m, sigma, spin[p + 1])
                if c == 0.0:
                    continue
                if sigma > 0:
                    next_terms.append(
                        (alpha | (1 << p), beta, m + sigma, coeff * c)
                    )
                else:
                    next_terms.append(
                        (alpha, beta | (1 << p), m + sigma, coeff * c)
                    )
        current = next_terms

    target_m = spin[-1]
    out: dict[tuple[int, int], float] = {}
    for alpha, beta, m, coeff in current:
        if m != target_m or coeff == 0.0:
            continue
        out[(alpha, beta)] = coeff * _canonical_sign(alpha, beta, norb)
    return out


def _guga_ref(
    h1: np.ndarray,
    eri: np.ndarray,
    ecore: float,
    dets: np.ndarray,
) -> np.ndarray:
    norb = h1.shape[0]
    rhf = libdet.Hamiltonian.rhf(h1, eri, ecore=ecore)
    terms = [_csf_terms(det, norb) for det in dets]
    out = np.zeros((len(dets), len(dets)), dtype=np.float64)

    for ibra, bra_terms in enumerate(terms):
        for iket, ket_terms in enumerate(terms):
            value = 0.0
            for (ba, bb), bc in bra_terms.items():
                for (ka, kb), kc in ket_terms.items():
                    value += bc * kc * rhf.hij(_as_det(ba, bb), _as_det(ka, kb))
            out[ibra, iket] = value

    return out


def _pair_index(p: int, q: int) -> int:
    hi, lo = max(p, q), min(p, q)
    return hi * (hi + 1) // 2 + lo


def _small_integrals(norb: int = 3) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(31)
    h1 = rng.normal(scale=0.5, size=(norb, norb))
    h1 = 0.5 * (h1 + h1.T)

    npair = norb * (norb + 1) // 2
    pair_eri = rng.normal(scale=0.15, size=(npair, npair))
    pair_eri = 0.5 * (pair_eri + pair_eri.T)

    eri = np.empty((norb, norb, norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    eri[p, q, r, s] = pair_eri[
                        _pair_index(p, q),
                        _pair_index(r, s),
                    ]

    return h1, eri, 0.17


def test_finite(toy) -> None:
    ham, dets, dense = toy.ham, toy.dets, toy.matrix
    x = np.linspace(-0.8, 1.1, len(dets))
    x2 = np.column_stack((x, x * x))

    np.testing.assert_allclose(ham.matrix(dets).toarray(), dense)
    np.testing.assert_allclose(ham.diag(dets), np.diag(dense))
    np.testing.assert_allclose(ham.matvec(dets, x), dense @ x)
    np.testing.assert_allclose(ham.matvec(dets, x2), dense @ x2)

    empty = dets[:0]
    assert ham.matrix(empty).shape == (0, 0)
    assert ham.matvec(empty, np.empty(0)).shape == (0,)


def test_project(toy) -> None:
    ham, dets, dense, eps = toy.ham, toy.dets, toy.matrix, toy.eps
    kets = dets[:5]
    coeffs = np.array([0.9, -0.7, 0.4, 0.0, 0.2])
    bras = dets[1::3]

    known = ham.project(bras, kets, coeffs)
    np.testing.assert_array_equal(known.bra, bras)
    np.testing.assert_allclose(known.hpsi, dense[1::3, :5] @ coeffs)

    terms = dense[:, :5] * coeffs
    expected = np.where(np.abs(terms) >= eps, terms, 0.0).sum(axis=1)
    external = {_key(det) for det in dets[5:]}
    expected = {
        _key(det): float(value)
        for det, value in zip(dets, expected, strict=True)
        if _key(det) in external and value != 0.0
    }

    expanded = ham.expand(kets, eps, coeffs=coeffs, exclude=kets)
    generated = ham.project(None, kets, coeffs, eps=eps, exclude=kets)

    assert {_key(det) for det in expanded} == set(expected)
    assert _values(generated.bra, generated.hpsi) == expected


def test_conns(toy) -> None:
    ham, dets, dense, eps = toy.ham, toy.dets, toy.matrix, toy.eps
    kets = dets[:6]
    conns = ham.conns(kets, eps)
    cached = ham.conns(kets, eps)
    weight, count = ham.degrees(kets, eps)

    for name in ("x", "ket_ptr", "bra_idx", "h", "weight"):
        np.testing.assert_array_equal(
            np.asarray(getattr(conns, name)),
            np.asarray(getattr(cached, name)),
        )

    pool = np.asarray(conns.x)
    ptr = np.asarray(conns.ket_ptr)
    bra_idx = np.asarray(conns.bra_idx)
    h = np.asarray(conns.h)

    for iket in range(len(kets)):
        expected = {
            _key(dets[ibra]): dense[ibra, iket]
            for ibra in range(len(dets))
            if ibra != iket and abs(dense[ibra, iket]) >= eps
        }
        actual = {
            _key(pool[bra_idx[pos]]): h[pos]
            for pos in range(ptr[iket], ptr[iket + 1])
        }
        assert actual == expected
        assert count[iket] == len(expected)
        np.testing.assert_allclose(weight[iket], sum(map(abs, expected.values())))


def test_sampling(toy) -> None:
    ham, dets, dense = toy.ham, toy.dets, toy.matrix
    kets = dets[:5]
    coeffs = np.array([0.9, -0.7, 0.4, 0.3, -0.2])
    counts = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]])

    conns = ham.sample_conns(kets, counts, eps1=np.inf, eps2=0.0, seed=19)
    again = ham.sample_conns(kets, counts, eps1=np.inf, eps2=0.0, seed=19)
    for name in ("x", "ket_ptr", "bra_idx", "h", "count", "weight"):
        np.testing.assert_array_equal(
            np.asarray(getattr(conns, name)),
            np.asarray(getattr(again, name)),
        )

    ptr = np.asarray(conns.ket_ptr)
    hits = np.asarray(conns.count)
    for stream in range(counts.shape[0]):
        for iket in range(len(kets)):
            block = stream * len(kets) + iket
            actual = hits[ptr[block] : ptr[block + 1]].sum()
            expected = counts[stream, iket] if conns.weight[iket] > 0 else 0
            assert actual == expected

    external = np.abs(dense[5:, :5] * coeffs)
    eps1 = float(np.quantile(external[external > 0.0], 0.65))
    sample = ham.sample_project(
        kets,
        coeffs,
        eps1,
        0.0,
        6,
        exclude=kets,
        n_rep=2,
        seed=29,
    )
    repeated = ham.sample_project(
        kets,
        coeffs,
        eps1,
        0.0,
        6,
        exclude=kets,
        n_rep=2,
        seed=29,
    )

    for name in ("rep_ptr", "bra", "hpsi_strong", "hpsi_a", "hpsi_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(sample, name)),
            np.asarray(getattr(repeated, name)),
        )

    exact = ham.project(sample.bra, kets, coeffs, eps=eps1)
    np.testing.assert_allclose(sample.hpsi_strong, exact.hpsi)
    np.testing.assert_allclose(sample.diag, exact.diag)


def test_guga_singlet() -> None:
    h1 = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    ham = libdet.Hamiltonian.guga(
        h1,
        eri,
        n_alpha=1,
        n_beta=1,
    )

    dets = np.zeros((3, 2, 1), dtype=np.uint64)
    dets[0, 0, 0] = np.uint64(1) << np.uint64(0)
    dets[0, 1, 0] = np.uint64(1) << np.uint64(0)
    dets[1, 0, 0] = np.uint64(1) << np.uint64(0)
    dets[1, 1, 0] = np.uint64(1) << np.uint64(1)
    dets[2, 0, 0] = np.uint64(1) << np.uint64(1)
    dets[2, 1, 0] = np.uint64(1) << np.uint64(1)

    expected = np.array(
        [
            [2.0 * h1[0, 0], np.sqrt(2.0) * h1[0, 1], 0.0],
            [
                np.sqrt(2.0) * h1[0, 1],
                h1[0, 0] + h1[1, 1],
                np.sqrt(2.0) * h1[0, 1],
            ],
            [0.0, np.sqrt(2.0) * h1[0, 1], 2.0 * h1[1, 1]],
        ],
        dtype=np.float64,
    )

    matrix = ham.matrix(dets).toarray()
    np.testing.assert_allclose(matrix, expected, atol=1.0e-12)
    np.testing.assert_allclose(ham.diag(dets), np.diag(expected), atol=1.0e-12)
    np.testing.assert_allclose(
        ham.matvec(dets, np.array([0.4, -0.2, 0.8])),
        expected @ np.array([0.4, -0.2, 0.8]),
        atol=1.0e-12,
    )

    coeffs = np.array([0.4, -0.2, 0.8], dtype=np.float64)
    known = ham.project(dets, dets, coeffs)
    np.testing.assert_array_equal(known.bra, dets)
    np.testing.assert_allclose(known.hpsi, expected @ coeffs, atol=1.0e-12)

    expanded = ham.expand(dets[:1], 0.0, exclude=dets[:1])
    assert expanded.shape == (1, 2, 1)
    np.testing.assert_array_equal(expanded[0], dets[1])


def test_guga_conns() -> None:
    h1 = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)
    ham = libdet.Hamiltonian.guga(
        h1,
        eri,
        n_alpha=1,
        n_beta=1,
    )

    kets = np.zeros((1, 2, 1), dtype=np.uint64)
    kets[0, 0, 0] = np.uint64(1) << np.uint64(0)
    kets[0, 1, 0] = np.uint64(1) << np.uint64(0)

    conns = ham.conns(kets, 0.0)
    assert np.asarray(conns.h).shape == (1,)
    np.testing.assert_allclose(conns.h, [np.sqrt(2.0) * h1[0, 1]])

    sample = ham.sample_conns(
        kets,
        np.array([6], dtype=np.int64),
        eps1=np.inf,
        eps2=0.0,
        seed=3,
    )
    assert int(np.asarray(sample.count).sum()) == 6


def test_guga_matrix() -> None:
    h1, eri, ecore = _small_integrals()
    norb = h1.shape[0]
    dets = _guga_paths(norb, nelec=3, spin_twice=1)
    ham = libdet.Hamiltonian.guga(h1, eri, n_alpha=2, n_beta=1, ecore=ecore)

    expected = _guga_ref(h1, eri, ecore, dets)
    actual = ham.matrix(dets).toarray()

    np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    np.testing.assert_allclose(ham.diag(dets), np.diag(expected), atol=1.0e-11)

    x = np.linspace(-0.3, 0.6, len(dets))
    x2 = np.column_stack((x, x[::-1]))
    np.testing.assert_allclose(ham.matvec(dets, x), expected @ x, atol=1.0e-11)
    np.testing.assert_allclose(ham.matvec(dets, x2), expected @ x2, atol=1.0e-11)

    coeffs = np.linspace(0.2, -0.4, len(dets))
    projection = ham.project(dets[::2], dets, coeffs)
    np.testing.assert_array_equal(projection.bra, dets[::2])
    np.testing.assert_allclose(projection.hpsi, expected[::2] @ coeffs, atol=1.0e-11)


def test_guga_external() -> None:
    h1, eri, ecore = _small_integrals()
    norb = h1.shape[0]
    dets = _guga_paths(norb, nelec=3, spin_twice=1)
    ham = libdet.Hamiltonian.guga(h1, eri, n_alpha=2, n_beta=1, ecore=ecore)
    dense = _guga_ref(h1, eri, ecore, dets)

    kets = dets[:4]
    coeffs = np.array([0.6, -0.3, 0.2, 0.5], dtype=np.float64)
    terms = np.abs(dense[:, : len(kets)] * coeffs)
    eps = float(np.quantile(terms[terms > 0.0], 0.45))

    expanded = ham.expand(kets, eps, coeffs=coeffs, exclude=kets)
    generated = ham.project(None, kets, coeffs, eps=eps, exclude=kets)

    expected_hpsi = np.where(
        terms >= eps,
        dense[:, : len(kets)] * coeffs,
        0.0,
    ).sum(axis=1)
    external = {_key(det) for det in dets[len(kets) :]}
    expected = {
        _key(det): float(value)
        for det, value in zip(dets, expected_hpsi, strict=True)
        if _key(det) in external and value != 0.0
    }

    assert {_key(det) for det in expanded} == set(expected)
    assert _values(generated.bra, generated.hpsi) == expected

    conns = ham.conns(kets, eps)
    weight, count = ham.degrees(kets, eps)
    pool = np.asarray(conns.x)
    ptr = np.asarray(conns.ket_ptr)
    bra_idx = np.asarray(conns.bra_idx)
    h = np.asarray(conns.h)

    for iket in range(len(kets)):
        expected_conn = {
            _key(dets[ibra]): dense[ibra, iket]
            for ibra in range(len(dets))
            if ibra != iket and abs(dense[ibra, iket]) >= eps
        }
        actual_conn = {
            _key(pool[bra_idx[pos]]): h[pos]
            for pos in range(ptr[iket], ptr[iket + 1])
        }
        assert actual_conn == expected_conn
        assert count[iket] == len(expected_conn)
        np.testing.assert_allclose(weight[iket], sum(map(abs, expected_conn.values())))

    sample = ham.sample_conns(
        kets,
        np.array([4, 3, 2, 1], dtype=np.int64),
        eps1=np.inf,
        eps2=0.0,
        seed=41,
    )
    hits = np.asarray(sample.count)
    ptr = np.asarray(sample.ket_ptr)
    for iket in range(len(kets)):
        actual = hits[ptr[iket] : ptr[iket + 1]].sum()
        expected_count = [4, 3, 2, 1][iket] if sample.weight[iket] > 0 else 0
        assert actual == expected_count
