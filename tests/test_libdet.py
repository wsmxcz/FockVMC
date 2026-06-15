from __future__ import annotations

import numpy as np


def _key(det: np.ndarray) -> bytes:
    return np.ascontiguousarray(det).tobytes()


def _values(dets: np.ndarray, values: np.ndarray) -> dict[bytes, float]:
    return {
        _key(det): float(value)
        for det, value in zip(dets, values, strict=True)
    }


def test_finite(toy) -> None:
    ham, dets, dense = toy.ham, toy.dets, toy.matrix
    x = np.linspace(-0.8, 1.1, len(dets))
    x2 = np.column_stack((x, x * x))

    np.testing.assert_allclose(ham.matrix(dets).toarray(), dense)
    np.testing.assert_allclose(ham.diags(dets), np.diag(dense))
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
    np.testing.assert_array_equal(known.bras, bras)
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
    assert _values(generated.bras, generated.hpsi) == expected


def test_conns(toy) -> None:
    ham, dets, dense, eps = toy.ham, toy.dets, toy.matrix, toy.eps
    kets = dets[:6]
    conns = ham.conns(kets, eps)
    cached = ham.conns(kets, eps)
    weight, count = ham.degrees(kets, eps)

    for name in ("dets", "ket_ptr", "bra_idx", "h", "weight"):
        np.testing.assert_array_equal(
            np.asarray(getattr(conns, name)),
            np.asarray(getattr(cached, name)),
        )

    pool = np.asarray(conns.dets)
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
    for name in ("dets", "ket_ptr", "bra_idx", "h", "count", "weight"):
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

    for name in ("rep_ptr", "bras", "hpsi_strong", "hpsi_a", "hpsi_b"):
        np.testing.assert_array_equal(
            np.asarray(getattr(sample, name)),
            np.asarray(getattr(repeated, name)),
        )

    exact = ham.project(sample.bras, kets, coeffs, eps=eps1)
    np.testing.assert_allclose(sample.hpsi_strong, exact.hpsi)
    np.testing.assert_allclose(sample.diags, exact.diags)
