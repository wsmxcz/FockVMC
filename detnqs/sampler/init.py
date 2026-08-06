from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..hilbert import Sector


def sample_slater(
    sector: Sector,
    ref_mat: ArrayLike,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample configurations from a real Slater determinant."""
    norb = int(sector.norb)
    n_alpha = int(sector.n_alpha)
    n_beta = int(sector.n_beta)
    ref_mat = np.asarray(ref_mat, dtype=np.float64, order="C")

    if ref_mat.shape != (n_alpha + n_beta, 2 * norb):
        raise ValueError("ref_mat shape must match sector")

    x = sector.zeros(n)
    rng = np.random.default_rng(seed)
    batch = np.arange(n, dtype=np.int64)
    alpha = ref_mat[:n_alpha, :norb].T
    beta = ref_mat[n_alpha:, norb:].T

    for spin, nelec, coeff in ((0, n_alpha, alpha), (1, n_beta, beta)):
        if nelec == 0:
            continue

        basis = np.linalg.qr(coeff, mode="reduced")[0]
        basis = np.broadcast_to(basis, (n, norb, nelec)).copy()
        occupied = np.empty((n, nelec), dtype=np.int64)

        for k in range(nelec):
            probability = np.sum(basis * basis, axis=-1)
            if k:
                probability[batch[:, None], occupied[:, :k]] = 0.0
            probability /= probability.sum(axis=1, keepdims=True)
            u = rng.random(n)
            occupied[:, k] = (
                np.cumsum(probability, axis=1) < u[:, None]
            ).sum(axis=1)

            if k + 1 == nelec:
                break

            row = basis[batch, occupied[:, k]].copy()
            pivot = np.argmax(np.abs(row), axis=1)
            value = row[batch, pivot]
            column = basis[batch, :, pivot].copy()
            basis -= (column / value[:, None])[:, :, None] * row[:, None, :]
            basis[batch, :, pivot] = basis[:, :, -1]
            basis = np.linalg.qr(basis[:, :, :-1], mode="reduced")[0]

        word = occupied >> 6
        bit = (occupied & 63).astype(np.uint64)
        np.bitwise_or.at(
            x,
            (batch[:, None], np.full_like(word, spin), word),
            np.uint64(1) << bit,
        )

    return np.ascontiguousarray(x)
