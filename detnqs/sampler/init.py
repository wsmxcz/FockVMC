from __future__ import annotations

from typing import Any

import numpy as np

from ..model import slater_reference


def init_chains(
    sector: Any,
    reference: Any | None = None,
    *,
    n_chains: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """Sample initial configurations from a Slater reference."""
    norb = int(sector.norb)
    n_alpha = int(sector.n_alpha)
    n_beta = int(sector.n_beta)
    ref = slater_reference(sector, reference, seed=seed)
    x = sector.zeros(n_chains)

    rng = np.random.default_rng(seed)
    batch = np.arange(n_chains, dtype=np.int64)
    alpha = ref[:n_alpha, :norb].T
    beta = ref[n_alpha:, norb:].T

    for spin, nelec, coeff in ((0, n_alpha, alpha), (1, n_beta, beta)):
        if nelec == 0:
            continue

        basis = np.linalg.qr(coeff[:, :nelec], mode="reduced")[0]
        basis = np.broadcast_to(basis, (n_chains, norb, nelec)).copy()
        occupied = np.empty((n_chains, nelec), dtype=np.int64)

        for k in range(nelec):
            probability = np.sum(basis * basis, axis=-1)
            if k:
                probability[batch[:, None], occupied[:, :k]] = 0.0
            probability /= probability.sum(axis=1, keepdims=True)
            u = rng.random(n_chains)
            occupied[:, k] = (
                np.cumsum(probability, axis=1) < u[:, None]
            ).sum(axis=1)

            if k + 1 == nelec:
                break

            row = basis[batch, occupied[:, k]].copy()
            pivot = np.argmax(np.abs(row), axis=1)
            pivot_value = row[batch, pivot]
            column = basis[batch, :, pivot].copy()
            basis -= (column / pivot_value[:, None])[:, :, None] * row[:, None, :]
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
