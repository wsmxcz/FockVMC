from __future__ import annotations

from typing import Any

import numpy as np
from pyscf import ao2mo


def slater_reference(
    sector: Any,
    source: Any | None = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Build a spin-orbital Slater reference.

    The result has shape ``(nelec, 2 * norb)``. ``source`` may be an
    ``(h1, eri)`` integral pair, a spatial-orbital coefficient matrix, or an
    already assembled spin-orbital reference.
    """
    norb = int(sector.norb)
    n_alpha = int(sector.n_alpha)
    n_beta = int(sector.n_beta)
    nelec = n_alpha + n_beta

    if source is None:
        coeff = np.eye(norb)
    elif isinstance(source, tuple):
        h1 = np.asarray(source[0], dtype=np.float64)
        eri = np.asarray(source[1], dtype=np.float64)
        eri = eri if eri.ndim == 4 else ao2mo.restore(1, eri, norb)

        rng = np.random.default_rng(seed)
        noise = rng.normal(size=(norb, norb))
        fock = 0.5 * (h1 + h1.T) + 1.0e-10 * (noise + noise.T)
        _, coeff = np.linalg.eigh(fock)
        density = np.zeros_like(h1)

        for _ in range(64):
            ca = coeff[:, :n_alpha]
            cb = coeff[:, :n_beta]
            target = ca @ ca.T + cb @ cb.T
            density = 0.65 * density + 0.35 * target
            coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
            exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
            fock = h1 + coulomb - 0.5 * exchange
            _, coeff = np.linalg.eigh(0.5 * (fock + fock.T))
    else:
        array = np.asarray(source, dtype=np.float64)
        if array.shape == (nelec, 2 * norb):
            return np.ascontiguousarray(array)
        coeff = np.linalg.qr(array, mode="reduced")[0]

    reference = np.zeros((nelec, 2 * norb), dtype=np.float64)
    reference[:n_alpha, :norb] = coeff[:, :n_alpha].T
    reference[n_alpha:, norb:] = coeff[:, :n_beta].T
    return np.ascontiguousarray(reference)
