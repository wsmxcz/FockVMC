from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pytest

import libdet


@dataclass(frozen=True)
class ToySystem:
    ham: libdet.Hamiltonian
    dets: np.ndarray
    matrix: np.ndarray
    eps: float


def _pair_index(p: int, q: int) -> int:
    hi, lo = max(p, q), min(p, q)
    return hi * (hi + 1) // 2 + lo


def _hamiltonian(norb: int = 4) -> libdet.Hamiltonian:
    rng = np.random.default_rng(7)
    h1 = rng.normal(size=(norb, norb))
    h1 = 0.5 * (h1 + h1.T)

    npair = norb * (norb + 1) // 2
    pair_eri = rng.normal(scale=0.2, size=(npair, npair))
    pair_eri = 0.5 * (pair_eri + pair_eri.T)
    eri = np.empty((norb, norb, norb, norb))

    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    eri[p, q, r, s] = pair_eri[
                        _pair_index(p, q),
                        _pair_index(r, s),
                    ]

    return libdet.Hamiltonian.rhf(h1, eri, ecore=0.3)


def _determinants(
    norb: int = 4,
    n_alpha: int = 2,
    n_beta: int = 1,
) -> np.ndarray:
    dets = []
    for alpha in combinations(range(norb), n_alpha):
        for beta in combinations(range(norb), n_beta):
            det = np.zeros((2, 1), dtype=np.uint64)
            for p in alpha:
                det[0, 0] |= np.uint64(1) << np.uint64(p)
            for p in beta:
                det[1, 0] |= np.uint64(1) << np.uint64(p)
            dets.append(det)
    return np.ascontiguousarray(dets)


@pytest.fixture(scope="session")
def toy() -> ToySystem:
    ham = _hamiltonian()
    dets = _determinants()
    matrix = np.array(
        [
            [ham.hij(bra[None], ket[None]) for ket in dets]
            for bra in dets
        ]
    )
    offdiag = np.abs(matrix[~np.eye(len(dets), dtype=bool)])
    eps = float(np.quantile(offdiag[offdiag > 0.0], 0.55))
    return ToySystem(ham, dets, matrix, eps)
