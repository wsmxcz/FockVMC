from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .sector import Sector


@dataclass(frozen=True, slots=True)
class DetSector(Sector):
    """Fixed alpha/beta determinant sector."""

    norb: int
    n_alpha: int
    n_beta: int

    def __post_init__(self) -> None:
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)
        if norb < 0:
            raise ValueError("norb must be nonnegative")
        if not (0 <= n_alpha <= norb and 0 <= n_beta <= norb):
            raise ValueError("electron counts must satisfy 0 <= n <= norb")
        object.__setattr__(self, "norb", norb)
        object.__setattr__(self, "n_alpha", n_alpha)
        object.__setattr__(self, "n_beta", n_beta)

    @property
    def nelec(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def nword(self) -> int:
        return max(1, (self.norb + 63) // 64)

    @property
    def shape(self) -> tuple[int, int]:
        return (2, self.nword)

    def reference(self, n: int = 1) -> np.ndarray:
        x = self.zeros(1)
        for p in range(self.n_alpha):
            x[0, 0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
        for p in range(self.n_beta):
            x[0, 1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
        return np.repeat(x, int(n), axis=0)

    def random(self, n: int, seed: int) -> np.ndarray:
        n = int(n)
        rng = np.random.default_rng(int(seed))
        x = self.zeros(n)
        item = np.arange(n, dtype=np.int64)[:, None]

        for spin, n_elec in enumerate((self.n_alpha, self.n_beta)):
            if n_elec == 0:
                continue
            if n_elec == self.norb:
                occ = np.broadcast_to(
                    np.arange(self.norb, dtype=np.int64),
                    (n, self.norb),
                )
            else:
                score = rng.random((n, self.norb))
                occ = np.argpartition(score, n_elec - 1, axis=1)[:, :n_elec]

            word = occ >> 6
            bit = (occ & 63).astype(np.uint64)
            np.bitwise_or.at(x[:, spin, :], (item, word), np.uint64(1) << bit)

        return np.ascontiguousarray(x)

    def enumerate(self) -> np.ndarray:
        basis = []
        for occ_a in itertools.combinations(range(self.norb), self.n_alpha):
            alpha = np.zeros(self.nword, dtype=np.uint64)
            for p in occ_a:
                alpha[p >> 6] |= np.uint64(1) << np.uint64(p & 63)

            for occ_b in itertools.combinations(range(self.norb), self.n_beta):
                beta = np.zeros(self.nword, dtype=np.uint64)
                for p in occ_b:
                    beta[p >> 6] |= np.uint64(1) << np.uint64(p & 63)
                basis.append(np.stack((alpha, beta), axis=0))

        return np.ascontiguousarray(np.stack(basis, axis=0))
