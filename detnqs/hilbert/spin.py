from __future__ import annotations

import numpy as np

from .sector import Sector


class SpinSector(Sector):
    """Fixed-spin sector."""

    __slots__ = ()

    def reference(self, n: int = 1) -> np.ndarray:
        x = self.zeros(1)

        if self.n_beta:
            occ = np.arange(self.n_beta, dtype=np.int64)
            bit = np.uint64(1) << ((occ & 63).astype(np.uint64))
            np.bitwise_or.at(x[0, 0], occ >> 6, bit)
            np.bitwise_or.at(x[0, 1], occ >> 6, bit)

        if self.spin:
            occ = np.arange(self.n_beta, self.n_beta + self.spin, dtype=np.int64)
            np.bitwise_or.at(
                x[0, 0],
                occ >> 6,
                np.uint64(1) << ((occ & 63).astype(np.uint64)),
            )

        return np.repeat(x, n, axis=0)

    def random(self, n: int, seed: int) -> np.ndarray:
        basis = self.enumerate()
        rng = np.random.default_rng(seed)
        return basis[rng.integers(basis.shape[0], size=n)]

    def enumerate(self) -> np.ndarray:
        paths = np.zeros((1, self.norb), dtype=np.uint8)
        nelec = np.zeros(1, dtype=np.int64)
        spin = np.zeros(1, dtype=np.int64)

        step = np.array([0, 1, 2, 3], dtype=np.uint8)
        dne = np.array([0, 1, 1, 2], dtype=np.int64)
        dsp = np.array([0, -1, 1, 0], dtype=np.int64)

        for p in range(self.norb):
            n_path = paths.shape[0]

            cand = np.repeat(paths, 4, axis=0)
            cand[:, p] = np.tile(step, n_path)

            ne = np.repeat(nelec, 4) + np.tile(dne, n_path)
            sp = np.repeat(spin, 4) + np.tile(dsp, n_path)

            rem = self.norb - p - 1
            keep = (ne <= self.nelec) & (ne + 2 * rem >= self.nelec) & (sp >= 0)

            paths = cand[keep]
            nelec = ne[keep]
            spin = sp[keep]

        paths = paths[(nelec == self.nelec) & (spin == self.spin)]
        basis = self.zeros(paths.shape[0])

        if paths.shape[0] == 0 or self.norb == 0:
            return basis

        occ = np.arange(self.norb, dtype=np.int64)
        word = occ >> 6
        bit = np.uint64(1) << ((occ & 63).astype(np.uint64))

        alpha = (paths == 2) | (paths == 3)
        beta = (paths == 1) | (paths == 3)

        for w in range(self.nword):
            col = word == w
            basis[:, 0, w] = alpha[:, col].astype(np.uint64) @ bit[col]
            basis[:, 1, w] = beta[:, col].astype(np.uint64) @ bit[col]

        return basis