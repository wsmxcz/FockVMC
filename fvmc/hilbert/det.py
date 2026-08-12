import itertools

import numpy as np

from .sector import Sector


class DetSector(Sector):
    """Fixed-Sz determinant sector."""

    __slots__ = ()

    def reference(self, n: int = 1) -> np.ndarray:
        x = self.zeros(1)

        for spin, n_elec in enumerate((self.n_alpha, self.n_beta)):
            if n_elec == 0:
                continue

            occ = np.arange(n_elec, dtype=np.int64)
            np.bitwise_or.at(
                x[0, spin],
                occ >> 6,
                np.uint64(1) << ((occ & 63).astype(np.uint64)),
            )

        return np.repeat(x, n, axis=0)

    def random(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = self.zeros(n)
        row = np.arange(n, dtype=np.int64)[:, None]

        for spin, n_elec in enumerate((self.n_alpha, self.n_beta)):
            if n_elec == 0:
                continue

            score = rng.random((n, self.norb))
            occ = np.argpartition(score, n_elec - 1, axis=1)[:, :n_elec]

            np.bitwise_or.at(
                x[:, spin],
                (row, occ >> 6),
                np.uint64(1) << ((occ & 63).astype(np.uint64)),
            )

        return x

    def enumerate(self) -> np.ndarray:
        spins = []
        for n_elec in (self.n_alpha, self.n_beta):
            occ = (
                np.empty((1, 0), dtype=np.int64)
                if n_elec == 0
                else np.array(
                    list(itertools.combinations(range(self.norb), n_elec)),
                    dtype=np.int64,
                ).reshape(-1, n_elec)
            )
            words = np.zeros((occ.shape[0], self.nword), dtype=np.uint64)
            if n_elec:
                row = np.arange(occ.shape[0], dtype=np.int64)[:, None]
                np.bitwise_or.at(
                    words,
                    (row, occ >> 6),
                    np.uint64(1) << ((occ & 63).astype(np.uint64)),
                )
            spins.append(words)

        alpha, beta = spins

        basis = np.empty(
            (alpha.shape[0] * beta.shape[0], 2, self.nword),
            dtype=np.uint64,
        )
        basis[:, 0] = np.repeat(alpha, beta.shape[0], axis=0)
        basis[:, 1] = np.tile(beta, (alpha.shape[0], 1))

        return basis
