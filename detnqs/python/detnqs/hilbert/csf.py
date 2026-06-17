from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Space

_EMPTY = 0
_LOWER = 1
_UPPER = 2
_DOUBLE = 3


@dataclass(frozen=True, slots=True)
class CsfSpace(Space):
    """Spin-adapted CSF sector represented by Shavitt paths."""

    norb: int
    nelec: int
    spin: int

    def __post_init__(self) -> None:
        norb = int(self.norb)
        nelec = int(self.nelec)
        spin = int(self.spin)
        if norb < 0 or nelec < 0 or spin < 0:
            raise ValueError("norb, nelec, and spin must be nonnegative")
        if nelec > 2 * norb:
            raise ValueError("nelec must not exceed 2 * norb")
        if spin > nelec or (nelec - spin) % 2:
            raise ValueError("spin must be a valid 2S value")
        object.__setattr__(self, "norb", norb)
        object.__setattr__(self, "nelec", nelec)
        object.__setattr__(self, "spin", spin)

    @property
    def n_alpha(self) -> int:
        return (self.nelec + self.spin) // 2

    @property
    def n_beta(self) -> int:
        return (self.nelec - self.spin) // 2

    @property
    def nword(self) -> int:
        return max(1, (self.norb + 63) // 64)

    @property
    def shape(self) -> tuple[int, int]:
        return (2, self.nword)

    def reference(self, n: int = 1) -> np.ndarray:
        path = [_EMPTY] * self.norb
        for p in range(self.n_beta):
            path[p] = _DOUBLE
        for p in range(self.n_beta, self.n_beta + self.spin):
            path[p] = _UPPER

        x = self.zeros(1)
        self._write_path(x[0], path)
        return np.repeat(x, int(n), axis=0)

    def random(self, n: int, seed: int) -> np.ndarray:
        basis = self.enumerate()
        if basis.shape[0] == 0:
            raise ValueError("empty CSF sector")
        rng = np.random.default_rng(int(seed))
        pick = rng.integers(basis.shape[0], size=int(n), dtype=np.int64)
        return np.ascontiguousarray(basis[pick])

    def enumerate(self) -> np.ndarray:
        path = [_EMPTY] * self.norb
        basis = []

        def rec(p: int, nelec: int, spin: int) -> None:
            if p == self.norb:
                if nelec == self.nelec and spin == self.spin:
                    x = self.zeros(1)[0]
                    self._write_path(x, path)
                    basis.append(x)
                return

            for step, dn, ds in (
                (_EMPTY, 0, 0),
                (_LOWER, 1, -1),
                (_UPPER, 1, 1),
                (_DOUBLE, 2, 0),
            ):
                if nelec + dn > self.nelec or spin + ds < 0:
                    continue
                path[p] = step
                rec(p + 1, nelec + dn, spin + ds)

        rec(0, 0, 0)
        if not basis:
            return self.zeros(0)
        return np.ascontiguousarray(np.stack(basis, axis=0))

    def _write_path(self, x: np.ndarray, path: list[int]) -> None:
        x[...] = 0
        for p, step in enumerate(path):
            if step in (_UPPER, _DOUBLE):
                x[0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
            if step in (_LOWER, _DOUBLE):
                x[1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
