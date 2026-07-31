from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Sector(ABC):
    """Fock-space sector using the PySCF convention.

    spin = n_alpha - n_beta.
    """

    norb: int
    nelec: int
    spin: int

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

    def zeros(self, n: int = 1) -> np.ndarray:
        return np.zeros((n, 2, self.nword), dtype=np.uint64)

    def asarray(self, x: Any) -> np.ndarray:
        return np.asarray(x, dtype=np.uint64, order="C")

    def unique(self, x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.asarray(x)
        if x.shape[0] == 0:
            index = np.empty(0, dtype=np.int64)
            return x, index, index

        _, first, inverse = np.unique(
            x.reshape(x.shape[0], -1),
            axis=0,
            return_index=True,
            return_inverse=True,
        )

        order = np.argsort(first, kind="stable")
        first = first[order].astype(np.int64, copy=False)

        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)

        return x[first], first, remap[inverse]

    @abstractmethod
    def reference(self, n: int = 1) -> np.ndarray:
        """Return a simple reference configuration."""

    @abstractmethod
    def random(self, n: int, seed: int) -> np.ndarray:
        """Return random valid configurations."""

    @abstractmethod
    def enumerate(self) -> np.ndarray:
        """Return the full finite sector."""
