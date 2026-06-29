from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Sector(ABC):
    """Discrete Fock-space sector."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of one configuration x."""

    def zeros(self, n: int = 1) -> np.ndarray:
        return np.zeros((int(n), *self.shape), dtype=np.uint64)

    def asarray(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint64)
        return np.ascontiguousarray(x)

    def unique(self, x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.asarray(x)
        n = int(x.shape[0])
        if n == 0:
            empty = np.empty(0, dtype=np.int64)
            return x, empty, empty

        flat = x.reshape(n, -1)
        packed = flat.view(
            np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
        ).ravel()
        _, first, inverse = np.unique(
            packed,
            return_index=True,
            return_inverse=True,
        )

        order = np.argsort(first, kind="stable")
        first = first[order].astype(np.int64, copy=False)
        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)
        inverse = remap[inverse]

        return np.ascontiguousarray(x[first]), first, inverse

    @abstractmethod
    def reference(self, n: int = 1) -> np.ndarray:
        """Return a simple reference configuration."""

    @abstractmethod
    def random(self, n: int, seed: int) -> np.ndarray:
        """Return random valid configurations."""

    @abstractmethod
    def enumerate(self) -> np.ndarray:
        """Return the full finite sector."""
