from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Sector(ABC):
    """Fock-space sector with spin = n_alpha - n_beta."""

    norb: int
    nelec: int
    spin: int

    def __post_init__(self) -> None:
        if self.norb <= 0:
            raise ValueError("norb must be positive")
        if (self.nelec + self.spin) % 2:
            raise ValueError("nelec and spin must have the same parity")
        if not (
            0 <= self.n_alpha <= self.norb
            and 0 <= self.n_beta <= self.norb
        ):
            raise ValueError("particle numbers must fit in the orbital space")

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
        x = np.asarray(x, dtype=np.uint64, order="C")
        if x.ndim != 3 or x.shape[1:] != self.shape:
            raise ValueError(
                f"configurations must have shape (batch, 2, {self.nword})"
            )
        return x

    def unique(self, x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.asarray(x)
        if x.shape[0] == 0:
            index = np.empty(0, dtype=np.int64)
            return x, index, index

        flat = x.reshape(x.shape[0], -1)
        row = flat.view(
            np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))
        ).reshape(-1)
        _, first, inverse = np.unique(
            row,
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
