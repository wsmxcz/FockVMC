from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hilbert import DetSector, SpinSector


def occupation(x: Any, spin: int, p: int) -> np.ndarray:
    """Return n_{spin,p}(x)."""
    spin = int(spin)
    p = int(p)
    word = p >> 6
    bit = np.uint64(1) << np.uint64(p & 63)
    return ((x[:, spin, word] & bit) != 0).astype(np.float64)


def annihilate(x: Any, spin: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply c_{spin,p} to each ket."""
    ket = x
    spin = int(spin)
    p = int(p)
    word = p >> 6
    bit = np.uint64(1) << np.uint64(p & 63)

    before = np.zeros(ket.shape[0], dtype=np.int64)
    if spin == 1:
        before += _count(ket[:, 0, :])
    if word:
        before += _count(ket[:, spin, :word])
    mask = (
        (np.uint64(1) << np.uint64(p & 63)) - np.uint64(1)
        if p & 63
        else np.uint64(0)
    )
    if mask:
        before += np.bitwise_count(ket[:, spin, word] & mask).astype(np.int64)

    active = (ket[:, spin, word] & bit) != 0
    bra = np.ascontiguousarray(ket.copy())
    bra[active, spin, word] &= ~bit
    return bra, np.where(before & 1, -1.0, 1.0), active


def create(x: Any, spin: int, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply creation operator c^dagger_{spin,p} to each ket."""
    ket = x
    spin = int(spin)
    p = int(p)
    word = p >> 6
    bit = np.uint64(1) << np.uint64(p & 63)

    before = np.zeros(ket.shape[0], dtype=np.int64)
    if spin == 1:
        before += _count(ket[:, 0, :])
    if word:
        before += _count(ket[:, spin, :word])
    mask = (
        (np.uint64(1) << np.uint64(p & 63)) - np.uint64(1)
        if p & 63
        else np.uint64(0)
    )
    if mask:
        before += np.bitwise_count(ket[:, spin, word] & mask).astype(np.int64)

    active = (ket[:, spin, word] & bit) == 0
    bra = np.ascontiguousarray(ket.copy())
    bra[active, spin, word] |= bit
    return bra, np.where(before & 1, -1.0, 1.0), active


def number(x: Any, spin: int | None = None, p: int | None = None) -> np.ndarray:
    """Return particle number on a spin-orbital, spin block, or full ket."""
    if p is not None:
        if spin is None:
            return occupation(x, 0, int(p)) + occupation(x, 1, int(p))
        return occupation(x, int(spin), int(p))

    if spin is not None:
        return _count(x[:, int(spin), :]).astype(np.float64)

    return (
        _count(x[:, 0, :]).astype(np.float64)
        + _count(x[:, 1, :]).astype(np.float64)
    )


def sz(x: Any) -> np.ndarray:
    """Return S_z(x)."""
    return 0.5 * (
        _count(x[:, 0, :]).astype(np.float64)
        - _count(x[:, 1, :]).astype(np.float64)
    )


def _count(words: np.ndarray) -> np.ndarray:
    return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Number:
    """Particle-number operator."""

    sector: Any

    def diag(self, x: Any) -> np.ndarray:
        return number(x)


@dataclass(frozen=True, slots=True)
class Sz:
    """Spin-z operator."""

    sector: Any

    def diag(self, x: Any) -> np.ndarray:
        return sz(x)


@dataclass(frozen=True, slots=True)
class S2:
    """Total spin-squared operator."""

    sector: Any

    def diag(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=np.uint64)

        if isinstance(self.sector, SpinSector):
            s = 0.5 * float(self.sector.spin)
            return np.full(x.shape[0], s * (s + 1.0), dtype=np.float64)

        if isinstance(self.sector, DetSector):
            n_alpha = _count(x[:, 0])
            n_beta = _count(x[:, 1])
            n_double = _count(x[:, 0] & x[:, 1])
            n_single = n_alpha + n_beta - 2 * n_double
            m = 0.5 * (n_alpha - n_beta)
            return m * m + 0.5 * n_single.astype(np.float64)

        raise TypeError(f"unsupported sector: {type(self.sector).__name__}")

    def local_conn(self, x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return diagonal, CSR pointer, spin-exchange bras, and values."""
        x = np.ascontiguousarray(x, dtype=np.uint64)
        diag = self.diag(x)
        n = int(x.shape[0])

        if isinstance(self.sector, SpinSector):
            return (
                diag,
                np.zeros(n + 1, dtype=np.int64),
                x[:0].copy(),
                np.empty(0, dtype=np.float64),
            )

        if not isinstance(self.sector, DetSector):
            raise TypeError(f"unsupported sector: {type(self.sector).__name__}")

        norb = int(self.sector.norb)
        orb = np.arange(norb, dtype=np.int64)
        word = orb >> 6
        bit = np.left_shift(np.uint64(1), (orb & 63).astype(np.uint64))

        alpha = (x[:, 0, word] & bit) != 0
        beta = (x[:, 1, word] & bit) != 0

        aonly = alpha & ~beta
        bonly = beta & ~alpha
        single = alpha ^ beta

        count = aonly.sum(axis=1, dtype=np.int64) * bonly.sum(
            axis=1,
            dtype=np.int64,
        )

        ptr = np.empty(n + 1, dtype=np.int64)
        ptr[0] = 0
        np.cumsum(count, out=ptr[1:])

        bra = np.empty((int(ptr[-1]),) + x.shape[1:], dtype=np.uint64)
        val = np.empty(int(ptr[-1]), dtype=np.float64)

        prefix = np.empty((n, norb + 1), dtype=np.int64)
        prefix[:, 0] = 0
        np.cumsum(single, axis=1, dtype=np.int64, out=prefix[:, 1:])

        for i in range(n):
            lo = int(ptr[i])
            hi = int(ptr[i + 1])
            if lo == hi:
                continue

            pa = np.flatnonzero(aonly[i])
            qb = np.flatnonzero(bonly[i])

            p = np.repeat(pa, qb.size)
            q = np.tile(qb, pa.size)
            r = np.arange(p.size)

            y = np.repeat(x[i : i + 1], p.size, axis=0)

            wp, bp = word[p], bit[p]
            wq, bq = word[q], bit[q]

            y[r, 0, wp] &= ~bp
            y[r, 1, wp] |= bp
            y[r, 1, wq] &= ~bq
            y[r, 0, wq] |= bq

            left = np.minimum(p, q)
            right = np.maximum(p, q)
            parity = (prefix[i, right] - prefix[i, left + 1]) & 1

            bra[lo:hi] = y
            val[lo:hi] = 2.0 * parity.astype(np.float64) - 1.0

        return diag, ptr, bra, val