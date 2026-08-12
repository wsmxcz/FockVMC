from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def annihilate(
    ket: Any,
    spin: int,
    p: int,
) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.bool_]]:
    """Apply c_{spin,p} to each ket."""
    ket = np.ascontiguousarray(ket, dtype=np.uint64)
    word = p >> 6
    bit = np.uint64(1) << np.uint64(p & 63)

    active = (ket[:, spin, word] & bit) != 0
    bra = ket.copy()
    bra[active, spin, word] &= ~bit

    before = np.zeros(ket.shape[0], dtype=np.int64)
    if spin == 1:
        before += np.bitwise_count(ket[:, 0]).sum(axis=1, dtype=np.int64)
    if word:
        before += np.bitwise_count(ket[:, spin, :word]).sum(
            axis=1,
            dtype=np.int64,
        )
    shift = p & 63
    if shift:
        mask = (np.uint64(1) << np.uint64(shift)) - np.uint64(1)
        before += np.bitwise_count(ket[:, spin, word] & mask)
    sign = np.where(before & 1, -1.0, 1.0)
    return bra, sign, active


def create(
    ket: Any,
    spin: int,
    p: int,
) -> tuple[NDArray[np.uint64], NDArray[np.float64], NDArray[np.bool_]]:
    """Apply c^dagger_{spin,p} to each ket."""
    ket = np.ascontiguousarray(ket, dtype=np.uint64)
    word = p >> 6
    bit = np.uint64(1) << np.uint64(p & 63)

    active = (ket[:, spin, word] & bit) == 0
    bra = ket.copy()
    bra[active, spin, word] |= bit

    before = np.zeros(ket.shape[0], dtype=np.int64)
    if spin == 1:
        before += np.bitwise_count(ket[:, 0]).sum(axis=1, dtype=np.int64)
    if word:
        before += np.bitwise_count(ket[:, spin, :word]).sum(
            axis=1,
            dtype=np.int64,
        )
    shift = p & 63
    if shift:
        mask = (np.uint64(1) << np.uint64(shift)) - np.uint64(1)
        before += np.bitwise_count(ket[:, spin, word] & mask)
    sign = np.where(before & 1, -1.0, 1.0)
    return bra, sign, active


def number(
    ket: Any,
    spin: int | None = None,
    p: int | None = None,
) -> NDArray[np.float64]:
    """Return particle number on an orbital, spin block, or full ket."""
    ket = np.asarray(ket, dtype=np.uint64)

    if p is not None:
        word = p >> 6
        bit = np.uint64(1) << np.uint64(p & 63)
        if spin is not None:
            return ((ket[:, spin, word] & bit) != 0).astype(np.float64)
        return (
            ((ket[:, 0, word] & bit) != 0).astype(np.float64)
            + ((ket[:, 1, word] & bit) != 0).astype(np.float64)
        )

    if spin is not None:
        return np.bitwise_count(ket[:, spin]).sum(
            axis=1,
            dtype=np.int64,
        ).astype(np.float64)

    return np.bitwise_count(ket).sum(
        axis=(1, 2),
        dtype=np.int64,
    ).astype(np.float64)
