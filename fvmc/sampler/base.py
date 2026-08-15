from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import comb

import jax
import numpy as np
from numpy.typing import ArrayLike

from ..hilbert import Sector


@dataclass(frozen=True, slots=True)
class ChainState:
    """Dynamic state of Markov chains."""

    key: jax.Array
    x: np.ndarray
    logabs: np.ndarray


@cache
def rank_table(
    norb: int,
    n_alpha: int,
    n_beta: int,
    rank: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build rank and spin-split distributions."""
    max_alpha = min(n_alpha, norb - n_alpha)
    max_beta = min(n_beta, norb - n_beta)
    max_rank = max_alpha + max_beta
    stop = max_rank if rank is None else min(rank, max_rank)
    ranks = np.arange(1, stop + 1, dtype=np.int32)

    if not stop:
        return ranks, np.empty(0), (), ()

    weight = np.exp2(-ranks.astype(np.float64))
    rank_prob = weight / weight.sum()
    splits = []
    split_prob = []

    for total in ranks:
        low = max(0, int(total) - max_beta)
        high = min(max_alpha, int(total))
        alpha = np.arange(low, high + 1, dtype=np.int32)
        count = np.array(
            [
                comb(n_alpha, int(a))
                * comb(norb - n_alpha, int(a))
                * comb(n_beta, int(total - a))
                * comb(norb - n_beta, int(total - a))
                for a in alpha
            ],
            dtype=np.float64,
        )
        splits.append(alpha)
        split_prob.append(count / count.sum())

    return ranks, rank_prob, tuple(splits), tuple(split_prob)


def sample_orbitals(
    words: np.ndarray,
    count: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample occupied orbital indices without replacement."""
    n = words.shape[0]
    width = int(count.max(initial=0))
    rank = np.full((n, width), -1, dtype=np.int32)

    for i in range(width):
        active = i < count
        last = size - count + i
        draw = (rng.random(n) * (last + 1)).astype(np.int32)
        if i:
            duplicate = (rank[:, :i] == draw[:, None]).any(axis=1)
            draw = np.where(duplicate, last, draw)
        rank[active, i] = draw[active]

    if not width:
        return rank

    prefix = np.cumulative_sum(
        np.bitwise_count(words),
        axis=1,
        dtype=np.int32,
        include_initial=True,
    )
    valid = rank >= 0
    rank = np.where(valid, rank, 0)
    word = np.sum(
        prefix[:, None, 1:] <= rank[:, :, None],
        axis=2,
        dtype=np.int32,
    )
    before = np.take_along_axis(prefix, word, axis=1)
    value = np.take_along_axis(words, word, axis=1)
    bits = np.unpackbits(
        np.ascontiguousarray(value).view(np.uint8).reshape(n, width, 8),
        axis=2,
        bitorder="little",
    )
    inside = rank - before
    bit = np.argmax(
        np.cumsum(bits, axis=2, dtype=np.int16) > inside[:, :, None],
        axis=2,
    )
    orbital = (word << 6) + bit
    return np.where(valid, orbital, -1)


def sample_slater(
    sector: Sector,
    ref_mat: ArrayLike,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample configurations from a real Slater determinant."""
    norb = sector.norb
    n_alpha = sector.n_alpha
    n_beta = sector.n_beta
    ref_mat = np.asarray(ref_mat, dtype=np.float64, order="C")

    if ref_mat.shape != (n_alpha + n_beta, 2 * norb):
        raise ValueError("ref_mat shape must match sector")

    x = sector.zeros(n)
    rng = np.random.default_rng(seed)
    batch = np.arange(n, dtype=np.int64)
    alpha = ref_mat[:n_alpha, :norb].T
    beta = ref_mat[n_alpha:, norb:].T

    for spin, nelec, coeff in ((0, n_alpha, alpha), (1, n_beta, beta)):
        if not nelec:
            continue

        basis = np.linalg.qr(coeff, mode="reduced")[0]
        basis = np.broadcast_to(basis, (n, norb, nelec)).copy()
        occupied = np.empty((n, nelec), dtype=np.int64)

        for k in range(nelec):
            probability = np.sum(basis * basis, axis=-1)
            if k:
                probability[batch[:, None], occupied[:, :k]] = 0.0
            probability /= probability.sum(axis=1, keepdims=True)
            u = rng.random(n)
            occupied[:, k] = (
                np.cumsum(probability, axis=1) < u[:, None]
            ).sum(axis=1)

            if k + 1 == nelec:
                break

            row = basis[batch, occupied[:, k]].copy()
            pivot = np.argmax(np.abs(row), axis=1)
            value = row[batch, pivot]
            column = basis[batch, :, pivot].copy()
            basis -= (column / value[:, None])[:, :, None] * row[:, None, :]
            basis[batch, :, pivot] = basis[:, :, -1]
            basis = np.linalg.qr(basis[:, :, :-1], mode="reduced")[0]

        word = occupied >> 6
        bit = (occupied & 63).astype(np.uint64)
        np.bitwise_or.at(
            x,
            (batch[:, None], np.full_like(word, spin), word),
            np.uint64(1) << bit,
        )

    return x
