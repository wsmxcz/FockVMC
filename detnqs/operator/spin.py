from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from numpy.typing import NDArray

from ..hilbert import DetSector
from ..model.base import to_ratio
from ..utils import batch, tree


@dataclass(frozen=True, slots=True)
class S2:
    """Total spin-squared operator."""

    sector: DetSector

    def diag(self, ket: Any) -> NDArray[np.float64]:
        ket = np.asarray(ket, dtype=np.uint64)
        n_double = np.bitwise_count(ket[:, 0] & ket[:, 1]).sum(
            axis=1,
            dtype=np.int64,
        )
        n_single = self.sector.nelec - 2 * n_double
        m = 0.5 * self.sector.spin
        return m * m + 0.5 * n_single.astype(np.float64)

    def local_conn(
        self,
        ket: Any,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.int64],
        NDArray[np.uint64],
        NDArray[np.float64],
    ]:
        ket = np.ascontiguousarray(ket, dtype=np.uint64)
        nket = ket.shape[0]
        norb = self.sector.norb
        orbital = np.arange(norb, dtype=np.int64)
        word = orbital >> 6
        bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))

        alpha = (ket[:, 0, word] & bit) != 0
        beta = (ket[:, 1, word] & bit) != 0
        aonly = alpha & ~beta
        bonly = beta & ~alpha
        single = alpha ^ beta
        diag = (
            0.25 * self.sector.spin**2
            + 0.5 * single.sum(axis=1, dtype=np.int64)
        )
        count = (
            aonly.sum(axis=1, dtype=np.int64)
            * bonly.sum(axis=1, dtype=np.int64)
        )
        ptr = np.empty(nket + 1, dtype=np.int64)
        ptr[0] = 0
        np.cumsum(count, out=ptr[1:])
        prefix = np.pad(np.cumsum(single, axis=1), ((0, 0), (1, 0)))

        source, p, q = np.nonzero(
            aonly[:, :, None] & bonly[:, None, :]
        )
        bra = ket[source].copy()
        item = np.arange(source.size)
        wp = word[p]
        wq = word[q]
        bp = bit[p]
        bq = bit[q]
        bra[item, 0, wp] &= ~bp
        bra[item, 1, wp] |= bp
        bra[item, 1, wq] &= ~bq
        bra[item, 0, wq] |= bq

        left = np.minimum(p, q)
        right = np.maximum(p, q)
        parity = (prefix[source, right] - prefix[source, left + 1]) & 1
        value = 2.0 * parity.astype(np.float64) - 1.0

        return diag, ptr, bra, value


def spin_correlation(state: Any, x: Any, weight: Any) -> np.ndarray:
    """Return the orbital spin-correlation matrix."""
    sector = state.hamiltonian.sector
    norb = sector.norb
    x = sector.asarray(x)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight /= np.sum(weight)
    out = np.zeros((norb, norb), dtype=np.complex128)

    ket_logpsi = batch.apply(state.model.logpsi, state.params, x)
    jax.block_until_ready(ket_logpsi)
    ket_logpsi = tree.host(ket_logpsi)

    orbital = np.arange(norb, dtype=np.int64)
    word = orbital >> 6
    bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))

    for slc in batch.slices(x.shape[0], sector.n_alpha * sector.n_beta):
        ket = x[slc]
        mass = weight[slc]
        alpha = (ket[:, 0, word] & bit) != 0
        beta = (ket[:, 1, word] & bit) != 0
        single = alpha ^ beta
        sz = 0.5 * (alpha.astype(np.float64) - beta.astype(np.float64))
        out += np.einsum("n,np,nq->pq", mass, sz, sz, optimize=True)
        out[orbital, orbital] += 0.5 * np.einsum(
            "n,np->p",
            mass,
            single,
            optimize=True,
        )

        source, p, q = np.nonzero(
            (alpha & ~beta)[:, :, None]
            & (beta & ~alpha)[:, None, :]
        )
        if source.size == 0:
            continue

        bra = ket[source].copy()
        item = np.arange(source.size)
        wp = p >> 6
        wq = q >> 6
        bp = np.left_shift(np.uint64(1), (p & 63).astype(np.uint64))
        bq = np.left_shift(np.uint64(1), (q & 63).astype(np.uint64))
        bra[item, 0, wp] &= ~bp
        bra[item, 1, wp] |= bp
        bra[item, 1, wq] &= ~bq
        bra[item, 0, wq] |= bq

        prefix = np.pad(np.cumsum(single, axis=1), ((0, 0), (1, 0)))
        left = np.minimum(p, q)
        right = np.maximum(p, q)
        parity = (prefix[source, right] - prefix[source, left + 1]) & 1
        value = parity.astype(np.float64) - 0.5

        raw, _, inverse = sector.unique(bra)
        bra_logpsi = batch.bucket(
            state.model.logpsi,
            state.params,
            raw,
            in_axes=(None, 0),
        )
        jax.block_until_ready(bra_logpsi)
        bra_logpsi = tree.host(bra_logpsi)
        ratio = np.asarray(
            to_ratio(
                jax.tree.map(lambda a: a[inverse], bra_logpsi),
                jax.tree.map(lambda a: a[slc][source], ket_logpsi),
            )
        ).reshape(-1)
        transverse = mass[source] * value * ratio
        np.add.at(out, (p, q), transverse)
        np.add.at(out, (q, p), transverse)

    return np.real_if_close(out)
