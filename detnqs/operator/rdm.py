from __future__ import annotations

from typing import Any

import jax
import numpy as np

from ..model.base import to_ratio
from ..utils import batch, tree


def rdm1(state: Any, x: Any, weight: Any) -> np.ndarray:
    """Return gamma[spin,p,q] = <a^dagger[p,spin] a[q,spin]>."""
    sector = state.hamiltonian.sector
    norb = sector.norb
    x = sector.asarray(x)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight /= np.sum(weight)
    out = np.zeros((2, norb, norb), dtype=np.complex128)

    ket_logpsi = batch.apply(state.model.logpsi, state.params, x)
    jax.block_until_ready(ket_logpsi)
    ket_logpsi = tree.host(ket_logpsi)

    orbital = np.arange(norb, dtype=np.int64)
    word = orbital >> 6
    bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))

    for spin, nelec in enumerate((sector.n_alpha, sector.n_beta)):
        degree = nelec * (norb - nelec)

        for slc in batch.chunk(x.shape[0], expansion=degree):
            ket = x[slc]
            mass = weight[slc]
            occ = (ket[:, spin, word] & bit) != 0
            source, p = np.nonzero(occ)
            np.add.at(out, (spin, p, p), mass[source])

            source, p, q = np.nonzero(
                occ[:, :, None] & ~occ[:, None, :]
            )
            if source.size == 0:
                continue

            bra = ket[source].copy()
            item = np.arange(source.size)
            pword = p >> 6
            qword = q >> 6
            pbit = np.left_shift(np.uint64(1), (p & 63).astype(np.uint64))
            qbit = np.left_shift(np.uint64(1), (q & 63).astype(np.uint64))
            bra[item, spin, pword] &= ~pbit
            bra[item, spin, qword] |= qbit

            prefix = np.pad(np.cumsum(occ, axis=1), ((0, 0), (1, 0)))
            left = np.minimum(p, q)
            right = np.maximum(p, q)
            parity = (prefix[source, right] - prefix[source, left + 1]) & 1

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
            value = 1.0 - 2.0 * parity
            np.add.at(out, (spin, p, q), mass[source] * value * ratio)

    return np.real_if_close(out)


def rdm2(state: Any, x: Any, weight: Any) -> np.ndarray:
    """Return the spin-resolved two-particle reduced density matrix.

    Gamma[spin,tau,p,q,r,s]
        = <a^dagger[p,spin] a^dagger[q,tau] a[s,tau] a[r,spin]>.
    """
    sector = state.hamiltonian.sector
    norb = sector.norb
    x = sector.asarray(x)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight /= np.sum(weight)
    out = np.zeros((2, 2, norb, norb, norb, norb), dtype=np.complex128)

    ket_logpsi = batch.apply(state.model.logpsi, state.params, x)
    jax.block_until_ready(ket_logpsi)
    ket_logpsi = tree.host(ket_logpsi)

    orbital = np.arange(norb, dtype=np.int64)
    word = orbital >> 6
    bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))

    for iket, ket in enumerate(x):
        occ = (ket[:, word] & bit) != 0
        occupied = tuple(np.flatnonzero(occ[spin]) for spin in range(2))
        vacant = tuple(np.flatnonzero(~occ[spin]) for spin in range(2))

        for spin in range(2):
            for tau in range(2):
                if spin == tau:
                    noccupied = occupied[spin].size
                    npair = noccupied * (noccupied - 1)
                    if npair == 0:
                        continue

                    pair = np.arange(npair, dtype=np.int64)
                    ip, iq = np.divmod(pair, noccupied - 1)
                    iq += iq >= ip
                    p = occupied[spin][ip]
                    q = occupied[tau][iq]

                    pool = np.empty(
                        (npair, vacant[spin].size + 2),
                        dtype=np.int64,
                    )
                    pool[:, : vacant[spin].size] = vacant[spin]
                    pool[:, -2] = p
                    pool[:, -1] = q
                    ncreate = pool.shape[1] * (pool.shape[1] - 1)
                else:
                    npair = occupied[spin].size * occupied[tau].size
                    if npair == 0:
                        continue

                    ip, iq = np.divmod(
                        np.arange(npair, dtype=np.int64),
                        occupied[tau].size,
                    )
                    p = occupied[spin][ip]
                    q = occupied[tau][iq]
                    rpool = np.empty(
                        (npair, vacant[spin].size + 1),
                        dtype=np.int64,
                    )
                    spool = np.empty(
                        (npair, vacant[tau].size + 1),
                        dtype=np.int64,
                    )
                    rpool[:, : vacant[spin].size] = vacant[spin]
                    spool[:, : vacant[tau].size] = vacant[tau]
                    rpool[:, -1] = p
                    spool[:, -1] = q
                    ncreate = rpool.shape[1] * spool.shape[1]

                for slc in batch.chunk(npair * ncreate):
                    index = np.arange(slc.start, slc.stop, dtype=np.int64)
                    pair, creation = np.divmod(index, ncreate)
                    if spin == tau:
                        irank, isrank = np.divmod(
                            creation,
                            pool.shape[1] - 1,
                        )
                        isrank += isrank >= irank
                        r = pool[pair, irank]
                        s = pool[pair, isrank]
                    else:
                        irank, isrank = np.divmod(creation, spool.shape[1])
                        r = rpool[pair, irank]
                        s = spool[pair, isrank]

                    ps = p[pair]
                    qs = q[pair]
                    source_p = spin * norb + ps
                    source_q = tau * norb + qs
                    ir = spin * norb + r
                    is_ = tau * norb + s

                    bra = np.repeat(ket[None, :], r.size, axis=0)
                    sign = np.ones(r.size, dtype=np.float64)
                    item = np.arange(r.size)
                    for target, add in (
                        (source_p, False),
                        (source_q, False),
                        (is_, True),
                        (ir, True),
                    ):
                        target_spin = target // norb
                        target_orbital = target - target_spin * norb
                        target_word = target_orbital >> 6
                        shift = (target_orbital & 63).astype(np.uint64)
                        parity = np.where(
                            target_spin == 1,
                            np.bitwise_count(bra[:, 0]).sum(axis=1),
                            0,
                        )
                        for w in range(bra.shape[2]):
                            parity += np.where(
                                target_word > w,
                                np.bitwise_count(bra[item, target_spin, w]),
                                0,
                            )
                        mask = np.left_shift(np.uint64(1), shift) - np.uint64(1)
                        parity += np.bitwise_count(
                            bra[item, target_spin, target_word] & mask
                        )
                        sign *= 1.0 - 2.0 * (parity & 1)

                        target_bit = np.left_shift(np.uint64(1), shift)
                        if add:
                            bra[item, target_spin, target_word] |= target_bit
                        else:
                            bra[item, target_spin, target_word] &= ~target_bit

                    diagonal = np.all(bra == ket, axis=(1, 2))
                    if np.any(diagonal):
                        np.add.at(
                            out,
                            (
                                spin,
                                tau,
                                ps[diagonal],
                                qs[diagonal],
                                r[diagonal],
                                s[diagonal],
                            ),
                            weight[iket] * sign[diagonal],
                        )
                    if np.any(~diagonal):
                        active = ~diagonal
                        raw, _, inverse = sector.unique(bra[active])
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
                                jax.tree.map(
                                    lambda a: a[inverse],
                                    bra_logpsi,
                                ),
                                jax.tree.map(lambda a: a[iket], ket_logpsi),
                            )
                        ).reshape(-1)
                        np.add.at(
                            out,
                            (
                                spin,
                                tau,
                                ps[active],
                                qs[active],
                                r[active],
                                s[active],
                            ),
                            weight[iket] * sign[active] * ratio,
                        )

    return np.real_if_close(out)


def density_correlation(sector: Any, x: Any, weight: Any) -> np.ndarray:
    """Return C[p,q] = <n[p] n[q]> - <n[p]><n[q]>."""
    norb = sector.norb
    x = sector.asarray(x)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    weight /= np.sum(weight)

    density = np.zeros(norb, dtype=np.float64)
    density_product = np.zeros((norb, norb), dtype=np.float64)
    orbital = np.arange(norb, dtype=np.int64)
    word = orbital >> 6
    bit = np.left_shift(np.uint64(1), (orbital & 63).astype(np.uint64))

    for slc in batch.chunk(x.shape[0]):
        occupation = np.sum(
            (x[slc, :, word] & bit) != 0,
            axis=1,
            dtype=np.float64,
        )
        weighted = weight[slc, None] * occupation
        density += np.sum(weighted, axis=0)
        density_product += weighted.T @ occupation

    out = density_product - np.outer(density, density)
    return 0.5 * (out + out.T)
