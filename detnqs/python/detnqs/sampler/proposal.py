from __future__ import annotations

from typing import Any

import libdet
import numpy as np

from ..utils import precision


def unique_dets(dets: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique determinants, first positions, and inverse indices."""
    dets = libdet.to_dets(dets)
    n_dets = int(dets.shape[0])

    if n_dets == 0:
        return (
            np.ascontiguousarray(dets),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    flat = np.ascontiguousarray(dets.reshape(n_dets, -1))
    key = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()

    _, first, inv = np.unique(key, return_index=True, return_inverse=True)

    order = np.argsort(first, kind="stable")
    first = first[order]

    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    inv = remap[inv].astype(np.int64, copy=False)

    return np.ascontiguousarray(dets[first]), first.astype(np.int64), inv


def propose(
    name: str,
    hamiltonian: Any,
    dets: np.ndarray,
    *,
    seed: int,
    eps: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Propose one move for every physical walker.

    Returns
    -------
    proposed:
        Proposed determinants aligned with ``dets``.
    log_qratio:
        ``log q(x|y) - log q(y|x)`` for every walker.
    active:
        Walkers for which a nontrivial proposal exists.
    n_conn:
        Number of Hamiltonian connections scanned.
    """
    dets = libdet.to_dets(dets)

    match str(name):
        case "single":
            return _single_proposal(hamiltonian, dets, seed=int(seed))
        case "ham":
            return _ham_proposal(
                hamiltonian,
                dets,
                seed=int(seed),
                eps=float(eps),
            )
        case _:
            raise ValueError("proposal must be 'ham' or 'single'")


def _single_proposal(
    hamiltonian: Any,
    dets: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Uniform single excitation in a fixed particle-number sector.

    The number of allowed moves is constant within the sector, so the
    proposal is symmetric and its Hastings correction is zero.
    """
    rng = np.random.default_rng(seed)
    n_walker = int(dets.shape[0])
    rdtype = precision.dtype("calc", "real", host=True)

    proposed = np.ascontiguousarray(dets.copy())
    log_qratio = np.zeros(n_walker, dtype=rdtype)
    active = np.zeros(n_walker, dtype=bool)

    if n_walker == 0:
        return proposed, log_qratio, active, 0

    norb = int(hamiltonian.norb)
    orbitals = np.arange(norb, dtype=np.int64)
    word = orbitals >> 6
    bit = (orbitals & 63).astype(np.uint64)

    occ = ((dets[:, :, word] >> bit) & np.uint64(1)).astype(bool)
    n_occ = occ.sum(axis=2)

    if not np.all(n_occ == n_occ[0]):
        raise ValueError("single proposal requires a fixed (n_alpha, n_beta) sector")

    n_alpha = int(n_occ[0, 0])
    n_beta = int(n_occ[0, 1])
    n_move_a = n_alpha * (norb - n_alpha)
    n_move_b = n_beta * (norb - n_beta)
    n_move = n_move_a + n_move_b

    if n_move <= 0:
        return proposed, log_qratio, active, 0

    move = rng.integers(n_move, size=n_walker, dtype=np.int64)
    spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
    move_spin = np.where(spin == 0, move, move - n_move_a)

    n_vir = np.where(spin == 0, norb - n_alpha, norb - n_beta)
    occ_rank = move_spin // n_vir
    vir_rank = move_spin % n_vir

    walker_occ = occ[np.arange(n_walker), spin]
    walker_vir = ~walker_occ
    occ_pos = np.cumsum(walker_occ, axis=1) - 1
    vir_pos = np.cumsum(walker_vir, axis=1) - 1

    occ_orb = np.argmax(
        walker_occ & (occ_pos == occ_rank[:, None]),
        axis=1,
    ).astype(np.int64)
    vir_orb = np.argmax(
        walker_vir & (vir_pos == vir_rank[:, None]),
        axis=1,
    ).astype(np.int64)

    row = np.arange(n_walker, dtype=np.int64)
    occ_word = occ_orb >> 6
    occ_bit = (occ_orb & 63).astype(np.uint64)
    vir_word = vir_orb >> 6
    vir_bit = (vir_orb & 63).astype(np.uint64)

    proposed[row, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
    proposed[row, spin, vir_word] |= np.uint64(1) << vir_bit
    active[:] = True

    return proposed, log_qratio, active, 0


def _ham_proposal(
    hamiltonian: Any,
    dets: np.ndarray,
    *,
    seed: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Heat-bath proposal ``q(y|x) = |H_yx| / d(x)``.

    Hermiticity cancels ``|H_yx|`` in the Hastings ratio, leaving
    ``log d(x) - log d(y)``.
    """
    n_walker = int(dets.shape[0])
    rdtype = precision.dtype("calc", "real", host=True)
    tiny = rdtype(precision.tiny("calc"))

    proposed = np.ascontiguousarray(dets.copy())
    log_qratio = np.zeros(n_walker, dtype=rdtype)
    active = np.zeros(n_walker, dtype=bool)

    if n_walker == 0:
        return proposed, log_qratio, active, 0

    # Scan each distinct Hamiltonian row once, but return one draw per walker.
    kets, _, walker_to_ket = unique_dets(dets)
    counts = np.bincount(walker_to_ket, minlength=kets.shape[0]).astype(np.int64)

    sample = hamiltonian.sample_conns(
        kets,
        counts,
        eps1=np.inf,
        eps2=float(eps),
        seed=int(seed),
    )

    ket_weight = precision.asarray(
        np.asarray(sample.ket_weight),
        "calc",
        "real",
        host=True,
    )
    n_conn = int(np.asarray(sample.ket_nconn, dtype=np.int64).sum())

    sampled_ket = np.asarray(sample.ket, dtype=np.int64)
    sampled_count = np.asarray(sample.counts, dtype=np.int64)
    sampled_bras = np.ascontiguousarray(np.asarray(sample.bras, dtype=np.uint64))

    if sampled_ket.size == 0:
        return proposed, log_qratio, active, n_conn

    rng = np.random.default_rng(int(seed) ^ 0x9E3779B9)

    # Random assignment preserves exchangeability of walkers sharing one ket.
    for iket in np.unique(sampled_ket):
        records = np.flatnonzero(sampled_ket == iket)
        draw = np.repeat(records, sampled_count[records])
        walkers = np.flatnonzero(walker_to_ket == iket)

        if draw.size != walkers.size:
            raise RuntimeError("libdet returned an inconsistent proposal count")

        rng.shuffle(draw)
        proposed[walkers] = sampled_bras[draw]
        active[walkers] = True

    bras, _, walker_to_bra = unique_dets(proposed[active])
    bra_weight = np.empty(bras.shape[0], dtype=rdtype)

    # Reuse degrees when a proposed determinant is already a current ket.
    _, first, lookup = unique_dets(np.concatenate([kets, bras], axis=0))
    bra_first = first[lookup[kets.shape[0] :]]
    known = bra_first < kets.shape[0]

    if known.any():
        bra_weight[known] = ket_weight[bra_first[known]]

    if (~known).any():
        degrees = hamiltonian.degrees(
            np.ascontiguousarray(bras[~known]),
            float(eps),
        )
        bra_weight[~known] = precision.asarray(
            np.asarray(degrees.ket_weight),
            "calc",
            "real",
            host=True,
        )

    source = ket_weight[walker_to_ket[active]]
    target = bra_weight[walker_to_bra]
    log_qratio[active] = (
        np.log(np.maximum(source, tiny))
        - np.log(np.maximum(target, tiny))
    )

    return proposed, log_qratio, active, n_conn
