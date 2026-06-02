from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import libdet
import numpy as np

from ..utils import precision


def unique_dets(dets: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique determinants in first-occurrence order.

    Determinants are treated as opaque uint64 bit patterns.

    Returns
    -------
    unique:
        Unique determinants with shape (U, 2, nword).
    first:
        Indices of the first occurrence of each unique determinant.
    inv:
        Input-to-unique index map.
    """
    dets = libdet.to_dets(dets)
    n_dets = int(dets.shape[0])

    if n_dets == 0:
        return (
            np.ascontiguousarray(dets),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    flat = np.ascontiguousarray(dets.reshape(n_dets, 2 * int(dets.shape[2])))
    key = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()

    _, first, inv = np.unique(key, return_index=True, return_inverse=True)

    # np.unique sorts by key. Restore first-occurrence order and remap inv.
    order = np.argsort(first, kind="stable")
    first = first[order]

    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    inv = remap[inv].astype(np.int64, copy=False)

    return np.ascontiguousarray(dets[first]), first.astype(np.int64), inv


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    """Grouped proposals for one raw Metropolis step.

    The counted walker state is {(ket_u, count_u)}. This batch groups all
    walkers proposing the same ket/bra move, so accept/reject needs only one
    binomial draw per grouped move.

    ket:
        Source determinant indices into the counted walker support.
    count:
        Number of walkers proposing each grouped move.
    dets:
        Unique proposed bra determinants.
    bra:
        Proposed bra indices into dets.
    log_qratio:
        log q(ket|bra) - log q(bra|ket) for each grouped move.
    n_conn:
        Number of Hamiltonian connections scanned while generating moves.
    """

    ket: np.ndarray
    count: np.ndarray
    dets: np.ndarray
    bra: np.ndarray
    log_qratio: np.ndarray
    n_conn: int = 0


def propose(
    name: str,
    hamiltonian: Any,
    dets: np.ndarray,
    count: np.ndarray,
    *,
    seed: int,
    eps: float = 1.0e-3,
) -> ProposalBatch:
    """Generate grouped proposal moves from a counted walker state.

    The proposal layer owns q(bra|ket) and the Metropolis-Hastings correction

        log q(ket|bra) - log q(bra|ket).

    It never evaluates the wavefunction and never accepts or rejects moves.
    """
    dets = libdet.to_dets(dets)
    count = np.asarray(count, dtype=np.int64)

    n_ket = int(dets.shape[0])
    nword = int(dets.shape[2])
    rdtype = precision.dtype("calc", "real", host=True)

    empty_dets = np.empty((0, 2, nword), dtype=np.uint64)
    empty_i64 = np.empty(0, dtype=np.int64)
    empty_real = np.empty(0, dtype=rdtype)

    match str(name):
        case "single":
            return _single_proposal(
                hamiltonian,
                dets,
                count,
                seed=int(seed),
                empty_dets=empty_dets,
                empty_i64=empty_i64,
                empty_real=empty_real,
            )

        case "ham":
            return _ham_proposal(
                hamiltonian,
                dets,
                count,
                seed=int(seed),
                eps=float(eps),
                n_ket=n_ket,
                empty_dets=empty_dets,
                empty_i64=empty_i64,
                empty_real=empty_real,
            )

        case _:
            raise ValueError("proposal must be 'ham' or 'single'")


def _single_proposal(
    hamiltonian: Any,
    dets: np.ndarray,
    count: np.ndarray,
    *,
    seed: int,
    empty_dets: np.ndarray,
    empty_i64: np.ndarray,
    empty_real: np.ndarray,
) -> ProposalBatch:
    """Uniform single excitation within a fixed (N_alpha, N_beta) sector."""
    rng = np.random.default_rng(seed)
    n_ket = int(dets.shape[0])

    ket = np.repeat(np.arange(n_ket, dtype=np.int64), count)
    if ket.size == 0:
        return ProposalBatch(
            ket=empty_i64,
            count=empty_i64,
            dets=empty_dets,
            bra=empty_i64,
            log_qratio=empty_real,
        )

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
        return ProposalBatch(
            ket=empty_i64,
            count=empty_i64,
            dets=empty_dets,
            bra=empty_i64,
            log_qratio=empty_real,
        )

    move = rng.integers(n_move, size=ket.size, dtype=np.int64)

    spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
    move_spin = np.where(spin == 0, move, move - n_move_a)

    n_vir = np.where(spin == 0, norb - n_alpha, norb - n_beta).astype(np.int64)
    occ_rank = move_spin // n_vir
    vir_rank = move_spin % n_vir

    ket_occ = occ[ket, spin]
    ket_vir = ~ket_occ

    occ_pos = np.cumsum(ket_occ, axis=1) - 1
    vir_pos = np.cumsum(ket_vir, axis=1) - 1

    occ_orb = np.argmax(ket_occ & (occ_pos == occ_rank[:, None]), axis=1).astype(
        np.int64
    )
    vir_orb = np.argmax(ket_vir & (vir_pos == vir_rank[:, None]), axis=1).astype(
        np.int64
    )

    bra_dets = np.ascontiguousarray(dets[ket].copy())
    rows = np.arange(ket.size, dtype=np.int64)

    occ_word = occ_orb >> 6
    occ_bit = (occ_orb & 63).astype(np.uint64)
    vir_word = vir_orb >> 6
    vir_bit = (vir_orb & 63).astype(np.uint64)

    bra_dets[rows, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
    bra_dets[rows, spin, vir_word] |= np.uint64(1) << vir_bit

    bras, _, bra = unique_dets(bra_dets)

    move_key = np.column_stack((ket, bra))
    _, first, inv0 = np.unique(move_key, axis=0, return_index=True, return_inverse=True)

    order = np.argsort(first, kind="stable")
    first = first[order]

    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    inv = remap[inv0]

    grouped_count = np.bincount(inv, minlength=order.size).astype(np.int64)
    grouped_ket = ket[first].astype(np.int64, copy=False)
    grouped_bra = bra[first].astype(np.int64, copy=False)

    return ProposalBatch(
        ket=grouped_ket,
        count=grouped_count,
        dets=bras,
        bra=grouped_bra,
        log_qratio=np.zeros(grouped_count.shape[0], dtype=empty_real.dtype),
        n_conn=0,
    )


def _ham_proposal(
    hamiltonian: Any,
    dets: np.ndarray,
    count: np.ndarray,
    *,
    seed: int,
    eps: float,
    n_ket: int,
    empty_dets: np.ndarray,
    empty_i64: np.ndarray,
    empty_real: np.ndarray,
) -> ProposalBatch:
    """Hamiltonian heat-bath proposal.

    q(bra|ket) = |H_bra,ket| / d_A(ket),
    d_A(ket)   = sum_bra |H_bra,ket| 1(|H_bra,ket| >= eps).

    For symmetric Hamiltonian connections, |H_bra,ket| cancels in the
    Metropolis-Hastings ratio, leaving log d_A(ket) - log d_A(bra).
    """
    rdtype = precision.dtype("calc", "real", host=True)
    tiny = rdtype(precision.tiny("calc"))

    sample = hamiltonian.sample_conns(
        dets,
        count,
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

    if sampled_ket.size == 0 or sampled_bras.shape[0] == 0:
        return ProposalBatch(
            ket=empty_i64,
            count=empty_i64,
            dets=empty_dets,
            bra=empty_i64,
            log_qratio=empty_real,
            n_conn=n_conn,
        )

    bras, _, bra = unique_dets(sampled_bras)
    bra_weight = np.empty(bras.shape[0], dtype=rdtype)

    # Reuse degrees for proposed bras already in the walker support.
    _, first, inv_lookup = unique_dets(np.concatenate([dets, bras], axis=0))
    bra_first = first[inv_lookup[n_ket:]]
    known = bra_first < n_ket

    if known.any():
        bra_weight[known] = ket_weight[bra_first[known]]

    if (~known).any():
        deg = hamiltonian.degrees(np.ascontiguousarray(bras[~known]), float(eps))
        bra_weight[~known] = precision.asarray(
            np.asarray(deg.ket_weight),
            "calc",
            "real",
            host=True,
        )

    log_qratio = (
        np.log(np.maximum(ket_weight[sampled_ket], tiny))
        - np.log(np.maximum(bra_weight[bra], tiny))
    )

    return ProposalBatch(
        ket=sampled_ket,
        count=sampled_count,
        dets=bras,
        bra=bra.astype(np.int64, copy=False),
        log_qratio=precision.asarray(log_qratio, "calc", "real", host=True),
        n_conn=n_conn,
    )