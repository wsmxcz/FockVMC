from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import libdet
import numpy as np

from ..utils import precision


def unique_dets(dets: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique determinants in first-occurrence order.

    Determinants are treated as opaque bit patterns. The returned order is
    stable with respect to the first appearance in the input.

    Returns
    -------
    unique:
        Unique determinants with shape ``(U, 2, nword)``.
    first:
        Indices of the first occurrence of each unique determinant.
    inv:
        Input-to-unique index map.
    """
    dets = libdet.to_dets(dets)
    n = int(dets.shape[0])

    if n == 0:
        return (
            np.ascontiguousarray(dets),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    flat = np.ascontiguousarray(dets.reshape(n, 2 * int(dets.shape[2])))
    key = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()

    _, first, inv = np.unique(key, return_index=True, return_inverse=True)

    # np.unique sorts by key. Reorder the unique table by first occurrence.
    order = np.argsort(first, kind="stable")
    first = first[order]

    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size, dtype=np.int64)
    inv = remap[inv].astype(np.int64, copy=False)

    return np.ascontiguousarray(dets[first]), first.astype(np.int64), inv


@dataclass(frozen=True, slots=True)
class ProposalBatch:
    """Grouped proposals for one raw Metropolis step.

    The sampler stores an empirical walker state ``{(x_u, m_u)}``.
    This batch groups all walkers that propose the same source/destination
    pair. The Metropolis accept/reject step can then use one binomial draw
    per grouped move.

    src:
        Source row indices into the counted walker state.
    count:
        Number of walkers proposing each grouped move.
    dets:
        Unique proposed determinants.
    dst:
        Destination row indices into ``dets``.
    log_qratio:
        ``log q(x|y) - log q(y|x)`` for each grouped move.
    n_edge:
        Number of Hamiltonian proposal edges touched while generating moves.
    """

    src: np.ndarray
    count: np.ndarray
    dets: np.ndarray
    dst: np.ndarray
    log_qratio: np.ndarray
    n_edge: int = 0


def propose(
    name: str,
    hamiltonian: Any,
    dets: np.ndarray,
    count: np.ndarray,
    *,
    seed: int,
    eps: float = 1.0e-3,
    timing: dict[str, float] | None = None,
) -> ProposalBatch:
    """Generate grouped proposal moves from a counted walker state.

    The proposal layer owns only the transition kernel ``q(y|x)`` and the
    reverse/forward correction ``log q(x|y) - log q(y|x)``. It never evaluates
    the wave function and never accepts or rejects moves.
    """
    dets = libdet.to_dets(dets)
    count = np.asarray(count, dtype=np.int64)
    name = str(name)

    n_row = int(dets.shape[0])
    nword = int(dets.shape[2])
    rdtype = precision.dtype("calc", "real", host=True)
    empty_dets = np.empty((0, 2, nword), dtype=np.uint64)

    if name == "single":
        # Uniform single excitation within a fixed (N_alpha, N_beta) sector.
        # The number of possible reverse moves is the same, so log_qratio = 0.
        rng = np.random.default_rng(int(seed))
        src = np.repeat(np.arange(n_row, dtype=np.int64), count)

        if src.size == 0:
            return ProposalBatch(
                src=np.empty(0, dtype=np.int64),
                count=np.empty(0, dtype=np.int64),
                dets=empty_dets,
                dst=np.empty(0, dtype=np.int64),
                log_qratio=np.empty(0, dtype=rdtype),
                n_edge=0,
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
        n_vir_a = norb - n_alpha
        n_vir_b = norb - n_beta

        n_move_a = n_alpha * n_vir_a
        n_move_b = n_beta * n_vir_b
        n_move = n_move_a + n_move_b

        if n_move <= 0:
            return ProposalBatch(
                src=np.empty(0, dtype=np.int64),
                count=np.empty(0, dtype=np.int64),
                dets=empty_dets,
                dst=np.empty(0, dtype=np.int64),
                log_qratio=np.empty(0, dtype=rdtype),
                n_edge=0,
            )

        move = rng.integers(n_move, size=src.size, dtype=np.int64)
        spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
        move_spin = np.where(spin == 0, move, move - n_move_a)

        n_vir = np.where(spin == 0, n_vir_a, n_vir_b).astype(np.int64)
        occ_rank = move_spin // n_vir
        vir_rank = move_spin % n_vir

        src_occ = occ[src, spin, :]
        src_vir = ~src_occ

        occ_pos = np.cumsum(src_occ, axis=1) - 1
        vir_pos = np.cumsum(src_vir, axis=1) - 1

        occ_orb = np.argmax(src_occ & (occ_pos == occ_rank[:, None]), axis=1).astype(np.int64)
        vir_orb = np.argmax(src_vir & (vir_pos == vir_rank[:, None]), axis=1).astype(np.int64)

        prop_dets = np.ascontiguousarray(dets[src].copy())
        rows = np.arange(src.size, dtype=np.int64)

        occ_word = occ_orb >> 6
        occ_bit = (occ_orb & 63).astype(np.uint64)
        vir_word = vir_orb >> 6
        vir_bit = (vir_orb & 63).astype(np.uint64)

        prop_dets[rows, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
        prop_dets[rows, spin, vir_word] |= np.uint64(1) << vir_bit

        prop, _, dst = unique_dets(prop_dets)

        pair = np.column_stack((src, dst))
        _, first, inv0 = np.unique(pair, axis=0, return_index=True, return_inverse=True)

        order = np.argsort(first, kind="stable")
        first = first[order]

        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size, dtype=np.int64)
        inv = remap[inv0]

        grouped_count = np.bincount(inv, minlength=order.size).astype(np.int64)
        grouped_src = src[first].astype(np.int64, copy=False)
        grouped_dst = dst[first].astype(np.int64, copy=False)

        return ProposalBatch(
            src=grouped_src,
            count=grouped_count,
            dets=prop,
            dst=grouped_dst,
            log_qratio=np.zeros(grouped_count.shape[0], dtype=rdtype),
            n_edge=0,
        )

    if name == "ham":
        # Hamiltonian heat-bath proposal:
        #
        #   q(y|x) = |H_xy| / d_A(x),
        #   d_A(x) = sum_z |H_xz| 1(|H_xz| >= eps).
        #
        # For a symmetric Hamiltonian graph, |H_xy| cancels in the
        # Metropolis-Hastings ratio, leaving log d_A(x) - log d_A(y).
        pa = precision.asarray
        tiny = rdtype(precision.tiny("calc"))

        t = perf_counter()
        sample = hamiltonian.sample_edges(
            dets,
            count,
            eps1=np.inf,
            eps2=float(eps),
            seed=int(seed),
        )
        degree_x = pa(np.asarray(sample.row_weight), "calc", "real", host=True)
        n_edge = int(np.asarray(sample.h).size)

        if timing is not None:
            timing["time_graph"] += perf_counter() - t

        if n_edge == 0:
            return ProposalBatch(
                src=np.empty(0, dtype=np.int64),
                count=np.empty(0, dtype=np.int64),
                dets=empty_dets,
                dst=np.empty(0, dtype=np.int64),
                log_qratio=np.empty(0, dtype=rdtype),
                n_edge=n_edge,
            )

        src = np.asarray(sample.rows, dtype=np.int64)
        grouped_count = np.asarray(sample.counts, dtype=np.int64)
        pair_dets = np.ascontiguousarray(np.asarray(sample.dets, dtype=np.uint64))

        if src.size == 0 or pair_dets.shape[0] == 0:
            return ProposalBatch(
                src=np.empty(0, dtype=np.int64),
                count=np.empty(0, dtype=np.int64),
                dets=empty_dets,
                dst=np.empty(0, dtype=np.int64),
                log_qratio=np.empty(0, dtype=rdtype),
                n_edge=n_edge,
            )

        prop, _, dst = unique_dets(pair_dets)

        # Reuse degrees if a proposed determinant is already in the current
        # walker support; otherwise query libdet once for the unknown support.
        prop_degree = np.empty(prop.shape[0], dtype=rdtype)

        _, first, inv_lookup = unique_dets(np.concatenate([dets, prop], axis=0))
        prop_first = first[inv_lookup[n_row:]]
        known = prop_first < n_row

        if known.any():
            prop_degree[known] = degree_x[prop_first[known]]

        if (~known).any():
            prop_unk = np.ascontiguousarray(prop[~known])

            t = perf_counter()
            deg = hamiltonian.degrees(prop_unk, float(eps))
            prop_degree[~known] = pa(np.asarray(deg.row_weight), "calc", "real", host=True)

            if timing is not None:
                timing["time_graph"] += perf_counter() - t

        log_qratio = (
            np.log(np.maximum(degree_x[src], tiny))
            - np.log(np.maximum(prop_degree[dst], tiny))
        )

        return ProposalBatch(
            src=src,
            count=grouped_count,
            dets=prop,
            dst=dst.astype(np.int64, copy=False),
            log_qratio=pa(log_qratio, "calc", "real", host=True),
            n_edge=n_edge,
        )

    raise ValueError("proposal must be 'ham' or 'single'")