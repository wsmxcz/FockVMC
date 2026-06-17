from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hilbert import DetSpace
from ..utils import precision


@dataclass(frozen=True, slots=True)
class HeatBath:
    """Sample bras from |H[bra, ket]|."""

    eps: float = 1.0e-3

    def __call__(
        self,
        theta: Any,
        H: Any,
        model: Any,
        state: Any,
        *,
        rng: np.random.Generator,
        timer: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        del theta, model

        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.x.shape[0])
        eps = float(self.eps)

        with timer("sample"):
            ket, _, ket_index = H.space.unique(state.x)
            n_ket = int(ket.shape[0])
            counts = np.bincount(ket_index, minlength=n_ket).astype(np.int64)

        with timer("conns"):
            conn = H.sample_conns(
                ket,
                np.ascontiguousarray(counts),
                eps1=np.inf,
                eps2=eps,
                seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
            )
            n_conn = int(np.asarray(conn.h).size)
            weight = precision.asarray(np.asarray(conn.weight), "calc", "real", host=True)
            bra_x = np.asarray(conn.x, dtype=np.uint64)
            ket_ptr = np.asarray(conn.ket_ptr, dtype=np.int64)
            bra_idx = np.asarray(conn.bra_idx, dtype=np.int64)
            count = np.asarray(conn.count, dtype=np.int64)

        with timer("sample"):
            bra = np.ascontiguousarray(state.x.copy())
            bra_pool = np.full(n_chain, -1, dtype=np.int64)
            order = np.argsort(ket_index, kind="stable")
            sorted_ket = ket_index[order]
            bounds = np.flatnonzero(
                np.concatenate(
                    (
                        np.ones(1, dtype=bool),
                        sorted_ket[1:] != sorted_ket[:-1],
                        np.ones(1, dtype=bool),
                    )
                )
            )

            for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
                k = int(sorted_ket[begin])
                records = np.arange(ket_ptr[k], ket_ptr[k + 1])
                if records.size == 0:
                    continue

                hits = np.repeat(records, count[records])
                rng.shuffle(hits)

                item = order[begin:end]
                bra[item] = bra_x[bra_idx[hits]]
                bra_pool[item] = bra_idx[hits]

            active = bra_pool >= 0

        log_q = np.zeros(n_chain, dtype=rdtype)
        if active.any():
            with timer("conns"):
                tiny = rdtype(precision.tiny("calc"))
                pool, inverse = np.unique(bra_pool[active], return_inverse=True)
                known = pool < n_ket
                bra_weight = np.empty(pool.size, dtype=rdtype)

                if known.any():
                    bra_weight[known] = weight[pool[known]]

                if (~known).any():
                    degree, _ = H.degrees(np.ascontiguousarray(bra_x[pool[~known]]), eps)
                    bra_weight[~known] = precision.asarray(
                        degree,
                        "calc",
                        "real",
                        host=True,
                    )

                ket_w = weight[ket_index[active]]
                bra_w = bra_weight[inverse]
                log_q[active] = np.log(np.maximum(ket_w, tiny)) - np.log(
                    np.maximum(bra_w, tiny)
                )

        return bra, active, log_q, {"n_conn": n_conn}


@dataclass(frozen=True, slots=True)
class Local:
    """Uniform single-orbital replacement in a determinant sector."""

    eps: float = 1.0e-3

    def __call__(
        self,
        theta: Any,
        H: Any,
        model: Any,
        state: Any,
        *,
        rng: np.random.Generator,
        timer: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
        del theta, model, timer

        space = H.space
        if not isinstance(space, DetSpace):
            raise NotImplementedError("Local proposal is defined only for DetSpace")

        x = state.x
        n = int(x.shape[0])
        bra = np.ascontiguousarray(x.copy())
        active = np.zeros(n, dtype=bool)
        log_q = np.zeros(n, dtype=precision.dtype("calc", "real", host=True))

        n_move_a = space.n_alpha * (space.norb - space.n_alpha)
        n_move_b = space.n_beta * (space.norb - space.n_beta)
        n_move = n_move_a + n_move_b
        if n_move == 0:
            return bra, active, log_q, {"n_conn": 0}

        orb = np.arange(space.norb, dtype=np.int64)
        word = orb >> 6
        bit = (orb & 63).astype(np.uint64)
        occ = ((x[:, :, word] >> bit) & np.uint64(1)).astype(bool)

        move = rng.integers(n_move, size=n, dtype=np.int64)
        spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
        move_spin = np.where(spin == 0, move, move - n_move_a)

        n_occ = np.where(spin == 0, space.n_alpha, space.n_beta)
        n_vir = space.norb - n_occ
        occ_rank = move_spin // n_vir
        vir_rank = move_spin % n_vir

        chain_occ = occ[np.arange(n), spin]
        chain_vir = ~chain_occ
        occ_pos = np.cumsum(chain_occ, axis=1) - 1
        vir_pos = np.cumsum(chain_vir, axis=1) - 1

        occ_orb = np.argmax(chain_occ & (occ_pos == occ_rank[:, None]), axis=1)
        vir_orb = np.argmax(chain_vir & (vir_pos == vir_rank[:, None]), axis=1)
        item = np.arange(n, dtype=np.int64)

        occ_word = occ_orb >> 6
        occ_bit = (occ_orb & 63).astype(np.uint64)
        vir_word = vir_orb >> 6
        vir_bit = (vir_orb & 63).astype(np.uint64)

        bra[item, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
        bra[item, spin, vir_word] |= np.uint64(1) << vir_bit
        active[:] = True

        return bra, active, log_q, {"n_conn": 0}
