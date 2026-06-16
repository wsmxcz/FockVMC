from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from .. import utils
from ..utils import precision


@dataclass(frozen=True, slots=True)
class Chains:
    """State of determinant Markov chains."""

    key: jax.Array
    dets: np.ndarray
    logabs: np.ndarray
    alpha: float = 1.0
    alpha_step: int = 0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Metropolis sampler for determinant chains.

    The chain targets ``|psi(det)|^alpha``. Optional Hamiltonian blur changes
    only the observed dets, not the chain transition.
    """

    n_samples: int = 1024
    n_chains: int = 1024

    thermal_steps: int = 32
    discard_steps: int = 0
    sweep_steps: int = 1
    reset_chains: bool = False

    alpha: float | None = None
    proposal: str = "ham"
    proposal_eps: float = 1.0e-3

    blur: float = 0.5
    blur_eps: float | None = 1.0e-3

    def init(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        *,
        key: jax.Array,
        n_alpha: int,
        n_beta: int,
        chain_init: str | Any = "hf",
        alpha: float | None = None,
        alpha_step: int = 0,
    ) -> Chains:
        """Initialize chains and run burn-in."""
        n_chains = int(self.n_chains)
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")

        alpha_value = (
            float(alpha)
            if self.alpha is None and alpha is not None
            else 1.0
            if self.alpha is None
            else float(self.alpha)
        )

        key, init_key = jax.random.split(key)
        nword = int(hamiltonian.nword)
        norb = int(hamiltonian.norb)
        n_alpha = int(n_alpha)
        n_beta = int(n_beta)

        if isinstance(chain_init, str):
            if chain_init == "hf":
                det = np.zeros((1, 2, nword), dtype=np.uint64)
                for p in range(n_alpha):
                    det[0, 0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
                for p in range(n_beta):
                    det[0, 1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
                dets = np.repeat(det, n_chains, axis=0)

            elif chain_init == "random":
                seed = int(jax.random.bits(init_key, (), dtype=jnp.uint32))
                rng = np.random.default_rng(seed)
                dets = np.zeros((n_chains, 2, nword), dtype=np.uint64)
                rows = np.arange(n_chains, dtype=np.int64)[:, None]

                for spin, n_elec in enumerate((n_alpha, n_beta)):
                    if n_elec < 0 or n_elec > norb:
                        raise ValueError(
                            "electron number must satisfy 0 <= n <= norb"
                        )
                    if n_elec == 0:
                        continue
                    if n_elec == norb:
                        occ = np.broadcast_to(
                            np.arange(norb, dtype=np.int64),
                            (n_chains, norb),
                        )
                    else:
                        score = rng.random((n_chains, norb))
                        occ = np.argpartition(score, n_elec - 1, axis=1)[
                            :, :n_elec
                        ]

                    word = occ >> 6
                    bit = (occ & 63).astype(np.uint64)
                    np.bitwise_or.at(
                        dets[:, spin, :],
                        (rows, word),
                        np.uint64(1) << bit,
                    )
            else:
                raise ValueError(
                    "chain_init must be 'hf', 'random', or a determinant batch"
                )
        else:
            dets = libdet.to_dets(chain_init)
            if dets.shape[0] == 0:
                raise ValueError("chain_init determinant batch must be non-empty")
            if dets.shape[0] != n_chains:
                reps = (n_chains + dets.shape[0] - 1) // dets.shape[0]
                dets = np.tile(dets, (reps, 1, 1))[:n_chains]
            dets = np.ascontiguousarray(dets)

        unique, _, inv = libdet.unique_dets(dets)
        value = utils.apply(model.logabs, theta, unique)
        jax.block_until_ready(value)
        unique_logabs = precision.asarray(
            np.asarray(utils.host(value)).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        logabs = precision.asarray(
            unique_logabs[inv],
            "calc",
            "real",
            host=True,
        )

        state = Chains(
            key=key,
            dets=dets,
            logabs=logabs,
            alpha=alpha_value,
            alpha_step=max(0, int(alpha_step)),
        )

        for _ in range(max(0, int(self.thermal_steps))):
            state, _, _, _ = self._step(theta, hamiltonian, model, state)

        return state

    def draw(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, float]]:
        """Advance chains and return observed dets with sample masses."""
        n_samples = int(self.n_samples)
        n_chains = int(self.n_chains)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if state.dets.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)
        alpha = float(state.alpha) if self.alpha is None else float(self.alpha)

        with timer("forward"):
            unique, _, inv = libdet.unique_dets(state.dets)
            value = utils.apply(model.logabs, theta, unique)
            jax.block_until_ready(value)
            unique_logabs = precision.asarray(
                np.asarray(utils.host(value)).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            state = replace(
                state,
                logabs=precision.asarray(
                    unique_logabs[inv],
                    "calc",
                    "real",
                    host=True,
                ),
                alpha=alpha,
            )

        accepted = proposed = 0
        n_conn_proposal = 0
        n_conn_blur = 0

        for _ in range(max(0, int(self.discard_steps))):
            state, _, _, info = self._step(
                theta,
                hamiltonian,
                model,
                state,
                timer=timer,
            )
            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn_proposal += info["n_conn_proposal"]

        samples = np.empty((n_samples, 2, state.dets.shape[2]), dtype=np.uint64)
        mass = np.empty(n_samples, dtype=rdtype)
        sweep_steps = max(1, int(self.sweep_steps))
        offset = 0

        while offset < n_samples:
            take = min(n_chains, n_samples - offset)
            state, dets, obs_mass, info = self._step(
                theta,
                hamiltonian,
                model,
                state,
                n_observe=take,
                timer=timer,
            )
            samples[offset : offset + take] = dets
            mass[offset : offset + take] = obs_mass
            offset += take

            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn_proposal += info["n_conn_proposal"]
            n_conn_blur += info["n_conn_blur"]

            for _ in range(sweep_steps - 1):
                state, _, _, info = self._step(
                    theta,
                    hamiltonian,
                    model,
                    state,
                    timer=timer,
                )
                accepted += info["accepted"]
                proposed += info["proposed"]
                n_conn_proposal += info["n_conn_proposal"]

        stats = {
            "accept": float(accepted / proposed if proposed else 0.0),
            "n_conn_proposal": float(n_conn_proposal),
            "n_conn_blur": float(n_conn_blur),
        }
        stats.update(timer.stats())

        return (
            state,
            samples,
            precision.asarray(mass, "calc", "real", host=True),
            stats,
        )

    def _step(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
        *,
        n_observe: int = 0,
        timer: utils.Timer | None = None,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, int]]:
        """Observe selected chains, then make one Metropolis transition."""
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])
        n_observe = int(n_observe)
        if n_observe < 0 or n_observe > n_chain:
            raise ValueError("n_observe must satisfy 0 <= n_observe <= n_chains")

        with timer("sample"):
            key, random_key = jax.random.split(state.key)
            seed = int(jax.random.bits(random_key, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)
            rows = (
                np.arange(n_chain, dtype=np.int64)
                if n_observe == n_chain
                else rng.choice(n_chain, size=n_observe, replace=False)
            )
            observed = np.ascontiguousarray(state.dets[rows])
            obs_mass = np.ones(n_observe, dtype=rdtype)

        beta = float(np.clip(self.blur, 0.0, 1.0))
        eps = float(self.proposal_eps)
        blur_eps = eps if self.blur_eps is None else float(self.blur_eps)
        shared_blur = (
            n_observe > 0
            and beta > 0.0
            and self.proposal == "ham"
            and blur_eps == eps
        )

        if self.proposal == "single":
            (
                trial,
                active,
                log_q,
                logabs_y,
                n_conn_proposal,
                cache,
            ) = self._single_proposal(
                theta,
                hamiltonian,
                model,
                state,
                rng=rng,
                timer=timer,
            )
        elif self.proposal == "ham":
            (
                trial,
                active,
                log_q,
                logabs_y,
                n_conn_proposal,
                cache,
            ) = self._ham_proposal(
                theta,
                hamiltonian,
                model,
                state,
                rows=rows,
                n_observe=n_observe,
                shared_blur=shared_blur,
                beta=beta,
                eps=eps,
                rng=rng,
                timer=timer,
            )
        else:
            raise ValueError("proposal must be 'ham' or 'single'")

        n_conn_blur = 0
        if n_observe > 0 and beta > 0.0:
            if shared_blur:
                observed, obs_mass, _ = self._blur(
                    hamiltonian,
                    observed,
                    beta=beta,
                    eps=blur_eps,
                    rng=rng,
                    timer=timer,
                    kets=cache["kets"],
                    ket_index=cache["ket_index"][rows],
                    sample=cache["sample"],
                    weight=cache["ket_weight"],
                    selected=cache["blur"],
                    stream=1,
                )
            else:
                kets = cache.get("kets") if self.proposal == "ham" else None
                ket_index = (
                    cache["ket_index"][rows]
                    if self.proposal == "ham"
                    else None
                )
                observed, obs_mass, n_conn_blur = self._blur(
                    hamiltonian,
                    observed,
                    beta=beta,
                    eps=blur_eps,
                    rng=rng,
                    timer=timer,
                    kets=kets,
                    ket_index=ket_index,
                )

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            with timer("sample"):
                log_accept = (
                    rdtype(state.alpha) * (logabs_y - state.logabs)
                    + precision.asarray(log_q, "calc", "real", host=True)
                )
                accept = active & (
                    np.log(rng.random(n_chain))
                    < np.minimum(rdtype(0.0), log_accept)
                )
                accepted = int(np.count_nonzero(accept))
                next_state = replace(
                    state,
                    key=key,
                    dets=np.ascontiguousarray(
                        np.where(accept.reshape((-1, 1, 1)), trial, state.dets)
                    ),
                    logabs=precision.asarray(
                        np.where(accept, logabs_y, state.logabs),
                        "calc",
                        "real",
                        host=True,
                    ),
                )

        return (
            next_state,
            observed,
            precision.asarray(obs_mass, "calc", "real", host=True),
            {
                "accepted": accepted,
                "proposed": proposed,
                "n_conn_proposal": n_conn_proposal,
                "n_conn_blur": n_conn_blur,
            },
        )

    def _single_proposal(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
        *,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
        """Uniform same-spin single excitations."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])

        with timer("sample"):
            trial = np.ascontiguousarray(state.dets.copy())
            log_q = np.zeros(n_chain, dtype=rdtype)
            active = np.zeros(n_chain, dtype=bool)
            norb = int(hamiltonian.norb)

            if n_chain > 0:
                orb = np.arange(norb, dtype=np.int64)
                word = orb >> 6
                bit = (orb & 63).astype(np.uint64)
                occ = ((state.dets[:, :, word] >> bit) & np.uint64(1)).astype(
                    bool
                )
                n_occ = occ.sum(axis=2)
                if not np.all(n_occ == n_occ[0]):
                    raise ValueError(
                        "single proposal requires a fixed particle-number sector"
                    )

                n_alpha = int(n_occ[0, 0])
                n_beta = int(n_occ[0, 1])
                n_move_a = n_alpha * (norb - n_alpha)
                n_move_b = n_beta * (norb - n_beta)
                n_move = n_move_a + n_move_b

                if n_move > 0:
                    move = rng.integers(n_move, size=n_chain, dtype=np.int64)
                    spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
                    move_spin = np.where(spin == 0, move, move - n_move_a)
                    n_vir = np.where(spin == 0, norb - n_alpha, norb - n_beta)
                    occ_rank = move_spin // n_vir
                    vir_rank = move_spin % n_vir

                    chain_occ = occ[np.arange(n_chain), spin]
                    chain_vir = ~chain_occ
                    occ_pos = np.cumsum(chain_occ, axis=1) - 1
                    vir_pos = np.cumsum(chain_vir, axis=1) - 1
                    occ_orb = np.argmax(
                        chain_occ & (occ_pos == occ_rank[:, None]),
                        axis=1,
                    ).astype(np.int64)
                    vir_orb = np.argmax(
                        chain_vir & (vir_pos == vir_rank[:, None]),
                        axis=1,
                    ).astype(np.int64)

                    row = np.arange(n_chain, dtype=np.int64)
                    occ_word = occ_orb >> 6
                    occ_bit = (occ_orb & 63).astype(np.uint64)
                    vir_word = vir_orb >> 6
                    vir_bit = (vir_orb & 63).astype(np.uint64)
                    trial[row, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
                    trial[row, spin, vir_word] |= np.uint64(1) << vir_bit
                    active[:] = True

        logabs_y = np.empty(n_chain, dtype=rdtype)
        logabs_y[~active] = state.logabs[~active]

        if active.any():
            unique_y, _, inverse_y = libdet.unique_dets(trial[active])
            _, first, lookup = libdet.unique_dets(
                np.concatenate((state.dets, unique_y), axis=0)
            )
            first_y = first[lookup[n_chain:]]
            known = first_y < n_chain
            unique_logabs = np.empty(unique_y.shape[0], dtype=rdtype)

            if known.any():
                unique_logabs[known] = state.logabs[first_y[known]]
            if (~known).any():
                with timer("forward"):
                    value = utils.apply(
                        model.logabs,
                        theta,
                        np.ascontiguousarray(unique_y[~known]),
                    )
                    jax.block_until_ready(value)
                unique_logabs[~known] = precision.asarray(
                    np.asarray(utils.host(value)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                )
            logabs_y[active] = unique_logabs[inverse_y]

        return trial, active, log_q, logabs_y, 0, {}

    def _ham_proposal(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
        *,
        rows: np.ndarray,
        n_observe: int,
        shared_blur: bool,
        beta: float,
        eps: float,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
        """Hamiltonian heat-bath proposal."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])

        with timer("sample"):
            kets, ket_first, ket_index = libdet.unique_dets(state.dets)
            ket_logabs = state.logabs[ket_first]
            counts = np.bincount(
                ket_index,
                minlength=kets.shape[0],
            ).astype(np.int64)

            blur = None
            if shared_blur:
                obs_index = ket_index[rows]
                blur = rng.random(n_observe) < beta
                blur_counts = np.bincount(
                    obs_index[blur],
                    minlength=kets.shape[0],
                ).astype(np.int64)
                counts = np.stack((counts, blur_counts))

        with timer("conns"):
            sample = hamiltonian.sample_conns(
                kets,
                np.ascontiguousarray(counts),
                eps1=np.inf,
                eps2=eps,
                seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
            )
            n_conn = int(np.asarray(sample.h).size)
            ket_weight = precision.asarray(
                np.asarray(sample.weight),
                "calc",
                "real",
                host=True,
            )

        with timer("sample"):
            trial, trial_bra = self._draw(
                state.dets,
                ket_index,
                np.ones(n_chain, dtype=bool),
                sample,
                stream=0,
                rng=rng,
            )
            active = trial_bra >= 0

        log_q = np.zeros(n_chain, dtype=rdtype)
        logabs_y = np.empty(n_chain, dtype=rdtype)
        logabs_y[~active] = state.logabs[~active]
        proposal_dets = np.asarray(sample.dets, dtype=np.uint64)

        if active.any():
            with timer("conns"):
                tiny = rdtype(precision.tiny("calc"))
                target_idx, target_inverse = np.unique(
                    trial_bra[active],
                    return_inverse=True,
                )
                target_known = target_idx < kets.shape[0]
                target_weight = np.empty(target_idx.size, dtype=rdtype)

                if target_known.any():
                    target_weight[target_known] = ket_weight[
                        target_idx[target_known]
                    ]

                if (~target_known).any():
                    weight, _ = hamiltonian.degrees(
                        np.ascontiguousarray(
                            proposal_dets[target_idx[~target_known]]
                        ),
                        eps,
                    )
                    target_weight[~target_known] = precision.asarray(
                        weight,
                        "calc",
                        "real",
                        host=True,
                    )

                source = ket_weight[ket_index[active]]
                target = target_weight[target_inverse]
                log_q[active] = (
                    np.log(np.maximum(source, tiny))
                    - np.log(np.maximum(target, tiny))
                )

            unique_logabs = np.empty(target_idx.size, dtype=rdtype)
            if target_known.any():
                unique_logabs[target_known] = ket_logabs[
                    target_idx[target_known]
                ]
            if (~target_known).any():
                target_dets = np.ascontiguousarray(
                    proposal_dets[target_idx[~target_known]]
                )
                with timer("forward"):
                    value = utils.apply(model.logabs, theta, target_dets)
                    jax.block_until_ready(value)
                unique_logabs[~target_known] = precision.asarray(
                    np.asarray(utils.host(value)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                )
            logabs_y[active] = unique_logabs[target_inverse]

        return (
            trial,
            active,
            log_q,
            logabs_y,
            n_conn,
            {
                "kets": kets,
                "ket_index": ket_index,
                "ket_weight": ket_weight,
                "sample": sample,
                "blur": blur,
            },
        )

    def _blur(
        self,
        hamiltonian: Any,
        dets: np.ndarray,
        *,
        beta: float,
        eps: float,
        rng: np.random.Generator,
        timer: utils.Timer,
        kets: np.ndarray | None = None,
        ket_index: np.ndarray | None = None,
        sample: Any | None = None,
        weight: np.ndarray | None = None,
        selected: np.ndarray | None = None,
        stream: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Observe either the ket itself or a Hamiltonian-connected bra."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_det = int(dets.shape[0])
        if n_det == 0 or beta <= 0.0:
            return dets, np.ones(n_det, dtype=rdtype), 0

        with timer("sample"):
            if kets is None or ket_index is None:
                kets, _, ket_index = libdet.unique_dets(dets)
            blur = (
                rng.random(n_det) < beta
                if selected is None
                else np.asarray(selected, dtype=bool)
            )

        n_conn = 0
        if sample is None:
            with timer("sample"):
                counts = np.bincount(
                    ket_index[blur],
                    minlength=kets.shape[0],
                ).astype(np.int64)

            with timer("conns"):
                sample = hamiltonian.sample_conns(
                    kets,
                    np.ascontiguousarray(counts),
                    eps1=np.inf,
                    eps2=eps,
                    seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                )
                n_conn = int(np.asarray(sample.h).size)

            weight = precision.asarray(
                np.asarray(sample.weight),
                "calc",
                "real",
                host=True,
            )
        elif weight is None:
            weight = precision.asarray(
                np.asarray(sample.weight),
                "calc",
                "real",
                host=True,
            )

        with timer("sample"):
            out, _ = self._draw(
                dets,
                ket_index,
                blur,
                sample,
                stream=stream,
                rng=rng,
            )
            mass = np.where(
                weight[ket_index] > 0.0,
                weight[ket_index],
                rdtype(1.0),
            )

        return out, mass, n_conn

    @staticmethod
    def _draw(
        dets: np.ndarray,
        ket_index: np.ndarray,
        selected: np.ndarray,
        sample: Any,
        *,
        stream: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign sampled bras to selected ket rows."""
        out = np.ascontiguousarray(dets.copy())
        bra_pool_idx = np.full(dets.shape[0], -1, dtype=np.int64)
        n_ket = int(sample.n_kets)
        ket_ptr = np.asarray(sample.ket_ptr, dtype=np.int64)
        bra_idx = np.asarray(sample.bra_idx, dtype=np.int64)
        count = np.asarray(sample.count, dtype=np.int64)
        pool = np.asarray(sample.dets, dtype=np.uint64)

        rows = np.flatnonzero(selected)
        if rows.size == 0:
            return out, bra_pool_idx

        order = np.argsort(ket_index[rows], kind="stable")
        rows = rows[order]
        kets = ket_index[rows]
        bounds = np.flatnonzero(
            np.concatenate(
                (
                    np.ones(1, dtype=bool),
                    kets[1:] != kets[:-1],
                    np.ones(1, dtype=bool),
                )
            )
        )

        for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
            ket = int(kets[begin])
            block = int(stream) * n_ket + ket
            records = np.arange(ket_ptr[block], ket_ptr[block + 1])
            if records.size == 0:
                continue
            draw = np.repeat(records, count[records])
            ket_rows = rows[begin:end]
            if draw.size != ket_rows.size:
                raise RuntimeError(
                    "libdet returned an inconsistent bra sample count"
                )
            rng.shuffle(draw)
            out[ket_rows] = pool[bra_idx[draw]]
            bra_pool_idx[ket_rows] = bra_idx[draw]

        return out, bra_pool_idx
