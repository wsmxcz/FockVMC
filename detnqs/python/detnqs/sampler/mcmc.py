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

    The chain targets |psi(det)|^alpha.

    Supported proposals:
        ham:
            Hamiltonian heat-bath proposal with |H_bra,ket| >= proposal_eps.

        single:
            Uniform same-spin single-excitation baseline.

    The Hamiltonian Fock-VMC path assumes

        proposal_eps == blur_eps == eloc_eps1,

    checked once by MCState.init.
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

        if self.proposal not in {"ham", "single"}:
            raise ValueError("proposal must be 'ham' or 'single'")

        alpha_value = (
            float(alpha)
            if self.alpha is None and alpha is not None
            else 2.0
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
        """Advance chains and return observed determinants."""
        n_samples = int(self.n_samples)
        n_chains = int(self.n_chains)

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if state.dets.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)
        alpha = float(state.alpha) if self.alpha is None else float(self.alpha)

        # Refresh chain amplitudes for the current parameters.
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

        accepted = 0
        proposed = 0
        n_conn = 0

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
            n_conn += info["n_conn"]

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
            n_conn += info["n_conn"]

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
                n_conn += info["n_conn"]

        stats = {
            "accept": float(accepted / proposed if proposed else 0.0),
            "n_conn": float(n_conn),
        }
        stats.update(timer.stats())

        return state, samples, mass, stats

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
        """Observe selected chains and make one Metropolis transition."""
        timer = utils.Timer() if timer is None else timer

        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])
        n_observe = int(n_observe)

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

        beta = float(self.blur)
        eps = float(self.proposal_eps)

        if self.proposal == "ham":
            (
                trial,
                active,
                log_q,
                logabs_y,
                observed,
                obs_mass,
                n_conn,
            ) = self._ham_proposal(
                theta,
                hamiltonian,
                model,
                state,
                rows=rows,
                observed=observed,
                beta=beta,
                eps=eps,
                rng=rng,
                timer=timer,
            )

        else:
            trial, active, log_q, logabs_y = self._single_proposal(
                theta,
                hamiltonian,
                model,
                state,
                rng=rng,
                timer=timer,
            )

            n_conn = 0
            if n_observe > 0 and beta > 0.0:
                observed, obs_mass, n_conn = self._blur(
                    hamiltonian,
                    observed,
                    beta=beta,
                    eps=eps,
                    rng=rng,
                    timer=timer,
                )

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            with timer("sample"):
                log_accept = (
                    rdtype(state.alpha) * (logabs_y - state.logabs)
                    + log_q
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
            obs_mass,
            {
                "accepted": accepted,
                "proposed": proposed,
                "n_conn": n_conn,
            },
        )

    def _ham_proposal(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
        *,
        rows: np.ndarray,
        observed: np.ndarray,
        beta: float,
        eps: float,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
    ]:
        """Hamiltonian proposal and optional blur from one connection sample."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])
        n_observe = int(observed.shape[0])

        with timer("sample"):
            kets, ket_first, ket_index = libdet.unique_dets(state.dets)
            n_ket = int(kets.shape[0])

            ket_logabs = state.logabs[ket_first]
            proposal_counts = np.bincount(
                ket_index,
                minlength=n_ket,
            ).astype(np.int64)

            blur = np.zeros(n_observe, dtype=bool)
            counts: np.ndarray

            if n_observe > 0 and beta > 0.0:
                obs_index = ket_index[rows]
                blur = rng.random(n_observe) < beta

                blur_counts = np.bincount(
                    obs_index[blur],
                    minlength=n_ket,
                ).astype(np.int64)

                counts = np.stack((proposal_counts, blur_counts))

            else:
                obs_index = ket_index[rows]
                counts = proposal_counts

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
            sample_dets = np.asarray(sample.dets, dtype=np.uint64)
            sample_ket_ptr = np.asarray(sample.ket_ptr, dtype=np.int64)
            sample_bra_idx = np.asarray(sample.bra_idx, dtype=np.int64)
            sample_count = np.asarray(sample.count, dtype=np.int64)

        with timer("sample"):
            trial, trial_bra = self._draw(
                state.dets,
                ket_index,
                np.ones(n_chain, dtype=bool),
                n_ket=n_ket,
                ket_ptr=sample_ket_ptr,
                bra_idx=sample_bra_idx,
                count=sample_count,
                bra_dets=sample_dets,
                stream=0,
                rng=rng,
            )
            active = trial_bra >= 0

            if n_observe > 0 and beta > 0.0:
                observed, _ = self._draw(
                    observed,
                    obs_index,
                    blur,
                    n_ket=n_ket,
                    ket_ptr=sample_ket_ptr,
                    bra_idx=sample_bra_idx,
                    count=sample_count,
                    bra_dets=sample_dets,
                    stream=1,
                    rng=rng,
                )
                obs_mass = np.where(
                    ket_weight[obs_index] > 0.0,
                    ket_weight[obs_index],
                    rdtype(1.0),
                )
            else:
                obs_mass = np.ones(n_observe, dtype=rdtype)

        log_q = np.zeros(n_chain, dtype=rdtype)
        logabs_y = np.empty(n_chain, dtype=rdtype)
        logabs_y[~active] = state.logabs[~active]

        if active.any():
            with timer("conns"):
                tiny = rdtype(precision.tiny("calc"))

                bra_pool, bra_inverse = np.unique(
                    trial_bra[active],
                    return_inverse=True,
                )
                bra_known = bra_pool < n_ket

                bra_weight = np.empty(bra_pool.size, dtype=rdtype)

                if bra_known.any():
                    bra_weight[bra_known] = ket_weight[
                        bra_pool[bra_known]
                    ]

                if (~bra_known).any():
                    weight, _ = hamiltonian.degrees(
                        np.ascontiguousarray(
                            sample_dets[bra_pool[~bra_known]]
                        ),
                        eps,
                    )
                    bra_weight[~bra_known] = precision.asarray(
                        weight,
                        "calc",
                        "real",
                        host=True,
                    )

                ket_w = ket_weight[ket_index[active]]
                bra_w = bra_weight[bra_inverse]

                log_q[active] = (
                    np.log(np.maximum(ket_w, tiny))
                    - np.log(np.maximum(bra_w, tiny))
                )

            unique_logabs = np.empty(bra_pool.size, dtype=rdtype)

            if bra_known.any():
                unique_logabs[bra_known] = ket_logabs[
                    bra_pool[bra_known]
                ]

            if (~bra_known).any():
                bra_dets = np.ascontiguousarray(
                    sample_dets[bra_pool[~bra_known]]
                )

                with timer("forward"):
                    value = utils.apply(model.logabs, theta, bra_dets)
                    jax.block_until_ready(value)

                unique_logabs[~bra_known] = precision.asarray(
                    np.asarray(utils.host(value)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                )

            logabs_y[active] = unique_logabs[bra_inverse]

        return trial, active, log_q, logabs_y, observed, obs_mass, n_conn

    def _single_proposal(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Chains,
        *,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform same-spin single-excitation baseline."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.dets.shape[0])

        with timer("sample"):
            trial = np.ascontiguousarray(state.dets.copy())
            log_q = np.zeros(n_chain, dtype=rdtype)
            active = np.zeros(n_chain, dtype=bool)

            norb = int(hamiltonian.norb)
            orb = np.arange(norb, dtype=np.int64)
            word = orb >> 6
            bit = (orb & 63).astype(np.uint64)

            occ = ((state.dets[:, :, word] >> bit) & np.uint64(1)).astype(bool)
            n_occ = occ.sum(axis=2)

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

        return trial, active, log_q, logabs_y

    def _blur(
        self,
        hamiltonian: Any,
        dets: np.ndarray,
        *,
        beta: float,
        eps: float,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Apply Hamiltonian blur to observed determinants."""
        rdtype = precision.dtype("calc", "real", host=True)
        n_det = int(dets.shape[0])

        if n_det == 0 or beta <= 0.0:
            return dets, np.ones(n_det, dtype=rdtype), 0

        with timer("sample"):
            kets, _, ket_index = libdet.unique_dets(dets)
            blur = rng.random(n_det) < beta

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
            sample_dets = np.asarray(sample.dets, dtype=np.uint64)
            sample_ket_ptr = np.asarray(sample.ket_ptr, dtype=np.int64)
            sample_bra_idx = np.asarray(sample.bra_idx, dtype=np.int64)
            sample_count = np.asarray(sample.count, dtype=np.int64)

        with timer("sample"):
            out, _ = self._draw(
                dets,
                ket_index,
                blur,
                n_ket=int(kets.shape[0]),
                ket_ptr=sample_ket_ptr,
                bra_idx=sample_bra_idx,
                count=sample_count,
                bra_dets=sample_dets,
                stream=0,
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
        *,
        n_ket: int,
        ket_ptr: np.ndarray,
        bra_idx: np.ndarray,
        count: np.ndarray,
        bra_dets: np.ndarray,
        stream: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign sampled bras to selected determinant rows."""
        out = np.ascontiguousarray(dets.copy())
        bra_pool_idx = np.full(dets.shape[0], -1, dtype=np.int64)

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
            rng.shuffle(draw)

            ket_rows = rows[begin:end]
            out[ket_rows] = bra_dets[bra_idx[draw]]
            bra_pool_idx[ket_rows] = bra_idx[draw]

        return out, bra_pool_idx
