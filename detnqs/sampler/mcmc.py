from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import Timer, batch, precision


@dataclass(frozen=True, slots=True)
class ChainState:
    """State of Markov chains in Fock space."""

    key: jax.Array
    x: np.ndarray
    logabs: np.ndarray
    alpha: float = 1.0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Metropolis sampler with Hamiltonian or single-excitation proposals."""

    n_samples: int = 1024
    n_chains: int = 1024

    burn_in: int = 32
    discard: int = 0
    sweep: int = 1
    reset: bool = False

    alpha: float | None = 1.0
    proposal: str = "ham"

    beta: float = 0.5

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive")
        if self.burn_in < 0 or self.discard < 0:
            raise ValueError("burn_in and discard must be nonnegative")
        if self.sweep <= 0:
            raise ValueError("sweep must be positive")
        if self.proposal not in {"ham", "single"}:
            raise ValueError("proposal must be 'ham' or 'single'")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must satisfy 0 <= beta <= 1")

        if self.alpha is not None and (
            not np.isfinite(self.alpha) or not 0.0 <= self.alpha <= 2.0
        ):
            raise ValueError("alpha must be None or satisfy 0 <= alpha <= 2")
        if self.proposal == "single" and (
            self.alpha != 2.0 or self.beta != 0.0
        ):
            raise ValueError(
                "proposal='single' requires alpha=2.0 and beta=0.0"
            )

    def init(
        self,
        params: Any,
        hamiltonian: Any,
        model: Any,
        *,
        key: jax.Array,
        eps1: float,
        chains: Any,
        alpha: float | None = None,
        timer: Timer | None = None,
    ) -> ChainState:
        """Initialize chains and run burn-in."""
        timer = Timer(timing=False) if timer is None else timer
        key, _ = jax.random.split(key)
        x = hamiltonian.sector.asarray(chains)
        if x.shape[0] != self.n_chains:
            raise ValueError("chains size must equal sampler.n_chains")

        with timer("unique"):
            unique, _, inv = hamiltonian.sector.unique(x)

        with timer("forward", n=unique.shape[0]):
            value = batch.apply(model.logabs, params, unique)
            unique_logabs = precision.host(value, "calc", "real").reshape(-1)

        logabs = unique_logabs[inv]

        if self.alpha is None:
            alpha0 = 2.0 if alpha is None else alpha
        else:
            alpha0 = self.alpha

        state = ChainState(
            key=key,
            x=x,
            logabs=logabs,
            alpha=float(np.clip(alpha0, 0.0, 2.0)),
        )

        for _ in range(self.burn_in):
            state, _, _, _ = self._step(
                params,
                hamiltonian,
                model,
                state,
                eps1=eps1,
                timer=timer,
            )

        return state

    def draw(
        self,
        params: Any,
        hamiltonian: Any,
        model: Any,
        state: ChainState,
        *,
        eps1: float,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, np.ndarray, dict[str, float | int]]:
        """Advance chains and return configurations, masses, and diagnostics."""
        n_samples = self.n_samples
        n_chains = self.n_chains

        if state.x.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = Timer(timing=False) if timer is None else timer
        rdtype = precision.real("calc", host=True)

        accepted = 0
        proposed = 0
        n_conn = 0

        for _ in range(self.discard):
            state, _, _, info = self._step(
                params,
                hamiltonian,
                model,
                state,
                eps1=eps1,
                timer=timer,
            )
            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn += info["n_conn"]

        observations = np.empty((n_samples, *state.x.shape[1:]), dtype=np.uint64)
        mass = np.empty(n_samples, dtype=rdtype)

        offset = 0

        while offset < n_samples:
            take = min(n_chains, n_samples - offset)
            state, x_obs, obs_mass, info = self._step(
                params,
                hamiltonian,
                model,
                state,
                eps1=eps1,
                n_observe=take,
                timer=timer,
            )

            observations[offset : offset + take] = x_obs
            mass[offset : offset + take] = obs_mass
            offset += take

            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn += info["n_conn"]

            for _ in range(self.sweep - 1):
                state, _, _, info = self._step(
                    params,
                    hamiltonian,
                    model,
                    state,
                    eps1=eps1,
                    timer=timer,
                )
                accepted += info["accepted"]
                proposed += info["proposed"]
                n_conn += info["n_conn"]

        rec = {
            "acceptance_rate": float(accepted / proposed if proposed else 0.0),
            "n_conn": n_conn,
        }

        return state, observations, mass, rec

    def _step(
        self,
        params: Any,
        hamiltonian: Any,
        model: Any,
        state: ChainState,
        *,
        eps1: float,
        n_observe: int = 0,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, np.ndarray, dict[str, int]]:
        """Observe selected chains and make one Metropolis transition."""
        timer = Timer(timing=False) if timer is None else timer
        rdtype = precision.real("calc", host=True)
        n_chain = state.x.shape[0]
        beta = self.beta

        with timer("sample"):
            key, subkey = jax.random.split(state.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)

            pick = (
                np.arange(n_chain, dtype=np.int64)
                if n_observe == n_chain
                else rng.choice(n_chain, size=n_observe, replace=False)
            )
            observed = state.x[pick]
            obs_mass = np.ones(n_observe, dtype=rdtype)

        if self.proposal == "single":
            sector = hamiltonian.sector
            move_a = sector.n_alpha * (sector.norb - sector.n_alpha)
            move_b = sector.n_beta * (sector.norb - sector.n_beta)
            n_move = move_a + move_b

            if n_move == 0:
                return (
                    replace(state, key=key),
                    observed,
                    obs_mass,
                    {"accepted": 0, "proposed": 0, "n_conn": 0},
                )

            with timer("sample"):
                item = np.arange(n_chain, dtype=np.int64)
                move = rng.integers(n_move, size=n_chain, dtype=np.int64)
                spin = (move >= move_a).astype(np.int64)
                move_spin = move - spin * move_a

                n_occ = np.where(spin == 0, sector.n_alpha, sector.n_beta)
                n_vir = sector.norb - n_occ
                occ_rank = move_spin // n_vir
                vir_rank = move_spin % n_vir

                words = state.x[item, spin]
                valid = np.full(
                    sector.nword,
                    np.iinfo(np.uint64).max,
                    dtype=np.uint64,
                )
                remainder = sector.norb & 63
                if remainder:
                    valid[-1] = (
                        np.uint64(1) << np.uint64(remainder)
                    ) - np.uint64(1)

                vir_words = (~words) & valid
                occ_prefix = np.cumulative_sum(
                    np.bitwise_count(words),
                    axis=1,
                    dtype=np.int32,
                    include_initial=True,
                )
                vir_prefix = np.cumulative_sum(
                    np.bitwise_count(vir_words),
                    axis=1,
                    dtype=np.int32,
                    include_initial=True,
                )
                occ_word = np.argmax(occ_prefix[:, 1:] > occ_rank[:, None], axis=1)
                vir_word = np.argmax(vir_prefix[:, 1:] > vir_rank[:, None], axis=1)
                occ_before = occ_prefix[item, occ_word]
                vir_before = vir_prefix[item, vir_word]

                selected = np.stack(
                    (words[item, occ_word], vir_words[item, vir_word]),
                    axis=1,
                )
                bits = np.unpackbits(
                    selected.view(np.uint8),
                    axis=1,
                    bitorder="little",
                ).reshape(n_chain, 2, 64)
                bit_prefix = np.cumsum(bits, axis=2, dtype=np.int16)

                occ_bit = np.argmax(
                    bit_prefix[:, 0] == (occ_rank - occ_before)[:, None] + 1,
                    axis=1,
                ).astype(np.uint64)
                vir_bit = np.argmax(
                    bit_prefix[:, 1] == (vir_rank - vir_before)[:, None] + 1,
                    axis=1,
                ).astype(np.uint64)

                candidate = state.x.copy()
                candidate[item, spin, occ_word] ^= np.uint64(1) << occ_bit
                candidate[item, spin, vir_word] ^= np.uint64(1) << vir_bit

            with timer("unique"):
                unique, first, inverse = sector.unique(
                    np.concatenate((state.x, candidate), axis=0)
                )

            with timer("reduce"):
                known = first < n_chain
                unique_logabs = np.empty(unique.shape[0], dtype=rdtype)
                unique_logabs[known] = state.logabs[first[known]]
                unknown = ~known

            if unknown.any():
                with timer("forward", n=np.count_nonzero(unknown)):
                    value = batch.apply(
                        model.logabs,
                        params,
                        unique[unknown],
                    )
                    unknown_logabs = precision.host(
                        value,
                        "calc",
                        "real",
                    ).reshape(-1)

                unique_logabs[unknown] = unknown_logabs

            with timer("sample"):
                candidate_logabs = unique_logabs[inverse[n_chain:]]
                log_accept = rdtype(2.0) * (
                    candidate_logabs - state.logabs
                )
                accept = np.log(rng.random(n_chain)) < np.minimum(
                    rdtype(0.0),
                    log_accept,
                )
                accepted = int(np.count_nonzero(accept))
                reject = ~accept
                candidate[reject] = state.x[reject]
                candidate_logabs[reject] = state.logabs[reject]

                next_state = replace(
                    state,
                    key=key,
                    x=candidate,
                    logabs=candidate_logabs,
                )

            return (
                next_state,
                observed,
                obs_mass,
                {
                    "accepted": accepted,
                    "proposed": n_chain,
                    "n_conn": 0,
                },
            )

        with timer("unique"):
            ket, _, ket_index = hamiltonian.sector.unique(state.x)
            n_ket = ket.shape[0]

        obs_ket = ket_index[pick] if n_observe else np.empty(0, dtype=np.int64)

        with timer("sample"):
            kernel_move = (
                rng.random(n_observe) < beta
                if n_observe > 0 and beta > 0.0
                else np.zeros(n_observe, dtype=bool)
            )

        with timer("reduce"):
            kernel_counts = np.bincount(
                obs_ket[kernel_move],
                minlength=n_ket,
            ).astype(np.int64)

        candidate = state.x.copy()
        candidate_pos = np.full(n_chain, -1, dtype=np.int64)

        with timer("reduce"):
            proposal_counts = np.bincount(
                ket_index,
                minlength=n_ket,
            ).astype(np.int64)
            counts = (
                np.stack((proposal_counts, kernel_counts))
                if kernel_move.any()
                else proposal_counts
            )

        with timer("conns"):
            conn = hamiltonian.sample_conn(
                ket,
                counts,
                eps1=np.inf,
                eps2=eps1,
                seed=rng.integers(0, 2**32, dtype=np.uint64),
            )
            conn_bra = np.asarray(conn.bra, dtype=np.uint64)
            conn_ptr = np.asarray(conn.ptr, dtype=np.int64)
            n_conn = int(conn_ptr[-1])

        with timer("sample"):
            proposal_ptr = conn_ptr[: n_ket + 1]
            take = np.diff(proposal_ptr)

            order = np.argsort(ket_index, kind="stable")
            sorted_ket = ket_index[order]
            active_order = order[take[sorted_ket] > 0]

            records = np.arange(proposal_ptr[0], proposal_ptr[-1], dtype=np.int64)
            if records.size:
                candidate_pos[active_order] = n_ket + records
                candidate[active_order] = conn_bra[candidate_pos[active_order]]

            active = candidate_pos >= 0
            # Degree tilting cancels proposal asymmetry.

        if kernel_move.any():
            with timer("sample"):
                kernel_ptr = conn_ptr[n_ket : 2 * n_ket + 1]
                take = np.diff(kernel_ptr)

                kernel_pick = np.flatnonzero(kernel_move)
                order = np.argsort(obs_ket[kernel_pick], kind="stable")
                kernel_pick = kernel_pick[order]
                sorted_ket = obs_ket[kernel_pick]
                active_pick = kernel_pick[take[sorted_ket] > 0]

                records = np.arange(kernel_ptr[0], kernel_ptr[-1], dtype=np.int64)
                if records.size:
                    observed[active_pick] = conn_bra[n_ket + records]

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            logabs_candidate = np.empty(n_chain, dtype=rdtype)
            logabs_candidate[~active] = state.logabs[~active]
            new = candidate_pos[active]

            with timer("forward", n=new.size):
                value = batch.apply(
                    model.logabs,
                    params,
                    conn_bra[new],
                )
                new_logabs = precision.host(value, "calc", "real").reshape(-1)

            with timer("reduce"):
                logabs_candidate[active] = new_logabs

            with timer("sample"):
                alpha = rdtype(state.alpha if self.alpha is None else self.alpha)
                log_accept = alpha * (logabs_candidate - state.logabs)
                accept = active & (
                    np.log(rng.random(n_chain)) < np.minimum(rdtype(0.0), log_accept)
                )
                accepted = int(np.count_nonzero(accept))

                next_state = replace(
                    state,
                    key=key,
                    x=np.where(accept[:, None, None], candidate, state.x),
                    logabs=np.where(accept, logabs_candidate, state.logabs),
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
