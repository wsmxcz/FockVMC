from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..hilbert import DetSector
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
    """Metropolis sampler for Fock-space chains.

    Hamiltonian proposals target the degree-tilted law
    ``rho_alpha(x) s(x)``, so no reverse ket degree is needed.
    """

    n_samples: int = 1024
    n_chains: int = 1024

    thermal_steps: int = 32
    discard_steps: int = 0
    sweep_steps: int = 1
    reset_chains: bool = False

    alpha: float | None = 1.0
    proposal: str = "ham"

    blur: float = 0.5

    def __post_init__(self) -> None:
        n_samples = int(self.n_samples)
        n_chains = int(self.n_chains)
        thermal_steps = int(self.thermal_steps)
        discard_steps = int(self.discard_steps)
        sweep_steps = int(self.sweep_steps)
        proposal = str(self.proposal)
        reset_chains = bool(self.reset_chains)
        blur = float(self.blur)

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")
        if thermal_steps < 0 or discard_steps < 0:
            raise ValueError("thermal_steps and discard_steps must be nonnegative")
        if sweep_steps <= 0:
            raise ValueError("sweep_steps must be positive")
        if proposal not in {"ham", "single"}:
            raise ValueError("proposal must be 'ham' or 'single'")
        if not 0.0 <= blur <= 1.0:
            raise ValueError("blur must satisfy 0 <= blur <= 1")

        alpha = None
        if self.alpha is not None:
            alpha = float(self.alpha)
            if not np.isfinite(alpha) or not 0.0 <= alpha <= 2.0:
                raise ValueError("alpha must be None or satisfy 0 <= alpha <= 2")

        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "n_chains", n_chains)
        object.__setattr__(self, "thermal_steps", thermal_steps)
        object.__setattr__(self, "discard_steps", discard_steps)
        object.__setattr__(self, "sweep_steps", sweep_steps)
        object.__setattr__(self, "reset_chains", reset_chains)
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "blur", blur)
        object.__setattr__(self, "alpha", alpha)

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
        timer = Timer(enabled=False) if timer is None else timer
        n_chains = self.n_chains

        key, _ = jax.random.split(key)
        x = hamiltonian.sector.asarray(chains)
        if x.shape[0] != n_chains:
            raise ValueError("chains size must equal sampler.n_chains")
        x = np.ascontiguousarray(x)

        with timer("unique"):
            unique, _, inv = hamiltonian.sector.unique(x)

        with timer("forward"):
            value = batch.apply(model.logabs, params, unique)
            jax.block_until_ready(value)
            unique_logabs = precision.host(value, "calc", "real").reshape(-1)

        with timer("reduce"):
            logabs = precision.cast(unique_logabs[inv], "calc", "real", host=True)

        if self.alpha is None:
            alpha0 = 2.0 if alpha is None else float(alpha)
        else:
            alpha0 = self.alpha

        state = ChainState(
            key=key,
            x=x,
            logabs=logabs,
            alpha=float(np.clip(alpha0, 0.0, 2.0)),
        )

        for _ in range(self.thermal_steps):
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
        profile: bool = False,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, np.ndarray, dict[str, float]]:
        """Advance chains and return observed configurations.

        The caller owns logabs synchronization. The sampler only advances
        chains and reports proposal health; estimator weights are built in
        vstate.
        """
        n_samples = self.n_samples
        n_chains = self.n_chains

        if state.x.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = Timer(enabled=profile) if timer is None else timer
        rdtype = precision.real("calc", host=True)

        accepted = 0
        proposed = 0
        n_conn = 0

        for _ in range(self.discard_steps):
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

        sweep_steps = self.sweep_steps
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

            for _ in range(sweep_steps - 1):
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

        stats = {"accept": float(accepted / proposed if proposed else 0.0)}
        if profile:
            stats["n_conn"] = float(n_conn)

        return state, observations, mass, stats

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
        timer = Timer(enabled=False) if timer is None else timer
        rdtype = precision.real("calc", host=True)
        n_chain = int(state.x.shape[0])
        n_observe = int(n_observe)
        beta = self.blur

        with timer("sample"):
            key, random_key = jax.random.split(state.key)
            seed = int(jax.random.bits(random_key, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)

            pick = (
                np.arange(n_chain, dtype=np.int64)
                if n_observe == n_chain
                else rng.choice(n_chain, size=n_observe, replace=False)
            )
            observed = np.ascontiguousarray(state.x[pick])
            obs_mass = np.ones(n_observe, dtype=rdtype)

        with timer("unique"):
            ket, first, ket_index = hamiltonian.sector.unique(state.x)
            n_ket = int(ket.shape[0])

        with timer("reduce"):
            ket_logabs = precision.cast(
                state.logabs[np.asarray(first, dtype=np.int64)],
                "calc",
                "real",
                host=True,
            )
            obs_ket = ket_index[pick] if n_observe else np.empty(0, dtype=np.int64)

        with timer("sample"):
            blur = (
                rng.random(n_observe) < beta
                if n_observe > 0 and beta > 0.0
                else np.zeros(n_observe, dtype=bool)
            )

        with timer("reduce"):
            blur_counts = np.bincount(
                obs_ket[blur],
                minlength=n_ket,
            ).astype(np.int64)

        candidate = np.ascontiguousarray(state.x.copy())
        active = np.zeros(n_chain, dtype=bool)
        candidate_pos = np.full(n_chain, -1, dtype=np.int64)

        conn_bra = None
        conn_ptr = None
        conn_degree = None
        n_conn = 0

        if self.proposal == "ham":
            with timer("reduce"):
                proposal_counts = np.bincount(
                    ket_index,
                    minlength=n_ket,
                ).astype(np.int64)
                counts = (
                    np.stack((proposal_counts, blur_counts))
                    if blur.any()
                    else proposal_counts
                )

            with timer("conns"):
                conn = hamiltonian.sample_conn(
                    ket,
                    counts,
                    eps1=np.inf,
                    eps2=float(eps1),
                    seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
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
                # Degree-tilted target cancels the heat-bath ket degree.

        else:
            with timer("sample"):
                sector = hamiltonian.sector
                if not isinstance(sector, DetSector):
                    raise NotImplementedError(
                        "proposal='single' is defined only for DetSector"
                    )

                n_move_a = sector.n_alpha * (sector.norb - sector.n_alpha)
                n_move_b = sector.n_beta * (sector.norb - sector.n_beta)
                n_move = n_move_a + n_move_b

                if n_move > 0:
                    orb = np.arange(sector.norb, dtype=np.int64)
                    word = orb >> 6
                    bit = (orb & 63).astype(np.uint64)
                    occ = ((state.x[:, :, word] >> bit) & np.uint64(1)).astype(bool)

                    move = rng.integers(n_move, size=n_chain, dtype=np.int64)
                    spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
                    move_spin = np.where(spin == 0, move, move - n_move_a)

                    n_occ = np.where(spin == 0, sector.n_alpha, sector.n_beta)
                    n_vir = sector.norb - n_occ
                    occ_rank = move_spin // n_vir
                    vir_rank = move_spin % n_vir

                    chain_occ = occ[np.arange(n_chain), spin]
                    chain_vir = ~chain_occ
                    occ_pos = np.cumsum(chain_occ, axis=1) - 1
                    vir_pos = np.cumsum(chain_vir, axis=1) - 1

                    occ_orb = np.argmax(
                        chain_occ & (occ_pos == occ_rank[:, None]),
                        axis=1,
                    )
                    vir_orb = np.argmax(
                        chain_vir & (vir_pos == vir_rank[:, None]),
                        axis=1,
                    )
                    item = np.arange(n_chain, dtype=np.int64)

                    occ_word = occ_orb >> 6
                    occ_bit = (occ_orb & 63).astype(np.uint64)
                    vir_word = vir_orb >> 6
                    vir_bit = (vir_orb & 63).astype(np.uint64)

                    candidate[item, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
                    candidate[item, spin, vir_word] |= np.uint64(1) << vir_bit
                    active[:] = True

        if n_observe > 0 and beta > 0.0:
            if self.proposal != "ham":
                with timer("conns"):
                    conn = hamiltonian.sample_conn(
                        ket,
                        blur_counts,
                        eps1=np.inf,
                        eps2=float(eps1),
                        seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                    )
                    conn_bra = np.asarray(conn.bra, dtype=np.uint64)
                    conn_ptr = np.asarray(conn.ptr, dtype=np.int64)
                    conn_degree = precision.cast(
                        np.asarray(conn.degree),
                        "calc",
                        "real",
                        host=True,
                    )
                    n_conn += int(conn_ptr[-1])

            with timer("sample"):
                if blur.any():
                    offset = n_ket if self.proposal == "ham" else 0
                    blur_ptr = conn_ptr[offset : offset + n_ket + 1]
                    take = np.diff(blur_ptr)

                    blur_pick = np.flatnonzero(blur)
                    order = np.argsort(obs_ket[blur_pick], kind="stable")
                    blur_pick = blur_pick[order]
                    sorted_ket = obs_ket[blur_pick]
                    active_pick = blur_pick[take[sorted_ket] > 0]

                    records = np.arange(blur_ptr[0], blur_ptr[-1], dtype=np.int64)
                    if records.size:
                        observed[active_pick] = conn_bra[n_ket + records]

                if self.proposal != "ham":
                    # Single-move chains need the blur degree as sample mass.
                    obs_degree = conn_degree[obs_ket]
                    obs_mass = np.where(
                        obs_degree > 0.0,
                        obs_degree,
                        rdtype(1.0),
                    )

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            logabs_candidate = np.empty(n_chain, dtype=rdtype)
            logabs_candidate[~active] = state.logabs[~active]

            if self.proposal == "ham":
                with timer("reduce"):
                    new = candidate_pos[active]

                with timer("forward"):
                    value = batch.apply(
                        model.logabs,
                        params,
                        np.ascontiguousarray(conn_bra[new]),
                    )
                    jax.block_until_ready(value)
                    new_logabs = precision.host(value, "calc", "real").reshape(-1)

                with timer("reduce"):
                    logabs_candidate[active] = new_logabs

            else:
                with timer("unique"):
                    unique_candidate, _, inverse = hamiltonian.sector.unique(
                        candidate[active]
                    )
                    _, first_all, lookup = hamiltonian.sector.unique(
                        np.concatenate((ket, unique_candidate), axis=0)
                    )

                with timer("reduce"):
                    first_candidate = first_all[lookup[n_ket:]]
                    known = first_candidate < n_ket

                    unique_logabs = np.empty(unique_candidate.shape[0], dtype=rdtype)
                    if known.any():
                        unique_logabs[known] = ket_logabs[first_candidate[known]]

                    unknown = ~known

                if unknown.any():
                    with timer("forward"):
                        value = batch.apply(
                            model.logabs,
                            params,
                            np.ascontiguousarray(unique_candidate[unknown]),
                        )
                        jax.block_until_ready(value)
                        unknown_logabs = precision.host(
                            value,
                            "calc",
                            "real",
                        ).reshape(-1)

                    with timer("reduce"):
                        unique_logabs[unknown] = unknown_logabs

                with timer("reduce"):
                    logabs_candidate[active] = unique_logabs[inverse]

            with timer("sample"):
                # Fixed alpha uses sampler.alpha; auto alpha uses chain state.
                alpha = rdtype(state.alpha if self.alpha is None else self.alpha)
                log_accept = alpha * (logabs_candidate - state.logabs)
                accept = active & (
                    np.log(rng.random(n_chain)) < np.minimum(rdtype(0.0), log_accept)
                )
                accepted = int(np.count_nonzero(accept))

                next_state = replace(
                    state,
                    key=key,
                    x=np.ascontiguousarray(
                        np.where(accept.reshape((-1, 1, 1)), candidate, state.x)
                    ),
                    logabs=precision.cast(
                        np.where(accept, logabs_candidate, state.logabs),
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
