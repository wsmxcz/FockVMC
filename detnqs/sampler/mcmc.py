from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import utils
from ..hilbert import DetSector
from ..utils import precision


@dataclass(frozen=True, slots=True)
class Chains:
    """State of Markov chains in Fock space."""

    key: jax.Array
    x: np.ndarray
    logabs: np.ndarray
    alpha: float = 1.0
    alpha_step: int = 0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Metropolis sampler for Fock-space chains."""

    n_samples: int = 1024
    n_chains: int = 1024

    thermal_steps: int = 32
    discard_steps: int = 0
    sweep_steps: int = 1
    reset_chains: bool = False

    alpha: float | None = None
    proposal: str = "ham"

    blur: float = 0.5

    def __post_init__(self) -> None:
        if self.proposal not in {"ham", "single"}:
            raise ValueError("proposal must be 'ham' or 'single'")
        if not 0.0 <= float(self.blur) <= 1.0:
            raise ValueError("blur must satisfy 0 <= blur <= 1")

    def init(
        self,
        theta: Any,
        H: Any,
        model: Any,
        *,
        key: jax.Array,
        screen_eps: float,
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
            else 2.0
            if self.alpha is None
            else float(self.alpha)
        )

        key, init_key = jax.random.split(key)

        if isinstance(chain_init, str):
            if chain_init in {"hf", "reference"}:
                x = H.sector.reference(n_chains)
            elif chain_init == "random":
                seed = int(jax.random.bits(init_key, (), dtype=jnp.uint32))
                x = H.sector.random(n_chains, seed)
            else:
                raise ValueError("chain_init must be 'hf', 'reference', 'random', or x")
        else:
            x = H.sector.asarray(chain_init)
            if x.shape[0] == 0:
                raise ValueError("chain_init must be non-empty")
            if x.shape[0] != n_chains:
                reps = (n_chains + x.shape[0] - 1) // x.shape[0]
                x = np.tile(x, (reps, 1, 1))[:n_chains]
            x = np.ascontiguousarray(x)

        unique, _, inv = H.sector.unique(x)
        value = utils.apply(model.logabs, theta, unique)
        jax.block_until_ready(value)

        unique_logabs = precision.asarray(
            np.asarray(utils.host(value)).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        logabs = precision.asarray(unique_logabs[inv], "calc", "real", host=True)

        state = Chains(
            key=key,
            x=x,
            logabs=logabs,
            alpha=alpha_value,
            alpha_step=max(0, int(alpha_step)),
        )

        for _ in range(max(0, int(self.thermal_steps))):
            state, _, _, _ = self._step(theta, H, model, state, screen_eps=screen_eps)

        return state

    def draw(
        self,
        theta: Any,
        H: Any,
        model: Any,
        state: Chains,
        *,
        screen_eps: float,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, float]]:
        """Advance chains and return observed configurations.

        The caller owns logabs synchronization. This sampler only consumes the
        current chain state and performs Markov transitions plus observations.
        """
        n_samples = int(self.n_samples)
        n_chains = int(self.n_chains)

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if state.x.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)
        alpha = float(state.alpha) if self.alpha is None else float(self.alpha)
        state = replace(state, alpha=alpha)

        accepted = 0
        proposed = 0
        n_conn = 0

        for _ in range(max(0, int(self.discard_steps))):
            state, _, _, info = self._step(
                theta,
                H,
                model,
                state,
                screen_eps=screen_eps,
                timer=timer,
            )
            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn += info["n_conn"]

        samples = np.empty((n_samples, *state.x.shape[1:]), dtype=np.uint64)
        mass = np.empty(n_samples, dtype=rdtype)

        sweep_steps = max(1, int(self.sweep_steps))
        offset = 0

        while offset < n_samples:
            take = min(n_chains, n_samples - offset)
            state, x_obs, obs_mass, info = self._step(
                theta,
                H,
                model,
                state,
                screen_eps=screen_eps,
                n_observe=take,
                timer=timer,
            )

            samples[offset : offset + take] = x_obs
            mass[offset : offset + take] = obs_mass
            offset += take

            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn += info["n_conn"]

            for _ in range(sweep_steps - 1):
                state, _, _, info = self._step(
                    theta,
                    H,
                    model,
                    state,
                    screen_eps=screen_eps,
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
        H: Any,
        model: Any,
        state: Chains,
        *,
        screen_eps: float,
        n_observe: int = 0,
        timer: utils.Timer | None = None,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, int]]:
        """Observe selected chains and make one Metropolis transition."""
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.x.shape[0])
        n_observe = int(n_observe)
        beta = float(self.blur)

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

            ket, first, ket_index = H.sector.unique(state.x)
            n_ket = int(ket.shape[0])
            ket_logabs = precision.asarray(
                state.logabs[np.asarray(first, dtype=np.int64)],
                "calc",
                "real",
                host=True,
            )

            obs_ket = ket_index[pick] if n_observe else np.empty(0, dtype=np.int64)
            blur = (
                rng.random(n_observe) < beta
                if n_observe > 0 and beta > 0.0
                else np.zeros(n_observe, dtype=bool)
            )
            blur_counts = np.bincount(
                obs_ket[blur],
                minlength=n_ket,
            ).astype(np.int64)

        candidate = np.ascontiguousarray(state.x.copy())
        active = np.zeros(n_chain, dtype=bool)
        candidate_pos = np.full(n_chain, -1, dtype=np.int64)
        log_q = np.zeros(n_chain, dtype=rdtype)

        conn_x = None
        conn_ptr = None
        conn_bra = None
        conn_weight = None
        n_conn = 0

        if self.proposal == "ham":
            with timer("sample"):
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
                conn = H.sample_conn(
                    ket,
                    np.ascontiguousarray(counts),
                    eps1=np.inf,
                    eps2=float(screen_eps),
                    seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                    bra_weight=True,
                )
                conn_x = np.asarray(conn.x, dtype=np.uint64)
                conn_ptr = np.asarray(conn.ptr, dtype=np.int64)
                conn_bra = np.asarray(conn.bra, dtype=np.int64)
                conn_weight = precision.asarray(
                    np.asarray(conn.weight),
                    "calc",
                    "real",
                    host=True,
                )
                n_conn = int(np.asarray(conn.h).size)

            with timer("sample"):
                proposal_ptr = conn_ptr[: n_ket + 1]
                take = np.diff(proposal_ptr)

                order = np.argsort(ket_index, kind="stable")
                sorted_ket = ket_index[order]
                active_order = order[take[sorted_ket] > 0]

                records = np.arange(proposal_ptr[0], proposal_ptr[-1], dtype=np.int64)
                if records.size:
                    candidate[active_order] = conn_x[conn_bra[records]]
                    candidate_pos[active_order] = conn_bra[records]

                active = candidate_pos >= 0
                if active.any():
                    tiny = rdtype(precision.tiny("calc"))
                    ket_w = conn_weight[ket_index[active]]
                    bra_w = conn_weight[candidate_pos[active]]
                    log_q[active] = (
                        np.log(np.maximum(ket_w, tiny))
                        - np.log(np.maximum(bra_w, tiny))
                    )

        else:
            with timer("sample"):
                sector = H.sector
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
                    conn = H.sample_conn(
                        ket,
                        np.ascontiguousarray(blur_counts),
                        eps1=np.inf,
                        eps2=float(screen_eps),
                        seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                    )
                    conn_x = np.asarray(conn.x, dtype=np.uint64)
                    conn_ptr = np.asarray(conn.ptr, dtype=np.int64)
                    conn_bra = np.asarray(conn.bra, dtype=np.int64)
                    conn_weight = precision.asarray(
                        np.asarray(conn.weight),
                        "calc",
                        "real",
                        host=True,
                    )
                    n_conn += int(np.asarray(conn.h).size)

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
                        observed[active_pick] = conn_x[conn_bra[records]]

                obs_weight = conn_weight[obs_ket]
                obs_mass = np.where(
                    obs_weight > 0.0,
                    obs_weight,
                    rdtype(1.0),
                )

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            logabs_candidate = np.empty(n_chain, dtype=rdtype)
            logabs_candidate[~active] = state.logabs[~active]

            if self.proposal == "ham":
                pool_logabs = np.empty(conn_x.shape[0], dtype=rdtype)
                pool_logabs[:n_ket] = ket_logabs

                needed = np.unique(candidate_pos[active])
                new = needed[needed >= n_ket]
                if new.size:
                    with timer("forward"):
                        value = utils.apply(
                            model.logabs,
                            theta,
                            np.ascontiguousarray(conn_x[new]),
                        )
                        jax.block_until_ready(value)

                    pool_logabs[new] = precision.asarray(
                        np.asarray(utils.host(value)).reshape(-1),
                        "calc",
                        "real",
                        host=True,
                    )

                logabs_candidate[active] = pool_logabs[candidate_pos[active]]

            else:
                unique_candidate, _, inverse = H.sector.unique(candidate[active])
                _, first_all, lookup = H.sector.unique(
                    np.concatenate((ket, unique_candidate), axis=0)
                )

                first_candidate = first_all[lookup[n_ket:]]
                known = first_candidate < n_ket

                unique_logabs = np.empty(unique_candidate.shape[0], dtype=rdtype)
                if known.any():
                    unique_logabs[known] = ket_logabs[first_candidate[known]]

                unknown = ~known
                if unknown.any():
                    with timer("forward"):
                        value = utils.apply(
                            model.logabs,
                            theta,
                            np.ascontiguousarray(unique_candidate[unknown]),
                        )
                        jax.block_until_ready(value)

                    unique_logabs[unknown] = precision.asarray(
                        np.asarray(utils.host(value)).reshape(-1),
                        "calc",
                        "real",
                        host=True,
                    )

                logabs_candidate[active] = unique_logabs[inverse]

            with timer("sample"):
                log_accept = (
                    rdtype(state.alpha) * (logabs_candidate - state.logabs)
                    + log_q
                )
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
                    logabs=precision.asarray(
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
