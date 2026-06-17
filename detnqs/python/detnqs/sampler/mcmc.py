from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import utils
from ..utils import precision
from .proposal import HeatBath


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
    proposal: Any = field(default_factory=HeatBath)

    blur: float = 0.5
    blur_eps: float | None = None

    @property
    def eps(self) -> float:
        return float(getattr(self.proposal, "eps", 0.0))

    def init(
        self,
        theta: Any,
        H: Any,
        model: Any,
        *,
        key: jax.Array,
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
                x = H.space.reference(n_chains)
            elif chain_init == "random":
                seed = int(jax.random.bits(init_key, (), dtype=jnp.uint32))
                x = H.space.random(n_chains, seed)
            else:
                raise ValueError("chain_init must be 'hf', 'reference', 'random', or x")
        else:
            x = H.space.asarray(chain_init)
            if x.shape[0] == 0:
                raise ValueError("chain_init must be non-empty")
            if x.shape[0] != n_chains:
                reps = (n_chains + x.shape[0] - 1) // x.shape[0]
                x = np.tile(x, (reps, 1, 1))[:n_chains]
            x = np.ascontiguousarray(x)

        unique, _, inv = H.space.unique(x)
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
            state, _, _, _ = self._step(theta, H, model, state)

        return state

    def draw(
        self,
        theta: Any,
        H: Any,
        model: Any,
        state: Chains,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, float]]:
        """Advance chains and return observed configurations."""
        n_samples = int(self.n_samples)
        n_chains = int(self.n_chains)

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if state.x.shape[0] != n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)
        alpha = float(state.alpha) if self.alpha is None else float(self.alpha)

        with timer("forward"):
            unique, _, inv = H.space.unique(state.x)
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
                logabs=precision.asarray(unique_logabs[inv], "calc", "real", host=True),
                alpha=alpha,
            )

        accepted = 0
        proposed = 0
        n_conn = 0

        for _ in range(max(0, int(self.discard_steps))):
            state, _, _, info = self._step(theta, H, model, state, timer=timer)
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
                state, _, _, info = self._step(theta, H, model, state, timer=timer)
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
        n_observe: int = 0,
        timer: utils.Timer | None = None,
    ) -> tuple[Chains, np.ndarray, np.ndarray, dict[str, int]]:
        """Observe selected chains and make one Metropolis transition."""
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)
        n_chain = int(state.x.shape[0])
        n_observe = int(n_observe)

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

        n_conn = 0
        if n_observe > 0 and float(self.blur) > 0.0:
            eps = float(self.blur_eps if self.blur_eps is not None else self.eps)
            observed, obs_mass, n_conn = self._blur(
                H,
                observed,
                beta=float(self.blur),
                eps=eps,
                rng=rng,
                timer=timer,
            )

        bra, active, log_q, prop_info = self.proposal(
            theta,
            H,
            model,
            state,
            rng=rng,
            timer=timer,
        )
        n_conn += int(prop_info.get("n_conn", 0))

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            logabs_bra = self._candidate_logabs(
                theta,
                H,
                model,
                state,
                bra,
                active,
                timer,
            )

            with timer("sample"):
                log_accept = rdtype(state.alpha) * (logabs_bra - state.logabs) + log_q
                accept = active & (
                    np.log(rng.random(n_chain)) < np.minimum(rdtype(0.0), log_accept)
                )
                accepted = int(np.count_nonzero(accept))

                next_state = replace(
                    state,
                    key=key,
                    x=np.ascontiguousarray(
                        np.where(accept.reshape((-1, 1, 1)), bra, state.x)
                    ),
                    logabs=precision.asarray(
                        np.where(accept, logabs_bra, state.logabs),
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

    def _candidate_logabs(
        self,
        theta: Any,
        H: Any,
        model: Any,
        state: Chains,
        bra: np.ndarray,
        active: np.ndarray,
        timer: utils.Timer,
    ) -> np.ndarray:
        """Evaluate log|psi| on active candidate bras."""
        rdtype = precision.dtype("calc", "real", host=True)
        out = np.empty(state.x.shape[0], dtype=rdtype)
        out[~active] = state.logabs[~active]

        unique_bra, _, inverse = H.space.unique(bra[active])
        _, first, lookup = H.space.unique(np.concatenate((state.x, unique_bra), axis=0))
        first_bra = first[lookup[state.x.shape[0] :]]
        known = first_bra < state.x.shape[0]
        unique_logabs = np.empty(unique_bra.shape[0], dtype=rdtype)

        if known.any():
            unique_logabs[known] = state.logabs[first_bra[known]]

        if (~known).any():
            with timer("forward"):
                value = utils.apply(model.logabs, theta, np.ascontiguousarray(unique_bra[~known]))
                jax.block_until_ready(value)

            unique_logabs[~known] = precision.asarray(
                np.asarray(utils.host(value)).reshape(-1),
                "calc",
                "real",
                host=True,
            )

        out[active] = unique_logabs[inverse]
        return out

    def _blur(
        self,
        H: Any,
        x: np.ndarray,
        *,
        beta: float,
        eps: float,
        rng: np.random.Generator,
        timer: utils.Timer,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Apply Hamiltonian blur to observed configurations."""
        rdtype = precision.dtype("calc", "real", host=True)
        n = int(x.shape[0])
        if n == 0 or beta <= 0.0:
            return x, np.ones(n, dtype=rdtype), 0

        with timer("sample"):
            ket, _, ket_index = H.space.unique(x)
            blur = rng.random(n) < beta
            counts = np.bincount(ket_index[blur], minlength=ket.shape[0]).astype(np.int64)

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
            conn_x = np.asarray(conn.x, dtype=np.uint64)
            conn_ket_ptr = np.asarray(conn.ket_ptr, dtype=np.int64)
            conn_bra_idx = np.asarray(conn.bra_idx, dtype=np.int64)
            conn_count = np.asarray(conn.count, dtype=np.int64)

        with timer("sample"):
            out = np.ascontiguousarray(x.copy())
            pick = np.flatnonzero(blur)
            if pick.size:
                order = np.argsort(ket_index[pick], kind="stable")
                pick = pick[order]
                sorted_ket = ket_index[pick]
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
                    records = np.arange(conn_ket_ptr[k], conn_ket_ptr[k + 1])
                    if records.size == 0:
                        continue

                    hits = np.repeat(records, conn_count[records])
                    rng.shuffle(hits)
                    item = pick[begin:end]
                    out[item] = conn_x[conn_bra_idx[hits]]

            mass = np.where(weight[ket_index] > 0.0, weight[ket_index], rdtype(1.0))

        return out, mass, n_conn
