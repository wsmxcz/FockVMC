from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..utils import Timer, batch, precision
from .base import ChainState


@dataclass(frozen=True, slots=True)
class HamSampler:
    """Metropolis sampler with Hamiltonian proposals."""

    n_samples: int = 1024
    n_chains: int = 1024
    thermal_steps: int = 32
    discard_steps: int = 1
    eps1: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive")
        if self.thermal_steps < 0:
            raise ValueError("thermal_steps must be nonnegative")
        if self.discard_steps <= 0:
            raise ValueError("discard_steps must be positive")
        if not np.isfinite(self.eps1) or self.eps1 < 0.0:
            raise ValueError("eps1 must be finite and nonnegative")

    def init(
        self,
        params: Any,
        model: Any,
        hamiltonian: Any,
        *,
        chains: Any,
        key: jax.Array,
        alpha: float,
        timer: Timer | None = None,
    ) -> ChainState:
        """Initialize chains and thermalize them."""
        timer = Timer(timing=False) if timer is None else timer
        x = hamiltonian.sector.asarray(chains)
        if x.shape[0] != self.n_chains:
            raise ValueError("chains size must equal sampler.n_chains")

        with timer("unique"):
            unique, _, index = hamiltonian.sector.unique(x)
        with timer("forward", n=unique.shape[0]):
            value = batch.apply(model.logabs, params, unique)
            logabs = precision.host(value, "calc", "real").reshape(-1)[index]

        chain = ChainState(key=key, x=x, logabs=logabs)
        for _ in range(self.thermal_steps):
            chain, _, _ = self.step(
                params,
                model,
                hamiltonian,
                chain,
                alpha=alpha,
                timer=timer,
            )
        return chain

    def draw(
        self,
        params: Any,
        model: Any,
        hamiltonian: Any,
        chain: ChainState,
        *,
        alpha: float,
        beta: float,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, dict[str, float]]:
        """Draw kernel observations and advance all chains."""
        if chain.x.shape[0] != self.n_chains:
            raise ValueError("chain state size must equal n_chains")

        timer = Timer(timing=False) if timer is None else timer
        samples = np.empty(
            (self.n_samples, *chain.x.shape[1:]),
            dtype=np.uint64,
        )
        accepted = 0
        proposed = 0
        offset = 0

        while offset < self.n_samples:
            take = min(self.n_chains, self.n_samples - offset)
            key, subkey = jax.random.split(chain.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)
            pick = (
                np.arange(self.n_chains, dtype=np.int64)
                if take == self.n_chains
                else rng.choice(self.n_chains, size=take, replace=False)
            )
            observe = rng.random(take) < beta
            parent = chain.x[pick]
            chain = replace(chain, key=key)
            chain, proposal, info = self.step(
                params,
                model,
                hamiltonian,
                chain,
                alpha=alpha,
                timer=timer,
            )
            samples[offset : offset + take] = np.where(
                observe[:, None, None],
                proposal[pick],
                parent,
            )
            offset += take
            accepted += info["accepted"]
            proposed += info["proposed"]

            for _ in range(self.discard_steps - 1):
                chain, _, info = self.step(
                    params,
                    model,
                    hamiltonian,
                    chain,
                    alpha=alpha,
                    timer=timer,
                )
                accepted += info["accepted"]
                proposed += info["proposed"]

        rec = {
            "acceptance_rate": float(accepted / proposed if proposed else 0.0),
        }
        return chain, samples, rec

    def step(
        self,
        params: Any,
        model: Any,
        hamiltonian: Any,
        chain: ChainState,
        *,
        alpha: float,
        timer: Timer | None = None,
    ) -> tuple[ChainState, np.ndarray, dict[str, int]]:
        """Generate one proposal and make one Metropolis transition."""
        timer = Timer(timing=False) if timer is None else timer
        dtype = precision.real("calc", host=True)
        n_chain = chain.x.shape[0]
        sector = hamiltonian.sector

        key, subkey = jax.random.split(chain.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng = np.random.default_rng(seed)

        with timer("unique"):
            ket, _, index = sector.unique(chain.x)
        with timer("reduce"):
            count = np.bincount(index, minlength=ket.shape[0]).astype(np.int64)
        with timer("conns"):
            conn = hamiltonian.sample_conn(
                ket,
                count,
                eps1=np.inf,
                eps2=self.eps1,
                seed=rng.integers(0, 2**32, dtype=np.uint64),
            )

        with timer("sample"):
            degree = np.asarray(conn.degree)
            order = np.argsort(index, kind="stable")
            active = degree[index[order]] > 0.0
            move = order[active]
            records = np.arange(conn.ptr[-1], dtype=np.int64)
            proposal = chain.x.copy()
            proposal[move] = np.asarray(conn.bra)[ket.shape[0] + records]

        proposed = int(move.size)
        if proposed == 0:
            return (
                replace(chain, key=key),
                proposal,
                {"accepted": 0, "proposed": 0},
            )

        with timer("unique"):
            unique, first, inverse = sector.unique(
                np.concatenate((chain.x, proposal), axis=0)
            )
        with timer("reduce"):
            known = first < n_chain
            logabs = np.empty(unique.shape[0], dtype=dtype)
            logabs[known] = chain.logabs[first[known]]
            unknown = ~known
        if unknown.any():
            with timer("forward", n=np.count_nonzero(unknown)):
                value = batch.apply(model.logabs, params, unique[unknown])
                logabs[unknown] = precision.host(
                    value,
                    "calc",
                    "real",
                ).reshape(-1)

        with timer("sample"):
            new_logabs = logabs[inverse[n_chain:]]
            log_accept = dtype(alpha) * (new_logabs - chain.logabs)
            accept = np.zeros(n_chain, dtype=bool)
            accept[move] = np.log(rng.random(proposed)) < np.minimum(
                dtype(0.0), log_accept[move]
            )
            accepted = int(np.count_nonzero(accept))

        next_chain = ChainState(
            key=key,
            x=np.where(accept[:, None, None], proposal, chain.x),
            logabs=np.where(accept, new_logabs, chain.logabs),
        )
        return (
            next_chain,
            proposal,
            {
                "accepted": accepted,
                "proposed": proposed,
            },
        )
