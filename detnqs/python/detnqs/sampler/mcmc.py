from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from .. import utils
from ..utils import precision
from .proposal import propose, unique_dets


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """Counted observed samples.

    The Markov chain state is ``x``. The observed sample is ``y`` after the
    optional observation kernel ``B_beta(y|x)``.
    """

    dets: np.ndarray
    count: np.ndarray


@dataclass(frozen=True, slots=True)
class WalkerState:
    """Counted empirical state of exchangeable Markov walkers.

    The public semantics are ``n_chains`` parallel walkers. Internally, walker
    identities are not stored because VMC estimators depend only on the current
    empirical measure.

        state = {(x_u, m_u, log|psi(x_u)|)},   sum_u m_u = n_chains.
    """

    key: jax.Array
    dets: np.ndarray
    count: np.ndarray
    logabs: np.ndarray
    accept: float = 0.0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Counted determinant-space Metropolis sampler.

    Physical target:
        pi_theta(x) proportional to |psi_theta(x)|^2.

    Markov reference:
        eta_alpha(x) proportional to |psi_theta(x)|^alpha.

    Proposal kernels:
        "ham":
            Hamiltonian heat-bath proposal over screened connected determinants.

        "single":
            Uniform single excitation within a fixed (N_alpha, N_beta) sector.

    Observation kernel:
        B_beta(y|x) = (1 - beta_x) delta_xy + beta_x K(y|x),
        K(y|x)      = |H_xy| / d_B(x).

        If d_B(x) = 0, beta_x is set to 0 and the walker stays at x.

    Step convention:
        A raw Metropolis step means one proposal attempt for each counted
        walker. ``thermal_steps`` are run after initialization/reset,
        ``discard_steps`` before each production draw, and ``sweep_steps``
        separate consecutive observation steps.
    """

    n_samples: int = 1024
    n_chains: int = 1024

    thermal_steps: int = 1024
    discard_steps: int = 0
    sweep_steps: int = 1
    reset_chains: bool = False

    alpha: float = 1.0
    proposal: str = "ham"
    proposal_eps: float = 1.0e-3

    blur: float = 0.5
    blur_kernel: str = "ham"
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
        init_method: str | Any = "hf",
    ) -> WalkerState:
        """Initialise counted walkers and run thermalization steps.

        init_method:
            "hf"     - lowest n_alpha/n_beta orbitals.
            "random" - random fixed-(n_alpha, n_beta) determinants.
            array    - user-provided determinant batch; tiled if needed.
        """
        n_chains = int(self.n_chains)
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")

        nword = int(hamiltonian.nword)
        norb = int(hamiltonian.norb)

        match init_method:
            case "hf":
                det = np.zeros((1, 2, nword), dtype=np.uint64)

                for p in range(int(n_alpha)):
                    det[0, 0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

                for p in range(int(n_beta)):
                    det[0, 1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

                dets = np.repeat(det, n_chains, axis=0)

            case "random":
                key, subkey = jax.random.split(key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                rng = np.random.default_rng(seed)

                dets = np.zeros((n_chains, 2, nword), dtype=np.uint64)

                for i in range(n_chains):
                    for spin, n_elec in enumerate((int(n_alpha), int(n_beta))):
                        for p in rng.choice(norb, size=n_elec, replace=False):
                            p = int(p)
                            dets[i, spin, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

            case str():
                raise ValueError("init_method must be 'hf', 'random', or a determinant batch")

            case _:
                dets = libdet.to_dets(init_method)

                if dets.shape[0] == 0:
                    raise ValueError("init_method determinant batch must be non-empty")

                if dets.shape[0] != n_chains:
                    reps = (n_chains + dets.shape[0] - 1) // dets.shape[0]
                    dets = np.tile(dets, (reps, 1, 1))[:n_chains]

        dets, _, inv = unique_dets(dets)
        count = np.bincount(inv, minlength=dets.shape[0]).astype(np.int64)

        logabs = precision.asarray(
            np.asarray(utils.host(utils.apply(model.logabs, theta, dets))).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        state = WalkerState(key=key, dets=dets, count=count, logabs=logabs)

        accepted = proposed = 0

        for _ in range(max(0, int(self.thermal_steps))):
            state, acc, prop, _ = self._move(theta, hamiltonian, model, state)
            accepted += acc
            proposed += prop

        return replace(state, accept=accepted / proposed if proposed else 0.0)

    def draw(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: WalkerState,
    ) -> tuple[WalkerState, SampleBatch, dict[str, float]]:
        """Draw counted observed samples and advance the walker state."""
        if int(self.n_samples) <= 0:
            raise ValueError("n_samples must be positive")

        t_draw = perf_counter()
        timing = {
            "time_graph": 0.0,
            "time_forward": 0.0,
        }

        # Parameters change during optimization. Refresh cached log|psi| on the
        # current counted support before any Metropolis ratio is formed.
        t = perf_counter()
        state = replace(
            state,
            logabs=precision.asarray(
                np.asarray(utils.host(utils.apply(model.logabs, theta, state.dets))).reshape(-1),
                "calc",
                "real",
                host=True,
            ),
        )
        timing["time_forward"] += perf_counter() - t

        accepted = proposed = n_proposal_edges = n_blur_edges = 0

        for _ in range(max(0, int(self.discard_steps))):
            state, acc, prop, n_prop = self._move(
                theta,
                hamiltonian,
                model,
                state,
                timing=timing,
            )
            accepted += acc
            proposed += prop
            n_proposal_edges += n_prop

        det_parts: list[np.ndarray] = []
        count_parts: list[np.ndarray] = []

        remaining = int(self.n_samples)
        n_chains = int(self.n_chains)
        sweep_steps = max(1, int(self.sweep_steps))

        while remaining > 0:
            take = min(n_chains, remaining)

            if take == n_chains:
                base_dets = state.dets
                base_count = state.count.astype(np.int64, copy=False)
            else:
                # A partial observation uses a hypergeometric draw from the
                # counted walker state: sample walkers without replacement.
                key, subkey = jax.random.split(state.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                rng = np.random.default_rng(seed)

                sample_count = rng.multivariate_hypergeometric(
                    state.count.astype(np.int64, copy=False),
                    take,
                ).astype(np.int64)

                keep = sample_count > 0
                base_dets = state.dets[keep]
                base_count = sample_count[keep]
                state = replace(state, key=key)

            state, obs_dets, obs_count, n_blur = self._observe(
                hamiltonian,
                state,
                base_dets,
                base_count,
                timing=timing,
            )

            det_parts.append(obs_dets)
            count_parts.append(obs_count)
            n_blur_edges += n_blur

            # Observations are separated by sweep_steps raw Metropolis moves.
            for _ in range(sweep_steps):
                state, acc, prop, n_prop = self._move(
                    theta,
                    hamiltonian,
                    model,
                    state,
                    timing=timing,
                )
                accepted += acc
                proposed += prop
                n_proposal_edges += n_prop

            remaining -= take

        obs_dets, _, inv = unique_dets(np.concatenate(det_parts, axis=0))
        raw_count = np.concatenate(count_parts).astype(np.int64, copy=False)

        obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
        np.add.at(obs_count, inv, raw_count)

        keep = obs_count > 0
        obs_dets = np.ascontiguousarray(obs_dets[keep])
        obs_count = obs_count[keep]

        accept = accepted / proposed if proposed else 0.0

        elapsed = perf_counter() - t_draw
        time_sampler = max(0.0, elapsed - timing["time_graph"] - timing["time_forward"])

        return (
            replace(state, accept=accept),
            SampleBatch(dets=obs_dets, count=obs_count),
            {
                "accept": float(accept),
                "n_proposal_edges": float(n_proposal_edges),
                "n_blur_edges": float(n_blur_edges),
                "time_sampler": float(time_sampler),
                "time_graph": float(timing["time_graph"]),
                "time_forward": float(timing["time_forward"]),
            },
        )

    def _move(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: WalkerState,
        *,
        timing: dict[str, float] | None = None,
    ) -> tuple[WalkerState, int, int, int]:
        """Advance the counted walker state by one raw Metropolis step.

        Acceptance probability:

            a(x -> y) = min(1, exp[
                alpha (log|psi(y)| - log|psi(x)|)
                + log q(x|y) - log q(y|x)
            ]).
        """
        pa = precision.asarray
        rdtype = precision.dtype("calc", "real", host=True)

        key, subkey = jax.random.split(state.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng = np.random.default_rng(seed)

        dets = state.dets
        count = state.count.astype(np.int64, copy=False)
        logabs = pa(state.logabs, "calc", "real", host=True)
        n_row = int(dets.shape[0])

        batch = propose(
            self.proposal,
            hamiltonian,
            dets,
            count,
            seed=seed,
            eps=float(self.proposal_eps),
            timing=timing,
        )

        proposed = int(np.sum(batch.count))

        if proposed == 0 or batch.dets.shape[0] == 0:
            return replace(state, key=key), 0, proposed, int(batch.n_edge)

        prop_logabs = np.empty(batch.dets.shape[0], dtype=rdtype)

        # Avoid duplicate model evaluation for proposals already in support.
        _, first, inv_lookup = unique_dets(np.concatenate([dets, batch.dets], axis=0))
        prop_first = first[inv_lookup[n_row:]]
        known = prop_first < n_row

        if known.any():
            prop_logabs[known] = logabs[prop_first[known]]

        if (~known).any():
            prop_unk = np.ascontiguousarray(batch.dets[~known])

            t = perf_counter()
            prop_logabs[~known] = pa(
                np.asarray(utils.host(utils.apply(model.logabs, theta, prop_unk))).reshape(-1),
                "calc",
                "real",
                host=True,
            )

            if timing is not None:
                timing["time_forward"] += perf_counter() - t

        log_ratio = (
            rdtype(self.alpha) * (prop_logabs[batch.dst] - logabs[batch.src])
            + pa(batch.log_qratio, "calc", "real", host=True)
        )

        accept_prob = np.clip(
            np.exp(np.minimum(rdtype(0.0), log_ratio)),
            rdtype(0.0),
            rdtype(1.0),
        )

        accepted_g = rng.binomial(batch.count, accept_prob).astype(np.int64)
        accepted = int(accepted_g.sum())

        next_count = count.copy()
        np.add.at(next_count, batch.src, -accepted_g)

        prop_count = np.zeros(batch.dets.shape[0], dtype=np.int64)
        np.add.at(prop_count, batch.dst, accepted_g)

        all_dets = np.concatenate([dets, batch.dets], axis=0)
        all_count = np.concatenate([next_count, prop_count])
        all_logabs = np.concatenate([logabs, prop_logabs])

        next_dets, first, inv = unique_dets(all_dets)

        merged_count = np.zeros(next_dets.shape[0], dtype=np.int64)
        np.add.at(merged_count, inv, all_count)

        keep = merged_count > 0

        return (
            WalkerState(
                key=key,
                dets=np.ascontiguousarray(next_dets[keep]),
                count=merged_count[keep],
                logabs=pa(all_logabs[first][keep], "calc", "real", host=True),
                accept=accepted / proposed if proposed else 0.0,
            ),
            accepted,
            proposed,
            int(batch.n_edge),
        )

    def _observe(
        self,
        hamiltonian: Any,
        state: WalkerState,
        base_dets: np.ndarray,
        base_count: np.ndarray,
        *,
        timing: dict[str, float] | None = None,
    ) -> tuple[WalkerState, np.ndarray, np.ndarray, int]:
        """Apply the observation kernel B_beta(y|x) to counted base walkers."""
        beta = float(np.clip(self.blur, 0.0, 1.0))

        if beta <= 0.0:
            return (
                state,
                np.ascontiguousarray(base_dets),
                base_count.astype(np.int64, copy=False),
                0,
            )

        if self.blur_kernel != "ham":
            raise ValueError("blur_kernel must be 'ham'")

        key, subkey = jax.random.split(state.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng = np.random.default_rng(seed)

        blur_eps = float(self.proposal_eps if self.blur_eps is None else self.blur_eps)
        base_count = base_count.astype(np.int64, copy=False)

        move = rng.binomial(base_count, beta).astype(np.int64)
        stay = base_count - move

        parts = [base_dets]
        counts = [stay]
        n_blur_edges = 0

        move_rows = np.flatnonzero(move > 0)

        if move_rows.size > 0:
            move_dets = np.ascontiguousarray(base_dets[move_rows])
            move_count = move[move_rows].astype(np.int64, copy=False)

            t = perf_counter()
            sample = hamiltonian.sample_edges(
                move_dets,
                move_count,
                eps1=np.inf,
                eps2=blur_eps,
                seed=seed,
            )
            n_blur_edges = int(np.asarray(sample.h).size)
            row_weight = precision.asarray(
                np.asarray(sample.row_weight),
                "calc",
                "real",
                host=True,
            )

            if timing is not None:
                timing["time_graph"] += perf_counter() - t

            # If K(.|x) is empty, the attempted blur move becomes a stay.
            dead = row_weight <= 0.0
            if dead.any():
                stay[move_rows[dead]] += move_count[dead]

            if n_blur_edges > 0:
                sampled_dets = np.ascontiguousarray(np.asarray(sample.dets, dtype=np.uint64))

                if sampled_dets.shape[0] > 0:
                    sampled_count = np.asarray(sample.counts, dtype=np.int64)

                    obs_dets, _, obs_inv = unique_dets(sampled_dets)
                    obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
                    np.add.at(obs_count, obs_inv, sampled_count)

                    parts.append(obs_dets)
                    counts.append(obs_count)

        obs_dets, _, inv = unique_dets(np.concatenate(parts, axis=0))
        raw_count = np.concatenate(counts).astype(np.int64, copy=False)

        obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
        np.add.at(obs_count, inv, raw_count)

        keep = obs_count > 0

        return (
            replace(state, key=key),
            np.ascontiguousarray(obs_dets[keep]),
            obs_count[keep],
            n_blur_edges,
        )