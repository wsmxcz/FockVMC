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
    """Compressed observed samples."""
    dets: np.ndarray   # (U, 2, nword) observed determinants y
    count: np.ndarray  # (U,) multiplicity of each


@dataclass(frozen=True, slots=True)
class SamplerState:
    """Compressed Markov-chain state.

    chain:   unique chain determinants x.
    count:   walker multiplicity per unique determinant.
    logabs:  cached log|psi_theta(x)|, refreshed at the start of each draw.
    accept:  acceptance rate from the last draw or burn-in.
    """
    key: jax.Array
    chain: np.ndarray
    count: np.ndarray
    logabs: np.ndarray
    accept: float = 0.0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Compressed determinant-space Metropolis sampler.

    Physical target:
        pi(x)  proportional to |psi(x)|^2

    Markov reference:
        eta_alpha(x)  proportional to |psi(x)|^alpha

    Proposals:
        ham:
            Hamiltonian-weighted screened proposal.

        single:
            Uniform single excitation in the fixed (N_alpha, N_beta) sector.

    Observation kernel:
        B_beta(y|x) = (1 - beta) delta_xy + beta K(y|x)
        K(y|x)      = |H_xy| / d_B(x)

    burn_in:
        raw Metropolis moves run after initialization or reset.

    n_discard:
        raw Metropolis moves discarded before each production draw.

    sweep_size:
        raw Metropolis moves between two observation steps.
    """

    n_sample:      int        = 1024
    n_chain:       int        = 1024
    burn_in:       int        = 1024
    n_discard:     int        = 0
    sweep_size:    int        = 1
    reset_chains:  bool       = False
    alpha:         float      = 1.0
    proposal:      str        = "ham"
    proposal_eps:  float      = 1.0e-3
    blur:          float      = 0.5
    blur_kernel:   str        = "ham"
    blur_eps:      float | None = 1.0e-3

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
    ) -> SamplerState:
        """Initialise chains and run burn-in.

        init_method:
            "hf"     - lowest n_alpha/n_beta orbitals (Hartree-Fock).
            "random" - random fixed-(n_alpha, n_beta) determinants.
            array    - user-provided batch shaped (N, 2, nword); tiled if N != n_chain.
        """
        n_chain = int(self.n_chain)
        nword   = int(hamiltonian.nword)
        norb    = int(hamiltonian.norb)

        match init_method:
            case "hf":
                det = np.zeros((1, 2, nword), dtype=np.uint64)
                for p in range(int(n_alpha)):
                    det[0, 0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
                for p in range(int(n_beta)):
                    det[0, 1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)
                chain = np.repeat(det, n_chain, axis=0)

            case "random":
                key, subkey = jax.random.split(key)
                rng   = np.random.default_rng(int(jax.random.bits(subkey, (), dtype=jnp.uint32)))
                chain = np.zeros((n_chain, 2, nword), dtype=np.uint64)
                for i in range(n_chain):
                    for spin, n_elec in enumerate((int(n_alpha), int(n_beta))):
                        for p in rng.choice(norb, size=n_elec, replace=False):
                            p = int(p)
                            chain[i, spin, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

            case str():
                raise ValueError("init_method must be 'hf', 'random', or a determinant batch")

            case _:
                chain = libdet.to_dets(init_method)
                if chain.shape[0] != n_chain:
                    reps  = (n_chain + chain.shape[0] - 1) // chain.shape[0]
                    chain = np.tile(chain, (reps, 1, 1))[:n_chain]

        chain, _, inv = unique_dets(chain)
        count  = np.bincount(inv, minlength=chain.shape[0]).astype(np.int64)
        logabs = precision.asarray(
            np.asarray(utils.host(utils.apply(model.logabs, theta, chain))).reshape(-1),
            "calc", "real", host=True,
        )

        state = SamplerState(key=key, chain=chain, count=count, logabs=logabs)

        accepted = proposed = 0
        for _ in range(max(0, int(self.burn_in))):
            state, acc, prop, _ = self._move(theta, hamiltonian, model, state)
            accepted += acc
            proposed += prop

        return replace(state, accept=accepted / proposed if proposed else 0.0)

    def draw(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: SamplerState,
    ) -> tuple[SamplerState, SampleBatch, dict[str, float]]:
        """Draw compressed observed samples and advance the Markov chain."""
        t_draw = perf_counter()
        timing = {
            "time_graph":   0.0,
            "time_forward": 0.0,
        }

        # Refresh logabs cache for updated parameters.
        t = perf_counter()
        state = replace(state, logabs=precision.asarray(
            np.asarray(utils.host(utils.apply(model.logabs, theta, state.chain))).reshape(-1),
            "calc", "real", host=True,
        ))
        timing["time_forward"] += perf_counter() - t

        accepted = proposed = n_prop_edge = n_blur_edge = 0

        for _ in range(max(0, int(self.n_discard))):
            state, acc, prop, n_prop = self._move(
                theta, hamiltonian, model, state, timing=timing,
            )
            accepted    += acc
            proposed    += prop
            n_prop_edge += n_prop

        det_parts:   list[np.ndarray] = []
        count_parts: list[np.ndarray] = []
        remaining = int(self.n_sample)
        sweep_size = max(1, int(self.sweep_size))

        while remaining > 0:
            take = min(int(self.n_chain), remaining)

            if take == int(self.n_chain):
                base       = state.chain
                base_count = state.count.astype(np.int64, copy=False)
            else:
                key, subkey = jax.random.split(state.key)
                rng = np.random.default_rng(int(jax.random.bits(subkey, (), dtype=jnp.uint32)))
                sample_count = rng.multivariate_hypergeometric(
                    state.count.astype(np.int64, copy=False), take,
                ).astype(np.int64)
                keep       = sample_count > 0
                base       = state.chain[keep]
                base_count = sample_count[keep]
                state      = replace(state, key=key)

            state, obs, obs_count, n_blur = self._observe(
                hamiltonian, state, base, base_count, timing=timing,
            )
            det_parts.append(obs)
            count_parts.append(obs_count)
            n_blur_edge += n_blur

            for _ in range(sweep_size):
                state, acc, prop, n_prop = self._move(
                    theta, hamiltonian, model, state, timing=timing,
                )
                accepted    += acc
                proposed    += prop
                n_prop_edge += n_prop

            remaining -= take

        obs, _, inv = unique_dets(np.concatenate(det_parts, axis=0))
        raw_count   = np.concatenate(count_parts).astype(np.int64, copy=False)
        count       = np.zeros(obs.shape[0], dtype=np.int64)
        np.add.at(count, inv, raw_count)
        keep = count > 0

        accept = accepted / proposed if proposed else 0.0
        elapsed = perf_counter() - t_draw
        time_sampler = max(0.0, elapsed - timing["time_graph"] - timing["time_forward"])

        return (
            replace(state, accept=accept),
            SampleBatch(np.ascontiguousarray(obs[keep]), count[keep]),
            {
                "accept":       float(accept),
                "n_prop_edge":  float(n_prop_edge),
                "n_blur_edge":  float(n_blur_edge),
                "time_sampler": float(time_sampler),
                "time_graph":   float(timing["time_graph"]),
                "time_forward": float(timing["time_forward"]),
            },
        )

    def _move(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: SamplerState,
        *,
        timing: dict[str, float] | None = None,
    ) -> tuple[SamplerState, int, int, int]:
        """Advance the compressed chain by one raw Metropolis move.

        Returns:
            (next_state, n_accepted, n_proposed, n_prop_edge)
        """
        pa     = precision.asarray
        rdtype = precision.dtype("calc", "real", host=True)

        key, subkey = jax.random.split(state.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng = np.random.default_rng(seed)

        chain  = state.chain
        count  = state.count.astype(np.int64, copy=False)
        logabs = pa(state.logabs, "calc", "real", host=True)
        n_row  = int(chain.shape[0])

        batch = propose(
            self.proposal,
            hamiltonian,
            chain,
            count,
            seed=seed,
            eps=float(self.proposal_eps),
            timing=timing,
        )

        proposed = int(np.sum(batch.count))
        if proposed == 0 or batch.dets.shape[0] == 0:
            return replace(state, key=key), 0, proposed, int(batch.n_edge)

        prop_logabs = np.empty(batch.dets.shape[0], dtype=rdtype)
        _, first, inv_lookup = unique_dets(np.concatenate([chain, batch.dets], axis=0))
        prop_first = first[inv_lookup[n_row:]]
        known = prop_first < n_row

        if known.any():
            prop_logabs[known] = logabs[prop_first[known]]

        if (~known).any():
            prop_unk = np.ascontiguousarray(batch.dets[~known])
            t = perf_counter()
            prop_logabs[~known] = pa(
                np.asarray(utils.host(utils.apply(model.logabs, theta, prop_unk))).reshape(-1),
                "calc", "real", host=True,
            )
            if timing is not None:
                timing["time_forward"] += perf_counter() - t

        log_ratio = (
            rdtype(self.alpha) * (prop_logabs[batch.dst] - logabs[batch.src])
            + pa(batch.log_qratio, "calc", "real", host=True)
        )
        accept_prob = np.clip(np.exp(np.minimum(rdtype(0.0), log_ratio)), rdtype(0.0), rdtype(1.0))
        accepted_g = rng.binomial(batch.count, accept_prob).astype(np.int64)
        accepted = int(accepted_g.sum())

        next_count = count.copy()
        np.add.at(next_count, batch.src, -accepted_g)

        prop_count = np.zeros(batch.dets.shape[0], dtype=np.int64)
        np.add.at(prop_count, batch.dst, accepted_g)

        all_dets   = np.concatenate([chain, batch.dets], axis=0)
        all_count  = np.concatenate([next_count, prop_count])
        all_logabs = np.concatenate([logabs, prop_logabs])

        chain, first, inv = unique_dets(all_dets)
        merged_count = np.zeros(chain.shape[0], dtype=np.int64)
        np.add.at(merged_count, inv, all_count)
        keep = merged_count > 0

        return (
            SamplerState(
                key    = key,
                chain  = np.ascontiguousarray(chain[keep]),
                count  = merged_count[keep],
                logabs = pa(all_logabs[first][keep], "calc", "real", host=True),
                accept = accepted / proposed if proposed else 0.0,
            ),
            accepted, proposed, int(batch.n_edge),
        )

    def _observe(
        self,
        hamiltonian: Any,
        state: SamplerState,
        base: np.ndarray,
        count: np.ndarray,
        *,
        timing: dict[str, float] | None = None,
    ) -> tuple[SamplerState, np.ndarray, np.ndarray, int]:
        """Apply observation kernel B_beta(y|x) to base samples.

        Returns:
            (next_state, obs_dets, obs_count, n_blur_edge)
        """
        beta = float(np.clip(self.blur, 0.0, 1.0))

        if beta <= 0.0:
            return (
                state,
                np.ascontiguousarray(base),
                count.astype(np.int64, copy=False),
                0,
            )

        if self.blur_kernel != "ham":
            raise ValueError("blur_kernel must be 'ham'")

        key, subkey = jax.random.split(state.key)
        seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
        rng  = np.random.default_rng(seed)

        blur_eps = float(self.proposal_eps if self.blur_eps is None else self.blur_eps)
        count    = count.astype(np.int64, copy=False)
        move     = rng.binomial(count, beta).astype(np.int64)
        stay     = count - move

        parts  = [base]
        counts = [stay]
        n_blur_edge = 0

        move_rows = np.flatnonzero(move > 0)
        if move_rows.size > 0:
            move_base  = np.ascontiguousarray(base[move_rows])
            move_count = move[move_rows].astype(np.int64, copy=False)

            t = perf_counter()
            sample = hamiltonian.sample_edges(
                move_base, move_count, eps1=np.inf, eps2=blur_eps, seed=seed,
            )
            n_blur_edge = int(np.asarray(sample.h).size)
            row_weight  = precision.asarray(
                np.asarray(sample.row_weight), "calc", "real", host=True,
            )
            if timing is not None:
                timing["time_graph"] += perf_counter() - t

            dead = row_weight <= 0.0
            if dead.any():
                stay[move_rows[dead]] += move_count[dead]

            if n_blur_edge > 0:
                sampled_dets  = np.ascontiguousarray(np.asarray(sample.dets, dtype=np.uint64))
                sampled_count = np.asarray(sample.counts, dtype=np.int64)
                obs, _, obs_inv = unique_dets(sampled_dets)
                obs_count = np.zeros(obs.shape[0], dtype=np.int64)
                np.add.at(obs_count, obs_inv, sampled_count)
                parts.append(obs)
                counts.append(obs_count)

        obs, _, inv = unique_dets(np.concatenate(parts, axis=0))
        raw_count   = np.concatenate(counts).astype(np.int64, copy=False)
        obs_count   = np.zeros(obs.shape[0], dtype=np.int64)
        np.add.at(obs_count, inv, raw_count)
        keep = obs_count > 0

        return (
            replace(state, key=key),
            np.ascontiguousarray(obs[keep]),
            obs_count[keep],
            n_blur_edge,
        )