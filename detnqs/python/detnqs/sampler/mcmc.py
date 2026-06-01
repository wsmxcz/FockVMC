from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from .. import utils
from ..utils import precision
from .proposal import propose
from .proposal import unique_dets


@dataclass(frozen=True, slots=True)
class SampleBatch:
    """Counted observed determinants.

    The Markov chain state is x. The observed determinant is y after the
    optional observation kernel B(y|x).

    count:
        Number of observed walkers at each determinant.

    mass:
        Statistical mass entering the reweighted estimator. For identity
        observation, mass == count. For degree-tilted blurred observation,

            mass(y) = sum_{i: Y_i = y} s(X_i),

        where s(x) = d_B(x) for non-empty blur kets and s(x) = 1 for empty
        blur kets.
    """

    dets: np.ndarray
    count: np.ndarray
    mass: np.ndarray


@dataclass(frozen=True, slots=True)
class WalkerState:
    """Counted empirical state of exchangeable Markov walkers.

    Public semantics:
        n_chains parallel walkers.

    Internal representation:
        {(x_u, m_u, log|psi(x_u)|)}, with sum_u m_u = n_chains.

    alpha:
        Numeric exponent used by the reference law in this state. When the
        sampler is configured with alpha="adaptive", MCState updates this
        scalar after each estimator pass.
    """

    key: jax.Array
    dets: np.ndarray
    count: np.ndarray
    logabs: np.ndarray
    accept: float = 0.0
    alpha: float = 1.0
    alpha_step: int = 0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Counted determinant-space Metropolis sampler.

    Reference law:
        eta_alpha(x) proportional to |psi_theta(x)|^alpha.

    Observation kernel:
        B_beta(y|x) = (1 - beta_x) delta_xy + beta_x K(y|x),
        K(y|x)      = |H_yx| / d_B(x).

    If d_B(x) = 0, beta_x is set to zero and the walker stays at x.

    Step convention:
        A raw Metropolis step means one proposal attempt for each counted
        walker. thermal_steps are run after initialization/reset,
        discard_steps before each production draw, and sweep_steps separate
        consecutive observations.

    alpha:
        A numeric value gives a fixed reference exponent. The string
        "adaptive" initializes the exponent at 1.0 and lets MCState update the
        numeric value stored in WalkerState.alpha.
    """

    n_samples: int = 1024
    n_chains: int = 1024

    thermal_steps: int = 1024
    discard_steps: int = 0
    sweep_steps: int = 1
    reset_chains: bool = False

    alpha: float | str = 1.0
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
        alpha: float | None = None,
        alpha_step: int = 0,
    ) -> WalkerState:
        """Initialize counted walkers and run thermalization.

        init_method:
            "hf"     - lowest n_alpha/n_beta orbitals.
            "random" - random fixed-(n_alpha, n_beta) determinants.
            array    - user-provided determinant batch, tiled if needed.

        alpha_step:
            Adaptive clock for alpha. It is preserved across chain resets so that
            the reference law changes with diminishing adaptation.
        """
        n_chains = int(self.n_chains)
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")

        if isinstance(self.alpha, str):
            if self.alpha != "adaptive":
                raise ValueError("alpha must be a float or 'adaptive'")
            alpha_value = 0.5 if alpha is None else float(alpha)
        else:
            alpha_value = float(self.alpha)

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

                for ichain in range(n_chains):
                    for spin, n_elec in enumerate((int(n_alpha), int(n_beta))):
                        for p in rng.choice(norb, size=n_elec, replace=False):
                            p = int(p)
                            dets[ichain, spin, p >> 6] |= (
                                np.uint64(1) << np.uint64(p & 63)
                            )

            case str():
                raise ValueError(
                    "init_method must be 'hf', 'random', or a determinant batch"
                )

            case _:
                dets = libdet.to_dets(init_method)

                if dets.shape[0] == 0:
                    raise ValueError("init_method determinant batch must be non-empty")

                if dets.shape[0] != n_chains:
                    reps = (n_chains + dets.shape[0] - 1) // dets.shape[0]
                    dets = np.tile(dets, (reps, 1, 1))[:n_chains]

        dets, _, inv = unique_dets(dets)
        count = np.bincount(inv, minlength=dets.shape[0]).astype(np.int64)

        logabs_jax = utils.apply(model.logabs, theta, dets)
        jax.block_until_ready(logabs_jax)

        logabs = precision.asarray(
            np.asarray(utils.host(logabs_jax)).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        state = WalkerState(
            key=key,
            dets=dets,
            count=count,
            logabs=logabs,
            alpha=alpha_value,
            alpha_step=max(0, int(alpha_step)),
        )

        accepted = 0
        proposed = 0

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

        if isinstance(self.alpha, str):
            if self.alpha != "adaptive":
                raise ValueError("alpha must be a float or 'adaptive'")
            alpha_value = float(state.alpha)
        else:
            alpha_value = float(self.alpha)

        timer = utils.Timer()

        # Parameters change during optimization. Refresh cached log|psi| before
        # any Metropolis ratio is formed.
        with timer("forward"):
            logabs_jax = utils.apply(model.logabs, theta, state.dets)
            jax.block_until_ready(logabs_jax)

            state = replace(
                state,
                logabs=precision.asarray(
                    np.asarray(utils.host(logabs_jax)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                ),
                alpha=alpha_value,
            )

        accepted = 0
        proposed = 0
        n_conn_proposal = 0
        n_conn_blur = 0

        for _ in range(max(0, int(self.discard_steps))):
            state, acc, prop, n_conn = self._move(
                theta,
                hamiltonian,
                model,
                state,
                timer=timer,
            )
            accepted += acc
            proposed += prop
            n_conn_proposal += n_conn

        det_parts: list[np.ndarray] = []
        count_parts: list[np.ndarray] = []
        mass_parts: list[np.ndarray] = []

        remaining = int(self.n_samples)
        n_chains = int(self.n_chains)
        sweep_steps = max(1, int(self.sweep_steps))

        while remaining > 0:
            with timer("sample"):
                take = min(n_chains, remaining)

                if take == n_chains:
                    base_dets = state.dets
                    base_count = state.count.astype(np.int64, copy=False)
                else:
                    # Partial observation samples counted walkers without
                    # replacement from the empirical walker measure.
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

            state, obs_dets, obs_count, obs_mass, n_conn = self._observe(
                hamiltonian,
                state,
                base_dets,
                base_count,
                timer=timer,
            )

            with timer("sample"):
                det_parts.append(obs_dets)
                count_parts.append(obs_count)
                mass_parts.append(obs_mass)
                n_conn_blur += n_conn

            # Observations are separated by sweep_steps raw Metropolis moves.
            for _ in range(sweep_steps):
                state, acc, prop, n_conn = self._move(
                    theta,
                    hamiltonian,
                    model,
                    state,
                    timer=timer,
                )
                accepted += acc
                proposed += prop
                n_conn_proposal += n_conn

            remaining -= take

        with timer("sample"):
            obs_dets, _, inv = unique_dets(np.concatenate(det_parts, axis=0))
            raw_count = np.concatenate(count_parts).astype(np.int64, copy=False)
            raw_mass = np.concatenate(mass_parts).astype(
                precision.dtype("calc", "real", host=True),
                copy=False,
            )

            obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
            obs_mass = np.zeros(obs_dets.shape[0], dtype=raw_mass.dtype)

            np.add.at(obs_count, inv, raw_count)
            np.add.at(obs_mass, inv, raw_mass)

            keep = obs_count > 0
            obs_dets = np.ascontiguousarray(obs_dets[keep])
            obs_count = obs_count[keep]
            obs_mass = obs_mass[keep]

            accept = accepted / proposed if proposed else 0.0

        stats = {
            "accept": float(accept),
            "n_conn_proposal": float(n_conn_proposal),
            "n_conn_blur": float(n_conn_blur),
        }
        stats.update(timer.stats())

        return (
            replace(state, accept=accept),
            SampleBatch(dets=obs_dets, count=obs_count, mass=obs_mass),
            stats,
        )

    def _move(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: WalkerState,
        *,
        timer: utils.Timer | None = None,
    ) -> tuple[WalkerState, int, int, int]:
        """Advance the counted walker state by one raw Metropolis step.

        Acceptance probability:

            a(ket -> bra) = min(1, exp[
                alpha (log|psi(bra)| - log|psi(ket)|)
                + log q(ket|bra) - log q(bra|ket)
            ]).
        """
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("sample"):
            key, subkey = jax.random.split(state.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)

            alpha = rdtype(state.alpha)
            dets = state.dets
            count = state.count.astype(np.int64, copy=False)
            logabs = precision.asarray(state.logabs, "calc", "real", host=True)
            n_ket = int(dets.shape[0])

        with timer("graph"):
            batch = propose(
                self.proposal,
                hamiltonian,
                dets,
                count,
                seed=seed,
                eps=float(self.proposal_eps),
            )

        with timer("sample"):
            proposed = int(np.sum(batch.count))

            if proposed == 0 or batch.dets.shape[0] == 0:
                return replace(state, key=key), 0, proposed, int(batch.n_conn)

            bra_logabs = np.empty(batch.dets.shape[0], dtype=rdtype)

            # Avoid duplicate model evaluations for proposals already present
            # in the current counted support.
            _, first, inv_lookup = unique_dets(
                np.concatenate([dets, batch.dets], axis=0)
            )
            bra_first = first[inv_lookup[n_ket:]]
            known = bra_first < n_ket

            if known.any():
                bra_logabs[known] = logabs[bra_first[known]]

        if (~known).any():
            unknown_bras = np.ascontiguousarray(batch.dets[~known])

            with timer("forward"):
                bra_logabs_jax = utils.apply(model.logabs, theta, unknown_bras)
                jax.block_until_ready(bra_logabs_jax)

                bra_logabs[~known] = precision.asarray(
                    np.asarray(utils.host(bra_logabs_jax)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                )

        with timer("sample"):
            log_ratio = (
                alpha * (bra_logabs[batch.bra] - logabs[batch.ket])
                + precision.asarray(batch.log_qratio, "calc", "real", host=True)
            )

            accept_prob = np.clip(
                np.exp(np.minimum(rdtype(0.0), log_ratio)),
                rdtype(0.0),
                rdtype(1.0),
            )

            accepted_g = rng.binomial(batch.count, accept_prob).astype(np.int64)
            accepted = int(accepted_g.sum())

            next_count = count.copy()
            np.add.at(next_count, batch.ket, -accepted_g)

            bra_count = np.zeros(batch.dets.shape[0], dtype=np.int64)
            np.add.at(bra_count, batch.bra, accepted_g)

            all_dets = np.concatenate([dets, batch.dets], axis=0)
            all_count = np.concatenate([next_count, bra_count])
            all_logabs = np.concatenate([logabs, bra_logabs])

            next_dets, first, inv = unique_dets(all_dets)

            merged_count = np.zeros(next_dets.shape[0], dtype=np.int64)
            np.add.at(merged_count, inv, all_count)

            keep = merged_count > 0

            next_state = WalkerState(
                key=key,
                dets=np.ascontiguousarray(next_dets[keep]),
                count=merged_count[keep],
                logabs=precision.asarray(
                    all_logabs[first][keep],
                    "calc",
                    "real",
                    host=True,
                ),
                accept=accepted / proposed if proposed else 0.0,
                alpha=float(state.alpha),
                alpha_step=int(state.alpha_step),
            )

        return next_state, accepted, proposed, int(batch.n_conn)

    def _observe(
        self,
        hamiltonian: Any,
        state: WalkerState,
        base_dets: np.ndarray,
        base_count: np.ndarray,
        *,
        timer: utils.Timer | None = None,
    ) -> tuple[WalkerState, np.ndarray, np.ndarray, np.ndarray, int]:
        """Apply the observation kernel B_beta(y|x) to counted base walkers.

        With blur disabled, this returns identity observations with mass=count.

        With Hamiltonian blur enabled, observations are drawn from

            B(y|x) = (1 - beta_x) delta_xy
                   + beta_x |H_yx| / d_B(x),

        and each realized observation carries source mass

            s(x) = d_B(x) if d_B(x) > 0 else 1.

        This degree-tilted mass removes the need for second-order degree
        lookups when the observed density is evaluated in MCState.
        """
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("sample"):
            beta = float(np.clip(self.blur, 0.0, 1.0))
            base_count = base_count.astype(np.int64, copy=False)

            if beta <= 0.0:
                return (
                    state,
                    np.ascontiguousarray(base_dets),
                    base_count,
                    base_count.astype(rdtype, copy=False),
                    0,
                )

            if self.blur_kernel != "ham":
                raise ValueError("blur_kernel must be 'ham'")

            key, subkey = jax.random.split(state.key)
            seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)

            blur_eps = float(
                self.proposal_eps if self.blur_eps is None else self.blur_eps
            )

            # Draw attempted blur moves first. Kets with d_B=0 are corrected
            # after ket weights have been computed by sample_conns.
            move = rng.binomial(base_count, beta).astype(np.int64)
            stay = base_count - move

        with timer("graph"):
            sample = hamiltonian.sample_conns(
                base_dets,
                move,
                eps1=np.inf,
                eps2=blur_eps,
                seed=seed,
            )

        with timer("sample"):
            ket_weight = precision.asarray(
                np.asarray(sample.ket_weight),
                "calc",
                "real",
                host=True,
            )

            # Empty blur kets use beta_x=0 and source mass s(x)=1.
            dead = ket_weight <= 0.0
            if dead.any():
                stay[dead] += move[dead]
                move[dead] = 0

            source_mass = np.where(ket_weight > 0.0, ket_weight, rdtype(1.0))
            stay_mass = stay.astype(rdtype, copy=False) * source_mass

            parts = [base_dets]
            counts = [stay]
            masses = [stay_mass]

            n_conn_blur = int(np.asarray(sample.ket_nconn, dtype=np.int64).sum())

            sampled_bras = np.ascontiguousarray(
                np.asarray(sample.bras, dtype=np.uint64)
            )

            if sampled_bras.shape[0] > 0:
                sampled_count = np.asarray(sample.counts, dtype=np.int64)
                sampled_ket = np.asarray(sample.ket, dtype=np.int64)

                sampled_mass_raw = (
                    sampled_count.astype(rdtype, copy=False)
                    * source_mass[sampled_ket]
                )

                obs_dets, _, obs_inv = unique_dets(sampled_bras)

                obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
                obs_mass = np.zeros(obs_dets.shape[0], dtype=rdtype)

                np.add.at(obs_count, obs_inv, sampled_count)
                np.add.at(obs_mass, obs_inv, sampled_mass_raw)

                parts.append(obs_dets)
                counts.append(obs_count)
                masses.append(obs_mass)

            obs_dets, _, inv = unique_dets(np.concatenate(parts, axis=0))
            raw_count = np.concatenate(counts).astype(np.int64, copy=False)
            raw_mass = np.concatenate(masses).astype(rdtype, copy=False)

            obs_count = np.zeros(obs_dets.shape[0], dtype=np.int64)
            obs_mass = np.zeros(obs_dets.shape[0], dtype=rdtype)

            np.add.at(obs_count, inv, raw_count)
            np.add.at(obs_mass, inv, raw_mass)

            keep = obs_count > 0

        return (
            replace(state, key=key),
            np.ascontiguousarray(obs_dets[keep]),
            obs_count[keep],
            obs_mass[keep],
            n_conn_blur,
        )