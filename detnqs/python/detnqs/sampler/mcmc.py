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
class Walkers:
    """Flattened state of the physical Markov chains.

    ``dets[i]`` and ``logabs[i]`` always describe the same walker. ``alpha``
    and ``alpha_step`` carry the complete state of adaptive reference sampling.
    """

    key: jax.Array
    dets: np.ndarray
    logabs: np.ndarray
    alpha: float = 1.0
    alpha_step: int = 0


@dataclass(frozen=True, slots=True)
class MCSampler:
    """Metropolis sampler for determinant-space walkers.

    The Markov reference law is

        eta_alpha(x) proportional to |psi(x)|^alpha.

    The optional Hamiltonian blur changes only the observation law. It does
    not modify the Markov-chain state.
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
        init_method: str | Any = "hf",
        alpha: float | None = None,
        alpha_step: int = 0,
    ) -> Walkers:
        """Initialize all physical chains and run thermalization."""
        n_chains = int(self.n_chains)
        if n_chains <= 0:
            raise ValueError("n_chains must be positive")

        if self.alpha is None:
            alpha_value = 1.0 if alpha is None else float(alpha)
        else:
            alpha_value = float(self.alpha)

        key, init_key = jax.random.split(key)
        nword = int(hamiltonian.nword)
        norb = int(hamiltonian.norb)
        n_alpha = int(n_alpha)
        n_beta = int(n_beta)

        if isinstance(init_method, str):
            if init_method == "hf":
                det = np.zeros((1, 2, nword), dtype=np.uint64)

                for p in range(n_alpha):
                    det[0, 0, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

                for p in range(n_beta):
                    det[0, 1, p >> 6] |= np.uint64(1) << np.uint64(p & 63)

                dets = np.repeat(det, n_chains, axis=0)

            elif init_method == "random":
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
                    "init_method must be 'hf', 'random', or a determinant batch"
                )
        else:
            dets = libdet.to_dets(init_method)
            if dets.shape[0] == 0:
                raise ValueError(
                    "init_method determinant batch must be non-empty"
                )

            if dets.shape[0] != n_chains:
                reps = (n_chains + dets.shape[0] - 1) // dets.shape[0]
                dets = np.tile(dets, (reps, 1, 1))[:n_chains]

            dets = np.ascontiguousarray(dets)

        logabs = self._eval_logabs(theta, model, dets)

        state = Walkers(
            key=key,
            dets=dets,
            logabs=logabs,
            alpha=alpha_value,
            alpha_step=max(0, int(alpha_step)),
        )

        # Burn-in advances the same physical chains used by later measurements.
        # A modest default avoids making initialization dominate large systems.
        for _ in range(max(0, int(self.thermal_steps))):
            state, _, _, _ = self._move(theta, hamiltonian, model, state)

        return state

    def draw(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Walkers,
    ) -> tuple[Walkers, np.ndarray, np.ndarray, dict[str, float]]:
        """Draw observations and advance the chains.

        Samples remain walker-wise here. Determinant multiplicities and
        statistical masses are reduced only at the estimator boundary.
        """
        n_samples = int(self.n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if state.dets.shape[0] != int(self.n_chains):
            raise ValueError("walker state size must equal n_chains")

        timer = utils.Timer()
        alpha = float(state.alpha) if self.alpha is None else float(self.alpha)

        # Parameters change between optimization steps; refresh the chain cache.
        with timer("forward"):
            state = replace(
                state,
                logabs=self._eval_logabs(theta, model, state.dets),
                alpha=alpha,
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
        mass_parts: list[np.ndarray] = []
        remaining = n_samples
        sweep_steps = max(1, int(self.sweep_steps))

        while remaining > 0:
            take = min(int(self.n_chains), remaining)

            if take == int(self.n_chains):
                base_dets = state.dets
            else:
                with timer("sample"):
                    key, subkey = jax.random.split(state.key)
                    seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                    rng = np.random.default_rng(seed)
                    index = rng.choice(
                        int(self.n_chains),
                        size=take,
                        replace=False,
                    )
                    base_dets = np.ascontiguousarray(state.dets[index])
                    state = replace(state, key=key)

            state, observed, mass, n_conn = self._observe(
                hamiltonian,
                state,
                base_dets,
                timer=timer,
            )
            det_parts.append(observed)
            mass_parts.append(mass)
            n_conn_blur += n_conn

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

        stats = {
            "accept": float(accepted / proposed if proposed else 0.0),
            "n_conn_proposal": float(n_conn_proposal),
            "n_conn_blur": float(n_conn_blur),
        }
        stats.update(timer.stats())

        return (
            state,
            np.ascontiguousarray(np.concatenate(det_parts, axis=0)),
            precision.asarray(
                np.concatenate(mass_parts),
                "calc",
                "real",
                host=True,
            ),
            stats,
        )

    def _move(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Walkers,
        *,
        timer: utils.Timer | None = None,
    ) -> tuple[Walkers, int, int, int]:
        """Advance every walker by one Metropolis-Hastings step.

        log A(x -> y) =
            alpha [log|psi(y)| - log|psi(x)|]
            + log q(x|y) - log q(y|x).
        """
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("sample"):
            key, proposal_key, accept_key = jax.random.split(state.key, 3)
            proposal_seed = int(
                jax.random.bits(proposal_key, (), dtype=jnp.uint32)
            )
            accept_seed = int(
                jax.random.bits(accept_key, (), dtype=jnp.uint32)
            )

        with timer("graph"):
            dets_y, log_qratio, active, n_conn = propose(
                self.proposal,
                hamiltonian,
                state.dets,
                seed=proposal_seed,
                eps=float(self.proposal_eps),
            )

        proposed = int(np.count_nonzero(active))
        if proposed == 0:
            return replace(state, key=key), 0, 0, int(n_conn)

        logabs_y = np.empty(state.dets.shape[0], dtype=rdtype)
        logabs_y[~active] = state.logabs[~active]

        active_dets = np.ascontiguousarray(dets_y[active])
        unique_y, _, inv_y = unique_dets(active_dets)

        # The model sees only new unique proposals. Current amplitudes are cached.
        _, first, lookup = unique_dets(
            np.concatenate([state.dets, unique_y], axis=0)
        )
        y_first = first[lookup[state.dets.shape[0] :]]
        known = y_first < state.dets.shape[0]
        unique_logabs = np.empty(unique_y.shape[0], dtype=rdtype)

        if known.any():
            unique_logabs[known] = state.logabs[y_first[known]]

        if (~known).any():
            with timer("forward"):
                unique_logabs[~known] = self._eval_logabs(
                    theta,
                    model,
                    np.ascontiguousarray(unique_y[~known]),
                )

        logabs_y[active] = unique_logabs[inv_y]

        with timer("sample"):
            log_ratio = (
                rdtype(state.alpha) * (logabs_y - state.logabs)
                + precision.asarray(log_qratio, "calc", "real", host=True)
            )

            rng = np.random.default_rng(accept_seed)
            log_uniform = np.log(rng.random(state.dets.shape[0]))
            accept = active & (log_uniform < np.minimum(rdtype(0.0), log_ratio))

            det_mask = accept.reshape((-1, 1, 1))
            next_dets = np.where(det_mask, dets_y, state.dets)
            next_logabs = np.where(accept, logabs_y, state.logabs)

        return (
            replace(
                state,
                key=key,
                dets=np.ascontiguousarray(next_dets),
                logabs=precision.asarray(
                    next_logabs,
                    "calc",
                    "real",
                    host=True,
                ),
            ),
            int(np.count_nonzero(accept)),
            proposed,
            int(n_conn),
        )

    def _observe(
        self,
        hamiltonian: Any,
        state: Walkers,
        base_dets: np.ndarray,
        *,
        timer: utils.Timer | None = None,
    ) -> tuple[Walkers, np.ndarray, np.ndarray, int]:
        """Apply the Hamiltonian blur independently to every observation.

        B(y|x) = (1-beta) delta_xy + beta |H_yx| / d_B(x).

        Each observation carries source mass d_B(x); an empty Hamiltonian row
        falls back to the identity kernel with unit mass.
        """
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)
        beta = float(np.clip(self.blur, 0.0, 1.0))

        if beta <= 0.0:
            return (
                state,
                np.ascontiguousarray(base_dets),
                np.ones(base_dets.shape[0], dtype=rdtype),
                0,
            )

        with timer("sample"):
            key, choice_key, conn_key = jax.random.split(state.key, 3)
            choice_seed = int(
                jax.random.bits(choice_key, (), dtype=jnp.uint32)
            )
            conn_seed = int(
                jax.random.bits(conn_key, (), dtype=jnp.uint32)
            )
            rng = np.random.default_rng(choice_seed)
            move = rng.random(base_dets.shape[0]) < beta

            kets, _, walker_to_ket = unique_dets(base_dets)
            counts = np.bincount(
                walker_to_ket[move],
                minlength=kets.shape[0],
            ).astype(np.int64)

        blur_eps = float(
            self.proposal_eps if self.blur_eps is None else self.blur_eps
        )

        with timer("graph"):
            sample = hamiltonian.sample_conns(
                kets,
                counts,
                eps1=np.inf,
                eps2=blur_eps,
                seed=conn_seed,
            )

        with timer("sample"):
            ket_weight = precision.asarray(
                np.asarray(sample.ket_weight),
                "calc",
                "real",
                host=True,
            )
            source_mass = np.where(
                ket_weight[walker_to_ket] > 0.0,
                ket_weight[walker_to_ket],
                rdtype(1.0),
            )

            observed = np.ascontiguousarray(base_dets.copy())
            sampled_ket = np.asarray(sample.ket, dtype=np.int64)
            sampled_count = np.asarray(sample.counts, dtype=np.int64)
            sampled_bras = np.ascontiguousarray(
                np.asarray(sample.bras, dtype=np.uint64)
            )

            assign_rng = np.random.default_rng(conn_seed ^ 0x85EBCA6B)

            for iket in np.unique(sampled_ket):
                records = np.flatnonzero(sampled_ket == iket)
                draw = np.repeat(records, sampled_count[records])
                walkers = np.flatnonzero(move & (walker_to_ket == iket))

                if draw.size != walkers.size:
                    raise RuntimeError("libdet returned an inconsistent blur count")

                assign_rng.shuffle(draw)
                observed[walkers] = sampled_bras[draw]

            n_conn = int(np.asarray(sample.ket_nconn, dtype=np.int64).sum())

        return (
            replace(state, key=key),
            observed,
            precision.asarray(source_mass, "calc", "real", host=True),
            n_conn,
        )

    @staticmethod
    def _eval_logabs(theta: Any, model: Any, dets: np.ndarray) -> np.ndarray:
        """Evaluate each distinct determinant once and scatter to walkers."""
        unique, _, inv = unique_dets(dets)
        value = utils.apply(model.logabs, theta, unique)
        jax.block_until_ready(value)

        unique_value = precision.asarray(
            np.asarray(utils.host(value)).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        return precision.asarray(
            unique_value[inv],
            "calc",
            "real",
            host=True,
        )
