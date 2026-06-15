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
            state, _, _, _ = self._step(
                theta,
                hamiltonian,
                model,
                state,
            )

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
            state, _, _, info = self._step(
                theta,
                hamiltonian,
                model,
                state,
                timer=timer,
            )
            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn_proposal += info["proposal_conn"]

        det_parts: list[np.ndarray] = []
        mass_parts: list[np.ndarray] = []
        remaining = n_samples
        sweep_steps = max(1, int(self.sweep_steps))

        while remaining > 0:
            take = min(int(self.n_chains), remaining)

            state, dets, mass, info = self._step(
                theta,
                hamiltonian,
                model,
                state,
                n_observe=take,
                timer=timer,
            )
            det_parts.append(dets)
            mass_parts.append(mass)
            accepted += info["accepted"]
            proposed += info["proposed"]
            n_conn_proposal += info["proposal_conn"]
            n_conn_blur += info["blur_conn"]

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
                n_conn_proposal += info["proposal_conn"]

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

    def _step(
        self,
        theta: Any,
        hamiltonian: Any,
        model: Any,
        state: Walkers,
        *,
        n_observe: int = 0,
        timer: utils.Timer | None = None,
    ) -> tuple[Walkers, np.ndarray, np.ndarray, dict[str, int]]:
        """Observe selected walkers, then advance every chain by one step.

        log A(x -> y) =
            alpha [log|psi(y)| - log|psi(x)|]
            + log q(x|y) - log q(y|x).
        """
        timer = utils.Timer() if timer is None else timer
        rdtype = precision.dtype("calc", "real", host=True)
        n_walker = int(state.dets.shape[0])
        n_observe = int(n_observe)
        if n_observe < 0 or n_observe > n_walker:
            raise ValueError("n_observe must be between zero and n_chains")

        beta = float(np.clip(self.blur, 0.0, 1.0))
        eps = float(self.proposal_eps)
        blur_eps = float(
            eps if self.blur_eps is None else self.blur_eps
        )

        with timer("sample"):
            key, random_key = jax.random.split(state.key)
            seed = int(jax.random.bits(random_key, (), dtype=jnp.uint32))
            rng = np.random.default_rng(seed)
            if n_observe == n_walker:
                observe_rows = np.arange(n_walker, dtype=np.int64)
            else:
                observe_rows = rng.choice(
                    n_walker,
                    size=n_observe,
                    replace=False,
                )
            observed = np.ascontiguousarray(state.dets[observe_rows])
            mass = np.ones(n_observe, dtype=rdtype)

        shared = (
            n_observe > 0
            and beta > 0.0
            and self.proposal == "ham"
            and blur_eps == eps
        )
        proposal_conn = 0
        blur_conn = 0

        if self.proposal == "single":
            with timer("sample"):
                trial, log_q, active = self._single_proposal(
                    hamiltonian,
                    state.dets,
                    rng,
                )
        elif self.proposal == "ham":
            with timer("sample"):
                kets, _, ket_index = libdet.unique_dets(state.dets)
                counts = np.bincount(
                    ket_index,
                    minlength=kets.shape[0],
                ).astype(np.int64)

                if shared:
                    obs_index = ket_index[observe_rows]
                    blur = rng.random(n_observe) < beta
                    blur_counts = np.bincount(
                        obs_index[blur],
                        minlength=kets.shape[0],
                    ).astype(np.int64)
                    counts = np.stack((counts, blur_counts))

            with timer("graph"):
                sample = hamiltonian.sample_conns(
                    kets,
                    np.ascontiguousarray(counts),
                    eps1=np.inf,
                    eps2=eps,
                    seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                )
                proposal_conn = int(np.asarray(sample.h).size)

            with timer("sample"):
                trial, active = self._draw_conns(
                    state.dets,
                    ket_index,
                    np.ones(n_walker, dtype=bool),
                    sample,
                    stream=0,
                    rng=rng,
                )

                if shared:
                    observed, _ = self._draw_conns(
                        observed,
                        obs_index,
                        blur,
                        sample,
                        stream=1,
                        rng=rng,
                    )
                    weight = precision.asarray(
                        np.asarray(sample.weight),
                        "calc",
                        "real",
                        host=True,
                    )
                    mass = np.where(
                        weight[obs_index] > 0.0,
                        weight[obs_index],
                        rdtype(1.0),
                    )

            with timer("graph"):
                log_q = self._ham_log_ratio(
                    hamiltonian,
                    kets,
                    ket_index,
                    trial,
                    active,
                    sample,
                    eps,
                )
        else:
            raise ValueError("proposal must be 'ham' or 'single'")

        if n_observe > 0 and beta > 0.0 and not shared:
            with timer("sample"):
                if self.proposal == "single":
                    blur_kets, _, obs_index = libdet.unique_dets(observed)
                else:
                    blur_kets = kets
                    obs_index = ket_index[observe_rows]

                blur = rng.random(n_observe) < beta
                counts = np.bincount(
                    obs_index[blur],
                    minlength=blur_kets.shape[0],
                ).astype(np.int64)

            with timer("graph"):
                sample = hamiltonian.sample_conns(
                    blur_kets,
                    np.ascontiguousarray(counts),
                    eps1=np.inf,
                    eps2=blur_eps,
                    seed=int(rng.integers(0, 2**32, dtype=np.uint64)),
                )
                blur_conn = int(np.asarray(sample.h).size)

            with timer("sample"):
                observed, _ = self._draw_conns(
                    observed,
                    obs_index,
                    blur,
                    sample,
                    stream=0,
                    rng=rng,
                )
                weight = precision.asarray(
                    np.asarray(sample.weight),
                    "calc",
                    "real",
                    host=True,
                )
                mass = np.where(
                    weight[obs_index] > 0.0,
                    weight[obs_index],
                    rdtype(1.0),
                )

        proposed = int(np.count_nonzero(active))
        accepted = 0
        next_state = replace(state, key=key)

        if proposed:
            logabs_y = np.empty(n_walker, dtype=rdtype)
            logabs_y[~active] = state.logabs[~active]
            unique_y, _, inverse_y = libdet.unique_dets(trial[active])

            # Reuse cached amplitudes when a proposal is a current walker.
            _, first, lookup = libdet.unique_dets(
                np.concatenate([state.dets, unique_y], axis=0)
            )
            first_y = first[lookup[n_walker:]]
            known = first_y < n_walker
            unique_logabs = np.empty(unique_y.shape[0], dtype=rdtype)

            if known.any():
                unique_logabs[known] = state.logabs[first_y[known]]

            if (~known).any():
                with timer("forward"):
                    unique_logabs[~known] = self._eval_logabs(
                        theta,
                        model,
                        np.ascontiguousarray(unique_y[~known]),
                    )

            logabs_y[active] = unique_logabs[inverse_y]

            with timer("sample"):
                log_accept = (
                    rdtype(state.alpha) * (logabs_y - state.logabs)
                    + precision.asarray(log_q, "calc", "real", host=True)
                )
                accept = active & (
                    np.log(rng.random(n_walker))
                    < np.minimum(rdtype(0.0), log_accept)
                )
                accepted = int(np.count_nonzero(accept))
                next_state = replace(
                    state,
                    key=key,
                    dets=np.ascontiguousarray(
                        np.where(
                            accept.reshape((-1, 1, 1)),
                            trial,
                            state.dets,
                        )
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
            precision.asarray(
                mass,
                "calc",
                "real",
                host=True,
            ),
            {
                "accepted": accepted,
                "proposed": proposed,
                "proposal_conn": proposal_conn,
                "blur_conn": blur_conn,
            },
        )

    @staticmethod
    def _draw_conns(
        dets: np.ndarray,
        index: np.ndarray,
        selected: np.ndarray,
        sample: Any,
        *,
        stream: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assign sampled connections to selected determinant rows."""
        out = np.ascontiguousarray(dets.copy())
        changed = np.zeros(dets.shape[0], dtype=bool)
        n_ket = int(sample.n_kets)
        ptr = np.asarray(sample.ptr, dtype=np.int64)
        col = np.asarray(sample.col, dtype=np.int64)
        count = np.asarray(sample.count, dtype=np.int64)
        pool = np.asarray(sample.dets, dtype=np.uint64)

        for i in np.unique(index[selected]):
            block = int(stream) * n_ket + int(i)
            records = np.arange(ptr[block], ptr[block + 1])
            if records.size == 0:
                continue
            draw = np.repeat(records, count[records])
            rows = np.flatnonzero(selected & (index == i))
            if draw.size != rows.size:
                raise RuntimeError(
                    "libdet returned an inconsistent sample count"
                )

            rng.shuffle(draw)
            out[rows] = pool[col[draw]]
            changed[rows] = True

        return out, changed

    @staticmethod
    def _ham_log_ratio(
        hamiltonian: Any,
        kets: np.ndarray,
        ket_index: np.ndarray,
        trial: np.ndarray,
        active: np.ndarray,
        sample: Any,
        eps: float,
    ) -> np.ndarray:
        """Return the heat-bath Hastings correction."""
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))
        log_q = np.zeros(trial.shape[0], dtype=rdtype)

        if not active.any():
            return log_q

        source_weight = precision.asarray(
            np.asarray(sample.weight),
            "calc",
            "real",
            host=True,
        )
        bras, _, bra_index = libdet.unique_dets(trial[active])
        bra_weight = np.empty(bras.shape[0], dtype=rdtype)

        # Current rows already have their degree from the proposal scan.
        _, first, lookup = libdet.unique_dets(
            np.concatenate([kets, bras], axis=0)
        )
        bra_first = first[lookup[kets.shape[0] :]]
        known = bra_first < kets.shape[0]
        if known.any():
            bra_weight[known] = source_weight[bra_first[known]]

        if (~known).any():
            weight, _ = hamiltonian.degrees(
                np.ascontiguousarray(bras[~known]),
                eps,
            )
            bra_weight[~known] = precision.asarray(
                weight,
                "calc",
                "real",
                host=True,
            )

        source = source_weight[ket_index[active]]
        target = bra_weight[bra_index]
        log_q[active] = (
            np.log(np.maximum(source, tiny))
            - np.log(np.maximum(target, tiny))
        )
        return log_q

    @staticmethod
    def _single_proposal(
        hamiltonian: Any,
        dets: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw a uniform single excitation in a fixed number sector."""
        n_walker = int(dets.shape[0])
        rdtype = precision.dtype("calc", "real", host=True)

        trial = np.ascontiguousarray(dets.copy())
        log_q = np.zeros(n_walker, dtype=rdtype)
        active = np.zeros(n_walker, dtype=bool)
        if n_walker == 0:
            return trial, log_q, active

        norb = int(hamiltonian.norb)
        orbitals = np.arange(norb, dtype=np.int64)
        word = orbitals >> 6
        bit = (orbitals & 63).astype(np.uint64)
        occ = ((dets[:, :, word] >> bit) & np.uint64(1)).astype(bool)
        n_occ = occ.sum(axis=2)

        if not np.all(n_occ == n_occ[0]):
            raise ValueError(
                "single proposal requires a fixed particle-number sector"
            )

        n_alpha = int(n_occ[0, 0])
        n_beta = int(n_occ[0, 1])
        n_move_a = n_alpha * (norb - n_alpha)
        n_move_b = n_beta * (norb - n_beta)
        n_move = n_move_a + n_move_b
        if n_move <= 0:
            return trial, log_q, active

        move = rng.integers(n_move, size=n_walker, dtype=np.int64)
        spin = np.where(move < n_move_a, 0, 1).astype(np.int64)
        move_spin = np.where(spin == 0, move, move - n_move_a)
        n_vir = np.where(spin == 0, norb - n_alpha, norb - n_beta)
        occ_rank = move_spin // n_vir
        vir_rank = move_spin % n_vir

        walker_occ = occ[np.arange(n_walker), spin]
        walker_vir = ~walker_occ
        occ_pos = np.cumsum(walker_occ, axis=1) - 1
        vir_pos = np.cumsum(walker_vir, axis=1) - 1
        occ_orb = np.argmax(
            walker_occ & (occ_pos == occ_rank[:, None]),
            axis=1,
        ).astype(np.int64)
        vir_orb = np.argmax(
            walker_vir & (vir_pos == vir_rank[:, None]),
            axis=1,
        ).astype(np.int64)

        row = np.arange(n_walker, dtype=np.int64)
        occ_word = occ_orb >> 6
        occ_bit = (occ_orb & 63).astype(np.uint64)
        vir_word = vir_orb >> 6
        vir_bit = (vir_orb & 63).astype(np.uint64)
        trial[row, spin, occ_word] &= ~(np.uint64(1) << occ_bit)
        trial[row, spin, vir_word] |= np.uint64(1) << vir_bit
        active[:] = True
        return trial, log_q, active

    @staticmethod
    def _eval_logabs(theta: Any, model: Any, dets: np.ndarray) -> np.ndarray:
        """Evaluate each distinct determinant once and scatter to walkers."""
        unique, _, inv = libdet.unique_dets(dets)
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
