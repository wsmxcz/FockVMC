from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from detnqs import utils
from ..model.base import Model
from ..model.base import to_logabs
from ..model.base import to_ratio
from ..optimizer import Geometry
from ..sampler.mcmc import MCSampler
from ..sampler.mcmc import Walkers
from ..sampler.proposal import unique_dets
from ..utils import precision
from .base import VState


@dataclass(frozen=True, slots=True)
class Edges:
    """Hamiltonian edges from observed kets to a unique bra pool.

    ``scale`` is an optional stochastic correction. Exact edges use unit
    scale; sampled weak edges use ``count / (n_sample * pgen)``.
    """

    ket: np.ndarray
    bras: np.ndarray
    bra: np.ndarray
    h: np.ndarray
    scale: np.ndarray | None = None


@dataclass(slots=True)
class MCState(VState):
    """Monte Carlo variational state.

    Physical target:
        pi_theta(x) proportional to |psi_theta(x)|^2.

    Markov reference:
        eta_alpha(x) proportional to |psi_theta(x)|^alpha.

    Observation law:
        x ~ eta_alpha, y ~ B(y|x).

    Hamiltonian blur:
        The observed sample carries source mass s(x). For non-empty blur rows,

            s(x) = d_B(x).

        The observed unnormalized density is

            r_tilde(y)
              = (1 - beta) d_B(y) |psi(y)|^alpha
                + beta sum_x |H_yx| |psi(x)|^alpha.

    Importance weight:
        omega(y) = sample_mass(y) |psi_theta(y)|^2 / r_tilde(y).

    The physical estimators remain Born-measure expectations; alpha and blur
    only define the auxiliary observed law.
    """

    model: Model
    params: Any
    hamiltonian: Any
    sampler: MCSampler
    sampler_state: Walkers

    n_alpha: int
    n_beta: int
    init_method: str | Any = "hf"

    eloc_eps1: float = 1.0e-3
    eloc_eps2: float = 1.0e-6
    eloc_sample: int = 256

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Any,
        *,
        sampler: MCSampler,
        n_alpha: int,
        n_beta: int,
        key: jax.Array,
        init_method: str | Any = "hf",
        eloc_eps1: float = 1.0e-3,
        eloc_eps2: float = 1.0e-6,
        eloc_sample: int = 256,
    ) -> MCState:
        _, init_key, sample_key = jax.random.split(key, 3)

        params = model.init(
            init_key,
            jnp.zeros((1, 2, hamiltonian.nword), dtype=jnp.uint64),
        )["params"]

        sampler_state = sampler.init(
            params,
            hamiltonian,
            model,
            key=sample_key,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            init_method=init_method,
        )

        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            sampler=sampler,
            sampler_state=sampler_state,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            init_method=init_method,
            eloc_eps1=float(eloc_eps1),
            eloc_eps2=float(eloc_eps2),
            eloc_sample=int(eloc_sample),
        )

    def replace(self, **updates: Any) -> MCState:
        return replace(self, **updates)

    def expect(self) -> tuple[MCState, dict[str, float]]:
        new_state, _, _, stats, _ = self._run(grad=False, geometry=False)
        return new_state, stats

    def expect_and_grad(self, *, geometry: bool = False):
        return self._run(grad=True, geometry=geometry)

    def _run(self, *, grad: bool, geometry: bool):
        """Run one VMC estimator pass.

        The data flow is fixed:

            walkers -> observed samples -> unique kets
            -> Hamiltonian edges -> unique model pool -> weighted estimator.
        """
        timer = utils.Timer()
        sampler_state = self.sampler_state

        if self.sampler.reset_chains:
            with timer("sample"):
                sampler_state = self.sampler.init(
                    self.params,
                    self.hamiltonian,
                    self.model,
                    key=sampler_state.key,
                    n_alpha=int(self.n_alpha),
                    n_beta=int(self.n_beta),
                    init_method=self.init_method,
                    alpha=(
                        float(sampler_state.alpha)
                        if self.sampler.alpha is None
                        else None
                    ),
                    alpha_step=(
                        int(sampler_state.alpha_step)
                        if self.sampler.alpha is None
                        else 0
                    ),
                )

        sampler_state, samples, sample_mass, sampler_stats = self.sampler.draw(
            self.params,
            self.hamiltonian,
            self.model,
            sampler_state,
        )
        alpha = float(sampler_state.alpha)

        timer.add("sample", sampler_stats.get("time_sample", 0.0))
        timer.add("graph", sampler_stats.get("time_graph", 0.0))
        timer.add("forward", sampler_stats.get("time_forward", 0.0))

        with timer("reduce"):
            # count and mass are estimator-local; they never enter chain state.
            kets, _, sample_to_ket = unique_dets(samples)
            n_ket = int(kets.shape[0])

            rdtype = precision.dtype("calc", "real", host=True)
            raw_mass = precision.asarray(
                sample_mass,
                "calc",
                "real",
                host=True,
            )
            mass = np.zeros(n_ket, dtype=rdtype)
            mass2 = np.zeros(n_ket, dtype=rdtype)
            np.add.at(mass, sample_to_ket, raw_mass)
            np.add.at(mass2, sample_to_ket, raw_mass * raw_mass)

            seed = 0
            if self.eloc_sample > 0:
                key, subkey = jax.random.split(sampler_state.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                sampler_state = replace(sampler_state, key=key)

        with timer("graph"):
            # Strong edges are summed exactly:
            # E_loc(i) = H_ii + sum_a H_ia psi(a) / psi(i).
            strong_raw = self.hamiltonian.conns(
                kets,
                float(self.eloc_eps1),
            )
            diag = precision.asarray(
                np.asarray(strong_raw.diags).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            strong_ptr = np.asarray(strong_raw.ket_ptr, dtype=np.int64)
            strong = Edges(
                ket=np.repeat(
                    np.arange(n_ket, dtype=np.int64),
                    np.diff(strong_ptr),
                ),
                bras=np.ascontiguousarray(
                    np.asarray(strong_raw.bras, dtype=np.uint64)
                ),
                bra=np.asarray(strong_raw.bra, dtype=np.int64),
                h=precision.asarray(
                    np.asarray(strong_raw.h),
                    "calc",
                    "real",
                    host=True,
                ),
            )

            empty_bras = np.empty(
                (0, 2, int(self.hamiltonian.nword)),
                dtype=np.uint64,
            )
            weak = Edges(
                ket=np.empty(0, dtype=np.int64),
                bras=empty_bras,
                bra=np.empty(0, dtype=np.int64),
                h=np.empty(0, dtype=rdtype),
            )
            n_conn_weak = 0

            if self.eloc_sample > 0:
                weak_raw = self.hamiltonian.sample_conns(
                    kets,
                    int(self.eloc_sample),
                    eps1=float(self.eloc_eps1),
                    eps2=float(self.eloc_eps2),
                    seed=seed,
                )
                n_conn_weak = int(
                    np.asarray(weak_raw.ket_nconn, dtype=np.int64).sum()
                )
                weak_bras = np.ascontiguousarray(
                    np.asarray(weak_raw.bras, dtype=np.uint64)
                )

                if weak_bras.shape[0] > 0:
                    weak_bras, _, weak_bra = unique_dets(weak_bras)
                    pgen = precision.asarray(
                        np.asarray(weak_raw.pgen),
                        "calc",
                        "real",
                        host=True,
                    )
                    count = precision.asarray(
                        np.asarray(weak_raw.counts),
                        "calc",
                        "real",
                        host=True,
                    )
                    denom = np.maximum(
                        rdtype(self.eloc_sample) * pgen,
                        rdtype(precision.tiny("calc")),
                    )
                    weak = Edges(
                        ket=np.asarray(weak_raw.ket, dtype=np.int64),
                        bras=weak_bras,
                        bra=weak_bra,
                        h=precision.asarray(
                            np.asarray(weak_raw.h),
                            "calc",
                            "real",
                            host=True,
                        ),
                        # Multiplicity / (S pgen) gives an unbiased weak sum.
                        scale=precision.asarray(
                            count / denom,
                            "calc",
                            "real",
                            host=True,
                        ),
                    )

            beta = float(np.clip(self.sampler.blur, 0.0, 1.0))
            blur = None
            blur_degree = np.zeros(n_ket, dtype=rdtype)

            if beta > 0.0:
                blur_eps = float(
                    self.sampler.proposal_eps
                    if self.sampler.blur_eps is None
                    else self.sampler.blur_eps
                )

                if blur_eps == float(self.eloc_eps1):
                    blur = strong
                    blur_degree = precision.asarray(
                        np.asarray(strong_raw.ket_weight),
                        "calc",
                        "real",
                        host=True,
                    )
                else:
                    blur_raw = self.hamiltonian.conns(kets, blur_eps)
                    blur_ptr = np.asarray(blur_raw.ket_ptr, dtype=np.int64)
                    blur = Edges(
                        ket=np.repeat(
                            np.arange(n_ket, dtype=np.int64),
                            np.diff(blur_ptr),
                        ),
                        bras=np.ascontiguousarray(
                            np.asarray(blur_raw.bras, dtype=np.uint64)
                        ),
                        bra=np.asarray(blur_raw.bra, dtype=np.int64),
                        h=precision.asarray(
                            np.asarray(blur_raw.h),
                            "calc",
                            "real",
                            host=True,
                        ),
                    )
                    blur_degree = precision.asarray(
                        np.asarray(blur_raw.ket_weight),
                        "calc",
                        "real",
                        host=True,
                    )

        with timer("reduce"):
            # Every expensive model call receives one global unique pool.
            parts = [kets, strong.bras, weak.bras]
            if blur is not None and blur is not strong:
                parts.append(blur.bras)

            pool, _, inv = unique_dets(np.concatenate(parts, axis=0))
            offset = n_ket
            ket_uid = inv[:offset]
            strong_uid = inv[offset : offset + strong.bras.shape[0]]
            offset += strong.bras.shape[0]
            weak_uid = inv[offset : offset + weak.bras.shape[0]]
            offset += weak.bras.shape[0]

            if blur is None:
                blur_uid = None
            elif blur is strong:
                blur_uid = strong_uid
            else:
                blur_uid = inv[offset : offset + blur.bras.shape[0]]

        with timer("forward"):
            logpsi_pool_jax = utils.apply(
                self.model.logpsi,
                self.params,
                pool,
            )
            jax.block_until_ready(logpsi_pool_jax)
            logpsi_pool = utils.host(logpsi_pool_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[ket_uid], logpsi_pool)
            ket_logabs = precision.asarray(
                np.asarray(to_logabs(ket_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )

            lognu = self._lognu(
                ket_logabs=ket_logabs,
                logpsi_pool=logpsi_pool,
                blur=blur,
                blur_uid=blur_uid,
                blur_degree=blur_degree,
                alpha=alpha,
            )
            eloc = self._eloc(
                diag=diag,
                logpsi_pool=logpsi_pool,
                ket_uid=ket_uid,
                strong=strong,
                strong_uid=strong_uid,
                weak=weak,
                weak_uid=weak_uid,
            )
            w, ess = self._weights(
                mass=mass,
                mass2=mass2,
                ket_logabs=ket_logabs,
                lognu=lognu,
            )

            energy = float(np.real(np.dot(w, eloc)))
            residual = eloc - energy
            variance = float(np.real(np.dot(w, np.abs(residual) ** 2)))

        gradient = None
        geom = None

        if grad:
            with timer("backward"):
                dlogpsi = rdtype(2.0) * w * residual
                cotangent = self.model.cotangent(ket_logpsi, dlogpsi)

                gradient = utils.vjp(
                    self.model.coord,
                    self.params,
                    kets,
                    utils.device(
                        precision.asarray(
                            cotangent,
                            "model",
                            "real",
                            host=True,
                        )
                    ),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    b_log = rdtype(2.0) * np.sqrt(w) * residual
                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=kets,
                        w=utils.device(
                            precision.asarray(
                                w,
                                "sr",
                                "real",
                                host=True,
                            )
                        ),
                        b=utils.device(
                            precision.asarray(
                                self.model.cotangent(ket_logpsi, b_log),
                                "sr",
                                "real",
                                host=True,
                            )
                        ),
                    )

        sampler_state = self._auto_alpha(
            sampler_state=sampler_state,
            ket_logabs=ket_logabs,
            eloc=eloc,
            energy=energy,
            w=w,
        )
        new_state = replace(self, sampler_state=sampler_state)

        n_sample = int(samples.shape[0])
        stats = {
            "energy": energy,
            "variance": variance,
            "accept": float(sampler_stats.get("accept", 0.0)),
            "ess": float(ess),
            "ess_frac": float(ess / max(1, n_sample)),
            "n_sample": float(n_sample),
            "n_unique": float(n_ket),
            "unique_frac": float(n_ket / max(1, n_sample)),
            "n_eval": float(pool.shape[0]),
            "n_conn_eloc": float(strong.h.size),
            "n_conn_weak": float(n_conn_weak),
            "n_conn_proposal": float(
                sampler_stats.get("n_conn_proposal", 0.0)
            ),
            "n_conn_blur": float(
                sampler_stats.get("n_conn_blur", 0.0)
                + (0 if blur is None else blur.h.size)
            ),
            "alpha": alpha,
        }
        stats.update(timer.stats())

        return new_state, energy, gradient, stats, geom

    def _lognu(
        self,
        *,
        ket_logabs: np.ndarray,
        logpsi_pool: Any,
        blur: Edges | None,
        blur_uid: np.ndarray | None,
        blur_degree: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Return the unnormalized density of blurred observations.

        r_tilde(y) =
            (1-beta) d_B(y) |psi(y)|^alpha
            + beta sum_x |H_yx| |psi(x)|^alpha.

        Empty blur rows use the identity observation with unit source mass.
        """
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))
        alpha_r = rdtype(alpha)
        beta = rdtype(np.clip(self.sampler.blur, 0.0, 1.0))
        logrho = precision.asarray(
            alpha_r * ket_logabs,
            "calc",
            "real",
            host=True,
        )

        if beta <= 0.0 or blur is None or blur_uid is None:
            return logrho

        # Empty Hamiltonian rows revert to the identity observation.
        stay_scale = np.where(
            blur_degree > 0.0,
            (rdtype(1.0) - beta) * blur_degree,
            rdtype(1.0),
        )
        log_stay = np.full(ket_logabs.shape[0], -np.inf, dtype=rdtype)
        keep = stay_scale > 0.0
        log_stay[keep] = (
            np.log(np.maximum(stay_scale[keep], tiny)) + logrho[keep]
        )

        if blur.h.size == 0:
            return log_stay

        bra_logabs = precision.asarray(
            np.asarray(
                to_logabs(
                    jax.tree.map(
                        lambda a: a[blur_uid[blur.bra]],
                        logpsi_pool,
                    )
                )
            ).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        abs_h = np.abs(blur.h)
        valid = abs_h > 0.0
        terms = np.full(blur.h.size, -np.inf, dtype=rdtype)
        terms[valid] = (
            alpha_r * bra_logabs[valid]
            + np.log(np.maximum(abs_h[valid], tiny))
        )

        row_count = np.bincount(
            blur.ket,
            minlength=ket_logabs.shape[0],
        ).astype(np.int64)
        row_ptr = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(row_count)]
        )
        log_blur = np.log(beta) + utils.segment_logsumexp(
            row_ptr,
            terms,
            ket_logabs.shape[0],
        )

        return precision.asarray(
            np.logaddexp(log_stay, log_blur),
            "calc",
            "real",
            host=True,
        )

    def _eloc(
        self,
        *,
        diag: np.ndarray,
        logpsi_pool: Any,
        ket_uid: np.ndarray,
        strong: Edges,
        strong_uid: np.ndarray,
        weak: Edges,
        weak_uid: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the semi-stochastic local energy.

        Strong edges are exact. A sampled weak edge contributes

            count * H_ia / (S p_ia) * psi(a) / psi(i).
        """
        eloc = precision.asarray(diag.copy(), "calc", host=True)

        for edge, bra_uid in (
            (strong, strong_uid),
            (weak, weak_uid),
        ):
            if edge.h.size == 0:
                continue

            ratio = precision.asarray(
                np.asarray(
                    to_ratio(
                        jax.tree.map(
                            lambda a: a[bra_uid[edge.bra]],
                            logpsi_pool,
                        ),
                        jax.tree.map(
                            lambda a: a[ket_uid[edge.ket]],
                            logpsi_pool,
                        ),
                    )
                ),
                "calc",
                host=True,
            )
            contribution = edge.h * ratio

            if edge.scale is not None:
                contribution = edge.scale * contribution

            eloc = eloc.astype(
                np.result_type(eloc, contribution),
                copy=False,
            )
            np.add.at(eloc, edge.ket, contribution)

        return precision.asarray(eloc, "calc", host=True)

    def _weights(
        self,
        *,
        mass: np.ndarray,
        mass2: np.ndarray,
        ket_logabs: np.ndarray,
        lognu: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Normalize Born weights and compute the sample-level ESS.

        u_i = |psi(D_i)|^2 / r_tilde(D_i),
        w_i = mass_i u_i / sum_j mass_j u_j.
        """
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        logu = rdtype(2.0) * ket_logabs - lognu
        finite = np.isfinite(logu)
        if not finite.any():
            raise FloatingPointError("all importance weights are non-finite")

        u = np.zeros_like(logu, dtype=rdtype)
        u[finite] = np.exp(logu[finite] - rdtype(np.max(logu[finite])))

        weighted_mass = mass * u
        norm = float(np.sum(weighted_mass))
        if not np.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError("importance weights have zero total mass")

        w = precision.asarray(
            weighted_mass / max(norm, float(tiny)),
            "calc",
            "real",
            host=True,
        )
        ess_denom = float(np.sum(mass2 * u * u))
        ess = float(norm * norm / max(ess_denom, float(tiny)))
        return w, ess

    def _auto_alpha(
        self,
        *,
        sampler_state: Walkers,
        ket_logabs: np.ndarray,
        eloc: np.ndarray,
        energy: float,
        w: np.ndarray,
    ) -> Walkers:
        """Update alpha by KL moment projection.

        The residual-tilted Born law is projected onto
        ``eta_alpha proportional to |psi|^alpha`` by matching
        ``E[log|psi|]``. Robbins-Monro averaging makes adaptation diminish.
        """
        if self.sampler.alpha is not None:
            return sampler_state

        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))
        alpha = float(np.clip(float(sampler_state.alpha), 0.0, 2.0))
        alpha_step = int(sampler_state.alpha_step) + 1

        ell = precision.asarray(
            np.asarray(ket_logabs).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        weight = precision.asarray(
            np.asarray(w).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        resid = precision.asarray(
            np.asarray(np.abs(eloc - energy)).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        valid = np.isfinite(ell) & np.isfinite(weight) & np.isfinite(resid)
        valid &= weight > 0.0
        if not valid.any():
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        ell = ell[valid]
        weight = weight[valid]
        resid = resid[valid]

        target = weight * resid
        target_norm = float(np.sum(target))
        if not np.isfinite(target_norm) or target_norm <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        target_moment = float(np.dot(target, ell) / target_norm)

        # eta_alpha / pi is proportional to exp((alpha - 2) log|psi|).
        log_ratio = (rdtype(alpha) - rdtype(2.0)) * ell
        reference = weight * np.exp(log_ratio - rdtype(np.max(log_ratio)))
        reference_norm = float(np.sum(reference))
        if not np.isfinite(reference_norm) or reference_norm <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        reference_moment = float(
            np.dot(reference, ell) / reference_norm
        )
        reference_var = float(
            np.dot(reference, (ell - rdtype(reference_moment)) ** 2)
            / reference_norm
        )
        if not np.isfinite(reference_var) or reference_var <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        alpha_hat = alpha + (
            target_moment - reference_moment
        ) / reference_var
        alpha_hat = float(np.clip(alpha_hat, 0.0, 2.0))

        rate = 1.0 / float(alpha_step + 1)
        alpha_next = (1.0 - rate) * alpha + rate * alpha_hat

        return replace(
            sampler_state,
            alpha=float(np.clip(alpha_next, 0.0, 2.0)),
            alpha_step=alpha_step,
        )
