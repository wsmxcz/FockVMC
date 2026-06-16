from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import libdet
import numpy as np

from detnqs import utils
from ..model.base import Model
from ..model.base import to_logabs
from ..model.base import to_ratio
from ..optimizer import Geometry
from ..sampler.mcmc import Chains
from ..sampler.mcmc import MCSampler
from ..utils import precision
from .base import VState


@dataclass(slots=True)
class MCState(VState):
    """Monte Carlo variational state.

    Physical measure:
        pi_theta(x) proportional to |psi_theta(x)|^2.

    Markov reference:
        eta_alpha(x) proportional to |psi_theta(x)|^alpha.

    Observation law:
        x ~ eta_alpha, y ~ B(y|x).

    Hamiltonian Fock-VMC path:
        proposal_eps == blur_eps == eloc_eps1.

    The same strong Hamiltonian connections define proposal, blur, and the
    deterministic part of the local energy.
    """

    model: Model
    params: Any
    hamiltonian: Any
    sampler: MCSampler
    sampler_state: Chains

    n_alpha: int
    n_beta: int
    chain_init: str | Any = "hf"

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
        chain_init: str | Any = "hf",
        eloc_eps1: float = 1.0e-3,
        eloc_eps2: float = 1.0e-6,
        eloc_sample: int = 256,
    ) -> MCState:
        """Initialize a Monte Carlo variational state."""
        eps = float(eloc_eps1)

        if sampler.proposal not in {"ham", "single"}:
            raise ValueError("sampler.proposal must be 'ham' or 'single'")

        if sampler.blur_eps is None:
            raise ValueError("sampler.blur_eps must be explicit")

        if float(sampler.proposal_eps) != eps or float(sampler.blur_eps) != eps:
            raise ValueError(
                "Fock-VMC requires proposal_eps == blur_eps == eloc_eps1"
            )

        if not 0.0 <= float(sampler.blur) <= 1.0:
            raise ValueError("sampler.blur must satisfy 0 <= blur <= 1")

        if float(eloc_eps2) > eps:
            raise ValueError("eloc_eps2 must not exceed eloc_eps1")

        if int(eloc_sample) < 0:
            raise ValueError("eloc_sample must be nonnegative")

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
            chain_init=chain_init,
        )

        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            sampler=sampler,
            sampler_state=sampler_state,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            chain_init=chain_init,
            eloc_eps1=eps,
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
        """Run one VMC estimator pass."""
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
                    chain_init=self.chain_init,
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
        timer.add("conns", sampler_stats.get("time_conns", 0.0))
        timer.add("forward", sampler_stats.get("time_forward", 0.0))

        with timer("reduce"):
            kets, _, sample_to_ket = libdet.unique_dets(samples)
            n_ket = int(kets.shape[0])

            raw_mass = precision.asarray(
                sample_mass,
                "calc",
                "real",
                host=True,
            )

            mass = np.bincount(
                sample_to_ket,
                weights=raw_mass,
                minlength=n_ket,
            )
            mass2 = np.bincount(
                sample_to_ket,
                weights=raw_mass * raw_mass,
                minlength=n_ket,
            )

            mass = precision.asarray(mass, "calc", "real", host=True)
            mass2 = precision.asarray(mass2, "calc", "real", host=True)

            seed = 0
            if self.eloc_sample > 0:
                key, subkey = jax.random.split(sampler_state.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                sampler_state = replace(sampler_state, key=key)

        with timer("conns"):
            conns = self.hamiltonian.conns(
                kets,
                float(self.eloc_eps1),
                sample=int(self.eloc_sample),
                sample_eps=float(self.eloc_eps2),
                seed=seed,
            )

            det_pool = np.ascontiguousarray(
                np.asarray(conns.dets, dtype=np.uint64)
            )

        with timer("forward"):
            pool_logpsi_jax = utils.apply(
                self.model.logpsi,
                self.params,
                det_pool,
            )
            jax.block_until_ready(pool_logpsi_jax)
            pool_logpsi = utils.host(pool_logpsi_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], pool_logpsi)

            pool_logabs = precision.asarray(
                np.asarray(to_logabs(pool_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            ket_logabs = pool_logabs[:n_ket]

            lognu, eloc = self._local_estimate(
                ket_logabs=ket_logabs,
                pool_logabs=pool_logabs,
                pool_logpsi=pool_logpsi,
                conns=conns,
                alpha=alpha,
                n_sample=int(self.eloc_sample),
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
                dlogpsi = precision.asarray(
                    2.0 * w * residual,
                    "calc",
                    host=True,
                )
                cotangent = self.model.cotangent(ket_logpsi, dlogpsi)

                gradient = utils.vjp(
                    self.model.coord,
                    self.params,
                    kets,
                    utils.device(
                        precision.asarray(
                            cotangent,
                            "model",
                            host=True,
                        )
                    ),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    b_log = precision.asarray(
                        2.0 * np.sqrt(w) * residual,
                        "sr",
                        "real",
                        host=True,
                    )

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

        n_forward = int(det_pool.shape[0])
        n_strong = int(np.asarray(conns.bra_idx).size)
        n_weak = int(np.asarray(conns.sample_bra_idx).size)
        n_strong_h = int(np.asarray(conns.h).size)
        n_weak_h = int(np.asarray(conns.sample_h).size)
        n_forward_raw = n_ket + n_strong + n_weak
        forward_frac = float(n_forward) / max(1, n_forward_raw)

        n_sample = int(samples.shape[0])
        unique_frac = 1.0 - float(n_ket) / max(1, n_sample)

        n_conn = (
            float(sampler_stats.get("n_conn", 0.0))
            + float(n_strong_h)
        )

        stats = {
            "energy": energy,
            "variance": variance,
            "accept": float(sampler_stats.get("accept", 0.0)),
            "ess": float(ess),
            "ess_frac": float(ess / max(1, n_sample)),
            "n_sample": float(n_sample),
            "n_unique": float(n_ket),
            "n_forward": float(det_pool.shape[0]),
            "unique_frac": float(unique_frac),
            "forward_frac": float(forward_frac),
            "n_conn": n_conn,
            "n_conn_weak": float(n_weak_h),
            "alpha": alpha,
        }
        stats.update(timer.stats())

        return new_state, energy, gradient, stats, geom

    def _local_estimate(
        self,
        *,
        ket_logabs: np.ndarray,
        pool_logabs: np.ndarray,
        pool_logpsi: Any,
        conns: Any,
        alpha: float,
        n_sample: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate observation density and local energy."""
        rdtype = precision.dtype("calc", "real", host=True)
        tiny = rdtype(precision.tiny("calc"))

        alpha_r = rdtype(alpha)
        beta = rdtype(float(self.sampler.blur))
        n_ket = int(ket_logabs.size)

        ket_ptr = np.asarray(conns.ket_ptr, dtype=np.int64)
        ket = np.repeat(
            np.arange(n_ket, dtype=np.int64),
            np.diff(ket_ptr),
        )
        bra = np.asarray(conns.bra_idx, dtype=np.int64)

        h = precision.asarray(
            np.asarray(conns.h),
            "calc",
            host=True,
        )
        diag = precision.asarray(
            np.asarray(conns.diag),
            "calc",
            host=True,
        )

        logrho = precision.asarray(
            alpha_r * ket_logabs,
            "calc",
            "real",
            host=True,
        )

        eloc = diag.copy()

        if h.size:
            bra_logpsi = jax.tree.map(lambda a: a[bra], pool_logpsi)
            ket_logpsi = jax.tree.map(lambda a: a[ket], pool_logpsi)

            ratio = precision.asarray(
                np.asarray(to_ratio(bra_logpsi, ket_logpsi)),
                "calc",
                host=True,
            )

            contribution = h * ratio
            eloc = eloc.astype(
                np.result_type(eloc, contribution),
                copy=False,
            )
            np.add.at(eloc, ket, contribution)

        if beta <= 0.0:
            lognu = logrho

        else:
            weight = precision.asarray(
                np.asarray(conns.weight),
                "calc",
                "real",
                host=True,
            )

            stay_scale = np.where(
                weight > 0.0,
                (rdtype(1.0) - beta) * weight,
                rdtype(1.0),
            )

            log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
            nonzero = stay_scale > 0.0
            log_stay[nonzero] = (
                np.log(np.maximum(stay_scale[nonzero], tiny))
                + logrho[nonzero]
            )

            if h.size:
                abs_h = np.abs(h)
                terms = (
                    alpha_r * pool_logabs[bra]
                    + np.log(np.maximum(abs_h, tiny))
                )

                log_blur = (
                    np.log(beta)
                    + utils.segment_logsumexp(ket_ptr, terms, n_ket)
                )
                lognu = precision.asarray(
                    np.logaddexp(log_stay, log_blur),
                    "calc",
                    "real",
                    host=True,
                )

            else:
                lognu = log_stay

        sample_h = precision.asarray(
            np.asarray(conns.sample_h) if n_sample > 0 else np.empty(0),
            "calc",
            host=True,
        )

        if sample_h.size and n_sample > 0:
            sample_ket_ptr = np.asarray(
                conns.sample_ket_ptr,
                dtype=np.int64,
            )
            sample_ket = np.repeat(
                np.arange(n_ket, dtype=np.int64),
                np.diff(sample_ket_ptr),
            )
            sample_bra = np.asarray(
                conns.sample_bra_idx,
                dtype=np.int64,
            )

            ratio = precision.asarray(
                np.asarray(
                    to_ratio(
                        jax.tree.map(lambda a: a[sample_bra], pool_logpsi),
                        jax.tree.map(lambda a: a[sample_ket], pool_logpsi),
                    )
                ),
                "calc",
                host=True,
            )

            count = precision.asarray(
                np.asarray(conns.sample_count),
                "calc",
                "real",
                host=True,
            )
            sample_weight = precision.asarray(
                np.asarray(conns.sample_weight),
                "calc",
                "real",
                host=True,
            )

            scale = count * sample_weight[sample_ket] / np.maximum(
                rdtype(n_sample) * np.abs(sample_h),
                tiny,
            )

            contribution = scale * sample_h * ratio
            eloc = eloc.astype(
                np.result_type(eloc, contribution),
                copy=False,
            )
            np.add.at(eloc, sample_ket, contribution)

        return lognu, precision.asarray(eloc, "calc", host=True)

    def _weights(
        self,
        *,
        mass: np.ndarray,
        mass2: np.ndarray,
        ket_logabs: np.ndarray,
        lognu: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Normalize Born weights and compute sample-level ESS."""
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
        sampler_state: Chains,
        ket_logabs: np.ndarray,
        eloc: np.ndarray,
        energy: float,
        w: np.ndarray,
    ) -> Chains:
        """Update alpha by KL moment projection."""
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

        resid_weight = weight * resid
        resid_norm = float(np.sum(resid_weight))

        if not np.isfinite(resid_norm) or resid_norm <= float(tiny):
            return replace(
                sampler_state,
                alpha=alpha,
                alpha_step=alpha_step,
            )

        resid_moment = float(np.dot(resid_weight, ell) / resid_norm)

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
            resid_moment - reference_moment
        ) / reference_var
        alpha_hat = float(np.clip(alpha_hat, 0.0, 2.0))

        rate = 1.0 / float(alpha_step + 1)
        alpha_next = (1.0 - rate) * alpha + rate * alpha_hat

        return replace(
            sampler_state,
            alpha=float(np.clip(alpha_next, 0.0, 2.0)),
            alpha_step=alpha_step,
        )
