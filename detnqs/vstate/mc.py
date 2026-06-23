from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
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
        Hamiltonian proposals use the degree-tilted law
        ``|psi_theta(x)|^alpha s(x)``.

    Observation law:
        Hamiltonian chains or blur use degree-tilted observed density.

    ``eps1`` defines Hamiltonian proposal, blur, and deterministic local
    energy. ``eps2`` is the sampled weak-window lower cutoff.
    """

    model: Model
    params: Any
    H: Any
    sampler: MCSampler
    sampler_state: Chains

    chains: np.ndarray

    eps1: float = 1.0e-3
    eps2: float = 1.0e-6
    eloc_sample: int = 256
    assemble_mode: str = "unique"

    @classmethod
    def init(
        cls,
        model: Model,
        H: Any,
        *,
        sampler: MCSampler,
        key: jax.Array,
        chains: Any | None = None,
        eps1: float = 1.0e-3,
        eps2: float = 1.0e-6,
        eloc_sample: int = 256,
        assemble_mode: str = "unique",
    ) -> MCState:
        """Initialize a Monte Carlo variational state."""
        eps1 = float(eps1)
        eps2 = float(eps2)
        if eps1 < 0.0:
            raise ValueError("eps1 must be nonnegative")
        if eps2 < 0.0:
            raise ValueError("eps2 must be nonnegative")
        if eps2 > eps1:
            raise ValueError("eps2 must be <= eps1")

        if not 0.0 <= float(sampler.blur) <= 1.0:
            raise ValueError("sampler.blur must satisfy 0 <= blur <= 1")

        if int(eloc_sample) < 0:
            raise ValueError("eloc_sample must be nonnegative")
        if assemble_mode not in {"unique", "flat"}:
            raise ValueError("assemble_mode must be 'unique' or 'flat'")

        _, init_key, sample_key = jax.random.split(key, 3)

        params = model.init(init_key, H.sector.zeros(1))["params"]

        if chains is None:
            chains_arr = H.sector.reference(sampler.n_chains)
        else:
            chains_arr = H.sector.asarray(chains)
        if chains_arr.shape[0] != int(sampler.n_chains):
            raise ValueError("chains size must equal sampler.n_chains")
        chains_arr = np.ascontiguousarray(chains_arr)

        sampler_state = sampler.init(
            params,
            H,
            model,
            key=sample_key,
            eps1=eps1,
            chains=chains_arr,
        )

        return cls(
            model=model,
            params=params,
            H=H,
            sampler=sampler,
            sampler_state=sampler_state,
            chains=chains_arr,
            eps1=eps1,
            eps2=eps2,
            eloc_sample=int(eloc_sample),
            assemble_mode=assemble_mode,
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
                    self.H,
                    self.model,
                    key=sampler_state.key,
                    eps1=float(self.eps1),
                    chains=self.chains,
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
        else:
            alpha = (
                float(sampler_state.alpha)
                if self.sampler.alpha is None
                else float(self.sampler.alpha)
            )

            with timer("forward"):
                unique, _, inv = self.H.sector.unique(sampler_state.x)
                value = utils.apply(self.model.logabs, self.params, unique)
                jax.block_until_ready(value)

                unique_logabs = precision.asarray(
                    np.asarray(utils.host(value)).reshape(-1),
                    "calc",
                    "real",
                    host=True,
                )
                sampler_state = replace(
                    sampler_state,
                    logabs=precision.asarray(
                        unique_logabs[inv],
                        "calc",
                        "real",
                        host=True,
                    ),
                    alpha=alpha,
                )

        sampler_state, samples, sample_mass, sampler_stats = self.sampler.draw(
            self.params,
            self.H,
            self.model,
            sampler_state,
            eps1=float(self.eps1),
        )
        alpha = float(sampler_state.alpha)

        timer.add("sample", sampler_stats.get("time_sample", 0.0))
        timer.add("conns", sampler_stats.get("time_conns", 0.0))
        timer.add("forward", sampler_stats.get("time_forward", 0.0))

        with timer("reduce"):
            ket, _, sample_to_ket = self.H.sector.unique(samples)
            n_ket = int(ket.shape[0])

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
            conn = self.H.local_conn(
                ket,
                float(self.eps1),
                float(self.eps2),
                int(self.eloc_sample),
                seed=seed,
                assemble_mode=self.assemble_mode,
            )
            bra = np.asarray(conn.bra, dtype=np.uint64)
            strong_ptr = np.asarray(conn.strong_ptr, dtype=np.int64)
            strong_bra = np.asarray(conn.strong_bra, dtype=np.int64)
            strong_h = precision.asarray(
                np.asarray(conn.strong_h),
                "calc",
                host=True,
            )
            strong_w = precision.asarray(
                np.asarray(conn.strong_degree),
                "calc",
                "real",
                host=True,
            )
            diag = precision.asarray(np.asarray(conn.diag), "calc", host=True)
            weak_ptr = np.asarray(conn.weak_ptr, dtype=np.int64)
            weak_bra = np.asarray(conn.weak_bra, dtype=np.int64)
            weak_h = precision.asarray(np.asarray(conn.weak_h), "calc", host=True)
            weak_count = precision.asarray(
                np.asarray(conn.weak_count),
                "calc",
                "real",
                host=True,
            )
            weak_w = precision.asarray(
                np.asarray(conn.weak_degree),
                "calc",
                "real",
                host=True,
            )

        with timer("forward"):
            bra_logpsi_jax = utils.apply(
                self.model.logpsi,
                self.params,
                bra,
            )
            jax.block_until_ready(bra_logpsi_jax)
            bra_logpsi = utils.host(bra_logpsi_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], bra_logpsi)

            bra_logabs = precision.asarray(
                np.asarray(to_logabs(bra_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            ket_logabs = bra_logabs[:n_ket]

            rdtype = precision.dtype("calc", "real", host=True)
            tiny = rdtype(precision.tiny("calc"))
            alpha_r = rdtype(alpha)
            beta = rdtype(float(self.sampler.blur))

            ket_idx = np.repeat(
                np.arange(n_ket, dtype=np.int64),
                np.diff(strong_ptr),
            )
            logrho = precision.asarray(
                alpha_r * ket_logabs,
                "calc",
                "real",
                host=True,
            )

            # Exact strong-window local energy.
            eloc = diag.copy()
            if strong_h.size:
                ratio = precision.asarray(
                    np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[strong_bra], bra_logpsi),
                            jax.tree.map(lambda a: a[ket_idx], bra_logpsi),
                        )
                    ),
                    "calc",
                    host=True,
                )
                contribution = strong_h * ratio
                eloc = eloc.astype(np.result_type(eloc, contribution), copy=False)
                np.add.at(eloc, ket_idx, contribution)

            # Degree-tilted observed density for Hamiltonian chains or blur.
            tilted_obs = self.sampler.proposal == "ham" or beta > 0.0
            if not tilted_obs:
                lognu = logrho
            else:
                stay_scale = np.where(
                    strong_w > 0.0,
                    (rdtype(1.0) - beta) * strong_w,
                    rdtype(1.0),
                )
                log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
                nonzero = stay_scale > 0.0
                log_stay[nonzero] = (
                    np.log(np.maximum(stay_scale[nonzero], tiny))
                    + logrho[nonzero]
                )

                if beta > 0.0 and strong_h.size:
                    terms = (
                        alpha_r * bra_logabs[strong_bra]
                        + np.log(np.maximum(np.abs(strong_h), tiny))
                    )
                    log_blur = (
                        np.log(beta)
                        + utils.segment_logsumexp(strong_ptr, terms, n_ket)
                    )
                    lognu = precision.asarray(
                        np.logaddexp(log_stay, log_blur),
                        "calc",
                        "real",
                        host=True,
                    )
                else:
                    lognu = log_stay

            # Unbiased weak-window local energy correction.
            if weak_h.size and int(self.eloc_sample) > 0:
                weak_ket = np.repeat(
                    np.arange(n_ket, dtype=np.int64),
                    np.diff(weak_ptr),
                )
                ratio = precision.asarray(
                    np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[weak_bra], bra_logpsi),
                            jax.tree.map(lambda a: a[weak_ket], bra_logpsi),
                        )
                    ),
                    "calc",
                    host=True,
                )
                scale = weak_w[weak_ket] / np.maximum(
                    rdtype(int(self.eloc_sample)) * np.abs(weak_h),
                    tiny,
                )
                contribution = weak_count * scale * weak_h * ratio
                eloc = eloc.astype(np.result_type(eloc, contribution), copy=False)
                np.add.at(eloc, weak_ket, contribution)

            eloc = precision.asarray(eloc, "calc", host=True)

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
                    ket,
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
                        x=ket,
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

        n_sample = int(samples.shape[0])
        n_strong = int(strong_bra.size)
        n_weak = int(weak_bra.size)
        n_forward_raw = n_ket + n_strong + n_weak
        forward_frac = float(bra.shape[0]) / max(1, n_forward_raw)

        unique_frac = 1.0 - float(n_ket) / max(1, n_sample)
        n_conn = (
            float(sampler_stats.get("n_conn", 0.0))
            + float(n_strong)
            + float(n_weak)
        )

        stats = {
            "energy": energy,
            "variance": variance,
            "accept": float(sampler_stats.get("accept", 0.0)),
            "ess": float(ess),
            "ess_frac": float(ess / max(1, n_sample)),
            "n_sample": float(n_sample),
            "n_unique": float(n_ket),
            "n_forward": float(bra.shape[0]),
            "unique_frac": float(unique_frac),
            "forward_frac": float(forward_frac),
            "n_conn": n_conn,
            "n_conn_weak": float(n_weak),
            "alpha": alpha,
        }
        stats.update(timer.stats())

        return new_state, energy, gradient, stats, geom

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

        reference_moment = float(np.dot(reference, ell) / reference_norm)
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
