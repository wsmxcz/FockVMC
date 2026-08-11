from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..model import Model
from ..model.base import to_logabs, to_ratio
from ..optimizer.base import Geometry
from ..sampler import ChainState, MCSampler
from ..utils import Timer, batch, math, precision, stats, tree


@dataclass(slots=True)
class MCState:
    """Monte Carlo variational state."""

    model: Model
    params: Any
    hamiltonian: Any
    sampler: MCSampler
    sampler_state: ChainState
    chains: np.ndarray

    eps1: float = 1.0e-3
    eps2: float = 1.0e-6
    eloc_sample: int = 256

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Any,
        *,
        sampler: MCSampler,
        chains: Any,
        key: jax.Array,
        eps1: float = 1.0e-3,
        eps2: float = 1.0e-6,
        eloc_sample: int = 256,
    ) -> MCState:
        eps1 = float(eps1)
        eps2 = float(eps2)
        eloc_sample = int(eloc_sample)
        if not 0.0 <= eps2 <= eps1:
            raise ValueError("screening requires 0 <= eps2 <= eps1")
        if eloc_sample < 0:
            raise ValueError("eloc_sample must be nonnegative")
        if eps2 < eps1 and (eps2 == 0.0 or eloc_sample == 0):
            raise ValueError("weak sampling requires eps2 > 0 and eloc_sample > 0")

        _, init_key, sample_key = jax.random.split(key, 3)
        params = model.init(init_key, hamiltonian.sector.zeros(1))["params"]

        chains_arr = hamiltonian.sector.asarray(chains)
        if chains_arr.shape[0] != sampler.n_chains:
            raise ValueError("chains size must equal sampler.n_chains")
        chains_arr = np.ascontiguousarray(chains_arr)

        sampler_state = sampler.init(
            params,
            hamiltonian,
            model,
            key=sample_key,
            eps1=eps1,
            chains=chains_arr,
        )

        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            sampler=sampler,
            sampler_state=sampler_state,
            chains=chains_arr,
            eps1=eps1,
            eps2=eps2,
            eloc_sample=eloc_sample,
        )

    def replace(self, **updates: Any) -> MCState:
        return replace(self, **updates)

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "sampler_state": {
                "key": self.sampler_state.key,
                "x": self.sampler_state.x,
                "logabs": self.sampler_state.logabs,
                "alpha": np.asarray(self.sampler_state.alpha),
            },
            "chains": self.chains,
        }

    def _restore(self, data: dict[str, Any]) -> MCState:
        saved = data["sampler_state"]
        sampler_state = ChainState(
            key=jax.device_put(saved["key"]),
            x=np.ascontiguousarray(saved["x"], dtype=np.uint64),
            logabs=precision.cast(saved["logabs"], "calc", "real", host=True),
            alpha=float(np.asarray(saved["alpha"])),
        )
        return replace(
            self,
            params=data["params"],
            sampler_state=sampler_state,
            chains=np.ascontiguousarray(data["chains"], dtype=np.uint64),
        )

    def expect(
        self,
        *,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
        data: bool = False,
    ):
        """Estimate energy and observables from auxiliary samples."""
        result = self._run(
            grad=False,
            geometry=False,
            obs=obs,
            profile=profile,
            data=data,
        )
        if data:
            new_state, _, _, out, _, sample_data = result
            return new_state, out, sample_data
        new_state, _, _, out, _ = result
        return new_state, out

    def expect_and_grad(
        self,
        *,
        geometry: bool = False,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
    ):
        """Estimate energy, gradient, and optional SR geometry."""
        return self._run(
            grad=True,
            geometry=geometry,
            obs=obs,
            profile=profile,
            data=False,
        )

    def _update_alpha(
        self,
        sampler_state: ChainState,
        mass: np.ndarray,
        score: np.ndarray,
        score2: np.ndarray,
        residual: np.ndarray,
        weight: np.ndarray,
    ) -> ChainState:
        """Update alpha by damped local KL moment projection."""
        if self.sampler.alpha is not None:
            return sampler_state

        rdtype = precision.real("calc", host=True)
        tiny = rdtype(precision.tiny("calc"))
        alpha = float(sampler_state.alpha)

        target = weight * np.abs(residual)
        norm_t = float(np.sum(target))
        norm_m = float(np.sum(mass))
        if norm_t <= float(tiny) or norm_m <= float(tiny):
            return sampler_state

        mu_s = float(np.dot(target, score) / norm_t)
        nu_s = float(np.dot(mass, score) / norm_m)
        nu_q = float(np.dot(mass, score2) / norm_m)
        info = nu_q - nu_s * nu_s
        if not np.isfinite(info) or info <= float(tiny):
            return sampler_state

        # KL projection with fixed under-relaxation.
        ahat = float(np.clip(alpha + (mu_s - nu_s) / info, 0.0, 2.0))
        rate = 0.02
        alpha = (1.0 - rate) * alpha + rate * ahat
        return replace(sampler_state, alpha=float(np.clip(alpha, 0.0, 2.0)))

    def _run(
        self,
        *,
        grad: bool,
        geometry: bool,
        obs: Mapping[str, Any] | None,
        profile: bool,
        data: bool = False,
    ):
        timer = Timer(enabled=profile)
        sampler_state = self.sampler_state
        obs = {} if obs is None else dict(obs)

        if self.sampler.reset_chains:
            sampler_state = self.sampler.init(
                self.params,
                self.hamiltonian,
                self.model,
                key=sampler_state.key,
                eps1=self.eps1,
                chains=self.chains,
                alpha=sampler_state.alpha,
                timer=timer,
            )
        else:
            with timer("unique"):
                unique, _, inv = self.hamiltonian.sector.unique(sampler_state.x)

            with timer("forward"):
                value = batch.apply(self.model.logabs, self.params, unique)
                jax.block_until_ready(value)
                logabs = precision.host(value, "calc", "real").reshape(-1)

            with timer("reduce"):
                sampler_state = replace(
                    sampler_state,
                    logabs=precision.cast(logabs[inv], "calc", "real", host=True),
                )

        sampler_state, observations, observation_mass, sample_stats = (
            self.sampler.draw(
                self.params,
                self.hamiltonian,
                self.model,
                sampler_state,
                eps1=float(self.eps1),
                profile=profile,
                timer=timer,
            )
        )

        with timer("unique"):
            ket, _, observation_to_ket = self.hamiltonian.sector.unique(observations)
            n_ket = int(ket.shape[0])

        with timer("reduce"):
            raw_mass = precision.cast(observation_mass, "calc", "real", host=True)
            mass = np.bincount(observation_to_ket, weights=raw_mass, minlength=n_ket)
            mass2 = np.bincount(
                observation_to_ket,
                weights=raw_mass * raw_mass,
                minlength=n_ket,
            )
            mass = precision.cast(mass, "calc", "real", host=True)
            mass2 = precision.cast(mass2, "calc", "real", host=True)

            seed = 0
            if self.eps2 < self.eps1:
                key, subkey = jax.random.split(sampler_state.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                sampler_state = replace(sampler_state, key=key)

        with timer("conns"):
            conn = self.hamiltonian.local_conn(
                ket,
                self.eps1,
                self.eps2,
                self.eloc_sample,
                seed=seed,
            )
            n_strong = conn.strong_h.size
            n_weak = conn.weak_coeff.size
            bra_parts = [conn.bra]
            obs_conn = []
            start = conn.bra.shape[0]

            for name, op in obs.items():
                o_diag, o_ptr, o_bra, o_value = op.local_conn(ket)
                stop = start + o_bra.shape[0]
                obs_conn.append((
                    str(name),
                    o_diag,
                    o_ptr,
                    slice(start, stop),
                    o_value,
                ))
                bra_parts.append(o_bra)
                start = stop

            bra = (
                bra_parts[0]
                if len(bra_parts) == 1
                else np.concatenate(bra_parts)
            )
            n_forward = bra.shape[0]

        with timer("forward"):
            bra_logpsi = batch.apply(self.model.logpsi, self.params, bra)
            jax.block_until_ready(bra_logpsi)
            bra_logpsi = tree.host(bra_logpsi)

        with timer("reduce"):
            rdtype = precision.real("calc", host=True)
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], bra_logpsi)
            ket_logabs = np.asarray(
                to_logabs(ket_logpsi),
                dtype=rdtype,
            ).reshape(-1)
            eloc_dtype = (
                precision.complex("calc", host=True)
                if any(np.iscomplexobj(a) for a in jax.tree.leaves(ket_logpsi))
                else rdtype
            )
            strong_h = np.asarray(conn.strong_h, dtype=rdtype)
            weak_coeff = np.asarray(conn.weak_coeff, dtype=rdtype)
            strong_degree = np.asarray(conn.strong_degree, dtype=rdtype)
            strong_slice = slice(n_ket, n_ket + n_strong)
            weak_slice = slice(
                n_ket + n_strong,
                n_ket + n_strong + n_weak,
            )
            strong_count = np.diff(conn.strong_ptr)
            strong_row = np.flatnonzero(strong_count)
            strong_start = conn.strong_ptr[strong_row]
            strong_source = np.repeat(
                np.arange(n_ket, dtype=np.int32),
                strong_count,
            )

            tiny = rdtype(precision.tiny("calc"))
            auto = self.sampler.alpha is None
            alpha_used = sampler_state.alpha if auto else self.sampler.alpha
            alpha = rdtype(alpha_used)
            beta = rdtype(self.sampler.blur)
            tilted = self.sampler.proposal == "ham" or beta > 0.0
            source_logamp = alpha * ket_logabs

            eloc = np.array(conn.diag, dtype=eloc_dtype, copy=True)
            if n_strong:
                strong_logpsi = jax.tree.map(
                    lambda a: a[strong_slice],
                    bra_logpsi,
                )
                ratio = np.asarray(
                    to_ratio(
                        strong_logpsi,
                        jax.tree.map(
                            lambda a: a[strong_source],
                            ket_logpsi,
                        ),
                    ),
                    dtype=eloc_dtype,
                )
                eloc[strong_row] += np.add.reduceat(
                    strong_h * ratio,
                    strong_start,
                )

            if not tilted:
                log_observation = source_logamp
                if auto:
                    score = ket_logabs
                    score2 = ket_logabs * ket_logabs
            else:
                stay = np.where(
                    strong_degree > 0.0,
                    (rdtype(1.0) - beta) * strong_degree,
                    rdtype(1.0),
                )
                log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
                active = stay > 0.0
                log_stay[active] = (
                    np.log(np.maximum(stay[active], tiny))
                    + source_logamp[active]
                )

                if beta > 0.0 and n_strong:
                    strong_logabs = np.asarray(
                        to_logabs(strong_logpsi),
                        dtype=rdtype,
                    ).reshape(-1)
                    terms = alpha * strong_logabs + np.log(
                        np.maximum(np.abs(strong_h), tiny)
                    )
                    log_blur = np.log(beta) + math.segment_logsumexp(
                        conn.strong_ptr,
                        terms,
                        n_ket,
                    )
                    log_observation = np.logaddexp(log_stay, log_blur)
                else:
                    log_observation = log_stay

                if auto:
                    # S=d_alpha log r_alpha is the latent log-amplitude moment.
                    finite = np.isfinite(log_stay) & np.isfinite(log_observation)
                    stay_weight = np.zeros(n_ket, dtype=rdtype)
                    stay_weight[finite] = np.exp(
                        log_stay[finite] - log_observation[finite]
                    )
                    score = stay_weight * ket_logabs
                    score2 = score * ket_logabs

                    if beta > 0.0 and n_strong:
                        term_log = (
                            np.log(beta)
                            + terms
                            - log_observation[strong_source]
                        )
                        finite = np.isfinite(term_log)
                        conn_weight = np.zeros_like(term_log, dtype=rdtype)
                        conn_weight[finite] = np.exp(term_log[finite])
                        score[strong_row] += np.add.reduceat(
                            conn_weight * strong_logabs,
                            strong_start,
                        )
                        score2[strong_row] += np.add.reduceat(
                            conn_weight * strong_logabs * strong_logabs,
                            strong_start,
                        )

            if n_weak:
                weak_count = np.diff(conn.weak_ptr)
                weak_row = np.flatnonzero(weak_count)
                weak_start = conn.weak_ptr[weak_row]
                weak_source = np.repeat(
                    np.arange(n_ket, dtype=np.int32),
                    weak_count,
                )
                ratio = np.asarray(
                    to_ratio(
                        jax.tree.map(
                            lambda a: a[weak_slice],
                            bra_logpsi,
                        ),
                        jax.tree.map(
                            lambda a: a[weak_source],
                            ket_logpsi,
                        ),
                    ),
                    dtype=eloc_dtype,
                )
                eloc[weak_row] += np.add.reduceat(
                    weak_coeff * ratio,
                    weak_start,
                )

            # Reweight the observed auxiliary law back to the Born target pi.
            weight, weight_stats = stats.weight(
                mass,
                mass2,
                ket_logabs,
                log_observation,
            )
            energy, energy_stats = stats.eloc(weight, eloc)
            residual = eloc - energy
            if auto:
                sampler_state = self._update_alpha(
                    sampler_state,
                    mass,
                    score,
                    score2,
                    residual,
                    weight,
                )

            obs_stats: dict[str, float] = {}
            obs_data = {}
            for name, o_diag, o_ptr, o_slice, o_value in obs_conn:
                o_dtype = (
                    precision.complex("calc", host=True)
                    if np.iscomplexobj(o_diag) or np.iscomplexobj(o_value)
                    else eloc_dtype
                )
                o_value = np.asarray(o_value, dtype=o_dtype)
                oloc = np.array(o_diag, dtype=o_dtype, copy=True)
                if o_value.size:
                    o_count = np.diff(o_ptr)
                    o_row = np.flatnonzero(o_count)
                    o_start = o_ptr[o_row]
                    o_source = np.repeat(
                        np.arange(n_ket, dtype=np.int32),
                        o_count,
                    )
                    ratio = np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[o_slice], bra_logpsi),
                            jax.tree.map(
                                lambda a: a[o_source],
                                ket_logpsi,
                            ),
                        ),
                        dtype=eloc_dtype,
                    )
                    oloc[o_row] += np.add.reduceat(
                        o_value * ratio,
                        o_start,
                    )

                obs_stats.update(stats.observable(name, weight, oloc))
                if data:
                    obs_data[name] = oloc

        gradient = None
        geom = None
        if grad:
            with timer("backward"):
                dlogpsi = precision.cast(
                    2.0 * weight * residual,
                    "calc",
                    host=True,
                )
                cotangent = self.model.cotangent(ket_logpsi, dlogpsi)
                gradient = batch.vjp(
                    self.model.coord,
                    self.params,
                    ket,
                    precision.device(cotangent, "model"),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    b_log = precision.cast(
                        2.0 * np.sqrt(weight) * residual,
                        "sr",
                        host=True,
                    )
                    geom = Geometry(
                        params=self.params,
                        coord=self.model.coord,
                        x=ket,
                        weight=precision.device(weight, "sr", "real"),
                        b=precision.device(
                            self.model.cotangent(ket_logpsi, b_log),
                            "sr",
                        ),
                    )

        new_state = replace(self, sampler_state=sampler_state)
        n_sample = int(observations.shape[0])

        out = {
            **energy_stats,
            **weight_stats,
            **obs_stats,
            "alpha": float(alpha_used),
            "accept": float(sample_stats.get("accept", 0.0)),
            "n_forward": float(n_forward),
            "unique_frac": float(n_ket) / max(1.0, float(n_sample)),
        }
        if profile:
            out.update(timer.stats())
            if "n_conn" in sample_stats:
                out["n_conn"] = float(sample_stats["n_conn"])

        if data:
            return (
                new_state,
                energy,
                gradient,
                out,
                geom,
                {
                    "x": ket,
                    "weight": weight,
                    "eloc": eloc,
                    "obs": obs_data,
                },
            )
        return new_state, energy, gradient, out, geom
