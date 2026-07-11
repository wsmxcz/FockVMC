from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..model import Model, to_logabs, to_ratio
from ..optimizer import Geometry
from ..sampler import Chains, MCSampler
from ..utils import Timer, batch, checkpoint, math, precision, stats, tree
from .base import VState


@dataclass(slots=True)
class MCState(VState):
    """Monte Carlo variational state."""

    model: Model
    params: Any
    H: Any
    sampler: MCSampler
    sampler_state: Chains
    chains: np.ndarray

    eps1: float = 1.0e-3
    eps2: float = 1.0e-6
    eloc_sample: int = 256

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
    ) -> MCState:
        eps1 = float(eps1)
        eps2 = float(eps2)
        eloc_sample = int(eloc_sample)
        if eps1 < 0.0:
            raise ValueError("eps1 must be nonnegative")
        if eps2 < 0.0:
            raise ValueError("eps2 must be nonnegative")
        if eps2 > eps1:
            raise ValueError("eps2 must be <= eps1")
        if eloc_sample < 0:
            raise ValueError("eloc_sample must be nonnegative")

        _, init_key, sample_key = jax.random.split(key, 3)
        params = model.init(init_key, H.sector.zeros(1))["params"]

        chains_arr = (
            H.sector.reference(sampler.n_chains)
            if chains is None
            else H.sector.asarray(chains)
        )
        if chains_arr.shape[0] != sampler.n_chains:
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
            eloc_sample=eloc_sample,
        )

    def replace(self, **updates: Any) -> MCState:
        return replace(self, **updates)

    def expect(
        self,
        *,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
        data: bool = False,
    ):
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
        return self._run(
            grad=True,
            geometry=geometry,
            obs=obs,
            profile=profile,
            data=False,
        )

    def save(self, file: str | Path) -> Path:
        data = {
            "params": self.params,
            "sampler_state": {
                "key": self.sampler_state.key,
                "x": self.sampler_state.x,
                "logabs": self.sampler_state.logabs,
                "alpha": np.asarray(self.sampler_state.alpha),
            },
            "chains": self.chains,
        }
        return checkpoint.save(file, data)

    def load(self, file: str | Path) -> MCState:
        data = checkpoint.load(file)
        saved = data["sampler_state"]
        sampler_state = Chains(
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

    def _auto_alpha(
        self,
        sampler_state: Chains,
        mass: np.ndarray,
        score: np.ndarray,
        score2: np.ndarray,
        residual: np.ndarray,
        w: np.ndarray,
    ) -> Chains:
        """Update alpha by damped local KL moment projection."""
        if self.sampler.alpha is not None:
            return sampler_state
    
        rdtype = precision.real("calc", host=True)
        tiny = rdtype(precision.tiny("calc"))
        alpha = float(sampler_state.alpha)
    
        target = w * np.abs(residual)
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
                self.H,
                self.model,
                key=sampler_state.key,
                eps1=self.eps1,
                chains=self.chains,
                alpha=sampler_state.alpha,
                timer=timer,
            )
        else:
            with timer("reduce"):
                unique, _, inv = self.H.sector.unique(sampler_state.x)

            with timer("forward"):
                value = batch.apply(self.model.logabs, self.params, unique)
                jax.block_until_ready(value)
                logabs = precision.host(value, "calc", "real").reshape(-1)

            with timer("reduce"):
                sampler_state = replace(
                    sampler_state,
                    logabs=precision.cast(logabs[inv], "calc", "real", host=True),
                )

        sampler_state, samples, sample_mass, sample_stats = self.sampler.draw(
            self.params,
            self.H,
            self.model,
            sampler_state,
            eps1=float(self.eps1),
            profile=profile,
            timer=timer,
        )

        with timer("reduce"):
            ket, _, sample_to_ket = self.H.sector.unique(samples)
            n_ket = int(ket.shape[0])

            raw_mass = precision.cast(sample_mass, "calc", "real", host=True)
            mass = np.bincount(sample_to_ket, weights=raw_mass, minlength=n_ket)
            mass2 = np.bincount(
                sample_to_ket,
                weights=raw_mass * raw_mass,
                minlength=n_ket,
            )
            mass = precision.cast(mass, "calc", "real", host=True)
            mass2 = precision.cast(mass2, "calc", "real", host=True)

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
            )

            h_bra = np.asarray(conn.bra, dtype=np.uint64)
            strong_ptr = np.asarray(conn.strong_ptr, dtype=np.int64)
            strong_h = precision.cast(np.asarray(conn.strong_h), "calc", host=True)
            strong_w = precision.cast(
                np.asarray(conn.strong_degree),
                "calc",
                "real",
                host=True,
            )
            diag = precision.cast(np.asarray(conn.diag), "calc", host=True)
            weak_ptr = np.asarray(conn.weak_ptr, dtype=np.int64)
            weak_h = precision.cast(np.asarray(conn.weak_h), "calc", host=True)
            weak_count = precision.cast(
                np.asarray(conn.weak_count),
                "calc",
                "real",
                host=True,
            )
            weak_w = precision.cast(
                np.asarray(conn.weak_degree),
                "calc",
                "real",
                host=True,
            )

            obs_conn = []
            raw_parts = [h_bra]
            for name, op in obs.items():
                o_diag, o_ptr, o_bra, o_val = op.local_conn(ket)
                o_bra = np.asarray(o_bra, dtype=np.uint64)
                obs_conn.append((
                    str(name),
                    np.asarray(o_diag),
                    np.asarray(o_ptr, dtype=np.int64),
                    o_bra,
                    np.asarray(o_val),
                ))
                raw_parts.append(o_bra)

        with timer("reduce"):
            # Share one forward pool across H and optional observables.
            bra = h_bra if len(raw_parts) == 1 else np.concatenate(raw_parts, axis=0)
            n_strong = int(strong_h.size)
            n_weak = int(weak_h.size)
            strong_slice = slice(n_ket, n_ket + n_strong)
            weak_slice = slice(n_ket + n_strong, n_ket + n_strong + n_weak)

            obs_mapped = []
            start = int(h_bra.shape[0])
            for name, o_diag, o_ptr, o_bra, o_val in obs_conn:
                end = start + int(o_bra.shape[0])
                obs_mapped.append((name, o_diag, o_ptr, slice(start, end), o_val))
                start = end

        with timer("forward"):
            bra_logpsi_jax = batch.apply(self.model.logpsi, self.params, bra)
            jax.block_until_ready(bra_logpsi_jax)
            bra_logpsi = tree.host(bra_logpsi_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], bra_logpsi)
            strong_logpsi = jax.tree.map(lambda a: a[strong_slice], bra_logpsi)
            weak_logpsi = jax.tree.map(lambda a: a[weak_slice], bra_logpsi)
            bra_logabs = precision.cast(
                np.asarray(to_logabs(bra_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            ket_logabs = bra_logabs[:n_ket]

            rdtype = precision.real("calc", host=True)
            tiny = rdtype(precision.tiny("calc"))
            auto = self.sampler.alpha is None
            alpha_used = sampler_state.alpha if auto else self.sampler.alpha
            alpha = rdtype(alpha_used)
            beta = rdtype(self.sampler.blur)

            ket_idx = np.repeat(np.arange(n_ket, dtype=np.int64), np.diff(strong_ptr))
            logamp = precision.cast(alpha * ket_logabs, "calc", "real", host=True)

            eloc = diag.copy()
            if strong_h.size:
                ratio = precision.cast(
                    np.asarray(
                        to_ratio(
                            strong_logpsi,
                            jax.tree.map(lambda a: a[ket_idx], ket_logpsi),
                        )
                    ),
                    "calc",
                    host=True,
                )
                contrib = strong_h * ratio
                eloc = eloc.astype(np.result_type(eloc, contrib), copy=False)
                np.add.at(eloc, ket_idx, contrib)

            tilted = self.sampler.proposal == "ham" or beta > 0.0
            if not tilted:
                logprob = logamp
                if auto:
                    score = ket_logabs
                    score2 = ket_logabs * ket_logabs
            else:
                stay = np.where(
                    strong_w > 0.0,
                    (rdtype(1.0) - beta) * strong_w,
                    rdtype(1.0),
                )
                log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
                ok = stay > 0.0
                log_stay[ok] = np.log(np.maximum(stay[ok], tiny)) + logamp[ok]

                if auto:
                    score = np.zeros(n_ket, dtype=rdtype)
                    score2 = np.zeros(n_ket, dtype=rdtype)

                if beta > 0.0 and strong_h.size:
                    terms = alpha * bra_logabs[strong_slice] + np.log(
                        np.maximum(np.abs(strong_h), tiny)
                    )
                    log_blur = np.log(beta) + math.segment_logsumexp(
                        strong_ptr,
                        terms,
                        n_ket,
                    )
                    logprob = precision.cast(
                        np.logaddexp(log_stay, log_blur),
                        "calc",
                        "real",
                        host=True,
                    )
                else:
                    logprob = log_stay

                if auto:
                    # S=d_alpha log r_alpha is the latent log-amplitude moment.
                    ok = np.isfinite(log_stay) & np.isfinite(logprob)
                    stay_w = np.zeros(n_ket, dtype=rdtype)
                    stay_w[ok] = np.exp(log_stay[ok] - logprob[ok])
                    score += stay_w * ket_logabs
                    score2 += stay_w * ket_logabs * ket_logabs

                    if beta > 0.0 and strong_h.size:
                        term_log = np.log(beta) + terms - logprob[ket_idx]
                        ok = np.isfinite(term_log)
                        conn_w = np.zeros_like(term_log, dtype=rdtype)
                        conn_w[ok] = np.exp(term_log[ok])

                        ell = bra_logabs[strong_slice]
                        np.add.at(score, ket_idx, conn_w * ell)
                        np.add.at(score2, ket_idx, conn_w * ell * ell)

            if weak_h.size and int(self.eloc_sample) > 0:
                weak_ket = np.repeat(
                    np.arange(n_ket, dtype=np.int64),
                    np.diff(weak_ptr),
                )
                ratio = precision.cast(
                    np.asarray(
                        to_ratio(
                            weak_logpsi,
                            jax.tree.map(lambda a: a[weak_ket], ket_logpsi),
                        )
                    ),
                    "calc",
                    host=True,
                )
                scale = weak_w[weak_ket] / np.maximum(
                    rdtype(int(self.eloc_sample)) * np.abs(weak_h),
                    tiny,
                )
                contrib = weak_count * scale * weak_h * ratio
                eloc = eloc.astype(np.result_type(eloc, contrib), copy=False)
                np.add.at(eloc, weak_ket, contrib)

            eloc = precision.cast(eloc, "calc", host=True)
            w, wstats = stats.weight(mass, mass2, ket_logabs, logprob)
            energy, estats = stats.eloc(w, eloc)
            residual = eloc - energy
            if auto:
                sampler_state = self._auto_alpha(
                    sampler_state,
                    mass,
                    score,
                    score2,
                    residual,
                    w,
                )

            obs_stats: dict[str, float] = {}
            obs_data = {}
            for name, o_diag, o_ptr, o_slice, o_val in obs_mapped:
                oloc = precision.cast(np.asarray(o_diag).copy(), "calc", host=True)
                if o_val.size:
                    o_ket = np.repeat(
                        np.arange(n_ket, dtype=np.int64),
                        np.diff(o_ptr),
                    )
                    ratio = precision.cast(
                        np.asarray(
                            to_ratio(
                                jax.tree.map(lambda a: a[o_slice], bra_logpsi),
                                jax.tree.map(lambda a: a[o_ket], ket_logpsi),
                            )
                        ),
                        "calc",
                        host=True,
                    )
                    contrib = precision.cast(o_val, "calc", host=True) * ratio
                    oloc = oloc.astype(np.result_type(oloc, contrib), copy=False)
                    np.add.at(oloc, o_ket, contrib)
                obs_stats.update(stats.observable(name, w, oloc))
                if data:
                    obs_data[name] = oloc

        gradient = None
        geom = None
        if grad:
            with timer("backward"):
                dlogpsi = precision.cast(2.0 * w * residual, "calc", host=True)
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
                        2.0 * np.sqrt(w) * residual,
                        "sr",
                        "real",
                        host=True,
                    )
                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=ket,
                        w=precision.device(w, "sr", "real"),
                        b=precision.device(
                            self.model.cotangent(ket_logpsi, b_log),
                            "sr",
                        ),
                    )

        new_state = replace(self, sampler_state=sampler_state)
        n_sample = int(samples.shape[0])
        n_forward = int(bra.shape[0])

        out = {
            **estats,
            **wstats,
            **obs_stats,
            "alpha": float(alpha_used),
            "accept": float(sample_stats.get("accept", 0.0)),
            "n_sample": float(n_sample),
            "n_unique": float(n_ket),
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
                    "w": w,
                    "eloc": eloc,
                    "obs": obs_data,
                },
            )
        return new_state, energy, gradient, out, geom
