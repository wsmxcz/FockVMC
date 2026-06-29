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
        eps1 = float(eps1)
        eps2 = float(eps2)
        if eps1 < 0.0:
            raise ValueError("eps1 must be nonnegative")
        if eps2 < 0.0:
            raise ValueError("eps2 must be nonnegative")
        if eps2 > eps1:
            raise ValueError("eps2 must be <= eps1")
        if int(eloc_sample) < 0:
            raise ValueError("eloc_sample must be nonnegative")
        if not 0.0 <= float(sampler.blur) <= 1.0:
            raise ValueError("sampler.blur must satisfy 0 <= blur <= 1")

        _, init_key, sample_key = jax.random.split(key, 3)
        params = model.init(init_key, H.sector.zeros(1))["params"]

        chains_arr = (
            H.sector.reference(sampler.n_chains)
            if chains is None
            else H.sector.asarray(chains)
        )
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
        )
        return replace(
            self,
            params=data["params"],
            sampler_state=sampler_state,
            chains=np.ascontiguousarray(data["chains"], dtype=np.uint64),
        )

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
            with timer("sample"):
                sampler_state = self.sampler.init(
                    self.params,
                    self.H,
                    self.model,
                    key=sampler_state.key,
                    eps1=float(self.eps1),
                    chains=self.chains,
                )
        else:
            with timer("forward"):
                unique, _, inv = self.H.sector.unique(sampler_state.x)
                value = batch.apply(self.model.logabs, self.params, unique)
                jax.block_until_ready(value)
                logabs = precision.host(value, "calc", "real").reshape(-1)
                sampler_state = replace(
                    sampler_state,
                    logabs=precision.cast(logabs[inv], "calc", "real", host=True),
                )

        with timer("sample"):
            sampler_state, samples, sample_mass, sample_stats = self.sampler.draw(
                self.params,
                self.H,
                self.model,
                sampler_state,
                eps1=float(self.eps1),
                profile=profile,
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
                assemble_mode=self.assemble_mode,
            )

            h_bra = np.asarray(conn.bra, dtype=np.uint64)
            strong_ptr = np.asarray(conn.strong_ptr, dtype=np.int64)
            strong_bra = np.asarray(conn.strong_bra, dtype=np.int64)
            strong_h = precision.cast(np.asarray(conn.strong_h), "calc", host=True)
            strong_w = precision.cast(
                np.asarray(conn.strong_degree),
                "calc",
                "real",
                host=True,
            )
            diag = precision.cast(np.asarray(conn.diag), "calc", host=True)
            weak_ptr = np.asarray(conn.weak_ptr, dtype=np.int64)
            weak_bra = np.asarray(conn.weak_bra, dtype=np.int64)
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

        # Share one forward pool across H and optional observables.
        raw_bra = np.concatenate(raw_parts, axis=0)
        h_size = int(h_bra.shape[0])

        if self.assemble_mode == "unique":
            bra, _, raw_to_bra = self.H.sector.unique(raw_bra)
            strong_bra = raw_to_bra[strong_bra]
            weak_bra = raw_to_bra[weak_bra]
            obs_mapped = []
            start = h_size
            for name, o_diag, o_ptr, o_bra, o_val in obs_conn:
                end = start + int(o_bra.shape[0])
                obs_mapped.append((
                    name,
                    o_diag,
                    o_ptr,
                    raw_to_bra[start:end],
                    o_val,
                ))
                start = end
        else:
            bra = raw_bra
            obs_mapped = []
            start = h_size
            for name, o_diag, o_ptr, o_bra, o_val in obs_conn:
                end = start + int(o_bra.shape[0])
                obs_mapped.append((
                    name,
                    o_diag,
                    o_ptr,
                    np.arange(start, end, dtype=np.int64),
                    o_val,
                ))
                start = end

        with timer("forward"):
            bra_logpsi_jax = batch.apply(self.model.logpsi, self.params, bra)
            jax.block_until_ready(bra_logpsi_jax)
            bra_logpsi = tree.host(bra_logpsi_jax)

        with timer("reduce"):
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], bra_logpsi)
            bra_logabs = precision.cast(
                np.asarray(to_logabs(bra_logpsi)).reshape(-1),
                "calc",
                "real",
                host=True,
            )
            ket_logabs = bra_logabs[:n_ket]

            rdtype = precision.real("calc", host=True)
            tiny = rdtype(precision.tiny("calc"))
            alpha = rdtype(float(self.sampler.alpha))
            beta = rdtype(float(self.sampler.blur))

            ket_idx = np.repeat(np.arange(n_ket, dtype=np.int64), np.diff(strong_ptr))
            logamp = precision.cast(alpha * ket_logabs, "calc", "real", host=True)

            eloc = diag.copy()
            if strong_h.size:
                ratio = precision.cast(
                    np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[strong_bra], bra_logpsi),
                            jax.tree.map(lambda a: a[ket_idx], bra_logpsi),
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
            else:
                stay = np.where(
                    strong_w > 0.0,
                    (rdtype(1.0) - beta) * strong_w,
                    rdtype(1.0),
                )
                log_stay = np.full(n_ket, -np.inf, dtype=rdtype)
                ok = stay > 0.0
                log_stay[ok] = np.log(np.maximum(stay[ok], tiny)) + logamp[ok]

                if beta > 0.0 and strong_h.size:
                    terms = alpha * bra_logabs[strong_bra] + np.log(
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

            if weak_h.size and int(self.eloc_sample) > 0:
                weak_ket = np.repeat(
                    np.arange(n_ket, dtype=np.int64),
                    np.diff(weak_ptr),
                )
                ratio = precision.cast(
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
                contrib = weak_count * scale * weak_h * ratio
                eloc = eloc.astype(np.result_type(eloc, contrib), copy=False)
                np.add.at(eloc, weak_ket, contrib)

            eloc = precision.cast(eloc, "calc", host=True)
            w, wstats = stats.weight(mass, mass2, ket_logabs, logprob)
            energy, estats = stats.eloc(w, eloc)
            residual = eloc - energy

            obs_stats: dict[str, float] = {}
            obs_data = {}
            for name, o_diag, o_ptr, o_bra, o_val in obs_mapped:
                oloc = precision.cast(np.asarray(o_diag).copy(), "calc", host=True)
                if o_val.size:
                    o_ket = np.repeat(
                        np.arange(n_ket, dtype=np.int64),
                        np.diff(o_ptr),
                    )
                    ratio = precision.cast(
                        np.asarray(
                            to_ratio(
                                jax.tree.map(lambda a: a[o_bra], bra_logpsi),
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
            "accept": float(sample_stats.get("accept", 0.0)),
            "n_sample": float(n_sample),
            "n_unique": float(n_ket),
            "n_forward": float(n_forward),
            "unique_frac": float(n_ket) / max(1.0, float(n_sample)),
        }
        if profile:
            out.update(timer.stats())
            out.update(
                {
                    k: float(v)
                    for k, v in sample_stats.items()
                    if k.startswith("time_") or k.startswith("n_")
                }
            )

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
