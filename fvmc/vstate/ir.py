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
from ..sampler import ChainState, HamSampler
from ..utils import Timer, batch, math, precision, stats, tree


@dataclass(slots=True)
class IRState:
    """Importance-reweighted variational state."""

    model: Model
    params: Any
    hamiltonian: Any
    sampler: HamSampler
    chain: ChainState
    alpha: float | None
    alpha_value: float
    beta: float
    eps1: float
    eps2: float
    n_eloc: int

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Any,
        *,
        sampler: HamSampler,
        chains: Any,
        key: jax.Array,
        alpha: float | None = None,
        beta: float = 0.5,
        eps1: float | None = None,
        eps2: float = 1.0e-12,
        n_eloc: int = 1024,
    ) -> IRState:
        eps1 = sampler.eps1 if eps1 is None else eps1
        if alpha is not None and (
            not np.isfinite(alpha) or not 0.0 <= alpha <= 2.0
        ):
            raise ValueError("alpha must be None or satisfy 0 <= alpha <= 2")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must satisfy 0 <= beta <= 1")
        if not 0.0 <= eps2 <= eps1 <= sampler.eps1:
            raise ValueError(
                "screening requires 0 <= eps2 <= eps1 <= sampler.eps1"
            )
        if n_eloc < 0:
            raise ValueError("n_eloc must be nonnegative")
        if eps2 < eps1 and (eps2 == 0.0 or n_eloc == 0):
            raise ValueError("weak sampling requires eps2 > 0 and n_eloc > 0")

        _, init_key, sample_key = jax.random.split(key, 3)
        params = model.init(init_key, hamiltonian.sector.zeros(1))["params"]
        alpha_value = 2.0 if alpha is None else float(alpha)
        chain = sampler.init(
            params,
            model,
            hamiltonian,
            chains=chains,
            key=sample_key,
            alpha=alpha_value,
        )
        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            sampler=sampler,
            chain=chain,
            alpha=alpha,
            alpha_value=alpha_value,
            beta=float(beta),
            eps1=float(eps1),
            eps2=float(eps2),
            n_eloc=n_eloc,
        )

    def replace(self, **updates: Any) -> IRState:
        return replace(self, **updates)

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic state for checkpointing."""
        return {
            "params": self.params,
            "chain": {
                "key": self.chain.key,
                "x": self.chain.x,
                "logabs": self.chain.logabs,
            },
            "alpha_value": self.alpha_value,
        }

    def load_state(self, data: dict[str, Any]) -> IRState:
        """Restore dynamic state from a checkpoint."""
        saved = data["chain"]
        chain = ChainState(
            key=saved["key"],
            x=saved["x"],
            logabs=saved["logabs"],
        )
        return replace(
            self,
            params=data["params"],
            chain=chain,
            alpha_value=float(data["alpha_value"]),
        )

    def expect(
        self,
        *,
        grad: bool = False,
        geometry: bool = False,
        obs: Mapping[str, Any] | None = None,
        timer: Timer | None = None,
        data: bool = False,
    ):
        """Evaluate the state and optional gradient or sample data."""
        timer = Timer(timing=False) if timer is None else timer
        obs = {} if obs is None else obs
        sector = self.hamiltonian.sector
        alpha = self.alpha_value if self.alpha is None else self.alpha

        with timer("unique"):
            unique, _, index = sector.unique(self.chain.x)
        with timer("forward", n=unique.shape[0]):
            value = batch.apply(self.model.logabs, self.params, unique)
            logabs = precision.host(value, "calc", "real").reshape(-1)
        chain = replace(self.chain, logabs=logabs[index])

        chain, samples, sample_rec = self.sampler.draw(
            self.params,
            self.model,
            self.hamiltonian,
            chain,
            alpha=alpha,
            beta=self.beta,
            timer=timer,
        )

        with timer("unique"):
            ket, _, sample_index = sector.unique(samples)
            n_ket = ket.shape[0]
        with timer("reduce"):
            count = np.bincount(sample_index, minlength=n_ket)
            seed = 0
            if self.eps2 < self.eps1:
                key, subkey = jax.random.split(chain.key)
                seed = int(jax.random.bits(subkey, (), dtype=jnp.uint32))
                chain = replace(chain, key=key)

        with timer("conns"):
            conn = self.hamiltonian.local_conn(
                ket,
                self.eps1,
                self.eps2,
                self.n_eloc,
                seed=seed,
            )
            n_strong = conn.strong_h.size
            n_weak = conn.weak_coeff.size
            bra_parts = [conn.bra]
            obs_conn = []
            start = conn.bra.shape[0]
            for name, op in obs.items():
                diag, ptr, obs_bra, value = op.local_conn(ket)
                stop = start + obs_bra.shape[0]
                obs_conn.append((name, diag, ptr, slice(start, stop), value))
                bra_parts.append(obs_bra)
                start = stop
            bra = conn.bra if len(bra_parts) == 1 else np.concatenate(bra_parts)

        with timer("forward", n=bra.shape[0]):
            bra_logpsi = tree.host(
                batch.apply(self.model.logpsi, self.params, bra)
            )

        with timer("reduce"):
            dtype = precision.real("calc", host=True)
            ket_logpsi = jax.tree.map(lambda a: a[:n_ket], bra_logpsi)
            ket_logabs = np.asarray(to_logabs(ket_logpsi), dtype=dtype).reshape(-1)
            eloc_dtype = (
                precision.complex("calc", host=True)
                if any(np.iscomplexobj(a) for a in jax.tree.leaves(ket_logpsi))
                else dtype
            )
            strong_h = np.asarray(conn.strong_h, dtype=dtype)
            weak_coeff = np.asarray(conn.weak_coeff, dtype=dtype)
            strong_slice = slice(n_ket, n_ket + n_strong)
            weak_slice = slice(n_ket + n_strong, n_ket + n_strong + n_weak)
            strong_count = np.diff(conn.strong_ptr)
            strong_row = np.flatnonzero(strong_count)
            strong_start = conn.strong_ptr[strong_row]
            strong_ket = np.repeat(
                np.arange(n_ket, dtype=np.int32),
                strong_count,
            )
            strong_logpsi = jax.tree.map(
                lambda a: a[strong_slice],
                bra_logpsi,
            )
            graph = np.abs(strong_h) >= self.sampler.eps1
            graph_h = strong_h[graph]
            graph_ket = strong_ket[graph]
            graph_count = np.bincount(graph_ket, minlength=n_ket)
            graph_ptr = np.cumulative_sum(graph_count, include_initial=True)
            graph_row = np.flatnonzero(graph_count)
            graph_start = graph_ptr[graph_row]
            degree = np.zeros(n_ket, dtype=dtype)
            degree[graph_row] = np.add.reduceat(np.abs(graph_h), graph_start)

            eloc = np.array(conn.diag, dtype=eloc_dtype, copy=True)
            if n_strong:
                ratio = np.asarray(
                    to_ratio(
                        strong_logpsi,
                        jax.tree.map(lambda a: a[strong_ket], ket_logpsi),
                    ),
                    dtype=eloc_dtype,
                )
                eloc[strong_row] += np.add.reduceat(
                    strong_h * ratio,
                    strong_start,
                )

            tiny = dtype(precision.tiny("calc"))
            alpha_logabs = dtype(alpha) * ket_logabs
            log_stay = np.full(n_ket, -np.inf, dtype=dtype)
            active = degree > 0.0
            if self.beta < 1.0:
                log_stay[active] = (
                    np.log1p(-dtype(self.beta))
                    + np.log(degree[active])
                    + alpha_logabs[active]
                )
            log_stay[~active] = alpha_logabs[~active]

            terms = np.empty(0, dtype=dtype)
            graph_logabs = np.empty(0, dtype=dtype)
            if self.beta > 0.0 and graph_h.size:
                graph_logpsi = jax.tree.map(lambda a: a[graph], strong_logpsi)
                graph_logabs = np.asarray(
                    to_logabs(graph_logpsi),
                    dtype=dtype,
                ).reshape(-1)
                terms = (
                    np.log(np.maximum(np.abs(graph_h), tiny))
                    + dtype(alpha) * graph_logabs
                )
                log_move = np.log(dtype(self.beta)) + math.segment_logsumexp(
                    graph_ptr,
                    terms,
                    n_ket,
                )
                log_r = np.logaddexp(log_stay, log_move)
            else:
                log_r = log_stay

            if n_weak:
                weak_count = np.diff(conn.weak_ptr)
                weak_row = np.flatnonzero(weak_count)
                weak_start = conn.weak_ptr[weak_row]
                weak_ket = np.repeat(
                    np.arange(n_ket, dtype=np.int32),
                    weak_count,
                )
                ratio = np.asarray(
                    to_ratio(
                        jax.tree.map(lambda a: a[weak_slice], bra_logpsi),
                        jax.tree.map(lambda a: a[weak_ket], ket_logpsi),
                    ),
                    dtype=eloc_dtype,
                )
                eloc[weak_row] += np.add.reduceat(
                    weak_coeff * ratio,
                    weak_start,
                )

            weight, weight_rec = stats.weight(count, ket_logabs, log_r)
            energy, eloc_var = stats.moments(weight, eloc)
            residual = eloc - energy

            alpha_value = self.alpha_value
            if self.alpha is None:
                finite = np.isfinite(log_stay) & np.isfinite(log_r)
                stay_weight = np.zeros(n_ket, dtype=dtype)
                stay_weight[finite] = np.exp(log_stay[finite] - log_r[finite])
                score = stay_weight * ket_logabs
                score2 = score * ket_logabs
                if self.beta > 0.0 and graph_h.size:
                    term_log = (
                        np.log(dtype(self.beta))
                        + terms
                        - log_r[graph_ket]
                    )
                    finite = np.isfinite(term_log)
                    conn_weight = np.zeros_like(term_log, dtype=dtype)
                    conn_weight[finite] = np.exp(term_log[finite])
                    score[graph_row] += np.add.reduceat(
                        conn_weight * graph_logabs,
                        graph_start,
                    )
                    score2[graph_row] += np.add.reduceat(
                        conn_weight * graph_logabs * graph_logabs,
                        graph_start,
                    )

                res_weight = weight * np.abs(residual)
                res_norm = float(np.sum(res_weight))
                count_norm = float(np.sum(count))
                if res_norm > tiny and count_norm > tiny:
                    score_res = float(np.dot(res_weight, score) / res_norm)
                    score_mean = float(np.dot(count, score) / count_norm)
                    score_sq = float(np.dot(count, score2) / count_norm)
                    info = score_sq - score_mean * score_mean
                    if np.isfinite(info) and info > tiny:
                        target = np.clip(
                            alpha + (score_res - score_mean) / info,
                            0.0,
                            2.0,
                        )
                        alpha_value = float(alpha + 0.02 * (target - alpha))

            obs_rec: dict[str, float] = {}
            obs_data = {}
            for name, diag, ptr, obs_slice, value in obs_conn:
                obs_dtype = (
                    precision.complex("calc", host=True)
                    if np.iscomplexobj(diag) or np.iscomplexobj(value)
                    else eloc_dtype
                )
                value = np.asarray(value, dtype=obs_dtype)
                local = np.array(diag, dtype=obs_dtype, copy=True)
                if value.size:
                    local_count = np.diff(ptr)
                    row = np.flatnonzero(local_count)
                    start = ptr[row]
                    local_ket = np.repeat(
                        np.arange(n_ket, dtype=np.int32),
                        local_count,
                    )
                    ratio = np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[obs_slice], bra_logpsi),
                            jax.tree.map(lambda a: a[local_ket], ket_logpsi),
                        ),
                        dtype=eloc_dtype,
                    )
                    local[row] += np.add.reduceat(value * ratio, start)
                mean, var = stats.moments(weight, local)
                obs_rec[name] = mean
                obs_rec[f"{name}_var"] = var
                if data:
                    obs_data[name] = local

        if grad:
            geom = None
            with timer("backward"):
                dlogpsi = 2.0 * weight * residual
                cotangent = self.model.cotangent(ket_logpsi, dlogpsi)
                gradient = batch.vjp(
                    self.model.coord,
                    self.params,
                    ket,
                    precision.device(cotangent, "model"),
                )
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
                if timer.timing:
                    jax.block_until_ready(gradient)

        new_state = replace(
            self,
            chain=chain,
            alpha_value=alpha_value,
        )
        rec = {
            "energy": energy,
            "eloc_var": eloc_var,
            **weight_rec,
            **obs_rec,
            "alpha": float(alpha),
            "acceptance_rate": sample_rec["acceptance_rate"],
            "unique_frac": n_ket / max(1, samples.shape[0]),
        }
        rec.update(timer.stats())
        if timer.timing:
            rec["n_conn"] = n_strong + n_weak

        if data:
            return new_state, rec, {
                "x": ket,
                "weight": weight,
                "eloc": eloc,
                "obs": obs_data,
            }
        if grad:
            return new_state, rec, gradient, geom
        return new_state, rec
