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
    n_eloc: int = 256

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
        n_eloc: int = 256,
    ) -> MCState:
        if not 0.0 <= eps2 <= eps1:
            raise ValueError("screening requires 0 <= eps2 <= eps1")
        if n_eloc < 0:
            raise ValueError("n_eloc must be nonnegative")
        if eps2 < eps1 and (eps2 == 0.0 or n_eloc == 0):
            raise ValueError("weak sampling requires eps2 > 0 and n_eloc > 0")

        _, init_key, sample_key = jax.random.split(key, 3)
        params = model.init(init_key, hamiltonian.sector.zeros(1))["params"]

        chains_arr = hamiltonian.sector.asarray(chains)
        if chains_arr.shape[0] != sampler.n_chains:
            raise ValueError("chains size must equal sampler.n_chains")

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
            n_eloc=n_eloc,
        )

    def replace(self, **updates: Any) -> MCState:
        return replace(self, **updates)

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic state for checkpointing."""
        return {
            "params": self.params,
            "sampler_state": {
                "key": self.sampler_state.key,
                "x": self.sampler_state.x,
                "logabs": self.sampler_state.logabs,
                "alpha": self.sampler_state.alpha,
            },
            "chains": self.chains,
        }

    def load_state(self, data: dict[str, Any]) -> MCState:
        """Restore dynamic state from a checkpoint."""
        saved = data["sampler_state"]
        sampler_state = ChainState(
            key=saved["key"],
            x=saved["x"],
            logabs=saved["logabs"],
            alpha=float(saved["alpha"]),
        )
        return replace(
            self,
            params=data["params"],
            sampler_state=sampler_state,
            chains=data["chains"],
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
        sampler_state = self.sampler_state
        obs = {} if obs is None else obs

        if self.sampler.reset:
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

            with timer("forward", n=unique.shape[0]):
                value = batch.apply(self.model.logabs, self.params, unique)
                logabs = precision.host(value, "calc", "real").reshape(-1)

            sampler_state = replace(sampler_state, logabs=logabs[inv])

        sampler_state, observations, observation_mass, sample_rec = (
            self.sampler.draw(
                self.params,
                self.hamiltonian,
                self.model,
                sampler_state,
                eps1=self.eps1,
                timer=timer,
            )
        )

        with timer("unique"):
            ket, _, obs_ket = self.hamiltonian.sector.unique(observations)
            n_ket = ket.shape[0]

        with timer("reduce"):
            mass = np.bincount(
                obs_ket,
                weights=observation_mass,
                minlength=n_ket,
            )
            mass2 = np.bincount(
                obs_ket,
                weights=observation_mass * observation_mass,
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

            bra = (
                bra_parts[0]
                if len(bra_parts) == 1
                else np.concatenate(bra_parts)
            )

        with timer("forward", n=bra.shape[0]):
            bra_logpsi = batch.apply(self.model.logpsi, self.params, bra)
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
            strong_ket = np.repeat(
                np.arange(n_ket, dtype=np.int32),
                strong_count,
            )

            tiny = rdtype(precision.tiny("calc"))
            auto = self.sampler.alpha is None
            alpha_used = sampler_state.alpha if auto else self.sampler.alpha
            alpha = rdtype(alpha_used)
            beta = rdtype(self.sampler.beta)
            tilted = self.sampler.proposal == "ham"
            aux_logamp = alpha * ket_logabs

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
                            lambda a: a[strong_ket],
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
                log_induced = aux_logamp
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
                    + aux_logamp[active]
                )

                if beta > 0.0 and n_strong:
                    strong_logabs = np.asarray(
                        to_logabs(strong_logpsi),
                        dtype=rdtype,
                    ).reshape(-1)
                    terms = alpha * strong_logabs + np.log(
                        np.maximum(np.abs(strong_h), tiny)
                    )
                    log_move = np.log(beta) + math.segment_logsumexp(
                        conn.strong_ptr,
                        terms,
                        n_ket,
                    )
                    log_induced = np.logaddexp(log_stay, log_move)
                else:
                    log_induced = log_stay

                if auto:
                    # S = d_alpha log r_alpha.
                    finite = np.isfinite(log_stay) & np.isfinite(log_induced)
                    stay_weight = np.zeros(n_ket, dtype=rdtype)
                    stay_weight[finite] = np.exp(
                        log_stay[finite] - log_induced[finite]
                    )
                    score = stay_weight * ket_logabs
                    score2 = score * ket_logabs

                    if beta > 0.0 and n_strong:
                        term_log = (
                            np.log(beta)
                            + terms
                            - log_induced[strong_ket]
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
                weak_ket = np.repeat(
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
                            lambda a: a[weak_ket],
                            ket_logpsi,
                        ),
                    ),
                    dtype=eloc_dtype,
                )
                eloc[weak_row] += np.add.reduceat(
                    weak_coeff * ratio,
                    weak_start,
                )

            weight, weight_rec = stats.weight(
                mass,
                mass2,
                ket_logabs,
                log_induced,
            )
            energy, eloc_var = stats.moments(weight, eloc)
            residual = eloc - energy
            if auto:
                res_mass = weight * np.abs(residual)
                norm_r = float(np.sum(res_mass))
                norm_m = float(np.sum(mass))
                if norm_r > tiny and norm_m > tiny:
                    mu_s = float(np.dot(res_mass, score) / norm_r)
                    nu_s = float(np.dot(mass, score) / norm_m)
                    nu_q = float(np.dot(mass, score2) / norm_m)
                    info = nu_q - nu_s * nu_s
                    if np.isfinite(info) and info > tiny:
                        alpha_hat = np.clip(
                            alpha_used + (mu_s - nu_s) / info,
                            0.0,
                            2.0,
                        )
                        alpha_next = alpha_used + 0.02 * (alpha_hat - alpha_used)
                        sampler_state = replace(
                            sampler_state,
                            alpha=float(alpha_next),
                        )

            obs_rec: dict[str, float] = {}
            obs_data = {}
            for name, diag, ptr, obs_slice, value in obs_conn:
                obs_dtype = (
                    precision.complex("calc", host=True)
                    if np.iscomplexobj(diag) or np.iscomplexobj(value)
                    else eloc_dtype
                )
                value = np.asarray(value, dtype=obs_dtype)
                oloc = np.array(diag, dtype=obs_dtype, copy=True)
                if value.size:
                    count = np.diff(ptr)
                    row = np.flatnonzero(count)
                    start = ptr[row]
                    local_ket = np.repeat(
                        np.arange(n_ket, dtype=np.int32),
                        count,
                    )
                    ratio = np.asarray(
                        to_ratio(
                            jax.tree.map(lambda a: a[obs_slice], bra_logpsi),
                            jax.tree.map(
                                lambda a: a[local_ket],
                                ket_logpsi,
                            ),
                        ),
                        dtype=eloc_dtype,
                    )
                    oloc[row] += np.add.reduceat(
                        value * ratio,
                        start,
                    )

                mean, var = stats.moments(weight, oloc)
                obs_rec[name] = mean
                obs_rec[f"{name}_var"] = var
                if data:
                    obs_data[name] = oloc

        gradient = None
        geom = None
        if grad:
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

        new_state = replace(self, sampler_state=sampler_state)
        rec = {
            "energy": energy,
            "eloc_var": eloc_var,
            **weight_rec,
            **obs_rec,
            "alpha": float(alpha_used),
            "acceptance_rate": sample_rec["acceptance_rate"],
            "unique_frac": n_ket / max(1, observations.shape[0]),
        }
        rec.update(timer.stats())
        if timer.timing:
            rec["n_conn"] = sample_rec["n_conn"]

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
