from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np
from scipy.sparse import csr_matrix

from ..model import Model, to_psi, to_ratio
from ..optimizer import Geometry
from ..utils import Timer, batch, checkpoint, precision, stats, tree


@dataclass(frozen=True, slots=True)
class ExactState:
    """Exact variational state on a full finite sector."""

    model: Model
    params: Any
    H: Any
    x: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(cls, model: Model, H: Any, *, key: jax.Array) -> Self:
        x = H.sector.enumerate()
        params = model.init(key, H.sector.zeros(1))["params"]
        return cls(model=model, params=params, H=H, x=x, hmat=H.matrix(x))

    @property
    def n_x(self) -> int:
        return int(self.x.shape[0])

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

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def save(self, file: str | Path) -> Path:
        return checkpoint.save(file, {"params": self.params})

    def load(self, file: str | Path) -> Self:
        data = checkpoint.load(file)
        return replace(self, params=data["params"])

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
        obs = {} if obs is None else dict(obs)
        rdtype = precision.real("calc", host=True)

        raw_parts = [self.x]
        obs_conn = []
        for name, op in obs.items():
            o_diag, o_ptr, o_bra, o_val = op.local_conn(self.x)
            o_bra = np.asarray(o_bra, dtype=np.uint64)
            obs_conn.append((
                str(name),
                np.asarray(o_diag),
                np.asarray(o_ptr, dtype=np.int64),
                o_bra,
                np.asarray(o_val),
            ))
            raw_parts.append(o_bra)

        # Share one forward pool across the basis and optional observables.
        raw_bra = np.concatenate(raw_parts, axis=0)
        bra, _, raw_to_bra = self.H.sector.unique(raw_bra)
        obs_mapped = []
        start = self.n_x
        for name, o_diag, o_ptr, o_bra, o_val in obs_conn:
            end = start + int(o_bra.shape[0])
            obs_mapped.append((name, o_diag, o_ptr, raw_to_bra[start:end], o_val))
            start = end

        with timer("forward"):
            logpsi_jax = batch.apply(self.model.logpsi, self.params, bra)
            jax.block_until_ready(logpsi_jax)
            logpsi_h = tree.host(logpsi_jax)

        with timer("reduce"):
            x_logpsi = jax.tree.map(lambda a: a[: self.n_x], logpsi_h)
            psi = precision.cast(
                np.asarray(to_psi(x_logpsi)).reshape(-1),
                "calc",
                host=True,
            )
            hpsi = precision.cast(
                np.asarray(self.hmat.dot(psi)).reshape(-1),
                "calc",
                host=True,
            )
            norm = max(float(np.vdot(psi, psi).real), precision.tiny("calc"))
            w = precision.cast(np.abs(psi) ** 2 / norm, "calc", "real", host=True)
            energy = float((np.vdot(psi, hpsi) / norm).real)
            residual = hpsi - rdtype(energy) * psi
            out = {
                "energy": energy,
                "eloc_var": float((np.vdot(residual, residual) / norm).real),
                "n_x": float(self.n_x),
                "n_forward": float(bra.shape[0]),
            }

            obs_data = {}
            for name, o_diag, o_ptr, o_bra, o_val in obs_mapped:
                oloc = precision.cast(np.asarray(o_diag).copy(), "calc", host=True)
                if o_val.size:
                    o_ket = np.repeat(
                        np.arange(self.n_x, dtype=np.int64),
                        np.diff(o_ptr),
                    )
                    ratio = precision.cast(
                        np.asarray(
                            to_ratio(
                                jax.tree.map(lambda a: a[o_bra], logpsi_h),
                                jax.tree.map(lambda a: a[o_ket], x_logpsi),
                            )
                        ),
                        "calc",
                        host=True,
                    )
                    contrib = precision.cast(o_val, "calc", host=True) * ratio
                    oloc = oloc.astype(np.result_type(oloc, contrib), copy=False)
                    np.add.at(oloc, o_ket, contrib)
                out.update(stats.observable(name, w, oloc))
                if data:
                    obs_data[name] = oloc

            if data:
                tiny = precision.tiny("calc")
                eloc = np.zeros_like(hpsi, dtype=np.result_type(hpsi, psi))
                np.divide(hpsi, psi, out=eloc, where=np.abs(psi) > tiny)
                sample_data = {
                    "x": self.x,
                    "w": w,
                    "eloc": eloc,
                    "obs": obs_data,
                }

        gradient = None
        geom = None
        if grad:
            with timer("backward"):
                dlogpsi = rdtype(2.0 / norm) * np.conjugate(psi) * residual
                cot = self.model.cotangent(x_logpsi, dlogpsi)
                gradient = batch.vjp(
                    self.model.coord,
                    self.params,
                    self.x,
                    precision.device(cot, "model", "real"),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    sqrt_w = np.sqrt(w)
                    b_log = np.zeros_like(dlogpsi)
                    np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)
                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=self.x,
                        w=precision.device(w, "sr", "real"),
                        b=precision.device(
                            self.model.cotangent(x_logpsi, b_log),
                            "sr",
                            "real",
                        ),
                    )

        if profile:
            out.update(timer.stats())
        if data:
            return self, energy, gradient, out, geom, sample_data
        return self, energy, gradient, out, geom
