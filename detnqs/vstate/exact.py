from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self

import jax
import numpy as np
from scipy.sparse import csr_matrix

from ..model import Model
from ..model.base import to_psi
from ..optimizer.base import Geometry
from ..utils import Timer, batch, precision, tree


@dataclass(frozen=True, slots=True)
class ExactState:
    """Exact variational state on a full finite sector.

    This state is a deterministic baseline. It evaluates the full projected
    Hamiltonian exactly and does not support instantaneous local operators.
    """

    model: Model
    params: Any
    hamiltonian: Any
    x: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(cls, model: Model, hamiltonian: Any, *, key: jax.Array) -> Self:
        x = hamiltonian.sector.enumerate()
        params = model.init(key, hamiltonian.sector.zeros(1))["params"]
        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            x=x,
            hmat=hamiltonian.matrix(x),
        )

    @property
    def n_x(self) -> int:
        return int(self.x.shape[0])

    def expect(
        self,
        *,
        obs: Any | None = None,
        profile: bool = False,
        data: bool = False,
    ):
        """Evaluate exact energy statistics on the complete sector."""
        result = self._run(grad=False, geometry=False, profile=profile, data=data)
        if data:
            new_state, _, _, out, _, sample = result
            return new_state, out, sample

        new_state, _, _, out, _ = result
        return new_state, out

    def expect_and_grad(
        self,
        *,
        geometry: bool = False,
        obs: Any | None = None,
        profile: bool = False,
    ):
        """Evaluate exact energy, gradient, and optional SR geometry."""
        return self._run(grad=True, geometry=geometry, profile=profile, data=False)

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def _run(
        self,
        *,
        grad: bool,
        geometry: bool,
        profile: bool,
        data: bool,
    ):
        timer = Timer(enabled=profile)
        rdtype = precision.real("calc", host=True)

        with timer("forward"):
            logpsi_jax = batch.apply(self.model.logpsi, self.params, self.x)
            jax.block_until_ready(logpsi_jax)
            logpsi = tree.host(logpsi_jax)

        with timer("reduce"):
            psi = precision.cast(
                np.asarray(to_psi(logpsi)).reshape(-1),
                "calc",
                host=True,
            )
            hpsi = precision.cast(
                np.asarray(self.hmat.dot(psi)).reshape(-1),
                "calc",
                host=True,
            )

            norm = max(float(np.vdot(psi, psi).real), precision.tiny("calc"))
            energy = float((np.vdot(psi, hpsi) / norm).real)
            residual = hpsi - rdtype(energy) * psi
            weight = precision.cast(
                np.abs(psi) ** 2 / norm,
                "calc",
                "real",
                host=True,
            )

            out = {
                "energy": energy,
                "eloc_var": float((np.vdot(residual, residual) / norm).real),
                "n_x": self.n_x,
                "n_forward": self.n_x,
            }

            if data:
                tiny = precision.tiny("calc")
                eloc = np.zeros_like(hpsi, dtype=np.result_type(hpsi, psi))
                np.divide(hpsi, psi, out=eloc, where=np.abs(psi) > tiny)
                sample = {
                    "x": self.x,
                    "weight": weight,
                    "eloc": eloc,
                }

        gradient = None
        geom = None

        if grad:
            with timer("backward"):
                dlogpsi = rdtype(2.0 / norm) * np.conjugate(psi) * residual
                cot = self.model.cotangent(logpsi, dlogpsi)

                gradient = batch.vjp(
                    self.model.coord,
                    self.params,
                    self.x,
                    precision.device(cot, "model", "real"),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    sqrt_w = np.sqrt(weight)
                    b_log = np.zeros_like(dlogpsi)
                    np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)

                    geom = Geometry(
                        params=self.params,
                        coord=self.model.coord,
                        x=self.x,
                        weight=precision.device(weight, "sr", "real"),
                        b=precision.device(
                            self.model.cotangent(logpsi, b_log),
                            "sr",
                            "real",
                        ),
                    )

        if profile:
            out.update(timer.stats())

        if data:
            return self, energy, gradient, out, geom, sample
        return self, energy, gradient, out, geom
