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
    """Exact variational state on a full finite sector."""

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
        return self.x.shape[0]

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic state for checkpointing."""
        return {"params": self.params}

    def load_state(self, data: dict[str, Any]) -> Self:
        """Restore dynamic state from a checkpoint."""
        return replace(self, params=data["params"])

    def expect(
        self,
        *,
        grad: bool = False,
        geometry: bool = False,
        obs: Any | None = None,
        timer: Timer | None = None,
        data: bool = False,
    ):
        """Evaluate the state and optional gradient or sample data."""
        timer = Timer(timing=False) if timer is None else timer
        rdtype = precision.real("calc", host=True)

        with timer("forward", n=self.n_x):
            logpsi_jax = batch.apply(self.model.logpsi, self.params, self.x)
            logpsi = tree.host(logpsi_jax)

        with timer("reduce"):
            psi = precision.cast(
                to_psi(logpsi),
                "calc",
                host=True,
            ).reshape(-1)
            hpsi = precision.cast(
                self.hmat.dot(psi),
                "calc",
                host=True,
            ).reshape(-1)

            norm = max(float(np.vdot(psi, psi).real), precision.tiny("calc"))
            energy = float((np.vdot(psi, hpsi) / norm).real)
            residual = hpsi - rdtype(energy) * psi
            weight = precision.cast(
                np.abs(psi) ** 2 / norm,
                "calc",
                "real",
                host=True,
            )

            rec = {
                "energy": energy,
                "eloc_var": float((np.vdot(residual, residual) / norm).real),
                "n_x": self.n_x,
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

                if timer.timing:
                    jax.block_until_ready(gradient)

        rec.update(timer.stats())

        if data:
            return self, rec, sample
        if grad:
            return self, rec, gradient, geom
        return self, rec
