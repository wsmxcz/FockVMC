from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self

import jax
import numpy as np
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model
from ..model.base import to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class ExactState:
    """Exact variational state on a full finite sector."""

    model: Model
    params: Any
    H: Any
    x: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(
        cls,
        model: Model,
        H: Any,
        *,
        key: jax.Array,
    ) -> Self:
        """Enumerate the sector and initialize model parameters."""
        x = H.sector.enumerate()
        params = model.init(key, H.sector.zeros(1))["params"]

        return cls(
            model=model,
            params=params,
            H=H,
            x=x,
            hmat=H.matrix(x),
        )

    @property
    def n_x(self) -> int:
        return int(self.x.shape[0])

    def expect(self) -> tuple[Self, dict[str, float]]:
        new_state, _, _, stats, _ = self._run(grad=False, geometry=False)
        return new_state, stats

    def expect_and_grad(self, *, geometry: bool = False):
        return self._run(grad=True, geometry=geometry)

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def _run(self, *, grad: bool, geometry: bool):
        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("forward"):
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.x)
            jax.block_until_ready(logpsi_jax)
            logpsi_h = utils.host(logpsi_jax)

        with timer("reduce"):
            psi = precision.asarray(np.asarray(to_psi(logpsi_h)).reshape(-1), "calc", host=True)
            hpsi = precision.asarray(np.asarray(self.hmat.dot(psi)).reshape(-1), "calc", host=True)

            norm = max(float(np.vdot(psi, psi).real), precision.tiny("calc"))
            energy = float((np.vdot(psi, hpsi) / norm).real)
            residual = hpsi - rdtype(energy) * psi
            variance = float((np.vdot(residual, residual) / norm).real)

        gradient = None
        geom = None

        if grad:
            with timer("backward"):
                # dE / dlogpsi = 2 / norm * conj(psi) * (H psi - E psi).
                dlogpsi = rdtype(2.0 / norm) * np.conjugate(psi) * residual
                cot = self.model.cotangent(logpsi_h, dlogpsi)

                gradient = utils.vjp(
                    self.model.coord,
                    self.params,
                    self.x,
                    utils.device(precision.asarray(cot, "model", "real", host=True)),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    w = precision.asarray(np.abs(psi) ** 2 / norm, "sr", "real", host=True)
                    sqrt_w = np.sqrt(w)

                    b_log = np.zeros_like(dlogpsi)
                    np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)

                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=self.x,
                        w=utils.device(w),
                        b=utils.device(
                            precision.asarray(
                                self.model.cotangent(logpsi_h, b_log),
                                "sr",
                                "real",
                                host=True,
                            )
                        ),
                    )

        stats = {
            "energy": float(energy),
            "variance": float(variance),
            "n_x": float(self.n_x),
            "time_forward": 0.0,
            "time_reduce": 0.0,
            "time_backward": 0.0,
        }
        stats.update(timer.stats())

        return self, energy, gradient, stats, geom
