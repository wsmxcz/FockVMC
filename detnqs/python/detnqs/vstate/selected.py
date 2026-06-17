from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Self

import jax
import numpy as np
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model
from ..model.base import to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class SelectedState:
    """Variational state on a selected Fock subspace."""

    model: Model
    params: Any
    H: Any
    basis: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(
        cls,
        model: Model,
        H: Any,
        init: Any = "hf",
        *,
        key: jax.Array,
    ) -> Self:
        """Initialize from a selected basis."""
        basis = H.space.reference(1) if isinstance(init, str) and init == "hf" else H.space.asarray(init)
        if basis.shape[0] == 0:
            raise ValueError("selected basis must be non-empty")

        params = model.init(key, H.space.zeros(1))["params"]

        return cls(
            model=model,
            params=params,
            H=H,
            basis=basis,
            hmat=H.matrix(basis),
        )

    @property
    def n_basis(self) -> int:
        return int(self.basis.shape[0])

    def evolve(
        self,
        selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        eps: float | None = None,
    ) -> Self:
        """Update the selected basis and rebuild H[basis, basis]."""
        basis = self.basis

        if eps is not None:
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.basis)
            jax.block_until_ready(logpsi_jax)

            psi = np.asarray(to_psi(utils.host(logpsi_jax))).reshape(-1)
            coeff = np.abs(psi).astype(np.float64, copy=False)

            norm = float(np.linalg.norm(coeff))
            if norm > 0.0:
                coeff = coeff / norm

            bra = self.H.expand(self.basis, float(eps), coeffs=coeff, exclude=self.basis)
            if bra.shape[0] > 0:
                basis = np.ascontiguousarray(np.concatenate([self.basis, bra], axis=0))

        logabs_jax = utils.apply(self.model.logabs, self.params, basis)
        jax.block_until_ready(logabs_jax)

        logabs = precision.asarray(
            np.asarray(utils.host(logabs_jax)).reshape(-1),
            "calc",
            "real",
            host=True,
        )
        basis = self.H.space.asarray(selector(logabs, basis))

        if basis.shape[0] == 0:
            raise ValueError("selector returned an empty basis")

        return replace(self, basis=basis, hmat=self.H.matrix(basis))

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
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.basis)
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
                    self.basis,
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
                        x=self.basis,
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
            "n_basis": float(self.n_basis),
            "time_forward": 0.0,
            "time_reduce": 0.0,
            "time_backward": 0.0,
        }
        stats.update(timer.stats())

        return self, energy, gradient, stats, geom


def topk_selector(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a selector that keeps the k largest amplitudes."""
    k = int(k)

    def select(logabs: np.ndarray, basis: np.ndarray) -> np.ndarray:
        logabs = np.asarray(logabs, dtype=np.float64).reshape(-1)
        if basis.shape[0] != logabs.shape[0]:
            raise ValueError("logabs length must match basis size")

        if basis.shape[0] == 0 or k <= 0:
            return np.ascontiguousarray(basis[:0])

        n = min(k, basis.shape[0])
        pick = np.argpartition(logabs, -n)[-n:]
        pick = pick[np.argsort(logabs[pick])[::-1]]
        return np.ascontiguousarray(basis[pick])

    return select
