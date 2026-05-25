from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Self

import jax
import jax.numpy as jnp
import libdet
import numpy as np
from libdet import Hamiltonian
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model, to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class SelectedState:
    """Variational state on a selected determinant space V."""

    model: Model
    params: Any
    hamiltonian: Hamiltonian
    v_dets: np.ndarray
    h_vv: csr_matrix

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Hamiltonian,
        init_v: Any,
        *,
        key: jax.Array,
    ) -> Self:
        v_dets = np.ascontiguousarray(libdet.to_dets(init_v))
        nword = int(v_dets.shape[2])

        variables = model.init(
            key,
            jnp.zeros((1, 2, nword), dtype=jnp.uint64),
        )

        return cls(
            model=model,
            params=variables["params"],
            hamiltonian=hamiltonian,
            v_dets=v_dets,
            h_vv=hamiltonian.matrix(v_dets, v_dets),
        )

    @property
    def n_v(self) -> int:
        return int(self.v_dets.shape[0])

    def evolve(
        self,
        selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        eps: float | None = None,
    ) -> Self:
        """Update the selected determinant space and rebuild H[V,V]."""

        dets = self.v_dets

        if eps is not None:
            logpsi = utils.apply(self.model.logpsi, self.params, self.v_dets)
            psi = np.asarray(utils.host(to_psi(logpsi))).reshape(-1)

            coeffs = np.abs(psi).astype(np.float64, copy=False)
            norm = float(np.linalg.norm(coeffs))
            if norm > 0.0:
                coeffs = coeffs / norm

            ext = self.hamiltonian.expand(
                self.v_dets,
                float(eps),
                coeffs=coeffs,
                exclude=self.v_dets,
            )

            ext = libdet.to_dets(ext)
            if ext.shape[0] > 0:
                dets = np.ascontiguousarray(np.concatenate([self.v_dets, ext], axis=0))

        logabs = np.asarray(
            utils.host(utils.apply(self.model.logabs, self.params, dets))
        ).reshape(-1)
        logabs = precision.asarray(logabs, "calc", "real", host=True)

        v_dets = selector(logabs, dets)

        return replace(
            self,
            v_dets=v_dets,
            h_vv=self.hamiltonian.matrix(v_dets, v_dets),
        )

    def expect(self) -> tuple[Self, dict[str, float]]:
        energy, norm, _, _ = self._energy_data(geometry=False)

        return self, {
            "energy": float(energy),
            "norm": float(norm),
            "n_v": float(self.n_v),
        }

    def expect_and_grad(self, *, geometry: bool = False):
        """Return energy, gradient, statistics, and optional optimizer geometry."""

        energy, norm, cot, geom = self._energy_data(geometry=geometry)

        grad = utils.vjp(
            self.model.coord,
            self.params,
            self.v_dets,
            utils.device(precision.asarray(cot, "model", "real", host=True)),
        )

        stats = {
            "energy": float(energy),
            "norm": float(norm),
            "n_v": float(self.n_v),
        }

        return self, energy, grad, stats, geom

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def _energy_data(self, *, geometry: bool):
        """Compute projected variational energy data on the selected space V."""

        real_dtype = precision.dtype("calc", "real", host=True)

        logpsi = utils.apply(self.model.logpsi, self.params, self.v_dets)
        logpsi_h = utils.host(logpsi)

        psi = precision.asarray(
            np.asarray(utils.host(to_psi(logpsi))).reshape(-1),
            "calc",
            host=True,
        )
        hpsi = precision.asarray(
            np.asarray(self.h_vv.dot(psi)).reshape(-1),
            "calc",
            host=True,
        )

        norm = max(float(np.vdot(psi, psi).real), precision.tiny("calc"))
        energy = float((np.vdot(psi, hpsi) / norm).real)

        # dE / dlogpsi = 2 / norm * conj(psi) * (H psi - E psi)
        residual = hpsi - real_dtype(energy) * psi
        dlogpsi = real_dtype(2.0 / norm) * np.conjugate(psi) * residual
        cot = self.model.cotangent(logpsi_h, dlogpsi)

        if not geometry:
            return energy, norm, cot, None

        w = precision.asarray(np.abs(psi) ** 2 / norm, "calc", "real", host=True)
        sqrt_w = np.sqrt(w)

        b_log = np.zeros_like(dlogpsi)
        np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)

        geom = Geometry(
            theta=self.params,
            coord=self.model.coord,
            x=self.v_dets,
            w=utils.device(precision.asarray(w, "sr", "real", host=True)),
            b=utils.device(
                precision.asarray(
                    self.model.cotangent(logpsi_h, b_log),
                    "sr",
                    "real",
                    host=True,
                )
            ),
        )

        return energy, norm, cot, geom


def topk_selector(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a selector that keeps the K largest log-amplitude determinants."""

    k = int(k)

    def select(logabs: np.ndarray, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint64)
        logabs = np.asarray(logabs, dtype=np.float64).reshape(-1)

        if dets.shape[0] != logabs.shape[0]:
            raise ValueError("logabs length must match number of determinants")

        if dets.shape[0] == 0 or k <= 0:
            return np.ascontiguousarray(dets[:0])

        n = min(k, dets.shape[0])
        idx = np.argpartition(logabs, -n)[-n:]
        idx = idx[np.argsort(logabs[idx])[::-1]]

        return np.ascontiguousarray(dets[idx])

    return select