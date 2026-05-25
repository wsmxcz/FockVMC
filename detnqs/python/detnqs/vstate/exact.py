from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from libdet import Hamiltonian
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model, to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class ExactState:
    """Exact variational state on the full fixed-particle determinant space."""

    model: Model
    params: Any
    hamiltonian: Hamiltonian
    n_alpha: int
    n_beta: int
    dets: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Hamiltonian,
        n_alpha: int,
        n_beta: int,
        *,
        key: jax.Array,
    ) -> Self:
        norb = int(hamiltonian.norb)
        nword = int(hamiltonian.nword)
        n_alpha = int(n_alpha)
        n_beta = int(n_beta)

        if not (0 <= n_alpha <= norb and 0 <= n_beta <= norb):
            raise ValueError("electron counts must satisfy 0 <= n_alpha,n_beta <= norb")

        dets = []

        for occ_a in itertools.combinations(range(norb), n_alpha):
            det_a = np.zeros(nword, dtype=np.uint64)
            for i in occ_a:
                det_a[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

            for occ_b in itertools.combinations(range(norb), n_beta):
                det_b = np.zeros(nword, dtype=np.uint64)
                for i in occ_b:
                    det_b[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

                dets.append(np.stack((det_a, det_b), axis=0))

        dets = np.ascontiguousarray(np.stack(dets, axis=0))

        variables = model.init(
            key,
            jnp.zeros((1, 2, nword), dtype=jnp.uint64),
        )

        return cls(
            model=model,
            params=variables["params"],
            hamiltonian=hamiltonian,
            n_alpha=n_alpha,
            n_beta=n_beta,
            dets=dets,
            hmat=hamiltonian.matrix(dets, dets),
        )

    @property
    def n_det(self) -> int:
        return int(self.dets.shape[0])

    def expect(self) -> tuple[Self, dict[str, float]]:
        energy, norm, _, _ = self._energy_data(geometry=False)

        return self, {
            "energy": float(energy),
            "norm": float(norm),
            "n_det": float(self.n_det),
        }

    def expect_and_grad(self, *, geometry: bool = False):
        """Return energy, gradient, statistics, and optional optimizer geometry."""

        energy, norm, cot, geom = self._energy_data(geometry=geometry)

        grad = utils.vjp(
            self.model.coord,
            self.params,
            self.dets,
            utils.device(precision.asarray(cot, "model", "real", host=True)),
        )

        stats = {
            "energy": float(energy),
            "norm": float(norm),
            "n_det": float(self.n_det),
        }

        return self, energy, grad, stats, geom

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def _energy_data(self, *, geometry: bool):
        """Compute exact energy data on the determinant space."""

        real_dtype = precision.dtype("calc", "real", host=True)

        logpsi = utils.apply(self.model.logpsi, self.params, self.dets)
        logpsi_h = utils.host(logpsi)

        psi = precision.asarray(
            np.asarray(utils.host(to_psi(logpsi))).reshape(-1),
            "calc",
            host=True,
        )
        hpsi = precision.asarray(
            np.asarray(self.hmat.dot(psi)).reshape(-1),
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
            x=self.dets,
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