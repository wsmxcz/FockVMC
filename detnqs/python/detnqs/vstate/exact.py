from __future__ import annotations

import itertools
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from libdet import Hamiltonian
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model
from ..model.base import to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class ExactState:
    """Exact variational state on the full fixed-particle determinant space.

    The determinant basis is enumerated once at initialization. The Hamiltonian
    H[dets, dets] is stored as a CSR matrix.

    Estimator:
        E = <psi|H|psi> / <psi|psi>.

    Variance:
        var = ||H psi - E psi||^2 / <psi|psi>.
    """

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
        """Build the full determinant sector and initialize model parameters."""
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
            hmat=hamiltonian.matrix(dets),
        )

    @property
    def n_det(self) -> int:
        """Number of determinants in the exact sector."""
        return int(self.dets.shape[0])

    def expect(self) -> tuple[Self, dict[str, float]]:
        """Return exact energy statistics without a gradient."""
        new_state, _, _, stats, _ = self._run(grad=False, geometry=False)
        return new_state, stats

    def expect_and_grad(self, *, geometry: bool = False):
        """Return energy, gradient, statistics, and optional geometry."""
        return self._run(grad=True, geometry=geometry)

    def replace(self, **updates: Any) -> Self:
        """Return a copy with updated fields."""
        return replace(self, **updates)

    def _run(self, *, grad: bool, geometry: bool):
        """Evaluate exact energy, optional gradient, and optional geometry."""
        timer = utils.Timer()
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("forward"):
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.dets)
            jax.block_until_ready(logpsi_jax)
            logpsi_h = utils.host(logpsi_jax)

        with timer("reduce"):
            psi = precision.asarray(
                np.asarray(to_psi(logpsi_h)).reshape(-1),
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
                    self.dets,
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
                        x=self.dets,
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
            "n_det": float(self.n_det),
            "time_forward": 0.0,
            "time_reduce": 0.0,
            "time_backward": 0.0,
        }
        stats.update(timer.stats())

        return self, energy, gradient, stats, geom