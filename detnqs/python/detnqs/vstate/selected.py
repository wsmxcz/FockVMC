from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Self

import jax
import jax.numpy as jnp
import libdet
import numpy as np
from libdet import Hamiltonian
from scipy.sparse import csr_matrix

from .. import utils
from ..model.base import Model
from ..model.base import to_psi
from ..optimizer import Geometry
from ..utils import precision


@dataclass(frozen=True, slots=True)
class SelectedState:
    """Variational state on a selected determinant space V.

    The Hamiltonian is projected to H[V, V]. The selected space can be evolved
    by expanding connected determinants and then applying a user-provided
    selector.

    Estimator:
        E_V = <psi_V|H[V,V]|psi_V> / <psi_V|psi_V>.

    Variance:
        var_V = ||H[V,V] psi_V - E_V psi_V||^2 / <psi_V|psi_V>.
    """

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
        """Initialize from a user-provided selected determinant space."""
        v_dets = np.ascontiguousarray(libdet.to_dets(init_v))

        if v_dets.shape[0] == 0:
            raise ValueError("init_v must contain at least one determinant")

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
        """Number of determinants in the selected space."""
        return int(self.v_dets.shape[0])

    def evolve(
        self,
        selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        eps: float | None = None,
    ) -> Self:
        """Update the selected space and rebuild H[V, V].

        If eps is provided, the current selected space is first expanded by
        screened Hamiltonian-connected determinants. The selector then receives
        log|psi| and the candidate determinant table, and returns the next V.
        """
        dets = self.v_dets

        if eps is not None:
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.v_dets)
            jax.block_until_ready(logpsi_jax)

            psi = np.asarray(to_psi(utils.host(logpsi_jax))).reshape(-1)
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

        logabs_jax = utils.apply(self.model.logabs, self.params, dets)
        jax.block_until_ready(logabs_jax)

        logabs = precision.asarray(
            np.asarray(utils.host(logabs_jax)).reshape(-1),
            "calc",
            "real",
            host=True,
        )

        v_dets = np.ascontiguousarray(libdet.to_dets(selector(logabs, dets)))

        if v_dets.shape[0] == 0:
            raise ValueError("selector returned an empty determinant space")

        return replace(
            self,
            v_dets=v_dets,
            h_vv=self.hamiltonian.matrix(v_dets, v_dets),
        )

    def expect(self) -> tuple[Self, dict[str, float]]:
        """Return projected energy statistics without a gradient."""
        new_state, _, _, stats, _ = self._run(grad=False, geometry=False)
        return new_state, stats

    def expect_and_grad(self, *, geometry: bool = False):
        """Return energy, gradient, statistics, and optional geometry."""
        return self._run(grad=True, geometry=geometry)

    def replace(self, **updates: Any) -> Self:
        """Return a copy with updated fields."""
        return replace(self, **updates)

    def _run(self, *, grad: bool, geometry: bool):
        """Evaluate projected energy, optional gradient, and optional geometry."""
        timer = utils.Timer()
        pa = precision.asarray
        rdtype = precision.dtype("calc", "real", host=True)

        with timer("forward"):
            logpsi_jax = utils.apply(self.model.logpsi, self.params, self.v_dets)
            jax.block_until_ready(logpsi_jax)
            logpsi_h = utils.host(logpsi_jax)

        with timer("reduce"):
            psi = pa(
                np.asarray(to_psi(logpsi_h)).reshape(-1),
                "calc",
                host=True,
            )

            hpsi = pa(
                np.asarray(self.h_vv.dot(psi)).reshape(-1),
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
                    self.v_dets,
                    utils.device(pa(cot, "model", "real", host=True)),
                )
                jax.block_until_ready(gradient)

                if geometry:
                    w = pa(np.abs(psi) ** 2 / norm, "sr", "real", host=True)
                    sqrt_w = np.sqrt(w)

                    b_log = np.zeros_like(dlogpsi)
                    np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)

                    geom = Geometry(
                        theta=self.params,
                        coord=self.model.coord,
                        x=self.v_dets,
                        w=utils.device(w),
                        b=utils.device(
                            pa(
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
            "n_v": float(self.n_v),
            "time_forward": 0.0,
            "time_reduce": 0.0,
            "time_backward": 0.0,
        }
        stats.update(timer.stats())

        return self, energy, gradient, stats, geom


def topk_selector(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a selector that keeps the k largest log-amplitude determinants."""
    k = int(k)

    def select(logabs: np.ndarray, dets: np.ndarray) -> np.ndarray:
        dets = libdet.to_dets(dets)
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