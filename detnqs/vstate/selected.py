from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np
from scipy.sparse import csr_matrix

from ..model import Model, to_psi
from ..optimizer import Geometry
from ..utils import Timer, batch, checkpoint, precision, tree


@dataclass(frozen=True, slots=True)
class SelectedState:
    """Variational state on a selected Fock subspace.

    This state is a deterministic projected-space baseline. It evaluates the
    selected Hamiltonian exactly and does not support instantaneous local
    operators.
    """

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
        *,
        key: jax.Array,
        basis: Any | None = None,
    ) -> Self:
        basis = H.sector.reference(1) if basis is None else H.sector.asarray(basis)
        if basis.shape[0] == 0:
            raise ValueError("selected basis must be non-empty")

        params = model.init(key, H.sector.zeros(1))["params"]
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
        basis = self.basis

        if eps is not None:
            logpsi_jax = batch.apply(self.model.logpsi, self.params, basis)
            jax.block_until_ready(logpsi_jax)

            psi = np.asarray(to_psi(tree.host(logpsi_jax))).reshape(-1)
            scale = np.abs(psi).astype(np.float64, copy=False)
            norm = float(np.linalg.norm(scale))

            if norm > 0.0:
                scale = scale / norm

            bra = self.H.expand(
                basis,
                float(eps),
                scale=scale,
                exclude=basis,
            )

            if bra.shape[0] > 0:
                basis = np.ascontiguousarray(
                    np.concatenate([basis, bra], axis=0),
                )

        logabs_jax = batch.apply(self.model.logabs, self.params, basis)
        jax.block_until_ready(logabs_jax)

        logabs = precision.host(logabs_jax, "calc", "real").reshape(-1)
        basis = self.H.sector.asarray(selector(logabs, basis))

        if basis.shape[0] == 0:
            raise ValueError("selector returned an empty basis")

        return replace(self, basis=basis, hmat=self.H.matrix(basis))

    def expect(
        self,
        *,
        obs: Any | None = None,
        profile: bool = False,
        data: bool = False,
    ):
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
        return self._run(grad=True, geometry=geometry, profile=profile, data=False)

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def save(self, file: str | Path) -> Path:
        return checkpoint.save(file, {"params": self.params, "basis": self.basis})

    def load(self, file: str | Path) -> Self:
        data = checkpoint.load(file)
        basis = np.ascontiguousarray(data["basis"], dtype=np.uint64)
        return replace(
            self,
            params=data["params"],
            basis=basis,
            hmat=self.H.matrix(basis),
        )

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
        n_basis = self.n_basis

        with timer("forward"):
            logpsi_jax = batch.apply(self.model.logpsi, self.params, self.basis)
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
            w = precision.cast(np.abs(psi) ** 2 / norm, "calc", "real", host=True)

            out = {
                "energy": energy,
                "eloc_var": float((np.vdot(residual, residual) / norm).real),
                "n_basis": n_basis,
                "n_forward": n_basis,
            }

            if data:
                tiny = precision.tiny("calc")
                eloc = np.zeros_like(hpsi, dtype=np.result_type(hpsi, psi))
                np.divide(hpsi, psi, out=eloc, where=np.abs(psi) > tiny)
                sample = {
                    "x": self.basis,
                    "w": w,
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
                    self.basis,
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
                        x=self.basis,
                        w=precision.device(w, "sr", "real"),
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


def topk_selector(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a selector that keeps the largest log-amplitudes."""
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