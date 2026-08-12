from __future__ import annotations

from collections.abc import Callable
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
class SelectedState:
    """Variational state on a selected Fock subspace."""

    model: Model
    params: Any
    hamiltonian: Any
    basis: np.ndarray
    hmat: csr_matrix

    @classmethod
    def init(
        cls,
        model: Model,
        hamiltonian: Any,
        *,
        key: jax.Array,
        basis: Any | None = None,
    ) -> Self:
        basis = (
            hamiltonian.sector.reference(1)
            if basis is None
            else hamiltonian.sector.asarray(basis)
        )
        if basis.shape[0] == 0:
            raise ValueError("selected basis must be non-empty")

        params = model.init(key, hamiltonian.sector.zeros(1))["params"]
        return cls(
            model=model,
            params=params,
            hamiltonian=hamiltonian,
            basis=basis,
            hmat=hamiltonian.matrix(basis),
        )

    @property
    def n_basis(self) -> int:
        return self.basis.shape[0]

    def evolve(
        self,
        selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        eps: float | None = None,
    ) -> Self:
        """Expand and select the finite variational basis."""
        basis = self.basis

        if eps is not None:
            logpsi_jax = batch.apply(self.model.logpsi, self.params, basis)

            psi = to_psi(tree.host(logpsi_jax)).reshape(-1)
            scale = np.abs(psi)
            norm = np.linalg.norm(scale)

            if norm > 0.0:
                scale = scale / norm

            bra = self.hamiltonian.expand(
                basis,
                eps,
                scale=scale,
                exclude=basis,
            )

            if bra.shape[0] > 0:
                basis = np.concatenate([basis, bra], axis=0)

        logabs_jax = batch.apply(self.model.logabs, self.params, basis)

        logabs = precision.host(logabs_jax, "calc", "real").reshape(-1)
        basis = self.hamiltonian.sector.asarray(selector(logabs, basis))

        if basis.shape[0] == 0:
            raise ValueError("selector returned an empty basis")

        return replace(self, basis=basis, hmat=self.hamiltonian.matrix(basis))

    def replace(self, **updates: Any) -> Self:
        return replace(self, **updates)

    def state_dict(self) -> dict[str, Any]:
        """Return dynamic state for checkpointing."""
        return {"params": self.params, "basis": self.basis}

    def load_state(self, data: dict[str, Any]) -> Self:
        """Restore parameters and selected basis from a checkpoint."""
        basis = data["basis"]
        return replace(
            self,
            params=data["params"],
            basis=basis,
            hmat=self.hamiltonian.matrix(basis),
        )

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
        n_basis = self.n_basis

        with timer("forward", n=n_basis):
            logpsi_jax = batch.apply(self.model.logpsi, self.params, self.basis)
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
                "n_basis": n_basis,
            }

            if data:
                tiny = precision.tiny("calc")
                eloc = np.zeros_like(hpsi, dtype=np.result_type(hpsi, psi))
                np.divide(hpsi, psi, out=eloc, where=np.abs(psi) > tiny)
                sample = {
                    "x": self.basis,
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
                    self.basis,
                    precision.device(cot, "model", "real"),
                )

                if geometry:
                    sqrt_w = np.sqrt(weight)
                    b_log = np.zeros_like(dlogpsi)
                    np.divide(dlogpsi, sqrt_w, out=b_log, where=sqrt_w > 0.0)

                    geom = Geometry(
                        params=self.params,
                        coord=self.model.coord,
                        x=self.basis,
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


def topk_selector(k: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a selector that keeps the largest log-amplitudes."""
    def select(logabs: np.ndarray, basis: np.ndarray) -> np.ndarray:
        logabs = np.asarray(logabs, dtype=np.float64).reshape(-1)

        if basis.shape[0] != logabs.shape[0]:
            raise ValueError("logabs length must match basis size")

        if basis.shape[0] == 0 or k <= 0:
            return basis[:0]

        n = min(k, basis.shape[0])
        pick = np.argpartition(logabs, -n)[-n:]
        pick = pick[np.argsort(logabs[pick])[::-1]]

        return basis[pick]

    return select
