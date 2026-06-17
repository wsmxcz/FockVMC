from __future__ import annotations

from typing import Any

import libdet
import numpy as np
from scipy.sparse import csr_matrix

from ..hilbert import CsfSpace, DetSpace, Space


class Hamiltonian:
    """Second-quantized electronic Hamiltonian on a Fock-space sector."""

    def __init__(self, space: Space, h1: Any, eri: Any, *, ecore: float = 0.0):
        self.space = space
        self._ecore = float(ecore)
        if isinstance(space, DetSpace):
            self._raw = libdet.Hamiltonian.rhf(h1, eri, ecore=self._ecore)
        elif isinstance(space, CsfSpace):
            self._raw = libdet.Hamiltonian.guga(
                h1,
                eri,
                n_alpha=space.n_alpha,
                n_beta=space.n_beta,
                ecore=self._ecore,
            )
        else:
            raise TypeError(f"unsupported space: {type(space).__name__}")

    @property
    def norb(self) -> int:
        return int(self.space.norb)

    @property
    def ecore(self) -> float:
        return self._ecore

    def hij(self, bra: Any, ket: Any) -> float:
        return self._raw.hij(bra, ket)

    def diag(self, x: Any) -> np.ndarray:
        return self._raw.diag(x)

    def expand(
        self,
        ket: Any,
        eps: float,
        *,
        coeffs: Any = None,
        exclude: Any = None,
    ) -> np.ndarray:
        return self._raw.expand(ket, float(eps), coeffs=coeffs, exclude=exclude)

    def project(
        self,
        bra: Any | None,
        ket: Any,
        coeffs: Any,
        *,
        eps: float = 0.0,
        exclude: Any = None,
    ) -> Any:
        return self._raw.project(bra, ket, coeffs, eps=float(eps), exclude=exclude)

    def conns(
        self,
        ket: Any,
        eps: float,
        *,
        sample: int = 0,
        sample_eps: float = 0.0,
        seed: int = 0,
    ) -> Any:
        return self._raw.conns(
            ket,
            float(eps),
            sample=int(sample),
            sample_eps=float(sample_eps),
            seed=int(seed),
        )

    def degrees(self, ket: Any, eps: float) -> tuple[np.ndarray, np.ndarray]:
        return self._raw.degrees(ket, float(eps))

    def matrix(self, bra: Any, ket: Any | None = None) -> csr_matrix:
        ket = bra if ket is None else ket
        return self._raw.matrix(bra, ket)

    def matvec(self, bra: Any, v: Any, *, ket: Any | None = None) -> np.ndarray:
        ket = bra if ket is None else ket
        return self._raw.matvec(bra, v, ket=ket)

    def sample_conns(
        self,
        ket: Any,
        counts: Any,
        *,
        eps1: float = 1.0e-6,
        eps2: float = 0.0,
        seed: int = 0,
    ) -> Any:
        return self._raw.sample_conns(
            ket,
            counts,
            eps1=float(eps1),
            eps2=float(eps2),
            seed=int(seed),
        )

    def sample_project(
        self,
        ket: Any,
        coeffs: Any,
        eps1: float,
        eps2: float,
        counts: Any,
        *,
        exclude: Any = None,
        n_rep: int = 1,
        seed: int = 0,
    ) -> Any:
        return self._raw.sample_project(
            ket,
            coeffs,
            float(eps1),
            float(eps2),
            counts,
            exclude=exclude,
            n_rep=int(n_rep),
            seed=int(seed),
        )
