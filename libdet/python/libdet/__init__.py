from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ._libdet_cpp import Conns
from ._libdet_cpp import ConnSamples
from ._libdet_cpp import Hamiltonian as _RawHamiltonian
from ._libdet_cpp import Projection
from ._libdet_cpp import ProjectSamples

__all__ = [
    "Conns",
    "ConnSamples",
    "Hamiltonian",
    "Projection",
    "ProjectSamples",
]


@dataclass(frozen=True, slots=True)
class Hamiltonian:
    """Determinant Hamiltonian primitives."""

    _raw: Any
    _ecore: float = 0.0

    @classmethod
    def rhf(cls, h1: Any, eri: Any, *, ecore: float = 0.0) -> Hamiltonian:
        """Build an RHF spatial-orbital Hamiltonian."""
        h1_arr = np.ascontiguousarray(np.asarray(h1, dtype=np.float64))
        eri_arr = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(-1))

        raw = _RawHamiltonian.rhf(h1_arr, eri_arr, float(ecore))

        return cls(raw, float(ecore))

    @classmethod
    def guga(
        cls,
        h1: Any,
        eri: Any,
        *,
        n_alpha: int,
        n_beta: int,
        ecore: float = 0.0,
    ) -> Hamiltonian:
        """Build a GUGA CSF spin-adapted Hamiltonian."""
        h1_arr = np.ascontiguousarray(np.asarray(h1, dtype=np.float64))
        eri_arr = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(-1))

        raw = _RawHamiltonian.guga(
            h1_arr,
            eri_arr,
            int(n_alpha),
            int(n_beta),
            float(ecore),
        )

        return cls(raw, float(ecore))

    @property
    def norb(self) -> int:
        """Number of spatial orbitals."""
        return int(self._raw.norb)

    @property
    def nword(self) -> int:
        """Number of 64-bit words per spin string."""
        return int(self._raw.nword)

    @property
    def ecore(self) -> float:
        """Scalar core energy."""
        return self._ecore

    def hij(self, bra: Any, ket: Any) -> float:
        """Return H[bra, ket]. Each input must contain exactly one determinant."""
        return float(self._raw.hij(bra, ket))

    def diag(self, x: Any) -> np.ndarray:
        """Return diagonal elements H[det, det] for a determinant batch."""
        return np.asarray(self._raw.diag(x), dtype=np.float64)

    def expand(
        self,
        ket: Any,
        eps: float,
        *,
        coeffs: Any = None,
        exclude: Any = None,
    ) -> np.ndarray:
        """Return unique screened bras connected from ket.

        With coeffs, screening uses |H_ai c_i| >= eps.
        Without coeffs, screening uses |H_ai| >= eps.
        The exclude space defaults to ket in the C++ layer when omitted.
        """
        coeff_arr = None
        if coeffs is not None:
            coeff_arr = np.ascontiguousarray(
                np.asarray(coeffs, dtype=np.float64).reshape(-1)
            )

            if coeff_arr.shape[0] != ket.shape[0]:
                raise ValueError("coeffs length must match number of kets")

        return np.asarray(
            self._raw.expand(
                ket,
                float(eps),
                coeff_arr,
                exclude,
            ),
            dtype=np.uint64,
        )

    def project(
        self,
        bra: Any | None,
        ket: Any,
        coeffs: Any,
        *,
        eps: float = 0.0,
        exclude: Any = None,
    ) -> Projection:
        """Return projected amplitudes on known or generated bras.

        If bra is given, compute H[bra, ket] @ coeffs.
        If bra is None, generate connected external bras from ket and
        accumulate screened H_ai c_i contributions.

        ket and coeffs are always aligned.
        """
        coeff_arr = np.ascontiguousarray(
            np.asarray(coeffs, dtype=np.float64).reshape(-1)
        )

        if coeff_arr.shape[0] != ket.shape[0]:
            raise ValueError("coeffs length must match number of ket")

        if bra is None:
            return self._raw.project(
                None,
                ket,
                coeff_arr,
                float(eps),
                exclude,
            )

        if exclude is not None:
            raise ValueError("exclude is only valid when bra is None")

        return self._raw.project(
            bra,
            ket,
            coeff_arr,
            float(eps),
            None,
        )

    def conns(
        self,
        ket: Any,
        eps: float,
        *,
        sample: int = 0,
        sample_eps: float = 0.0,
        seed: int = 0,
    ) -> Conns:
        """Return exact and optionally sampled bra connections for each ket."""
        return self._raw.conns(
            ket,
            float(eps),
            int(sample),
            float(sample_eps),
            int(seed),
        )

    def degrees(
        self,
        ket: Any,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return absolute coupling weights and bra counts for each ket."""
        weight, nconn = self._raw.degrees(ket, float(eps))
        return (
            np.asarray(weight, dtype=np.float64),
            np.asarray(nconn, dtype=np.int64),
        )

    def matrix(
        self,
        bra: Any,
        ket: Any | None = None,
    ) -> csr_matrix:
        """Return exact sparse H[bra, ket]."""
        ket = bra if ket is None else ket

        indptr, indices, data, shape = self._raw.matrix(bra, ket)
        return csr_matrix((data, indices, indptr), shape=tuple(shape))

    def matvec(
        self,
        bra: Any,
        x: Any,
        *,
        ket: Any | None = None,
    ) -> np.ndarray:
        """Return exact H[bra, ket] @ x.

        ket defaults to bra. A 2D x is forwarded to the C++ matmat kernel.
        """
        ket = bra if ket is None else ket
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim == 1:
            x_vec = np.ascontiguousarray(x_arr)

            if x_vec.shape[0] != ket.shape[0]:
                raise ValueError("x length must match number of ket")

            return np.asarray(
                self._raw.matvec(bra, ket, x_vec),
                dtype=np.float64,
            )

        if x_arr.ndim == 2:
            x_mat = np.ascontiguousarray(x_arr)

            if x_mat.shape[0] != ket.shape[0]:
                raise ValueError("X must have shape (n_ket, n_rhs)")

            return np.asarray(
                self._raw.matmat(bra, ket, x_mat),
                dtype=np.float64,
            )

        raise ValueError("x must be a 1D vector or a 2D matrix")

    def sample_conns(
        self,
        ket: Any,
        counts: Any,
        *,
        eps1: float = 1.0e-6,
        eps2: float = 0.0,
        seed: int = 0,
    ) -> ConnSamples:
        """Sample bra connections in ``eps2 <= |H| < eps1``.

        ``counts`` has shape ``(N,)`` or ``(S, N)`` for independent streams.
        """
        counts_arr = np.ascontiguousarray(
            np.asarray(counts, dtype=np.int64)
        )

        if counts_arr.ndim not in (1, 2):
            raise ValueError("counts must have shape (N,) or (S, N)")
        if counts_arr.shape[-1] != ket.shape[0]:
            raise ValueError("counts last dimension must match kets")
        if np.any(counts_arr < 0):
            raise ValueError("counts must be nonnegative")

        return self._raw.sample_conns(
            ket,
            counts_arr,
            float(eps1),
            float(eps2),
            int(seed),
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
    ) -> ProjectSamples:
        """Return sampled projected amplitudes by replica.

        ket and coeffs define the source wavefunction. The weak window is
        eps2 <= |H_ai c_i| < eps1. A scalar counts broadcasts to all ket.
        """
        coeff_arr = np.ascontiguousarray(
            np.asarray(coeffs, dtype=np.float64).reshape(-1)
        )

        if coeff_arr.shape[0] != ket.shape[0]:
            raise ValueError("coeffs length must match number of kets")

        if np.isscalar(counts):
            counts_arr = np.full(ket.shape[0], int(counts), dtype=np.int64)
        else:
            counts_arr = np.ascontiguousarray(
                np.asarray(counts, dtype=np.int64).reshape(-1)
            )

            if counts_arr.shape[0] != ket.shape[0]:
                raise ValueError("counts length must match number of kets")

        return self._raw.sample_project(
            ket,
            coeff_arr,
            float(eps1),
            float(eps2),
            counts_arr,
            exclude,
            int(n_rep),
            int(seed),
        )
