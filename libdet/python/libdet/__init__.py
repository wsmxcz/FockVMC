from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ._libdet_cpp import Degrees
from ._libdet_cpp import EdgeSamples
from ._libdet_cpp import Edges
from ._libdet_cpp import Hamiltonian as _RawHamiltonian
from ._libdet_cpp import Matrix
from ._libdet_cpp import Projection
from ._libdet_cpp import ShellSamples

__all__ = [
    "Degrees",
    "EdgeSamples",
    "Edges",
    "Hamiltonian",
    "Matrix",
    "Projection",
    "ShellSamples",
    "to_dets",
]


def to_dets(dets: Any) -> np.ndarray:
    """Return a contiguous determinant batch with shape (N, 2, nword)."""
    arr = np.asarray(dets, dtype=np.uint64)
    if arr.ndim != 3 or arr.shape[1] != 2 or arr.shape[2] <= 0:
        raise ValueError("determinants must have shape (N, 2, nword)")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True, slots=True)
class Hamiltonian:
    """Python wrapper for determinant-driven Hamiltonian primitives."""

    _raw: _RawHamiltonian
    _ecore: float = 0.0

    @classmethod
    def rhf(cls, h1: Any, eri: Any, *, ecore: float = 0.0) -> "Hamiltonian":
        """Build a Hamiltonian from RHF one- and two-body integrals."""
        h1_arr  = np.ascontiguousarray(np.asarray(h1,  dtype=np.float64))
        eri_arr = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(-1))
        raw = _RawHamiltonian.rhf(h1_arr, eri_arr, float(ecore))
        return cls(raw, float(ecore))

    @property
    def norb(self) -> int:
        return int(self._raw.norb)

    @property
    def nword(self) -> int:
        return int(self._raw.nword)

    @property
    def ecore(self) -> float:
        return self._ecore

    def hij(self, bra: Any, ket: Any) -> float:
        """Return <bra|H|ket>. Each input must hold exactly one determinant."""
        bras = to_dets(bra)
        kets = to_dets(ket)
        if bras.shape[0] != 1 or kets.shape[0] != 1:
            raise ValueError("hij expects exactly one bra and one ket determinant")
        return float(self._raw.hij(bras, kets))

    def diags(self, dets: Any) -> np.ndarray:
        """Return diagonal elements H_ii for a determinant batch."""
        return np.asarray(self._raw.diags(to_dets(dets)), dtype=np.float64)

    def expand(
        self,
        kets: Any,
        eps: float,
        *,
        coeffs: Any = None,
        exclude: Any = None,
    ) -> np.ndarray:
        """Return unique screened bra determinants connected from source kets.

        With coeffs, screening uses |H_ai c_i| >= eps over i in kets.
        Without coeffs, screening uses |H_ai| >= eps.
        The exclude space defaults to kets.
        """
        ket_arr = to_dets(kets)

        coeff_arr = None
        if coeffs is not None:
            coeff_arr = np.ascontiguousarray(
                np.asarray(coeffs, dtype=np.float64).reshape(-1)
            )
            if coeff_arr.shape[0] != ket_arr.shape[0]:
                raise ValueError("coeffs length must match number of kets")

        exclude_arr = None if exclude is None else to_dets(exclude)
        out = self._raw.expand(ket_arr, float(eps), coeff_arr, exclude_arr)
        return np.array(out.dets, dtype=np.uint64, copy=True)

    def project(
        self,
        bras: Any,
        kets: Any,
        coeffs: Any,
        *,
        eps: float = 0.0,
    ) -> Projection:
        """Return screened H[bras, kets] @ coeffs.

        bras define the output axis; kets and coeffs are aligned.
        Contributions with absolute value below eps are skipped.
        """
        bras_arr  = to_dets(bras)
        kets_arr  = to_dets(kets)
        coeff_arr = np.ascontiguousarray(
            np.asarray(coeffs, dtype=np.float64).reshape(-1)
        )
        if coeff_arr.shape[0] != kets_arr.shape[0]:
            raise ValueError("coeffs length must match number of kets")
        return self._raw.project(bras_arr, kets_arr, coeff_arr, float(eps))

    def edges(self, dets: Any, eps: float) -> Edges:
        """Return screened row connectivity for each determinant in dets."""
        return self._raw.edges(to_dets(dets), float(eps))
    
    def degrees(self, dets: Any, eps: float) -> Degrees:
        """Return screened row degrees and absolute Hamiltonian row weights.

        This is the lightweight counterpart of edges(). It computes only
        row_nnz and row_weight without materializing connected determinants.
        """
        return self._raw.degrees(to_dets(dets), float(eps))
    
    def matrix(
        self,
        bras: Any,
        kets: Any | None = None,
        *,
        raw: bool = False,
    ) -> Matrix | csr_matrix:
        """Return exact sparse H[bras, kets].

        When raw=True the internal Matrix object is returned directly.
        Otherwise a scipy csr_matrix is returned.
        """
        bras_arr = to_dets(bras)
        kets_arr = bras_arr if kets is None else to_dets(kets)
        out = self._raw.matrix(bras_arr, kets_arr)

        if raw:
            return out

        row_ptr = np.array(out.row_ptr, dtype=np.int32,   copy=True)
        col     = np.array(out.col,     dtype=np.int32,   copy=True)
        data    = np.array(out.h,       dtype=np.float64, copy=True)
        return csr_matrix((data, col, row_ptr), shape=tuple(out.shape))

    def matvec(
        self,
        bras: Any,
        x: Any,
        *,
        kets: Any | None = None,
    ) -> np.ndarray:
        """Return exact H[bras, kets] @ x.

        kets defaults to bras when omitted.
        A 2D x is forwarded to the C++ matmat kernel.
        """
        bras_arr = to_dets(bras)
        kets_arr = bras_arr if kets is None else to_dets(kets)
        x_arr    = np.asarray(x, dtype=np.float64)

        if x_arr.ndim == 1:
            x_vec = np.ascontiguousarray(x_arr)
            if x_vec.shape[0] != kets_arr.shape[0]:
                raise ValueError("x length must match number of kets")
            return np.asarray(self._raw.matvec(bras_arr, kets_arr, x_vec), dtype=np.float64)

        if x_arr.ndim == 2:
            x_mat = np.ascontiguousarray(x_arr)
            if x_mat.shape[0] != kets_arr.shape[0]:
                raise ValueError("X must have shape (n_ket, n_rhs)")
            return np.asarray(self._raw.matmat(bras_arr, kets_arr, x_mat), dtype=np.float64)

        raise ValueError("x must be a 1D vector or a 2D matrix")

    def sample_edges(
        self,
        dets: Any,
        counts: Any | None = None,
        *,
        eps1: float = 1.0e-6,
        eps2: float = 0.0,
        seed: int = 0,
    ) -> EdgeSamples:
        """Return row statistics and optional sampled edges.

        The sampled window is eps2 <= |H_ai| < eps1.  Passing eps1=np.inf
        samples from the full screened row |H_ai| >= eps2.
        """
        det_arr = to_dets(dets)

        counts_arr = None
        if counts is not None:
            if np.isscalar(counts):
                counts_arr = np.full(det_arr.shape[0], int(counts), dtype=np.int64)
            else:
                counts_arr = np.ascontiguousarray(
                    np.asarray(counts, dtype=np.int64).reshape(-1)
                )
                if counts_arr.shape[0] != det_arr.shape[0]:
                    raise ValueError("counts length must match number of determinants")

        return self._raw.sample_edges(det_arr, counts_arr, float(eps1), float(eps2), int(seed))

    def sample_shell(
        self,
        kets: Any,
        coeffs: Any,
        eps1: float,
        eps2: float,
        counts: Any,
        *,
        exclude: Any = None,
        n_rep: int = 1,
        seed: int = 0,
    ) -> ShellSamples:
        """Return aggregated weak-shell PT2 samples by replica.

        kets and coeffs define the source wave function.
        A scalar counts broadcasts to all kets.
        Sampled external determinants are returned in ShellSamples.dets.
        """
        ket_arr   = to_dets(kets)
        coeff_arr = np.ascontiguousarray(
            np.asarray(coeffs, dtype=np.float64).reshape(-1)
        )
        if coeff_arr.shape[0] != ket_arr.shape[0]:
            raise ValueError("coeffs length must match number of kets")

        if np.isscalar(counts):
            counts_arr = np.full(ket_arr.shape[0], int(counts), dtype=np.int64)
        else:
            counts_arr = np.ascontiguousarray(
                np.asarray(counts, dtype=np.int64).reshape(-1)
            )
            if counts_arr.shape[0] != ket_arr.shape[0]:
                raise ValueError("counts length must match number of kets")

        exclude_arr = None if exclude is None else to_dets(exclude)
        return self._raw.sample_shell(
            ket_arr,
            coeff_arr,
            float(eps1),
            float(eps2),
            counts_arr,
            exclude_arr,
            int(n_rep),
            int(seed),
        )