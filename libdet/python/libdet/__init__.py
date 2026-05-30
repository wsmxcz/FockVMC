from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ._libdet_cpp import Conns
from ._libdet_cpp import ConnSamples
from ._libdet_cpp import Degrees
from ._libdet_cpp import Hamiltonian as _RawHamiltonian
from ._libdet_cpp import Matrix
from ._libdet_cpp import Projection
from ._libdet_cpp import ProjectSamples

__all__ = [
    "Conns",
    "ConnSamples",
    "Degrees",
    "Hamiltonian",
    "Matrix",
    "Projection",
    "ProjectSamples",
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
    """Python wrapper for row-local determinant Hamiltonian primitives."""

    _raw: _RawHamiltonian
    _ecore: float = 0.0

    @classmethod
    def rhf(cls, h1: Any, eri: Any, *, ecore: float = 0.0) -> Hamiltonian:
        """Build an RHF spatial-orbital Hamiltonian."""
        h1_arr = np.ascontiguousarray(np.asarray(h1, dtype=np.float64))
        eri_arr = np.ascontiguousarray(np.asarray(eri, dtype=np.float64).reshape(-1))

        raw = _RawHamiltonian.rhf(h1_arr, eri_arr, float(ecore))

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
        bra_arr = to_dets(bra)
        ket_arr = to_dets(ket)

        if bra_arr.shape[0] != 1 or ket_arr.shape[0] != 1:
            raise ValueError("hij expects exactly one bra and one ket determinant")

        return float(self._raw.hij(bra_arr, ket_arr))

    def diags(self, dets: Any) -> np.ndarray:
        """Return diagonal elements H[det, det] for a determinant batch."""
        return np.asarray(self._raw.diags(to_dets(dets)), dtype=np.float64)

    def expand(
        self,
        kets: Any,
        eps: float,
        *,
        coeffs: Any = None,
        exclude: Any = None,
    ) -> np.ndarray:
        """Return unique screened bras connected from kets.

        With coeffs, screening uses |H_ai c_i| >= eps.
        Without coeffs, screening uses |H_ai| >= eps.
        The exclude space defaults to kets in the C++ layer when omitted.
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

        out = self._raw.expand(
            ket_arr,
            float(eps),
            coeff_arr,
            exclude_arr,
        )

        return np.array(out.dets, dtype=np.uint64, copy=True)

    def project(
        self,
        bras: Any | None,
        kets: Any,
        coeffs: Any,
        *,
        eps: float = 0.0,
        exclude: Any = None,
    ) -> Projection:
        """Return projected amplitudes on known or generated bras.

        If bras is given, compute H[bras, kets] @ coeffs.
        If bras is None, generate connected external bras from kets and
        accumulate screened H_ai c_i contributions.

        kets and coeffs are always aligned.
        """
        kets_arr = to_dets(kets)
        coeff_arr = np.ascontiguousarray(
            np.asarray(coeffs, dtype=np.float64).reshape(-1)
        )

        if coeff_arr.shape[0] != kets_arr.shape[0]:
            raise ValueError("coeffs length must match number of kets")

        exclude_arr = None if exclude is None else to_dets(exclude)

        if bras is None:
            return self._raw.project(
                None,
                kets_arr,
                coeff_arr,
                float(eps),
                exclude_arr,
            )

        if exclude is not None:
            raise ValueError("exclude is only valid when bras is None")

        return self._raw.project(
            to_dets(bras),
            kets_arr,
            coeff_arr,
            float(eps),
            None,
        )

    def conns(self, kets: Any, eps: float) -> Conns:
        """Return screened Hamiltonian connections generated from kets."""
        return self._raw.conns(to_dets(kets), float(eps))

    def degrees(self, kets: Any, eps: float) -> Degrees:
        """Return per-ket connection counts and absolute Hamiltonian weights.

        This is the lightweight counterpart of conns(). It computes ket_nconn
        and ket_weight without materializing connected bras.
        """
        return self._raw.degrees(to_dets(kets), float(eps))

    def matrix(
        self,
        bras: Any,
        kets: Any | None = None,
        *,
        raw: bool = False,
    ) -> Matrix | csr_matrix:
        """Return exact sparse H[bras, kets].

        When raw=True, return the internal Matrix object.
        Otherwise return a scipy.sparse.csr_matrix.
        """
        bras_arr = to_dets(bras)
        kets_arr = bras_arr if kets is None else to_dets(kets)

        out = self._raw.matrix(bras_arr, kets_arr)

        if raw:
            return out

        indptr = np.array(out.indptr, dtype=np.int32, copy=True)
        indices = np.array(out.indices, dtype=np.int32, copy=True)
        data = np.array(out.data, dtype=np.float64, copy=True)

        return csr_matrix((data, indices, indptr), shape=tuple(out.shape))

    def matvec(
        self,
        bras: Any,
        x: Any,
        *,
        kets: Any | None = None,
    ) -> np.ndarray:
        """Return exact H[bras, kets] @ x.

        kets defaults to bras. A 2D x is forwarded to the C++ matmat kernel.
        """
        bras_arr = to_dets(bras)
        kets_arr = bras_arr if kets is None else to_dets(kets)
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim == 1:
            x_vec = np.ascontiguousarray(x_arr)

            if x_vec.shape[0] != kets_arr.shape[0]:
                raise ValueError("x length must match number of kets")

            return np.asarray(
                self._raw.matvec(bras_arr, kets_arr, x_vec),
                dtype=np.float64,
            )

        if x_arr.ndim == 2:
            x_mat = np.ascontiguousarray(x_arr)

            if x_mat.shape[0] != kets_arr.shape[0]:
                raise ValueError("X must have shape (n_ket, n_rhs)")

            return np.asarray(
                self._raw.matmat(bras_arr, kets_arr, x_mat),
                dtype=np.float64,
            )

        raise ValueError("x must be a 1D vector or a 2D matrix")

    def sample_conns(
        self,
        kets: Any,
        counts: Any | None = None,
        *,
        eps1: float = 1.0e-6,
        eps2: float = 0.0,
        seed: int = 0,
    ) -> ConnSamples:
        """Return window connection statistics and optional sampled connections.

        The sampled window is eps2 <= |H_ai| < eps1.

        If counts is omitted, only ket_nconn and ket_weight are returned.
        If counts is provided, connected bras are sampled with probability
        |H_ai| / ket_weight within each ket.
        """
        ket_arr = to_dets(kets)

        counts_arr = None
        if counts is not None:
            if np.isscalar(counts):
                counts_arr = np.full(ket_arr.shape[0], int(counts), dtype=np.int64)
            else:
                counts_arr = np.ascontiguousarray(
                    np.asarray(counts, dtype=np.int64).reshape(-1)
                )

                if counts_arr.shape[0] != ket_arr.shape[0]:
                    raise ValueError("counts length must match number of kets")

        return self._raw.sample_conns(
            ket_arr,
            counts_arr,
            float(eps1),
            float(eps2),
            int(seed),
        )

    def sample_project(
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
    ) -> ProjectSamples:
        """Return sampled projected amplitudes by replica.

        kets and coeffs define the source wavefunction. The weak window is
        eps2 <= |H_ai c_i| < eps1. A scalar counts broadcasts to all kets.
        """
        ket_arr = to_dets(kets)
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

        return self._raw.sample_project(
            ket_arr,
            coeff_arr,
            float(eps1),
            float(eps2),
            counts_arr,
            exclude_arr,
            int(n_rep),
            int(seed),
        )