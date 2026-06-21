from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..hilbert import DetSector, Sector, SpinSector
from . import libdet

FloatArray = NDArray[np.float64]
UInt64Array = NDArray[np.uint64]

Conns = libdet.Conns
Projection = libdet.Projection
Projections = libdet.Projections


class Hamiltonian:
    """Electronic Hamiltonian acting on a Fock-space sector."""

    def __init__(
        self,
        sector: Sector,
        h1: ArrayLike,
        eri: ArrayLike,
        *,
        ecore: float = 0.0,
    ) -> None:
        self.sector = sector
        self._ecore = float(ecore)

        h1_arr = np.asarray(h1, dtype=np.float64)
        if h1_arr.shape != (sector.norb, sector.norb):
            raise ValueError("h1 must have shape (sector.norb, sector.norb)")
        h1_arr = np.ascontiguousarray(h1_arr)

        eri_arr = np.asarray(eri, dtype=np.float64)
        npair = sector.norb * (sector.norb + 1) // 2
        if eri_arr.shape != (npair * (npair + 1) // 2,):
            raise ValueError("eri must be a 1D PySCF chemist 8-fold array")
        eri_arr = np.ascontiguousarray(eri_arr)

        if isinstance(sector, DetSector):
            self._raw = libdet.Hamiltonian.det(h1_arr, eri_arr, self._ecore)
        elif isinstance(sector, SpinSector):
            self._raw = libdet.Hamiltonian.spin(
                h1_arr,
                eri_arr,
                int(sector.n_alpha),
                int(sector.n_beta),
                self._ecore,
            )
        else:
            raise TypeError(f"unsupported sector: {type(sector).__name__}")

    @property
    def norb(self) -> int:
        return int(self.sector.norb)

    @property
    def ecore(self) -> float:
        return self._ecore

    def hij(self, bra: ArrayLike, ket: ArrayLike) -> float:
        """Return a single matrix element `H[bra, ket]`."""
        return float(self._raw.hij(bra, ket))

    def diag(self, x: ArrayLike) -> FloatArray:
        """Return diagonal elements for a batch of basis states."""
        return np.asarray(self._raw.diag(x), dtype=np.float64)

    def expand(
        self,
        ket: ArrayLike,
        eps: float,
        *,
        scale: ArrayLike | None = None,
        exclude: ArrayLike | None = None,
    ) -> UInt64Array:
        """Generate unique connected `bra` states above the screening cutoff."""
        scale_arr = None
        if scale is not None:
            scale_arr = np.ascontiguousarray(
                np.asarray(scale, dtype=np.float64).reshape(-1)
            )
            if scale_arr.shape[0] != ket.shape[0]:
                raise ValueError("scale length must match ket")

        return np.asarray(
            self._raw.expand(ket, float(eps), scale_arr, exclude),
            dtype=np.uint64,
        )

    def project(
        self,
        bra: ArrayLike | None,
        ket: ArrayLike,
        scale: ArrayLike,
        *,
        eps: float = 0.0,
        exclude: ArrayLike | None = None,
    ) -> Projection:
        """Contract `H[bra, ket] scale[ket]`.

        If `bra` is `None`, connected external `bra` states are generated with
        the same screening rule used by `expand`.
        """
        scale_arr = np.ascontiguousarray(
            np.asarray(scale, dtype=np.float64).reshape(-1)
        )
        if scale_arr.shape[0] != ket.shape[0]:
            raise ValueError("scale length must match ket")

        if bra is None:
            return self._raw.project(None, ket, scale_arr, float(eps), exclude)

        if exclude is not None:
            raise ValueError("exclude is only valid when bra is None")

        return self._raw.project(bra, ket, scale_arr, float(eps), None)

    def conn(
        self,
        ket: ArrayLike,
        eps: float = 0.0,
        *,
        include: ArrayLike | None = None,
    ) -> Conns:
        """Return screened off-diagonal connections for each `ket`."""
        return self._raw.conn(ket, float(eps), include)

    def sample_conn(
        self,
        ket: ArrayLike,
        counts: ArrayLike | int,
        *,
        eps1: float = np.inf,
        eps2: float = 0.0,
        seed: int = 0,
        bra_weight: bool = False,
        include: ArrayLike | None = None,
    ) -> Conns:
        """Sample connections in the screened window `eps2 <= |H| < eps1`."""
        n_ket = ket.shape[0]
        if np.isscalar(counts):
            counts_arr = np.full(n_ket, int(counts), dtype=np.int64)
        else:
            counts_arr = np.ascontiguousarray(np.asarray(counts, dtype=np.int64))
            if counts_arr.ndim not in (1, 2):
                raise ValueError("counts must have shape (N,) or (S, N)")
            if counts_arr.shape[-1] != n_ket:
                raise ValueError("counts last dimension must match ket")

        if np.any(counts_arr < 0):
            raise ValueError("counts must be nonnegative")

        return self._raw.sample_conn(
            ket,
            counts_arr,
            float(eps1),
            float(eps2),
            int(seed),
            bool(bra_weight),
            include,
        )

    def sample_project(
        self,
        ket: ArrayLike,
        scale: ArrayLike,
        counts: ArrayLike | int,
        *,
        eps1: float,
        eps2: float = 0.0,
        exclude: ArrayLike | None = None,
        seed: int = 0,
    ) -> Projections:
        """Sample projected amplitudes in the screened external window."""
        n_ket = ket.shape[0]
        scale_arr = np.ascontiguousarray(
            np.asarray(scale, dtype=np.float64).reshape(-1)
        )
        if scale_arr.shape[0] != n_ket:
            raise ValueError("scale length must match ket")

        if np.isscalar(counts):
            counts_arr = np.full(n_ket, int(counts), dtype=np.int64)
        else:
            counts_arr = np.ascontiguousarray(np.asarray(counts, dtype=np.int64))
            if counts_arr.ndim not in (1, 2):
                raise ValueError("counts must have shape (N,) or (S, N)")
            if counts_arr.shape[-1] != n_ket:
                raise ValueError("counts last dimension must match ket")

        if np.any(counts_arr < 0):
            raise ValueError("counts must be nonnegative")

        return self._raw.sample_project(
            ket,
            scale_arr,
            counts_arr,
            float(eps1),
            float(eps2),
            exclude,
            int(seed),
        )

    def matrix(self, bra: ArrayLike, ket: ArrayLike | None = None) -> csr_matrix:
        """Build the sparse matrix `H[bra, ket]`."""
        ket = bra if ket is None else ket
        indptr, indices, data, shape = self._raw.matrix(bra, ket)
        return csr_matrix((data, indices, indptr), shape=tuple(shape))

    def matvec(
        self,
        bra: ArrayLike,
        coeff: ArrayLike,
        *,
        ket: ArrayLike | None = None,
    ) -> FloatArray:
        """Apply `H[bra, ket]` to one or more coefficient vectors."""
        ket = bra if ket is None else ket
        coeff_arr = np.asarray(coeff, dtype=np.float64)

        if coeff_arr.ndim == 1:
            coeff_vec = np.ascontiguousarray(coeff_arr)
            if coeff_vec.shape[0] != ket.shape[0]:
                raise ValueError("coeff length must match ket")
            return np.asarray(
                self._raw.matvec(bra, ket, coeff_vec),
                dtype=np.float64,
            )

        if coeff_arr.ndim == 2:
            coeff_mat = np.ascontiguousarray(coeff_arr)
            if coeff_mat.shape[0] != ket.shape[0]:
                raise ValueError("coeff must have shape (n_ket, n_rhs)")
            return np.asarray(
                self._raw.matmat(bra, ket, coeff_mat),
                dtype=np.float64,
            )

        raise ValueError("coeff must be a vector or matrix")
