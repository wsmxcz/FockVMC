from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix

from ..hilbert import DetSector, Sector, SpinSector
from . import libdet

FloatArray = NDArray[np.float64]
UInt64Array = NDArray[np.uint64]

Conns = libdet.Conns
LocalConn = libdet.LocalConn
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

        h1_arr = np.ascontiguousarray(h1, dtype=np.float64)
        if h1_arr.shape != (sector.norb, sector.norb):
            raise ValueError("h1 must have shape (sector.norb, sector.norb)")

        eri_arr = np.ascontiguousarray(eri, dtype=np.float64).reshape(-1)
        npair = sector.norb * (sector.norb + 1) // 2
        if eri_arr.shape != (npair * (npair + 1) // 2,):
            raise ValueError("eri must be a 1D PySCF chemist 8-fold array")

        if isinstance(sector, DetSector):
            self._raw = libdet.Hamiltonian.det(
                h1_arr,
                eri_arr,
                self._ecore,
            )
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
        return float(
            self._raw.hij(
                np.ascontiguousarray(bra, dtype=np.uint64),
                np.ascontiguousarray(ket, dtype=np.uint64),
            )
        )

    def diag(self, x: ArrayLike) -> FloatArray:
        """Return diagonal elements for a batch of basis states."""
        return self._raw.diag(np.ascontiguousarray(x, dtype=np.uint64))

    def expand(
        self,
        kets: ArrayLike,
        eps: float,
        *,
        scale: ArrayLike | None = None,
        exclude: ArrayLike | None = None,
    ) -> UInt64Array:
        """Generate unique connected `bra` states above the screening cutoff."""
        return self._raw.expand(
            np.ascontiguousarray(kets, dtype=np.uint64),
            float(eps),
            None
            if scale is None
            else np.ascontiguousarray(scale, dtype=np.float64).reshape(-1),
            None
            if exclude is None
            else np.ascontiguousarray(exclude, dtype=np.uint64),
        )

    def project(
        self,
        bras: ArrayLike | None,
        kets: ArrayLike,
        scale: ArrayLike,
        *,
        eps: float = 0.0,
        exclude: ArrayLike | None = None,
    ) -> Projection:
        """Contract `H[bras, kets] scale[kets]`.

        If `bras` is `None`, connected external `bra` states are generated with
        the same screening rule used by `expand`.
        """
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        scale_arr = np.ascontiguousarray(scale, dtype=np.float64).reshape(-1)

        if bras is None:
            exclude_arr = (
                None
                if exclude is None
                else np.ascontiguousarray(exclude, dtype=np.uint64)
            )
            return self._raw.project(None, kets_arr, scale_arr, float(eps), exclude_arr)

        if exclude is not None:
            raise ValueError("exclude is only valid when bras is None")

        return self._raw.project(
            np.ascontiguousarray(bras, dtype=np.uint64),
            kets_arr,
            scale_arr,
            float(eps),
            None,
        )

    def conn(
        self,
        kets: ArrayLike,
        eps: float = 0.0,
        *,
        assemble_mode: str = "unique",
    ) -> Conns:
        """Return screened off-diagonal connections for each ket."""
        return self._raw.conn(
            np.ascontiguousarray(kets, dtype=np.uint64),
            float(eps),
            assemble_mode=assemble_mode,
        )

    def sample_conn(
        self,
        kets: ArrayLike,
        counts: ArrayLike | int,
        *,
        eps1: float = np.inf,
        eps2: float = 0.0,
        seed: int = 0,
        assemble_mode: str = "unique",
    ) -> Conns:
        """Sample connections in the screened span `eps2 <= |H| < eps1`."""
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        counts_arr = (
            np.full(kets_arr.shape[0], int(counts), dtype=np.int64)
            if np.isscalar(counts)
            else np.ascontiguousarray(counts, dtype=np.int64)
        )
        return self._raw.sample_conn(
            kets_arr,
            counts_arr,
            float(eps1),
            float(eps2),
            int(seed),
            assemble_mode=assemble_mode,
        )


    def local_conn(
        self,
        kets: ArrayLike,
        eps1: float,
        eps2: float,
        counts: ArrayLike | int,
        *,
        seed: int = 0,
        assemble_mode: str = "unique",
    ) -> LocalConn:
        """Return strong connections and weak-window samples for local energy."""
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        counts_arr = (
            np.full(kets_arr.shape[0], int(counts), dtype=np.int64)
            if np.isscalar(counts)
            else np.ascontiguousarray(counts, dtype=np.int64).reshape(-1)
        )
        return self._raw.local_conn(
            kets_arr,
            float(eps1),
            float(eps2),
            counts_arr,
            seed=int(seed),
            assemble_mode=assemble_mode,
        )

    def sample_project(
        self,
        kets: ArrayLike,
        scale: ArrayLike,
        counts: ArrayLike | int,
        *,
        eps1: float,
        eps2: float = 0.0,
        exclude: ArrayLike | None = None,
        seed: int = 0,
    ) -> Projections:
        """Sample projected amplitudes in the screened external span."""
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        scale_arr = np.ascontiguousarray(scale, dtype=np.float64).reshape(-1)
        counts_arr = (
            np.full(kets_arr.shape[0], int(counts), dtype=np.int64)
            if np.isscalar(counts)
            else np.ascontiguousarray(counts, dtype=np.int64)
        )

        exclude_arr = (
            None
            if exclude is None
            else np.ascontiguousarray(exclude, dtype=np.uint64)
        )
        return self._raw.sample_project(
            kets_arr, scale_arr, counts_arr, float(eps1), float(eps2),
            exclude_arr, int(seed),
        )

    def matrix(self, bras: ArrayLike, kets: ArrayLike | None = None) -> csr_matrix:
        """Build the sparse matrix `H[bras, kets]`."""
        bras_arr = np.ascontiguousarray(bras, dtype=np.uint64)
        kets_arr = (
            bras_arr
            if kets is None
            else np.ascontiguousarray(kets, dtype=np.uint64)
        )
        indptr, indices, data, shape = self._raw.matrix(bras_arr, kets_arr)
        return csr_matrix((data, indices, indptr), shape=tuple(shape))

    def matvec(
        self,
        bras: ArrayLike,
        x: ArrayLike,
        *,
        kets: ArrayLike | None = None,
    ) -> FloatArray:
        """Apply `H[bras, kets]` to one or more vectors."""
        bras_arr = np.ascontiguousarray(bras, dtype=np.uint64)
        kets_arr = (
            bras_arr
            if kets is None
            else np.ascontiguousarray(kets, dtype=np.uint64)
        )
        x_arr = np.ascontiguousarray(x, dtype=np.float64)

        if x_arr.ndim == 1:
            return self._raw.matvec(bras_arr, kets_arr, x_arr)

        if x_arr.ndim == 2:
            return self._raw.matmat(bras_arr, kets_arr, x_arr)

        raise ValueError("x must be a vector or matrix")
