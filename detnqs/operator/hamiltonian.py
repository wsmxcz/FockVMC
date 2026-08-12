from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from pyscf.tools import fcidump

from ..hilbert import DetSector
from . import libdet


class Hamiltonian:
    """Electronic Hamiltonian acting on a Fock-space sector."""

    def __init__(
        self,
        sector: DetSector,
        h1: ArrayLike,
        eri: ArrayLike,
        *,
        ecore: float = 0.0,
    ) -> None:
        if not isinstance(sector, DetSector):
            raise TypeError(f"unsupported sector: {type(sector).__name__}")

        self.sector = sector
        self.ecore = ecore

        h1 = np.ascontiguousarray(h1, dtype=np.float64)
        eri = np.ascontiguousarray(eri, dtype=np.float64).reshape(-1)

        if h1.shape != (sector.norb, sector.norb):
            raise ValueError("h1 shape must match sector.norb")

        self.integrals = h1, eri

        self._raw = libdet.Hamiltonian(h1, eri, self.ecore)

    @classmethod
    def load(
        cls,
        file: str | Path,
        *,
        sector: type[DetSector] = DetSector,
        spin: int | None = None,
    ) -> Self:
        """Load a standard FCIDUMP Hamiltonian."""
        data = fcidump.read(str(file))

        norb = int(data["NORB"])
        nelec = int(data["NELEC"])
        spin = int(data["MS2"] if spin is None else spin)

        return cls(
            sector(norb, nelec, spin),
            data["H1"],
            data["H2"],
            ecore=float(data["ECORE"]),
        )

    def save(self, file: str | Path) -> Path:
        """Save a PySCF FCIDUMP Hamiltonian."""
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        h1, eri = self.integrals
        fcidump.from_integrals(
            str(path),
            h1,
            eri,
            self.sector.norb,
            self.sector.nelec,
            nuc=self.ecore,
            ms=self.sector.spin,
        )
        return path

    def hij(self, bra: ArrayLike, ket: ArrayLike) -> float:
        """Return a single matrix element `H[bra, ket]`."""
        return self._raw.hij(
            np.ascontiguousarray(bra, dtype=np.uint64),
            np.ascontiguousarray(ket, dtype=np.uint64),
        )

    def diag(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return diagonal elements for a batch of basis states."""
        return self._raw.diag(np.ascontiguousarray(x, dtype=np.uint64))

    def expand(
        self,
        kets: ArrayLike,
        eps: float,
        *,
        scale: ArrayLike | None = None,
        exclude: ArrayLike | None = None,
    ) -> NDArray[np.uint64]:
        """Generate unique connected `bra` states above the screening cutoff."""
        return self._raw.expand(
            np.ascontiguousarray(kets, dtype=np.uint64),
            eps,
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
    ) -> libdet.Projection:
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
            return self._raw.project(None, kets_arr, scale_arr, eps, exclude_arr)

        if exclude is not None:
            raise ValueError("exclude is only valid when bras is None")

        return self._raw.project(
            np.ascontiguousarray(bras, dtype=np.uint64),
            kets_arr,
            scale_arr,
            eps,
            None,
        )

    def conn(
        self,
        kets: ArrayLike,
        eps: float = 0.0,
    ) -> libdet.Conns:
        """Return screened off-diagonal connections for each ket."""
        return self._raw.conn(
            np.ascontiguousarray(kets, dtype=np.uint64),
            eps,
        )

    def sample_conn(
        self,
        kets: ArrayLike,
        counts: ArrayLike | int,
        *,
        eps1: float = np.inf,
        eps2: float = 0.0,
        seed: int = 0,
    ) -> libdet.Conns:
        """Sample connections in the screened span `eps2 <= |H| < eps1`."""
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        counts_arr = (
            np.full(kets_arr.shape[0], counts, dtype=np.int64)
            if np.isscalar(counts)
            else np.ascontiguousarray(counts, dtype=np.int64)
        )
        return self._raw.sample_conn(
            kets_arr,
            counts_arr,
            eps1,
            eps2,
            seed,
        )

    def local_conn(
        self,
        kets: ArrayLike,
        eps1: float,
        eps2: float,
        n_draw: int,
        *,
        seed: int = 0,
    ) -> libdet.LocalConn:
        """Return exact strong terms and sampled weak coefficients."""
        return self._raw.local_conn(
            np.ascontiguousarray(kets, dtype=np.uint64),
            eps1,
            eps2,
            n_draw,
            seed=seed,
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
    ) -> libdet.Projections:
        """Sample projected amplitudes in the screened external span."""
        kets_arr = np.ascontiguousarray(kets, dtype=np.uint64)
        scale_arr = np.ascontiguousarray(scale, dtype=np.float64).reshape(-1)
        counts_arr = (
            np.full(kets_arr.shape[0], counts, dtype=np.int64)
            if np.isscalar(counts)
            else np.ascontiguousarray(counts, dtype=np.int64)
        )

        exclude_arr = (
            None
            if exclude is None
            else np.ascontiguousarray(exclude, dtype=np.uint64)
        )
        return self._raw.sample_project(
            kets_arr,
            scale_arr,
            counts_arr,
            eps1,
            eps2,
            exclude_arr,
            seed,
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
        return csr_matrix((data, indices, indptr), shape=shape)

    def matvec(
        self,
        bras: ArrayLike,
        x: ArrayLike,
        *,
        kets: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
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
