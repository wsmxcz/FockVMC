from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HydrogenLattice:
    coords: np.ndarray

    @property
    def atom(self) -> str:
        return "; ".join(f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in self.coords)

    @classmethod
    def chain(cls, n: int, R: float) -> HydrogenLattice:
        coords = np.array([(0.0, 0.0, i * R) for i in range(n)], dtype=float)
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def square(cls, shape: int | tuple[int, int], R: float) -> HydrogenLattice:
        nx, ny = (shape, shape) if isinstance(shape, int) else shape
        coords = np.array(
            [(ix * R, iy * R, 0.0) for ix in range(nx) for iy in range(ny)],
            dtype=float,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def cubic(cls, shape: int | tuple[int, int, int], R: float) -> HydrogenLattice:
        nx, ny, nz = (shape, shape, shape) if isinstance(shape, int) else shape
        coords = np.array(
            [
                (ix * R, iy * R, iz * R)
                for ix in range(nx)
                for iy in range(ny)
                for iz in range(nz)
            ],
            dtype=float,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def sheet(cls, shape: int | tuple[int, int], R: float) -> HydrogenLattice:
        nx, ny = (shape, shape) if isinstance(shape, int) else shape
        a1 = np.array([R, 0.0, 0.0])
        a2 = np.array([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        coords = np.array(
            [ix * a1 + iy * a2 for ix in range(nx) for iy in range(ny)],
            dtype=float,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def pyramid(cls, n: int, R: float) -> HydrogenLattice:
        a1 = np.array([R, 0.0, 0.0])
        a2 = np.array([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        a3 = np.array(
            [0.5 * R, R / (2.0 * np.sqrt(3.0)), np.sqrt(2.0 / 3.0) * R]
        )

        coords = np.array(
            [
                ix * a1 + iy * a2 + iz * a3
                for ix in range(n)
                for iy in range(n - ix)
                for iz in range(n - ix - iy)
            ],
            dtype=float,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)


def chain_init(sector, mo_coeff, n_chains: int, seed: int = 0) -> np.ndarray:
    """Sample determinant chains from a real restricted Slater determinant."""
    rng = np.random.default_rng(seed)
    n_chains = int(n_chains)

    coeff = np.asarray(mo_coeff, dtype=np.float64)
    chains = sector.zeros(n_chains)
    chain = np.arange(n_chains, dtype=np.int64)
    item = chain[:, None]

    for spin, n_elec in enumerate((sector.n_alpha, sector.n_beta)):
        n_elec = int(n_elec)
        basis = np.linalg.qr(coeff[:, :n_elec], mode="reduced")[0]
        basis = np.broadcast_to(basis, (n_chains, *basis.shape)).copy()
        occ = np.empty((n_chains, n_elec), dtype=np.int64)

        for k in range(n_elec):
            probs = np.vecdot(basis, basis, axis=-1)
            probs[chain[:, None], occ[:, :k]] = 0.0
            probs /= probs.sum(axis=1, keepdims=True)

            u = rng.random((n_chains, 1))
            p = (np.cumsum(probs, axis=1) < u).sum(axis=1)
            occ[:, k] = np.minimum(p, sector.norb - 1)

            if k + 1 == n_elec:
                break

            row = basis[chain, occ[:, k]].copy()
            col = np.argmax(np.abs(row), axis=1)
            pivot = row[chain, col]

            basis -= (
                basis[chain, :, col] / pivot[:, None]
            )[:, :, None] * row[:, None, :]

            last = basis.shape[2] - 1
            basis[chain, :, col] = basis[:, :, last]
            basis = np.linalg.qr(basis[:, :, :last], mode="reduced")[0]

        word = occ >> 6
        bit = (occ & 63).astype(np.uint64)
        np.bitwise_or.at(chains[:, spin, :], (item, word), np.uint64(1) << bit)

    return np.ascontiguousarray(chains)
