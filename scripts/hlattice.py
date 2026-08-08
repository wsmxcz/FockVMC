from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import optax
from pyscf import ao2mo, gto, lo, scf

from detnqs import Hamiltonian, MCState, VMC
from detnqs.hilbert import DetSector
from detnqs.model import PBackflow, slater_reference
from detnqs.operator import S2
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler, sample_slater
from detnqs.utils import Logger, batch, precision


@dataclass(frozen=True, slots=True)
class HydrogenLattice:
    """Coordinates for a finite hydrogen lattice."""

    coords: np.ndarray

    @property
    def atom(self) -> str:
        return "; ".join(
            f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in self.coords
        )

    def afm_sites(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a balanced AFM partition of the nearest-neighbor graph."""
        nsite = int(self.coords.shape[0])
        if nsite < 2 or nsite % 2:
            raise ValueError("AFM partition requires an even number of sites")

        delta = self.coords[:, None, :] - self.coords[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        positive = distance[distance > 0.0]
        if positive.size == 0:
            raise ValueError("lattice sites must have distinct coordinates")

        bond = float(positive.min())
        neighbor = np.isclose(distance, bond, rtol=1.0e-7, atol=1.0e-10)
        laplacian = np.diag(neighbor.sum(axis=1)) - neighbor
        score = np.linalg.eigh(laplacian)[1][:, -1]

        order = np.argsort(score, kind="stable")
        spin = np.ones(nsite, dtype=np.int8)
        spin[order[nsite // 2 :]] = -1

        while True:
            alpha = np.flatnonzero(spin > 0)
            beta = np.flatnonzero(spin < 0)
            field = neighbor @ spin
            gain = (
                spin[alpha, None] * field[alpha, None]
                + spin[beta] * field[beta]
                - 2
                * spin[alpha, None]
                * spin[beta]
                * neighbor[np.ix_(alpha, beta)]
            )
            pick = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[pick] <= 0:
                break

            spin[alpha[pick[0]]] *= -1
            spin[beta[pick[1]]] *= -1

        alpha = np.flatnonzero(spin > 0)
        beta = np.flatnonzero(spin < 0)

        if 0 not in alpha:
            alpha, beta = beta, alpha

        return np.sort(alpha), np.sort(beta)

    @classmethod
    def chain(cls, n: int, R: float) -> HydrogenLattice:
        coords = np.asarray([(0.0, 0.0, i * R) for i in range(n)], dtype=float)
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def square(cls, shape: int | tuple[int, int], R: float) -> HydrogenLattice:
        nx, ny = (shape, shape) if isinstance(shape, int) else shape
        coords = np.asarray(
            [(ix * R, iy * R, 0.0) for ix in range(nx) for iy in range(ny)],
            dtype=float,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def cubic(
        cls,
        shape: int | tuple[int, int, int],
        R: float,
    ) -> HydrogenLattice:
        nx, ny, nz = (shape, shape, shape) if isinstance(shape, int) else shape
        coords = np.asarray(
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
        a1 = np.asarray([R, 0.0, 0.0])
        a2 = np.asarray([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        coords = np.asarray(
            [ix * a1 + iy * a2 for ix in range(nx) for iy in range(ny)],
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def pyramid(cls, n: int, R: float) -> HydrogenLattice:
        a1 = np.asarray([R, 0.0, 0.0])
        a2 = np.asarray([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        a3 = np.asarray(
            [0.5 * R, R / (2.0 * np.sqrt(3.0)), np.sqrt(2.0 / 3.0) * R]
        )
        coords = np.asarray(
            [
                ix * a1 + iy * a2 + iz * a3
                for ix in range(n)
                for iy in range(n - ix)
                for iz in range(n - ix - iy)
            ],
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)


def main() -> None:
    batch.configure(
        forward_chunk=32768,
        backward_chunk=4096,
        param_chunk=None,
        bucket_min=4096,
    )
    precision.configure("double")

    name = "H36chain"
    seed = 0
    lattice = HydrogenLattice.chain(36, 2.0)
    mol = gto.M(
        atom=lattice.atom,
        basis="sto-6g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )

    norb = int(mol.nao_nr())
    n_alpha, n_beta = map(int, mol.nelec)

    s = mol.intor_symmetric("int1e_ovlp")
    coeff = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)

    hcore = scf.hf.get_hcore(mol)
    h1 = np.asarray(coeff.T @ hcore @ coeff, dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(8, ao2mo.kernel(mol, coeff), norb),
        dtype=np.float64,
    )

    h1[np.abs(h1) < 1.0e-8] = 0.0
    eri[np.abs(eri) < 1.0e-8] = 0.0

    sector = DetSector(norb, n_alpha + n_beta, n_alpha - n_beta)
    hamiltonian = Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())
    hamiltonian.save(f"{name}.FCIDUMP")

    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")
    print(f"orth err   : {np.linalg.norm(coeff.T @ s @ coeff - np.eye(norb)):.3e}")

    nsite = lattice.coords.shape[0]
    if norb != nsite or (n_alpha, n_beta) != (nsite // 2, nsite // 2):
        raise ValueError("AFM reference requires one electron and one OAO per H atom")

    occ_a, occ_b = lattice.afm_sites()
    orbitals = np.eye(norb)
    ref_mat = slater_reference(
        orbitals[:, occ_a],
        orbitals[:, occ_b],
    )

    print(f"AF alpha   : {occ_a.tolist()}")
    print(f"AF beta    : {occ_b.tolist()}")

    model = PBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(256, 256),
        ref_mat=ref_mat,
    )

    sampler = MCSampler(
        n_samples=4096,
        n_chains=4096,
        thermal_steps=4096,
        discard_steps=16,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = sample_slater(sector, ref_mat, n=sampler.n_chains, seed=seed)

    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(seed),
        eps1=1.0e-3,
        eps2=1.0e-12,
        eloc_sample=32768,
    )

    optimizer = optax.chain(
        psr(shift=1.0e-3, mu=0.95),
        optax.scale_by_learning_rate(5.0e-2),
    )
    vmc = VMC.init(state, optimizer)

    log = Logger(file=f"{name}.jsonl", every=10)
    obs = {"s2": S2(sector)}

    vmc.run(
        5000,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=f"{name}_{{step:05d}}.npz",
        checkpoint_every=1000,
    )


if __name__ == "__main__":
    main()
