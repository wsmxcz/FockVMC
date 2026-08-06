from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import optax
from pyscf import ao2mo, gto, lo, scf

from detnqs import Hamiltonian, MCState, VMC
from detnqs.hilbert import DetSector
from detnqs.model import GBackflow, slater_reference
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
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    name = "h16"
    mol = gto.M(
        atom=HydrogenLattice.chain(16, 2.0).atom,
        basis="sto-6g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = int(mf.mo_coeff.shape[1])
    n_alpha, n_beta = map(int, mol.nelec)

    # Build OAO integrals.
    s = mol.intor_symmetric("int1e_ovlp")
    coeff = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)

    h1 = np.asarray(coeff.T @ mf.get_hcore() @ coeff, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, coeff), norb), dtype=np.float64)

    h1[np.abs(h1) < 1.0e-8] = 0.0
    eri[np.abs(eri) < 1.0e-8] = 0.0

    # Build and save the FCIDUMP Hamiltonian.
    sector = DetSector(norb, n_alpha + n_beta, n_alpha - n_beta)
    hamiltonian = Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())
    hamiltonian.save(f"{name}.FCIDUMP")

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")
    print(f"orth err   : {np.linalg.norm(coeff.T @ s @ coeff - np.eye(norb)):.3e}")

    # Express the occupied SCF orbitals in the OAO Hamiltonian basis.
    orbitals = coeff.T @ s @ mf.mo_coeff
    ref_mat = slater_reference(
        orbitals[:, :n_alpha],
        orbitals[:, :n_beta],
    )

    model = GBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(256,),
        ref_mat=ref_mat,
    )

    sampler = MCSampler(
        n_samples=4096,
        n_chains=4096,
        thermal_steps=256,
        discard_steps=8,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = sample_slater(sector, ref_mat, n=sampler.n_chains, seed=0)

    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
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
        checkpoint_every=500,
    )


if __name__ == "__main__":
    main()
