from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pyscf import ao2mo, gto, lo, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


@dataclass(frozen=True, slots=True)
class HydrogenLattice:
    """Simple hydrogen lattice builders."""

    coords: np.ndarray

    @property
    def atom(self) -> str:
        return "; ".join(f"H {x:.12f} {y:.12f} {z:.12f}" for x, y, z in self.coords)

    @classmethod
    def chain(cls, n: int, R: float) -> HydrogenLattice:
        # Build a centered 1D hydrogen chain.
        coords = np.array([(0.0, 0.0, i * R) for i in range(int(n))], dtype=np.float64)
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def square(cls, shape: int | tuple[int, int], R: float) -> HydrogenLattice:
        # Build a centered square lattice.
        nx, ny = (shape, shape) if isinstance(shape, int) else shape
        coords = np.array([(ix * R, iy * R, 0.0) for ix in range(nx) for iy in range(ny)], dtype=np.float64)
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def cubic(cls, shape: int | tuple[int, int, int], R: float) -> HydrogenLattice:
        # Build a centered cubic lattice.
        nx, ny, nz = (shape, shape, shape) if isinstance(shape, int) else shape
        coords = np.array(
            [(ix * R, iy * R, iz * R) for ix in range(nx) for iy in range(ny) for iz in range(nz)],
            dtype=np.float64,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def sheet(cls, shape: int | tuple[int, int], R: float) -> HydrogenLattice:
        # Build a centered triangular sheet.
        nx, ny = (shape, shape) if isinstance(shape, int) else shape
        a1 = np.array([R, 0.0, 0.0])
        a2 = np.array([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        coords = np.array([ix * a1 + iy * a2 for ix in range(nx) for iy in range(ny)], dtype=np.float64)
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)

    @classmethod
    def pyramid(cls, n: int, R: float) -> HydrogenLattice:
        # Build a centered tetrahedral pyramid.
        a1 = np.array([R, 0.0, 0.0])
        a2 = np.array([0.5 * R, 0.5 * np.sqrt(3.0) * R, 0.0])
        a3 = np.array([0.5 * R, R / (2.0 * np.sqrt(3.0)), np.sqrt(2.0 / 3.0) * R])
        coords = np.array(
            [ix * a1 + iy * a2 + iz * a3 for ix in range(n) for iy in range(n - ix) for iz in range(n - ix - iy)],
            dtype=np.float64,
        )
        coords -= coords.mean(axis=0, keepdims=True)
        return cls(coords)


NAME = "h16"
STEPS = 1000
CHECKPOINT_EVERY = 1000


def configure() -> None:
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)


def main() -> None:
    configure()

    # Build the hydrogen lattice molecule.
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
    sector = hilbert.DetSector(norb, n_alpha + n_beta, n_alpha - n_beta)
    H = operator.Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())
    H.save(f"{NAME}.FCIDUMP")

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")
    print(f"orth err   : {np.linalg.norm(coeff.T @ s @ coeff - np.eye(norb)):.3e}")

    # Build a spin-orbital reference in the OAO basis.
    ref_mat = utils.ref_init(sector, np.linalg.solve(coeff, mf.mo_coeff))

    model = GBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64,),
        ref_mat=jnp.asarray(ref_mat),
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = utils.chain_init(sector, ref_mat, n_chains=sampler.n_chains, seed=0)

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    scale = optax.linear_schedule(
        init_value=0.0,
        end_value=-5.0e-2,
        transition_steps=100,
    )
    optimizer = psr(shift=1.0e-3, mu=0.95, scale=scale)
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(file=f"{NAME}.jsonl", every=10)
    obs = {"s2": operator.S2(sector)}

    vmc.run(
        STEPS,
        obs=obs,
        logger=log,
        profile=True,
        checkpoint=f"{NAME}_{{step:05d}}.npz",
        checkpoint_every=CHECKPOINT_EVERY,
    )


if __name__ == "__main__":
    main()
