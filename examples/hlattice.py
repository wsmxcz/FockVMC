from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyscf import ao2mo, gto, lo, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

from helper import HydrogenLattice, chain_init, ref_init


def main() -> None:
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=32768,
        backward_chunk=32768,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Build molecule.
    mol = gto.M(
        atom=HydrogenLattice.cubic([4, 4, 4], 2.00).atom,
        basis="sto-6g",
        unit="Angstrom",
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    # Build OAO integrals.
    S = mol.intor_symmetric("int1e_ovlp")
    C = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)
    assert np.allclose(C.T @ S @ C, np.eye(norb), atol=1e-10)

    h1e = np.asarray(C.T @ mf.get_hcore() @ C, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, C), norb), dtype=np.float64)

    h1e[np.abs(h1e) < 1e-8] = 0.0
    eri[np.abs(eri) < 1e-8] = 0.0

    # Build Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # Build reference.
    mo_oao = np.linalg.solve(C, mf.mo_coeff)
    ref_mat = ref_init(sector, mo_oao)

    model = Backflow(
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
    )

    chains = chain_init(H.sector, mo_oao, n_chains=sampler.n_chains, seed=0)
    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1e-3,
        eps2=1e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    steps = 1000

    optimizer = optax.chain(
        psr(shift=1e-3, mu=0.95),
        optax.scale(-5e-2),
    )
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(file="hchain_log.jsonl", every=10, verbose=2)

    # Run optimization.
    for step in range(steps):
        log.add(step, vmc.step())


if __name__ == "__main__":
    main()
