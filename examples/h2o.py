from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import optax

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow, GBackflow, SBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState
from helper import ref_init, chain_init


def main() -> None:
    # Runtime.
    utils.batch.configure(
        forward_chunk=131072,
        backward_chunk=8192,
        param_chunk=None,
        bucket_min=8192,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Molecule.
    mol = gto.M(
        atom=
        """
        O   0.000000   0.000000  -0.009000
        H   0.000000   1.515263  -1.058898
        H   0.000000  -1.515263  -1.058898
        """,
#        atom=
#        """
#        O   0.000000   0.000000  -0.027000
#        H   0.000000   4.545789  -3.176694
#        H   0.000000  -4.545789  -3.176694
#        """,
        basis="cc-pvdz",
        unit="Bohr",
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    mf = mf.newton()  
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb),
        dtype=np.float64,
    )
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())
    H.save("H2O_ccpvdz_1.0Re.npz")

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")

    model = GBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(256, 256),
    )

    sampler = MCSampler(
        n_samples=8192,
        n_chains=8192,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    chains = H.sector.reference(sampler.n_chains)

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="flat",
    )

    # Optimizer.
    steps = 5000
    checkpoint_every = 500

    lr = optax.linear_schedule(
        init_value=0.0,
        end_value=-5.0e-2,
        transition_steps=1000,
    )

    optimizer = psr(
        shift=1.0e-3,
        mu=0.95,
        scale=lr,
    )

    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="H2O_ccpvdz_1.0Re.jsonl",
        every=10,
    )

    obs = {
        "s2": operator.S2(sector),
    }

    # Optimization.
    for _ in range(steps):
        rec = vmc.step(obs=obs, profile=True)
        log.add(rec)

        step = int(rec["step"])
        if step % checkpoint_every == 0 or step == steps:
            vmc.save(f"H2O_ccpvdz_1.0Re_{step:05d}.npz")


if __name__ == "__main__":
    main()