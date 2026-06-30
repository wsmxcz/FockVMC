from __future__ import annotations

import jax
import numpy as np
import optax

from pyscf import ao2mo, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


def main() -> None:
    # Runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Molecule.
    mol = gto.M(
        atom="""
        N   0.53920000,  0.00000000,  0.00000000
        N  -0.53920000,  0.00000000,  0.00000000
        """,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = int(mf.mo_coeff.shape[1])
    n_alpha, n_beta = map(int, mol.nelec)

    h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())
    H.save("N2_ham.npz")

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")

    model = GBackflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=None,
    )

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=sector.reference(sampler.n_chains),
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    steps = 1000
    checkpoint_every = 1000
    scale = optax.linear_schedule(0.0, -5.0e-2, transition_steps=100)
    optimizer = psr(shift=1.0e-3, mu=0.95, scale=scale)
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(file="N2.jsonl", every=10)
    obs = {"s2": operator.S2(sector)}

    for _ in range(steps):
        rec = vmc.step(obs=obs, profile=True)
        log.add(rec)

        step = int(rec["step"])
        if step % checkpoint_every == 0 or step == steps:
            vmc.save(f"N2_{step:05d}.npz")


if __name__ == "__main__":
    main()
