from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState
from helper import warmup


def main():
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
        atom="""
        N   1.53920000,  0.00000000,  0.0000000
        N   -1.539200000,  0.00000000,  0.0000000
        """,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb),
        dtype=np.float64,
    )

    # Build Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # Solve benchmark.
    solver = fci.direct_spin0.FCI(mol)
    e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"FCI energy : {e_fci:.12f}")
    print(f"S^2        : {s2:.6f}")

    # Initialize VMC.
    model = Backflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64,),
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=256,
        proposal="ham",
        blur=0.5,
    )

    chains = H.sector.reference(sampler.n_chains)
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

    lr = partial(warmup, start=0.0, end=5.0e-2, steps=100)

    optimizer = optax.chain(
        psr(shift=1e-3, mu=0.95, beta=0.995),
        optax.scale_by_schedule(lr),
        optax.scale(-1.0),
    )
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="vmc_log.jsonl",
        every=1,
        keys=[
            "step",
            "energy",
            "error",
            "variance",
            "accept",
            "ess_frac",
            "n_unique",
            "n_forward",
            "forward_frac",
            "alpha",
            "s2",
        ],
    )

    # Run optimization.
    for step in range(steps):
        stats = dict(vmc.step())
        stats["error"] = abs(float(stats["energy"]) - float(e_fci))
        log.add(step, stats)

    log.plot("energy", benchmark=e_fci)
    plt.savefig("convergence1.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
