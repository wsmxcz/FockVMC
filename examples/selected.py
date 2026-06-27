from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.vstate import SelectedState, topk_selector
from helper import warmup


def main():
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=32768,
        backward_chunk=32768,
        param_chunk=32768,
        bucket_min=1024,
    )
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Build molecule.
    mol = gto.M(
        atom="""
        O   0.00000000,  0.00000000,  0.00000000
        H   0.75700000,  0.00000000,  0.58590000
        H  -0.75700000,  0.00000000,  0.58590000
        """,
        basis="sto-3g",
        unit="Angstrom",
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

    # Initialize state.
    model = Backflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
    basis = H.sector.reference(1)
    state = SelectedState.init(
        model=model,
        H=H,
        basis=basis,
        key=jax.random.key(0),
    )

    outer_steps = 5
    inner_steps = 100
    vmc = VMC.init(state, optax.adamw(1e-3), geometry=False)

    log = utils.Logger(
        every=1,
        keys=["outer", "energy", "error", "variance", "n_basis"],
    )

    # Run optimization.
    for outer in range(outer_steps):
        vmc.state = vmc.state.evolve(topk_selector(k=128), eps=1e-6)

        for _ in range(inner_steps):
            stats = dict(vmc.step())

        stats["outer"] = outer
        stats["error"] = abs(float(stats["energy"]) - float(e_fci))
        stats["n_basis"] = vmc.state.n_basis
        log.add(outer, stats)

    log.plot("energy", x="outer", benchmark=e_fci)
    plt.show()


if __name__ == "__main__":
    main()
