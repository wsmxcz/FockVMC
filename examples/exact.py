from __future__ import annotations

import jax
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.vstate import ExactState


def main() -> None:
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
    state = ExactState.init(
        model=model,
        H=H,
        key=jax.random.key(0),
    )

    steps = 500

    optimizer = optax.chain(
        psr(shift=1e-3, mu=0.95),
        optax.scale(-5e-2),
    )
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(every=10, verbose=2)

    # Run optimization.
    for step in range(steps):
        log.add(step, vmc.step())


if __name__ == "__main__":
    main()
