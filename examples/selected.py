import jax
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, scf

from detnqs import utils as dq_utils
from detnqs import hilbert, operator
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.vstate import SelectedState
from detnqs.vstate.selected import topk_selector

import utils as run_utils


def main():
    # numerical defaults.
    dq_utils.batch.configure(chunk=8192)
    dq_utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # problem and integral.
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
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

    # sector and Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    solver = fci.direct_spin0.FCI(mol)
    e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"FCI energy : {e_fci:.12f}")
    print(f"S^2        : {s2:.6f}")


    # variational state and optimizer.
    model = Backflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
    state = SelectedState.init(
        model=model,
        H=H,
        init="hf",
        key=jax.random.key(0),
    )

    vmc = VMC.init(state, optax.adamw(1e-3), geometry=False)

    metrics = {
        "outer": ("Outer", "d", 5),
        "energy": ("Energy", ".8f", 16),
        "error": ("|E-E0|", ".2e", 10),
        "variance": ("Var", ".2e", 10),
        "n_basis": ("N_basis", "d", 8),
    }

    line = run_utils.print_metrics(metrics)

    history = []
    outer_steps = 5
    inner_steps = 100

    # optimization loops
    for outer in range(outer_steps):
        vmc.state = vmc.state.evolve(topk_selector(k=128), eps=1e-6)

        times = {}

        for _ in range(inner_steps):
            stats = dict(vmc.step())
            history.append(float(stats["energy"]))

            for key, value in stats.items():
                if key.startswith("time_"):
                    times[key] = times.get(key, 0.0) + float(value)

        stats["outer"] = outer
        stats["error"] = abs(float(stats["energy"]) - float(e_fci))
        stats["n_basis"] = vmc.state.n_basis

        run_utils.print_metrics(metrics, stats)
        run_utils.print_times(times)

    print(line)

    run_utils.plot_convergence(history, e_fci)
    plt.show()


if __name__ == "__main__":
    main()
