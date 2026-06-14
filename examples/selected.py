import jax
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, scf

from libdet import Hamiltonian
from detnqs.model import Backflow, RBM
from detnqs.vstate import SelectedState
from detnqs.vstate.selected import topk_selector
from detnqs.driver import VMC
from detnqs import utils as dq_utils

import utils as run_utils


dq_utils.batch.configure(chunk=8192)
dq_utils.precision.configure("double")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_log_compiles", False)


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

# Canonical MO integrals.
h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

ham = Hamiltonian.rhf(h1e, eri, ecore=mol.energy_nuc())

solver = fci.direct_spin0.FCI(mol)
e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

print(f"SCF energy : {mf.e_tot:.12f}")
print(f"FCI energy : {e_fci:.12f}")
print(f"S^2        : {s2:.6f}")


model = Backflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
# model = RBM(norb=norb, alpha=1)

nword = (norb + 63) // 64
init_v = np.zeros((1, 2, nword), dtype=np.uint64)

for i in range(n_alpha):
    init_v[0, 0, i >> 6] |= np.uint64(1) << np.uint64(i & 63)

for i in range(n_beta):
    init_v[0, 1, i >> 6] |= np.uint64(1) << np.uint64(i & 63)

state = SelectedState.init(
    model=model,
    hamiltonian=ham,
    init_v=init_v,
    key=jax.random.key(0),
)

vmc = VMC.init(state, optax.adamw(1e-3), geometry=False)

metrics = {
    "outer": ("Outer", "d", 5),
    "energy": ("Energy", ".8f", 16),
    "error": ("|E-E0|", ".2e", 10),
    "variance": ("Var", ".2e", 10),
    "n_v": ("N_v", "d", 8),
}

line = run_utils.print_metrics(metrics)

history = []
outer_steps = 5
inner_steps = 100

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
    stats["n_v"] = vmc.state.n_v

    run_utils.print_metrics(metrics, stats)
    run_utils.print_times(times)

print(line)

run_utils.plot_convergence(history, e_fci)
plt.show()