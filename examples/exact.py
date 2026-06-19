import jax
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, scf

from detnqs import utils as dq_utils
from detnqs import hilbert, operator
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.vstate import ExactState

import utils as run_utils


# numerical defaults.
dq_utils.batch.configure(chunk=8192)
dq_utils.precision.configure("double")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_log_compiles", False)


# problem and integral tensors.
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
state = ExactState.init(
    model=model,
    H=H,
    key=jax.random.key(0),
)

optimizer = optax.chain(
    psr(shift=1e-3),
    optax.scale(-5e-2),
)
vmc = VMC.init(state, optimizer)

metrics = {
    "step": ("Step", "d", 5),
    "energy": ("Energy", ".8f", 16),
    "error": ("|E-E0|", ".2e", 10),
    "variance": ("Var", ".2e", 10),
}

line = run_utils.print_metrics(metrics)

history = []
steps = 500

# optimization loop.
for step in range(steps):
    stats = dict(vmc.step())
    stats["step"] = step
    stats["error"] = abs(float(stats["energy"]) - float(e_fci))

    history.append(float(stats["energy"]))

    if step % 10 == 0 or step == steps - 1:
        run_utils.print_metrics(metrics, stats)

print(line)

run_utils.plot_convergence(history, e_fci)
plt.show()
