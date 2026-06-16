from collections import defaultdict

import jax
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, scf

from libdet import Hamiltonian
from detnqs.model import Backflow, RBackflow, RBM
from detnqs.vstate import MCState
from detnqs.sampler.mcmc import MCSampler
from detnqs.driver import VMC
from detnqs.optimizer import psr, minsr
from detnqs import utils as dq_utils

import utils as run_utils


dq_utils.batch.configure(chunk=8192)
dq_utils.precision.configure("double")
jax.config.update("jax_debug_nans", False)
jax.config.update("jax_log_compiles", False)


mol = gto.M(
    atom=
    '''
    N   0.53920000,  0.00000000,  0.0000000
    N   -0.539200000,  0.00000000,  0.0000000
    ''',
    # atom=
    # '''
    # O   0.00000000,  0.00000000,  0.00000000
    # H   0.75700000,  0.00000000,  0.58590000
    # H  -0.75700000,  0.00000000,  0.58590000
    # ''',
    # atom=
    # '''
    # Li   3.732, 0.25, 0.0
    # Li   2.0, 0.25, 0.0
    # O  2.866, -0.25, 0.0
    # ''',
    basis="sto-3g",
    unit="Angstrom",
    spin=0,
    verbose=0,
)

mf = scf.RHF(mol).run()
norb = mf.mo_coeff.shape[1]
n_alpha, n_beta = mol.nelec

# Canonical MO integrals.
h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

ham = Hamiltonian.rhf(h1e, eri, ecore=mol.energy_nuc())

# e_fci = -109.099941428008 # N2_631g
# e_fci = -87.892693 # Li2O_sto3g
# e_fci = -76.243769 # H2O_ccpvdz

solver = fci.direct_spin0.FCI(mol)
e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

print(f"SCF energy : {mf.e_tot:.12f}")
print(f"FCI energy : {e_fci:.12f}")
print(f"S^2        : {s2:.6f}")


model = RBackflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
# model = Backflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))
# model = RBM(norb=norb, alpha=1)

sampler = MCSampler(
    n_samples=4096,
    n_chains=4096,
    thermal_steps=128,
    proposal="ham",
    blur=0.5,
)

state = MCState.init(
    model=model,
    hamiltonian=ham,
    sampler=sampler,
    n_alpha=n_alpha,
    n_beta=n_beta,
    chain_init="hf",
    key=jax.random.key(0),
    eloc_sample=1024
)

optimizer = optax.chain(
    psr(shift=1e-3),
    # minsr(shift=1e-3),
    optax.scale(-5e-2),
)

vmc = VMC.init(state, optimizer)

metrics = {
    "step": ("Step", "d", 5),
    "energy": ("Energy", ".8f", 16),
    "error": ("|E-E0|", ".2e", 10),
    "variance": ("Var", ".2e", 10),
    "accept": ("Acc", ".1%", 7),
    "ess_frac": ("ESS", ".1%", 7),
    "n_unique": ("N_u", ".0f", 7),
    "n_eval": ("N_f", ".0f", 7),
    "alpha": ("alpha", ".3f", 7),
}

line = run_utils.print_metrics(metrics)

history = []
total_times = defaultdict(float)
steps = 1000

for step in range(steps):
    stats = dict(vmc.step())
    stats["step"] = step
    stats["error"] = abs(float(stats["energy"]) - float(e_fci))

    history.append(float(stats["energy"]))

    for key, value in stats.items():
        if key.startswith("time_"):
            total_times[key] += float(value)

    if step % 10 == 0 or step == steps - 1:
        run_utils.print_metrics(metrics, stats)

print(line)
run_utils.print_times(total_times)
print(line)

run_utils.plot_convergence(history, e_fci)
plt.savefig('convergence.pdf', dpi=300, bbox_inches='tight')
plt.close()