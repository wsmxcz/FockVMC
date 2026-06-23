from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, lo, scf

from detnqs import utils as dq_utils
from detnqs import hilbert, operator
from detnqs.driver import VMC
from detnqs.model import RBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

import utils as run_utils


def main():
    # numerical defaults.
    dq_utils.batch.configure(chunk=8192)
    dq_utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # problem and integral tensors.
    mol = gto.M(
        atom="; ".join(f"H 0 0 {i * 2.00}" for i in range(16)),
        basis="sto-6g",
        unit="Angstrom",
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    # Lowdin OAO / site basis.
    S = mol.intor_symmetric("int1e_ovlp")
    C = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)
    assert np.allclose(C.T @ S @ C, np.eye(norb), atol=1e-10)

    h1e = np.asarray(C.T @ mf.get_hcore() @ C, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, C), norb), dtype=np.float64)

    h1e[np.abs(h1e) < 1e-8] = 0.0
    eri[np.abs(eri) < 1e-8] = 0.0

    # sector and Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # reference matrix in the OAO basis.
    mo_oao = np.linalg.solve(C, mf.mo_coeff)
    assert np.allclose(mo_oao.T @ mo_oao, np.eye(norb), atol=1e-10)

    ref_mat = np.zeros((n_alpha + n_beta, 2 * norb), dtype=np.float64)
    ref_mat[:n_alpha, :norb] = mo_oao[:, :n_alpha].T
    ref_mat[n_alpha:, norb:] = mo_oao[:, :n_beta].T

    e_fci = -7.66653 # H16_2.00A
    # e_fci = -14.46061 # H30_3.60Bohr
    # e_fci = -24.10276	# H50_3.60Bohr

    # solver = fci.direct_spin0.FCI(mol)
    # e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    # s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    # print(f"SCF energy : {mf.e_tot:.12f}")
    # print(f"FCI energy : {e_fci:.12f}")
    # print(f"S^2        : {s2:.6f}")

    # variational state and optimizer.
    model = RBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64,),
        ref_mat=jnp.asarray(ref_mat),
    )

    sampler = MCSampler(
        n_samples=4096,
        n_chains=4096,
        thermal_steps=1024,
        proposal="ham",
        blur=0.5,
    )

    chains = H.sector.random(sampler.n_chains, 0)
    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1e-3,
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
        "accept": ("Acc", ".1%", 8),
        "ess_frac": ("ESS", ".1%", 8),
        "n_unique": ("N_u", ".0f", 8),
        "n_forward": ("N_f", ".0f", 8),
        "unique_frac": ("R_u", ".1%", 8),
        "forward_frac": ("R_f", ".1%", 8),
        "alpha": ("alpha", ".3f", 8),
    }

    line = run_utils.print_metrics(metrics)

    history = []
    total_times = defaultdict(float)
    steps = 1000

    # optimization loop.
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
    plt.savefig("convergence_standard.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
