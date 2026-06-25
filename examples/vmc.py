import jax
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow, RBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState



def main():
    # numerical defaults.
    utils.batch.configure(forward_chunk=8192, backward_chunk=4096)
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # problem and integral tensors.
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

    h1e = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb), dtype=np.float64)

    # sector and Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # e_fci = -109.099941428008 # N2_631g
    # e_fci = -87.892693 # Li2O_sto3g
    # e_fci = -76.243769 # H2O_ccpvdz

    solver = fci.direct_spin0.FCI(mol)
    e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"FCI energy : {e_fci:.12f}")
    print(f"S^2        : {s2:.6f}")

    # variational state and optimizer.
    model = RBackflow(norb=norb, n_alpha=n_alpha, n_beta=n_beta, hidden=(64,))

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=16,
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
        eloc_sample=1024,
        assemble_mode="unique",
    )

    optimizer = optax.chain(
        psr(shift=1e-3),
        optax.scale(-5e-2),
    )
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="vmc_log.jsonl",
        every=1,
        keys=[
            "step", "energy", "error", "variance", "accept",
            "ess_frac", "n_unique", "n_forward", "forward_frac", "alpha",
        ],
    )
    steps = 1000

    # optimization loop.
    for step in range(steps):
        stats = dict(vmc.step())
        stats["error"] = abs(float(stats["energy"]) - float(e_fci))
        log.add(step, stats)

    log.plot("energy", benchmark=e_fci)
    plt.savefig("convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
