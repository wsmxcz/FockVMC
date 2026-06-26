import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, lo, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import PBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

from helper import HydrogenLattice, chain_init


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
        atom=HydrogenLattice.cubic([4, 4, 4], 2.00).atom,
        basis="sto-6g",
        unit="Angstrom",
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    # Build OAO integrals.
    S = mol.intor_symmetric("int1e_ovlp")
    C = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)
    assert np.allclose(C.T @ S @ C, np.eye(norb), atol=1e-10)

    h1e = np.asarray(C.T @ mf.get_hcore() @ C, dtype=np.float64)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, C), norb), dtype=np.float64)

    h1e[np.abs(h1e) < 1e-8] = 0.0
    eri[np.abs(eri) < 1e-8] = 0.0

    # Build Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())

    # Build reference.
    mo_oao = np.linalg.solve(C, mf.mo_coeff)
    assert np.allclose(mo_oao.T @ mo_oao, np.eye(norb), atol=1e-10)

    ref_mat = np.zeros((n_alpha + n_beta, 2 * norb), dtype=np.float64)
    ref_mat[:n_alpha, :norb] = mo_oao[:, :n_alpha].T
    ref_mat[n_alpha:, norb:] = mo_oao[:, :n_beta].T

    # Solve benchmark.
    # solver = fci.direct_spin0.FCI(mol)
    # e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    # s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    # print(f"SCF energy : {mf.e_tot:.12f}")
    # print(f"FCI energy : {e_fci:.12f}")
    # print(f"S^2        : {s2:.6f}")

    # Initialize VMC.
    model = PBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64, 64),
        ref_mat=jnp.asarray(ref_mat),
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
    )

    chains = chain_init(H.sector, mo_oao, sampler.n_chains, seed=0)
    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1e-3,
        eps2=1e-6,
        eloc_sample=1024,
        assemble_mode="flat",
    )

    optimizer = optax.chain(
        psr(shift=1e-3),
        optax.scale(-5e-2),
    )

    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="hchain_log.jsonl",
        every=10,
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
        ],
    )
    steps = 1000

    # Run optimization.
    for step in range(steps):
        stats = dict(vmc.step())
        # stats["error"] = abs(float(stats["energy"]) - float(e_fci))
        log.add(step, stats)

    # log.plot("energy", benchmark=e_fci)
    plt.savefig("convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
