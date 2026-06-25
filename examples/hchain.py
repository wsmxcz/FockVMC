import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from pyscf import ao2mo, fci, gto, lo, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import RBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


def chain_init(sector, mo_coeff, n_chains, seed=0):
    """Sample determinant chains from an RHF Slater determinant."""
    rng = np.random.default_rng(seed)
    n_chains = int(n_chains)

    coeff = np.asarray(mo_coeff, dtype=np.float64)
    chains = sector.zeros(n_chains)

    chain = np.arange(n_chains, dtype=np.int64)
    item = chain[:, None]

    for spin, n_elec in enumerate((sector.n_alpha, sector.n_beta)):
        n_elec = int(n_elec)

        q = np.linalg.qr(coeff[:, :n_elec], mode="reduced")[0]
        occ = np.empty((n_chains, n_elec), dtype=np.int64)
        basis = np.broadcast_to(q, (n_chains, *q.shape)).copy()

        for k in range(n_elec):
            probs = np.vecdot(basis, basis, axis=-1)
            probs[chain[:, None], occ[:, :k]] = 0.0
            probs /= probs.sum(axis=1, keepdims=True)

            u = rng.random((n_chains, 1))
            p = (np.cumsum(probs, axis=1) < u).sum(axis=1)
            occ[:, k] = np.minimum(p, sector.norb - 1)

            if k + 1 == n_elec:
                break

            row = basis[chain, occ[:, k]].copy()
            col = np.argmax(np.abs(row), axis=1)
            pivot = row[chain, col]

            basis -= (basis[chain, :, col] / pivot[:, None])[:, :, None] * row[:, None, :]

            last = basis.shape[2] - 1
            basis[chain, :, col] = basis[:, :, last]
            basis = np.linalg.qr(basis[:, :, :last], mode="reduced")[0]

        word = occ >> 6
        bit = (occ & 63).astype(np.uint64)
        np.bitwise_or.at(chains[:, spin, :], (item, word), np.uint64(1) << bit)

    return np.ascontiguousarray(chains)


def main():
    # numerical defaults.
    utils.batch.configure(forward_chunk=8192, param_chunk=32648)
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # problem and integral tensors.
    mol = gto.M(
        atom="; ".join(f"H 0 0 {i * 2.00}" for i in range(10)),
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

    # e_fci = -7.66653 # H16_2.00A
    # e_fci = -14.46061 # H30_3.60Bohr
    # e_fci = -24.10276	# H50_3.60Bohr

    solver = fci.direct_spin0.FCI(mol)
    e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    s2, _ = fci.spin_op.spin_square(ci, norb, mol.nelec)

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"FCI energy : {e_fci:.12f}")
    print(f"S^2        : {s2:.6f}")

    # variational state and optimizer.
    model = RBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64,),
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
    plt.savefig("convergence_standard.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
