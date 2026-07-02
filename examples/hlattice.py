from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyscf import ao2mo, gto, lo, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

from helper import HydrogenLattice, chain_init, ref_init


def main() -> None:
    # Runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("single")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Molecule.
    mol = gto.M(
        atom=HydrogenLattice.chain(16, 2.0).atom,
        basis="sto-6g",
        unit="Angstrom",
        spin=0,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    norb = mf.mo_coeff.shape[1]
    n_alpha, n_beta = mol.nelec

    # OAO integrals.
    S = mol.intor_symmetric("int1e_ovlp")
    C = lo.orth.orth_ao(mol, method="lowdin", pre_orth_ao=None)
    assert np.allclose(C.T @ S @ C, np.eye(norb), atol=1.0e-10)

    h1e = np.asarray(C.T @ mf.get_hcore() @ C, dtype=np.float64)
    eri = np.asarray(
        ao2mo.restore(8, ao2mo.kernel(mol, C), norb),
        dtype=np.float64,
    )

    h1e[np.abs(h1e) < 1.0e-8] = 0.0
    eri[np.abs(eri) < 1.0e-8] = 0.0

    # Hamiltonian.
    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=mol.energy_nuc())
    H.save("h10.npz")

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"active     : ({n_alpha + n_beta}e, {norb}o)")

    # Reference.
    mo_oao = np.linalg.solve(C, mf.mo_coeff)
    ref_mat = ref_init(sector, mo_oao)

    model = GBackflow(
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
        alpha=None,
    )

    chains = chain_init(sector, mo_oao, n_chains=sampler.n_chains, seed=0)

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    # Optimizer.
    steps = 1000
    checkpoint_every = 1000

    lr = optax.linear_schedule(
        init_value=0.0,
        end_value=-5.0e-2,
        transition_steps=100,
    )

    optimizer = psr(
        shift=1.0e-3,
        mu=0.95,
        scale=lr,
    )

    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="h10.jsonl",
        every=10,
    )

    obs = {
        "s2": operator.S2(sector),
    }

    # Optimization.
    for _ in range(steps):
        rec = vmc.step(obs=obs, profile=True)
        log.add(rec)

        step = int(rec["step"])
        if step % checkpoint_every == 0 or step == steps:
            vmc.save(f"h10_{step:05d}.npz")


if __name__ == "__main__":
    main()