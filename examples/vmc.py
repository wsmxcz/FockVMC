from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


def main() -> None:
    # Configure runtime.
    utils.batch.configure(
        forward_chunk=8192,
        backward_chunk=1024,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Build molecule.
    mol = gto.M(
        atom="""
        N   0.53920000,  0.00000000,  0.0000000
        N   -0.539200000,  0.00000000,  0.0000000
        """,
        basis="sto-3g",
        unit="Angstrom",
        spin=0,
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
    solver.nroots = 2
    e_fci, ci = solver.kernel(h1e, eri, norb, mol.nelec, ecore=mol.energy_nuc())
    s2, _ = fci.spin_op.spin_square(ci[0], norb, mol.nelec)

    print(f"SCF energy : {mf.e_tot:.12f}")
    print(f"FCI energy : {e_fci[0]:.12f}")
    print(f"S^2        : {s2:.6f}")

    # Initialize VMC.
    model = GBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(64,),
    )

    sampler = MCSampler(
        n_samples=1024,
        n_chains=1024,
        thermal_steps=0,
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
        eps2=1e-6,
        eloc_sample=1024,
        assemble_mode="unique",
    )

    steps = 500

    lr = optax.linear_schedule(
        init_value=0.0,
        end_value=5.0e-2,
        transition_steps=500,
    )

    optimizer = optax.chain(
        psr(shift=1e-3, mu=0.95),
        optax.scale_by_schedule(lr),
        optax.scale(-1.0),
    )

    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file="N2_GBackflow.jsonl",
        every=10,
        keys=(
            "step",
            "energy",
            "eloc_var",
            "ess_frac",
            "accept",
            "s2",
            "s2_var",
            "n_forward",
        ),
    )

    # Run with explicit spin diagnostics.
    obs = {"s2": operator.S2(sector), "sz": operator.Sz(sector)}
    for step in range(steps):
        log.add(step, vmc.step(obs=obs))

    # Posterior analysis with a larger independent sample.
    post_state = vmc.state.replace(
        sampler=replace(vmc.state.sampler, n_samples=131072)
    )
    post_state, post, data = post_state.expect(obs=obs, data=True)
    del post_state

    sup = utils.analysis.support(data["w"])
    exc = utils.analysis.excitation(data["x"], data["w"], sector.reference(1))
    tail = utils.analysis.tail(data["w"], data["eloc"], post["energy"])

    print("\nPosterior")
    print(f"Energy     : {post['energy']:.12f}")
    print(f"Eloc var   : {post['eloc_var']:.6e}")
    print(f"S^2        : {post['s2']:.8f}")
    print(f"Sz         : {post['sz']:.8f}")

    print("\nSupport")
    print(f"N2         : {sup['n2']:.3f}")
    print(f"Entropy    : {sup['entropy']:.6f}")
    print(f"Top 1      : {sup['top1']:.3%}")
    print(f"Top 5      : {sup['top5']:.3%}")
    print(f"Top 10     : {sup['top10']:.3%}")

    print("\nExcitation")
    for key in sorted(exc, key=lambda s: int(s[3:])):
        print(f"{key:<10s} : {exc[key]:.3%}")

    print("\nTail")
    print(f"R q90      : {tail['r_q90']:.6e}")
    print(f"R q99      : {tail['r_q99']:.6e}")
    print(f"R q999     : {tail['r_q999']:.6e}")
    print(f"Var max    : {tail['var_max']:.6e}")


if __name__ == "__main__":
    main()
