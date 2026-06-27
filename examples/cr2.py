"""
Cr2 (Ahlrichs VDZ) active space models.

Active Spaces:
    - (48e, 42o): All-electron space.
    - (24e, 30o): Frozen Mg-core space (12 spatial orbitals frozen).

Reference:
    Li et al., Phys. Rev. Research 2, 012015 (2020)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, lib, mcscf, scf, symm
from pyscf.gto.basis import bse

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import Backflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

from helper import chain_init, ref_init, warmup


def main():
    space = "24e30o"
    steps = 1000

    # Configure runtime.
    utils.batch.configure(
        forward_chunk=32768,
        backward_chunk=32768,
        param_chunk=None,
        bucket_min=1024,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Build molecule.
    mol = gto.Mole()
    mol.unit = "Angstrom"
    mol.atom = [
        ("Cr", (0.0, 0.0, -0.75)),
        ("Cr", (0.0, 0.0, +0.75)),
    ]
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = True
    mol.symmetry_subgroup = "D2h"
    mol.basis = bse.get_basis("Ahlrichs VDZ", elements="Cr")
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()

    # Build CASSCF orbitals.
    cas_irrep = {
        "Ag": 3,
        "B1u": 3,
        "B2g": 1,
        "B3g": 1,
        "B2u": 1,
        "B3u": 1,
        "B1g": 1,
        "Au": 1,
    }

    mc = mcscf.CASSCF(mf, ncas=12, nelecas=(6, 6))
    mc.conv_tol = 1e-8
    mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
    mc.fcisolver.wfnsym = symm.irrep_name2id(mol.groupname, "Ag")
    mc.natorb = False
    mc.canonicalization = False
    mc.sorting_mo_energy = False

    mo0 = mc.sort_mo_by_irrep(
        [cas_irrep.get(name, 0) for name in mol.irrep_name],
        mo_coeff=mf.mo_coeff,
    )
    mc.kernel(mo0)

    s = mf.get_ovlp()
    orbsym = symm.label_orb_symm(
        mol,
        mol.irrep_id,
        mol.symm_orb,
        mc.mo_coeff,
        s=s,
        check=True,
    )
    mo_tag = lib.tag_array(mc.mo_coeff, orbsym=np.asarray(orbsym, dtype=int))
    mo_cas, _, eps = mc.canonicalize(
        mo_coeff=mo_tag,
        ci=mc.ci,
        sort=False,
        cas_natorb=True,
    )

    n_core = int(mc.ncore)
    n_cas = int(mc.ncas)
    assert mo_cas.shape[1] == 42
    assert n_core == 18
    assert n_cas == 12

    # Select active space.
    if space == "48e42o":
        mo = mo_cas
        ncore = 0
        norb = 42
        n_alpha = n_beta = 24

    elif space == "24e30o":
        core = np.arange(n_core)
        order = np.argsort(np.asarray(eps)[core], kind="stable")

        frozen = core[order[:12]]
        kept_core = core[order[12:]]
        active = np.arange(n_core, n_core + n_cas)
        virtual = np.arange(n_core + n_cas, mo_cas.shape[1])

        mo = mo_cas[:, np.concatenate([frozen, kept_core, active, virtual])]
        ncore = 12
        norb = 30
        n_alpha = n_beta = 12

    else:
        raise ValueError("space must be '48e42o' or '24e30o'")

    # Build Hamiltonian.
    mc_eff = mcscf.CASCI(mf, ncas=norb, nelecas=(n_alpha, n_beta))
    mc_eff.ncore = ncore
    mc_eff.mo_coeff = mo

    h1e, ecore = mc_eff.get_h1eff(mo_coeff=mo)
    eri = ao2mo.restore(8, mc_eff.get_h2eff(mo_coeff=mo), norb)

    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    h1e[np.abs(h1e) < 1e-10] = 0.0
    eri[np.abs(eri) < 1e-10] = 0.0

    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=float(ecore))

    print(f"space        : {space}")
    print(f"RHF energy   : {mf.e_tot:.12f}")
    print(f"CASSCF energy: {mc.e_tot:.12f}")
    print(f"active space : ({2 * n_alpha}e, {norb}o)")
    print(f"frozen core  : {ncore}")
    print(f"ecore        : {float(ecore):.12f}")

    # Build reference.
    occ_mo = mf.mo_coeff[:, : mol.nelectron // 2]
    act_mo = mo[:, ncore:]
    proj = act_mo.T @ s @ occ_mo
    ref_coeff = np.linalg.svd(proj, full_matrices=False)[0][:, :n_alpha]
    ref_mat = ref_init(sector, ref_coeff)

    model = Backflow(
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

    chains = chain_init(H.sector, ref_coeff, n_chains=sampler.n_chains, seed=0)

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

    lr = partial(warmup, start=0.0, end=5.0e-2, steps=100)

    optimizer = optax.chain(
        psr(shift=1e-3, mu=0.95, beta=0.995),
        optax.scale_by_schedule(lr),
        optax.scale(-1.0),
    )
    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file=f"cr2_{space}.jsonl",
        every=10,
        keys=[
            "step",
            "energy",
            "variance",
            "accept",
            "ess_frac",
            "n_unique",
            "n_forward",
            "forward_frac",
            "alpha",
        ],
    )

    # Run optimization.
    for step in range(steps):
        log.add(step, dict(vmc.step()))

    log.plot("energy")
    plt.savefig(f"cr2_{space}.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
