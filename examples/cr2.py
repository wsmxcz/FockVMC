"""
Cr2 (Ahlrichs VDZ) active space models.

Active Spaces:
    - (48e, 42o): All-electron space.
    - (24e, 30o): Frozen Mg-core space (12 spatial orbitals frozen).

Reference:
    Li et al., Phys. Rev. Research 2, 012015 (2020)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pyscf import ao2mo, fci, gto, lib, mcscf, scf, symm
from pyscf.gto.basis import bse

from detnqs import hilbert, operator, utils
from detnqs.driver import VMC
from detnqs.model import GBackflow
from detnqs.optimizer import psr
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState

from helper import chain_init, ref_init


def main() -> None:
    case = "24e30o"  # "24e30o" or "48e42o"

    # Runtime.
    utils.batch.configure(
        forward_chunk=131072,
        backward_chunk=8192,
        param_chunk=None,
        bucket_min=8192,
    )
    utils.precision.configure("double")
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_log_compiles", False)

    # Molecule.
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
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 200
    mf.kernel()

    # CASSCF orbitals.
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
    mc.conv_tol = 1.0e-8
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

    if case == "24e30o":
        # Freeze the lowest twelve core orbitals.
        n_core = int(mc.ncore)
        n_cas = int(mc.ncas)

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

    elif case == "48e42o":
        # All-electron active Hamiltonian.
        mo = mo_cas

        ncore = 0
        norb = 42
        n_alpha = n_beta = 24

    else:
        raise ValueError(f"Unknown case: {case!r}")

    # Hamiltonian.
    mc_eff = mcscf.CASCI(mf, ncas=norb, nelecas=(n_alpha, n_beta))
    mc_eff.ncore = ncore
    mc_eff.mo_coeff = mo

    h1e, ecore = mc_eff.get_h1eff(mo_coeff=mo)
    eri = ao2mo.restore(8, mc_eff.get_h2eff(mo_coeff=mo), norb)

    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    h1e[np.abs(h1e) < 1.0e-12] = 0.0
    eri[np.abs(eri) < 1.0e-12] = 0.0

    sector = hilbert.DetSector(norb, n_alpha, n_beta)
    H = operator.Hamiltonian(sector, h1e, eri, ecore=float(ecore))
    H.save(f"cr2_{case}_ham.npz")

    print(f"RHF energy    : {mf.e_tot:.12f}")
    print(f"CASSCF energy : {mc.e_tot:.12f}")
    print(f"active        : ({n_alpha + n_beta}e, {norb}o)")
    print(f"frozen core   : {ncore}")
    print(f"ecore         : {float(ecore):.12f}")

    # Reference.
    occ_mo = mf.mo_coeff[:, : mol.nelectron // 2]
    act_mo = mo[:, ncore:]
    proj = act_mo.T @ s @ occ_mo
    ref_coeff = np.linalg.svd(proj, full_matrices=False)[0][:, :n_alpha]
    ref_mat = ref_init(sector, ref_coeff)

    # State.
    model = GBackflow(
        norb=norb,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=(256, 256),
        ref_mat=jnp.asarray(ref_mat),
    )

    sampler = MCSampler(
        n_samples=8192,
        n_chains=8192,
        thermal_steps=0,
        proposal="ham",
        blur=0.5,
        alpha=1.0,
    )

    chains = chain_init(sector, ref_coeff, n_chains=sampler.n_chains, seed=0)

    state = MCState.init(
        model=model,
        H=H,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(0),
        eps1=1.0e-3,
        eps2=1.0e-6,
        eloc_sample=1024,
        assemble_mode="flat",
    )

    # Optimizer.
    steps = 5000
    checkpoint_every = 500

    lr = optax.linear_schedule(
        init_value=0.0,
        end_value=-5.0e-2,
        transition_steps=1000,
    )

    optimizer = psr(
        shift=1.0e-3,
        mu=0.95,
        scale=lr,
    )

    vmc = VMC.init(state, optimizer)

    log = utils.Logger(
        file=f"cr2_{case}.jsonl",
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
            vmc.save(f"cr2_{case}_{step:05d}.npz")


if __name__ == "__main__":
    main()