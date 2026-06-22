from __future__ import annotations

import jax
import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert
from detnqs import operator
from detnqs.model import RBM
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


def test_h2_workflow() -> None:
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    norb = int(mf.mo_coeff.shape[1])
    nelec = tuple(map(int, mol.nelec))
    h1 = np.asarray(mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff)
    eri = np.asarray(ao2mo.restore(8, ao2mo.kernel(mol, mf.mo_coeff), norb))
    sector = hilbert.DetSector(norb, nelec[0], nelec[1])
    H = operator.Hamiltonian(sector, h1, eri, ecore=mol.energy_nuc())

    energy, _ = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelec,
        ecore=mol.energy_nuc(),
    )

    basis = sector.reference(1)
    for _ in range(4):
        matrix = H.matrix(basis).toarray()
        _, vec = np.linalg.eigh(matrix)
        bra = H.expand(basis, 0.0, scale=vec[:, 0], exclude=basis)
        if bra.shape[0] == 0:
            break
        basis = np.ascontiguousarray(np.concatenate((basis, bra)))

    actual = np.linalg.eigvalsh(H.matrix(basis).toarray())[0]
    assert actual == pytest.approx(float(energy), abs=1.0e-10)

    sampler = MCSampler(n_samples=16, n_chains=8, thermal_steps=1, blur=0.5)
    state = MCState.init(
        model=RBM(norb=norb, alpha=1),
        H=H,
        sampler=sampler,
        key=jax.random.key(5),
        eps1=1.0e-3,
        eloc_sample=8,
    )
    _, stats = state.expect()
    assert np.isfinite(stats["energy"])
    assert np.isfinite(stats["variance"])
    assert stats["n_sample"] == 16
