from __future__ import annotations

import jax
import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf

import libdet
from detnqs.model import RBM
from detnqs.sampler import MCSampler
from detnqs.vstate import MCState


@pytest.fixture(scope="module")
def h2():
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
    ham = libdet.Hamiltonian.rhf(h1, eri, ecore=mol.energy_nuc())
    energy, _ = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelec,
        ecore=mol.energy_nuc(),
    )
    return ham, norb, nelec, float(energy)


def _hf_det(norb: int, nelec: tuple[int, int]) -> np.ndarray:
    det = np.zeros((1, 2, (norb + 63) // 64), dtype=np.uint64)
    for spin, nocc in enumerate(nelec):
        for p in range(nocc):
            det[0, spin, p // 64] |= np.uint64(1) << np.uint64(p % 64)
    return det


def test_hci(h2) -> None:
    ham, norb, nelec, fci_energy = h2
    dets = _hf_det(norb, nelec)

    for _ in range(4):
        matrix = ham.matrix(dets).toarray()
        _, vectors = np.linalg.eigh(matrix)
        candidates = ham.expand(
            dets,
            0.0,
            coeffs=vectors[:, 0],
            exclude=dets,
        )
        if len(candidates) == 0:
            break
        dets = np.ascontiguousarray(np.concatenate((dets, candidates)))

    energy = np.linalg.eigvalsh(ham.matrix(dets).toarray())[0]
    assert energy == pytest.approx(fci_energy, abs=1.0e-10)


def test_detnqs(h2) -> None:
    ham, norb, nelec, _ = h2
    sampler = MCSampler(
        n_samples=16,
        n_chains=8,
        thermal_steps=1,
        proposal="ham",
        proposal_eps=1.0e-8,
        blur=0.5,
        blur_eps=1.0e-8,
    )
    state = MCState.init(
        model=RBM(norb=norb, alpha=1),
        hamiltonian=ham,
        sampler=sampler,
        n_alpha=nelec[0],
        n_beta=nelec[1],
        key=jax.random.key(5),
        eloc_eps1=1.0e-8,
        eloc_eps2=0.0,
        eloc_sample=8,
    )

    state, stats = state.expect()
    assert np.isfinite(stats["energy"])
    assert np.isfinite(stats["variance"])
    assert stats["n_sample"] == 16

    _, repeated = state.expect()
    assert np.isfinite(repeated["energy"])
