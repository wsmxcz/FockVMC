from __future__ import annotations

import jax
import numpy as np
import pytest
from pyscf import ao2mo, fci, gto, scf

from detnqs import hilbert
from detnqs import operator
from detnqs.model import RBM
from detnqs.sampler import HeatBath, MCSampler
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
    hi = hilbert.DetSpace(norb, nelec[0], nelec[1])
    H = operator.Hamiltonian(hi, h1, eri, ecore=mol.energy_nuc())
    energy, _ = fci.direct_spin1.kernel(
        h1,
        eri,
        norb,
        nelec,
        ecore=mol.energy_nuc(),
    )
    return H, norb, float(energy)


def test_hci(h2) -> None:
    H, _, fci_energy = h2
    basis = H.space.reference(1)

    for _ in range(4):
        matrix = H.matrix(basis).toarray()
        _, vectors = np.linalg.eigh(matrix)
        candidates = H.expand(
            basis,
            0.0,
            coeffs=vectors[:, 0],
            exclude=basis,
        )
        if len(candidates) == 0:
            break
        basis = np.ascontiguousarray(np.concatenate((basis, candidates)))

    energy = np.linalg.eigvalsh(H.matrix(basis).toarray())[0]
    assert energy == pytest.approx(fci_energy, abs=1.0e-10)


def test_detnqs(h2) -> None:
    H, norb, _ = h2
    sampler = MCSampler(
        n_samples=16,
        n_chains=8,
        thermal_steps=1,
        proposal=HeatBath(1.0e-8),
        blur=0.5,
        blur_eps=1.0e-8,
    )
    state = MCState.init(
        model=RBM(norb=norb, alpha=1),
        H=H,
        sampler=sampler,
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


def test_csf_operator() -> None:
    h1 = np.array([[0.3, 0.2], [0.2, 0.7]], dtype=np.float64)
    eri = np.zeros((2, 2, 2, 2), dtype=np.float64)

    hi = hilbert.CsfSpace(norb=2, nelec=2, spin=0)
    H = operator.Hamiltonian(hi, h1, eri)
    basis = hi.enumerate()
    matrix = H.matrix(basis).toarray()

    assert basis.shape == (3, 2, 1)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(H.diag(basis), np.diag(matrix))
    np.testing.assert_allclose(operator.S2(hi).diag(basis), 0.0)


def test_fermion_primitives() -> None:
    space = hilbert.DetSpace(norb=4, n_alpha=2, n_beta=1)
    x = space.reference(1)

    bra, sign, active = operator.annihilate(x, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_allclose(operator.number(bra), [2.0])

    ket, sign, active = operator.create(bra, 0, 1)
    assert active.tolist() == [True]
    np.testing.assert_allclose(sign, [-1.0])
    np.testing.assert_array_equal(ket, x)

    np.testing.assert_allclose(operator.Sz(space).diag(x), [0.5])
    np.testing.assert_allclose(operator.S2(space).diag(x), [0.75])
