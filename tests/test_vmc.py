from pathlib import Path

import jax
import numpy as np
import optax

from detnqs import ExactState, Hamiltonian, MCState, SelectedState, VMC, psr
from detnqs.model import RBM, slater_reference
from detnqs.operator import number
from detnqs.sampler import MCSampler, sample_slater
from detnqs.utils import batch


FCIDUMP = Path(__file__).parents[1] / "scripts" / "FCIDUMP" / "H2.FCIDUMP"


def test_chains() -> None:
    sector = Hamiltonian.load(FCIDUMP).sector
    orbitals = np.eye(sector.norb)
    ref_mat = slater_reference(
        orbitals[:, :sector.n_alpha],
        orbitals[:, :sector.n_beta],
    )
    chains = sample_slater(sector, ref_mat, n=16, seed=0)

    assert chains.shape == (16, 2, sector.nword)
    np.testing.assert_array_equal(number(chains, spin=0), sector.n_alpha)
    np.testing.assert_array_equal(number(chains, spin=1), sector.n_beta)


def test_states() -> None:
    batch.configure(
        forward_chunk=128,
        backward_chunk=128,
        param_chunk=None,
        bucket_min=128,
    )
    hamiltonian = Hamiltonian.load(FCIDUMP)
    sector = hamiltonian.sector
    basis = sector.enumerate()
    model = RBM(norb=sector.norb, alpha=1)

    exact = ExactState.init(
        model=model,
        hamiltonian=hamiltonian,
        key=jax.random.key(0),
    )
    exact, exact_stats = exact.expect()

    selected = SelectedState.init(
        model=model,
        hamiltonian=hamiltonian,
        basis=basis,
        key=jax.random.key(1),
    ).replace(params=exact.params)
    _, selected_stats = selected.expect()
    np.testing.assert_allclose(selected_stats["energy"], exact_stats["energy"])

    sampler = MCSampler(
        n_samples=128,
        n_chains=32,
        thermal_steps=4,
        proposal="ham",
        blur=0.5,
        alpha=2.0,
    )
    mc = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=sector.random(sampler.n_chains, seed=2),
        key=jax.random.key(2),
        eps1=0.0,
        eps2=0.0,
        eloc_sample=0,
    ).replace(params=exact.params)
    _, energy, grad, stats, geometry = mc.expect_and_grad(geometry=True)

    assert abs(energy - exact_stats["energy"]) < 0.25
    assert np.isfinite(stats["eloc_var"])
    assert stats["n_forward"] <= len(basis)
    assert geometry is not None
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))


def test_checkpoint(tmp_path: Path) -> None:
    batch.configure(
        forward_chunk=64,
        backward_chunk=64,
        param_chunk=None,
        bucket_min=64,
    )
    hamiltonian = Hamiltonian.load(FCIDUMP)
    model = RBM(norb=hamiltonian.sector.norb, alpha=1)
    sampler = MCSampler(n_samples=64, n_chains=16, thermal_steps=2)
    state = MCState.init(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        chains=hamiltonian.sector.random(sampler.n_chains, seed=3),
        key=jax.random.key(3),
        eps1=0.0,
        eps2=0.0,
        eloc_sample=0,
    )
    optimizer = optax.chain(
        psr(mu=0.0, shift=1.0e-2),
        optax.scale_by_learning_rate(1.0e-3),
    )
    vmc = VMC.init(state, optimizer)
    record = vmc.step()
    assert "sr_force" in record

    path = tmp_path / "vmc.npz"
    vmc.save(path)
    saved_params = jax.tree.map(np.asarray, vmc.state.params)
    saved_x = vmc.state.sampler_state.x.copy()
    saved_step = vmc.step_count

    vmc.step()
    vmc.load(path)

    assert vmc.step_count == saved_step
    np.testing.assert_array_equal(vmc.state.sampler_state.x, saved_x)
    for actual, expected in zip(
        jax.tree.leaves(vmc.state.params),
        jax.tree.leaves(saved_params),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
