from pathlib import Path

import jax
import numpy as np
import optax
import pytest

from detnqs import ExactState, Hamiltonian, MCState, SelectedState, VMC
from detnqs.model import RBM
from detnqs.operator import number
from detnqs.sampler import MCSampler
from detnqs.utils import batch


FCIDUMP = Path(__file__).parents[1] / "scripts" / "FCIDUMP" / "H2.FCIDUMP"


def test_selected() -> None:
    batch.configure(
        forward_chunk=None,
        backward_chunk=None,
        param_chunk=None,
        bucket_min=16,
    )
    ham = Hamiltonian.load(FCIDUMP)
    model = RBM(norb=ham.sector.norb, alpha=1)
    exact = ExactState.init(model, ham, key=jax.random.key(0))
    selected = SelectedState.init(
        model,
        ham,
        basis=exact.x,
        key=jax.random.key(1),
    ).replace(params=exact.params)

    _, rec_a, grad_a, _ = exact.expect(grad=True)
    _, rec_b, grad_b, _ = selected.expect(grad=True)

    np.testing.assert_allclose(rec_b["energy"], rec_a["energy"])
    np.testing.assert_allclose(rec_b["eloc_var"], rec_a["eloc_var"])
    for a, b in zip(
        jax.tree.leaves(grad_a),
        jax.tree.leaves(grad_b),
        strict=True,
    ):
        np.testing.assert_allclose(b, a)


@pytest.mark.parametrize("proposal", ("ham", "single"))
def test_sampler(proposal: str) -> None:
    batch.configure(
        forward_chunk=None,
        backward_chunk=None,
        param_chunk=None,
        bucket_min=16,
    )
    ham = Hamiltonian.load(FCIDUMP)
    model = RBM(norb=ham.sector.norb, alpha=1)
    sampler = MCSampler(
        n_samples=16,
        n_chains=8,
        burn_in=1,
        proposal=proposal,
        alpha=2.0,
        beta=0.0 if proposal == "single" else 0.5,
    )
    chains = ham.sector.random(sampler.n_chains, seed=2)
    state_a = MCState.init(
        model,
        ham,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(2),
        eps1=0.0,
        eps2=0.0,
        n_eloc=0,
    )
    state_b = MCState.init(
        model,
        ham,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(2),
        eps1=0.0,
        eps2=0.0,
        n_eloc=0,
    )

    out_a = sampler.draw(
        state_a.params,
        ham,
        model,
        state_a.sampler_state,
        eps1=0.0,
    )
    out_b = sampler.draw(
        state_b.params,
        ham,
        model,
        state_b.sampler_state,
        eps1=0.0,
    )
    next_a, x_a, mass_a, rec = out_a
    next_b, x_b, mass_b, _ = out_b

    np.testing.assert_array_equal(next_b.x, next_a.x)
    np.testing.assert_array_equal(next_b.logabs, next_a.logabs)
    np.testing.assert_array_equal(
        jax.random.key_data(next_b.key),
        jax.random.key_data(next_a.key),
    )
    np.testing.assert_array_equal(x_b, x_a)
    np.testing.assert_array_equal(mass_b, mass_a)
    np.testing.assert_array_equal(number(x_a, spin=0), ham.sector.n_alpha)
    np.testing.assert_array_equal(number(x_a, spin=1), ham.sector.n_beta)
    assert np.isfinite(mass_a).all() and (mass_a > 0.0).all()
    assert 0.0 <= rec["acceptance_rate"] <= 1.0


def test_driver(tmp_path: Path) -> None:
    batch.configure(
        forward_chunk=None,
        backward_chunk=None,
        param_chunk=None,
        bucket_min=16,
    )
    ham = Hamiltonian.load(FCIDUMP)
    model = RBM(norb=ham.sector.norb, alpha=1)
    sampler = MCSampler(
        n_samples=16,
        n_chains=8,
        burn_in=0,
        proposal="single",
        alpha=2.0,
        beta=0.0,
    )
    state = MCState.init(
        model,
        ham,
        sampler=sampler,
        chains=ham.sector.random(sampler.n_chains, seed=3),
        key=jax.random.key(3),
        eps1=0.0,
        eps2=0.0,
        n_eloc=0,
    )
    vmc = VMC.init(state, optax.adam(1.0e-3), geometry=False)
    vmc.step()

    saved = vmc.state.state_dict()
    saved_opt = vmc.opt_state
    step = vmc.step_count
    path = tmp_path / "vmc.npz"
    vmc.save(path)
    vmc.step()
    vmc.load(path)

    assert vmc.step_count == step
    np.testing.assert_array_equal(vmc.state.chains, saved["chains"])
    np.testing.assert_array_equal(
        vmc.state.sampler_state.x,
        saved["sampler_state"]["x"],
    )
    np.testing.assert_array_equal(
        vmc.state.sampler_state.logabs,
        saved["sampler_state"]["logabs"],
    )
    assert vmc.state.sampler_state.alpha == saved["sampler_state"]["alpha"]
    np.testing.assert_array_equal(
        jax.random.key_data(vmc.state.sampler_state.key),
        jax.random.key_data(saved["sampler_state"]["key"]),
    )
    for actual, expected in zip(
        jax.tree.leaves(vmc.state.params),
        jax.tree.leaves(saved["params"]),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        jax.tree.leaves(vmc.opt_state),
        jax.tree.leaves(saved_opt),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
