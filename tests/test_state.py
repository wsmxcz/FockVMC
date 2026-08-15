from pathlib import Path

import jax
import numpy as np
import optax
import pytest

from fvmc import ExactState, Hamiltonian, IRState, MCState, SelectedState, VMC
from fvmc.hilbert import DetSector
from fvmc.model import RBM
from fvmc.operator import number
from fvmc.sampler import HamSampler, MCSampler
from fvmc.utils import batch


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


@pytest.mark.parametrize("rank", (1, 2, None))
def test_mc(rank: int | None) -> None:
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
        thermal_steps=1,
        rank=rank,
    )
    chains = ham.sector.random(sampler.n_chains, seed=2)
    state_a = MCState.init(
        model,
        ham,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(2),
        alpha=2.0,
    )
    state_b = MCState.init(
        model,
        ham,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(2),
        alpha=2.0,
    )

    out_a = sampler.draw(
        state_a.params,
        model,
        ham.sector,
        state_a.chain,
        alpha=2.0,
    )
    out_b = sampler.draw(
        state_b.params,
        model,
        ham.sector,
        state_b.chain,
        alpha=2.0,
    )
    next_a, samples_a, rec = out_a
    next_b, samples_b, _ = out_b

    np.testing.assert_array_equal(next_b.x, next_a.x)
    np.testing.assert_array_equal(next_b.logabs, next_a.logabs)
    np.testing.assert_array_equal(
        jax.random.key_data(next_b.key),
        jax.random.key_data(next_a.key),
    )
    np.testing.assert_array_equal(samples_b, samples_a)
    np.testing.assert_array_equal(number(samples_a, spin=0), ham.sector.n_alpha)
    np.testing.assert_array_equal(number(samples_a, spin=1), ham.sector.n_beta)
    assert 0.0 <= rec["acceptance_rate"] <= 1.0
    _, state_rec = state_a.expect()
    assert np.isfinite(state_rec["energy"])


def test_ham() -> None:
    sector = DetSector(norb=2, nelec=2, spin=0)
    ham = Hamiltonian(
        sector,
        np.zeros((2, 2)),
        np.zeros(6),
    )
    model = RBM(norb=sector.norb, alpha=1)
    sampler = HamSampler(
        n_samples=8,
        n_chains=8,
        thermal_steps=0,
        eps1=0.0,
    )
    chains = sector.random(sampler.n_chains, seed=7)
    state = IRState.init(
        model,
        ham,
        sampler=sampler,
        chains=chains,
        key=jax.random.key(7),
        alpha=2.0,
        beta=1.0,
        eps2=0.0,
        n_eloc=0,
    )
    chain, samples, _ = sampler.draw(
        state.params,
        model,
        ham,
        state.chain,
        alpha=2.0,
        beta=1.0,
    )
    np.testing.assert_array_equal(samples, chains)
    np.testing.assert_array_equal(chain.x, chains)
    _, rec = state.expect()
    assert rec["energy"] == 0.0


def test_driver(tmp_path: Path) -> None:
    batch.configure(
        forward_chunk=None,
        backward_chunk=None,
        param_chunk=None,
        bucket_min=16,
    )
    ham = Hamiltonian.load(FCIDUMP)
    model = RBM(norb=ham.sector.norb, alpha=1)
    chains = ham.sector.random(8, seed=4)
    state = IRState.init(
        model,
        ham,
        sampler=HamSampler(
            n_samples=16,
            n_chains=8,
            thermal_steps=0,
            eps1=0.0,
        ),
        chains=chains,
        key=jax.random.key(4),
        alpha=2.0,
        eps2=0.0,
        n_eloc=0,
    )

    vmc = VMC.init(state, optax.adam(1.0e-3), geometry=False)
    vmc.run(1)

    saved = vmc.state.state_dict()
    saved_opt = vmc.opt_state
    step = vmc.step_count
    path = tmp_path / "vmc.npz"
    vmc.save(path)
    vmc.step()
    vmc.load(path)

    assert vmc.step_count == step
    np.testing.assert_array_equal(vmc.state.chain.x, saved["chain"]["x"])
    np.testing.assert_array_equal(
        vmc.state.chain.logabs,
        saved["chain"]["logabs"],
    )
    assert vmc.state.alpha_value == saved["alpha_value"]
    np.testing.assert_array_equal(
        jax.random.key_data(vmc.state.chain.key),
        jax.random.key_data(saved["chain"]["key"]),
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
