from pathlib import Path

import jax
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from detnqs import ExactState, Hamiltonian, psr, sr
from detnqs.model import RBM
from detnqs.utils import batch


FCIDUMP = Path(__file__).parents[1] / "scripts" / "FCIDUMP" / "H2.FCIDUMP"


def test_optimizers() -> None:
    batch.configure(
        forward_chunk=32,
        backward_chunk=32,
        param_chunk=None,
        bucket_min=4,
    )
    hamiltonian = Hamiltonian.load(FCIDUMP)
    state = ExactState.init(
        model=RBM(norb=hamiltonian.sector.norb, alpha=1),
        hamiltonian=hamiltonian,
        key=jax.random.key(0),
    )
    _, _, grad, _, geometry = state.expect_and_grad(geometry=True)

    parameter_sr = sr(mode="dense", shift=1.0e-2)
    sample_sr = psr(mu=0.0, shift=1.0e-2)

    update_sr, _ = parameter_sr.update(
        grad,
        parameter_sr.init(state.params),
        state.params,
        geometry=geometry,
    )
    update_psr, _ = sample_sr.update(
        grad,
        sample_sr.init(state.params),
        state.params,
        geometry=geometry,
    )

    flat_sr, _ = ravel_pytree(update_sr)
    flat_psr, _ = ravel_pytree(update_psr)
    assert np.isfinite(flat_sr).all()
    assert np.isfinite(flat_psr).all()
    np.testing.assert_allclose(
        flat_psr,
        flat_sr,
        rtol=2e-3,
        atol=2e-4,
    )

    optimizer = optax.chain(sample_sr, optax.scale_by_learning_rate(1.0e-2))
    scaled, _ = optimizer.update(
        grad,
        optimizer.init(state.params),
        state.params,
        geometry=geometry,
    )
    flat_scaled, _ = ravel_pytree(scaled)
    np.testing.assert_allclose(flat_scaled, -1.0e-2 * flat_psr)

    params = optax.apply_updates(state.params, scaled)
    assert jax.tree.structure(params) == jax.tree.structure(state.params)

    predictive = psr(mu=0.5, shift=1.0e-2)
    predictive_state = predictive.init(state.params)
    _, predictive_state = predictive.update(
        grad,
        predictive_state,
        state.params,
        geometry=geometry,
    )
    second_update, _ = predictive.update(
        grad,
        predictive_state,
        state.params,
        geometry=geometry,
    )
    flat_second, _ = ravel_pytree(second_update)
    assert np.isfinite(flat_second).all()
