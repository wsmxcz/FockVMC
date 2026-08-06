from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from detnqs.hilbert import DetSector
from detnqs.model import (
    Backflow,
    GBackflow,
    PBackflow,
    RBM,
    SBackflow,
    slater_reference,
)


def test_reference() -> None:
    alpha = np.eye(3)[:, :2]
    beta = np.eye(3)[:, :1]
    ref_mat = slater_reference(alpha, beta)

    assert ref_mat.shape == (3, 6)
    np.testing.assert_array_equal(ref_mat[:2, :3], alpha.T)
    np.testing.assert_array_equal(ref_mat[2:, 3:], beta.T)
    np.testing.assert_array_equal(ref_mat[:2, 3:], 0.0)
    np.testing.assert_array_equal(ref_mat[2:, :3], 0.0)


def test_models() -> None:
    sector = DetSector(norb=2, nelec=2, spin=0)
    x = sector.enumerate()
    models = (
        RBM(norb=2, alpha=1),
        Backflow(norb=2, n_alpha=1, n_beta=1, hidden=(4,)),
        GBackflow(norb=2, n_alpha=1, n_beta=1, hidden=(4,)),
        SBackflow(norb=2, n_alpha=1, n_beta=1, hidden=(4,)),
        PBackflow(norb=2, n_alpha=1, n_beta=1, hidden=(4,)),
    )

    for i, model in enumerate(models):
        params = model.init(jax.random.key(i), x[:1])["params"]
        logpsi = model.apply(params, x)
        logabs = model.logabs(params, x)
        grad = jax.grad(lambda p: jnp.mean(model.logabs(p, x)))(params)

        leaves = jax.tree.leaves(grad)
        assert np.asarray(logabs).shape == (len(x),)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)

        if isinstance(logpsi, tuple):
            assert np.asarray(logpsi[0]).shape == (len(x),)
            assert np.asarray(logpsi[1]).shape == (len(x),)
        else:
            assert np.asarray(logpsi).shape == (len(x),)
