import jax
import jax.numpy as jnp
import jax.scipy as jsp
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

    hole_sector = DetSector(norb=4, nelec=5, spin=1)
    hole_x = hole_sector.enumerate()
    hole_ref = slater_reference(np.eye(4)[:, :3], np.eye(4)[:, :2])
    hole_models = (
        GBackflow(norb=4, n_alpha=3, n_beta=2, hidden=(4,), ref_mat=hole_ref),
        SBackflow(norb=4, n_alpha=3, n_beta=2, hidden=(4,), ref_mat=hole_ref),
        PBackflow(norb=4, n_alpha=3, n_beta=2, hidden=(4,), ref_mat=hole_ref),
    )

    seed = 0
    for inputs, group in ((x, models), (hole_x, hole_models)):
        for model in group:
            params = model.init(jax.random.key(seed), inputs[:1])["params"]
            logpsi = model.apply(params, inputs)
            logabs = model.logabs(params, inputs)
            grad = jax.grad(lambda p: jnp.mean(model.logabs(p, inputs)))(params)
            seed += 1

            assert logabs.shape == (len(inputs),)
            assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(grad))

            if isinstance(logpsi, tuple):
                assert logpsi[0].shape == (len(inputs),)
                assert logpsi[1].shape == (len(inputs),)
            else:
                assert logpsi.shape == (len(inputs),)


def test_projection() -> None:
    keys = jax.random.split(jax.random.key(0), 4)
    a = jax.random.normal(keys[0], (3, 3, 3))
    b = jax.random.normal(keys[1], (3, 3, 2))
    c = jax.random.normal(keys[2], (3, 2, 3))
    d = jax.random.normal(keys[3], (3, 2, 2))
    z = jnp.array((0.2, 1.0, 3.0))
    log_weight = jnp.log(jnp.array((0.2, 0.3, 0.5)))

    lu, piv = jsp.linalg.lu_factor(a)
    k = c @ jsp.linalg.lu_solve((lu, piv), b)
    diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
    a_logabs = jnp.sum(jnp.log(jnp.abs(diag)), axis=-1)
    n_swap = jnp.sum(
        piv != jnp.arange(a.shape[-1], dtype=piv.dtype), axis=-1
    )
    a_sign = (
        jnp.prod(jnp.sign(diag), axis=-1)
        * (1 - 2 * (n_swap & 1)).astype(a.dtype)
    )

    quad = d[:, None] + z[None, :, None, None] * k[:, None]
    det_sign, det_logabs = jnp.linalg.slogdet(quad)
    logabs, sign = jax.nn.logsumexp(
        det_logabs + log_weight[None, :],
        b=det_sign,
        axis=1,
        return_sign=True,
    )
    sign = sign * a_sign
    logabs = logabs + a_logabs

    root = jnp.sqrt(z)[:, None, None, None]
    full_a = jnp.broadcast_to(a, (len(z),) + a.shape)
    full_d = jnp.broadcast_to(d, (len(z),) + d.shape)
    top = jnp.concatenate((full_a, root * b[None]), axis=-1)
    bottom = jnp.concatenate((-root * c[None], full_d), axis=-1)
    full_sign, full_logabs = jnp.linalg.slogdet(
        jnp.concatenate((top, bottom), axis=-2)
    )
    expected_logabs, expected_sign = jax.nn.logsumexp(
        full_logabs + log_weight[:, None],
        b=full_sign,
        axis=0,
        return_sign=True,
    )

    np.testing.assert_array_equal(sign, expected_sign)
    np.testing.assert_allclose(logabs, expected_logabs, rtol=1e-11, atol=1e-11)
