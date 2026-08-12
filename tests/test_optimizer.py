import jax.numpy as jnp
import numpy as np

from detnqs import psr, sr
from detnqs.optimizer.base import Geometry
from detnqs.utils import batch


def test_sr() -> None:
    batch.configure(
        forward_chunk=None,
        backward_chunk=None,
        param_chunk=None,
        bucket_min=4,
    )
    params = {"w": jnp.array([0.2, -0.4])}
    x = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
    )
    weight = jnp.array([0.1, 0.2, 0.3, 0.4])
    b = jnp.array([0.3, -0.5, 0.2, 0.4])

    def coord(theta, sample):
        return sample @ theta["w"]

    centered = (x - weight @ x) * jnp.sqrt(weight)[:, None]
    grad = {"w": centered.T @ b}
    geometry = Geometry(params=params, coord=coord, x=x, weight=weight, b=b)

    updates = []
    for optimizer in (
        sr(mode="dense", shift=0.1),
        sr(mode="matvec", shift=0.1, max_iter=32),
        psr(mu=0.0, shift=0.1),
    ):
        update, _ = optimizer.update(
            grad,
            optimizer.init(params),
            params,
            geometry=geometry,
        )
        updates.append(update["w"])

    np.testing.assert_allclose(updates[1], updates[0], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(updates[2], updates[0], rtol=1e-8, atol=1e-10)
