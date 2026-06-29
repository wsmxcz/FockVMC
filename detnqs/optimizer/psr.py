from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..utils import batch
from ..utils import math
from ..utils import precision
from ..utils import tree
from . import linalg
from .base import Geometry


class PSRState(NamedTuple):
    """Optimizer state for predictive SR.

    PSR is a residual-corrected minSR update:

        p = mu * delta_prev
        r = b - O p
        (O O^dagger + shift I) a = r
        q = O^dagger a
        delta = p + q

    The predictor is never applied as an external momentum step.  It is
    corrected inside the current tangent-space SR equation.
    """

    step: jax.Array
    p: Any
    stats: dict[str, jax.Array]


def psr(
    *,
    shift: float = 1.0e-3,
    mu: float = 0.95,
) -> optax.GradientTransformationExtraArgs:
    """Predictive stochastic reconfiguration."""
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must satisfy 0 <= mu < 1")

    def init_fn(params: Any) -> PSRState:
        dtype = precision.real("sr")
        zero = jnp.asarray(0.0, dtype=dtype)

        return PSRState(
            step=jnp.zeros((), dtype=jnp.int32),
            p=jax.tree.map(jnp.zeros_like, params),
            stats={"sr_force": zero, "sr_damp": zero},
        )

    def update_fn(
        updates: Any,
        state: PSRState,
        params: Any | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Any, PSRState]:
        del updates, params, extra_args

        if geometry is None:
            raise ValueError("optimizer.psr requires geometry")

        shift_t = jnp.asarray(shift, dtype=precision.real("sr"))

        delta_raw, info = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            state.p,
            geometry.x,
            precision.cast(math.normalize(geometry.w), "model", "real"),
            precision.cast(geometry.b, "model", "real"),
            shift_t,
            in_axes=(None, None, None, 0, 0, 0, None),
            out_axes=None,
            static_argnums=0,
        )

        delta = jax.tree.map(
            lambda d, theta: d.astype(theta.dtype),
            delta_raw,
            geometry.theta,
        )

        p_next = jax.tree.map(
            lambda d: jnp.asarray(mu, dtype=d.dtype) * d,
            delta,
        )

        return delta, PSRState(
            step=state.step + 1,
            p=p_next,
            stats={"sr_force": info["force"], "sr_damp": info["damp"]},
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord,
    theta: Any,
    p: Any,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    """Dense sample-space PSR solve."""
    b_flat, unravel_b = ravel_pytree(b)
    nrow = b_flat.size
    nsample = w.shape[0]

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(theta)],
        *[leaf.dtype for leaf in jax.tree.leaves(p)],
    )

    sqrt_w = jnp.sqrt(w)

    K = jnp.zeros((nrow, nrow), dtype=dtype)

    for block, put in tree.blocks(theta, batch.config["param_chunk"]):
        def coord_block(block, sample):
            sample = jax.tree.map(lambda z: z[None, ...], sample)
            out = coord(put(block), sample)
            return jax.tree.map(lambda z: jnp.asarray(z)[0], out)

        J = jax.vmap(
            jax.jacrev(coord_block),
            in_axes=(None, 0),
        )(block, x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * sqrt_w[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        K = K + O @ O.conj().T

    K = precision.cast(K, "sr")
    rhs = precision.cast(b_flat, "sr").astype(K.dtype)

    # Predict current tangent response.
    _, Jp = batch.jvp(coord, theta, p, x)

    def center(tangent):
        shape = (nsample,) + (1,) * (tangent.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * tangent, axis=0)
        return sqrt_w.reshape(shape) * (tangent - mean)

    Op, _ = ravel_pytree(jax.tree.map(center, Jp))
    Op = precision.cast(Op, "sr").astype(K.dtype)

    rhs = rhs - Op

    a = linalg.solve_dense(K, rhs, shift)
    real_dtype = precision.real("sr")
    tiny = precision.tiny("sr")
    force = jnp.real(jnp.vdot(rhs, a)).astype(real_dtype)
    damp = (shift * jnp.real(jnp.vdot(a, a))) / jnp.maximum(force, tiny)
    a_tree = unravel_b(a.astype(dtype))

    def cotangent(a_leaf):
        shape = (nsample,) + (1,) * (a_leaf.ndim - 1)
        weighted = sqrt_w.reshape(shape) * a_leaf
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    q = batch.vjp(
        coord,
        theta,
        x,
        jax.tree.map(cotangent, a_tree),
    )

    delta = jax.tree.map(
        lambda pred, corr: jnp.asarray(pred).astype(corr.dtype) + corr,
        p,
        q,
    )

    return delta, {"force": force, "damp": damp.astype(real_dtype)}
