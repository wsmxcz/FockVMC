from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..utils import batch, math, precision, tree
from . import linalg
from .base import Geometry


class PSRState(NamedTuple):
    """Previous unscaled PSR direction."""

    direction: Any


def psr(
    *,
    shift: float = 1.0e-3,
    mu: float = 0.95,
) -> optax.GradientTransformationExtraArgs:
    """Precondition updates with predictive sample-space SR."""
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must satisfy 0 <= mu < 1")

    def init_fn(params: Any) -> PSRState:
        return PSRState(direction=jax.tree.map(jnp.zeros_like, params))

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

        pred = jax.tree.map(
            lambda d: jnp.asarray(mu, dtype=d.dtype) * d,
            state.direction,
        )

        delta = batch.bucket(
            _step,
            geometry.coord,
            geometry.params,
            pred,
            geometry.x,
            precision.cast(math.normalize(geometry.weight), "model", "real"),
            precision.cast(geometry.b, "model", "real"),
            shift_t,
            in_axes=(None, None, None, 0, 0, 0, None),
            out_axes=None,
            static_argnums=0,
        )

        delta = jax.tree.map(
            lambda d, p: d.astype(p.dtype),
            delta,
            geometry.params,
        )

        return delta, PSRState(direction=delta)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord: Any,
    theta: Any,
    pred: Any,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
) -> Any:
    """Solve dense sample-space PSR."""
    b_flat, _ = ravel_pytree(b)

    nsample = w.shape[0]
    nrow = b_flat.size
    sqrt_w = jnp.sqrt(w)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(theta)],
        *[leaf.dtype for leaf in jax.tree.leaves(pred)],
    )

    def tangent_block(block: jax.Array, put: Any) -> jax.Array:
        def coord_block(block_leaf: jax.Array, ket: Any) -> Any:
            ket = jax.tree.map(lambda z: z[None, ...], ket)
            val = coord(put(block_leaf), ket)
            return jax.tree.map(lambda z: jnp.asarray(z)[0], val)

        J = jax.vmap(jax.jacrev(coord_block), in_axes=(None, 0))(block, x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * sqrt_w[:, None, None]
        return O.reshape(nrow, block.size).astype(dtype)

    K = jnp.zeros((nrow, nrow), dtype=dtype)
    for block, put in tree.blocks(theta, batch.config["param_chunk"]):
        O = tangent_block(block, put)
        K = K + O @ O.conj().T

    K = precision.cast(K, "sr")
    rhs = precision.cast(b_flat, "sr").astype(K.dtype)

    _, Jp = batch.jvp(coord, theta, pred, x)

    def center(val: jax.Array) -> jax.Array:
        shape = (nsample,) + (1,) * (val.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * val, axis=0)
        return sqrt_w.reshape(shape) * (val - mean)

    Op, _ = ravel_pytree(jax.tree.map(center, Jp))
    rhs = rhs - precision.cast(Op, "sr").astype(K.dtype)

    a = linalg.solve_dense(K, rhs, shift)

    # A real-output VJP would discard the complex sample-space correction.
    corr = jax.tree.map(jnp.zeros_like, theta)
    for block, put in tree.blocks(theta, batch.config["param_chunk"]):
        O = tangent_block(block, put)
        block_corr = (O.conj().T @ a).astype(block.dtype)
        contribution = jax.tree.map(
            jnp.subtract,
            put(block + block_corr),
            theta,
        )
        corr = jax.tree.map(jnp.add, corr, contribution)

    delta = jax.tree.map(
        lambda p, q: jnp.asarray(p).astype(q.dtype) + q,
        pred,
        corr,
    )
    return delta
