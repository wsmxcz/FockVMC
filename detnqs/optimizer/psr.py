from __future__ import annotations

from collections.abc import Callable
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
    """State for predictive sample-space SR.

    The stored `delta` is the previous actual parameter displacement.
    With signed scale `eta`, PSR solves

        p = mu delta_prev,
        r = eta b - O p,
        (O O^dagger + shift I) a = r,
        delta = p + O^dagger a.

    The returned update is the actual parameter displacement.
    """

    step: jax.Array
    delta: Any
    stats: dict[str, jax.Array]


def psr(
    *,
    shift: float = 1.0e-3,
    mu: float = 0.95,
    scale: float | Callable[[jax.Array], Any] = -1.0,
) -> optax.GradientTransformationExtraArgs:
    """Predictive stochastic reconfiguration."""
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must satisfy 0 <= mu < 1")

    def init_fn(params: Any) -> PSRState:
        zero = jnp.asarray(0.0, dtype=precision.real("sr"))

        return PSRState(
            step=jnp.zeros((), dtype=jnp.int32),
            delta=jax.tree.map(jnp.zeros_like, params),
            stats={
                "step_scale": zero,
                "sr_force": zero,
                "sr_damp": zero,
            },
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

        eta = jnp.asarray(
            scale(state.step) if callable(scale) else scale,
            dtype=precision.real("sr"),
        )
        shift_t = jnp.asarray(shift, dtype=precision.real("sr"))

        pred = jax.tree.map(
            lambda d: jnp.asarray(mu, dtype=d.dtype) * d,
            state.delta,
        )

        delta, info = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            pred,
            geometry.x,
            precision.cast(math.normalize(geometry.w), "model", "real"),
            precision.cast(geometry.b, "model", "real"),
            shift_t,
            eta,
            in_axes=(None, None, None, 0, 0, 0, None, None),
            out_axes=None,
            static_argnums=0,
        )

        delta = jax.tree.map(
            lambda d, p: d.astype(p.dtype),
            delta,
            geometry.theta,
        )

        return delta, PSRState(
            step=state.step + 1,
            delta=delta,
            stats={
                "step_scale": eta,
                "sr_force": info["force"],
                "sr_damp": info["damp"],
            },
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord: Any,
    theta: Any,
    pred: Any,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
    scale: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    """Solve dense sample-space PSR."""
    b_flat, unravel_b = ravel_pytree(b)

    nsample = w.shape[0]
    nrow = b_flat.size
    sqrt_w = jnp.sqrt(w)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(theta)],
        *[leaf.dtype for leaf in jax.tree.leaves(pred)],
    )

    K = jnp.zeros((nrow, nrow), dtype=dtype)

    for block, put in tree.blocks(theta, batch.config["param_chunk"]):

        def coord_block(block_leaf: jax.Array, ket: Any) -> Any:
            ket = jax.tree.map(lambda z: z[None, ...], ket)
            val = coord(put(block_leaf), ket)
            return jax.tree.map(lambda z: jnp.asarray(z)[0], val)

        J = jax.vmap(jax.jacrev(coord_block), in_axes=(None, 0))(block, x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * sqrt_w[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        K = K + O @ O.conj().T

    K = precision.cast(K, "sr")
    rhs = scale * precision.cast(b_flat, "sr").astype(K.dtype)

    # Subtract the predicted tangent response.
    _, Jp = batch.jvp(coord, theta, pred, x)

    def center(val: jax.Array) -> jax.Array:
        shape = (nsample,) + (1,) * (val.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * val, axis=0)
        return sqrt_w.reshape(shape) * (val - mean)

    Op, _ = ravel_pytree(jax.tree.map(center, Jp))
    rhs = rhs - precision.cast(Op, "sr").astype(K.dtype)

    a = linalg.solve_dense(K, rhs, shift)

    force = jnp.real(jnp.vdot(rhs, a)).astype(precision.real("sr"))
    damp = shift * jnp.real(jnp.vdot(a, a))
    damp = damp / jnp.maximum(force, precision.tiny("sr"))

    a_tree = unravel_b(a.astype(dtype))

    def cotangent(val: jax.Array) -> jax.Array:
        shape = (nsample,) + (1,) * (val.ndim - 1)
        weighted = sqrt_w.reshape(shape) * val
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    corr = batch.vjp(
        coord,
        theta,
        x,
        jax.tree.map(cotangent, a_tree),
    )

    delta = jax.tree.map(
        lambda p, q: jnp.asarray(p).astype(q.dtype) + q,
        pred,
        corr,
    )

    return delta, {
        "force": force,
        "damp": damp.astype(precision.real("sr")),
    }