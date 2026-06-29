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


class MinSRState(NamedTuple):
    """Optimizer state for sample-space minSR."""

    step: jax.Array
    stats: dict[str, jax.Array]


def minsr(
    *,
    shift: float | Callable[[jax.Array], Any] = 1.0e-3,
) -> optax.GradientTransformationExtraArgs:
    """Dense sample-space minimum-step stochastic reconfiguration.

    minSR solves the damped sample-space SR system

        (K + shift I) a = b,
        K = O O^dagger,
        delta = O^dagger a.

    Here O is the centered weighted log-derivative matrix and b is the
    sample-space SR force. This form avoids building the parameter-space
    QGT S = O^dagger O explicitly.
    """
    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Any) -> MinSRState:
        del params

        dtype = precision.real("sr")
        zero = jnp.asarray(0.0, dtype=dtype)

        return MinSRState(
            step=jnp.zeros((), dtype=jnp.int32),
            stats={"sr_force": zero, "sr_damp": zero},
        )

    def update_fn(
        updates: Any,
        state: MinSRState,
        params: Any | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Any, MinSRState]:
        del updates, params, extra_args

        if geometry is None:
            raise ValueError("optimizer.minsr requires geometry")

        shift_value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(shift_value, dtype=precision.real("sr"))

        delta_raw, info = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            geometry.x,
            precision.cast(math.normalize(geometry.w), "model", "real"),
            precision.cast(geometry.b, "model", "real"),
            shift_t,
            in_axes=(None, None, 0, 0, 0, None),
            out_axes=None,
            static_argnums=0,
        )

        delta = jax.tree.map(
            lambda d, theta: d.astype(theta.dtype),
            delta_raw,
            geometry.theta,
        )

        return delta, MinSRState(
            step=state.step + 1,
            stats={"sr_force": info["force"], "sr_damp": info["damp"]},
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord,
    theta: Any,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    """One dense sample-space minSR solve on one fixed-shape bucket."""
    b_flat, unravel_b = ravel_pytree(b)
    nrow = b_flat.size
    nsample = w.shape[0]

    dtype = jnp.result_type(b_flat, *[leaf.dtype for leaf in jax.tree.leaves(theta)])
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
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        K = K + O @ O.conj().T

    K = precision.cast(K, "sr")
    rhs = precision.cast(b_flat, "sr").astype(K.dtype)

    a = linalg.solve_dense(K, rhs, shift)
    real_dtype = precision.real("sr")
    tiny = precision.tiny("sr")
    force = jnp.real(jnp.vdot(rhs, a)).astype(real_dtype)
    damp = (shift * jnp.real(jnp.vdot(a, a))) / jnp.maximum(force, tiny)
    a_tree = unravel_b(a.astype(dtype))

    def cotangent(a_leaf):
        shape = (nsample,) + (1,) * (a_leaf.ndim - 1)
        sqrt_w = jnp.sqrt(w).reshape(shape)
        weighted = sqrt_w * a_leaf
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    delta = batch.vjp(
        coord,
        theta,
        x,
        jax.tree.map(cotangent, a_tree),
    )

    return delta, {"force": force, "damp": damp.astype(real_dtype)}
