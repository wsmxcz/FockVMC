from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from jax.flatten_util import ravel_pytree

from ..utils import batch
from ..utils import normalize
from ..utils import precision
from . import linalg
from .base import Geometry

Tree = Any
Shift = float | Callable[[jax.Array], Any]


class MinSRState(NamedTuple):
    """Optimizer state for sample-space minSR."""

    step: jax.Array
    shift: jax.Array


def minsr(
    *,
    shift: Shift = 1.0e-3,
    fallback: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Dense sample-space minimum-step SR.

    Given

        O = sqrt(w) * (J - <J>_w),
        K = O O^dagger,

    minSR solves

        (K + shift I) a = b,

    and returns

        delta = O^dagger a.

    The incoming Optax update is ignored. The right hand side is geometry.b.
    """
    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Tree) -> MinSRState:
        del params

        step = jnp.zeros((), dtype=jnp.int32)
        value = shift(step) if callable(shift) else shift

        return MinSRState(
            step=step,
            shift=jnp.asarray(value, dtype=precision.dtype("sr", "real")),
        )

    def update_fn(
        updates: Tree,
        state: MinSRState,
        params: Tree | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Tree, MinSRState]:
        del updates, params, extra_args

        if geometry is None:
            raise ValueError("optimizer.minsr requires geometry")

        value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(value, dtype=precision.dtype("sr", "real"))

        delta = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            geometry.x,
            precision.asarray(normalize(geometry.w), "model", "real"),
            precision.asarray(geometry.b, "model", "real"),
            shift_t,
            bool(fallback),
            in_axes=(None, None, 0, 0, 0, None, None),
            out_axes=None,
            static_argnums=(0, 6),
        )

        delta = jax.tree.map(lambda d, p: d.astype(p.dtype), delta, geometry.theta)

        return delta, MinSRState(
            step=state.step + 1,
            shift=shift_t,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord,
    theta: Tree,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
    fallback: bool,
) -> Tree:
    """One minSR solve on one fixed-shape bucket."""
    b_flat, _ = ravel_pytree(b)
    nrow = b_flat.size

    jac = jax.jacrev(lambda p: coord(p, x))(theta)

    theta_leaves, treedef = tree_util.tree_flatten(theta)
    jac_leaves = tree_util.tree_leaves(jac)

    dtype = jnp.result_type(b_flat, *[p.dtype for p in theta_leaves])
    K = jnp.zeros((nrow, nrow), dtype=dtype)
    blocks = []

    for J, p in zip(jac_leaves, theta_leaves, strict=True):
        wb = w.reshape((w.shape[0],) + (1,) * (J.ndim - 1))
        mean = jnp.sum(wb * J, axis=0, keepdims=True)
        O = (jnp.sqrt(wb) * (J - mean)).reshape(-1, p.size).astype(dtype)

        blocks.append(O)
        K = K + O @ O.conj().T

    K = precision.asarray(K, "sr")
    rhs = precision.asarray(b_flat, "sr").astype(K.dtype)

    a = linalg.solve_dense(K, rhs, shift, fallback=fallback).astype(dtype)

    leaves = [
        (O.conj().T @ a).reshape(p.shape)
        for O, p in zip(blocks, theta_leaves, strict=True)
    ]

    return tree_util.tree_unflatten(treedef, leaves)