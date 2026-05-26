from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import tree_util
from jax.flatten_util import ravel_pytree

from ..utils import normalize
from ..utils import precision
from . import linalg
from .base import Geometry

Tree = Any
Shift = float | Callable[[jax.Array], Any]
Mode = Literal["dense", "matvec"]


class SRState(NamedTuple):
    """Optimizer state for parameter-space SR."""

    step: jax.Array
    shift: jax.Array


def sr(
    *,
    shift: Shift = 1.0e-3,
    mode: Mode = "matvec",
    maxiter: int = 64,
    fallback: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Parameter-space stochastic reconfiguration.

    Given

        O = sqrt(w) * (J - <J>_w),
        S = O^dagger O,

    SR maps the incoming Optax update g to delta by solving

        (S + shift I) delta = g.

    This transform does not choose the optimization sign or learning rate.
    Use optax.scale, optax.scale_by_schedule, clipping, or momentum outside it.
    """
    if mode not in {"dense", "matvec"}:
        raise ValueError("sr mode must be 'dense' or 'matvec'")

    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Tree) -> SRState:
        del params

        step = jnp.zeros((), dtype=jnp.int32)
        value = shift(step) if callable(shift) else shift

        return SRState(
            step=step,
            shift=jnp.asarray(value, dtype=precision.dtype("sr", "real")),
        )

    def update_fn(
        updates: Tree,
        state: SRState,
        params: Tree | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Tree, SRState]:
        del params, extra_args

        if geometry is None:
            raise ValueError("optimizer.sr requires geometry")

        value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(value, dtype=precision.dtype("sr", "real"))

        theta = geometry.theta
        w = precision.asarray(normalize(geometry.w), "model", "real")
        updates = jax.tree.map(lambda g, p: jnp.asarray(g).astype(p.dtype), updates, theta)

        if mode == "dense":
            delta = _dense_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                updates,
                shift_t,
                bool(fallback),
            )
        else:
            delta = _matvec_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                updates,
                shift_t,
                int(maxiter),
            )

        delta = jax.tree.map(lambda d, p: d.astype(p.dtype), delta, theta)

        return delta, SRState(
            step=state.step + 1,
            shift=shift_t,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@partial(jax.jit, static_argnums=(0, 6))
def _dense_step(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
    updates: Tree,
    shift: jax.Array,
    fallback: bool,
) -> Tree:
    updates_flat, unravel = ravel_pytree(updates)
    blocks = _blocks(coord, theta, x, w)

    O = jnp.concatenate(blocks, axis=1)
    S = precision.asarray(O.conj().T @ O, "sr")
    rhs = precision.asarray(updates_flat, "sr").astype(S.dtype)

    delta_flat = linalg.solve_dense(S, rhs, shift, fallback=fallback)
    return unravel(delta_flat.astype(updates_flat.dtype))


@partial(jax.jit, static_argnums=(0, 6))
def _matvec_step(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
    updates: Tree,
    shift: jax.Array,
    maxiter: int,
) -> Tree:
    updates_flat, unravel = ravel_pytree(updates)
    blocks = _blocks(coord, theta, x, w)
    sizes = tuple(block.shape[1] for block in blocks)

    def matvec(v_flat: jax.Array) -> jax.Array:
        y = jnp.zeros((blocks[0].shape[0],), dtype=v_flat.dtype)

        lo = 0
        for block, size in zip(blocks, sizes, strict=True):
            y = y + block @ v_flat[lo : lo + size]
            lo += size

        parts = [block.conj().T @ y for block in blocks]
        return jnp.concatenate(parts, axis=0) + shift.astype(v_flat.dtype) * v_flat

    delta_flat = linalg.solve_matvec(matvec, updates_flat, maxiter=maxiter)
    return unravel(delta_flat.astype(updates_flat.dtype))


def _blocks(coord, theta: Tree, x: Any, w: jax.Array) -> list[jax.Array]:
    """Build parameter blocks of O = sqrt(w) * (J - <J>_w)."""
    jac = jax.jacrev(lambda p: coord(p, x))(theta)

    blocks = []
    for J, p in zip(tree_util.tree_leaves(jac), tree_util.tree_leaves(theta), strict=True):
        wb = w.reshape((w.shape[0],) + (1,) * (J.ndim - 1))
        mean = jnp.sum(wb * J, axis=0, keepdims=True)
        blocks.append((jnp.sqrt(wb) * (J - mean)).reshape(-1, p.size))

    return blocks