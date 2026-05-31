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
    x0: jax.Array
    stats: dict[str, jax.Array]


def sr(
    *,
    shift: Shift = 1.0e-3,
    mode: Mode = "matvec",
    maxiter: int = 64,
) -> optax.GradientTransformationExtraArgs:
    """Parameter-space stochastic reconfiguration.

    SR solves

        (S + shift I) delta = g,

    with S = O^dagger O and O = sqrt(w) * (J - <J>_w).
    """
    if mode not in {"dense", "matvec"}:
        raise ValueError("sr mode must be 'dense' or 'matvec'")

    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Tree) -> SRState:
        flat, _ = ravel_pytree(params)

        step = jnp.zeros((), dtype=jnp.int32)
        value = shift(step) if callable(shift) else shift
        shift_t = jnp.asarray(value, dtype=precision.dtype("sr", "real"))

        stats = {
            "sr_shift": shift_t,
            "sr_residual": jnp.asarray(0.0, dtype=shift_t.dtype),
            "sr_step_norm": jnp.asarray(0.0, dtype=shift_t.dtype),
        }

        if mode == "dense":
            stats.update(
                {
                    "sr_eig_min": jnp.asarray(0.0, dtype=shift_t.dtype),
                    "sr_eig_max": jnp.asarray(0.0, dtype=shift_t.dtype),
                    "sr_trace": jnp.asarray(0.0, dtype=shift_t.dtype),
                    "sr_rank_eff": jnp.asarray(0.0, dtype=shift_t.dtype),
                    "sr_dof": jnp.asarray(0.0, dtype=shift_t.dtype),
                    "sr_cond": jnp.asarray(0.0, dtype=shift_t.dtype),
                }
            )

        return SRState(
            step=step,
            shift=shift_t,
            x0=jnp.zeros_like(flat),
            stats=stats,
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
            delta, info = _dense_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                updates,
                shift_t,
            )
            x0_next = state.x0
        else:
            delta, x0_next, info = _matvec_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                updates,
                shift_t,
                state.x0,
                int(maxiter),
            )

        delta = jax.tree.map(lambda d, p: d.astype(p.dtype), delta, theta)

        real_dtype = precision.dtype("sr", "real")
        step_norm2 = sum(
            (
                jnp.sum(jnp.real(jnp.asarray(leaf) * jnp.conj(jnp.asarray(leaf))))
                for leaf in jax.tree.leaves(delta)
            ),
            jnp.asarray(0.0, dtype=real_dtype),
        )
        step_norm = jnp.sqrt(step_norm2)

        stats = {
            "sr_shift": info["shift"],
            "sr_residual": info["residual"],
            "sr_step_norm": step_norm,
        }

        if mode == "dense":
            stats.update(
                {
                    "sr_eig_min": info["eig_min"],
                    "sr_eig_max": info["eig_max"],
                    "sr_trace": info["trace"],
                    "sr_rank_eff": info["rank_eff"],
                    "sr_dof": info["dof"],
                    "sr_cond": info["cond"],
                }
            )

        return delta, SRState(
            step=state.step + 1,
            shift=info["shift"],
            x0=x0_next,
            stats=stats,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@partial(jax.jit, static_argnums=0)
def _dense_step(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
    updates: Tree,
    shift: jax.Array,
) -> tuple[Tree, dict[str, jax.Array]]:
    updates_flat, unravel = ravel_pytree(updates)
    blocks = _blocks(coord, theta, x, w)

    O = jnp.concatenate(blocks, axis=1)
    S = precision.asarray(O.conj().T @ O, "sr")
    rhs = precision.asarray(updates_flat, "sr").astype(S.dtype)

    delta_flat, info = linalg.solve_dense(S, rhs, shift)
    return unravel(delta_flat.astype(updates_flat.dtype)), info


@partial(jax.jit, static_argnums=(0, 7))
def _matvec_step(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
    updates: Tree,
    shift: jax.Array,
    x0: jax.Array,
    maxiter: int,
) -> tuple[Tree, jax.Array, dict[str, jax.Array]]:
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
        return jnp.concatenate(parts, axis=0)

    diag = jnp.concatenate(
        [jnp.sum(jnp.real(block.conj() * block), axis=0) for block in blocks],
        axis=0,
    )

    rhs = precision.asarray(updates_flat, "sr")

    delta_flat, info = linalg.solve_matvec(
        matvec,
        rhs,
        shift,
        x0=precision.asarray(x0, "sr").astype(rhs.dtype),
        diag=diag,
        maxiter=maxiter,
    )

    return (
        unravel(delta_flat.astype(updates_flat.dtype)),
        delta_flat.astype(x0.dtype),
        info,
    )


def _blocks(coord, theta: Tree, x: Any, w: jax.Array) -> list[jax.Array]:
    """Build parameter blocks of O = sqrt(w) * (J - <J>_w)."""
    jac = jax.jacrev(lambda p: coord(p, x))(theta)

    blocks = []
    for J, p in zip(tree_util.tree_leaves(jac), tree_util.tree_leaves(theta), strict=True):
        wb = w.reshape((w.shape[0],) + (1,) * (J.ndim - 1))
        mean = jnp.sum(wb * J, axis=0, keepdims=True)
        blocks.append((jnp.sqrt(wb) * (J - mean)).reshape(-1, p.size))

    return blocks