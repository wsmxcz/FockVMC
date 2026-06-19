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
    x0: jax.Array
    stats: dict[str, jax.Array]


def sr(
    *,
    shift: Shift = 1.0e-3,
    mode: Mode = "matvec",
    maxiter: int = 64,
) -> optax.GradientTransformationExtraArgs:
    """Parameter-space stochastic reconfiguration.

    SR solves the damped QGT system

        (S + shift I) delta = g,
        S = O^dagger O,

    where O is the centered weighted log-derivative matrix and g is the
    parameter-space force.

    mode="dense":
        Build S explicitly and solve by spectral decomposition.

    mode="matvec":
        Apply S as O^dagger O and solve by warm-started CG.
    """
    if mode not in {"dense", "matvec"}:
        raise ValueError("mode must be 'dense' or 'matvec'")

    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Tree) -> SRState:
        flat, _ = ravel_pytree(params)

        dtype = precision.dtype("sr", "real")
        zero = jnp.asarray(0.0, dtype=dtype)

        stats = {
            "sr_shift": zero,
            "sr_residual": zero,
            "sr_step_norm": zero,
        }

        if mode == "dense":
            stats["sr_cond"] = zero

        return SRState(
            step=jnp.zeros((), dtype=jnp.int32),
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

        shift_value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(shift_value, dtype=precision.dtype("sr", "real"))

        theta = geometry.theta
        w = precision.asarray(normalize(geometry.w), "model", "real")

        # The input updates are interpreted as the parameter-space force g.
        g = jax.tree.map(
            lambda x, theta_leaf: jnp.asarray(x).astype(theta_leaf.dtype),
            updates,
            theta,
        )

        if mode == "dense":
            delta_raw, info = _dense_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                g,
                shift_t,
            )
            x0_next = state.x0
        else:
            delta_raw, x0_next, info = _matvec_step(
                geometry.coord,
                theta,
                geometry.x,
                w,
                g,
                shift_t,
                state.x0,
                int(maxiter),
            )

        delta = jax.tree.map(
            lambda d, theta_leaf: d.astype(theta_leaf.dtype),
            delta_raw,
            theta,
        )

        dtype = precision.dtype("sr", "real")
        step_norm2 = sum(
            (
                jnp.sum(
                    jnp.real(
                        jnp.asarray(x) * jnp.conj(jnp.asarray(x))
                    )
                )
                for x in jax.tree.leaves(delta)
            ),
            jnp.asarray(0.0, dtype=dtype),
        )

        stats = {
            "sr_shift": info["shift"],
            "sr_residual": info["residual"],
            "sr_step_norm": jnp.sqrt(step_norm2),
        }

        if mode == "dense":
            stats["sr_cond"] = info["cond"]

        return delta, SRState(
            step=state.step + 1,
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
    g: Tree,
    shift: jax.Array,
) -> tuple[Tree, dict[str, jax.Array]]:
    """One dense parameter-space SR solve."""
    g_flat, unravel = ravel_pytree(g)

    O = jnp.concatenate(_blocks(coord, theta, x, w), axis=1)

    # S = O^dagger O.
    S = precision.asarray(O.conj().T @ O, "sr")

    rhs = precision.asarray(g_flat, "sr").astype(S.dtype)

    delta_flat, info = linalg.solve_dense(S, rhs, shift)
    delta = unravel(delta_flat.astype(g_flat.dtype))

    return delta, info


@partial(jax.jit, static_argnums=(0, 7))
def _matvec_step(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
    g: Tree,
    shift: jax.Array,
    x0: jax.Array,
    maxiter: int,
) -> tuple[Tree, jax.Array, dict[str, jax.Array]]:
    """One matrix-free parameter-space SR solve."""
    g_flat, unravel = ravel_pytree(g)

    blocks = _blocks(coord, theta, x, w)
    sizes = tuple(block.shape[1] for block in blocks)

    def matvec(v: jax.Array) -> jax.Array:
        # Apply S v = O^dagger O v without forming S.
        y = jnp.zeros((blocks[0].shape[0],), dtype=v.dtype)

        lo = 0
        for block, size in zip(blocks, sizes, strict=True):
            y = y + block @ v[lo : lo + size]
            lo += size

        parts = [block.conj().T @ y for block in blocks]
        return jnp.concatenate(parts, axis=0)

    # Jacobi preconditioner diagonal: diag(S).
    diag = jnp.concatenate(
        [
            jnp.sum(jnp.real(block.conj() * block), axis=0)
            for block in blocks
        ],
        axis=0,
    )

    rhs = precision.asarray(g_flat, "sr")
    x0 = precision.asarray(x0, "sr").astype(rhs.dtype)

    delta_flat, info = linalg.solve_matvec(
        matvec,
        rhs,
        shift,
        x0=x0,
        diag=diag,
        maxiter=maxiter,
    )

    delta = unravel(delta_flat.astype(g_flat.dtype))
    return delta, delta_flat.astype(x0.dtype), info


def _blocks(
    coord,
    theta: Tree,
    x: Any,
    w: jax.Array,
) -> list[jax.Array]:
    """Build parameter blocks of O = sqrt(w) * (J - <J>_w).

    For each parameter leaf,

        J: [N, ..., *theta.shape] -> [N, C, P],

    where N is the sample size, C is the flattened output channel size,
    and P is the flattened parameter-leaf size.

    Each returned block has shape [N * C, P].
    """
    jac = jax.jacrev(lambda params: coord(params, x))(theta)

    blocks = []

    for J, theta_leaf in zip(
        tree_util.tree_leaves(jac),
        tree_util.tree_leaves(theta),
        strict=True,
    ):
        J = J.reshape((w.shape[0], -1, theta_leaf.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]

        blocks.append(O.reshape(-1, theta_leaf.size))

    return blocks