from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..utils import batch
from ..utils import math
from ..utils import precision
from . import linalg
from .base import Geometry


class SRState(NamedTuple):
    """Optimizer state for parameter-space SR."""

    step: jax.Array
    x0: jax.Array
    stats: dict[str, jax.Array]


def sr(
    *,
    shift: float | Callable[[jax.Array], Any] = 1.0e-3,
    mode: Literal["dense", "matvec"] = "matvec",
    maxiter: int = 64,
) -> optax.GradientTransformationExtraArgs:
    """Parameter-space stochastic reconfiguration.

    SR solves

        (S + shift I) delta = g,
        S = O^dagger O,

    where O is the centered weighted log-derivative matrix.
    """
    if mode not in {"dense", "matvec"}:
        raise ValueError("mode must be 'dense' or 'matvec'")
    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")

    def init_fn(params: Any) -> SRState:
        flat, _ = ravel_pytree(params)
        dtype = precision.real("sr")
        zero = jnp.asarray(0.0, dtype=dtype)
        stats = {"sr_force": zero, "sr_damp": zero}

        return SRState(
            step=jnp.zeros((), dtype=jnp.int32),
            x0=jnp.zeros_like(flat),
            stats=stats,
        )

    def update_fn(
        updates: Any,
        state: SRState,
        params: Any | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Any, SRState]:
        del params, extra_args

        if geometry is None:
            raise ValueError("optimizer.sr requires geometry")

        shift_value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(shift_value, dtype=precision.real("sr"))

        theta = geometry.theta
        w = precision.cast(math.normalize(geometry.w), "model", "real")
        g = jax.tree.map(
            lambda x, leaf: jnp.asarray(x).astype(leaf.dtype),
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
            lambda d, leaf: d.astype(leaf.dtype),
            delta_raw,
            theta,
        )

        return delta, SRState(
            step=state.step + 1,
            x0=x0_next,
            stats={"sr_force": info["force"], "sr_damp": info["damp"]},
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@partial(jax.jit, static_argnums=0)
def _dense_step(
    coord,
    theta: Any,
    x: Any,
    w: jax.Array,
    g: Any,
    shift: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    """One explicit small-parameter SR solve."""
    g_flat, unravel = ravel_pytree(g)
    jac = jax.jacrev(lambda params: coord(params, x))(theta)
    blocks = []
    for J, leaf in zip(jax.tree.leaves(jac), jax.tree.leaves(theta), strict=True):
        J = J.reshape((w.shape[0], -1, leaf.size))
        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        blocks.append(O.reshape(-1, leaf.size))

    O = jnp.concatenate(blocks, axis=1)
    S = precision.cast(O.conj().T @ O, "sr")
    rhs = precision.cast(g_flat, "sr").astype(S.dtype)
    delta_flat = linalg.solve_dense(S, rhs, shift)
    real_dtype = precision.real("sr")
    tiny = precision.tiny("sr")
    force = jnp.real(jnp.vdot(rhs, delta_flat)).astype(real_dtype)
    damp = (shift * jnp.real(jnp.vdot(delta_flat, delta_flat))) / jnp.maximum(
        force,
        tiny,
    )
    return (
        unravel(delta_flat.astype(g_flat.dtype)),
        {"force": force, "damp": damp.astype(real_dtype)},
    )


@partial(jax.jit, static_argnums=(0, 7))
def _matvec_step(
    coord,
    theta: Any,
    x: Any,
    w: jax.Array,
    g: Any,
    shift: jax.Array,
    x0: jax.Array,
    maxiter: int,
) -> tuple[Any, jax.Array, dict[str, jax.Array]]:
    """One matrix-free parameter-space SR solve."""
    g_flat, unravel = ravel_pytree(g)
    nsample = w.shape[0]

    def center(y):
        shape = (nsample,) + (1,) * (y.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * y, axis=0)
        return jnp.sqrt(weight) * (y - mean)

    def cotangent(y):
        shape = (nsample,) + (1,) * (y.ndim - 1)
        sqrt_w = jnp.sqrt(w).reshape(shape)
        weighted = sqrt_w * y
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    def matvec(v: jax.Array) -> jax.Array:
        tangent = unravel(v.astype(g_flat.dtype))
        _, Jv = batch.jvp(coord, theta, tangent, x)
        y = jax.tree.map(center, Jv)
        out = batch.vjp(coord, theta, x, jax.tree.map(cotangent, y))
        out_flat, _ = ravel_pytree(out)
        return precision.cast(out_flat, "sr").astype(v.dtype)

    rhs = precision.cast(g_flat, "sr")
    x0 = precision.cast(x0, "sr").astype(rhs.dtype)

    delta_flat = linalg.solve_matvec(
        matvec,
        rhs,
        shift,
        x0=x0,
        diag=None,
        maxiter=maxiter,
    )

    real_dtype = precision.real("sr")
    tiny = precision.tiny("sr")
    force = jnp.real(jnp.vdot(rhs, delta_flat)).astype(real_dtype)
    damp = (shift * jnp.real(jnp.vdot(delta_flat, delta_flat))) / jnp.maximum(
        force,
        tiny,
    )
    info = {"force": force, "damp": damp.astype(real_dtype)}
    return unravel(delta_flat.astype(g_flat.dtype)), delta_flat.astype(x0.dtype), info
