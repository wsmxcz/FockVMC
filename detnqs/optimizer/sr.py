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
    """State for parameter-space stochastic reconfiguration.

    SR solves

        (S + shift I) delta = eta grad,

    where `eta = scale(step)` is the signed update scale and
    `S = O^dagger O` is the centered tangent-space metric.
    """

    step: jax.Array
    x0: jax.Array
    stats: dict[str, jax.Array]


def sr(
    *,
    shift: float = 1.0e-3,
    scale: float | Callable[[jax.Array], Any] = -1.0,
    mode: Literal["dense", "matvec"] = "matvec",
    maxiter: int = 64,
) -> optax.GradientTransformationExtraArgs:
    """Parameter-space stochastic reconfiguration."""
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if mode not in {"dense", "matvec"}:
        raise ValueError("mode must be 'dense' or 'matvec'")

    def init_fn(params: Any) -> SRState:
        flat, _ = ravel_pytree(params)
        zero = jnp.asarray(0.0, dtype=precision.real("sr"))

        return SRState(
            step=jnp.zeros((), dtype=jnp.int32),
            x0=jnp.zeros_like(flat),
            stats={
                "step_scale": zero,
                "sr_force": zero,
                "sr_damp": zero,
            },
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

        eta = jnp.asarray(
            scale(state.step) if callable(scale) else scale,
            dtype=precision.real("sr"),
        )
        shift_t = jnp.asarray(shift, dtype=precision.real("sr"))

        theta = geometry.theta
        x = geometry.x
        w = precision.cast(math.normalize(geometry.w), "model", "real")
        grad = jax.tree.map(
            lambda g, p: jnp.asarray(g).astype(p.dtype),
            updates,
            theta,
        )

        if mode == "dense":
            delta, info = _dense_step(
                geometry.coord,
                theta,
                x,
                w,
                grad,
                shift_t,
                eta,
            )
            x0 = state.x0
        else:
            delta, x0, info = _matvec_step(
                geometry.coord,
                theta,
                x,
                w,
                grad,
                shift_t,
                eta,
                state.x0,
                int(maxiter),
            )

        delta = jax.tree.map(
            lambda d, p: d.astype(p.dtype),
            delta,
            theta,
        )

        return delta, SRState(
            step=state.step + 1,
            x0=x0,
            stats={
                "step_scale": eta,
                "sr_force": info["force"],
                "sr_damp": info["damp"],
            },
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@partial(jax.jit, static_argnums=0)
def _dense_step(
    coord: Any,
    theta: Any,
    x: Any,
    w: jax.Array,
    grad: Any,
    shift: jax.Array,
    scale: jax.Array,
) -> tuple[Any, dict[str, jax.Array]]:
    """Solve dense parameter-space SR."""
    grad_flat, unravel = ravel_pytree(grad)

    jac = jax.jacrev(lambda params: coord(params, x))(theta)
    blocks = []

    for J, leaf in zip(jax.tree.leaves(jac), jax.tree.leaves(theta), strict=True):
        J = J.reshape((w.shape[0], -1, leaf.size))
        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        blocks.append(O.reshape(-1, leaf.size))

    O = jnp.concatenate(blocks, axis=1)
    S = precision.cast(O.conj().T @ O, "sr")

    rhs = scale * precision.cast(grad_flat, "sr").astype(S.dtype)
    delta = linalg.solve_dense(S, rhs, shift)

    force = jnp.real(jnp.vdot(rhs, delta)).astype(precision.real("sr"))
    damp = shift * jnp.real(jnp.vdot(delta, delta))
    damp = damp / jnp.maximum(force, precision.tiny("sr"))

    return unravel(delta.astype(grad_flat.dtype)), {
        "force": force,
        "damp": damp.astype(precision.real("sr")),
    }


@partial(jax.jit, static_argnums=(0, 8))
def _matvec_step(
    coord: Any,
    theta: Any,
    x: Any,
    w: jax.Array,
    grad: Any,
    shift: jax.Array,
    scale: jax.Array,
    x0: jax.Array,
    maxiter: int,
) -> tuple[Any, jax.Array, dict[str, jax.Array]]:
    """Solve matrix-free parameter-space SR."""
    grad_flat, unravel = ravel_pytree(grad)
    nsample = w.shape[0]
    sqrt_w = jnp.sqrt(w)

    def center(val: jax.Array) -> jax.Array:
        shape = (nsample,) + (1,) * (val.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * val, axis=0)
        return sqrt_w.reshape(shape) * (val - mean)

    def cotangent(val: jax.Array) -> jax.Array:
        shape = (nsample,) + (1,) * (val.ndim - 1)
        weighted = sqrt_w.reshape(shape) * val
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    def matvec(vec: jax.Array) -> jax.Array:
        tangent = unravel(vec.astype(grad_flat.dtype))
        _, Jv = batch.jvp(coord, theta, tangent, x)
        out = batch.vjp(
            coord,
            theta,
            x,
            jax.tree.map(cotangent, jax.tree.map(center, Jv)),
        )
        flat, _ = ravel_pytree(out)
        return precision.cast(flat, "sr").astype(vec.dtype)

    rhs = scale * precision.cast(grad_flat, "sr")
    x0 = precision.cast(x0, "sr").astype(rhs.dtype)

    delta = linalg.solve_matvec(
        matvec,
        rhs,
        shift,
        x0=x0,
        maxiter=maxiter,
    )

    force = jnp.real(jnp.vdot(rhs, delta)).astype(precision.real("sr"))
    damp = shift * jnp.real(jnp.vdot(delta, delta))
    damp = damp / jnp.maximum(force, precision.tiny("sr"))

    return (
        unravel(delta.astype(grad_flat.dtype)),
        delta.astype(x0.dtype),
        {
            "force": force,
            "damp": damp.astype(precision.real("sr")),
        },
    )