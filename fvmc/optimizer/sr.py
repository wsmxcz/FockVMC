from __future__ import annotations

from functools import partial
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree

from ..utils import batch, math, precision
from . import linalg
from .base import Geometry


class SRState(NamedTuple):
    """Warm start for parameter-space SR."""

    guess: jax.Array


def sr(
    *,
    shift: float = 1.0e-3,
    mode: Literal["dense", "matvec"] = "matvec",
    max_iter: int = 64,
) -> optax.GradientTransformationExtraArgs:
    """Precondition updates with the parameter-space SR geometry."""
    if shift < 0.0:
        raise ValueError("shift must be non-negative")
    if mode not in {"dense", "matvec"}:
        raise ValueError("mode must be 'dense' or 'matvec'")

    def init_fn(params: Any) -> SRState:
        flat, _ = ravel_pytree(params)
        return SRState(guess=jnp.zeros_like(flat))

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

        shift_t = jnp.asarray(shift, dtype=precision.real("sr"))

        theta = geometry.params
        x = geometry.x
        w = precision.cast(math.normalize(geometry.weight), "model", "real")
        grad = jax.tree.map(
            lambda g, p: jnp.asarray(g).astype(p.dtype),
            updates,
            theta,
        )

        if mode == "dense":
            delta = _dense_step(
                geometry.coord,
                theta,
                x,
                w,
                grad,
                shift_t,
            )
            guess = state.guess
        else:
            delta, guess = _matvec_step(
                geometry.coord,
                theta,
                x,
                w,
                grad,
                shift_t,
                state.guess,
                max_iter,
            )

        delta = jax.tree.map(
            lambda d, p: d.astype(p.dtype),
            delta,
            theta,
        )

        return delta, SRState(guess=guess)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@partial(jax.jit, static_argnums=0)
def _dense_step(
    coord: Any,
    theta: Any,
    x: Any,
    w: jax.Array,
    grad: Any,
    shift: jax.Array,
) -> Any:
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

    rhs = precision.cast(grad_flat, "sr").astype(S.dtype)
    delta = linalg.solve_dense(S, rhs, shift)
    return unravel(delta.astype(grad_flat.dtype))


@partial(jax.jit, static_argnums=(0, 7))
def _matvec_step(
    coord: Any,
    theta: Any,
    x: Any,
    w: jax.Array,
    grad: Any,
    shift: jax.Array,
    guess: jax.Array,
    max_iter: int,
) -> tuple[Any, jax.Array]:
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

    rhs = precision.cast(grad_flat, "sr")
    guess = precision.cast(guess, "sr").astype(rhs.dtype)

    delta = linalg.solve_matvec(
        matvec,
        rhs,
        shift,
        x0=guess,
        max_iter=max_iter,
    )
    return (
        unravel(delta.astype(grad_flat.dtype)),
        delta.astype(guess.dtype),
    )
