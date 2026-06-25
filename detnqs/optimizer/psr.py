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

Tree = Any
Shift = float | Callable[[jax.Array], Any]


class PSRState(NamedTuple):
    """Optimizer state for predictive SR."""

    step: jax.Array
    p: Tree
    v: Tree
    stats: dict[str, jax.Array]


def psr(
    *,
    shift: Shift = 1.0e-3,
    mu: float = 0.95,
    beta: float = 0.995,
) -> optax.GradientTransformationExtraArgs:
    """Predictive stochastic reconfiguration.

    PSR solves a damped sample-space SR equation around a predicted natural
    step p:

        r = b - O p,
        K = O C O^dagger,
        (K + shift I) a = r,
        q = C O^dagger a,
        delta = p + q.
    """
    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must satisfy 0 <= mu < 1")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")

    def init_fn(params: Tree) -> PSRState:
        dtype = precision.real("sr")
        zero = jnp.asarray(0.0, dtype=dtype)

        return PSRState(
            step=jnp.zeros((), dtype=jnp.int32),
            p=jax.tree.map(jnp.zeros_like, params),
            v=jax.tree.map(
                lambda x: jnp.zeros_like(jnp.real(jnp.asarray(x))),
                params,
            ),
            stats={
                "sr_shift": zero,
                "sr_residual": zero,
                "sr_step_norm": zero,
                "sr_cond": zero,
                "sr_fallback": zero,
            },
        )

    def update_fn(
        updates: Tree,
        state: PSRState,
        params: Tree | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Tree, PSRState]:
        del updates, params, extra_args

        if geometry is None:
            raise ValueError("optimizer.psr requires geometry")

        shift_value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(shift_value, dtype=precision.real("sr"))

        delta_raw, q_raw, info = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            state.p,
            state.v,
            geometry.x,
            precision.cast(math.normalize(geometry.w), "model", "real"),
            precision.cast(geometry.b, "model", "real"),
            shift_t,
            in_axes=(None, None, None, None, 0, 0, 0, None),
            out_axes=None,
            static_argnums=0,
        )

        delta = jax.tree.map(
            lambda d, theta: d.astype(theta.dtype),
            delta_raw,
            geometry.theta,
        )

        p_next = jax.tree.map(
            lambda d: jnp.asarray(mu, dtype=d.dtype) * d,
            delta,
        )

        v_next = jax.tree.map(
            lambda old, q: (
                jnp.asarray(beta, dtype=old.dtype) * old
                + (1.0 - jnp.asarray(beta, dtype=old.dtype))
                * jnp.real(q * q.conj()).astype(old.dtype)
            ),
            state.v,
            q_raw,
        )

        dtype = precision.real("sr")
        step_norm2 = sum(
            (
                jnp.sum(jnp.real(jnp.asarray(x) * jnp.conj(jnp.asarray(x))))
                for x in jax.tree.leaves(delta)
            ),
            jnp.asarray(0.0, dtype=dtype),
        )

        return delta, PSRState(
            step=state.step + 1,
            p=p_next,
            v=v_next,
            stats={
                "sr_shift": info["shift"],
                "sr_residual": info["residual"],
                "sr_step_norm": jnp.sqrt(step_norm2),
                "sr_cond": info["cond"],
                "sr_fallback": info["fallback"],
            },
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _step(
    coord,
    theta: Tree,
    p: Tree,
    v: Tree,
    x: jax.Array,
    w: jax.Array,
    b: jax.Array,
    shift: jax.Array,
) -> tuple[Tree, Tree, dict[str, jax.Array]]:
    """One dense sample-space PSR solve on one fixed-shape bucket."""
    b_flat, unravel_b = ravel_pytree(b)
    nrow = b_flat.size
    nsample = w.shape[0]

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(theta)],
        *[leaf.dtype for leaf in jax.tree.leaves(p)],
    )

    v_sum = jnp.asarray(0.0, dtype=precision.real("sr"))
    size = 0
    for leaf in jax.tree.leaves(v):
        leaf = jnp.maximum(jnp.asarray(leaf), 0.0)
        v_sum = v_sum + jnp.sum(leaf)
        size += leaf.size

    v_mean = v_sum / float(size)
    cold = v_mean == 0.0

    def cov(leaf):
        leaf = jnp.maximum(jnp.asarray(leaf), 0.0)
        numer = jnp.asarray(v_mean, dtype=leaf.dtype)
        denom = leaf + numer
        numer = jnp.where(cold, jnp.ones_like(leaf), numer)
        denom = jnp.where(cold, jnp.ones_like(leaf), denom)
        return numer / denom

    def coord_one(params: Tree, sample: Tree):
        sample = jax.tree.map(lambda z: z[None, ...], sample)
        out = coord(params, sample)
        return jax.tree.map(lambda z: jnp.asarray(z)[0], out)

    K = jnp.zeros((nrow, nrow), dtype=dtype)
    for block, put, v_block in tree.blocks(theta, batch.config["param_chunk"], v):
        def coord_block(block, sample):
            return coord_one(put(block), sample)

        J = jax.vmap(
            jax.jacrev(coord_block),
            in_axes=(None, 0),
        )(block, x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        OC = O * jnp.sqrt(cov(v_block).astype(dtype))[None, :]
        K = K + OC @ OC.conj().T

    K = precision.cast(K, "sr")
    rhs = precision.cast(b_flat, "sr").astype(K.dtype)

    _, Jp = batch.jvp(coord, theta, p, x)

    def center(tangent):
        shape = (nsample,) + (1,) * (tangent.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * tangent, axis=0)
        return jnp.sqrt(weight) * (tangent - mean)

    Op, _ = ravel_pytree(jax.tree.map(center, Jp))
    rhs = rhs - precision.cast(Op, "sr").astype(K.dtype)

    a, info = linalg.solve_dense(K, rhs, shift)
    a_tree = unravel_b(a.astype(dtype))

    def cotangent(a_leaf):
        shape = (nsample,) + (1,) * (a_leaf.ndim - 1)
        sqrt_w = jnp.sqrt(w).reshape(shape)
        weighted = sqrt_w * a_leaf
        return weighted - w.reshape(shape) * jnp.sum(weighted, axis=0)

    gradient = batch.vjp(
        coord,
        theta,
        x,
        jax.tree.map(cotangent, a_tree),
    )

    q = jax.tree.map(
        lambda g, leaf: cov(leaf).astype(g.dtype) * g,
        gradient,
        v,
    )
    delta = jax.tree.map(
        lambda pred, corr: jnp.asarray(pred).astype(corr.dtype) + corr,
        p,
        q,
    )

    return delta, q, info
