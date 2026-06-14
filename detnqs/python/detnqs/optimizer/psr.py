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


class PSRState(NamedTuple):
    """Optimizer state for Predictive SR."""

    step: jax.Array
    p: Tree
    v: Tree
    stats: dict[str, jax.Array]


def psr(
    *,
    shift: Shift = 1.0e-2,
    mu: float = 0.9,
    beta: float = 0.999,
) -> optax.GradientTransformationExtraArgs:
    """Predictive stochastic reconfiguration.

    PSR solves a damped sample-space SR equation around a predicted
    natural step p.

        r = b - O p,
        K = O C O^dagger,
        (K + shift I) a = r,
        q = C O^dagger a,
        delta = p + q.

    Here O is the centered weighted log-derivative matrix, b is the
    sample-space SR force, p is the predicted step, q is the residual
    correction, and C is a conservative diagonal covariance.

    The state variables are updated as

        p <- mu * delta,
        v <- beta * v + (1 - beta) * |q|^2.

    The covariance is defined by

        C_i = mean(v) / (v_i + mean(v)),

    with C = I at initialization. Thus directions with large historical
    residual corrections are shrunk.
    """
    if not callable(shift) and shift < 0.0:
        raise ValueError("shift must be non-negative")
    if not 0.0 <= mu < 1.0:
        raise ValueError("mu must satisfy 0 <= mu < 1")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must satisfy 0 <= beta < 1")

    def init_fn(params: Tree) -> PSRState:
        dtype = precision.dtype("sr", "real")
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
        shift_t = jnp.asarray(shift_value, dtype=precision.dtype("sr", "real"))

        delta_raw, q_raw, info = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            state.p,
            state.v,
            geometry.x,
            precision.asarray(normalize(geometry.w), "model", "real"),
            precision.asarray(geometry.b, "model", "real"),
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

        # Damped prediction of the next natural update.
        p_next = jax.tree.map(
            lambda d: jnp.asarray(mu, dtype=d.dtype) * d,
            delta,
        )

        # Track the residual correction, not the raw gradient.
        v_next = jax.tree.map(
            lambda v_old, q: (
                jnp.asarray(beta, dtype=v_old.dtype) * v_old
                + (1.0 - jnp.asarray(beta, dtype=v_old.dtype))
                * jnp.real(q * q.conj()).astype(v_old.dtype)
            ),
            state.v,
            q_raw,
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
            "sr_cond": info["cond"],
            "sr_fallback": info["fallback"],
        }

        return delta, PSRState(
            step=state.step + 1,
            p=p_next,
            v=v_next,
            stats=stats,
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

    theta_leaves, treedef = tree_util.tree_flatten(theta)
    p_leaves = tree_util.tree_leaves(p)
    v_leaves = tree_util.tree_leaves(v)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in theta_leaves],
        *[leaf.dtype for leaf in p_leaves],
    )

    # The global scale of the correction variance defines the covariance
    # reference level.
    v_sum = jnp.asarray(0.0, dtype=precision.dtype("sr", "real"))
    size = 0

    for v_leaf in v_leaves:
        v_leaf = jnp.maximum(jnp.asarray(v_leaf), 0.0)
        v_sum = v_sum + jnp.sum(v_leaf)
        size += v_leaf.size

    v_mean = v_sum / float(size)
    cold_start = v_mean == 0.0

    K = jnp.zeros((nrow, nrow), dtype=dtype)

    def coord_one(params: Tree, sample: Tree):
        sample = jax.tree.map(lambda z: z[None, ...], sample)
        out = coord(params, sample)
        return jax.tree.map(lambda z: jnp.asarray(z)[0], out)

    for i, (theta_leaf, v_leaf) in enumerate(
        zip(theta_leaves, v_leaves, strict=True)
    ):
        def coord_leaf(leaf, sample):
            leaves = list(theta_leaves)
            leaves[i] = leaf
            params = tree_util.tree_unflatten(treedef, leaves)
            return coord_one(params, sample)

        J = jax.vmap(
            jax.jacrev(coord_leaf),
            in_axes=(None, 0),
        )(theta_leaf, x)
        J = J.reshape((nsample, -1, theta_leaf.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        O = O.reshape(nrow, theta_leaf.size).astype(dtype)

        v_flat = jnp.maximum(jnp.asarray(v_leaf).reshape(-1), 0.0)

        numer = jnp.asarray(v_mean, dtype=v_flat.dtype)
        denom = v_flat + numer

        numer = jnp.where(cold_start, jnp.ones_like(v_flat), numer)
        denom = jnp.where(cold_start, jnp.ones_like(v_flat), denom)

        c = (numer / denom).astype(dtype)
        OC_sqrt = O * jnp.sqrt(c)[None, :]
        K = K + OC_sqrt @ OC_sqrt.conj().T

    K = precision.asarray(K, "sr")

    # Residual SR equation around the predicted step p.
    rhs = precision.asarray(b_flat, "sr").astype(K.dtype)

    _, Jp = batch.jvp(coord, theta, p, x)

    def center(tangent):
        shape = (nsample,) + (1,) * (tangent.ndim - 1)
        weight = w.reshape(shape)
        mean = jnp.sum(weight * tangent, axis=0)
        return jnp.sqrt(weight) * (tangent - mean)

    Op, _ = ravel_pytree(jax.tree.map(center, Jp))
    Op = precision.asarray(Op, "sr")
    rhs = rhs - Op.astype(K.dtype)

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

    q_leaves = []
    delta_leaves = []

    for g_leaf, p_leaf, v_leaf in zip(
        tree_util.tree_leaves(gradient),
        p_leaves,
        v_leaves,
        strict=True,
    ):
        v_leaf = jnp.maximum(jnp.asarray(v_leaf), 0.0)
        numer = jnp.asarray(v_mean, dtype=v_leaf.dtype)
        denom = v_leaf + numer

        numer = jnp.where(cold_start, jnp.ones_like(v_leaf), numer)
        denom = jnp.where(cold_start, jnp.ones_like(v_leaf), denom)

        q_leaf = (numer / denom).astype(g_leaf.dtype) * g_leaf
        q_leaves.append(q_leaf)
        delta_leaves.append(jnp.asarray(p_leaf).astype(q_leaf.dtype) + q_leaf)

    q = tree_util.tree_unflatten(treedef, q_leaves)
    delta = tree_util.tree_unflatten(treedef, delta_leaves)

    return delta, q, info
