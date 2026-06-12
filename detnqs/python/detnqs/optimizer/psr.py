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

        # Damped constant-step prediction of the next natural update.
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
    b_flat, _ = ravel_pytree(b)
    nrow = b_flat.size

    jac = jax.jacrev(lambda params: coord(params, x))(theta)

    theta_leaves, treedef = tree_util.tree_flatten(theta)
    jac_leaves = tree_util.tree_leaves(jac)
    p_leaves = tree_util.tree_leaves(p)
    v_leaves = tree_util.tree_leaves(v)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in theta_leaves],
        *[leaf.dtype for leaf in jac_leaves],
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
    Op = jnp.zeros((nrow,), dtype=dtype)

    blocks = []
    covariances = []

    for J, p_leaf, v_leaf in zip(
        jac_leaves,
        p_leaves,
        v_leaves,
        strict=True,
    ):
        J = J.reshape((w.shape[0], -1, p_leaf.size))

        # Centered weighted log-derivatives define the SR geometry.
        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        O = O.reshape(nrow, p_leaf.size).astype(dtype)

        # Conservative diagonal covariance:
        #
        #     C_i = mean(v) / (v_i + mean(v)).
        #
        # Directions with large historical corrections are shrunk.
        v_flat = jnp.maximum(jnp.asarray(v_leaf).reshape(-1), 0.0)

        numer = jnp.asarray(v_mean, dtype=v_flat.dtype)
        denom = v_flat + numer

        numer = jnp.where(cold_start, jnp.ones_like(v_flat), numer)
        denom = jnp.where(cold_start, jnp.ones_like(v_flat), denom)

        c = (numer / denom).astype(dtype)
        d = jnp.sqrt(c)

        p_flat = jnp.asarray(p_leaf).reshape(-1).astype(dtype)

        # K = O C O^dagger = (O sqrt(C)) (O sqrt(C))^dagger.
        OC_sqrt = O * d[None, :]

        blocks.append(O)
        covariances.append(c)

        K = K + OC_sqrt @ OC_sqrt.conj().T
        Op = Op + O @ p_flat

    K = precision.asarray(K, "sr")

    # Residual SR equation around the predicted step p.
    rhs = precision.asarray(b_flat, "sr").astype(K.dtype)
    rhs = rhs - precision.asarray(Op, "sr").astype(K.dtype)

    a, info = linalg.solve_dense(K, rhs, shift)
    a = a.astype(dtype)

    q_leaves = []
    delta_leaves = []

    for O, c, p_leaf in zip(blocks, covariances, p_leaves, strict=True):
        # q = C O^dagger a.
        q_flat = c * (O.conj().T @ a)
        q = q_flat.reshape(p_leaf.shape)

        p_cast = jnp.asarray(p_leaf).astype(q.dtype)

        q_leaves.append(q)
        delta_leaves.append(p_cast + q)

    return (
        tree_util.tree_unflatten(treedef, delta_leaves),
        tree_util.tree_unflatten(treedef, q_leaves),
        info,
    )