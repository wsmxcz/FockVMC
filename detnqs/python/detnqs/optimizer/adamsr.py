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


class AdamSRState(NamedTuple):
    """Optimizer state for native Adam-style minSR."""

    step: jax.Array
    p: Tree
    v: Tree
    shift: jax.Array


def adamsr(
    *,
    shift: Shift = 1.0e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    fallback: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Adam-style sample-space SR.

    The carried predictor p is first projected into sample space:

        r = b - O p.

    The adaptive diagonal metric is

        D_i = sqrt(v_mean / (v_hat_i + v_mean)),

    giving the sample-space system

        (O D^2 O^† + shift I) a = r.

    The innovation and final update are

        q = D^2 O^† a,
        delta = p + q.

    The second moment tracks the innovation q, not the carried predictor p.
    """

    if shift < 0.0 and not callable(shift):
        raise ValueError("shift must be non-negative")
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("beta1 must satisfy 0 <= beta1 < 1")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("beta2 must satisfy 0 <= beta2 < 1")

    def init_fn(params: Tree) -> AdamSRState:
        step = jnp.zeros((), dtype=jnp.int32)
        value = shift(step) if callable(shift) else shift

        return AdamSRState(
            step=step,
            p=jax.tree.map(jnp.zeros_like, params),
            v=jax.tree.map(lambda x: jnp.zeros_like(jnp.real(jnp.asarray(x))), params),
            shift=jnp.asarray(value, dtype=precision.dtype("sr", "real")),
        )

    def update_fn(
        updates: Tree,
        state: AdamSRState,
        params: Tree | None = None,
        *,
        geometry: Geometry | None = None,
        **extra_args: Any,
    ) -> tuple[Tree, AdamSRState]:
        del updates, params, extra_args

        if geometry is None:
            raise ValueError("optimizer.adamsr requires geometry")

        value = shift(state.step) if callable(shift) else shift
        shift_t = jnp.asarray(value, dtype=precision.dtype("sr", "real"))

        delta_raw, q_raw = batch.bucket(
            _step,
            geometry.coord,
            geometry.theta,
            state.p,
            state.v,
            geometry.x,
            precision.asarray(normalize(geometry.w), "model", "real"),
            precision.asarray(geometry.b, "model", "real"),
            shift_t,
            jnp.asarray(beta2, dtype=precision.dtype("sr", "real")),
            state.step,
            bool(fallback),
            in_axes=(None, None, None, None, 0, 0, 0, None, None, None, None),
            out_axes=None,
            static_argnums=(0, 10),
        )

        v_next = jax.tree.map(
            lambda v_old, q: (
                jnp.asarray(beta2, dtype=v_old.dtype) * v_old
                + (1.0 - jnp.asarray(beta2, dtype=v_old.dtype))
                * jnp.real(q * q.conj()).astype(v_old.dtype)
            ),
            state.v,
            q_raw,
        )

        p_next = jax.tree.map(
            lambda d: jnp.asarray(beta1, dtype=d.dtype) * d,
            delta_raw,
        )

        delta = jax.tree.map(lambda d, p: d.astype(p.dtype), delta_raw, geometry.theta)

        return delta, AdamSRState(
            step=state.step + 1,
            p=p_next,
            v=v_next,
            shift=shift_t,
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
    beta2: jax.Array,
    step: jax.Array,
    fallback: bool,
) -> tuple[Tree, Tree]:
    """One AdamSR solve on one fixed-shape bucket."""

    b_flat, _ = ravel_pytree(b)
    nrow = b_flat.size

    jac = jax.jacrev(lambda params: coord(params, x))(theta)

    theta_leaves, treedef = tree_util.tree_flatten(theta)
    jac_leaves = tree_util.tree_leaves(jac)
    p_leaves = tree_util.tree_leaves(p)
    v_leaves = tree_util.tree_leaves(v)

    K_dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in theta_leaves],
        *[leaf.dtype for leaf in jac_leaves],
    )

    bias = 1.0 - beta2**step
    bias = jnp.where(step == 0, 1.0, bias)

    v_hat_leaves = []
    v_hat_sum = 0.0
    total_count = 0

    for v_leaf in v_leaves:
        v_hat = jnp.asarray(v_leaf) / bias
        v_hat = jnp.maximum(v_hat, 0.0)

        v_hat_leaves.append(v_hat)
        v_hat_sum = v_hat_sum + jnp.sum(v_hat)
        total_count += v_hat.size

    v_mean = v_hat_sum / float(total_count)
    cold_start = (step == 0) | (v_mean == 0.0)

    K = jnp.zeros((nrow, nrow), dtype=K_dtype)
    Op = jnp.zeros((nrow,), dtype=K_dtype)

    blocks = []
    scales = []

    for J, p_theta, p_leaf, v_hat in zip(
        jac_leaves,
        theta_leaves,
        p_leaves,
        v_hat_leaves,
        strict=True,
    ):
        wb = w.reshape((w.shape[0],) + (1,) * (J.ndim - 1))
        mean = jnp.sum(wb * J, axis=0, keepdims=True)
        O = (jnp.sqrt(wb) * (J - mean)).reshape(-1, p_theta.size).astype(K_dtype)

        v_mean_leaf = jnp.asarray(v_mean, dtype=v_hat.dtype)
        denom = jnp.where(cold_start, 1.0, v_hat + v_mean_leaf)

        d = jnp.sqrt(v_mean_leaf / denom)
        d = jnp.where(cold_start, jnp.ones_like(d), d)
        d = d.reshape(-1).astype(K_dtype)

        p_flat = jnp.asarray(p_leaf).reshape(-1).astype(K_dtype)
        OD = O * d.reshape((1, -1))

        blocks.append(O)
        scales.append(d)

        K = K + OD @ OD.conj().T
        Op = Op + O @ p_flat

    K = precision.asarray(K, "sr")
    rhs = precision.asarray(b_flat, "sr").astype(K.dtype)
    rhs = rhs - precision.asarray(Op, "sr").astype(K.dtype)

    a = linalg.solve_dense(K, rhs, shift, fallback=fallback).astype(K_dtype)

    q_leaves = []
    delta_leaves = []

    for O, d, p_leaf in zip(blocks, scales, p_leaves, strict=True):
        q_flat = (d**2) * (O.conj().T @ a)
        q_leaf = q_flat.reshape(p_leaf.shape)

        p_cast = jnp.asarray(p_leaf).astype(q_leaf.dtype)
        q_leaves.append(q_leaf)
        delta_leaves.append(p_cast + q_leaf)

    return (
        tree_util.tree_unflatten(treedef, delta_leaves),
        tree_util.tree_unflatten(treedef, q_leaves),
    )