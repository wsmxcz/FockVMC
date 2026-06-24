from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def normalize(w, mask=None) -> jax.Array:
    """Return non-negative weights normalized to sum one."""
    w = jnp.asarray(w)

    if not jnp.issubdtype(w.dtype, jnp.inexact):
        w = w.astype(jnp.float64)

    if mask is not None:
        w = jnp.where(jnp.asarray(mask), w, 0.0)

    tiny = jnp.asarray(jnp.finfo(w.dtype).tiny, dtype=w.dtype)
    return w / jnp.maximum(jnp.sum(w), tiny)


def masked_sum(x, mask, axis: int = 0):
    """Sum a PyTree after zeroing inactive entries."""
    mask = jnp.asarray(mask, dtype=bool)

    def one(a):
        a = jnp.asarray(a)
        shape = [1] * a.ndim
        shape[int(axis)] = mask.shape[0]
        return jnp.sum(jnp.where(mask.reshape(shape), a, 0), axis=axis)

    return jax.tree.map(one, x)


def masked_logsumexp(x, mask, axis: int = -1) -> jax.Array:
    """Log-sum-exp over active entries, returning -inf for empty segments."""
    x = jnp.asarray(x)
    mask = jnp.asarray(mask, dtype=bool)
    return jax.nn.logsumexp(jnp.where(mask, x, -jnp.inf), axis=axis)


def segment_logsumexp(ptr: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Host log-sum-exp over contiguous segments."""
    ptr = np.asarray(ptr, dtype=np.int64)
    values = np.asarray(values)

    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    values = values.astype(dtype, copy=False)

    out = np.full(int(n), -np.inf, dtype=dtype)
    count = np.diff(ptr)
    active = np.flatnonzero(count > 0)

    if active.size == 0:
        return out

    start = ptr[active]
    seg_max = np.maximum.reduceat(values, start)

    label = np.repeat(np.arange(active.size, dtype=np.int64), count[active])
    shifted = np.exp(values - seg_max[label])
    seg_sum = np.add.reduceat(shifted, start)

    out[active] = seg_max + np.log(seg_sum)
    return out
