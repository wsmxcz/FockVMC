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


def segment_logsumexp(row_ptr: np.ndarray, values: np.ndarray, n_row: int) -> np.ndarray:
    """CSR row-wise logsumexp on host."""
    row_ptr = np.asarray(row_ptr, dtype=np.int64)
    values = np.asarray(values)

    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.float64
    values = values.astype(dtype, copy=False)

    out = np.full(int(n_row), -np.inf, dtype=dtype)
    count = np.diff(row_ptr)
    rows = np.flatnonzero(count > 0)

    if rows.size == 0:
        return out

    start = row_ptr[rows]
    row_max = np.maximum.reduceat(values, start)

    label = np.repeat(np.arange(rows.size, dtype=np.int64), count[rows])
    shifted = np.exp(values - row_max[label])
    row_sum = np.add.reduceat(shifted, start)

    out[rows] = row_max + np.log(row_sum)
    return out