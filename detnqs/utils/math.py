from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray


def normalize(w: Any, mask: Any | None = None) -> jax.Array:
    """Normalize non-negative weights to unit total mass."""
    w = jnp.asarray(w)
    if not jnp.issubdtype(w.dtype, jnp.inexact):
        w = w.astype(jnp.float64)

    if mask is not None:
        mask = jnp.asarray(mask, dtype=bool)
        w = jnp.where(mask, w, 0.0)

    tiny = jnp.asarray(jnp.finfo(w.dtype).tiny, dtype=w.dtype)
    norm = jnp.maximum(jnp.sum(w), tiny)

    return w / norm


def segment_logsumexp(
    ptr: NDArray[Any],
    values: NDArray[Any],
    n: int,
) -> NDArray[Any]:
    """Host log-sum-exp over contiguous segments."""
    ptr = np.asarray(ptr, dtype=np.int64)
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    n = int(n)

    out = np.full(n, -np.inf, dtype=values.dtype)
    count = np.diff(ptr)
    active = np.flatnonzero(count > 0)

    if active.size == 0:
        return out

    start = ptr[active]
    seg_max = np.maximum.reduceat(values, start)
    label = np.repeat(np.arange(active.size, dtype=np.int64), count[active])
    seg_sum = np.add.reduceat(np.exp(values - seg_max[label]), start)

    out[active] = seg_max + np.log(seg_sum)
    return out
