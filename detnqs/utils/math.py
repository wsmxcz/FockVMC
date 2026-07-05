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


def masked_sum(x: Any, mask: Any, axis: int = 0) -> Any:
    """Sum PyTree leaves after masking along one axis."""
    axis = int(axis)
    mask = jnp.asarray(mask, dtype=bool)

    def one(a: Any) -> jax.Array:
        a = jnp.asarray(a)
        shape = [1] * a.ndim
        shape[axis] = mask.shape[0]
        keep = mask.reshape(shape)
        return jnp.sum(jnp.where(keep, a, 0), axis=axis)

    return jax.tree.map(one, x)


def masked_logsumexp(x: Any, mask: Any, axis: int = -1) -> jax.Array:
    """Log-sum-exp over active entries, returning -inf for empty segments."""
    axis = int(axis)
    x = jnp.asarray(x)
    if not jnp.issubdtype(x.dtype, jnp.inexact):
        x = x.astype(jnp.float64)
    mask = jnp.asarray(mask, dtype=bool)

    return jax.nn.logsumexp(jnp.where(mask, x, -jnp.inf), axis=axis)


def signed_logsumexp(
    sign: Any,
    logabs: Any,
    axis: int = -1,
) -> tuple[jax.Array, jax.Array]:
    """Stable log of a signed exponential sum."""
    axis = int(axis)
    sign = jnp.asarray(sign)
    logabs = jnp.asarray(logabs)
    if not jnp.issubdtype(logabs.dtype, jnp.inexact):
        logabs = logabs.astype(jnp.float64)

    dtype = logabs.dtype
    sign = sign.astype(dtype)

    zero = jnp.asarray(0.0, dtype=dtype)
    ninf = jnp.asarray(-jnp.inf, dtype=dtype)

    valid = (sign != 0) & jnp.isfinite(sign) & jnp.isfinite(logabs)
    safe_sign = jnp.where(valid, sign, zero)
    safe_log = jnp.where(valid, logabs, ninf)

    scale = jnp.max(safe_log, axis=axis, keepdims=True)
    scale = jnp.where(jnp.isfinite(scale), scale, zero)

    amp = jnp.sum(
        safe_sign * jnp.exp(safe_log - scale),
        axis=axis,
        keepdims=True,
    )

    out_sign = jnp.sign(amp).astype(dtype)
    out_log = jnp.where(amp != 0, scale + jnp.log(jnp.abs(amp)), ninf)

    return (
        jnp.squeeze(out_sign, axis=axis),
        jnp.squeeze(out_log, axis=axis),
    )


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
