"""Shape control for JAX kernels."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp


config = {
    "forward_chunk": 8192,
    "backward_chunk": 8192,
    "param_chunk": None,
    "bucket_min": 1024,
}


def configure(
    *,
    forward_chunk: int | None = 8192,
    backward_chunk: int | None = 8192,
    param_chunk: int | None = None,
    bucket_min: int = 1024,
) -> None:
    """Set chunk sizes for JAX kernels."""
    for key, value in {
        "forward_chunk": forward_chunk,
        "backward_chunk": backward_chunk,
        "param_chunk": param_chunk,
    }.items():
        if value is not None:
            value = int(value)
            if value <= 0:
                raise ValueError(f"{key} must be positive or None")
        config[key] = value

    bucket_min = int(bucket_min)
    if bucket_min <= 0:
        raise ValueError("bucket_min must be positive")

    config["bucket_min"] = bucket_min
    _jit.cache_clear()
    jax.clear_caches()


def chunks(tree: Any, size: int | None):
    """Yield leading-axis chunks and their true sizes."""
    n = int(jax.tree.leaves(tree)[0].shape[0])

    if size is None:
        yield tree, n
        return

    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive or None")

    if n == 0:
        yield pad(tree, size), 0
        return

    for lo in range(0, n, size):
        hi = min(lo + size, n)
        chunk = jax.tree.map(lambda x: x[lo:hi], tree)
        yield (pad(chunk, size) if hi - lo < size else chunk), hi - lo


def slices(n: int, expansion: int = 1):
    """Yield source slices whose expanded work fits one forward chunk."""
    n = int(n)
    expansion = int(expansion)
    if n < 0 or expansion < 0:
        raise ValueError("n and expansion must be nonnegative")
    if n == 0:
        return

    expansion = max(1, expansion)
    limit = config["forward_chunk"]
    step = n if limit is None else max(1, int(limit) // expansion)

    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def apply(fun: Callable[[Any, Any], Any], theta: Any, x: Any) -> Any:
    """Evaluate `fun(theta, x)` over the leading axis."""
    size = config["forward_chunk"]
    if size is None:
        return _apply(fun, theta, x)

    out = [trim(_apply(fun, theta, xb), n) for xb, n in chunks(x, size)]
    return out[0] if len(out) == 1 else jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *out,
    )


def jvp(
    fun: Callable[[Any, Any], Any],
    theta: Any,
    tangent: Any,
    x: Any,
) -> tuple[Any, Any]:
    """Evaluate `fun(theta, x)` and its parameter JVP."""
    size = config["backward_chunk"]
    if size is None:
        return _jvp(fun, theta, tangent, x)

    vals = []
    tangents = []
    for xb, n in chunks(x, size):
        val, tan = _jvp(fun, theta, tangent, xb)
        vals.append(trim(val, n))
        tangents.append(trim(tan, n))

    if len(vals) == 1:
        return vals[0], tangents[0]

    return (
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *vals),
        jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *tangents),
    )


def vjp(
    fun: Callable[[Any, Any], Any],
    theta: Any,
    x: Any,
    cotangent: Any,
) -> Any:
    """Evaluate `sum_i J_i^dagger cotangent_i`."""
    size = config["backward_chunk"]
    if size is None:
        return jax.tree.map(jnp.conj, _vjp(fun, theta, x, cotangent))

    grad = jax.tree.map(jnp.zeros_like, theta)
    for (xb, cb), _ in chunks((x, cotangent), size):
        grad = jax.tree.map(jnp.add, grad, _vjp(fun, theta, xb, cb))

    return jax.tree.map(jnp.conj, grad)


def bucket(
    fun: Callable[..., Any],
    *args: Any,
    in_axes: int | tuple[int | None, ...] | None = 0,
    out_axes: int | tuple[int | None, ...] | None = 0,
    static_argnums: int | tuple[int, ...] = (),
) -> Any:
    """Run a coupled leading-axis kernel on one padded power-of-two bucket."""
    static_argnums = (
        (int(static_argnums),)
        if not isinstance(static_argnums, tuple)
        else tuple(int(i) for i in static_argnums)
    )
    in_axes = (in_axes,) * len(args) if not isinstance(in_axes, tuple) else in_axes

    if len(in_axes) != len(args):
        raise ValueError("in_axes length must match number of args")

    n = None
    for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True)):
        if i in static_argnums or axis is None:
            continue
        n = int(jax.tree.leaves(arg)[0].shape[int(axis)])
        break

    if n is None:
        return _jit(fun, static_argnums)(*args)

    size = max(n, int(config["bucket_min"]))
    size = 1 << (size - 1).bit_length()

    padded = tuple(
        arg if i in static_argnums or axis is None else pad(arg, size, int(axis))
        for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True))
    )

    out = _jit(fun, static_argnums)(*padded)

    if out_axes is None:
        return out

    if isinstance(out_axes, tuple):
        return tuple(
            y if axis is None else trim(y, n, int(axis))
            for y, axis in zip(out, out_axes, strict=True)
        )

    return trim(out, n, int(out_axes))


def pad(tree: Any, size: int, axis: int = 0) -> Any:
    """Pad a PyTree along one axis."""
    axis = int(axis)
    size = int(size)

    def one(x: Any) -> jax.Array:
        x = jnp.asarray(x)
        n = size - int(x.shape[axis])
        if n <= 0:
            return x

        width = [(0, 0)] * x.ndim
        width[axis] = (0, n)
        return jnp.pad(x, width)

    return jax.tree.map(one, tree)


def trim(tree: Any, size: int, axis: int = 0) -> Any:
    """Trim a PyTree along one axis."""
    axis = int(axis)
    size = int(size)

    def one(x: Any) -> jax.Array:
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, size)
        return x[tuple(slc)]

    return jax.tree.map(one, tree)


@lru_cache(maxsize=16)
def _jit(fun: Callable[..., Any], static_argnums: tuple[int, ...]):
    return jax.jit(fun, static_argnums=static_argnums)


@partial(jax.jit, static_argnums=0)
def _apply(fun: Callable[[Any, Any], Any], theta: Any, x: Any) -> Any:
    return fun(theta, x)


@partial(jax.jit, static_argnums=0)
def _jvp(
    fun: Callable[[Any, Any], Any],
    theta: Any,
    tangent: Any,
    x: Any,
) -> tuple[Any, Any]:
    return jax.jvp(lambda p: fun(p, x), (theta,), (tangent,))


@partial(jax.jit, static_argnums=0)
def _vjp(
    fun: Callable[[Any, Any], Any],
    theta: Any,
    x: Any,
    cotangent: Any,
) -> Any:
    _, pullback = jax.vjp(lambda p: fun(p, x), theta)
    return pullback(cotangent)[0]
