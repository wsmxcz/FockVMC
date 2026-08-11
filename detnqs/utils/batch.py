"""Shape control for JAX kernels."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    jax.clear_caches()


def chunk(
    n: int,
    expansion: int = 1,
    size: int | None = None,
) -> Iterator[slice]:
    """Partition source items by their estimated downstream work."""
    n = int(n)
    expansion = int(expansion)
    if n < 0 or expansion < 0:
        raise ValueError("n and expansion must be nonnegative")

    size = config["forward_chunk"] if size is None else int(size)
    if size is not None and size <= 0:
        raise ValueError("size must be positive or None")
    if n == 0:
        return

    expansion = max(1, expansion)
    step = n if size is None else max(1, size // expansion)

    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def apply(fun: Callable[[Any, Any], Any], theta: Any, x: Any) -> Any:
    """Evaluate `fun(theta, x)` over the leading axis."""
    run = jax.jit(fun)
    size = config["forward_chunk"]
    if size is None:
        return run(theta, x)

    n = int(jax.tree.leaves(x)[0].shape[0])
    out = []
    slices = (slice(0, 0),) if n == 0 else chunk(n, size=size)
    for slc in slices:
        n_chunk = slc.stop - slc.start
        xb = jax.tree.map(
            lambda a: jnp.pad(
                jnp.asarray(a[slc]),
                ((0, size - n_chunk),) + ((0, 0),) * (a.ndim - 1),
            ),
            x,
        )
        out.append(jax.tree.map(lambda a: a[:n_chunk], run(theta, xb)))

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
    run = jax.jit(
        lambda p, t, y: jax.jvp(lambda q: fun(q, y), (p,), (t,))
    )
    size = config["backward_chunk"]
    if size is None:
        return run(theta, tangent, x)

    n = int(jax.tree.leaves(x)[0].shape[0])
    vals = []
    tangents = []
    slices = (slice(0, 0),) if n == 0 else chunk(n, size=size)
    for slc in slices:
        n_chunk = slc.stop - slc.start
        xb = jax.tree.map(
            lambda a: jnp.pad(
                jnp.asarray(a[slc]),
                ((0, size - n_chunk),) + ((0, 0),) * (a.ndim - 1),
            ),
            x,
        )
        val, tan = run(theta, tangent, xb)
        vals.append(jax.tree.map(lambda a: a[:n_chunk], val))
        tangents.append(jax.tree.map(lambda a: a[:n_chunk], tan))

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
    run = jax.jit(
        lambda p, y, c: jax.vjp(lambda q: fun(q, y), p)[1](c)[0]
    )
    size = config["backward_chunk"]
    if size is None:
        return jax.tree.map(jnp.conj, run(theta, x, cotangent))

    n = int(jax.tree.leaves(x)[0].shape[0])
    grad = jax.tree.map(jnp.zeros_like, theta)
    slices = (slice(0, 0),) if n == 0 else chunk(n, size=size)
    for slc in slices:
        n_chunk = slc.stop - slc.start
        xb = jax.tree.map(
            lambda a: jnp.pad(
                jnp.asarray(a[slc]),
                ((0, size - n_chunk),) + ((0, 0),) * (a.ndim - 1),
            ),
            x,
        )
        cb = jax.tree.map(
            lambda a: jnp.pad(
                jnp.asarray(a[slc]),
                ((0, size - n_chunk),) + ((0, 0),) * (a.ndim - 1),
            ),
            cotangent,
        )
        grad = jax.tree.map(jnp.add, grad, run(theta, xb, cb))

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

    run = jax.jit(fun, static_argnums=static_argnums)
    if n is None:
        return run(*args)

    size = max(n, int(config["bucket_min"]))
    size = 1 << (size - 1).bit_length()

    padded = tuple(
        arg
        if i in static_argnums or axis is None
        else jax.tree.map(
            lambda a, axis=int(axis): jnp.pad(
                jnp.asarray(a),
                [
                    (0, size - a.shape[axis]) if j == axis % a.ndim else (0, 0)
                    for j in range(a.ndim)
                ],
            ),
            arg,
        )
        for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True))
    )

    out = run(*padded)

    if out_axes is None:
        return out

    if isinstance(out_axes, tuple):
        return tuple(
            y
            if axis is None
            else jax.tree.map(
                lambda a, axis=int(axis): a[
                    (slice(None),) * (axis % a.ndim) + (slice(0, n),)
                ],
                y,
            )
            for y, axis in zip(out, out_axes, strict=True)
        )

    axis = int(out_axes)
    return jax.tree.map(
        lambda a: a[(slice(None),) * (axis % a.ndim) + (slice(0, n),)],
        out,
    )
