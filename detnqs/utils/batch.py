from __future__ import annotations

"""Shape control for JAX kernels.

DetNQS uses three independent chunk sizes.

forward_chunk:
    Leading-axis chunks for model forward evaluations.

backward_chunk:
    Leading-axis chunks for JVP and VJP evaluations.

param_chunk:
    Flattened parameter-leaf column chunks used by PyTree utilities.

Bucketed kernels are padded to a power-of-two size before JIT compilation.
Padding is only valid when the caller makes padded entries inactive.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp

Tree = Any

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
    """Set the global JAX shape policy."""
    values = {
        "forward_chunk": forward_chunk,
        "backward_chunk": backward_chunk,
        "param_chunk": param_chunk,
    }

    for key, value in values.items():
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


def bucket_size(n: int) -> int:
    """Return the power-of-two bucket used for a true leading size."""
    size = max(int(n), int(config["bucket_min"]))
    return 1 << (size - 1).bit_length()


def chunks(tree: Tree, size: int | None):
    """Yield leading-axis chunks and their true sizes."""
    n = int(jax.tree.leaves(tree)[0].shape[0])

    if size is None:
        yield tree, n
        return

    size = int(size)
    if n == 0:
        yield pad(tree, size, axis=0), 0
        return

    for lo in range(0, n, size):
        hi = min(lo + size, n)
        block = jax.tree.map(lambda a: a[lo:hi], tree)
        true = hi - lo
        yield (pad(block, size, axis=0) if true < size else block), true


def mask(n: int, size: int | None = None) -> jax.Array:
    """Return a leading-axis validity mask."""
    n = int(n)
    size = n if size is None else int(size)
    return jnp.arange(size) < n


def apply(fun: Callable[[Tree, Tree], Tree], theta: Tree, x: Tree) -> Tree:
    """Evaluate y_i = f_theta(x_i) over the leading axis."""
    chunk = config["forward_chunk"]
    if chunk is None:
        return _apply(fun, theta, x)

    ys = [trim(_apply(fun, theta, xb), n, axis=0) for xb, n in chunks(x, chunk)]
    if len(ys) == 1:
        return ys[0]
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *ys)


def jvp(
    fun: Callable[[Tree, Tree], Tree],
    theta: Tree,
    tangent: Tree,
    x: Tree,
) -> tuple[Tree, Tree]:
    """Evaluate y_i = f_theta(x_i) and dy_i = J_i tangent."""
    chunk = config["backward_chunk"]
    if chunk is None:
        return _jvp(fun, theta, tangent, x)

    ys = []
    dys = []
    for xb, n in chunks(x, chunk):
        yb, dyb = _jvp(fun, theta, tangent, xb)
        ys.append(trim(yb, n, axis=0))
        dys.append(trim(dyb, n, axis=0))

    if len(ys) == 1:
        return ys[0], dys[0]

    y = jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *ys)
    dy = jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *dys)
    return y, dy


def vjp(
    fun: Callable[[Tree, Tree], Tree],
    theta: Tree,
    x: Tree,
    cotangent: Tree,
) -> Tree:
    """Evaluate sum_i J_i^dagger cotangent_i."""
    chunk = config["backward_chunk"]
    if chunk is None:
        return jax.tree.map(jnp.conj, _vjp(fun, theta, x, cotangent))

    grad = jax.tree.map(jnp.zeros_like, theta)
    for (xb, cb), _ in chunks((x, cotangent), chunk):
        grad = jax.tree.map(jnp.add, grad, _vjp(fun, theta, xb, cb))
    return jax.tree.map(jnp.conj, grad)


def bucket(
    fun: Callable[..., Tree],
    *args: Tree,
    in_axes: int | tuple[int | None, ...] | None = 0,
    out_axes: int | tuple[int | None, ...] | None = 0,
    static_argnums: int | tuple[int, ...] = (),
) -> Tree:
    """Run a coupled leading-axis kernel on a padded power-of-two bucket."""
    if not isinstance(static_argnums, tuple):
        static_argnums = (int(static_argnums),)

    if not isinstance(in_axes, tuple):
        in_axes = (in_axes,) * len(args)

    if len(in_axes) != len(args):
        raise ValueError("in_axes length must match number of args")

    n = None
    for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True)):
        if i in static_argnums or axis is None:
            continue
        n = int(jax.tree.leaves(arg)[0].shape[axis])
        break

    if n is None:
        return _jit(fun, static_argnums)(*args)

    size = bucket_size(n)
    padded = tuple(
        arg if i in static_argnums or axis is None else pad(arg, size, axis)
        for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True))
    )

    out = _jit(fun, static_argnums)(*padded)

    if out_axes is None:
        return out

    if isinstance(out_axes, tuple):
        return tuple(
            y if axis is None else trim(y, n, axis)
            for y, axis in zip(out, out_axes, strict=True)
        )

    return trim(out, n, out_axes)


def pad(tree: Tree, size: int, axis: int = 0) -> Tree:
    def one(a):
        a = jnp.asarray(a)
        n = int(size) - int(a.shape[axis])
        if n <= 0:
            return a
        width = [(0, 0)] * a.ndim
        width[int(axis)] = (0, n)
        return jnp.pad(a, width)

    return jax.tree.map(one, tree)


def trim(tree: Tree, size: int, axis: int = 0) -> Tree:
    def one(a):
        slc = [slice(None)] * a.ndim
        slc[int(axis)] = slice(0, int(size))
        return a[tuple(slc)]

    return jax.tree.map(one, tree)


@lru_cache(maxsize=16)
def _jit(fun: Callable[..., Tree], static_argnums: tuple[int, ...]):
    return jax.jit(fun, static_argnums=static_argnums)


@partial(jax.jit, static_argnums=0)
def _apply(fun: Callable[[Tree, Tree], Tree], theta: Tree, x: Tree) -> Tree:
    return fun(theta, x)


@partial(jax.jit, static_argnums=0)
def _jvp(
    fun: Callable[[Tree, Tree], Tree],
    theta: Tree,
    tangent: Tree,
    x: Tree,
) -> tuple[Tree, Tree]:
    return jax.jvp(lambda p: fun(p, x), (theta,), (tangent,))


@partial(jax.jit, static_argnums=0)
def _vjp(
    fun: Callable[[Tree, Tree], Tree],
    theta: Tree,
    x: Tree,
    cotangent: Tree,
) -> Tree:
    _, pullback = jax.vjp(lambda p: fun(p, x), theta)
    return pullback(cotangent)[0]
