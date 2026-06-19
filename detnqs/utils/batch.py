from __future__ import annotations

"""Shape control for JAX kernels.

Two patterns are used throughout DetNQS.

1. Streaming kernels
   The map is separable over the leading sample axis:

       y_i = f_theta(x_i).

   These kernels can be evaluated chunk by chunk without changing the result.
   This is used for model forward passes, JVPs, and VJPs.

2. Bucketed kernels
   The kernel couples samples inside the batch:

       y = F_theta(x_1, ..., x_N).

   These kernels are padded to a power-of-two bucket before JIT compilation.
   Padding is only valid when the caller makes padded rows inactive.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp

Tree = Any

_CONFIG = {
    "chunk": 8192,
    "bucket_min": 1024,
}


def configure(*, chunk: int | None = 8192, bucket_min: int = 1024) -> None:
    """Set global shape policy for streaming and bucketed kernels."""
    if chunk is not None:
        chunk = int(chunk)
        if chunk <= 0:
            raise ValueError("chunk must be positive or None")

    bucket_min = int(bucket_min)
    if bucket_min <= 0:
        raise ValueError("bucket_min must be positive")

    _CONFIG["chunk"] = chunk
    _CONFIG["bucket_min"] = bucket_min
    jax.clear_caches()


def bucket_size(n: int) -> int:
    """Return the power-of-two bucket used for a true leading size."""
    size = max(int(n), int(_CONFIG["bucket_min"]))
    return 1 << (size - 1).bit_length()


def chunks(tree: Tree, size: int | None = None):
    """Yield padded leading-axis chunks and their true sizes."""
    size = int(_CONFIG["chunk"] if size is None else size)
    n = int(jax.tree.leaves(tree)[0].shape[0])

    if n == 0:
        yield pad(tree, size, axis=0), 0
        return

    for lo in range(0, n, size):
        hi = min(lo + size, n)
        block = jax.tree.map(lambda a: a[lo:hi], tree)
        true_size = hi - lo

        if true_size < size:
            block = pad(block, size, axis=0)

        yield block, true_size


def mask(n: int, size: int | None = None) -> jax.Array:
    """Return a leading-axis validity mask."""
    n = int(n)
    size = n if size is None else int(size)
    return jnp.arange(size) < n


def apply(fun: Callable[[Tree, Tree], Tree], theta: Tree, x: Tree) -> Tree:
    """Evaluate y_i = f_theta(x_i) over the leading axis."""
    chunk = _CONFIG["chunk"]

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
    chunk = _CONFIG["chunk"]

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
    chunk = _CONFIG["chunk"]

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
    """Run a non-streaming kernel on a padded power-of-two batch.

    The first non-static mapped argument defines the true batch size N.
    Mapped arguments are padded to

        N_hat = 2 ** ceil(log2(max(N, bucket_min))).

    Padded rows are not masked here. The caller must make them inactive.
    """
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
    def pad_leaf(a):
        a = jnp.asarray(a)
        pad = int(size) - int(a.shape[axis])

        if pad <= 0:
            return a

        width = [(0, 0)] * a.ndim
        width[axis] = (0, pad)
        return jnp.pad(a, width)

    return jax.tree.map(pad_leaf, tree)


def trim(tree: Tree, size: int, axis: int = 0) -> Tree:
    def trim_leaf(a):
        slc = [slice(None)] * a.ndim
        slc[axis] = slice(0, int(size))
        return a[tuple(slc)]

    return jax.tree.map(trim_leaf, tree)


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
