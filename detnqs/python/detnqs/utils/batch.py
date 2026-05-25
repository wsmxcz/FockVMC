from __future__ import annotations

"""Shape regularization for JAX kernels.

This module centralizes the two shape-control patterns used by DetNQS.

1. Streaming kernels
   Functions such as model forward, Jrd, JVP, and VJP are separable over the
   leading sample axis. They can be evaluated chunk by chunk and stitched
   back together without changing the mathematical result.

2. Bucketed kernels
   Functions such as dense SR/minSR are not separable over samples. They need
   the full batch, but can be padded to a small set of power-of-two shapes to
   reduce recompilation pressure.
"""

from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

type Tree = Any

_CONFIG = {
    "chunk": 1024,
    "bucket_min": 64,
}


def configure(*, chunk: int | None = 1024, bucket_min: int = 64) -> None:
    """Set the global shape policy.

    Args:
        chunk:
            Leading-axis chunk size for streaming kernels.
            Use None to disable streaming chunking.

        bucket_min:
            Minimum bucket size for non-streaming full-batch kernels.
    """
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


def apply(fun: Callable[[Tree, Tree], Tree], theta: Tree, x: Tree) -> Tree:
    """Evaluate a leading-axis separable map.

    Computes:
        y = fun(theta, x)

    If chunking is enabled, x is split along axis 0, each block is padded to
    a fixed chunk size, evaluated under JIT, trimmed, and concatenated.
    """
    chunk = _CONFIG["chunk"]

    if chunk is None:
        return _apply(fun, theta, x)

    ys = [
        _trim(_apply(fun, theta, xb), n, axis=0)
        for xb, n in _chunks(x, chunk)
    ]

    if len(ys) == 1:
        return ys[0]

    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=0), *ys)


def jvp(
    fun: Callable[[Tree, Tree], Tree],
    theta: Tree,
    tangent: Tree,
    x: Tree,
) -> tuple[Tree, Tree]:
    """Evaluate a leading-axis separable JVP.

    Computes:
        y, dy = fun(theta, x), J(theta, x) tangent
    """
    chunk = _CONFIG["chunk"]

    if chunk is None:
        return _jvp(fun, theta, tangent, x)

    ys = []
    dys = []

    for xb, n in _chunks(x, chunk):
        yb, dyb = _jvp(fun, theta, tangent, xb)
        ys.append(_trim(yb, n, axis=0))
        dys.append(_trim(dyb, n, axis=0))

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
    """Evaluate a leading-axis separable VJP.

    Computes:
        g = J(theta, x)^T cotangent

    For chunked execution, each block contributes one parameter cotangent and
    the block cotangents are summed.
    """
    chunk = _CONFIG["chunk"]

    if chunk is None:
        return jax.tree.map(jnp.conj, _vjp(fun, theta, x, cotangent))

    grad = jax.tree.map(jnp.zeros_like, theta)

    for (xb, cb), _ in _chunks((x, cotangent), chunk):
        grad = jax.tree.map(jnp.add, grad, _vjp(fun, theta, xb, cb))

    return jax.tree.map(jnp.conj, grad)


def bucket(
    fun: Callable[..., Tree],
    *args: Tree,
    in_axes: int | tuple[int | None, ...] | None = 0,
    out_axes: int | tuple[int | None, ...] | None = 0,
    static_argnums: int | tuple[int, ...] = (),
) -> Tree:
    """Run a non-streaming kernel on a power-of-two bucket.

    The first non-static argument with a mapped axis determines the true batch
    size. All mapped arguments are padded to the same bucket size before JIT.

    Use this for full-batch kernels whose samples interact with each other. 
    Padding is valid only when padded rows are made inactive by the caller.
    """
    if not isinstance(static_argnums, tuple):
        static_argnums = (int(static_argnums),)

    if not isinstance(in_axes, tuple):
        in_axes = (in_axes,) * len(args)

    n = None

    for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True)):
        if i in static_argnums or axis is None:
            continue

        n = int(jax.tree.leaves(arg)[0].shape[axis])
        break

    if n is None:
        return _jit(fun, static_argnums)(*args)

    bucket_size = max(n, int(_CONFIG["bucket_min"]))
    bucket_size = 1 << (bucket_size - 1).bit_length()

    padded = tuple(
        arg if i in static_argnums or axis is None else _pad(arg, bucket_size, axis)
        for i, (arg, axis) in enumerate(zip(args, in_axes, strict=True))
    )

    out = _jit(fun, static_argnums)(*padded)

    if out_axes is None:
        return out

    if isinstance(out_axes, tuple):
        return tuple(
            y if axis is None else _trim(y, n, axis)
            for y, axis in zip(out, out_axes, strict=True)
        )

    return _trim(out, n, out_axes)


def _chunks(tree: Tree, size: int):
    n = int(jax.tree.leaves(tree)[0].shape[0])

    if n == 0:
        yield _pad(tree, size, axis=0), 0
        return

    for lo in range(0, n, size):
        hi = min(lo + size, n)
        block = jax.tree.map(lambda a: a[lo:hi], tree)
        true_size = hi - lo

        if true_size < size:
            block = _pad(block, size, axis=0)

        yield block, true_size


def _pad(tree: Tree, size: int, axis: int) -> Tree:
    def pad_leaf(a):
        a = np.asarray(jax.device_get(a))
        pad_size = int(size) - int(a.shape[axis])

        if pad_size <= 0:
            return jnp.asarray(a)

        shape = list(a.shape)
        shape[axis] = int(size)

        out = np.zeros(shape, dtype=a.dtype)
        slc = [slice(None)] * a.ndim
        slc[axis] = slice(0, a.shape[axis])
        out[tuple(slc)] = a

        return jnp.asarray(out)

    return jax.tree.map(pad_leaf, tree)


def _trim(tree: Tree, size: int, axis: int) -> Tree:
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