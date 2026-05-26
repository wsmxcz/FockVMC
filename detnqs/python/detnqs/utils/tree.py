from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Tree = Any


def host(tree: Tree) -> Tree:
    """Move a PyTree to host NumPy arrays."""
    return jax.tree.map(np.asarray, jax.device_get(tree))


def device(tree: Tree) -> Tree:
    """Move a PyTree to the default JAX device."""
    return jax.tree.map(jnp.asarray, tree)


def vdot(a: Tree, b: Tree) -> jax.Array:
    """Real part of the Hermitian PyTree inner product."""
    out = None

    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        term = jnp.real(jnp.vdot(x, y))
        out = term if out is None else out + term

    return jnp.asarray(0.0) if out is None else out