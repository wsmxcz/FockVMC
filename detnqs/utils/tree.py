from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util


def host(tree: Any) -> Any:
    """Move a PyTree to host NumPy arrays."""
    return jax.tree.map(np.asarray, jax.device_get(tree))


def blocks(tree: Any, size: int | None = None, *aligned: Any):
    """Yield flattened leaf blocks and a local tree updater.

    aligned PyTrees must have the same structure and leaf shapes as tree.
    Each yielded item is

        block, put, *aligned_blocks

    where put(new_block) returns the full primary tree with the current block
    replaced by new_block.
    """
    leaves, spec = tree_util.tree_flatten(tree)
    aligned_leaves = []

    for other in aligned:
        other_leaves, other_spec = tree_util.tree_flatten(other)
        if other_spec != spec:
            raise ValueError("aligned trees must have the same structure")
        aligned_leaves.append(other_leaves)

    if size is not None:
        size = int(size)
        if size <= 0:
            raise ValueError("size must be positive or None")

    for i, leaf in enumerate(leaves):
        leaf = jnp.asarray(leaf)
        flat = leaf.reshape(-1)
        if flat.size == 0:
            continue
        step = flat.size if size is None else size

        for lo in range(0, flat.size, step):
            hi = min(lo + step, flat.size)
            extra = []

            for other in aligned_leaves:
                other_leaf = jnp.asarray(other[i])
                if other_leaf.shape != leaf.shape:
                    raise ValueError("aligned leaves must have the same shapes")
                extra.append(other_leaf.reshape(-1)[lo:hi])

            def put(block, *, i=i, lo=lo, hi=hi, shape=leaf.shape):
                new = list(leaves)
                flat_leaf = jnp.asarray(new[i]).reshape(-1)
                new[i] = flat_leaf.at[lo:hi].set(block).reshape(shape)
                return tree_util.tree_unflatten(spec, new)

            yield (flat[lo:hi], put, *extra)
