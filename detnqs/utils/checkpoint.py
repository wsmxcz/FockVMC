from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def save(file: str | Path, tree: Any) -> Path:
    """Save a numerical tree."""
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    leaves, treedef = jax.tree_util.tree_flatten(jax.device_get(tree))

    is_key = np.asarray(
        [
            hasattr(leaf, "dtype")
            and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            for leaf in leaves
        ],
        dtype=bool,
    )
    arrays: dict[str, np.ndarray] = {
        "treedef": np.frombuffer(pickle.dumps(treedef), dtype=np.uint8),
        "is_key": is_key,
    }

    for i, leaf in enumerate(leaves):
        value = jax.random.key_data(leaf) if is_key[i] else leaf
        arrays[f"leaf_{i}"] = np.asarray(value)

    np.savez(path, **arrays)
    return path


def load(file: str | Path, *, key: str | None = None) -> Any:
    """Load a numerical tree or one value from a top-level mapping."""
    with np.load(Path(file), allow_pickle=False) as data:
        treedef = pickle.loads(
            bytes(np.asarray(data["treedef"], dtype=np.uint8))
        )
        is_key = np.asarray(data["is_key"], dtype=bool)

        start = 0
        stop = treedef.num_leaves
        if key is not None:
            node = treedef.node_data()
            if node is None or node[0] is not dict:
                raise ValueError("checkpoint root is not a mapping")

            names = node[1]
            if key not in names:
                raise KeyError(key)

            children = treedef.children()
            index = names.index(key)
            start = sum(child.num_leaves for child in children[:index])
            treedef = children[index]
            stop = start + treedef.num_leaves

        leaves = [
            jax.random.wrap_key_data(np.asarray(data[f"leaf_{i}"]))
            if is_prng
            else np.asarray(data[f"leaf_{i}"])
            for i, is_prng in enumerate(is_key[start:stop], start=start)
        ]

    return jax.tree_util.tree_unflatten(treedef, leaves)
