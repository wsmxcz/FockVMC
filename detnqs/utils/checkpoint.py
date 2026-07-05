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
        if is_key[i]:
            arrays[f"leaf_{i}"] = np.asarray(jax.random.key_data(leaf))
        else:
            arrays[f"leaf_{i}"] = np.asarray(leaf)

    np.savez_compressed(path, **arrays)
    return path


def load(file: str | Path) -> Any:
    """Load a numerical tree."""
    with np.load(Path(file), allow_pickle=False) as data:
        treedef = pickle.loads(bytes(np.asarray(data["treedef"], dtype=np.uint8)))
        is_key = np.asarray(data["is_key"], dtype=bool)

        leaves = [
            jax.random.wrap_key_data(np.asarray(data[f"leaf_{i}"]))
            if key
            else np.asarray(data[f"leaf_{i}"])
            for i, key in enumerate(is_key)
        ]

    return jax.tree_util.tree_unflatten(treedef, leaves)
