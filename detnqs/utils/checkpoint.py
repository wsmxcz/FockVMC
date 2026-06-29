from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def save(file: str | Path, tree: Any) -> Path:
    """Save a numerical PyTree."""
    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    leaves, treedef = jax.tree_util.tree_flatten(jax.device_get(tree))

    arrays: dict[str, np.ndarray] = {
        "treedef": np.frombuffer(pickle.dumps(treedef), dtype=np.uint8),
        "leaf_kind": np.asarray(
            [
                "key"
                if hasattr(leaf, "dtype")
                and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)
                else "array"
                for leaf in leaves
            ],
        ),
    }

    for i, leaf in enumerate(leaves):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(
            leaf.dtype,
            jax.dtypes.prng_key,
        ):
            arrays[f"leaf_{i}"] = np.asarray(jax.random.key_data(leaf))
        else:
            arrays[f"leaf_{i}"] = np.asarray(leaf)

    np.savez_compressed(path, **arrays)
    return path


def load(file: str | Path) -> Any:
    """Load a numerical PyTree."""
    with np.load(Path(file), allow_pickle=False) as data:
        treedef = pickle.loads(bytes(np.asarray(data["treedef"], dtype=np.uint8)))
        kinds = np.asarray(data["leaf_kind"]).astype(str).tolist()

        leaves = [
            jax.random.wrap_key_data(np.asarray(data[f"leaf_{i}"]))
            if kind == "key"
            else np.asarray(data[f"leaf_{i}"])
            for i, kind in enumerate(kinds)
        ]

    return jax.tree_util.tree_unflatten(treedef, leaves)