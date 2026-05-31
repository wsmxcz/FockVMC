from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .adamsr import adamsr
from .base import Geometry
from .minsr import minsr
from .sr import sr


def stats(opt_state: Any) -> dict[str, float]:
    """Collect scalar optimizer diagnostics from an Optax state tree."""
    out: dict[str, float] = {}

    def visit(node: Any) -> None:
        if hasattr(node, "stats") and isinstance(node.stats, dict):
            for key, value in node.stats.items():
                value = jax.device_get(value)
                out[str(key)] = float(jnp.asarray(value))

        if isinstance(node, dict):
            for value in node.values():
                visit(value)
        elif isinstance(node, tuple | list):
            for value in node:
                visit(value)

    visit(opt_state)
    return out


__all__ = [
    "Geometry",
    "adamsr",
    "minsr",
    "sr",
    "stats",
]