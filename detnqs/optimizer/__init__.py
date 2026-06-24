from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .base import Geometry
from .minsr import minsr
from .psr import psr
from .sr import sr


def stats(opt_state: Any) -> dict[str, float]:
    """Collect scalar optimizer diagnostics from an Optax state tree."""
    out: dict[str, float] = {}
    stack = [opt_state]

    while stack:
        node = stack.pop()
        if hasattr(node, "stats") and isinstance(node.stats, dict):
            for key, value in node.stats.items():
                value = jax.device_get(value)
                out[str(key)] = float(jnp.asarray(value))

        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (tuple, list)):
            stack.extend(node)

    return out


__all__ = (
    "Geometry",
    "minsr",
    "psr",
    "sr",
    "stats",
)
