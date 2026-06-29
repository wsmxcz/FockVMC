from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import precision


def weight(
    mass: NDArray[Any],
    mass2: NDArray[Any],
    logabs: NDArray[Any],
    logprob: NDArray[Any],
) -> tuple[NDArray[Any], dict[str, float]]:
    """Normalize Born weights from an auxiliary law.

    Unique-ket convention:

        W_x = M_x exp(2 log|psi(x)| - log nu(x)),
        w_x = W_x / sum_y W_y.

    `ess` uses the raw sample second mass M_x^(2). `essu` is the
    participation of the normalized unique-ket Born weights.
    """
    rdtype = precision.real("calc", host=True)
    tiny = rdtype(precision.tiny("calc"))

    mass = precision.cast(np.asarray(mass).reshape(-1), "calc", "real", host=True)
    mass2 = precision.cast(np.asarray(mass2).reshape(-1), "calc", "real", host=True)
    logabs = precision.cast(np.asarray(logabs).reshape(-1), "calc", "real", host=True)
    logprob = precision.cast(np.asarray(logprob).reshape(-1), "calc", "real", host=True)

    logu = rdtype(2.0) * logabs - logprob
    finite = (
        np.isfinite(logu)
        & np.isfinite(mass)
        & np.isfinite(mass2)
        & (mass > 0.0)
    )
    if not finite.any():
        raise FloatingPointError("all importance weights are non-finite")

    u = np.zeros_like(logu, dtype=rdtype)
    u[finite] = np.exp(logu[finite] - rdtype(np.max(logu[finite])))

    wm = mass * u
    norm = float(np.sum(wm))
    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("importance weights have zero total mass")

    w = precision.cast(wm / norm, "calc", "real", host=True)
    ess = float(norm * norm / max(float(np.sum(mass2 * u * u)), float(tiny)))
    essu = float(1.0 / max(float(np.sum(w * w)), float(tiny)))

    return w, {
        "ess": ess,
        "ess_frac": ess / max(1.0, float(np.sum(mass))),
        "essu": essu,
        "essu_frac": essu / max(1.0, float(w.shape[0])),
        "w_max": float(np.max(w)) if w.size else 0.0,
    }


def eloc(
    w: NDArray[Any],
    eloc: NDArray[Any],
    *,
    blocks: NDArray[Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Reduce weighted local-energy statistics.

    `eloc_var` is the variance of the local-energy estimator supplied by the
    caller, including any stochastic weak-window contribution already present.
    """
    w = precision.cast(np.asarray(w).reshape(-1), "calc", "real", host=True)
    eloc = precision.cast(np.asarray(eloc).reshape(-1), "calc", host=True)

    energy = float(np.real(np.dot(w, eloc)))
    residual = eloc - energy
    out = {
        "energy": energy,
        "eloc_var": float(np.real(np.dot(w, np.abs(residual) ** 2))),
    }

    if blocks is not None:
        values = precision.cast(np.asarray(blocks).reshape(-1), "calc", host=True)
        if values.size > 1:
            mean = np.mean(values)
            norm = values.size * (values.size - 1)
            variance = np.sum(np.abs(values - mean) ** 2).real / norm
            out["energy_se"] = float(np.sqrt(variance))

    return energy, out


def observable(name: str, w: NDArray[Any], oloc: NDArray[Any]) -> dict[str, float]:
    """Reduce one local observable estimator `O_L(x)`."""
    w = precision.cast(np.asarray(w).reshape(-1), "calc", "real", host=True)
    oloc = precision.cast(np.asarray(oloc).reshape(-1), "calc", host=True)

    mean = float(np.real(np.dot(w, oloc)))
    return {
        str(name): mean,
        f"{name}_var": float(np.real(np.dot(w, np.abs(oloc - mean) ** 2))),
    }


def update(grad: Any, updates: Any) -> dict[str, float]:
    """Return diagnostics for the final applied Optax update.

    `dE_lin = Re <grad, update>` is the first-order energy change. A descent
    step has `dE_lin < 0` after all Optax transforms.
    """
    dE_lin = 0.0
    norm2 = 0.0
    for g, u in zip(jax.tree.leaves(grad), jax.tree.leaves(updates), strict=True):
        gv = np.asarray(jax.device_get(g)).reshape(-1)
        uv = np.asarray(jax.device_get(u)).reshape(-1)
        dE_lin += float(np.real(np.vdot(gv, uv)))
        norm2 += float(np.real(np.vdot(uv, uv)))
    return {"dE_lin": dE_lin, "update_norm": float(np.sqrt(max(norm2, 0.0)))}


def collect(opt_state: Any) -> dict[str, float]:
    """Collect scalar `.stats` dictionaries from an Optax state PyTree."""
    out: dict[str, float] = {}
    stack = [opt_state]

    while stack:
        node = stack.pop()
        node_stats = getattr(node, "stats", None)
        if isinstance(node_stats, Mapping):
            for key, value in node_stats.items():
                arr = jnp.asarray(jax.device_get(value))
                if arr.ndim == 0 and not jnp.issubdtype(arr.dtype, jnp.complexfloating):
                    out[str(key)] = float(arr)

        if isinstance(node, Mapping):
            stack.extend(node.values())
        elif isinstance(node, tuple | list):
            stack.extend(node)

    return out
