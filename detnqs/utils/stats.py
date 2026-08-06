from __future__ import annotations

from typing import Any

import jax
import numpy as np
from numpy.typing import NDArray

from . import precision


def weight(
    mass: NDArray[Any],
    mass2: NDArray[Any],
    logabs: NDArray[Any],
    log_observation: NDArray[Any],
) -> tuple[NDArray[Any], dict[str, float]]:
    """Normalize Born weights from an auxiliary law.

    Unique-ket convention:

        W_x = M_x exp(2 log|psi(x)| - log nu(x)),
        w_x = W_x / sum_y W_y.
    """
    dtype = precision.real("calc", host=True)
    tiny = dtype(precision.tiny("calc"))

    mass = precision.cast(np.asarray(mass).reshape(-1), "calc", "real", host=True)
    mass2 = precision.cast(np.asarray(mass2).reshape(-1), "calc", "real", host=True)
    logabs = precision.cast(np.asarray(logabs).reshape(-1), "calc", "real", host=True)
    log_observation = precision.cast(
        np.asarray(log_observation).reshape(-1),
        "calc",
        "real",
        host=True,
    )

    logw = dtype(2.0) * logabs - log_observation
    valid = (
        np.isfinite(logw)
        & np.isfinite(mass)
        & np.isfinite(mass2)
        & (mass > 0.0)
    )

    if not valid.any():
        raise FloatingPointError("all importance weights are non-finite")

    scale = dtype(np.max(logw[valid]))
    unorm = np.zeros_like(logw, dtype=dtype)
    unorm[valid] = np.exp(logw[valid] - scale)

    weighted = mass * unorm
    norm = float(np.sum(weighted))

    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("importance weights have zero total mass")

    weight = precision.cast(weighted / norm, "calc", "real", host=True)

    ess = float(norm * norm / max(float(np.sum(mass2 * unorm * unorm)), float(tiny)))
    essu = float(1.0 / max(float(np.sum(weight * weight)), float(tiny)))

    return weight, {
        "ess": ess,
        "ess_frac": ess / max(1.0, float(np.sum(mass))),
        "essu": essu,
        "essu_frac": essu / max(1.0, float(weight.shape[0])),
        "w_max": float(np.max(weight)) if weight.size else 0.0,
    }


def eloc(
    weight: NDArray[Any],
    eloc: NDArray[Any],
    *,
    blocks: NDArray[Any] | None = None,
) -> tuple[float, dict[str, float]]:
    """Reduce weighted local-energy statistics."""
    weight = precision.cast(
        np.asarray(weight).reshape(-1),
        "calc",
        "real",
        host=True,
    )
    eloc = precision.cast(np.asarray(eloc).reshape(-1), "calc", host=True)

    energy = float(np.real(np.dot(weight, eloc)))
    resid = eloc - energy

    out = {
        "energy": energy,
        "eloc_var": float(np.real(np.dot(weight, np.abs(resid) ** 2))),
    }

    if blocks is not None:
        block = precision.cast(np.asarray(blocks).reshape(-1), "calc", host=True)
        if block.size > 1:
            mean = np.mean(block)
            var = np.sum(np.abs(block - mean) ** 2).real
            out["energy_se"] = float(np.sqrt(var / (block.size * (block.size - 1))))

    return energy, out


def observable(
    name: str,
    weight: NDArray[Any],
    oloc: NDArray[Any],
) -> dict[str, float]:
    """Reduce one local observable."""
    weight = precision.cast(
        np.asarray(weight).reshape(-1),
        "calc",
        "real",
        host=True,
    )
    oloc = precision.cast(np.asarray(oloc).reshape(-1), "calc", host=True)

    mean = float(np.real(np.dot(weight, oloc)))
    resid = oloc - mean

    return {
        str(name): mean,
        f"{name}_var": float(np.real(np.dot(weight, np.abs(resid) ** 2))),
    }


def update(grad: Any, updates: Any) -> dict[str, float]:
    """Return diagnostics for the final applied parameter update.

    `dE_lin = Re <grad, update>` is the first-order energy change.
    """
    dE_lin = 0.0
    norm2 = 0.0

    for grad_leaf, update_leaf in zip(
        jax.tree.leaves(grad),
        jax.tree.leaves(updates),
        strict=True,
    ):
        g = np.asarray(jax.device_get(grad_leaf)).reshape(-1)
        u = np.asarray(jax.device_get(update_leaf)).reshape(-1)

        dE_lin += float(np.real(np.vdot(g, u)))
        norm2 += float(np.real(np.vdot(u, u)))

    return {
        "dE_lin": dE_lin,
        "update_norm": float(np.sqrt(max(norm2, 0.0))),
    }
