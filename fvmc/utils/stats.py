from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import precision


def weight(
    mass: NDArray[Any],
    mass2: NDArray[Any],
    logabs: NDArray[Any],
    log_induced: NDArray[Any],
) -> tuple[NDArray[Any], dict[str, float]]:
    """Construct normalized importance weights.

    Unique-ket convention:

        omega_x = M_x exp(2 log|psi(x)| - log r(x)),
        w_x = omega_x / sum_y omega_y.
    """
    dtype = precision.real("calc", host=True)
    tiny = dtype(precision.tiny("calc"))

    mass = precision.cast(mass, "calc", "real", host=True).reshape(-1)
    mass2 = precision.cast(mass2, "calc", "real", host=True).reshape(-1)
    logabs = precision.cast(logabs, "calc", "real", host=True).reshape(-1)
    log_induced = precision.cast(
        log_induced,
        "calc",
        "real",
        host=True,
    ).reshape(-1)

    logw = dtype(2.0) * logabs - log_induced
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
    norm = np.sum(weighted)

    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("importance weights have zero total mass")

    weight = weighted / norm
    ess = norm * norm / max(np.sum(mass2 * unorm * unorm), tiny)
    unique_ess = 1.0 / max(np.sum(weight * weight), tiny)

    return weight, {
        "ess_frac": float(ess / max(1.0, np.sum(mass))),
        "unique_eff": float(unique_ess / max(1, weight.size)),
        "w_max": float(np.max(weight)) if weight.size else 0.0,
    }


def moments(
    weight: NDArray[Any],
    value: NDArray[Any],
) -> tuple[float, float]:
    """Return the real weighted mean and absolute variance."""
    mean = float(np.real(np.dot(weight, value)))
    resid = value - mean
    var = float(np.real(np.dot(weight, np.abs(resid) ** 2)))
    return mean, var
