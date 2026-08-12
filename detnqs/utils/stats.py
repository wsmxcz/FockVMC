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

    mass = precision.cast(np.asarray(mass).reshape(-1), "calc", "real", host=True)
    mass2 = precision.cast(np.asarray(mass2).reshape(-1), "calc", "real", host=True)
    logabs = precision.cast(np.asarray(logabs).reshape(-1), "calc", "real", host=True)
    log_induced = precision.cast(
        np.asarray(log_induced).reshape(-1),
        "calc",
        "real",
        host=True,
    )

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
    norm = float(np.sum(weighted))

    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("importance weights have zero total mass")

    weight = precision.cast(weighted / norm, "calc", "real", host=True)

    ess = float(norm * norm / max(float(np.sum(mass2 * unorm * unorm)), float(tiny)))
    essu = float(1.0 / max(float(np.sum(weight * weight)), float(tiny)))

    return weight, {
        "ess_frac": ess / max(1.0, float(np.sum(mass))),
        "essu_frac": essu / max(1.0, float(weight.shape[0])),
        "w_max": float(np.max(weight)) if weight.size else 0.0,
    }


def eloc(
    weight: NDArray[Any],
    eloc: NDArray[Any],
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

    return energy, {
        "energy": energy,
        "eloc_var": float(np.real(np.dot(weight, np.abs(resid) ** 2))),
    }


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
