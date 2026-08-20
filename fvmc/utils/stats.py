from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import rankdata

from . import precision


def weight(
    count: NDArray[Any],
    logabs: NDArray[Any],
    log_r: NDArray[Any],
) -> tuple[NDArray[Any], dict[str, float]]:
    """Construct normalized importance weights.

    Unique-ket convention:

        omega_x = N_x exp(2 log|psi(x)| - log r(x)),
        w_x = omega_x / sum_y omega_y.
    """
    dtype = precision.real("calc", host=True)
    tiny = dtype(precision.tiny("calc"))

    count = precision.cast(count, "calc", "real", host=True).reshape(-1)
    logabs = precision.cast(logabs, "calc", "real", host=True).reshape(-1)
    log_r = precision.cast(log_r, "calc", "real", host=True).reshape(-1)

    logw = dtype(2.0) * logabs - log_r
    valid = (
        np.isfinite(logw)
        & np.isfinite(count)
        & (count > 0.0)
    )

    if not valid.any():
        raise FloatingPointError("all importance weights are non-finite")

    scale = dtype(np.max(logw[valid]))
    unorm = np.zeros_like(logw, dtype=dtype)
    unorm[valid] = np.exp(logw[valid] - scale)

    weighted = count * unorm
    norm = np.sum(weighted)

    if norm <= 0.0 or not np.isfinite(norm):
        raise FloatingPointError("importance weights have zero total mass")

    weight = weighted / norm
    ess = norm * norm / max(np.sum(count * unorm * unorm), tiny)

    return weight, {
        "ess_frac": float(ess / max(1.0, np.sum(count))),
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


def autocorr(value: NDArray[Any]) -> NDArray[np.float64]:
    """Return the mean autocorrelation of equal-length chains."""
    n_step = value.shape[1]
    n_fft = 1 << (2 * n_step - 1).bit_length()
    resid = value - np.mean(value, axis=1, keepdims=True)
    freq = np.fft.rfft(resid, n=n_fft, axis=1)
    cov = np.fft.irfft(freq * freq.conj(), n=n_fft, axis=1)
    cov = np.mean(cov[:, :n_step], axis=0)
    cov /= np.arange(n_step, 0, -1)
    return cov / cov[0]


def int_time(acf: NDArray[Any]) -> float:
    """Estimate integrated autocorrelation time with Geyer's sequence."""
    size = acf.size // 2
    pair = acf[: 2 * size].reshape(size, 2).sum(axis=1)
    stop = np.flatnonzero(pair <= 0.0)
    if stop.size:
        pair = pair[: stop[0]]
    pair = np.minimum.accumulate(pair)
    return float(2.0 * np.sum(pair) - 1.0)


def rhat(value: NDArray[Any]) -> float:
    """Return rank-normalized folded split-Rhat for multiple chains."""
    half = value.shape[1] // 2
    value = value[:, : 2 * half]
    split = np.concatenate((value[:, :half], value[:, half:]), axis=0)
    folded = np.abs(split - np.median(split))
    result = []

    for draw in (split, folded):
        rank = rankdata(draw, method="average").reshape(draw.shape)
        size = draw.size
        score = ndtri((rank - 0.375) / (size + 0.25))
        mean = np.mean(score, axis=1)
        within = np.mean(np.var(score, axis=1, ddof=1))
        between = half * np.var(mean, ddof=1)
        var = (half - 1.0) * within / half + between / half
        result.append(np.sqrt(var / within))

    return float(max(result))
