from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpy.typing import NDArray

from . import batch, math, precision, tree


def estimate(
    state: Any,
    *,
    n_samples: int | None = None,
    n_blocks: int = 1,
    obs: Mapping[str, Any] | None = None,
    profile: bool = False,
) -> dict[str, float]:
    """Return scalar estimates at fixed wavefunction and sampling law.

    If `n_blocks > 1`, the current sample count is split into block estimates
    and `energy_se` is estimated from block energies.
    """
    if hasattr(state, "sampler"):
        sampler = state.sampler
        if getattr(sampler, "alpha", 1.0) is None and hasattr(state, "sampler_state"):
            sampler = replace(sampler, alpha=float(state.sampler_state.alpha))
        if n_samples is not None:
            sampler = replace(sampler, n_samples=int(n_samples))
        state = state.replace(sampler=sampler)

    n_blocks = int(n_blocks)
    if n_blocks <= 1:
        _, out = (
            state.expect(profile=profile)
            if obs is None
            else state.expect(obs=obs, profile=profile)
        )
        return out

    if hasattr(state, "sampler"):
        state = state.replace(
            sampler=replace(
                state.sampler,
                n_samples=max(1, int(state.sampler.n_samples) // n_blocks),
            ),
        )

    blocks = []
    for _ in range(n_blocks):
        state, out = (
            state.expect(profile=profile)
            if obs is None
            else state.expect(obs=obs, profile=profile)
        )
        blocks.append(out)

    out: dict[str, float] = {}
    for key in sorted(set().union(*(block.keys() for block in blocks))):
        if key == "energy":
            continue

        values = []
        for block in blocks:
            if key not in block:
                continue
            value = np.asarray(block[key])
            if value.ndim == 0 and not np.iscomplexobj(value):
                values.append(float(value))

        if not values:
            continue

        out[key] = float(
            np.sum(values)
            if key == "n_forward" or key.startswith("time_")
            else np.mean(values)
        )

    energy = np.asarray([block["energy"] for block in blocks], dtype=np.float64)
    out["energy"] = float(np.mean(energy))
    out["energy_se"] = float(np.std(energy, ddof=1) / np.sqrt(n_blocks))

    return out


def support(
    w: NDArray[Any],
    *,
    top: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Return Born-support diagnostics.

    `n2 = 1 / sum_x w_x^2` is the inverse participation number.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w) & (w > 0.0)]

    norm = float(np.sum(w))
    if norm <= 0.0:
        raise ValueError("weights have zero mass")

    w = w / norm
    w_sort = np.sort(w)[::-1]

    out = {
        "n2": float(1.0 / np.sum(w * w)),
        "entropy": float(-np.sum(w * np.log(w))),
    }

    for k in top:
        k = int(k)
        n = max(1, min(k, w_sort.size))
        out[f"top{k}"] = float(np.sum(w_sort[:n]))

    return out


def excitation(
    x: NDArray[Any],
    w: NDArray[Any],
    ref: NDArray[Any],
) -> dict[str, float]:
    """Return excitation-degree weights relative to one reference ket.

    The degree is half the Hamming distance between occupation bitstrings.
    """
    x = np.asarray(x, dtype=np.uint64)
    ref = np.asarray(ref, dtype=np.uint64)

    if ref.ndim == x.ndim:
        ref = ref[0]

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.shape[0] != w.shape[0]:
        raise ValueError("x and w length mismatch")

    diff = np.bitwise_xor(x, ref[None])
    deg = np.bitwise_count(diff).reshape(x.shape[0], -1).sum(axis=1) // 2

    return {
        f"exc{int(d)}": float(np.sum(w[deg == d]))
        for d in np.unique(deg)
    }


def tail(
    w: NDArray[Any],
    eloc: NDArray[Any],
    energy: float,
    *,
    q: Sequence[float] = (0.9, 0.99, 0.999),
) -> dict[str, float]:
    """Return weighted residual quantiles and largest variance contribution."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    r = np.abs(np.asarray(eloc).reshape(-1) - energy)

    if w.shape[0] != r.shape[0]:
        raise ValueError("w and eloc length mismatch")

    norm = float(np.sum(w))
    if norm <= 0.0:
        raise ValueError("weights have zero mass")

    order = np.argsort(r)
    r = r[order]
    w = w[order] / norm
    cdf = np.cumsum(w)

    out: dict[str, float] = {}
    for value in q:
        value = float(value)
        idx = min(np.searchsorted(cdf, value, side="left"), r.size - 1)
        label = int(round((1000 if value > 0.995 else 100) * value))
        out[f"r_q{label}"] = float(r[idx])

    out["var_max"] = float(np.max(w * r * r)) if r.size else 0.0
    return out


def sr(geometry: Any, *, shift: float = 1.0e-3) -> dict[str, float]:
    """Analyze the dense sample-space SR system.

    The system is `(K + shift I) a = b`, with
    `K = O O^dagger` and `O = sqrt(w) (J - <J>_w)`.
    """
    w = precision.cast(math.normalize(geometry.w), "model", "real")
    b = precision.cast(geometry.b, "model", "real")

    b_flat, _ = ravel_pytree(b)
    nrow = int(b_flat.size)
    nsample = int(w.shape[0])
    sqrt_w = jnp.sqrt(w)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(geometry.theta)],
    )
    K = jnp.zeros((nrow, nrow), dtype=dtype)

    for block, put in tree.blocks(geometry.theta, batch.config["param_chunk"]):

        def coord(block_leaf: jax.Array, ket: Any) -> Any:
            ket = jax.tree.map(lambda z: z[None, ...], ket)
            val = geometry.coord(put(block_leaf), ket)
            return jax.tree.map(lambda z: jnp.asarray(z)[0], val)

        J = jax.vmap(jax.jacrev(coord), in_axes=(None, 0))(block, geometry.x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * sqrt_w[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        K = K + O @ O.conj().T

    K = np.asarray(jax.device_get(precision.cast(K, "sr")))
    b = np.asarray(jax.device_get(precision.cast(b_flat, "sr"))).reshape(-1)

    shift = float(shift)
    a = np.linalg.solve(K + shift * np.eye(K.shape[0], dtype=K.dtype), b)

    force = float(np.real(np.vdot(b, a)))
    damp = float(
        shift * np.real(np.vdot(a, a)) / max(force, precision.tiny("sr")),
    )

    tr = float(np.real(np.trace(K)))
    tr2 = float(np.real(np.vdot(K, K)))

    return {
        "tr_qgt": tr,
        "rank_eff": float(tr * tr / max(tr2, precision.tiny("sr"))),
        "sr_force": force,
        "sr_damp": damp,
    }