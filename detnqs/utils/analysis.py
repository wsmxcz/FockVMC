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
    """Return posterior scalar estimates.

    The posterior state is evaluated with an optional larger sample count.
    If `n_blocks > 1`, the total sample count is split into independent
    self-normalized estimates and `energy_se` is estimated from block energies.
    """
    eval_state = state
    if n_samples is not None and hasattr(state, "sampler"):
        sampler = replace(state.sampler, n_samples=int(n_samples))
        eval_state = state.replace(sampler=sampler)

    n_blocks = int(n_blocks)
    if n_blocks <= 1:
        _, out = eval_state.expect(obs=obs, profile=profile)
        return out

    if hasattr(eval_state, "sampler"):
        sampler = replace(
            eval_state.sampler,
            n_samples=max(1, int(eval_state.sampler.n_samples) // n_blocks),
        )
        eval_state = eval_state.replace(sampler=sampler)

    blocks: list[dict[str, float]] = []
    for _ in range(n_blocks):
        eval_state, out = eval_state.expect(obs=obs, profile=profile)
        blocks.append(out)

    out = {}
    for key in sorted(set().union(*(block.keys() for block in blocks))):
        if key == "energy":
            continue
        values: list[float] = []
        for block in blocks:
            if key not in block:
                continue
            arr = np.asarray(block[key])
            if arr.ndim == 0 and not np.iscomplexobj(arr):
                values.append(float(arr))
        if not values:
            continue
        if key == "n_forward" or key.startswith("time_"):
            out[key] = float(np.sum(values))
        else:
            out[key] = float(np.mean(values))

    energy = np.asarray([block["energy"] for block in blocks], dtype=np.float64)
    out["energy"] = float(np.mean(energy))
    out["energy_se"] = float(np.std(energy, ddof=1) / np.sqrt(n_blocks))
    return out


def support(
    w: NDArray[Any],
    *,
    top: Sequence[int] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Return Born-support diagnostics from normalized weights.

    `n2 = 1 / sum_x w_x^2` is the inverse participation number and
    `top{k}` is the cumulative mass of the largest `k` weights.
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
        kk = max(1, min(int(k), w_sort.size))
        out[f"top{int(k)}"] = float(np.sum(w_sort[:kk]))
    return out


def excitation(
    x: NDArray[Any],
    w: NDArray[Any],
    ref: NDArray[Any],
) -> dict[str, float]:
    """Return excitation-degree weights relative to one reference ket.

    The degree is `|x xor ref| / 2`, i.e. the number of orbital replacements.
    """
    x = np.asarray(x, dtype=np.uint64)
    ref = np.asarray(ref, dtype=np.uint64)
    if ref.ndim == x.ndim:
        ref = ref[0]

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if x.shape[0] != w.shape[0]:
        raise ValueError("x and w length mismatch")

    diff = np.bitwise_xor(x, ref[None])
    axes = tuple(range(1, diff.ndim))
    deg = (np.bitwise_count(diff).sum(axis=axes) // 2).astype(np.int64)

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
    """Return weighted residual quantiles and the largest eloc_var term."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    r = np.abs(np.asarray(eloc).reshape(-1) - energy)

    if w.shape[0] != r.shape[0]:
        raise ValueError("w and eloc length mismatch")

    order = np.argsort(r)
    r = r[order]
    w = w[order]

    norm = float(np.sum(w))
    if norm <= 0.0:
        raise ValueError("weights have zero mass")

    cdf = np.cumsum(w / norm)
    out: dict[str, float] = {}
    for qq in q:
        qv = float(qq)
        idx = min(np.searchsorted(cdf, qv, side="left"), r.size - 1)
        label = int(round((1000 if qv > 0.995 else 100) * qv))
        out[f"r_q{label}"] = float(r[idx])

    out["var_max"] = float(np.max(w * r * r)) if r.size else 0.0
    return out


def sr(geometry: Any, *, shift: float = 1.0e-3) -> dict[str, float]:
    """Analyze the dense fixed-shift sample-space SR system.

    The posterior system is `(K + shift I) a = b`,
    where `K = O O^dagger` and `O = sqrt(w) (J - <J>_w)`.
    This routine is dense and belongs to checkpoint analysis.
    """
    w = precision.cast(math.normalize(geometry.w), "model", "real")
    b = precision.cast(geometry.b, "model", "real")

    b_flat, _ = ravel_pytree(b)
    nsample = int(w.shape[0])
    nrow = int(b_flat.size)

    dtype = jnp.result_type(
        b_flat,
        *[leaf.dtype for leaf in jax.tree.leaves(geometry.theta)],
    )
    K = jnp.zeros((nrow, nrow), dtype=dtype)

    for block, put in tree.blocks(geometry.theta, batch.config["param_chunk"]):

        def coord_block(block_leaf: jax.Array, ket: Any) -> Any:
            ket = jax.tree.map(lambda z: z[None, ...], ket)
            val = geometry.coord(put(block_leaf), ket)
            return jax.tree.map(lambda z: jnp.asarray(z)[0], val)

        J = jax.vmap(jax.jacrev(coord_block), in_axes=(None, 0))(block, geometry.x)
        J = J.reshape((nsample, -1, block.size))

        mean = jnp.einsum("n,ncp->cp", w, J)
        O = (J - mean[None]) * jnp.sqrt(w)[:, None, None]
        O = O.reshape(nrow, block.size).astype(dtype)

        K = K + O @ O.conj().T

    K = np.asarray(jax.device_get(precision.cast(K, "sr")))
    b = np.asarray(jax.device_get(precision.cast(b_flat, "sr"))).reshape(-1)

    shift = float(shift)
    a = np.linalg.solve(K + shift * np.eye(K.shape[0], dtype=K.dtype), b)

    force = float(np.real(np.vdot(b, a)))
    damp = float(shift * np.real(np.vdot(a, a)) / max(force, precision.tiny("sr")))

    tr = float(np.real(np.trace(K)))
    tr2 = float(np.real(np.vdot(K, K)))

    return {
        "tr_qgt": tr,
        "rank_eff": float(tr * tr / max(tr2, precision.tiny("sr"))),
        "sr_force": force,
        "sr_damp": damp,
    }
