from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse


def solve_dense(
    matrix: jax.Array,
    rhs: jax.Array,
    shift: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve a Hermitian PSD ridge system by spectral filtering."""
    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)

    matrix = 0.5 * (matrix + matrix.conj().T)

    real_dtype = jnp.real(jnp.zeros((), dtype=matrix.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)

    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    tiny = jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype)

    eig, vec = jnp.linalg.eigh(matrix)
    eig = jnp.maximum(eig.astype(real_dtype), 0.0)

    coeff = vec.conj().T @ rhs

    eig_max = jnp.max(eig)
    cutoff = eps * jnp.maximum(eig_max, 1.0)
    positive = eig > cutoff

    # For shift > 0 this is ridge filtering.
    # For shift = 0 this becomes a truncated minimum-norm solve.
    denom = jnp.maximum(eig + shift, tiny)
    active = (shift > 0.0) | positive
    inv = jnp.where(active, 1.0 / denom, 0.0)

    sol = vec @ (inv.astype(coeff.dtype) * coeff)

    filt = eig * inv
    trace = jnp.sum(eig)
    eig_min = jnp.min(jnp.where(positive, eig, jnp.inf))
    eig_min = jnp.where(jnp.isfinite(eig_min), eig_min, 0.0)

    residual_coeff = ((eig + shift) * inv - 1.0).astype(coeff.dtype) * coeff
    residual = jnp.linalg.norm(residual_coeff) / jnp.maximum(
        jnp.linalg.norm(rhs),
        tiny,
    )

    rank_eff = trace * trace / jnp.maximum(jnp.sum(eig * eig), tiny)
    cond = (eig_max + shift) / jnp.maximum(eig_min + shift, tiny)

    info = {
        "shift": shift,
        "residual": residual.astype(real_dtype),
        "eig_min": eig_min,
        "eig_max": eig_max,
        "trace": trace,
        "rank_eff": rank_eff,
        "dof": jnp.sum(filt),
        "cond": cond,
    }
    return sol, info


def solve_matvec(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    shift: jax.Array,
    *,
    x0: jax.Array | None = None,
    diag: jax.Array | None = None,
    maxiter: int = 64,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve (A + shift I) x = rhs by warm-started Jacobi-CG."""
    rhs = jnp.asarray(rhs)

    real_dtype = jnp.real(jnp.zeros((), dtype=rhs.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)

    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    tiny = jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=real_dtype)

    tol = 1.0e-4 if real_dtype == jnp.float32 else 1.0e-8

    def system(v: jax.Array) -> jax.Array:
        return matvec(v) + shift.astype(v.dtype) * v

    if diag is None:
        precond = None
    else:
        diag = jnp.asarray(diag, dtype=real_dtype)
        denom = jnp.maximum(diag + shift, eps)

        def precond(v: jax.Array) -> jax.Array:
            return v / denom.astype(v.dtype)

    sol, _ = jsp_sparse.cg(
        system,
        rhs,
        x0=x0,
        M=precond,
        tol=tol,
        atol=0.0,
        maxiter=int(maxiter),
    )

    rhs_norm = jnp.linalg.norm(rhs)
    residual = jnp.linalg.norm(system(sol) - rhs) / jnp.maximum(rhs_norm, tiny)

    return sol, {
        "shift": shift,
        "residual": residual.astype(real_dtype),
    }