from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsp_sparse


def solve_dense(
    matrix: jax.Array,
    rhs: jax.Array,
    shift: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve a dense Hermitian PSD SR system.

    The regularized system is

        (A + shift I) x = rhs.

    For shift > 0 this is damped SR. For shift = 0, zero modes are removed
    by a spectral cutoff.
    """
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
    active = (shift > 0.0) | (eig > cutoff)

    denom = jnp.maximum(eig + shift, tiny)
    inv = jnp.where(active, 1.0 / denom, 0.0)

    sol = vec @ (inv.astype(coeff.dtype) * coeff)

    filt = (eig + shift) * inv
    residual = jnp.linalg.norm((filt.astype(coeff.dtype) - 1.0) * coeff)
    residual = residual / jnp.maximum(jnp.linalg.norm(rhs), tiny)

    eig_min = jnp.min(jnp.where(eig > cutoff, eig, jnp.inf))
    eig_min = jnp.where(jnp.isfinite(eig_min), eig_min, 0.0)

    cond = (eig_max + shift) / jnp.maximum(eig_min + shift, tiny)

    return sol, {
        "shift": shift,
        "residual": residual.astype(real_dtype),
        "cond": cond.astype(real_dtype),
    }


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

    def system(x: jax.Array) -> jax.Array:
        return matvec(x) + shift.astype(x.dtype) * x

    if diag is None:
        precond = None
    else:
        diag = jnp.asarray(diag, dtype=real_dtype)
        denom = jnp.maximum(diag + shift, eps)

        def precond(x: jax.Array) -> jax.Array:
            return x / denom.astype(x.dtype)

    sol, _ = jsp_sparse.cg(
        system,
        rhs,
        x0=x0,
        M=precond,
        tol=tol,
        atol=0.0,
        maxiter=int(maxiter),
    )

    residual = jnp.linalg.norm(system(sol) - rhs)
    residual = residual / jnp.maximum(jnp.linalg.norm(rhs), tiny)

    return sol, {
        "shift": shift,
        "residual": residual.astype(real_dtype),
    }