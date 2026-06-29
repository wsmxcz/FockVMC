from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg as jsp_sparse


def solve_dense(
    matrix: jax.Array,
    rhs: jax.Array,
    shift: jax.Array,
) -> jax.Array:
    """Solve `(A + shift I) x = rhs` for a dense Hermitian SR matrix."""
    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)
    real_dtype = jnp.real(jnp.zeros((), dtype=matrix.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)

    eye = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    system = 0.5 * (matrix + matrix.conj().T) + shift.astype(matrix.dtype) * eye
    return jnp.linalg.solve(system, rhs)


def solve_matvec(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    shift: jax.Array,
    *,
    x0: jax.Array | None = None,
    diag: jax.Array | None = None,
    maxiter: int = 64,
) -> jax.Array:
    """Solve `(A + shift I) x = rhs` by Jacobi-preconditioned CG."""
    rhs = jnp.asarray(rhs)
    real_dtype = jnp.real(jnp.zeros((), dtype=rhs.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)
    eps = jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype)
    tol = 1.0e-4 if real_dtype == jnp.float32 else 1.0e-8

    def system(x: jax.Array) -> jax.Array:
        return matvec(x) + shift.astype(x.dtype) * x

    precond = None
    if diag is not None:
        denom = jnp.maximum(jnp.asarray(diag, dtype=real_dtype) + shift, eps)

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
    return sol
