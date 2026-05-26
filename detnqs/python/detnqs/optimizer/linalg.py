from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import jax.scipy.sparse.linalg as jsp_sparse


def solve_dense(
    matrix: jax.Array,
    rhs: jax.Array,
    shift: jax.Array,
    *,
    fallback: bool = False,
) -> jax.Array:
    """Solve (A + shift I) x = rhs for Hermitian positive semidefinite A."""
    matrix = jnp.asarray(matrix)
    rhs = jnp.asarray(rhs)

    matrix = 0.5 * (matrix + matrix.conj().T)
    real_dtype = jnp.real(jnp.zeros((), dtype=matrix.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)

    system = matrix + shift.astype(matrix.dtype) * jnp.eye(
        matrix.shape[0],
        dtype=matrix.dtype,
    )

    chol = jnp.linalg.cholesky(system)
    y = jsp_linalg.solve_triangular(chol, rhs, lower=True)
    x_chol = jsp_linalg.solve_triangular(chol.conj().T, y, lower=False)

    if not fallback:
        return x_chol

    ok = jnp.all(jnp.isfinite(chol)) & jnp.all(jnp.isfinite(x_chol))

    def pinv_solve(args):
        mat, vec = args
        eigval, eigvec = jnp.linalg.eigh(mat)

        scale = jnp.max(jnp.abs(eigval))
        eps = jnp.asarray(jnp.finfo(eigval.dtype).eps, dtype=eigval.dtype)
        tiny = jnp.asarray(jnp.finfo(eigval.dtype).tiny, dtype=eigval.dtype)
        cutoff = jnp.maximum(100.0 * eps * scale, tiny)

        inv = jnp.where(eigval > cutoff, 1.0 / eigval, 0.0)
        return eigvec @ (inv * (eigvec.conj().T @ vec))

    return jax.lax.cond(ok, lambda _: x_chol, pinv_solve, (system, rhs))


def solve_matvec(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    maxiter: int = 64,
) -> jax.Array:
    """Solve A x = rhs by conjugate gradient."""
    rhs = jnp.asarray(rhs)

    real_dtype = jnp.real(jnp.zeros((), dtype=rhs.dtype)).dtype
    tol = 1.0e-4 if real_dtype == jnp.float32 else 1.0e-8

    sol, _ = jsp_sparse.cg(
        matvec,
        rhs,
        tol=tol,
        atol=0.0,
        maxiter=int(maxiter),
    )
    return sol