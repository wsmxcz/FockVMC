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
    max_iter: int = 64,
) -> jax.Array:
    """Solve `(A + shift I) x = rhs` by conjugate gradients."""
    rhs = jnp.asarray(rhs)
    real_dtype = jnp.real(jnp.zeros((), dtype=rhs.dtype)).dtype
    shift = jnp.asarray(shift, dtype=real_dtype)
    tol = 1.0e-4 if real_dtype == jnp.float32 else 1.0e-8

    def system(x: jax.Array) -> jax.Array:
        return matvec(x) + shift.astype(x.dtype) * x

    sol, _ = jsp_sparse.cg(
        system,
        rhs,
        x0=x0,
        tol=tol,
        atol=0.0,
        maxiter=max_iter,
    )
    return sol
