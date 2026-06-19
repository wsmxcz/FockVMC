from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..utils import precision
from .base import Model


# Reference convention
# --------------------
# ref_mat is always a Slater reference matrix with shape
#
#     (n_alpha + n_beta, 2 * norb).
#
# Columns are ordered as
#
#     [alpha spatial orbitals | beta spatial orbitals].
#
# The alpha block ref_mat[:n_alpha, :norb] and beta block
# ref_mat[n_alpha:, norb:] contain occupied spatial orbital
# coefficients in the same one-particle basis as the Hamiltonian.


class Backflow(Model):
    """Dense spin-orbital neural backflow Slater ansatz.

    This represents the general neural network backflow (NNBF)
    form where the network predicts determinant-dependent spin orbitals:

        M(x) = M0 + Delta(x)

    and evaluates the occupied spin-orbital determinant:

        psi(x) = det M(x)[occ_rows, occ_cols]
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.dtype("model", "real") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)
        norb = int(self.norb)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        shifts = jnp.arange(64, dtype=jnp.uint64)

        alpha = (x[:, 0, :, None] >> shifts[None, None, :]) & jnp.uint64(1)
        beta = (x[:, 1, :, None] >> shifts[None, None, :]) & jnp.uint64(1)

        alpha = alpha.reshape(batch, nword * 64)[:, :norb].astype(jnp.int32)
        beta = beta.reshape(batch, nword * 64)[:, :norb].astype(jnp.int32)

        # Spin-orbital token: empty, alpha, beta, double.
        token = alpha + 2 * beta
        x = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            x = nn.Dense(width, dtype=dtype, param_dtype=dtype)(x)
            x = nn.silu(x)

        delta = nn.Dense(
            n_elec * n_sorb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(1.0e-6),
            bias_init=nn.initializers.zeros,
        )(x)

        delta = delta.reshape(batch, n_elec, n_sorb)

        if self.ref_mat is None:
            ref = jnp.zeros((n_elec, n_sorb), dtype=dtype)

            ref = ref.at[:n_alpha, :norb].set(
                jnp.eye(norb, n_alpha, dtype=dtype).T
            )
            ref = ref.at[n_alpha:, norb:].set(
                jnp.eye(norb, n_beta, dtype=dtype).T
            )
        else:
            ref = jnp.asarray(self.ref_mat, dtype=dtype)

        mat = ref[None] + delta

        occ = jnp.concatenate([alpha, beta], axis=-1)
        col = jnp.argsort(
            jnp.where(occ.astype(bool), jnp.arange(n_sorb), n_sorb),
            axis=-1,
            stable=True,
        )[:, :n_elec]

        submat = mat[
            jnp.arange(batch)[:, None, None],
            jnp.arange(n_elec)[None, :, None],
            col[:, None, :],
        ]

        sign, logabs = jnp.linalg.slogdet(submat)
        return sign.reshape((batch,)), logabs.reshape((batch,))


class RBackflow(Model):
    """Restricted paired spatial-orbital neural backflow ansatz.

    This represents a singlet-pair or neural antisymmetric geminal power (AGP)
    ansatz. The wavefunction is evaluated as:

        psi(A, B) = det F(s)[A, B]

    where A and B are the indices of the occupied alpha and beta spatial
    orbitals, respectively.

    The symmetric pairing matrix F(s) is parameterized as:

        F(s) = F0 + Delta(s)
        F(s)^T = F(s)

    Requirements:
        Must satisfy n_alpha == n_beta == n_pair.

    Notes:
        - The network takes spatial orbital occupations as inputs:
          s_p = n_{p, alpha} + n_{p, beta} in the set {0, 1, 2}.
        - The symmetric pairing matrix enforces alpha/beta spin-flip invariance:
          psi(A, B) = psi(B, A), but does not implement full S^2 spin adaptation.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.dtype("model", "real") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)
        norb = int(self.norb)

        if n_alpha != n_beta:
            raise ValueError("RBackflow requires n_alpha == n_beta")

        n_pair = n_alpha
        shifts = jnp.arange(64, dtype=jnp.uint64)

        alpha = (x[:, 0, :, None] >> shifts[None, None, :]) & jnp.uint64(1)
        beta = (x[:, 1, :, None] >> shifts[None, None, :]) & jnp.uint64(1)

        alpha = alpha.reshape(batch, nword * 64)[:, :norb].astype(jnp.int32)
        beta = beta.reshape(batch, nword * 64)[:, :norb].astype(jnp.int32)

        # Spatial token: empty, single, double. No alpha/beta label is exposed.
        spatial = alpha + beta
        x = jax.nn.one_hot(spatial, 3, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            x = nn.Dense(width, dtype=dtype, param_dtype=dtype)(x)
            x = nn.silu(x)

        delta = nn.Dense(
            norb * norb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(1.0e-6),
            bias_init=nn.initializers.zeros,
        )(x)

        delta = delta.reshape(batch, norb, norb)

        # Symmetric pair matrix gives spin-flip invariance.
        delta = 0.5 * (delta + jnp.swapaxes(delta, -1, -2))

        if self.ref_mat is None:
            coeff = jnp.eye(norb, n_pair, dtype=dtype)
        else:
            ref = jnp.asarray(self.ref_mat, dtype=dtype)
            alpha_ref = ref[:n_pair, :norb].T
            beta_ref = ref[n_pair:, norb:].T
            coeff = 0.5 * (alpha_ref + beta_ref)

        ref = coeff @ coeff.T
        pair = ref[None] + delta

        alpha_col = jnp.argsort(
            jnp.where(alpha.astype(bool), jnp.arange(norb), norb),
            axis=-1,
            stable=True,
        )[:, :n_pair]

        beta_col = jnp.argsort(
            jnp.where(beta.astype(bool), jnp.arange(norb), norb),
            axis=-1,
            stable=True,
        )[:, :n_pair]

        mat = pair[
            jnp.arange(batch)[:, None, None],
            alpha_col[:, :, None],
            beta_col[:, None, :],
        ]

        sign, logabs = jnp.linalg.slogdet(mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))
