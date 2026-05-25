from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..utils import precision
from .base import Model


class Backflow(Model):
    """Backflow Slater determinant ansatz."""

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    dtype: Any | None = None

    @nn.compact
    def __call__(self, dets: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.dtype("model", "real") if self.dtype is None else self.dtype

        batch = dets.shape[0]
        nword = dets.shape[2]
        n_elec = int(self.n_alpha) + int(self.n_beta)
        n_sorb = 2 * int(self.norb)

        shifts = jnp.arange(64, dtype=jnp.uint64)

        alpha = (dets[:, 0, :, None] >> shifts[None, None, :]) & jnp.uint64(1)
        beta = (dets[:, 1, :, None] >> shifts[None, None, :]) & jnp.uint64(1)

        alpha = alpha.reshape(batch, nword * 64)[:, : self.norb].astype(jnp.int32)
        beta = beta.reshape(batch, nword * 64)[:, : self.norb].astype(jnp.int32)

        token = alpha + 2 * beta
        occ = jnp.concatenate([alpha, beta], axis=-1)

        x = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            x = nn.relu(nn.Dense(width, dtype=dtype, param_dtype=dtype)(x))

        delta = nn.Dense(
            n_elec * n_sorb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(1.0e-3),
            bias_init=nn.initializers.normal(1.0e-3),
        )(x).reshape(batch, n_elec, n_sorb)

        if self.ref_mat is None:
            mo_coeff = jnp.eye(self.norb, dtype=dtype)
            
            ref = jnp.zeros((n_elec, n_sorb), dtype=dtype)
            ref = ref.at[:self.n_alpha, :self.norb].set(mo_coeff[:, :self.n_alpha].T)
            ref = ref.at[self.n_alpha:, self.norb:].set(mo_coeff[:, :self.n_beta].T)
        else:
            ref = self.ref_mat.astype(dtype)

        mat = ref[None] + delta

        occ_col = jnp.argsort(
            jnp.where(occ.astype(bool), jnp.arange(n_sorb), n_sorb),
            axis=-1,
            stable=True,
        )[:, :n_elec]

        occ_mat = mat[
            jnp.arange(batch)[:, None, None],
            jnp.arange(n_elec)[None, :, None],
            occ_col[:, None, :],
        ]

        sign, logabs = jnp.linalg.slogdet(occ_mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))
    
    
class Backflow1D(Model):
    norb: int
    n_alpha: int
    n_beta: int
    k: int = 3
    hidden: tuple[int, ...] = (64, )
    ref_mat: jax.Array | None = None
    dtype: Any | None = None

    @nn.compact
    def __call__(self, dets: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.dtype("model", "real") if self.dtype is None else self.dtype

        batch = dets.shape[0]
        nword = dets.shape[2]
        n_elec = int(self.n_alpha) + int(self.n_beta)
        n_sorb = 2 * int(self.norb)

        shifts = jnp.arange(64, dtype=jnp.uint64)
        alpha = (dets[:, 0, :, None] >> shifts[None, None, :]) & jnp.uint64(1)
        beta  = (dets[:, 1, :, None] >> shifts[None, None, :]) & jnp.uint64(1)

        alpha = alpha.reshape(batch, nword * 64)[:, : self.norb].astype(jnp.int32)
        beta  = beta.reshape(batch, nword * 64)[:, : self.norb].astype(jnp.int32)

        token = alpha + 2 * beta
        x = jax.nn.one_hot(token, 4, dtype=dtype)  # (batch, norb, 4)

        for i, width in enumerate(self.hidden):
            k_size = (2 * self.k + 1,) if i == 0 else (1,)
            x = nn.relu(
                nn.Conv(
                    features=width,
                    kernel_size=k_size,
                    padding="SAME",
                    dtype=dtype,
                    param_dtype=dtype,
                )(x)
            )

        # Locally read out to (batch, norb, 2 * n_elec) using a 1x1 convolution
        delta_local = nn.Conv(
            features=2 * n_elec,
            kernel_size=(1,),
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(1.0e-3),
            bias_init=nn.initializers.normal(1.0e-3),
        )(x)

        # Reshape to (batch, n_elec, n_sorb) without losing spatial locality
        delta_alpha = delta_local[:, :, :n_elec]
        delta_beta  = delta_local[:, :, n_elec:]
        delta_alpha = jnp.transpose(delta_alpha, (0, 2, 1))
        delta_beta  = jnp.transpose(delta_beta, (0, 2, 1))
        delta = jnp.concatenate([delta_alpha, delta_beta], axis=-1)

        if self.ref_mat is None:
            mo_coeff = jnp.eye(self.norb, dtype=dtype)
            ref = jnp.zeros((n_elec, n_sorb), dtype=dtype)
            ref = ref.at[: self.n_alpha, : self.norb].set(mo_coeff[:, : self.n_alpha].T)
            ref = ref.at[self.n_alpha :, self.norb :].set(mo_coeff[:, : self.n_beta].T)
        else:
            ref = self.ref_mat.astype(dtype)

        mat = ref[None] + delta
        occ = jnp.concatenate([alpha, beta], axis=-1)
        occ_col = jnp.argsort(
            jnp.where(occ.astype(bool), jnp.arange(n_sorb), n_sorb),
            axis=-1,
            stable=True,
        )[:, :n_elec]

        occ_mat = mat[
            jnp.arange(batch)[:, None, None],
            jnp.arange(n_elec)[None, :, None],
            occ_col[:, None, :],
        ]

        sign, logabs = jnp.linalg.slogdet(occ_mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))