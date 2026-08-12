from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..utils import precision
from .base import Model


class RBM(Model):
    """Complex RBM on spin-orbital occupation bitstrings."""

    norb: int
    alpha: int = 1
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = precision.complex("model") if self.dtype is None else self.dtype

        n_sorb = 2 * self.norb
        n_hidden = self.alpha * n_sorb

        orb = jnp.arange(self.norb, dtype=jnp.uint32)
        word = (orb >> jnp.uint32(6)).astype(jnp.int32)
        shift = (orb & jnp.uint32(63)).astype(jnp.uint64)

        alpha_words = jnp.take(x[:, 0], word, axis=1)
        beta_words = jnp.take(x[:, 1], word, axis=1)

        alpha_occ = (alpha_words >> shift) & jnp.uint64(1)
        beta_occ = (beta_words >> shift) & jnp.uint64(1)

        occ = jnp.concatenate([alpha_occ, beta_occ], axis=-1)
        x = 2 * occ.astype(dtype) - 1

        init_std = 1.0e-2 / math.sqrt(n_sorb)

        kernel = self.param(
            "kernel",
            nn.initializers.normal(init_std),
            (n_sorb, n_hidden),
            dtype,
        )
        hidden_bias = self.param(
            "hidden_bias",
            nn.initializers.zeros,
            (n_hidden,),
            dtype,
        )
        visible_bias = self.param(
            "visible_bias",
            nn.initializers.zeros,
            (n_sorb,),
            dtype,
        )

        z = x @ kernel + hidden_bias

        # Stable complex log(cosh(z)).
        s = jnp.where(jnp.real(z) >= 0, z, -z)
        hidden = jnp.sum(
            s + jnp.log1p(jnp.exp(-2 * s)) - math.log(2.0),
            axis=-1,
        )
        visible = x @ visible_bias

        return (visible + hidden).reshape((x.shape[0],))
