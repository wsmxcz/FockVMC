from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..utils import precision
from .base import Model


class Backflow(Model):
    """
    Dense spin-orbital neural backflow Slater determinant.

    Fixed sector:
        N = n_alpha + n_beta,  K = 2 * norb.

    Input:
        t_p in {0, alpha, beta, double}, encoded as 4-state one-hot.

    Backflow:
        M(x) = M0 + dM(x).

    Amplitude:
        psi(C) = det M(x)[:, C].

    This is the standard electron-only NNBF baseline.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        if n_elec > n_sorb:
            raise ValueError("Backflow requires n_alpha + n_beta <= 2 * norb")

        if self.ref_mat is None:
            ref = np.zeros((n_elec, n_sorb), dtype=np.float64)
            col = np.concatenate((np.arange(n_alpha), norb + np.arange(n_beta)))
            ref[np.arange(n_elec), col] = 1.0
        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

        self.ref = np.ascontiguousarray(ref)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        # Unpack occupations.
        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        # Build 4-state input.
        token = alpha.astype(jnp.int32) + 2 * beta.astype(jnp.int32)
        h = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        # Apply backbone.
        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        # Build backflow matrix.
        out = nn.Dense(
            n_elec * n_sorb,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
            name="out",
        )(h)

        mat = out.reshape(batch, n_elec, n_sorb)
        mat = mat + jnp.asarray(self.ref, dtype=dtype)[None]

        # Select occupied columns.
        occ = jnp.concatenate((alpha, beta), axis=-1)
        sorb = jnp.arange(n_sorb, dtype=jnp.int32)

        score = jnp.where(occ, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_elec)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_elec, n_elec))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))


class GBackflow(Model):
    """
    Generalized spin-orbital backflow Slater determinant.

    Fixed sector:
        N = n_alpha + n_beta,  K = 2 * norb,  d = min(N, K - N).

    Input:
        t_p in {0, alpha, beta, double}, encoded as 4-state one-hot.
        In the hole representation, the input is particle-hole transformed.

    Backflow:
        M(x) = M0 + dM(x).

    Amplitude:
        electron representation:
            psi(C) = det M(x)[:, C].

        hole representation:
            psi(C) = chi_ph(C) det M(1 - x)[:, C^c].

    The head is fully spin-orbital resolved. S_z and SU(2) are not enforced.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        if n_elec > n_sorb:
            raise ValueError("GBackflow requires n_alpha + n_beta <= 2 * norb")

        use_hole = n_elec > norb
        n_det = n_sorb - n_elec if use_hole else n_elec

        if self.ref_mat is None:
            ref = np.zeros((n_det, n_sorb), dtype=np.float64)

            if use_hole:
                col = np.concatenate(
                    (np.arange(n_alpha, norb), norb + np.arange(n_beta, norb))
                )
            else:
                col = np.concatenate((np.arange(n_alpha), norb + np.arange(n_beta)))

            ref[np.arange(n_det), col] = 1.0

        else:
            ref_e = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if use_hole:
                q = np.linalg.qr(ref_e.T, mode="complete")[0]
                ref = q[:, n_elec:].T
            else:
                ref = ref_e

        self.ref = np.ascontiguousarray(ref)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        use_hole = n_elec > norb
        n_det = n_sorb - n_elec if use_hole else n_elec

        # Unpack occupations.
        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        # Build electron/hole input.
        net_alpha = ~alpha if use_hole else alpha
        net_beta = ~beta if use_hole else beta

        token = net_alpha.astype(jnp.int32) + 2 * net_beta.astype(jnp.int32)
        h = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        # Apply backbone.
        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        # Build backflow matrix.
        out = nn.Dense(
            n_det * n_sorb,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
            name="out",
        )(h)

        mat = out.reshape(batch, n_det, n_sorb)
        mat = mat + jnp.asarray(self.ref, dtype=dtype)[None]

        # Select electron/hole columns.
        occ = jnp.concatenate((alpha, beta), axis=-1)
        mask = ~occ if use_hole else occ

        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(mask, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_det)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_det, n_det))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)

        # Apply particle-hole phase.
        if use_hole:
            exponent = (
                jnp.sum(mask.astype(jnp.int32) * sorb[None, :], axis=-1)
                - n_det * (n_det - 1) // 2
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign.reshape((batch,)), logabs.reshape((batch,))


class SBackflow(Model):
    """
    SU(2)-adapted paired-backflow determinant.

    Fixed sector:
        n_alpha >= n_beta,  S = (n_alpha - n_beta) / 2,  M = S.

    Input:
        s_p = n_{p,alpha} + n_{p,beta} in {0, 1, 2}.

    Particle-hole normalized channels:
        z_p = s_p - 1       in electron representation,
        z_p = 1 - s_p       in hole representation,
        r_p = 1_{s_p = 1}.

    Backflow:
        (z, r) -> g(s),
        K(s) = K0 + dK(s),  K(s) = K(s)^T,
        U(s) = U0 + dU(s).

    Amplitude:
        psi(A, B) = det [K(s)[A, B]  U(s)[A]].

    High filling uses the particle-hole paired dual.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (256,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        if n_alpha < n_beta:
            raise ValueError("SBackflow requires n_alpha >= n_beta")

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb
        n_open = n_alpha - n_beta
        use_hole = n_elec > norb

        if self.ref_mat is None:
            pair_ref = np.zeros((norb, norb), dtype=np.float64)
            open_ref = np.zeros((norb, n_open), dtype=np.float64)

            p = np.arange(n_alpha, norb) if use_hole else np.arange(n_beta)
            pair_ref[p, p] = 1.0

            if n_open > 0:
                p = np.arange(n_beta, n_alpha)
                open_ref[p, np.arange(n_open)] = 1.0

        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            alpha_occ = ref[:n_alpha, :norb].T
            open_ref = alpha_occ[:, n_beta:n_alpha]

            if use_hole:
                q = np.linalg.qr(alpha_occ, mode="complete")[0]
                pair_orb = q[:, n_alpha:]
            else:
                pair_orb = alpha_occ[:, :n_beta]

            pair_ref = pair_orb @ pair_orb.T

        self.pair_ref = np.ascontiguousarray(pair_ref)
        self.open_ref = np.ascontiguousarray(open_ref)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_open = n_alpha - n_beta
        use_hole = (n_alpha + n_beta) > norb

        n_row = norb - n_beta if use_hole else n_alpha
        n_pair = norb - n_alpha if use_hole else n_beta

        # Unpack occupations.
        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        # Build spin-free input.
        af = alpha.astype(dtype)
        bf = beta.astype(dtype)

        z = (1.0 - af - bf) if use_hole else (af + bf - 1.0)
        r = jnp.logical_xor(alpha, beta).astype(dtype)
        h = jnp.concatenate((z, r), axis=-1)

        # Apply backbone.
        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        # Build pair/open heads.
        n_tri = norb * (norb + 1) // 2
        n_out = n_tri + norb * n_open

        out = nn.Dense(
            n_out,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
            name="out",
        )(h)

        d_tri = out[:, :n_tri]

        # Select electron/hole rows and columns.
        orb = jnp.arange(norb, dtype=jnp.int32)

        row_mask = ~beta if use_hole else alpha
        col_mask = ~alpha if use_hole else beta

        def select(mask: jax.Array, size: int) -> jax.Array:
            if size == 0:
                return jnp.zeros((batch, 0), dtype=jnp.int32)

            score = jnp.where(mask, norb - 1 - orb[None, :], -1)
            _, idx = jax.lax.top_k(score, size)
            return idx.astype(jnp.int32)

        row = select(row_mask, n_row)
        col = select(col_mask, n_pair)

        # Gather symmetric dK[row, col].
        p = row[:, :, None]
        q = col[:, None, :]

        i = jnp.minimum(p, q)
        j = jnp.maximum(p, q)

        tri = i * norb - i * (i + 1) // 2 + j
        bid3 = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
        pair_mat = d_tri[bid3, tri]

        # Add K0[row, col].
        pair_ref = jnp.asarray(self.pair_ref, dtype=dtype)
        pair_mat = pair_mat + pair_ref[row[:, :, None], col[:, None, :]]

        # Add U0[row].
        if n_open == 0:
            mat = pair_mat
        else:
            d_open = out[:, n_tri:].reshape(batch, norb, n_open)
            bid2 = jnp.arange(batch, dtype=jnp.int32)[:, None]
            open_mat = d_open[bid2, row]

            open_ref = jnp.asarray(self.open_ref, dtype=dtype)
            open_mat = open_mat + open_ref[row]

            mat = jnp.concatenate((pair_mat, open_mat), axis=-1)

        sign, logabs = jnp.linalg.slogdet(mat)

        # Apply particle-hole phase.
        if use_hole:
            exponent = (
                n_pair
                + n_row * norb
                + jnp.sum(row, axis=-1)
                + jnp.sum(col, axis=-1)
                - n_row * (n_row - 1) // 2
                - n_pair * (n_pair - 1) // 2
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign.reshape((batch,)), logabs.reshape((batch,))