from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from flax import linen as nn

from ..utils import precision
from .base import Model


class Backflow(Model):
    """Dense spin-orbital backflow determinant, psi(C) = det M[:, C]."""

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self) -> None:
        norb = self.norb
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        if not (0 <= n_alpha <= norb and 0 <= n_beta <= norb):
            raise ValueError("Backflow requires 0 <= n_alpha, n_beta <= norb")

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        self.n_elec = n_elec
        self.n_sorb = n_sorb

        if self.ref_mat is None:
            self.ref = None
        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            self.ref = ref.reshape(-1)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = self.norb
        n_elec = self.n_elec
        n_sorb = self.n_sorb

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        af = alpha.astype(dtype)
        bf = beta.astype(dtype)

        q = af + bf - 1.0
        mu = af - bf
        nu = jnp.logical_xor(alpha, beta).astype(dtype)
        h = jnp.concatenate((q, mu, nu), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                width,
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        if self.ref is None:
            bias_init = nn.initializers.normal(self.init_scale)
        else:
            ref = jnp.asarray(self.ref, dtype=dtype)
            bias_init = lambda _, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

        out = nn.Dense(
            n_elec * n_sorb,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=bias_init,
            name="out",
        )(h)

        mat = out.reshape(batch, n_elec, n_sorb)

        occ = jnp.concatenate((alpha, beta), axis=-1)
        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(occ, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_elec)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_elec, n_elec))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)
        return sign, logabs


class GBackflow(Model):
    """Electron/hole generalized backflow for n_alpha >= n_beta."""

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self) -> None:
        norb = self.norb
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        if not (0 <= n_beta <= n_alpha <= norb):
            raise ValueError("GBackflow requires 0 <= n_beta <= n_alpha <= norb")

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        use_hole = n_elec > norb

        if use_hole:
            d_alpha = norb - n_beta
            d_beta = norb - n_alpha
        else:
            d_alpha = n_alpha
            d_beta = n_beta

        n_det = d_alpha + d_beta

        self.n_sorb = n_sorb
        self.n_det = n_det
        self.use_hole = use_hole
        self.d_alpha = d_alpha
        self.d_beta = d_beta

        if self.ref_mat is None:
            self.ref = None
        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            if use_hole:
                basis = np.linalg.qr(ref.T, mode="complete")[0]
                ref = basis[:, n_elec:].T
                ref = np.concatenate((ref[:, norb:], ref[:, :norb]), axis=-1)

            self.ref = ref.reshape(-1)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = self.norb
        n_sorb = self.n_sorb
        n_det = self.n_det
        use_hole = self.use_hole
        d_alpha = self.d_alpha
        d_beta = self.d_beta

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        if use_hole:
            eff_alpha = jnp.logical_not(beta)
            eff_beta = jnp.logical_not(alpha)
        else:
            eff_alpha = alpha
            eff_beta = beta

        af = eff_alpha.astype(dtype)
        bf = eff_beta.astype(dtype)

        q = af + bf - 1.0
        mu = af - bf
        nu = jnp.logical_xor(eff_alpha, eff_beta).astype(dtype)
        h = jnp.concatenate((q, mu, nu), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                width,
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        if self.ref is None:
            bias_init = nn.initializers.normal(self.init_scale)
        else:
            ref = jnp.asarray(self.ref, dtype=dtype)
            bias_init = lambda _, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

        out = nn.Dense(
            n_det * n_sorb,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=bias_init,
            name="out",
        )(h)

        mat = out.reshape(batch, n_det, n_sorb)

        mask = jnp.concatenate((eff_alpha, eff_beta), axis=-1)

        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(mask, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_det)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_det, n_det))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)

        if use_hole:
            hole = jnp.concatenate(
                (jnp.logical_not(alpha), jnp.logical_not(beta)), axis=-1
            )
            exponent = (
                jnp.sum(hole.astype(jnp.int32) * sorb[None, :], axis=-1)
                - n_det * (n_det - 1) // 2
                + d_alpha * d_beta
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign, logabs


class SBackflow(Model):
    """Spin-adapted paired determinant det[K[A, B], U[A]], K = K.T."""

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self) -> None:
        norb = self.norb
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        if not (0 <= n_beta <= n_alpha <= norb):
            raise ValueError("SBackflow requires 0 <= n_beta <= n_alpha <= norb")

        n_elec = n_alpha + n_beta
        n_open = n_alpha - n_beta
        use_hole = n_elec > norb

        n_row = norb - n_beta if use_hole else n_alpha
        n_pair = norb - n_alpha if use_hole else n_beta
        n_out = norb * (norb + n_open)

        self.n_open = n_open
        self.n_row = n_row
        self.n_pair = n_pair
        self.n_out = n_out
        self.use_hole = use_hole

        if self.ref_mat is None:
            self.ref = None
        else:
            n_sorb = 2 * norb
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            alpha_occ = ref[:n_alpha, :norb].T
            open_ref = alpha_occ[:, n_beta:n_alpha]

            if use_hole:
                basis = np.linalg.qr(alpha_occ, mode="complete")[0]
                pair_orb = basis[:, n_alpha:]
            else:
                pair_orb = alpha_occ[:, :n_beta]

            pair_ref = pair_orb @ pair_orb.T
            ref = np.concatenate((pair_ref.reshape(-1), open_ref.reshape(-1)))

            self.ref = ref

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = self.norb
        n_open = self.n_open
        n_row = self.n_row
        n_pair = self.n_pair
        n_out = self.n_out
        use_hole = self.use_hole

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        if use_hole:
            eff_alpha = jnp.logical_not(beta)
            eff_beta = jnp.logical_not(alpha)
        else:
            eff_alpha = alpha
            eff_beta = beta

        af = eff_alpha.astype(dtype)
        bf = eff_beta.astype(dtype)

        q = af + bf - 1.0
        nu = jnp.logical_xor(eff_alpha, eff_beta).astype(dtype)
        h = jnp.concatenate((q, nu), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                width,
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        if self.ref is None:
            bias_init = nn.initializers.normal(self.init_scale)
        else:
            ref = jnp.asarray(self.ref, dtype=dtype)
            bias_init = lambda _, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

        out = nn.Dense(
            n_out,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=bias_init,
            name="out",
        )(h)

        k = out[:, : norb * norb].reshape(batch, norb, norb)
        k = 0.5 * (k + jnp.swapaxes(k, -1, -2))
        open_val = out[:, norb * norb :].reshape(batch, norb, n_open)

        orb = jnp.arange(norb, dtype=jnp.int32)
        row_mask = eff_alpha
        col_mask = eff_beta

        row_score = jnp.where(row_mask, norb - 1 - orb[None, :], -1)
        _, row = jax.lax.top_k(row_score, n_row)

        col_score = jnp.where(col_mask, norb - 1 - orb[None, :], -1)
        _, col = jax.lax.top_k(col_score, n_pair)

        bid = jnp.arange(batch, dtype=jnp.int32)
        pair_mat = k[bid[:, None, None], row[:, :, None], col[:, None, :]]
        open_mat = open_val[bid[:, None], row]
        mat = jnp.concatenate((pair_mat, open_mat), axis=-1)

        sign, logabs = jnp.linalg.slogdet(mat)

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

        return sign, logabs


class PBackflow(Model):
    """Spin-projected spatial backflow for n_alpha >= n_beta."""

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self) -> None:
        norb = self.norb
        n_alpha = self.n_alpha
        n_beta = self.n_beta

        if not (0 <= n_beta <= n_alpha <= norb):
            raise ValueError("PBackflow requires 0 <= n_beta <= n_alpha <= norb")

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb
        use_hole = n_elec > norb

        if use_hole:
            d_alpha = norb - n_beta
            d_beta = norb - n_alpha
        else:
            d_alpha = n_alpha
            d_beta = n_beta

        n_det = d_alpha + d_beta

        if n_det <= 0:
            raise ValueError("PBackflow requires a non-empty effective sector")

        self.d_alpha = d_alpha
        self.d_beta = d_beta
        self.n_det = n_det
        self.use_hole = use_hole

        spin2 = d_alpha - d_beta
        n_quad = (d_alpha + 2) // 2

        x, w = np.polynomial.legendre.leggauss(n_quad)
        t = 0.5 * (1.0 + x)

        self.quad_logw = (
            np.log(0.5 * (spin2 + 1) * w) + d_alpha * np.log(t)
        )
        self.quad_z = (1.0 - x) / (1.0 + x)

        if self.ref_mat is None:
            self.ref = None
        else:
            full = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if full.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            alpha_ref = full[:n_alpha, :norb]
            beta_ref = full[n_alpha:n_elec, norb:]

            if use_hole:
                qa = np.linalg.qr(alpha_ref.T, mode="complete")[0]
                qb = np.linalg.qr(beta_ref.T, mode="complete")[0]

                alpha_hole = qa[:, n_alpha:].T
                beta_hole = qb[:, n_beta:].T

                ref = np.concatenate((beta_hole, alpha_hole), axis=0)
            else:
                ref = np.concatenate((alpha_ref, beta_ref), axis=0)

            self.ref = ref.reshape(-1)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = self.norb
        n_det = self.n_det
        use_hole = self.use_hole
        d_alpha = self.d_alpha
        d_beta = self.d_beta

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        if use_hole:
            eff_alpha = jnp.logical_not(beta)
            eff_beta = jnp.logical_not(alpha)
        else:
            eff_alpha = alpha
            eff_beta = beta

        af = eff_alpha.astype(dtype)
        bf = eff_beta.astype(dtype)

        q = af + bf - 1.0
        nu = jnp.logical_xor(eff_alpha, eff_beta).astype(dtype)
        h = jnp.concatenate((q, nu), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                width,
                use_bias=True,
                dtype=dtype,
                param_dtype=dtype,
                name=f"hidden_{i}",
            )(h)
            h = nn.silu(h)

        if self.ref is None:
            bias_init = nn.initializers.normal(self.init_scale)
        else:
            ref = jnp.asarray(self.ref, dtype=dtype)
            bias_init = lambda _, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

        out = nn.Dense(
            n_det * norb,
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=bias_init,
            name="out",
        )(h)

        y = out.reshape(batch, n_det, norb)

        orb = jnp.arange(norb, dtype=jnp.int32)
        alpha_score = jnp.where(eff_alpha, norb - 1 - orb[None, :], -1)
        beta_score = jnp.where(eff_beta, norb - 1 - orb[None, :], -1)
        _, alpha_col = jax.lax.top_k(alpha_score, d_alpha)
        _, beta_col = jax.lax.top_k(beta_score, d_beta)

        col = jnp.concatenate((alpha_col, beta_col), axis=-1)
        mat = jnp.take_along_axis(y, col[:, None, :], axis=2)

        a = mat[:, :d_alpha, :d_alpha]
        b = mat[:, :d_alpha, d_alpha:]
        c = mat[:, d_alpha:, :d_alpha]
        d = mat[:, d_alpha:, d_alpha:]

        lu, piv = jsp.linalg.lu_factor(a)
        k = c @ jsp.linalg.lu_solve((lu, piv), b)

        diag = jnp.diagonal(lu, axis1=-2, axis2=-1)
        a_logabs = jnp.sum(jnp.log(jnp.abs(diag)), axis=-1)
        n_swap = jnp.sum(
            piv != jnp.arange(d_alpha, dtype=piv.dtype),
            axis=-1,
        )
        a_sign = (
            jnp.prod(jnp.sign(diag), axis=-1)
            * (1 - 2 * (n_swap & 1)).astype(dtype)
        )

        z = jnp.asarray(self.quad_z, dtype=dtype)
        quad = d[:, None, :, :] + z[None, :, None, None] * k[:, None, :, :]
        det_sign, det_logabs = jnp.linalg.slogdet(quad)

        log_weight = jnp.asarray(self.quad_logw, dtype=dtype)
        logabs, sign = jax.nn.logsumexp(
            det_logabs + log_weight[None, :],
            b=det_sign,
            axis=1,
            return_sign=True,
        )
        sign = sign * a_sign
        logabs = logabs + a_logabs

        if use_hole:
            n_sorb = 2 * norb
            sorb = jnp.arange(n_sorb, dtype=jnp.int32)
            hole = jnp.concatenate(
                (jnp.logical_not(alpha), jnp.logical_not(beta)), axis=-1
            )
            exponent = (
                jnp.sum(hole.astype(jnp.int32) * sorb[None, :], axis=-1)
                - n_det * (n_det - 1) // 2
                + d_alpha * d_beta
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign, logabs
