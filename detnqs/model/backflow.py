from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..utils import math, precision
from .base import Model


class Backflow(Model):
    """
    Dense spin-orbital backflow determinant.

    Sector:
        N = n_alpha + n_beta, K = 2 * norb.

    Network:
        u_i = 2 n_i - 1,
        u -> M(u) in R^{N x K}.

    Amplitude:
        psi(C) = det M(u)[:, C].
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        if n_elec > n_sorb:
            raise ValueError("Backflow requires n_alpha + n_beta <= 2 * norb")

        self.n_elec = n_elec
        self.n_sorb = n_sorb

        if self.ref_mat is None:
            self.ref = None
        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            self.ref = np.ascontiguousarray(ref.reshape(-1))

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_elec = int(self.n_elec)
        n_sorb = int(self.n_sorb)

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]
        occ = jnp.concatenate((alpha, beta), axis=-1)

        h = 2.0 * occ.astype(dtype) - 1.0

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
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
            bias_init = lambda key, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

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

        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(occ, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_elec)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_elec, n_elec))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))


class GBackflow(Model):
    """
    Electron/hole generalized spin-orbital backflow determinant.

    Sector:
        N = n_alpha + n_beta, K = 2 * norb, d = min(N, K - N).

    Network:
        m = n in electron representation,
        m = 1 - n in hole representation,
        u_i = 2 m_i - 1,
        u -> M(u) in R^{d x K}.

    Amplitude:
        electron: psi(C) = det M(u)[:, C],
        hole:     psi(C) = chi_ph(C) det M(u)[:, C^c].
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
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

        self.n_elec = n_elec
        self.n_sorb = n_sorb
        self.n_det = n_det
        self.use_hole = use_hole

        if self.ref_mat is None:
            self.ref = None
        else:
            ref = np.asarray(jax.device_get(self.ref_mat), dtype=np.float64)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError("ref_mat must have shape (n_alpha + n_beta, 2 * norb)")

            if use_hole:
                q = np.linalg.qr(ref.T, mode="complete")[0]
                ref = q[:, n_elec:].T

            self.ref = np.ascontiguousarray(ref.reshape(-1))

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_sorb = int(self.n_sorb)
        n_det = int(self.n_det)
        use_hole = bool(self.use_hole)

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]
        occ = jnp.concatenate((alpha, beta), axis=-1)

        mask = jnp.logical_not(occ) if use_hole else occ
        h = 2.0 * mask.astype(dtype) - 1.0

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
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
            bias_init = lambda key, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

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

        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(mask, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_det)

        col = jnp.broadcast_to(col[:, None, :], (batch, n_det, n_det))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)

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
    Spin-adapted paired backflow determinant.

    Sector:
        n_alpha >= n_beta,
        n_open = n_alpha - n_beta.

    Network:
        s_p = n_{p alpha} + n_{p beta},
        z_p = s_p - 1 in electron representation,
        z_p = 1 - s_p in hole representation,
        r_p = 1_{s_p = 1},
        (z, r) -> K(s), U(s).

    Amplitude:
        psi(A, B) = det [K(s)[A, B], U(s)[A]],
        K(s) = K(s)^T.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        if n_alpha < n_beta:
            raise ValueError("SBackflow requires n_alpha >= n_beta")

        n_elec = n_alpha + n_beta
        n_open = n_alpha - n_beta
        use_hole = n_elec > norb

        n_row = norb - n_beta if use_hole else n_alpha
        n_pair = norb - n_alpha if use_hole else n_beta
        n_out = norb * norb + norb * n_open

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
                q = np.linalg.qr(alpha_occ, mode="complete")[0]
                pair_orb = q[:, n_alpha:]
            else:
                pair_orb = alpha_occ[:, :n_beta]

            pair_ref = pair_orb @ pair_orb.T
            ref = np.concatenate((pair_ref.reshape(-1), open_ref.reshape(-1)))

            self.ref = np.ascontiguousarray(ref)

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_open = int(self.n_open)
        n_row = int(self.n_row)
        n_pair = int(self.n_pair)
        n_out = int(self.n_out)
        use_hole = bool(self.use_hole)

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        af = alpha.astype(dtype)
        bf = beta.astype(dtype)

        z = (1.0 - af - bf) if use_hole else (af + bf - 1.0)
        r = jnp.logical_xor(alpha, beta).astype(dtype)
        h = jnp.concatenate((z, r), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
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
            bias_init = lambda key, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

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

        row_mask = jnp.logical_not(beta) if use_hole else alpha
        col_mask = jnp.logical_not(alpha) if use_hole else beta

        row_score = jnp.where(row_mask, norb - 1 - orb[None, :], -1)
        _, row = jax.lax.top_k(row_score, n_row)

        if n_pair == 0:
            col = jnp.zeros((batch, 0), dtype=jnp.int32)
        else:
            col_score = jnp.where(col_mask, norb - 1 - orb[None, :], -1)
            _, col = jax.lax.top_k(col_score, n_pair)

        bid = jnp.arange(batch, dtype=jnp.int32)
        pair_mat = k[bid[:, None, None], row[:, :, None], col[:, None, :]]

        if n_open == 0:
            mat = pair_mat
        else:
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

        return sign.reshape((batch,)), logabs.reshape((batch,))


class PBackflow(Model):
    """
    Spin-projected spatial backflow determinant.

    Sector:
        n_alpha >= n_beta,
        S = M = (n_alpha - n_beta) / 2.

    Network:
        effective occupation m is electron or hole occupation,
        z_p = m_{p alpha} + m_{p beta} - 1,
        r_p = 1_{m_{p alpha} != m_{p beta}},
        (z, r) -> Y(m) in R^{d x norb}.

    Amplitude:
        psi(C) = sum_q c_q det M_q(m)[:, C],
        M_q[i, p alpha] = Y[i, p] a[q, i],
        M_q[i, p beta ] = Y[i, p] b[q, i].

    Note:
        in hole representation, beta holes are effective alpha particles.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: Any | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-3
    dtype: Any | None = None

    def setup(self):
        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        if n_alpha < n_beta:
            raise ValueError("PBackflow requires n_alpha >= n_beta")

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
        u = np.sqrt(0.5 * (1.0 + x))
        v = np.sqrt(0.5 * (1.0 - x))

        coef = 0.5 * (spin2 + 1) * w * (u**spin2)
        row_is_alpha = np.arange(n_det) < d_alpha

        spin_a = np.where(row_is_alpha[None, :], u[:, None], -v[:, None])
        spin_b = np.where(row_is_alpha[None, :], v[:, None], u[:, None])

        self.log_coef = np.ascontiguousarray(np.log(coef).astype(np.float64))
        self.spin_a = np.ascontiguousarray(spin_a.astype(np.float64))
        self.spin_b = np.ascontiguousarray(spin_b.astype(np.float64))

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

            self.ref = np.ascontiguousarray(ref.reshape(-1))

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch = x.shape[0]
        nword = x.shape[2]

        norb = int(self.norb)
        n_det = int(self.n_det)
        n_sorb = 2 * norb
        use_hole = bool(self.use_hole)

        shifts = jnp.arange(64, dtype=jnp.uint64)
        bits = ((x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)) != 0
        bits = bits.reshape(batch, 2, nword * 64)[:, :, :norb]

        alpha = bits[:, 0]
        beta = bits[:, 1]

        eff_alpha = jnp.logical_not(alpha) if use_hole else alpha
        eff_beta = jnp.logical_not(beta) if use_hole else beta

        af = eff_alpha.astype(dtype)
        bf = eff_beta.astype(dtype)

        z = af + bf - 1.0
        r = jnp.logical_xor(eff_alpha, eff_beta).astype(dtype)
        h = jnp.concatenate((z, r), axis=-1)

        for i, width in enumerate(self.hidden):
            h = nn.Dense(
                int(width),
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
            bias_init = lambda key, shape, dtype=dtype: ref.astype(dtype).reshape(shape)

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

        occ = jnp.concatenate((alpha, beta), axis=-1)
        mask = jnp.logical_not(occ) if use_hole else occ

        sorb = jnp.arange(n_sorb, dtype=jnp.int32)
        score = jnp.where(mask, n_sorb - 1 - sorb[None, :], -1)
        _, col = jax.lax.top_k(score, n_det)

        orb = jnp.where(col < norb, col, col - norb)
        is_alpha_col = col < norb

        spin_is_alpha = jnp.logical_not(is_alpha_col) if use_hole else is_alpha_col

        orb = jnp.broadcast_to(orb[:, None, :], (batch, n_det, n_det))
        mat = jnp.take_along_axis(y, orb, axis=2)

        spin_a = jnp.asarray(self.spin_a, dtype=dtype)
        spin_b = jnp.asarray(self.spin_b, dtype=dtype)

        spin = jnp.where(
            spin_is_alpha[:, None, None, :],
            spin_a[None, :, :, None],
            spin_b[None, :, :, None],
        )

        mat = mat[:, None, :, :] * spin

        det_sign, det_logabs = jnp.linalg.slogdet(mat)

        log_coef = jnp.asarray(self.log_coef, dtype=dtype)
        term_log = det_logabs + log_coef[None, :]

        sign, logabs = math.signed_logsumexp(det_sign, term_log, axis=-1)

        if use_hole:
            exponent = (
                jnp.sum(mask.astype(jnp.int32) * sorb[None, :], axis=-1)
                - n_det * (n_det - 1) // 2
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign.reshape((batch,)), logabs.reshape((batch,))