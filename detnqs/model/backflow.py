from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
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

    Ansatz:
        M(x) = M0 + dM(x),
        psi(C) = det M(x)[:, C].

    This is the standard electron-only NNBF baseline.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch, nword = x.shape[0], x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        if n_elec > n_sorb:
            raise ValueError("Backflow requires n_alpha + n_beta <= 2 * norb")

        orb = jnp.arange(norb)
        sorb = jnp.arange(n_sorb)
        shifts = jnp.arange(64, dtype=jnp.uint64)

        # Packed bits -> alpha/beta occupations.
        occ = (x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)
        occ = occ.reshape(batch, 2, nword * 64)[:, :, :norb].astype(jnp.int32)

        alpha = occ[:, 0]
        beta = occ[:, 1]

        # Local 4-state input.
        token = alpha + 2 * beta
        h = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            h = nn.Dense(width, use_bias=False, dtype=dtype, param_dtype=dtype)(h)
            h = nn.LayerNorm(dtype=dtype, param_dtype=dtype)(h)
            h = nn.silu(h)

        out = nn.Dense(
            n_elec * n_sorb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
        )(h)

        mat = out.reshape(batch, n_elec, n_sorb)

        if self.ref_mat is None:
            # Canonical electron reference.
            row = jnp.concatenate((orb[:n_alpha], norb + orb[:n_beta]), axis=0)
            ref = jnp.eye(n_sorb, dtype=dtype)[row]
        else:
            # Electron reference.
            ref = jnp.asarray(self.ref_mat, dtype=dtype)

            if ref.shape != (n_elec, n_sorb):
                raise ValueError(
                    f"Backflow ref_mat must have shape {(n_elec, n_sorb)}, "
                    f"got {ref.shape}"
                )

        mat = mat + ref[None]

        # Occupied spin-orbital columns.
        occ_sorb = jnp.concatenate((alpha, beta), axis=-1).astype(bool)
        col = jnp.argsort(
            jnp.where(occ_sorb, sorb, n_sorb),
            axis=-1,
            stable=True,
        )[:, :n_elec]

        # Gather determinant matrix.
        col = jnp.broadcast_to(col[:, None, :], (batch, n_elec, n_elec))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)
        return sign.reshape((batch,)), logabs.reshape((batch,))


class UBackflow(Model):
    """
    UHF-like spin-block backflow Slater determinant.

    Fixed sector:
        n_alpha, n_beta,  d_sigma = min(n_sigma, norb - n_sigma).

    Input:
        t_p in {0, alpha, beta, double}, encoded as 4-state one-hot.

    Ansatz:
        M_alpha(x) = M_alpha0 + dM_alpha(x),
        M_beta(x)  = M_beta0  + dM_beta(x),

        psi(A, B)
        = det M_alpha(x)[:, A] det M_beta(x)[:, B],

    with independent particle-hole duals for alpha and beta blocks.

    The ansatz preserves S_z but does not enforce SU(2).
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch, nword = x.shape[0], x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        if n_alpha > norb or n_beta > norb:
            raise ValueError("UBackflow requires n_alpha, n_beta <= norb")

        use_alpha_hole = n_alpha > norb - n_alpha
        use_beta_hole = n_beta > norb - n_beta

        d_alpha = norb - n_alpha if use_alpha_hole else n_alpha
        d_beta = norb - n_beta if use_beta_hole else n_beta

        orb = jnp.arange(norb)
        shifts = jnp.arange(64, dtype=jnp.uint64)

        # Packed bits -> alpha/beta occupations.
        occ = (x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)
        occ = occ.reshape(batch, 2, nword * 64)[:, :, :norb].astype(jnp.int32)

        alpha = occ[:, 0]
        beta = occ[:, 1]

        # Spin-block electron/hole input.
        alpha_in = 1 - alpha if use_alpha_hole else alpha
        beta_in = 1 - beta if use_beta_hole else beta

        token = alpha_in + 2 * beta_in
        h = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            h = nn.Dense(width, use_bias=False, dtype=dtype, param_dtype=dtype)(h)
            h = nn.LayerNorm(dtype=dtype, param_dtype=dtype)(h)
            h = nn.silu(h)

        n_out = (d_alpha + d_beta) * norb

        out = nn.Dense(
            n_out,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
        )(h)

        alpha_out = out[:, : d_alpha * norb].reshape(batch, d_alpha, norb)
        beta_out = out[:, d_alpha * norb :].reshape(batch, d_beta, norb)

        eye = jnp.eye(norb, dtype=dtype)

        if self.ref_mat is None:
            if use_alpha_hole:
                # Canonical alpha-hole reference.
                alpha_ref = eye[orb[n_alpha:]]
            else:
                # Canonical alpha-electron reference.
                alpha_ref = eye[orb[:n_alpha]]

            if use_beta_hole:
                # Canonical beta-hole reference.
                beta_ref = eye[orb[n_beta:]]
            else:
                # Canonical beta-electron reference.
                beta_ref = eye[orb[:n_beta]]
        else:
            # Path-compatible spin-block reference.
            ref = jnp.asarray(self.ref_mat, dtype=dtype)
            ref_shape = (d_alpha + d_beta, 2 * norb)

            if ref.shape != ref_shape:
                raise ValueError(
                    f"UBackflow ref_mat must have shape {ref_shape}, "
                    f"got {ref.shape}"
                )

            alpha_ref = ref[:d_alpha, :norb]
            beta_ref = ref[d_alpha:, norb:]

        alpha_mat = alpha_ref[None] + alpha_out
        beta_mat = beta_ref[None] + beta_out

        alpha_occ = alpha.astype(bool)
        beta_occ = beta.astype(bool)

        if use_alpha_hole:
            # Alpha-hole columns.
            alpha_col = jnp.argsort(
                jnp.where(alpha_occ, norb, orb),
                axis=-1,
                stable=True,
            )[:, :d_alpha]
        else:
            # Alpha-electron columns.
            alpha_col = jnp.argsort(
                jnp.where(alpha_occ, orb, norb),
                axis=-1,
                stable=True,
            )[:, :d_alpha]

        if use_beta_hole:
            # Beta-hole columns.
            beta_col = jnp.argsort(
                jnp.where(beta_occ, norb, orb),
                axis=-1,
                stable=True,
            )[:, :d_beta]
        else:
            # Beta-electron columns.
            beta_col = jnp.argsort(
                jnp.where(beta_occ, orb, norb),
                axis=-1,
                stable=True,
            )[:, :d_beta]

        if d_alpha == 0:
            sign_alpha = jnp.ones((batch,), dtype=dtype)
            log_alpha = jnp.zeros((batch,), dtype=dtype)
        else:
            # Gather alpha determinant.
            alpha_col = jnp.broadcast_to(
                alpha_col[:, None, :],
                (batch, d_alpha, d_alpha),
            )
            alpha_mat = jnp.take_along_axis(alpha_mat, alpha_col, axis=2)
            sign_alpha, log_alpha = jnp.linalg.slogdet(alpha_mat)

        if d_beta == 0:
            sign_beta = jnp.ones((batch,), dtype=dtype)
            log_beta = jnp.zeros((batch,), dtype=dtype)
        else:
            # Gather beta determinant.
            beta_col = jnp.broadcast_to(
                beta_col[:, None, :],
                (batch, d_beta, d_beta),
            )
            beta_mat = jnp.take_along_axis(beta_mat, beta_col, axis=2)
            sign_beta, log_beta = jnp.linalg.slogdet(beta_mat)

        sign = sign_alpha * sign_beta
        logabs = log_alpha + log_beta

        orb_i = orb.astype(jnp.int32)

        if use_alpha_hole:
            # Alpha particle-hole phase.
            alpha_hole = 1 - alpha

            exp_alpha = (
                jnp.sum(alpha_hole * orb_i[None], axis=-1)
                - d_alpha * (d_alpha - 1) // 2
            )

            sign = sign * (1 - 2 * (exp_alpha & 1)).astype(dtype)

        if use_beta_hole:
            # Beta particle-hole phase.
            beta_hole = 1 - beta

            exp_beta = (
                jnp.sum(beta_hole * orb_i[None], axis=-1)
                - d_beta * (d_beta - 1) // 2
            )

            sign = sign * (1 - 2 * (exp_beta & 1)).astype(dtype)

        return sign.reshape((batch,)), logabs.reshape((batch,))


class GBackflow(Model):
    """
    Generalized spin-orbital backflow Slater determinant.

    Fixed sector:
        N = n_alpha + n_beta,  K = 2 * norb,  d = min(N, K - N).

    Input:
        t_p in {0, alpha, beta, double}, encoded as 4-state one-hot.

    Ansatz:
        M(x) = M0 + dM(x),
        psi(C) = det M(x)[:, C],
        or chi_ph(C) det M(1 - x)[:, C^c].

    The head is fully spin-orbital resolved; S_z and SU(2) are not enforced.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch, nword = x.shape[0], x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        n_elec = n_alpha + n_beta
        n_sorb = 2 * norb

        if n_elec > n_sorb:
            raise ValueError("GBackflow requires n_alpha + n_beta <= 2 * norb")

        use_hole = n_elec > norb
        n_det = n_sorb - n_elec if use_hole else n_elec

        orb = jnp.arange(norb)
        sorb = jnp.arange(n_sorb)
        shifts = jnp.arange(64, dtype=jnp.uint64)

        # Packed bits -> alpha/beta occupations.
        occ = (x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)
        occ = occ.reshape(batch, 2, nword * 64)[:, :, :norb].astype(jnp.int32)

        alpha = occ[:, 0]
        beta = occ[:, 1]
        occ_sorb = jnp.concatenate((alpha, beta), axis=-1)

        # Electron/hole 4-state input.
        net_occ = 1 - occ_sorb if use_hole else occ_sorb
        token = net_occ[:, :norb] + 2 * net_occ[:, norb:]
        h = jax.nn.one_hot(token, 4, dtype=dtype).reshape(batch, -1)

        for width in self.hidden:
            h = nn.Dense(width, use_bias=False, dtype=dtype, param_dtype=dtype)(h)
            h = nn.LayerNorm(dtype=dtype, param_dtype=dtype)(h)
            h = nn.silu(h)

        out = nn.Dense(
            n_det * n_sorb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
        )(h)

        mat = out.reshape(batch, n_det, n_sorb)

        if self.ref_mat is None:
            if use_hole:
                # Canonical hole reference.
                row = jnp.concatenate((orb[n_alpha:], norb + orb[n_beta:]), axis=0)
            else:
                # Canonical electron reference.
                row = jnp.concatenate((orb[:n_alpha], norb + orb[:n_beta]), axis=0)

            ref = jnp.eye(n_sorb, dtype=dtype)[row]
        else:
            # Path-compatible reference.
            ref = jnp.asarray(self.ref_mat, dtype=dtype)

            if ref.shape != (n_det, n_sorb):
                raise ValueError(
                    f"GBackflow ref_mat must have shape {(n_det, n_sorb)}, "
                    f"got {ref.shape}"
                )

        mat = mat + ref[None]
        occ_bool = occ_sorb.astype(bool)

        if use_hole:
            # Hole columns.
            col = jnp.argsort(
                jnp.where(occ_bool, n_sorb, sorb),
                axis=-1,
                stable=True,
            )[:, :n_det]
        else:
            # Electron columns.
            col = jnp.argsort(
                jnp.where(occ_bool, sorb, n_sorb),
                axis=-1,
                stable=True,
            )[:, :n_det]

        # Gather determinant matrix.
        col = jnp.broadcast_to(col[:, None, :], (batch, n_det, n_det))
        mat = jnp.take_along_axis(mat, col, axis=2)

        sign, logabs = jnp.linalg.slogdet(mat)

        if use_hole:
            # Particle-hole basis phase.
            holes = 1 - occ_sorb
            sorb_i = sorb.astype(jnp.int32)

            exponent = (
                jnp.sum(holes * sorb_i[None], axis=-1)
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

    Head:
        K(s) = K(s)^T,  U(s) in R^{norb x (n_alpha - n_beta)}.

    Amplitude:
        psi(A, B) = det [K(s)[A, B]  U(s)[A]].

    High filling uses the particle-hole paired dual.
    """

    norb: int
    n_alpha: int
    n_beta: int
    ref_mat: jax.Array | None = None
    hidden: tuple[int, ...] = (64,)
    init_scale: float = 1.0e-6
    dtype: Any | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        dtype = precision.real("model") if self.dtype is None else self.dtype

        batch, nword = x.shape[0], x.shape[2]

        norb = int(self.norb)
        n_alpha = int(self.n_alpha)
        n_beta = int(self.n_beta)

        if n_alpha < n_beta:
            raise ValueError("SBackflow requires n_alpha >= n_beta")

        n_open = n_alpha - n_beta
        use_hole = (n_alpha + n_beta) > norb

        n_row = norb - n_beta if use_hole else n_alpha
        n_pair = norb - n_alpha if use_hole else n_beta

        orb = jnp.arange(norb)
        shifts = jnp.arange(64, dtype=jnp.uint64)

        # Packed bits -> alpha/beta occupations.
        occ = (x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)
        occ = occ.reshape(batch, 2, nword * 64)[:, :, :norb].astype(jnp.int32)

        alpha = occ[:, 0]
        beta = occ[:, 1]

        # Spin-free input.
        spatial = alpha + beta
        h = (1 - spatial if use_hole else spatial - 1).astype(dtype)

        for width in self.hidden:
            h = nn.Dense(width, use_bias=False, dtype=dtype, param_dtype=dtype)(h)
            h = nn.LayerNorm(dtype=dtype, param_dtype=dtype)(h)
            h = nn.silu(h)

        n_tri = norb * (norb + 1) // 2
        n_out = n_tri + norb * n_open

        out = nn.Dense(
            n_out,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
        )(h)

        # Triangular head -> symmetric K.
        tri_i, tri_j = jnp.triu_indices(norb)
        upper = jnp.zeros((batch, norb, norb), dtype=dtype)
        upper = upper.at[:, tri_i, tri_j].set(out[:, :n_tri])

        eye = jnp.eye(norb, dtype=dtype)
        diag = jnp.diagonal(upper, axis1=-2, axis2=-1)

        d_pair = upper + jnp.swapaxes(upper, -1, -2)
        d_pair = d_pair - eye[None] * diag[:, None, :]

        # Open-shell head.
        d_open = out[:, n_tri:].reshape(batch, norb, n_open)

        if self.ref_mat is None:
            if use_hole:
                # Canonical hole reference.
                pair_ref = eye * (orb >= n_alpha).astype(dtype)[None]
                open_ref = eye[:, n_beta:n_alpha]
            else:
                # Canonical electron reference.
                pair_ref = eye * (orb < n_beta).astype(dtype)[None]
                open_ref = eye[:, n_beta:n_alpha]
        else:
            # Path-compatible paired reference.
            ref = jnp.asarray(self.ref_mat, dtype=dtype)
            n_ref = n_row + n_pair

            if ref.shape != (n_ref, 2 * norb):
                raise ValueError(
                    f"SBackflow ref_mat must have shape {(n_ref, 2 * norb)}, "
                    f"got {ref.shape}"
                )

            alpha_ref = ref[:n_row, :norb].T
            beta_ref = ref[n_row:, norb:].T

            pair = 0.5 * (alpha_ref[:, :n_pair] + beta_ref[:, :n_pair])
            pair_ref = pair @ pair.T
            open_ref = alpha_ref[:, n_pair:n_row]

        pair = pair_ref[None] + d_pair
        open_orb = open_ref[None] + d_open

        alpha_occ = alpha.astype(bool)
        beta_occ = beta.astype(bool)

        if use_hole:
            # Hole rows/columns.
            row = jnp.argsort(
                jnp.where(beta_occ, norb, orb),
                axis=-1,
                stable=True,
            )[:, :n_row]

            col = jnp.argsort(
                jnp.where(alpha_occ, norb, orb),
                axis=-1,
                stable=True,
            )[:, :n_pair]
        else:
            # Electron rows/columns.
            row = jnp.argsort(
                jnp.where(alpha_occ, orb, norb),
                axis=-1,
                stable=True,
            )[:, :n_row]

            col = jnp.argsort(
                jnp.where(beta_occ, orb, norb),
                axis=-1,
                stable=True,
            )[:, :n_pair]

        # Gather K[row, col].
        row_idx = jnp.broadcast_to(row[:, :, None], (batch, n_row, norb))
        pair_row = jnp.take_along_axis(pair, row_idx, axis=1)

        col_idx = jnp.broadcast_to(col[:, None, :], (batch, n_row, n_pair))
        pair_mat = jnp.take_along_axis(pair_row, col_idx, axis=2)

        # Gather U[row].
        open_idx = jnp.broadcast_to(row[:, :, None], (batch, n_row, n_open))
        open_mat = jnp.take_along_axis(open_orb, open_idx, axis=1)

        mat = jnp.concatenate((pair_mat, open_mat), axis=-1)
        sign, logabs = jnp.linalg.slogdet(mat)

        if use_hole:
            # Block-ordered particle-hole phase.
            hole_alpha = 1 - beta
            hole_beta = 1 - alpha
            orb_i = orb.astype(jnp.int32)

            exponent = (
                n_pair
                + n_row * norb
                + jnp.sum(hole_alpha * orb_i[None], axis=-1)
                + jnp.sum(hole_beta * orb_i[None], axis=-1)
                - n_row * (n_row - 1) // 2
                - n_pair * (n_pair - 1) // 2
            )

            phase = (1 - 2 * (exponent & 1)).astype(dtype)
            sign = sign * phase

        return sign.reshape((batch,)), logabs.reshape((batch,))
