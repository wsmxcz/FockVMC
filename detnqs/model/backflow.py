from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..utils import precision
from .base import Model

class Backflow(Model):
    """
    Symmetry-broken spin-orbital backflow Slater determinant.

    Fixed sector:
        N = n_alpha + n_beta,  K = 2 * norb,  d = min(N, K - N).

    Input:
        c_i in {0, 1},  i = 0, ..., K - 1.

    Amplitude:
        psi(C) = det [M0 + dM(c)][:, C],
        or chi_ph(C) det [M0 + dM(1 - c)][:, C^c].

    The head is fully spin-orbital resolved; SU(2) is not enforced.
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

        use_hole = n_elec > norb
        n_det = n_sorb - n_elec if use_hole else n_elec

        sorb = jnp.arange(n_sorb)
        shifts = jnp.arange(64, dtype=jnp.uint64)

        # Packed bits -> spin-orbital occupations.
        occ = (x[:, :, :, None] >> shifts[None, None, None, :]) & jnp.uint64(1)
        occ = occ.reshape(batch, 2, nword * 64)[:, :, :norb].astype(jnp.int32)
        occ = jnp.concatenate((occ[:, 0], occ[:, 1]), axis=-1)

        alpha = occ[:, :norb]
        beta = occ[:, norb:]

        # Local spin-orbital features.
        za = 2 * alpha - 1
        zb = 2 * beta - 1

        if use_hole:
            za = -za
            zb = -zb

        h = jnp.concatenate((za, zb, za * zb), axis=-1).astype(dtype)

        for width in self.hidden:
            h = nn.Dense(width, dtype=dtype, param_dtype=dtype)(h)
            h = nn.silu(h)

        out = nn.Dense(
            n_det * n_sorb,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nn.initializers.normal(self.init_scale),
            bias_init=nn.initializers.zeros,
        )(h)

        delta = out.reshape(batch, n_det, n_sorb)
        eye = jnp.eye(n_sorb, dtype=dtype)

        if self.ref_mat is None:
            alpha0 = jnp.arange(n_alpha)
            beta0 = norb + jnp.arange(n_beta)

            if use_hole:
                # Canonical hole reference.
                alpha0 = jnp.arange(n_alpha, norb)
                beta0 = norb + jnp.arange(n_beta, norb)

            ref = eye[jnp.concatenate((alpha0, beta0), axis=0)]
        else:
            # Path-compatible reference.
            ref = jnp.asarray(self.ref_mat, dtype=dtype)

            if ref.shape != (n_det, n_sorb):
                raise ValueError(
                    f"Backflow ref_mat must have shape {(n_det, n_sorb)}, "
                    f"got {ref.shape}"
                )

        mat = ref[None] + delta
        occ_bool = occ.astype(bool)

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

        # Gather determinant columns.
        col_idx = jnp.broadcast_to(col[:, None, :], (batch, n_det, n_det))
        submat = jnp.take_along_axis(mat, col_idx, axis=2)

        sign, logabs = jnp.linalg.slogdet(submat)

        if use_hole:
            # Particle-hole basis phase.
            holes = 1 - occ
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
    SU(2)-adapted single paired-backflow determinant.

    Fixed sector:
        n_alpha >= n_beta,  S = (n_alpha - n_beta) / 2,  M = S.

    Input:
        s_p = n_{p,alpha} + n_{p,beta} in {0, 1, 2}.

    Head:
        K(s) = K(s)^T,  U(s) in R^{norb x (n_alpha - n_beta)}.

    Amplitude:
        psi(A, B) = det [K(s)[A, B]  U(s)[A]].

    High filling uses the particle-hole dual.
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

        alpha_occ = alpha.astype(bool)
        beta_occ = beta.astype(bool)

        # Spin-free scalar input.
        spatial = alpha + beta
        h = (1 - spatial if use_hole else spatial - 1).astype(dtype)

        for width in self.hidden:
            h = nn.Dense(width, dtype=dtype, param_dtype=dtype)(h)
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

        # Triangular head -> symmetric pair update.
        tri_i, tri_j = jnp.triu_indices(norb)
        upper = jnp.zeros((batch, norb, norb), dtype=dtype)
        upper = upper.at[:, tri_i, tri_j].set(out[:, :n_tri])

        diag = jnp.diagonal(upper, axis1=-2, axis2=-1)
        eye = jnp.eye(norb, dtype=dtype)

        delta_pair = upper + jnp.swapaxes(upper, -1, -2)
        delta_pair = delta_pair - eye[None] * diag[:, None, :]

        # Open-shell update.
        delta_open = out[:, n_tri:].reshape(batch, norb, n_open)

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

            pair_coeff = 0.5 * (alpha_ref[:, :n_pair] + beta_ref[:, :n_pair])
            pair_ref = pair_coeff @ pair_coeff.T
            open_ref = alpha_ref[:, n_pair:n_row]

        pair = pair_ref[None] + delta_pair
        open_orb = open_ref[None] + delta_open

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