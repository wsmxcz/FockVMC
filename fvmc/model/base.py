from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from numpy.typing import ArrayLike


def slater_reference(alpha: ArrayLike, beta: ArrayLike) -> np.ndarray:
    """Assemble occupied alpha/beta orbitals into a Slater reference."""
    alpha = np.asarray(alpha, dtype=np.float64, order="C")
    beta = np.asarray(beta, dtype=np.float64, order="C")

    if alpha.ndim != 2 or beta.ndim != 2 or alpha.shape[0] != beta.shape[0]:
        raise ValueError("alpha and beta must have shape (norb, n_elec)")

    norb = alpha.shape[0]
    n_alpha = alpha.shape[1]
    n_beta = beta.shape[1]
    ref_mat = np.zeros((n_alpha + n_beta, 2 * norb), dtype=np.float64)
    ref_mat[:n_alpha, :norb] = alpha.T
    ref_mat[n_alpha:, norb:] = beta.T
    return ref_mat


def to_logabs(logpsi: Any) -> Any:
    """Extract log|psi| from real, complex, or signed output."""
    if isinstance(logpsi, tuple):
        return logpsi[1]

    xp = np if isinstance(logpsi, np.ndarray) else jnp
    if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
        return xp.real(logpsi)

    return logpsi


def to_psi(logpsi: Any) -> Any:
    """Exponentiate after a numerically stable global shift."""
    if isinstance(logpsi, tuple):
        sign, logabs = logpsi
        xp = np if isinstance(logabs, np.ndarray) else jnp
        return sign * xp.exp(logabs - xp.max(logabs))

    xp = np if isinstance(logpsi, np.ndarray) else jnp

    if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
        return xp.exp(logpsi - xp.max(xp.real(logpsi)))

    return xp.exp(logpsi - xp.max(logpsi))


def to_ratio(num: Any, den: Any) -> Any:
    """Return psi(num) / psi(den)."""
    if isinstance(num, tuple):
        sign_num, logabs_num = num
        sign_den, logabs_den = den
        xp = np if isinstance(logabs_num, np.ndarray) else jnp

        safe_den = xp.where(sign_den != 0, sign_den, 1)
        return (sign_num / safe_den) * xp.exp(logabs_num - logabs_den)

    xp = np if isinstance(num, np.ndarray) else jnp
    return xp.exp(num - den)


class Model(nn.Module):
    """Base class for real, complex, and signed-log wavefunctions."""

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError

    def apply(self, params: Any, x: Any) -> Any:
        """Evaluate ``logpsi`` for configurations ``x``."""
        return nn.Module.apply(self, {"params": params}, x)

    def logpsi(self, theta: Any, x: Any) -> Any:
        """Evaluate the raw wavefunction representation."""
        out = self.apply(theta, x)

        if isinstance(out, tuple):
            sign, logabs = out
            return jax.lax.stop_gradient(sign), logabs

        return out

    def logabs(self, theta: Any, x: Any) -> Any:
        """Evaluate log|psi_theta(x)|."""
        return to_logabs(self.logpsi(theta, x))

    def coord(self, theta: Any, x: Any) -> Any:
        """Evaluate the real coordinate differentiated by autodiff."""
        logpsi = self.logpsi(theta, x)

        if isinstance(logpsi, tuple):
            return logpsi[1]

        if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
            return jnp.stack((jnp.real(logpsi), jnp.imag(logpsi)), axis=-1)

        return logpsi

    def cotangent(self, logpsi: Any, dlogpsi: Any) -> Any:
        """Map wavefunction cotangents to real autodiff coordinates."""
        xp = np if isinstance(dlogpsi, np.ndarray) else jnp

        if isinstance(logpsi, tuple):
            return xp.real(dlogpsi)

        if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
            return xp.stack((xp.real(dlogpsi), xp.imag(dlogpsi)), axis=-1)

        return xp.real(dlogpsi)
