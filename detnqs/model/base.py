from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def to_logabs(logpsi: Any) -> Any:
    """Return log|psi| from a supported wavefunction representation.

    Supported representations:
        real array:
            log|psi|.

        complex array:
            log|psi| + i phase.

        (sign, logabs):
            signed real wavefunction representation.
    """
    if isinstance(logpsi, tuple):
        return logpsi[1]

    if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
        return (
            np.real(logpsi)
            if isinstance(logpsi, np.ndarray)
            else jnp.real(logpsi)
        )

    return logpsi


def to_psi(logpsi: Any) -> Any:
    """Return shifted wavefunction values.

    The global shift does not change Rayleigh quotients or amplitude ratios.
    It only improves numerical stability when exponentiating log-amplitudes.
    """
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
    """Base class for Fock-space neural quantum states.

    Raw model output:
        real:
            log|psi|.

        complex:
            log|psi| + i phase.

        (sign, logabs):
            signed real wavefunction.

    Differentiated coordinate:
        real:
            log|psi|.

        complex:
            [log|psi|, phase].

        signed real:
            logabs with sign stopped.
    """

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
        """Map dL/dlogpsi to dL/dcoord.

        This method keeps optimizer-facing autodiff coordinates real. Complex
        cotangents are represented by real and imaginary channels.
        """
        if isinstance(logpsi, tuple):
            return (
                np.real(dlogpsi)
                if isinstance(dlogpsi, np.ndarray)
                else jnp.real(dlogpsi)
            )

        if jnp.issubdtype(jnp.asarray(logpsi).dtype, jnp.complexfloating):
            if isinstance(dlogpsi, np.ndarray):
                return np.stack((np.real(dlogpsi), np.imag(dlogpsi)), axis=-1)

            return jnp.stack((jnp.real(dlogpsi), jnp.imag(dlogpsi)), axis=-1)

        return (
            np.real(dlogpsi)
            if isinstance(dlogpsi, np.ndarray)
            else jnp.real(dlogpsi)
        )
