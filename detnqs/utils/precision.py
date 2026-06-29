"""Global dtype policy.

Precision is a boundary policy. Kernels should cast inputs at their boundary
and then run the mathematical calculation without scattered dtype decisions.

Roles:
    model: neural-network forward pass and autodiff.
    calc: energies, probabilities, and host reductions.
    sr: stochastic-reconfiguration matrices, RHS vectors, and solves.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_ROLES = {
    "model": "double",
    "calc": "double",
    "sr": "double",
}

_DTYPES = {
    "single": {
        "real": (jnp.float32, np.float32),
        "complex": (jnp.complex64, np.complex64),
    },
    "double": {
        "real": (jnp.float64, np.float64),
        "complex": (jnp.complex128, np.complex128),
    },
}


def configure(
    profile: str = "double",
    *,
    model: str | None = None,
    calc: str | None = None,
    sr: str | None = None,
) -> None:
    """Set the global precision profile."""
    global _ROLES

    profile = str(profile)
    if profile not in {"single", "double", "mixed"}:
        raise ValueError("profile must be 'single', 'double', or 'mixed'")

    if profile in {"single", "double"}:
        if model is not None or calc is not None or sr is not None:
            raise ValueError("role overrides require profile='mixed'")
        roles = {"model": profile, "calc": profile, "sr": profile}
    else:
        roles = {
            "model": "single" if model is None else str(model),
            "calc": "double" if calc is None else str(calc),
            "sr": "double" if sr is None else str(sr),
        }
        bad = [
            name
            for name, value in roles.items()
            if value not in {"single", "double"}
        ]
        if bad:
            raise ValueError("role precision must be 'single' or 'double'")

    _ROLES = roles
    jax.config.update("jax_enable_x64", True)
    jax.clear_caches()


def policy() -> dict[str, str]:
    """Return the active role precision policy."""
    return dict(_ROLES)


def _dtype(role: str, kind: str, *, host: bool = False) -> Any:
    if role not in _ROLES:
        raise ValueError("role must be 'model', 'calc', or 'sr'")
    if kind not in {"real", "complex"}:
        raise ValueError("kind must be 'real' or 'complex'")
    jax_dtype, np_dtype = _DTYPES[_ROLES[role]][kind]
    return np_dtype if host else jax_dtype


def real(role: str = "calc", *, host: bool = False) -> Any:
    """Return the real dtype assigned to a role."""
    return _dtype(role, "real", host=host)


def complex(role: str = "calc", *, host: bool = False) -> Any:
    """Return the complex dtype assigned to a role."""
    return _dtype(role, "complex", host=host)


def cast(
    x: Any,
    role: str = "calc",
    kind: str | None = None,
    *,
    host: bool = False,
) -> Any:
    """Cast floating leaves according to the precision policy.

    With ``kind=None``, real and complex floating leaves are cast to the role
    dtype while integer and boolean leaves are preserved. Passing
    ``kind='real'`` or ``kind='complex'`` forces all leaves to that dtype.
    """
    if kind not in {None, "real", "complex"}:
        raise ValueError("kind must be None, 'real', or 'complex'")

    def cast_leaf(a):
        arr = np.asarray(a) if host else jnp.asarray(a)

        if kind is None:
            if np.issubdtype(arr.dtype, np.complexfloating):
                leaf_kind = "complex"
            elif np.issubdtype(arr.dtype, np.floating):
                leaf_kind = "real"
            else:
                return arr
        else:
            leaf_kind = kind

        return arr.astype(_dtype(role, leaf_kind, host=host), copy=False)

    return jax.tree.map(cast_leaf, x)


def host(x: Any, role: str = "calc", kind: str | None = None) -> Any:
    """Move a PyTree to host and cast floating leaves by role."""
    return cast(jax.device_get(x), role, kind, host=True)


def device(x: Any, role: str = "model", kind: str | None = None) -> Any:
    """Move a PyTree to the default JAX device and cast floating leaves."""
    return cast(x, role, kind, host=False)


def tiny(role: str = "calc") -> float:
    """Return the smallest positive normal real number for a role."""
    return float(np.finfo(real(role, host=True)).tiny)


def eps(role: str = "calc") -> float:
    """Return the machine epsilon of the real dtype assigned to a role."""
    return float(np.finfo(real(role, host=True)).eps)
