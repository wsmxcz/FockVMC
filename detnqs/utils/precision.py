"""Global dtype policy for model, calculation, and SR boundaries."""

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

    if profile not in {"single", "double", "mixed"}:
        raise ValueError("profile must be 'single', 'double', or 'mixed'")

    if profile in {"single", "double"}:
        if model is not None or calc is not None or sr is not None:
            raise ValueError("role overrides require profile='mixed'")
        roles = {"model": profile, "calc": profile, "sr": profile}
    else:
        roles = {
            "model": "single" if model is None else model,
            "calc": "double" if calc is None else calc,
            "sr": "double" if sr is None else sr,
        }
        if any(value not in {"single", "double"} for value in roles.values()):
            raise ValueError("role precision must be 'single' or 'double'")

    _ROLES = roles
    jax.clear_caches()


def real(role: str = "calc", *, host: bool = False) -> Any:
    """Return the real dtype assigned to a role."""
    if role not in _ROLES:
        raise ValueError("role must be 'model', 'calc', or 'sr'")
    jax_dtype, np_dtype = _DTYPES[_ROLES[role]]["real"]
    return np_dtype if host else jax_dtype


def complex(role: str = "calc", *, host: bool = False) -> Any:
    """Return the complex dtype assigned to a role."""
    if role not in _ROLES:
        raise ValueError("role must be 'model', 'calc', or 'sr'")
    jax_dtype, np_dtype = _DTYPES[_ROLES[role]]["complex"]
    return np_dtype if host else jax_dtype


def cast(
    x: Any,
    role: str = "calc",
    kind: str | None = None,
    *,
    host: bool = False,
) -> Any:
    """Cast floating leaves by role, preserving integers unless forced."""
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

        dtype = (
            real(role, host=host)
            if leaf_kind == "real"
            else complex(role, host=host)
        )
        return arr.astype(dtype, copy=False)

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
