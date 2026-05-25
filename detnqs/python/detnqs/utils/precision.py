from __future__ import annotations

"""Global precision policy for DetNQS.

Precision is a boundary policy, not an algorithmic detail. Core estimators
should cast inputs at their boundaries and then run the mathematical kernel
without scattered dtype decisions.

Roles:
    model:
        Neural-network forward pass and autodiff.

    calc:
        Energy, local energy, probabilities, and host reductions.

    sr:
        SR/minSR matrices, RHS vectors, and linear solves.

JAX x64 is always enabled because determinant bitstrings use uint64 and quantum
chemistry calculations commonly require float64 arithmetic.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_PROFILE = "double"

_ROLES = {
    "model": "double",
    "calc": "double",
    "sr": "double",
}

_LEVELS = {"single", "double"}
_PROFILES = {"single", "double", "mixed"}

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
    """Set the global precision profile.

    Profiles:
        single:
            model = calc = sr = single

        double:
            model = calc = sr = double

        mixed:
            role precisions may be set independently. Defaults are
            model=single, calc=double, sr=double.
    """
    global _PROFILE, _ROLES

    profile = str(profile)

    if profile not in _PROFILES:
        raise ValueError("profile must be 'single', 'double', or 'mixed'")

    if profile in {"single", "double"}:
        if model is not None or calc is not None or sr is not None:
            raise ValueError("role overrides require profile='mixed'")

        roles = {
            "model": profile,
            "calc": profile,
            "sr": profile,
        }

    else:
        roles = {
            "model": "single" if model is None else str(model),
            "calc": "double" if calc is None else str(calc),
            "sr": "double" if sr is None else str(sr),
        }

        bad = [name for name, value in roles.items() if value not in _LEVELS]
        if bad:
            raise ValueError("role precision must be 'single' or 'double'")

    _PROFILE = profile
    _ROLES = roles

    jax.config.update("jax_enable_x64", True)
    jax.clear_caches()


def dtype(role: str = "calc", kind: str = "real", *, host: bool = False) -> Any:
    """Return the dtype assigned to a role.

    Args:
        role:
            One of model, calc, or sr.

        kind:
            real or complex.

        host:
            If True, return a NumPy dtype. Otherwise return a JAX dtype.
    """
    if role not in _ROLES:
        raise ValueError("role must be 'model', 'calc', or 'sr'")

    if kind not in {"real", "complex"}:
        raise ValueError("kind must be 'real' or 'complex'")

    jax_dtype, np_dtype = _DTYPES[_ROLES[role]][kind]
    return np_dtype if host else jax_dtype


def asarray(
    x: Any,
    role: str = "calc",
    kind: str | None = None,
    *,
    host: bool = False,
) -> Any:
    """Cast an array or PyTree according to the precision policy.

    If kind is None, each leaf keeps its real/complex character.
    """
    def cast_leaf(a):
        arr = np.asarray(a) if host else jnp.asarray(a)

        leaf_kind = kind
        if leaf_kind is None:
            leaf_kind = "complex" if jnp.issubdtype(arr.dtype, jnp.complexfloating) else "real"

        out_dtype = dtype(role, leaf_kind, host=host)
        return arr.astype(out_dtype, copy=False) if host else arr.astype(out_dtype)

    return jax.tree.map(cast_leaf, x)


def tiny(role: str = "calc") -> float:
    """Return the smallest positive normal real number for a role."""
    return float(np.finfo(dtype(role, "real", host=True)).tiny)


def eps(role: str = "calc") -> float:
    """Return the machine epsilon of the real dtype assigned to a role."""
    return float(np.finfo(dtype(role, "real", host=True)).eps)