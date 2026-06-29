from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Geometry:
    """Local geometry consumed by SR-like optimizers.

    The centered weighted Jacobian is

        O = sqrt(w) * (J - <J>_w).

    Fields:
        theta:
            Parameters used to build the geometry.

        coord:
            Real autodiff coordinate, coord(theta, x).

        x:
            Sample axis used by the geometry.

        w:
            Non-negative sample weights. They may be normalized or
            unnormalized; optimizers normalize them internally.

        b:
            Sample-space right-hand side. minSR and PSR use b directly.
            Parameter-space SR uses the incoming Optax update instead.
    """

    theta: Any
    coord: Callable[[Any, Any], Any]
    x: Any
    w: Any
    b: Any
