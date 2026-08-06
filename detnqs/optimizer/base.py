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
        params:
            Parameters used to build the geometry.

        coord:
            Real autodiff coordinate, coord(params, x).

        x:
            Sample axis used by the geometry.

        weight:
            Non-negative sample weights. They may be normalized or
            unnormalized; optimizers normalize them internally.

        b:
            Sample-space right-hand side used by PSR.
    """

    params: Any
    coord: Callable[[Any, Any], Any]
    x: Any
    weight: Any
    b: Any
