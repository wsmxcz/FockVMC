from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Geometry:
    """SR geometry with O = sqrt(w) (J - <J>_w)."""

    params: Any
    coord: Callable[[Any, Any], Any]
    x: Any
    weight: Any
    b: Any
