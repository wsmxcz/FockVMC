from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Tree = Any


@dataclass(frozen=True, slots=True)
class Geometry:
    """Local variational geometry consumed by geometry-aware optimizers.

    The variational state owns all physics and sampling.  The optimizer only sees
    the local score geometry

        O = sqrt(w) * (J - <J>_w)

    through a lightweight recipe: parameters, coordinate function, samples,
    normalized or unnormalized weights, and the sample-space right hand side b.

    No Hamiltonian, sampler, local-energy graph, or Markov-chain state belongs
    to the optimizer.
    """

    theta: Tree
    coord: Callable[[Tree, Any], Any]
    x: Any
    w: Any
    b: Any