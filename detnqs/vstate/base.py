from __future__ import annotations

from typing import Any
from typing import Protocol

from ..model import Model

Tree = Any


class VState(Protocol):
    """Variational-state protocol consumed by the VMC driver.

    A variational state owns the physics and estimator. The driver only needs
    three operations:

        expect:
            Return scalar estimator statistics without a gradient.

        expect_and_grad:
            Return energy, gradient, statistics, and optional optimizer
            geometry.

        replace:
            Return a copy with updated fields, usually updated parameters.
    """

    model: Model
    params: Tree

    def expect(self) -> tuple["VState", dict[str, float]]: ...

    def expect_and_grad(self, *, geometry: bool = False) -> Any: ...

    def replace(self, **updates: Any) -> "VState": ...
