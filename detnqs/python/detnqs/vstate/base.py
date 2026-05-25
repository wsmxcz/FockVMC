from __future__ import annotations

from typing import Any, Protocol

from ..model.base import Model

Tree = Any


class VState(Protocol):
    """Variational-state protocol consumed by the driver."""

    model: Model
    params: Tree

    def expect(self) -> tuple["VState", dict[str, float]]: ...

    def expect_and_grad(self, *, geometry: bool = False) -> Any: ...

    def replace(self, **updates: Any) -> "VState": ...