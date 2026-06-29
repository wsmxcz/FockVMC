from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ..model import Model


class VState(Protocol):
    """Variational-state protocol consumed by the VMC driver."""

    model: Model
    params: Any

    def expect(
        self,
        *,
        obs: Any = None,
        profile: bool = False,
        data: bool = False,
    ) -> Any: ...

    def expect_and_grad(
        self,
        *,
        geometry: bool = False,
        obs: Any = None,
        profile: bool = False,
    ) -> Any: ...

    def save(self, file: str | Path) -> Path: ...

    def load(self, file: str | Path) -> "VState": ...

    def replace(self, **updates: Any) -> "VState": ...
