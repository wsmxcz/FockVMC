from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class Timer:
    """Accumulate named wall-clock blocks when profiling is enabled."""

    enabled: bool = True
    times: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        """Time one named block; call block_until_ready inside JAX blocks."""
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + perf_counter() - start

    def stats(self) -> dict[str, float]:
        """Return `time_<name>` scalars for logged profile records."""
        return {f"time_{name}": float(value) for name, value in self.times.items()}
