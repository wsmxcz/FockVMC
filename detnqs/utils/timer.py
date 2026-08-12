from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class Timer:
    """Accumulate named wall-clock blocks and work counts."""

    enabled: bool = True
    times: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def __call__(self, name: str, *, n: int = 0) -> Iterator[None]:
        """Time one named block; call block_until_ready inside JAX blocks."""
        if n:
            self.counts[name] = self.counts.get(name, 0) + int(n)
        if not self.enabled:
            yield
            return
        start = perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + perf_counter() - start

    def stats(self) -> dict[str, float | int]:
        """Return accumulated times and counts."""
        return {
            **{f"time_{name}": value for name, value in self.times.items()},
            **{f"n_{name}": value for name, value in self.counts.items()},
        }
