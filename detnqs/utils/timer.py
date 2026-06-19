from __future__ import annotations

"""Minimal wall-clock timer.

Use this only to accumulate named wall-clock blocks. For JAX kernels, call
jax.block_until_ready(result) inside the timed block when an accurate device
execution time is needed.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class Timer:
    """Accumulate named wall-clock times in seconds."""

    times: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - t0)

    def add(self, name: str, value: float) -> None:
        self.times[name] = self.times.get(name, 0.0) + float(value)

    def stats(self) -> dict[str, float]:
        return {f"time_{name}": float(value) for name, value in self.times.items()}