from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import optax

Callback = Callable[[int, dict[str, float], "VMC"], bool | None]


@dataclass(slots=True)
class VMC:
    """Minimal VMC driver.

    The driver owns only optimizer state and the training loop.

    The variational state owns:
        - Hamiltonian physics,
        - sampling state,
        - estimator state,
        - local variational geometry.

    The optimizer owns:
        - gradient/update transformations,
        - optimizer state.

    The update convention follows Optax:

        updates, opt_state = optimizer.update(...)
        params = optax.apply_updates(params, updates)

    Geometry-aware DetNQS optimizers receive geometry as an Optax extra arg.
    Ordinary Optax transforms ignore it when wrapped with extra-args support.
    """

    state: Any
    optimizer: optax.GradientTransformationExtraArgs
    opt_state: Any
    step_count: int = 0
    geometry: bool = True

    @classmethod
    def init(
        cls,
        state: Any,
        optimizer: optax.GradientTransformation,
        *,
        geometry: bool = True,
    ) -> "VMC":
        optimizer = optax.with_extra_args_support(optimizer)
        return cls(
            state=state,
            optimizer=optimizer,
            opt_state=optimizer.init(state.params),
            geometry=bool(geometry),
        )

    def step(self) -> dict[str, float]:
        """Run one optimization step."""

        state, loss, grad, stats, geometry = self.state.expect_and_grad(
            geometry=self.geometry,
        )

        updates, self.opt_state = self.optimizer.update(
            grad,
            self.opt_state,
            state.params,
            geometry=geometry,
            value=loss,
            stats=stats,
        )

        params = optax.apply_updates(state.params, updates)
        self.state = state.replace(params=params)

        out = dict(stats)
        out["loss"] = float(loss)
        out["step"] = float(self.step_count)

        self.step_count += 1
        return out

    def run(
        self,
        steps: int,
        callbacks: Sequence[Callback] = (),
    ) -> dict[str, float]:
        """Run several optimization steps."""

        stats: dict[str, float] = {}

        for _ in range(int(steps)):
            stats = self.step()
            step = self.step_count - 1

            for callback in callbacks:
                if callback(step, stats, self):
                    return stats

        return stats

    def reset_optimizer(self) -> None:
        """Reset optimizer state."""

        self.opt_state = self.optimizer.init(self.state.params)