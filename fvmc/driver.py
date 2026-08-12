from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
import optax

from .utils import Timer, checkpoint as ckpt


@dataclass(slots=True)
class VMC:
    """Variational Monte Carlo optimization driver."""

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
    ) -> Self:
        optimizer = optax.with_extra_args_support(optimizer)
        return cls(
            state=state,
            optimizer=optimizer,
            opt_state=optimizer.init(state.params),
            geometry=geometry,
        )

    def step(
        self,
        *,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
    ) -> dict[str, float | int]:
        """Apply one update and return the pre-update scalar record."""
        timer = Timer(timing=profile)
        with timer("step"):
            state, rec, grad, geom = self.state.expect(
                grad=True,
                geometry=self.geometry,
                obs=obs,
                timer=timer,
            )

            with timer("optimizer"):
                updates, self.opt_state = self.optimizer.update(
                    grad,
                    self.opt_state,
                    state.params,
                    geometry=geom,
                )
                params = optax.apply_updates(state.params, updates)
                if timer.timing:
                    jax.block_until_ready(params)

            self.state = state.replace(params=params)

        self.step_count += 1
        rec.update(timer.stats())
        rec["step"] = self.step_count
        return rec

    def run(
        self,
        steps: int,
        *,
        obs: Mapping[str, Any] | None = None,
        log: Callable[[Mapping[str, float | int]], None] | None = None,
        profile: bool = False,
        checkpoint: str | Path | None = None,
        checkpoint_every: int = 0,
    ) -> dict[str, float | int]:
        rec: dict[str, float | int] = {}
        for _ in range(steps):
            rec = self.step(obs=obs, profile=profile)

            if log is not None:
                log(rec)

            if (
                checkpoint is not None
                and checkpoint_every > 0
                and self.step_count % checkpoint_every == 0
            ):
                self.save(str(checkpoint).format(step=self.step_count))

        return rec

    def save(self, file: str | Path) -> Path:
        return ckpt.save(
            file,
            {
                "step": self.step_count,
                "state": self.state.state_dict(),
                "opt_state": self.opt_state,
            },
        )

    def load(self, file: str | Path) -> Self:
        data = ckpt.load(file)
        self.state = self.state.load_state(data["state"])
        self.opt_state = data["opt_state"]
        self.step_count = int(data["step"])
        return self
