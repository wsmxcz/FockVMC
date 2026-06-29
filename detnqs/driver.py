from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
import optax

from .utils import Timer, checkpoint, stats as stats_util
from .utils.logger import Logger


@dataclass(slots=True)
class VMC:
    """Minimal variational Monte Carlo driver.

    The driver owns iteration, optimizer state, and checkpointable numerical
    state. Physics remains in `state`; SR details remain in the optimizer.
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
    ) -> Self:
        """Create a driver and initialize the Optax state from parameters."""
        optimizer = optax.with_extra_args_support(optimizer)
        return cls(
            state=state,
            optimizer=optimizer,
            opt_state=optimizer.init(state.params),
            geometry=bool(geometry),
        )

    def step(
        self,
        *,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
    ) -> dict[str, float]:
        """Run one optimization step and return a flat scalar record."""
        timer = Timer(enabled=profile)
        with timer("total"):
            state, energy, grad, out, geometry = self.state.expect_and_grad(
                geometry=self.geometry,
                obs=obs,
                profile=profile,
            )
            jax.block_until_ready(grad)

            with timer("optimizer"):
                updates, self.opt_state = self.optimizer.update(
                    grad,
                    self.opt_state,
                    state.params,
                    geometry=geometry,
                    value=energy,
                    stats=out,
                )
                params = optax.apply_updates(state.params, updates)
                jax.block_until_ready(params)

            self.state = state.replace(params=params)

        rec = dict(out)
        rec["step"] = float(self.step_count)
        rec.update(stats_util.update(grad, updates))
        rec.update(stats_util.collect(self.opt_state))
        if profile:
            rec.update(timer.stats())

        self.step_count += 1
        return rec

    def run(
        self,
        steps: int,
        *,
        obs: Mapping[str, Any] | None = None,
        logger: Logger | None = None,
        callbacks: Sequence[
            Callable[[int, dict[str, float], "VMC"], bool | None]
        ] = (),
        verbose: int = 1,
        profile: bool = False,
    ) -> dict[str, float]:
        """Run several steps; logging and callbacks are user-level hooks."""
        if logger is None and int(verbose) > 0:
            logger = Logger(verbose=int(verbose))

        rec: dict[str, float] = {}
        for _ in range(int(steps)):
            rec = self.step(obs=obs, profile=profile)
            step = self.step_count - 1

            if logger is not None:
                logger(step, rec, self)
            for callback in callbacks:
                if callback(step, rec, self):
                    return rec
        return rec

    def reset_optimizer(self) -> None:
        """Reset optimizer state while keeping the current wavefunction."""
        self.opt_state = self.optimizer.init(self.state.params)

    def save(self, file: str | Path) -> Path:
        """Save optimizer and variational-state numerical state."""
        state_data = {"params": self.state.params}
        if hasattr(self.state, "sampler_state"):
            state_data["sampler_state"] = {
                "key": self.state.sampler_state.key,
                "x": self.state.sampler_state.x,
                "logabs": self.state.sampler_state.logabs,
            }
        if hasattr(self.state, "chains"):
            state_data["chains"] = self.state.chains

        data = {
            "step": self.step_count,
            "state": state_data,
            "opt_state": self.opt_state,
        }
        return checkpoint.save(file, data)

    def load(self, file: str | Path) -> Self:
        """Restore numerical driver state into an explicitly rebuilt object."""
        data = checkpoint.load(file)
        state_data = data["state"]
        updates = {"params": state_data["params"]}
        if "sampler_state" in state_data and hasattr(self.state, "sampler_state"):
            saved = state_data["sampler_state"]
            updates["sampler_state"] = type(self.state.sampler_state)(
                key=jax.device_put(saved["key"]),
                x=saved["x"],
                logabs=saved["logabs"],
            )
        if "chains" in state_data and hasattr(self.state, "chains"):
            updates["chains"] = state_data["chains"]
        self.state = self.state.replace(**updates)
        self.opt_state = data["opt_state"]
        self.step_count = int(data["step"])
        return self
