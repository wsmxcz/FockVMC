from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
import optax

from .utils import Timer, checkpoint, stats as stats_util
from .utils.logger import Logger


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
            geometry=bool(geometry),
        )

    def step(
        self,
        *,
        obs: Mapping[str, Any] | None = None,
        profile: bool = False,
    ) -> dict[str, float]:
        """Apply one update and return the pre-update scalar record."""
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
        rec["step"] = float(self.step_count + 1)
        rec.update(stats_util.update(grad, updates))

        opt_stats = getattr(self.opt_state, "stats", None)
        if isinstance(opt_stats, Mapping):
            for key, value in opt_stats.items():
                arr = jnp.asarray(jax.device_get(value))
                if arr.ndim == 0 and not jnp.issubdtype(arr.dtype, jnp.complexfloating):
                    rec[str(key)] = float(arr)

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
        profile: bool = False,
    ) -> dict[str, float]:
        rec: dict[str, float] = {}
        for _ in range(int(steps)):
            rec = self.step(obs=obs, profile=profile)
            if logger is not None:
                logger.add(rec)
        return rec

    def save(self, file: str | Path) -> Path:
        state = {"params": self.state.params}
        if hasattr(self.state, "sampler_state"):
            state["sampler_state"] = {
                "key": self.state.sampler_state.key,
                "x": self.state.sampler_state.x,
                "logabs": self.state.sampler_state.logabs,
            }
        if hasattr(self.state, "chains"):
            state["chains"] = self.state.chains
        if hasattr(self.state, "basis"):
            state["basis"] = self.state.basis

        return checkpoint.save(
            file,
            {
                "step": self.step_count,
                "state": state,
                "opt_state": self.opt_state,
            },
        )

    def load(self, file: str | Path) -> Self:
        data = checkpoint.load(file)
        saved = data["state"]
        updates = {"params": saved["params"]}

        if "sampler_state" in saved and hasattr(self.state, "sampler_state"):
            old = saved["sampler_state"]
            updates["sampler_state"] = type(self.state.sampler_state)(
                key=jax.device_put(old["key"]),
                x=old["x"],
                logabs=old["logabs"],
            )
        if "chains" in saved and hasattr(self.state, "chains"):
            updates["chains"] = saved["chains"]
        if "basis" in saved and hasattr(self.state, "basis"):
            basis = saved["basis"]
            updates["basis"] = basis
            if hasattr(self.state, "hmat"):
                updates["hmat"] = self.state.H.matrix(basis)

        self.state = self.state.replace(**updates)
        self.opt_state = data["opt_state"]
        self.step_count = int(data["step"])
        return self
