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
            state, energy, grad, stats, geometry = self.state.expect_and_grad(
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
                    stats=stats,
                )
                params = optax.apply_updates(state.params, updates)
                jax.block_until_ready(params)

            self.state = state.replace(params=params)

        rec = dict(stats)
        rec["step"] = float(self.step_count + 1)
        rec.update(stats_util.update(grad, updates))

        opt_states = jax.tree.leaves(
            self.opt_state,
            is_leaf=lambda x: isinstance(getattr(x, "stats", None), Mapping),
        )
        for opt_state in opt_states:
            opt_stats = getattr(opt_state, "stats", None)
            if isinstance(opt_stats, Mapping):
                for key, value in opt_stats.items():
                    arr = jnp.asarray(jax.device_get(value))
                    if arr.ndim == 0 and not jnp.issubdtype(
                        arr.dtype,
                        jnp.complexfloating,
                    ):
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
        checkpoint: str | Path | None = None,
        checkpoint_every: int = 0,
    ) -> dict[str, float]:
        rec: dict[str, float] = {}
        checkpoint_every = int(checkpoint_every)

        for _ in range(int(steps)):
            rec = self.step(obs=obs, profile=profile)

            if logger is not None:
                logger.add(rec)

            if checkpoint is not None and checkpoint_every > 0:
                step = int(rec["step"])
                if step % checkpoint_every == 0:
                    self.save(str(checkpoint).format(step=step))

        return rec

    def save(self, file: str | Path) -> Path:
        return checkpoint.save(
            file,
            {
                "step": self.step_count,
                "state": self.state._checkpoint(),
                "opt_state": self.opt_state,
            },
        )

    def load(self, file: str | Path) -> Self:
        data = checkpoint.load(file)
        self.state = self.state._restore(data["state"])
        self.opt_state = data["opt_state"]
        self.step_count = int(data["step"])
        return self
