"""Plot convergence data from local VMC logs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    path = Path("fe2s2.jsonl")
    benchmark = -116.6056091

    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    keys = ("energy", "eloc_var", "accept", "alpha", "s2", "time_total")
    data = {
        key: np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
        for key in keys
    }
    step = np.asarray(
        [row.get("step", i + 1) for i, row in enumerate(rows)],
        dtype=float,
    )
    energy = data["energy"]

    finite = energy[np.isfinite(energy)]
    print(f"{path}: {len(rows)} records")
    if finite.size:
        print(f"final energy: {finite[-1]:.8f}")
        print(f"final error : {finite[-1] - benchmark:+.8f}")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.0),
        layout="constrained",
    )
    axes = axes.ravel()
    alpha_axis = axes[3].twinx()
    color = "#347ED8"

    axes[0].plot(step, energy, color=color, label=path.stem)
    axes[1].plot(step, np.abs(energy - benchmark), color=color)
    axes[2].plot(step, data["eloc_var"], color=color)
    axes[3].plot(step, data["accept"], color=color)
    alpha_axis.plot(step, data["alpha"], color=color, linestyle="--")
    axes[4].plot(step, data["s2"], color=color)
    axes[5].plot(step, np.cumsum(data["time_total"]), color=color)

    axes[0].axhline(
        benchmark,
        color="#1E2433",
        linestyle="--",
        label="benchmark",
    )
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    labels = (
        "energy",
        "energy error",
        "variance",
        "accept",
        r"$S^2$",
        "wall time",
    )
    for axis, label in zip(axes, labels, strict=True):
        axis.set_xlabel("step")
        axis.set_ylabel(label)
        axis.grid(True, color="#F4F7FB")

    alpha_axis.set_ylabel("alpha")
    axes[0].legend(frameon=False)
    figure.savefig("convergence.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
