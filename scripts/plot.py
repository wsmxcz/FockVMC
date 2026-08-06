"""Plot convergence data from local VMC logs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    paths = (
        Path("fe2s2/fe2s2_GBackflow.jsonl"),
        Path("fe2s2/fe2s2_SBackflow.jsonl"),
        Path("fe2s2/fe2s2_PBackflow.jsonl"),
    )
    colors = ("#EE4B96", "#347ED8", "#E8BB32")
    benchmark = -116.6056091
    runs = []

    for path, color in zip(paths, colors, strict=True):
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]

        keys = ("energy", "eloc_var", "accept", "alpha", "s2", "time_total")
        data = {
            key: np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
            for key in keys
        }
        step = np.asarray(
            [row.get("step", i) for i, row in enumerate(rows)],
            dtype=float,
        )
        energy = data["energy"]
        runs.append(
            {
                "label": path.stem,
                "color": color,
                "step": step,
                "energy": energy,
                "error": np.abs(energy - benchmark),
                "variance": data["eloc_var"],
                "accept": data["accept"],
                "alpha": data["alpha"],
                "s2": data["s2"],
                "time": np.cumsum(data["time_total"]),
            }
        )

        finite = energy[np.isfinite(energy)]
        print(f"{path}: {len(rows)} records")
        if finite.size:
            print(f"  final energy: {finite[-1]:.8f}")
            print(f"  final error: {finite[-1] - benchmark:+.8f}")

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(10.5, 6.0),
        layout="constrained",
    )
    axes = axes.ravel()
    alpha_axis = axes[3].twinx()

    for run in runs:
        color = run["color"]
        step = run["step"]
        axes[0].plot(step, run["energy"], color=color, label=run["label"])
        axes[1].plot(step, run["error"], color=color)
        axes[2].plot(step, run["variance"], color=color)
        axes[3].plot(step, run["accept"], color=color)
        alpha_axis.plot(step, run["alpha"], color=color, linestyle="--")
        axes[4].plot(step, run["s2"], color=color)
        axes[5].plot(step, run["time"], color=color)

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
