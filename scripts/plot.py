"""Plot and compare convergence data from VMC logs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(y: np.ndarray, window: int = 11) -> np.ndarray:
    mask = np.isfinite(y)
    kernel = np.ones(window)
    value = np.convolve(np.where(mask, y, 0.0), kernel, mode="same")
    count = np.convolve(mask.astype(float), kernel, mode="same")
    return np.divide(value, count, out=np.full_like(y, np.nan), where=count > 0)


def main() -> None:
    runs = {
        "32768": Path("fe2s2.jsonl"),
        "8192": Path("fe2s2_eloc8192.jsonl"),
        # "1024": Path("fe2s2_eps1e-3.jsonl"),
        "1e-4": Path("fe2s2_eps1e-4.jsonl"),
        "1e-5": Path("fe2s2_eps1e-5.jsonl"),
        "1e-6": Path("fe2s2_eps1e-6.jsonl"),
    }
    benchmark = -116.6056091  # Fe2S2
    # benchmark = -327.248858  # Fe4S4
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.0, 6.0),
        layout="constrained",
    )
    axes = axes.ravel()
    alpha_axis = axes[3].twinx()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for (label, path), color in zip(runs.items(), colors, strict=False):
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
        energy_smooth = smooth(energy)
        error = 1e3 * np.abs(energy - benchmark)
        error_smooth = 1e3 * np.abs(energy_smooth - benchmark)

        finite = energy[np.isfinite(energy)]
        print(f"{label}: {path} ({len(rows)} records)")
        if finite.size:
            print(f"  final energy: {finite[-1]:.8f}")
            print(f"  final error : {1e3 * abs(finite[-1] - benchmark):.3f} mEh")

        # Energy
        axes[0].plot(step, energy, color=color, lw=0.6, alpha=0.18)
        axes[0].plot(
            step,
            energy_smooth,
            color=color,
            lw=1.6,
            label=label,
        )

        # Energy error
        axes[1].plot(step, error, color=color, lw=0.6, alpha=0.18)
        axes[1].plot(step, error_smooth, color=color, lw=1.6)

        # Local-energy variance
        axes[2].plot(
            step,
            data["eloc_var"],
            color=color,
            lw=0.6,
            alpha=0.18,
        )
        axes[2].plot(
            step,
            smooth(data["eloc_var"]),
            color=color,
            lw=1.6,
        )

        # Acceptance and tempering
        axes[3].plot(
            step,
            smooth(data["accept"]),
            color=color,
            lw=1.5,
        )
        alpha_axis.plot(
            step,
            smooth(data["alpha"]),
            color=color,
            lw=1.2,
            ls="--",
        )

        # Spin
        axes[4].plot(step, data["s2"], color=color, lw=0.6, alpha=0.18)
        axes[4].plot(
            step,
            smooth(data["s2"]),
            color=color,
            lw=1.6,
        )

        # Wall time
        axes[5].plot(
            step,
            np.nancumsum(data["time_total"]) / 60.0,
            color=color,
            lw=1.6,
        )

    axes[0].axhline(
        benchmark,
        color="0.2",
        lw=0.9,
        ls="--",
        label="benchmark",
    )

    axes[1].set_yscale("log")
    axes[2].set_yscale("log")

    labels = (
        r"$E$ ($E_h$)",
        r"$|E-E_{\mathrm{ref}}|$ (m$E_h$)",
        r"$\mathrm{Var}(E_{\mathrm{L}})$",
        "acceptance",
        r"$\langle S^2\rangle$",
        "wall time (min)",
    )

    for axis, label in zip(axes, labels, strict=True):
        axis.set_xlabel("optimization step")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="0.92", lw=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    alpha_axis.set_ylabel(r"$\alpha$")
    alpha_axis.spines["top"].set_visible(False)

    axes[0].legend(frameon=False)

    figure.savefig("convergence.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()