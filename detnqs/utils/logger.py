from __future__ import annotations

"""Scalar history logger."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

COLORS = {
    "ref1": "#2B2D42",
    "ref2": "#5C677D",
    "ref3": "#8D99AE",
    "method1": "#4A809D",
    "method2": "#5FA89B",
    "method3": "#8EAF74",
    "method4": "#E2AF5E",
    "method5": "#E38562",
    "method6": "#D67B8C",
    "method7": "#A38CB4",
    "method8": "#7FA1C3",
    "aux": "#A2C9E4",
    "bar": "#EAECEF",
    "bar_edge": "#C4CCD3",
}

COUNT_KEYS = {"step", "outer"}
FIXED_KEYS = {"energy", "loss"}
SCI_KEYS = {
    "error",
    "variance",
    "sr_shift",
    "sr_residual",
    "sr_step_norm",
    "sr_cond",
}
RATIO_KEYS = {"accept", "ess_frac", "forward_frac", "unique_frac"}
FLAG_KEYS = {"sr_fallback"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    arr = np.asarray(value)
    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, complex):
            return {"real": float(item.real), "imag": float(item.imag)}
        return item
    if arr.size <= 16:
        return arr.tolist()
    return {"shape": list(arr.shape)}


def _format_quantity(key: str, value: Any) -> str:
    if value is None:
        return ""

    arr = np.asarray(value)
    if arr.ndim != 0 or np.iscomplexobj(arr):
        return str(value)

    x = float(arr)
    if not np.isfinite(x):
        return str(x)

    if key in FLAG_KEYS:
        return "yes" if bool(round(x)) else "no"
    if key in COUNT_KEYS or key.startswith("n_"):
        return str(int(round(x)))
    if key.startswith("time_"):
        return f"{x:.3f}s"
    if key in RATIO_KEYS or key.endswith("_frac"):
        return f"{100.0 * x:.2f}%"
    if key == "alpha":
        return f"{x:.3f}"
    if key in SCI_KEYS or key.startswith("sr_"):
        return f"{x:.3e}"
    if key in FIXED_KEYS:
        return f"{x:.10f}"
    return f"{x:.6g}"


@dataclass(slots=True)
class Logger:
    """Record scalar statistics, print aligned summaries, and write JSONL."""

    file: str | Path | None = None
    every: int = 1
    keys: Iterable[str] | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    _cols: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _head: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.every = max(1, int(self.every))
        self.keys = None if self.keys is None else tuple(str(k) for k in self.keys)
        if self.file is not None:
            path = Path(self.file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
            self.file = path

    def __call__(self, step: int, stats: Mapping[str, Any], driver: Any = None) -> bool:
        del driver
        self.add(step, stats)
        return False

    def add(self, step: int, stats: Mapping[str, Any]) -> dict[str, Any]:
        rec = {"step": int(step)}
        rec.update({str(k): _jsonable(v) for k, v in stats.items() if k != "step"})
        self.records.append(rec)

        if self.file is not None:
            with Path(self.file).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

        if int(step) % self.every == 0:
            self.print(rec)
        return rec

    def print(self, rec: Mapping[str, Any] | None = None) -> None:
        rec = self.records[-1] if rec is None and self.records else rec
        if rec is None:
            return

        if self._cols is None:
            if self.keys is not None:
                self._cols = tuple(self.keys)
            else:
                cols = []
                for key, value in rec.items():
                    if value is None:
                        continue
                    arr = np.asarray(value)
                    if arr.ndim == 0 and not np.iscomplexobj(arr):
                        cols.append(key)
                self._cols = tuple(cols)

        cols = self._cols
        cells = [_format_quantity(k, rec.get(k)) for k in cols]
        widths = [max(len(k), len(v), 8) for k, v in zip(cols, cells, strict=True)]

        if not self._head:
            print("  ".join(k.rjust(width) for k, width in zip(cols, widths, strict=True)))
            print("  ".join("-" * width for width in widths))
            self._head = True
        print("  ".join(v.rjust(width) for v, width in zip(cells, widths, strict=True)))

    def save(self, file: str | Path | None = None) -> Path:
        if file is None and self.file is None:
            raise ValueError("file must be provided")
        path = Path(self.file if file is None else file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        return path

    def array(self, key: str) -> np.ndarray:
        values = []
        for rec in self.records:
            value = rec.get(key)
            if value is None:
                values.append(np.nan)
                continue

            arr = np.asarray(value)
            values.append(
                float(arr) if arr.ndim == 0 and not np.iscomplexobj(arr) else np.nan
            )
        return np.asarray(values, dtype=float)

    def plot(
        self,
        y: str,
        *,
        x: str = "step",
        benchmark: float | None = None,
        ax: Any = None,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 3.2))

        xv = self.array(x)
        yv = self.array(y)
        ylabel = y
        if benchmark is not None:
            yv = np.abs(yv - float(benchmark))
            ylabel = f"|{y} - ref|"
            keep = np.isfinite(xv) & np.isfinite(yv) & (yv > 0)
            xv, yv = xv[keep], yv[keep]
            ax.set_yscale("log")

        ax.plot(xv, yv, color=COLORS["method1"], linewidth=1.6)
        ax.set_xlabel(x)
        ax.set_ylabel(ylabel)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(color=COLORS["bar"], linestyle="-", linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["bar_edge"])
        return ax.figure, ax
