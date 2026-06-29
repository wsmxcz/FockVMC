from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    """Convert small numeric objects to readable JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list | tuple):
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


@dataclass(slots=True)
class Logger:
    """Write flat records to stdout and optional JSONL."""

    file: str | Path | None = None
    every: int = 1
    keys: Iterable[str] | None = None
    verbose: int = 1
    append: bool = False

    _cols: tuple[str, ...] | None = field(default=None, init=False, repr=False)
    _widths: tuple[int, ...] | None = field(default=None, init=False, repr=False)
    _head: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.every = max(1, int(self.every))
        self.verbose = max(0, int(self.verbose))
        self.keys = None if self.keys is None else tuple(str(k) for k in self.keys)
        if self.file is not None:
            path = Path(self.file)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self.append:
                path.write_text("", encoding="utf-8")
            self.file = path

    def __call__(
        self,
        step: int,
        values: Mapping[str, Any],
        driver: Any = None,
    ) -> bool:
        del driver
        self.add(step, values)
        return False

    def add(self, step: int, values: Mapping[str, Any]) -> dict[str, Any]:
        rec = {"step": int(step)}
        rec.update({str(k): _jsonable(v) for k, v in values.items() if k != "step"})

        if self.file is not None:
            with Path(self.file).open("a", encoding="utf-8") as fh:
                text = json.dumps(
                    rec,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                fh.write(text + "\n")

        if self.verbose > 0 and int(step) % self.every == 0:
            self.print(rec)
        return rec

    def print(self, rec: Mapping[str, Any]) -> None:
        if self.keys is not None:
            cols = tuple(k for k in self.keys if k in rec)
        elif self.verbose <= 0:
            cols = ()
        elif self.verbose == 1:
            cols = tuple(
                k
                for k in ("step", "energy", "eloc_var", "ess_frac", "accept", "dE_lin")
                if k in rec
            )
        elif self.verbose == 2:
            cols = tuple(
                k
                for k in (
                    "step",
                    "energy",
                    "energy_se",
                    "eloc_var",
                    "ess_frac",
                    "essu_frac",
                    "w_max",
                    "accept",
                    "unique_frac",
                    "dE_lin",
                    "update_norm",
                    "sr_force",
                    "sr_damp",
                    "n_forward",
                )
                if k in rec
            )
        else:
            cols_list = []
            for key, value in rec.items():
                if value is None:
                    continue
                arr = np.asarray(value)
                if arr.ndim == 0 and not np.iscomplexobj(arr):
                    cols_list.append(key)
            cols = tuple(cols_list)
        if not cols:
            return
        if self._cols != cols:
            self._cols = cols
            self._widths = None
            self._head = False

        cells = []
        for key in cols:
            value = rec.get(key)
            if value is None:
                cells.append("")
                continue

            arr = np.asarray(value)
            if arr.ndim != 0 or np.iscomplexobj(arr):
                cells.append(str(value))
                continue

            x = float(arr)
            if not np.isfinite(x):
                cells.append(str(x))
            elif key == "step" or key == "outer" or key.startswith("n_"):
                cells.append(str(int(round(x))))
            elif key.startswith("time_"):
                cells.append(f"{x:.3f}s")
            elif key == "accept" or key.endswith("_frac"):
                cells.append(f"{100.0 * x:.2f}%")
            elif key == "energy" or key == "energy_se":
                cells.append(f"{x:.8f}")
            else:
                cells.append(f"{x:.6f}")

        widths_list = []
        for key, value in zip(cols, cells, strict=True):
            if key == "step" or key == "outer" or key.startswith("n_"):
                width = max(len(key), 8)
            elif key.startswith("time_"):
                width = max(len(key), 9)
            elif key == "accept" or key.endswith("_frac"):
                width = max(len(key), 8)
            elif key == "energy" or key == "energy_se":
                width = max(len(key), 15)
            elif key in (
                "eloc_var",
                "s2",
                "s2_var",
                "sr_force",
                "sr_damp",
                "dE_lin",
                "update_norm",
                "w_max",
            ):
                width = max(len(key), 13)
            else:
                width = max(len(key), 14)
            widths_list.append(max(width, len(value)))
        widths = tuple(widths_list)
        if self._widths is None:
            self._widths = widths
        else:
            self._widths = tuple(
                max(old, new) for old, new in zip(self._widths, widths, strict=True)
            )
        widths = self._widths

        if not self._head:
            print(
                "  ".join(
                    k.rjust(width)
                    for k, width in zip(cols, widths, strict=True)
                )
            )
            print("  ".join("-" * width for width in widths))
            self._head = True
        print(
            "  ".join(
                v.rjust(width)
                for v, width in zip(cells, widths, strict=True)
            )
        )
