from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


class Logger:
    """Write flat step records to JSONL and stdout."""

    def __init__(
        self,
        file: str | Path | None = None,
        *,
        every: int = 1,
        keys: Iterable[str] | None = None,
        verbose: bool = True,
        append: bool = False,
    ) -> None:
        self.file = None if file is None else Path(file)
        self.every = max(1, every)
        self.keys = (
            tuple(keys)
            if keys is not None
            else (
                "step",
                "energy",
                "eloc_var",
                "ess_frac",
                "unique_eff",
                "acceptance_rate",
                "alpha",
            )
        )
        self.verbose = verbose

        self._cols: tuple[str, ...] | None = None

        if self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.file.write_text("", encoding="utf-8")

    def __call__(self, record: Mapping[str, float | int]) -> None:
        """Append one step record."""
        step = record["step"]

        if self.file is not None:
            with self.file.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        if not self.verbose or step % self.every:
            return

        cols = tuple(key for key in self.keys if key in record)
        if not cols:
            return

        widths = []
        for key in cols:
            if key == "energy":
                width = 15
            elif key.startswith("time_"):
                width = 9
            elif (
                key in {"step", "outer"}
                or key.startswith("n_")
                or key.endswith(("_frac", "_rate", "_eff"))
            ):
                width = 8
            else:
                width = 13
            widths.append(max(len(key), width))

        if self._cols != cols:
            self._cols = cols
            header = (
                key.rjust(width)
                for key, width in zip(cols, widths, strict=True)
            )
            print("  ".join(header))
            print("  ".join("-" * w for w in widths))

        cells = []
        for key in cols:
            x = float(record[key])
            if not np.isfinite(x):
                value = str(x)
            elif key in {"step", "outer"} or key.startswith("n_"):
                value = str(int(round(x)))
            elif key.startswith("time_"):
                value = f"{x:.3f}s"
            elif key.endswith(("_frac", "_rate", "_eff")):
                value = f"{100.0 * x:.3f}%"
            elif key == "eloc_var" or key == "w_max" or key.endswith("_var"):
                value = f"{x:.3e}"
            elif key in {"alpha", "beta"}:
                value = f"{x:.3f}"
            else:
                value = f"{x:.6f}"
            cells.append(value)

        row = (
            cell.rjust(width)
            for cell, width in zip(cells, widths, strict=True)
        )
        print("  ".join(row))
