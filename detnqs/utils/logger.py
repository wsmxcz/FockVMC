from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

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
        self.every = max(1, int(every))
        self.keys = (
            tuple(keys)
            if keys is not None
            else (
                "step",
                "energy",
                "eloc_var",
                "ess_frac",
                "acceptance_rate",
                "alpha",
            )
        )
        self.verbose = bool(verbose)

        self._cols: tuple[str, ...] | None = None

        if self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.file.write_text("", encoding="utf-8")

    def add(self, record: Mapping[str, Any]) -> None:
        """Append one step record."""
        rec = {key: np.asarray(value).item() for key, value in record.items()}
        step = int(rec["step"])

        if self.file is not None:
            with self.file.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        if self.verbose and step % self.every == 0:
            self._print(rec)

    def _print(self, rec: Mapping[str, Any]) -> None:
        cols = tuple(key for key in self.keys if key in rec)
        if not cols:
            return

        widths = []
        for key in cols:
            if key == "energy":
                width = 15
            elif key.startswith("time_"):
                width = 9
            elif (
                key in {"step", "outer", "acceptance_rate"}
                or key.startswith("n_")
                or key.endswith("_frac")
            ):
                width = 8
            else:
                width = 13
            widths.append(max(len(key), width))

        if self._cols != cols:
            self._cols = cols
            print("  ".join(k.rjust(w) for k, w in zip(cols, widths, strict=True)))
            print("  ".join("-" * w for w in widths))

        cells = tuple(self._format(key, rec[key]) for key in cols)
        print("  ".join(v.rjust(w) for v, w in zip(cells, widths, strict=True)))

    @staticmethod
    def _format(key: str, value: Any) -> str:
        x = float(value)

        if not np.isfinite(x):
            return str(x)
        if key in {"step", "outer"} or key.startswith("n_"):
            return str(int(round(x)))
        if key.startswith("time_"):
            return f"{x:.3f}s"
        if key == "acceptance_rate" or key.endswith("_frac"):
            return f"{100.0 * x:.3f}%"
        if key == "eloc_var" or key == "w_max" or key.endswith("_var"):
            return f"{x:.3e}"
        if key in {"alpha", "beta"}:
            return f"{x:.3f}"

        return f"{x:.6f}"
