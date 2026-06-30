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
        verbose: int = 1,
        append: bool = False,
    ) -> None:
        self.file = None if file is None else Path(file)
        self.every = max(1, int(every))
        self.keys = None if keys is None else tuple(str(k) for k in keys)
        self.verbose = max(0, int(verbose))

        self._cols: tuple[str, ...] | None = None
        self._widths: tuple[int, ...] | None = None
        self._head = False

        if self.file is not None:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            if not append:
                self.file.write_text("", encoding="utf-8")

    def add(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one step record."""
        rec = {str(k): self._json(v) for k, v in record.items()}
        step = int(rec["step"])

        if self.file is not None:
            with self.file.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        rec,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        if self.verbose and step % self.every == 0:
            self._print(rec)

        return rec

    def _print(self, rec: Mapping[str, Any]) -> None:
        cols = self._select(rec)
        if not cols:
            return

        if self._cols != cols:
            self._cols = cols
            self._widths = None
            self._head = False

        cells = tuple(self._format(k, rec[k]) for k in cols)
        widths = tuple(
            max(self._width(k), len(v))
            for k, v in zip(cols, cells, strict=True)
        )

        self._widths = (
            widths
            if self._widths is None
            else tuple(max(a, b) for a, b in zip(self._widths, widths, strict=True))
        )
        widths = self._widths

        if not self._head:
            print("  ".join(k.rjust(w) for k, w in zip(cols, widths, strict=True)))
            print("  ".join("-" * w for w in widths))
            self._head = True

        print("  ".join(v.rjust(w) for v, w in zip(cells, widths, strict=True)))

    def _select(self, rec: Mapping[str, Any]) -> tuple[str, ...]:
        if self.keys is not None:
            return tuple(k for k in self.keys if k in rec)

        if self.verbose == 1:
            return tuple(
                k
                for k in ("step", "energy", "eloc_var", "ess_frac", "accept", "alpha")
                if k in rec
            )

        if self.verbose == 2:
            return tuple(
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
                    "step_scale",
                    "sr_force",
                    "sr_damp",
                    "n_forward",
                )
                if k in rec
            )

        return tuple(
            k
            for k, v in rec.items()
            if np.asarray(v).ndim == 0 and not np.iscomplexobj(np.asarray(v))
        )

    @staticmethod
    def _format(key: str, value: Any) -> str:
        arr = np.asarray(value)

        if arr.ndim != 0 or np.iscomplexobj(arr):
            return str(value)

        x = float(arr)

        if not np.isfinite(x):
            return str(x)
        if key == "step" or key.startswith("n_"):
            return str(int(round(x)))
        if key.startswith("time_"):
            return f"{x:.3f}s"
        if key == "accept" or key.endswith("_frac"):
            return f"{100.0 * x:.2f}%"
        if key in {"energy", "energy_se"}:
            return f"{x:.8f}"

        return f"{x:.6f}"

    @staticmethod
    def _width(key: str) -> int:
        if key == "step" or key.startswith("n_"):
            return max(len(key), 8)
        if key.startswith("time_"):
            return max(len(key), 9)
        if key == "accept" or key.endswith("_frac"):
            return max(len(key), 8)
        if key in {"energy", "energy_se"}:
            return max(len(key), 15)
        return max(len(key), 13)

    @staticmethod
    def _json(value: Any) -> Any:
        if isinstance(value, str | int | float | bool) or value is None:
            return value

        if isinstance(value, Mapping):
            return {str(k): Logger._json(v) for k, v in value.items()}

        if isinstance(value, list | tuple):
            return [Logger._json(v) for v in value]

        arr = np.asarray(value)

        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, complex):
                return {"real": float(item.real), "imag": float(item.imag)}
            return item

        if arr.size <= 16:
            return arr.tolist()

        return {"shape": list(arr.shape)}
