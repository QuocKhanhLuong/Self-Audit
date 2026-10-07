"""Optional W&B telemetry for the image-only v3 pipeline.

The v3 scientific artifacts are authoritative.  This adapter is deliberately
best-effort: a missing SDK, unavailable network, or SDK error becomes a local
warning and never changes the training/evaluation control flow.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _safe_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return json.dumps([_safe_value(item) for item in value], sort_keys=True)
    return str(value)


def flatten_payload(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten JSON-like values into W&B-safe namespaced scalar fields."""

    flattened: dict[str, Any] = {}

    def visit(key: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                visit(f"{key}/{child_key}" if key else str(child_key), child_value)
            return
        flattened[key] = _safe_value(item)

    visit(prefix, value)
    return flattened


class WandbV3Tracker:
    """Best-effort single-run W&B tracker owned by the v3 parent process."""

    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        project: str,
        entity: str | None,
        run_name: str,
        run_dir: str | Path | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        self.requested_enabled = bool(enabled)
        self.enabled = False
        self.mode = str(mode or "disabled")
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.run_dir = str(run_dir) if run_dir is not None else None
        self.warnings: list[str] = []
        self._wandb: Any = None
        self._run: Any = None
        self._finished = False

        if not self.requested_enabled or self.mode == "disabled":
            return

        try:
            import wandb  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import depends on runtime
            self._warn("import", exc)
            return

        kwargs: dict[str, Any] = {
            "project": self.project,
            "entity": self.entity,
            "name": self.run_name,
            "mode": self.mode,
            "dir": self.run_dir,
            "config": dict(config or {}),
        }
        try:
            self._wandb = wandb
            self._run = wandb.init(**kwargs)
            self.enabled = True
        except Exception as exc:
            self._warn("init", exc)
            self._wandb = None
            self._run = None

    def _warn(self, operation: str, exc: BaseException | str) -> None:
        message = str(exc)
        secret = os.environ.get("WANDB_API_KEY")
        if secret:
            message = message.replace(secret, "<redacted>")
        self.warnings.append(f"{operation}: {message}")

    def log(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self._wandb is None:
            return
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            cleaned.update(flatten_payload(str(key), value))
        if not cleaned:
            return
        try:
            self._wandb.log(cleaned)
        except Exception as exc:
            self._warn("log", exc)

    def log_history(self, stage: str, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            self.log({stage: row})

    def log_json(self, stage: str, path: str | Path, *, include_history: bool = True) -> None:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._warn(f"read {stage}", exc)
            return
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, Mapping)]
            if include_history:
                self.log_history(stage, rows)
        elif isinstance(payload, Mapping):
            history = payload.get("history")
            if isinstance(history, list):
                rows = [row for row in history if isinstance(row, Mapping)]
                if include_history:
                    self.log_history(stage, rows)
                payload = {key: value for key, value in payload.items() if key != "history"}
            self.log({stage: payload})
        else:
            self._warn(f"log {stage}", "JSON root must be an object or list")

    def set_summary(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            summary = getattr(self._run, "summary", None)
            if summary is None:
                return
            for key, value in flatten_payload("", payload).items():
                summary[key] = value
        except Exception as exc:
            self._warn("summary", exc)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._wandb is None or self._run is None:
            return
        try:
            self._wandb.finish()
        except Exception as exc:
            self._warn("finish", exc)
        finally:
            self._run = None
