"""Live, flushed terminal telemetry; independent of torch, RNG and checkpoints.

Append-only text works inside GUI terminals and through ``tee`` (no ANSI cursor
control). A heartbeat reports the innermost active operation even before the
first batch completes. JSONL is diagnostic only, never a resume authority.
"""
from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class _QuietProgress:
    enabled = False

    @contextmanager
    def stage(self, name, **details):
        yield

    def event(self, name, **details):
        pass

    def update(self, **details):
        pass

    def dashboard(self, **details):
        pass


_CURRENT = _QuietProgress()


def current_progress():
    return _CURRENT


class TerminalProgress:
    """One process-wide reporter installed by a CLI, absent in library callers.

    Repeated short stages are throttled; first occurrence, errors, explicit
    events, and epoch summaries are immediate. The heartbeat always uses fresh
    context. It measures wall time, not GPU utilization or estimated completion.
    """
    enabled = True

    def __init__(self, *, stream=None, heartbeat_seconds=None, interval_seconds=5.0):
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_seconds = float(heartbeat_seconds if heartbeat_seconds is not None
                                       else os.environ.get("MASKFREE_HEARTBEAT_SECONDS", "10"))
        if not math.isfinite(self.heartbeat_seconds) or self.heartbeat_seconds <= 0:
            raise ValueError("MASKFREE_HEARTBEAT_SECONDS must be finite and positive")
        self.interval = interval_seconds
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._stack = []
        self._context = {}
        self._last_stage = {}
        self._last_dashboard = -float("inf")
        self._started = time.monotonic()
        self._journal = None
        self._previous = None
        self._thread = None

    def attach(self, path):
        """Append on resume; never rewrite the experiment's metric stream."""
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._journal is not None:
                self._journal.close()
            self._journal = path.open("a", encoding="utf-8", buffering=1)
            self.event("telemetry.attached", path=str(path))

    def __enter__(self):
        global _CURRENT
        self._previous, _CURRENT = _CURRENT, self
        self._thread = threading.Thread(target=self._heartbeat_loop, name="maskfree-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, kind, exc, tb):
        global _CURRENT
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if exc is not None:
            self.event("process.error", error_type=type(exc).__name__, error=str(exc))
        with self._lock:
            if self._journal is not None:
                self._journal.close()
                self._journal = None
        _CURRENT = self._previous

    def _write(self, payload):
        payload = _safe(payload)
        line = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        if self._journal is not None:
            self._journal.write(line + "\n")
            self._journal.flush()
        if payload["event"] in {"metrics", "epoch.summary"}:
            self.stream.write(self._render_dashboard(payload))
        else:
            # JSON values keep paths/linebreaks unambiguous in captured GUI logs.
            fields = " ".join(f"{key}={json.dumps(value, ensure_ascii=True)}"
                              for key, value in payload.items() if key not in {"event", "timestamp"})
            self.stream.write(f"[maskfree {payload['timestamp']}] {payload['event']} {fields}\n")
        self.stream.flush()

    @staticmethod
    def _render_dashboard(payload):
        def display(value):
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.5g}"
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

        rows = [f"[maskfree {payload['timestamp']}] {payload['event']} "
                f"dataset={payload.get('dataset')} epoch={payload.get('epoch')}/{payload.get('epochs')} "
                f"batch={payload.get('batch_cursor', payload.get('batch'))}/{payload.get('batches')}"]

        def group(name, values):
            pairs = [f"{k}={display(v)}" for k, v in values.items()]
            for i in range(0, len(pairs), 4):
                rows.append(f"  {name}: " + "  ".join(pairs[i:i + 4]))

        group("schedule", {key: payload[key] for key in (
            "lr", "label_ramp", "global_optimizer_steps", "component_steps", "epoch_complete",
            "physical_batch", "accumulation", "metric_scope") if key in payload})
        metrics = payload.get("metrics", {})
        if payload["event"] == "epoch.summary":
            metrics = {**payload.get("producer", {}), **payload.get("students", {})}
        for prefix in ("producer", "student_no_audit", "student_audited", "audit", "coverage"):
            group(prefix, {k.removeprefix(prefix + "/"): v for k, v in metrics.items()
                           if k.startswith(prefix + "/")})
        for name in ("audit", "coverage", "metric_counts", "class_occupancy", "batch_stage_seconds", "gpu_stats"):
            value = payload.get(name)
            if isinstance(value, dict):
                group(name, value)
            elif name == "gpu_stats":
                group("gpu_memory_bytes", {"available": False})
        group("timing", {key: payload[key] for key in ("batch_seconds", "epoch_seconds", "timing_note")
                         if key in payload})
        if payload.get("verification_status"):
            group("verification", {"status": payload["verification_status"]})
        return "\n".join(rows) + "\n"

    def event(self, name, **details):
        with self._lock:
            self._write({**self._context, **details, "event": name,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def update(self, **details):
        with self._lock:
            self._context.update(_safe(details))

    @contextmanager
    def stage(self, name, **details):
        start = time.monotonic()
        frame = {"name": name, "start": start, "details": details}
        with self._lock:
            parent_context = dict(self._context)
            self._stack.append(frame)
            visible = start - self._last_stage.get(name, -float("inf")) >= self.interval
            if visible:
                self._last_stage[name] = start
                self.event("stage.start", phase=name, **details)
        try:
            yield
        except BaseException as exc:
            self.event("stage.error", phase=name, **details, seconds=time.monotonic() - start,
                       error_type=type(exc).__name__, error=str(exc))
            raise
        else:
            elapsed = time.monotonic() - start
            if visible or elapsed >= self.interval:
                self.event("stage.done", phase=name, **details, seconds=elapsed)
        finally:
            with self._lock:
                self._stack.remove(frame)
                # File/frame updates belong to this operation. They must not
                # label a later model step with the last inventoried filename.
                self._context = parent_context

    def heartbeat(self):
        with self._lock:
            now = time.monotonic()
            details = {"process_seconds": now - self._started}
            if self._stack:
                frame = self._stack[-1]
                details.update(frame["details"])
                details.update(phase=frame["name"], phase_seconds=now - frame["start"],
                               phase_path=" > ".join(f["name"] for f in self._stack))
            self.event("heartbeat", **details)

    def _heartbeat_loop(self):
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat()

    def dashboard(self, *, force=False, **details):
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_dashboard < self.interval:
                return
            self._last_dashboard = now
            self.event("metrics", **details)
