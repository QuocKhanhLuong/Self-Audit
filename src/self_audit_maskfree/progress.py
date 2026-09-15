"""Compact tqdm display plus a detailed, non-authoritative JSONL journal."""
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

    The default console is one progress bar and one summary per epoch. Detailed
    events stay in JSONL; MASKFREE_PROGRESS=verbose restores diagnostic output.
    """
    enabled = True

    def __init__(self, *, stream=None, heartbeat_seconds=None, interval_seconds=5.0, mode=None):
        self.stream = stream if stream is not None else sys.stderr
        self.mode = mode or os.environ.get("MASKFREE_PROGRESS", "compact")
        if self.mode not in {"compact", "verbose"}:
            raise ValueError("MASKFREE_PROGRESS must be compact or verbose")
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
        self._bar = None
        self._bar_kind = None
        self._bar_owner = None
        self._last_refresh = 0.0
        self._bar_started = 0.0
        self._loss_postfix = ""

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
            self._close_bar()
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
        if self.mode == "compact":
            self._compact_event(payload)
        elif payload["event"] in {"metrics", "epoch.summary"}:
            self.stream.write(self._render_dashboard(payload))
        else:
            # JSON values keep paths/linebreaks unambiguous in captured GUI logs.
            fields = " ".join(f"{key}={json.dumps(value, ensure_ascii=True)}"
                              for key, value in payload.items() if key not in {"event", "timestamp"})
            self.stream.write(f"[maskfree {payload['timestamp']}] {payload['event']} {fields}\n")
        self.stream.flush()

    def _open_bar(self, description, *, kind, total=None, initial=0, owner=None):
        from tqdm import tqdm
        self._close_bar()
        self._bar_kind, self._bar_owner = kind, owner
        self._bar_started = time.monotonic()
        self._bar = tqdm(total=total, initial=initial, desc=description, file=self.stream,
                         unit="batch" if kind == "epoch" else "file", ascii=True,
                         mininterval=1.0, dynamic_ncols=True, leave=True, disable=False,
                         delay=0 if kind == "epoch" else 0.5,
                         bar_format="{desc} [{elapsed}{postfix}]" if kind == "phase" else None)

    def _close_bar(self):
        if self._bar is not None:
            self._bar.close()
        self._bar = self._bar_kind = self._bar_owner = None

    def _line(self, text):
        if self._bar is not None:
            self._bar.write(text, file=self.stream)
        else:
            self.stream.write(text + "\n")

    @staticmethod
    def _number(value):
        return "--" if value is None else f"{value:.4f}"

    def _advance_epoch(self, payload):
        if self._bar is None or self._bar_kind != "epoch":
            return
        completed = payload.get("batch_cursor", payload.get("batch", self._bar.n))
        self._bar.update(max(0, completed - self._bar.n))
        metrics = payload.get("metrics", {})
        self._loss_postfix = " ".join(
            f"{short}={self._number(metrics[key])}" for short, key in (
                ("P", "producer/loss"), ("U", "student_no_audit/loss"), ("A", "student_audited/loss")
            ) if key in metrics)
        self._refresh_bar()

    def _refresh_bar(self, *, force=False):
        if self._bar is None:
            return
        now = time.monotonic()
        if self._bar_kind != "epoch" and now - self._bar_started < 0.5:
            return
        if not force and now - self._last_refresh < 1.0:
            return
        self._last_refresh = now
        # Once explicitly refreshed, tqdm must also terminate the displayed
        # line on close, even if this phase has no counter updates.
        self._bar.delay = 0
        phase = ""
        if self._stack:
            frame = self._stack[-1]
            phase = f"{frame['name']} {now - frame['start']:.0f}s"
        postfix = " | ".join(part for part in (self._loss_postfix if self._bar_kind == "epoch" else "", phase) if part)
        self._bar.set_postfix_str(postfix, refresh=False)
        self._bar.refresh()

    def _compact_event(self, payload):
        event = payload["event"]
        if event == "epoch.start":
            self._loss_postfix = ""
            self._open_bar(f"{payload.get('dataset', '')} Epoch {payload['epoch']}/{payload['epochs']} Train",
                           kind="epoch", total=payload["batches"], initial=payload.get("resumed_from_batch", 0))
        elif event == "metrics":
            self._advance_epoch(payload)
        elif event == "epoch.summary":
            self._advance_epoch(payload)
            self._close_bar()
            producer, students = payload["producer"], payload["students"]
            audit = payload.get("audit", {})
            status = "" if payload.get("epoch_complete") else " PARTIAL"
            self._line(
                f"Epoch {payload['global_epoch'] + 1}/{payload.get('epochs', '?')}{status} | "
                f"time={payload['epoch_seconds']:.1f}s | "
                f"loss P/U/A={self._number(producer.get('producer/loss'))}/"
                f"{self._number(students.get('student_no_audit/loss'))}/"
                f"{self._number(students.get('student_audited/loss'))} | "
                f"select NLL={self._number(audit.get('select_nll_mean'))} | "
                "val Dice=-- (reference evaluation not configured)")
        elif event == "discovery.files_listed" and self._bar_kind == "inventory":
            self._bar.total = payload["files_total"]
        elif event == "stage.done" and payload.get("phase") == "discovery.header" and self._bar_kind == "inventory":
            self._bar.update(max(0, payload["file_index"] - self._bar.n))
        elif event == "run.start":
            self._line(f"[maskfree] {payload['dataset']} {payload['mode']} | device={payload['device']} | "
                       f"batch={payload['physical_batch']} x {payload['accumulation']} | epochs={payload['epochs']}")
        elif event in {"run.result", "preflight.result"}:
            self._line(f"[maskfree] {event}: {payload['status']} | report={payload.get('report', '--')}")
        elif event == "reports.ready":
            self._line(f"[maskfree] reports: {payload['paths']}")
        elif event in {"inventory.failed", "run.failed", "finalization.failed", "process.error"}:
            self._close_bar()
            reason = str(payload.get("error", payload.get("reason", ""))).split("\n", 1)[0][:180]
            diagnostic = payload.get("diagnostic_path", payload.get("failure_report", payload.get("path", "")))
            self._line(f"[maskfree] {event}: {reason}\n  details: {diagnostic or 'diagnostic journal'}")
        self._refresh_bar()

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
            if self._bar_kind == "inventory" and "files_read" in details:
                completed = details["files_read"] + details.get("files_skipped", 0)
                self._bar.update(max(0, completed - self._bar.n))
            self._refresh_bar()

    @contextmanager
    def stage(self, name, **details):
        start = time.monotonic()
        frame = {"name": name, "start": start, "details": details}
        with self._lock:
            parent_context = dict(self._context)
            self._stack.append(frame)
            if self.mode == "compact" and self._bar is None and len(self._stack) == 1:
                kind = "inventory" if name in {"inventory.discover", "data_discovery"} else "phase"
                self._open_bar(name, kind=kind, owner=frame)
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
                if self._bar_owner is frame:
                    self._close_bar()
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
            if self.mode == "compact":
                self._advance_epoch({**self._context, **details})
            now = time.monotonic()
            if not force and now - self._last_dashboard < self.interval:
                return
            self._last_dashboard = now
            self.event("metrics", **details)
