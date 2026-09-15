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
        self._validation_operation = ""

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
        dynamic, columns, rows = True, None, None
        try:
            size = os.get_terminal_size(self.stream.fileno())
            if size.columns == 0 or size.lines <= 1:
                dynamic, columns, rows = False, size.columns or 120, max(size.lines, 24)
        except (OSError, AttributeError, ValueError):
            pass
        self._bar = tqdm(total=total, initial=initial, desc=description, file=self.stream,
                         unit="batch" if kind in {"epoch", "validation"} else "file", ascii=True,
                         mininterval=1.0, dynamic_ncols=dynamic, ncols=columns, nrows=rows, leave=True, disable=False,
                         delay=0 if kind in {"epoch", "validation"} else 0.5,
                         bar_format="{desc} [{elapsed}{postfix}]" if kind == "phase" else None)

    def _close_bar(self):
        if self._bar is not None:
            self._bar.close()
        self._bar = self._bar_kind = self._bar_owner = None
        self._validation_operation = ""

    def _line(self, text):
        if self._bar is not None:
            self._bar.write(text, file=self.stream)
        else:
            self.stream.write(text + "\n")

    @staticmethod
    def _number(value):
        if value is None:
            return "--"
        if isinstance(value, bool):
            return str(value)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return "--" if not math.isfinite(number) else f"{number:.4f}"

    @staticmethod
    def _seconds(value):
        if value is None:
            return "--"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"{value}s"
        return "--" if not math.isfinite(number) else f"{number:.1f}s"

    @staticmethod
    def _epoch_label(payload):
        """Return the human-facing one-based epoch and timeline length."""
        # A resumed run may retain an old one-based ``epoch`` in the shared
        # progress context. The explicit zero-based global counter is the
        # authoritative label whenever the caller supplies both fields.
        epoch = payload.get("global_epoch")
        if epoch is not None:
            try:
                epoch = int(epoch) + 1
            except (TypeError, ValueError):
                pass
        else:
            epoch = payload.get("epoch")
        return ("?" if epoch is None else epoch,
                payload.get("total_epochs", payload.get("epochs", "?")))

    @staticmethod
    def _metric(mapping, name, metric):
        if not isinstance(mapping, dict):
            return None
        if name:
            nested = mapping.get(name)
            if isinstance(nested, dict) and metric in nested:
                return nested.get(metric)
            return mapping.get(f"{name}/{metric}")
        return mapping.get(metric, mapping.get(f"producer/{metric}"))

    @staticmethod
    def _short_reason(reason, default):
        if reason is None or reason == "":
            reason = default
        return str(reason).split("\n", 1)[0][:180]

    def _advance_validation(self, payload):
        if self._bar is None or self._bar_kind != "validation":
            return
        completed = payload.get("batch", payload.get("batch_cursor", self._bar.n))
        try:
            completed = int(completed)
        except (TypeError, ValueError):
            completed = self._bar.n
        self._bar.update(max(0, completed - self._bar.n))
        operation = payload.get("operation")
        changed = operation is not None and str(operation) != self._validation_operation
        if operation is not None:
            self._validation_operation = str(operation)
        self._refresh_bar(force=changed)

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
        if self._bar_kind == "phase" and now - self._bar_started < 0.5:
            return
        if not force and now - self._last_refresh < 1.0:
            return
        self._last_refresh = now
        # Once explicitly refreshed, tqdm must also terminate the displayed
        # line on close, even if this phase has no counter updates.
        self._bar.delay = 0
        phase = ""
        if self._bar_kind == "validation":
            operation = self._validation_operation
            started = self._bar_started
            frames = [frame for frame in self._stack if frame["name"].startswith("validation.")]
            if frames:
                frame = frames[-1]
                started = frame["start"]
                frame_operation = frame["details"].get("operation")
                if frame_operation:
                    operation = frame_operation
                elif frame["name"] != "validation" or not operation:
                    operation = str(frame["name"]).removeprefix("validation.") or operation
            if operation:
                phase = f"{operation} {now - started:.0f}s"
        elif self._stack:
            frame = self._stack[-1]
            phase = f"{frame['name']} {now - frame['start']:.0f}s"
        postfix = " | ".join(part for part in (self._loss_postfix if self._bar_kind == "epoch" else "", phase) if part)
        self._bar.set_postfix_str(postfix, refresh=False)
        self._bar.refresh()

    def _compact_event(self, payload):
        event = payload["event"]
        if event == "epoch.start":
            self._loss_postfix = ""
            epoch, epochs = self._epoch_label(payload)
            self._open_bar(f"{payload.get('dataset', '')} Epoch {epoch}/{epochs} Train",
                           kind="epoch", total=payload.get("batches"),
                           initial=payload.get("resumed_from_batch", 0))
        elif event == "epoch.train_done":
            # Training and report-only validation have separate bars. The
            # summary is deliberately emitted after validation by the caller.
            if self._bar_kind == "epoch":
                self._close_bar()
        elif event == "validation.start":
            epoch, epochs = self._epoch_label(payload)
            self._open_bar(f"{payload.get('dataset', '')} Epoch {epoch}/{epochs} Val",
                           kind="validation", total=payload.get("batches", payload.get("units")))
            self._validation_operation = str(payload["operation"]) if payload.get("operation") else ""
            self._refresh_bar(force=True)
        elif event == "validation.batch":
            self._advance_validation(payload)
        elif event == "validation.end":
            if self._bar_kind == "validation":
                self._close_bar()
        elif event in {"stage.start", "stage.done", "stage.error"} and self._bar_kind == "validation":
            operation = payload.get("operation")
            phase = payload.get("phase")
            previous_operation = self._validation_operation
            if operation is not None:
                self._validation_operation = str(operation)
            elif phase:
                self._validation_operation = str(phase).removeprefix("validation.")
            if self._validation_operation != previous_operation and str(phase or "").startswith("validation."):
                self._refresh_bar(force=True)
        elif event == "metrics":
            self._advance_epoch(payload)
        elif event == "epoch.summary":
            self._close_bar()
            producer = payload.get("producer", {})
            students = payload.get("students", {})
            audit = payload.get("audit", {})
            status = "" if payload.get("epoch_complete") else " PARTIAL"
            epoch, epochs = self._epoch_label(payload)
            validation = payload.get("validation")
            validation_available = isinstance(validation, dict) and bool(validation.get("available"))
            validation_reason = self._short_reason(
                validation.get("reason") if isinstance(validation, dict) else None,
                "reference evaluation not configured",
            )
            train_seconds = self._seconds(payload.get("epoch_seconds"))
            validation_seconds = self._seconds(
                validation.get("elapsed_seconds") if isinstance(validation, dict) else None
            )
            total_seconds = self._seconds(payload.get("total_elapsed_seconds"))
            if validation_available or isinstance(validation, dict):
                timing = f"train={train_seconds} val={validation_seconds} total={total_seconds}"
            else:
                # Keep the long-standing unavailable-validation summary token
                # for operators and existing log consumers.
                timing = f"time={train_seconds} val={validation_seconds} total={total_seconds}"
            summary = (
                f"Epoch {epoch}/{epochs}{status} | {timing} | "
                f"loss P/U/A={self._number(self._metric(producer, '', 'loss'))}/"
                f"{self._number(self._metric(students, 'student_no_audit', 'loss'))}/"
                f"{self._number(self._metric(students, 'student_audited', 'loss'))} | "
                f"select NLL={self._number(audit.get('select_nll_mean') if isinstance(audit, dict) else None)}"
            )
            if not validation_available:
                summary += f" | val Dice=-- ({validation_reason})"
            self._line(summary)
            if validation_available:
                validation_students = validation.get("students", {})
                for name in ("student_no_audit", "student_audited"):
                    student = validation_students.get(name, {}) if isinstance(validation_students, dict) else {}
                    available_student = isinstance(student, dict)
                    dice = student.get("dice") if available_student else None
                    iou = student.get("iou") if available_student else None
                    per_class = student.get("dice_per_class", {}) if available_student else {}
                    if not isinstance(per_class, dict):
                        per_class = {}
                    classes = "/".join(self._number(per_class.get(key)) for key in ("RV", "MYO", "LV"))
                    self._line(
                        f"  {name}: Dice={self._number(dice)} IoU={self._number(iou)} "
                        f"Dice(RV/MYO/LV)={classes} val={validation_seconds}"
                    )
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

        rows = [(f"[maskfree {payload['timestamp']}] {payload['event']} "
                 f"dataset={payload.get('dataset')} epoch={payload.get('epoch')}/{payload.get('epochs')} "
                 f"batch={payload.get('batch_cursor', payload.get('batch'))}/{payload.get('batches')}")]

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
