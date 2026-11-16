#!/usr/bin/env python3
"""Durable, single-writer supervisor for the canonical PG stage pipeline.

This is a local foreground queue. It does not submit server jobs. Stage order
is shared S1/S2/S3, the canonical four-arm stage, then canonical finalization
within that arms transaction.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "self_audit.candidate_pg.server_queue.v1"
STAGES = ("S1", "S2", "S3", "arm:A4", "arm:A3", "arm:A2", "arm:A1", "finalize")
ARM_ORDER = ("A4", "A3", "A2", "A1")


def tee_run(command, *, cwd, stdout, stderr, check=False):
    """Keep durable per-stage logs and the single tmux pipeline pane in sync."""
    del stderr, check
    proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout.write(line)
        stdout.flush()
        print(line.decode('utf-8', errors='replace'), end='', flush=True)
    return subprocess.CompletedProcess(command, proc.wait())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_file_pin(path_value: str, expected: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} artifact missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}: {path}")
    return path


def verify_ready_receipt(path: Path, kind: str, config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    """Require a READY receipt and verify optional artifact/config pins."""
    receipt = read_json(path)
    status = receipt.get("status")
    if status == "FAILED":
        raise RuntimeError(f"{kind} readiness receipt reports FAILED")
    if status not in {"READY", "WAITING", "BUILDING", "PREPARING", "RUNNING"}:
        raise ValueError(f"{kind} readiness receipt has malformed status {status!r}")
    if status != "READY":
        raise ReadinessPending(f"{kind} readiness status is {status}")
    if receipt.get("kind") not in (None, kind):
        raise ValueError(f"{kind} readiness receipt kind mismatch")
    pins = receipt.get("config_pins", receipt.get("config", {}))
    if isinstance(pins, dict):
        expected = pins.get("path") or pins.get("config_path")
        expected_sha = pins.get("sha256") or pins.get("config_sha256")
        if expected_sha and expected_sha != config_sha:
            raise ValueError(f"{kind} readiness receipt config SHA mismatch")
        if expected and Path(expected).expanduser().resolve() != Path(config.get("_queue_config_path", "")).resolve():
            raise ValueError(f"{kind} readiness receipt config path mismatch")
    artifacts = receipt.get("artifacts", [])
    if isinstance(artifacts, dict):
        artifacts = [dict(path=k, **(v if isinstance(v, dict) else {"sha256": v})) for k, v in artifacts.items()]
    if not isinstance(artifacts, list):
        raise ValueError(f"{kind} readiness artifacts must be a list or object")
    verified = []
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"{kind} readiness artifact pin is malformed")
        expected = item.get("sha256")
        if expected:
            try:
                artifact = _verify_file_pin(item["path"], expected, kind)
            except ValueError as exc:
                if "artifact missing" in str(exc):
                    raise ReadinessPending(str(exc)) from exc
                raise
        else:
            artifact = Path(item["path"]).expanduser().resolve()
            if not artifact.exists():
                raise ReadinessPending(f"{kind} readiness artifact missing: {artifact}")
        verified.append({"path": str(artifact), "sha256": sha256(artifact) if artifact.is_file() else None})
    # Readiness data receipts may pin manifest hashes directly.
    manifest_pins = receipt.get("manifests", {})
    expected_manifests = config.get("data", {}).get("manifests", {})
    if isinstance(manifest_pins, dict):
        for role, pin in manifest_pins.items():
            if role not in expected_manifests or not isinstance(pin, dict):
                continue
            expected = expected_manifests[role]
            if pin.get("sha256") and pin["sha256"] != expected.get("sha256"):
                raise ValueError(f"data readiness receipt {role} manifest SHA conflicts with config")
            if pin.get("path") and Path(pin["path"]).expanduser().resolve() != Path(expected["path"]).expanduser().resolve():
                raise ValueError(f"data readiness receipt {role} manifest path conflicts with config")
    if kind == "data":
        # The READY file is a gate, not a substitute for the config's pinned
        # inputs. Refuse to launch if any of the three manifest bytes are not
        # already present and match the checked-in identity.
        if set(expected_manifests) != {"train", "val", "test"}:
            raise ValueError("data config must pin train, val, and test manifests")
        for role, pin in expected_manifests.items():
            if not isinstance(pin, dict) or not isinstance(pin.get("path"), str) or not isinstance(pin.get("sha256"), str):
                raise ValueError(f"data config {role} manifest pin is malformed")
            try:
                _verify_file_pin(pin["path"], pin["sha256"], f"data {role} manifest")
            except ValueError as exc:
                if "artifact missing" in str(exc):
                    raise ReadinessPending(str(exc)) from exc
                raise
    if kind == "source" and receipt.get("source_sha256"):
        known = {
            "config": config_sha,
            "split_lock": config.get("data", {}).get("split_lock_sha256"),
            "source_k8": config.get("data", {}).get("source_k8_sha256"),
        }
        for name, digest in receipt["source_sha256"].items():
            if name in known and digest != known[name]:
                raise ValueError(f"source readiness receipt {name} SHA conflicts with config")
    return {"path": str(path.resolve()), "sha256": sha256(path), "status": "READY", "verified_artifacts": verified}


class ReadinessPending(Exception):
    """Valid receipt is not ready yet or a pinned expected artifact is absent."""


def readiness(data_path: Path, source_path: Path, config: dict[str, Any], config_sha: str) -> dict[str, Any]:
    return {
        "data": verify_ready_receipt(data_path, "data", config, config_sha),
        "source": verify_ready_receipt(source_path, "source", config, config_sha),
    }


def planned_commands(config_path: Path, out: Path, device: str) -> list[list[str]]:
    base = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path), "--out", str(out), "--device", device]
    # S2's resume contract requires teacher_latest.pt. A new stage (including
    # an empty directory left by pre-training validation) must start fresh.
    # Any retained S2 artifact keeps strict resume enabled, so a missing or
    # corrupt checkpoint after training cannot silently reset the budget.
    s2_dir = out / "S2"
    s2_has_state = (s2_dir.exists() and any(s2_dir.iterdir())) or (out / "S2_RECEIPT.json").exists()
    commands = [base + ["--pg-stage", stage] +
                (["--pg-resume"] if stage != "S2" or s2_has_state else [])
                for stage in ("S1", "S2", "S3")]
    commands.extend(base + ["--pg-stage", "arms", "--pg-arm", arm, "--pg-resume"]
                    for arm in ARM_ORDER)
    commands.append(base + ["--pg-stage", "arms", "--pg-resume"])
    return commands


class Queue:
    def __init__(self, *, config_path: Path, out: Path, data_ready: Path, source_ready: Path,
                 device: str, poll_seconds: float, max_wait_seconds: float,
                 runner: Callable[..., Any] = tee_run,
                 resume_failed_reason: str | None = None,
                 before_stage_script: Path | None = None):
        self.config_path = config_path.resolve()
        self.out = out.resolve()
        self.data_ready = data_ready.resolve()
        self.source_ready = source_ready.resolve()
        self.device = device
        self.poll_seconds = poll_seconds
        self.max_wait_seconds = max_wait_seconds
        self.runner = runner
        self.resume_failed_reason = resume_failed_reason
        self.before_stage_script = before_stage_script.resolve() if before_stage_script else None

    def run(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        lock_path = self.out / "QUEUE.lock"
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another PG queue writer holds QUEUE.lock") from exc
            raw = read_json(self.config_path)
            config_sha = sha256(self.config_path)
            raw["_queue_config_path"] = str(self.config_path)
            state_path = self.out / "QUEUE_STATE.json"
            state = read_json(state_path) if state_path.exists() else {
                "schema": SCHEMA, "status": "WAITING", "created_utc": _now(),
                "config_path": str(self.config_path), "config_sha256": config_sha,
                "stages": {}, "arm_order": list(ARM_ORDER),
            }
            if state.get("schema") != SCHEMA or state.get("config_sha256") != config_sha:
                raise ValueError("queue state/config identity mismatch; use a new output directory")
            prior_status = state.get("status")
            has_interrupted_stage = any(item.get("status") in {"FAILED", "RUNNING"} for item in state.get("stages", {}).values())
            if prior_status in {"FAILED", "RUNNING"} or has_interrupted_stage:
                if not self.resume_failed_reason or not self.resume_failed_reason.strip():
                    raise RuntimeError("queue has a prior FAILED/RUNNING attempt; pass --resume-failed with root-approved reason after reviewing its logs/checkpoints")
                state.setdefault("attempt_history", []).append({
                    "attempt": len(state.get("attempt_history", [])) + 1,
                    "status": prior_status,
                    "active_stage": state.get("active_stage"),
                    "stages": copy.deepcopy(state.get("stages", {})),
                    "recovery_reason": self.resume_failed_reason.strip(),
                    "recorded_utc": _now(),
                })
                state["status"] = "WAITING"
                state.pop("active_stage", None)
            if state.get("status") == "COMPLETED":
                return state
            state.update(status="WAITING", updated_utc=_now(), data_ready_path=str(self.data_ready), source_ready_path=str(self.source_ready))
            atomic_json(state_path, state)
            deadline = time.monotonic() + self.max_wait_seconds
            while True:
                missing = [str(p) for p in (self.data_ready, self.source_ready) if not p.is_file()]
                if not missing:
                    try:
                        ready = readiness(self.data_ready, self.source_ready, raw, config_sha)
                    except ReadinessPending as exc:
                        state.update(status="WAITING", readiness_error=str(exc), updated_utc=_now())
                        atomic_json(state_path, state)
                        ready = None
                    except (RuntimeError, ValueError, json.JSONDecodeError, OSError) as exc:
                        state.update(status="FAILED", readiness_error=str(exc), updated_utc=_now())
                        atomic_json(state_path, state)
                        return state
                    if ready:
                        break
                else:
                    state.update(status="WAITING", waiting_for=missing, updated_utc=_now())
                    atomic_json(state_path, state)
                if time.monotonic() >= deadline:
                    state.update(status="WAITING", timed_out=True, updated_utc=_now())
                    atomic_json(state_path, state)
                    return state
                time.sleep(self.poll_seconds)
            state.update(status="RUNNING", readiness=ready, waiting_for=[], updated_utc=_now())
            atomic_json(state_path, state)
            commands = planned_commands(self.config_path, self.out, self.device)
            state["source_hashes"] = source_hashes()
            state["commands"] = [{"stage": s, "argv": c, "config_sha256": config_sha,
                                   "source_hashes": state["source_hashes"]} for s, c in zip(STAGES, commands)]
            atomic_json(state_path, state)
            for stage, command in zip(STAGES, commands):
                prior = state["stages"].get(stage)
                if prior and prior.get("status") == "COMPLETED":
                    continue
                if self.before_stage_script is not None and stage in {'S1', 'S2'}:
                    hook = [sys.executable, str(self.before_stage_script), '--stage', stage,
                            '--out', str(self.out), '--repo', str(ROOT)]
                    print(f'[PG-QUEUE] boundary hook stage={stage}', flush=True)
                    result = subprocess.run(hook, cwd=ROOT, check=False)
                    if result.returncode:
                        state.update(status='FAILED', active_stage=stage, updated_utc=_now())
                        state['stages'][stage] = {'status': 'FAILED', 'phase': 'boundary_hook',
                                                 'returncode': result.returncode, 'argv': hook}
                        atomic_json(state_path, state)
                        return state
                state['source_hashes'] = source_hashes()
                if stage == "finalize":
                    try:
                        reuse = self._committed_finalization(state_path)
                    except RuntimeError as exc:
                        state["status"] = "FAILED"
                        state["active_stage"] = stage
                        state["stages"][stage] = {"status": "FAILED", "reason": str(exc),
                                                  "updated_utc": _now(), "argv": command}
                        state["updated_utc"] = _now()
                        atomic_json(state_path, state)
                        return state
                    if reuse is not None:
                        state["stages"][stage] = reuse
                        state["status"] = "RUNNING"
                        state["active_stage"] = None
                        state["updated_utc"] = _now()
                        atomic_json(state_path, state)
                        continue
                log_path = self.out / "logs" / f"{stage}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                started = _now()
                state["status"] = "RUNNING"
                state["active_stage"] = stage
                state["stages"][stage] = {"status": "RUNNING", "started_utc": started,
                                           "argv": command, "config_sha256": config_sha,
                                           "source_hashes": state["source_hashes"],
                                           "log_path": str(log_path)}
                state["updated_utc"] = _now()
                atomic_json(state_path, state)
                print(f"[PG-QUEUE] launching stage={stage} argv={json.dumps(command)} log={log_path}", flush=True)
                with log_path.open("ab") as log:
                    try:
                        completed = self.runner(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                        rc = int(completed.returncode)
                    except Exception as exc:
                        rc = -1
                        error = repr(exc)
                    else:
                        error = None
                record = {"status": "COMPLETED" if rc == 0 else "FAILED", "started_utc": started,
                          "finished_utc": _now(), "argv": command, "config_sha256": config_sha,
                          "source_hashes": state["source_hashes"], "log_path": str(log_path),
                          "log_sha256": sha256(log_path), "returncode": rc}
                if error:
                    record["error"] = error
                state["stages"][stage] = record
                state["status"] = "RUNNING" if rc == 0 else "FAILED"
                state["active_stage"] = None if rc == 0 else stage
                state["updated_utc"] = _now()
                atomic_json(state_path, state)
                if rc != 0:
                    return state
            state.update(status="COMPLETED", completed_utc=_now(), updated_utc=_now())
            atomic_json(state_path, state)
            return state

    def _committed_finalization(self, state_path: Path) -> dict[str, Any] | None:
        txn_path = self.out / "TEST_TRANSACTION.json"
        if not txn_path.exists():
            return None
        txn = read_json(txn_path)
        if txn.get("status") != "COMMITTED":
            raise RuntimeError(f"TEST transaction exists with status {txn.get('status')!r}; refusing duplicate finalization")
        config = read_json(self.config_path)
        test_pin = config.get("data", {}).get("manifests", {}).get("test", {})
        if (txn.get("manifest_sha256") != test_pin.get("sha256")
                or Path(txn.get("manifest", "")).expanduser().resolve()
                != Path(test_pin.get("path", "")).expanduser().resolve()):
            raise RuntimeError("committed TEST transaction does not match the queue's pinned TEST manifest")
        artifacts = txn.get("artifacts", {})
        if set(artifacts) != {"A1", "A2", "A3", "A4"}:
            raise RuntimeError("committed TEST transaction does not contain exactly all four arm exports")
        for arm, artifact in artifacts.items():
            if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
                raise RuntimeError(f"committed TEST {arm} export receipt is malformed")
            _verify_file_pin(artifact["path"], artifact["sha256"], f"committed TEST {arm} export")
        finalization_path = self.out / "FINALIZATION.json"
        metrics_path = self.out / "FINAL_METRICS.json"
        if not finalization_path.is_file() or not metrics_path.is_file():
            raise RuntimeError("TEST is committed but finalization receipts are incomplete; reuse TEST manually, do not rerun finalizer")
        final = read_json(finalization_path)
        metrics = read_json(metrics_path)
        if final.get("test_transaction", {}).get("status") != "COMMITTED" or metrics.get("test_transaction", {}).get("status") != "COMMITTED":
            raise RuntimeError("TEST is committed but finalization receipt does not bind the committed TEST; refusing rerun")
        return {"status": "COMPLETED", "reused": True, "returncode": 0,
                "completed_utc": _now(), "log_path": None, "log_sha256": None,
                "test_transaction_path": str(txn_path.resolve()),
                "test_transaction_sha256": sha256(txn_path),
                "finalization_path": str(finalization_path.resolve()),
                "finalization_sha256": sha256(finalization_path)}


def source_hashes() -> dict[str, str]:
    paths = (ROOT / "train.py", ROOT / "scripts" / "pg_campaign.py", ROOT / "self_audit" / "candidate_pg" / "engine.py")
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/candidate_pg/pg_k8_base.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-ready", required=True)
    parser.add_argument("--source-ready", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=86400.0)
    parser.add_argument("--before-stage-script", type=Path,
                        help="checked stage-boundary maintenance hook; never runs during an update")
    parser.add_argument("--resume-failed", metavar="ROOT_APPROVED_REASON",
                        help="explicitly recover a prior FAILED/RUNNING queue after root reviews state/logs/checkpoints")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.max_wait_seconds < 0:
        parser.error("poll interval must be positive and max wait nonnegative")
    try:
        state = Queue(config_path=Path(args.config), out=Path(args.out), data_ready=Path(args.data_ready),
                      source_ready=Path(args.source_ready), device=args.device,
                      poll_seconds=args.poll_seconds, max_wait_seconds=args.max_wait_seconds,
                      resume_failed_reason=args.resume_failed,
                      before_stage_script=args.before_stage_script).run()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"[PG-QUEUE] status={state['status']} out={Path(args.out).resolve()}")
    return 0 if state["status"] == "COMPLETED" else (2 if state["status"] == "FAILED" else 3)


if __name__ == "__main__":
    raise SystemExit(main())
