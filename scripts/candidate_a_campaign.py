#!/usr/bin/env python3
"""Run several hash-pinned A-3D few-shot plans on one GPU.

The few-shot planner deliberately refuses to resume a partially executed child
plan.  This supervisor keeps that property while allowing a durable, serial
campaign: it records every job transition, runs the shared teacher stages and
the ``full`` arm for all requested plans first, and then runs the remaining
arms.  An existing output is accepted only when its final checkpoint and
report pass the same provenance checks as a newly completed job; partial
outputs are never overwritten or resumed.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.candidate_a_fewshot import (
    _resolve,
    _validate_child_plan,
    _verify_root_plan,
)
from self_audit.candidate_a3d import CLASS_NAMES
from self_audit.candidate_a3d.io import source_inventory
from self_audit.model import digest

STATE_SCHEMA = "self_audit.candidate_a3d.campaign_state.v1"
PHASES = ("full", "remaining")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain one object: {path}")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{Path(__file__).stat().st_ino}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def _campaign_lock(lock_path: Path):
    lock_path = lock_path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another campaign already holds the GPU lock: {lock_path}") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _plan_hash(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(path)}


def _job_key(child_path: Path, job_id: str) -> str:
    return f"{child_path.resolve()}::{job_id}"


def _arm_name(job: dict[str, Any]) -> str:
    job_id = str(job["id"])
    if "_" not in job_id:
        return job_id
    return job_id.split("_", 1)[1]


def _validate_job_definition(job: dict[str, Any], *, plan: dict[str, Any],
                             child_path: Path) -> None:
    required = {"id", "config", "config_sha256", "out", "teacher_from"}
    if not required.issubset(job):
        raise ValueError(f"job is missing required fields: {child_path}")
    config = _resolve(job["config"])
    if not config.is_file() or digest(config) != job["config_sha256"]:
        raise ValueError(f"ablation recipe changed: {config}")
    recipe = _read_json(config)
    if recipe.get("method") != "candidate_a3d":
        raise ValueError(f"job recipe is not candidate_a3d: {config}")
    if recipe.get("stage") not in {"teacher_pretrain", "teacher_warmup", "joint"}:
        raise ValueError(f"unsupported job stage: {config}")
    if recipe.get("dataset", {}).get("name") != plan["dataset"]:
        raise ValueError(f"job dataset differs from plan: {config}")


def _collect_plans(paths: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_paths: set[Path] = set()
    for raw_path in paths:
        root_path = _resolve(raw_path)
        if root_path in seen_paths:
            raise ValueError(f"duplicate few-shot plan: {root_path}")
        seen_paths.add(root_path)
        root = _verify_root_plan(root_path)
        roots.append({"path": root_path, "plan": root})
        for child_record in sorted(root["child_plans"], key=lambda item: item["k"]):
            child_path = _resolve(child_record["plan"])
            manifest = next(item for item in root["manifests"] if item["k"] == child_record["k"])
            child = _validate_child_plan(
                child_path,
                expected_train=_resolve(manifest["path"]),
                expected_dev=_resolve(root["source_development"]["path"]),
                k=child_record["k"],
                arms=root["arms"],
                seed=root["seed"],
                checkpoint_every_epochs=root["checkpoint_every_epochs"],
                decoder_chunk_size=root["decoder_chunk_size"],
            )
            for job in child["jobs"]:
                _validate_job_definition(job, plan=root, child_path=child_path)
                key = _job_key(child_path, job["id"])
                if key in seen_keys:
                    raise ValueError(f"duplicate job key: {key}")
                seen_keys.add(key)
                records.append({
                    "key": key,
                    "dataset": root["dataset"],
                    "k": child_record["k"],
                    "arm": _arm_name(job),
                    "stage": _read_json(_resolve(job["config"]))["stage"],
                    "root_path": str(root_path),
                    "child_path": str(child_path),
                    "plan": root,
                    "child": child,
                    "job": job,
                })
    if not roots:
        raise ValueError("at least one few-shot plan is required")
    return roots, records


def _ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return full (teacher + full arm) jobs before all other arms."""
    full: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for record in records:
        if record["stage"] in {"teacher_pretrain", "teacher_warmup"} or record["arm"] == "full":
            full.append(record)
        else:
            remaining.append(record)

    def order(item: dict[str, Any]) -> tuple[int, int, str, int]:
        phase_rank = 0 if item["stage"] == "teacher_pretrain" else 1 if item["stage"] == "teacher_warmup" else 2
        dataset_rank = 0 if item["dataset"] == "acdc" else 1
        return (dataset_rank, item["k"], phase_rank, str(item["job"]["id"]))

    def arm_order(item: dict[str, Any]) -> tuple[int, int, str]:
        dataset_rank = 0 if item["dataset"] == "acdc" else 1
        return (dataset_rank, item["k"], str(item["job"]["id"]))

    return sorted(full, key=order) + sorted(remaining, key=arm_order)


def build_schedule(paths: Iterable[str | Path]) -> dict[str, Any]:
    paths = list(paths)
    roots, records = _collect_plans(paths)
    ordered = _ordered_records(records)
    return {
        "schema": STATE_SCHEMA,
        "status": "PENDING",
        "created_at": _now(),
        "updated_at": _now(),
        "phase_order": list(PHASES),
        "plans": [_plan_hash(item["path"]) for item in roots],
        "total_jobs": len(ordered),
        "jobs": [
            {
                "key": item["key"],
                "dataset": item["dataset"],
                "k": item["k"],
                "arm": item["arm"],
                "stage": item["stage"],
                "phase": "full" if item["stage"] in {"teacher_pretrain", "teacher_warmup"} or item["arm"] == "full" else "remaining",
                "out": str(_resolve(item["job"]["out"])),
                "planned_out": str(_resolve(item["job"]["out"])),
                "attempt": 0,
                "config": str(_resolve(item["job"]["config"])),
                "config_sha256": item["job"]["config_sha256"],
                "status": "PENDING",
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "log": None,
                "error": None,
            }
            for item in ordered
        ],
    }


def _expected_recipe(record: dict[str, Any], dependency_state: dict[str, Any] | None = None) -> dict[str, Any]:
    recipe = _read_json(_resolve(record["job"]["config"]))
    if dependency_state is not None:
        checkpoint = Path(dependency_state["out"]).resolve() / "final"
        manifest = checkpoint / "MANIFEST.json"
        if not manifest.is_file():
            raise ValueError(f"teacher checkpoint manifest is missing: {manifest}")
        recipe["teacher_init"] = {
            "kind": "candidate_a3d",
            "path": str(checkpoint),
            "sha256": digest(manifest),
        }
    return recipe


def _artifact_status(state_job: dict[str, Any], *, record: dict[str, Any], plan: dict[str, Any],
                     dependency_state: dict[str, Any] | None = None) -> tuple[bool, str]:
    out = Path(state_job["out"])
    final = out / "final"
    manifest_path, payload_path, report_path = final / "MANIFEST.json", final / "weights.pt", out / "REPORT.json"
    if not out.is_dir() or not final.is_dir():
        return False, "missing output/final directory"
    if not manifest_path.is_file() or not payload_path.is_file() or not report_path.is_file():
        return False, "missing final MANIFEST.json, weights.pt or REPORT.json"
    try:
        manifest = _read_json(manifest_path)
        report = _read_json(report_path)
    except ValueError as exc:
        return False, str(exc)
    if report.get("status") != "COMPLETED":
        return False, "REPORT.json does not mark the job COMPLETED"
    expected_stage = record["stage"]
    expected_recipe = _expected_recipe(record, dependency_state)
    if report.get("stage") != expected_stage or manifest.get("config") != expected_recipe:
        return False, "checkpoint/report stage does not match the planned stage"
    if report.get("epochs") != expected_recipe.get("epochs"):
        return False, "REPORT.json epoch count differs from the planned recipe"
    if report.get("final_manifest_sha256") != digest(manifest_path):
        return False, "REPORT.json final manifest hash mismatch"
    if manifest.get("payload_sha256") != digest(payload_path):
        return False, "checkpoint payload hash mismatch"
    if manifest.get("schema") != "self_audit.candidate_a3d.checkpoint.v1":
        return False, "checkpoint schema is not the A-3D checkpoint schema"
    if manifest.get("model_class") != "CandidateA3DStudent":
        return False, "checkpoint model class is not CandidateA3DStudent"
    if manifest.get("data_sha256") != plan["train_sha256"]:
        return False, "checkpoint training manifest hash mismatch"
    if manifest.get("label_budget") != plan["label_budget"]:
        return False, "checkpoint label budget mismatch"
    if manifest.get("epoch") != expected_recipe.get("epochs"):
        return False, "checkpoint epoch differs from the planned recipe"
    if manifest.get("synthetic") != bool(plan.get("synthetic", False)):
        return False, "checkpoint synthetic flag differs from the plan"
    if manifest.get("class_mapping") != list(CLASS_NAMES):
        return False, "checkpoint class mapping differs from the declared ontology"
    train = _read_json(_resolve(plan["train"]))
    if manifest.get("labeled_ids") != [item["id"] for item in train["labeled"]]:
        return False, "checkpoint labeled IDs differ from the planned manifest"
    if manifest.get("unlabeled_ids") != [item["id"] for item in train["unlabeled"]]:
        return False, "checkpoint unlabeled IDs differ from the planned manifest"
    current_sources = source_inventory()
    current_source_sha = hashlib.sha256(
        json.dumps(current_sources, sort_keys=True).encode()
    ).hexdigest()
    if manifest.get("source_files") != current_sources or manifest.get("source_sha256") != current_source_sha:
        return False, "checkpoint source inventory differs from this checkout"
    development_sha = plan.get("development_sha256")
    if report.get("validation_manifest_sha256") != development_sha:
        return False, "REPORT.json development manifest hash differs from the plan"
    return True, "validated final checkpoint"


def _state_matches(existing: dict[str, Any], fresh: dict[str, Any]) -> bool:
    if existing.get("schema") != STATE_SCHEMA or existing.get("total_jobs") != fresh["total_jobs"]:
        return False
    if existing.get("plans") != fresh["plans"]:
        return False
    old = {job["key"]: job for job in existing.get("jobs", [])}
    new = {job["key"]: job for job in fresh["jobs"]}
    if set(old) != set(new):
        return False
    return all(old[key].get("planned_out", old[key].get("out")) == new[key]["planned_out"]
               and old[key].get("config_sha256") == new[key]["config_sha256"]
               for key in new)


def _command(record: dict[str, Any], *, device: str, out: str | Path | None = None) -> list[str]:
    plan = record["child"]
    job = record["job"]
    output = _resolve(out if out is not None else job["out"])
    command = [sys.executable, str(ROOT / "train.py"), "--config", str(_resolve(job["config"])),
               "--data", str(_resolve(plan["train"])), "--data-sha256", plan["train_sha256"],
               "--out", str(output), "--device", device, "--allow-full-run"]
    if plan.get("synthetic"):
        command.append("--synthetic")
    if plan.get("development") is not None:
        command.extend(["--validation-data", str(_resolve(plan["development"])),
                        "--validation-data-sha256", plan["development_sha256"]])
    return command


def _dependency(record: dict[str, Any], records_by_key: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    dependency = record["job"].get("teacher_from")
    if dependency is None:
        return None
    key = _job_key(Path(record["child_path"]), dependency)
    try:
        return records_by_key[key]
    except KeyError as exc:
        raise ValueError(f"teacher dependency is absent: {key}") from exc


def _archive_failed_output(state_job: dict[str, Any]) -> None:
    """Move a failed attempt aside before an explicitly requested retry."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    old_out = Path(state_job["out"])
    history = state_job.setdefault("history", [])
    if old_out.exists():
        archive = old_out.with_name(f"{old_out.name}.failed-{timestamp}")
        suffix = 0
        while archive.exists():
            suffix += 1
            archive = old_out.with_name(f"{old_out.name}.failed-{timestamp}-{suffix}")
        old_out.rename(archive)
        history.append({"status": state_job.get("status"), "out": str(old_out),
                        "archive": str(archive), "error": state_job.get("error"),
                        "log": state_job.get("log"), "finished_at": state_job.get("finished_at")})
    else:
        history.append({"status": state_job.get("status"), "out": str(old_out),
                        "error": state_job.get("error"), "log": state_job.get("log"),
                        "finished_at": state_job.get("finished_at")})
    planned = Path(state_job.get("planned_out", state_job["out"]))
    retry = planned.with_name(f"{planned.name}.retry-{timestamp}")
    suffix = 0
    while retry.exists():
        suffix += 1
        retry = planned.with_name(f"{planned.name}.retry-{timestamp}-{suffix}")
    state_job.update(out=str(retry), status="PENDING", started_at=None,
                     finished_at=None, returncode=None, log=None, error=None)
    state_job["attempt"] = int(state_job.get("attempt", 0)) + 1


def _mark_failed(state_job: dict[str, Any], error: str) -> None:
    state_job.update(status="FAILED", error=error, finished_at=_now())


def _free_gb(path: Path) -> float:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return shutil.disk_usage(candidate).free / (1024 ** 3)


def _default_gpu_lock(device: str) -> Path:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{visible}-{device}")
    return Path("/tmp") / f"self-audit-gpu-{tag}.lock"


def run_campaign(paths: Iterable[str | Path], *, state_path: str | Path,
                 execute: bool = False, device: str = "cuda", log_dir: str | Path | None = None,
                 restart_failed: bool = False, min_free_gb: float = 15.0,
                 gpu_lock: str | Path | None = None) -> dict[str, Any]:
    if isinstance(min_free_gb, bool) or not math.isfinite(min_free_gb) or min_free_gb < 0:
        raise ValueError("min_free_gb must be a finite non-negative number")
    paths = list(paths)
    fresh = build_schedule(paths)
    state_file = _resolve(state_path)
    if state_file.exists():
        state = _read_json(state_file)
        if not _state_matches(state, fresh):
            raise ValueError("campaign state does not match the supplied plans; use a new state path")
    else:
        state = fresh
    if not execute:
        return {"status": "PLAN_VALIDATED_NOT_RUN", "total_jobs": state["total_jobs"],
                "jobs": [{"key": job["key"], "phase": job["phase"], "status": job["status"]} for job in state["jobs"]]}

    _roots, records = _collect_plans(paths)
    records_by_key = {item["key"]: item for item in records}
    state_jobs = {job["key"]: job for job in state["jobs"]}
    log_root = _resolve(log_dir) if log_dir is not None else state_file.parent / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_lock = state_file.with_name(f".{state_file.name}.lock")
    gpu_lock_path = _resolve(gpu_lock) if gpu_lock is not None else _default_gpu_lock(device)
    with contextlib.ExitStack() as locks:
        locks.enter_context(_campaign_lock(state_lock))
        if gpu_lock_path != state_lock.resolve():
            locks.enter_context(_campaign_lock(gpu_lock_path))
        # A process dying after marking a job RUNNING cannot be resumed safely.
        # Convert it into a recorded failure; --restart-failed is the explicit
        # opt-in that creates a new attempt directory.
        for state_job in state["jobs"]:
            if state_job["status"] == "RUNNING":
                _mark_failed(state_job, "campaign interrupted while RUNNING; resume is unsupported")
            if state_job["status"] == "FAILED" and restart_failed:
                _archive_failed_output(state_job)
        state["status"] = "RUNNING"
        state["updated_at"] = _now()
        _atomic_write(state_file, state)
        blocked_storage = False
        for scheduled in state["jobs"]:
            state_job = state_jobs[scheduled["key"]]
            record = records_by_key[scheduled["key"]]
            plan = record["child"]
            dependency = _dependency(record, records_by_key)
            dependency_state = state_jobs[dependency["key"]] if dependency is not None else None

            if state_job["status"] == "DONE":
                if dependency_state is not None and dependency_state["status"] != "DONE":
                    _mark_failed(state_job, f"teacher dependency not DONE: {dependency['key']}")
                else:
                    try:
                        valid, reason = _artifact_status(state_job, record=record, plan=plan,
                                                          dependency_state=dependency_state)
                    except (OSError, TypeError, ValueError, KeyError) as exc:
                        valid, reason = False, repr(exc)
                    if not valid:
                        _mark_failed(state_job, f"previously DONE but now invalid: {reason}")
                state["updated_at"] = _now()
                _atomic_write(state_file, state)
                continue
            if state_job["status"] == "FAILED":
                # Failed jobs are retained as evidence. Independent jobs may
                # continue; dependent jobs below receive a deterministic fail.
                continue
            if state_job["status"] == "RUNNING":
                _mark_failed(state_job, "unexpected RUNNING state; resume is unsupported")
                state["updated_at"] = _now()
                _atomic_write(state_file, state)
                continue
            if dependency is not None:
                if dependency_state is None or dependency_state["status"] != "DONE":
                    _mark_failed(state_job, f"teacher dependency not DONE: {dependency['key']}")
                    state["updated_at"] = _now()
                    _atomic_write(state_file, state)
                    continue
                dependency_dependency = _dependency(dependency, records_by_key)
                dependency_dependency_state = (
                    state_jobs[dependency_dependency["key"]] if dependency_dependency is not None else None
                )
                try:
                    valid, reason = _artifact_status(
                        dependency_state, record=dependency, plan=dependency["child"],
                        dependency_state=dependency_dependency_state,
                    )
                except (OSError, TypeError, ValueError, KeyError) as exc:
                    valid, reason = False, repr(exc)
                if not valid:
                    _mark_failed(state_job, f"invalid teacher dependency: {reason}")
                    state["updated_at"] = _now()
                    _atomic_write(state_file, state)
                    continue
            out = Path(state_job["out"])
            if out.exists():
                try:
                    valid, reason = _artifact_status(state_job, record=record, plan=plan,
                                                      dependency_state=dependency_state)
                except (OSError, TypeError, ValueError, KeyError) as exc:
                    valid, reason = False, repr(exc)
                if valid:
                    state_job.update(status="DONE", finished_at=_now(), error=None)
                else:
                    _mark_failed(state_job, f"existing output is not complete; no overwrite/resume: {reason}")
                state["updated_at"] = _now()
                _atomic_write(state_file, state)
                continue
            free_gb = _free_gb(out.parent)
            if free_gb < min_free_gb:
                state["status"] = "BLOCKED_STORAGE"
                state["blocked_storage"] = {
                    "path": str(out.parent), "free_gb": free_gb,
                    "min_free_gb": min_free_gb, "job": scheduled["key"],
                }
                state["updated_at"] = _now()
                _atomic_write(state_file, state)
                blocked_storage = True
                break
            state_job["free_gb_before"] = free_gb
            log_path = log_root / (scheduled["key"].replace("/", "_").replace(":", "_") + f".attempt{state_job.get('attempt', 0)}.log")
            command = _command(record, device=device, out=state_job["out"])
            if dependency is not None:
                checkpoint = Path(dependency_state["out"]) / "final"
                command.extend(["--teacher-checkpoint", str(checkpoint),
                                "--teacher-checkpoint-sha256", digest(checkpoint / "MANIFEST.json")])
            state_job.update(status="RUNNING", started_at=_now(), log=str(log_path), error=None)
            state["updated_at"] = _now()
            _atomic_write(state_file, state)
            with log_path.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
            state_job["returncode"] = result.returncode
            if result.returncode != 0:
                _mark_failed(state_job, f"train.py exited with code {result.returncode}")
                state["updated_at"] = _now()
                _atomic_write(state_file, state)
                continue
            try:
                valid, reason = _artifact_status(state_job, record=record, plan=plan,
                                                  dependency_state=dependency_state)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                valid, reason = False, repr(exc)
            if not valid:
                _mark_failed(state_job, f"training returned success but artifact is invalid: {reason}")
            else:
                state_job.update(status="DONE", finished_at=_now(), error=None)
            state["updated_at"] = _now()
            _atomic_write(state_file, state)
        if not blocked_storage:
            state["status"] = "COMPLETED" if not any(job["status"] == "FAILED" for job in state["jobs"]) else "FAILED"
        state["updated_at"] = _now()
        _atomic_write(state_file, state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", required=True,
                        help="few-shot FEWSHOT_PLAN.json; repeat for datasets")
    parser.add_argument("--state", default="campaign_state.json",
                        help="durable state JSON path (default: ./campaign_state.json)")
    parser.add_argument("--log-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--restart-failed", action="store_true",
                        help="archive failed attempts and create new output directories")
    parser.add_argument("--min-free-gb", type=float, default=15.0,
                        help="minimum free space before launching each job (default: 15 GB)")
    parser.add_argument("--gpu-lock",
                        help="shared GPU lock path (default: /tmp/self-audit-gpu-<visible>-<device>.lock)")
    args = parser.parse_args(argv)
    result = run_campaign(args.plan, state_path=args.state, execute=args.execute,
                          device=args.device, log_dir=args.log_dir,
                          restart_failed=args.restart_failed, min_free_gb=args.min_free_gb,
                          gpu_lock=args.gpu_lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] in {"FAILED", "BLOCKED_STORAGE"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
