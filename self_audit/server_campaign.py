"""Fail-closed, finite, single-GPU campaign queue for server experiments.

This module is deliberately independent of trainer implementations. A campaign
contains exact argv vectors and hash pins; this controller never imports a
trainer, edits a recipe, or constructs training arguments.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any
import uuid


SCHEMA = "self_audit.server_campaign.v1"
STATE_SCHEMA = "self_audit.server_campaign_state.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API.?KEY|AUTH)", re.I)
_SECRET_ARG_RE = re.compile(r"^--?(?:ACCESS[-_]?TOKEN|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API[-_]?KEY)(?:$|=|-)", re.I)
_VAR_RE = re.compile(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%")
_OOM_RE = re.compile(
    rb"cuda out of memory|out of memory|cudnn_status_alloc_failed|oom.?kill|"
    rb"torch\.cuda\.outofmemoryerror",
    re.I,
)
_TRAINING_ROLES = {"supervised", "joint", "teacher", "kd"}
_ROLES = _TRAINING_ROLES | {"evaluation", "backup", "maintenance"}
_TERMINAL_STAGE_STATES = {
    "succeeded", "failed", "blocked", "stopped", "interrupted_no_resume",
    "failed_hash_drift", "launch_unknown",
}


class CampaignError(ValueError):
    """Invalid or unsafe campaign state."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read JSON file {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("short append to JSONL receipt")
        os.fsync(fd)
    finally:
        os.close(fd)


def _require_keys(value: dict[str, Any], required: set[str], optional: set[str], where: str) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise CampaignError(f"{where} missing keys: {sorted(missing)}")
    if unknown:
        raise CampaignError(f"{where} has unsupported keys: {sorted(unknown)}")


def _absolute_path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CampaignError(f"{where} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise CampaignError(f"{where} must be absolute")
    return path


def _dotted_get(value: Any, dotted: str) -> Any:
    current = value
    for component in dotted.split("."):
        if not isinstance(current, dict) or component not in current:
            return _MISSING
        current = current[component]
    return current


_MISSING = object()


def _validate_argv(argv: Any, where: str) -> list[str]:
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise CampaignError(f"{where} must be a nonempty argv string array")
    if any(not item or "\x00" in item or _VAR_RE.search(item) for item in argv):
        raise CampaignError(f"{where} has empty, NUL, or interpolation-like arguments")
    if any(_SECRET_ARG_RE.search(item) for item in argv):
        raise CampaignError(f"{where} may not pass credentials or token-like values in argv")
    executable = Path(argv[0])
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise CampaignError(f"{where}[0] must be an absolute executable path")
    base = executable.name.lower()
    if base in {"sh", "bash", "zsh", "dash", "fish", "csh", "tcsh", "cmd", "powershell"}:
        raise CampaignError(f"{where} may not invoke a shell")
    if base.startswith("python") and any(arg in {"-c", "--command"} for arg in argv[1:3]):
        raise CampaignError(f"{where} may not use interpreter eval/command strings")
    if base == "env":
        raise CampaignError(f"{where} must use the campaign env_allowlist instead of an env wrapper")
    eval_flags = {
        "node": {"-e", "--eval"}, "ruby": {"-e"}, "perl": {"-e"}, "php": {"-r"},
    }
    if any(arg in eval_flags.get(base, set()) for arg in argv[1:3]):
        raise CampaignError(f"{where} may not use interpreter eval strings")
    try:
        first_line = executable.open("rb").read(256).splitlines()[0]
    except (OSError, IndexError):
        first_line = b""
    if first_line.startswith(b"#!"):
        interpreter = first_line[2:].decode("utf-8", errors="ignore").strip().split()
        interpreter_names = [Path(item).name.lower() for item in interpreter]
        if any(name in {"sh", "bash", "zsh", "dash", "fish", "csh", "tcsh", "cmd", "powershell"} for name in interpreter_names):
            raise CampaignError(f"{where}[0] is a shell script")
    return argv


def _validate_pin(pin: Any, where: str) -> dict[str, Any]:
    if not isinstance(pin, dict):
        raise CampaignError(f"{where} must be an object")
    _require_keys(pin, {"id", "kind", "path", "sha256"}, set(), where)
    if not isinstance(pin["id"], str) or not _ID_RE.fullmatch(pin["id"]):
        raise CampaignError(f"{where}.id is invalid")
    if not isinstance(pin["kind"], str) or pin["kind"] not in {"source", "recipe", "asset"}:
        raise CampaignError(f"{where}.kind must be source, recipe, or asset")
    _absolute_path(pin["path"], f"{where}.path")
    if not isinstance(pin["sha256"], str) or not _SHA256_RE.fullmatch(pin["sha256"]):
        raise CampaignError(f"{where}.sha256 must be lowercase SHA-256")
    return pin


def _validate_receipt_spec(spec: Any, where: str) -> None:
    if not isinstance(spec, dict):
        raise CampaignError(f"{where} must be an object")
    _require_keys(spec, {"path", "field", "equals"}, set(), where)
    _absolute_path(spec["path"], f"{where}.path")
    if not isinstance(spec["field"], str) or not spec["field"] or any(not p for p in spec["field"].split(".")):
        raise CampaignError(f"{where}.field must be a dotted JSON field")
    try:
        json.dumps(spec["equals"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{where}.equals must be finite JSON data") from exc


def _validate_training(training: Any, pin_by_id: dict[str, dict[str, Any]], where: str) -> None:
    if not isinstance(training, dict):
        raise CampaignError(f"{where} training role requires a training declaration")
    _require_keys(
        training,
        {"steps", "recipe_pin", "source_pin_ids", "asset_pin_ids", "progress_source"},
        set(),
        f"{where}.training",
    )
    if training["steps"] != 10000 or isinstance(training["steps"], bool):
        raise CampaignError(f"{where}.training.steps must be exactly 10000")
    recipe_id = training["recipe_pin"]
    if not isinstance(recipe_id, str):
        raise CampaignError(f"{where}.training.recipe_pin must be a string pin ID")
    if recipe_id not in pin_by_id or pin_by_id[recipe_id]["kind"] != "recipe":
        raise CampaignError(f"{where}.training.recipe_pin must name a declared recipe pin")
    for key, expected_kind in (("source_pin_ids", "source"), ("asset_pin_ids", "asset")):
        ids = training[key]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(item, str) for item in ids)
                or len(ids) != len(set(ids))):
            raise CampaignError(f"{where}.training.{key} must be a nonempty unique list")
        if any(item not in pin_by_id or pin_by_id[item]["kind"] != expected_kind for item in ids):
            raise CampaignError(f"{where}.training.{key} references the wrong pin kind")
    progress = training["progress_source"]
    if not isinstance(progress, dict):
        raise CampaignError(f"{where}.training.progress_source must be an object")
    _require_keys(
        progress,
        {"path", "step_field"},
        {"eta_seconds_field", "step_duration_seconds_field", "elapsed_seconds_field", "step_offset"},
        f"{where}.training.progress_source",
    )
    if sum(k in progress for k in ("eta_seconds_field", "step_duration_seconds_field", "elapsed_seconds_field")) != 1:
        raise CampaignError(f"{where}.training.progress_source needs exactly one ETA, duration or elapsed field")
    _absolute_path(progress["path"], f"{where}.training.progress_source.path")
    if type(progress.get("step_offset", 0)) is not int or progress.get("step_offset", 0) not in (0, 1):
        raise CampaignError("progress step_offset must be zero or one")
    for key in ("step_field", "eta_seconds_field", "step_duration_seconds_field", "elapsed_seconds_field"):
        if key not in progress:
            continue
        if not isinstance(progress[key], str) or not progress[key] or any(not p for p in progress[key].split(".")):
            raise CampaignError(f"{where}.training.progress_source.{key} must be a dotted JSON field")


def validate_campaign(plan: Any) -> None:
    if not isinstance(plan, dict):
        raise CampaignError("campaign must be a JSON object")
    required = {
        "schema", "campaign_id", "campaign_mode", "state_dir", "progress_receipts_jsonl", "max_wall_clock_seconds",
        "poll_seconds", "continue_independent_on_failure", "gpu", "pins", "stages",
    }
    optional = {"env_allowlist", "tmux_target"}
    _require_keys(plan, required, optional, "campaign")
    if plan["schema"] != SCHEMA:
        raise CampaignError(f"campaign.schema must equal {SCHEMA}")
    if not isinstance(plan["campaign_mode"], str) or plan["campaign_mode"] not in {"training", "backup_recovery"}:
        raise CampaignError("campaign_mode must be training or backup_recovery")
    if not isinstance(plan["campaign_id"], str) or not _ID_RE.fullmatch(plan["campaign_id"]):
        raise CampaignError("campaign_id is invalid")
    _absolute_path(plan["state_dir"], "campaign.state_dir")
    _absolute_path(plan["progress_receipts_jsonl"], "campaign.progress_receipts_jsonl")
    max_wall = plan["max_wall_clock_seconds"]
    if isinstance(max_wall, bool) or not isinstance(max_wall, int) or not 1 <= max_wall <= 14 * 24 * 3600:
        raise CampaignError("max_wall_clock_seconds must be an integer from 1 second through 14 days")
    poll = plan["poll_seconds"]
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or not 0.1 <= poll <= 300:
        raise CampaignError("poll_seconds must be between 0.1 and 300")
    if not isinstance(plan["continue_independent_on_failure"], bool):
        raise CampaignError("continue_independent_on_failure must be boolean")

    gpu = plan["gpu"]
    if not isinstance(gpu, dict):
        raise CampaignError("gpu must be an object")
    _require_keys(gpu, {"visible_device"}, set(), "campaign.gpu")
    if not isinstance(gpu["visible_device"], str) or not re.fullmatch(r"\d+", gpu["visible_device"]):
        raise CampaignError("gpu.visible_device must select exactly one numeric device")

    env = plan.get("env_allowlist", {})
    if not isinstance(env, dict):
        raise CampaignError("env_allowlist must map names to nonsecret config paths")
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key) or _SECRET_NAME_RE.search(key):
            raise CampaignError(f"env_allowlist contains a forbidden or invalid name: {key}")
        path = _absolute_path(value, f"env_allowlist.{key}")
        if not path.exists():
            raise CampaignError(f"env_allowlist.{key} config path does not exist")
    if "CUDA_VISIBLE_DEVICES" in env:
        raise CampaignError("CUDA_VISIBLE_DEVICES is set only by gpu.visible_device")

    if "tmux_target" in plan:
        target = plan["tmux_target"]
        if not isinstance(target, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?", target):
            raise CampaignError("tmux_target must be a simple session[:window] name")

    if not isinstance(plan["pins"], list):
        raise CampaignError("pins must be a list")
    pin_by_id: dict[str, dict[str, Any]] = {}
    pin_paths: set[str] = set()
    for i, pin in enumerate(plan["pins"]):
        validated = _validate_pin(pin, f"pins[{i}]")
        if validated["id"] in pin_by_id:
            raise CampaignError(f"duplicate pin id: {validated['id']}")
        if validated["path"] in pin_paths:
            raise CampaignError(f"duplicate pinned path: {validated['path']}")
        pin_paths.add(validated["path"])
        pin_by_id[validated["id"]] = validated
    if not isinstance(plan["stages"], list) or not plan["stages"] or len(plan["stages"]) > 128:
        raise CampaignError("stages must contain between 1 and 128 finite named stages")
    stage_ids: list[str] = []
    stage_by_id: dict[str, dict[str, Any]] = {}
    completion_paths: set[str] = set()
    for i, stage in enumerate(plan["stages"]):
        where = f"stages[{i}]"
        if not isinstance(stage, dict):
            raise CampaignError(f"{where} must be an object")
        stage_required = {"id", "kind", "role", "requires", "completion"}
        stage_optional = {"training", "argv", "cwd", "process", "resume", "quality_gate", "pin_ids", "failure_outcomes"}
        _require_keys(stage, stage_required, stage_optional, where)
        if not isinstance(stage["id"], str) or not _ID_RE.fullmatch(stage["id"]):
            raise CampaignError(f"{where}.id is invalid")
        if stage["id"] in stage_by_id:
            raise CampaignError(f"duplicate stage id: {stage['id']}")
        if not isinstance(stage["kind"], str) or stage["kind"] not in {"command", "adopt"}:
            raise CampaignError(f"{where}.kind must be command or adopt")
        role = stage["role"]
        if not isinstance(role, str) or role not in _ROLES:
            raise CampaignError(f"{where}.role is unsupported")
        if (not isinstance(stage["requires"], list)
                or any(not isinstance(item, str) for item in stage["requires"])
                or len(stage["requires"]) != len(set(stage["requires"]))):
            raise CampaignError(f"{where}.requires must be a unique stage-id list")
        _validate_receipt_spec(stage["completion"], f"{where}.completion")
        receipt_path = stage["completion"]["path"]
        if receipt_path in completion_paths:
            raise CampaignError(f"completion receipt paths must be unique: {receipt_path}")
        completion_paths.add(receipt_path)
        if role in _TRAINING_ROLES:
            _validate_training(stage.get("training"), pin_by_id, where)
        elif "training" in stage:
            raise CampaignError(f"{where}.training is only valid for training roles")

        if stage["kind"] == "command":
            if "process" in stage:
                raise CampaignError(f"{where}.command may not declare an adopted process")
            _validate_argv(stage.get("argv"), f"{where}.argv")
            _absolute_path(stage.get("cwd"), f"{where}.cwd")
            if role in _TRAINING_ROLES and not isinstance(stage.get("pin_ids"), list):
                raise CampaignError(f"{where}.pin_ids must be present for a training command")
            if "resume" in stage:
                resume = stage["resume"]
                if not isinstance(resume, dict):
                    raise CampaignError(f"{where}.resume must be an object")
                _require_keys(resume, {"argv", "checkpoint_pins", "required_argv_values"}, set(), f"{where}.resume")
                if role not in _TRAINING_ROLES:
                    raise CampaignError(f"{where}.resume is only valid for training stages")
                resume_argv = _validate_argv(resume["argv"], f"{where}.resume.argv")
                pins = resume["checkpoint_pins"]
                if not isinstance(pins, list) or not pins:
                    raise CampaignError(f"{where}.resume.checkpoint_pins must be nonempty")
                for j, resume_pin in enumerate(pins):
                    if not isinstance(resume_pin, dict):
                        raise CampaignError(f"{where}.resume.checkpoint_pins[{j}] must be an object")
                    _require_keys(resume_pin, {"path", "sha256"}, set(), f"{where}.resume.checkpoint_pins[{j}]")
                    _absolute_path(resume_pin["path"], f"{where}.resume.checkpoint_pins[{j}].path")
                    if not isinstance(resume_pin["sha256"], str) or not _SHA256_RE.fullmatch(resume_pin["sha256"]):
                        raise CampaignError(f"{where}.resume.checkpoint_pins[{j}].sha256 must be SHA-256")
                required_values = resume["required_argv_values"]
                if not isinstance(required_values, list) or not required_values or not all(isinstance(x, str) and x for x in required_values):
                    raise CampaignError(f"{where}.resume.required_argv_values must be a nonempty string list")
                for resume_pin in pins:
                    if resume_pin["path"] not in resume_argv or resume_pin["sha256"] not in resume_argv:
                        raise CampaignError(f"{where}.resume.argv must include every checkpoint path and SHA exactly")
                if any(value not in resume_argv for value in required_values):
                    raise CampaignError(f"{where}.resume.argv omits a required explicitly listed value")
            if "process" in stage or "resume" in stage and stage["kind"] != "command":
                raise CampaignError(f"{where} has incompatible command fields")
        else:
            if "argv" in stage or "cwd" in stage or "resume" in stage or "pin_ids" in stage:
                raise CampaignError(f"{where}.adopt must not submit or resume a command")
            process = stage.get("process")
            if not isinstance(process, dict):
                raise CampaignError(f"{where}.adopt requires an exact process identity")
            _require_keys(
                process,
                {"pid", "start_ticks", "cwd", "argv_sha256", "log_path", "receipt_sha256_before"},
                set(),
                f"{where}.process",
            )
            if isinstance(process["pid"], bool) or not isinstance(process["pid"], int) or process["pid"] <= 1:
                raise CampaignError(f"{where}.process.pid is invalid")
            if isinstance(process["start_ticks"], bool) or not isinstance(process["start_ticks"], int) or process["start_ticks"] <= 0:
                raise CampaignError(f"{where}.process.start_ticks is invalid")
            _absolute_path(process["cwd"], f"{where}.process.cwd")
            _absolute_path(process["log_path"], f"{where}.process.log_path")
            if not isinstance(process["argv_sha256"], str) or not _SHA256_RE.fullmatch(process["argv_sha256"]):
                raise CampaignError(f"{where}.process.argv_sha256 must be lowercase SHA-256")
            if (process["receipt_sha256_before"] is not None
                    and (not isinstance(process["receipt_sha256_before"], str)
                         or not _SHA256_RE.fullmatch(process["receipt_sha256_before"]))):
                raise CampaignError(f"{where}.process.receipt_sha256_before must be SHA-256 or null")

        if "failure_outcomes" in stage:
            outcomes = stage["failure_outcomes"]
            if not isinstance(outcomes, list):
                raise CampaignError(f"{where}.failure_outcomes must be a list")
            for j, outcome in enumerate(outcomes):
                if not isinstance(outcome, dict):
                    raise CampaignError(f"{where}.failure_outcomes[{j}] must be an object")
                _require_keys(outcome, {"field", "equals", "label"}, set(), f"{where}.failure_outcomes[{j}]")
                if not isinstance(outcome["field"], str) or not outcome["field"]:
                    raise CampaignError(f"{where}.failure_outcomes[{j}].field must be a JSON path")
                if not isinstance(outcome["label"], str) or not _ID_RE.fullmatch(outcome["label"]):
                    raise CampaignError(f"{where}.failure_outcomes[{j}].label is invalid")

        if role == "kd":
            gate = stage.get("quality_gate")
            if not isinstance(gate, dict):
                raise CampaignError(f"{where}.kd requires the teacher Macro4 quality gate")
            _require_keys(gate, {"stage", "metric", "field", "minimum"}, set(), f"{where}.quality_gate")
            if gate["metric"] != "Macro4" or gate["minimum"] != 0.87 or isinstance(gate["minimum"], bool):
                raise CampaignError(f"{where}.quality_gate must enforce Macro4 >= 0.87 exactly")
            if not isinstance(gate["stage"], str) or not isinstance(gate["field"], str) or not gate["field"]:
                raise CampaignError(f"{where}.quality_gate must name a stage and receipt field")
            if gate["stage"] not in stage["requires"]:
                raise CampaignError(f"{where}.quality_gate stage must also be a prerequisite")
        elif "quality_gate" in stage:
            raise CampaignError(f"{where}.quality_gate is reserved for KD stages")
        stage_ids.append(stage["id"])
        stage_by_id[stage["id"]] = stage

    index_by_id = {stage_id: index for index, stage_id in enumerate(stage_ids)}
    for index, stage in enumerate(plan["stages"]):
        for dep in stage["requires"]:
            if dep not in index_by_id or index_by_id[dep] >= index:
                raise CampaignError(f"stage {stage['id']} prerequisites must be declared earlier")
        if stage["role"] in _TRAINING_ROLES and stage["kind"] == "command":
            pin_ids = stage.get("pin_ids", [])
            if (not pin_ids or any(not isinstance(item, str) for item in pin_ids)
                    or len(pin_ids) != len(set(pin_ids))):
                raise CampaignError(f"training stage {stage['id']} needs unique pin_ids")
            expected_ids = {stage["training"]["recipe_pin"], *stage["training"]["source_pin_ids"], *stage["training"]["asset_pin_ids"]}
            if set(pin_ids) != expected_ids:
                raise CampaignError(f"training stage {stage['id']} pin_ids must exactly equal its recipe/source/asset pins")
        elif stage["kind"] == "command" and "pin_ids" in stage:
            if (not isinstance(stage["pin_ids"], list)
                    or any(not isinstance(item, str) for item in stage["pin_ids"])
                    or len(stage["pin_ids"]) != len(set(stage["pin_ids"]))):
                raise CampaignError(f"stage {stage['id']}.pin_ids must be a unique list")
            if any(pin_id not in pin_by_id for pin_id in stage["pin_ids"]):
                raise CampaignError(f"stage {stage['id']}.pin_ids references an undeclared pin")
        if stage["kind"] == "command" and stage["role"] in {"evaluation", "backup"}:
            if not stage.get("pin_ids"):
                raise CampaignError(f"{stage['role']} stage {stage['id']} must pin its inputs")
        if stage["role"] == "kd":
            gate_id = stage["quality_gate"]["stage"]
            if gate_id not in index_by_id or index_by_id[gate_id] >= index:
                raise CampaignError(f"KD quality gate for {stage['id']} must reference an earlier stage")
            if stage_by_id[gate_id]["role"] != "evaluation":
                raise CampaignError(f"KD quality gate must use an evaluation stage receipt")

    if plan["campaign_mode"] == "backup_recovery" and any(stage["role"] in _TRAINING_ROLES for stage in plan["stages"]):
        raise CampaignError("backup_recovery campaigns may not contain training stages")

    progress_path = Path(plan["progress_receipts_jsonl"])
    if str(progress_path) in completion_paths:
        raise CampaignError("progress receipt output may not overwrite a stage completion receipt")
    if any(
        stage.get("training", {}).get("progress_source", {}).get("path") == str(progress_path)
        for stage in plan["stages"]
    ):
        raise CampaignError("progress receipt output may not be a training progress input")


def load_campaign(plan_path: str | Path, expected_sha256: str | None = None) -> tuple[dict[str, Any], str, Path]:
    path = Path(plan_path).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not _SHA256_RE.fullmatch(expected_sha256) or actual != expected_sha256:
            raise CampaignError("campaign plan hash does not match the externally supplied SHA-256")
    try:
        plan = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"campaign plan is not valid UTF-8 JSON: {type(exc).__name__}") from exc
    validate_campaign(plan)
    return plan, actual, path


def _safe_environment(plan: dict[str, Any]) -> dict[str, str]:
    environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    environment.update(plan.get("env_allowlist", {}))
    environment["CUDA_VISIBLE_DEVICES"] = plan["gpu"]["visible_device"]
    return environment


def process_identity(pid: int) -> dict[str, Any] | None:
    """Return a verifiable process identity without exposing argv contents."""
    proc = Path("/proc") / str(pid)
    try:
        stat_text = (proc / "stat").read_text(encoding="ascii")
        close = stat_text.rfind(")")
        if close < 0:
            return None
        fields = stat_text[close + 2 :].split()  # starts with field 3 (state)
        state = fields[0]
        start_ticks = int(fields[19])  # /proc/<pid>/stat field 22
        argv = (proc / "cmdline").read_bytes()
        cwd = os.readlink(proc / "cwd")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
        if Path("/proc").exists():
            return None
        # macOS has no /proc. Use separate ps/lsof queries with shell-free argv;
        # the start timestamp is second-resolution there, while PID, cwd, and
        # the complete displayed command fingerprint still guard adoption.
        try:
            ps = subprocess.run(
                ["/bin/ps", "-ww", "-p", str(pid), "-o", "command="],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=2, check=False, env={"PATH": os.defpath},
            )
            started = subprocess.run(
                ["/bin/ps", "-p", str(pid), "-o", "lstart="],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=2, check=False, env={"PATH": os.defpath},
            )
            lsof_path = next((p for p in (Path("/usr/sbin/lsof"), Path("/usr/bin/lsof")) if p.is_file()), None)
            if not lsof_path or ps.returncode != 0 or started.returncode != 0:
                return None
            lsof = subprocess.run(
                [str(lsof_path), "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=2, check=False, env={"PATH": os.defpath},
            )
            command = ps.stdout.strip()
            start_text = started.stdout.strip()
            if not command or not start_text or lsof.returncode != 0:
                return None
            start_dt = datetime.strptime(start_text, "%a %b %d %H:%M:%S %Y")
            cwd = next((line[1:] for line in lsof.stdout.splitlines() if line.startswith("n")), None)
            if not cwd:
                return None
            return {
                "pid": pid,
                "start_ticks": int(start_dt.timestamp()),
                "cwd": str(Path(cwd).resolve()),
                "argv_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "state": "R",
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    if state == "Z" or not argv:
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "cwd": str(Path(cwd).resolve()),
        "argv_sha256": hashlib.sha256(argv).hexdigest(),
        "state": state,
    }


def _identity_matches(actual: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if actual is None:
        return False
    return all(actual.get(key) == expected.get(key) for key in ("pid", "start_ticks", "cwd", "argv_sha256"))


def _verify_pin_set(plan: dict[str, Any], pin_ids: list[str]) -> None:
    by_id = {pin["id"]: pin for pin in plan["pins"]}
    for pin_id in pin_ids:
        pin = by_id[pin_id]
        path = Path(pin["path"])
        try:
            actual = sha256_file(path)
        except OSError as exc:
            raise CampaignError(f"pinned {pin['kind']} file is unavailable: {pin_id}") from exc
        if actual != pin["sha256"]:
            raise CampaignError(f"pinned {pin['kind']} hash changed: {pin_id}")


def _read_receipt(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(spec["path"])
    if not path.is_file():
        return None, None
    try:
        value = _read_json(path)
        digest = sha256_file(path)
    except (CampaignError, OSError):
        return None, None
    return value, digest


def _receipt_matches(spec: dict[str, Any], value: dict[str, Any] | None) -> bool:
    if value is None:
        return False
    actual = _dotted_get(value, spec["field"])
    expected = spec["equals"]
    return type(actual) is type(expected) and actual == expected


def _failure_outcome(stage: dict[str, Any], receipt: dict[str, Any] | None) -> str | None:
    if receipt is None:
        return None
    for outcome in stage.get("failure_outcomes", []):
        if _dotted_get(receipt, outcome["field"]) == outcome["equals"]:
            return outcome["label"]
    return None


def _campaign_locked(plan: dict[str, Any]) -> int:
    state_dir = Path(plan["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "campaign.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return -1
    return fd


def _initial_state(plan: dict[str, Any], plan_sha: str) -> dict[str, Any]:
    now = time.time()
    return {
        "schema": STATE_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "campaign_mode": plan["campaign_mode"],
        "plan_sha256": plan_sha,
        "started_utc": _utc_now(),
        "started_epoch": now,
        "status": "running",
        "supervisor": None,
        "stop_requested": False,
        "stop_reason": None,
        "active_stage": None,
        "stages": {
            stage["id"]: {
                "status": "pending",
                "attempt": 0,
                "progress_offset": 0,
                "progress_last_step": -1,
            }
            for stage in plan["stages"]
        },
    }


class CampaignController:
    def __init__(self, plan_path: str | Path, plan_sha256: str):
        self.plan, self.plan_sha256, self.plan_path = load_campaign(plan_path, plan_sha256)
        self.state_dir = Path(self.plan["state_dir"])
        self.state_path = self.state_dir / "STATE.json"
        self.events_path = self.state_dir / "EVENTS.jsonl"
        self.stop_path = self.state_dir / "STOP_REQUESTED.json"
        self._state = self._load_or_initialize()

    def _load_or_initialize(self) -> dict[str, Any]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            state = _read_json(self.state_path)
            if state.get("schema") != STATE_SCHEMA or state.get("campaign_id") != self.plan["campaign_id"]:
                raise CampaignError("persistent state belongs to a different campaign")
            if state.get("plan_sha256") != self.plan_sha256:
                raise CampaignError("persistent state is pinned to a different campaign JSON")
            return state
        state = _initial_state(self.plan, self.plan_sha256)
        _atomic_json(self.state_path, state)
        self._event("campaign_created", plan_sha256=self.plan_sha256)
        return state

    def _save(self) -> None:
        _atomic_json(self.state_path, self._state)

    def _event(self, event: str, **fields: Any) -> None:
        row = {
            "schema": "self_audit.server_campaign_event.v1",
            "campaign_id": self.plan["campaign_id"],
            "utc": _utc_now(),
            "event": event,
            **fields,
        }
        _append_jsonl(self.events_path, row)

    def set_supervisor(self, pid: int) -> None:
        identity = process_identity(pid)
        if identity is None:
            raise CampaignError("cannot record supervisor process identity on this host")
        self._state["supervisor"] = {key: identity[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")}
        self._state["status"] = "running"
        self._save()

    def _recheck_plan_and_active_pins(self) -> bool:
        try:
            current = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        except OSError:
            current = "missing"
        if current != self.plan_sha256:
            self._state["stop_reason"] = "campaign_plan_hash_changed"
            self._state["hash_drift_detected"] = True
            self._save()
            return False
        active_id = self._state.get("active_stage")
        if active_id:
            stage = self._stage_by_id()[active_id]
            pin_ids = self._stage_pin_ids(stage)
            try:
                _verify_pin_set(self.plan, pin_ids)
            except CampaignError:
                self._state["stop_reason"] = "active_stage_pin_changed"
                self._state["hash_drift_detected"] = True
                self._save()
                return False
        return True

    def _stage_by_id(self) -> dict[str, dict[str, Any]]:
        return {stage["id"]: stage for stage in self.plan["stages"]}

    @staticmethod
    def _stage_pin_ids(stage: dict[str, Any]) -> list[str]:
        if stage["role"] in _TRAINING_ROLES:
            train = stage["training"]
            return [train["recipe_pin"], *train["source_pin_ids"], *train["asset_pin_ids"]]
        return stage.get("pin_ids", [])

    def _stop_requested(self) -> bool:
        if self.stop_path.exists():
            self._state["stop_requested"] = True
            try:
                request = _read_json(self.stop_path)
                self._state["stop_reason"] = request.get("reason", "operator_stop")
            except CampaignError:
                self._state["stop_reason"] = "operator_stop"
            self._save()
        return bool(self._state.get("stop_requested"))

    def _deadline_reached(self) -> bool:
        started = float(self._state["started_epoch"])
        return time.time() - started >= self.plan["max_wall_clock_seconds"]

    def _start_or_monitor_adopt(self, stage: dict[str, Any]) -> None:
        stage_id = stage["id"]
        state = self._state["stages"][stage_id]
        process = stage["process"]
        expected = {key: process[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")}
        actual = process_identity(expected["pid"])
        if actual is not None:
            if not _identity_matches(actual, expected):
                self._fail_stage(stage_id, "adopted_process_identity_mismatch")
                return
            if state["status"] == "pending":
                receipt_path = Path(stage["completion"]["path"])
                current_receipt_sha = sha256_file(receipt_path) if receipt_path.is_file() else None
                if current_receipt_sha != process["receipt_sha256_before"]:
                    self._fail_stage(stage_id, "adoption_receipt_changed_since_plan")
                    return
                log_path = process["log_path"]
                if not Path(log_path).is_file():
                    self._fail_stage(stage_id, "adopted_process_log_missing")
                    return
                self._verify_stage_pins(stage)
                state.update({
                    "status": "running",
                    "mode": "adopted",
                    "pid": expected["pid"],
                    "process_identity": expected,
                    "receipt_sha256_before": process["receipt_sha256_before"],
                    "stdout_log": log_path,
                    "started_utc": _utc_now(),
                    "started_epoch": time.time(),
                    "attempt": 1,
                    "progress_offset": Path(stage["training"]["progress_source"]["path"]).stat().st_size
                    if Path(stage["training"]["progress_source"]["path"]).is_file() else 0,
                })
                self._state["active_stage"] = stage_id
                self._state["status"] = "running"
                self._save()
                self._event("stage_adopted", stage_id=stage_id, pid=expected["pid"], stdout_log=log_path)
                self._link_tmux_log(stage, log_path, 1)
            self._poll_progress(stage)
            return

        receipt, receipt_sha = _read_receipt(stage["completion"])
        fresh = process["receipt_sha256_before"] is None or receipt_sha != process["receipt_sha256_before"]
        if _receipt_matches(stage["completion"], receipt) and fresh:
            self._complete_stage(stage_id, receipt_sha, receipt, adopted=True)
        else:
            self._fail_stage(stage_id, "adopted_process_gone_without_success_receipt", receipt, receipt_sha)

    def _verify_stage_pins(self, stage: dict[str, Any]) -> None:
        try:
            _verify_pin_set(self.plan, self._stage_pin_ids(stage))
        except CampaignError as exc:
            raise CampaignError(f"stage {stage['id']} input pin check failed: {exc}") from exc

    def _quality_gate_passes(self, stage: dict[str, Any]) -> tuple[bool, str | None]:
        gate = stage.get("quality_gate")
        if gate is None:
            return True, None
        upstream_state = self._state["stages"].get(gate["stage"], {})
        expected_receipt_sha = upstream_state.get("receipt_sha256")
        upstream_stage = self._stage_by_id()[gate["stage"]]
        receipt, current_sha = _read_receipt(upstream_stage["completion"])
        if upstream_state.get("status") != "succeeded" or receipt is None or current_sha != expected_receipt_sha:
            return False, "teacher_evaluation_receipt_missing_or_changed"
        score = _dotted_get(receipt, gate["field"])
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            return False, "teacher_Macro4_missing_or_nonfinite"
        if float(score) < 0.87:
            return False, f"teacher_Macro4_below_0.87:{score}"
        return True, None

    def _mark_dependency_blocks(self) -> None:
        changed = False
        for stage in self.plan["stages"]:
            state = self._state["stages"][stage["id"]]
            if state["status"] != "pending":
                continue
            bad = [dep for dep in stage["requires"] if self._state["stages"][dep]["status"] in _TERMINAL_STAGE_STATES and self._state["stages"][dep]["status"] != "succeeded"]
            if bad:
                state.update({"status": "blocked", "blocked_by": bad, "reason": "prerequisite_not_succeeded", "completed_utc": _utc_now()})
                self._event("stage_blocked", stage_id=stage["id"], blocked_by=bad)
                changed = True
        if changed:
            self._save()

    def _next_eligible_stage(self) -> dict[str, Any] | None:
        self._mark_dependency_blocks()
        failures = [
            stage["id"] for stage in self.plan["stages"]
            if self._state["stages"][stage["id"]]["status"] in _TERMINAL_STAGE_STATES
            and self._state["stages"][stage["id"]]["status"] != "succeeded"
        ]
        if failures and not self.plan["continue_independent_on_failure"]:
            for stage in self.plan["stages"]:
                state = self._state["stages"][stage["id"]]
                if state["status"] == "pending":
                    state.update({"status": "blocked", "blocked_by": failures, "reason": "campaign_stops_after_failure", "completed_utc": _utc_now()})
            self._save()
            return None
        for stage in self.plan["stages"]:
            state = self._state["stages"][stage["id"]]
            if state["status"] != "pending":
                continue
            if any(self._state["stages"][dep]["status"] != "succeeded" for dep in stage["requires"]):
                continue
            gate_ok, gate_reason = self._quality_gate_passes(stage)
            if not gate_ok:
                state.update({"status": "blocked", "reason": gate_reason, "completed_utc": _utc_now()})
                self._event("stage_blocked", stage_id=stage["id"], reason=gate_reason)
                self._save()
                continue
            return stage
        return None

    def _mark_no_more_work(self) -> None:
        stages = self._state["stages"].values()
        if all(state["status"] in _TERMINAL_STAGE_STATES for state in stages):
            success = all(state["status"] == "succeeded" for state in self._state["stages"].values())
            self._state["status"] = "completed" if success else "completed_with_blocks"
            self._state["completed_utc"] = _utc_now()
        elif self._state.get("stop_requested"):
            for state in self._state["stages"].values():
                if state["status"] == "pending":
                    state.update({"status": "stopped", "reason": self._state.get("stop_reason") or "operator_stop", "completed_utc": _utc_now()})
            self._state["status"] = "stopped"
            self._state["completed_utc"] = _utc_now()
        elif self._deadline_reached():
            for state in self._state["stages"].values():
                if state["status"] == "pending":
                    state.update({"status": "stopped", "reason": "max_wall_clock_deadline", "completed_utc": _utc_now()})
            self._state["status"] = "deadline_exceeded"
            self._state["completed_utc"] = _utc_now()
        self._save()

    def _launch_stage(self, stage: dict[str, Any], resume: bool = False) -> None:
        stage_id = stage["id"]
        self._verify_stage_pins(stage)
        if resume:
            resume_cfg = stage.get("resume")
            if not resume_cfg:
                self._fail_stage(stage_id, "interrupted_started_stage_has_no_explicit_resume")
                return
            for pin in resume_cfg["checkpoint_pins"]:
                path = Path(pin["path"])
                if not path.is_file() or sha256_file(path) != pin["sha256"]:
                    self._fail_stage(stage_id, "exact_resume_checkpoint_hash_mismatch")
                    return
            if any(state.get("status") == "pending" for key, state in self._state["stages"].items() if key != stage_id):
                # Other pending stages are fine; this guard only protects the fixed stage identity.
                pass
        else:
            existing, _ = _read_receipt(stage["completion"])
            if _receipt_matches(stage["completion"], existing):
                self._fail_stage(stage_id, "preexisting_success_receipt_refused")
                return

        state = self._state["stages"][stage_id]
        attempt = 2 if resume else 1
        if state["attempt"] >= attempt:
            self._fail_stage(stage_id, "attempt_already_consumed")
            return
        attempt_dir = self.state_dir / "stages" / stage_id / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "stdout.log"
        execution_path = attempt_dir / "EXECUTION.json"
        result_path = attempt_dir / "RESULT.json"
        for path in (log_path, execution_path, result_path):
            if path.exists():
                self._fail_stage(stage_id, "attempt_artifact_already_exists_refused")
                return

        state.update({
            "status": "launching",
            "attempt": attempt,
            "mode": "resume" if resume else "initial",
            "stdout_log": str(log_path),
            "execution_path": str(execution_path),
            "result_path": str(result_path),
            "attempt_dir": str(attempt_dir),
            "started_utc": _utc_now(),
            "started_epoch": time.time(),
            "progress_offset": (
                Path(stage["training"]["progress_source"]["path"]).stat().st_size
                if resume and Path(stage["training"]["progress_source"]["path"]).is_file()
                else 0
            ),
            "progress_last_step": state.get("progress_last_step", -1) if resume else -1,
            "step_duration_samples": state.get("step_duration_samples", []) if resume else [],
        })
        self._state["active_stage"] = stage_id
        self._state["status"] = "running"
        self._save()
        self._event("stage_launching", stage_id=stage_id, attempt=attempt, mode=state["mode"])
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_server_campaign.py"
        command = [
            sys.executable, str(script), "_stage_child", "--plan", str(self.plan_path),
            "--plan-sha256", self.plan_sha256, "--stage", stage_id, "--attempt", str(attempt),
            "--log", str(log_path), "--execution", str(execution_path), "--result", str(result_path),
        ]
        worker_log = (self.state_dir / "controller.log").open("ab", buffering=0)
        try:
            child = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.DEVNULL,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                close_fds=True,
                start_new_session=True,
            )
        finally:
            worker_log.close()
        identity = self._wait_identity(child.pid)
        if identity is None:
            self._fail_stage(stage_id, "stage_wrapper_identity_unavailable")
            return
        state.update({
            "status": "running",
            "wrapper_pid": child.pid,
            "wrapper_identity": {key: identity[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")},
        })
        self._save()
        self._link_tmux_log(stage, str(log_path), attempt)
        self._event("stage_started", stage_id=stage_id, attempt=attempt, wrapper_pid=child.pid, stdout_log=str(log_path))

    @staticmethod
    def _wait_identity(pid: int) -> dict[str, Any] | None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            identity = process_identity(pid)
            if identity is not None:
                return identity
            time.sleep(0.01)
        return None

    def _link_tmux_log(self, stage: dict[str, Any], log_path: str, attempt: int) -> None:
        target = self.plan.get("tmux_target")
        if not target:
            return
        tmux = "/usr/bin/tmux" if Path("/usr/bin/tmux").is_file() else "/bin/tmux"
        if not Path(tmux).is_file():
            return
        window_name = f"camp_{self.plan['campaign_id'][:16]}_{stage['id'][:20]}_{attempt}"
        viewer = f"tail -n +1 -F -- {shlex.quote(log_path)}"
        try:
            result = subprocess.run(
                [tmux, "new-window", "-d", "-t", target, "-n", window_name, viewer],
                cwd=str(Path(__file__).resolve().parents[1]),
                env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        self._state["stages"][stage["id"]]["tmux_log_window"] = window_name if result and result.returncode == 0 else None
        self._save()

    def _poll_progress(self, stage: dict[str, Any]) -> None:
        if stage["role"] not in _TRAINING_ROLES:
            return
        stage_state = self._state["stages"][stage["id"]]
        source = stage["training"]["progress_source"]
        path = Path(source["path"])
        if not path.is_file():
            return
        offset = int(stage_state.get("progress_offset", 0))
        if path.stat().st_size < offset:
            stage_state["progress_error"] = "progress source truncated"
            self._state["safety_stop_reason"] = "invalid_training_progress_receipt"
            self._save()
            return
        with path.open("rb") as stream:
            stream.seek(offset)
            chunk = stream.read(8 << 20)
        complete_bytes = 0
        try:
            for line in chunk.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    break
                complete_bytes += len(line)
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, object_pairs_hook=_unique_object)
                except (json.JSONDecodeError, CampaignError) as exc:
                    raise CampaignError(f"training progress JSONL malformed for {stage['id']}") from exc
                step = _dotted_get(record, source["step_field"])
                if type(step) is int:
                    step += source.get("step_offset", 0)
                if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > 10000:
                    raise CampaignError(f"training progress step is invalid for {stage['id']}")
                if "eta_seconds_field" in source:
                    eta = _dotted_get(record, source["eta_seconds_field"])
                elif "elapsed_seconds_field" in source:
                    elapsed = _dotted_get(record, source["elapsed_seconds_field"])
                    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
                            or step <= 0):
                        raise CampaignError(f"training cumulative elapsed/step invalid for {stage['id']}")
                    eta = max(0, 10000 - step) * elapsed / step
                else:
                    duration = _dotted_get(record, source["step_duration_seconds_field"])
                    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                            or not math.isfinite(float(duration)) or duration <= 0):
                        raise CampaignError(f"training step duration is invalid for {stage['id']}")
                    samples = stage_state.setdefault("step_duration_samples", [])
                    samples.append(float(duration))
                    del samples[:-100]
                    eta = max(0, 10000 - step) * (sum(samples) / len(samples))
                if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(float(eta)) or eta < 0:
                    raise CampaignError(f"training ETA is invalid for {stage['id']}")
                last_step = int(stage_state.get("progress_last_step", -1))
                if step <= last_step:
                    continue
                row = {
                    "schema": "self_audit.server_campaign_progress.v1",
                    "campaign_id": self.plan["campaign_id"],
                    "stage_id": stage["id"],
                    "attempt": stage_state.get("attempt", 1),
                    "event_id": f"{self.plan['campaign_id']}:{stage['id']}:{stage_state.get('attempt', 1)}:{step}",
                    "utc": _utc_now(),
                    "step": step,
                    "total_steps": 10000,
                    "eta_seconds": float(eta),
                }
                _append_jsonl(self.plan["progress_receipts_jsonl"], row)
                stage_state["progress_last_step"] = step
                self._event("training_progress", stage_id=stage["id"], attempt=stage_state.get("attempt", 1), step=step, eta_seconds=float(eta))
        except CampaignError as exc:
            stage_state["progress_error"] = str(exc)
            self._state["safety_stop_reason"] = "invalid_training_progress_receipt"
            self._save()
            return
        stage_state["progress_offset"] = offset + complete_bytes
        self._save()

    def _poll_active(self) -> bool:
        stage_id = self._state.get("active_stage")
        if not stage_id:
            return False
        stage = self._stage_by_id()[stage_id]
        state = self._state["stages"][stage_id]
        if stage["kind"] == "adopt":
            self._poll_progress(stage)
            process = stage["process"]
            expected = {key: process[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")}
            actual = process_identity(expected["pid"])
            if actual is not None:
                if not _identity_matches(actual, expected):
                    self._fail_stage(stage_id, "adopted_process_identity_changed")
                    return False
                return True
            receipt, receipt_sha = _read_receipt(stage["completion"])
            baseline_sha = state.get("receipt_sha256_before")
            fresh = baseline_sha is None or receipt_sha != baseline_sha
            if _receipt_matches(stage["completion"], receipt) and fresh:
                if state.get("progress_error"):
                    self._fail_stage(stage_id, "invalid_training_progress_receipt", receipt, receipt_sha)
                elif self._state.get("hash_drift_detected"):
                    self._fail_stage(stage_id, "hash_drift_while_adopting", receipt, receipt_sha)
                else:
                    self._complete_stage(stage_id, receipt_sha, receipt, adopted=True)
            else:
                self._fail_stage(stage_id, "adopted_process_exited_without_success_receipt", receipt, receipt_sha)
            return False

        # Stage commands run inside a tiny supervisor wrapper. The wrapper and
        # command PID identities survive controller restarts and avoid resubmit.
        execution_path = Path(state["execution_path"])
        result_path = Path(state["result_path"])
        wrapper_expected = state.get("wrapper_identity")
        wrapper_actual = process_identity(int(state["wrapper_pid"])) if state.get("wrapper_pid") else None
        try:
            execution = _read_json(execution_path) if execution_path.is_file() else None
        except CampaignError:
            self._fail_stage(stage_id, "stage_execution_identity_malformed")
            return False
        recorded_wrapper = execution.get("wrapper_identity") if execution else None
        if wrapper_expected is None:
            if recorded_wrapper and _identity_matches(process_identity(int(recorded_wrapper["pid"])), recorded_wrapper):
                state["wrapper_pid"] = recorded_wrapper["pid"]
                state["wrapper_identity"] = recorded_wrapper
                self._save()
                return True
            if state.get("status") == "launching" and not result_path.is_file():
                self._fail_stage(stage_id, "launch_unknown_refused_to_prevent_duplicate")
                return False
        if self._state.get("hash_drift_detected"):
            self._fail_stage(stage_id, "hash_drift_during_stage")
            return False
        if wrapper_expected and _identity_matches(wrapper_actual, wrapper_expected):
            self._poll_progress(stage)
            return True
        if result_path.is_file():
            try:
                result = _read_json(result_path)
            except CampaignError:
                self._fail_stage(stage_id, "stage_result_receipt_malformed")
                return False
            if result.get("stage_id") != stage_id or result.get("attempt") != state.get("attempt"):
                self._fail_stage(stage_id, "stage_result_identity_mismatch")
                return False
            receipt, receipt_sha = _read_receipt(stage["completion"])
            self._poll_progress(stage)
            exit_code = result.get("exit_code")
            oom = result.get("oom_detected") is True
            if state.get("progress_error"):
                self._fail_stage(stage_id, "invalid_training_progress_receipt", receipt, receipt_sha, exit_code=exit_code)
            elif oom:
                self._fail_stage(stage_id, "OOM_detected", receipt, receipt_sha)
            elif exit_code != 0:
                self._fail_stage(stage_id, "stage_exit_nonzero", receipt, receipt_sha, exit_code=exit_code,
                                 failure_outcome=_failure_outcome(stage, receipt))
            elif not _receipt_matches(stage["completion"], receipt):
                self._fail_stage(stage_id, "completion_receipt_mismatch", receipt, receipt_sha, exit_code=exit_code,
                                 failure_outcome=_failure_outcome(stage, receipt))
            elif self._state.get("hash_drift_detected"):
                self._fail_stage(stage_id, "hash_drift_during_stage", receipt, receipt_sha, exit_code=exit_code)
            else:
                self._complete_stage(stage_id, receipt_sha, receipt, exit_code=exit_code)
            return False

        command_identity = execution.get("command_identity") if execution else None
        if command_identity:
            command_actual = process_identity(int(command_identity["pid"]))
            if _identity_matches(command_actual, command_identity):
                self._poll_progress(stage)
                return True

        # The wrapper vanished without a result. Only a hash-verified explicit
        # resume vector may recover a started training stage; no zero restart.
        log_path = Path(state["stdout_log"])
        oom = log_path.is_file() and _OOM_RE.search(log_path.read_bytes()) is not None
        if oom:
            self._fail_stage(stage_id, "OOM_detected_without_wrapper_result")
        elif state.get("attempt") == 1 and stage.get("resume"):
            self._event("stage_interrupted_resume_authorized", stage_id=stage_id, attempt=1)
            self._state["active_stage"] = None
            self._save()
            self._launch_stage(stage, resume=True)
        elif state.get("attempt") == 1:
            self._fail_stage(stage_id, "interrupted_started_stage_has_no_explicit_resume")
        else:
            self._fail_stage(stage_id, "explicit_resume_attempt_interrupted")
        return False

    def _complete_stage(self, stage_id: str, receipt_sha: str | None, receipt: dict[str, Any] | None, **extra: Any) -> None:
        state = self._state["stages"][stage_id]
        state.update({
            "status": "succeeded",
            "completed_utc": _utc_now(),
            "receipt_sha256": receipt_sha,
            "receipt_summary": self._receipt_summary(receipt),
            **extra,
        })
        self._state["active_stage"] = None
        self._save()
        self._event("stage_succeeded", stage_id=stage_id, receipt_sha256=receipt_sha, **extra)

    @staticmethod
    def _receipt_summary(receipt: dict[str, Any] | None) -> dict[str, Any] | None:
        if receipt is None:
            return None
        # Retain only scalar status and quality values; do not copy arbitrary receipt payloads into state.
        summary: dict[str, Any] = {}
        for key in ("status", "training_complete", "remote_verified", "steps", "Macro4", "macro4"):
            value = receipt.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
        metrics = receipt.get("metrics")
        if isinstance(metrics, dict):
            for key in ("Macro4", "macro4"):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    summary[f"metrics.{key}"] = float(value)
        return summary

    def _fail_stage(
        self,
        stage_id: str,
        reason: str,
        receipt: dict[str, Any] | None = None,
        receipt_sha: str | None = None,
        **extra: Any,
    ) -> None:
        state = self._state["stages"][stage_id]
        state.update({
            "status": "failed",
            "reason": reason,
            "completed_utc": _utc_now(),
            "receipt_sha256": receipt_sha,
            "receipt_summary": self._receipt_summary(receipt),
            **{key: value for key, value in extra.items() if value is not None},
        })
        if extra.get("failure_outcome") == "UPLOAD_PENDING":
            state["upload_state"] = "UPLOAD_PENDING"
            state["remote_verified"] = False
            state["local_artifacts_retained"] = True
        self._state["active_stage"] = None
        self._save()
        self._event("stage_failed", stage_id=stage_id, reason=reason, **{k: v for k, v in extra.items() if v is not None})

    def run(self) -> dict[str, Any]:
        if self._state["status"] not in {"running"}:
            return self._state
        self.set_supervisor(os.getpid())
        while True:
            active = self._state.get("active_stage")
            hash_ok = self._recheck_plan_and_active_pins()
            stop = self._stop_requested()
            deadline = self._deadline_reached()
            if active:
                still_active = self._poll_active()
                if still_active:
                    time.sleep(float(self.plan["poll_seconds"]))
                    continue
                if self._state.get("hash_drift_detected"):
                    self._state["status"] = "blocked_hash_drift"
                    self._state["completed_utc"] = _utc_now()
                    self._state["active_stage"] = None
                    self._save()
                    return self._state
                if stop or deadline or not hash_ok:
                    self._state["stop_requested"] = stop
                    if deadline:
                        self._state["stop_reason"] = "max_wall_clock_deadline"
                    self._mark_no_more_work()
                    return self._state
                continue

            if self._state.get("safety_stop_reason"):
                for stage_state in self._state["stages"].values():
                    if stage_state["status"] == "pending":
                        stage_state.update({"status": "blocked", "reason": self._state["safety_stop_reason"], "completed_utc": _utc_now()})
                self._state["status"] = "blocked_safety"
                self._state["completed_utc"] = _utc_now()
                self._save()
                return self._state

            if self._state.get("hash_drift_detected") or not hash_ok:
                self._state["status"] = "blocked_hash_drift"
                self._state["completed_utc"] = _utc_now()
                self._save()
                return self._state
            if stop or deadline:
                self._state["stop_requested"] = stop
                if deadline:
                    self._state["stop_reason"] = "max_wall_clock_deadline"
                self._mark_no_more_work()
                return self._state

            stage = self._next_eligible_stage()
            if stage is None:
                self._mark_no_more_work()
                return self._state
            try:
                self._verify_stage_pins(stage)
                if stage["kind"] == "adopt":
                    self._start_or_monitor_adopt(stage)
                else:
                    self._launch_stage(stage)
            except CampaignError as exc:
                self._fail_stage(stage["id"], f"preflight_failed:{exc}")
            if self._state.get("active_stage"):
                time.sleep(float(self.plan["poll_seconds"]))
            else:
                continue


def _initialize_controller(plan_path: str, plan_sha256: str) -> CampaignController:
    return CampaignController(plan_path, plan_sha256)


def start_campaign(plan_path: str, plan_sha256: str) -> dict[str, Any]:
    plan, actual_sha, path = load_campaign(plan_path, plan_sha256)
    state_dir = Path(plan["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = _campaign_locked(plan)
    if lock_fd < 0:
        return {"status": "already_running", "state_dir": str(state_dir)}
    try:
        controller = _initialize_controller(str(path), actual_sha)
        if controller._state["status"] != "running":
            return {"status": controller._state["status"], "state_dir": str(state_dir)}
        old_supervisor = controller._state.get("supervisor")
        if old_supervisor and _identity_matches(process_identity(old_supervisor["pid"]), old_supervisor):
            raise CampaignError("supervisor identity is alive without holding the queue lock; refusing duplicate")
        read_fd, write_fd = os.pipe()
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_server_campaign.py"
        command = [
            sys.executable, str(script), "_worker", "--plan", str(path), "--plan-sha256", actual_sha,
            "--lock-fd", str(lock_fd), "--ready-fd", str(read_fd),
        ]
        worker_log = (state_dir / "controller.log").open("ab", buffering=0)
        try:
            worker = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.DEVNULL,
                stdout=worker_log,
                stderr=subprocess.STDOUT,
                env={"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                pass_fds=(lock_fd, read_fd),
                close_fds=True,
                start_new_session=True,
            )
        finally:
            worker_log.close()
            os.close(read_fd)
        identity = CampaignController._wait_identity(worker.pid)
        if identity is None:
            os.close(write_fd)
            raise CampaignError("cannot record detached supervisor identity on this host")
        controller._state["supervisor"] = {key: identity[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")}
        controller._state["status"] = "running"
        controller._save()
        os.write(write_fd, b"1")
        os.close(write_fd)
        os.close(lock_fd)
        return {"status": "started", "supervisor_pid": worker.pid, "state_dir": str(state_dir)}
    except BaseException:
        try:
            os.close(lock_fd)
        except OSError:
            pass
        raise


def run_foreground(plan_path: str, plan_sha256: str) -> dict[str, Any]:
    plan, actual_sha, path = load_campaign(plan_path, plan_sha256)
    lock_fd = _campaign_locked(plan)
    if lock_fd < 0:
        raise CampaignError("campaign queue lock is already held")
    try:
        controller = _initialize_controller(str(path), actual_sha)
        if controller._state["status"] != "running":
            return controller._state
        controller.run()
        return controller._state
    finally:
        os.close(lock_fd)


def _worker_main(plan_path: str, plan_sha256: str, lock_fd: int, ready_fd: int) -> int:
    try:
        ready = os.read(ready_fd, 1)
        os.close(ready_fd)
        if ready != b"1":
            return 2
        controller = _initialize_controller(plan_path, plan_sha256)
        controller.run()
        return 0
    finally:
        try:
            os.close(lock_fd)
        except OSError:
            pass


def _stage_child_main(
    plan_path: str,
    plan_sha256: str,
    stage_id: str,
    attempt: int,
    log_path: str,
    execution_path: str,
    result_path: str,
) -> int:
    plan, actual_sha, _ = load_campaign(plan_path, plan_sha256)
    if actual_sha != plan_sha256:
        return 2
    stages = {stage["id"]: stage for stage in plan["stages"]}
    if stage_id not in stages or stages[stage_id]["kind"] != "command":
        return 2
    stage = stages[stage_id]
    if attempt not in {1, 2} or (attempt == 2 and "resume" not in stage):
        return 2
    _verify_pin_set(plan, CampaignController._stage_pin_ids(stage))
    argv = stage["argv"] if attempt == 1 else stage["resume"]["argv"]
    if attempt == 2:
        for pin in stage["resume"]["checkpoint_pins"]:
            if not Path(pin["path"]).is_file() or sha256_file(pin["path"]) != pin["sha256"]:
                return 2
    wrapper_identity = process_identity(os.getpid())
    if wrapper_identity is None:
        return 2
    execution = {
        "schema": "self_audit.server_campaign_execution.v1",
        "stage_id": stage_id,
        "attempt": attempt,
        "wrapper_identity": {key: wrapper_identity[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")},
        "command_identity": None,
        "started_utc": _utc_now(),
    }
    _atomic_json(execution_path, execution)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    process: subprocess.Popen[bytes] | None = None
    wrapper_error: str | None = None
    with os.fdopen(fd, "wb", buffering=0) as output:
        try:
            process = subprocess.Popen(
                argv,
                cwd=stage["cwd"],
                env=_safe_environment(plan),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                shell=False,
                close_fds=True,
                start_new_session=False,
            )
            identity = CampaignController._wait_identity(process.pid)
            if identity is None:
                # An owned short-lived child can finish before /proc or ps is
                # observed. Popen's wait status is authoritative for that child;
                # it is never used to adopt or identify an unrelated process.
                if process.poll() is not None:
                    exit_code = process.wait()
                    execution["command_finished_before_identity_observation"] = True
                    _atomic_json(execution_path, execution)
                else:
                    process.kill()
                    process.wait()
                    wrapper_error = "command_identity_unavailable"
                    exit_code = 127
            else:
                execution["command_identity"] = {key: identity[key] for key in ("pid", "start_ticks", "cwd", "argv_sha256")}
                execution["command_started_utc"] = _utc_now()
                _atomic_json(execution_path, execution)
                exit_code = process.wait()
        except (OSError, subprocess.SubprocessError) as exc:
            wrapper_error = type(exc).__name__
            exit_code = 127
        output.flush()
        os.fsync(output.fileno())
    log_bytes = Path(log_path).read_bytes()
    result = {
        "schema": "self_audit.server_campaign_result.v1",
        "stage_id": stage_id,
        "attempt": attempt,
        "wrapper_pid": os.getpid(),
        "command_pid": process.pid if process is not None else None,
        "exit_code": exit_code,
        "oom_detected": _OOM_RE.search(log_bytes) is not None,
        "stdout_log": log_path,
        "completed_utc": _utc_now(),
    }
    if wrapper_error:
        result["wrapper_error"] = wrapper_error
    _atomic_json(result_path, result)
    return 0


def status_state(state_dir: str | Path) -> dict[str, Any]:
    path = Path(state_dir).resolve() / "STATE.json"
    if not path.is_file():
        raise CampaignError(f"campaign state file does not exist: {path}")
    return _read_json(path)


def request_stop(state_dir: str | Path, reason: str = "operator_stop") -> dict[str, Any]:
    path = Path(state_dir).resolve() / "STATE.json"
    if not path.is_file():
        raise CampaignError(f"campaign state file does not exist: {path}")
    state = _read_json(path)
    request = {"schema": "self_audit.server_campaign_stop.v1", "reason": reason, "requested_utc": _utc_now()}
    _atomic_json(path.parent / "STOP_REQUESTED.json", request)
    return {"status": "stop_requested", "active_stage": state.get("active_stage"), "state_dir": str(path.parent)}


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detached sequential one-GPU server campaign controller")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "validate the exact campaign and print its SHA-256"),
        ("start", "start one detached campaign supervisor"),
        ("run", "run the queue in the current process; intended for local fixtures"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--plan", required=True)
        if name != "validate":
            command.add_argument("--plan-sha256", required=True)
        else:
            command.add_argument("--plan-sha256")
    status = sub.add_parser("status", help="read persistent campaign state")
    status.add_argument("--state-dir", required=True)
    stop = sub.add_parser("stop", help="stop future scheduling after the active stage ends")
    stop.add_argument("--state-dir", required=True)
    identify = sub.add_parser("identify", help="print a process identity for an explicit adoption plan")
    identify.add_argument("--pid", required=True, type=int)
    identify.add_argument("--receipt", help="completion receipt to hash before adoption")

    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--plan", required=True)
    worker.add_argument("--plan-sha256", required=True)
    worker.add_argument("--lock-fd", required=True, type=int)
    worker.add_argument("--ready-fd", required=True, type=int)
    child = sub.add_parser("_stage_child", help=argparse.SUPPRESS)
    child.add_argument("--plan", required=True)
    child.add_argument("--plan-sha256", required=True)
    child.add_argument("--stage", required=True)
    child.add_argument("--attempt", required=True, type=int)
    child.add_argument("--log", required=True)
    child.add_argument("--execution", required=True)
    child.add_argument("--result", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            plan, plan_sha, _ = load_campaign(args.plan, args.plan_sha256)
            print(json.dumps({
                "status": "VALID",
                "campaign_id": plan["campaign_id"],
                "plan_sha256": plan_sha,
                "stage_order": [stage["id"] for stage in plan["stages"]],
                "training_steps": {stage["id"]: 10000 for stage in plan["stages"] if stage["role"] in _TRAINING_ROLES},
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "start":
            result = start_campaign(args.plan, args.plan_sha256)
        elif args.command == "run":
            result = run_foreground(args.plan, args.plan_sha256)
        elif args.command == "status":
            result = status_state(args.state_dir)
        elif args.command == "stop":
            result = request_stop(args.state_dir)
        elif args.command == "identify":
            result = process_identity(args.pid)
            if result is None:
                raise CampaignError("PID is not a live verifiable process")
            result.pop("state", None)
            if args.receipt:
                receipt_path = Path(args.receipt).resolve()
                result["receipt_sha256_before"] = sha256_file(receipt_path) if receipt_path.is_file() else None
        elif args.command == "_worker":
            return _worker_main(args.plan, args.plan_sha256, args.lock_fd, args.ready_fd)
        else:
            return _stage_child_main(
                args.plan, args.plan_sha256, args.stage, args.attempt,
                args.log, args.execution, args.result,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CampaignError, OSError, subprocess.SubprocessError) as exc:
        print(f"server campaign error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
