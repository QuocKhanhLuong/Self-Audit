#!/usr/bin/env python3
"""Fail-closed offload of completed joint A-3D checkpoints.

The training supervisor keeps teacher-stage checkpoints on the server because
later student jobs depend on them.  A completed joint job has no downstream
training dependency, so its large training payload may be copied to an
external backup.  This module validates the local copy and then performs one
remote, hash-checked receipt-and-unlink transaction.  The receipt is kept next
to the checkpoint manifest and is the only accepted substitute for a missing
``weights.pt`` payload.

The command has a local mode for tests and a ``--remote-host`` mode for the
server.  Paths are passed as argument-list values to ``ssh``/``scp``; no shell
interpolation is used.  ``--execute`` is required before any remote payload is
removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_audit.model import digest

OFFLOAD_SCHEMA = "self_audit.candidate_a3d.offload_receipt.v1"
OFFLOAD_RECEIPT_NAME = "OFFLOAD_RECEIPT.json"
JOINT_STAGE = "joint"


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"JSON object expected: {path}")
    return value


def _sha256(path: Path) -> str:
    return digest(path)


def _bytes(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError as exc:
        raise ValueError(f"file is not readable: {path}") from exc


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _safe_relative(path: Path, root: Path) -> Path:
    path = path.resolve()
    root = root.resolve()
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside declared root: {path}") from exc


def _remote_is_within(path: str, root: str) -> bool:
    """Check remote POSIX paths lexically; do not resolve them on macOS.

    The server paths do not exist in the local filesystem, and on macOS
    ``Path.resolve`` rewrites ``/home`` through a local system symlink.  Remote
    containment must therefore use POSIX path semantics until the remote
    transaction performs its own realpath and symlink checks.
    """

    try:
        PurePosixPath(path).relative_to(PurePosixPath(root))
    except ValueError:
        return False
    return True


def _slug(job_key: str) -> str:
    if not isinstance(job_key, str) or not job_key:
        raise ValueError("job key must be a non-empty string")
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", job_key.split("::")[-1]).strip("._") or "job"
    return f"{readable}-{hashlib.sha256(job_key.encode()).hexdigest()[:12]}"


def _expected(
    *,
    state_job: Mapping[str, Any],
    record: Mapping[str, Any],
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    report_sha256: str,
) -> dict[str, Any]:
    """Build the immutable identity bound into an offload receipt."""

    required = ("key", "dataset", "k", "arm", "stage", "config_sha256")
    missing = [name for name in required if name not in state_job]
    if missing:
        raise ValueError(f"campaign state job is missing {missing}")
    expected = {
        "job_key": str(state_job["key"]),
        "dataset": str(state_job["dataset"]),
        "k": int(state_job["k"]),
        "arm": str(state_job["arm"]),
        "stage": str(state_job["stage"]),
        "config_sha256": str(state_job["config_sha256"]),
        "source_sha256": str(manifest.get("source_sha256", "")),
        "data_sha256": str(plan.get("train_sha256", "")),
        "manifest_sha256": manifest_sha256,
        "report_sha256": report_sha256,
        "payload_sha256": str(manifest.get("payload_sha256", "")),
    }
    if (not _is_sha256(expected["config_sha256"])
            or not _is_sha256(expected["source_sha256"])
            or not _is_sha256(expected["data_sha256"])
            or not _is_sha256(expected["manifest_sha256"])
            or not _is_sha256(expected["report_sha256"])
            or not _is_sha256(expected["payload_sha256"])):
        raise ValueError("manifest/plan lacks source, data or payload hash")
    if expected["stage"] != JOINT_STAGE:
        raise ValueError("only completed joint jobs may be offloaded")
    if record.get("stage") not in (None, JOINT_STAGE):
        raise ValueError("job record stage disagrees with campaign state")
    return expected


def _student_export_meta(export_dir: Path, *, parent_manifest_sha256: str) -> dict[str, Any]:
    """Validate a self-contained student export before it becomes a receipt."""

    export_dir = export_dir.resolve()
    manifest_path = export_dir / "MANIFEST.json"
    payload_path = export_dir / "weights.pt"
    if not manifest_path.is_file() or not payload_path.is_file():
        raise ValueError(f"student export is incomplete: {export_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != "self_audit.candidate_a3d.student.v1":
        raise ValueError("student export is not a self-contained A-3D export")
    if manifest.get("parent_manifest_sha256") != parent_manifest_sha256:
        raise ValueError("student export is not derived from this final checkpoint")
    payload_sha = _sha256(payload_path)
    if manifest.get("payload_sha256") != payload_sha:
        raise ValueError("student export payload hash mismatch")
    return {
        "path": str(export_dir),
        "manifest_sha256": _sha256(manifest_path),
        "payload_sha256": payload_sha,
        "payload_bytes": _bytes(payload_path),
        "parent_manifest_sha256": parent_manifest_sha256,
    }


def validate_offload_receipt(
    receipt_path: Path,
    *,
    expected: Mapping[str, Any],
    verify_local: bool = False,
) -> tuple[bool, str]:
    """Validate an OFFLOADED receipt without treating missing payload as valid.

    ``verify_local`` is used by the transfer helper.  The remote supervisor and
    report generator set it to false because the local backup path is outside
    the server filesystem; the receipt still binds the exact hashes and job
    identity that the transfer transaction verified before unlinking.
    """

    if not receipt_path.is_file() or receipt_path.is_symlink():
        return False, f"missing offload receipt: {receipt_path}"
    try:
        receipt = _read_json(receipt_path)
    except (OSError, TypeError, ValueError) as exc:
        return False, str(exc)
    required = {
        "schema", "status", "job_key", "dataset", "k", "arm", "stage",
        "config_sha256", "source_sha256", "data_sha256", "manifest_sha256",
        "report_sha256", "payload_sha256", "payload_bytes", "local_backup_path",
        "local_backup_sha256", "local_backup_bytes", "remote_payload_sha256",
        "remote_payload_bytes", "remote_payload_removed", "student_export",
        "created_at", "remote_verified_at",
    }
    missing = sorted(required - set(receipt))
    if missing:
        return False, f"offload receipt missing {missing}"
    if receipt.get("schema") != OFFLOAD_SCHEMA or receipt.get("status") != "OFFLOADED":
        return False, "offload receipt schema/status mismatch"
    for key in ("job_key", "dataset", "arm", "stage"):
        if str(receipt.get(key)) != str(expected.get(key)):
            return False, f"offload receipt {key} mismatch"
    for key in ("config_sha256", "source_sha256", "data_sha256", "manifest_sha256",
                "report_sha256", "payload_sha256", "local_backup_sha256",
                "remote_payload_sha256"):
        if not _is_sha256(receipt.get(key)):
            return False, f"offload receipt {key} is not a SHA-256"
        expected_value = expected.get(key)
        if key in {"local_backup_sha256", "remote_payload_sha256"}:
            expected_value = expected.get("payload_sha256")
        if str(receipt.get(key)) != str(expected_value):
            return False, f"offload receipt {key} mismatch"
    if receipt.get("k") != expected.get("k"):
        return False, "offload receipt k mismatch"
    if receipt.get("stage") != JOINT_STAGE:
        return False, "teacher-stage checkpoint cannot be offloaded"
    for key in ("payload_bytes", "local_backup_bytes", "remote_payload_bytes"):
        value = receipt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False, f"offload receipt {key} is invalid"
    if not (receipt["payload_bytes"] == receipt["local_backup_bytes"] == receipt["remote_payload_bytes"]):
        return False, "offload receipt payload sizes disagree"
    if isinstance(expected.get("payload_bytes"), int) and receipt["payload_bytes"] != expected["payload_bytes"]:
        return False, "offload receipt payload size mismatch"
    if not receipt.get("remote_payload_removed"):
        return False, "offload receipt does not attest remote payload removal"
    for key in ("local_backup_path", "created_at", "remote_verified_at"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            return False, f"offload receipt {key} is invalid"
    export = receipt.get("student_export")
    if not isinstance(export, Mapping):
        return False, "offload receipt has no student export attestation"
    for key in ("path", "manifest_sha256", "payload_sha256", "parent_manifest_sha256"):
        if not isinstance(export.get(key), str) or not export[key]:
            return False, f"student export attestation {key} is invalid"
    if export.get("parent_manifest_sha256") != expected["manifest_sha256"]:
        return False, "student export parent manifest mismatch"
    if not isinstance(export.get("payload_bytes"), int) or export["payload_bytes"] < 0:
        return False, "student export payload size is invalid"
    if verify_local:
        payload = Path(str(receipt["local_backup_path"])).resolve()
        if not payload.is_file() or payload.is_symlink():
            return False, f"local backup payload is missing: {payload}"
        if _bytes(payload) != receipt["local_backup_bytes"] or _sha256(payload) != receipt["local_backup_sha256"]:
            return False, "local backup payload changed after transfer"
        export_dir = Path(str(export["path"])).resolve()
        try:
            verified = _student_export_meta(export_dir, parent_manifest_sha256=expected["manifest_sha256"])
        except (OSError, TypeError, ValueError) as exc:
            return False, str(exc)
        for key in ("manifest_sha256", "payload_sha256", "payload_bytes", "parent_manifest_sha256"):
            if verified[key] != export[key]:
                return False, f"student export {key} changed after transfer"
    return True, "validated offload receipt"


def _discover_export(out: Path) -> Path:
    candidates = (out / "student", out / "student_export", out / "export")
    found = [candidate for candidate in candidates if candidate.is_dir()]
    if len(found) != 1:
        raise ValueError("joint job must have exactly one student export under student/, student_export/ or export/")
    return found[0]


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _require_backup_space(root: Path, required_bytes: int) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(root).free)
    # Keep a small operational margin so a receipt/manifest write cannot fill
    # the filesystem after the payload copy succeeds.
    margin = 64 * 1024 * 1024
    if free < int(required_bytes) + margin:
        raise RuntimeError(
            f"local backup free space is insufficient: free={free} required={required_bytes + margin}"
        )


def _local_offload(
    *,
    state_job: Mapping[str, Any],
    record: Mapping[str, Any],
    plan: Mapping[str, Any],
    out: Path,
    backup_root: Path,
    execute: bool,
) -> dict[str, Any]:
    final = out / "final"
    manifest_path, payload_path, report_path = final / "MANIFEST.json", final / "weights.pt", out / "REPORT.json"
    if state_job.get("status") != "DONE":
        raise ValueError("only a DONE job can be offloaded")
    if state_job.get("stage") != JOINT_STAGE:
        raise ValueError("teacher-stage checkpoints are retained and cannot be offloaded")
    if not manifest_path.is_file() or not payload_path.is_file() or not report_path.is_file():
        raise ValueError("joint job is missing final manifest, payload or report")
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    if report.get("status") != "COMPLETED" or report.get("stage") != JOINT_STAGE:
        raise ValueError("joint job is not COMPLETED")
    manifest_sha = _sha256(manifest_path)
    expected = _expected(state_job=state_job, record=record, plan=plan, manifest=manifest,
                         manifest_sha256=manifest_sha, report_sha256=_sha256(report_path))
    if _sha256(payload_path) != expected["payload_sha256"]:
        raise ValueError("remote payload hash does not match manifest")
    export_dir = _discover_export(out)
    slug = _slug(str(state_job["key"]))
    backup_root = backup_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    export_bytes = sum(_bytes(child) for child in export_dir.iterdir()
                       if child.is_file() and not child.is_symlink()
                       and child.name in {"MANIFEST.json", "weights.pt"})
    _require_backup_space(backup_root, _bytes(payload_path) + export_bytes)
    target = backup_root / slug
    if target.exists():
        raise ValueError(f"backup target already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".offload-{slug}-", dir=backup_root))
    try:
        local_final = temporary / "final"
        local_export = temporary / "student"
        _copy_file(manifest_path, local_final / "MANIFEST.json")
        _copy_file(report_path, temporary / "REPORT.json")
        _copy_file(payload_path, local_final / "weights.pt")
        for child in export_dir.iterdir():
            if child.is_symlink() or not child.is_file() or child.name not in {"MANIFEST.json", "weights.pt"}:
                continue
            _copy_file(child, local_export / child.name)
        local_export_meta = _student_export_meta(local_export, parent_manifest_sha256=manifest_sha)
        receipt = {
            "schema": OFFLOAD_SCHEMA,
            "status": "OFFLOADED",
            **{key: expected[key] for key in ("job_key", "dataset", "k", "arm", "stage",
                                                "config_sha256", "source_sha256", "data_sha256",
                                                "manifest_sha256", "report_sha256", "payload_sha256")},
            "payload_bytes": _bytes(payload_path),
            "local_backup_path": str((target / "final" / "weights.pt").resolve()),
            "local_backup_sha256": _sha256(local_final / "weights.pt"),
            "local_backup_bytes": _bytes(local_final / "weights.pt"),
            "remote_payload_sha256": _sha256(payload_path),
            "remote_payload_bytes": _bytes(payload_path),
            "remote_payload_removed": True,
            "student_export": {**local_export_meta, "path": str((target / "student").resolve())},
            "created_at": _now(),
            "remote_verified_at": _now(),
            "transfer": {"method": "local_copy_verified", "source": str(payload_path)},
        }
        _write_json_atomic(temporary / OFFLOAD_RECEIPT_NAME, receipt)
        if not execute:
            shutil.rmtree(temporary)
            return {"status": "DRY_RUN", "job_key": expected["job_key"],
                    "payload_bytes": receipt["payload_bytes"], "backup_target": str(target)}
        # Make the local directory durable before deleting the remote payload.
        temporary.rename(target)
        receipt_path = target / OFFLOAD_RECEIPT_NAME
        # Recheck the copied files after the directory rename, then delete only
        # the exact joint payload.  Manifests, reports and exports remain.
        ok, reason = validate_offload_receipt(receipt_path, expected=expected, verify_local=True)
        if not ok:
            raise ValueError(reason)
        if _sha256(payload_path) != expected["payload_sha256"]:
            raise ValueError("remote payload changed before unlink")
        # The receipt is written beside the remote manifest immediately before
        # the exact payload unlink.  If the process dies between these two
        # operations, the intact payload remains a valid ordinary checkpoint;
        # a later verifier will never accept a missing payload without this
        # receipt.
        _write_json_atomic(final / OFFLOAD_RECEIPT_NAME, receipt)
        payload_path.unlink()
        if payload_path.exists():
            raise RuntimeError("remote payload still exists after offload")
        # The receipt beside the remote manifest is authoritative; mirror it
        # into the local backup for offline report generation.
        _write_json_atomic(target / OFFLOAD_RECEIPT_NAME, receipt)
        return {"status": "OFFLOADED", "job_key": expected["job_key"],
                "backup_target": str(target), "receipt": str(receipt_path),
                "payload_bytes": receipt["payload_bytes"]}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


_REMOTE_TRANSACTION = r'''
import hashlib, json, os, pathlib, re, sys, datetime

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require_sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise SystemExit("invalid expected " + name)
    return value

def reject_symlink(path, name):
    if path.is_symlink():
        raise SystemExit(name + " must not be a symlink")

def reject_symlink_components(path, root, name):
    reject_symlink(root, "remote campaign root")
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise SystemExit(name + " is outside the declared campaign root")
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        reject_symlink(cursor, name)

out_raw = pathlib.Path(sys.argv[1])
root_raw = pathlib.Path(sys.argv[2])
state_path = pathlib.Path(sys.argv[3])
job_key = sys.argv[4]
export_raw = pathlib.Path(sys.argv[5])
if out_raw.is_symlink() or root_raw.is_symlink() or state_path.is_symlink():
    raise SystemExit("remote campaign/state paths must not be symlinks")
out = out_raw.resolve()
root = root_raw.resolve()
reject_symlink_components(out, root, "remote output")
if not out.is_relative_to(root):
    raise SystemExit("remote output is outside the declared campaign root")
export = export_raw.resolve()
reject_symlink_components(export, root, "student export")
if not export.is_relative_to(out):
    raise SystemExit("student export is outside the job output")
if not state_path.resolve().is_relative_to(root):
    raise SystemExit("remote state is outside the declared campaign root")
reject_symlink_components(state_path.resolve(), root, "remote state")
if not state_path.is_file():
    raise SystemExit("remote campaign state is missing")
state = json.loads(state_path.read_text())
jobs = {str(item.get("key")): item for item in state.get("jobs", []) if isinstance(item, dict)}
job = jobs.get(job_key)
if job is None or job.get("status") != "DONE" or job.get("stage") != "joint":
    raise SystemExit("remote state no longer marks this job DONE/joint")
if pathlib.Path(str(job.get("out", ""))).resolve() != out:
    raise SystemExit("remote state output changed")
expected = json.load(sys.stdin)
if expected.get("job_key") != job_key or expected.get("stage") != "joint":
    raise SystemExit("offload identity does not match the remote state")
for field in ("dataset", "k", "arm", "stage"):
    if str(job.get(field)) != str(expected.get(field)):
        raise SystemExit("offload " + field + " does not match the remote state")
for field in ("config_sha256", "source_sha256", "data_sha256", "manifest_sha256",
              "report_sha256", "payload_sha256", "local_backup_sha256",
              "remote_payload_sha256"):
    require_sha(expected.get(field), field)
if not require_sha(job.get("data_sha256"), "state data_sha256"):
    raise SystemExit("remote state has no pinned data hash")
if job.get("data_sha256") != expected.get("data_sha256"):
    raise SystemExit("remote state data hash changed before offload")
for field in ("payload_bytes", "local_backup_bytes", "remote_payload_bytes"):
    if isinstance(expected.get(field), bool) or not isinstance(expected.get(field), int) or expected[field] < 0:
        raise SystemExit("invalid expected payload size")
final = out / "final"
manifest_path, payload_path, report_path = final / "MANIFEST.json", final / "weights.pt", out / "REPORT.json"
if final.is_symlink() or any(path.is_symlink() for path in (manifest_path, payload_path, report_path)):
    raise SystemExit("remote checkpoint files must not be symlinks")
if not payload_path.is_file() or not manifest_path.is_file() or not report_path.is_file():
    raise SystemExit("remote joint artifact is incomplete")
report = json.loads(report_path.read_text())
if report.get("status") != "COMPLETED" or report.get("stage") != "joint":
    raise SystemExit("remote report is not a completed joint report")
manifest_sha = sha(manifest_path)
report_sha = sha(report_path)
payload_sha = sha(payload_path)
if manifest_sha != expected["manifest_sha256"] or report_sha != expected["report_sha256"]:
    raise SystemExit("remote manifest/report changed before offload")
if payload_sha != expected["payload_sha256"]:
    raise SystemExit("remote payload changed before offload")
if expected["remote_payload_sha256"] != payload_sha or expected["local_backup_sha256"] != payload_sha:
    raise SystemExit("remote/local payload hash attestation disagrees")
if int(payload_path.stat().st_size) != int(expected["payload_bytes"]):
    raise SystemExit("remote payload size changed before offload")
if not (int(expected["local_backup_bytes"]) == int(expected["remote_payload_bytes"]) == int(expected["payload_bytes"])):
    raise SystemExit("payload size attestation disagrees")
manifest = json.loads(manifest_path.read_text())
if manifest.get("source_sha256") != expected["source_sha256"] or manifest.get("data_sha256") != expected["data_sha256"]:
    raise SystemExit("remote manifest provenance changed before offload")
if manifest.get("payload_sha256") != payload_sha:
    raise SystemExit("remote manifest payload hash mismatch")
if str(job.get("config_sha256")) != str(expected["config_sha256"]):
    raise SystemExit("remote state config hash changed before offload")
if manifest.get("config") is None:
    raise SystemExit("remote manifest has no declared config")
export_manifest = export / "MANIFEST.json"
export_payload = export / "weights.pt"
if not export_manifest.is_file() or not export_payload.is_file():
    raise SystemExit("remote student export is incomplete")
if export_manifest.is_symlink() or export_payload.is_symlink():
    raise SystemExit("remote student export files must not be symlinks")
if sha(export_manifest) != expected["student_export"]["manifest_sha256"]:
    raise SystemExit("remote student export manifest changed before offload")
if sha(export_payload) != expected["student_export"]["payload_sha256"]:
    raise SystemExit("remote student export payload changed before offload")
if int(export_payload.stat().st_size) != int(expected["student_export"]["payload_bytes"]):
    raise SystemExit("remote student export payload size changed before offload")
export_meta = json.loads(export_manifest.read_text())
if export_meta.get("parent_manifest_sha256") != expected["manifest_sha256"]:
    raise SystemExit("remote student export parent manifest changed before offload")
receipt = dict(expected)
receipt.update(schema="self_audit.candidate_a3d.offload_receipt.v1", status="OFFLOADED")
receipt.update(remote_payload_sha256=payload_sha,
              remote_payload_bytes=int(payload_path.stat().st_size),
              remote_payload_removed=True, remote_verified_at=now())
receipt_path = final / "OFFLOAD_RECEIPT.json"
if receipt_path.exists() and receipt_path.is_symlink():
    raise SystemExit("remote offload receipt must not be a symlink")
temp = receipt_path.with_name(".OFFLOAD_RECEIPT.json.tmp-%d" % os.getpid())
temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
temp.replace(receipt_path)
payload_path.unlink()
if payload_path.exists():
    raise SystemExit("remote payload was not removed")
print(json.dumps({"receipt": str(receipt_path), "payload_sha256": payload_sha,
                  "payload_bytes": int(receipt["payload_bytes"])}, sort_keys=True))
'''


def _run_remote(argv: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, input=input_bytes, text=False, capture_output=True, check=True)


def _validate_host(host: str) -> None:
    if not host or host.startswith("-") or any(char in host for char in "\r\n \t"):
        raise ValueError("remote host must be a single SSH host token")


def _remote_cat(host: str, path: str) -> bytes:
    _validate_host(host)
    if not path.startswith("/") or "\n" in path or "\r" in path:
        raise ValueError("remote paths must be absolute and newline-free")
    command = shlex.join(["cat", "--", path])
    return _run_remote(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
                        command]).stdout


def _remote_copy(host: str, source: str, target: Path) -> None:
    _validate_host(host)
    if not source.startswith("/") or "\n" in source or "\r" in source:
        raise ValueError("remote source path must be absolute and newline-free")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["cat", "--", source])
    with target.open("wb") as stream:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", host,
                        command], stdout=stream, stderr=subprocess.PIPE, check=True)


_REMOTE_SIZE_QUERY = r'''
import json, pathlib, sys
total = 0
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    if path.is_symlink() or not path.is_file():
        raise SystemExit("remote transfer source is missing or symlinked: " + raw)
    total += path.stat().st_size
print(json.dumps({"bytes": total}, sort_keys=True))
'''


def offload_remote_job(
    *,
    state: Mapping[str, Any],
    state_path: Path,
    job_key: str,
    remote_host: str,
    remote_campaign_root: str,
    backup_root: Path,
    execute: bool,
    student_export_path: str | None = None,
    remote_state_path: str | None = None,
    python_bin: str = "/venv/main/bin/python",
) -> dict[str, Any]:
    """Copy and attest one remote joint job, then unlink its remote payload."""

    jobs = {str(job.get("key")): job for job in state.get("jobs", []) if isinstance(job, Mapping)}
    try:
        state_job = jobs[job_key]
    except KeyError as exc:
        raise ValueError(f"job key is not present in campaign state: {job_key}") from exc
    if state_job.get("status") != "DONE":
        raise ValueError("remote offload requires a DONE job")
    if state_job.get("stage") != JOINT_STAGE:
        raise ValueError("remote offload refuses teacher-stage jobs")
    out = str(state_job.get("out", ""))
    if not out.startswith("/") or not remote_campaign_root.startswith("/"):
        raise ValueError("remote campaign/output paths must be absolute")
    _validate_host(remote_host)
    if not remote_state_path or not remote_state_path.startswith("/"):
        raise ValueError("remote campaign state path must be absolute")
    if not _remote_is_within(out, remote_campaign_root):
        raise ValueError("job output is outside the declared remote campaign root")
    remote_final = f"{out.rstrip('/')}/final"
    slug = _slug(job_key)
    backup_root = backup_root.resolve()
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / slug
    if target.exists():
        raise ValueError(f"backup target already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".offload-{slug}-", dir=backup_root))
    try:
        export_path = student_export_path or f"{out.rstrip('/')}/student"
        if not export_path.startswith("/"):
            raise ValueError("student export path must be absolute")
        # Query the exact remote inputs before allocating a temporary local
        # copy.  The query rejects symlinks and avoids discovering an out of
        # space condition only after a multi-gigabyte stream has started.
        remote_sources = [
            f"{remote_final}/MANIFEST.json", f"{out.rstrip('/')}/REPORT.json",
            f"{remote_final}/weights.pt", f"{export_path.rstrip('/')}/MANIFEST.json",
            f"{export_path.rstrip('/')}/weights.pt",
        ]
        size_command = shlex.join([python_bin, "-c", _REMOTE_SIZE_QUERY, *remote_sources])
        size_result = _run_remote([
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", remote_host,
            size_command,
        ])
        try:
            required_bytes = int(json.loads(size_result.stdout.decode())["bytes"])
        except (UnicodeDecodeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("remote size preflight returned invalid JSON") from exc
        _require_backup_space(backup_root, required_bytes)
        local_final = temporary / "final"
        _remote_copy(remote_host, f"{remote_final}/MANIFEST.json", local_final / "MANIFEST.json")
        _remote_copy(remote_host, f"{out.rstrip('/')}/REPORT.json", temporary / "REPORT.json")
        _remote_copy(remote_host, f"{remote_final}/weights.pt", local_final / "weights.pt")
        local_export = temporary / "student"
        _remote_copy(remote_host, f"{export_path.rstrip('/')}/MANIFEST.json", local_export / "MANIFEST.json")
        _remote_copy(remote_host, f"{export_path.rstrip('/')}/weights.pt", local_export / "weights.pt")
        manifest = _read_json(local_final / "MANIFEST.json")
        report = _read_json(temporary / "REPORT.json")
        if report.get("status") != "COMPLETED" or report.get("stage") != JOINT_STAGE:
            raise ValueError("remote joint report is not COMPLETED")
        manifest_sha = _sha256(local_final / "MANIFEST.json")
        data_sha = str(state_job.get("data_sha256", ""))
        if not _is_sha256(data_sha) or manifest.get("data_sha256") != data_sha:
            raise ValueError("remote state/checkpoint data manifest hash mismatch")
        expected = {
            "job_key": job_key, "dataset": str(state_job["dataset"]), "k": int(state_job["k"]),
            "arm": str(state_job["arm"]), "stage": JOINT_STAGE,
            "config_sha256": str(state_job["config_sha256"]),
            "source_sha256": str(manifest.get("source_sha256", "")),
            "data_sha256": data_sha,
            "manifest_sha256": manifest_sha, "report_sha256": _sha256(temporary / "REPORT.json"),
            "payload_sha256": str(manifest.get("payload_sha256", "")),
            "payload_bytes": _bytes(local_final / "weights.pt"),
            "local_backup_path": str((target / "final" / "weights.pt").resolve()),
            "local_backup_sha256": _sha256(local_final / "weights.pt"),
            "local_backup_bytes": _bytes(local_final / "weights.pt"),
        }
        for field in ("config_sha256", "source_sha256", "data_sha256", "manifest_sha256",
                      "report_sha256", "payload_sha256"):
            if not _is_sha256(expected[field]):
                raise ValueError(f"remote offload provenance field is not SHA-256: {field}")
        export_meta = _student_export_meta(local_export, parent_manifest_sha256=manifest_sha)
        expected["student_export"] = {**export_meta, "path": str((target / "student").resolve())}
        expected.update(created_at=_now(), remote_verified_at=_now(), remote_payload_sha256=expected["payload_sha256"],
                        remote_payload_bytes=expected["payload_bytes"], remote_payload_removed=True,
                        transfer={"method": "scp+ssh_transaction", "remote_host": remote_host})
        # The local copy must be complete before the server transaction is
        # allowed to delete its payload.
        # Validate against the temporary paths first; after the durable rename,
        # the transaction receipt is rebound to the final backup paths.
        local_expected = dict(expected)
        local_expected["local_backup_path"] = str((temporary / "final" / "weights.pt").resolve())
        local_expected["student_export"] = {**export_meta, "path": str(local_export.resolve())}
        _write_json_atomic(temporary / OFFLOAD_RECEIPT_NAME,
                           {"schema": OFFLOAD_SCHEMA, "status": "OFFLOADED", **local_expected})
        ok, reason = validate_offload_receipt(
            temporary / OFFLOAD_RECEIPT_NAME, expected=local_expected, verify_local=True)
        if not ok:
            raise ValueError(reason)
        if not execute:
            shutil.rmtree(temporary)
            return {"status": "DRY_RUN", "job_key": job_key, "backup_target": str(target),
                    "payload_bytes": expected["payload_bytes"]}
        temporary.rename(target)
        transaction = {"schema": OFFLOAD_SCHEMA, "status": "OFFLOADED", **expected}
        transaction["local_backup_path"] = str((target / "final" / "weights.pt").resolve())
        transaction["student_export"] = {**export_meta, "path": str((target / "student").resolve())}
        # Pass the receipt to the remote transaction over stdin.  Paths are
        # argv values and are checked there again immediately before unlink.
        script = _REMOTE_TRANSACTION
        # OpenSSH concatenates arguments after the host into a remote shell
        # command.  Quote the complete command as one shell-safe token so a
        # path or job key can never become shell syntax on the server.
        remote_command = shlex.join([
            python_bin, "-c", script, out, remote_campaign_root,
            remote_state_path, job_key, export_path,
        ])
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", remote_host,
                   remote_command]
        result = _run_remote(command, input_bytes=json.dumps(transaction, sort_keys=True).encode())
        try:
            remote_result = json.loads(result.stdout.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("remote offload transaction returned invalid JSON") from exc
        _remote_copy(remote_host, f"{remote_final}/{OFFLOAD_RECEIPT_NAME}", target / OFFLOAD_RECEIPT_NAME)
        ok, reason = validate_offload_receipt(target / OFFLOAD_RECEIPT_NAME, expected=expected, verify_local=True)
        if not ok:
            raise ValueError(reason)
        if remote_result.get("payload_sha256") != expected["payload_sha256"]:
            raise ValueError("remote transaction payload hash mismatch")
        return {"status": "OFFLOADED", "job_key": job_key, "backup_target": str(target),
                "receipt": str(target / OFFLOAD_RECEIPT_NAME), "payload_bytes": expected["payload_bytes"]}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _load_local_state(path: Path) -> dict[str, Any]:
    return _read_json(path.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="local campaign state JSON")
    parser.add_argument("--job-key", action="append", required=True, help="exact state job key; repeatable")
    parser.add_argument("--backup-root", required=True, help="local durable backup root")
    parser.add_argument("--execute", action="store_true", help="delete validated remote/local payload")
    parser.add_argument("--remote-host", help="SSH host; omit for local fixture mode")
    parser.add_argument("--remote-campaign-root", help="absolute remote campaign root")
    parser.add_argument("--remote-state", help="absolute remote campaign state JSON")
    parser.add_argument("--student-export", help="absolute remote student export path (single job mode)")
    parser.add_argument("--python-bin", default="/venv/main/bin/python")
    args = parser.parse_args(argv)
    state_path = Path(args.state).resolve()
    state = _load_local_state(state_path)
    results = []
    for key in args.job_key:
        jobs = {str(job.get("key")): job for job in state.get("jobs", []) if isinstance(job, Mapping)}
        if key not in jobs:
            raise SystemExit(f"job key is not present in state: {key}")
        if args.remote_host:
            if not args.remote_campaign_root:
                raise SystemExit("--remote-campaign-root is required with --remote-host")
            if not args.remote_state:
                raise SystemExit("--remote-state is required with --remote-host")
            result = offload_remote_job(
                state=state, state_path=state_path, job_key=key,
                remote_host=args.remote_host, remote_campaign_root=args.remote_campaign_root,
                backup_root=Path(args.backup_root), execute=args.execute,
                student_export_path=args.student_export, remote_state_path=args.remote_state,
                python_bin=args.python_bin,
            )
        else:
            job = jobs[key]
            out = Path(str(job.get("out", ""))).resolve()
            # Local fixture mode uses the state record's output and a minimal
            # plan record stored next to it when present.
            plan_path = Path(str(job.get("plan", ""))).resolve() if job.get("plan") else None
            plan = _read_json(plan_path) if plan_path and plan_path.is_file() else {
                "train_sha256": _read_json(out / "final" / "MANIFEST.json").get("data_sha256", "")}
            result = _local_offload(
                state_job=job, record=job, plan=plan, out=out,
                backup_root=Path(args.backup_root), execute=args.execute,
            )
        results.append(result)
    print(json.dumps({"status": "OFFLOADED" if args.execute else "DRY_RUN", "jobs": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
