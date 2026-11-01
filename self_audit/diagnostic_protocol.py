"""Append-once diagnostic round records; prediction and label access are separate.

These are provenance checks, not a security sandbox. Callers must validate their
scientific config before locking and call authorize_score before opening GT.
Only image/model/source bytes are hashed here; reference masks are not assets.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _time():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    """Publish a complete JSON file atomically without replacing existing data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".diagnostic-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # fails on existing destination, unlike replace
    finally:
        os.unlink(temporary)
    return value


def _read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("diagnostic JSON must contain an object")
    canonical(value)
    return value


def _stage(stage):
    if stage not in ("A", "B"):
        raise ValueError("diagnostic stage must be A or B")
    return stage


def source_snapshot():
    files = sorted((ROOT / "self_audit").glob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.py")) + [ROOT / "train.py"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    environment = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ("torch", "numpy"):
        try:
            environment[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            environment[distribution] = None
    return {"git_head": commit, "environment": environment,
            "files": {str(p.relative_to(ROOT)): sha256(p) for p in files}}


def _check_source(record):
    # Content is authoritative. Committing the exact tested bytes does not alter
    # execution; record HEAD is provenance, not permission to ignore dirty code.
    current = source_snapshot()
    if record["source"]["files"] != current["files"]:
        raise ValueError("diagnostic source changed after lock")
    if record["source"].get("environment") != current.get("environment"):
        raise ValueError("diagnostic software environment changed after lock")


def _asset_record(assets):
    result = {}
    for name, asset in assets.items():
        if not isinstance(name, str) or not name:
            raise ValueError("asset name required")
        if any(token in name.lower() for token in ("mask", "ground_truth", "gt_payload")):
            raise ValueError("reference mask payloads may not be locking assets")
        path = Path(asset["path"]).resolve()
        expected = asset["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"asset hash mismatch: {name}")
        result[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    return result


def _seal(path, record):
    payload = {**record, "record_sha256": hashlib.sha256(canonical(record)).hexdigest()}
    return write_once(path, payload)


def _unseal(path):
    record = _read(path)
    expected = record.pop("record_sha256", None)
    if expected != hashlib.sha256(canonical(record)).hexdigest():
        raise ValueError("diagnostic record checksum mismatch")
    return {**record, "record_sha256": expected}


def create_lock(round_dir, stage, protocol, assets):
    """Bind a caller-validated scientific protocol to physical assets and code."""
    stage = _stage(stage)
    directory = Path(round_dir).resolve()
    if (directory / "VAL_EXPOSURE.json").exists():
        raise ValueError("VAL already exposed; cannot create a late protocol lock")
    if not isinstance(protocol, dict) or not protocol:
        raise ValueError("complete validated protocol required")
    canonical(protocol)
    previous = None
    if stage == "B":
        previous = verify_frozen(directory, "A")["record_sha256"]
    record = {
        "schema": "self_audit.diagnostics.lock.v1", "stage": stage,
        "created_utc": _time(), "protocol": protocol,
        "assets": _asset_record(assets), "source": source_snapshot(),
        "A_frozen_sha256": previous, "labels_opened": False,
    }
    return _seal(directory / f"protocol-{stage}.lock.json", record)


def read_lock(round_dir, stage, verify_assets=True):
    stage = _stage(stage)
    record = _unseal(Path(round_dir) / f"protocol-{stage}.lock.json")
    if record.get("stage") != stage or record.get("schema") != "self_audit.diagnostics.lock.v1":
        raise ValueError("wrong protocol lock schema/stage")
    _check_source(record)
    if verify_assets:
        _asset_record(record["assets"])
    return record


def freeze_predictions(round_dir, stage, files, metadata):
    """Freeze every output shard and receipt after complete, image-only prediction."""
    stage = _stage(stage)
    directory = Path(round_dir).resolve()
    if (directory / "VAL_EXPOSURE.json").exists():
        raise ValueError("cannot freeze new predictions after VAL exposure")
    lock = read_lock(directory, stage)
    paths = [Path(p).resolve() for p in files]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("prediction inventory must be nonempty and unique")
    records = []
    for path in paths:
        records.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return _seal(directory / f"predictions-{stage}.frozen.json", {
        "schema": "self_audit.diagnostics.freeze.v1", "stage": stage,
        "created_utc": _time(), "lock_sha256": lock["record_sha256"],
        "files": records, "metadata": metadata, "labels_opened": False,
    })


def verify_frozen(round_dir, stage):
    stage = _stage(stage)
    lock = read_lock(round_dir, stage)
    record = _unseal(Path(round_dir) / f"predictions-{stage}.frozen.json")
    if (record.get("schema") != "self_audit.diagnostics.freeze.v1"
            or record.get("stage") != stage or record.get("lock_sha256") != lock["record_sha256"]):
        raise ValueError("prediction freeze/lock mismatch")
    if not record.get("files"):
        raise ValueError("empty prediction freeze")
    for item in record["files"]:
        path = Path(item["path"])
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("frozen prediction changed")
    return record


def authorize_score(round_dir, stage, b_blocked_reason=None):
    """Verify both diagnostics before first GT access, or explicitly expose A alone.

    A failed/negative result never blocks B. Only an explicit missing/invalid B
    prerequisite permits the A-only route; that round can never later lock B.
    """
    stage = _stage(stage)
    directory = Path(round_dir).resolve()
    frozen = {"A": verify_frozen(directory, "A")}
    if b_blocked_reason is not None:
        if stage != "A" or not isinstance(b_blocked_reason, str) or not b_blocked_reason.strip():
            raise ValueError("a concrete B blocker permits only independent A scoring")
        if (directory / "predictions-B.frozen.json").exists():
            raise ValueError("B predictions exist; use the shared-round scoring path")
    else:
        frozen["B"] = verify_frozen(directory, "B")
        b_lock = read_lock(directory, "B")
        if b_lock["A_frozen_sha256"] != frozen["A"]["record_sha256"]:
            raise ValueError("B protocol is not bound to this A prediction round")
    path = directory / "VAL_EXPOSURE.json"
    identities = {key: value["record_sha256"] for key, value in frozen.items()}
    if path.exists():
        previous = _unseal(path)
        if previous["prediction_freezes"] != identities or previous["B_blocked_reason"] != b_blocked_reason:
            raise ValueError("score request conflicts with prior VAL exposure ledger")
        return previous
    return _seal(path, {
        "schema": "self_audit.diagnostics.exposure.v1", "created_utc": _time(),
        "prediction_freezes": identities, "B_blocked_reason": b_blocked_reason,
        "both_protocols_locked_before_first_GT": b_blocked_reason is None,
        "first_score_stage": stage, "meaning": "GT access authorized only for scoring these frozen predictions",
    })
