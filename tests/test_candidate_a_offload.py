"""Fail-closed tests for the semi-supervised joint-checkpoint offload."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.candidate_a_offload import (
    _REMOTE_TRANSACTION,
    _expected,
    _local_offload,
    _student_export_meta,
    offload_remote_job,
    validate_offload_receipt,
)
from self_audit.model import digest


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, dict, dict, Path, Path]:
    out = tmp_path / "job"
    final = out / "final"
    student = out / "student"
    final.mkdir(parents=True)
    student.mkdir()
    payload = final / "weights.pt"
    payload.write_bytes(b"joint-checkpoint")
    manifest = {
        "schema": "self_audit.candidate_a3d.checkpoint.v1",
        "config": {"stage": "joint", "epochs": 1},
        "source_sha256": _canonical_hash({"checkout": "fixture"}),
        "data_sha256": _canonical_hash({"train": "fixture"}),
        "payload_sha256": digest(payload),
        "epoch": 1,
    }
    (final / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"status": "COMPLETED", "stage": "joint", "epochs": 1}
    (out / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    student_payload = student / "weights.pt"
    student_payload.write_bytes(b"student-only")
    student_manifest = {
        "schema": "self_audit.candidate_a3d.student.v1",
        "parent_manifest_sha256": digest(final / "MANIFEST.json"),
        "payload_sha256": digest(student_payload),
    }
    (student / "MANIFEST.json").write_text(json.dumps(student_manifest, indent=2) + "\n")
    state_job = {
        "key": "campaign/k2/PLAN.json::s42_full",
        "dataset": "acdc",
        "k": 2,
        "arm": "full",
        "stage": "joint",
        "config_sha256": "c" * 64,
        "status": "DONE",
        "out": str(out),
    }
    plan = {"train_sha256": manifest["data_sha256"]}
    return state_job, state_job, plan, out, tmp_path / "backup"


def test_joint_offload_verifies_backup_then_removes_only_payload(tmp_path):
    state_job, record, plan, out, backup = _fixture(tmp_path)
    result = _local_offload(
        state_job=state_job, record=record, plan=plan, out=out,
        backup_root=backup, execute=True,
    )
    assert result["status"] == "OFFLOADED"
    assert not (out / "final" / "weights.pt").exists()
    assert (out / "final" / "MANIFEST.json").is_file()
    assert (out / "final" / "OFFLOAD_RECEIPT.json").is_file()
    expected = _expected(
        state_job=state_job, record=record, plan=plan,
        manifest=json.loads((out / "final" / "MANIFEST.json").read_text()),
        manifest_sha256=digest(out / "final" / "MANIFEST.json"),
        report_sha256=digest(out / "REPORT.json"),
    )
    ok, reason = validate_offload_receipt(
        out / "final" / "OFFLOAD_RECEIPT.json", expected=expected, verify_local=False,
    )
    assert ok, reason
    local_receipt = backup / next(path.name for path in backup.iterdir()) / "OFFLOAD_RECEIPT.json"
    ok, reason = validate_offload_receipt(local_receipt, expected=expected, verify_local=True)
    assert ok, reason


def test_teacher_stage_cannot_be_offloaded(tmp_path):
    state_job, record, plan, out, backup = _fixture(tmp_path)
    state_job["stage"] = record["stage"] = "teacher_warmup"
    with pytest.raises(ValueError, match="teacher-stage"):
        _local_offload(
            state_job=state_job, record=record, plan=plan, out=out,
            backup_root=backup, execute=True,
        )
    assert (out / "final" / "weights.pt").is_file()


def test_tampered_receipt_never_validates_missing_payload(tmp_path):
    state_job, record, plan, out, backup = _fixture(tmp_path)
    _local_offload(
        state_job=state_job, record=record, plan=plan, out=out,
        backup_root=backup, execute=True,
    )
    receipt_path = out / "final" / "OFFLOAD_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["local_backup_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    expected = _expected(
        state_job=state_job, record=record, plan=plan,
        manifest=json.loads((out / "final" / "MANIFEST.json").read_text()),
        manifest_sha256=digest(out / "final" / "MANIFEST.json"),
        report_sha256=digest(out / "REPORT.json"),
    )
    ok, _reason = validate_offload_receipt(receipt_path, expected=expected, verify_local=False)
    assert not ok


def test_tampered_local_backup_never_validates(tmp_path):
    state_job, record, plan, out, backup = _fixture(tmp_path)
    _local_offload(
        state_job=state_job, record=record, plan=plan, out=out,
        backup_root=backup, execute=True,
    )
    expected = _expected(
        state_job=state_job, record=record, plan=plan,
        manifest=json.loads((out / "final" / "MANIFEST.json").read_text()),
        manifest_sha256=digest(out / "final" / "MANIFEST.json"),
        report_sha256=digest(out / "REPORT.json"),
    )
    target = next(path for path in backup.iterdir() if path.is_dir())
    payload = target / "final" / "weights.pt"
    payload.write_bytes(b"tampered")
    ok, _reason = validate_offload_receipt(
        target / "OFFLOAD_RECEIPT.json", expected=expected, verify_local=True,
    )
    assert not ok


def test_remote_offload_rejects_output_outside_campaign_before_ssh(tmp_path):
    state_job, _record, _plan, _out, backup = _fixture(tmp_path)
    state_job["out"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="outside"):
        offload_remote_job(
            state={"jobs": [state_job]}, state_path=tmp_path / "state.json",
            job_key=state_job["key"], remote_host="example.test",
            remote_campaign_root=str(tmp_path / "campaign"),
            remote_state_path=str(tmp_path / "campaign" / "STATE.json"),
            backup_root=backup, execute=False,
        )


def test_remote_transaction_rechecks_state_and_removes_joint_payload(tmp_path):
    state_job, record, plan, out, _backup = _fixture(tmp_path)
    state_job["data_sha256"] = plan["train_sha256"]
    state_path = tmp_path / "STATE.json"
    state_path.write_text(json.dumps({"jobs": [state_job]}) + "\n")
    manifest_path = out / "final" / "MANIFEST.json"
    payload_path = out / "final" / "weights.pt"
    manifest = json.loads(manifest_path.read_text())
    expected = _expected(
        state_job=state_job, record=record, plan=plan, manifest=manifest,
        manifest_sha256=digest(manifest_path), report_sha256=digest(out / "REPORT.json"),
    )
    export_meta = _student_export_meta(out / "student", parent_manifest_sha256=expected["manifest_sha256"])
    expected.update(
        payload_bytes=payload_path.stat().st_size,
        local_backup_path=str(tmp_path / "backup" / "weights.pt"),
        local_backup_sha256=digest(payload_path),
        local_backup_bytes=payload_path.stat().st_size,
        remote_payload_sha256=digest(payload_path),
        remote_payload_bytes=payload_path.stat().st_size,
        remote_payload_removed=True,
        student_export=export_meta,
        created_at="now", remote_verified_at="now",
    )
    result = subprocess.run(
        [sys.executable, "-c", _REMOTE_TRANSACTION, str(out), str(tmp_path),
         str(state_path), state_job["key"], str(out / "student")],
        input=json.dumps(expected).encode(), capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["payload_sha256"] == expected["payload_sha256"]
    assert not payload_path.exists()
    assert (out / "final" / "OFFLOAD_RECEIPT.json").is_file()
