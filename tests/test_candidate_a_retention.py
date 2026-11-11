from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import candidate_a_retention as retention
from self_audit.model import digest


def _student_eval(root: Path, *, payload: bytes = b"prediction") -> tuple[Path, Path]:
    evaluation = root / "validation" / "epoch_0001"
    prediction_dir = evaluation / "predictions"
    prediction_dir.mkdir(parents=True)
    prediction = prediction_dir / "patient001.npy"
    prediction.write_bytes(payload)
    frozen = {
        "epoch": 1,
        "data_sha256": "d" * 64,
        "files": [{"id": "patient001", "file": prediction.name, "sha256": digest(prediction)}],
        "codes": [{"patient": "patient001", "frame": 0, "bytes": 8}],
    }
    (evaluation / "PREDICTIONS_FROZEN.json").write_text(json.dumps(frozen))
    metrics = {
        "epoch": 1,
        "scope": "development",
        "per_class": {},
        "code": {"packets": 1},
        "profile": {"frames": [], "total_ms": 1.0},
        "labels_used_for": "development scoring only; no target, gate, checkpoint or early-stop decision",
    }
    (evaluation / "EPOCH_METRICS.json").write_text(json.dumps(metrics))
    return evaluation, prediction


def _teacher_eval(root: Path) -> tuple[Path, list[Path]]:
    evaluation = root / "validation" / "epoch_0001"
    entries = {}
    files = []
    for branch in ("region", "boundary"):
        directory = evaluation / f"predictions_{branch}"
        directory.mkdir(parents=True)
        prediction = directory / "patient001.npy"
        prediction.write_bytes(branch.encode())
        files.append(prediction)
        entries[branch] = [{"id": "patient001", "file": prediction.name, "sha256": digest(prediction)}]
    evaluation.mkdir(exist_ok=True)
    (evaluation / "PREDICTIONS_FROZEN.json").write_text(json.dumps({"epoch": 1, "branches": entries}))
    (evaluation / "TEACHER_EPOCH_METRICS.json").write_text(json.dumps({
        "epoch": 1, "scope": "development", "branches": {}, "profile": {"total_ms": 1.0},
        "labels_used_for": "development scoring only; no target, gate, checkpoint or early-stop decision",
    }))
    return evaluation, files


def test_student_inventory_is_verified_and_only_prediction_is_removed(tmp_path: Path):
    evaluation, prediction = _student_eval(tmp_path)
    finding = retention._inspect_eval(evaluation, {"key": "job", "out": tmp_path})
    assert finding["score_marker"] == "EPOCH_METRICS.json"
    assert prediction.exists()

    # Exercise the public retention path without inventing a campaign plan.
    receipt = tmp_path / "receipt.json"
    plan = tmp_path / "FEWSHOT_PLAN.json"
    plan.write_text("{}")
    original_discover = retention.discover
    try:
        retention.discover = lambda paths, state_path=None: [finding]
        result = retention.retain([plan], execute=True, receipt=receipt)
    finally:
        retention.discover = original_discover
    assert result["status"] == "APPLIED"
    assert result["removed_count"] == 1
    assert not prediction.exists()
    assert (evaluation / "PREDICTIONS_FROZEN.json").exists()
    assert (evaluation / "EPOCH_METRICS.json").exists()
    assert receipt.exists()


def test_teacher_branches_are_supported(tmp_path: Path):
    evaluation, files = _teacher_eval(tmp_path)
    finding = retention._inspect_eval(evaluation, {"key": "teacher", "out": tmp_path})
    assert finding["score_marker"] == "TEACHER_EPOCH_METRICS.json"
    assert {Path(item["path"]) for item in finding["files"]} == set(files)


def test_invalid_hash_and_nonfinite_marker_fail_closed(tmp_path: Path):
    evaluation, prediction = _student_eval(tmp_path)
    frozen_path = evaluation / "PREDICTIONS_FROZEN.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["files"][0]["sha256"] = "0" * 64
    frozen_path.write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="hash mismatch"):
        retention._inspect_eval(evaluation, {"key": "job", "out": tmp_path})
    assert prediction.exists()

    _student_eval(tmp_path / "second")
    metrics_path = tmp_path / "second" / "validation" / "epoch_0001" / "EPOCH_METRICS.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["profile"]["total_ms"] = float("nan")
    metrics_path.write_text(json.dumps(metrics, allow_nan=True))
    with pytest.raises(ValueError, match="non-finite"):
        retention._inspect_eval(metrics_path.parent, {"key": "job", "out": tmp_path / "second"})


def test_running_campaign_job_is_allowlisted_but_pending_job_is_not(tmp_path: Path, monkeypatch):
    child = tmp_path / "child.json"
    child.write_text(json.dumps({
        "schema": "self_audit.candidate_a3d.plan.v1",
        "jobs": [{"id": "s42_full", "out": str(tmp_path / "run_full")},
                 {"id": "s42_pending", "out": str(tmp_path / "run_pending")}],
    }))
    root = tmp_path / "FEWSHOT_PLAN.json"
    root.write_text("{}")
    root_payload = {
        "dataset": "acdc",
        "child_plans": [{"plan": str(child), "plan_sha256": digest(child), "k": 2}],
    }
    monkeypatch.setattr(retention, "_verify_root_plan", lambda path: root_payload)
    state = {
        "schema": "self_audit.candidate_a3d.campaign_state.v1",
        "jobs": [
            {"key": f"{child.resolve()}::s42_full", "status": "RUNNING"},
            {"key": f"{child.resolve()}::s42_pending", "status": "PENDING"},
        ],
    }
    records = retention._job_records([root], state)
    assert [record["arm"] for record in records] == ["s42_full"]

    state["jobs"][0]["out"] = str(tmp_path / "run_full.retry-20261111")
    records = retention._job_records([root], state)
    assert records[0]["out"].name == "run_full.retry-20261111"
    state["jobs"][0]["out"] = str(tmp_path / "unrelated-output")
    with pytest.raises(ValueError, match="outside its planned retry path"):
        retention._job_records([root], state)
