"""Durable A-3D campaign scheduling and fail-closed output checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.candidate_a_campaign import (
    _archive_failed_output,
    _campaign_lock,
    build_schedule,
    run_campaign,
)
from scripts.candidate_a_fewshot import create_fewshot_plan
from self_audit.candidate_a3d.data import synthetic_fixture


def _plan(tmp_path: Path, dataset: str = "acdc") -> Path:
    fixture = synthetic_fixture(tmp_path / dataset, n_labeled=10, n_unlabeled=2,
                                n_evaluation=1, shape_tzyx=(1, 2, 4, 5))
    train = Path(fixture["train"])
    payload = json.loads(train.read_text())
    payload["dataset"] = dataset
    train.write_text(json.dumps(payload, indent=2) + "\n")
    return create_fewshot_plan(
        dataset=dataset,
        train_manifest=train,
        development_manifest=fixture["evaluation"],
        model_config=None,
        out=tmp_path / f"{dataset}-campaign",
        synthetic=True,
    )


def test_schedule_puts_all_full_stages_before_other_arms(tmp_path):
    plan = _plan(tmp_path)
    schedule = build_schedule([plan])
    assert schedule["total_jobs"] == 92
    full = [job for job in schedule["jobs"] if job["phase"] == "full"]
    remaining = [job for job in schedule["jobs"] if job["phase"] == "remaining"]
    assert len(full) == 12
    assert len(remaining) == 80
    assert all(job["phase"] == "full" for job in schedule["jobs"][:12])
    assert {job["arm"] for job in full} == {"pretrain", "warmup", "full"}
    assert all(job["arm"] != "full" for job in remaining)
    assert [job["k"] for job in full[:3]] == [2, 2, 2]
    assert [job["stage"] for job in full[:3]] == ["teacher_pretrain", "teacher_warmup", "joint"]


def test_two_dataset_schedule_has_184_jobs_and_two_full_blocks(tmp_path):
    acdc = _plan(tmp_path, "acdc")
    mnms = _plan(tmp_path, "mnms")
    schedule = build_schedule([acdc, mnms])
    assert schedule["total_jobs"] == 184
    assert sum(job["phase"] == "full" for job in schedule["jobs"]) == 24
    assert sum(job["phase"] == "remaining" for job in schedule["jobs"]) == 160
    assert all(job["phase"] == "full" for job in schedule["jobs"][:24])
    assert {job["dataset"] for job in schedule["jobs"][:24]} == {"acdc", "mnms"}


def test_existing_partial_output_is_failed_without_overwrite(tmp_path):
    plan = _plan(tmp_path)
    schedule = build_schedule([plan])
    for scheduled in schedule["jobs"]:
        Path(scheduled["out"]).mkdir(parents=True)
    state = tmp_path / "state.json"
    result = run_campaign([plan], state_path=state, execute=True, device="cpu")
    assert result["status"] == "FAILED"
    saved = json.loads(state.read_text())
    assert saved["status"] == "FAILED"
    assert saved["jobs"][0]["status"] == "FAILED"
    assert all(job["status"] == "FAILED" for job in saved["jobs"])


def test_restart_failed_archives_attempt_and_uses_new_output(tmp_path):
    old = tmp_path / "job"
    old.mkdir()
    state_job = {
        "status": "FAILED", "out": str(old), "planned_out": str(old),
        "attempt": 0, "error": "synthetic failure", "log": "old.log",
        "finished_at": "now",
    }
    _archive_failed_output(state_job)
    assert not old.exists()
    retry = Path(state_job["out"])
    assert retry.name.startswith("job.retry-")
    assert state_job["status"] == "PENDING"
    assert state_job["history"][0]["archive"].startswith(str(old) + ".failed-")


def test_low_storage_blocks_without_marking_pending_job_failed(tmp_path):
    plan = _plan(tmp_path)
    state = tmp_path / "state.json"
    result = run_campaign([plan], state_path=state, execute=True, device="cpu",
                          min_free_gb=10**9)
    assert result["status"] == "BLOCKED_STORAGE"
    saved = json.loads(state.read_text())
    assert saved["jobs"][0]["status"] == "PENDING"
    assert saved["blocked_storage"]["job"] == saved["jobs"][0]["key"]


def test_shared_gpu_lock_rejects_second_state_owner(tmp_path):
    lock = tmp_path / "gpu.lock"
    def attempt():
        with _campaign_lock(lock):
            pass
    with _campaign_lock(lock):  # noqa: SIM117
        with pytest.raises(RuntimeError, match="GPU lock"):
            attempt()


def test_plan_only_initializes_durable_pending_state(tmp_path):
    plan = _plan(tmp_path)
    state = tmp_path / "state.json"
    result = run_campaign([plan], state_path=state, execute=False)
    assert result["status"] == "PLAN_VALIDATED_NOT_RUN"
    assert result["total_jobs"] == 92
    assert not state.exists()


def test_tampered_plan_is_rejected_before_state_creation(tmp_path):
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text())
    child = Path(payload["child_plans"][0]["plan"])
    child_payload = json.loads(child.read_text())
    child_payload["training_seeds"] = [43]
    child.write_text(json.dumps(child_payload, indent=2) + "\n")
    with pytest.raises(ValueError, match="child plan changed"):
        build_schedule([plan])
    assert not (tmp_path / "state.json").exists()
