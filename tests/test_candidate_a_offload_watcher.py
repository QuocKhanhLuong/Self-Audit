"""Tests for the opt-in, fail-closed offload watcher."""

from __future__ import annotations

import json

import pytest

import scripts.candidate_a_offload_watcher as watcher


def test_eligible_jobs_only_done_joint_and_sorted():
    state = {"jobs": [
        {"key": "b", "status": "DONE", "stage": "joint"},
        {"key": "teacher", "status": "DONE", "stage": "teacher_warmup"},
        {"key": "running", "status": "RUNNING", "stage": "joint"},
        {"key": "a", "status": "DONE", "stage": "joint"},
    ]}
    assert [job["key"] for job in watcher._eligible_jobs(state)] == ["a", "b"]


def test_execute_requires_exact_enable_token(tmp_path):
    enable = tmp_path / "ENABLE"
    assert not watcher._enabled(enable)
    enable.write_text("wrong")
    assert not watcher._enabled(enable)
    enable.write_text(watcher.ENABLE_TOKEN + "\n")
    assert watcher._enabled(enable)


def test_poll_once_dry_run_skips_teacher_and_passes_execute_false(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"jobs": [
        {"key": "joint", "status": "DONE", "stage": "joint"},
        {"key": "teacher", "status": "DONE", "stage": "teacher_pretrain"},
    ]}))
    seen = []

    def fake_offload_remote_job(**kwargs):
        seen.append(kwargs)
        return {"status": "DRY_RUN", "job_key": kwargs["job_key"]}

    monkeypatch.setattr(watcher, "offload_remote_job", fake_offload_remote_job)
    result = watcher.poll_once(
        state_path=state_path, remote_state_path="/remote/STATE.json",
        remote_host="vast-gpu", remote_campaign_root="/remote/campaign",
        backup_root=tmp_path / "backup", enable_file=tmp_path / "ENABLE",
        execute=False, python_bin="python",
    )
    assert result["enabled"] is False
    assert result["jobs"][0]["status"] == "DRY_RUN"
    assert [call["job_key"] for call in seen] == ["joint"]
    assert seen[0]["execute"] is False


def test_execute_without_enable_file_is_rejected(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"jobs": []}))
    with pytest.raises(PermissionError, match="enable file"):
        watcher.poll_once(
            state_path=state_path, remote_state_path="/remote/STATE.json",
            remote_host="vast-gpu", remote_campaign_root="/remote/campaign",
            backup_root=tmp_path / "backup", enable_file=tmp_path / "ENABLE",
            execute=True, python_bin="python",
        )
