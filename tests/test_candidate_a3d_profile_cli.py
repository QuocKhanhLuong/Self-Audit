"""CPU-only contracts for the A-3D preflight profiler."""

import json
import torch
import pytest

import scripts.profile_candidate_a3d as profile_module
from scripts.profile_candidate_a3d import main
from self_audit.candidate_a3d.data import synthetic_fixture
from self_audit.model import digest


def test_profile_records_b1_and_rejects_larger_batches(tmp_path):
    fixture = synthetic_fixture(tmp_path / "data", n_labeled=2, n_unlabeled=0,
                                n_evaluation=1, shape_tzyx=(1, 2, 4, 5))
    out = tmp_path / "profile"
    main([
        "--config", "configs/candidate_a3d/smoke.json",
        "--data", str(fixture["train"]), "--data-sha256", digest(fixture["train"]),
        "--out", str(out), "--device", "cpu", "--synthetic", "--chunk-sizes", "512",
    ])
    result = json.loads((out / "PROFILE.json").read_text())
    assert result["status"] == "PASS_PREFLIGHT"
    assert result["protocol"]["batch_size"] == 1
    assert [trial["batch_size"] for trial in result["batch_trials"]] == [1, 2, 4, 8, 16, 32, 64]
    assert all(trial["status"] == "UNSUPPORTED" for trial in result["batch_trials"][1:])
    assert result["teacher_phase"]["status"] == "PASS"
    assert result["student_phases"][0]["status"] == "PASS"
    assert result["sample"]["voxel_count"] == 40


def test_failed_teacher_preflight_is_receipted_and_nonzero(tmp_path, monkeypatch):
    fixture = synthetic_fixture(tmp_path / "data", n_labeled=2, n_unlabeled=0,
                                n_evaluation=1, shape_tzyx=(1, 2, 4, 5))
    out = tmp_path / "failed-profile"

    def failed_teacher(*args, **kwargs):
        del args, kwargs
        return ({"name": "teacher_forward_backward_optimizer", "status": "OOM",
                 "error": "synthetic OOM"}, {"quality": "synthetic"},
                torch.nn.Identity(), object())

    monkeypatch.setattr(profile_module, "_teacher_trial", failed_teacher)
    with pytest.raises(SystemExit, match="2"):
        main([
            "--config", "configs/candidate_a3d/smoke.json",
            "--data", str(fixture["train"]), "--data-sha256", digest(fixture["train"]),
            "--out", str(out), "--device", "cpu", "--synthetic", "--chunk-sizes", "512",
        ])
    result = json.loads((out / "PROFILE.json").read_text())
    assert result["status"] == "FAILED_PREFLIGHT"
    assert result["teacher_phase"]["status"] == "OOM"
    assert result["teacher_snapshot_phase"]["status"] == "NOT_RUN"
    assert result["student_phases"] == []
    assert result["batch_trials"][0]["status"] == "NOT_RUN"
