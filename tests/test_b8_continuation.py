import json

import pytest

from scripts.continue_b8_transfer_pilot import teacher_gate


def setup(tmp_path, status="TRAINING", quality="GO"):
    (tmp_path / "STATUS.json").write_text(json.dumps({"status": status, "quality_gate": quality,
                                                     "plan_sha256": "pinned"}))
    return tmp_path


def test_waits_without_teacher_complete(tmp_path):
    assert teacher_gate(setup(tmp_path), "pinned") is False


@pytest.mark.parametrize("state", ["BLOCKED", "UPLOAD_PENDING"])
def test_failed_backup_never_starts_compact(tmp_path, state):
    with pytest.raises(RuntimeError, match="stopped"):
        teacher_gate(setup(tmp_path, state), "pinned")


def test_failed_quality_never_starts_compact(tmp_path):
    with pytest.raises(RuntimeError, match="NO_GO"):
        teacher_gate(setup(tmp_path, "TEACHER_COMPLETED_REMOTE_VERIFIED", "NO_GO"), "pinned")


def test_other_teacher_plan_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        teacher_gate(setup(tmp_path), "other")


def test_complete_requires_matching_raw_identity_and_readback(tmp_path):
    setup(tmp_path, "TEACHER_COMPLETED_REMOTE_VERIFIED")
    identity = {"quality_gate": "GO", "inference_state": "student", "remote_files": [{"sha256": "a"}]}
    (tmp_path / "TEACHER_IDENTITY.json").write_text(json.dumps(identity))
    (tmp_path / "REMOTE_WEIGHTS_VERIFIED.json").write_text(json.dumps({"files": [{"sha256": "b"}]}))
    with pytest.raises(ValueError, match="readback"):
        teacher_gate(tmp_path, "pinned")
    (tmp_path / "REMOTE_WEIGHTS_VERIFIED.json").write_text(json.dumps({"files": identity["remote_files"]}))
    assert teacher_gate(tmp_path, "pinned") is True
