import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_context_diagnostic as diag
from self_audit import diagnostic_protocol as shared_protocol


def _fixed_source_snapshot():
    return {"git_head": "synthetic-fixture", "files": {"fixture.py": "f" * 64}}


@pytest.fixture
def frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_protocol, "source_snapshot", _fixed_source_snapshot)
    fixture_dir = tmp_path / "fixture_inputs"
    config_path, config_sha = diag._write_synthetic_fixture(fixture_dir)
    round_dir = tmp_path / "round"
    report = diag.preflight_stage(config_path, config_sha, round_dir)
    assert report["status"] == "READY_FOR_INFERENCE"
    lock = diag.lock_stage(config_path, config_sha, round_dir)
    freeze = diag.predict_stage(config_path, config_sha, round_dir)
    config = diag._read_config(config_path, config_sha)
    val_manifest = diag._load_json(Path(config["val_manifest"]["path"]))
    first_patient = val_manifest["patients"][0]
    gt_path = Path(config["val_manifest"]["path"]).parent / first_patient["masks"]["0"]["path"]
    return config_path, config_sha, config, round_dir, gt_path, lock, freeze


def _freeze_b(round_dir):
    lock_a = shared_protocol.read_lock(round_dir, "A")
    shared_protocol.create_lock(round_dir, "B", {"fixture": True}, lock_a["assets"])
    prediction = Path(round_dir) / "fixture-B-prediction.bin"
    prediction.write_bytes(b"synthetic B prediction")
    return shared_protocol.freeze_predictions(round_dir, "B", [prediction], {"fixture": True})


def test_pair_rejects_reproduced_teacher_and_ema_state():
    teacher = {"payload_sha256": diag.TEACHER_PAYLOAD}
    student = {"lineage": {
        "data_sha256": diag.TRAIN_MANIFEST, "label_budget": 8, "lids": ["p1"],
        "teacher_identity": {
            "payload_sha256": diag.TEACHER_PAYLOAD, "manifest_sha256": diag.TEACHER_MANIFEST,
            "inference_state": "student", "train_manifest_sha256": diag.TRAIN_MANIFEST,
            "labeled_ids": ["p1"],
        },
    }}
    diag.validate_pair(teacher, student)
    with pytest.raises(ValueError, match="not distilled"):
        diag.validate_pair({"payload_sha256": "f" * 64}, student)
    other = json.loads(json.dumps(student))
    other["lineage"]["teacher_identity"]["inference_state"] = "ema"
    with pytest.raises(ValueError, match="raw student"):
        diag.validate_pair(teacher, other)


def test_image_only_predict_does_not_open_fixture_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_protocol, "source_snapshot", _fixed_source_snapshot)
    fixture_dir = tmp_path / "fixture_inputs"
    config_path, config_sha = diag._write_synthetic_fixture(fixture_dir)
    round_dir = tmp_path / "round"
    diag.preflight_stage(config_path, config_sha, round_dir)
    diag.lock_stage(config_path, config_sha, round_dir)
    config = diag._read_config(config_path, config_sha)
    manifest_path = Path(config["val_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    gt_paths = {(manifest_path.parent / mask["path"]).resolve()
                for patient in manifest["patients"] for mask in patient["masks"].values()}
    assert len(gt_paths) == 2 and all(path.is_file() for path in gt_paths)
    original_open = Path.open

    def no_gt_open(path, *args, **kwargs):
        assert path.resolve() not in gt_paths, "prediction opened GT bytes"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_gt_open)
    frozen = diag.predict_stage(config_path, config_sha, round_dir)
    assert frozen["stage"] == "A"
    assert frozen["metadata"]["ground_truth_opened_before_freeze"] is False


def test_shared_fixture_serializes_and_strictly_loads_actual_models(tmp_path):
    import torch
    from self_audit.diagnostic_fixture import build_diagnostic_fixture

    fixture = build_diagnostic_fixture(tmp_path / "fixture")
    fixture_again = build_diagnostic_fixture(tmp_path / "fixture_again")
    teacher_spec = fixture["teacher_spec"]
    student_spec = fixture["student_spec"]
    teacher, teacher_meta = diag._worker._load_runtime_model(teacher_spec, "teacher", "cpu")
    student, student_meta = diag._worker._load_runtime_model(student_spec, "student", "cpu")
    assert teacher.__class__.__name__ == "NativeStudent"
    assert teacher.encoder.__class__.__name__ == "TinyEncoder"
    assert student.__class__.__name__ == "CompactStudent"
    assert teacher_meta["synthetic"] is student_meta["synthetic"] is True
    assert len(fixture["train_patient_ids"]) == 2
    assert len(fixture["development_patient_ids"]) == 2
    manifest_path = Path(fixture["val_manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    image = np.load(manifest_path.parent / manifest["patients"][0]["image"])
    assert image.shape == (1, 3, 32, 32)  # edge + interior context, not repeated one-slice input
    assert not np.array_equal(image[0, 0], image[0, 1])
    assert Path(fixture["train_manifest_path"]).read_bytes() == Path(fixture_again["train_manifest_path"]).read_bytes()
    assert Path(fixture["val_manifest_path"]).read_bytes() == Path(fixture_again["val_manifest_path"]).read_bytes()
    for role, spec in (("teacher", teacher_spec), ("student", student_spec)):
        repeated_spec = fixture_again["model_specs"][role]
        payload = torch.load(Path(spec["path"]) / "weights.pt", map_location="cpu", weights_only=True)
        repeated = torch.load(Path(repeated_spec["path"]) / "weights.pt", map_location="cpu", weights_only=True)
        assert set(payload) == {"student"}
        assert payload["student"]
        assert all(isinstance(tensor, torch.Tensor) for tensor in payload["student"].values())
        assert payload["student"].keys() == repeated["student"].keys()
        assert all(torch.equal(payload["student"][name], repeated["student"][name])
                   for name in payload["student"])


def test_missing_val_manifest_preflight_writes_structured_missing_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_protocol, "source_snapshot", _fixed_source_snapshot)
    fixture_dir = tmp_path / "fixture_inputs"
    config_path, _ = diag._write_synthetic_fixture(fixture_dir)
    config = json.loads(config_path.read_text())
    missing_manifest = fixture_dir / "not_created" / "development.json"
    config["val_manifest"] = {"path": str(missing_manifest), "sha256": "0" * 64}
    missing_config = fixture_dir / "missing_val_config.json"
    diag.write_json(missing_config, config)
    config_sha = diag._sha256(missing_config)
    original_load_json = diag._load_json

    def guard_missing_manifest(path):
        assert Path(path).resolve() != missing_manifest.resolve(), "missing VAL manifest was opened"
        return original_load_json(path)

    monkeypatch.setattr(diag, "_load_json", guard_missing_manifest)
    round_dir = tmp_path / "missing-val-round"
    report = diag.preflight_stage(missing_config, config_sha, round_dir)
    assert report["status"] == "NOT_RUN_MISSING_ASSETS"
    assert {item["asset"] for item in report["missing"]} >= {"development manifest"}
    assert any(Path(item["path"]).resolve() == missing_manifest.resolve() for item in report["missing"])
    record = json.loads((round_dir / "PREFLIGHT_A.json").read_text())
    assert record["report"]["missing"] == report["missing"]


def test_missing_b_blocks_score_before_any_gt_read(frozen_fixture, monkeypatch):
    _, config_sha, config, round_dir, gt_path, _, _ = frozen_fixture
    original_sha = diag._sha256

    def no_gt_hash(path):
        assert Path(path) != gt_path, "GT hash was checked before shared-round authorization"
        return original_sha(path)

    monkeypatch.setattr(diag, "_sha256", no_gt_hash)
    with pytest.raises(FileNotFoundError):
        diag.score_frozen(config, round_dir, config_sha)
    assert not (round_dir / "VAL_EXPOSURE.json").exists()
    assert not (round_dir / "A/RESULT.json").exists()


def test_score_prevalidates_prediction_payload_shapes_before_authorizing_gt(frozen_fixture, monkeypatch):
    _, config_sha, config, round_dir, gt_path, _, _ = frozen_fixture
    receipt = json.loads((round_dir / "A/teacher/PREDICTIONS_FROZEN.json").read_text())
    bad_path = Path(receipt["records"][0]["path"])
    original_load = np.load

    def wrong_payload(path, *args, **kwargs):
        if Path(path) == bad_path:
            return np.zeros((2, 2, 2), dtype=np.float32)
        assert Path(path) != gt_path
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", wrong_payload)
    with pytest.raises(ValueError, match="shape differs from native image geometry"):
        diag.score_frozen(config, round_dir, config_sha,
                          b_blocked_reason="B BLOCKED: no B prediction in this test")
    assert not (round_dir / "VAL_EXPOSURE.json").exists()


def test_explicit_b_blocker_is_recorded_before_gt_open_and_score_is_fixture_only(frozen_fixture, monkeypatch):
    _, config_sha, config, round_dir, gt_path, _, freeze = frozen_fixture
    original_sha = diag._sha256
    original_load = np.load

    def assert_exposed(path):
        if Path(path) == gt_path:
            assert (round_dir / "VAL_EXPOSURE.json").is_file()
        return original_sha(path)

    def assert_exposed_load(path, *args, **kwargs):
        if Path(path) == gt_path:
            assert (round_dir / "VAL_EXPOSURE.json").is_file()
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(diag, "_sha256", assert_exposed)
    monkeypatch.setattr(np, "load", assert_exposed_load)
    with pytest.raises(ValueError, match="B BLOCKED:"):
        diag.score_frozen(config, round_dir, config_sha, b_blocked_reason="not ready")
    result = diag.score_frozen(
        config, round_dir, config_sha,
        b_blocked_reason="B BLOCKED: synthetic fixture intentionally omits gradient diagnostic",
    )
    assert result["status"] == "FIXTURE_ONLY"
    assert result["synthetic_fixture"] is True
    assert result["A_freeze_sha256"] == freeze["record_sha256"]
    assert result["summary"]["patient_count"] == 2
    assert result["B_blocked_reason"].startswith("B BLOCKED:")
    assert (round_dir / "A/patient_class_common_failure.csv").is_file()
    exposure = json.loads((round_dir / "VAL_EXPOSURE.json").read_text())
    assert exposure["both_protocols_locked_before_first_GT"] is False
    assert exposure["B_blocked_reason"].startswith("B BLOCKED:")
    with pytest.raises(ValueError, match="append-once"):
        diag.score_frozen(config, round_dir, config_sha,
                         b_blocked_reason="B BLOCKED: synthetic fixture intentionally omits gradient diagnostic")


def test_both_locks_and_prediction_freezes_are_required_for_joint_score(frozen_fixture):
    _, config_sha, config, round_dir, _, _, _ = frozen_fixture
    _freeze_b(round_dir)
    result = diag.score_frozen(config, round_dir, config_sha)
    assert result["status"] == "FIXTURE_ONLY"
    exposure = json.loads((round_dir / "VAL_EXPOSURE.json").read_text())
    assert exposure["both_protocols_locked_before_first_GT"] is True
    assert set(exposure["prediction_freezes"]) == {"A", "B"}


def test_incomplete_or_modified_prediction_fails_before_gt(frozen_fixture, monkeypatch):
    _, config_sha, config, round_dir, gt_path, _, _ = frozen_fixture
    receipt_path = round_dir / "A/student/PREDICTIONS_FROZEN.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["records"].pop()
    receipt_path.write_text(json.dumps(receipt))
    original_sha = diag._sha256

    def no_gt(path):
        assert Path(path) != gt_path
        return original_sha(path)

    monkeypatch.setattr(diag, "_sha256", no_gt)
    with pytest.raises(ValueError, match="incomplete"):
        diag.score_frozen(config, round_dir, config_sha,
                          b_blocked_reason="B BLOCKED: no B prediction in this test")
    assert not (round_dir / "VAL_EXPOSURE.json").exists()


def test_protocol_tampering_and_config_hash_mismatch_fail_closed(frozen_fixture, monkeypatch):
    config_path, config_sha, config, round_dir, gt_path, _, _ = frozen_fixture
    lock_path = round_dir / "protocol-A.lock.json"
    lock = json.loads(lock_path.read_text())
    lock["protocol"]["config_sha256"] = "a" * 64
    lock_path.write_text(json.dumps(lock))
    original_sha = diag._sha256

    def no_gt(path):
        assert Path(path) != gt_path
        return original_sha(path)

    monkeypatch.setattr(diag, "_sha256", no_gt)
    with pytest.raises(ValueError, match="checksum mismatch"):
        diag.score_frozen(config, round_dir, config_sha,
                          b_blocked_reason="B BLOCKED: no B prediction in this test")
    assert not (round_dir / "VAL_EXPOSURE.json").exists()
    with pytest.raises(ValueError, match="config hash mismatch"):
        diag.preflight_stage(config_path, "0" * 64, round_dir.parent / "other-round")


def test_cli_fixture_smoke_is_runnable_and_explicitly_synthetic(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(shared_protocol, "source_snapshot", _fixed_source_snapshot)
    round_dir = tmp_path / "cli-smoke"
    code = diag.main(["fixture-smoke", "--round-dir", str(round_dir)])
    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["status"] == "FIXTURE_ONLY"
    assert output["patient_count"] == 2
    assert (round_dir / "VAL_EXPOSURE.json").is_file()
    assert (round_dir / "A/patient_class_common_failure.csv").is_file()
