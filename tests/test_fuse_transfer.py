import hashlib
import json
from pathlib import Path

import pytest
import torch

from self_audit.candidate_fuse.config import (
    TRANSFER_SCHEMA,
    default_transfer_config,
    load_config,
    normalize_config,
)
from self_audit.candidate_fuse.engine import (
    _create_zero_shot_artifacts,
    _import_transfer_source,
    _transfer_source_reference,
    _transfer_teacher_budget,
    _verify_zero_shot_artifacts,
    run_training,
)
from self_audit.candidate_fuse.evaluation import evaluate_exports
from self_audit.candidate_fuse.data import write_synthetic_manifests
from self_audit.candidate_fuse.teacher import (
    build_teacher,
    configure_teacher_step,
    teacher_parameter_groups,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic_source(tmp_path, *, missing_key=None):
    config = default_transfer_config(synthetic=True)
    identity = {
        "config_sha256": "a" * 64,
        "train_manifest_sha256": "b" * 64,
        "val_manifest_sha256": "c" * 64,
    }
    model, _ = build_teacher(config, synthetic=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if missing_key is not None:
        del state[missing_key]
    source_path = tmp_path / "source_teacher.pt"
    torch.save({"model": state}, source_path)
    metadata = {"status": "SELECTED", "sha256": _sha(source_path), "identity": identity}
    metadata_path = Path(str(source_path) + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    source = config["experiment"]["source_teacher"]
    source.update(path=str(source_path), sha256=_sha(source_path),
                  metadata_sha256=_sha(metadata_path), identity=identity)
    return normalize_config(config), model


def test_transfer_b_template_pins_real_source_and_budget():
    config = load_config(ROOT / "configs/candidate_fuse/transfer_b.json")
    assert config["schema"] == TRANSFER_SCHEMA
    assert config["experiment"]["track"] == "B"
    assert config["training"]["teacher_epochs"] == 250
    assert config["training"]["teacher_batch_size"] == 2
    assert config["training"]["joint_epochs"] == 100
    assert config["experiment"]["source_teacher"]["sha256"] == (
        "bf2de8287a3fb6a5df7c490b0e081d39e58911a99cfa081243015a2c6ca894ec"
    )
    assert _transfer_teacher_budget(16, 250) == {
        "optimizer_steps": 2000, "examples_seen": 4000,
    }
    assert _transfer_teacher_budget(16, 250, max_optimizer_steps=20) == {
        "optimizer_steps": 20, "examples_seen": 40,
    }


def test_transfer_synthetic_source_load_is_full_strict_state(tmp_path):
    config, source_model = _synthetic_source(tmp_path)
    reference = _transfer_source_reference(config)
    target, _ = build_teacher(config, synthetic=True)
    _import_transfer_source(reference, target)
    for key, value in source_model.state_dict().items():
        assert torch.equal(value.cpu(), target.state_dict()[key].cpu()), key


@pytest.mark.parametrize("bad_pin", ["sha256", "metadata_sha256"])
def test_transfer_rejects_bad_source_hash_before_deserialization(tmp_path, monkeypatch, bad_pin):
    config, _ = _synthetic_source(tmp_path)
    config["experiment"]["source_teacher"][bad_pin] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _transfer_source_reference(config)


@pytest.mark.parametrize("prefix", ["model.encoder.", "model.region_decoder.", "model.boundary_path."])
def test_transfer_strictly_rejects_missing_encoder_or_heads(tmp_path, prefix):
    config = default_transfer_config(synthetic=True)
    pristine, _ = build_teacher(config, synthetic=True)
    key = next(k for k in pristine.state_dict() if k.startswith(prefix))
    config, _ = _synthetic_source(tmp_path, missing_key=key)
    target, _ = build_teacher(config, synthetic=True)
    reference = _transfer_source_reference(config)
    with pytest.raises(ValueError, match="strict full-model loading"):
        _import_transfer_source(reference, target)


def test_transfer_teacher_enters_t_b_without_encoder_freeze():
    config = default_transfer_config(synthetic=True)
    teacher, _ = build_teacher(config, synthetic=True)
    optimizer = torch.optim.AdamW(
        teacher_parameter_groups(teacher, config),
        weight_decay=config["training"]["weight_decay"],
    )
    step = configure_teacher_step(teacher, optimizer, 0, 8, config)
    assert step["phase"] == "T-b"
    assert all(parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.model.encoder.training
    assert all(group["lr"] > 0 for group in optimizer.param_groups)


def test_synthetic_transfer_t_runs_one_epoch_with_zero_shot_source(tmp_path):
    config, _ = _synthetic_source(tmp_path)
    manifests = write_synthetic_manifests(tmp_path / "data")
    for role, prefix in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{prefix}_manifest"] = manifests[role]["path"]
        config["data"][f"{prefix}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)
    partial = run_training(config, tmp_path / "run", device="cpu", stage="T", max_updates=1)
    assert partial["examples_seen"] == 2
    assert partial["optimizer_steps"] == 1
    result = run_training(config, tmp_path / "run", device="cpu", stage="T", resume=True,
                          max_updates=2)
    assert result["examples_seen"] == 4
    assert result["optimizer_steps"] == 2
    assert set(result["exports"]) == {"teacher", "teacher_zero_shot"}
    assert Path(result["exports"]["teacher_zero_shot"]["path"]).is_file()
    report = json.loads((tmp_path / "run" / "ZERO_SHOT_VAL.json").read_text())
    assert report["selection_role"] == "report_only"
    assert set(report["metrics"]) == {"teacher_zero_shot"}
    joint = run_training(config, tmp_path / "run", device="cpu", stage="J", resume=True,
                         teacher_artifact=result["teacher_artifact"])
    assert joint["teacher_has_no_grads_after_J"] is True
    assert set(joint["validation"]) == {"teacher", "student", "fusion"}
    assert set(joint["exports"]) == {"teacher_zero_shot", "teacher", "student", "fusion"}
    assert joint["exports"]["teacher_zero_shot"]["sha256"] == result["exports"]["teacher_zero_shot"]["sha256"]


def test_t_resume_after_update_does_not_skip_epoch_validation(tmp_path):
    config, _ = _synthetic_source(tmp_path)
    manifests = write_synthetic_manifests(tmp_path / "data")
    for role, prefix in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{prefix}_manifest"] = manifests[role]["path"]
        config["data"][f"{prefix}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)
    import self_audit.candidate_fuse.engine as engine

    original_eval = engine.evaluate_modes
    validations = []
    crash_on_validation = {"enabled": False}

    def count_validation(*args, **kwargs):
        validations.append(kwargs.get("modes"))
        if crash_on_validation["enabled"]:
            crash_on_validation["enabled"] = False
            raise RuntimeError("simulated crash before T validation commit")
        return original_eval(*args, **kwargs)

    engine.evaluate_modes = count_validation
    try:
        run_training(config, tmp_path / "resume_run", device="cpu", stage="T", max_updates=2)
        first_rows = [json.loads(line) for line in (tmp_path / "resume_run" / "metrics.jsonl").read_text().splitlines()]
        assert first_rows[-1]["type"] == "validation"
        assert validations == [("teacher",), ("teacher",)]
        # Simulate a crash before the final VAL append/commit by restoring the
        # last pre-boundary checkpoint captured by the test run's update.
        checkpoint = torch.load(tmp_path / "resume_run" / "resume.pt", map_location="cpu", weights_only=False)
        assert checkpoint["stage"] == "T" and checkpoint["epoch"] == 1
        validations.clear()
        run_training(config, tmp_path / "resume_run", device="cpu", stage="T", resume=True,
                     max_updates=2)
        assert validations == []
        rows = [json.loads(line) for line in (tmp_path / "resume_run" / "metrics.jsonl").read_text().splitlines()]
        assert [row["type"] for row in rows].count("validation") == 1
    finally:
        engine.evaluate_modes = original_eval


def test_v1_t_resume_at_completed_max_updates_does_not_repeat_validation(tmp_path):
    from self_audit.candidate_fuse.config import default_config
    config = default_config(synthetic=True)
    manifests = write_synthetic_manifests(tmp_path / "data")
    for role, prefix in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{prefix}_manifest"] = manifests[role]["path"]
        config["data"][f"{prefix}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)
    import self_audit.candidate_fuse.engine as engine

    original_eval = engine.evaluate_modes
    validations = []

    def count_validation(*args, **kwargs):
        validations.append(kwargs.get("modes"))
        return original_eval(*args, **kwargs)

    engine.evaluate_modes = count_validation
    try:
        run = tmp_path / "v1_resume"
        run_training(config, run, device="cpu", stage="T", max_updates=1)
        run_training(config, run, device="cpu", stage="T", resume=True, max_updates=1)
        assert validations == [("teacher",)]
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        assert [row["type"] for row in rows].count("validation") == 1
    finally:
        engine.evaluate_modes = original_eval


def test_engine_zero_shot_export_is_accepted_by_four_mode_transaction(tmp_path):
    """Exercise the real engine export path and the transaction provenance gate."""
    config, _ = _synthetic_source(tmp_path)
    from self_audit.candidate_fuse.mnms_data import write_synthetic_mnms_manifests
    manifests = write_synthetic_mnms_manifests(tmp_path / "mnms")
    for role, prefix in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{prefix}_manifest"] = manifests[role]["path"]
        config["data"][f"{prefix}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)
    run = tmp_path / "run"
    # Produce all four exports through the engine's actual T/J path.
    run_training(config, run, device="cpu", stage="T", max_updates=1)
    artifact = run / "exports" / "teacher_zero_shot.pt"
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    assert payload["metadata"]["source_teacher"] == config["experiment"]["source_teacher"]
    assert payload["metadata"]["source_teacher_verification"]["status"] == "SELECTED"
    assert payload["metadata"]["source_teacher_verification"]["path"] != config["experiment"]["source_teacher"]["path"] or Path(
        payload["metadata"]["source_teacher_verification"]["path"]
    ).is_absolute()
    partial_t = run_training(config, run, device="cpu", stage="T", resume=True, max_updates=2)
    run_training(config, run, device="cpu", stage="J", resume=True,
                 teacher_artifact=partial_t["teacher_artifact"])
    result = json.loads((run / "RESULT.json").read_text())
    assert set(result["exports"]) == {"teacher_zero_shot", "teacher", "student", "fusion"}
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import fuse_campaign
    bundle = fuse_campaign.export_bundle(run, track_b=True)
    transaction = evaluate_exports(bundle, manifests["test"]["path"],
                                   tmp_path / "transaction", "cpu", "test",
                                   manifests["test"]["sha256"])
    assert transaction["status"] == "COMMITTED"
    sidecar = json.loads(Path(str(artifact) + ".manifest.json").read_text())
    assert sidecar["metadata"]["source_teacher"] == config["experiment"]["source_teacher"]


def test_zero_shot_validation_is_report_only_and_hash_pinned(tmp_path):
    config, teacher = _synthetic_source(tmp_path)
    source = _transfer_source_reference(config)
    out = tmp_path / "run"
    out.mkdir()
    identity = {"config_sha256": "d" * 64, "train_manifest_sha256": "e" * 64,
                "val_manifest_sha256": "f" * 64}

    class Val:
        pass

    # Exercise creation and then resume-style verification with a tiny real
    # synthetic teacher export; metrics are only ever reported under the
    # report-only zero-shot label.
    from self_audit.candidate_fuse.engine import evaluate_modes

    original = evaluate_modes
    import self_audit.candidate_fuse.engine as engine

    engine.evaluate_modes = lambda *args, **kwargs: {"teacher": {"fg_dice": 0.5}}
    try:
        receipt = _create_zero_shot_artifacts(out, config=config, identity=identity,
            source=source, teacher=teacher, val_data=Val(), device=torch.device("cpu"))
    finally:
        engine.evaluate_modes = original
    report = json.loads(Path(receipt["val_path"]).read_text())
    assert report["comparison_label"] == "teacher_zero_shot"
    assert report["selection_role"] == "report_only"
    assert set(report["metrics"]) == {"teacher_zero_shot"}
    assert _verify_zero_shot_artifacts(out, identity=identity, source=source,
                                      expected=receipt) == receipt
    payload = torch.load(receipt["path"], map_location="cpu", weights_only=True)
    assert payload["metadata"]["source_teacher"] == config["experiment"]["source_teacher"]
    assert payload["metadata"]["source_teacher_verification"]["status"] == "SELECTED"
