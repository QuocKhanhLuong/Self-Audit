"""Bounded CPU evidence for epoch snapshots, isolation and exact continuation."""
from __future__ import annotations

import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from self_audit_maskfree import epoch_validation, runtime
from self_audit_maskfree.config import MaskfreeConfig
from self_audit_maskfree.export import validate_freeze
from self_audit_maskfree.progress import TerminalProgress
from self_audit_maskfree.trainer import MaskfreeTrainer


def _images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    rng = np.random.default_rng(617)
    for patient in range(4):
        folder = root / "training" / f"patient{patient:03d}"
        folder.mkdir(parents=True)
        volume = rng.normal(size=(24, 24, 2)).astype(np.float32)
        volume[6:18, 6:18] += 3
        reference = np.zeros(volume.shape, np.uint8)
        reference[5:19, 5:19] = 2
        reference[8:16, 8:16] = 3
        reference[2:6, 8:14] = 1
        for suffix, values in (("", volume), ("_gt", reference)):
            image = nib.Nifti1Image(values, np.diag([1.5, 1.5, 8., 1.]))
            image.header.set_xyzt_units("mm", "sec")
            nib.save(image, folder / f"patient{patient:03d}_frame01{suffix}.nii.gz")
    return root


def _config(tmp_path: Path, data: Path, run_id: str, **overrides) -> MaskfreeConfig:
    payload = dict(dataset="acdc", data_root=str(data), output_dir=str(tmp_path),
                   total_epochs=3, warmup_epochs=0, batch_size=8, image_size=24,
                   width=4, feature_dim=4, device="cpu", allow_cpu=True, amp=False,
                   wandb_mode="disabled", run_id=run_id, max_epochs=2,
                   epoch_validation=True)
    payload.update(overrides)
    return MaskfreeConfig(**payload)


def _assert_states_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_states_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_states_equal(a, b)
    else:
        assert left == right


def test_native_epoch_dice_isolated_and_exact_resume(tmp_path, monkeypatch):
    """Real child evaluation; all GT opens in the parent are forbidden."""
    torch.set_num_threads(1)
    data = _images(tmp_path)
    native_load = nib.load

    def image_only_load(path, *args, **kwargs):
        assert "_gt" not in str(path), "generating process opened a reference mask"
        return native_load(path, *args, **kwargs)

    monkeypatch.setattr(nib, "load", image_only_load)
    enabled = MaskfreeTrainer(_config(tmp_path, data, "enabled"))
    console = io.StringIO()
    with TerminalProgress(stream=console, mode="compact"):
        report = enabled.run()
    assert report["epoch_validation"]["epoch_counts"]["COMPLETED"] == 2, report
    assert "Val U" in console.getvalue() or "val U" in console.getvalue() or "Dice" in console.getvalue()
    for epoch in (1, 2):
        root = enabled.paths.root / "validation" / f"epoch_{epoch:04d}"
        receipt = json.loads((root / "receipt.json").read_text())
        assert receipt["status"] == "COMPLETED", receipt
        assert receipt["physical_batch"] == 8
        for arm in epoch_validation.ARMS:
            student = receipt["students"][arm]
            assert student["dice"] is not None and student["iou"] is not None
            assert student["patients"] > 0 and student["volumes"] > 0
            assert set(student["dice_per_class"]) == {"RV", "MYO", "LV"}
        frozen = json.loads(Path(receipt["freeze_manifest"]).read_text())
        validate_freeze(frozen, require_complete=True)
        assert set(frozen["completeness"]["required"]) == set(epoch_validation.METHODS)
        assert all(entry["split"] == "dev" for entry in frozen["predictions"] if entry["kind"] == "volume")
        assert all("last.pt" not in row["path"] for row in frozen["checkpoints"])

    cached = epoch_validation.observe_epoch(enabled, 1)
    assert cached["cached"]
    freeze_bytes = Path(cached["freeze_manifest"]).read_bytes()
    reference_report = Path(cached["reference_report"])
    reference_report.rename(reference_report.with_suffix(".lost.json"))
    assert epoch_validation.validation_status(enabled.paths.root, enabled=True, epochs=2)["status"] == "FAILED"
    repaired = epoch_validation.observe_epoch(enabled, 1)
    assert repaired["status"] == "COMPLETED" and not repaired["cached"], repaired
    assert Path(repaired["freeze_manifest"]).read_bytes() == freeze_bytes
    assert Path(repaired["reference_report"]).is_file()

    disabled = MaskfreeTrainer(_config(tmp_path, data, "disabled", epoch_validation=False))
    disabled.run()
    left = torch.load(enabled.paths.last_checkpoint, weights_only=False)
    right = torch.load(disabled.paths.last_checkpoint, weights_only=False)
    for key in ("models", "optimizers", "scaler", "rng", "sampling_generator", "component_steps"):
        _assert_states_equal(left[key], right[key])
    assert all("validation" not in row for row in left["history"])

    # Simulate a process interruption immediately after the training checkpoint,
    # before any snapshot exists. Resume must validate that saved boundary first.
    interrupted = MaskfreeTrainer(_config(tmp_path, data, "interrupted", max_epochs=1))
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("injected after epoch checkpoint")

    monkeypatch.setattr(interrupted, "_observe_epoch", interrupt)
    with pytest.raises(KeyboardInterrupt):
        interrupted.run()
    assert interrupted.paths.last_checkpoint.exists()
    resumed = MaskfreeTrainer(interrupted.config.replace(
        resume=str(interrupted.paths.last_checkpoint), max_epochs=2))
    resumed_report = resumed.run()
    assert resumed_report["epoch_validation"]["epoch_counts"]["COMPLETED"] == 2
    continued = torch.load(resumed.paths.last_checkpoint, weights_only=False)
    for key in ("models", "optimizers", "scaler", "rng", "sampling_generator", "component_steps"):
        _assert_states_equal(left[key], continued[key])


def test_validation_failure_restores_rng_modes_and_never_scores_unfrozen(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    trainer = MaskfreeTrainer(_config(tmp_path, _images(tmp_path), "failed"))
    trainer.setup()
    before_rng = runtime.capture_rng_state()
    before_models = {arm: epoch_validation._state_hash(model.state_dict())
                     for arm, model in trainer.models.items()}
    called = []

    def fail_export(*args, **kwargs):
        torch.rand(4)
        raise RuntimeError("injected export failure")

    from dataclasses import replace
    trainer.components = replace(trainer.components, export_prediction=fail_export)
    monkeypatch.setattr(epoch_validation, "_run_reference", lambda *a: called.append(True))
    result = epoch_validation.observe_epoch(trainer, 0)
    assert result["status"] == "FAILED"
    assert not called
    _assert_states_equal(before_rng, runtime.capture_rng_state())
    assert all(model.training for model in trainer.models.values())
    assert before_models == {arm: epoch_validation._state_hash(model.state_dict())
                             for arm, model in trainer.models.items()}


def test_interrupted_export_reuses_shards_then_freezes_both_arms(tmp_path, monkeypatch):
    from dataclasses import replace

    torch.set_num_threads(1)
    trainer = MaskfreeTrainer(_config(tmp_path, _images(tmp_path), "snapshot-retry"))
    trainer.setup()
    export = trainer.components.export_prediction
    calls = []

    def interrupted_export(*args, **kwargs):
        calls.append(kwargs["prediction_name"])
        if len(calls) == 3:
            raise RuntimeError("interrupted between native shards")
        return export(*args, **kwargs)

    scored = []

    def reference_after_freeze(freeze_path, image_path, output, reference_config):
        freeze = json.loads(freeze_path.read_text())
        validate_freeze(freeze, require_complete=True)
        assert set(freeze["compared_methods"]) == set(epoch_validation.METHODS)
        scored.append(freeze["freeze_id"])
        return {"epoch": 0, "freeze_id": freeze["freeze_id"], "status": "UNAVAILABLE",
                "available": False, "reason": "test does not score references", "students": {}}

    monkeypatch.setattr(epoch_validation, "_run_reference", reference_after_freeze)
    trainer.components = replace(trainer.components, export_prediction=interrupted_export)
    assert epoch_validation.observe_epoch(trainer, 0)["status"] == "FAILED"
    assert not scored
    trainer.components = replace(trainer.components, export_prediction=export)
    recovered = epoch_validation.observe_epoch(trainer, 0)
    assert recovered["status"] == "UNAVAILABLE", recovered
    assert len(scored) == 1
    status = epoch_validation.validation_status(trainer.paths.root, enabled=True, epochs=1)
    assert status["epoch_counts"] == {"COMPLETED": 0, "FAILED": 0, "UNAVAILABLE": 1, "NOT_STARTED": 0}
    epoch_root = trainer.paths.root / "validation" / "epoch_0001"
    assert list(epoch_root.glob("recovered_failure_*.json"))
    freeze = json.loads(Path(recovered["freeze_manifest"]).read_text())
    checkpoint = Path(next(row["path"] for row in freeze["checkpoints"]
                           if row["name"] == "student_audited"))
    checkpoint.write_bytes(b"tampered epoch checkpoint")
    rejected = epoch_validation.observe_epoch(trainer, 0)
    assert rejected["status"] == "FAILED"
    assert len(scored) == 1, "tampered freeze reached reference evaluation"
    assert epoch_validation.validation_status(trainer.paths.root, enabled=True, epochs=1)["status"] == "FAILED"
