"""Small validation-batch contract checks for the mask-free epoch observer.

The tests use tiny deterministic student doubles so they exercise ordering,
tail batches, native export records, snapshot identity and cohort/firewall
guards without running a training epoch or a reference evaluator.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from self_audit_maskfree import epoch_validation, runtime


class _Student(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Three discrete channels with a deterministic class-2 winner.
        return inputs[:, :3] + self.offset


def _trainer(tmp_path: Path, *, n_dev: int = 5, overlap: bool = False):
    calls: list[str] = []
    exported: list[dict[str, object]] = []
    freeze_calls: list[dict[str, object]] = []
    records = []
    for index in range(n_dev):
        records.append(
            {
                "unit_id": f"unit-{index:02d}",
                "volume_id": f"volume-{index:02d}",
                "study_id": f"study-{index:02d}",
                "patient_id": "patient-train" if overlap and index == 0 else f"patient-{index:02d}",
                "split": "dev",
                "path": str(tmp_path / f"source-{index:02d}.npy"),
            }
        )
    if overlap:
        records.insert(
            0,
            {
                "unit_id": "train-unit",
                "volume_id": "train-volume",
                "study_id": "train-study",
                "patient_id": "patient-train",
                "split": "train",
                "path": str(tmp_path / "train.npy"),
            },
        )
    manifest = {
        "schema_version": "maskfree150.manifest.v1",
        "manifest_id": "manifest-performance",
        "dataset": "acdc",
        "resolved_protocol": "spatial_predictive",
        "records": records,
    }

    def load_full_input(current_manifest, unit_id, *, image_size):
        calls.append(unit_id)
        value = float(next(row for row in current_manifest["records"] if row["unit_id"] == unit_id)["unit_id"][-2:])
        image = torch.stack(
            [torch.full((image_size, image_size), value + channel) for channel in range(3)]
        )
        row = next(row for row in current_manifest["records"] if row["unit_id"] == unit_id)
        record = dict(row)
        record.update(
            {
                "dataset": "acdc",
                "protocol": "spatial_predictive",
                "manifest_id": current_manifest["manifest_id"],
                "partition_id": "partition-performance",
                "native_export_available": True,
            }
        )
        return image, record

    def export_prediction(output, **kwargs):
        exported.append(
            {
                "prediction_name": kwargs["prediction_name"],
                "record": dict(kwargs["record"]),
                "labels": kwargs["labels"].detach().clone(),
            }
        )
        return {
            "kind": "volume",
            "prediction_name": kwargs["prediction_name"],
            "unit_id": kwargs["record"]["unit_id"],
        }

    def freeze_predictions(output, entries, checkpoints, **kwargs):
        freeze_calls.append(
            {
                "entries": list(entries),
                "required_methods": list(kwargs["required_methods"]),
                "required_unit_ids": list(kwargs["required_unit_ids"]),
            }
        )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        frozen = {
            "freeze_id": "freeze-performance",
            "required_methods": list(kwargs["required_methods"]),
            "completeness": {"required": list(kwargs["required_methods"])},
            "predictions": list(entries),
        }
        (output / "freeze_manifest.json").write_text(
            __import__("json").dumps(frozen), encoding="utf-8"
        )
        return frozen

    components = SimpleNamespace(
        load_full_input=load_full_input,
        export_prediction=export_prediction,
        freeze_predictions=freeze_predictions,
        validate_freeze=lambda frozen, require_complete: None,
    )
    config = SimpleNamespace(
        dataset="acdc",
        batch_size=2,
        image_size=4,
        total_epochs=2,
        epoch_reference_config=None,
        scientific_identity=lambda: {"test": "performance"},
    )
    trainer = SimpleNamespace(
        paths=SimpleNamespace(root=tmp_path),
        run_id="performance-run",
        manifest=manifest,
        config=config,
        source={"package": {"combined": "package-performance"}},
        models={"student_no_audit": _Student(0.0), "student_audited": _Student(0.5)},
        components=components,
        device=torch.device("cpu"),
        pseudo_label_version=lambda epoch: f"version-{epoch}",
    )
    return trainer, calls, exported, freeze_calls


def test_validation_batches_preserve_sorted_ids_tail_and_native_exports(tmp_path: Path, monkeypatch) -> None:
    trainer, calls, exported, freeze_calls = _trainer(tmp_path, n_dev=5)
    monkeypatch.setattr(
        epoch_validation,
        "_run_reference",
        lambda freeze_path, image_manifest, output, reference_config: {
            "epoch": 0,
            "freeze_id": "freeze-performance",
            "status": "UNAVAILABLE",
            "available": False,
            "reason": "test-only reference bypass",
            "students": {},
        },
    )

    result = epoch_validation.observe_epoch(trainer, 0)

    assert result["status"] == "UNAVAILABLE"
    assert calls == ["unit-00", "unit-01", "unit-02", "unit-03", "unit-04"]
    assert [len(calls[start : start + trainer.config.batch_size]) for start in (0, 2, 4)] == [2, 2, 1]
    assert len(exported) == 2 * len(calls)
    assert all(item["labels"].dtype == torch.long for item in exported)
    assert all(item["record"]["split"] == "dev" for item in exported)
    assert freeze_calls[0]["required_unit_ids"] == calls
    assert set(freeze_calls[0]["required_methods"]) == set(epoch_validation.METHODS)
    assert result["physical_batch"] == 2
    assert result["source_cache"] is not None


def test_validation_rejects_train_dev_patient_overlap_before_loading(tmp_path: Path, monkeypatch) -> None:
    trainer, calls, exported, _ = _trainer(tmp_path, n_dev=3, overlap=True)
    monkeypatch.setattr(
        epoch_validation,
        "_run_reference",
        lambda *args: pytest.fail("reference must stay behind cohort firewall"),
    )

    result = epoch_validation.observe_epoch(trainer, 0)

    assert result["status"] == "FAILED"
    assert "overlap" in result["reason"]
    assert calls == []
    assert exported == []


def test_validation_restores_rng_modes_and_snapshot_identity(tmp_path: Path, monkeypatch) -> None:
    trainer, _, _, _ = _trainer(tmp_path, n_dev=1)
    monkeypatch.setattr(
        epoch_validation,
        "_run_reference",
        lambda *args: {
            "epoch": 0,
            "freeze_id": "freeze-performance",
            "status": "UNAVAILABLE",
            "available": False,
            "reason": "test-only reference bypass",
            "students": {},
        },
    )
    torch.manual_seed(901)
    before_rng = runtime.capture_rng_state()
    before_modes = {name: model.training for name, model in trainer.models.items()}

    first = epoch_validation.observe_epoch(trainer, 0)
    assert first["status"] == "UNAVAILABLE"
    assert runtime.capture_rng_state()["torch"].equal(before_rng["torch"])
    assert {name: model.training for name, model in trainer.models.items()} == before_modes

    with torch.no_grad():
        trainer.models["student_no_audit"].offset.add_(1.0)
    second = epoch_validation.observe_epoch(trainer, 0)
    assert second["status"] == "FAILED"
    assert "snapshot identity" in second["reason"]


def test_validation_cache_receipt_exposes_hit_miss_bytes_evictions(tmp_path: Path, monkeypatch) -> None:
    trainer, _, _, _ = _trainer(tmp_path, n_dev=0)
    monkeypatch.setattr(epoch_validation, "_run_reference", lambda *args: pytest.fail("no dev units"))

    result = epoch_validation.observe_epoch(trainer, 0)

    stats = result["source_cache"]["after"]
    assert {"hits", "misses", "bytes", "evictions"}.issubset(stats)


def test_custom_kwargs_loader_is_not_given_private_digest(tmp_path: Path, monkeypatch) -> None:
    trainer, _, _, _ = _trainer(tmp_path, n_dev=1)
    original = trainer.components.load_full_input
    received: dict[str, object] = {}

    def custom_loader(current_manifest, unit_id, *, image_size, **kwargs):
        received.update(kwargs)
        return original(current_manifest, unit_id, image_size=image_size)

    trainer.components.load_full_input = custom_loader
    monkeypatch.setattr(
        epoch_validation,
        "_run_reference",
        lambda *args: {
            "epoch": 0,
            "freeze_id": "freeze-performance",
            "status": "UNAVAILABLE",
            "available": False,
            "reason": "test-only reference bypass",
            "students": {},
        },
    )

    result = epoch_validation.observe_epoch(trainer, 0)

    assert result["status"] == "UNAVAILABLE"
    assert received == {}
