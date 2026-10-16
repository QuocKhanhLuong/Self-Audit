"""Manufactured, CPU-only tests for the opt-in Astra training runner.

These unit tests replace the bank reader, constructor and checkpoint writer.
The separate end-to-end test exercises the actual reader and checkpoint code.  They
never exercise a random fallback in production code and do not touch the
historical trainer or baseline artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from training_v1 import runner
from training_v1.config import DEFAULT_RECIPE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TinyApprovedStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 8, 1)
        self.feature_project = nn.Conv2d(8, 64, 1)
        self.coarse_head = nn.Conv2d(8, 4, 1)
        self.detail_head = nn.Conv2d(8, 4, 1)
        self.integration_identity = {
            "recipe": {"architecture": DEFAULT_RECIPE["architecture"]},
            "backbone": {
                "model": "dinov3_convnext_tiny",
                "verified_weights_sha256": "__WEIGHTS__",
            },
        }

    def forward(self, images):
        hidden = self.encoder(images)
        features = self.feature_project(
            torch.nn.functional.adaptive_avg_pool2d(hidden, (1, 1))
        )
        coarse = self.coarse_head(torch.nn.functional.avg_pool2d(hidden, 4))
        detail = self.detail_head(hidden)
        return {
            "features": features,
            "coarse_logits": coarse,
            "bounded_detail": detail.tanh(),
            "logits": detail,
            "logits_padded": detail,
        }


class FakeBank:
    def __init__(self, root: Path, *, active: bool):
        self.path = root / "bank"
        self.path.mkdir()
        self.sha256 = "b" * 64
        self.patients = ("patient_a",)
        self.allowed_classes = (True, True, True, True)
        self.manifest = {
            "role": "guidance_bank",
            "split": "train",
            "patients": ["patient_a"],
            "locked_splits": {
                "train": ["patient_a"],
                "val": ["patient_v"],
                "test": ["patient_t"],
            },
            "image_records": {"patient_a": {"shape_xyzt": [16, 16, 1, 1]}},
            "teacher": {
                "frozen": True,
                "checkpoint_sha256": "c" * 64,
                "source": "test@commit",
            },
        }
        self.active = active

    def load_patient(self, patient_id):
        assert patient_id == "patient_a"
        return {"patient_id": patient_id}

    def batch(self, patient, indices, device):
        return {
            "images": torch.ones(len(indices), 3, 16, 16, device=device),
            "semantic_active": self.active,
            "semantic_active_mask": torch.full(
                (len(indices), 4, 4), self.active, dtype=torch.bool, device=device
            ),
        }

    def prepare_targets(self, batch):
        n = batch["images"].shape[0]
        device = batch["images"].device
        return {
            "feature_target": torch.ones(n, 64, 1, 1, device=device),
            "feature_valid": torch.ones(n, 1, 1, dtype=torch.bool, device=device),
            "coarse_target": torch.full((n, 4, 4, 4), 0.25, device=device),
            "coarse_weights": torch.ones(n, 4, 4, device=device),
            "semantic_active": self.active,
        }

    def reverify(self):
        return self.manifest


@pytest.fixture
def approval(tmp_path):
    source = tmp_path / "official_dinov3"
    source.mkdir()
    (source / "hubconf.py").write_text(
        "# manufactured source used only after constructor patch\n"
    )
    weights = tmp_path / "dinov3.pth"
    weights.write_bytes(b"manufactured approved weight bytes")
    payload = {
        "schema": "layer_guided.backbone_approval.v1",
        "approved": True,
        "model": "dinov3_convnext_tiny",
        "lineage": "official_self_supervised_dinov3",
        "repo": str(source),
        "weights": str(weights),
        "weights_sha256": _sha(weights),
        "source_commit": "a" * 40,
        "review_record": "TEST_FIXTURE_ONLY",
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path, payload


def _config(**overrides):
    result = dict(DEFAULT_RECIPE)
    result.update({"epochs": 2, "batch_size": 1, "max_steps": 1, "threads": 1})
    result.update(overrides)
    return result


def _patch_runtime(monkeypatch, approval_payload):
    def construct(recipe, approval, device):
        model = TinyApprovedStudent().to(device)
        model.integration_identity["recipe"] = dict(recipe)
        model.integration_identity["backbone"]["verified_weights_sha256"] = (
            approval_payload["weights_sha256"]
        )
        return model, {"verified_weights_sha256": approval_payload["weights_sha256"]}

    def checkpoint(model, optimizer, folder, config, lineage, *, training_state):
        folder.mkdir(parents=True)
        torch.save(model.state_dict(), folder / "model.pt")
        (folder / "MANIFEST.json").write_text(
            json.dumps({"lineage": lineage, "state": training_state})
        )
        return "d" * 64

    monkeypatch.setattr(runner, "construct_student", construct)
    monkeypatch.setattr(runner, "save_checkpoint", checkpoint)


def test_runner_applies_new_student_and_writes_partial_pilot(
    tmp_path, monkeypatch, approval
):
    approval_path, payload = approval
    _patch_runtime(monkeypatch, payload)
    bank = FakeBank(tmp_path, active=True)
    monkeypatch.setattr(runner, "FrozenTrainingBank", lambda path, expected: bank)
    out = tmp_path / "run"

    report = runner.run_training(
        bank=bank.path,
        expected_bank_sha256=bank.sha256,
        approval=approval_path,
        expected_approval_sha256=_sha(approval_path),
        config=_config(),
        out=out,
        device="cpu",
    )

    assert report["status"] == "COMPLETED_MAX_STEPS"
    assert report["optimizer_steps"] == 1
    assert report["semantic_updates"] == 1
    assert (
        json.loads((out / "STARTUP.json").read_text())["class"] == "TinyApprovedStudent"
    )
    assert (out / "checkpoints" / "epoch_0001" / "MANIFEST.json").is_file()
    assert (out / "final" / "model.pt").is_file()
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 1
    assert (
        json.loads((out / "REPORT.json").read_text())["native_detail"]
        == "NOT_TRAINED_NATIVE_DETAIL"
    )


def test_inactive_semantics_do_not_decay_coarse_head(tmp_path, monkeypatch, approval):
    approval_path, payload = approval
    _patch_runtime(monkeypatch, payload)
    bank = FakeBank(tmp_path, active=False)
    monkeypatch.setattr(runner, "FrozenTrainingBank", lambda path, expected: bank)
    model_holder = {}

    def construct(recipe, approval, device):
        model = TinyApprovedStudent().to(device)
        model.integration_identity["recipe"] = dict(recipe)
        model.integration_identity["backbone"]["verified_weights_sha256"] = (
            approval_payload["weights_sha256"]
        )
        model_holder["model"] = model
        model_holder["before"] = model.coarse_head.weight.detach().clone()
        return model, {"verified_weights_sha256": approval_payload["weights_sha256"]}

    approval_payload = payload
    monkeypatch.setattr(runner, "construct_student", construct)
    out = tmp_path / "inactive"
    runner.run_training(
        bank=bank.path,
        expected_bank_sha256=bank.sha256,
        approval=approval_path,
        expected_approval_sha256=_sha(approval_path),
        config=_config(),
        out=out,
        device="cpu",
    )
    after = model_holder["model"].coarse_head.weight.detach().clone()
    # The fixture starts with a fresh model and semantic is inactive.  Because
    # the loss graph omits coarse logits, AdamW receives grad=None and cannot
    # apply weight decay to the semantic-only head.
    torch.testing.assert_close(after, model_holder["before"])
    assert json.loads((out / "REPORT.json").read_text())["semantic_updates"] == 0


def test_missing_approval_fails_before_output(tmp_path):
    bank = FakeBank(tmp_path, active=True)
    out = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="approval"):
        runner.run_training(
            bank=bank.path,
            expected_bank_sha256=bank.sha256,
            approval=tmp_path / "does-not-exist.json",
            expected_approval_sha256="a" * 64,
            config=_config(),
            out=out,
            device="cpu",
        )
    assert not out.exists()


def test_output_inside_bank_is_rejected_before_output(tmp_path, approval, monkeypatch):
    approval_path, _ = approval
    bank = FakeBank(tmp_path, active=True)
    monkeypatch.setattr(runner, "FrozenTrainingBank", lambda path, expected: bank)
    out = bank.path / "nested-run"
    with pytest.raises(ValueError, match="inside"):
        runner.run_training(
            bank=bank.path,
            expected_bank_sha256=bank.sha256,
            approval=approval_path,
            expected_approval_sha256=_sha(approval_path),
            config=_config(),
            out=out,
            device="cpu",
        )
    assert not out.exists()


def test_duck_typed_bank_cannot_bypass_provenance(tmp_path):
    bank = FakeBank(tmp_path, active=True)
    with pytest.raises(TypeError, match="on-disk path"):
        runner._validate_bank(bank, bank.sha256)
