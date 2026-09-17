from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def _write_mnms_fixture(root: Path) -> None:
    volumes = root / "testing" / "volumes"
    masks = root / "testing" / "masks"
    volumes.mkdir(parents=True)
    masks.mkdir(parents=True)
    volume = np.zeros((16, 16, 2), dtype=np.float32)
    mask = np.zeros((16, 16, 2), dtype=np.uint8)
    mask[2:6, 2:6, 0] = 1
    np.save(volumes / "A100_t00.npy", volume)
    np.save(masks / "A100_t00.npy", mask)


def test_external_evaluator_report_is_fixed_tau_and_independent(tmp_path, monkeypatch) -> None:
    _write_mnms_fixture(tmp_path)
    torch.save({"training_dataset": "acdc", "model": {}}, tmp_path / "checkpoint.pt")
    import scripts.evaluate_external_mnms as evaluator

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.num_classes = 4

    class FakeBinding:
        state_digest = "live-digest"
        model_identity = {"architecture": "fake"}

        def as_dict(self):
            return {
                "checkpoint_path": "checkpoint.pt",
                "checkpoint_sha256": "file-hash",
                "state_digest": self.state_digest,
                "model_identity": self.model_identity,
            }

    monkeypatch.setattr(evaluator, "build_model_from_config", lambda config, device: FakeModel().to(device))
    monkeypatch.setattr(evaluator, "bind_evaluation_checkpoint", lambda *args, **kwargs: FakeBinding())
    monkeypatch.setattr(evaluator, "verify_bound_state", lambda *args, **kwargs: "live-digest")
    monkeypatch.setattr(
        evaluator,
        "evaluate_volume_cohort",
        lambda *args, **kwargs: {
            "metric_space": "volume_resized",
            "native_dice_available": False,
            "volumes_evaluated": 1,
            "volumes_available": 1,
            "patients": {},
            "cohorts": {"unknown": {"case_ids": ["A100_t00"]}},
        },
    )

    payload = evaluator.run_external_evaluation(
        config={
            "external_test": {
                "dataset": "mnms",
                "split": "testing",
                "raw_to_acdc": {0: 0, 1: 3, 2: 2, 3: 1},
            },
            "image_size": 16,
            "num_classes": 4,
            "depth_axis": 2,
            "audit": {"tau_accept": 0.25, "t_max": 1},
            "model": {"num_classes": 4},
        },
        checkpoint=tmp_path / "checkpoint.pt",
        data_root=tmp_path,
        split="testing",
        tau_accept=None,
        device=torch.device("cpu"),
        output=tmp_path / "external_mnms.json",
    )

    assert payload["evidence_class"] == "independent_external_evaluation"
    assert payload["dataset"] == "mnms"
    assert payload["split"] == "testing"
    assert payload["metric_space"] == "volume_resized"
    assert payload["phase_partition"] == "unavailable_without_authoritative_metadata"
    assert payload["tau_accept"] == 0.25
    assert payload["tau_accept_source"] == "config.audit.tau_accept"
    assert payload["comparison_modes"] == ["initial_only", "always_accept_refinement", "self_audit", "oracle_accept"]
    assert (tmp_path / "external_mnms.json").is_file()

