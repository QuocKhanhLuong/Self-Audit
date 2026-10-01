from __future__ import annotations

import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BASELINE_ROOT.parents[1]
for name in list(sys.modules):
    if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
        sys.modules.pop(name)
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT / "src"):
    if str(import_path) in sys.path:
        sys.path.remove(str(import_path))
for import_path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(import_path))

import numpy as np
import pytest
import torch

from cardiac_benchmark.config import STEGOConfig
from cardiac_benchmark.stego_runner import run_inference
from scripts import train_stego_sa224_fair


def test_stego_cpu_inference_with_mock_model_outputs_shared_grid_partition():
    class MockModel(torch.nn.Module):
        def forward(self, image):
            batch = image.shape[0]
            code = torch.zeros((batch, 4, 2, 2), dtype=torch.float32)
            code[:, 2] = 1.0
            return torch.zeros((batch, 1, 2, 2)), code

    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    partition = run_inference(MockModel(), image, target_hw=(8, 8), device="cpu")
    assert partition.shape == (8, 8)
    assert partition.dtype == np.int32
    assert set(np.unique(partition)) == {2}


def test_stego_fair_inference_uses_trained_cluster_probe_path():
    class MockNet(torch.nn.Module):
        def forward(self, image):
            batch = image.shape[0]
            code = torch.zeros((batch, 4, 2, 2), dtype=torch.float32)
            code[:, 0] = 1.0
            return torch.zeros((batch, 1, 2, 2)), code

    class MockClusterProbe(torch.nn.Module):
        def forward(self, code, alpha, log_probs=False):
            logits = torch.full((code.shape[0], 4, code.shape[2], code.shape[3]), -10.0, dtype=torch.float32)
            logits[:, 1] = 10.0
            if log_probs:
                return logits
            return torch.tensor(0.0), torch.softmax(logits, dim=1)

    class MockFairModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = MockNet()
            self.cluster_probe = MockClusterProbe()

    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    partition = run_inference(
        MockFairModel(),
        image,
        target_hw=(8, 8),
        config=STEGOConfig(profile="STEGO-SA224-FAIR"),
        device="cpu",
    )
    assert partition.shape == (8, 8)
    assert partition.dtype == np.int32
    assert set(np.unique(partition)) == {1}


def test_stego_fair_prior_rejects_downstream_checkpoint_without_override(tmp_path):
    checkpoint = tmp_path / "downstream.ckpt"
    torch.save(
        {
            "state_dict": {
                "net.model.block.weight": torch.zeros(1),
                "net.cluster1.0.weight": torch.zeros(1),
                "net.cluster2.0.weight": torch.zeros(1),
            }
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="--allow-stego-downstream-prior"):
        train_stego_sa224_fair.resolve_fair_prior_policy(
            checkpoint,
            allow_stego_downstream_prior=False,
        )


def test_stego_fair_prior_accepts_dino_teacher_checkpoint(tmp_path):
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"teacher": {"module.backbone.block.weight": torch.zeros(1)}}, checkpoint)
    prior = train_stego_sa224_fair.resolve_fair_prior_policy(
        checkpoint,
        allow_stego_downstream_prior=False,
    )
    assert prior["checkpoint_format"] == "dino_teacher"
    assert prior["fair_status"] == "matched"
    assert prior["allowed_for_fair_mode"] is True
