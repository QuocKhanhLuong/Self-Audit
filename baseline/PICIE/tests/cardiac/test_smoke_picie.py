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
import torch

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.picie_runner import run_inference
from scripts import train_picie_sa224_fair


def test_picie_cpu_inference_with_mock_modules_outputs_shared_grid_partition():
    class MockModel(torch.nn.Module):
        def forward(self, image):
            return torch.ones((image.shape[0], 128, 2, 2), dtype=torch.float32)

    class MockClassifier(torch.nn.Module):
        def forward(self, feats):
            logits = torch.zeros((feats.shape[0], 4, feats.shape[2], feats.shape[3]), dtype=torch.float32)
            logits[:, 3] = 1.0
            return logits

    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    partition = run_inference(MockModel(), MockClassifier(), image, target_hw=(8, 8), config=PICIEConfig(), device="cpu")
    assert partition.shape == (8, 8)
    assert partition.dtype == np.int32
    assert set(np.unique(partition)) == {3}


def test_picie_recipe_audit_exposes_known_fidelity_gaps():
    audit = train_picie_sa224_fair.build_recipe_audit()
    statuses = {item["item"]: item["status"] for item in audit["items"]}
    assert statuses["two_view_generation"] == "matched"
    assert statuses["classifier_update_path"] == "matched"
    assert statuses["epoch_iteration_semantics"] == "intentional_adaptation"
    assert audit["known_fidelity_gaps"] == []
