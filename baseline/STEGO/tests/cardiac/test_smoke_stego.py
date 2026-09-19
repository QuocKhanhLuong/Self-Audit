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

from cardiac_benchmark.stego_runner import run_inference


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
