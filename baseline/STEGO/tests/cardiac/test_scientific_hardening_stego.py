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

import types

import torch

from cardiac_benchmark.config import STEGOConfig
from cardiac_benchmark.stego_runner import load_stego_model


def test_stego_load_model_uses_explicit_device_without_implicit_cuda(monkeypatch, tmp_path):
    class FakeDino(torch.nn.Module):
        def __init__(self, dim, cfg):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.dim = dim
            self.cfg = cfg

    fake_modules = types.ModuleType("modules")
    fake_modules.DinoFeaturizer = FakeDino
    monkeypatch.setitem(sys.modules, "modules", fake_modules)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {}}, checkpoint)
    model = load_stego_model(STEGOConfig(checkpoint_path=str(checkpoint), scientific_run=True), device="cpu")
    assert next(model.parameters()).device.type == "cpu"
