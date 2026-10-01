# SPDX-License-Identifier: GPL-3.0
"""Parity against pinned official functions, not a substitute scientific profile."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
sys.path[:0] = [str(BASE / "src"), str(ROOT / "src")]
from sgscn.model import CSNet
from sgscn.losses import loss_terms
from sgscn.producer import predict_native
from shared_benchmark.native_protocol import load_lock, ProtocolBlocked
from shared_benchmark.native_artifacts import NativeContractError

torch.set_num_threads(1)
SETTINGS = load_lock(BASE / "config/native/ph2_official_reference.yaml")["scientific"]


def original_functions():
    module = types.ModuleType("official_functions")
    module.args = types.SimpleNamespace(nChannel=100, nConv=2)
    module.nn, module.F, module.torch, module.np = nn, F, torch, np
    spec = importlib.util.spec_from_file_location("official_center", BASE / "upstream/src/center.py")
    center = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(center)
    module.center = center
    tree = ast.parse((BASE / "upstream/demo_final.py").read_text())
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))], type_ignores=[])
    exec(compile(selected, "official_functions", "exec"), module.__dict__)
    return module


def test_architecture_loss_and_gradient_parity():
    source = original_functions()
    torch.manual_seed(123)
    original = source.CSNet(3)
    adapted = CSNet(3, n_channels=100, n_conv=2)
    adapted.load_state_dict(original.state_dict())
    image = torch.rand(1, 3, 12, 10)
    a = original(image)[0]
    b = adapted(image)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    with patch.object(torch.Tensor, "cuda", lambda self: self):
        locations = source.center.get_centers(a)
        total = 0
        for channel in range(100):
            plane = a[channel] + .001
            pdf = plane / plane.sum()
            vx, vy = source.get_variance(pdf, *locations[channel])
            total = total + vx + vy
    flat = a.permute(1, 2, 0).contiguous().view(-1, 100)
    target = flat.argmax(1)
    grid = flat.reshape(12, 10, 100)
    spatial = F.l1_loss(grid[1:] - grid[:-1], torch.zeros_like(grid[1:])) + F.l1_loss(
        grid[:, 1:] - grid[:, :-1], torch.zeros_like(grid[:, 1:]))
    expected = F.cross_entropy(flat, target) + 5 * spatial + total
    actual, adapted_target, _ = loss_terms(b, SETTINGS)
    assert torch.equal(target, adapted_target)
    torch.testing.assert_close(expected, actual, rtol=1e-6, atol=1e-4)
    expected.backward()
    actual.backward()
    for p, q in zip(original.parameters(), adapted.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-3)


def test_native_full_reference_settings_and_same_seed():
    image = np.random.default_rng(42).integers(0, 256, (16, 20, 3), dtype=np.uint8)
    x, a = predict_native(image, SETTINGS, seed=1)
    y, b = predict_native(image, SETTINGS, seed=1)
    assert x.dtype == np.int32 and x.shape == image.shape[:2]
    assert np.array_equal(x, y)
    assert 1 <= a["iterations"] <= 50
    assert a["final_forward_mode"] == "train"
    assert a["stop_reason"] in {"max_iterations", "min_labels"}
    assert [r["loss"] for r in a["trajectory"]] == [r["loss"] for r in b["trajectory"]]


def test_update_then_stop_and_final_train_forward(monkeypatch):
    import sgscn.producer as producer
    calls = []
    class FakeNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
        def forward(self, x):
            calls.append(("forward", float(self.weight.detach()), self.training))
            result = self.weight.expand(1, 100, 4, 4).clone()
            result[:, 0] = result[:, 0] + 1
            return result
    def fake_loss(output, settings):
        calls.append(("loss", float(output[0, 0, 0].detach())))
        return output[0].mean(), torch.zeros(16, dtype=torch.long), {"ce": output[0].mean()}
    monkeypatch.setattr(producer, "CSNet", FakeNet)
    monkeypatch.setattr(producer, "loss_terms", fake_loss)
    partition, receipt = producer.predict_native(np.zeros((4, 4, 3), np.uint8), SETTINGS, seed=1)
    assert receipt["iterations"] == 1 and receipt["stop_reason"] == "min_labels"
    forwards = [c for c in calls if c[0] == "forward"]
    assert len(forwards) == 2 and forwards[1][1] < forwards[0][1]
    assert all(c[2] for c in forwards)  # final pass remains train mode


def test_nonfinite_loss_is_not_repaired(monkeypatch):
    import sgscn.producer as producer
    monkeypatch.setattr(producer, "loss_terms", lambda value, settings:
                        (value.sum() * float("nan"), value.flatten().long(), {}))
    with pytest.raises(FloatingPointError, match="no repair"):
        predict_native(np.zeros((4, 4, 3), np.uint8), SETTINGS, seed=1)


def test_paper_evaluation_tie_remains_blocked():
    spec = importlib.util.spec_from_file_location("native_metric", BASE / "evaluation/track_b/metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.max_overlap_dice(np.array([[2, 2], [7, 7]]), np.array([[True, True], [False, False]]))
    assert result["selected_raw_id"] == 2 and result["dice"] == 1
    with pytest.raises(ProtocolBlocked):
        module.max_overlap_dice(np.array([[2, 7]]), np.ones((1, 2), bool))
