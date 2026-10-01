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

import inspect

from cardiac_benchmark import picie_runner


def test_picie_producer_does_not_import_legacy_metrics_or_hungarian():
    source = inspect.getsource(picie_runner).lower()
    for token in ("medical_metrics", "compute_medical_metrics", "linear_sum_assignment", "hungarian", "dice", "iou", "hd95", "assd", "label", "mask"):
        assert token not in source


def test_picie_producer_outputs_shared_grid_not_native_geometry_contract():
    signature = inspect.signature(picie_runner.run_inference)
    assert "target_hw" in signature.parameters
    source = inspect.getsource(picie_runner.run_inference).lower()
    assert "target_hw" in source
    assert "native" not in source
