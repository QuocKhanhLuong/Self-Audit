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

from cardiac_benchmark import stego_runner


def test_stego_producer_path_has_no_metric_or_semantic_mapping_imports():
    source = inspect.getsource(stego_runner).lower()
    for token in ("dice", "iou", "hd95", "assd", "hungarian", "medical_metrics", "label", "mask"):
        assert token not in source
