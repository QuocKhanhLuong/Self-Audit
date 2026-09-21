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

import torch

from cardiac_benchmark.config import PICIEConfig
from cardiac_benchmark.picie_runner import build_classifier, normalize_state_dict_keys


def test_picie_checkpoint_key_normalization_strips_parallel_prefixes():
    value = torch.ones(1)
    assert normalize_state_dict_keys({"module.model.weight": value}) == {"weight": value}


def test_picie_classifier_is_cpu_constructible():
    classifier = build_classifier(PICIEConfig())
    assert next(classifier.parameters()).device.type == "cpu"
