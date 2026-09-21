"""Test configuration for PICIE cardiac benchmark tests."""
from __future__ import annotations

import sys
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BASELINE_ROOT.parents[1]
for name in list(sys.modules):
    if name == "cardiac_benchmark" or name.startswith("cardiac_benchmark."):
        sys.modules.pop(name)
for path in (REPO_ROOT / "src", BASELINE_ROOT / "src", BASELINE_ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
for path in (BASELINE_ROOT / "src", BASELINE_ROOT, REPO_ROOT / "src"):
    sys.path.insert(0, str(path))
