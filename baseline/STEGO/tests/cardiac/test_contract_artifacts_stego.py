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

from scripts import run_stego_scientific


def test_stego_runner_exposes_shared_artifact_cli():
    parser = run_stego_scientific.build_parser()
    options = {action.dest for action in parser._actions}
    required = {
        "manifest", "image_root", "output_root", "checkpoint", "split", "device",
        "limit", "sample_list", "config_hash", "apply_adapter", "adapter_spec",
        "semantic_root", "retry_failed",
    }
    assert required <= options
