#!/usr/bin/env python3
"""Check the canonical layout and prevent stale legacy training code from returning."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION = "experiments/layer_guided_student_v1"
REMOVED = ("baseline", "external", "scratch", "run_configs", "environments", ".omc", ".agenteam",
           "experiments/guidance_signal_20261014", "src/self_audit", "src/self_audit_pseudolabel",
           "src/self_audit_guidance/student.py", "src/self_audit_guidance/full.py",
           "src/self_audit_guidance/pipeline.py")
REQUIRED = ("train.py", "ACTIVE_PIPELINE.json", "configs/pilot.json", "configs/full50.json",
            "src/self_audit_guidance/io.py", f"{IMPLEMENTATION}/layer_upgrade/model.py",
            f"{IMPLEMENTATION}/training_v1/runner.py", f"{IMPLEMENTATION}/training_v1/checkpoint.py",
            f"{IMPLEMENTATION}/integration/artifacts.py", "docs/CURRENT.md", "docs/RUNBOOK.md")
LEGACY_STUBS = ("run_stable_guidance.py", "run_stable_guidance_full.py", "run_stable_guidance_wandb.py",
                "run_full_pipeline_v3.py", "train_self_audit.py", "train_pseudolabel_v3.py")


def check_layout(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    missing = [path for path in REQUIRED if not (root / path).is_file()]
    stale = [path for path in REMOVED if (root / path).exists()]
    if missing or stale:
        raise ValueError(f"Repository layout mismatch: missing={missing}, retired={stale}. Create a fresh worktree; do not run git clean on datasets/runs.")
    active = json.loads((root / "ACTIVE_PIPELINE.json").read_text())
    if active.get("architecture") != "layer_guided_convnext_tiny_v1" or active.get("entrypoint") != "train.py":
        raise ValueError("Active model identity differs from the canonical DINOv3 student")
    source_files = list((root / IMPLEMENTATION).rglob("*.py")) + list((root / "src").rglob("*.py"))
    for path in source_files:
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GuidanceStudent":
                raise ValueError(f"Retired CNN implementation found: {path}")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("self_audit_guidance"):
                if node.module != "self_audit_guidance.io":
                    raise ValueError(f"Unexpected legacy dependency in {path}: {node.module}")
    for name in LEGACY_STUBS:
        path = root / "scripts" / name
        if path.exists() and "LEGACY_ENTRYPOINT_DISABLED" not in path.read_text():
            raise ValueError(f"Retired entrypoint is executable again: {path}")
    return {"status": "PASS_CANONICAL_LAYOUT", "architecture": active["architecture"],
            "implementation": IMPLEMENTATION, "removed_surfaces_checked": len(REMOVED)}


def main():
    try:
        print(json.dumps(check_layout(), sort_keys=True))
        return 0
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"[LAYOUT_FAILED] {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
