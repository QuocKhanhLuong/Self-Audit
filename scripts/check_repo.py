#!/usr/bin/env python3
"""Check the single active Candidate A runtime and concrete recipe contracts."""
from pathlib import Path
import json
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from self_audit.candidate_a.config import validate_config
active = json.loads((root / "ACTIVE_PIPELINE.json").read_text())
assert active["method"] == "candidate_a" and active["entrypoint"] == "train.py"
for relative in ("self_audit/engine.py", "self_audit/control.py", "self_audit/decoder_control.py",
                 "self_audit/evaluation.py", "configs/pilot.json", "configs/full.json"):
    if (root / relative).exists():
        raise SystemExit(f"RETIRED_RUNTIME_PRESENT: {relative}")
for relative in ("self_audit/candidate_a/engine.py", "self_audit/candidate_a/teacher.py",
                 "self_audit/metrics.py", "prepare_acdc.py", "prepare_mnms.py", "docs/METHOD.md"):
    if not (root / relative).is_file():
        raise SystemExit(f"MISSING_CANONICAL_FILE: {relative}")
recipes = list((root / "configs/candidate_a").glob("*.json"))
for recipe in recipes:
    validate_config(json.loads(recipe.read_text()))
print(json.dumps({"status": "PASS", "method": active["method"], "recipes": len(recipes),
                  "supervision": active["supervision"]}))
