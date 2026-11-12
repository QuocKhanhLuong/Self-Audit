#!/usr/bin/env python3
"""Validate the active Q3D contract and preserve explicit A/A-3D baselines."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    from self_audit.candidate_a.config import validate_config as validate_a
    from self_audit.candidate_a3d.config import validate_config as validate_a3d
    from self_audit.candidate_q3d.config import validate_config as validate_q3d

    active = json.loads((ROOT / "ACTIVE_PIPELINE.json").read_text())
    _require(active.get("entrypoint") == "train.py", "ACTIVE_ENTRYPOINT_MISMATCH")
    _require(active.get("evaluation_entrypoint") == "evaluate.py", "ACTIVE_EVALUATION_ENTRYPOINT_MISMATCH")
    _require(active.get("default_track") == "candidate_q3d", "Q3D_NOT_DEFAULT")
    _require(active.get("method") == "candidate_q3d", "Q3D_NOT_ACTIVE_METHOD")
    tracks = active.get("tracks")
    _require(isinstance(tracks, dict), "ACTIVE_TRACKS_MISSING")
    _require(set(tracks) == {"candidate_a", "candidate_a3d", "candidate_q3d"},
             "ACTIVE_TRACK_INVENTORY_MISMATCH")
    for name in ("candidate_a", "candidate_a3d"):
        _require(tracks[name].get("status") == "explicit_baseline_unchanged",
                 f"BASELINE_STATUS_MISMATCH: {name}")
    q3d = tracks["candidate_q3d"]
    _require(q3d.get("package") == "self_audit.candidate_q3d", "Q3D_PACKAGE_MISMATCH")
    _require(q3d.get("config_schema") == "self_audit.candidate_q3d.v1", "Q3D_CONFIG_SCHEMA_MISMATCH")
    _require(q3d.get("checkpoint_schema") == "self_audit.candidate_q3d.resume.v2",
             "Q3D_CHECKPOINT_SCHEMA_MISMATCH")
    _require(q3d.get("export_schema") == "q3d.student_export.v1",
             "Q3D_EXPORT_SCHEMA_MISMATCH")

    for relative in (
        "self_audit/engine.py", "self_audit/control.py", "self_audit/decoder_control.py",
        "self_audit/evaluation.py", "configs/pilot.json", "configs/full.json",
    ):
        _require(not (ROOT / relative).exists(), f"RETIRED_RUNTIME_PRESENT: {relative}")

    for relative in (
        "self_audit/candidate_a/engine.py", "self_audit/candidate_a/teacher.py",
        "self_audit/candidate_a3d/engine.py", "self_audit/candidate_a3d/teacher.py",
        "self_audit/metrics.py", "prepare_acdc.py", "prepare_mnms.py",
        "self_audit/candidate_q3d/__init__.py", "self_audit/candidate_q3d/engine.py",
        "self_audit/candidate_q3d/config.py", "self_audit/candidate_q3d/student.py",
        "self_audit/candidate_q3d/tokenizer.py", "self_audit/candidate_q3d/alignment.py",
        "self_audit/candidate_q3d/teacher_features.py", "self_audit/candidate_q3d/losses.py",
        "self_audit/candidate_q3d/physiology.py", "self_audit/candidate_q3d/data.py",
        "self_audit/candidate_q3d/augmentation.py", "self_audit/candidate_q3d/evaluation.py",
        "self_audit/candidate_q3d/export.py", "scripts/smoke_q3d.py",
    ):
        _require((ROOT / relative).is_file(), f"MISSING_CANONICAL_FILE: {relative}")

    baseline_changes = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "self_audit/candidate_a", "self_audit/candidate_a3d"],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    _require(baseline_changes.returncode == 0, "BASELINE_CHANGE_CHECK_FAILED")
    _require(not baseline_changes.stdout.strip(),
             f"BASELINE_SOURCE_MODIFIED: {baseline_changes.stdout.strip().splitlines()}")

    docs = sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "docs").rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    )
    _require(docs == ["docs/CANDIDATE_Q3D_SPEC.md"], f"Q3D_SOLE_DOC_CONTRACT_MISMATCH: {docs}")
    _require((ROOT / docs[0]).is_file(), "MISSING_AUTHORITATIVE_Q3D_SPEC")

    tests = sorted(
        path.name for path in (ROOT / "tests").glob("test*.py")
        if path.name != "__init__.py"
    )
    contract_tests = {"test_candidate_q3d_weighting.py"}
    old_tests = [name for name in tests
                 if not name.startswith("test_q3d") and name not in contract_tests]
    _require(not old_tests, f"NON_Q3D_TESTS_REMAIN: {old_tests}")

    recipes_a = list((ROOT / "configs/candidate_a").glob("*.json"))
    _require(bool(recipes_a), "MISSING_A_RECIPES")
    for recipe in recipes_a:
        validate_a(json.loads(recipe.read_text()))
    recipes_a3d = list((ROOT / "configs/candidate_a3d").glob("*.json"))
    _require(bool(recipes_a3d), "MISSING_A3D_RECIPES")
    for recipe in recipes_a3d:
        validate_a3d(json.loads(recipe.read_text()))
    recipes_q3d = list((ROOT / "configs/candidate_q3d").glob("*.json"))
    _require(bool(recipes_q3d), "MISSING_Q3D_RECIPES")
    for recipe in recipes_q3d:
        validate_q3d(json.loads(recipe.read_text()))
    arms = {str(json.loads(path.read_text()).get("arm")) for path in recipes_q3d}
    _require({"c0", "c1", "v1", "v1u"}.issubset(arms), "MISSING_Q3D_PRIMARY_ABLATION_RECIPES")

    print(json.dumps({
        "status": "PASS",
        "method": active["method"],
        "docs": docs,
        "tests": tests,
        "recipes": {
            "candidate_a": len(recipes_a),
            "candidate_a3d": len(recipes_a3d),
            "candidate_q3d": len(recipes_q3d),
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
