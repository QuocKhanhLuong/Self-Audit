"""Presentation terminology must not migrate persisted Q3D contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECIPE_SHA256 = {
    "acdc_a_k10_seed42_v1.json": "17780f8037d3e3276bb14fe4ef8e42d40e478abe8b8d46a7c03b54124f012942",
    "acdc_a_k5_seed42_v1.json": "48978954a11f824a48cd1e007f4fc7bdf17cdf0c9a20a593e9078eafda37c08b",
    "acdc_a_k8_seed42_c0.json": "eacb5a23c9f265c1391d27c02b58c1a6cf46341b501aab895690a8ac93e22b1b",
    "acdc_a_k8_seed42_c1.json": "caae30e898f426171891a80dfc1406b08d79ae6d425b8ba9edfc42609f25c798",
    "acdc_a_k8_seed42_teacher_supervised.json": "74440125026bac851737715773c8d43cc81ae21273a7ce81a5d843548a897643",
    "acdc_a_k8_seed42_v1.json": "727d778acaab0f675148f495de9c643fc6514af74f0223f90de213ac874f6302",
    "acdc_a_k8_seed42_v1u.json": "32972464239f06060bfca8e72cb3029d30cc1093e91ad23e698cd7f5c5409db4",
}


@pytest.mark.parametrize("record", [
    pytest.param({
        "event": "development_selection", "stage": "T", "update": 400,
        "teacher_dev_per_class": {"RV": 0.71, "MYO": 0.82, "LV": 0.90},
        "score": 0.81,
    }, id="legacy-dev-selection"),
    pytest.param({
        # New VAL presentation keeps these development JSON field names.
        "event": "development_epoch", "stage": "T", "update": 3216,
        "epoch": 201, "per_class": {"RV": 0.71, "MYO": 0.82, "LV": 0.90},
        "score": 0.81,
    }, id="current-val-epoch"),
])
def test_log_view_displays_val_for_legacy_and_current_records(tmp_path, record):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(record) + "\n")
    plan = tmp_path / "PLAN.json"
    plan.write_text(json.dumps({"stage_jobs": [
        {"output": str(tmp_path), "setting": "v1", "k": 8},
    ]}))
    before = {p: p.read_bytes() for p in (metrics, plan)}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/q3d_watch_metrics.py"),
         "--plan", str(plan), "--once"],
        check=True, capture_output=True, text=True,
    )
    assert "VAL patient-macro Dice (%)" in result.stdout
    assert "DEV" not in result.stdout and "development" not in result.stdout
    assert "RV  71.00%" in result.stdout
    assert "MYO  82.00%" in result.stdout
    assert "LV  90.00%" in result.stdout
    assert "val FG Dice  81.00%" in result.stdout
    assert f"update={record['update']:>5}" in result.stdout
    assert f"epoch={record.get('epoch', '-'):>4}" in result.stdout
    assert {p: p.read_bytes() for p in before} == before


def test_display_amendment_keeps_every_plan_v2_recipe_byte_identical():
    # Captured before the terminology edit at main@3b8e3ab42d8cf5d729e7924ba2a91bb77a39013c.
    recipes = ROOT / "reports/q3d-resenc-plan-v2-20261112/recipes"
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in recipes.glob("*.json")}
    assert actual == RECIPE_SHA256
