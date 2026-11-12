"""Focused receipt aggregation checks for the A-3D campaign reporter."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.candidate_a_campaign_report import build_report, write_report


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job_artifacts(root: Path, name: str, *, dice: float, status: str = "DONE") -> dict:
    out = root / name
    out.mkdir(parents=True)
    (out / "validation" / "epoch_0002").mkdir(parents=True)
    (out / "final").mkdir()
    (out / "final" / "weights.pt").write_bytes(b"weights-" + name.encode())
    payload_hash = _digest(out / "final" / "weights.pt")
    manifest = {
        "schema": "self_audit.candidate_a3d.checkpoint.v1",
        "payload_sha256": payload_hash,
        "source_sha256": "source-hash",
        "synthetic": True,
        "epoch": 2,
    }
    (out / "final" / "MANIFEST.json").write_text(json.dumps(manifest) + "\n")
    metrics = {
        "epoch": 2,
        "synthetic": True,
        "per_class": {
            "BG": {"dice": 0.9, "f1": 0.9, "precision": 0.9, "recall": 0.9},
            "RV": {"dice": dice, "f1": dice, "precision": dice, "recall": dice},
            "MYO": {"dice": dice + 0.05, "f1": dice + 0.05, "precision": dice + 0.05, "recall": dice + 0.05},
            "LV": {"dice": dice + 0.1, "f1": dice + 0.1, "precision": dice + 0.1, "recall": dice + 0.1},
        },
        "macro_patient": {
            "BG": {"dice": 0.9, "f1": 0.9, "precision": 0.9, "recall": 0.9},
            "RV": {"dice": dice, "f1": dice, "precision": dice, "recall": dice},
            "MYO": {"dice": dice + 0.05, "f1": dice + 0.05, "precision": dice + 0.05, "recall": dice + 0.05},
            "LV": {"dice": dice + 0.1, "f1": dice + 0.1, "precision": dice + 0.1, "recall": dice + 0.1},
        },
        "boundary": {"RV": {"surface_dice": 0.7, "assd_mm": 1.2, "hd95_mm": 2.0}},
        "code": {"packets": 2, "total_bytes": 128, "mean_bytes": 64, "query_counts": [2, 3]},
        "profile": {"total_ms": 20.0, "mean_ms": 10.0},
    }
    (out / "validation" / "epoch_0002" / "EPOCH_METRICS.json").write_text(json.dumps(metrics) + "\n")
    report = {"status": "COMPLETED", "epochs": 2, "elapsed_seconds": 3.5}
    (out / "REPORT.json").write_text(json.dumps(report) + "\n")
    return {"status": status, "out": str(out)}


def _state(tmp_path: Path) -> Path:
    train = tmp_path / "train.json"
    dev = tmp_path / "development.json"
    train.write_text("{}\n")
    dev.write_text("{}\n")
    child = tmp_path / "PLAN.json"
    child.write_text(json.dumps({
        "train_sha256": _digest(train), "development_sha256": _digest(dev),
        "training_seeds": [42], "jobs": [],
    }) + "\n")
    full = _job_artifacts(tmp_path, "full", dice=0.6)
    ablation = _job_artifacts(tmp_path, "teacher_free", dice=0.5)
    jobs = []
    for arm, artifacts in (("full", full), ("teacher_free", ablation)):
        jobs.append({
            "key": f"{child}::s42_{arm}", "dataset": "acdc", "k": 2, "arm": arm,
            "stage": "joint", "phase": "full" if arm == "full" else "remaining",
            "status": artifacts["status"], "out": artifacts["out"], "config": "config.json",
            "config_sha256": "config-hash", "error": None,
        })
    jobs.append({
        "key": f"{child}::s42_boundary_only", "dataset": "acdc", "k": 2, "arm": "boundary_only",
        "stage": "joint", "phase": "remaining", "status": "PENDING", "out": str(tmp_path / "never"),
        "config": "config.json", "config_sha256": "config-hash", "error": None,
    })
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"schema": "self_audit.candidate_a3d.campaign_state.v1",
                                 "status": "FAILED", "jobs": jobs}) + "\n")
    return state


def test_report_preserves_missing_job_and_delta(tmp_path):
    state = _state(tmp_path)
    report = build_report(state)
    assert report["total_jobs"] == 3
    rows = {row["arm"]: row for row in report["rows"]}
    assert rows["boundary_only"]["status"] == "PENDING"
    assert rows["boundary_only"]["metrics"]["validation"] == "MISSING"
    assert rows["boundary_only"]["metrics"]["primary"] is None
    delta = rows["teacher_free"]["delta_vs_full"]
    assert round(delta["foreground_macro_dice"], 6) == -0.1
    assert rows["full"]["metrics"]["primary"]["foreground_macro"]["dice"] == 0.65
    assert rows["full"]["metrics"]["boundary"]["RV"]["surface_dice"] == 0.7


def test_write_report_emits_json_csv_markdown_atomically(tmp_path):
    state = _state(tmp_path)
    report = build_report(state)
    paths = write_report(report, tmp_path / "report")
    for path in paths.values():
        assert Path(path).is_file()
    text = Path(paths["csv"]).read_text()
    assert "foreground_macro_dice" in text
    assert "null" not in text.splitlines()[1]  # observed full row has a finite headline
    assert "one seed" in Path(paths["markdown"]).read_text()


def test_cli_help_runs_outside_repository_root(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "candidate_a_campaign_report.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--state STATE" in result.stdout
