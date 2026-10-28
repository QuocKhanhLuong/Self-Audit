"""Public runner and offline receipt-chain regression tests, manufactured CPU only."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from self_audit.data import synthetic_fixture
from self_audit.model import digest
from self_audit.automatic_evaluation import evaluate_automatic
from scripts.run_automatic_pilot import collate_evaluations, run_experiment_matrix
from scripts.run_automatic_pilot import compute_resource_accounting

ROOT = Path(__file__).resolve().parents[1]


def test_resume_timing_cannot_claim_complete_cold_cost():
    costs = compute_resource_accounting(
        {"steps": 4, "step_latencies_ms": [10., 11.], "elapsed_seconds": 0.5},
        {"elapsed_seconds": 0.2}, {"total_seconds": 0.1})
    assert costs["cold_total_cost"]["status"] == "INCOMPLETE"


def test_cli_warm_start_only_and_no_overwrite(tmp_path):
    out = tmp_path / "warm"
    cmd = [sys.executable, str(ROOT / "scripts/run_automatic_pilot.py"),
           "--mode", "warm_start", "--synthetic", "--steps", "1", "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["receipt"]["phase"] == "warm_start"
    assert not (out / "joint_dice").exists()
    before = digest(out / "summary.json")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert digest(out / "summary.json") == before


def test_real_receipt_chain_collation_and_tamper_rejection(tmp_path):
    manifest, sha = synthetic_fixture(tmp_path / "train")
    names = ["joint_dice", "scalar_dice", "smaller_lr", "extra_supervised"]
    matrix_dir = tmp_path / "matrix"
    matrix = run_experiment_matrix(names, manifest, sha, matrix_dir,
                                   synthetic=True, device="cpu", steps=1,
                                   pseudo_threshold=0, allow_full_run=True)
    matrix_path = matrix_dir / "summary.json"
    matrix_path.write_text(json.dumps(matrix))
    query = tmp_path / "query"
    query.mkdir()
    image = query / "image.npy"
    np.save(image, np.random.default_rng(91).random((2, 2, 35, 37)).astype("float32"))
    mask = query / "mask.npy"
    labels = np.zeros((2, 35, 37), dtype="uint8")
    labels[:, :8] = 1
    labels[:, 8:16] = 2
    labels[:, 16:24] = 3
    labels[:, -1] = 255
    np.save(mask, labels)
    q_manifest = query / "query.json"
    q_manifest.write_text(json.dumps({
        "schema": "self_audit.data.v2", "role": "external_frozen", "synthetic": True,
        "patients": [{"id": "synthetic_query", "image": image.name, "image_sha256": digest(image),
                      "masks": {"0": {"path": mask.name, "sha256": digest(mask)}}}]}))
    index = {"schema": "self_audit.automatic_evaluation_index.v1",
             "matrix_report": {"path": str(matrix_path), "sha256": digest(matrix_path)},
             "evaluations": {}}
    for arm in names:
        dest = tmp_path / ("eval_" + arm)
        evaluate_automatic(matrix_dir / arm / "final", matrix["arms"][arm]["receipt"]["final_manifest_sha256"],
                           q_manifest, digest(q_manifest), dest, device="cpu", batch_size=2)
        index["evaluations"][arm] = {"directory": str(dest), "metrics_sha256": digest(dest / "METRICS.json")}
    index_path = tmp_path / "evaluations.json"
    index_path.write_text(json.dumps(index))
    report = collate_evaluations(index_path, digest(index_path))
    assert report["decision_gate"]["decision"] == "NO_GO_SOFTWARE_ONLY"
    assert report["verified_context"]["matched_initialization"] is True
    assert report["verified_context"]["matched_lineage"] is True
    assert report["verified_context"]["matched_compute"] is False  # equal-update is not compute matching
    assert all(a["accounting"]["cold_total_cost"]["status"] == "COMPLETE" for a in report["arms"].values())
    bad_index = copy.deepcopy(index)
    bad_index["evaluations"].pop("scalar_dice")
    index_path.write_text(json.dumps(bad_index))
    assert collate_evaluations(index_path, digest(index_path))["decision_gate"]["decision"] == "INCONCLUSIVE_NOT_RUN"
    index_path.write_text(json.dumps(index))
    pred = tmp_path / "eval_joint_dice" / "predictions" / "synthetic_query.npy"
    pred.write_bytes(pred.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        collate_evaluations(index_path, digest(index_path))
