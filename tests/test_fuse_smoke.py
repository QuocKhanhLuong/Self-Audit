"""Focused end-to-end CPU coverage for the FUSE synthetic pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_smoke_runs_complete_pipeline_and_cold_load(tmp_path):
    out = tmp_path / "fuse-smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_fuse.py"), "--out", str(out)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads((out / "SOFTWARE_ONLY.json").read_text())
    assert receipt["status"] == "SOFTWARE_ONLY"
    assert receipt["seed"] == 42
    assert receipt["gpu_evidence"] == "NOT_RUN"
    assert receipt["teacher_has_no_grads_after_J"] is True
    assert receipt["export_clean"] is True
    assert receipt["teacher_runtime_invariant"] is True
    assert set(receipt["exports"]) == {"teacher", "student", "fusion"}
    assert receipt["deployment"]["mode"] in {"student", "fusion"}
    assert receipt["deployment"]["path"] == receipt["exports"][receipt["deployment"]["mode"]]["path"]
    assert set(receipt["cold_load"]) == {"teacher", "student", "fusion"}
    assert all(value["bitwise_equal"] is True for value in receipt["cold_load"].values())
    assert {value["mode"] for value in receipt["cold_load"].values()} == {"teacher", "student", "fusion"}
    shapes = [tuple(shape) for shape in receipt["native_shapes"]]
    assert len(shapes) == 3
    assert len(set(shapes)) == 3
    assert all(h % 2 == 1 and w % 2 == 1 for _, h, w in shapes)
    output_json = proc.stdout[proc.stdout.find("{\n"):]
    assert json.loads(output_json)["status"] == "SOFTWARE_ONLY"
