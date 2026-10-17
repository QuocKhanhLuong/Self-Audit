#!/usr/bin/env python3
"""Run current software checks into a new receipt directory, without patient data."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-smoke", action="store_true", help="Also run the random full-size CPU architecture smoke")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    if out.exists() or out.is_relative_to(ROOT) or ROOT.is_relative_to(out):
        parser.error("receipt directory must be new and outside this checkout")
    out.mkdir(parents=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    commands = [
        ("layout", [sys.executable, str(ROOT / "scripts/check_repo.py")]),
        ("identity", [sys.executable, str(ROOT / "train.py"), "--describe"]),
        ("tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml="+str(out / "TEST_RESULTS.xml")]),
    ]
    if args.full_smoke:
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "experiments/layer_guided_student_v1"), str(ROOT / "src")))
        commands.append(("random_full_smoke", [sys.executable, str(ROOT / "experiments/layer_guided_student_v1/verify_full_model.py")]))
    receipt = {"schema": "self_audit.current_software_verification.v1", "status": "RUNNING", "commands": [],
               "pretrained_weights_loaded": False, "patient_training": False, "gpu_execution": False,
               "active_pipeline": json.loads((ROOT / "ACTIVE_PIPELINE.json").read_text())}
    for label, command in commands:
        with (out / (label + ".log")).open("w") as log:
            completed = subprocess.run(command, cwd=out if label == "random_full_smoke" else ROOT,
                                       env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt["commands"].append({"name": label, "argv": command, "exit_code": completed.returncode})
        receipt["status"] = "FAILED" if completed.returncode else "RUNNING"
        (out / "VERIFICATION.json").write_text(json.dumps(receipt, indent=2)+"\n")
        if completed.returncode:
            return completed.returncode
    receipt["status"] = "PASS_CPU_SOFTWARE_ONLY"
    receipt["log_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.log"))}
    (out / "VERIFICATION.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
