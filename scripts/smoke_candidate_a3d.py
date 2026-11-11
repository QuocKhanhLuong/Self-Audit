#!/usr/bin/env python3
"""Manufactured CPU smoke run for native full-volume Candidate-A 3-D."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_audit.candidate_a3d.data import CandidateA3DData, synthetic_fixture
from self_audit.candidate_a3d.evaluation import evaluate_epoch
from self_audit.candidate_a3d.io import export_student, load_student
from self_audit.candidate_a3d.k1 import K1Config, student_rollout
from self_audit.data import write_json
from self_audit.model import digest


def run_smoke(out: str | Path, semi_config: str | Path | None = None) -> dict:
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    fixture = synthetic_fixture(
        out / "data", n_labeled=2, n_unlabeled=1, n_evaluation=1,
        # The tiny 3-D trunk downsamples three times; keep at least two
        # voxels on every final axis so GroupNorm remains well-defined.
        shape_tzyx=(2, 8, 16, 16),
    )
    config = json.loads((root / "configs/candidate_a3d/smoke.json").read_text())
    if semi_config is not None:
        semi_recipe = json.loads(Path(semi_config).read_text())
        config["semi_supervised"] = semi_recipe.get("semi_supervised", semi_recipe)
    stages: dict[str, dict] = {}
    previous = None
    for stage in ("teacher_pretrain", "teacher_warmup", "joint"):
        current = copy.deepcopy(config)
        current.update(stage=stage, epochs=1)
        stage_config = out / f"{stage}.json"
        write_json(stage_config, current)
        command = [
            sys.executable, str(root / "train.py"), "--config", str(stage_config),
            "--data", str(fixture["train"]), "--data-sha256", fixture["train_sha256"],
            "--out", str(out / stage), "--device", "cpu", "--synthetic",
            "--validation-data", str(fixture["evaluation"]),
            "--validation-data-sha256", fixture["evaluation_sha256"],
        ]
        if previous is not None:
            command.extend([
                "--teacher-checkpoint", str(previous),
                "--teacher-checkpoint-sha256", digest(previous / "MANIFEST.json"),
            ])
        try:
            result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            # Keep the child traceback in the smoke failure.  A bare
            # CalledProcessError hides the stage that failed and makes a
            # synthetic contract failure needlessly hard to diagnose.
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            raise RuntimeError(
                f"A-3D smoke stage {stage!r} failed with exit code {exc.returncode}\n"
                f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
            ) from exc
        (out / f"{stage}.stdout.txt").write_text(result.stdout)
        stages[stage] = json.loads((out / stage / "REPORT.json").read_text())
        previous = out / stage / "final"

    exported = export_student(previous, digest(previous / "MANIFEST.json"), out / "student")
    model_a, bank_a, codec_a, metadata_a = load_student(
        previous, digest(previous / "MANIFEST.json"), device="cpu")
    model_b, bank_b, codec_b, metadata_b = load_student(
        out / "student", exported["manifest_sha256"], device="cpu")
    data = CandidateA3DData(fixture["train"], fixture["train_sha256"], synthetic=True)
    key = data.labeled_frame_keys()[0]
    x, _, geometry, _ = data.load_labeled_frame(key)
    rollout_config = K1Config(**metadata_a["config"]["rollout"])
    torch.set_num_threads(1)
    with torch.inference_mode():
        first = student_rollout(
            model_a, x, bank_a, codec=codec_a, geometry=geometry,
            config=rollout_config,
        )
        second = student_rollout(
            model_b, x, bank_b, codec=codec_b, geometry=geometry,
            config=K1Config(**metadata_b["config"]["rollout"]),
        )
    if not torch.equal(first.final_logits, second.final_logits) or first.code_ids != second.code_ids:
        raise AssertionError("A-3D student-only export prediction/code differs")
    payload = torch.load(out / "student" / "weights.pt", weights_only=True)
    if set(payload) != {"student"}:
        raise AssertionError("A-3D export contains teacher or optimizer payload")
    evaluation = evaluate_epoch(
        model_b, bank_b, codec_b, rollout_config, fixture["evaluation"],
        fixture["evaluation_sha256"], out / "evaluation", epoch=1,
        device="cpu", train_ids=fixture["train_ids"], synthetic=True,
    )
    receipt = {
        "status": "PASS_SYNTHETIC_ONLY",
        "patient_training": "NOT_RUN", "gpu_validation": "NOT_RUN",
        "mednext_weights": "NOT_AVAILABLE_LOCALLY",
        "stages": stages, "export": exported,
        "semi_supervised": config.get("semi_supervised", {"enabled": False}),
        "export_bitwise_parity": True, "export_payload_keys": sorted(payload),
        "evaluation_metrics": str(out / "evaluation" / "EPOCH_METRICS.json"),
        "evaluation_synthetic": evaluation["synthetic"], "output": str(out),
    }
    write_json(out / "SMOKE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new output folder")
    parser.add_argument("--semi-config", help="optional A-3D image-only U consistency JSON block")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.out, args.semi_config), indent=2))
