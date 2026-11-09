#!/usr/bin/env python3
"""Manufactured CPU integration check, never a patient experiment."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from self_audit.candidate_a.data import synthetic_candidate_a_fixture
from self_audit.candidate_a.io import export_student, load_student
from self_audit.candidate_a.k1 import K1Config, student_rollout
from self_audit.candidate_a.evaluation import evaluate
from self_audit.data import write_json, TrainingData
from self_audit.model import digest


def run_smoke(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    fixture = synthetic_candidate_a_fixture(out / "data", n_labeled=2, n_unlabeled=1,
        n_evaluation=1, shape_tzyx=(1, 2, 12, 13))
    config = json.loads((root / "configs/candidate_a/smoke.json").read_text())
    stages = {}
    previous = None
    for name in ("teacher_pretrain", "teacher_warmup", "joint"):
        c = copy.deepcopy(config)
        c["stage"] = name
        c["epochs"] = 1
        path = out / (name + ".json")
        write_json(path, c)
        command = [sys.executable, str(root / "train.py"), "--config", str(path),
            "--data", str(fixture["train"]), "--data-sha256", fixture["train_sha256"],
            "--out", str(out / name), "--device", "cpu", "--synthetic",
            "--validation-data", str(fixture["evaluation"]),
            "--validation-data-sha256", fixture["evaluation_sha256"]]
        if previous is not None:
            command += ["--teacher-checkpoint", str(previous), "--teacher-checkpoint-sha256", digest(previous / "MANIFEST.json")]
        result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        (out / (name + ".stdout.txt")).write_text(result.stdout)
        stages[name] = json.loads((out / name / "REPORT.json").read_text())
        previous = out / name / "final"
    exported = export_student(previous, digest(previous / "MANIFEST.json"), out / "student")
    model1, bank1, codec1, meta1 = load_student(previous, digest(previous / "MANIFEST.json"))
    model2, bank2, codec2, meta2 = load_student(out / "student", exported["manifest_sha256"])
    train_manifest = json.loads(fixture["train"].read_text())
    image = np.load(fixture["train"].parent / train_manifest["labeled"][0]["image"], allow_pickle=False)
    x = TrainingData.context(image, 0, 0)
    geometry = {"shape": tuple(x.shape[-2:]), "spacing_yx": (0.7, 1.3)}
    torch.set_num_threads(1)
    with torch.inference_mode():
        r1 = student_rollout(model1, x, bank1, codec=codec1, geometry=geometry, config=K1Config(**meta1["config"]["rollout"]))
        r2 = student_rollout(model2, x, bank2, codec=codec2, geometry=geometry, config=K1Config(**meta2["config"]["rollout"]))
    if not torch.equal(r1.final_logits, r2.final_logits) or r1.code_ids != r2.code_ids:
        raise AssertionError("student-only export prediction/code differs")
    payload = torch.load(out / "student" / "weights.pt", weights_only=True)
    if set(payload) != {"student"}:
        raise AssertionError("export contains non-student payload")
    metrics = evaluate(out / "student", exported["manifest_sha256"], fixture["evaluation"],
        fixture["evaluation_sha256"], out / "evaluation", device="cpu")
    np.save(out / "single_image.npy", x[0].numpy())
    receipt = {"status": "PASS_SYNTHETIC_ONLY", "patient_training": "NOT_RUN",
        "gpu_validation": "NOT_RUN", "teacher_pretrained_weights": "NOT_AVAILABLE_LOCALLY",
        "stages": stages, "export": exported, "export_bitwise_parity": True,
        "export_payload_keys": sorted(payload), "evaluation_metrics": str(out / "evaluation" / "METRICS.json"),
        "evaluation_synthetic": metrics["synthetic"], "output": str(out)}
    write_json(out / "SMOKE_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new output folder")
    print(json.dumps(run_smoke(parser.parse_args().out), indent=2))
