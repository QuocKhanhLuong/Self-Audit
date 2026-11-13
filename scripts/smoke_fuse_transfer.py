#!/usr/bin/env python3
"""Run a synthetic Track B T/J/four-export CPU smoke (SOFTWARE_ONLY)."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(out: Path, threads: int = 1) -> dict:
    from self_audit.candidate_fuse import engine
    from self_audit.candidate_fuse.config import default_transfer_config, normalize_config
    from self_audit.candidate_fuse.mnms_data import write_synthetic_mnms_manifests
    from self_audit.candidate_fuse.evaluation import evaluate_exports
    from self_audit.candidate_fuse.export import verify_cold_load

    out = out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("transfer smoke output must be new or empty; preserve prior smoke receipts")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    manifests = write_synthetic_mnms_manifests(out / "data", seed=42)
    config = default_transfer_config(synthetic=True)
    source_model, _ = engine.build_teacher(config, synthetic=True)
    source_path = out / "source_teacher.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in source_model.state_dict().items()}}, source_path)
    metadata = {"status": "SELECTED", "sha256": __import__("hashlib").sha256(source_path.read_bytes()).hexdigest(),
                "identity": config["experiment"]["source_teacher"]["identity"]}
    metadata_path = Path(str(source_path) + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
    source = config["experiment"]["source_teacher"]
    source.update(path=str(source_path), sha256=metadata["sha256"],
                  metadata_sha256=__import__("hashlib").sha256(metadata_path.read_bytes()).hexdigest())
    config["runtime"].update(synthetic=True, amp_dtype="float32", num_threads=threads)
    for role, key in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{key}_manifest"] = manifests[role]["path"]
        config["data"][f"{key}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)
    teachers, students, zero_shot = [], [], []
    original_build_teacher = engine.build_teacher
    original_make_zero_shot = engine._create_zero_shot_artifacts
    from self_audit.candidate_fuse import student as student_module
    original_student = engine.FuseStudent
    def capture_teacher(*args, **kwargs):
        model, metadata = original_build_teacher(*args, **kwargs)
        teachers.append(model)
        return model, metadata
    def capture_student(*args, **kwargs):
        model = original_student(*args, **kwargs)
        students.append(model)
        return model
    def capture_zero_shot(*args, **kwargs):
        zero_shot.append(copy.deepcopy(kwargs["teacher"]))
        return original_make_zero_shot(*args, **kwargs)
    engine.build_teacher = capture_teacher
    engine._create_zero_shot_artifacts = capture_zero_shot
    engine.FuseStudent = capture_student
    try:
        result = engine.run_training(config, out / "run", device="cpu", stage="all")
    finally:
        engine.build_teacher = original_build_teacher
        engine._create_zero_shot_artifacts = original_make_zero_shot
        engine.FuseStudent = original_student
    expected = {"teacher_zero_shot", "teacher", "student", "fusion"}
    exports = result.get("exports", {})
    if set(exports) != expected:
        raise RuntimeError(f"B synthetic smoke expected four exports, got {sorted(exports)}")
    volumes = []
    for i, shape in enumerate(((5, 9, 11), (6, 11, 13), (7, 13, 15))):
        spacing = (5.0, 1.25, 1.1)
        geometry = {"shape": shape, "spacing_zyx": spacing,
            "affine_zyx": [[spacing[0],0,0,0],[0,spacing[1],0,0],[0,0,spacing[2],0],[0,0,0,1]],
            "context": "full_volume"}
        volumes.append((torch.rand((1,1,*shape), generator=torch.Generator().manual_seed(42+i)), geometry))
    if len(teachers) != 1 or len(students) != 1 or len(zero_shot) != 1:
        raise RuntimeError("expected one target teacher/student and one zero-shot source snapshot")
    refs = {"teacher_zero_shot": zero_shot[0], "teacher": teachers[0],
            "student": students[0], "fusion": {"teacher": teachers[0], "student": students[0]}}
    cold = {mode: verify_cold_load(exports[mode]["path"], volumes,
                                  original_model=refs[mode], device="cpu") for mode in sorted(expected)}
    if any(value.get("bitwise_equal") is not True for value in cold.values()):
        raise RuntimeError(f"cold-load parity failed: {cold}")
    test_result = evaluate_exports(
        {mode: {"path": exports[mode]["path"], "sha256": exports[mode]["sha256"]}
         for mode in expected},
        config["data"]["test_manifest"], out / "run" / "TEST_ONCE", "cpu", "test",
        config["data"]["test_manifest_sha256"],
    )
    if test_result.get("status") != "COMMITTED" or test_result.get("role") != "test":
        raise RuntimeError(f"four-mode synthetic TEST transaction did not commit: {test_result}")
    receipt = {"schema": "self_audit.candidate_fuse.transfer_smoke.v1", "status": "SOFTWARE_ONLY",
        "device": "cpu", "seed": 42, "stages": result.get("stages"), "exports": exports,
        "test_transaction": test_result,
        "cold_load": cold, "real_data_evidence": "NOT_RUN", "gpu_evidence": "NOT_RUN"}
    (out / "SOFTWARE_ONLY.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    print(json.dumps(run(args.out, args.threads), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
