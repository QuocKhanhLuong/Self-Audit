#!/usr/bin/env python3
"""Run the complete synthetic CPU FUSE pipeline and write a SOFTWARE_ONLY receipt."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(out: Path, threads: int = 1) -> dict:
    # Imports stay inside run so --help remains usable while the FUSE modules land.
    from self_audit.candidate_fuse.config import default_config, normalize_config
    from self_audit.candidate_fuse.data import write_synthetic_manifests
    from self_audit.candidate_fuse import engine
    from self_audit.candidate_fuse.export import verify_cold_load

    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

    manifests = write_synthetic_manifests(out / "data", seed=42)
    config = default_config(synthetic=True)
    config["seed"] = 42
    config["runtime"].update(synthetic=True, deterministic=True, num_threads=threads, amp_dtype="float32")
    for role, prefix in (("train", "train"), ("development", "val"), ("test", "test")):
        config["data"][f"{prefix}_manifest"] = manifests[role]["path"]
        config["data"][f"{prefix}_manifest_sha256"] = manifests[role]["sha256"]
    config = normalize_config(config)

    teachers = []
    students = []
    build_teacher = engine.build_teacher
    student_module = __import__("self_audit.candidate_fuse.student", fromlist=["FuseStudent"])
    build_student = student_module.FuseStudent

    def capture_teacher(*args, **kwargs):
        teacher, metadata = build_teacher(*args, **kwargs)
        teachers.append(teacher)
        return teacher, metadata

    def capture_student(*args, **kwargs):
        student = build_student(*args, **kwargs)
        students.append(student)
        return student

    engine.build_teacher = capture_teacher
    engine.FuseStudent = capture_student
    try:
        result = engine.run_training(config, out / "run", device="cpu", stage="all")
    finally:
        engine.build_teacher = build_teacher
        engine.FuseStudent = build_student
    if len(teachers) != 1 or len(students) != 1:
        raise RuntimeError(f"expected one training teacher and student, found {len(teachers)} and {len(students)}")
    teacher_has_no_grads = all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in teachers[0].parameters()
    )
    exports = result.get("exports")
    expected_modes = {"teacher", "student", "fusion"}
    if not isinstance(exports, dict) or set(exports) != expected_modes:
        raise RuntimeError(f"FUSE must emit exactly the three approved comparison exports, got: {exports}")
    export_clean = result.get("export_clean") is True
    if not export_clean:
        raise RuntimeError("engine did not confirm clean comparison export inventories")
    for mode in sorted(expected_modes):
        receipt = exports[mode]
        if receipt.get("mode") != mode or not Path(receipt.get("path", "")).is_file():
            raise RuntimeError(f"FUSE {mode} export receipt is missing or malformed: {receipt}")
        payload = torch.load(receipt["path"], map_location="cpu", weights_only=True)
        if payload.get("mode") != mode:
            raise RuntimeError(f"FUSE {mode} receipt points to {payload.get('mode')} payload")
        expected_keys = {"schema", "mode", "config", "metadata"}
        if mode in {"student", "fusion"}:
            expected_keys.add("student_state")
        if mode in {"teacher", "fusion"}:
            expected_keys.add("teacher_state")
        if set(payload) != expected_keys:
            raise RuntimeError(f"FUSE {mode} export has unexpected payload keys: {sorted(payload)}")
        state = payload.get("student_state", payload.get("teacher_state", {}))
        if any("mimic" in name.lower() for name in state):
            raise RuntimeError(f"FUSE {mode} export contains training-only mimic weights")
    deployment = result.get("deployment")
    if not isinstance(deployment, dict) or deployment.get("mode") not in {"student", "fusion"}:
        raise RuntimeError(f"FUSE deployment receipt is malformed: {deployment}")
    export_path = deployment.get("path")
    if export_path != exports[deployment["mode"]]["path"]:
        raise RuntimeError("deployment path does not match its selected mode receipt")

    # Cold-load fixtures are deterministic native volumes with valid physical
    # spacing.  Keep them separate from the training manifest so the fixture
    # explicitly exercises three distinct geometries, including odd H/W.
    volumes = []
    for index, shape in enumerate(((5, 9, 11), (6, 11, 13), (7, 13, 15))):
        spacing = (5.0, 1.25, 1.1)
        geometry = {
            "shape": shape,
            "spacing_zyx": spacing,
            "affine_zyx": [
                [spacing[0], 0.0, 0.0, 0.0],
                [0.0, spacing[1], 0.0, 0.0],
                [0.0, 0.0, spacing[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "context": "full_volume",
        }
        generator = torch.Generator(device="cpu").manual_seed(42 + index)
        image = torch.rand((1, 1, *shape), generator=generator, dtype=torch.float32)
        volumes.append((image, geometry))
    cold = {}
    for mode in sorted(expected_modes):
        reference = {"teacher": teachers[0], "student": students[0]} if mode == "fusion" else (
            students[0] if mode == "student" else teachers[0]
        )
        cold[mode] = verify_cold_load(
            exports[mode]["path"], volumes,
            original_model=reference,
            device="cpu",
        )
        if cold[mode].get("bitwise_equal") is not True or cold[mode].get("mode") != mode:
            raise RuntimeError(f"fresh-process {mode} parity failed: {cold[mode]}")
    if cold["student"]["mode"] != "student":
        raise RuntimeError(f"student cold-load did not load the selected student route: {cold['student']}")
    if set(cold) != expected_modes or any(value.get("bitwise_equal") is not True for value in cold.values()):
        raise RuntimeError(f"fresh-process comparison-export parity failed: {cold}")
    if not teacher_has_no_grads:
        parameters = tuple(teachers[0].parameters())
        grad_count = sum(parameter.grad is not None for parameter in parameters)
        trainable_count = sum(parameter.requires_grad for parameter in parameters)
        raise RuntimeError(
            "training teacher invariant failed after J: "
            f"{grad_count} parameters retain gradients, {trainable_count} remain trainable"
        )

    receipt = {
        "schema": "self_audit.candidate_fuse.smoke.v1",
        "status": "SOFTWARE_ONLY",
        "seed": 42,
        "device": "cpu",
        "threads": threads,
        "patient_training": "SYNTHETIC_ONLY",
        "gpu_evidence": "NOT_RUN",
        "real_data_evidence": "NOT_RUN",
        "stages": result.get("stages", ["T", "J", "VAL3", "deploy_export"]),
        "teacher_has_no_grads_after_J": teacher_has_no_grads,
        "teacher_runtime_invariant": result.get("teacher_has_no_grads_after_J") is True,
        "deployment": deployment,
        "exports": exports,
        "export_clean": export_clean,
        "cold_load": cold,
        "native_shapes": [list(x[0].shape[-3:]) for x in volumes],
        "output": str(out),
    }
    if receipt["export_clean"] is not True or not receipt["teacher_runtime_invariant"]:
        raise RuntimeError("engine did not confirm clean exports and frozen teacher after J")
    (out / "SOFTWARE_ONLY.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Explicit output directory (use /tmp)")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    print(json.dumps(run(args.out, args.threads), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
