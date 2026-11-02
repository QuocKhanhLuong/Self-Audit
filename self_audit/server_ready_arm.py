"""Bind future teacher outputs, then profile/lock/run the finite server arms.

This is orchestration around the canonical trainer, never a second trainer.
The input plan pins source, recipes and data before the campaign starts. Only
hashes of the declared B8 run's future outputs may be filled in later.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .server_runner import (
    ROOT, _read_pinned_json, _train_metadata, atomic_json, canonical_sha256,
    create_protocol_lock, profile_arm_config, run_locked_experiment,
    sha256_file, source_identity, _validate_server_arm_config,
)


def load_plan(path: str | Path, expected_sha: str) -> dict[str, Any]:
    plan = _read_pinned_json(path, expected_sha, "ready-arm plan")
    if plan.get("schema") != "self_audit.server_ready_arm.v1":
        raise ValueError("unsupported ready-arm plan")
    source = source_identity()
    if (plan.get("git_sha") != source["git_head"]
            or plan.get("source_closure_sha256") != source["closure_sha256"]):
        raise ValueError("ready-arm source differs from the pinned commit/closure")
    for key in ("train", "joint_train", "val", "mnm_val", "test", "b8_plan",
                "s1_template", "s2_template", "joint_template"):
        pin = plan[key]
        if set(pin) != {"path", "sha256"} or not Path(pin["path"]).is_absolute():
            raise ValueError(f"{key} requires an absolute path and SHA256")
        _read_pinned_json(pin["path"], pin["sha256"], key)
    b8 = _read_pinned_json(**_pin_args(plan["b8_plan"], "B8 plan"))
    if str(Path(b8["run_root"]).resolve()) != plan["teacher_root"]:
        raise ValueError("teacher root must be the exact preregistered B8 run")
    if b8["data_sha256"] != plan["train"]["sha256"]:
        raise ValueError("B8 and compact TRAIN hashes differ")
    for key in ("output_root", "teacher_root", "cache_root", "cache_python"):
        if not Path(plan[key]).is_absolute():
            raise ValueError(f"{key} must be absolute")
    if plan.get("profile") != {"warmup_steps": 2, "measured_steps": 5}:
        raise ValueError("bounded profile is locked to two warmup plus five measured steps")
    return plan


def _pin_args(pin: dict[str, str], label: str) -> dict[str, str]:
    return {"path": pin["path"], "expected_sha256": pin["sha256"], "label": label}


def assert_gpu_idle() -> None:
    """Do not overlap a training or unrelated compute process on the one GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError("GPU has an existing compute process; refusing a concurrent launch")


def verify_teacher(plan: dict[str, Any]) -> dict[str, Any]:
    from .compact_teacher import validate_reproduction_receipt, verify_reproduction_source
    root = Path(plan["teacher_root"])
    b8 = _read_pinned_json(**_pin_args(plan["b8_plan"], "B8 plan"))
    status = json.loads((root / "STATUS.json").read_text())
    if status.get("status") != "TEACHER_COMPLETED_REMOTE_VERIFIED":
        raise ValueError("B8 has not completed with verified weight backup")
    receipt = root / "TEACHER_IDENTITY.json"
    identity = validate_reproduction_receipt(
        receipt, expected_receipt_sha256=sha256_file(receipt),
        checkpoint_dir=root / "training/final", raw_student_path=root / "raw_student/raw_student.pt",
    )
    if (identity["source_commit"] != b8["source_commit"]
            or identity["source_sha256"] != b8["source_sha256"]):
        raise ValueError("B8 identity source differs from preregistered reproduction")
    if identity["quality_gate"] != "GO" or identity["metrics"]["test50"]["Macro4"] < 0.87:
        raise ValueError("new B8 failed the preregistered Macro4>=0.87 teacher gate")
    verify_reproduction_source(b8["source"], expected_commit=identity["source_commit"],
                               expected_closure=identity["source_sha256"])
    # Rebind receipt summaries to the actual historical evaluator output.
    for domain, count in (("val18", 18), ("test50", 50)):
        metrics = json.loads((root / (domain + "_evaluation") / "METRICS.json").read_text())
        classes = {c: metrics["macro_patient"][c]["dice"] for c in ("BG", "RV", "MYO", "LV")}
        actual = sum(classes.values()) / 4
        if actual != identity["metrics"][domain]["Macro4"] or len(metrics["patients"]) != count:
            raise ValueError("B8 evaluation summary does not match physical evaluation output")
    return identity


def teacher_binding(plan: dict[str, Any]) -> dict[str, Any]:
    identity = verify_teacher(plan)
    receipt = Path(plan["teacher_root"]) / "TEACHER_IDENTITY.json"
    result = {"status": "COMPLETED", "metrics": {"Macro4": identity["metrics"]["test50"]["Macro4"]},
              "identity_sha256": canonical_sha256(identity), "teacher_identity": identity,
              "source_receipt": str(receipt), "source_receipt_sha256": sha256_file(receipt),
              "physical_raw_equals_checkpoint_student": True, "remote_verified": True}
    atomic_json(Path(plan["output_root"]) / "TEACHER_BINDING.json", result)
    return result


def cache_disk_bytes(train_path: Path, train_sha: str) -> int:
    """Read only image headers: float32 four-class native TRAIN-U targets."""
    import numpy as np
    train, _, _ = _train_metadata(train_path, train_sha)
    required = 0
    for record in train["unlabeled"]:
        image = Path(record["image"])
        if not image.is_absolute():
            image = train_path.parent / image
        array = np.load(image, mmap_mode="r", allow_pickle=False)
        if array.ndim != 4:
            raise ValueError("teacher cache requires prepared native TZYX images")
        required += int(array.size) * 4 * 4 + 4096
        del array
    return required


def build_cache(plan: dict[str, Any]) -> dict[str, Any]:
    identity = verify_teacher(plan)
    root, cache = Path(plan["teacher_root"]), Path(plan["cache_root"])
    if cache.exists():
        raise FileExistsError("cache output already exists; verify or resume it explicitly, never overwrite")
    estimate = cache_disk_bytes(Path(plan["train"]["path"]), plan["train"]["sha256"])
    # Keep a fixed reserve for three compact runs, logs, and resume checkpoints.
    reserve = 12 << 30
    free = shutil.disk_usage(cache.parent).free
    if free < estimate + reserve:
        raise RuntimeError(f"cache needs {estimate} bytes plus {reserve} reserve; only {free} free")
    assert_gpu_idle()
    b8 = _read_pinned_json(**_pin_args(plan["b8_plan"], "B8 plan"))
    receipt = root / "TEACHER_IDENTITY.json"
    argv = [plan["cache_python"], str(ROOT / "scripts/build_compact_teacher_cache.py"),
            "--data-manifest", plan["train"]["path"], "--data-sha256", plan["train"]["sha256"],
            "--output-dir", str(cache), "--checkpoint-dir", str(root / "training/final"),
            "--reproduction-receipt", str(receipt), "--reproduction-receipt-sha256", sha256_file(receipt),
            "--reproduction-source", b8["source"], "--precision", "float32", "--device", "cuda",
            "--batch-size", "8"]
    subprocess.run(argv, cwd=ROOT, check=True)
    manifest = cache / "CACHE_MANIFEST.json"
    value = json.loads(manifest.read_text())
    if value["teacher_identity"]["payload_sha256"] != identity["payload_sha256"]:
        raise ValueError("cache is not bound to this teacher payload")
    result = {"status": "COMPLETED", "manifest": str(manifest), "sha256": sha256_file(manifest),
              "estimated_cache_bytes": estimate, "reserve_bytes": reserve}
    atomic_json(Path(plan["output_root"]) / "CACHE_READY.json", result)
    return result


def joint_eval_manifests(plan: dict[str, Any], out: Path) -> tuple[dict, dict]:
    """Metadata-only views; retain payload hashes/geometry and namespace IDs."""
    from .server_evaluation import _normalized_domain_manifest, _validate_shared_l4_l4_train
    train, _ = _validate_shared_l4_l4_train(Path(plan["joint_train"]["path"]), plan["joint_train"]["sha256"])
    train_ids = {r["id"] for r in train["labeled"] + train["unlabeled"]}
    patients, sources = [], {}
    for domain, key in (("acdc", "val"), ("mnm", "mnm_val")):
        pin = plan[key]
        manifest = _read_pinned_json(**_pin_args(pin, key))
        if bool(manifest.get("synthetic", False)) != bool(train.get("synthetic", False)):
            raise ValueError("joint evaluation synthetic flag differs from TRAIN")
        records, ids = _normalized_domain_manifest(domain, Path(pin["path"]), manifest)
        if set(ids) & train_ids:
            raise ValueError("joint development overlaps TRAIN")
        patients.extend(records)
        sources[domain] = pin
    combined = {"schema": "self_audit.native_manifest.v1", "role": "development",
                "synthetic": bool(train.get("synthetic", False)), "patients": patients,
                "source_manifests": sources, "selection_metric": "equal_domain_class_first_macro4"}
    val_path = out / "shared_development.json"
    atomic_json(val_path, combined)
    test = _read_pinned_json(**_pin_args(plan["test"], "ACDC TEST metadata"))
    if test["role"] != "test":
        raise ValueError("final TEST manifest must have role=test")
    # The normalizer enforces development for its normal caller. Here it only
    # resolves metadata; preserve the original test role in the resulting view.
    records, ids = _normalized_domain_manifest("acdc", Path(plan["test"]["path"]), {**test, "role": "development"})
    if set(ids) & (train_ids | {r["id"] for r in patients}):
        raise ValueError("joint TEST overlaps TRAIN/development")
    test_path = out / "acdc_test50_namespaced.json"
    atomic_json(test_path, {**test, "patients": records, "source_manifest": plan["test"]})
    return ({"path": str(val_path), "sha256": sha256_file(val_path)},
            {"path": str(test_path), "sha256": sha256_file(test_path)})


def bind_config(plan: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm not in {"S1", "S2", "joint_supervised"}:
        raise ValueError("only the three preregistered compact arms are permitted")
    key = "joint_template" if arm == "joint_supervised" else arm.lower() + "_template"
    config = copy.deepcopy(_read_pinned_json(**_pin_args(plan[key], "arm template")))
    train_pin = plan["joint_train"] if arm == "joint_supervised" else plan["train"]
    _, lids, _ = _train_metadata(train_pin["path"], train_pin["sha256"])
    config["experiment"]["launchable"] = True
    config["data"] = {"manifest_sha256": train_pin["sha256"], "budget_k": len(lids), "labeled_ids": lids}
    if arm != "joint_supervised":
        identity = verify_teacher(plan)
        cache_pin = json.loads((Path(plan["output_root"]) / "CACHE_READY.json").read_text())
        cache = _read_pinned_json(cache_pin["manifest"], cache_pin["sha256"], "cache")
        cached_identity = cache["teacher_identity"]
        if cached_identity["payload_sha256"] != identity["payload_sha256"]:
            raise ValueError("KD cache is from a different B8 teacher")
        report = json.loads((Path(plan["teacher_root"]) / "training/REPORT.json").read_text())
        config["teacher"].update(
            source=str(Path(plan["teacher_root"]) / "training/final"),
            source_manifest_sha256=identity["manifest_sha256"], source_payload_sha256=identity["payload_sha256"],
            source_training_cost={"report": str(Path(plan["teacher_root"]) / "training/REPORT.json"),
                                  "wall_seconds": report.get("elapsed_seconds", "NOT_KNOWN")},
            cache_path=cache_pin["manifest"], cache_sha256=cache_pin["sha256"],
            teacher_identity=cached_identity, teacher_identity_sha256=canonical_sha256(cached_identity),
            lineage={"matching_lineage_required": True, "budget_k": len(lids), "exact_labeled_ids": lids},
        )
    return config


def run_arm(plan: dict[str, Any], arm: str) -> dict[str, Any]:
    config = bind_config(plan, arm)  # All lineage gates precede output or GPU use.
    work = Path(plan["output_root"]) / arm
    if work.exists():
        raise FileExistsError("arm already has outputs; exact resume is required, no second fresh run")
    assert_gpu_idle()
    work.mkdir(parents=True)
    train = plan["joint_train"] if arm == "joint_supervised" else plan["train"]
    val, test = joint_eval_manifests(plan, work) if arm == "joint_supervised" else (plan["val"], plan["test"])
    config_path = work / "CONFIG.json"
    atomic_json(config_path, config)
    config_sha = sha256_file(config_path)
    _validate_server_arm_config(config_path, config_sha, train["path"], train["sha256"])
    profile_arm_config(work / "profile", arm_config_path=config_path, arm_config_sha256=config_sha,
                       train_manifest=train["path"], train_sha256=train["sha256"], device="cuda", **plan["profile"])
    profile_report = json.loads((work / "profile/arm_profile/REPORT.json").read_text())
    if profile_report["initialization"]["initial_state_sha256"] != plan["initial_state_sha256"]:
        raise ValueError("compact initialization differs from the pinned seed-42 S0 initialization")
    profile_path = work / "profile/PROFILE.json"
    lock_path = work / "LOCK.json"
    create_protocol_lock(lock_path, profile_path=profile_path, profile_sha256=sha256_file(profile_path),
                         train_manifest=train["path"], train_sha256=train["sha256"],
                         val_manifest=val["path"], val_sha256=val["sha256"],
                         test_manifest=test["path"], test_sha256=test["sha256"], batch_size=32,
                         arm_config_path=config_path, arm_config_sha256=config_sha)
    # Profiles never supply initialization: canonical trainer starts seed 42 anew.
    return run_locked_experiment(lock_path, sha256_file(lock_path), work / "run", device="cuda",
                                 rclone_base=plan["backup_destination"], arm_config_path=config_path,
                                 arm_config_sha256=config_sha)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "teacher-binding", "cache", "S1", "S2", "joint_supervised"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan, args.plan_sha256)
    if args.action == "verify":
        result = {"status": "VERIFIED_PLAN", "teacher": "NOT_CHECKED", "gpu": "NOT_ALLOCATED"}
    elif args.action == "teacher-binding":
        result = teacher_binding(plan)
    elif args.action == "cache":
        result = build_cache(plan)
    else:
        result = run_arm(plan, args.action)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("status") == "UPLOAD_PENDING" else 0
