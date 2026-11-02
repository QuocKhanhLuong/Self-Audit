"""Locked one-run server shell around the canonical compact trainer.

The profile subcommand measures bounded canonical supervised steps at B8/B32/B64.
The run subcommand launches one S0 only; it does not fan out arms or seeds.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping
import uuid

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "configs/server_compact/s0_wide_seed42.json"
PROFILE_BATCHES = (8, 32, 64)
LOCK_BATCHES = (32, 64)
TOTAL_STEPS = 10000
CHECKPOINT_EVERY = 1000
NORMALIZATION = "prepared"
EVALUATION_BATCH_SIZE = 8
RCLONE_HARD_TIMEOUT_SECONDS = 900.0
HISTORICAL_WIDE_KD = {
    "pseudo_lr": 0.001,
    "temperature": 1.0,
    "coefficient": 1.0,
    "pseudo_threshold": 0.7,
}
JOINT_SUPERVISED_MNM_IDS = {f"mnm:{pid}" for pid in ("A9E3G9", "B9E0Q1", "D3F9H9", "E5F5V7")}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def source_identity(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    files: dict[str, str] = {}
    candidates = set((root / "self_audit").rglob("*.py"))
    candidates.update((root / "scripts").glob("*.py"))
    candidates.update((root / "configs").rglob("*"))
    candidates.update({root / "train.py", root / "prepare_acdc.py", root / "ACTIVE_PIPELINE.json"})
    for path in sorted(candidates):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    required = {
        "train.py", "prepare_acdc.py", "scripts/run_server_compact.py", "scripts/export_compact_student.py",
        "scripts/evaluate_compact_student.py", "self_audit/server_runner.py",
        "configs/server_compact/s0_wide_seed42.json",
        "configs/server_compact/s1_plain_kd.json", "configs/server_compact/s2_plain_kd.json",
        "configs/fresh_s0/supervised_wide_seed42.json", "ACTIVE_PIPELINE.json",
    }
    missing = sorted(required - files.keys())
    if missing:
        raise FileNotFoundError(f"locked server source closure is missing required files: {missing}")
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
            text=True, timeout=10,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Git HEAD is required to pin the server source identity") from exc
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", git_head):
        raise RuntimeError("Git HEAD is not a full hexadecimal commit identity")
    closure_material = {"git_head": git_head.lower(), "files": files}
    closure = canonical_sha256(closure_material)
    return {"git_head": git_head, "git_status_paths": dirty, "files": files, "closure_sha256": closure}


def _read_pinned_json(path: str | Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} SHA-256 must be a mandatory 64-character pin")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"{label} hash mismatch: expected {expected_sha256}, got {actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _train_metadata(path: str | Path, expected_sha256: str) -> tuple[dict[str, Any], list[str], bool]:
    manifest = _read_pinned_json(path, expected_sha256, "TRAIN manifest")
    if manifest.get("schema") != "self_audit.data.v2" or manifest.get("role") != "training":
        raise ValueError("TRAIN must be a hash-pinned self_audit.data.v2 training manifest")
    synthetic_flag = manifest.get("synthetic", False)
    if type(synthetic_flag) is not bool:
        raise ValueError("TRAIN manifest synthetic flag must be boolean when present")
    labeled, unlabeled = manifest.get("labeled"), manifest.get("unlabeled")
    if not isinstance(labeled, list) or not labeled or not isinstance(unlabeled, list) or not unlabeled:
        raise ValueError("TRAIN must contain nonempty labeled and image-only unlabeled records")
    lids = [record.get("id") for record in labeled if isinstance(record, Mapping)]
    uids = [record.get("id") for record in unlabeled if isinstance(record, Mapping)]
    if len(lids) != len(labeled) or len(uids) != len(unlabeled) or not all(isinstance(x, str) and x for x in lids + uids):
        raise ValueError("TRAIN records must have nonempty string IDs")
    if len(set(lids)) != len(lids) or len(set(uids)) != len(uids) or set(lids) & set(uids):
        raise ValueError("TRAIN L/U IDs must be unique and disjoint")
    if type(manifest.get("label_budget")) is not int or manifest["label_budget"] != len(lids):
        raise ValueError("TRAIN label_budget must equal its exact dense labeled patient count")
    if any("mask" in rec or "masks" in rec for rec in unlabeled):
        raise ValueError("unlabeled TRAIN records must be image-only and cannot expose masks")
    return manifest, lids, synthetic_flag


def _eval_metadata(path: str | Path, expected_sha256: str, name: str, synthetic: bool) -> dict[str, Any]:
    manifest = _read_pinned_json(path, expected_sha256, name)
    from .compact_inference import VALID_EVAL_ROLES

    if manifest.get("role") not in VALID_EVAL_ROLES:
        raise ValueError(f"{name} manifest role is not an accepted compact evaluation role")
    if type(manifest.get("synthetic")) is not bool or manifest["synthetic"] != synthetic:
        raise ValueError(f"{name} synthetic flag must match TRAIN")
    patients = manifest.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError(f"{name} must contain at least one patient")
    ids = [record.get("id") for record in patients if isinstance(record, Mapping)]
    if len(ids) != len(patients) or not all(isinstance(x, str) and x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} patient IDs must be nonempty and unique")
    return manifest


def build_s0_config(batch_size: int, root: str | Path = ROOT) -> dict[str, Any]:
    if type(batch_size) is not int or batch_size not in (*PROFILE_BATCHES,):
        raise ValueError("S0 batch profile must be one of 8, 32, or 64")
    config = json.loads((Path(root) / "configs/server_compact/s0_wide_seed42.json").read_text())
    if config.get("schema") != "self_audit.compact.v1" or config.get("method") != "supervised":
        raise ValueError("server recipe must remain canonical compact supervised S0")
    model = config.get("model", {})
    locked = (
        config.get("steps") == TOTAL_STEPS
        and config.get("seed") == 42
        and config.get("crop_size") == 160
        and config.get("save_every") == CHECKPOINT_EVERY
        and config.get("warmup_steps") == 8
        and model.get("widths") == [32, 64, 128, 256]
        and model.get("init") == "kaiming_normal"
        and not model.get("pretrained")
        and config.get("teacher") == {}
        and config.get("equal_wall_time", {}).get("enabled") is False
    )
    if not locked:
        raise ValueError("server S0 recipe changed from its locked new-experiment protocol")
    config["batch_size"] = batch_size
    config["labeled_batch_size"] = batch_size
    config["unlabeled_batch_size"] = batch_size
    # Runner-only setting persisted with the trainer config for exact source/run
    # provenance; validation always uses the same bounded batch for every arm.
    config["eval_batch_size"] = EVALUATION_BATCH_SIZE
    from .compact_training import validate_compact_config

    normalized = validate_compact_config(config, allow_full_run=True)
    if normalized["method"] != "supervised" or normalized["steps"] != TOTAL_STEPS:
        raise ValueError("server runner only permits one canonical 10,000-step supervised S0")
    return normalized


def _cache_manifest_path(cache_path: str | Path) -> Path:
    path = Path(cache_path).resolve()
    if path.is_dir():
        candidates = (path / "CACHE_MANIFEST.json", path / "MANIFEST.json")
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return path


def _identity_core(identity: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "teacher_kind", "inference_state", "labeled_ids", "label_budget", "manifest_sha256",
        "payload_sha256", "source_sha256", "training_method",
    )
    return {key: identity.get(key) for key in keys}


def _validate_server_arm_config(
    config_path: str | Path,
    config_sha256: str,
    train_manifest: str | Path,
    train_sha256: str,
    *,
    root: str | Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], bool]:
    """Validate an immutable S1/S2-style config and metadata-only TRAIN/teacher lineage."""
    config_path = Path(config_path).resolve()
    raw = _read_pinned_json(config_path, config_sha256, "arm config")
    from .compact_training import validate_compact_config

    config = validate_compact_config(raw, allow_full_run=True)
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping) or experiment.get("launchable") is not True:
        raise ValueError("server arm config is a template until experiment.launchable is explicitly true")
    arm_id = raw.get("server_arm")
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError("server arm config requires a nonempty server_arm identifier")
    if config["method"] not in {"supervised", "plain_kd"}:
        raise ValueError("server arm config permits only canonical supervised or plain_kd methods")
    if config["steps"] != TOTAL_STEPS or config["save_every"] != CHECKPOINT_EVERY:
        raise ValueError("server arm steps and checkpoint cadence must remain 10,000 and 1,000")
    if raw.get("stop_steps") not in (None, False) or raw.get("stop_after_step") not in (None, False):
        raise ValueError("server arm config cannot override its immutable 10,000-step total")
    if config["seed"] != 42 or config["crop_size"] not in {160, 256}:
        raise ValueError("server arm seed must be 42 and crop size must be 160 or 256")
    if config["batch_size"] != 32 or config["labeled_batch_size"] != 32 or config["unlabeled_batch_size"] != 32:
        raise ValueError("server arm batch, labeled batch, and unlabeled batch must all be 32")
    if config.get("eval_batch_size") != EVALUATION_BATCH_SIZE:
        raise ValueError(f"server arm eval_batch_size must be {EVALUATION_BATCH_SIZE}")
    model = config.get("model", {})
    if (
        model.get("name") != "CompactStudent"
        or model.get("widths") != [32, 64, 128, 256]
        or model.get("in_channels") != 3
        or model.get("num_classes") != 4
        or model.get("init") != "kaiming_normal"
        or model.get("pretrained") not in (None, False)
    ):
        raise ValueError("server arms require wide seed-42 random initialization with no pretrained weights")
    if any(raw.get(key) not in (None, False) for key in (
        "initial_checkpoint", "initial_weights", "init_checkpoint", "pretrained_checkpoint", "warm_start", "resume"
    )) or any(model.get(key) not in (None, False) for key in ("checkpoint_path", "weights_path", "pretrained_checkpoint")):
        raise ValueError("server arms cannot initialize from trained weights")
    if config["lr"] != 0.0003 or config["weight_decay"] != 0.0001:
        raise ValueError("server arm supervised optimizer must match the historical wide recipe")
    if config["equal_wall_time"].get("enabled") is not False:
        raise ValueError("server arm equal-wall-time must remain disabled")

    if config["method"] == "plain_kd":
        expected_crop = {"S1": 160, "S2": 256}.get(arm_id)
        if expected_crop is None or config["crop_size"] != expected_crop:
            raise ValueError("plain_kd server_arm must be S1/crop160 or S2/crop256")
        if config["warmup_steps"] != 1000:
            raise ValueError("S1/S2 plain_kd requires the historical 1,000-step supervised warmup")
        for key, expected in HISTORICAL_WIDE_KD.items():
            if config[key] != expected:
                raise ValueError(f"historical wide plain_kd requires {key}={expected!r}")
        if config.get("selector") != "original" or any(key in raw for key in ("class_weights", "denominator", "confidence", "kd")):
            raise ValueError("plain_kd must retain the historical unweighted common-pixel soft-KD objective")
    elif config["warmup_steps"] > config["steps"]:
        raise ValueError("server arm warmup cannot exceed its immutable total step budget")

    train_path = Path(train_manifest).resolve()
    train, labeled_ids, synthetic = _train_metadata(train_path, train_sha256)
    unlabeled_ids = [record["id"] for record in train["unlabeled"]]
    data_cfg = raw.get("data")
    if not isinstance(data_cfg, Mapping):
        raise ValueError("server arm config must pin its TRAIN manifest and dense labeled lineage")
    if (
        data_cfg.get("manifest_sha256") != train_sha256.lower()
        or data_cfg.get("budget_k") != len(labeled_ids)
        or data_cfg.get("labeled_ids") != labeled_ids
    ):
        raise ValueError("arm config data lineage must exactly match the hash-pinned TRAIN manifest")

    if arm_id == "joint_supervised":
        from .server_evaluation import _validate_shared_l4_l4_train
        _validate_shared_l4_l4_train(train_path, train_sha256)
        non_mnm_ids = set(labeled_ids) - JOINT_SUPERVISED_MNM_IDS
        if (
            config["method"] != "supervised"
            or config["crop_size"] != 160
            or len(labeled_ids) != 8
            or set(labeled_ids) & JOINT_SUPERVISED_MNM_IDS != JOINT_SUPERVISED_MNM_IDS
            or len(non_mnm_ids) != 4
            or non_mnm_ids != {f"acdc:patient{pid}" for pid in ("007", "024", "026", "041")}
        ):
            raise ValueError("joint_supervised requires crop160 and exactly ACDC4 plus MNM4 in one L8 TRAIN manifest")

    if config["method"] == "supervised":
        teacher = raw.get("teacher") or {}
        if not isinstance(teacher, Mapping) or teacher:
            raise ValueError("supervised server arms cannot consume a teacher or teacher cache")
        if arm_id != "joint_supervised":
            raise ValueError("generic supervised arm locks are reserved for joint_supervised; S0 retains its existing lock path")
        return config, train, labeled_ids, unlabeled_ids, synthetic

    teacher = raw.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("plain_kd server arm requires an explicitly pinned fixed teacher and cache")
    identity = teacher.get("teacher_identity")
    identity_sha = teacher.get("teacher_identity_sha256")
    if not isinstance(identity, Mapping) or identity_sha != canonical_sha256(identity):
        raise ValueError("teacher identity requires an exact canonical teacher_identity_sha256 pin")
    teacher_kind = identity.get("teacher_kind")
    if not synthetic and teacher_kind != "reproduction":
        raise ValueError("real S1/S2 requires the newly reproduced B8 identity, not a historical baseline")
    if teacher_kind not in {"reproduction", "synthetic"}:
        raise ValueError("server plain_kd requires a reproduction identity for real data or synthetic identity for fixtures")
    if identity.get("training_method") != "ema_full" or identity.get("inference_state") != "student":
        raise ValueError("server plain_kd requires the historical full-EMA raw-student teacher semantics")
    teacher_lineage = teacher.get("lineage")
    if not isinstance(teacher_lineage, Mapping):
        raise ValueError("fixed teacher label lineage is mandatory")
    if teacher_lineage.get("budget_k") != len(labeled_ids) or teacher_lineage.get("exact_labeled_ids") != labeled_ids:
        raise ValueError("fixed teacher label lineage must exactly match the locked TRAIN-L support")
    declared_identity_sha = teacher.get("source_manifest_sha256")
    if declared_identity_sha is not None and declared_identity_sha != identity.get("manifest_sha256"):
        raise ValueError("teacher source manifest SHA differs from the pinned teacher identity")
    declared_payload_sha = teacher.get("source_payload_sha256")
    if declared_payload_sha is not None and declared_payload_sha not in {
        identity.get("payload_sha256"), identity.get("raw_student_sha256")
    }:
        raise ValueError("teacher source payload SHA differs from the pinned teacher identity")

    from .compact_teacher import preflight_teacher_lineage, validate_teacher_identity

    lineage_data = SimpleNamespace(manifest=train, lids=labeled_ids)
    validated_identity = validate_teacher_identity(
        identity,
        declared_k=len(labeled_ids),
        declared_labeled_ids=labeled_ids,
        data=lineage_data,
    )
    preflight_teacher_lineage(
        teacher_labeled_ids=validated_identity["labeled_ids"],
        student_labeled_ids=labeled_ids,
        data=lineage_data,
        warmstart_labeled_ids=labeled_ids,
        audit_labeled_ids=labeled_ids,
        label_budget=len(labeled_ids),
        declared_labeled_ids=labeled_ids,
        teacher_kind=validated_identity["teacher_kind"],
    )

    cache_path = teacher.get("cache_path")
    cache_sha = teacher.get("cache_sha256")
    if not isinstance(cache_path, str) or not cache_path:
        raise ValueError("plain_kd requires a fixed teacher cache path")
    if not isinstance(cache_sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", cache_sha):
        raise ValueError("plain_kd requires the fixed teacher cache manifest SHA-256")
    cache_manifest_path = _cache_manifest_path(cache_path)
    cache = _read_pinned_json(cache_manifest_path, cache_sha, "fixed teacher cache manifest")
    if cache.get("schema") != "self_audit.compact_teacher_cache.v1":
        raise ValueError("fixed teacher cache schema is not canonical compact_teacher_cache.v1")
    if cache.get("native_axes") != "TZYX" or cache.get("num_classes") != 4:
        raise ValueError("fixed teacher cache must pin native TZYX axes and four canonical classes")
    if cache.get("train_manifest_sha256") != train_sha256.lower():
        raise ValueError("fixed teacher cache was not built from the exact locked TRAIN manifest")
    cache_identity = cache.get("teacher_identity")
    if not isinstance(cache_identity, Mapping) or _identity_core(cache_identity) != _identity_core(validated_identity):
        raise ValueError("fixed teacher cache identity differs from the pinned reproduced B8 identity")
    if not synthetic and (
        cache_identity.get("verified") is not True
        or cache_identity.get("physical_payload_status") not in {"VERIFIED_ON_DISK", "REMOTE_DRIVE_VERIFIED"}
    ):
        raise ValueError("fixed teacher cache does not carry verified physical teacher provenance")
    cache_patients = cache.get("patients")
    if not isinstance(cache_patients, Mapping) or set(cache_patients) != set(unlabeled_ids):
        raise ValueError("fixed teacher cache must contain exactly every locked TRAIN-U patient")
    return config, train, labeled_ids, unlabeled_ids, synthetic


def profile_batches(
    out_dir: str | Path,
    *,
    train_manifest: str | Path | None = None,
    train_sha256: str | None = None,
    synthetic: bool = False,
    device: str = "cuda",
    warmup_steps: int = 2,
    measured_steps: int = 5,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Measure canonical supervised steps at B8/B32/B64 on one pinned TRAIN."""
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("profile warmup_steps must be a non-negative integer")
    if type(measured_steps) is not int or measured_steps < 1:
        raise ValueError("profile measured_steps must be a positive integer")
    total = warmup_steps + measured_steps
    if total > 128:
        raise ValueError("batch profiling is bounded to at most 128 canonical steps per batch")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if train_manifest is None:
        if not synthetic or train_sha256 is not None:
            raise ValueError("profile requires a hash-pinned TRAIN or explicit --synthetic fixture generation")
        from .compact_data import synthetic_compact_fixture

        train_manifest, train_sha256 = synthetic_compact_fixture(
            out / "synthetic_train", seed=42, n_labeled=2, n_unlabeled=2, size=32
        )
        synthetic = True
    elif train_sha256 is None:
        raise ValueError("provided TRAIN requires --train-sha256")
    train_manifest = Path(train_manifest).resolve()
    train_obj, _, actual_synthetic = _train_metadata(train_manifest, train_sha256)
    train_sha256 = sha256_file(train_manifest)
    if bool(synthetic) != actual_synthetic:
        raise ValueError("--synthetic must agree with the pinned TRAIN manifest")
    import torch
    from .compact_training import train_compact

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but CUDA is unavailable")
    src = source_identity(root)
    recipe_sha = sha256_file(Path(root) / "configs/server_compact/s0_wide_seed42.json")
    results = []
    for batch in PROFILE_BATCHES:
        config = build_s0_config(batch, root)
        profile_out = out / f"batch_{batch:02d}"
        try:
            report = train_compact(
                config,
                train_manifest,
                train_sha256,
                profile_out,
                device=device,
                synthetic=actual_synthetic,
                stop_after_step=total,
                allow_full_run=True,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            allocated = reserved = None
            if str(device).startswith("cuda") and torch.cuda.is_available():
                allocated = int(torch.cuda.max_memory_allocated(device))
                reserved = int(torch.cuda.max_memory_reserved(device))
                torch.cuda.empty_cache()
            results.append({
                "batch_size": batch, "status": "OOM", "steps_measured": 0,
                "warmup_steps_excluded": warmup_steps, "step_latency_seconds_median": None,
                "step_latency_seconds_p95": None, "labeled_slices_per_second": None,
                "sampled_l_plus_u_slices_per_second": None,
                "cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved,
                "error": f"{type(exc).__name__}: {exc}", "output": str(profile_out),
            })
            continue
        if report.get("steps") != total or report.get("status") != "STOPPED_BOUNDARY":
            raise RuntimeError(f"canonical profile for batch {batch} did not stop at its requested boundary")
        rows = [json.loads(line) for line in (profile_out / "TRAIN_LOG.jsonl").read_text().splitlines() if line.strip()]
        measured = rows[warmup_steps:]
        latencies = [float(row["timings"]["step_elapsed_seconds"]) for row in measured]
        if len(latencies) != measured_steps or not all(math.isfinite(x) and x > 0 for x in latencies):
            raise RuntimeError(f"canonical profile for batch {batch} lacks finite step timings")
        elapsed = sum(latencies)
        labeled_draws = sum(int(row["effective_batch_counts"]["labeled"]) for row in measured)
        all_draws = sum(
            int(row["effective_batch_counts"]["labeled"]) + int(row["effective_batch_counts"]["unlabeled"])
            for row in measured
        )
        allocated = [row.get("cuda_memory", {}).get("cuda_peak_allocated_bytes") for row in measured]
        reserved = [row.get("cuda_memory", {}).get("cuda_peak_reserved_bytes") for row in measured]
        allocated = [int(x) for x in allocated if type(x) is int]
        reserved = [int(x) for x in reserved if type(x) is int]
        results.append({
            "batch_size": batch,
            "status": "MEASURED",
            "steps_measured": measured_steps,
            "warmup_steps_excluded": warmup_steps,
            "step_latency_seconds_median": statistics.median(latencies),
            "step_latency_seconds_p95": _percentile(latencies, 95),
            "labeled_slices_per_second": labeled_draws / elapsed,
            "sampled_l_plus_u_slices_per_second": all_draws / elapsed,
            "cuda_peak_allocated_bytes": max(allocated) if allocated else None,
            "cuda_peak_reserved_bytes": max(reserved) if reserved else None,
            "trainer_report_sha256": sha256_file(profile_out / "REPORT.json"),
            "train_log_sha256": sha256_file(profile_out / "TRAIN_LOG.jsonl"),
            "output": str(profile_out),
        })
    profile = {
        "schema": "self_audit.server_compact.profile.v1",
        "status": "PROFILED",
        "synthetic": actual_synthetic,
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_sha256,
        "profile_device": device,
        "profile_steps_total": total,
        "measured_steps_per_batch": measured_steps,
        "batch_sizes": list(PROFILE_BATCHES),
        "results": results,
        "recipe_path": str(Path(root) / "configs/server_compact/s0_wide_seed42.json"),
        "recipe_sha256": recipe_sha,
        "source": src,
        "claim_boundary": "short canonical trainer profile only; no patient or full-run evidence is inferred",
    }
    atomic_json(out / "PROFILE.json", profile)
    profile["profile_sha256"] = sha256_file(out / "PROFILE.json")
    return profile


def profile_arm_config(
    out_dir: str | Path,
    *,
    arm_config_path: str | Path,
    arm_config_sha256: str,
    train_manifest: str | Path,
    train_sha256: str,
    synthetic: bool = False,
    device: str = "cuda",
    warmup_steps: int = 2,
    measured_steps: int = 5,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Measure a short canonical profile of one immutable arm at its configured batch/crop."""
    if type(warmup_steps) is not int or warmup_steps < 1:
        raise ValueError("arm profile requires at least one profile-only warmup step")
    if type(measured_steps) is not int or measured_steps < 1:
        raise ValueError("arm profile measured_steps must be a positive integer")
    total = warmup_steps + measured_steps
    if total > 128:
        raise ValueError("arm profiling is bounded to at most 128 canonical steps")
    config_path = Path(arm_config_path).resolve()
    config_sha256 = arm_config_sha256.lower()
    config, _train, _lids, _uids, actual_synthetic = _validate_server_arm_config(
        config_path, config_sha256, train_manifest, train_sha256, root=root,
    )
    if bool(synthetic) != actual_synthetic:
        raise ValueError("--synthetic must agree with the pinned TRAIN manifest")
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA profile requested but CUDA is unavailable")

    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    train_path = Path(train_manifest).resolve()
    train_sha256 = train_sha256.lower()
    profile_out = out / "arm_profile"
    profile_config = copy.deepcopy(config)
    profile_config["steps"] = total
    # This short, profile-only override measures a KD step after a small supervised
    # warmup. The locked arm retains its full 1,000-step historical warmup.
    profile_config["warmup_steps"] = warmup_steps
    try:
        from .compact_training import train_compact

        report = train_compact(
            profile_config,
            train_path,
            train_sha256,
            profile_out,
            device=device,
            synthetic=actual_synthetic,
            stop_after_step=total,
            allow_full_run=True,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        allocated = reserved = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            torch.cuda.empty_cache()
        results = [{
            "batch_size": config["batch_size"], "status": "OOM", "steps_measured": 0,
            "profile_warmup_steps": warmup_steps, "step_latency_seconds_median": None,
            "step_latency_seconds_p95": None, "labeled_slices_per_second": None,
            "sampled_l_plus_u_slices_per_second": None,
            "cuda_peak_allocated_bytes": allocated, "cuda_peak_reserved_bytes": reserved,
            "error": f"{type(exc).__name__}: {exc}", "output": str(profile_out),
        }]
    else:
        # This disposable profile has a shortened total budget. The canonical
        # trainer reports COMPLETED when that exact budget is reached, even
        # when stop_after_step points to the same boundary. Never confuse this
        # profile completion with a completed 10,000-step experiment.
        bounded_status = report.get("status") == "STOPPED_BOUNDARY" or (
            report.get("status") == "COMPLETED" and report.get("total_budget") == total
        )
        if report.get("steps") != total or not bounded_status:
            raise RuntimeError("canonical arm profile did not stop at its requested boundary")
        rows = [
            json.loads(line) for line in (profile_out / "TRAIN_LOG.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if [row.get("step") for row in rows] != list(range(total)):
            raise RuntimeError("canonical arm profile has incomplete or inconsistent step coverage")
        measured = rows[warmup_steps:]
        latencies = [float(row["timings"]["step_elapsed_seconds"]) for row in measured]
        if len(latencies) != measured_steps or not all(math.isfinite(x) and x > 0 for x in latencies):
            raise RuntimeError("canonical arm profile lacks finite measured step timings")
        elapsed = sum(latencies)
        labeled_draws = sum(int(row["effective_batch_counts"]["labeled"]) for row in measured)
        all_draws = sum(
            int(row["effective_batch_counts"]["labeled"]) + int(row["effective_batch_counts"]["unlabeled"])
            for row in measured
        )
        allocated = [row.get("cuda_memory", {}).get("cuda_peak_allocated_bytes") for row in measured]
        reserved = [row.get("cuda_memory", {}).get("cuda_peak_reserved_bytes") for row in measured]
        allocated = [int(x) for x in allocated if type(x) is int]
        reserved = [int(x) for x in reserved if type(x) is int]
        results = [{
            "batch_size": config["batch_size"], "status": "MEASURED", "steps_measured": measured_steps,
            "profile_warmup_steps": warmup_steps,
            "step_latency_seconds_median": statistics.median(latencies),
            "step_latency_seconds_p95": _percentile(latencies, 95),
            "labeled_slices_per_second": labeled_draws / elapsed,
            "sampled_l_plus_u_slices_per_second": all_draws / elapsed,
            "cuda_peak_allocated_bytes": max(allocated) if allocated else None,
            "cuda_peak_reserved_bytes": max(reserved) if reserved else None,
            "trainer_report_sha256": sha256_file(profile_out / "REPORT.json"),
            "train_log_sha256": sha256_file(profile_out / "TRAIN_LOG.jsonl"),
            "output": str(profile_out),
        }]
    src = source_identity(root)
    profile = {
        "schema": "self_audit.server_compact.profile.v1",
        "profile_kind": "arm",
        "status": "PROFILED",
        "synthetic": actual_synthetic,
        "train_manifest": str(train_path),
        "train_manifest_sha256": train_sha256,
        "profile_device": device,
        "profile_steps_total": total,
        "measured_steps_per_batch": measured_steps,
        "batch_sizes": [config["batch_size"]],
        "results": results,
        "arm_id": config["server_arm"],
        "method": config["method"],
        "crop_size": config["crop_size"],
        "recipe_warmup_steps": config["warmup_steps"],
        "profile_warmup_steps": warmup_steps,
        "profile_only_random_initialization": True,
        "arm_config_path": str(config_path),
        "arm_config_sha256": config_sha256,
        "arm_config_canonical_sha256": canonical_sha256(config),
        "teacher_identity_sha256": config.get("teacher", {}).get("teacher_identity_sha256"),
        "teacher_cache_sha256": config.get("teacher", {}).get("cache_sha256"),
        "recipe_path": str(config_path),
        "recipe_sha256": config_sha256,
        "source": src,
        "claim_boundary": "short profile-only run with random initialization; no patient quality or full-run evidence",
    }
    atomic_json(out / "PROFILE.json", profile)
    profile["profile_sha256"] = sha256_file(out / "PROFILE.json")
    return profile


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _server_arm_protocol(config: Mapping[str, Any], arm_config_sha256: str) -> dict[str, Any]:
    teacher = config.get("teacher", {})
    protocol = {
        "arm_id": config["server_arm"],
        "method": config["method"],
        "architecture": "CompactStudent",
        "widths": [32, 64, 128, 256],
        "pretrained": False,
        "initialization": "kaiming_normal_seed42",
        "seed": 42,
        "crop_size": config["crop_size"],
        "steps": TOTAL_STEPS,
        "warmup_steps": config["warmup_steps"],
        "save_every": CHECKPOINT_EVERY,
        "batch_size": config["batch_size"],
        "labeled_batch_size": config["labeled_batch_size"],
        "unlabeled_batch_size": config["unlabeled_batch_size"],
        "eval_batch_size": config["eval_batch_size"],
        "arm_config_sha256": arm_config_sha256.lower(),
        "teacher_identity_sha256": teacher.get("teacher_identity_sha256"),
        "teacher_cache_sha256": teacher.get("cache_sha256"),
        "best_rule": ("highest equal-domain VAL Macro4; exact ties keep earliest step"
                      if config["server_arm"] == "joint_supervised" else
                      "highest VAL class_first_macro4; exact ties keep earliest step"),
        "selection_metric": ("equal_domain_class_first_macro4"
                             if config["server_arm"] == "joint_supervised" else "class_first_macro4"),
        "test_policy": "optional TEST only after best selection and frozen best-VAL predictions",
    }
    if config["method"] == "plain_kd":
        protocol["kd_loss"] = "soft_cross_entropy"
        protocol["kd_denominator"] = "common_valid_confident_pixel_count"
        protocol["confidence_threshold_domain"] = "raw_teacher_max_probability_before_temperature"
        protocol["teacher_cache_semantics"] = "FixedTeacherCache; exact TRAIN manifest and full TRAIN-U identity"
    return protocol


def _validate_arm_profile_summary(
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm_config_path: Path,
    arm_config_sha256: str,
    train_sha256: str,
) -> None:
    if (
        profile.get("schema") != "self_audit.server_compact.profile.v1"
        or profile.get("profile_kind") != "arm"
        or profile.get("status") != "PROFILED"
    ):
        raise ValueError("generic arm lock requires a completed arm-specific profile receipt")
    if (
        profile.get("batch_sizes") != [32]
        or profile.get("train_manifest_sha256") != train_sha256.lower()
        or profile.get("arm_config_path") != str(arm_config_path.resolve())
        or profile.get("arm_config_sha256") != arm_config_sha256.lower()
        or profile.get("arm_config_canonical_sha256") != canonical_sha256(config)
        or profile.get("arm_id") != config.get("server_arm")
        or profile.get("method") != config.get("method")
        or profile.get("crop_size") != config.get("crop_size")
        or profile.get("recipe_warmup_steps") != config.get("warmup_steps")
        or profile.get("teacher_identity_sha256") != config.get("teacher", {}).get("teacher_identity_sha256")
        or profile.get("teacher_cache_sha256") != config.get("teacher", {}).get("cache_sha256")
        or profile.get("profile_only_random_initialization") is not True
    ):
        raise ValueError("arm profile is not bound to the exact config, batch, crop, teacher, and cache")
    profile_warmup = profile.get("profile_warmup_steps")
    measured_steps = profile.get("measured_steps_per_batch")
    if (
        type(profile_warmup) is not int or profile_warmup < 1
        or type(measured_steps) is not int or measured_steps < 1
        or profile.get("profile_steps_total") != profile_warmup + measured_steps
        or profile.get("profile_steps_total") > 128
    ):
        raise ValueError("arm profile exceeds its bounded warmup and measurement budget")
    if config.get("method") == "plain_kd" and profile_warmup >= config.get("warmup_steps", 0):
        raise ValueError("KD profile must use a short profile-only warmup below the locked recipe warmup")
    result = profile.get("results")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], Mapping):
        raise ValueError("arm profile must contain exactly one B32 result")
    item = result[0]
    if item.get("batch_size") != 32 or item.get("status") != "MEASURED" or item.get("steps_measured") != measured_steps:
        raise ValueError("arm profile must contain a measured B32 result for its declared step count")
    for key in ("step_latency_seconds_median", "step_latency_seconds_p95", "labeled_slices_per_second"):
        value = item.get(key)
        if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("arm profile must contain finite positive timing and throughput measurements")


def _create_arm_protocol_lock(
    out_path: str | Path,
    *,
    profile_path: str | Path,
    profile_sha256: str,
    train_manifest: str | Path,
    train_sha256: str,
    val_manifest: str | Path,
    val_sha256: str,
    batch_size: int,
    arm_config_path: str | Path,
    arm_config_sha256: str,
    test_manifest: str | Path | None = None,
    test_sha256: str | None = None,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    if batch_size != 32:
        raise ValueError("generic server arm locks require the common batch size 32")
    profile_sha256, train_sha256, val_sha256 = (
        profile_sha256.lower(), train_sha256.lower(), val_sha256.lower()
    )
    if test_sha256 is not None:
        test_sha256 = test_sha256.lower()
    profile = _read_pinned_json(profile_path, profile_sha256, "profile receipt")
    config_path = Path(arm_config_path).resolve()
    config_sha256 = arm_config_sha256.lower()
    config, train, labeled_ids, _unlabeled_ids, synthetic = _validate_server_arm_config(
        config_path, config_sha256, train_manifest, train_sha256, root=root,
    )
    _validate_arm_profile_summary(
        profile, config, arm_config_path=config_path,
        arm_config_sha256=config_sha256, train_sha256=train_sha256,
    )
    if profile.get("source", {}).get("closure_sha256") != source_identity(root)["closure_sha256"]:
        raise ValueError("arm profile source closure differs from current locked server code")

    train_path = Path(train_manifest).resolve()
    val_path = Path(val_manifest).resolve()
    val = _eval_metadata(val_path, val_sha256, "VAL", synthetic)
    train_ids = set(labeled_ids) | {str(record["id"]) for record in train["unlabeled"]}
    val_ids = {str(record["id"]) for record in val["patients"]}
    if train_ids & val_ids:
        raise ValueError("VAL patient IDs overlap the complete TRAIN L/U union")
    test_input = None
    if test_manifest is not None or test_sha256 is not None:
        if test_manifest is None or test_sha256 is None:
            raise ValueError("optional TEST requires both manifest path and SHA-256")
        test_path = Path(test_manifest).resolve()
        test = _eval_metadata(test_path, test_sha256, "TEST", synthetic)
        test_ids = {str(record["id"]) for record in test["patients"]}
        if test_ids & (train_ids | val_ids):
            raise ValueError("TEST IDs must be disjoint from TRAIN and VAL")
        test_input = {"path": str(test_path), "sha256": test_sha256}

    base_recipe_path = Path(root) / "configs/server_compact/s0_wide_seed42.json"
    src = source_identity(root)
    lock = {
        "schema": "self_audit.server_compact.lock.v2",
        "status": "LOCKED",
        "new_experiment": True,
        "protocol": _server_arm_protocol(config, config_sha256),
        "config": config,
        "config_sha256": canonical_sha256(config),
        "arm_config_path": str(config_path),
        "arm_config_sha256": config_sha256,
        "base_recipe_path": str(base_recipe_path.resolve()),
        "base_recipe_sha256": sha256_file(base_recipe_path),
        "profile_path": str(Path(profile_path).resolve()),
        "profile_sha256": profile_sha256,
        "inputs": {
            "train": {"path": str(train_path), "sha256": train_sha256},
            "val": {"path": str(val_path), "sha256": val_sha256},
            "test": test_input,
            "synthetic": synthetic,
        },
        "source": src,
        "training_claim_boundary": "VAL is development-only equal-domain checkpoint selection; no VAL/TEST mask enters training or a quality gate",
    }
    out_path = Path(out_path).resolve()
    if out_path.exists():
        raise FileExistsError(f"protocol lock already exists: {out_path}")
    atomic_json(out_path, lock)
    lock["lock_sha256"] = sha256_file(out_path)
    return lock


def create_protocol_lock(
    out_path: str | Path,
    *,
    profile_path: str | Path,
    profile_sha256: str,
    train_manifest: str | Path,
    train_sha256: str,
    val_manifest: str | Path,
    val_sha256: str,
    batch_size: int,
    test_manifest: str | Path | None = None,
    test_sha256: str | None = None,
    arm_config_path: str | Path | None = None,
    arm_config_sha256: str | None = None,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Freeze one B32/B64 choice and exact TRAIN/VAL/optional TEST pins."""
    if arm_config_path is not None or arm_config_sha256 is not None:
        if arm_config_path is None or arm_config_sha256 is None:
            raise ValueError("generic arm lock requires both arm_config_path and arm_config_sha256")
        return _create_arm_protocol_lock(
            out_path,
            profile_path=profile_path,
            profile_sha256=profile_sha256,
            train_manifest=train_manifest,
            train_sha256=train_sha256,
            val_manifest=val_manifest,
            val_sha256=val_sha256,
            test_manifest=test_manifest,
            test_sha256=test_sha256,
            batch_size=batch_size,
            arm_config_path=arm_config_path,
            arm_config_sha256=arm_config_sha256,
            root=root,
        )
    if type(batch_size) is not int or batch_size not in LOCK_BATCHES:
        raise ValueError("protocol lock must select batch 32 or 64")
    profile_sha256 = profile_sha256.lower()
    train_sha256 = train_sha256.lower()
    val_sha256 = val_sha256.lower()
    if test_sha256 is not None:
        test_sha256 = test_sha256.lower()
    profile = _read_pinned_json(profile_path, profile_sha256, "profile receipt")
    if profile.get("schema") != "self_audit.server_compact.profile.v1" or profile.get("status") != "PROFILED":
        raise ValueError("profile receipt is not a completed server compact profile")
    if profile.get("batch_sizes") != list(PROFILE_BATCHES):
        raise ValueError("profile must include exactly B8/B32/B64")
    if profile.get("train_manifest_sha256") != train_sha256:
        raise ValueError("profile TRAIN identity differs from the requested protocol lock")
    profile_results = profile.get("results")
    if not isinstance(profile_results, list) or {item.get("batch_size") for item in profile_results} != set(PROFILE_BATCHES):
        raise ValueError("profile results do not cover B8/B32/B64")
    for item in profile_results:
        if item.get("status") == "OOM":
            continue
        if item.get("status") != "MEASURED" or not all(
            type(item.get(key)) in (float, int) and math.isfinite(float(item[key])) and float(item[key]) > 0
            for key in ("step_latency_seconds_median", "step_latency_seconds_p95", "labeled_slices_per_second")
        ):
            raise ValueError("profile latency and throughput must be finite positive measurements or explicit OOM")
    selected_profile = next(item for item in profile_results if item.get("batch_size") == batch_size)
    if selected_profile.get("status") != "MEASURED":
        raise ValueError("selected batch did not complete its profile; lock a measured B32 or B64")
    train_path = Path(train_manifest).resolve()
    train, lids, synthetic = _train_metadata(train_path, train_sha256)
    val_path = Path(val_manifest).resolve()
    val = _eval_metadata(val_path, val_sha256, "VAL", synthetic)
    train_ids = set(lids) | {str(rec["id"]) for rec in train["unlabeled"]}
    val_ids = {str(record["id"]) for record in val["patients"]}
    if train_ids & val_ids:
        raise ValueError("VAL patient IDs overlap the complete TRAIN L/U union")
    test_input = None
    if test_manifest is not None or test_sha256 is not None:
        if test_manifest is None or test_sha256 is None:
            raise ValueError("optional TEST requires both manifest path and SHA-256")
        test_path = Path(test_manifest).resolve()
        test = _eval_metadata(test_path, test_sha256, "TEST", synthetic)
        test_ids = {str(record["id"]) for record in test["patients"]}
        if test_ids & (train_ids | val_ids):
            raise ValueError("TEST IDs must be disjoint from TRAIN and VAL")
        test_input = {"path": str(test_path), "sha256": test_sha256.lower()}
    recipe_path = Path(root) / "configs/server_compact/s0_wide_seed42.json"
    if profile.get("recipe_sha256") != sha256_file(recipe_path):
        raise ValueError("profile source recipe bytes differ from current server recipe")
    src = source_identity(root)
    if profile.get("source", {}).get("closure_sha256") != src["closure_sha256"]:
        raise ValueError("profile source closure differs from current locked server code")
    config = build_s0_config(batch_size, root)
    lock = {
        "schema": "self_audit.server_compact.lock.v1",
        "status": "LOCKED",
        "new_experiment": True,
        "protocol": {
            "method": "supervised",
            "architecture": "CompactStudent",
            "widths": [32, 64, 128, 256],
            "pretrained": False,
            "initialization": "kaiming_normal_seed42",
            "seed": 42,
            "crop_size": 160,
            "steps": TOTAL_STEPS,
            "save_every": CHECKPOINT_EVERY,
            "batch_size": batch_size,
            "labeled_batch_size": batch_size,
            "unlabeled_batch_size": batch_size,
            "eval_batch_size": EVALUATION_BATCH_SIZE,
            "best_rule": "highest VAL class_first_macro4; exact ties keep earliest step",
            "test_policy": "optional TEST only after best selection and frozen best-VAL predictions",
        },
        "config": config,
        "config_sha256": canonical_sha256(config),
        "base_recipe_path": str(recipe_path.resolve()),
        "base_recipe_sha256": sha256_file(recipe_path),
        "profile_path": str(Path(profile_path).resolve()),
        "profile_sha256": profile_sha256.lower(),
        "inputs": {
            "train": {"path": str(train_path), "sha256": train_sha256.lower()},
            "val": {"path": str(val_path), "sha256": val_sha256.lower()},
            "test": test_input,
            "synthetic": synthetic,
        },
        "source": src,
        "training_claim_boundary": "VAL is development-only checkpoint selection; no VAL/TEST mask enters the trainer or a quality gate",
    }
    out_path = Path(out_path).resolve()
    if out_path.exists():
        raise FileExistsError(f"protocol lock already exists: {out_path}")
    atomic_json(out_path, lock)
    lock["lock_sha256"] = sha256_file(out_path)
    return lock


def verify_protocol_lock(path: str | Path, expected_sha256: str, root: str | Path = ROOT) -> dict[str, Any]:
    lock = _read_pinned_json(path, expected_sha256, "protocol lock")
    schema = lock.get("schema")
    if schema not in {"self_audit.server_compact.lock.v1", "self_audit.server_compact.lock.v2"} or lock.get("status") != "LOCKED":
        raise ValueError("protocol lock schema/status invalid")
    if canonical_sha256(lock.get("config")) != lock.get("config_sha256"):
        raise ValueError("protocol config changed after the lock was written")
    if sha256_file(lock["base_recipe_path"]) != lock["base_recipe_sha256"]:
        raise ValueError("base server recipe changed after protocol lock")
    current_source = source_identity(root)
    if current_source["closure_sha256"] != lock.get("source", {}).get("closure_sha256"):
        raise ValueError("server source closure changed after protocol lock")
    if schema == "self_audit.server_compact.lock.v2":
        arm_config_path = lock.get("arm_config_path")
        arm_config_sha256 = lock.get("arm_config_sha256")
        if not isinstance(arm_config_path, str) or not isinstance(arm_config_sha256, str):
            raise ValueError("generic arm lock is missing its explicit arm config path/SHA pin")
        train_ref = lock["inputs"]["train"]
        config, train, labeled_ids, _unlabeled_ids, synthetic = _validate_server_arm_config(
            arm_config_path, arm_config_sha256, train_ref["path"], train_ref["sha256"], root=root,
        )
        if config != lock["config"] or canonical_sha256(config) != lock["config_sha256"]:
            raise ValueError("locked config differs from the exact SHA-pinned arm config")
        if lock["protocol"] != _server_arm_protocol(config, arm_config_sha256):
            raise ValueError("arm protocol summary differs from the immutable arm config")
        profile_path = lock.get("profile_path")
        profile_sha256 = lock.get("profile_sha256")
        if not isinstance(profile_path, str) or not isinstance(profile_sha256, str):
            raise ValueError("generic arm lock is missing its exact profile receipt pin")
        profile = _read_pinned_json(profile_path, profile_sha256, "profile receipt")
        _validate_arm_profile_summary(
            profile, config, arm_config_path=Path(arm_config_path),
            arm_config_sha256=arm_config_sha256, train_sha256=train_ref["sha256"],
        )
        if profile.get("source", {}).get("closure_sha256") != current_source["closure_sha256"]:
            raise ValueError("arm profile is not bound to the locked config, TRAIN manifest, and current source")
        if lock["inputs"].get("synthetic") is not synthetic:
            raise ValueError("generic arm lock synthetic flag differs from TRAIN")
        val_ref = lock["inputs"]["val"]
        val = _eval_metadata(val_ref["path"], val_ref["sha256"], "VAL", synthetic)
        train_ids = set(labeled_ids) | {str(record["id"]) for record in train["unlabeled"]}
        val_ids = {str(record["id"]) for record in val["patients"]}
        if train_ids & val_ids:
            raise ValueError("TRAIN/VAL patient ID overlap detected")
        test_ref = lock["inputs"].get("test")
        if test_ref is not None:
            test = _eval_metadata(test_ref["path"], test_ref["sha256"], "TEST", synthetic)
            test_ids = {str(record["id"]) for record in test["patients"]}
            if test_ids & (train_ids | val_ids):
                raise ValueError("TRAIN/VAL/TEST patient ID overlap detected")
        return lock
    batch = lock["protocol"].get("batch_size")
    if batch not in LOCK_BATCHES or lock["config"].get("labeled_batch_size") != batch or lock["config"].get("unlabeled_batch_size") != batch:
        raise ValueError("protocol lock batch fields are inconsistent")
    if lock["config"] != build_s0_config(batch, root):
        raise ValueError("locked config differs from the one allowed new S0 recipe")
    train_ref = lock["inputs"]["train"]
    train, lids, synthetic = _train_metadata(train_ref["path"], train_ref["sha256"])
    val_ref = lock["inputs"]["val"]
    val = _eval_metadata(val_ref["path"], val_ref["sha256"], "VAL", synthetic)
    train_ids = set(lids) | {str(rec["id"]) for rec in train["unlabeled"]}
    val_ids = {str(rec["id"]) for rec in val["patients"]}
    if train_ids & val_ids:
        raise ValueError("TRAIN/VAL patient ID overlap detected")
    test_ref = lock["inputs"].get("test")
    if test_ref is not None:
        test = _eval_metadata(test_ref["path"], test_ref["sha256"], "TEST", synthetic)
        test_ids = {str(rec["id"]) for rec in test["patients"]}
        if test_ids & (train_ids | val_ids):
            raise ValueError("TRAIN/VAL/TEST patient ID overlap detected")
    return lock


def _to_cpu_clone(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_clone(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_cpu_clone(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_clone(child) for child in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (left.device.type == "cpu" and right.device.type == "cpu"
                and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
                and torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    try:
        return left == right
    except Exception:
        return False


class preserve_training_state:
    """Restore RNG, model, optimizer, gradients and module modes after evaluation."""

    def __init__(self, model: Any, optimizer: Any):
        self.model, self.optimizer = model, optimizer

    def __enter__(self):
        import numpy as np
        import torch

        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state().clone()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.model_state = _to_cpu_clone(self.model.state_dict())
        self.optimizer_state = _to_cpu_clone(self.optimizer.state_dict())
        self.gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                          for name, p in self.model.named_parameters()}
        self.module_modes = {name: module.training for name, module in self.model.named_modules()}
        self.torch_threads = torch.get_num_threads()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        self.deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.allow_tf32)
        self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        return self

    def __exit__(self, exc_type, exc, tb):
        import numpy as np
        import torch

        try:
            if not _tree_equal(_to_cpu_clone(self.model.state_dict()), self.model_state):
                self.model.load_state_dict(self.model_state, strict=True)
            if not _tree_equal(_to_cpu_clone(self.optimizer.state_dict()), self.optimizer_state):
                self.optimizer.load_state_dict(self.optimizer_state)
            for name, parameter in self.model.named_parameters():
                saved = self.gradients[name]
                if saved is None:
                    parameter.grad = None
                elif parameter.grad is None or not torch.equal(parameter.grad.detach().cpu(), saved):
                    parameter.grad = saved.to(device=parameter.device, dtype=parameter.dtype).clone()
            for name, module in self.model.named_modules():
                module.training = self.module_modes[name]
            if torch.get_num_threads() != self.torch_threads:
                torch.set_num_threads(self.torch_threads)
            if (torch.are_deterministic_algorithms_enabled() != self.deterministic
                    or torch.is_deterministic_algorithms_warn_only_enabled() != self.deterministic_warn_only):
                torch.use_deterministic_algorithms(self.deterministic, warn_only=self.deterministic_warn_only)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.backends.cudnn.allow_tf32 = self.cudnn
            torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
        finally:
            random.setstate(self.python_rng)
            np.random.set_state(self.numpy_rng)
            torch.set_rng_state(self.torch_rng)
            if self.cuda_rng is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.cuda_rng)
        return False


def _read_validation_log(run_root: Path, *, verify_artifacts: bool = True) -> list[dict[str, Any]]:
    path = run_root / "VALIDATION_LOG.jsonl"
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for entry in entries:
        key = (int(entry["step"]), entry["checkpoint_manifest_sha256"])
        if key in seen:
            raise ValueError(f"duplicate VAL selection record for step/hash {key}")
        seen.add(key)
        score = entry.get("macro4")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("VAL log contains a non-finite Macro4")
        if verify_artifacts:
            eval_dir = Path(entry["validation_dir"])
            for filename, key_name in (("METRICS.json", "metrics_sha256"),
                                       ("PREDICTIONS_FROZEN.json", "predictions_frozen_sha256")):
                file = eval_dir / filename
                if not file.is_file() or sha256_file(file) != entry[key_name]:
                    raise ValueError(f"VAL artifact hash mismatch: {file}")
            metrics = json.loads((eval_dir / "METRICS.json").read_text())
            if metrics.get("class_first_macro4") != score:
                raise ValueError("VAL log Macro4 differs from the bound metrics receipt")
    return entries


def _select_best(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        raise ValueError("no finite VAL Macro4 checkpoint is available for selection")
    # Sorting plus strict comparison keeps the earliest checkpoint on exact ties.
    best = None
    for entry in sorted(entries, key=lambda row: int(row["step"])):
        score = float(entry["macro4"])
        if best is None or score > float(best["macro4"]):
            best = entry
    return best


def _next_validation_dir(run_root: Path, step: int) -> Path:
    parent = run_root / "validation"
    parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        candidate = parent / f"step_{step:06d}_attempt_{attempt:02d}"
        if not candidate.exists():
            return candidate
        attempt += 1


def _write_best_weight_snapshot(run_root: Path, best: Mapping[str, Any]) -> None:
    """Atomically publish the currently selected real exported state dict."""
    source = Path(best["export_path"]) / "weights.pt"
    destination = run_root / "best.pt"
    tmp = run_root / f"best.pt.tmp.{uuid.uuid4().hex}"
    shutil.copyfile(source, tmp)
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, destination)
    if sha256_file(destination) != sha256_file(source):
        raise RuntimeError("atomic best.pt publication does not match the selected raw export")


def evaluate_checkpoint(
    checkpoint: str | Path,
    checkpoint_manifest_sha256: str,
    *,
    lock: Mapping[str, Any],
    run_root: str | Path,
    device: str,
    model: Any | None = None,
    optimizer: Any | None = None,
) -> dict[str, Any]:
    """Export/reload one raw checkpoint and score only its pinned VAL manifest."""
    run_root = Path(run_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    actual_checkpoint_sha = sha256_file(checkpoint / "MANIFEST.json")
    if actual_checkpoint_sha != checkpoint_manifest_sha256:
        raise ValueError("checkpoint manifest SHA does not match the observer's returned pin")
    manifest = json.loads((checkpoint / "MANIFEST.json").read_text())
    step = int(manifest.get("step", -1))
    checkpoint_config = dict(manifest.get("config") or {})
    checkpoint_data = checkpoint_config.pop("data", None)
    expected_config = dict(lock["config"])
    expected_data = expected_config.pop("data", None)
    if step < 1 or checkpoint_config != expected_config:
        raise ValueError("checkpoint step/config differs from the locked server protocol")
    if not isinstance(checkpoint_data, Mapping) or checkpoint_data.get("manifest_sha256") != lock["inputs"]["train"]["sha256"]:
        raise ValueError("checkpoint config does not bind the locked TRAIN manifest")
    if expected_data is not None and checkpoint_data != expected_data:
        raise ValueError("checkpoint TRAIN label lineage differs from the immutable arm config")
    entries = _read_validation_log(run_root)
    for entry in entries:
        if int(entry["step"]) == step and entry["checkpoint_manifest_sha256"] == checkpoint_manifest_sha256:
            return entry
    export_and_eval = _evaluate_checkpoint_payload
    context = preserve_training_state(model, optimizer) if model is not None and optimizer is not None else _null_context()
    with context:
        result = export_and_eval(
            checkpoint, checkpoint_manifest_sha256, lock=lock,
            validation_dir=_next_validation_dir(run_root, step), device=device,
        )
    result.update({
        "step": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "checkpoint_payload_sha256": sha256_file(checkpoint / "weights.pt"),
        "validation_manifest_sha256": lock["inputs"]["val"]["sha256"],
    })
    append_jsonl(run_root / "VALIDATION_LOG.jsonl", result)
    best = _select_best(_read_validation_log(run_root))
    _write_best_weight_snapshot(run_root, best)
    best_export_meta = json.loads((Path(best["export_path"]) / "MANIFEST.json").read_text())
    atomic_json(run_root / "BEST_SELECTION.json", {
        "schema": "self_audit.server_compact.best_selection.v1",
        "rule": lock["protocol"]["best_rule"],
        "scope": "development checkpoint selection only; no acceptance gate",
        "selected_step": int(best["step"]),
        "selected_macro4": float(best["macro4"]),
        "selected_checkpoint_manifest_sha256": best["checkpoint_manifest_sha256"],
        "selected_export_manifest_sha256": best["export_manifest_sha256"],
        "best_pt": str(run_root / "best.pt"),
        "best_pt_sha256": sha256_file(run_root / "best.pt"),
        "architecture": best_export_meta["architecture"],
        "preprocessing": best_export_meta["preprocessing"],
        "class_map": best_export_meta["class_map"],
        "validation_dir": best["validation_dir"],
        "predictions_frozen_path": str(Path(best["validation_dir"]) / "PREDICTIONS_FROZEN.json"),
        "predictions_frozen_sha256": best["predictions_frozen_sha256"],
    })
    return result


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _evaluate_checkpoint_payload(
    checkpoint: Path,
    checkpoint_manifest_sha256: str,
    *,
    lock: Mapping[str, Any],
    validation_dir: Path,
    device: str,
) -> dict[str, Any]:
    from .compact_artifacts import export_compact_student, load_compact_export
    from .compact_inference import run_two_phase_evaluation

    export_dir = validation_dir / "export"
    eval_dir = validation_dir / "val"
    export_sha = export_compact_student(checkpoint, checkpoint_manifest_sha256, export_dir)
    reloaded, export_meta = load_compact_export(export_dir, export_sha, device="cpu")
    if export_meta.get("source_checkpoint_manifest_sha256") != checkpoint_manifest_sha256:
        raise ValueError("raw export lineage does not bind to the selected full checkpoint")
    del reloaded
    inputs = lock["inputs"]
    metrics = run_two_phase_evaluation(
        export_path=export_dir,
        export_sha256=export_sha,
        manifest_path=inputs["val"]["path"],
        manifest_sha256=inputs["val"]["sha256"],
        train_manifest_path=inputs["train"]["path"],
        train_manifest_sha256=inputs["train"]["sha256"],
        out_dir=eval_dir,
        device=device,
        batch_size=int(lock["protocol"]["eval_batch_size"]),
        normalization=NORMALIZATION,
    )
    shared_summary = None
    if lock["config"].get("server_arm") == "joint_supervised":
        from .server_evaluation import shared_selection_summary
        shared_summary = shared_selection_summary(metrics)
        atomic_json(eval_dir / "DOMAIN_EVALUATION.json", shared_summary)
    macro4 = shared_summary["macro4"] if shared_summary else metrics.get("class_first_macro4")
    if type(macro4) not in (float, int) or not math.isfinite(macro4):
        raise ValueError("VAL did not return a finite class_first_macro4 for checkpoint selection")
    metrics_path = eval_dir / "METRICS.json"
    frozen_path = eval_dir / "PREDICTIONS_FROZEN.json"
    if not metrics_path.is_file() or not frozen_path.is_file():
        raise RuntimeError("canonical VAL evaluation did not persist metrics and frozen predictions")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("evaluation_manifest_sha256") != inputs["val"]["sha256"]:
        raise ValueError("frozen VAL predictions are not bound to the locked validation manifest")
    return {
        "schema": "self_audit.server_compact.validation.v1",
        "status": "VERIFIED",
        "macro4": float(macro4),
        "foreground_macro": shared_summary["foreground_macro"] if shared_summary else metrics.get("class_first_foreground_macro"),
        "domain_results": shared_summary["domain_results"] if shared_summary else metrics.get("datasets"),
        "synthetic": bool(inputs["synthetic"]),
        "export_path": str(export_dir),
        "export_manifest_sha256": export_sha,
        "validation_dir": str(eval_dir),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_frozen_sha256": sha256_file(frozen_path),
        "prediction_freeze_status": frozen.get("status"),
        "selection_metric": shared_summary["selection_metric"] if shared_summary else "class_first_macro4",
        "selection_note": "VAL is development-only; no go/no-go threshold was applied",
    }


class ProgressWatcher:
    """Read-only per-step tqdm progress with explicit labeled-patient coverage epochs."""

    def __init__(self, log_path: str | Path, *, total: int, start_step: int, labeled_ids: list[str], label: str = "S0"):
        from tqdm import tqdm
        self.progress_factory = tqdm
        self.log_path = Path(log_path)
        self.total = total
        self.start_step = start_step
        self.labeled_ids = set(labeled_ids)
        self.label = label
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="server-compact-progress", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()

    def _run(self) -> None:
        bar = self.progress_factory(total=self.total, initial=self.start_step, desc=self.label, unit="step", dynamic_ncols=True)
        offset = 0
        last_done = self.start_step
        completed_cycles = 0
        covered: set[str] = set()
        try:
            while True:
                if self.log_path.is_file():
                    with self.log_path.open("r", encoding="utf-8") as stream:
                        stream.seek(offset)
                        while True:
                            before = stream.tell()
                            line = stream.readline()
                            if not line or not line.endswith("\n"):
                                offset = before
                                break
                            offset = stream.tell()
                            row = json.loads(line)
                            done = int(row["step"]) + 1
                            schedule = row.get("data_schedule", {}).get("supervised", [])
                            if isinstance(schedule, Mapping):
                                schedule = [schedule]
                            for sample in schedule:
                                pid = str(sample.get("patient", ""))
                                if pid in self.labeled_ids:
                                    covered.add(pid)
                                    if covered >= self.labeled_ids:
                                        completed_cycles += 1
                                        covered.clear()
                            if done > last_done:
                                if done > self.start_step:
                                    bar.update(done - max(last_done, self.start_step))
                                last_done = done
                            fraction = len(covered) / len(self.labeled_ids) if self.labeled_ids else 0.0
                            bar.set_postfix({
                                "L-patient coverage epoch": f"{completed_cycles + fraction:.2f}",
                                "covered": f"{len(covered)}/{len(self.labeled_ids)}",
                            }, refresh=False)
                if self.stop_event.is_set():
                    break
                self.stop_event.wait(0.25)
        finally:
            bar.close()


def _install_checkpoint_observer(callback: Callable[..., Any]):
    from . import compact_training as trainer

    class Observer:
        def __enter__(self):
            self.original = trainer.save_compact_checkpoint

            def wrapped(*args, **kwargs):
                manifest_sha = self.original(*args, **kwargs)
                checkpoint = Path(args[4] if len(args) > 4 else kwargs["out"])
                callback(
                    model=args[0], optimizer=args[1], checkpoint=checkpoint,
                    checkpoint_manifest_sha256=manifest_sha, step=int(kwargs["step"]),
                )
                return manifest_sha

            trainer.save_compact_checkpoint = wrapped
            return self

        def __exit__(self, exc_type, exc, tb):
            trainer.save_compact_checkpoint = self.original
            return False

    return Observer()


def _write_best_artifacts(run_root: Path, lock: Mapping[str, Any], final_checkpoint: Path,
                          final_checkpoint_manifest_sha256: str) -> dict[str, Any]:
    from .compact_artifacts import load_compact_export

    entries = _read_validation_log(run_root)
    best = _select_best(entries)
    best_export_src = Path(best["export_path"])
    best_dir = run_root / "best_export"
    if best_dir.exists():
        existing_manifest = best_dir / "MANIFEST.json"
        if not existing_manifest.is_file() or sha256_file(existing_manifest) != best["export_manifest_sha256"]:
            prior_step = "unknown"
            best_path = run_root / "BEST.json"
            if best_path.is_file():
                try:
                    prior_step = str(json.loads(best_path.read_text()).get("step", "unknown"))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            archive = run_root / f"best_export_previous_step_{prior_step}_{uuid.uuid4().hex[:8]}"
            os.replace(best_dir, archive)
            temporary = run_root / f"best_export.tmp.{uuid.uuid4().hex}"
            shutil.copytree(best_export_src, temporary)
            os.replace(temporary, best_dir)
    else:
        shutil.copytree(best_export_src, best_dir)
    export_manifest_path = best_dir / "MANIFEST.json"
    if sha256_file(export_manifest_path) != best["export_manifest_sha256"]:
        raise ValueError("selected best export manifest changed before packaging")
    best_pt = run_root / "best.pt"
    tmp = run_root / f"best.pt.tmp.{uuid.uuid4().hex}"
    shutil.copyfile(best_dir / "weights.pt", tmp)
    with tmp.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, best_pt)
    if sha256_file(best_pt) != sha256_file(best_dir / "weights.pt"):
        raise ValueError("best.pt differs from selected raw export weights")
    reloaded, metadata = load_compact_export(best_dir, best["export_manifest_sha256"], device="cpu")
    del reloaded
    final_manifest = json.loads((final_checkpoint / "MANIFEST.json").read_text())
    if int(final_manifest.get("step", -1)) != TOTAL_STEPS:
        raise ValueError("full resume checkpoint is not the locked 10,000-step terminal checkpoint")
    full_resume = {
        "schema": "self_audit.server_compact.full_resume.v1",
        "checkpoint_path": str(final_checkpoint),
        "manifest_sha256": final_checkpoint_manifest_sha256,
        "payload_sha256": sha256_file(final_checkpoint / "weights.pt"),
        "step": TOTAL_STEPS,
        "config_sha256": canonical_sha256(lock["config"]),
        "train_manifest_sha256": lock["inputs"]["train"]["sha256"],
        "source_closure_sha256": lock["source"]["closure_sha256"],
    }
    atomic_json(run_root / "FULL_RESUME_CHECKPOINT.json", full_resume)
    best_receipt = {
        "schema": "self_audit.server_compact.best.v1",
        "status": "SELECTED",
        "selection_rule": "highest VAL class_first_macro4; exact ties keep earliest step",
        "selection_scope": "development-only model selection; no quality gate",
        "step": int(best["step"]),
        "val_macro4": float(best["macro4"]),
        "foreground_macro": best.get("foreground_macro"),
        "checkpoint_path": best["checkpoint_path"],
        "checkpoint_manifest_sha256": best["checkpoint_manifest_sha256"],
        "checkpoint_payload_sha256": best["checkpoint_payload_sha256"],
        "best_pt": str(best_pt),
        "best_pt_sha256": sha256_file(best_pt),
        "export_path": str(best_dir),
        "export_manifest_sha256": best["export_manifest_sha256"],
        "architecture": metadata["architecture"],
        "preprocessing": metadata["preprocessing"],
        "class_map": metadata["class_map"],
        "val_manifest_sha256": lock["inputs"]["val"]["sha256"],
        "val_metrics_sha256": best["metrics_sha256"],
        "validation_dir": best["validation_dir"],
        "predictions_frozen_path": str(Path(best["validation_dir"]) / "PREDICTIONS_FROZEN.json"),
        "predictions_frozen_sha256": best["predictions_frozen_sha256"],
        "full_resume_checkpoint": full_resume,
        "synthetic": bool(lock["inputs"]["synthetic"]),
    }
    atomic_json(run_root / "BEST.json", best_receipt)
    return best_receipt


def _latest_checkpoint(run_root: Path) -> tuple[Path, str, int] | None:
    candidates = []
    for manifest_path in (run_root / "segments").glob("segment_*/step_*/MANIFEST.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            candidates.append((int(manifest["step"]), manifest_path.parent, sha256_file(manifest_path)))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    for manifest_path in (run_root / "segments").glob("segment_*/final/MANIFEST.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
            candidates.append((int(manifest["step"]), manifest_path.parent, sha256_file(manifest_path)))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    if not candidates:
        return None
    step, path, sha = max(candidates, key=lambda item: (item[0], str(item[1])))
    return path, sha, step


def _resume_optimizer(model: Any, payload: Mapping[str, Any], config: Mapping[str, Any]):
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    optimizer.load_state_dict(payload["optimizer"])
    return optimizer


def _add_backup_tree(files: list[tuple[Path, str]], folder: Path, prefix: str,
                     *, reports_only: bool = False) -> None:
    if not folder.is_dir():
        raise FileNotFoundError(f"backup input folder missing: {folder}")
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"backup input cannot contain symlinks: {path}")
        if path.is_file() and (not reports_only or path.suffix.lower() in
                               {".json", ".jsonl", ".csv", ".txt", ".log", ".png", ".pdf"}):
            files.append((path, f"{prefix}/{path.relative_to(folder).as_posix()}"))


def _make_backup_files(run_root: Path, final_checkpoint: Path, phase: str) -> list[tuple[Path, str]]:
    if phase not in {"core", "post_validation", "post_test"}:
        raise ValueError(f"unknown backup phase: {phase}")
    files: list[tuple[Path, str]] = []
    if phase == "core":
        for prefix, folder in (("full_resume_checkpoint", final_checkpoint),
                               ("final_export", run_root / "final_export"),
                               ("best_export", run_root / "best_export")):
            _add_backup_tree(files, folder, prefix)
        names = (
            "best.pt", "BEST.json", "BEST_SELECTION.json", "FULL_RESUME_CHECKPOINT.json",
            "RUN_LOCK.json", "LOCKED_CONFIG.json", "LOCKED_PROFILE.json", "LOCKED_BASE_RECIPE.json",
            "PROVENANCE.json", "SHA256SUMS.txt",
        )
    else:
        names = (
            "RUN_LOCK.json", "LOCKED_CONFIG.json", "LOCKED_PROFILE.json", "LOCKED_BASE_RECIPE.json",
            "PROVENANCE.json", "TRAIN_LOG.jsonl", "REPORT.json", "VALIDATION_LOG.jsonl",
            "BEST_SELECTION.json", "best.pt", "BEST.json", "FULL_RESUME_CHECKPOINT.json",
            "SUPPLEMENTAL_EVALUATION_REPORT.json", "TEST_RESULT.json", "SHA256SUMS.txt",
        )
        # Completed validation and TEST products are copied with stable paths.
        for folder_name, prefix in (("validation", "validation"),):
            _add_backup_tree(files, run_root / folder_name, prefix, reports_only=True)
        if phase == "post_validation":
            _add_backup_tree(files, run_root / "best_export", "best_export")
        if phase == "post_test":
            for folder in sorted(run_root.glob("test_evaluation_attempt_*")):
                _add_backup_tree(files, folder, folder.name, reports_only=True)
    if (run_root / "inputs").is_dir():
        _add_backup_tree(files, run_root / "inputs", "inputs")
    for segment in sorted((run_root / "segments").glob("segment_*")) if (run_root / "segments").is_dir() else ():
        segment_names = ("TRAIN_LOG.jsonl", "config.json") if phase == "core" else (
            "TRAIN_LOG.jsonl", "REPORT.json", "config.json"
        )
        for name in segment_names:
            path = segment / name
            if path.is_file():
                files.append((path, f"training/{segment.name}/{name}"))
    for name in names:
        path = run_root / name
        if path.is_file():
            files.append((path, name))
        elif name in {"best.pt", "BEST.json", "FULL_RESUME_CHECKPOINT.json", "RUN_LOCK.json",
                      "LOCKED_CONFIG.json", "LOCKED_PROFILE.json", "LOCKED_BASE_RECIPE.json",
                      "PROVENANCE.json", "SHA256SUMS.txt"}:
            raise FileNotFoundError(f"backup metadata/weight missing: {path}")
    # Keep first occurrence when a directory inventory and root receipt overlap.
    deduplicated: dict[str, tuple[Path, str]] = {}
    for path, relative in files:
        deduplicated[relative] = (path, relative)
    if not deduplicated:
        raise FileNotFoundError(f"no files available for backup phase {phase}")
    return list(deduplicated.values())


def _write_sha256sums(run_root: Path, *, include_test: bool = False) -> dict[str, str]:
    candidates: dict[str, Path] = {}

    def add_file(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"checksum input cannot contain symlinks: {path}")
        if path.is_file():
            candidates[path.relative_to(run_root).as_posix()] = path

    def add_tree(folder: Path) -> None:
        if not folder.exists():
            return
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"checksum input must be a directory without symlinks: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"checksum input cannot contain symlinks: {path}")
            if path.is_file():
                add_file(path)

    root_names = (
        "RUN_LOCK.json", "LOCKED_CONFIG.json", "LOCKED_PROFILE.json", "LOCKED_BASE_RECIPE.json",
        "PROVENANCE.json", "VALIDATION_LOG.jsonl", "BEST_SELECTION.json", "BEST.json", "best.pt",
        "FULL_RESUME_CHECKPOINT.json", "SUPPLEMENTAL_EVALUATION_REPORT.json",
    )
    if include_test:
        root_names += ("TEST_RESULT.json",)
    for name in root_names:
        add_file(run_root / name)
    for segment in sorted((run_root / "segments").glob("segment_*")) if (run_root / "segments").is_dir() else ():
        for name in ("TRAIN_LOG.jsonl", "REPORT.json", "config.json"):
            add_file(segment / name)
    resume_receipt = run_root / "FULL_RESUME_CHECKPOINT.json"
    if resume_receipt.is_file():
        checkpoint = Path(json.loads(resume_receipt.read_text())["checkpoint_path"]).resolve()
        if not checkpoint.is_relative_to(run_root.resolve()):
            raise ValueError("terminal full checkpoint escapes the locked run directory")
        add_tree(checkpoint)
    for folder_name in ("final_export", "best_export", "validation", "inputs"):
        add_tree(run_root / folder_name)
    if include_test:
        for folder in sorted(run_root.glob("test_evaluation_attempt_*")):
            add_tree(folder)
    lines = [f"{sha256_file(path)}  {relative}" for relative, path in sorted(candidates.items())]
    target = run_root / "SHA256SUMS.txt"
    temporary = run_root / f"SHA256SUMS.txt.tmp.{uuid.uuid4().hex}"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return {relative: sha256_file(path) for relative, path in sorted(candidates.items())}


class SubprocessRclone:
    FLAGS = ("--retries", "1", "--low-level-retries", "2", "--timeout", "60s", "--contimeout", "15s",
             "--checkers", "1", "--transfers", "1")

    def __init__(self, log_path: str | Path, *, hard_timeout_seconds: float = RCLONE_HARD_TIMEOUT_SECONDS):
        if (type(hard_timeout_seconds) not in (int, float) or not math.isfinite(hard_timeout_seconds)
                or hard_timeout_seconds <= 0):
            raise ValueError("rclone hard timeout must be a finite positive number of seconds")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.hard_timeout_seconds = float(hard_timeout_seconds)

    def _log_operation(self, operation: str) -> None:
        # Do not persist argv, stdout, or stderr: any may contain remote config
        # material supplied by rclone or a credential-bearing remote string.
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"program": "rclone", "operation": operation,
                                  "time": time.time(), "hard_timeout_seconds": self.hard_timeout_seconds}) + "\n")
            log.flush()

    @staticmethod
    def _kill_process_group(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass

    def _deadline_timer(self, process: subprocess.Popen) -> tuple[threading.Event, threading.Timer]:
        expired = threading.Event()

        def expire() -> None:
            # Kill the process group even if rclone's parent exited while a
            # child still holds a captured pipe open.
            expired.set()
            self._kill_process_group(process)

        timer = threading.Timer(self.hard_timeout_seconds, expire)
        timer.daemon = True
        timer.start()
        return expired, timer

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        operation = str(argv[1]) if len(argv) > 1 else "unknown"
        self._log_operation(operation)
        process = subprocess.Popen(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=(os.name == "posix"))
        expired, timer = self._deadline_timer(process)
        try:
            stdout, _stderr = process.communicate()
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                self._kill_process_group(process)
                process.wait()
        if expired.is_set():
            raise TimeoutError(f"rclone {operation} exceeded hard deadline ({self.hard_timeout_seconds:g}s)")
        if process.returncode != 0:
            raise RuntimeError(f"rclone {operation} failed with exit code {process.returncode}; stderr omitted")
        return subprocess.CompletedProcess(argv, process.returncode, stdout, "")

    def ensure_fresh_destination(self, base: str, destination: str) -> None:
        # The caller's base must already exist and be listable. A random child is
        # selected and checked before mkdir, so no prior object is overwritten.
        raw = self._run(["rclone", "lsjson", base, *self.FLAGS]).stdout
        entries = json.loads(raw)
        child = destination.rsplit("/", 1)[-1]
        if any(item.get("Name") == child for item in entries):
            raise FileExistsError(f"fresh rclone destination already exists: {destination}")
        self._run(["rclone", "mkdir", destination, *self.FLAGS])
        raw_child = self._run(["rclone", "lsjson", destination, "--recursive", "--files-only", *self.FLAGS]).stdout
        if json.loads(raw_child):
            raise FileExistsError(f"fresh rclone destination is not empty: {destination}")

    def copyto(self, local_path: Path, remote_path: str) -> None:
        self._run(["rclone", "copyto", str(local_path), remote_path, *self.FLAGS])

    def readback_sha256(self, remote_path: str) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        argv = ["rclone", "cat", remote_path, *self.FLAGS]
        self._log_operation("cat")
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=(os.name == "posix"))
        expired, timer = self._deadline_timer(process)
        assert process.stderr is not None

        def drain_stderr() -> None:
            try:
                while process.stderr.read(64 << 10):
                    pass
            except (OSError, ValueError):
                pass

        stderr_thread = threading.Thread(target=drain_stderr, name="rclone-stderr-drain", daemon=True)
        stderr_thread.start()
        try:
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
            return_code = process.wait()
        finally:
            timer.cancel()
            timer.join()
            if process.poll() is None:
                self._kill_process_group(process)
                process.wait()
            stderr_thread.join(timeout=1.0)
            if process.stdout is not None:
                process.stdout.close()
            process.stderr.close()
        if expired.is_set():
            raise TimeoutError(f"rclone cat exceeded hard deadline ({self.hard_timeout_seconds:g}s)")
        if return_code != 0:
            raise RuntimeError(f"rclone cat failed with exit code {return_code}; stderr omitted")
        return size, digest.hexdigest()


def backup_artifacts(
    run_root: str | Path,
    *,
    final_checkpoint: str | Path,
    rclone_base: str,
    lock_sha256: str,
    client: Any | None = None,
    phase: str = "core",
    attempts: int = 3,
    retry_delay_seconds: float = 15.0,
    sleep_fn: Callable[[float], Any] = time.sleep,
) -> dict[str, Any]:
    """Copy one bounded backup phase and stream every remote file for SHA-256 verification."""
    if type(attempts) is not int or attempts < 1 or attempts > 5:
        raise ValueError("backup attempts must be between 1 and 5")
    if type(retry_delay_seconds) not in (int, float) or retry_delay_seconds < 0 or not math.isfinite(retry_delay_seconds):
        raise ValueError("retry delay must be finite and non-negative")
    run_root = Path(run_root).resolve()
    files = _make_backup_files(run_root, Path(final_checkpoint).resolve(), phase)
    if client is None:
        client = SubprocessRclone(run_root / "rclone.log")
    plan_path = run_root / ("BACKUP_PLAN.json" if phase == "core" else f"BACKUP_PLAN_{phase}.json")
    phase_status_path = run_root / f"BACKUP_STATUS_{phase}.json"
    status_path = run_root / "BACKUP_STATUS.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (plan.get("rclone_base") != rclone_base or plan.get("lock_sha256") != lock_sha256
                or plan.get("phase", "core") != phase):
            raise ValueError("pending backup plan identity differs; refusing to switch destinations")
        expected_plan = [{"local_path": str(path), "remote_path": rel,
                          "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path, rel in files]
        if plan.get("files") != expected_plan:
            raise ValueError("backup inputs changed after the first upload attempt")
        if phase_status_path.is_file():
            prior = json.loads(phase_status_path.read_text())
            if prior.get("status") == "VERIFIED":
                return prior
            if prior.get("status") != "UPLOAD_PENDING":
                raise ValueError("backup plan status does not permit a retry")
        destination = plan["destination"]
    else:
        if phase == "core":
            child = f"server_compact_{lock_sha256[:12]}_{uuid.uuid4().hex[:12]}"
            destination = rclone_base.rstrip("/") + "/" + child
        else:
            core_plan_path = run_root / "BACKUP_PLAN.json"
            core_status_path = run_root / "BACKUP_STATUS_core.json"
            if not core_plan_path.is_file() or not core_status_path.is_file():
                raise ValueError("supplemental backup requires a pinned core backup plan")
            core_status = json.loads(core_status_path.read_text())
            if core_status.get("status") != "VERIFIED":
                raise ValueError("supplemental backup requires a verified core checkpoint backup")
            core_plan = json.loads(core_plan_path.read_text())
            if core_plan.get("rclone_base") != rclone_base or core_plan.get("lock_sha256") != lock_sha256:
                raise ValueError("supplemental backup remote identity differs from core backup")
            destination = str(core_plan["destination"])
        plan = {
            "schema": "self_audit.server_compact.backup_plan.v1",
            "phase": phase,
            "rclone_base": rclone_base,
            "destination": destination,
            "lock_sha256": lock_sha256,
            "files": [{"local_path": str(path), "remote_path": rel,
                       "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path, rel in files],
        }
        if phase == "core":
            try:
                client.ensure_fresh_destination(rclone_base, destination)
            except Exception as exc:
                status = {
                    "schema": "self_audit.server_compact.backup.v1", "phase": phase,
                    "status": "UPLOAD_PENDING", "remote_verified": False,
                    "destination": destination, "lock_sha256": lock_sha256,
                    "files_verified": [], "pending": [{"path": "<destination-preflight>",
                        "error": f"{type(exc).__name__}: destination preflight failed", "attempts": 1}],
                    "local_retained": True, "verification": "not started; fresh destination could not be established",
                }
                atomic_json(phase_status_path, status)
                atomic_json(status_path, status)
                return status
        atomic_json(plan_path, plan)
    receipts = []
    pending = []
    for local_path, relative in files:
        expected_size, expected_sha = local_path.stat().st_size, sha256_file(local_path)
        remote_path = destination.rstrip("/") + "/" + relative
        verified = None
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                client.copyto(local_path, remote_path)
                remote_size, remote_sha = client.readback_sha256(remote_path)
                if remote_size != expected_size or remote_sha != expected_sha:
                    raise RuntimeError("remote bytes differ from the local SHA-256-pinned file")
                verified = {"path": relative, "bytes": expected_size, "sha256": expected_sha,
                            "remote_sha256": remote_sha, "remote_verified": True, "attempts": attempt}
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: backup operation failed; diagnostic output omitted"
                if attempt < attempts:
                    sleep_fn(retry_delay_seconds * attempt)
        if verified is None:
            pending.append({"path": relative, "error": last_error, "attempts": attempts})
        else:
            receipts.append(verified)
    status = {
        "schema": "self_audit.server_compact.backup.v1",
        "phase": phase,
        "status": "UPLOAD_PENDING" if pending else "VERIFIED",
        "remote_verified": not pending,
        "destination": destination,
        "lock_sha256": lock_sha256,
        "files_verified": receipts,
        "pending": pending,
        "local_retained": True,
        "verification": "streamed rclone cat SHA-256 and byte count per file",
    }
    atomic_json(phase_status_path, status)
    atomic_json(status_path, status)
    return status


def _write_run_status(run_root: Path, status: str, **extra: Any) -> dict[str, Any]:
    value = {"schema": "self_audit.server_compact.run_status.v1", "status": status,
             "updated_at_unix": time.time(), **extra}
    atomic_json(run_root / "RUN_STATUS.json", value)
    return value


def _copy_locked_run_receipts(run_root: Path, lock_path: Path, lock: Mapping[str, Any], lock_sha256: str) -> None:
    receipts = [
        (lock_path, run_root / "RUN_LOCK.json", lock_sha256),
        (Path(lock["profile_path"]), run_root / "LOCKED_PROFILE.json", lock["profile_sha256"]),
        (Path(lock["base_recipe_path"]), run_root / "LOCKED_BASE_RECIPE.json", lock["base_recipe_sha256"]),
    ]
    if lock.get("arm_config_path") is not None:
        receipts.append((
            Path(lock["arm_config_path"]),
            run_root / "LOCKED_ARM_CONFIG.json",
            lock["arm_config_sha256"],
        ))
    for source, destination, expected in receipts:
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"locked receipt changed or disappeared: {source}")
        if destination.exists():
            if sha256_file(destination) != expected:
                raise ValueError(f"existing run receipt differs from its protocol pin: {destination}")
        else:
            shutil.copyfile(source, destination)
    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for role in ("train", "val", "test"):
        reference = lock["inputs"].get(role)
        if reference is None:
            continue
        source = Path(reference["path"])
        destination = inputs_dir / f"{role}.json"
        if sha256_file(source) != reference["sha256"]:
            raise ValueError(f"locked {role} manifest bytes changed before backup packaging")
        if destination.exists():
            if sha256_file(destination) != reference["sha256"]:
                raise ValueError(f"packaged {role} manifest differs from the protocol pin")
        else:
            shutil.copyfile(source, destination)
    config_path = run_root / "LOCKED_CONFIG.json"
    if config_path.exists():
        if canonical_sha256(json.loads(config_path.read_text())) != lock["config_sha256"]:
            raise ValueError("existing locked run config differs from protocol pin")
    else:
        atomic_json(config_path, lock["config"])
    provenance = {
        "schema": "self_audit.server_compact.provenance.v1",
        "protocol_lock_sha256": lock_sha256,
        "source": lock["source"],
        "inputs": lock["inputs"],
        "config_sha256": lock["config_sha256"],
        "profile_sha256": lock["profile_sha256"],
        "base_recipe_sha256": lock["base_recipe_sha256"],
    }
    if lock.get("arm_config_path") is not None:
        provenance["arm_config_path"] = lock["arm_config_path"]
        provenance["arm_config_sha256"] = lock["arm_config_sha256"]
    provenance_path = run_root / "PROVENANCE.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("existing run provenance differs from the protocol lock")
    else:
        atomic_json(provenance_path, provenance)


def _export_terminal_checkpoint(checkpoint: Path, checkpoint_sha: str, run_root: Path) -> Path:
    from .compact_artifacts import export_compact_student, load_compact_export

    export_dir = run_root / "final_export"
    if export_dir.exists():
        manifest = export_dir / "MANIFEST.json"
        if not manifest.is_file():
            raise ValueError("existing final export is incomplete")
        metadata = json.loads(manifest.read_text())
        if metadata.get("source_checkpoint_manifest_sha256") != checkpoint_sha:
            raise ValueError("existing final export is not bound to the terminal checkpoint")
        load_compact_export(export_dir, sha256_file(manifest), device="cpu")
        return export_dir
    export_sha = export_compact_student(checkpoint, checkpoint_sha, export_dir)
    _model, metadata = load_compact_export(export_dir, export_sha, device="cpu")
    if metadata.get("source_checkpoint_manifest_sha256") != checkpoint_sha:
        raise ValueError("reloaded terminal raw export lost its checkpoint lineage")
    return export_dir


def _write_supplemental_evaluation_report(run_root: Path, lock: Mapping[str, Any], best: Mapping[str, Any],
                                          *, test_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validations = _read_validation_log(run_root)
    report = {
        "schema": "self_audit.server_compact.supplemental_evaluation.v1",
        "status": "TEST_SCORED" if test_metrics is not None else "VAL_SELECTION_COMPLETE",
        "synthetic": bool(lock["inputs"]["synthetic"]),
        "validation_manifest_sha256": lock["inputs"]["val"]["sha256"],
        "validation_checkpoint_count": len(validations),
        "selected_step": int(best["step"]),
        "selected_val_macro4": float(best["val_macro4"]),
        "selected_validation_dir": best["validation_dir"],
        "selected_frozen_predictions_sha256": best["predictions_frozen_sha256"],
        "test_manifest_sha256": lock["inputs"].get("test", {}).get("sha256") if lock["inputs"].get("test") else None,
        "test_metrics": dict(test_metrics) if test_metrics is not None else None,
        "test_masks_used_for_selection_or_gates": False,
        "claim_boundary": "development selection and optional post-freeze TEST only; no patient success is inferred from synthetic runs",
    }
    atomic_json(run_root / "SUPPLEMENTAL_EVALUATION_REPORT.json", report)
    return report


def _verify_best_val_freeze(run_root: Path, lock: Mapping[str, Any], best: Mapping[str, Any]) -> Path:
    validation_dir = Path(best.get("validation_dir", "")).resolve()
    frozen_path = Path(best.get("predictions_frozen_path", "")).resolve()
    expected_path = validation_dir / "PREDICTIONS_FROZEN.json"
    if frozen_path != expected_path or not frozen_path.is_file():
        raise ValueError("BEST receipt does not point to its exact frozen VAL prediction artifact")
    if not frozen_path.is_relative_to(run_root.resolve()):
        raise ValueError("frozen VAL prediction artifact escapes the locked run directory")
    if sha256_file(frozen_path) != best.get("predictions_frozen_sha256"):
        raise ValueError("best VAL prediction freeze hash changed before optional TEST")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("evaluation_manifest_sha256") != lock["inputs"]["val"]["sha256"]:
        raise ValueError("frozen VAL predictions are not bound to the locked VAL manifest")
    if frozen.get("export_manifest_sha256") != best.get("export_manifest_sha256"):
        raise ValueError("frozen VAL predictions are not bound to the selected best export")
    coverage = frozen.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        raise ValueError("best VAL frozen predictions do not declare complete patient coverage")
    return frozen_path


def _run_optional_test(run_root: Path, lock: Mapping[str, Any], best: Mapping[str, Any], *, device: str) -> dict[str, Any] | None:
    test_ref = lock["inputs"].get("test")
    if test_ref is None:
        return None
    frozen_path = _verify_best_val_freeze(run_root, lock, best)
    from .compact_inference import run_two_phase_evaluation

    prior_result_path = run_root / "TEST_RESULT.json"
    if prior_result_path.is_file():
        prior = json.loads(prior_result_path.read_text())
        prior_dir = Path(prior.get("test_evaluation_dir", "")).resolve()
        if (prior.get("test_manifest_sha256") != test_ref["sha256"]
                or prior.get("best_export_manifest_sha256") != best["export_manifest_sha256"]
                or prior.get("best_val_frozen_predictions_sha256") != sha256_file(frozen_path)
                or not prior_dir.is_relative_to(run_root.resolve())
                or not (prior_dir / "METRICS.json").is_file()
                or sha256_file(prior_dir / "METRICS.json") != prior.get("test_metrics_sha256")):
            raise ValueError("existing TEST result is not bound to the locked manifest and selected frozen best")
        return dict(prior.get("metrics") or {})

    test_dir = _next_test_dir(run_root)
    metrics = run_two_phase_evaluation(
        export_path=run_root / "best_export", export_sha256=best["export_manifest_sha256"],
        manifest_path=test_ref["path"], manifest_sha256=test_ref["sha256"],
        train_manifest_path=lock["inputs"]["train"]["path"],
        train_manifest_sha256=lock["inputs"]["train"]["sha256"],
        out_dir=test_dir, device=device, batch_size=EVALUATION_BATCH_SIZE, normalization=NORMALIZATION,
    )
    atomic_json(run_root / "TEST_RESULT.json", {
        "schema": "self_audit.server_compact.test_result.v1",
        "status": "SCORED_AFTER_BEST_FREEZE",
        "test_manifest_sha256": test_ref["sha256"],
        "best_export_manifest_sha256": best["export_manifest_sha256"],
        "best_val_frozen_predictions_path": str(frozen_path),
        "best_val_frozen_predictions_sha256": sha256_file(frozen_path),
        "test_evaluation_dir": str(test_dir),
        "test_metrics_sha256": sha256_file(test_dir / "METRICS.json"),
        "metrics": metrics,
        "test_masks_used_for_selection_or_gates": False,
    })
    return metrics


def _complete_after_core_backup(run_root: Path, lock: Mapping[str, Any], lock_sha256: str,
                                final_checkpoint: Path, final_checkpoint_sha: str, *, device: str,
                                backup_client: Any | None, retry_delay_seconds: float) -> dict[str, Any]:
    entries = _read_validation_log(run_root)
    if not any(int(row["step"]) == TOTAL_STEPS and
               row["checkpoint_manifest_sha256"] == final_checkpoint_sha for row in entries):
        import torch
        from .compact_artifacts import load_exact_compact_checkpoint
        model, payload, _meta = load_exact_compact_checkpoint(
            final_checkpoint, final_checkpoint_sha, device=device, expected_config=lock["config"]
        )
        optimizer = _resume_optimizer(model, payload, lock["config"])
        evaluate_checkpoint(final_checkpoint, final_checkpoint_sha, lock=lock, run_root=run_root,
                            device=device, model=model, optimizer=optimizer)
    best = _write_best_artifacts(run_root, lock, final_checkpoint, final_checkpoint_sha)
    _write_supplemental_evaluation_report(run_root, lock, best)
    _write_sha256sums(run_root)
    post_validation = backup_artifacts(
        run_root, final_checkpoint=final_checkpoint, rclone_base=str(json.loads((run_root / "BACKUP_PLAN.json").read_text())["rclone_base"]),
        lock_sha256=lock_sha256, client=backup_client, phase="post_validation",
        retry_delay_seconds=retry_delay_seconds,
    )
    if post_validation["status"] != "VERIFIED":
        return _write_run_status(
            run_root, "UPLOAD_PENDING", training_complete=True, final_checkpoint=str(final_checkpoint),
            final_checkpoint_manifest_sha256=final_checkpoint_sha, protocol_lock_sha256=lock_sha256,
            best_step=best["step"], best_macro4=best["val_macro4"], backup=post_validation,
            backup_phase="post_validation", test_status="NOT_RUN_BACKUP_PENDING",
        )
    return _complete_after_validation_backup(
        run_root, lock, lock_sha256, final_checkpoint, final_checkpoint_sha, best,
        device=device, backup_client=backup_client, retry_delay_seconds=retry_delay_seconds,
        post_validation=post_validation,
    )


def _complete_after_validation_backup(
    run_root: Path, lock: Mapping[str, Any], lock_sha256: str, final_checkpoint: Path,
    final_checkpoint_sha: str, best: Mapping[str, Any], *, device: str,
    backup_client: Any | None, retry_delay_seconds: float,
    post_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    test_metrics = _run_optional_test(run_root, lock, best, device=device)
    if test_metrics is None:
        return _write_run_status(
            run_root, "COMPLETED", training_complete=True, final_checkpoint=str(final_checkpoint),
            final_checkpoint_manifest_sha256=final_checkpoint_sha, best_step=best["step"],
            best_macro4=best["val_macro4"], backup=post_validation,
            test_status="NOT_REQUESTED", protocol_lock_sha256=lock_sha256,
        )
    _write_supplemental_evaluation_report(run_root, lock, best, test_metrics=test_metrics)
    _write_sha256sums(run_root, include_test=True)
    core_plan = json.loads((run_root / "BACKUP_PLAN.json").read_text())
    post_test = backup_artifacts(
        run_root, final_checkpoint=final_checkpoint, rclone_base=core_plan["rclone_base"],
        lock_sha256=lock_sha256, client=backup_client, phase="post_test",
        retry_delay_seconds=retry_delay_seconds,
    )
    if post_test["status"] != "VERIFIED":
        return _write_run_status(
            run_root, "UPLOAD_PENDING", training_complete=True, final_checkpoint=str(final_checkpoint),
            final_checkpoint_manifest_sha256=final_checkpoint_sha, protocol_lock_sha256=lock_sha256,
            best_step=best["step"], best_macro4=best["val_macro4"], backup=post_test,
            backup_phase="post_test", test_status="SCORED_POST_FREEZE_BACKUP_PENDING",
            test_macro4=test_metrics.get("class_first_macro4"),
        )
    return _write_run_status(
        run_root, "COMPLETED", training_complete=True, final_checkpoint=str(final_checkpoint),
        final_checkpoint_manifest_sha256=final_checkpoint_sha, best_step=best["step"],
        best_macro4=best["val_macro4"], backup=post_test, test_status="COMPLETED",
        test_macro4=test_metrics.get("class_first_macro4"), protocol_lock_sha256=lock_sha256,
    )


def run_locked_experiment(
    lock_path: str | Path,
    lock_sha256: str,
    out_dir: str | Path,
    *,
    device: str = "cuda",
    rclone_base: str,
    resume_checkpoint: str | Path | None = None,
    resume_sha256: str | None = None,
    arm_config_path: str | Path | None = None,
    arm_config_sha256: str | None = None,
    backup_client: Any | None = None,
    retry_delay_seconds: float = 15.0,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Train exactly one locked S0, select best on VAL, backup, then optional TEST."""
    lock = verify_protocol_lock(lock_path, lock_sha256, root)
    lock_sha256 = lock_sha256.lower()
    if lock.get("schema") == "self_audit.server_compact.lock.v2":
        if arm_config_path is None or arm_config_sha256 is None:
            raise ValueError("generic arm run requires --arm-config and --arm-config-sha256")
        if (
            Path(arm_config_path).resolve() != Path(lock["arm_config_path"]).resolve()
            or arm_config_sha256.lower() != lock["arm_config_sha256"]
        ):
            raise ValueError("run arm config path/SHA differs from the verified protocol lock")
    elif arm_config_path is not None or arm_config_sha256 is not None:
        raise ValueError("--arm-config is only supported with a generic v2 arm lock; S0 retains its existing CLI")
    run_root = Path(out_dir).resolve()
    if str(device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA run requested but CUDA is unavailable")
    was_new_run = not run_root.exists()
    if run_root.exists():
        pinned_lock = run_root / "RUN_LOCK.json"
        if not pinned_lock.is_file() or sha256_file(pinned_lock) != lock_sha256:
            raise ValueError("existing output belongs to a different protocol lock")
    else:
        if resume_checkpoint is not None or resume_sha256 is not None:
            raise ValueError("resume arguments require an existing locked output directory")
        run_root.mkdir(parents=True)
        shutil.copyfile(Path(lock_path).resolve(), run_root / "RUN_LOCK.json")
    _copy_locked_run_receipts(run_root, Path(lock_path).resolve(), lock, lock_sha256)
    if resume_checkpoint is None and resume_sha256 is not None:
        raise ValueError("resume SHA pin requires --resume-checkpoint")
    if resume_sha256 is not None:
        resume_sha256 = resume_sha256.lower()
    latest = _latest_checkpoint(run_root)
    if latest is None:
        if not was_new_run:
            raise ValueError("existing run has no resumable checkpoint; refusing a fresh replacement")
        starting_step = 0
    else:
        if resume_checkpoint is None or resume_sha256 is None:
            status = json.loads((run_root / "RUN_STATUS.json").read_text()) if (run_root / "RUN_STATUS.json").exists() else {}
            if not (latest[2] == TOTAL_STEPS and status.get("training_complete") is True):
                raise ValueError("existing training checkpoint requires explicit exact --resume-checkpoint and --resume-sha256")
        if resume_checkpoint is not None:
            chosen = Path(resume_checkpoint).resolve()
            if chosen != latest[0] or resume_sha256 != latest[1]:
                raise ValueError("resume must use the latest checkpoint path and its exact MANIFEST SHA-256")
        starting_step = latest[2]
    if latest is not None and latest[2] < TOTAL_STEPS:
        if resume_checkpoint is None or resume_sha256 is None:
            raise ValueError("exact resume requires latest checkpoint and mandatory manifest SHA pin")
    if (run_root / "RUN_STATUS.json").is_file():
        existing_status = json.loads((run_root / "RUN_STATUS.json").read_text())
        if existing_status.get("status") == "COMPLETED":
            return existing_status

    from .compact_training import train_compact
    train_ref = lock["inputs"]["train"]
    cfg = lock["config"]
    _manifest, labeled_ids, synthetic = _train_metadata(train_ref["path"], train_ref["sha256"])
    segments = run_root / "segments"
    segments.mkdir(exist_ok=True)
    segment_index = max([int(p.name.split("_")[-1]) for p in segments.glob("segment_*") if p.is_dir()] + [0]) + 1
    segment_out = segments / f"segment_{segment_index:04d}"
    final_checkpoint = None
    final_checkpoint_sha = None
    report = None
    callback_state: dict[str, Any] = {}

    if starting_step < TOTAL_STEPS:
        if segment_out.exists():
            raise FileExistsError(f"training segment destination already exists: {segment_out}")
        _write_run_status(run_root, "TRAINING", training_complete=False, segment=str(segment_out),
                          resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
                          protocol_lock_sha256=lock_sha256)
        watcher = ProgressWatcher(run_root / "segments" / "active" / "TRAIN_LOG.jsonl",
                                  total=TOTAL_STEPS, start_step=starting_step, labeled_ids=labeled_ids,
                                  label=cfg.get("server_arm", "S0"))
        # The actual per-segment log path is known before train_compact creates it.
        watcher.log_path = segment_out / "TRAIN_LOG.jsonl"
        callback_errors: list[BaseException] = []

        def on_checkpoint(model, optimizer, checkpoint, checkpoint_manifest_sha256, step):
            try:
                if step < TOTAL_STEPS:
                    evaluate_checkpoint(checkpoint, checkpoint_manifest_sha256, lock=lock, run_root=run_root,
                                        device=device, model=model, optimizer=optimizer)
                    return
                checkpoint = Path(checkpoint).resolve()
                _export_terminal_checkpoint(checkpoint, checkpoint_manifest_sha256, run_root)
                if not _read_validation_log(run_root):
                    raise ValueError("no earlier VAL checkpoint is available before core backup")
                _write_best_artifacts(run_root, lock, checkpoint, checkpoint_manifest_sha256)
                _write_sha256sums(run_root)
                _write_run_status(
                    run_root, "BACKUP", training_complete=False, final_checkpoint=str(checkpoint),
                    final_checkpoint_manifest_sha256=checkpoint_manifest_sha256,
                    protocol_lock_sha256=lock_sha256, backup_phase="core_before_final_val",
                )
                try:
                    core_backup = backup_artifacts(
                        run_root, final_checkpoint=checkpoint, rclone_base=rclone_base,
                        lock_sha256=lock_sha256, client=backup_client, phase="core",
                        retry_delay_seconds=retry_delay_seconds,
                    )
                except Exception as exc:
                    core_backup = {
                        "schema": "self_audit.server_compact.backup.v1", "phase": "core",
                        "status": "UPLOAD_PENDING", "remote_verified": False,
                        "local_retained": True,
                        "pending": [{"path": "<core-backup>",
                                      "error": f"{type(exc).__name__}: backup setup failed; details omitted",
                                      "attempts": 1}],
                    }
                    atomic_json(run_root / "BACKUP_STATUS_core.json", core_backup)
                    atomic_json(run_root / "BACKUP_STATUS.json", core_backup)
                callback_state["core_backup"] = core_backup
            except BaseException as exc:
                callback_errors.append(exc)
                raise

        watcher.start()
        try:
            with _install_checkpoint_observer(on_checkpoint):
                report = train_compact(
                    cfg, train_ref["path"], train_ref["sha256"], segment_out,
                    device=device, synthetic=synthetic, allow_full_run=True,
                    resume=resume_checkpoint, resume_sha256=resume_sha256,
                )
        except BaseException as exc:
            watcher.stop()
            latest_after_error = _latest_checkpoint(run_root)
            extras = {"training_complete": False, "error": f"{type(exc).__name__}: {exc}"}
            if latest_after_error is not None:
                extras.update({"resume_checkpoint": str(latest_after_error[0]),
                               "resume_manifest_sha256": latest_after_error[1], "resume_step": latest_after_error[2]})
            _write_run_status(run_root, "RESUME_AVAILABLE" if latest_after_error else "FAILED", **extras)
            raise
        finally:
            if watcher.thread.is_alive():
                watcher.stop()
        if callback_errors:
            raise callback_errors[0]
        final_checkpoint = Path(report["final_checkpoint_path"]).resolve()
        final_checkpoint_sha = report["final_manifest_sha256"]
        starting_step = int(report["steps"])
        if report.get("status") != "COMPLETED" or starting_step != TOTAL_STEPS:
            _write_run_status(run_root, "STOPPED", training_complete=False,
                              checkpoint=str(final_checkpoint), checkpoint_manifest_sha256=final_checkpoint_sha,
                              steps=starting_step)
            return json.loads((run_root / "RUN_STATUS.json").read_text())
    else:
        assert latest is not None
        final_checkpoint, final_checkpoint_sha, _ = latest

    if final_checkpoint is None or final_checkpoint_sha is None:
        latest = _latest_checkpoint(run_root)
        if latest is None or latest[2] != TOTAL_STEPS:
            raise RuntimeError("no locked terminal full resume checkpoint")
        final_checkpoint, final_checkpoint_sha, _ = latest
    _export_terminal_checkpoint(final_checkpoint, final_checkpoint_sha, run_root)
    if not (run_root / "BEST.json").is_file():
        _write_best_artifacts(run_root, lock, final_checkpoint, final_checkpoint_sha)
    core_backup = callback_state.get("core_backup")
    failed_in_terminal_callback = core_backup is not None and core_backup.get("status") != "VERIFIED"
    if core_backup is None:
        prior_core = run_root / "BACKUP_STATUS_core.json"
        core_backup = json.loads(prior_core.read_text()) if prior_core.is_file() else None
    if failed_in_terminal_callback:
        best = json.loads((run_root / "BEST.json").read_text())
        return _write_run_status(
            run_root, "UPLOAD_PENDING", training_complete=True, final_checkpoint=str(final_checkpoint),
            final_checkpoint_manifest_sha256=final_checkpoint_sha, best_step=best["step"],
            best_macro4=best["val_macro4"], backup=core_backup, backup_phase="core",
            test_status="NOT_RUN_BACKUP_PENDING", protocol_lock_sha256=lock_sha256,
        )
    if not isinstance(core_backup, Mapping) or core_backup.get("status") != "VERIFIED":
        _write_sha256sums(run_root)
        _write_run_status(run_root, "BACKUP", training_complete=True,
                          final_checkpoint=str(final_checkpoint), final_checkpoint_manifest_sha256=final_checkpoint_sha,
                          protocol_lock_sha256=lock_sha256, backup_phase="core")
        try:
            core_backup = backup_artifacts(
                run_root, final_checkpoint=final_checkpoint, rclone_base=rclone_base,
                lock_sha256=lock_sha256, client=backup_client, phase="core",
                retry_delay_seconds=retry_delay_seconds,
            )
        except Exception as exc:
            core_backup = {
                "schema": "self_audit.server_compact.backup.v1", "phase": "core",
                "status": "UPLOAD_PENDING", "remote_verified": False, "local_retained": True,
                "pending": [{"path": "<core-backup>",
                              "error": f"{type(exc).__name__}: backup setup failed; details omitted",
                              "attempts": 1}],
            }
            atomic_json(run_root / "BACKUP_STATUS_core.json", core_backup)
            atomic_json(run_root / "BACKUP_STATUS.json", core_backup)
    if core_backup["status"] != "VERIFIED":
        best = json.loads((run_root / "BEST.json").read_text())
        return _write_run_status(
            run_root, "UPLOAD_PENDING", training_complete=True, final_checkpoint=str(final_checkpoint),
            final_checkpoint_manifest_sha256=final_checkpoint_sha, best_step=best["step"],
            best_macro4=best["val_macro4"], backup=core_backup, backup_phase="core",
            test_status="NOT_RUN_BACKUP_PENDING", protocol_lock_sha256=lock_sha256,
        )
    return _complete_after_core_backup(
        run_root, lock, lock_sha256, final_checkpoint, final_checkpoint_sha,
        device=device, backup_client=backup_client, retry_delay_seconds=retry_delay_seconds,
    )


def retry_backup(run_root: str | Path, *, rclone_base: str, client: Any | None = None,
                 retry_delay_seconds: float = 15.0, device: str = "cuda") -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    status = json.loads((run_root / "RUN_STATUS.json").read_text())
    if status.get("status") != "UPLOAD_PENDING" or status.get("training_complete") is not True:
        raise ValueError("backup retry requires a completed training run with UPLOAD_PENDING")
    if not status.get("final_checkpoint") or not status.get("final_checkpoint_manifest_sha256"):
        raise ValueError("UPLOAD_PENDING status is missing the exact terminal checkpoint path and manifest pin")
    checkpoint = Path(status["final_checkpoint"]).resolve()
    lock_sha = status["protocol_lock_sha256"]
    if sha256_file(checkpoint / "MANIFEST.json") != status["final_checkpoint_manifest_sha256"]:
        raise ValueError("pending backup terminal checkpoint manifest changed")
    lock_path = run_root / "RUN_LOCK.json"
    lock = verify_protocol_lock(lock_path, lock_sha)
    phase = status.get("backup_phase") or status.get("backup", {}).get("phase", "core")
    result = backup_artifacts(run_root, final_checkpoint=checkpoint, rclone_base=rclone_base,
                              lock_sha256=lock_sha, client=client,
                              phase=phase, retry_delay_seconds=retry_delay_seconds)
    common = {
        "training_complete": True, "final_checkpoint": str(checkpoint),
        "final_checkpoint_manifest_sha256": status["final_checkpoint_manifest_sha256"],
        "best_step": status.get("best_step"), "best_macro4": status.get("best_macro4"),
        "protocol_lock_sha256": lock_sha,
    }
    if result["status"] != "VERIFIED":
        return _write_run_status(run_root, "UPLOAD_PENDING", **common, backup=result,
                                 backup_phase=phase, test_status=status.get("test_status", "NOT_RUN_BACKUP_PENDING"))
    if str(device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA retry requested but CUDA is unavailable")
    if phase == "core":
        return _complete_after_core_backup(
            run_root, lock, lock_sha, checkpoint, status["final_checkpoint_manifest_sha256"],
            device=device, backup_client=client, retry_delay_seconds=retry_delay_seconds,
        )
    if phase == "post_validation":
        best = json.loads((run_root / "BEST.json").read_text())
        return _complete_after_validation_backup(
            run_root, lock, lock_sha, checkpoint, status["final_checkpoint_manifest_sha256"], best,
            device=device, backup_client=client, retry_delay_seconds=retry_delay_seconds,
            post_validation=result,
        )
    if phase == "post_test":
        best = json.loads((run_root / "BEST.json").read_text())
        test = json.loads((run_root / "TEST_RESULT.json").read_text())
        return _write_run_status(
            run_root, "COMPLETED", **{**common, "best_step": best["step"],
                "best_macro4": best["val_macro4"]}, backup=result, test_status="COMPLETED",
            test_macro4=(test.get("metrics") or {}).get("class_first_macro4"),
        )
    raise ValueError(f"unknown pending backup phase: {phase}")


def _next_test_dir(run_root: Path) -> Path:
    attempt = 1
    while True:
        candidate = run_root / f"test_evaluation_attempt_{attempt:02d}"
        if not candidate.exists():
            return candidate
        attempt += 1
