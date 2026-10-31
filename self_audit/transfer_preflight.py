"""Rigorous real-asset, GPU occupancy, and immutable protocol preflight for scalar transfer utility.

Contracted in docs/CPTU_IMPLEMENTATION_CONTRACT.md and docs/TRANSFER_UTILITY_RUNBOOK.md.
Exported interfaces:
  - check_preflight
  - validate_locked_protocol
  - check_gpu_occupancy
  - check_active_training_jobs

Strict failclosed validation:
  - Refuses non-empty output directories (no writes, fresh output only).
  - Queries nvidia-smi without initializing a CUDA context.
  - Queries requested GPU UUID only, respects CUDA_VISIBLE_DEVICES, excludes own PID/PPID.
  - Failureclosed ps for active training jobs; never kills jobs.
  - Compares ALL locked budget, hyperparameters, selector parameters, architecture, L8 pins.
  - Validates exact L/U role, no U masks, budget K=8, and teacher/student label union BEFORE image/mask cache I/O.
  - Inspects physical weights payload via find_checkpoint_payload(t_dir, expected_sha).
  - All teacher identity read from hash-pinned cache / manifest, never invented or fallbacked.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .compact_artifacts import digest, compact_training_source_sha256
from .compact_teacher import (
    find_checkpoint_payload,
    preflight_teacher_lineage,
    validate_reproduction_receipt,
    validate_teacher_identity,
)

__all__ = [
    "PINNED_B8_TEACHER_MANIFEST_SHA256",
    "PINNED_L8_PATIENT_IDS",
    "PINNED_LABEL_BUDGET_K",
    "PINNED_RAW_PAYLOAD_SHA256",
    "PINNED_TEACHER_CACHE_SHA256",
    "PINNED_TRAIN_MANIFEST_SHA256",
    "PINNED_VAL18_PATIENT_COUNT",
    "PINNED_VAL_MANIFEST_SHA256",
    "check_active_training_jobs",
    "check_gpu_occupancy",
    "check_preflight",
    "derive_reproduction_protocol",
    "validate_locked_protocol",
]

# Immutable Protocol & Asset SHA-256 Pins
PINNED_TRAIN_MANIFEST_SHA256 = "e64f3b4009e7250632491ac634e15657692b58d38b97a0a2d975396865e80de4"
PINNED_B8_TEACHER_MANIFEST_SHA256 = "ae36d026ce26ca188627e6ea4699695e9eb99501312ce05d5324b81e44d3ddaa"
PINNED_RAW_PAYLOAD_SHA256 = "0ae593ecb6b6f9dcb88d0a41d4a77d7b3b3a3edd87396624c3648afb1281f1b6"
PINNED_TEACHER_CACHE_SHA256 = "783624211e760fbeb2c3a80ded7b8dacb9d1407f0dcc1e484010cda2c343897b"
PINNED_VAL_MANIFEST_SHA256 = "cac4426393ca11959f56df2876869e825443916cdec02c6d95ec8da3fac0ca50"

PINNED_L8_PATIENT_IDS: Tuple[str, ...] = (
    "patient007",
    "patient010",
    "patient011",
    "patient024",
    "patient026",
    "patient041",
    "patient077",
    "patient099",
)
PINNED_VAL18_PATIENT_COUNT = 18
PINNED_LABEL_BUDGET_K = 8

# Locked Protocol Constants
LOCKED_PROTOCOL_VERSION = "20261030"
LOCKED_STUDY_NAME = "scalar_transfer_utility_pilot"
LOCKED_TOTAL_STEPS = 2000
LOCKED_WARMUP_STEPS = 1000
LOCKED_EVAL_STEPS = (1000, 1500, 2000)
LOCKED_FINAL_STEP = 2000
LOCKED_SAVE_EVERY = 500
LOCKED_WATCHDOG_MAX_SECONDS = 7200
LOCKED_PROFILE_MAX_STEPS = 32

LOCKED_SEED = 42
LOCKED_LR = 0.0003
LOCKED_WEIGHT_DECAY = 0.0001
LOCKED_PSEUDO_LR = 0.001
LOCKED_TEMPERATURE = 1.0
LOCKED_COEFFICIENT = 1.0
LOCKED_PSEUDO_THRESHOLD = 0.7
LOCKED_RIDGE = 1e-06
LOCKED_RISK_BUDGET = 0.0
LOCKED_ABSENT_FP_BUDGET = 0.0
LOCKED_GUARD_TOLERANCE = 1e-06
LOCKED_MAX_TRIALS = 8
LOCKED_TRIAL_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)
LOCKED_BATCH_SIZE = 8
LOCKED_LABELED_BATCH_SIZE = 8
LOCKED_UNLABELED_BATCH_SIZE = 8
LOCKED_CROP_SIZE = 160

LOCKED_SELECTOR_DICE_TOLERANCE = 0.0
LOCKED_UTILITY_GAIN_TOLERANCE = 1e-07
LOCKED_UTILITY_TIE_TOLERANCE = 1e-07
LOCKED_PROBE_SEED = 42017
LOCKED_PROBE_BATCH_SIZE = 8

LOCKED_ARCHITECTURE_NAME = "CompactStudent"
LOCKED_ARCHITECTURE_WIDTHS = (16, 32, 64, 128)
LOCKED_IN_CHANNELS = 3
LOCKED_NUM_CLASSES = 4
LOCKED_INIT = "kaiming_normal"

LOCKED_SCREENING_THRESHOLD = 0.002


def _require_sha256(value: Any, name: str) -> str:
    """Validate 64-hex SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64 or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a valid 64-hex SHA-256 string, got {value!r}")
    return value.lower()


def _read_json(path: Union[Path, str]) -> Dict[str, Any]:
    """Read JSON file with strict parsing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text())


# ==============================================================================
# Protocol Verification
# ==============================================================================

def validate_locked_protocol(
    protocol: Mapping[str, Any],
    *,
    reproduction_receipt: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Exact typed lock; missing values, booleans and nonfinite numbers cannot pass."""
    repro_spec = None
    if reproduction_receipt is not None:
        try:
            repro_receipt_val = validate_reproduction_receipt(reproduction_receipt)
            repro_spec = {
                "source_manifest_sha256": repro_receipt_val["checkpoint_manifest_sha256"],
                "source_payload_sha256": repro_receipt_val["checkpoint_payload_sha256"],
                "state": "student",
                "lineage": dict(budget_k=8, exact_labeled_ids=list(PINNED_L8_PATIENT_IDS)),
                "reproduction": {
                    "schema": "self_audit.b8.reproduction.v1",
                    "source_commit": repro_receipt_val["source_commit"],
                    "source_sha256": repro_receipt_val["source_sha256"],
                    "raw_student_sha256": repro_receipt_val["raw_student_sha256"],
                    "checkpoint_payload_sha256": repro_receipt_val["checkpoint_payload_sha256"],
                    "quality_gate": repro_receipt_val["quality_gate"],
                    "Macro4": repro_receipt_val["metrics"]["test50"]["Macro4"],
                    "foreground": repro_receipt_val["metrics"]["test50"]["foreground"],
                },
            }
        except Exception as exc:
            return [f"invalid reproduction_receipt for locked protocol: {exc}"]
    elif isinstance(protocol.get("teacher"), Mapping) and "reproduction" in protocol["teacher"]:
        repro_block = protocol["teacher"]["reproduction"]
        if not isinstance(repro_block, Mapping) or repro_block.get("schema") != "self_audit.b8.reproduction.v1":
            return ["teacher.reproduction mismatch: schema must be 'self_audit.b8.reproduction.v1'"]
        if repro_block.get("quality_gate") != "GO" or repro_block.get("Macro4", 0) < 0.87:
            return ["teacher.reproduction mismatch: quality_gate must be 'GO' and Macro4 >= 0.87"]
        repro_spec = {
            "source_manifest_sha256": protocol["teacher"].get("source_manifest_sha256"),
            "source_payload_sha256": protocol["teacher"].get("source_payload_sha256"),
            "state": "student",
            "lineage": dict(budget_k=8, exact_labeled_ids=list(PINNED_L8_PATIENT_IDS)),
            "reproduction": repro_block,
        }

    if repro_spec is not None:
        teacher_expected = dict(
            source_manifest_sha256=repro_spec["source_manifest_sha256"],
            source_payload_sha256=repro_spec["source_payload_sha256"],
            state="student",
            lineage=dict(budget_k=8, exact_labeled_ids=list(PINNED_L8_PATIENT_IDS)),
            cache_sha256=protocol.get("teacher", {}).get("cache_sha256") if isinstance(protocol.get("teacher"), Mapping) else "",
            reproduction=repro_spec["reproduction"],
        )
    else:
        teacher_expected = dict(
            source_manifest_sha256=PINNED_B8_TEACHER_MANIFEST_SHA256,
            source_payload_sha256=PINNED_RAW_PAYLOAD_SHA256,
            state="student",
            lineage=dict(budget_k=8, exact_labeled_ids=list(PINNED_L8_PATIENT_IDS)),
            cache_sha256=PINNED_TEACHER_CACHE_SHA256,
        )

    expected: Dict[str, Any] = {
        "schema": "self_audit.compact.transfer_utility.protocol.v1",
        "protocol_version": "20261030", "study_name": "scalar_transfer_utility_pilot",
        "budget": dict(total_steps=2000, warmup_steps=1000, eval_steps=[1000,1500,2000],
            final_step=2000, save_every=500, watchdog_max_seconds=7200, profile_max_steps=32),
        "hyperparameters": dict(seed=42, lr=3e-4, weight_decay=1e-4, pseudo_lr=1e-3,
            temperature=1., coefficient=1., pseudo_threshold=.7, ridge=1e-6,
            risk_budget=0., absent_fp_budget=0., guard_tolerance=1e-6, max_trials=8,
            trial_scales=list(LOCKED_TRIAL_SCALES), batch_size=8, labeled_batch_size=8,
            unlabeled_batch_size=8, crop_size=160),
        "selector_parameters": dict(selector_dice_tolerance=0., utility_gain_tolerance=1e-7,
            utility_tie_tolerance=1e-7, probe_seed=42017, probe_batch_size=8),
        "architecture": dict(name="CompactStudent", widths=[16,32,64,128], in_channels=3,
            num_classes=4, init="kaiming_normal"),
        "data": dict(manifest_sha256=PINNED_TRAIN_MANIFEST_SHA256, budget_k=8,
            labeled_ids=list(PINNED_L8_PATIENT_IDS)),
        "teacher": teacher_expected,
        "evaluation": dict(manifest_sha256=PINNED_VAL_MANIFEST_SHA256, expected_patients=18,
            checkpoint_selection="final_step_2000", primary_metric="class_first_macro4"),
        "decision_rules": dict(screening_threshold_delta_macro4=.002),
        "arms": {
            "arm1": dict(name="plain_kd", method="plain_kd", selector=None),
            "arm2": dict(name="scalar_guard_original", method="scalar_guard", selector="original"),
            "arm3": dict(name="scalar_guard_larger_tie", method="scalar_guard", selector="larger_tie"),
            "arm4": dict(name="scalar_guard_cross_patient_utility", method="scalar_guard", selector="cross_patient_utility")},
        "conditional_arms": {"arm5": dict(name="scalar_guard_same_batch_control", method="scalar_guard",selector="same_batch_utility")}}
    if "runtime_policy" in protocol:
        rp = protocol["runtime_policy"]
        if isinstance(rp, Mapping) and isinstance(rp.get("deterministic_algorithms"), bool):
            expected["runtime_policy"] = dict(deterministic_algorithms=rp["deterministic_algorithms"])
    errors = []
    def compare(actual, required, path):
        if isinstance(required, dict):
            if not isinstance(actual, Mapping):
                errors.append(f"{path} mismatch: mapping required")
                return
            for key, val in required.items():
                if key not in actual:
                    errors.append(f"{path + '.' if path else ''}{key} mismatch: missing")
                else:
                    compare(actual[key], val, f"{path + '.' if path else ''}{key}")
            if path in ("arms", "conditional_arms") and set(actual) != set(required):
                errors.append(f"{path} mismatch: extra experiments prohibited")
        elif isinstance(required, list):
            if type(actual) is not list or len(actual) != len(required) or any(
                type(a) is bool or a != b for a,b in zip(actual, required)):
                errors.append(f"{path} mismatch: expected {required}")
        elif isinstance(required, (int,float)) and not isinstance(required,bool):
            allowed = (int,) if type(required) is int else (int,float)
            if type(actual) not in allowed or actual != required:
                errors.append(f"{path} mismatch: expected {required}, got {actual!r}")
        elif type(actual) is not type(required) or actual != required:
            errors.append(f"{path} mismatch: expected {required!r}")
    compare(protocol, expected, "")
    if repro_spec is not None:
        c_sha = protocol.get("teacher", {}).get("cache_sha256")
        if not isinstance(c_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", c_sha) is None:
            errors.append("teacher.cache_sha256 must be a valid 64-hex SHA-256 string")
    return errors


def derive_reproduction_protocol(
    base_protocol: Mapping[str, Any],
    reproduction_receipt: Mapping[str, Any],
    teacher_cache_sha256: str,
    *,
    deterministic_algorithms: bool = True,
) -> Dict[str, Any]:
    """Derive an immutable protocol bound to the reproduction B8 teacher identity.

    Preserves all locked pilot hyperparameters, budgets, architecture, L8 pins,
    evaluation cohorts, screening thresholds, and arm definitions.
    ONLY updates the teacher section to bind the reproduction checkpoint,
    full student weights, teacher cache hash, and adds runtime_policy.
    """
    base_errors = validate_locked_protocol(base_protocol)
    if base_errors:
        raise ValueError(f"base protocol is not a valid locked protocol: {base_errors}")

    receipt = validate_reproduction_receipt(reproduction_receipt)
    cache_sha = _require_sha256(teacher_cache_sha256, "teacher_cache_sha256")

    derived = json.loads(json.dumps(base_protocol))

    derived["runtime_policy"] = {
        "deterministic_algorithms": bool(deterministic_algorithms),
    }

    derived["teacher"] = {
        "source_manifest_sha256": receipt["checkpoint_manifest_sha256"],
        "source_payload_sha256": receipt["checkpoint_payload_sha256"],
        "state": "student",
        "lineage": {
            "budget_k": 8,
            "exact_labeled_ids": list(PINNED_L8_PATIENT_IDS),
        },
        "cache_sha256": cache_sha,
        "reproduction": {
            "schema": "self_audit.b8.reproduction.v1",
            "source_commit": receipt["source_commit"],
            "source_sha256": receipt["source_sha256"],
            "raw_student_sha256": receipt["raw_student_sha256"],
            "checkpoint_payload_sha256": receipt["checkpoint_payload_sha256"],
            "quality_gate": receipt["quality_gate"],
            "Macro4": receipt["metrics"]["test50"]["Macro4"],
            "foreground": receipt["metrics"]["test50"]["foreground"],
        },
    }
    if "source_training_cost" in receipt:
        derived["teacher"]["source_training_cost"] = receipt["source_training_cost"]
    else:
        derived["teacher"]["source_training_cost"] = "NOT_KNOWN"
    if "parent_receipt_sha256" in receipt:
        derived["teacher"]["parent_receipt_sha256"] = receipt["parent_receipt_sha256"]

    derived_errors = validate_locked_protocol(derived, reproduction_receipt=receipt)
    if derived_errors:
        raise ValueError(f"derived reproduction protocol failed validation: {derived_errors}")

    return derived


# ==============================================================================
# GPU Occupancy & Active Jobs (Read-Only)
# ==============================================================================

def check_gpu_occupancy(device: str = "cuda") -> Dict[str, Any]:
    """Read-only GPU occupancy check querying nvidia-smi before any CUDA initialization.

    Queries the requested GPU index or UUID only, handles CUDA_VISIBLE_DEVICES,
    and excludes the caller's own PID/PPID. Never kills jobs.
    """
    dev_str = str(device).strip()
    if not dev_str.startswith("cuda"):
        return {
            "status": "PASS",
            "device": dev_str,
            "gpu_available": False,
            "is_busy": False,
            "message": "Non-CUDA device selected; occupancy check bypassed.",
        }

    # Parse requested local CUDA index (default 0)
    requested_idx = 0
    if ":" in dev_str:
        try:
            requested_idx = int(dev_str.split(":")[1])
        except ValueError:
            requested_idx = 0

    # Resolve physical target through CUDA_VISIBLE_DEVICES if set
    target_spec: Optional[str] = None
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_vis:
        entries = [e.strip() for e in cuda_vis.split(",") if e.strip()]
        if not entries or requested_idx >= len(entries):
            return {
                "status": "UNVERIFIABLE",
                "device": dev_str,
                "gpu_available": False,
                "is_busy": True,
                "error": f"Requested device index {requested_idx} out of range for CUDA_VISIBLE_DEVICES={cuda_vis!r}",
            }
        target_spec = entries[requested_idx]
    else:
        target_spec = str(requested_idx)

    own_pids = {os.getpid(), os.getppid()}

    try:
        # First, resolve target GPU UUID
        id_arg = f"--id={target_spec}"
        uuid_cmd = ["nvidia-smi", id_arg, "--query-gpu=uuid,index,name", "--format=csv,noheader,nounits"]
        uuid_res = subprocess.run(uuid_cmd, capture_output=True, text=True, timeout=5)
        if uuid_res.returncode != 0:
            return {
                "status": "UNVERIFIABLE",
                "device": dev_str,
                "gpu_available": False,
                "is_busy": True,
                "error": f"nvidia-smi UUID query failed: {uuid_res.stderr.strip()}",
            }

        uuid_lines = [l.strip() for l in uuid_res.stdout.strip().splitlines() if l.strip()]
        if not uuid_lines:
            return {
                "status": "UNVERIFIABLE",
                "device": dev_str,
                "gpu_available": False,
                "is_busy": True,
                "error": "nvidia-smi returned empty GPU output",
            }

        target_uuid, phys_idx_str, gpu_name = [p.strip() for p in uuid_lines[0].split(",")]
        phys_idx = int(phys_idx_str)

        # 1. Query memory usage of the target UUID only
        smi_mem = subprocess.run(
            ["nvidia-smi", f"--id={target_uuid}", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if smi_mem.returncode != 0:
            return {
                "status": "UNVERIFIABLE",
                "device": dev_str,
                "gpu_available": False,
                "is_busy": True,
                "error": f"nvidia-smi memory query failed: {smi_mem.stderr.strip()}",
            }

        mem_lines = [l.strip() for l in smi_mem.stdout.strip().splitlines() if l.strip()]
        if not mem_lines:
            return {
                "status": "UNVERIFIABLE",
                "device": dev_str,
                "gpu_available": False,
                "is_busy": True,
                "error": "nvidia-smi returned empty memory output",
            }

        used_s, total_s, free_s = [p.strip() for p in mem_lines[0].split(",")]
        used_mb = float(used_s)
        total_mb = float(total_s)
        free_mb = float(free_s)
        fraction_used = used_mb / max(total_mb, 1.0)

        # 2. Query compute processes on requested UUID only
        smi_apps = subprocess.run(
            ["nvidia-smi", f"--id={target_uuid}", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if smi_apps.returncode != 0:
            return {"status":"UNVERIFIABLE", "gpu_available":True, "is_busy":True,
                    "error":"nvidia-smi compute-process query failed"}
        active_processes: List[Dict[str, Any]] = []
        if smi_apps.returncode == 0:
            for row in smi_apps.stdout.strip().splitlines():
                if row.strip():
                    parts = [p.strip() for p in row.split(",")]
                    try:
                        proc_pid = int(parts[0])
                    except ValueError:
                        continue
                    if proc_pid in own_pids:
                        continue
                    active_processes.append({
                        "pid": proc_pid,
                        "name": parts[1] if len(parts) > 1 else "",
                        "used_memory_mb": parts[2] if len(parts) > 2 else "",
                    })

        is_busy = used_mb > 500.0 or fraction_used > 0.10 or len(active_processes) > 0

        return {
            "status": "BUSY" if is_busy else "PASS",
            "device": dev_str,
            "device_index": requested_idx,
            "physical_index": phys_idx,
            "uuid": target_uuid,
            "name": gpu_name,
            "total_mb": total_mb,
            "used_mb": used_mb,
            "free_mb": free_mb,
            "fraction_used": round(fraction_used, 4),
            "active_processes": active_processes,
            "is_busy": is_busy,
            "gpu_available": True,
        }
    except FileNotFoundError:
        return {
            "status": "UNVERIFIABLE",
            "device": dev_str,
            "gpu_available": False,
            "is_busy": True,
            "error": "nvidia-smi not found on system; GPU occupancy cannot be verified (refusing failclosed)",
        }
    except Exception as exc:
        return {
            "status": "UNVERIFIABLE",
            "device": dev_str,
            "gpu_available": False,
            "is_busy": True,
            "error": f"nvidia-smi occupancy check error: {exc}",
        }


def check_active_training_jobs() -> Dict[str, Any]:
    """Read-only check for active training jobs in the workspace. Failureclosed."""
    active_jobs: List[Dict[str, Any]] = []
    own_pids = {os.getpid(), os.getppid()}

    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode != 0:
            return {
                "status": "UNVERIFIABLE",
                "active_job_count": 0,
                "active_jobs": [],
                "error": f"ps returned non-zero code {res.returncode}; failclosed",
            }

        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                pid_str, cmd = parts
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid in own_pids:
                    continue
                is_training = (
                    ("train.py" in cmd or "run_transfer_utility_pilot.py" in cmd or "run_automatic_pilot.py" in cmd)
                    and ("python" in cmd or "rtk" in cmd)
                )
                if is_training:
                    active_jobs.append({"pid": pid, "command": cmd[:160]})

        is_clean = len(active_jobs) == 0
        return {
            "status": "CLEAN" if is_clean else "ACTIVE_JOBS_PRESENT",
            "active_job_count": len(active_jobs),
            "active_jobs": active_jobs,
        }
    except Exception as exc:
        return {
            "status": "UNVERIFIABLE",
            "active_job_count": 0,
            "active_jobs": [],
            "error": f"ps check failed with exception: {exc}; failclosed",
        }


# ==============================================================================
# Comprehensive Transfer Preflight
# ==============================================================================

def _verify_physical_teacher_and_cache(
    manifest_path,
    payload_path,
    cache_path,
    cache_sha,
    identity,
    train_path,
    train_sha,
    receipt,
    *,
    reproduction_receipt=None,
    reproduction_source_dir=None,
):
    """Read pinned teacher/quality/cache assets only after the label-union gate."""
    from .compact_data import CompactTrainingData
    from .compact_teacher import FixedTeacherCache, HISTORICAL_SOURCE_CLOSURE, HISTORICAL_SOURCE_COMMIT
    import torch

    manifest = _read_json(manifest_path)
    if reproduction_receipt is not None:
        repro = validate_reproduction_receipt(reproduction_receipt)
        if (
            manifest.get("schema") not in ("self_audit.checkpoint.v2", "self_audit.b8.raw_student.v1")
            or manifest.get("source_sha256") != repro["source_sha256"]
            or identity.get("source_sha256") != repro["source_sha256"]
            or identity.get("source_commit") != repro["source_commit"]
            or manifest.get("synthetic", False)
        ):
            raise ValueError("physical reproduction B8 teacher source/config/state identity mismatch")
        if reproduction_source_dir is not None:
            from .compact_teacher import verify_reproduction_source
            verify_reproduction_source(
                reproduction_source_dir,
                expected_commit=repro["source_commit"],
                expected_closure=repro["source_sha256"],
            )
        macro4 = repro["metrics"]["test50"]["Macro4"]
        if macro4 < 0.87 or repro["quality_gate"] != "GO":
            raise ValueError("reproduction teacher quality gate failed: Macro4 < 0.87")
        quality_sha = repro.get("receipt_file_sha256", "reproduction_receipt_verified")
    else:
        if (
            manifest.get("schema") != "self_audit.checkpoint.v2"
            or manifest.get("source_sha256") != HISTORICAL_SOURCE_CLOSURE
            or identity.get("source_sha256") != HISTORICAL_SOURCE_CLOSURE
            or identity.get("source_commit") != HISTORICAL_SOURCE_COMMIT
            or manifest.get("config", {}).get("method") != "ema_full"
            or manifest.get("synthetic", False)
        ):
            raise ValueError("physical B8 teacher source/config/state identity mismatch")
        quality = identity["quality_receipt"]
        quality_path = Path(quality.get("path", ""))
        quality_sha = _require_sha256(quality.get("sha256"), "quality receipt")
        if not quality_path.is_file() or digest(quality_path) != quality_sha:
            raise ValueError("physical quality receipt missing or hash mismatch")
        record = _read_json(quality_path)
        if (
            record.get("checkpoint_manifest_sha256") != PINNED_B8_TEACHER_MANIFEST_SHA256
            or record.get("inference_state") != "student"
            or type(record.get("class_first_macro4")) not in (int, float)
            or not 0 <= record["class_first_macro4"] <= 1
        ):
            raise ValueError("quality receipt does not bind the raw B8 checkpoint")

    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "student" in payload:
        state = payload["student"]
    elif isinstance(payload, Mapping):
        state = payload
    else:
        state = None
    if not isinstance(state, Mapping) or not state:
        raise ValueError("physical teacher payload lacks raw student state")
    if any(not torch.isfinite(value).all() for value in state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("nonfinite physical teacher raw student weights")
    del payload, state
    data = CompactTrainingData(train_path, train_sha, synthetic=False, allow_k1=False)
    try:
        cache = FixedTeacherCache(cache_path, cache_sha, data, identity)
        cache.close()
    finally:
        data.close()
    receipt["checks"]["quality_receipt"] = {
        "status": "PASS",
        "sha256": quality_sha,
        "scope": "previously observed receipt only; no new TEST/VAL scoring",
    }
    receipt["checks"]["cache_payloads"] = {"status": "PASS", "manifest_sha256": cache_sha}


def check_preflight(
    protocol: Mapping[str, Any],
    data_manifest: Optional[Union[str, Path]],
    data_manifest_sha256: Optional[str],
    out_dir: Union[str, Path],
    *,
    device: str = "cuda",
    synthetic: bool = False,
    check_jobs: bool = True,
    teacher_checkpoint: Optional[Union[str, Path]] = None,
    teacher_checkpoint_sha256: Optional[str] = None,
    teacher_payload_sha256: Optional[str] = None,
    teacher_cache: Optional[Union[str, Path]] = None,
    teacher_cache_sha256: Optional[str] = None,
    val_manifest: Optional[Union[str, Path]] = None,
    val_manifest_sha256: Optional[str] = None,
    reproduction_receipt: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    reproduction_receipt_sha256: Optional[str] = None,
    reproduction_source_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute rigorous preflight validation for scalar transfer-utility study.

    Returns JSON receipt with status:
      PASS | NOT_RUN | INVALID_PROTOCOL | REFUSED_EXISTING_OUTPUT | REFUSED_GPU_BUSY | REFUSED_ACTIVE_JOBS
    """
    receipt: Dict[str, Any] = {
        "schema": "self_audit.compact.transfer_utility.preflight.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "synthetic": bool(synthetic),
        "device": str(device),
        "source_receipts": {
            "compact_training_source_sha256": compact_training_source_sha256(),
            "transfer_preflight_sha256": digest(Path(__file__).resolve()),
        },
        "out_dir": str(out_dir),
        "checks": {},
        "status": "PASS",
        "reasons": [],
    }

    # 1. Output directory safety: strict fresh output, reject if anything exists. NO writes.
    out_p = Path(out_dir).resolve()
    if out_p.exists():
        if out_p.is_file():
            receipt["checks"]["output_directory"] = {"status": "REFUSED_EXISTING", "path": str(out_p)}
            receipt["reasons"].append(f"Output path exists as a regular file: {out_p}")
            receipt["status"] = "REFUSED_EXISTING_OUTPUT"
            return receipt
        else:
            receipt["checks"]["output_directory"] = {"status": "REFUSED_EXISTING", "path": str(out_p)}
            receipt["reasons"].append(f"Output directory exists and contains or may own run state: {out_p}")
            receipt["status"] = "REFUSED_EXISTING_OUTPUT"
            return receipt
    receipt["checks"]["output_directory"] = {"status": "PASS", "path": str(out_p)}

    # 2. Active jobs check (read-only failureclosed)
    if check_jobs:
        jobs_chk = check_active_training_jobs()
        receipt["checks"]["active_jobs"] = jobs_chk
        if jobs_chk["status"] != "CLEAN":
            receipt["reasons"].append(
                f"Active training jobs check refused: {jobs_chk.get('status')} "
                f"({jobs_chk.get('active_job_count', 0)} active job(s) detected: {jobs_chk.get('error', '')})"
            )
            receipt["status"] = "REFUSED_ACTIVE_JOBS"
            return receipt
    else:
        receipt["checks"]["active_jobs"] = {"status": "SKIPPED"}

    # 3. GPU occupancy check (requested UUID only, handle CUDA_VISIBLE_DEVICES, failclosed)
    if not synthetic and str(device).startswith("cuda"):
        gpu_chk = check_gpu_occupancy(device)
        receipt["checks"]["gpu_occupancy"] = gpu_chk
        if gpu_chk["status"] != "PASS":
            receipt["reasons"].append(
                f"GPU {device} occupancy check refused: {gpu_chk.get('error') or 'GPU is busy or unverifiable'}"
            )
            receipt["status"] = "REFUSED_GPU_BUSY"
            return receipt
    else:
        receipt["checks"]["gpu_occupancy"] = {"status": "SKIPPED", "synthetic": bool(synthetic), "device": str(device)}

    # Validate reproduction receipt if provided
    validated_repro = None
    if reproduction_receipt is not None:
        try:
            validated_repro = validate_reproduction_receipt(
                reproduction_receipt,
                expected_receipt_sha256=reproduction_receipt_sha256,
            )
            receipt["checks"]["reproduction_receipt"] = {
                "status": "PASS",
                "schema": validated_repro["schema"],
                "quality_gate": validated_repro["quality_gate"],
                "Macro4": validated_repro["metrics"]["test50"]["Macro4"],
                "remote_verified": True,
            }
        except Exception as exc:
            receipt["reasons"].append(f"Reproduction receipt validation failed: {exc}")
            receipt["status"] = "NOT_RUN"
            return receipt

    # 4. Strict protocol lock validation
    proto_errors = validate_locked_protocol(protocol, reproduction_receipt=validated_repro)
    if proto_errors:
        receipt["checks"]["protocol"] = {"status": "INVALID_PROTOCOL", "errors": proto_errors}
        receipt["reasons"].extend(proto_errors)
        receipt["status"] = "INVALID_PROTOCOL"
        return receipt
    receipt["checks"]["protocol"] = {"status": "PASS"}

    # 5. Asset and Lineage Validation (Metadata & disjoint checks BEFORE array / cache I/O)
    if not synthetic:
        # 5a. Physical TRAIN manifest
        if not data_manifest or not data_manifest_sha256:
            receipt["reasons"].append("Real patient training requires --data and --data-sha256; no implicit data discovery.")
            receipt["status"] = "NOT_RUN"
            return receipt

        data_p = Path(data_manifest).resolve()
        if not data_p.is_file():
            receipt["reasons"].append(f"Data manifest file not found: {data_p}")
            receipt["status"] = "NOT_RUN"
            return receipt

        actual_dsha = digest(data_p)
        if actual_dsha.lower() != data_manifest_sha256.lower():
            receipt["reasons"].append(f"Data manifest SHA-256 mismatch: expected {data_manifest_sha256}, got {actual_dsha}")
            receipt["status"] = "NOT_RUN"
            return receipt

        if actual_dsha.lower() != PINNED_TRAIN_MANIFEST_SHA256.lower():
            receipt["reasons"].append(f"Data manifest SHA-256 does not match immutable protocol pin {PINNED_TRAIN_MANIFEST_SHA256}")
            receipt["status"] = "NOT_RUN"
            return receipt

        # Validate TRAIN manifest metadata: roles, L/U separation, no U masks
        try:
            train_data = _read_json(data_p)
        except Exception as exc:
            receipt["reasons"].append(f"Failed to parse TRAIN manifest JSON: {exc}")
            receipt["status"] = "NOT_RUN"
            return receipt

        if train_data.get("role") != "training" or train_data.get("schema") != "self_audit.data.v2":
            receipt["reasons"].append("TRAIN manifest must declare role='training' and schema='self_audit.data.v2'")
            receipt["status"] = "NOT_RUN"
            return receipt

        if train_data.get("label_budget") != PINNED_LABEL_BUDGET_K:
            receipt["reasons"].append(f"TRAIN manifest label_budget {train_data.get('label_budget')} != {PINNED_LABEL_BUDGET_K}")
            receipt["status"] = "NOT_RUN"
            return receipt

        labeled_records = train_data.get("labeled", [])
        unlabeled_records = train_data.get("unlabeled", [])
        train_lids = [r["id"] for r in labeled_records if isinstance(r, dict) and "id" in r]
        train_uids = [r["id"] for r in unlabeled_records if isinstance(r, dict) and "id" in r]

        if tuple(sorted(train_lids)) != tuple(sorted(PINNED_L8_PATIENT_IDS)):
            receipt["reasons"].append(f"TRAIN labeled patient IDs mismatch: expected {list(PINNED_L8_PATIENT_IDS)}, got {train_lids}")
            receipt["status"] = "NOT_RUN"
            return receipt

        if len(unlabeled_records) != 72:
            receipt["reasons"].append(f"TRAIN unlabeled patient count {len(unlabeled_records)} != 72")
            receipt["status"] = "NOT_RUN"
            return receipt

        if set(train_lids) & set(train_uids):
            receipt["reasons"].append("TRAIN manifest labeled and unlabeled patient sets overlap")
            receipt["status"] = "NOT_RUN"
            return receipt

        # Strict: unlabeled records must contain image only, NO masks
        for u_rec in unlabeled_records:
            if not isinstance(u_rec, dict) or "masks" in u_rec or set(u_rec.keys()) != {"id", "image", "image_sha256"}:
                receipt["reasons"].append(f"Unlabeled record {u_rec.get('id')} contains mask or forbidden fields; strict no U masks")
                receipt["status"] = "NOT_RUN"
                return receipt

        for l_rec in labeled_records:
            if not isinstance(l_rec, dict) or set(l_rec.keys()) != {"id", "image", "image_sha256", "masks"} or not l_rec.get("masks"):
                receipt["reasons"].append(f"Labeled record {l_rec.get('id')} invalid or lacks masks")
                receipt["status"] = "NOT_RUN"
                return receipt

        # No VAL masks are read; exact development cohort identity is mandatory.
        if val_manifest is None or val_manifest_sha256 != PINNED_VAL_MANIFEST_SHA256:
            receipt["status"] = "NOT_RUN"
            receipt["reasons"].append("Pinned VAL18 manifest and SHA are mandatory")
            return receipt
        if val_manifest is not None:
            val_p = Path(val_manifest).resolve()
            if not val_p.is_file():
                receipt["reasons"].append(f"Validation manifest file not found: {val_p}")
                receipt["status"] = "NOT_RUN"
                return receipt

            actual_vsha = digest(val_p)
            expected_vsha = val_manifest_sha256 or PINNED_VAL_MANIFEST_SHA256
            if actual_vsha.lower() != expected_vsha.lower():
                receipt["reasons"].append(f"Validation manifest SHA-256 mismatch: expected {expected_vsha}, got {actual_vsha}")
                receipt["status"] = "NOT_RUN"
                return receipt

            try:
                val_data = _read_json(val_p)
            except Exception as exc:
                receipt["reasons"].append(f"Failed to parse VAL manifest JSON: {exc}")
                receipt["status"] = "NOT_RUN"
                return receipt

            val_role = val_data.get("role")
            if val_role != "development" or val_data.get("schema") not in ("self_audit.data.v2", "self_audit.native_manifest.v1"):
                receipt["reasons"].append(f"Validation manifest role mismatch: got {val_role!r}")
                receipt["status"] = "NOT_RUN"
                return receipt

            val_patients = val_data.get("patients", [])
            val_ids = [p["id"] if isinstance(p, dict) else str(p) for p in val_patients]
            if len(val_ids) != PINNED_VAL18_PATIENT_COUNT or len(set(val_ids)) != len(val_ids):
                receipt["reasons"].append(f"Validation manifest patient count {len(val_ids)} != {PINNED_VAL18_PATIENT_COUNT}")
                receipt["status"] = "NOT_RUN"
                return receipt

            # Disjoint check: VAL patients strictly disjoint from TRAIN patients
            overlap = set(val_ids) & set(train_lids + train_uids)
            if overlap:
                receipt["reasons"].append(f"VAL/TRAIN patient overlap detected: {sorted(overlap)}")
                receipt["status"] = "NOT_RUN"
                return receipt
            # NOTE: "noVALmasksread" enforced: we do not open/read any VAL mask files.

        # 5c. Physical Teacher Checkpoint Validation
        t_checkpoint = teacher_checkpoint or protocol.get("teacher", {}).get("source")
        if validated_repro is not None:
            t_manifest_sha = (
                teacher_checkpoint_sha256
                or protocol.get("teacher", {}).get("source_manifest_sha256")
                or validated_repro["checkpoint_manifest_sha256"]
            )
            t_payload_sha = (
                teacher_payload_sha256
                or protocol.get("teacher", {}).get("source_payload_sha256")
                or validated_repro["raw_student_sha256"]
            )
        else:
            t_manifest_sha = teacher_checkpoint_sha256 or protocol.get("teacher", {}).get("source_manifest_sha256") or PINNED_B8_TEACHER_MANIFEST_SHA256
            t_payload_sha = teacher_payload_sha256 or protocol.get("teacher", {}).get("source_payload_sha256") or PINNED_RAW_PAYLOAD_SHA256

        if not t_checkpoint:
            receipt["reasons"].append("Protocol missing teacher checkpoint path; execution blocked.")
            receipt["status"] = "NOT_RUN"
            return receipt

        t_dir = Path(t_checkpoint).resolve()
        if not t_dir.is_dir():
            receipt["reasons"].append(f"Physical teacher checkpoint directory not found on disk: {t_dir}")
            receipt["status"] = "NOT_RUN"
            return receipt

        t_manifest_file = t_dir / "MANIFEST.json"
        if not t_manifest_file.is_file():
            if validated_repro is not None:
                receipt["reasons"].append(f"Full checkpoint MANIFEST.json not found in reproduction checkpoint dir: {t_dir}")
                receipt["status"] = "NOT_RUN"
                return receipt
            t_manifest_file = t_dir / "RAW_STUDENT.json"
        if not t_manifest_file.is_file():
            receipt["reasons"].append(f"Teacher manifest not found in checkpoint dir: {t_dir}")
            receipt["status"] = "NOT_RUN"
            return receipt

        actual_t_msha = digest(t_manifest_file)
        if actual_t_msha.lower() != t_manifest_sha.lower():
            receipt["reasons"].append(f"Teacher manifest SHA mismatch: expected {t_manifest_sha}, got {actual_t_msha}")
            receipt["status"] = "NOT_RUN"
            return receipt

        if validated_repro is None:
            if actual_t_msha.lower() != PINNED_B8_TEACHER_MANIFEST_SHA256.lower():
                receipt["reasons"].append(f"Teacher manifest SHA does not match immutable B8 pin {PINNED_B8_TEACHER_MANIFEST_SHA256}")
                receipt["status"] = "NOT_RUN"
                return receipt

            # Validate physical payload weights exist and match expected SHA
            # find_checkpoint_payload takes positional expected SHA
            if t_payload_sha != PINNED_RAW_PAYLOAD_SHA256:
                receipt["status"] = "NOT_RUN"
                receipt["reasons"].append("teacher payload differs from pinned raw B8 payload")
                return receipt
        else:
            if actual_t_msha.lower() != validated_repro["checkpoint_manifest_sha256"].lower():
                receipt["reasons"].append(f"Teacher manifest SHA does not match reproduction receipt {validated_repro['checkpoint_manifest_sha256']}")
                receipt["status"] = "NOT_RUN"
                return receipt
            expected_repro_psha = validated_repro["checkpoint_payload_sha256"].lower()
            if t_payload_sha.lower() != expected_repro_psha:
                receipt["status"] = "NOT_RUN"
                receipt["reasons"].append(f"reproduction teacher payload {t_payload_sha} must equal full checkpoint payload {expected_repro_psha}")
                return receipt

        payload_p: Optional[Path] = None
        try:
            payload_p = find_checkpoint_payload(t_dir, t_payload_sha)
            receipt["checks"]["teacher_payload"] = {"status": "PASS", "path": str(payload_p)}
        except FileNotFoundError as fnf:
            receipt["reasons"].append(f"Physical teacher weights payload validation failed: {fnf}")
            receipt["status"] = "NOT_RUN"
            return receipt

        # 5d. Teacher Cache Validation & Strict Identity
        t_cache = teacher_cache or protocol.get("teacher", {}).get("cache_path")
        if validated_repro is not None:
            t_cache_sha = teacher_cache_sha256 or protocol.get("teacher", {}).get("cache_sha256")
        else:
            t_cache_sha = teacher_cache_sha256 or protocol.get("teacher", {}).get("cache_sha256") or PINNED_TEACHER_CACHE_SHA256

        if t_cache is None:
            receipt["reasons"].append("Teacher cache directory is required for transfer utility distillation pilot.")
            receipt["status"] = "NOT_RUN"
            return receipt
        if not t_cache_sha:
            receipt["reasons"].append("Teacher cache SHA256 is required.")
            receipt["status"] = "NOT_RUN"
            return receipt

        c_dir = Path(t_cache).resolve()
        c_manifest_file = c_dir / "CACHE_MANIFEST.json" if c_dir.is_dir() else c_dir
        if not c_manifest_file.is_file():
            receipt["reasons"].append(f"Physical teacher cache directory or CACHE_MANIFEST.json not found: {c_dir}")
            receipt["status"] = "NOT_RUN"
            return receipt

        actual_c_sha = digest(c_manifest_file)
        if actual_c_sha.lower() != t_cache_sha.lower():
            receipt["reasons"].append(f"Teacher cache manifest SHA mismatch: expected {t_cache_sha}, got {actual_c_sha}")
            receipt["status"] = "NOT_RUN"
            return receipt

        if validated_repro is None:
            if actual_c_sha.lower() != PINNED_TEACHER_CACHE_SHA256.lower():
                receipt["reasons"].append(f"Teacher cache manifest SHA does not match immutable pin {PINNED_TEACHER_CACHE_SHA256}")
                receipt["status"] = "NOT_RUN"
                return receipt

        # Read exact teacher identity from pinned cache manifest; NEVER invent or fallback
        try:
            cache_manifest = _read_json(c_manifest_file)
            teacher_identity = cache_manifest.get("teacher_identity")
            if not teacher_identity or not isinstance(teacher_identity, Mapping):
                receipt["reasons"].append("CACHE_MANIFEST.json lacks valid 'teacher_identity' mapping")
                receipt["status"] = "NOT_RUN"
                return receipt

            # Validate teacher identity kind
            expected_kind = "reproduction" if validated_repro is not None else "canonical"
            if teacher_identity.get("teacher_kind") != expected_kind:
                receipt["reasons"].append(f"Teacher identity kind mismatch: expected '{expected_kind}', got {teacher_identity.get('teacher_kind')!r}")
                receipt["status"] = "NOT_RUN"
                return receipt

            if teacher_identity.get("inference_state") != "student":
                receipt["reasons"].append(f"Teacher identity inference_state mismatch: expected 'student', got {teacher_identity.get('inference_state')!r}")
                receipt["status"] = "NOT_RUN"
                return receipt

            if teacher_identity.get("training_method") != "ema_full":
                receipt["reasons"].append(f"Teacher identity training_method mismatch: expected 'ema_full', got {teacher_identity.get('training_method')!r}")
                receipt["status"] = "NOT_RUN"
                return receipt

            # Validate lineage and quality receipt with existing APIs
            ident_receipt = validate_teacher_identity(
                teacher_identity,
                declared_k=PINNED_LABEL_BUDGET_K,
                declared_labeled_ids=PINNED_L8_PATIENT_IDS,
                manifest_path=t_manifest_file,
                payload_path=payload_p,
            )
            receipt["checks"]["teacher_identity"] = ident_receipt

            lineage_receipt = preflight_teacher_lineage(
                teacher_labeled_ids=ident_receipt["labeled_ids"],
                audit_labeled_ids=train_lids,
                student_labeled_ids=PINNED_L8_PATIENT_IDS,
                label_budget=PINNED_LABEL_BUDGET_K,
                declared_labeled_ids=PINNED_L8_PATIENT_IDS,
                teacher_kind=expected_kind,
            )
            receipt["checks"]["teacher_lineage"] = lineage_receipt
            _verify_physical_teacher_and_cache(
                t_manifest_file,
                payload_p,
                c_manifest_file,
                actual_c_sha,
                ident_receipt,
                data_p,
                actual_dsha,
                receipt,
                reproduction_receipt=validated_repro,
                reproduction_source_dir=reproduction_source_dir,
            )

        except Exception as exc:
            receipt["reasons"].append(f"Teacher identity or lineage preflight failed: {exc}")
            receipt["status"] = "NOT_RUN"
            return receipt

        receipt["checks"]["assets"] = {"status": "PASS"}

    else:
        # Synthetic unit mode allows skipping physical patient assets and GPU
        receipt["checks"]["assets"] = {
            "status": "PASS",
            "synthetic": True,
            "disclaimer": "Synthetic unit mode: valid for test scaffolding only, does not certify scientific or physical asset PASS.",
        }

    receipt["status"] = "PASS"
    return receipt
