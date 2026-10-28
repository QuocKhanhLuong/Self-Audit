"""Engine integration for self_audit.automatic.v1.

Implements the locked contract from docs/AUTOMATIC_PILOT_CONTRACT.md:
- Schema: self_audit.automatic.v1
- Phase contract: warm_start (support-only exit) vs adaptation (requires SHA-pinned warm_start checkpoint).
- NativeStudent fixed-window features frozen during head adaptation.
- Complete fixed TRAIN-L support volumes (all Z slices), never random audit slices.
- Strict checkpoint loading with target label lineage accounting and mandatory manifest-only expected_sha256.
- Strict raw-student load helper load_automatic_student for evaluation.
- Shared support-only warm-start and 10 named arms:
  scalar_brier, joint_brier, scalar_dice, joint_dice, frozen, support_only,
  plain_head_u, extra_supervised, smaller_lr, full_ema.
- Per-arm teacher, atomic controller commit/rollback, and EMA once per step.
- Exact bit-for-bit resume continuation with local np_rng.bit_generator.state restoration.
- Enforced time budget with CUDA synchronization and latency/VRAM accounting.
- Output directory NEW only with pre-mkdir validation against actual manifest/data/checkpoints.
- No checkpoint overwrites.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .automatic_control import (
    compute_margins,
    controlled_head_step,
    make_candidates,
)
from .automatic_data import (
    SupportFeatureCache,
    SupportVolumeRecord,
    compute_model_feature_fingerprint,
    compute_support_feature_cache,
    extract_complete_support_volumes,
)
from .data import TrainingData, write_json
from .engine import ema_update, full_pseudo_step
from .losses import strong_intensity, supervised_loss
from .model import build_model, digest
from .window import WINDOW_DEFAULT, validate_window_config

SCHEMA = "self_audit.automatic.v1"
CHECKPOINT_SCHEMA = "self_audit.automatic_checkpoint.v1"

NAMED_ARMS = {
    "scalar_brier",
    "joint_brier",
    "scalar_dice",
    "joint_dice",
    "frozen",
    "support_only",
    "plain_head_u",
    "extra_supervised",
    "smaller_lr",
    "full_ema",
}

PHASES = {"warm_start", "adaptation"}

DEFAULT_AUTOMATIC_CONFIG: dict[str, Any] = {
    "schema": SCHEMA,
    "phase": "adaptation",
    "arm": "joint_dice",
    "steps": 32,
    "warmup_steps": 0,
    "seed": 42,
    "head_lr": 0.003,
    "body_lr": 0.0,
    "pseudo_lr": 0.001,
    "weight_decay": 0.0001,
    "ema_decay": 0.99,
    "pseudo_threshold": 0.7,
    "ridge": 1e-06,
    "risk_budget": 0.0,
    "guard_tolerance": 1e-08,
    "dice_tolerance": 0.0,
    "fp_tolerance": 0.0,
    "backtracks": 8,
    "save_every": 100,
    "threads": 2,
    "batch_size": 1,
    "crop_size": 256,
    "max_support_voxels": 10_000_000,
    "max_extract_batch": 8,
    "time_budget_seconds": None,
    "max_cache_bytes": 2 * 1024 * 1024 * 1024,
    "max_controller_workspace_bytes": 4 * 1024 * 1024 * 1024,
    "support": {
        "patient_ids": None,
        "k_reuse": 1,
    },
    "initialization": {
        "warm_start_checkpoint": None,
        "warm_start_sha256": None,
        "declared_target_budget": 1,
    },
    "window": {
        "mode": "fixed",
        "dim": 32,
        "radius_min": 0.5,
        "radius_max": 4.0,
        "offset_limit": 2.0,
        "initial_radius": 1.0,
        "gamma_init": 0.001,
    },
    "model": {
        "backbone": "dinov3_convnext_tiny",
        "source_repo": "",
        "source_commit": "",
        "weights": "",
        "weights_sha256": "",
    },
}


def automatic_source_digest() -> str:
    """Stable SHA-256 digest of executed source files including engine/control dependencies."""
    root = Path(__file__).resolve().parent.parent
    canonical = [
        root / "train.py",
        root / "self_audit" / "__init__.py",
        root / "self_audit" / "model.py",
        root / "self_audit" / "window.py",
        root / "self_audit" / "losses.py",
        root / "self_audit" / "data.py",
        root / "self_audit" / "engine.py",
        root / "self_audit" / "control.py",
        root / "self_audit" / "automatic.py",
        root / "self_audit" / "automatic_data.py",
        root / "self_audit" / "automatic_control.py",
    ]
    h = hashlib.sha256()
    for p in sorted(canonical, key=lambda x: x.name):
        if p.is_file():
            h.update(p.name.encode("ascii"))
            h.update(digest(p).encode("ascii"))
    return h.hexdigest()


def validate_automatic_config(
    c: dict[str, Any],
    *,
    allow_full_run: bool = False,
    stop_steps: int | None = None,
) -> dict[str, Any]:
    """Validate a self_audit.automatic.v1 configuration dictionary strictly against schema."""
    if not isinstance(c, dict):
        raise ValueError("config must be a dictionary")
    c = copy.deepcopy(c)

    if c.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA}, got {c.get('schema')}")

    # Check for unknown keys at top level (strict schema adherence)
    allowed_top_keys = set(DEFAULT_AUTOMATIC_CONFIG.keys())
    extra_top = set(c.keys()) - allowed_top_keys
    if extra_top:
        raise ValueError(f"unknown config keys: {sorted(extra_top)}")

    # Check for unknown keys in nested dictionaries
    for section in ("support", "initialization", "window", "model"):
        if section in c:
            if not isinstance(c[section], dict):
                raise ValueError(f"{section} must be a dictionary")
            allowed_sec = set(DEFAULT_AUTOMATIC_CONFIG[section].keys())
            extra_sec = set(c[section].keys()) - allowed_sec
            if extra_sec:
                raise ValueError(f"unknown keys in {section}: {sorted(extra_sec)}")

    arm = c.get("arm")
    if arm not in NAMED_ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {sorted(NAMED_ARMS)}")

    # Default phase inference for backward compatibility if omitted
    if "phase" not in c:
        if c.get("arm") == "support_only" and c.get("initialization", {}).get("warm_start_checkpoint") is None:
            c["phase"] = "warm_start"
        else:
            c["phase"] = "adaptation"

    # Populate defaults for missing keys
    for k, v in DEFAULT_AUTOMATIC_CONFIG.items():
        if k not in c:
            c[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(c[k], dict):
            for sub_k, sub_v in v.items():
                if sub_k not in c[k]:
                    c[k][sub_k] = copy.deepcopy(sub_v)

    # Validate phase
    phase = c.get("phase")
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase!r}, expected one of {sorted(PHASES)}")

    if phase == "warm_start":
        if c["arm"] != "support_only":
            raise ValueError(f"phase 'warm_start' requires arm='support_only', got {c['arm']!r}")
        if c["initialization"].get("warm_start_checkpoint") is not None:
            raise ValueError("phase 'warm_start' cannot load an existing warm_start_checkpoint")

    # Validate time_budget_seconds
    tb = c.get("time_budget_seconds")
    if tb is not None:
        if isinstance(tb, bool) or not isinstance(tb, (int, float)) or not np.isfinite(tb) or tb <= 0:
            raise ValueError(f"time_budget_seconds must be a positive finite number, got {tb!r}")
        c["time_budget_seconds"] = float(tb)

    # Validate max_cache_bytes
    mcb = c.get("max_cache_bytes")
    if mcb is None:
        c["max_cache_bytes"] = mcb = DEFAULT_AUTOMATIC_CONFIG["max_cache_bytes"]
    if mcb is not None:
        if type(mcb) is not int or mcb <= 0:
            raise ValueError(f"max_cache_bytes must be a positive integer, got {mcb!r}")

    # Validate max_controller_workspace_bytes
    mcwb = c.get("max_controller_workspace_bytes")
    if mcwb is None:
        c["max_controller_workspace_bytes"] = mcwb = DEFAULT_AUTOMATIC_CONFIG["max_controller_workspace_bytes"]
    if mcwb is not None:
        if type(mcwb) is not int or mcwb <= 0:
            raise ValueError(f"max_controller_workspace_bytes must be a positive integer, got {mcwb!r}")

    # Validate integer fields
    for k in ("steps", "warmup_steps", "seed", "backtracks", "save_every", "threads",
              "batch_size", "crop_size", "max_support_voxels", "max_extract_batch"):
        val = c.get(k)
        if type(val) is not int or val < (0 if k in ("warmup_steps", "crop_size") else 1):
            raise ValueError(f"invalid {k}: {val}")

    if stop_steps is not None:
        if type(stop_steps) is not int or stop_steps < 1:
            raise ValueError("stop_steps must be a positive integer")
        c["steps"] = stop_steps

    if c["warmup_steps"] > c["steps"] and phase != "warm_start":
        raise ValueError(f"warmup_steps ({c['warmup_steps']}) exceeds total steps ({c['steps']})")

    # Bounded steps guard: > 32 steps requires allow_full_run unless synthetic
    if c["steps"] > 32 and not allow_full_run and c["model"].get("backbone") != "tiny_test":
        raise ValueError(f"steps ({c['steps']}) > 32 requires --allow-full-run for patient data")

    # Validate float fields
    for k in ("head_lr", "pseudo_lr", "weight_decay", "ema_decay", "pseudo_threshold",
              "ridge", "risk_budget", "guard_tolerance", "dice_tolerance", "fp_tolerance"):
        val = c.get(k)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
            raise ValueError(f"invalid {k}: {val}")

    if c["head_lr"] <= 0 or c["ridge"] <= 0 or not 0 <= c["ema_decay"] < 1 or c["pseudo_threshold"] > 1:
        raise ValueError("invalid optimization range in config")

    # Body LR check: body_lr > 0 strictly required for full_ema, prohibited elsewhere (no silent fallback)
    body_lr = c.get("body_lr", 0.0)
    if isinstance(body_lr, bool) or not isinstance(body_lr, (int, float)) or not np.isfinite(body_lr) or body_lr < 0:
        raise ValueError(f"invalid body_lr: {body_lr}")
    if arm == "full_ema":
        if body_lr <= 0:
            raise ValueError("full_ema requires body_lr > 0; silent bodylr fallback is prohibited")
    else:
        if body_lr > 0:
            raise ValueError(f"body_lr must be 0 for head adaptation arm {arm}")

    # Validate window config: must be fixed mode
    c["window"] = validate_window_config(c.get("window"))
    if c["window"]["mode"] != "fixed":
        raise ValueError(f"self_audit.automatic.v1 requires fixed window mode, got {c['window']['mode']!r}")

    # Validate support config
    supp = c.get("support", {})
    pids = supp.get("patient_ids")
    if pids is not None:
        if not isinstance(pids, list) or not all(isinstance(p, str) for p in pids) or not pids:
            raise ValueError("support.patient_ids must be a non-empty list of string patient IDs or null")
        if len(set(pids)) != len(pids):
            raise ValueError("duplicate patient IDs in support.patient_ids")
    k_reuse = supp.get("k_reuse", 1)
    if type(k_reuse) is not int or k_reuse < 1:
        raise ValueError("support.k_reuse must be a positive integer")

    # Validate initialization config
    init = c.get("initialization", {})
    target_budget = init.get("declared_target_budget")
    if type(target_budget) is not int or target_budget < 1:
        raise ValueError("initialization.declared_target_budget must be a positive integer")

    ckpt_path = init.get("warm_start_checkpoint")
    ckpt_sha = init.get("warm_start_sha256")
    if ckpt_path is not None:
        if not isinstance(ckpt_path, str) or not ckpt_path:
            raise ValueError("warm_start_checkpoint must be a non-empty path string")
        if ckpt_sha is None or not isinstance(ckpt_sha, str) or len(ckpt_sha) != 64:
            raise ValueError("warm_start_sha256 must be a 64-char hex string")

    # Validate model config
    model_cfg = c.get("model", {})
    if "backbone" not in model_cfg:
        raise ValueError("model dictionary with backbone required")

    return copy.deepcopy(c)


def build_lineage(
    data: TrainingData,
    support_patient_ids: Sequence[str],
    k_reuse: int,
    prior_lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct deterministic target label lineage metadata with zero-leakage checks."""
    labeled_map = {r["id"]: r for r in data.labeled}
    support_frames: dict[str, list[int]] = {}
    for pid in sorted(support_patient_ids):
        if pid not in labeled_map:
            raise ValueError(f"support patient {pid} not in labeled training pool: {sorted(labeled_map.keys())}")
        r = labeled_map[pid]
        support_frames[pid] = sorted([int(k) for k in r["masks"].keys()])

    prior_patients = prior_lineage.get("patients", []) if prior_lineage else []
    for pid in prior_patients:
        if pid not in labeled_map:
            raise ValueError(f"prior lineage patient {pid} not in labeled pool")

    all_patients = sorted(list(set(prior_patients + list(support_patient_ids))))

    # Check for any unlabeled/query/eval overlap
    if set(all_patients) & set(data.uids):
        raise ValueError("target label lineage overlaps with unlabeled pool")
    eval_ids = set(data.manifest.get("evaluation_ids", []))
    if set(all_patients) & eval_ids:
        raise ValueError("target label lineage overlaps with evaluation set")

    return {
        "patients": all_patients,
        "count": len(all_patients),
        "support_patient_ids": sorted(list(support_patient_ids)),
        "support_frames": support_frames,
        "k_reuse": k_reuse,
        "prior_lineage": prior_lineage,
    }


def compute_config_invariants_digest(c: dict[str, Any], data_sha256: str) -> str:
    """Compute deterministic SHA-256 of canonical JSON config, excluding only steps and save_every."""
    c_copy = copy.deepcopy(c)
    c_copy.pop("steps", None)
    c_copy.pop("save_every", None)
    canonical_json = json.dumps(c_copy, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256()
    h.update(canonical_json.encode("utf-8"))
    h.update(data_sha256.encode("ascii"))
    return h.hexdigest()


def _validate_output_directory(
    out_dir: Path,
    *,
    manifest_path: str | Path | None = None,
    image_paths: Sequence[str | Path] | None = None,
    mask_paths: Sequence[str | Path] | None = None,
    warm_start_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
) -> None:
    """Ensure output directory is NEW only and validate outside actual manifest/data/checkpoint directories BEFORE mkdir."""
    out_dir = Path(out_dir).resolve()

    if out_dir.exists():
        raise ValueError(
            f"Output directory {out_dir} already exists; output directory must be new only (overwrites are prohibited)"
        )

    protected_paths: list[Path] = []
    if manifest_path is not None:
        mp = Path(manifest_path).resolve()
        protected_paths.append(mp)
        protected_paths.append(mp.parent)

    if image_paths:
        for ip in image_paths:
            p = Path(ip).resolve()
            protected_paths.append(p)
            protected_paths.append(p.parent)

    if mask_paths:
        for mp in mask_paths:
            p = Path(mp).resolve()
            protected_paths.append(p)
            protected_paths.append(p.parent)

    if warm_start_checkpoint is not None:
        ws = Path(warm_start_checkpoint).resolve()
        protected_paths.append(ws)
        if ws.is_file():
            protected_paths.append(ws.parent)

    if resume is not None:
        res = Path(resume).resolve()
        protected_paths.append(res)
        if res.is_file():
            protected_paths.append(res.parent)

    for prot in protected_paths:
        if out_dir == prot or prot in out_dir.parents or out_dir in prot.parents:
            raise ValueError(f"Output directory {out_dir} overlaps with protected path {prot}")


def save_automatic_checkpoint(
    folder: str | Path,
    model: nn.Module,
    teacher: nn.Module,
    optimizers: Sequence[torch.optim.Optimizer],
    config: dict[str, Any],
    step: int,
    provenance: dict[str, Any],
    lineage: dict[str, Any],
    *,
    data: TrainingData,
    phase: str,
    np_rng: np.random.Generator,
    counts: dict[str, Any],
    cumulative_seconds: float,
    cumulative_startup_seconds: float = 0.0,
    cumulative_extraction_seconds: float = 0.0,
    cumulative_transaction_seconds: float = 0.0,
    cumulative_final_save_seconds: float = 0.0,
    support_fingerprint: dict[str, Any] | None = None,
    is_final: bool = False,
) -> str:
    """Save weights.pt and MANIFEST.json with strict provenance, lineage, and exact RNG states. Fails closed on overwrite."""
    folder = Path(folder)
    payload_path = folder / "weights.pt"
    manifest_path = folder / "MANIFEST.json"

    if payload_path.exists() or manifest_path.exists():
        raise ValueError(f"Checkpoint already exists at {folder}; checkpoint overwrites are prohibited")

    folder.mkdir(parents=True, exist_ok=True)

    payload = {
        "student": model.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "np_rng_state": np_rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_rng": random.getstate(),
        "counters": copy.deepcopy(counts),
        "cumulative_seconds": float(cumulative_seconds),
        "cumulative_startup_seconds": float(cumulative_startup_seconds),
        "cumulative_extraction_seconds": float(cumulative_extraction_seconds),
        "cumulative_transaction_seconds": float(cumulative_transaction_seconds),
        "cumulative_final_save_seconds": float(cumulative_final_save_seconds),
        "support_fingerprint": copy.deepcopy(support_fingerprint),
    }
    torch.save(payload, payload_path)
    payload_sha = digest(payload_path)

    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "phase": phase,
        "arm": config["arm"],
        "step": step,
        "is_final": is_final,
        "prediction_model": "student",
        "training_manifest_sha256": getattr(data, "manifest_sha256", digest(data.path)),
        "labeled_ids": sorted(data.lids),
        "unlabeled_ids": sorted(data.uids),
        "declared_target_budget": config["initialization"]["declared_target_budget"],
        "support_patient_ids": sorted(lineage.get("support_patient_ids", lineage.get("patients", []))),
        "config": config,
        "target_label_lineage": lineage,
        "backbone": provenance,
        "payload_sha256": payload_sha,
        "source_sha256": automatic_source_digest(),
        "synthetic": config["model"].get("backbone") == "tiny_test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(manifest_path, manifest)
    return digest(manifest_path)


def _verify_and_read_checkpoint(
    folder: str | Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any], Path, str]:
    """Strictly verify checkpoint existence, manifest sha (MANIFEST ONLY), payload sha, lineage, and source digest."""
    if not expected_sha256 or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("expected_sha256 is mandatory and must be a 64-character hex string")

    folder = Path(folder)
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.is_file() and (folder / "final" / "MANIFEST.json").is_file():
        folder = folder / "final"
        manifest_path = folder / "MANIFEST.json"

    if not manifest_path.is_file():
        raise ValueError(f"checkpoint manifest not found at {manifest_path}")

    actual_manifest_sha = digest(manifest_path)
    if actual_manifest_sha != expected_sha256:
        raise ValueError(
            f"checkpoint hash mismatch: expected manifest {expected_sha256}, got {actual_manifest_sha}"
        )

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"expected checkpoint schema {CHECKPOINT_SCHEMA}, got {manifest.get('schema')}")

    phase = manifest.get("phase")
    if phase not in PHASES:
        raise ValueError(f"checkpoint phase must be in {sorted(PHASES)}, got {phase!r}")

    arm = manifest.get("arm")
    if arm not in NAMED_ARMS:
        raise ValueError(f"checkpoint arm must be in {sorted(NAMED_ARMS)}, got {arm!r}")

    if not isinstance(manifest.get("config"), dict):
        raise ValueError("checkpoint config must be a dictionary")
    cfg = validate_automatic_config(manifest["config"], allow_full_run=True)
    if cfg["arm"] != arm or cfg["phase"] != phase:
        raise ValueError("checkpoint configuration and phase/arm metadata differ")
    data_sha = manifest.get("training_manifest_sha256")
    if not isinstance(data_sha, str) or len(data_sha) != 64:
        raise ValueError("checkpoint training_manifest_sha256 is required")

    labeled_ids = manifest.get("labeled_ids")
    if not isinstance(labeled_ids, list) or not labeled_ids:
        raise ValueError("checkpoint labeled_ids must be a non-empty list of string IDs")

    unlabeled_ids = manifest.get("unlabeled_ids")
    if not isinstance(unlabeled_ids, list):
        raise ValueError("checkpoint unlabeled_ids must be a list of string IDs")

    if set(labeled_ids) & set(unlabeled_ids):
        raise ValueError("checkpoint labeled_ids and unlabeled_ids overlap")

    target_budget = manifest.get("declared_target_budget")
    if type(target_budget) is not int or target_budget < 1:
        raise ValueError("checkpoint declared_target_budget must be a positive integer")

    lineage = manifest.get("target_label_lineage")
    if not lineage or not isinstance(lineage, dict):
        raise ValueError("checkpoint target_label_lineage is missing or corrupted")

    patients = lineage.get("patients")
    if not isinstance(patients, list) or not patients:
        raise ValueError("checkpoint target_label_lineage patients must be a non-empty list")

    if lineage.get("count") != len(patients):
        raise ValueError("checkpoint target_label_lineage count does not match patients length")
    if target_budget != len(patients) or target_budget != cfg["initialization"]["declared_target_budget"]:
        raise ValueError("checkpoint label lineage budget mismatch")

    if not set(patients).issubset(set(labeled_ids)):
        raise ValueError("lineage patients are not a subset of labeled_ids")

    if set(patients) & set(unlabeled_ids):
        raise ValueError("lineage patients overlap with unlabeled_ids")

    payload_path = folder / "weights.pt"
    if not payload_path.is_file():
        raise ValueError(f"checkpoint payload weights.pt not found at {payload_path}")

    actual_payload_sha = digest(payload_path)
    if actual_payload_sha != manifest.get("payload_sha256"):
        raise ValueError(
            f"checkpoint payload hash mismatch: expected {manifest.get('payload_sha256')}, got {actual_payload_sha}"
        )

    current_source_sha = automatic_source_digest()
    if manifest.get("source_sha256") != current_source_sha:
        raise ValueError(
            f"executed source SHA256 {current_source_sha} differs from checkpoint {manifest.get('source_sha256')}"
        )

    if manifest.get("prediction_model") != "student":
        raise ValueError(f"checkpoint prediction_model must be 'student', got {manifest.get('prediction_model')!r}")

    return folder, manifest, payload_path, actual_manifest_sha


def load_automatic_student(
    folder: str | Path,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly load raw NativeStudent only for evaluation or deployment without teacher memory."""
    folder, manifest, payload_path, _ = _verify_and_read_checkpoint(folder, expected_sha256)
    cfg = manifest["config"]
    synthetic = manifest.get("synthetic", cfg["model"].get("backbone") == "tiny_test")

    student, _ = build_model(cfg["model"], synthetic=synthetic, window=cfg.get("window"))
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if "student" not in payload:
        raise ValueError("checkpoint payload missing 'student' weights")
    student.load_state_dict(payload["student"], strict=True)
    dev = torch.device(device)
    student = student.to(dev).eval()
    return student, manifest


def load_automatic_checkpoint(
    folder: str | Path,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, nn.Module, dict[str, Any], dict[str, Any]]:
    """Strictly load a self_audit.automatic_checkpoint.v1 checkpoint.

    Fails closed on manifest hash mismatch, payload mismatch, source digest mismatch, corrupted lineage, or schema mismatch.
    Returns: (student_model, teacher_model, manifest, payload)
    """
    folder, manifest, payload_path, _ = _verify_and_read_checkpoint(folder, expected_sha256)
    cfg = manifest["config"]
    synthetic = manifest.get("synthetic", cfg["model"].get("backbone") == "tiny_test")

    student, _ = build_model(cfg["model"], synthetic=synthetic, window=cfg.get("window"))
    teacher, _ = build_model(cfg["model"], synthetic=synthetic, window=cfg.get("window"))

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if "student" not in payload or "teacher" not in payload:
        raise ValueError("checkpoint payload missing student or teacher weights")
    student.load_state_dict(payload["student"], strict=True)
    teacher.load_state_dict(payload["teacher"], strict=True)

    dev = torch.device(device)
    return student.to(dev), teacher.to(dev), manifest, payload


def ema_head_update(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """Update teacher head parameters and buffers ONLY; feature path remains frozen."""
    with torch.no_grad():
        for t, s in zip(teacher.head.parameters(), student.head.parameters(), strict=True):
            t.mul_(decay).add_(s, alpha=1.0 - decay)
        for t, s in zip(teacher.head.buffers(), student.head.buffers(), strict=True):
            t.copy_(s)


def compute_complete_support_margins(
    model: nn.Module,
    is_head_only: bool,
    support_cache: SupportFeatureCache,
    support_records: list[SupportVolumeRecord],
    dev: torch.device,
    max_batch: int = 8,
) -> dict[str, Any]:
    """Track support margins on the same complete support for all arms (fullEMA native forward each volume)."""
    with torch.no_grad():
        was_train = model.training
        model.eval()
        if is_head_only:
            logits = model.head(support_cache.features.to(dev))
            labels = support_cache.labels.to(dev)
            val = support_cache.valid_mask.to(dev)
            m, _ = compute_margins(logits, labels, valid_mask=val)
        else:
            all_logits = []
            all_labels = []
            all_val = []
            for rec in support_records:
                for b_start in range(0, rec.num_slices, max_batch):
                    b_end = min(b_start + max_batch, rec.num_slices)
                    bx = rec.images[b_start:b_end].to(dev)
                    by = rec.labels[b_start:b_end].to(dev)
                    bl = model(bx)
                    all_logits.append(bl)
                    all_labels.append(by)
                    all_val.append(by != 255)
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            cat_val = torch.cat(all_val, dim=0)
            m, _ = compute_margins(cat_logits, cat_labels, valid_mask=cat_val)
        if was_train:
            model.train(True)
        return m


def warm_start_support(
    model: nn.Module,
    teacher: nn.Module,
    support_records: list[SupportVolumeRecord],
    config: dict[str, Any],
    data: TrainingData,
    out_dir: Path,
    device: str | torch.device,
    provenance: dict[str, Any],
    lineage: dict[str, Any],
    np_rng: np.random.Generator,
    total_steps: int,
    phase_entry_monotonic: float,
) -> dict[str, Any]:
    """Execute support-only warm start iterations and exit with a support-only checkpoint."""
    dev = torch.device(device)
    model.train()

    # Head SGD optimizer
    head_opt = torch.optim.SGD(model.head.parameters(), lr=config["head_lr"])
    body_opt = None
    if config["body_lr"] > 0 and config["arm"] == "full_ema":
        body_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        body_opt = torch.optim.AdamW(body_params, lr=config["body_lr"], weight_decay=config["weight_decay"])

    opts = [head_opt] + ([body_opt] if body_opt is not None else [])

    all_slices = []
    for rec in support_records:
        for z in range(rec.num_slices):
            all_slices.append((rec.images[z:z+1], rec.labels[z:z+1], rec.patient_id, rec.frame_id, z))

    if not all_slices:
        raise ValueError("support records have zero slices")

    counts = {
        "supervised_updates": 0,
        "pseudo_transactions_proposed": 0,
        "pseudo_transactions_accepted": 0,
        "pseudo_directions_proposed": 0,
        "pseudo_directions_accepted": 0,
        "pseudo_proposals": 0,
        "pseudo_accepted": 0,
        "nonzero_displacement_updates": 0,
        "zero_displacement_updates": 0,
        "ema_updates": 0,
        "rejected_trials": 0,
    }

    startup_seconds = time.monotonic() - phase_entry_monotonic
    step_latencies: list[float] = []
    step_loop_start = time.monotonic()

    for step in range(total_steps):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        step_start = time.monotonic()

        idx = step % len(all_slices)
        sx, sy, pid, fid, z = all_slices[idx]
        sx = sx.to(dev)
        sy = sy.to(dev)

        model.zero_grad(set_to_none=True)
        logits = model(sx)
        loss = supervised_loss(logits, sy)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss during support warm-start")
        loss.backward()

        before_head_w = model.head.weight.detach().clone()
        before_head_b = model.head.bias.detach().clone()
        before_all = [p.detach().clone() for p in model.parameters()] if config["arm"] == "full_ema" else None

        for opt in opts:
            opt.step()

        if config["arm"] == "full_ema" and before_all is not None:
            disp_norm = float(torch.sqrt(sum((p.detach() - bp).double().norm()**2 for p, bp in zip(model.parameters(), before_all))).item())
        else:
            disp_norm = float(torch.sqrt((model.head.weight - before_head_w).double().norm()**2 + (model.head.bias - before_head_b).double().norm()**2).item())

        counts["supervised_updates"] += 1
        if disp_norm > 0:
            counts["nonzero_displacement_updates"] += 1
        else:
            counts["zero_displacement_updates"] += 1

        if config["arm"] == "full_ema":
            ema_update(teacher, model, config["ema_decay"])
        else:
            ema_head_update(teacher, model, config["ema_decay"])
        counts["ema_updates"] += 1

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        step_latencies.append(time.monotonic() - step_start)

    step_loop_seconds = time.monotonic() - step_loop_start
    final_dir = out_dir / "final"

    final_save_start = time.monotonic()
    elapsed_so_far = time.monotonic() - phase_entry_monotonic
    manifest_sha = save_automatic_checkpoint(
        final_dir,
        model,
        teacher,
        opts,
        config,
        step=total_steps,
        provenance=provenance,
        lineage=lineage,
        data=data,
        phase="warm_start",
        np_rng=np_rng,
        counts=counts,
        cumulative_seconds=elapsed_so_far,
        cumulative_startup_seconds=startup_seconds,
        cumulative_extraction_seconds=0.0,
        cumulative_transaction_seconds=step_loop_seconds,
        cumulative_final_save_seconds=0.0,
        support_fingerprint=None,
        is_final=True,
    )
    final_save_seconds = time.monotonic() - final_save_start
    total_elapsed = time.monotonic() - phase_entry_monotonic

    step_latencies_ms = [round(t * 1000.0, 3) for t in step_latencies]
    peak_alloc = int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else "NOT_MEASURED"
    peak_res = int(torch.cuda.max_memory_reserved(dev)) if dev.type == "cuda" else "NOT_MEASURED"
    peak_alloc_mb = float(round(peak_alloc / (1024.0 * 1024.0), 3)) if dev.type == "cuda" else "NOT_MEASURED"
    peak_res_mb = float(round(peak_res / (1024.0 * 1024.0), 3)) if dev.type == "cuda" else "NOT_MEASURED"

    report = {
        "schema": SCHEMA,
        "phase": "warm_start",
        "status": "COMPLETED",
        "arm": config["arm"],
        "synthetic": config["model"].get("backbone") == "tiny_test",
        "quality": "NOT_EVALUATED",
        "steps": total_steps,
        "elapsed_seconds": total_elapsed,
        "phase_cost_seconds": total_elapsed,
        "marginal_adaptation_seconds": total_elapsed,
        "startup_seconds": startup_seconds,
        "extraction_seconds": 0.0,
        "transaction_seconds": step_loop_seconds,
        "final_save_seconds": final_save_seconds,
        "pseudo_transactions_proposed": 0,
        "pseudo_transactions_accepted": 0,
        "pseudo_directions_proposed": 0,
        "pseudo_directions_accepted": 0,
        "pseudo_proposals": 0,
        "pseudo_accepted": 0,
        "nonzero_displacement_updates": counts["nonzero_displacement_updates"],
        "zero_displacement_updates": counts["zero_displacement_updates"],
        "acceptance_rate": "NOT_APPLICABLE",
        "step_latencies_ms": step_latencies_ms,
        "latency_p50_ms": float(np.percentile(step_latencies_ms, 50)) if step_latencies_ms else "NOT_MEASURED",
        "latency_p95_ms": float(np.percentile(step_latencies_ms, 95)) if step_latencies_ms else "NOT_MEASURED",
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_res,
        "peak_vram_allocated_mb": peak_alloc_mb,
        "peak_vram_reserved_mb": peak_res_mb,
        "throughput_updates_per_sec": float(total_steps / total_elapsed) if total_elapsed > 0 else "NOT_MEASURED",
        "termination_reason": "STEP_LIMIT_REACHED",
        "equal_time_certified": False,
        "final_manifest_sha256": manifest_sha,
        "warm_start_manifest_sha256": manifest_sha,
        "warm_start_path": str(final_dir),
        "target_budget": config["initialization"]["declared_target_budget"],
        "lineage": lineage,
        "provenance": provenance,
        "source_sha256": automatic_source_digest(),
        "prediction_model": "student",
        "invocation_timing": {
            "startup_seconds": startup_seconds,
            "extraction_seconds": 0.0,
            "transaction_seconds": step_loop_seconds,
            "final_save_seconds": final_save_seconds,
            "total_seconds": total_elapsed,
        },
        "cumulative_timing": {
            "startup_seconds": startup_seconds,
            "extraction_seconds": 0.0,
            "transaction_seconds": step_loop_seconds,
            "final_save_seconds": final_save_seconds,
            "total_seconds": total_elapsed,
        },
    }
    write_json(out_dir / "REPORT.json", report)
    return report


def run(
    config: dict[str, Any] | str | Path,
    output: str | Path,
    *,
    synthetic: bool = False,
    resume: str | Path | None = None,
    resume_sha256: str | None = None,
    device: str = "cpu",
    allow_full_run: bool = False,
    manifest: str | Path | None = None,
    manifest_sha256: str | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """Public execution entrypoint for self_audit.automatic.v1.

    Enforces strict phase distinction (warm_start vs adaptation), exact SHA pinning,
    support feature caching with bounds, atomic controller trials, exact resume,
    and enforced time budget.
    """
    phase_entry_monotonic = time.monotonic()

    if isinstance(config, (str, Path)):
        c = json.loads(Path(config).read_text())
    else:
        c = copy.deepcopy(config)

    if synthetic:
        c["model"]["backbone"] = "tiny_test"

    c = validate_automatic_config(c, allow_full_run=allow_full_run, stop_steps=steps)
    phase = c["phase"]
    arm = c["arm"]
    dev = torch.device(device)

    # Apply thread setting
    torch.set_num_threads(c["threads"])

    # Reset CUDA peak memory at phase entry
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    # Initialize PRNGs
    torch.manual_seed(c["seed"])
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(c["seed"])
    np_rng = np.random.default_rng(c["seed"])
    random.seed(c["seed"])

    # Load TrainingData manifest and paths BEFORE directory creation
    if manifest is None:
        raise ValueError("manifest path required; no implicit data discovery")
    if manifest_sha256 is None:
        raise ValueError("manifest_sha256 required")

    manifest_file = Path(manifest).resolve()
    if not manifest_file.is_file():
        raise ValueError(f"manifest file not found at {manifest_file}")
    raw_manifest_dict = json.loads(manifest_file.read_text())
    manifest_dir = manifest_file.parent

    actual_image_paths = []
    actual_mask_paths = []
    for item in raw_manifest_dict.get("labeled", []) + raw_manifest_dict.get("unlabeled", []):
        if "image" in item:
            img_p = Path(item["image"])
            actual_image_paths.append(img_p if img_p.is_absolute() else (manifest_dir / img_p))
        for mask_info in item.get("masks", {}).values():
            if isinstance(mask_info, dict) and "path" in mask_info:
                mask_p = Path(mask_info["path"])
                actual_mask_paths.append(mask_p if mask_p.is_absolute() else (manifest_dir / mask_p))

    ws_ckpt_candidate = c["initialization"].get("warm_start_checkpoint")

    # Validate output directory NEW only and outside actual data/manifest/checkpoint directories BEFORE mkdir
    out_dir = Path(output).resolve()
    _validate_output_directory(
        out_dir,
        manifest_path=manifest_file,
        image_paths=actual_image_paths,
        mask_paths=actual_mask_paths,
        warm_start_checkpoint=ws_ckpt_candidate,
        resume=resume,
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    data = TrainingData(manifest, manifest_sha256, synthetic=synthetic)

    declared_support_ids = c["support"].get("patient_ids")
    if not declared_support_ids:
        # Fixed support primary: default to all labeled patients
        declared_support_ids = sorted(data.lids)
        # In >2L synthetic/testing fixtures without declared patient_ids, adapt default target budget to all declared L
        if c["initialization"].get("declared_target_budget") == DEFAULT_AUTOMATIC_CONFIG["initialization"]["declared_target_budget"]:
            c["initialization"]["declared_target_budget"] = len(declared_support_ids)

    target_budget = c["initialization"]["declared_target_budget"]

    # -------------------------------------------------------------------------
    # PHASE A: WARM_START ONLY
    # -------------------------------------------------------------------------
    if phase == "warm_start":
        model, provenance = build_model(c["model"], synthetic=synthetic, window=c.get("window"))
        teacher, _ = build_model(c["model"], synthetic=synthetic, window=c.get("window"))
        model.to(dev)
        teacher.to(dev)
        teacher.load_state_dict(model.state_dict(), strict=True)

        lineage = build_lineage(data, declared_support_ids, c["support"]["k_reuse"])
        if lineage["count"] != target_budget:
            raise ValueError(
                f"target label lineage count ({lineage['count']}) does not match "
                f"declared_target_budget ({target_budget}); failing closed"
            )

        support_records = extract_complete_support_volumes(
            data,
            declared_support_ids,
            max_support_voxels=c["max_support_voxels"],
        )
        total_steps = c["steps"] if c["steps"] > 0 else (c["warmup_steps"] if c["warmup_steps"] > 0 else 8)
        return warm_start_support(
            model,
            teacher,
            support_records,
            c,
            data,
            out_dir,
            dev,
            provenance,
            lineage,
            np_rng,
            total_steps=total_steps,
            phase_entry_monotonic=phase_entry_monotonic,
        )

    # -------------------------------------------------------------------------
    # PHASE B: ADAPTATION
    # -------------------------------------------------------------------------
    start_step = 0
    prior_lineage = None
    cumulative_prior_seconds = 0.0
    cumulative_prior_startup = 0.0
    cumulative_prior_extraction = 0.0
    cumulative_prior_transaction = 0.0
    payload: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    counts = {
        "supervised_updates": 0,
        "pseudo_transactions_proposed": 0,
        "pseudo_transactions_accepted": 0,
        "pseudo_directions_proposed": 0,
        "pseudo_directions_accepted": 0,
        "pseudo_proposals": 0,
        "pseudo_accepted": 0,
        "nonzero_displacement_updates": 0,
        "zero_displacement_updates": 0,
        "ema_updates": 0,
        "rejected_trials": 0,
    }

    if resume is not None:
        if phase != "adaptation":
            raise ValueError("resume requires adaptation phase")
        if resume_sha256 is None:
            raise ValueError("resume_sha256 is mandatory when --resume is specified")
        model, teacher, ckpt_m, payload = load_automatic_checkpoint(
            resume,
            expected_sha256=resume_sha256,
            device=dev,
        )
        if ckpt_m.get("phase") != "adaptation":
            raise ValueError(
                f"checkpoint phase must be 'adaptation' for resume continuation, got {ckpt_m.get('phase')!r}"
            )

        start_step = ckpt_m["step"]
        prior_lineage = ckpt_m.get("target_label_lineage")
        provenance = ckpt_m.get("backbone", {})
        cumulative_prior_seconds = float(payload.get("cumulative_seconds", 0.0))
        cumulative_prior_startup = float(payload.get("cumulative_startup_seconds", 0.0))
        cumulative_prior_extraction = float(payload.get("cumulative_extraction_seconds", 0.0))
        cumulative_prior_transaction = float(payload.get("cumulative_transaction_seconds", 0.0))

        # Check time budget cannot be reset or extended on resume
        time_budget = c.get("time_budget_seconds")
        if time_budget is not None and cumulative_prior_seconds >= time_budget:
            raise ValueError(
                f"Cannot resume run: prior cumulative seconds ({cumulative_prior_seconds:.3f}s) "
                f"already reached or exceeded time_budget_seconds ({time_budget:.3f}s); "
                "time budget cannot be reset or extended on resume"
            )

        # Mandatory resume schema checks
        if "np_rng_state" not in payload or not isinstance(payload["np_rng_state"], dict):
            raise ValueError("checkpoint payload missing mandatory 'np_rng_state'")
        if "torch_rng" not in payload or not isinstance(payload["torch_rng"], torch.Tensor):
            raise ValueError("checkpoint payload missing mandatory 'torch_rng'")
        if "python_rng" not in payload or not isinstance(payload["python_rng"], (tuple, list)):
            raise ValueError("checkpoint payload missing mandatory 'python_rng'")
        if "counters" not in payload or not isinstance(payload["counters"], dict):
            raise ValueError("checkpoint payload missing mandatory 'counters'")
        for req_c in ("supervised_updates", "nonzero_displacement_updates", "ema_updates", "rejected_trials"):
            if req_c not in payload["counters"]:
                raise ValueError(f"checkpoint payload counters missing mandatory key {req_c!r}")
        counts.update(payload["counters"])

        if "optimizers" not in payload or not isinstance(payload["optimizers"], list):
            raise ValueError("checkpoint payload missing mandatory 'optimizers'")
        if "support_fingerprint" not in payload or not isinstance(payload["support_fingerprint"], dict):
            raise ValueError("checkpoint payload missing mandatory 'support_fingerprint'")

        # Verify canonical JSON config digest all except explicitly allowed steps and save_every
        expected_invariants_hash = compute_config_invariants_digest(
            ckpt_m["config"], ckpt_m.get("training_manifest_sha256", manifest_sha256)
        )
        current_invariants_hash = compute_config_invariants_digest(c, manifest_sha256)
        if expected_invariants_hash != current_invariants_hash:
            raise ValueError(
                f"Resumed config invariants hash ({current_invariants_hash}) does not match "
                f"checkpoint config hash ({expected_invariants_hash}); arbitrary config mutation on resume prohibited"
            )

        if c["steps"] <= start_step:
            raise ValueError(f"target steps ({c['steps']}) must be greater than resumed step ({start_step})")

    else:
        # Adaptation REQUIRES shared externally SHA-pinned warm-start phase checkpoint
        ws_ckpt = c["initialization"].get("warm_start_checkpoint")
        ws_sha = c["initialization"].get("warm_start_sha256")
        if not ws_ckpt or not ws_sha:
            raise ValueError(
                "phase 'adaptation' requires an externally SHA-pinned warm-start checkpoint in "
                "initialization.warm_start_checkpoint and initialization.warm_start_sha256"
            )

        model, teacher, ckpt_m, payload = load_automatic_checkpoint(
            ws_ckpt,
            expected_sha256=ws_sha,
            device=dev,
        )
        if ckpt_m.get("phase") != "warm_start":
            raise ValueError(
                f"warm_start_checkpoint must have phase='warm_start', got {ckpt_m.get('phase')!r}; "
                "adaptation requires a warm_start phase checkpoint and rejects arbitrary adapted checkpoints."
            )
        if ckpt_m.get("arm") != "support_only":
            raise ValueError("warm-start must be support_only")
        if ckpt_m["training_manifest_sha256"] != manifest_sha256:
            raise ValueError("warm-start training manifest differs from adaptation data")
        for key in ("model", "window"):
            if ckpt_m["config"][key] != c[key]:
                raise ValueError(f"warm-start {key} differs from adaptation configuration")
        prior_lineage = ckpt_m.get("target_label_lineage")
        provenance = ckpt_m.get("backbone", {})

    lineage = build_lineage(
        data,
        declared_support_ids,
        c["support"]["k_reuse"],
        prior_lineage=prior_lineage,
    )

    if lineage["count"] != target_budget:
        raise ValueError(
            f"target label lineage count ({lineage['count']}) does not match "
            f"declared_target_budget ({target_budget}); failing closed"
        )

    # Extract complete fixed support volumes (all Z slices)
    support_records = extract_complete_support_volumes(
        data,
        declared_support_ids,
        max_support_voxels=c["max_support_voxels"],
    )

    # 1. Reset teacher feature path to student at split
    if resume is None:
        with torch.no_grad():
            for (s_name, s_p), (t_name, t_p) in zip(model.named_parameters(), teacher.named_parameters(), strict=True):
                if not s_name.startswith("head."):
                    t_p.copy_(s_p)
            for (s_name, s_b), (t_name, t_b) in zip(model.named_buffers(), teacher.named_buffers(), strict=True):
                if not s_name.startswith("head."):
                    t_b.copy_(s_b)

    # 2. Adaptation freeze: student feature parameters/buffers AND teacher feature path
    is_head_only = arm != "full_ema"
    if is_head_only:
        for name, p in model.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False
        if arm == "frozen":
            for p in model.head.parameters():
                p.requires_grad = False

    for p in teacher.parameters():
        p.requires_grad = False

    model.eval()
    teacher.eval()

    # Pre-extraction startup timing
    startup_pre_extract = time.monotonic() - phase_entry_monotonic

    # 3. Support feature extraction and timing
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    extraction_start = time.monotonic()

    support_cache = compute_support_feature_cache(
        model,
        support_records,
        device=dev,
        max_extract_batch=c["max_extract_batch"],
        max_cache_bytes=c.get("max_cache_bytes"),
        max_controller_workspace_bytes=c.get("max_controller_workspace_bytes"),
    )

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    invocation_extraction_seconds = time.monotonic() - extraction_start

    startup_post_extract_start = time.monotonic()

    # Verify teacher and student feature parity for head-only arms (must reject NaN/nonfinite)
    if is_head_only:
        with torch.no_grad():
            probe_x = support_cache.probe_sample["x"].to(dev)
            st_feats = model.features(probe_x)
            tc_feats = teacher.features(probe_x)
            if not torch.isfinite(st_feats).all() or not torch.isfinite(tc_feats).all():
                raise RuntimeError("Nonfinite features detected in teacher/student feature parity check; failing closed")
            diff_tensor = (st_feats - tc_feats).abs()
            if not torch.isfinite(diff_tensor).all():
                raise RuntimeError("Nonfinite difference in teacher/student feature parity check; failing closed")
            feat_diff = float(diff_tensor.max().item())
            if not np.isfinite(feat_diff) or feat_diff > 1e-4:
                raise RuntimeError(f"Teacher and student feature mismatch: nonfinite or diff {feat_diff:.3e}")

    # Set up optimizer for adaptation
    if arm == "full_ema":
        body_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
        opts = [
            torch.optim.SGD(model.head.parameters(), lr=c["head_lr"]),
            torch.optim.AdamW(body_params, lr=c["body_lr"], weight_decay=c["weight_decay"]),
        ]
    else:
        opts = [torch.optim.SGD(model.head.parameters(), lr=c["head_lr"])]

    # 4. Compare saved support fingerprint to rebuilt and restore states if resuming
    if resume is not None:
        saved_fp = payload["support_fingerprint"]
        rebuilt_fp = support_cache.fingerprint

        # Compare immutable support identity
        for k in ("support_inputs_sha256", "records_fingerprints", "model_config", "total_slices", "max_H", "max_W"):
            if saved_fp.get(k) != rebuilt_fp.get(k):
                raise ValueError(
                    f"Support fingerprint immutable identity mismatch for {k!r}: saved={saved_fp.get(k)!r}, rebuilt={rebuilt_fp.get(k)!r}"
                )

        if is_head_only:
            for k in ("feature_parameters_sha256", "cache_features_hash", "cache_labels_hash", "cache_valid_mask_hash"):
                if saved_fp.get(k) != rebuilt_fp.get(k):
                    raise ValueError(
                        f"Support feature cache fingerprint mismatch on resume for {k!r}: saved={saved_fp.get(k)!r}, rebuilt={rebuilt_fp.get(k)!r}"
                    )
        else:
            if saved_fp.get("feature_parameters_sha256") != rebuilt_fp.get("feature_parameters_sha256"):
                raise ValueError(
                    f"FullEMA support feature parameters mismatch on resume: saved={saved_fp.get('feature_parameters_sha256')}, rebuilt={rebuilt_fp.get('feature_parameters_sha256')}"
                )

        if len(payload["optimizers"]) != len(opts):
            raise ValueError(
                f"checkpoint optimizer count ({len(payload['optimizers'])}) does not match model optimizer count ({len(opts)})"
            )

        np_rng.bit_generator.state = payload["np_rng_state"]
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng") is not None and dev.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(tuple(payload["python_rng"]))
        for opt, opt_state in zip(opts, payload["optimizers"]):
            opt.load_state_dict(opt_state)

    # Flatten support slices for matched schedule
    support_slice_meta: list[tuple[int, int]] = []
    for rec_idx, rec in enumerate(support_records):
        for z in range(rec.num_slices):
            support_slice_meta.append((rec_idx, z))
    total_support_slices = len(support_slice_meta)
    if total_support_slices == 0:
        raise ValueError("support records have 0 slices")

    # Time budget, latency tracking, and metrics
    time_budget = c.get("time_budget_seconds")
    step_latencies: list[float] = []
    termination_reason = "STEP_LIMIT_REACHED"
    actual_final_step = start_step

    startup_post_extract = time.monotonic() - startup_post_extract_start
    invocation_startup_seconds = startup_pre_extract + startup_post_extract

    log_file = (out_dir / "steps.jsonl").open("a" if resume else "w")

    write_json(out_dir / "CONFIG.json", c)
    write_json(out_dir / "STARTUP.json", {
        "schema": SCHEMA,
        "phase": "adaptation",
        "arm": arm,
        "source_sha256": automatic_source_digest(),
        "device": device,
        "synthetic": synthetic,
        "target_budget": target_budget,
        "support_patients": sorted(declared_support_ids),
        "total_support_slices": support_cache.features.shape[0],
        "feature_cache_fingerprint": support_cache.fingerprint,
        "teacher_feature_split_policy": "reset_to_student_at_split",
        "adaptation_scope": "decoder" if arm == "full_ema" else "head",
        "extraction_seconds": invocation_extraction_seconds,
        "startup_seconds": invocation_startup_seconds,
    })

    initial_support_margins: dict[str, Any] | None = None
    final_support_margins: dict[str, Any] | None = None
    step_loop_start = time.monotonic()

    try:
        for step in range(start_step, c["steps"]):
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

            # Check time budget before starting transaction (includes startup + extraction)
            current_cumulative_seconds = cumulative_prior_seconds + (time.monotonic() - phase_entry_monotonic)
            if time_budget is not None and current_cumulative_seconds >= time_budget:
                termination_reason = "TIME_BUDGET_EXCEEDED"
                break

            step_start = time.monotonic()

            # Matched support slice index
            s_idx = step % total_support_slices
            rec_idx, z_idx = support_slice_meta[s_idx]
            rec = support_records[rec_idx]

            # Sample unlabeled slice
            ui = step % len(data.uids)
            xu, um, uvalid = data.sample_batch(
                ui,
                np_rng,
                c["batch_size"],
                labeled=False,
                crop_size=c["crop_size"],
                return_valid=True,
            )
            xu = xu.to(dev)
            uvalid = uvalid.to(dev)
            strong = strong_intensity(xu, np_rng).to(dev)

            row: dict[str, Any] = {
                "step": step + 1,
                "arm": arm,
                "unlabeled": um,
                "unlabeled_valid_pixels": int(uvalid.sum().item()),
            }

            # -----------------------------------------------------------------
            # 1. Supervised update: MATCHED NATIVE SPATIAL SLICE & NORMALIZATION
            # -----------------------------------------------------------------
            if arm not in ("frozen",):
                if is_head_only:
                    model.eval()
                    model.head.zero_grad(set_to_none=True)
                    # Slice native unpadded spatial extent to match fullEMA normalization exactly
                    sf = support_cache.features[s_idx : s_idx + 1, :, :rec.height, :rec.width].to(dev)
                    sy = support_cache.labels[s_idx : s_idx + 1, :rec.height, :rec.width].to(dev)

                    sup_logits = model.head(sf)
                    sup_loss = supervised_loss(sup_logits, sy)
                    if not torch.isfinite(sup_loss):
                        raise FloatingPointError("nonfinite support supervised loss")

                    gw, gb = torch.autograd.grad(sup_loss, (model.head.weight, model.head.bias))
                    with torch.no_grad():
                        model.head.weight.sub_(c["head_lr"] * gw)
                        model.head.bias.sub_(c["head_lr"] * gb)
                    counts["supervised_updates"] += 1
                    row["supervised_loss"] = float(sup_loss.item())
                else:
                    # full_ema arm: full model supervised step on the exact same slice
                    sx = support_records[rec_idx].images[z_idx : z_idx + 1].to(dev)
                    sy = support_records[rec_idx].labels[z_idx : z_idx + 1].to(dev)
                    model.train()
                    model.zero_grad(set_to_none=True)
                    logits = model(sx)
                    sup_loss = supervised_loss(logits, sy)
                    if not torch.isfinite(sup_loss):
                        raise FloatingPointError("nonfinite support supervised loss")
                    sup_loss.backward()
                    for opt in opts:
                        opt.step()
                    counts["supervised_updates"] += 1
                    row["supervised_loss"] = float(sup_loss.item())

            # Measure complete support margins at theta_sup (before extra step)
            margins_theta_sup = compute_complete_support_margins(
                model, is_head_only, support_cache, support_records, dev, max_batch=c["max_extract_batch"]
            )
            if initial_support_margins is None:
                initial_support_margins = margins_theta_sup

            # -----------------------------------------------------------------
            # 2. Teacher generates candidate targets on clean Xu
            # -----------------------------------------------------------------
            with torch.no_grad():
                teacher.eval()
                teacher_logits = teacher(xu, valid_mask=uvalid)
                teacher_probs = teacher_logits.softmax(1)

            # -----------------------------------------------------------------
            # 3. Unlabeled transaction by arm
            # -----------------------------------------------------------------
            control_diag: dict[str, Any] | None = None

            if arm == "frozen":
                row["status"] = "FROZEN_NO_UPDATE"

            elif arm == "support_only":
                row["status"] = "SUPPORT_ONLY_STEP"

            elif arm == "extra_supervised":
                model.eval()
                model.head.zero_grad(set_to_none=True)
                sf_extra = support_cache.features[s_idx : s_idx + 1, :, :rec.height, :rec.width].to(dev)
                sy_extra = support_cache.labels[s_idx : s_idx + 1, :rec.height, :rec.width].to(dev)
                extra_logits = model.head(sf_extra)
                extra_loss = supervised_loss(extra_logits, sy_extra)
                if not torch.isfinite(extra_loss):
                    raise FloatingPointError("nonfinite extra support supervised loss")
                gw, gb = torch.autograd.grad(extra_loss, (model.head.weight, model.head.bias))
                before_w = model.head.weight.detach().clone()
                before_b = model.head.bias.detach().clone()
                with torch.no_grad():
                    model.head.weight.sub_(c["head_lr"] * gw)
                    model.head.bias.sub_(c["head_lr"] * gb)
                disp_norm = float(torch.sqrt((model.head.weight - before_w).double().norm()**2 + (model.head.bias - before_b).double().norm()**2).item())
                counts["supervised_updates"] += 1
                if disp_norm > 0:
                    counts["nonzero_displacement_updates"] += 1
                else:
                    counts["zero_displacement_updates"] += 1
                row["extra_supervised_loss"] = float(extra_loss.item())
                row["status"] = "EXTRA_SUPERVISED_STEP"

            elif arm in ("plain_head_u", "smaller_lr"):
                # Use configured pseudo_lr directly without double multiplying
                effective_step_size = c["pseudo_lr"]
                with torch.no_grad():
                    xu_f = model.features(strong, valid_mask=uvalid)
                directions, ids, cand_info = make_candidates(
                    model.head,
                    xu_f,
                    teacher_probs,
                    c["pseudo_threshold"],
                    effective_step_size,
                    valid_mask=uvalid,
                )
                counts["pseudo_directions_proposed"] += len(directions)
                counts["pseudo_proposals"] += len(directions)
                row["proposal"] = cand_info
                if directions:
                    counts["pseudo_transactions_proposed"] += 1
                    dw = sum(d[0] for d in directions)
                    db = sum(d[1] for d in directions)
                    before_w = model.head.weight.detach().clone()
                    before_b = model.head.bias.detach().clone()
                    next_w, next_b = before_w + dw, before_b + db
                    if not torch.isfinite(next_w).all() or not torch.isfinite(next_b).all():
                        raise FloatingPointError("Nonfinite unguarded head proposal")
                    step_disp_norm = float(torch.sqrt((next_w - before_w).double().square().sum()
                                                     + (next_b - before_b).double().square().sum()).item())
                    with torch.no_grad():
                        model.head.weight.copy_(next_w)
                        model.head.bias.copy_(next_b)
                    counts["pseudo_transactions_accepted"] += 1
                    counts["pseudo_directions_accepted"] += len(directions)
                    counts["pseudo_accepted"] += 1
                    if step_disp_norm > 0:
                        counts["nonzero_displacement_updates"] += 1
                    else:
                        counts["zero_displacement_updates"] += 1
                    row["status"] = "PLAIN_UNCONSTRAINED_COMMITTED"
                else:
                    row["status"] = "NO_CANDIDATES"

            elif arm in ("scalar_brier", "joint_brier", "scalar_dice", "joint_dice"):
                with torch.no_grad():
                    xu_f = model.features(strong, valid_mask=uvalid)
                directions, ids, cand_info = make_candidates(
                    model.head,
                    xu_f,
                    teacher_probs,
                    c["pseudo_threshold"],
                    c["pseudo_lr"],
                    valid_mask=uvalid,
                )
                counts["pseudo_directions_proposed"] += len(directions)
                counts["pseudo_proposals"] += len(directions)
                row["proposal"] = cand_info

                allocation = "scalar" if "scalar" in arm else "joint"
                dice_guard = "dice" in arm

                if not support_cache.verify_fingerprint(model):
                    raise RuntimeError("Support feature cache fingerprint mismatch before controller call")

                if directions:
                    counts["pseudo_transactions_proposed"] += 1

                control_diag = controlled_head_step(
                    model.head,
                    support_cache.features.to(dev),
                    support_cache.labels.to(dev),
                    directions,
                    ids,
                    patient_ids=support_cache.patient_ids,
                    frame_ids=support_cache.frame_ids,
                    valid_mask=support_cache.valid_mask.to(dev),
                    allocation=allocation,
                    dice_guard=dice_guard,
                    ridge=c["ridge"],
                    budget=c["risk_budget"],
                    backtracks=c["backtracks"],
                    tolerance=c["guard_tolerance"],
                    dice_tolerance=c["dice_tolerance"],
                    fp_tolerance=c["fp_tolerance"],
                )
                row["control"] = control_diag
                row["status"] = control_diag["status"]
                if control_diag.get("accepted", False):
                    counts["pseudo_transactions_accepted"] += 1
                    counts["pseudo_accepted"] += 1
                    alpha_arr = np.asarray(control_diag.get("alpha", []))
                    counts["pseudo_directions_accepted"] += int(np.count_nonzero(alpha_arr > 0))
                    # Diagnostic key must match controller: realized_displacement_norm
                    disp_norm = float(control_diag.get("realized_displacement_norm", 0.0))
                    if disp_norm > 0:
                        counts["nonzero_displacement_updates"] += 1
                    else:
                        counts["zero_displacement_updates"] += 1
                else:
                    counts["rejected_trials"] += 1

            elif arm == "full_ema":
                conf, py = teacher_probs.max(1)
                valid = (conf >= c["pseudo_threshold"]) & uvalid
                row["accepted_pixels"] = int(valid.sum().item())
                if valid.any():
                    counts["pseudo_transactions_proposed"] += 1
                    counts["pseudo_directions_proposed"] += 1
                    before_params = [p.detach().clone() for p in model.parameters()]
                    row["pseudo_loss"], row["pseudo_gradient_norm"] = full_pseudo_step(
                        model,
                        opts,
                        strong,
                        py.detach(),
                        valid,
                        input_valid=uvalid,
                    )
                    disp_norm = float(torch.sqrt(sum((p.detach() - bp).double().norm()**2 for p, bp in zip(model.parameters(), before_params))).item())
                    counts["pseudo_transactions_accepted"] += 1
                    counts["pseudo_directions_accepted"] += 1
                    counts["pseudo_accepted"] += 1
                    if disp_norm > 0:
                        counts["nonzero_displacement_updates"] += 1
                    else:
                        counts["zero_displacement_updates"] += 1
                    row["status"] = "FULL_EMA_STEP"
                else:
                    row["status"] = "SKIPPED_EMPTY_PSEUDO_SUPPORT"

            # -----------------------------------------------------------------
            # 4. Support margin tracking across complete support after extra step
            # -----------------------------------------------------------------
            margins_after_extra = compute_complete_support_margins(
                model, is_head_only, support_cache, support_records, dev, max_batch=c["max_extract_batch"]
            )
            final_support_margins = margins_after_extra
            row["support_margins"] = margins_after_extra
            m_before = margins_theta_sup.get("mean_true_margin")
            m_after = margins_after_extra.get("mean_true_margin")
            if m_before is not None and m_after is not None:
                row["step_extra_margin_change"] = float(m_after - m_before)

            # -----------------------------------------------------------------
            # 5. EMA update: Head ONLY for head arms, full EMA for full_ema
            # -----------------------------------------------------------------
            if is_head_only:
                ema_head_update(teacher, model, c["ema_decay"])
            else:
                ema_update(teacher, model, c["ema_decay"])
            counts["ema_updates"] += 1

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)

            step_duration = time.monotonic() - step_start
            step_latencies.append(step_duration)

            row["step_seconds"] = step_duration
            row["elapsed_seconds"] = cumulative_prior_seconds + (time.monotonic() - phase_entry_monotonic)
            log_file.write(json.dumps(row, allow_nan=False) + "\n")
            log_file.flush()

            actual_final_step = step + 1

            # Save periodic checkpoints
            if (step + 1) % c["save_every"] == 0 and step + 1 < c["steps"]:
                if arm == "full_ema":
                    curr_fp = compute_support_feature_cache(
                        model,
                        support_records,
                        device=dev,
                        max_extract_batch=c["max_extract_batch"],
                        max_cache_bytes=c.get("max_cache_bytes"),
                        max_controller_workspace_bytes=c.get("max_controller_workspace_bytes"),
                    ).fingerprint
                else:
                    curr_fp = support_cache.fingerprint

                save_automatic_checkpoint(
                    out_dir / f"step_{step+1:06d}",
                    model,
                    teacher,
                    opts,
                    c,
                    step=step + 1,
                    provenance=provenance,
                    lineage=lineage,
                    data=data,
                    phase="adaptation",
                    np_rng=np_rng,
                    counts=counts,
                    cumulative_seconds=row["elapsed_seconds"],
                    cumulative_startup_seconds=cumulative_prior_startup + invocation_startup_seconds,
                    cumulative_extraction_seconds=cumulative_prior_extraction + invocation_extraction_seconds,
                    cumulative_transaction_seconds=cumulative_prior_transaction + (time.monotonic() - step_loop_start),
                    cumulative_final_save_seconds=0.0,
                    support_fingerprint=curr_fp,
                    is_final=False,
                )

            # Check bounded time overshoot at step boundary
            if time_budget is not None and row["elapsed_seconds"] >= time_budget and (step + 1 < c["steps"]):
                termination_reason = "TIME_BUDGET_EXCEEDED"
                break

        invocation_transaction_seconds = time.monotonic() - step_loop_start

        # For full_ema final checkpoint, recompute updated features fingerprint
        if arm == "full_ema":
            final_support_fp = compute_support_feature_cache(
                model,
                support_records,
                device=dev,
                max_extract_batch=c["max_extract_batch"],
                max_cache_bytes=c.get("max_cache_bytes"),
                max_controller_workspace_bytes=c.get("max_controller_workspace_bytes"),
            ).fingerprint
        else:
            final_support_fp = support_cache.fingerprint

        # Save final checkpoint
        final_save_start = time.monotonic()
        interim_cumulative_seconds = cumulative_prior_seconds + (time.monotonic() - phase_entry_monotonic)
        final_sha = save_automatic_checkpoint(
            out_dir / "final",
            model,
            teacher,
            opts,
            c,
            step=actual_final_step,
            provenance=provenance,
            lineage=lineage,
            data=data,
            phase="adaptation",
            np_rng=np_rng,
            counts=counts,
            cumulative_seconds=interim_cumulative_seconds,
            cumulative_startup_seconds=cumulative_prior_startup + invocation_startup_seconds,
            cumulative_extraction_seconds=cumulative_prior_extraction + invocation_extraction_seconds,
            cumulative_transaction_seconds=cumulative_prior_transaction + invocation_transaction_seconds,
            cumulative_final_save_seconds=0.0,
            support_fingerprint=final_support_fp,
            is_final=True,
        )
        final_save_seconds = time.monotonic() - final_save_start
        invocation_total_seconds = time.monotonic() - phase_entry_monotonic
        total_cumulative_seconds = cumulative_prior_seconds + invocation_total_seconds

        # Latency statistics
        step_latencies_ms = [round(t * 1000.0, 3) for t in step_latencies]
        p50 = float(np.percentile(step_latencies_ms, 50)) if step_latencies_ms else "NOT_MEASURED"
        p95 = float(np.percentile(step_latencies_ms, 95)) if step_latencies_ms else "NOT_MEASURED"

        # VRAM measurements (CUDA ONLY)
        if dev.type == "cuda":
            peak_alloc = int(torch.cuda.max_memory_allocated(dev))
            peak_res = int(torch.cuda.max_memory_reserved(dev))
            peak_alloc_mb = float(round(peak_alloc / (1024.0 * 1024.0), 3))
            peak_res_mb = float(round(peak_res / (1024.0 * 1024.0), 3))
        else:
            peak_alloc = "NOT_MEASURED"
            peak_res = "NOT_MEASURED"
            peak_alloc_mb = "NOT_MEASURED"
            peak_res_mb = "NOT_MEASURED"

        # Throughput
        steps_executed = actual_final_step - start_step
        throughput = (
            float(round(steps_executed / invocation_total_seconds, 3))
            if invocation_total_seconds > 0 and steps_executed > 0
            else "NOT_MEASURED"
        )

        # Separate transactions and directions acceptance rates
        tx_proposals = counts["pseudo_transactions_proposed"]
        tx_accepted = counts["pseudo_transactions_accepted"]
        dir_proposals = counts["pseudo_directions_proposed"]
        dir_accepted = counts["pseudo_directions_accepted"]

        if tx_proposals > 0:
            acceptance_rate: Any = float(tx_accepted / tx_proposals)
        else:
            acceptance_rate = "NOT_APPLICABLE" if arm in ("frozen", "support_only", "extra_supervised") else "NOT_MEASURED"

        # Margin changes summary: tracked at theta_sup before extra step to final post-extra step
        if initial_support_margins is not None and final_support_margins is not None:
            init_m = initial_support_margins.get("mean_true_margin")
            fin_m = final_support_margins.get("mean_true_margin")
            if init_m is not None and fin_m is not None:
                margin_changes: Any = {
                    "initial_mean_true_margin": float(init_m),
                    "final_mean_true_margin": float(fin_m),
                    "net_margin_change": float(fin_m - init_m),
                }
            else:
                margin_changes = "NOT_MEASURED"
        else:
            margin_changes = "NOT_MEASURED"

        # Equal-time certification gate
        equal_time_certified = bool(termination_reason == "TIME_BUDGET_EXCEEDED")

        report = {
            "schema": SCHEMA,
            "phase": "adaptation",
            "status": "COMPLETED",
            "arm": arm,
            "synthetic": synthetic,
            "quality": "NOT_EVALUATED",
            "steps": actual_final_step,
            "elapsed_seconds": total_cumulative_seconds,
            "phase_cost_seconds": total_cumulative_seconds,
            "marginal_adaptation_seconds": total_cumulative_seconds,
            "startup_seconds": cumulative_prior_startup + invocation_startup_seconds,
            "extraction_seconds": cumulative_prior_extraction + invocation_extraction_seconds,
            "transaction_seconds": cumulative_prior_transaction + invocation_transaction_seconds,
            "final_save_seconds": final_save_seconds,
            "pseudo_transactions_proposed": tx_proposals,
            "pseudo_transactions_accepted": tx_accepted,
            "pseudo_directions_proposed": dir_proposals,
            "pseudo_directions_accepted": dir_accepted,
            "pseudo_proposals": counts["pseudo_proposals"],
            "pseudo_accepted": counts["pseudo_accepted"],
            "nonzero_displacement_updates": counts["nonzero_displacement_updates"],
            "zero_displacement_updates": counts["zero_displacement_updates"],
            "acceptance_rate": acceptance_rate,
            "margin_changes": margin_changes,
            "step_latencies_ms": step_latencies_ms,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_res,
            "peak_vram_allocated_mb": peak_alloc_mb,
            "peak_vram_reserved_mb": peak_res_mb,
            "throughput_updates_per_sec": throughput,
            "termination_reason": termination_reason,
            "time_budget_seconds": time_budget,
            "time_budget_overshoot_seconds": (
                float(max(0.0, total_cumulative_seconds - time_budget)) if time_budget is not None else "NOT_APPLICABLE"
            ),
            "equal_time_certified": equal_time_certified,
            "final_manifest_sha256": final_sha,
            "target_budget": target_budget,
            "lineage": lineage,
            "provenance": provenance,
            "source_sha256": automatic_source_digest(),
            "prediction_model": "student",
            "invocation_timing": {
                "startup_seconds": invocation_startup_seconds,
                "extraction_seconds": invocation_extraction_seconds,
                "transaction_seconds": invocation_transaction_seconds,
                "final_save_seconds": final_save_seconds,
                "total_seconds": invocation_total_seconds,
            },
            "cumulative_timing": {
                "startup_seconds": cumulative_prior_startup + invocation_startup_seconds,
                "extraction_seconds": cumulative_prior_extraction + invocation_extraction_seconds,
                "transaction_seconds": cumulative_prior_transaction + invocation_transaction_seconds,
                "final_save_seconds": final_save_seconds,
                "total_seconds": total_cumulative_seconds,
            },
            **counts,
        }
        write_json(out_dir / "REPORT.json", report)
        return report

    except Exception as exc:
        write_json(out_dir / "FAILURE.json", {
            "schema": SCHEMA,
            "status": "FAILED",
            "phase": phase,
            "arm": arm,
            "error": repr(exc),
            "step": actual_final_step,
            **counts,
        })
        raise
    finally:
        log_file.close()
