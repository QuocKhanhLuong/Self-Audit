"""Compact student distillation training pipeline with post-supervised audited transfer.

Mode schema: self_audit.compact.v1
Timeline:
  1. Supervised AdamW step on labeled batch -> counterfactual reference theta_sup.
  2. Independent fixed teacher cache targets for unlabeled batch Xu.
  3. Class SGD candidate displacement computation via compact_control.
  4. Audited whole-network functional trials on full patient frames (or plain SGD U step).
  5. Commit accepted displacement directly to student parameters; NO student EMA.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .compact_artifacts import save_compact_checkpoint
from .compact_control import (
    CLASS_NAMES,
    FOREGROUND_CLASS_NAMES,
    apply_compact_update,
    build_class_candidates,
)
from .compact_data import CompactTrainingData, synthetic_compact_fixture
from .compact_model import CompactStudent, build_compact_model
from .compact_teacher import FixedTeacherCache, preflight_teacher_lineage, validate_teacher_identity
from .data import write_json
from .losses import supervised_loss
from .model import digest

def _stable_synthetic_seed(patient: str, frame: int, slice_index: int) -> int:
    payload = f"{patient}\0{frame}\0{slice_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _state_sha256(module: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _group_l2(values: Iterable[Tensor]) -> float:
    tensors = [value.detach().float() for value in values]
    if not tensors:
        return 0.0
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        raise ValueError("module parameters span devices; compact training requires one device")
    total = torch.zeros((), dtype=torch.float32, device=device)
    for value in tensors:
        total.add_(value.square().sum())
    return float(total.sqrt().item())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_schedule(metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    samples = metadata.get("samples")
    if not isinstance(samples, list):
        samples = [metadata]
    return [
        {
            "patient": str(sample.get("patient", "")),
            "frame": int(sample.get("frame", 0)),
            "slice": int(sample.get("slice", 0)),
            "crop_sha256": _canonical_sha256(sample.get("crop")),
        }
        for sample in samples
    ]


class SyntheticTeacherCache:
    """In-memory synthetic teacher probability cache for CPU tests and smoke validation."""

    def __init__(self, data: Any = None):
        self.data = data

    def sample_batch(
        self,
        metadata: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        *,
        device: str = "cpu",
    ) -> Tuple[Tensor, Tensor]:
        samples = metadata if isinstance(metadata, list) else metadata.get("samples", [metadata])
        targets_list: List[Tensor] = []
        valid_list: List[Tensor] = []

        for s in samples:
            crop = s.get("crop")
            pid = str(s.get("patient", "patient_u"))
            if crop is not None:
                H = W = int(crop["size"])
            elif self.data is not None and hasattr(self.data, "unlabeled"):
                matching = [r for r in self.data.unlabeled if r["id"] == pid]
                if matching:
                    cine = self.data.array(matching[0]["image"], matching[0]["image_sha256"])
                    H, W = int(cine.shape[-2]), int(cine.shape[-1])
                else:
                    H = W = 32
            else:
                H = W = 32
            frame = int(s.get("frame", 0))
            z = int(s.get("slice", 0))
            rng_seed = _stable_synthetic_seed(pid, frame, z)
            t_rng = np.random.default_rng(rng_seed)
            raw_p = t_rng.uniform(0.1, 1.0, size=(4, H, W)).astype(np.float32)
            probs = raw_p / raw_p.sum(axis=0, keepdims=True)
            val = np.ones((H, W), dtype=bool)
            targets_list.append(torch.from_numpy(probs))
            valid_list.append(torch.from_numpy(val))

        return torch.stack(targets_list).to(device=device), torch.stack(valid_list).to(device=device)

    def close(self) -> None:
        pass


METHODS = {"supervised", "plain_kd", "scalar_guard", "joint_guard"}
DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": "self_audit.compact.v1",
    "method": "joint_guard",
    "steps": 32,
    "warmup_steps": 8,
    "seed": 42,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "pseudo_lr": 1e-3,
    "temperature": 1.0,
    "coefficient": 1.0,
    "pseudo_threshold": 0.7,
    "ridge": 1e-6,
    "risk_budget": 0.0,
    "absent_fp_budget": 0.0,
    "guard_tolerance": 1e-6,
    "max_trials": 8,
    "batch_size": 1,
    "labeled_batch_size": 1,
    "unlabeled_batch_size": 1,
    "crop_size": 0,
    "save_every": 1000,
    "equal_wall_time": {"enabled": False},
    "model": {
        "name": "CompactStudent",
        "widths": [16, 32, 64, 128],
        "in_channels": 3,
        "num_classes": 4,
        "init": "kaiming_normal",
    },
    "teacher": {
        "source_training_cost": "NOT_KNOWN",
        "cache_path": None,
        "cache_sha256": None,
        "state": "student",
        "lineage": {
            "budget_k": 8,
            "exact_labeled_ids": [
                "patient007", "patient010", "patient011", "patient024",
                "patient026", "patient041", "patient077", "patient099",
            ],
        },
    },
    "acceptance": {
        "metric": "Macro4",
        "min_macro4": 0.87,
        "target_patient_dice": 0.87,
    },
}


def validate_compact_config(
    c: Mapping[str, Any],
    *,
    allow_full_run: bool = False,
    stop_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Strict opt-in config validation for self_audit.compact.v1 schema."""
    if not isinstance(c, Mapping):
        raise TypeError("config must be a dictionary or Mapping")

    cfg = copy.deepcopy(dict(c))
    if cfg.get("schema") != "self_audit.compact.v1":
        raise ValueError(
            f"invalid schema '{cfg.get('schema')}'; expected 'self_audit.compact.v1'"
        )

    method = cfg.get("method")
    if method not in METHODS:
        raise ValueError(f"invalid compact method '{method}'; expected one of {sorted(METHODS)}")

    experiment = cfg.get("experiment")
    if isinstance(experiment, Mapping) and experiment.get("launchable") is False:
        raise ValueError("compact experiment is a placeholder and is not launchable")

    # Reject enabled experimental mechanisms at any config depth.
    forbidden_features = {
        "ema", "ema_decay", "student_ema", "prototype", "prototypes",
        "teacher_memory", "window", "meta_window", "bank",
    }

    def walk_forbidden(value: Any, path: str = "") -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            key_text = str(key).lower()
            child_path = f"{path}.{key_text}" if path else key_text
            if key_text in forbidden_features:
                if isinstance(child, Mapping):
                    enabled = child.get("enabled", bool(child))
                elif key_text == "ema_decay":
                    enabled = child is not None and child is not False and child != 0
                else:
                    enabled = bool(child)
                if enabled:
                    if key_text == "student_ema":
                        raise ValueError("compact student EMA is prohibited; raw student deployment only")
                    raise ValueError(f"compact config cannot enable forbidden feature '{child_path}'")
            walk_forbidden(child, child_path)

    walk_forbidden(cfg)

    # Step budget guards
    steps = stop_steps if stop_steps is not None else cfg.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if steps > 128 and not allow_full_run:
        raise ValueError("more than 128 updates requires --allow-full-run")
    cfg["steps"] = steps

    warmup = cfg.get("warmup_steps", 8)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if stop_steps is not None and warmup > steps:
        warmup = steps
    if warmup > steps:
        raise ValueError(f"warmup_steps ({warmup}) exceeds total steps budget ({steps})")
    cfg["warmup_steps"] = warmup

    # Numeric optimization hyperparameters
    for key in ("lr", "pseudo_lr", "temperature", "coefficient", "ridge"):
        val = cfg.get(key, DEFAULT_CONFIG.get(key))
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val <= 0:
            raise ValueError(f"{key} must be a positive finite float")
        cfg[key] = float(val)

    for key in ("weight_decay", "risk_budget", "absent_fp_budget", "guard_tolerance"):
        val = cfg.get(key, DEFAULT_CONFIG.get(key))
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0:
            raise ValueError(f"{key} must be a non-negative finite float")
        cfg[key] = float(val)

    pseudo_threshold = cfg.get("pseudo_threshold", 0.7)
    if isinstance(pseudo_threshold, bool) or not isinstance(pseudo_threshold, (int, float)) or not 0 <= pseudo_threshold <= 1:
        raise ValueError("pseudo_threshold must be a float between 0.0 and 1.0")
    cfg["pseudo_threshold"] = float(pseudo_threshold)

    max_trials = cfg.get("max_trials", 8)
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 1:
        raise ValueError("max_trials must be a positive integer")
    cfg["max_trials"] = max_trials

    batch_size = cfg.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    cfg["batch_size"] = batch_size

    for key in ("labeled_batch_size", "unlabeled_batch_size"):
        value = cfg.get(key, batch_size)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        cfg[key] = value

    crop_size = cfg.get("crop_size", 0)
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size < 0:
        raise ValueError("crop_size must be a non-negative integer")
    cfg["crop_size"] = crop_size

    save_every = cfg.get("save_every", 1000)
    if isinstance(save_every, bool) or not isinstance(save_every, int) or save_every < 1:
        raise ValueError("save_every must be a positive integer")
    cfg["save_every"] = save_every

    # Locked Macro4 acceptance metadata (exactly 0.87, finite).
    acceptance = cfg.setdefault("acceptance", {})
    if not isinstance(acceptance, Mapping):
        raise TypeError("acceptance must be a dictionary")
    acceptance = dict(acceptance)
    metric = acceptance.get("metric", "Macro4")
    if metric != "Macro4":
        raise ValueError("acceptance metric must be exactly Macro4")
    raw_min_macro4 = acceptance.get("min_macro4", 0.87)
    if isinstance(raw_min_macro4, bool) or not isinstance(raw_min_macro4, (int, float)):
        raise ValueError("acceptance min_macro4 must be finite and exactly 0.87")
    min_macro4 = float(raw_min_macro4)
    if not np.isfinite(min_macro4) or min_macro4 != 0.87:
        raise ValueError("acceptance min_macro4 must be finite and exactly 0.87")
    acceptance["min_macro4"] = min_macro4
    acceptance["metric"] = "Macro4"
    cfg["acceptance"] = acceptance

    equal_wall_time = cfg.get("equal_wall_time", {"enabled": False})
    if not isinstance(equal_wall_time, Mapping):
        raise TypeError("equal_wall_time must be a mapping")
    equal_wall_time = dict(equal_wall_time)
    enabled = equal_wall_time.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("equal_wall_time.enabled must be boolean")
    if enabled:
        max_steps = equal_wall_time.get("max_steps")
        budget_seconds = equal_wall_time.get("budget_seconds")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("equal_wall_time.max_steps must be a positive integer")
        if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) or not np.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("equal_wall_time.budget_seconds must be positive and finite")
        equal_wall_time["budget_seconds"] = float(budget_seconds)
        if stop_steps is not None and max_steps > steps:
            raise ValueError("equal_wall_time.max_steps cannot exceed the immutable steps limit")
        if stop_steps is None and max_steps > steps:
            raise ValueError("equal_wall_time.max_steps cannot exceed the immutable steps limit")
    cfg["equal_wall_time"] = equal_wall_time

    # Teacher lineage requirement check
    teacher = cfg.get("teacher")
    if teacher is not None and isinstance(teacher, Mapping):
        lineage = teacher.get("lineage")
        if lineage and isinstance(lineage, Mapping) and lineage.get("matching_lineage_required"):
            budget_k = lineage.get("budget_k")
            exact_ids = lineage.get("exact_labeled_ids")
            if (
                type(budget_k) is not int
                or budget_k < 1
                or not isinstance(exact_ids, list)
                or len(exact_ids) != budget_k
                or not all(isinstance(pid, str) and pid for pid in exact_ids)
                or len(exact_ids) != len(set(exact_ids))
            ):
                raise ValueError(
                    f"placeholder for K={budget_k} requires matching teacher label lineage with explicit patient IDs; no default sweeps"
                )

    return cfg


def train_compact(
    config: Mapping[str, Any],
    manifest_path: Union[str, Path],
    manifest_sha256: str,
    out_dir: Union[str, Path],
    *,
    device: str = "cpu",
    synthetic: bool = False,
    stop_steps: Optional[int] = None,
    allow_full_run: bool = False,
) -> Dict[str, Any]:
    """Execute canonical compact student distillation and audited transfer.

    Returns:
        JSON-safe report dictionary.
    """
    start_walltime = time.monotonic()
    c = validate_compact_config(config, allow_full_run=allow_full_run, stop_steps=stop_steps)
    effective_steps = stop_steps if stop_steps is not None else c["steps"]

    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"compact output directory must not already exist: {out}")
    out.mkdir(parents=True, exist_ok=False)

    # Deterministic seeds
    seed = int(c.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_rng = np.random.default_rng(seed)

    # Resolve only the hashed manifest metadata before lineage preflight. The
    # CompactTrainingData constructor validates image arrays, so it must run
    # after the complete labeled-patient union has passed.
    if synthetic and (manifest_path is None or not Path(manifest_path).exists()):
        synth_dir = out / "synthetic_fixture"
        fix_manifest, fix_sha = synthetic_compact_fixture(
            synth_dir, seed=seed, n_labeled=1, n_unlabeled=2, size=32
        )
        manifest_path, manifest_sha256 = fix_manifest, fix_sha
    if manifest_path is None:
        raise ValueError("compact training requires an explicit TRAIN manifest")
    manifest_path = Path(manifest_path).resolve()
    actual_manifest_sha = digest(manifest_path)
    if not isinstance(manifest_sha256, str) or actual_manifest_sha != manifest_sha256.lower():
        raise ValueError(
            f"TRAIN manifest hash mismatch: expected {manifest_sha256!r}, got {actual_manifest_sha}"
        )
    manifest_obj = json.loads(manifest_path.read_text())
    if manifest_obj.get("schema") != "self_audit.data.v2" or manifest_obj.get("role") != "training":
        raise ValueError("compact TRAIN manifest must use self_audit.data.v2 with role training")
    if bool(manifest_obj.get("synthetic", False)) != synthetic:
        raise ValueError("synthetic/patient identity mismatch")
    labeled_records = manifest_obj.get("labeled")
    if not isinstance(labeled_records, list) or not labeled_records:
        raise ValueError("TRAIN manifest must declare a nonempty labeled patient list")
    manifest_lids = [record.get("id") for record in labeled_records if isinstance(record, Mapping)]
    if len(manifest_lids) != len(labeled_records) or not all(isinstance(pid, str) for pid in manifest_lids):
        raise ValueError("TRAIN manifest labeled records must have string patient IDs")
    if len(manifest_lids) != len(set(manifest_lids)):
        raise ValueError("TRAIN manifest labeled patient IDs must be unique")
    actual_budget = manifest_obj.get("label_budget")
    if type(actual_budget) is not int or actual_budget < 1 or actual_budget != len(manifest_lids):
        raise ValueError(
            f"TRAIN manifest label_budget {actual_budget!r} does not match its exact labeled IDs ({len(manifest_lids)})"
        )

    data_cfg = c.get("data")
    if data_cfg is None and synthetic:
        data_cfg = {
            "manifest_sha256": actual_manifest_sha,
            "budget_k": actual_budget,
            "labeled_ids": list(manifest_lids),
        }
        c["data"] = data_cfg
    if not isinstance(data_cfg, Mapping):
        raise ValueError("compact training requires a data mapping with manifest_sha256, budget_k, and exact labeled_ids")
    if data_cfg.get("manifest_sha256") != actual_manifest_sha:
        raise ValueError(
            f"config data.manifest_sha256 mismatch: {data_cfg.get('manifest_sha256')!r} != {actual_manifest_sha}"
        )
    cfg_budget = data_cfg.get("budget_k")
    if type(cfg_budget) is not int or cfg_budget != actual_budget:
        raise ValueError(f"config data.budget_k {cfg_budget!r} != TRAIN manifest budget K={actual_budget}")
    cfg_ids = data_cfg.get("labeled_ids")
    if (
        not isinstance(cfg_ids, list)
        or not all(isinstance(pid, str) for pid in cfg_ids)
        or len(cfg_ids) != len(set(cfg_ids))
        or set(cfg_ids) != set(manifest_lids)
    ):
        raise ValueError("config data.labeled_ids must exactly match unique TRAIN manifest labeled IDs")
    data_cfg = {
        "manifest_sha256": actual_manifest_sha,
        "budget_k": actual_budget,
        "labeled_ids": list(manifest_lids),
    }
    c["data"] = data_cfg

    # Strict teacher identity and complete label-union preflight happen before
    # any image/mask array reads, cache-array reads, model construction, or forward pass.
    lineage_data = SimpleNamespace(manifest=manifest_obj, lids=manifest_lids)
    teacher_cfg = dict(c.get("teacher") or {})
    lineage_preflight: Optional[Dict[str, Any]] = None
    validated_teacher_identity: Optional[Dict[str, Any]] = None
    if c["method"] != "supervised":
        t_id = teacher_cfg.get("teacher_identity")
        validated_teacher_identity = validate_teacher_identity(
            t_id,
            declared_k=actual_budget,
            declared_labeled_ids=manifest_lids,
            data=lineage_data,
        )
        teacher_ids = list(validated_teacher_identity["labeled_ids"])
        if set(teacher_ids) != set(manifest_lids):
            raise ValueError("teacher exact labeled IDs must match the configured TRAIN support for matched compact arms")
        declared_lineage = teacher_cfg.get("lineage")
        if declared_lineage is not None:
            if not isinstance(declared_lineage, Mapping):
                raise ValueError("teacher.lineage must be a mapping")
            lineage_ids = declared_lineage.get("exact_labeled_ids")
            if type(declared_lineage.get("budget_k")) is not int or declared_lineage["budget_k"] != actual_budget:
                raise ValueError("teacher.lineage.budget_k must match config data.budget_k")
            if (
                not isinstance(lineage_ids, list)
                or not all(isinstance(pid, str) for pid in lineage_ids)
                or set(lineage_ids) != set(teacher_ids)
                or len(lineage_ids) != len(set(lineage_ids))
            ):
                raise ValueError("teacher.lineage.exact_labeled_ids must match the verified teacher identity")

        warmstart_ids = c.get("warmstart_labeled_ids", manifest_lids)
        audit_ids = c.get("audit_labeled_ids", manifest_lids)
        auxiliary_ids: List[str] = []
        auxiliary_declarations = [
            c.get("auxiliary_labeled_ids", []),
            teacher_cfg.get("auxiliary_labeled_ids", []),
            declared_lineage.get("auxiliary_labeled_ids", []) if isinstance(declared_lineage, Mapping) else [],
            t_id.get("auxiliary_labeled_ids", []) if isinstance(t_id, Mapping) else [],
            (t_id.get("lineage", {}).get("auxiliary_labeled_ids", [])
             if isinstance(t_id, Mapping) and isinstance(t_id.get("lineage"), Mapping) else []),
        ]
        for declared_auxiliary in auxiliary_declarations:
            if not isinstance(declared_auxiliary, list) or not all(
                isinstance(pid, str) for pid in declared_auxiliary
            ):
                raise ValueError("auxiliary_labeled_ids declarations must be lists of patient IDs")
            auxiliary_ids.extend(declared_auxiliary)
        for name, ids in (("warmstart_labeled_ids", warmstart_ids), ("audit_labeled_ids", audit_ids)):
            if not isinstance(ids, list) or not all(isinstance(pid, str) for pid in ids) or set(ids) != set(manifest_lids):
                raise ValueError(f"{name} must exactly match the configured TRAIN labeled support")
        lineage_preflight = preflight_teacher_lineage(
            teacher_labeled_ids=teacher_ids,
            student_labeled_ids=manifest_lids,
            data=lineage_data,
            warmstart_labeled_ids=warmstart_ids,
            audit_labeled_ids=audit_ids,
            auxiliary_labeled_ids=auxiliary_ids,
            label_budget=actual_budget,
            declared_labeled_ids=manifest_lids,
            teacher_kind=validated_teacher_identity.get("teacher_kind"),
        )
        lineage_preflight["train_manifest_sha256"] = actual_manifest_sha

    # Actual image/mask geometry validation happens only after label lineage is bound.
    data = CompactTrainingData(
        manifest_path, actual_manifest_sha, synthetic=synthetic, allow_k1=True
    )
    if data.lids != manifest_lids or data.manifest.get("label_budget") != actual_budget:
        raise ValueError("loaded compact data differs from preflighted TRAIN manifest metadata")

    # Teacher probability cache
    cache = None
    if c["method"] != "supervised":
        cache_path = teacher_cfg.get("cache_path")
        cache_sha = teacher_cfg.get("cache_sha256")
        if cache_path is not None:
            if not isinstance(cache_sha, str) or len(cache_sha) != 64:
                raise ValueError("configured teacher cache requires its exact 64-character cache_sha256")
            cache = FixedTeacherCache(
                cache_path,
                cache_sha,
                data,
                validated_teacher_identity,
            )
        elif cache_sha is not None:
            raise ValueError("teacher.cache_sha256 requires teacher.cache_path")
        elif synthetic:
            cache = SyntheticTeacherCache(data)
        else:
            raise ValueError(
                "non-synthetic compact KD training requires an explicit, verified teacher cache; cache protection active"
            )

    if isinstance(cache, FixedTeacherCache):
        validated_teacher_identity = copy.deepcopy(cache.teacher_identity)
        lineage_preflight["teacher_verification"] = validated_teacher_identity["verification_status"]
    cache_generation = copy.deepcopy(getattr(cache, "raw_manifest", {}).get("telemetry"))
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # Build Compact Student only after the full teacher/data preflight.
    model_cfg = c.get("model", {})
    model = build_compact_model(model_cfg)
    model.to(device)
    initial_state_sha256 = _state_sha256(model)

    # Parameter groups and named ownership for initialization receipts.
    module_params = model.module_parameters()
    owner_by_param = {
        id(parameter): module_name
        for module_name, parameters in module_params.items()
        for parameter in parameters
    }
    parameter_ownership = {
        name: owner_by_param.get(id(parameter), "unassigned")
        for name, parameter in model.named_parameters()
    }

    lr = float(c["lr"])
    wd = float(c["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    optimizer_metadata = {
        "name": type(optimizer).__name__,
        "owner": "labeled_supervised_update_only",
        "parameter_ownership": parameter_ownership,
    }

    # Store the resolved, immutable config before the first optimization step.
    write_json(out / "config.json", c)

    # Telemetry and logging
    log_path = out / "TRAIN_LOG.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    slices_sampled = {"labeled": 0, "unlabeled": 0, "audit": 0}
    slices_optimized = {"labeled": 0, "unlabeled": 0}
    optimizer_calls = {"adamw": 0, "sgd": 0}
    counts = {
        "supervised_updates": 0,
        "pseudo_accepted": 0,
        "pseudo_rejected": 0,
        "warmup_steps": 0,
        "no_candidates": 0,
        "rejected_all": 0,
        "zero_displacement": 0,
    }
    seen_l: set[str] = set()
    seen_a: set[str] = set()
    seen_u: set[str] = set()
    total_cache_time = 0.0
    warm_start_logs: List[Dict[str, Any]] = []
    audit_totals = {"slice_reads": 0, "frame_reads": 0, "patient_reads": 0, "stream_replays": 0}
    audit_unique_frames: set[Tuple[str, int]] = set()
    equal_wall_time = c["equal_wall_time"]
    wall_time_enabled = bool(equal_wall_time.get("enabled", False))
    wall_step_limit = min(
        effective_steps,
        equal_wall_time["max_steps"] if wall_time_enabled else effective_steps,
    )
    wall_budget_seconds = equal_wall_time.get("budget_seconds") if wall_time_enabled else None
    _sync_device(device)
    training_wall_start = time.monotonic()
    completed_steps = 0
    termination_reason = "max_steps_reached"

    try:
        for step in range(wall_step_limit):
            if wall_time_enabled:
                _sync_device(device)
                if time.monotonic() - training_wall_start >= wall_budget_seconds:
                    termination_reason = "wall_time_budget_reached"
                    break
            step_start = time.monotonic()

            # Matched sampling RNG schedule across all compact arms
            li, ai, ui = data.sample_indices(step, data_rng)

            # Sample supervised batch (L only; foreground sampling active)
            Xs, ys, l_meta = data.sample_batch(
                li, data_rng, c["labeled_batch_size"], labeled=True, crop_size=c["crop_size"]
            )
            Xs = Xs.to(device)
            ys = ys.to(device)
            slices_sampled["labeled"] += len(Xs)
            p_sup_id = str(l_meta.get("patient", data.labeled[li]["id"]))
            seen_l.add(p_sup_id)

            # Select the audit patient in every arm to retain an identical data
            # schedule, but read its frames only for declared guarded methods.
            audit_pid = str(data.labeled[ai]["id"])
            audit_step_counts = {"slice_reads": 0, "frame_reads": 0, "patient_reads": 0, "stream_replays": 0}
            audit_frames_this_step: set[Tuple[str, int]] = set()

            # Sample unlabeled batch (Image-only U; NO foreground sampling)
            Xu, u_meta = data.sample_batch(
                ui, data_rng, c["unlabeled_batch_size"], labeled=False, crop_size=c["crop_size"]
            )
            Xu = Xu.to(device)
            slices_sampled["unlabeled"] += len(Xu)
            p_unl_id = str(u_meta.get("patient", data.unlabeled[ui]["id"]))
            seen_u.add(p_unl_id)
            schedule = {
                "step": step,
                "supervised": _sample_schedule(l_meta),
                "unlabeled": _sample_schedule(u_meta),
                "audit_patient_selected": audit_pid,
            }
            schedule_sha256 = _canonical_sha256(schedule)

            audit_batches_callable = None
            if c["method"] in {"scalar_guard", "joint_guard"} and step >= c["warmup_steps"]:
                raw_audit_batches = data.create_audit_batches(patient_index=ai, device=device)

                def tracked_audit_batches() -> Iterable[Dict[str, Any]]:
                    audit_step_counts["stream_replays"] += 1
                    audit_totals["stream_replays"] += 1
                    replay_patients: set[str] = set()
                    replay_frames: set[Tuple[str, int]] = set()
                    for batch in raw_audit_batches():
                        patient_id = str(batch["patient"])
                        frame_id = int(batch["frame"])
                        slice_id = int(batch["slice"])
                        frame_key = (patient_id, frame_id)
                        seen_a.add(patient_id)
                        audit_unique_frames.add(frame_key)
                        audit_frames_this_step.add(frame_key)
                        if patient_id not in replay_patients:
                            replay_patients.add(patient_id)
                            audit_step_counts["patient_reads"] += 1
                            audit_totals["patient_reads"] += 1
                        if frame_key not in replay_frames:
                            replay_frames.add(frame_key)
                            audit_step_counts["frame_reads"] += 1
                            audit_totals["frame_reads"] += 1
                        n_slices = int(batch["x"].shape[0])
                        audit_step_counts["slice_reads"] += n_slices
                        audit_totals["slice_reads"] += n_slices
                        slices_sampled["audit"] += n_slices
                        yield batch

                audit_batches_callable = tracked_audit_batches

            # Step 1: Learned Supervised AdamW update
            _sync_device(device)
            t_s0 = time.monotonic()
            model.train()
            optimizer.zero_grad()

            logits_s = model(Xs)
            loss_s = supervised_loss(logits_s, ys)
            loss_s.backward()

            # Warm-start module gradient norms (encoder, decoder, head)
            grad_norms = {
                mod: _group_l2(p.grad for p in params if p.grad is not None)
                for mod, params in module_params.items()
            }

            # Parameter snapshot before step to observe weight changes
            weights_before = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.step()

            # Warm-start module weight change norms (encoder, decoder, head)
            named_params_dict = dict(model.named_parameters())
            weight_change_norms = {}
            for mod, params in module_params.items():
                p_ids = {id(p) for p in params}
                changes = []
                for name, p in named_params_dict.items():
                    if id(p) in p_ids:
                        changes.append(p.detach() - weights_before[name])
                weight_change_norms[mod] = _group_l2(changes)

            _sync_device(device)
            supervised_time = time.monotonic() - t_s0
            slices_optimized["labeled"] += len(Xs)
            counts["supervised_updates"] += 1
            optimizer_calls["adamw"] += 1

            warm_start_rec = {
                "step": step,
                "gradient_norms": grad_norms,
                "weight_changes": weight_change_norms,
                "optimizer": optimizer_metadata if step == 0 else None,
            }
            warm_start_logs.append(warm_start_rec)

            # Post-supervised reference state is now the live model parameters.
            # Step 2: Unlabeled pseudo-update transaction
            t_cache = 0.0
            t_prop = 0.0
            t_audit = 0.0
            u_receipt: Dict[str, Any] = {}
            candidate_receipt = None

            if c["method"] == "supervised":
                u_receipt = {
                    "method": "supervised",
                    "status": "zero_displacement",
                    "accepted": False,
                    "scale": 0.0,
                    "proposed_displacement_norm": 0.0,
                    "realized_displacement_norm": 0.0,
                    "margin_change": 0.0,
                    "mean_margin_decreased": False,
                    "margin_decreased_fraction": None,
                    "trials": [],
                }
                counts["zero_displacement"] += 1
            elif step < c["warmup_steps"]:
                u_receipt = {
                    "method": c["method"],
                    "status": "warmup",
                    "accepted": False,
                    "scale": 0.0,
                    "proposed_displacement_norm": 0.0,
                    "realized_displacement_norm": 0.0,
                    "margin_change": 0.0,
                    "mean_margin_decreased": False,
                    "margin_decreased_fraction": None,
                    "trials": [],
                }
                counts["warmup_steps"] += 1
            else:
                # Sample fixed teacher cache targets for Xu
                _sync_device(device)
                t_c0 = time.monotonic()
                teacher_targets, valid_mask = cache.sample_batch(u_meta, device=device)
                _sync_device(device)
                t_cache = time.monotonic() - t_c0
                total_cache_time += t_cache

                # Build class-grouped KD displacement candidates (non-mutating)
                t_p0 = time.monotonic()
                candidates = build_class_candidates(
                    model,
                    Xu,
                    teacher_targets,
                    valid_mask,
                    lr=float(c["pseudo_lr"]),
                    temperature=float(c["temperature"]),
                    coefficient=float(c["coefficient"]),
                    confidence=float(c["pseudo_threshold"]),
                )
                candidate_receipt = {
                    "source_ids": list(candidates["source_ids"]),
                    "denominator": candidates["denominator"],
                    "telemetry": copy.deepcopy(candidates["telemetry"]),
                }
                _sync_device(device)
                t_prop = time.monotonic() - t_p0

                # Audit trial search and atomic commit
                t_a0 = time.monotonic()
                audit_cfg = {
                    "eps": float(c["risk_budget"]),
                    "absent_fp_budget": float(c["absent_fp_budget"]),
                    "tolerance": float(c["guard_tolerance"]),
                    "ridge": float(c["ridge"]),
                    "max_trials": int(c["max_trials"]),
                }
                u_receipt = apply_compact_update(
                    model,
                    candidates,
                    method=c["method"],
                    audit_batches=audit_batches_callable,
                    config=audit_cfg,
                )
                _sync_device(device)
                t_audit = time.monotonic() - t_a0

                if u_receipt.get("accepted"):
                    slices_optimized["unlabeled"] += len(Xu)
                    counts["pseudo_accepted"] += 1
                    optimizer_calls["sgd"] += 1
                else:
                    counts["pseudo_rejected"] += 1

                if u_receipt.get("status") == "no_candidate":
                    counts["no_candidates"] += 1
                elif u_receipt.get("status") == "zero_displacement":
                    counts["zero_displacement"] += 1
                elif u_receipt.get("status") == "rejected_all":
                    counts["rejected_all"] += 1

            # NO COMPACT EMA UPDATE! Student is raw exported model.
            _sync_device(device)
            step_elapsed = time.monotonic() - step_start
            completed_steps = step + 1
            if wall_time_enabled and time.monotonic() - training_wall_start >= wall_budget_seconds:
                termination_reason = "wall_time_budget_reached"

            # Write step telemetry log
            step_row = {
                "step": step,
                "method": c["method"],
                "status": u_receipt.get("status", "unknown"),
                "accepted": bool(u_receipt.get("accepted", False)),
                "supervised_loss": float(loss_s.item()),
                "initial_state_sha256": initial_state_sha256 if step == 0 else None,
                "first_step_optimizer": optimizer_metadata if step == 0 else None,
                "warm_start": warm_start_rec,
                "batch_sizes": {
                    "labeled": int(c["labeled_batch_size"]),
                    "unlabeled": int(c["unlabeled_batch_size"]),
                },
                "effective_batch_counts": {"labeled": len(Xs), "unlabeled": len(Xu)},
                "data_schedule": schedule,
                "data_schedule_sha256": schedule_sha256,
                "audit_consumption": {
                    **audit_step_counts,
                    "unique_frames": [list(item) for item in sorted(audit_frames_this_step)],
                    "selected_patient": audit_pid,
                },
                "scale": float(u_receipt.get("scale", 0.0)),
                "proposed_displacement_norm": float(u_receipt.get("proposed_displacement_norm", 0.0)),
                "realized_displacement_norm": float(u_receipt.get("realized_displacement_norm", 0.0)),
                "margins_before": u_receipt.get("margins_before", {}),
                "margins_after": u_receipt.get("margins_after", {}),
                "margin_change": u_receipt.get("margin_change"),
                "mean_margin_decreased": bool(u_receipt.get("mean_margin_decreased", False)),
                "margin_decreased_fraction": u_receipt.get("margin_decreased_fraction"),
                "trial_count": len(u_receipt.get("trials", [])),
                "candidate_receipt": candidate_receipt,
                "transaction": u_receipt,
                "timings": {
                    "supervised_seconds": float(supervised_time),
                    "teacher_cache_seconds": float(t_cache),
                    "proposal_seconds": float(t_prop),
                    "audit_trials_seconds": float(t_audit),
                    "step_elapsed_seconds": float(step_elapsed),
                    "cuda_synchronized": str(device).startswith("cuda"),
                },
                "runtime_scope": (
                    "AUDIT_OVERHEAD_INCLUDED" if c["method"] in {"scalar_guard", "joint_guard"}
                    else "PLAIN_KD_EXCLUDES_AUDIT_NOT_LATENCY_MATCHED" if c["method"] == "plain_kd"
                    else "SUPERVISED_NO_AUDIT"
                ),
                "slices_sampled": copy.deepcopy(slices_sampled),
                "slices_optimized": copy.deepcopy(slices_optimized),
                "optimizer_calls": copy.deepcopy(optimizer_calls),
                "cumulative_updates": copy.deepcopy(counts),
            }
            log_file.write(json.dumps(step_row, allow_nan=False) + "\n")
            log_file.flush()

            # Periodic checkpoint save
            if (step + 1) % c["save_every"] == 0 and (step + 1) < wall_step_limit:
                save_compact_checkpoint(
                    model,
                    optimizer,
                    c,
                    data,
                    out / f"step_{step + 1:06d}",
                    step=step + 1,
                    teacher_identity=validated_teacher_identity,
                )

            if wall_time_enabled and termination_reason == "wall_time_budget_reached":
                break

        # Final checkpoint save
        final_manifest_sha = save_compact_checkpoint(
            model,
            optimizer,
            c,
            data,
            out / "final",
            step=completed_steps,
            teacher_identity=validated_teacher_identity,
            extra={"completed": True, "total_steps": completed_steps},
        )

        total_elapsed = time.monotonic() - start_walltime
        if completed_steps >= wall_step_limit and termination_reason != "wall_time_budget_reached":
            termination_reason = "max_steps_reached"

        # Teacher training cost reporting contract:
        # Report NOT_KNOWN if unavailable, never 0.
        teacher_training_cost = teacher_cfg.get("source_training_cost")
        if teacher_training_cost is None or teacher_training_cost == 0:
            teacher_training_cost = "NOT_KNOWN"

        final_payload_sha = digest(out / "final" / "weights.pt")
        from .engine import source_digest

        source_hashes = {
            "self_audit/compact_training.py": digest(Path(__file__)),
            "self_audit/engine.py": digest(Path(__file__).with_name("engine.py")),
            "train.py": digest(Path(__file__).resolve().parents[1] / "train.py"),
        }

        report: Dict[str, Any] = {
            "schema": "self_audit.compact.report.v1",
            "status": "COMPLETED",
            "method": c["method"],
            "synthetic": bool(synthetic),
            "quality": "NOT_EVALUATED",
            "steps": completed_steps,
            "steps_requested": effective_steps,
            "max_steps": wall_step_limit,
            "termination_reason": termination_reason,
            "equal_wall_time": copy.deepcopy(equal_wall_time),
            "counts": counts,
            "slices_sampled": slices_sampled,
            "slices_optimized": slices_optimized,
            "batch_sizes": {
                "labeled": int(c["labeled_batch_size"]),
                "unlabeled": int(c["unlabeled_batch_size"]),
            },
            "effective_batch_counts": {
                "labeled": slices_sampled["labeled"],
                "unlabeled": slices_sampled["unlabeled"],
            },
            "optimizer_calls": optimizer_calls,
            "teacher_costs": {
                "teacher_cache_total_seconds": float(total_cache_time),
                "teacher_cache_generation_cost": cache_generation or "NOT_KNOWN",
                "teacher_training_cost": teacher_training_cost,
            },
            "audit_consumption": {
                **audit_totals,
                "unique_frames": [list(item) for item in sorted(audit_unique_frames)],
                "unique_patients": sorted(seen_a),
            },
            "runtime_comparability": (
                "AUDIT_OVERHEAD_INCLUDED" if c["method"] in {"scalar_guard", "joint_guard"}
                else "PLAIN_KD_EXCLUDES_AUDIT_NOT_LATENCY_MATCHED" if c["method"] == "plain_kd"
                else "SUPERVISED_NO_AUDIT"
            ),
            "quality_gate": {
                "threshold_macro4": float(c["acceptance"]["min_macro4"]),
                "status": "GATE_METADATA_LOCKED",
                "notes": "Acceptance requires evaluation of exported compact student with Macro4 >= 0.87",
            },
            "warm_start_summary": {
                "initial_gradient_norms": warm_start_logs[0]["gradient_norms"] if warm_start_logs else {},
                "initial_weight_changes": warm_start_logs[0]["weight_changes"] if warm_start_logs else {},
                "final_gradient_norms": warm_start_logs[-1]["gradient_norms"] if warm_start_logs else {},
                "final_weight_changes": warm_start_logs[-1]["weight_changes"] if warm_start_logs else {},
            },
            "initialization": {
                "seed": seed,
                "initial_state_sha256": initial_state_sha256,
                "first_step_optimizer": optimizer_metadata,
            },
            "teacher_preflight": lineage_preflight,
            "train_manifest_sha256": actual_manifest_sha,
            "teacher_cache_manifest_sha256": teacher_cfg.get("cache_sha256"),
            "source_hashes": source_hashes,
            "source_digest": source_digest(),
            "final_manifest_sha256": final_manifest_sha,
            "final_weights_sha256": final_payload_sha,
            "final_checkpoint_path": str(out / "final"),
            "final_weights_path": str(out / "final" / "weights.pt"),
            "report_path": str(out / "REPORT.json"),
            "output_dir": str(out),
            "label_budget": len(data.lids),
            "seen_supervised": sorted(seen_l),
            "seen_audit": sorted(seen_a),
            "seen_unlabeled": sorted(seen_u),
            "elapsed_seconds": float(total_elapsed),
            "profile": {
                "device": str(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if str(device).startswith("cuda") else None,
                "optimizer_steps_per_second": completed_steps / max(time.monotonic() - training_wall_start, 1e-12),
                "supervised_slices_per_second": slices_optimized["labeled"] / max(time.monotonic() - training_wall_start, 1e-12),
                "scope": "student training including synchronous telemetry, proposal and audit; teacher/cache build separate",
            },
            "training_elapsed_seconds": float(time.monotonic() - training_wall_start),
        }
        write_json(out / "REPORT.json", report)
        return report

    except Exception as exc:
        failure_record = {
            "schema": "self_audit.compact.failure.v1",
            "status": "FAILED",
            "error": repr(exc),
            "counts": counts,
            "slices_sampled": slices_sampled,
            "slices_optimized": slices_optimized,
            "optimizer_calls": optimizer_calls,
            "seen_supervised": sorted(seen_l),
            "seen_audit": sorted(seen_a),
            "seen_unlabeled": sorted(seen_u),
        }
        write_json(out / "FAILURE.json", failure_record)
        raise
    finally:
        log_file.close()
        if cache is not None:
            cache.close()
        data.close()
