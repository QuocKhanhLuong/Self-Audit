"""Candidate PG stage S2 (teacher) training.

The implementation in this module is intentionally small and explicit.  It
owns the S2 update, checkpoint, and epoch-receipt contracts; model construction,
data manifests, and patient metric primitives remain in their respective PG
modules.  The function is usable with an injected tiny model/fixture for CPU
software checks and fails closed when a real configuration is incomplete.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .logging import EpochLogger, epoch_console_line
from .reliability import ClassAdaptiveThreshold
from .views import StatefulViewGenerator, center_crop_pad, cutmix_box, mix_packet, mix_tensor


SCHEMA = "self_audit.candidate_pg.teacher_training.v1"
_CLASS_NAMES = ("RV", "MYO", "LV")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name} requires explicit key {key!r}")
    return mapping[key]


def _config_hash(config: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("teacher config is not JSON serializable") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_all(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("PG seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        out["torch_cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return out


def _restore_rng(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or not {"python", "numpy", "torch_cpu"}.issubset(state):
        raise ValueError("teacher checkpoint is missing RNG state")
    random.setstate(state["python"])
    np_state = state["numpy"]
    if isinstance(np_state, Mapping):
        np_state = (
            str(np_state["generator"]),
            np.asarray(np_state["keys"], dtype=np.uint32),
            int(np_state["position"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    np.random.set_state(np_state)
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(x, dtype=torch.uint8, device="cpu") for x in state["torch_cuda"]])


def _atomic_save(value: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temp)
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _frame_value(frame: Any, key: str, default: Any = None) -> Any:
    if isinstance(frame, Mapping):
        return frame.get(key, default)
    return getattr(frame, key, default)


def _as_image_target(frame: Any, device: torch.device, *, labeled: bool) -> tuple[Tensor, Tensor | None, dict[str, Any], str, int]:
    image = _frame_value(frame, "image")
    if image is None:
        raise ValueError("PG teacher frame is missing image")
    image = torch.as_tensor(image, dtype=torch.float32, device=device)
    if image.ndim == 3:
        image = image[None, None]
    elif image.ndim == 4:
        image = image[None] if image.shape[0] != 1 else image[:, None]
    if image.ndim != 5 or image.shape[1] != 1 or not bool(torch.isfinite(image).all()):
        raise ValueError("PG teacher image must be finite [B,1,D,H,W]")
    if bool((image < 0).any()) or bool((image > 1).any()):
        raise ValueError("PG teacher image must be normalized to [0,1]")
    geometry = _frame_value(frame, "geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("PG teacher frame requires native geometry")
    geometry = dict(geometry)
    shape = tuple(int(v) for v in _required(geometry, "shape", "geometry"))
    if shape != tuple(int(v) for v in image.shape[-3:]):
        raise ValueError("PG teacher frame shape differs from native geometry")
    spacing = _required(geometry, "spacing_zyx", "geometry")
    if len(spacing) != 3 or any(not math.isfinite(float(v)) or float(v) <= 0 for v in spacing):
        raise ValueError("PG teacher geometry spacing must be finite and positive")
    target = _frame_value(frame, "target")
    if labeled:
        if target is None:
            raise ValueError("labeled PG teacher frame is missing target")
        target = torch.as_tensor(target, dtype=torch.long, device=device)
        if target.ndim == 3:
            target = target.unsqueeze(0)
        elif target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0]
        if tuple(target.shape) != (image.shape[0], *image.shape[-3:]):
            raise ValueError("PG teacher target differs from native image shape")
        if bool(((target < 0) | (target > 3)).any()):
            raise ValueError("PG teacher labels must be class IDs 0..3")
    patient_value = _frame_value(frame, "patient_id", _frame_value(frame, "patient"))
    frame_value = _frame_value(frame, "frame_index", _frame_value(frame, "frame"))
    if patient_value is None or frame_value is None:
        raise ValueError("PG teacher frame requires patient_id and frame_index")
    patient = str(patient_value)
    frame_index = int(frame_value)
    return image, target, geometry, patient, frame_index


def _source_items(data: Any, *, labeled: bool) -> tuple[Any, ...]:
    """Materialize only the requested role when a sequence fixture is used."""
    if data is None:
        return tuple()
    if isinstance(data, Mapping):
        key = "labeled" if labeled else "unlabeled"
        if key in data:
            return tuple(data[key])
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return tuple(data)
    return tuple()


def _load_frame(data: Any, key: Any, *, labeled: bool, index: int | None = None) -> Any:
    if hasattr(data, "load") and callable(data.load):
        return data.load(key, labeled=labeled)
    items = _source_items(data, labeled=labeled)
    if not items:
        raise ValueError("PG teacher data has no requested frames")
    if not isinstance(key, int):
        raise TypeError("sequence fixture sampler keys must be integer indices")
    return items[key]


class _SequenceTripleSampler:
    """Tiny-fixture analogue of PGSampler with an exact checkpoint state."""

    SCHEMA = "self_audit.candidate_pg.teacher_training.sequence_sampler.v1"

    def __init__(self, train_data: Any, *, seed: int) -> None:
        self.labeled = _source_items(train_data, labeled=True)
        self.unlabeled = _source_items(train_data, labeled=False)
        if not self.labeled or len(self.unlabeled) < 2:
            raise ValueError("PG teacher sampler needs labeled frames and two U frames")
        self.seed = int(seed)
        self.rng = random.Random(seed ^ 0x50475332)
        self.cursor = 0

    def next_step(self) -> tuple[int, int, int]:
        l = self.cursor % len(self.labeled)
        a = self.cursor % len(self.unlabeled)
        b = (self.cursor + 1) % len(self.unlabeled)
        self.cursor += 1
        return l, a, b

    def state_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "seed": self.seed, "cursor": self.cursor, "rng": self.rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != self.SCHEMA or int(state.get("seed")) != self.seed:
            raise ValueError("teacher sampler checkpoint identity mismatch")
        self.cursor = int(state["cursor"])
        self.rng.setstate(state["rng"])


def _next_triple(sampler: Any) -> tuple[Any, Any, Any]:
    method = getattr(sampler, "next_step", None)
    if not callable(method):
        raise TypeError("PG teacher sampler must expose exact next_step API")
    value = method()
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("PG teacher sampler must return one L and two U keys")
    return value


def _sampler_state(sampler: Any) -> Any:
    method = getattr(sampler, "state_dict", None)
    if not callable(method):
        raise TypeError("PG teacher sampler must expose state_dict for resumability")
    return copy.deepcopy(method())


def _load_sampler_state(sampler: Any, state: Any) -> None:
    method = getattr(sampler, "load_state_dict", None)
    if not callable(method):
        raise TypeError("PG teacher sampler must expose load_state_dict for resumability")
    method(state)


def _model_output(teacher: nn.Module, image: Tensor, geometry: Mapping[str, Any], *, return_features: bool, skip_dropout: float | None = None, dropout_generator: torch.Generator | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"return_features": return_features}
    if skip_dropout is not None:
        kwargs["skip_dropout"] = float(skip_dropout)
    if dropout_generator is not None:
        kwargs["dropout_generator"] = dropout_generator
    output = teacher(image, geometry, **kwargs)
    if not isinstance(output, Mapping):
        raise TypeError("PG teacher forward must return a mapping")
    result = dict(output)
    for key in ("region_logits", "boundary_logits", "sdf_all"):
        if not isinstance(result.get(key), Tensor):
            raise ValueError(f"PG teacher output requires {key}")
    if result["region_logits"].shape[1] != 4 or result["boundary_logits"].shape != result["region_logits"].shape:
        raise ValueError("PG teacher region/boundary heads must be [B,4,D,H,W]")
    if result["sdf_all"].shape != result["region_logits"].shape:
        raise ValueError("PG teacher sdf_all must be four channels aligned to region logits")
    return result


def _sdf_targets4(labels: Tensor, geometry: Mapping[str, Any], *, radius_mm: float = 5.0) -> Tensor:
    """Compute the exact Q3D four-channel normalized SDF target."""

    if labels.ndim == 5 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 4 or labels.dtype.is_floating_point or labels.dtype.is_complex:
        raise ValueError("SDF labels must be integer [B,D,H,W]")
    if "spacing_zyx" not in geometry:
        raise ValueError("SDF target geometry requires spacing_zyx")
    from self_audit.candidate_q3d.engine import _sdf_targets

    target = _sdf_targets(labels.long(), geometry, radius_mm=float(radius_mm))
    if target.ndim != 5 or target.shape[1] != 4 or tuple(target.shape[2:]) != tuple(labels.shape[1:]):
        raise ValueError("Q3D SDF target must be [B,4,D,H,W]")
    return target


def _q3d_teacher_loss(output: Mapping[str, Any], labels: Tensor, geometry: Mapping[str, Any], *, sdf_radius: float, sdf_near: float, sdf_weight: float) -> tuple[Tensor, dict[str, Tensor]]:
    from self_audit.candidate_q3d.losses import q3d_teacher_loss

    # This adapter intentionally has exactly the three historical keys.  The
    # compatibility four-channel SDF is passed under the old ``sdf`` key.
    adapted = {
        "region_logits": output["region_logits"],
        "boundary_logits": output["boundary_logits"],
        "sdf": output["sdf_all"],
    }
    if set(adapted) != {"region_logits", "boundary_logits", "sdf"}:
        raise AssertionError("Q3D teacher adapter key contract changed")
    sdf_target = _sdf_targets4(labels, geometry, radius_mm=sdf_radius).to(device=labels.device)
    return q3d_teacher_loss(
        adapted, labels, sdf_target, lambda_distance=float(sdf_weight),
        radius_mm=float(sdf_radius), near_mm=float(sdf_near), alpha=0.5,
    )


def _public_sdf(output: Mapping[str, Any]) -> Tensor:
    sdf = output.get("sdf")
    if isinstance(sdf, Tensor) and sdf.ndim == 5 and sdf.shape[1] == 3:
        return sdf
    sdf_all = output.get("sdf_all")
    if isinstance(sdf_all, Tensor) and sdf_all.ndim == 5 and sdf_all.shape[1] == 4:
        return sdf_all[:, 1:]
    raise ValueError("PG teacher output requires public three-channel sdf or sdf_all")


def _soft_consistency(logits: Tensor, target_prob: Tensor, confident: Tensor) -> Tensor:
    if logits.shape != target_prob.shape:
        raise ValueError("teacher consistency logits/target probability shapes differ")
    # UniMatchV1 uses a detached hard argmax pseudo-label for CE.  Keeping the
    # confidence mask and denominator unchanged makes this a true masked hard
    # CE rather than a soft-distribution distillation loss.
    pseudo = target_prob.detach().argmax(dim=1)
    per = F.cross_entropy(logits, pseudo, reduction="none")
    support = confident.to(dtype=per.dtype)
    denominator = support.sum()
    if float(denominator.detach()) == 0.0:
        return logits.sum() * 0.0
    return (per * support).sum() / denominator


def _sdf_consistency(pred: Tensor, target: Tensor, confident: Tensor, *, weight: float) -> Tensor:
    if pred.shape != target.shape:
        raise ValueError("teacher SDF prediction/target shapes differ")
    support = confident.unsqueeze(1).expand_as(pred)
    denom = support.sum()
    if float(denom.detach()) == 0.0:
        return pred.sum() * 0.0
    return float(weight) * (pred - target.detach()).abs().masked_select(support).mean()


def _lambda_u(update: int, total_updates: int, *, warmup: int, ramp_fraction: float, maximum: float) -> float:
    if update < warmup:
        return 0.0
    ramp = max(1, int(math.ceil(float(total_updates) * float(ramp_fraction))))
    return float(min(float(maximum), (update - warmup + 1) / ramp * float(maximum)))


def _frame_record(frame: Any, *, patient: str, frame_index: int) -> dict[str, Any]:
    image = _frame_value(frame, "image")
    target = _frame_value(frame, "target")
    geometry = _frame_value(frame, "geometry")
    if target is None:
        raise ValueError("evaluation frame is missing target")
    return {"patient": patient, "frame": frame_index, "image": image, "target": target, "geometry": geometry}


def _evaluation_records(data: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    if isinstance(data, Mapping) and "records" in data:
        values = list(data["records"])
    elif isinstance(data, Mapping) and "labeled" in data:
        values = list(data["labeled"])
    elif hasattr(data, "labeled_keys") and callable(data.labeled_keys):
        keys = list(data.labeled_keys())
        values = [data.load(key, labeled=True) for key in keys]
    else:
        values = list(data) if isinstance(data, Iterable) and not isinstance(data, (str, bytes)) else []
    records = []
    for frame in values[:limit] if limit is not None else values:
        patient_value = _frame_value(frame, "patient_id", _frame_value(frame, "patient"))
        frame_value = _frame_value(frame, "frame_index", _frame_value(frame, "frame"))
        if patient_value is None or frame_value is None:
            raise ValueError("PG evaluation frame requires patient_id and frame_index")
        patient = str(patient_value)
        index = int(frame_value)
        records.append(_frame_record(frame, patient=patient, frame_index=index))
    return records


def _evaluate_teacher(teacher: nn.Module, data: Any, split: str, *, device: torch.device, epoch: int, update: int, evaluator: Any = None) -> dict[str, Any]:
    if evaluator is not None:
        if not callable(evaluator):
            raise TypeError("teacher evaluator must be callable")
        result = evaluator(teacher, data, split=split, path="teacher", device=device, epoch=epoch, update=update)
        if not isinstance(result, Mapping):
            raise TypeError("teacher evaluator must return a mapping")
        return dict(result)
    from .evaluation import evaluate_epoch

    records = _evaluation_records(data, limit=16 if split in {"TRAIN16", "train"} else None)
    if not records:
        raise ValueError(f"{split} evaluation has no labeled records")
    return evaluate_epoch(teacher, data=records, epoch=epoch, update=update, split=split, path="teacher", device=device)


def _metric_row(metrics: Mapping[str, Any], *, epoch: int, update: int, split: str, loss: Mapping[str, float] | None, lr: float, seconds: float, best_val: bool = False) -> dict[str, Any]:
    per = metrics.get("per_class", {})
    def value(name: str, field: str) -> Any:
        item = per.get(name, {}) if isinstance(per, Mapping) else {}
        return item.get(field)
    row: dict[str, Any] = {
        "stage": "S2", "arm": "teacher", "epoch": int(epoch), "update": int(update), "split": split, "path": "teacher",
        "loss_total": None, "loss_sup": None, "loss_pg": None, "loss_cbkd": None,
        "fg_dice": metrics.get("fg_dice"), "rv_dice": value("RV", "dice"), "myo_dice": value("MYO", "dice"), "lv_dice": value("LV", "dice"),
        "fg_hd95": metrics.get("fg_hd95"), "rv_hd95": value("RV", "hd95_mm"), "myo_hd95": value("MYO", "hd95_mm"), "lv_hd95": value("LV", "hd95_mm"),
        "fg_assd": metrics.get("fg_assd"), "rv_assd": value("RV", "assd_mm"), "myo_assd": value("MYO", "assd_mm"), "lv_assd": value("LV", "assd_mm"),
        "lr": float(lr), "seconds": float(seconds), "best_val": bool(best_val),
    }
    if split in {"TRAIN16", "train"} and loss is not None:
        row.update({"loss_total": float(loss["total"]), "loss_sup": float(loss["sup"]), "loss_pg": float(loss["u"]), "loss_cbkd": None})
    return row


def _logger_checkpoint_state(logger: EpochLogger, *, epoch: int, update: int) -> dict[str, Any]:
    """Serialize the logger boundary carried by a teacher checkpoint.

    ``EpochLogger`` predates the PG stage checkpoint contract and therefore
    does not expose a public ``state_dict`` method.  Keep the state explicit
    here rather than inferring it from receipt files during resume.  The
    row-count/offset pair is enough for :meth:`recover_from_checkpoint` to
    truncate a suffix written immediately before a crash.
    """

    return {
        "schema": "self_audit.candidate_pg.epoch_logger.v1",
        "row_count": int(len(logger.rows)),
        "epoch": int(epoch),
        "update": int(update),
        "csv_offset": int(logger.csv_path.stat().st_size if logger.csv_path.exists() else 0),
        "jsonl_offset": int(logger.jsonl_path.stat().st_size if logger.jsonl_path.exists() else 0),
    }


def _open_logger_for_resume(stage_dir: Path, payload: Mapping[str, Any] | None) -> EpochLogger:
    """Open S2's logger without losing rows from a checkpoint-before-commit crash.

    A checkpoint is intentionally written after epoch rows but before
    ``commit_epoch``.  On restart the logger constructor would otherwise use
    the older commit receipt to archive those rows before the newer checkpoint
    gets a chance to claim them.  Temporarily suppress that stale receipt when
    the checkpoint row count is newer; ``recover_from_checkpoint`` then writes
    the authoritative boundary atomically.
    """

    commit_path = stage_dir / "epochs.commit.json"
    pending = None
    if isinstance(payload, Mapping):
        pending = payload.get("epoch_logger", payload.get("metrics_receipt"))
    suppress = False
    if isinstance(pending, Mapping) and commit_path.is_file():
        try:
            committed = json.loads(commit_path.read_text(encoding="utf-8"))
            suppress = int(pending["row_count"]) > int(committed["row_count"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            suppress = False
    saved = commit_path.read_bytes() if suppress else None
    if suppress:
        commit_path.unlink()
    try:
        logger = EpochLogger(stage_dir, csv_name="epochs.csv", jsonl_name="epochs.jsonl", stage="S2", arm="teacher")
        if isinstance(pending, Mapping):
            logger.recover_from_checkpoint(pending)
        return logger
    except Exception:
        if saved is not None and not commit_path.exists():
            commit_path.write_bytes(saved)
        raise


def _patient_rows(metrics: Mapping[str, Any], *, epoch: int, split: str) -> list[dict[str, Any]]:
    rows = []
    for patient, value in sorted((metrics.get("per_patient") or {}).items()):
        per = value.get("per_class", {}) if isinstance(value, Mapping) else {}
        rows.append({"epoch": epoch, "split": split, "path": "teacher", "patient_id": str(patient),
                     "fg_dice": value.get("fg_dice"), "fg_hd95": value.get("fg_hd95"), "fg_assd": value.get("fg_assd"),
                     "rv_dice": (per.get("RV") or {}).get("dice"), "myo_dice": (per.get("MYO") or {}).get("dice"), "lv_dice": (per.get("LV") or {}).get("dice"),
                     "rv_hd95": (per.get("RV") or {}).get("hd95_mm"), "myo_hd95": (per.get("MYO") or {}).get("hd95_mm"), "lv_hd95": (per.get("LV") or {}).get("hd95_mm"),
                     "rv_assd": (per.get("RV") or {}).get("assd_mm"), "myo_assd": (per.get("MYO") or {}).get("assd_mm"), "lv_assd": (per.get("LV") or {}).get("assd_mm")})
    return rows


def _valid_for(image: Tensor) -> Tensor:
    return torch.ones((image.shape[0], *image.shape[-3:]), dtype=torch.bool, device=image.device)


def _backward_loss(loss: Tensor) -> None:
    """Accumulate one finite stream; the caller steps only after all streams."""
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("PG teacher stage produced nonfinite loss")
    loss.backward()


def run_teacher_stage(
    config: Mapping[str, Any],
    out: str | Path,
    train_data: Any,
    val_data: Any,
    test_data: Any,
    device: str | torch.device = "cpu",
    *,
    teacher: nn.Module | None = None,
    sampler: Any | None = None,
    view_generator: StatefulViewGenerator | None = None,
    resume: bool = False,
    max_updates: int | None = None,
    evaluator: Any = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Candidate PG S2 with exact labeled/U/FP stream contracts."""

    cfg = _mapping(config, "config")
    if cfg.get("schema") != "self_audit.candidate_pg.v2":
        raise ValueError("PG teacher stage requires schema self_audit.candidate_pg.v2")
    if cfg.get("seed") != 42 or seed != int(cfg["seed"]):
        raise ValueError("PG teacher stage seed is locked to 42")
    seed_value = int(cfg["seed"])
    teacher_cfg = _mapping(_required(cfg, "teacher", "config"), "teacher")
    semi = _mapping(_required(teacher_cfg, "semi", "teacher"), "teacher.semi")
    threshold_cfg = _mapping(_required(semi, "threshold", "teacher.semi"), "teacher.semi.threshold")
    lambda_cfg = _mapping(_required(semi, "lambda_u", "teacher.semi"), "teacher.semi.lambda_u")
    streams = _mapping(_required(semi, "stream_weights", "teacher.semi"), "teacher.semi.stream_weights")
    if {"s1", "s2", "fp"} - set(streams):
        raise ValueError("teacher stream_weights must contain s1, s2 and fp")
    augmentation_cfg = _mapping(_required(cfg, "augmentation", "config"), "augmentation")
    n_strong_views = int(_required(augmentation_cfg, "n_strong_views", "augmentation"))
    if n_strong_views not in {1, 2}:
        raise ValueError("PG teacher n_strong_views must be 1 or 2")
    weights = {key: float(streams[key]) for key in ("s1", "s2", "fp")}
    if n_strong_views == 1:
        # The only permitted one-view adaptation is common .5 strong/.5 FP.
        weights = {"s1": 0.5, "s2": 0.0, "fp": 0.5}
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("teacher stream weights must sum to one")
    optimizer_cfg = _mapping(_required(teacher_cfg, "optimizer", "teacher"), "teacher.optimizer")
    lr = float(_required(optimizer_cfg, "lr", "teacher.optimizer"))
    weight_decay = float(_required(optimizer_cfg, "weight_decay", "teacher.optimizer"))
    schedule_updates = int(_required(teacher_cfg, "updates", "teacher"))
    if schedule_updates <= 0 or lr <= 0 or weight_decay < 0:
        raise ValueError("teacher updates/optimizer values are invalid")
    runtime = _mapping(_required(cfg, "runtime", "config"), "runtime")
    deterministic = _required(runtime, "deterministic", "runtime")
    tf32 = _required(runtime, "tf32", "runtime")
    if deterministic is not True or tf32 is not False:
        raise ValueError("PG teacher requires deterministic=true and tf32=false")
    clip_norm = float(_required(runtime, "gradient_clip_norm", "runtime"))
    if clip_norm <= 0:
        raise ValueError("runtime.gradient_clip_norm must be positive")
    radius = float(_mapping(_required(teacher_cfg, "heads", "teacher"), "teacher.heads")["sdf"]["radius_mm"])
    sdf_consistency_cfg = _mapping(_required(semi, "sdf_consistency", "teacher.semi"), "teacher.semi.sdf_consistency")
    sdf_weight = float(_required(sdf_consistency_cfg, "weight", "teacher.semi.sdf_consistency"))
    if _required(sdf_consistency_cfg, "on", "teacher.semi.sdf_consistency") != "confident_voxels" or _required(sdf_consistency_cfg, "loss", "teacher.semi.sdf_consistency") != "l1_normalized":
        raise ValueError("PG teacher SDF consistency contract is fixed to confident_voxels/l1_normalized")
    pg_cfg = _mapping(_required(cfg, "pg", "config"), "pg")
    boundary_cfg = _mapping(_required(pg_cfg, "boundary", "pg"), "pg.boundary")
    sdf_near = float(_required(boundary_cfg, "near_mm", "pg.boundary"))
    student_cfg = _mapping(_required(cfg, "student", "config"), "student")
    fp_dropout = float(_required(student_cfg, "fp_skip_dropout", "student"))
    cpu_amp_dtype = _required(runtime, "cpu_amp_dtype", "runtime")
    gpu_amp_dtype = _required(runtime, "gpu_amp_dtype", "runtime")
    if cpu_amp_dtype != "float32" or gpu_amp_dtype != "bfloat16":
        raise ValueError("PG teacher precision is locked to CPU float32 and GPU bfloat16")
    cublas_config = str(_required(runtime, "cublas_workspace_config", "runtime"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    warmup = int(_required(lambda_cfg, "warmup_updates_labeled_only", "teacher.semi.lambda_u"))
    ramp_fraction = float(_required(lambda_cfg, "ramp_fraction", "teacher.semi.lambda_u"))
    lambda_max = float(_required(lambda_cfg, "max", "teacher.semi.lambda_u"))
    if warmup < 0 or ramp_fraction <= 0 or lambda_max < 0:
        raise ValueError("teacher lambda_u configuration is invalid")
    total_updates = schedule_updates
    if max_updates is not None:
        total_updates = min(schedule_updates, int(max_updates))
    if total_updates <= 0:
        raise ValueError("teacher stage update budget must be positive")
    target_device = torch.device(device)
    from .runtime import configure_runtime
    runtime_flags = configure_runtime(target_device, cublas_config)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_context = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if target_device.type == "cuda" else (lambda: nullcontext())
    _seed_all(seed_value)
    out_path = Path(out).resolve(); out_path.mkdir(parents=True, exist_ok=True)
    stage_dir = out_path / "S2"; stage_dir.mkdir(parents=True, exist_ok=True)
    if teacher is None:
        from .teacher_mednext import build_teacher
        teacher, _ = build_teacher(cfg)
    if not isinstance(teacher, nn.Module):
        raise TypeError("teacher must be a torch module")
    teacher.to(target_device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    if sampler is None:
        if hasattr(train_data, "labeled_keys") and hasattr(train_data, "unlabeled_keys"):
            from .data import PGSampler
            sampler = PGSampler(train_data, seed=seed_value)
        else:
            sampler = _SequenceTripleSampler(train_data, seed=seed_value)
    if view_generator is None:
        view_generator = StatefulViewGenerator(seed=seed_value + 2003)
    labeled_view = StatefulViewGenerator(seed=seed_value + 1009)
    fp_generator = torch.Generator(device=target_device.type)
    fp_generator.manual_seed(seed_value + 7003)
    threshold = ClassAdaptiveThreshold(4, ema=float(_required(threshold_cfg, "ema", "teacher.semi.threshold")), clip=_required(threshold_cfg, "clip", "teacher.semi.threshold"))
    latest = stage_dir / "teacher_latest.pt"
    start = 0; best_score = -float("inf"); best_update: int | None = None; best_state: dict[str, Tensor] | None = None
    epoch_loss_sum = {"total": 0.0, "sup": 0.0, "u": 0.0}; epoch_loss_count = 0
    history: list[dict[str, Any]] = []
    resume_payload: Mapping[str, Any] | None = None
    if resume:
        if not latest.is_file():
            raise FileNotFoundError(f"teacher resume checkpoint missing: {latest}")
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("config_sha256") != _config_hash(cfg):
            raise ValueError("teacher resume checkpoint schema/config identity mismatch")
        resume_payload = payload
        teacher.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        _load_sampler_state(sampler, payload["sampler"]); view_generator.load_state_dict(payload["augmentation"]["u"]); labeled_view.load_state_dict(payload["augmentation"]["l"])
        threshold.load_state_dict(payload["threshold"])
        fp_generator.set_state(payload["fp_generator_state"].cpu())
        _restore_rng(payload["rng"])
        start = int(payload["update"]); best_score = float(payload["best_score"]); best_update = payload.get("best_update"); history = list(payload.get("history", []))
        epoch_loss_sum = {key: float(payload["epoch_loss_sum"][key]) for key in ("total", "sup", "u")}; epoch_loss_count = int(payload["epoch_loss_count"])
        if payload.get("best_model") is not None: best_state = payload["best_model"]
    logger = _open_logger_for_resume(stage_dir, resume_payload) if resume else EpochLogger(
        stage_dir, csv_name="epochs.csv", jsonl_name="epochs.jsonl", stage="S2", arm="teacher"
    )
    print("[S2] start", flush=True)
    run_log = stage_dir / "run.log"; diag_path = stage_dir / "diag.jsonl"
    epochs = max(1, math.ceil(total_updates / 80))
    epoch_started = time.perf_counter()
    for update in range(start, total_updates):
        t0 = time.perf_counter()
        lkey, akey, bkey = _next_triple(sampler)
        lf = _load_frame(train_data, lkey, labeled=True, index=update); af = _load_frame(train_data, akey, labeled=False, index=update); bf = _load_frame(train_data, bkey, labeled=False, index=update)
        li, ly, lg, _, _ = _as_image_target(lf, target_device, labeled=True)
        lv, lstrong, ly_aug, lvalid, lg_aug = labeled_view.joint_views(li, _valid_for(li), lg, ly)
        # The ramp is locked to the full 8,000-update teacher recipe.  A
        # software-only ``max_updates`` truncation must never retune its
        # denominator or make a short smoke appear to be a different schedule.
        lambda_u = _lambda_u(update, schedule_updates, warmup=warmup, ramp_fraction=ramp_fraction, maximum=lambda_max)
        loss_u = li.sum() * 0.0; loss_s1 = loss_u; loss_s2 = loss_u; loss_fp = loss_u; loss_sdf = loss_u
        confident_fraction = 0.0
        if update >= warmup and lambda_u > 0.0:
            ai, _, ag, _, _ = _as_image_target(af, target_device, labeled=False)
            bi, _, bg, _, _ = _as_image_target(bf, target_device, labeled=False)
            aw, _astr, av, ag_aug = view_generator.joint_views(ai, _valid_for(ai), ag)
            if tuple(bi.shape[-3:]) != tuple(ai.shape[-3:]):
                anchor_shape = tuple(int(v) for v in ai.shape[-3:])
                original_bvalid = _valid_for(bi)
                bi = center_crop_pad(bi, anchor_shape, pad_value=0.0)
                bvalid = center_crop_pad(original_bvalid.to(dtype=torch.bool), anchor_shape, pad_value=0).bool()
                bg = dict(bg); bg["shape"] = anchor_shape
            else:
                bvalid = _valid_for(bi)
            bw, _bstr, bv, bg_aug = view_generator.joint_views(bi, bvalid, bg)
            # The weak A/B targets are detached before any packet or threshold work.
            with torch.no_grad(), amp_context():
                a_target = _model_output(teacher, aw, ag_aug, return_features=True)
                b_target = _model_output(teacher, bw, bg_aug, return_features=True)
                pa = a_target["region_logits"].softmax(1).detach(); pb = b_target["region_logits"].softmax(1).detach()
                ya = pa.argmax(1); yb = pb.argmax(1)
                wa = pa.max(1).values; wb = pb.max(1).values
                cutmix_cfg = _mapping(_required(augmentation_cfg, "cutmix", "augmentation"), "augmentation.cutmix")
                if _required(cutmix_cfg, "full_depth", "augmentation.cutmix") is not True:
                    raise ValueError("PG teacher CutMix must span full depth")
                box = cutmix_box(av & bv, view_generator.cutmix_generator, partner_valid=bv, area_fraction=_required(cutmix_cfg, "area_fraction", "augmentation.cutmix"), align_px=int(_required(cutmix_cfg, "align_px", "augmentation.cutmix")))
                sdf_a = _public_sdf(a_target).detach(); sdf_b = _public_sdf(b_target).detach()
                packet_s1 = mix_packet(pa, ya, wa, av, box, partner_p_W=pb, partner_y=yb, partner_w=wb, partner_valid=bv, extra={"sdf": sdf_a}, partner_extra={"sdf": sdf_b})
                # Strong view 2 owns a separate CutMix draw and packet.  The
                # image box, target/probability packet, SDF target, and valid
                # support must all agree for that stream; reusing box 1 would
                # silently make the two strong losses the same spatial view.
                if n_strong_views == 2:
                    box_s2 = cutmix_box(av & bv, view_generator.cutmix_generator, partner_valid=bv, area_fraction=_required(cutmix_cfg, "area_fraction", "augmentation.cutmix"), align_px=int(_required(cutmix_cfg, "align_px", "augmentation.cutmix")))
                    packet_s2 = mix_packet(pa, ya, wa, av, box_s2, partner_p_W=pb, partner_y=yb, partner_w=wb, partner_valid=bv, extra={"sdf": sdf_a}, partner_extra={"sdf": sdf_b})
                else:
                    box_s2 = box
                    packet_s2 = packet_s1
                tau = threshold.thresholds.to(device=pa.device)
                y_u = packet_s1["y"]; conf_u = packet_s1["p_W"].max(1).values
                confident = (conf_u >= tau[y_u]) & packet_s1["valid"]
                y_u_s2 = packet_s2["y"]; conf_u_s2 = packet_s2["p_W"].max(1).values
                confident_s2 = (conf_u_s2 >= tau[y_u_s2]) & packet_s2["valid"]
            # Targets retain only the detached tensors used by the losses,
            # not the unused five-level feature outputs of both weak passes.
            del a_target, b_target
            # Each strong stream applies its own intensity draw and its own
            # CutMix box.  FP remains the clean weak-A image by contract.
            s1_img = mix_tensor(view_generator.strong(aw), view_generator.strong(bw), box)
            s2_img = mix_tensor(view_generator.strong(aw), view_generator.strong(bw), box_s2) if n_strong_views == 2 else s1_img
            # FP is weak-A plus feature perturbation; it is not an intensity-strong view.
            fp_img = aw
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            # Supervision is evaluated on the labeled strong view.  Its target
            # and SDF geometry remain the spatially aligned weak-view pair;
            # intensity-only strong augmentation must not alter either.
            l_out = _model_output(teacher, lstrong, lg_aug, return_features=True)
            loss_sup, parts = _q3d_teacher_loss(l_out, ly_aug, lg_aug, sdf_radius=radius, sdf_near=sdf_near, sdf_weight=1.0)
        # The independent labeled/strong/FP graphs previously coexisted until
        # one combined backward, exceeding VRAM at the first post-warmup
        # update. Accumulate the same weighted gradients one stream at a time;
        # parameters, clipping, and optimizer state change only once below.
        _backward_loss(loss_sup)
        loss_sup = loss_sup.detach()
        del l_out, parts
        if update >= warmup and lambda_u > 0.0:
            confident_a = (wa >= tau[ya]) & av
            streams_to_run = [("s1", s1_img, packet_s1["p_W"], packet_s1["sdf"], confident)]
            if n_strong_views == 2:
                streams_to_run.append(("s2", s2_img, packet_s2["p_W"], packet_s2["sdf"], confident_s2))
            streams_to_run.append(("fp", fp_img, pa, sdf_a, confident_a))
            region_losses = {"s2": loss_sup.new_zeros(())}
            sdf_losses = {"s2": loss_sup.new_zeros(())}
            for name, stream_image, target_p, target_sdf, support in streams_to_run:
                kwargs = {"skip_dropout": fp_dropout, "dropout_generator": fp_generator} if name == "fp" else {}
                with amp_context():
                    stream_out = _model_output(teacher, stream_image, ag_aug, return_features=True, **kwargs)
                region_loss = _soft_consistency(stream_out["region_logits"], target_p, support)
                sdf_loss = _sdf_consistency(_public_sdf(stream_out), target_sdf, support, weight=1.0)
                _backward_loss(float(lambda_u) * weights[name] * (region_loss + float(sdf_weight) * sdf_loss))
                region_losses[name] = region_loss.detach()
                sdf_losses[name] = sdf_loss.detach()
                del stream_out, region_loss, sdf_loss
            loss_s1, loss_s2, loss_fp = (region_losses[name] for name in ("s1", "s2", "fp"))
            sdf_s1, sdf_s2, sdf_fp = (sdf_losses[name] for name in ("s1", "s2", "fp"))
            loss_sdf = float(sdf_weight) * (weights["s1"] * sdf_s1 + weights["s2"] * sdf_s2 + weights["fp"] * sdf_fp)
            loss_u = weights["s1"] * loss_s1 + weights["s2"] * loss_s2 + weights["fp"] * loss_fp + loss_sdf
            with torch.no_grad():
                threshold.update(torch.cat((pa, pb), dim=0), torch.cat((av, bv), dim=0))
            confident_fraction = float(confident.float().mean())
        total_loss = loss_sup + float(lambda_u) * loss_u
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError("PG teacher stage produced nonfinite loss")
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), clip_norm)
        optimizer.step()
        update_no = update + 1
        elapsed = time.perf_counter() - t0
        epoch_loss_sum["total"] += float(total_loss.detach()); epoch_loss_sum["sup"] += float(loss_sup.detach()); epoch_loss_sum["u"] += float(loss_u.detach()); epoch_loss_count += 1
        diag = {"update": update_no, "loss_total": float(total_loss.detach()), "loss_sup": float(loss_sup.detach()), "loss_u": float(loss_u.detach()), "lambda_u": float(lambda_u), "lambda_denominator_updates": int(schedule_updates), "tau_g": float(threshold.tau_g), "tau_c": [float(v) for v in threshold.thresholds], "confident_fraction": float(confident_fraction), "seconds": float(elapsed)}
        if update_no % 50 == 0 or update_no == total_updates:
            diag["memory_peak_bytes"] = torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else 0
            diag["memory_reserved_peak_bytes"] = torch.cuda.max_memory_reserved(target_device) if target_device.type == "cuda" else 0
            with diag_path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(diag, allow_nan=False, sort_keys=True) + "\n")
            with run_log.open("a", encoding="utf-8") as stream: stream.write("DEBUG " + json.dumps(diag, allow_nan=False, sort_keys=True) + "\n")
        is_epoch_boundary = update_no % 80 == 0
        if is_epoch_boundary:
            epoch = update_no // 80
            metrics_by_split = (("train", train_data), ("val", val_data), ("test", test_data))
            vals: dict[str, Any] = {}
            all_patient_rows: list[dict[str, Any]] = []
            for split, dataset in metrics_by_split:
                metrics = _evaluate_teacher(teacher, dataset, split, device=target_device, epoch=epoch, update=update_no, evaluator=evaluator)
                vals[split] = metrics
                epoch_loss = {key: epoch_loss_sum[key] / max(1, epoch_loss_count) for key in epoch_loss_sum}
                row = _metric_row(metrics, epoch=epoch, update=update_no, split=split, loss=epoch_loss, lr=lr, seconds=elapsed)
                logger.log_epoch(row)
                all_patient_rows.extend(_patient_rows(metrics, epoch=epoch, split=split))
            if all_patient_rows:
                logger.log_patient_rows(epoch, all_patient_rows)
            val_score = vals["val"].get("fg_dice")
            if val_score is None or not math.isfinite(float(val_score)):
                raise FloatingPointError("val teacher fg_dice is nonfinite")
            val_score = float(val_score)
            if best_update is None or val_score > best_score:  # strict > preserves earliest tie
                best_score = val_score; best_update = update_no; best_state = copy.deepcopy(teacher.state_dict())
            logger.mark_best(epoch=epoch if best_update == update_no else int(best_update // 80), split="val", path="teacher")
            history.append({"epoch": epoch, "update": update_no, "val_fg_dice": val_score, "test_fg_dice": vals["test"].get("fg_dice")})
            epoch_elapsed = time.perf_counter() - epoch_started
            print(epoch_console_line(
                "S2", epoch, epochs, epoch_loss["total"],
                vals["train"].get("fg_dice"), val_score, best_score,
                0 if best_update is None else int(best_update // 80),
                vals["test"].get("fg_dice"), epoch_elapsed,
            ), flush=True)
            epoch_started = time.perf_counter()
            epoch_loss_sum = {"total": 0.0, "sup": 0.0, "u": 0.0}; epoch_loss_count = 0
        # Checkpoint every non-boundary 50 updates to preserve partial-epoch
        # loss accumulators, and at every epoch boundary.  At a boundary the
        # order is deliberately rows -> atomic checkpoint -> commit receipt;
        # either crash window is deterministic on resume.
        if update_no % 50 == 0 or is_epoch_boundary or update_no == total_updates:
            logger_state = _logger_checkpoint_state(logger, epoch=update_no // 80, update=update_no)
            payload = {
                "schema": SCHEMA, "config_sha256": _config_hash(cfg), "update": update_no,
                "model": teacher.state_dict(), "optimizer": optimizer.state_dict(),
                "sampler": _sampler_state(sampler),
                "augmentation": {"l": labeled_view.state_dict(), "u": view_generator.state_dict()},
                "fp_generator_state": fp_generator.get_state().clone(), "threshold": threshold.state_dict(), "rng": _rng_state(),
                "best_score": best_score, "best_update": best_update, "best_model": best_state, "history": history,
                "epoch_loss_sum": epoch_loss_sum, "epoch_loss_count": epoch_loss_count,
                "epoch_logger": logger_state, "metrics_receipt": logger_state,
            }
            sha = _atomic_save(payload, latest)
            with (stage_dir / "teacher_latest.sha256.json").open("w", encoding="utf-8") as stream: json.dump({"path": str(latest), "sha256": sha, "update": update_no}, stream, indent=2); stream.write("\n")
            if is_epoch_boundary:
                logger.commit_epoch(epoch=update_no // 80, update=update_no)
    if best_state is not None:
        teacher.load_state_dict(best_state)
    if hasattr(teacher, "freeze") and callable(teacher.freeze): teacher.freeze()
    else: teacher.requires_grad_(False); teacher.eval()
    frozen = stage_dir / "teacher_best.pt"; frozen_sha = _atomic_save({"schema": SCHEMA, "config_sha256": _config_hash(cfg), "model": teacher.state_dict(), "selected_update": best_update, "selected_val_fg_dice": None if best_update is None else best_score}, frozen)
    receipt = {
        "schema": SCHEMA, "status": "COMPLETED", "updates": total_updates,
        "selected_update": best_update, "selected_val_fg_dice": None if best_update is None else best_score,
        "checkpoint": str(frozen), "checkpoint_sha256": frozen_sha,
        "selected_checkpoint": str(frozen), "selected_checkpoint_sha256": frozen_sha,
        "checkpoint_path": str(frozen), "latest_checkpoint": str(latest), "history": history,
        "seed": seed_value,
        "runtime": {
            "device": str(target_device), "synthetic": bool(runtime.get("synthetic", False)),
            "deterministic": bool(deterministic), "tf32": bool(tf32),
            **runtime_flags,
        },
    }
    (stage_dir / "TEACHER_BEST_RECEIPT.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("[S2] end", flush=True)
    return receipt


__all__ = ["SCHEMA", "_lambda_u", "_q3d_teacher_loss", "_sdf_targets4", "run_teacher_stage"]
