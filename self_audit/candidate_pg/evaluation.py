"""Evaluation primitives for Candidate PG.

The PG track deliberately keeps its evaluator independent from the training
engine.  It accepts either frozen prediction records or a small data/model
adapter and always computes metrics at the native patient level.  No resize or
crop is performed here; a shape mismatch is an error.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_audit.candidate_q3d.evaluation import (
    FOREGROUND_NAMES,
    aggregate_patient_class_dice,
    frame_class_dice,
    surface_distances_3d,
)

CLASS_NAMES = ("RV", "MYO", "LV")
PATHS = ("teacher", "student_only", "fusion", "qT")
SPLITS = ("TRAIN16", "VAL18", "TEST50")
METRIC_COLUMNS = (
    "stage", "arm", "epoch", "update", "split", "path", "loss_total", "loss_sup",
    "loss_pg", "loss_cbkd", "fg_dice", "rv_dice", "myo_dice", "lv_dice",
    "fg_hd95", "rv_hd95", "myo_hd95", "lv_hd95", "fg_assd", "rv_assd", "myo_assd",
    "lv_assd", "lr", "seconds", "best_val",
)


def _spacing(geometry: Mapping[str, Any] | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(geometry, Mapping):
        value = geometry.get("spacing_zyx", geometry.get("spacing_zyx_mm"))
    else:
        value = geometry
    if value is None or len(value) != 3:
        raise ValueError("native PG geometry requires spacing_zyx in mm")
    result = tuple(float(v) for v in value)
    if any(not math.isfinite(v) or v <= 0 for v in result):
        raise ValueError("PG spacing must be finite and positive")
    return result


def _as_label(value: Any, shape: Sequence[int] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 5 and array.shape[:2] == (1, 1):
        array = array[0, 0]
    elif array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("PG labels/predictions must be native [D,H,W]")
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError("PG prediction and target native shapes differ")
    if not np.isfinite(array).all():
        raise ValueError("PG labels/predictions contain NaN or Inf")
    rounded = array.astype(np.int64, copy=False)
    if not np.equal(array, rounded).all() or not np.isin(rounded, (0, 1, 2, 3)).all():
        raise ValueError("PG labels/predictions must use class IDs 0..3")
    return rounded.astype(np.uint8, copy=False)


def _extract_logits(output: Any, path: str) -> Any:
    if torch.is_tensor(output):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("PG model output must be a Tensor or mapping")
    aliases = {
        "teacher": ("teacher_logits", "region_logits", "logits"),
        "student_only": ("student_only_logits", "student_logits", "logits"),
        "fusion": ("fusion_logits", "fusion_weak_logits", "logits"),
        "fusion_weak": ("fusion_weak_logits", "fusion_logits", "logits"),
        "qT": ("qT_logits", "qt_logits", "q_logits", "logits"),
    }
    for key in aliases.get(path, ("logits",)):
        if torch.is_tensor(output.get(key)):
            return output[key]
    raise ValueError(f"PG output does not contain logits for path {path!r}")


def _prediction_from_output(output: Any, path: str, shape: Sequence[int]) -> np.ndarray:
    logits = _extract_logits(output, path)
    if logits.ndim == 5:
        if logits.shape[0] != 1 or logits.shape[1] != 4:
            raise ValueError("PG logits must have shape [1,4,D,H,W]")
        if tuple(logits.shape[-3:]) != tuple(shape):
            raise ValueError("PG logits and native target shapes differ")
        result = logits.argmax(dim=1)[0].detach().cpu().numpy()
    elif logits.ndim == 3:
        result = logits.detach().cpu().numpy()
    else:
        raise ValueError("PG logits must have shape [1,4,D,H,W] or labels [D,H,W]")
    return _as_label(result, shape)


def _frame_metrics(reference: np.ndarray, prediction: np.ndarray,
                   spacing_zyx: Sequence[float]) -> dict[str, Any]:
    reference = _as_label(reference)
    prediction = _as_label(prediction, reference.shape)
    dice = frame_class_dice(reference, prediction)
    per_class: dict[str, Any] = {}
    for class_id, name in enumerate(CLASS_NAMES, 1):
        boundary = surface_distances_3d(reference == class_id, prediction == class_id,
                                        spacing_zyx)
        per_class[name] = {
            **dice[name],
            "hd95_mm": boundary["hd95_mm"],
            "assd_mm": boundary["assd_mm"],
            "surface_status": boundary["status"],
            "reference_voxels": int(np.count_nonzero(reference == class_id)),
            "prediction_voxels": int(np.count_nonzero(prediction == class_id)),
        }
    fg = [v["dice"] for v in per_class.values() if v["dice"] is not None]
    h95 = [v["hd95_mm"] for v in per_class.values() if v["hd95_mm"] is not None]
    assd = [v["assd_mm"] for v in per_class.values() if v["assd_mm"] is not None]
    return {
        "per_class": per_class,
        "fg_dice": float(np.mean(fg)) if fg else None,
        "fg_hd95": float(np.mean(h95)) if h95 else None,
        "fg_assd": float(np.mean(assd)) if assd else None,
    }


def aggregate_patient_metrics(frame_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average ED/ES (or all labelled) frames within patient, then patients."""
    if not frame_rows:
        return {"per_class": {name: {"dice": None, "hd95_mm": None, "assd_mm": None,
                                      "frames_with_reference": 0} for name in CLASS_NAMES},
                "fg_dice": None, "fg_hd95": None, "fg_assd": None,
                "frames": 0}
    per_class: dict[str, Any] = {}
    for name in CLASS_NAMES:
        values = [row["metrics"]["per_class"][name]["dice"] for row in frame_rows
                  if row["metrics"]["per_class"][name].get("reference_present", False)
                  and row["metrics"]["per_class"][name]["dice"] is not None]
        h95 = [row["metrics"]["per_class"][name]["hd95_mm"] for row in frame_rows
               if row["metrics"]["per_class"][name]["hd95_mm"] is not None]
        assd = [row["metrics"]["per_class"][name]["assd_mm"] for row in frame_rows
                if row["metrics"]["per_class"][name]["assd_mm"] is not None]
        per_class[name] = {
            "dice": float(np.mean(values)) if values else None,
            "hd95_mm": float(np.mean(h95)) if h95 else None,
            "assd_mm": float(np.mean(assd)) if assd else None,
            "frames_with_reference": len(values),
        }
    fg = [v["dice"] for v in per_class.values() if v["dice"] is not None]
    h95 = [v["hd95_mm"] for v in per_class.values() if v["hd95_mm"] is not None]
    assd = [v["assd_mm"] for v in per_class.values() if v["assd_mm"] is not None]
    return {
        "per_class": per_class,
        "fg_dice": float(np.mean(fg)) if fg else None,
        "fg_hd95": float(np.mean(h95)) if h95 else None,
        "fg_assd": float(np.mean(assd)) if assd else None,
        "frames": len(frame_rows),
    }


def patient_macro(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return patient-macro metrics from frame rows grouped by ``patient``."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        patient = row.get("patient", row.get("patient_id"))
        if patient is None:
            raise ValueError("PG evaluation row requires patient/patient_id")
        grouped[str(patient)].append(row)
    per_patient = {patient: aggregate_patient_metrics(values)
                   for patient, values in sorted(grouped.items())}
    def macro(field: str) -> float | None:
        values = [float(item[field]) for item in per_patient.values()
                  if item.get(field) is not None and math.isfinite(float(item[field]))]
        return float(np.mean(values)) if values else None
    classes = {}
    for name in CLASS_NAMES:
        classes[name] = {
            "dice": macro_class(per_patient, name, "dice"),
            "hd95_mm": macro_class(per_patient, name, "hd95_mm"),
            "assd_mm": macro_class(per_patient, name, "assd_mm"),
            "patient_count": sum(item["per_class"][name]["dice"] is not None
                                  for item in per_patient.values()),
        }
    return {
        "fg_dice": macro("fg_dice"), "fg_hd95": macro("fg_hd95"),
        "fg_assd": macro("fg_assd"), "per_class": classes,
        "patient_count": len(per_patient), "per_patient": per_patient,
    }


def macro_class(per_patient: Mapping[str, Mapping[str, Any]], name: str, field: str) -> float | None:
    values = [entry["per_class"][name].get(field) for entry in per_patient.values()
              if entry["per_class"][name].get(field) is not None]
    return float(np.mean(values)) if values else None


def pooled_grid_metrics(reference: Any, prediction: Any, spacing_zyx: Sequence[float],
                        pool_factor: Sequence[int] = (1, 1, 1),
                        valid_mask: Any | None = None) -> dict[str, Any]:
    """Score qT on its pooled lattice using physical cell spacing.

    ``prediction`` and ``reference`` must already be on the qT lattice.  The
    only geometry adjustment is the declared integer cell factor, so physical
    HD95/ASSD are never accidentally reported in native-voxel units.
    """
    if len(pool_factor) != 3 or any(int(v) <= 0 for v in pool_factor):
        raise ValueError("qT pool_factor must contain three positive integers")
    spacing = tuple(a * int(b) for a, b in zip(_spacing(spacing_zyx), pool_factor))
    ref = _as_label(reference); pred = _as_label(prediction, ref.shape)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != ref.shape:
            raise ValueError("qT valid_mask shape differs from pooled grid")
        # Internal pad is outside the physical lattice and contributes neither
        # foreground nor background evidence to qT metrics.
        ref = np.where(valid, ref, 0).astype(np.uint8, copy=False)
        pred = np.where(valid, pred, 0).astype(np.uint8, copy=False)
    return _frame_metrics(ref, pred, spacing)


def _iter_records(records: Any, *, split: str | None = None) -> list[dict[str, Any]]:
    if records is None:
        return []
    if isinstance(records, Mapping):
        if "records" in records:
            records = records["records"]
        else:
            # Mapping of patient -> frame records.
            flattened = []
            for patient, values in records.items():
                for value in values if isinstance(values, Sequence) and not isinstance(values, (str, bytes, Mapping)) else [values]:
                    row = dict(value) if isinstance(value, Mapping) else {"prediction": value}
                    row.setdefault("patient", str(patient)); flattened.append(row)
            records = flattened
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes)):
        raise TypeError("PG records must be an iterable of mappings")
    result = []
    for value in records:
        if not isinstance(value, Mapping):
            raise TypeError("each PG record must be a mapping")
        row = dict(value)
        if split is None or row.get("split", split) == split:
            result.append(row)
    return result


def evaluate_records(records: Iterable[Mapping[str, Any]], *, path: str = "student_only",
                     split: str | None = None, epoch: int | None = None,
                     pool_factor: Sequence[int] = (1, 1, 1)) -> dict[str, Any]:
    """Evaluate frozen frame records; useful for logs and export recomputation."""
    if path == "fusion_weak":
        path = "fusion"
    rows = []
    for raw in _iter_records(records, split=split):
        patient = raw.get("patient", raw.get("patient_id"))
        if patient is None:
            raise ValueError("PG record requires patient/patient_id")
        target = raw.get("target", raw.get("reference", raw.get("label")))
        prediction = raw.get("prediction", raw.get("predicted", raw.get("mask")))
        if target is None or prediction is None:
            raise ValueError("PG record requires target and prediction")
        reference = _as_label(target)
        pred_array = np.asarray(prediction)
        if pred_array.ndim == 5 and pred_array.shape[:2] == (1, 4):
            prediction = _prediction_from_output(torch.as_tensor(pred_array), path, reference.shape)
        geometry = raw.get("geometry", raw)
        spacing = _spacing(geometry)
        if path == "qT":
            metrics = pooled_grid_metrics(reference, prediction, spacing, pool_factor,
                                          valid_mask=raw.get("valid_mask"))
        else:
            metrics = _frame_metrics(reference, _as_label(prediction, reference.shape), spacing)
        rows.append({"patient": str(patient), "frame": raw.get("frame"), "metrics": metrics,
                     "split": raw.get("split", split), "path": path,
                     "epoch": raw.get("epoch", epoch)})
    result = patient_macro(rows)
    result.update({"schema": "self_audit.candidate_pg.evaluation.v2", "split": split,
                   "epoch": epoch, "path": path, "frame_metrics": rows})
    return result


def evaluate_epoch(model: Any = None, data: Any = None, *, records: Any = None,
                   epoch: int | None = None, update: int | None = None,
                   split: str | None = None, path: str = "student_only",
                   device: str | torch.device = "cpu", out_dir: str | Path | None = None,
                   pool_factor: Sequence[int] = (1, 1, 1), metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one epoch/path, preserving model modes and all RNG streams.

    ``records`` is preferred for deterministic replay.  For a data adapter,
    each item must expose ``patient``, ``image``, ``target`` and ``geometry``.
    """
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    module_flags = []
    if isinstance(model, torch.nn.Module):
        module_flags = [(module, module.training) for module in model.modules()]
    try:
        if records is None:
            if data is None:
                raise ValueError("PG evaluate_epoch requires records or data")
            source = data.values() if isinstance(data, Mapping) else data
            generated = []
            for item in source:
                item = dict(item) if isinstance(item, Mapping) else item
                if not isinstance(item, Mapping):
                    raise TypeError("PG data items must be mappings")
                if item.get("target", item.get("reference")) is None:
                    raise ValueError("PG evaluation data requires labels")
                pred = item.get("prediction", item.get("predicted"))
                if pred is None:
                    if model is None:
                        raise ValueError("PG evaluation requires model when predictions are absent")
                    image = torch.as_tensor(item["image"], dtype=torch.float32, device=device)
                    if image.ndim == 3:
                        image = image[None, None]
                    old = model.training if isinstance(model, torch.nn.Module) else None
                    if isinstance(model, torch.nn.Module):
                        model.eval()
                    try:
                        with torch.inference_mode():
                            pred = _prediction_from_output(
                                model(image, geometry=item["geometry"]), path,
                                _as_label(item.get("target", item.get("reference"))).shape,
                            )
                    finally:
                        if isinstance(model, torch.nn.Module):
                            model.train(old)
                generated.append({**item, "prediction": pred})
            records = generated
        result = evaluate_records(records, path=path, split=split, epoch=epoch,
                                  pool_factor=pool_factor)
        result.update({"update": update if update is not None else int(epoch or 0) * 80,
                       **(dict(metadata or {}))})
        if out_dir is not None:
            write_epoch_metrics(out_dir, [result], stage="evaluation")
        return result
    finally:
        for module, flag in module_flags:
            module.training = flag
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        if rng_state[3] is not None:
            torch.cuda.set_rng_state_all(rng_state[3])


def evaluate_epoch_all(models: Any, records_by_split_path: Mapping[str, Any], *,
                       epoch: int, update: int | None = None,
                       device: str | torch.device = "cpu",
                       pool_factor: Sequence[int] = (1, 1, 1)) -> list[dict[str, Any]]:
    """Evaluate every requested split/path for one epoch.

    ``records_by_split_path`` accepts ``{split: {path: records}}`` or
    ``{(split, path): records}``; model may be one model or the same nested
    mapping.  The returned list is intentionally one row per split/path.
    """
    results = []
    if not isinstance(records_by_split_path, Mapping):
        raise TypeError("PG records_by_split_path must be a mapping")
    nested = all(isinstance(key, tuple) and len(key) == 2 for key in records_by_split_path)
    items = []
    if nested:
        items = [((str(split), str(path)), records) for (split, path), records in records_by_split_path.items()]
    else:
        for split, values in records_by_split_path.items():
            if isinstance(values, Mapping) and any(path in values for path in PATHS):
                items.extend(((str(split), str(path)), records) for path, records in values.items())
            else:
                items.append(((str(split), "student_only"), values))
    for (split, path), records in items:
        active_model = model
        if isinstance(models, Mapping):
            active_model = models.get(path, models.get(split, models.get((split, path))))
        results.append(evaluate_epoch(active_model, records=records, epoch=epoch, update=update,
                                      split=split, path=path, device=device,
                                      pool_factor=pool_factor))
    return results


def evaluate_manifest(model: Any, manifest: Any, *, epoch: int | None = None,
                      split: str | None = None, path: str = "student_only",
                      device: str | torch.device = "cpu", out_dir: str | Path | None = None,
                      pool_factor: Sequence[int] = (1, 1, 1)) -> dict[str, Any]:
    """Compatibility adapter for a frozen record/manifest payload.

    A PG manifest adapter may materialize records directly or expose a
    ``records`` attribute.  It is intentionally not a training-data loader.
    """
    if hasattr(manifest, "records") and not isinstance(manifest, Mapping):
        manifest = manifest.records
    if isinstance(manifest, (str, Path)):
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        manifest = payload.get("records", payload.get("frames", payload))
    if isinstance(manifest, Mapping) and "records" in manifest:
        manifest = manifest["records"]
    return evaluate_epoch(model, records=manifest, epoch=epoch, split=split, path=path,
                          device=device, out_dir=out_dir, pool_factor=pool_factor)


def select_best_epoch(history: Sequence[Mapping[str, Any]], *, metric: str = "fg_dice") -> dict[str, Any]:
    """Compatibility alias for the central PG logger selector."""
    from .logging import select_best_epoch as central_select
    normalized = [dict(row, update=int(row.get("update", int(row.get("epoch", 0)) * 80)))
                  for row in history]
    return central_select(normalized, split="VAL18", path="student_only", metric=metric)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("PG metric cannot contain NaN/Inf")
    return str(value)


def _summary_row(result: Mapping[str, Any], *, stage: str, arm: str,
                 best_val: bool = False) -> dict[str, Any]:
    per = result.get("per_class", {})
    values = {
        "stage": stage, "arm": arm, "epoch": result.get("epoch"), "update": result.get("update"),
        "split": result.get("split"), "path": result.get("path"), "loss_total": result.get("loss_total"),
        "loss_sup": result.get("loss_sup"), "loss_pg": result.get("loss_pg"), "loss_cbkd": result.get("loss_cbkd"),
        "fg_dice": result.get("fg_dice"), "rv_dice": per.get("RV", {}).get("dice"),
        "myo_dice": per.get("MYO", {}).get("dice"), "lv_dice": per.get("LV", {}).get("dice"),
        "fg_hd95": result.get("fg_hd95"), "rv_hd95": per.get("RV", {}).get("hd95_mm"),
        "myo_hd95": per.get("MYO", {}).get("hd95_mm"), "lv_hd95": per.get("LV", {}).get("hd95_mm"),
        "fg_assd": result.get("fg_assd"), "rv_assd": per.get("RV", {}).get("assd_mm"),
        "myo_assd": per.get("MYO", {}).get("assd_mm"), "lv_assd": per.get("LV", {}).get("assd_mm"),
        "lr": result.get("lr"), "seconds": result.get("seconds"), "best_val": bool(best_val),
    }
    return values


def write_epoch_metrics(out_dir: str | Path, results: Sequence[Mapping[str, Any]], *, stage: str = "evaluation",
                        arm: str = "A1", best_epoch: int | None = None) -> dict[str, str]:
    """Write unique epoch/patient rows through the shared :class:`EpochLogger`."""
    from .logging import EpochLogger

    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    summaries = [_summary_row(result, stage=stage, arm=arm, best_val=False)
                 for result in results]
    key_seen: set[tuple[Any, ...]] = set()
    for row in summaries:
        key = tuple(row.get(name) for name in ("epoch", "split", "path"))
        if key in key_seen:
            raise ValueError(f"duplicate PG epoch row {key}")
        key_seen.add(key)
    logger = EpochLogger(root, csv_name="epochs.csv", jsonl_name="epochs.jsonl")
    for row in summaries:
        normalized = {key: row.get(key) for key in METRIC_COLUMNS}
        # EpochLogger's contract requires explicit integer update; standalone
        # evaluation rows use the canonical 80 updates/epoch boundary.
        normalized["epoch"] = int(normalized["epoch"] or 0)
        normalized["update"] = int(normalized["update"] if normalized["update"] is not None else normalized["epoch"] * 80)
        normalized["split"] = normalized["split"] or "VAL18"
        normalized["path"] = "fusion" if normalized["path"] == "fusion_weak" else (normalized["path"] or "student_only")
        normalized["stage"] = normalized["stage"] or stage
        normalized["arm"] = normalized["arm"] or arm
        logger.log_epoch(normalized)
    if best_epoch is not None:
        try:
            logger.mark_best(epoch=int(best_epoch), split="VAL18", path="student_only")
        except ValueError:
            # A standalone result set may not contain the selected VAL row;
            # never invent a marker or silently move selection to TEST.
            pass
    csv_path = root / "epochs.csv"
    jsonl_path = root / "epochs.jsonl"
    patient_paths = []
    by_epoch: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        by_epoch[result.get("epoch")].append(result)
    for epoch, epoch_results in by_epoch.items():
        patient_file = root / f"patients_epoch{int(epoch):03d}.csv" if epoch is not None else root / "patients.csv"
        fields = ["epoch", "split", "path", "patient_id", "fg_dice", "rv_dice", "myo_dice", "lv_dice",
                  "rv_hd95", "myo_hd95", "lv_hd95", "rv_assd", "myo_assd", "lv_assd"]
        patient_rows = []
        seen_patients: set[tuple[Any, ...]] = set()
        for result in epoch_results:
            patients = result.get("per_patient", {})
            for patient, value in sorted(patients.items()):
                identity = (epoch, result.get("split"), result.get("path"), str(patient))
                if identity in seen_patients:
                    raise ValueError(f"duplicate PG patient row {identity}")
                seen_patients.add(identity)
                per = value.get("per_class", {})
                patient_rows.append({"epoch": epoch, "split": result.get("split"), "path": result.get("path"),
                                     "patient_id": patient, "fg_dice": _csv_value(value.get("fg_dice")),
                                     "rv_dice": _csv_value(per.get("RV", {}).get("dice")),
                                     "myo_dice": _csv_value(per.get("MYO", {}).get("dice")),
                                     "lv_dice": _csv_value(per.get("LV", {}).get("dice")),
                                     "rv_hd95": _csv_value(per.get("RV", {}).get("hd95_mm")),
                                     "myo_hd95": _csv_value(per.get("MYO", {}).get("hd95_mm")),
                                     "lv_hd95": _csv_value(per.get("LV", {}).get("hd95_mm")),
                                     "rv_assd": _csv_value(per.get("RV", {}).get("assd_mm")),
                                     "myo_assd": _csv_value(per.get("MYO", {}).get("assd_mm")),
                                     "lv_assd": _csv_value(per.get("LV", {}).get("assd_mm"))})
        if patient_rows:
            # EpochLogger owns the patient CSV schema and atomic write.
            logger.log_patient_rows(int(epoch or 0), patient_rows)
            patient_paths.append(str(patient_file))
    return {"csv": str(csv_path), "jsonl": str(jsonl_path), "patient_files": patient_paths[0] if len(patient_paths) == 1 else json.dumps(patient_paths)}


__all__ = [
    "CLASS_NAMES", "FOREGROUND_NAMES", "METRIC_COLUMNS", "PATHS", "SPLITS",
    "aggregate_patient_metrics", "evaluate_epoch", "evaluate_epoch_all", "evaluate_manifest",
    "evaluate_records", "frame_class_dice",
    "patient_macro", "pooled_grid_metrics", "select_best_epoch", "surface_distances_3d",
    "write_epoch_metrics",
]
