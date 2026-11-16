"""Locked finalisation for a Candidate PG campaign.

The training engine owns checkpoints and epoch monitoring.  This module owns
the small, explicit hand-off from selected in-memory students to deployment
artifacts and the one final TEST transaction.  It deliberately does not train,
select, or manufacture predictions: every metric below is recomputed from the
selected model bytes and native labelled frames.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .analysis import (
    code_spatial_shuffle_diagnostic,
    compare_logged_export,
    paired_comparisons,
)
from .evaluation import evaluate_epoch
from .export import (
    ARM_NAMES,
    evaluate_all4_transaction,
    export_student,
    load_student_export,
    sha256_file,
    verify_cold_load,
)
from .student_pg import PGStudent
from .training_step import _pooled_gt

_METRIC_FIELDS = (
    "fg_dice", "rv_dice", "myo_dice", "lv_dice",
    "fg_hd95", "rv_hd95", "myo_hd95", "lv_hd95",
    "fg_assd", "rv_assd", "myo_assd", "lv_assd",
)
_SPLIT_ALIASES = {
    "val": "VAL18", "VAL": "VAL18", "VAL18": "VAL18",
    "dev": "VAL18", "development": "VAL18",
    "test": "TEST50", "TEST": "TEST50", "TEST50": "TEST50",
    "train": "TRAIN16", "TRAIN": "TRAIN16", "TRAIN16": "TRAIN16",
}


def _json_safe(value: Any) -> Any:
    """Convert torch/numpy scalars without turning non-finite values into JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _json_safe(value.item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _capture_rng() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _restore_rng(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state[3])


@contextlib.contextmanager
def _preserve_runtime(models: Iterable[Any], teacher: Any = None):
    rng = _capture_rng()
    modules = []
    seen: set[int] = set()
    for candidate in (*list(models), teacher):
        if isinstance(candidate, nn.Module) and id(candidate) not in seen:
            seen.add(id(candidate))
            modules.extend((module, module.training) for module in candidate.modules())
    try:
        yield
    finally:
        for module, training in modules:
            module.training = training
        _restore_rng(rng)


def _normalise_split(value: Any) -> str:
    result = _SPLIT_ALIASES.get(str(value), str(value))
    if result not in {"TRAIN16", "VAL18", "TEST50"}:
        raise ValueError(f"unsupported PG finalisation split: {value!r}")
    return result


def _frame_value(frame: Any, name: str, default: Any = None) -> Any:
    if isinstance(frame, Mapping):
        return frame.get(name, default)
    return getattr(frame, name, default)


def _frames_for_split(data: Any, split: str) -> list[Any]:
    """Materialise only the labelled native frames required for finalisation."""

    canonical = _normalise_split(split)
    if hasattr(data, "train") and hasattr(data, "val") and hasattr(data, "test"):
        role = {"TRAIN16": "train", "VAL18": "val", "TEST50": "test"}[canonical]
        view = getattr(data, role)
        if not hasattr(view, "labeled_keys") or not hasattr(view, "load"):
            raise TypeError("PGData role does not expose the labelled frame API")
        return [view.load(key, labeled=True) for key in view.labeled_keys()]
    if not isinstance(data, Mapping):
        raise TypeError("PG finalisation data must be PGData or an explicit synthetic mapping")
    names = {
        "TRAIN16": ("labeled", "train", "TRAIN16"),
        "VAL18": ("val", "validation", "development", "VAL18"),
        "TEST50": ("test", "TEST50"),
    }[canonical]
    value: Any = None
    for name in names:
        if name in data:
            value = data[name]
            break
    if value is None and isinstance(data.get("frames"), Mapping):
        for name in names:
            if name in data["frames"]:
                value = data["frames"][name]
                break
    if value is None:
        raise ValueError(f"synthetic PG data is missing {canonical} frames")
    if isinstance(value, Mapping) and "frames" in value:
        value = value["frames"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"PG {canonical} data must be an iterable of frames")
    return list(value)


def _record(frame: Any, split: str) -> dict[str, Any]:
    image = _frame_value(frame, "image")
    target = _frame_value(frame, "target", _frame_value(frame, "label"))
    geometry = _frame_value(frame, "geometry")
    patient = _frame_value(frame, "patient_id", _frame_value(frame, "patient"))
    frame_index = _frame_value(frame, "frame_index", _frame_value(frame, "frame"))
    if image is None or target is None or geometry is None or patient is None:
        raise ValueError("PG finalisation frame requires image, target, geometry and patient_id")
    if not isinstance(geometry, Mapping):
        raise TypeError("PG finalisation frame geometry must be a mapping")
    # Deliberately omit any supplied prediction: finalisation must execute the
    # selected model, never replay an identity/prediction fixture.
    return {
        "patient": str(patient),
        "frame": frame_index,
        "image": image,
        "target": target,
        "geometry": dict(geometry),
        "split": _normalise_split(split),
    }


def _records(data: Any, split: str) -> list[dict[str, Any]]:
    rows = [_record(frame, split) for frame in _frames_for_split(data, split)]
    if not rows:
        raise ValueError(f"PG finalisation requires at least one {split} frame")
    return rows


def _native_volumes(records: Sequence[Mapping[str, Any]]) -> list[tuple[Any, Mapping[str, Any]]]:
    volumes: list[tuple[Any, Mapping[str, Any]]] = []
    for row in records:
        image = torch.as_tensor(row["image"], dtype=torch.float32)
        if image.ndim == 3:
            image = image[None, None]
        elif image.ndim == 4:
            image = image[None]
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("PG finalisation native volume must have [B,1,D,H,W] shape")
        volumes.append((image, dict(row["geometry"])))
    return volumes


def _selected_row(rows: Sequence[Mapping[str, Any]], arm: str, epoch: int, split: str) -> dict[str, Any]:
    wanted = _normalise_split(split)
    matches = [
        dict(row)
        for row in rows
        if _normalise_split(row.get("split")) == wanted
        and str(row.get("path")) in {"student_only", "student"}
        and int(row.get("epoch")) == epoch
    ]
    if len(matches) != 1:
        raise ValueError(f"PG {arm} requires exactly one student-only {wanted} row at epoch {epoch}")
    return matches[0]


def _compact_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    # Keep patient aggregates for paired analysis but omit the frame-level copy
    # from the receipt to avoid a second large metrics ledger.
    return {
        key: value
        for key, value in result.items()
        if key not in {"frame_metrics"}
    }


def _compare_all12(logged: Mapping[str, Any], recomputed: Mapping[str, Any]) -> dict[str, Any]:
    """Compare every fixed metric, preserving explicit absent-class ``None``."""

    comparison = compare_logged_export(logged, recomputed, fields=_METRIC_FIELDS)
    fields = comparison["fields"]
    for field, item in fields.items():
        if item.get("status") == "FAIL_MISSING_METRIC" and item.get("logged") is None and item.get("recomputed") is None:
            fields[field] = {"status": "PASS", "logged": None, "recomputed": None}
    comparison["status"] = "PASS" if all(item.get("status") == "PASS" for item in fields.values()) else "FAIL"
    return comparison


def _param_counts(model: nn.Module) -> dict[str, int]:
    return {
        "parameters_total": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameters_trainable": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
    }


def _manifest_for_transaction(data: Any, out: Path, test_records: Sequence[Mapping[str, Any]]) -> tuple[Path, str, bool]:
    """Resolve a hash-pinned TEST manifest, materialising only a declared fixture."""

    if hasattr(data, "test"):
        view = data.test
        path = Path(view.path).resolve()
        digest = str(getattr(view, "manifest_sha256", ""))
        if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
            raise ValueError("PG TEST manifest is missing or its SHA-256 is not pinned")
        return path, digest, bool(getattr(view, "synthetic", False))
    if isinstance(data, Mapping):
        candidates = ("test_manifest", "manifest_path", "manifest")
        path_value = next((data.get(name) for name in candidates
                           if isinstance(data.get(name), (str, Path)) and data.get(name)), None)
        if path_value is not None:
            path = Path(path_value).expanduser().resolve()
            digest = str(data.get("test_manifest_sha256", data.get("manifest_sha256", "")))
            if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
                raise ValueError("synthetic PG manifest path/SHA-256 mismatch")
            return path, digest, bool(data.get("synthetic", True))
    # Explicit runtime.synthetic data may not have an on-disk manifest.  Create
    # a deterministic metadata-only fixture manifest and label it clearly; no
    # labels, predictions, or efficacy values are invented here.
    payload = {
        "schema": "self_audit.candidate_pg.synthetic_test_manifest.v1",
        "role": "test", "synthetic": True, "software_only": True,
        "frames": [
            {"patient": str(row["patient"]), "frame": row.get("frame"),
             "shape": list(torch.as_tensor(row["image"]).shape[-3:]),
             "geometry": dict(row["geometry"])}
            for row in test_records
        ],
    }
    path = out / "TEST_MANIFEST.synthetic.json"
    _write_json(path, payload)
    return path, sha256_file(path), True


class _CodeHeadAdapter:
    def __init__(self, head: Any, valid: np.ndarray, device: torch.device):
        # ``code_spatial_shuffle_diagnostic`` intentionally accepts a model
        # exposing the public H_S_Q head.  Keep the adapter's public surface
        # identical while adding valid-cell masking around that real head.
        self.head, self.valid = head, torch.as_tensor(valid, dtype=torch.bool, device=device)
        self.H_S_Q = self

    def __call__(self, values: Any) -> np.ndarray:
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.valid.device)
        with torch.inference_mode():
            logits = self.head(tensor)
            if isinstance(logits, Mapping):
                logits = logits.get("q_logits", logits.get("logits"))
            if not torch.is_tensor(logits):
                raise TypeError("H_S_Q returned no logits")
            logits = logits.detach().clone()
            # qS Dice is reported over valid cells only; padded cells are forced
            # to background for both baseline and shuffled calls.
            valid = self.valid
            if valid.shape[1] == 1 and logits.shape[1] != 1:
                invalid = ~valid
                logits[:, 1:] = logits[:, 1:].masked_fill(invalid.expand(-1, logits.shape[1] - 1, -1, -1, -1), -1e9)
        return logits.cpu().numpy()


def _masked_qs_dice(logits: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float | None:
    pred = np.argmax(logits, axis=1)
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    truth = np.asarray(target)
    if truth.ndim == 4 and truth.shape[0] == 1:
        truth = truth[0]
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim == 5:
        mask = mask[:, 0]
    if mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if pred.shape != truth.shape or pred.shape != mask.shape:
        raise ValueError("qS diagnostic tensors have mismatched pooled shapes")
    scores = []
    for class_id in (1, 2, 3):
        ref = (truth == class_id) & mask
        guess = (pred == class_id) & mask
        denominator = int(ref.sum() + guess.sum())
        if denominator:
            scores.append(float(2 * np.count_nonzero(ref & guess) / denominator))
    return float(np.mean(scores)) if scores else None


def _shuffle_arm(model: Any, records: Sequence[Mapping[str, Any]], *, device: str | torch.device) -> dict[str, Any]:
    branch = getattr(model, "code_branch", None)
    head = getattr(branch, "H_S_Q", None)
    if branch is None or not callable(head):
        raise ValueError("A3/A4 code-shuffle requires the selected student's code branch")
    module_flags = [(module, module.training) for module in model.modules()] if isinstance(model, nn.Module) else []
    rows: list[dict[str, Any]] = []
    try:
        if isinstance(model, nn.Module):
            # PGStudent only emits the train-only code branch while training;
            # all stochastic paths are disabled and original flags restored.
            model.train(True)
        for index, row in enumerate(records):
            image = torch.as_tensor(row["image"], dtype=torch.float32, device=device)
            if image.ndim == 3:
                image = image[None, None]
            elif image.ndim == 4:
                image = image[None]
            target = torch.as_tensor(row["target"], dtype=torch.long, device=device)
            if target.ndim == 3:
                target = target[None]
            with torch.inference_mode():
                output = model(image, row["geometry"], return_stage3=True, return_code=True)
            code = output.get("code_branch") if isinstance(output, Mapping) else None
            if not isinstance(code, Mapping):
                raise ValueError("selected PG student did not return its code branch")
            values = code.get("e_nearest", code.get("quantized"))
            valid = code.get("valid_mask")
            if not torch.is_tensor(values) or not torch.is_tensor(valid):
                raise ValueError("PG qS diagnostic requires code vectors and valid_mask")
            pooled_target, support = _pooled_gt(target, row["geometry"], model, values.shape[-3:])
            target_q = pooled_target.argmax(1)
            valid = valid.bool() & (support[:, None] if valid.ndim == 5 else support)
            values_np = values.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy().astype(bool)
            if valid_np.ndim == 4:
                valid_np = valid_np[:, None]
            if valid_np.shape[1] == 1:
                valid_for_shuffle = np.broadcast_to(valid_np, values_np.shape)
            else:
                valid_for_shuffle = valid_np
            adapter = _CodeHeadAdapter(head, valid_np, torch.device(device))
            baseline_logits = adapter(values_np)
            target_for_helper = target_q.detach().cpu().numpy()
            if valid_np.shape[1] == 1:
                target_for_helper = np.where(valid_np[:, 0], target_for_helper, 0)
            diagnostic = code_spatial_shuffle_diagnostic(
                adapter, values_np, valid_mask=valid_for_shuffle,
                baseline_logits=baseline_logits, target=target_for_helper[0],
                seed=42 + index,
            )
            base_dice = _masked_qs_dice(baseline_logits, target_q.detach().cpu().numpy(), valid_np)
            # The helper intentionally returns summary deltas, not shuffled
            # logits.  Its valid-cell mask is the contract for the shuffle; use
            # its reported qS Dice and retain an explicit drop receipt.
            shuffled_dice = diagnostic.get("shuffled_fg_dice")
            rows.append({
                "patient": str(row["patient"]), "frame": row.get("frame"),
                "changed_positions": int(diagnostic.get("changed_positions", 0)),
                "baseline_qs_dice": base_dice if base_dice is not None else diagnostic.get("baseline_fg_dice"),
                "shuffled_qs_dice": shuffled_dice,
                "drop_qs_dice": ((base_dice if base_dice is not None else diagnostic.get("baseline_fg_dice")) - shuffled_dice)
                if shuffled_dice is not None and (base_dice is not None or diagnostic.get("baseline_fg_dice") is not None) else None,
            })
    finally:
        for module, training in module_flags:
            module.training = training
    baseline = [row["baseline_qs_dice"] for row in rows if row.get("baseline_qs_dice") is not None]
    shuffled = [row["shuffled_qs_dice"] for row in rows if row.get("shuffled_qs_dice") is not None]
    return {
        "status": "PASS" if rows else "NOT_RUN_NO_QS_OUTPUT",
        "frames": len(rows), "rows": rows,
        "baseline_qs_dice": float(np.mean(baseline)) if baseline else None,
        "shuffled_qs_dice": float(np.mean(shuffled)) if shuffled else None,
        "drop_qs_dice": (float(np.mean(baseline)) - float(np.mean(shuffled))) if baseline and shuffled else None,
        "valid_cells_only": True, "training": False,
    }


def _write_plot_report(out: Path, analysis_path: Path, epoch_logs: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    from scripts.pg_report import build_report
    candidates = [out / "arms" / arm for arm in ARM_NAMES if (out / "arms" / arm).is_dir()]
    if not candidates:
        candidates = [out / "finalization" / "metrics"]
    reports = {}
    for candidate in candidates:
        arm = candidate.name if candidate.name in ARM_NAMES else "campaign"
        reports[arm] = build_report(metrics_dir=candidate, paired_json=analysis_path,
                                    out_dir=out / "report" / arm)
    return {"status": "PASS" if reports else "NOT_RUN", "reports": reports}


def finalize_campaign(
    config: Mapping[str, Any],
    out: str | Path,
    selected_models: Mapping[str, nn.Module],
    selected_epochs: Mapping[str, int],
    epoch_logs: Mapping[str, Sequence[Mapping[str, Any]]],
    data: Any,
    *,
    teacher: Any,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Finalize all four PG arms from already-selected trained models.

    The function is intentionally strict about the four-arm/selected-epoch
    contract and all twelve logged metrics.  A mismatch raises before analysis
    or TEST commit; the existing TEST transaction helper remains the only
    writer of ``TEST_TRANSACTION.json``.
    """

    if not isinstance(config, Mapping):
        raise TypeError("PG finalisation config must be a mapping")
    if set(selected_models) != set(ARM_NAMES) or set(selected_epochs) != set(ARM_NAMES) or set(epoch_logs) != set(ARM_NAMES):
        raise ValueError("PG finalisation requires exactly selected models, epochs and logs for A1/A2/A3/A4")
    for arm in ARM_NAMES:
        if not isinstance(selected_models[arm], PGStudent):
            raise TypeError(f"PG {arm} selected model must be an actual PGStudent")
        if type(selected_epochs[arm]) is not int or selected_epochs[arm] < 0:
            raise ValueError(f"PG {arm} selected epoch must be a non-negative integer")
        if not isinstance(epoch_logs[arm], Sequence) or isinstance(epoch_logs[arm], (str, bytes)):
            raise TypeError(f"PG {arm} epoch logs must be a sequence")
    root = Path(out).resolve()
    exports_dir = root / "exports"
    root.mkdir(parents=True, exist_ok=True)

    with _preserve_runtime(selected_models.values(), teacher):
        # Materialise the labeled TRAIN view as a contract check.  Final
        # metrics intentionally use only VAL/TEST; TRAIN never influences
        # selection or the one-shot TEST transaction.
        _records(data, "TRAIN16")
        val_records = _records(data, "VAL18")
        test_records = _records(data, "TEST50")
        all_volumes = _native_volumes(val_records[:3])
        if not all_volumes:
            raise ValueError("PG finalisation requires native VAL/TEST volumes")

        receipts: dict[str, Any] = {}
        recomputed: dict[str, dict[str, Any]] = {}
        for arm in ARM_NAMES:
            epoch = selected_epochs[arm]
            rows = list(epoch_logs[arm])
            # Validate the selected row exists before writing an artifact.
            logged_val = _selected_row(rows, arm, epoch, "VAL18")
            _selected_row(rows, arm, epoch, "TEST50")
            model = selected_models[arm]
            arm_config = copy.deepcopy(dict(config))
            arm_config["pg"]["enabled"] = arm in {"A2", "A4"}
            arm_config["cbkd"]["enabled"] = arm in {"A3", "A4"}
            receipt = export_student(
                model, arm_config, exports_dir / f"{arm}.pt", epoch=epoch,
                metadata={"arm": arm, "selected_epoch": epoch,
                          "synthetic": bool(config.get("runtime", {}).get("synthetic", False))
                          if isinstance(config.get("runtime", {}), Mapping) else False,
                          **_param_counts(model)},
                verify=True,
            )
            receipt["cold_load"] = verify_cold_load(model, receipt["path"], all_volumes, device=device)
            exported_model = load_student_export(receipt["path"], device=device)
            val_result = evaluate_epoch(exported_model, data=val_records, epoch=epoch,
                                        split="VAL18", path="student_only", device=device)
            comparison = _compare_all12(logged_val, val_result)
            if comparison["status"] != "PASS":
                raise ValueError(f"PG {arm} VAL18 selected export metrics mismatch: {json.dumps(_json_safe(comparison), sort_keys=True)}")
            receipts[arm] = receipt
            recomputed[arm] = {"epoch": epoch, "parameter_counts": _param_counts(model),
                               "VAL18": _compact_metrics(val_result)}

        manifest_path, manifest_sha, synthetic_manifest = _manifest_for_transaction(data, root, test_records)

        def _final_test_evaluator(models: Mapping[str, nn.Module], *, manifest_path: Path, device: str | torch.device) -> dict[str, Any]:
            values = {}
            for arm in ARM_NAMES:
                values[arm] = evaluate_epoch(models[arm], data=test_records, epoch=selected_epochs[arm],
                                             split="TEST50", path="student_only", device=device)
                logged = _selected_row(epoch_logs[arm], arm, selected_epochs[arm], "TEST50")
                comparison = _compare_all12(logged, values[arm])
                if comparison["status"] != "PASS":
                    raise ValueError(f"PG {arm} TEST50 selected export metrics mismatch: {json.dumps(_json_safe(comparison), sort_keys=True)}")
                recomputed[arm]["TEST50"] = _compact_metrics(values[arm])
            return {arm: _compact_metrics(value) for arm, value in values.items()}

        transaction = evaluate_all4_transaction(
            receipts, root, _final_test_evaluator, role="test", device=device,
            expected_manifest_sha256=manifest_sha, manifest_path=manifest_path,
            monitoring_receipt={"status": "AUTHORIZED_PER_EPOCH_MONITORING", "synthetic": synthetic_manifest},
        )
        final_test = transaction.get("metrics")
        if not isinstance(final_test, Mapping):
            raise TypeError("PG TEST transaction did not return per-arm metrics")
        paired_input = {arm: {"TEST50": final_test[arm]} for arm in ARM_NAMES}
        requested_metrics = ("FG", "RV.dice", "MYO.dice", "LV.dice",
                             "RV.hd95_mm", "MYO.hd95_mm", "LV.hd95_mm",
                             "RV.assd_mm", "MYO.assd_mm", "LV.assd_mm")
        # Existing patient-macro conventions intentionally emit ``None`` for a
        # class absent from every frame.  Keep the five predeclared pairings,
        # but pass only metrics with finite paired values to the existing
        # bootstrap/Wilcoxon helper rather than inventing a distance of zero.
        def _available(metric: str) -> bool:
            for value in final_test.values():
                patients = value.get("per_patient", {}) if isinstance(value, Mapping) else {}
                for patient in patients.values():
                    if metric == "FG":
                        scalar = patient.get("fg_dice")
                    else:
                        cls, field = metric.split(".", 1)
                        scalar = patient.get("per_class", {}).get(cls, {}).get(field)
                    if scalar is None or not math.isfinite(float(scalar)):
                        return False
            return True
        available_metrics = tuple(metric for metric in requested_metrics if _available(metric))
        if not available_metrics:
            raise ValueError("PG final TEST lacks any finite patient metrics for paired analysis")
        analysis = paired_comparisons(paired_input, split="TEST50", metrics=available_metrics,
                                      bootstrap_resamples=10_000, seed=42, opened_test=True)
        analysis["metrics_requested"] = list(requested_metrics)
        analysis["metrics_skipped_absent_class"] = [metric for metric in requested_metrics if metric not in available_metrics]
        analysis_path = root / "ANALYSIS.json"
        _write_json(analysis_path, analysis)

        shuffle: dict[str, Any] = {}
        for arm in ("A3", "A4"):
            result = _shuffle_arm(selected_models[arm], val_records, device=device)
            result["arm"] = arm
            shuffle[arm] = result
        shuffle_payload = {"schema": "self_audit.candidate_pg.shuffle.v1", "status": "SOFTWARE_ONLY" if synthetic_manifest else "REAL_DATA", "arms": shuffle}
        shuffle_path = root / "SHUFFLE.json"
        _write_json(shuffle_path, shuffle_payload)

        final_metrics = {
            "schema": "self_audit.candidate_pg.final_metrics.v1",
            "status": "SOFTWARE_ONLY" if synthetic_manifest else "COMMITTED",
            "synthetic": synthetic_manifest,
            "selected_epochs": dict(selected_epochs),
            "arms": recomputed,
            "exports": receipts,
            "test_transaction": {"status": transaction.get("status"), "path": str(root / "TEST_TRANSACTION.json"),
                                 "manifest": str(manifest_path), "manifest_sha256": manifest_sha},
        }
        final_metrics_path = root / "FINAL_METRICS.json"
        _write_json(final_metrics_path, final_metrics)
        report = _write_plot_report(root, analysis_path, epoch_logs)
        result = {
            "schema": "self_audit.candidate_pg.finalization.v1",
            "status": final_metrics["status"], "synthetic": synthetic_manifest,
            "selected_epochs": dict(selected_epochs), "exports": receipts,
            "final_metrics": str(final_metrics_path), "analysis": str(analysis_path),
            "shuffle": str(shuffle_path), "report": report,
            "test_transaction": transaction,
        }
        _write_json(root / "FINALIZATION.json", result)
        return result


__all__ = ["finalize_campaign"]
