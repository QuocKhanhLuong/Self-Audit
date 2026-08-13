"""Stack slice-level Self-Audit predictions back into patient volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F


COMPARISON_MODES = (
    "initial_only",
    "always_accept_refinement",
    "self_audit",
    "oracle_accept",
)
ComparisonMode = Literal[
    "initial_only",
    "always_accept_refinement",
    "self_audit",
    "oracle_accept",
]


@dataclass
class VolumeInferenceResult:
    """Prediction and optional recurrent trace for one patient volume."""

    prediction: torch.Tensor
    initial_prediction: torch.Tensor
    accepted_turns: list[int]
    halted_turns: list[int]
    details: list[dict[str, Any]]


def canonicalize_depth_first(volume: Any, depth_axis: int | None = None) -> np.ndarray:
    """Return a 3-D array in ``[Z,H,W]`` order without through-plane resize."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {array.shape}")
    if depth_axis is None:
        candidates = [axis for axis, size in enumerate(array.shape) if size <= 64]
        depth_axis = candidates[0] if len(candidates) == 1 else int(np.argmin(array.shape))
    if depth_axis not in (0, 1, 2):
        raise ValueError(f"depth_axis must be 0, 1, or 2, got {depth_axis}")
    return np.moveaxis(array, depth_axis, 0).astype(np.float32, copy=False)


def normalize_volume(volume: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Apply the locked volume-wise percentile clip and z-score."""

    finite = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(finite).all():
        finite = np.nan_to_num(finite, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(finite, [float(lower), float(upper)])
    clipped = np.clip(finite, low, high)
    mean = float(clipped.mean())
    std = float(clipped.std())
    return (clipped - mean) / max(std, 1e-6)


def build_25d_batch(volume_zhw: np.ndarray, image_size: int = 256) -> torch.Tensor:
    """Build ``[Z,3,H,W]`` inputs with replicated boundary neighbors."""

    volume = np.asarray(volume_zhw, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected [Z,H,W], got {volume.shape}")
    z, _, _ = volume.shape
    rows = []
    for index in range(z):
        previous = max(index - 1, 0)
        following = min(index + 1, z - 1)
        rows.append(np.stack([volume[previous], volume[index], volume[following]], axis=0))
    batch = torch.from_numpy(np.stack(rows, axis=0)).float()
    if image_size > 0 and tuple(batch.shape[-2:]) != (image_size, image_size):
        batch = F.interpolate(batch, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return batch


def reconstruct_volume(predictions: Any, num_slices: int | None = None) -> torch.Tensor:
    """Reconstruct ``[Z,H,W]`` labels from 2-D predictions.

    ``predictions`` may be a list of ``[H,W]`` labels, ``[Z,H,W]`` labels, or
    logits/probabilities with a class dimension ``[Z,C,H,W]``.
    """

    if isinstance(predictions, (list, tuple)):
        if not predictions:
            raise ValueError("Cannot reconstruct a volume from zero predictions")
        tensor = torch.stack([
            value if torch.is_tensor(value) else torch.as_tensor(value)
            for value in predictions
        ])
    else:
        tensor = predictions if torch.is_tensor(predictions) else torch.as_tensor(predictions)
    if tensor.ndim == 4:
        tensor = tensor.argmax(dim=1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected [Z,H,W] or [Z,C,H,W], got {tuple(tensor.shape)}")
    if num_slices is not None and int(num_slices) != tensor.shape[0]:
        raise ValueError(f"Expected {num_slices} slices, got {tensor.shape[0]}")
    return tensor.long()


def _extract_logits(output: Any, initial: bool = False) -> torch.Tensor:
    if isinstance(output, dict):
        keys = (
            ("initial_logits", "a0_logits", "A0_logits", "logits")
            if initial
            else ("logits", "A_t", "annotation_logits", "output", "candidate_logits")
        )
        for key in keys:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(output):
        return output
    raise TypeError("Model output does not contain tensor annotation logits")


def _model_predict(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: ComparisonMode,
    tau_accept: float,
    t_max: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Call either the new ``infer`` API or a plain module."""

    with torch.no_grad():
        if hasattr(model, "infer"):
            output = model.infer(batch, mode=mode, tau_accept=tau_accept, t_max=t_max)
        else:
            output = model(batch)
    logits = _extract_logits(output, initial=False)
    initial_logits = _extract_logits(output, initial=True)
    detail = output if isinstance(output, dict) else {"output": output}
    return logits, initial_logits, detail


def infer_patient_volume(
    model: torch.nn.Module,
    volume: Any,
    *,
    image_size: int = 256,
    depth_axis: int | None = None,
    mode: ComparisonMode = "self_audit",
    tau_accept: float = 0.0,
    t_max: int = 3,
    device: str | torch.device | None = None,
    batch_size: int = 8,
) -> VolumeInferenceResult:
    """Run 2.5-D slice inference and return a stacked patient volume.

    The function never accepts a GT argument.  ``oracle_accept`` is reserved
    for an evaluation wrapper that supplies GT after this deployable path and
    must not be used as a model inference input.
    """

    if mode not in COMPARISON_MODES:
        raise ValueError(f"mode must be one of {COMPARISON_MODES}, got {mode!r}")
    if mode == "oracle_accept":
        raise ValueError("oracle_accept requires an explicit analysis wrapper; GT is not an inference input")
    if t_max < 0:
        raise ValueError("t_max must be non-negative")
    prepared = normalize_volume(canonicalize_depth_first(volume, depth_axis=depth_axis))
    inputs = build_25d_batch(prepared, image_size=image_size)
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    predictions: list[torch.Tensor] = []
    initial_predictions: list[torch.Tensor] = []
    details: list[dict[str, Any]] = []
    try:
        for start in range(0, inputs.shape[0], max(int(batch_size), 1)):
            batch = inputs[start : start + max(int(batch_size), 1)].to(target_device)
            logits, initial_logits, detail = _model_predict(model, batch, mode, tau_accept, t_max)
            predictions.append(logits.argmax(dim=1).cpu())
            initial_predictions.append(initial_logits.argmax(dim=1).cpu())
            details.append(detail)
    finally:
        if model_was_training:
            model.train()
    prediction = reconstruct_volume(torch.cat(predictions, dim=0), num_slices=inputs.shape[0])
    initial_prediction = reconstruct_volume(torch.cat(initial_predictions, dim=0), num_slices=inputs.shape[0])
    accepted = []
    halted = []
    for item in details:
        values = item.get("accepted_turns", []) if isinstance(item, dict) else []
        accepted.extend(int(v) for v in values)
        values = item.get("halted_turns", []) if isinstance(item, dict) else []
        halted.extend(int(v) for v in values)
    return VolumeInferenceResult(prediction, initial_prediction, accepted, halted, details)


def evaluate_comparison_modes(
    model: torch.nn.Module,
    volume: Any,
    ground_truth: Any,
    *,
    image_size: int = 256,
    depth_axis: int | None = None,
    tau_accept: float = 0.0,
    t_max: int = 3,
    device: str | torch.device | None = None,
    batch_size: int = 8,
    metrics_fn: Any | None = None,
) -> dict[str, Any]:
    """Evaluate all requested comparison modes, including GT-only oracle analysis.

    The oracle path is intentionally isolated here: it is an analysis helper
    and never part of ``infer_patient_volume`` or deployable model inference.
    """

    from .metrics import annotation_metrics

    metric_function = metrics_fn or annotation_metrics
    results: dict[str, Any] = {}
    for mode in ("initial_only", "always_accept_refinement", "self_audit"):
        result = infer_patient_volume(
            model,
            volume,
            image_size=image_size,
            depth_axis=depth_axis,
            mode=mode,
            tau_accept=tau_accept,
            t_max=t_max,
            device=device,
            batch_size=batch_size,
        )
        results[mode] = {"inference": result}

    prepared = normalize_volume(canonicalize_depth_first(volume, depth_axis=depth_axis))
    target = canonicalize_depth_first(ground_truth, depth_axis=depth_axis).astype(np.int64, copy=False)
    target_tensor = torch.from_numpy(target)
    if tuple(target_tensor.shape[-2:]) != (image_size, image_size):
        target_tensor = F.interpolate(
            target_tensor.unsqueeze(1).float(),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(1).long()
    inputs = build_25d_batch(prepared, image_size=image_size)
    oracle_predictions: list[Tensor] = []
    oracle_initial: list[Tensor] = []
    oracle_details: list[dict[str, Any]] = []
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for start in range(0, inputs.shape[0], max(int(batch_size), 1)):
            batch = inputs[start : start + max(int(batch_size), 1)].to(target_device)
            gt_batch = target_tensor[start : start + max(int(batch_size), 1)].to(target_device)
            with torch.no_grad():
                output = model.infer(
                    batch,
                    mode="oracle_accept",
                    oracle_target=gt_batch,
                    tau_accept=tau_accept,
                    t_max=t_max,
                )
            logits = _extract_logits(output, initial=False)
            initial_logits = _extract_logits(output, initial=True)
            oracle_predictions.append(logits.argmax(dim=1).cpu())
            oracle_initial.append(initial_logits.argmax(dim=1).cpu())
            oracle_details.append(output)
    finally:
        if was_training:
            model.train()
    oracle_result = VolumeInferenceResult(
        reconstruct_volume(torch.cat(oracle_predictions, dim=0), num_slices=target.shape[0]),
        reconstruct_volume(torch.cat(oracle_initial, dim=0), num_slices=target.shape[0]),
        [],
        [],
        oracle_details,
    )
    resized_ground_truth = target_tensor.numpy()
    for mode in ("initial_only", "always_accept_refinement", "self_audit"):
        results[mode]["metrics"] = metric_function(results[mode]["inference"].prediction, resized_ground_truth)
    results["oracle_accept"] = {"inference": oracle_result, "metrics": metric_function(oracle_result.prediction, resized_ground_truth)}
    return results


def compare_initial_and_audited(
    initial_prediction: Any,
    audited_prediction: Any,
    target: Any,
    *,
    metrics_fn: Any,
) -> dict[str, Any]:
    """Small analysis helper for deployable-vs-oracle reporting."""

    initial = metrics_fn(initial_prediction, target)
    audited = metrics_fn(audited_prediction, target)
    return {
        "initial_only": initial,
        "audited": audited,
        "delta_dice": float(audited["dice"] - initial["dice"]),
    }
