"""Stack slice-level Self-Audit predictions back into patient volumes.

Metric spaces
-------------
Every number produced here is stamped with exactly one of the three canonical
metric spaces defined in :mod:`self_audit.audit.semantics`.  They are *not*
interchangeable and a caller must never have to infer which one it holds:

``METRIC_SPACE_SLICE_PROXY`` (``"slice_proxy"``)
    2-D per-slice Dice on the network grid.  A training/monitoring proxy.  It
    is not produced by this module; it is named here only so the contrast is
    explicit.

``METRIC_SPACE_VOLUME_RESIZED`` (``"volume_resized"``)
    3-D per-patient Dice computed on the resized network grid (256x256
    in-plane).  This is what :func:`evaluate_comparison_modes` returns.  The
    ground truth is nearest-resized *down* to the network grid before scoring,
    so the number describes the model on the preprocessed grid and nothing
    else.

``METRIC_SPACE_VOLUME_NATIVE`` (``"volume_native"``)
    3-D per-patient Dice after inverse-mapping the prediction back to the
    original acquisition grid.  Produced only by
    :func:`evaluate_volume_native`.

Why ``volume_native`` is not derivable from ``preprocessed_data/``
------------------------------------------------------------------
``scripts/preprocess_acdc.py`` records ``orig_shape``, ``orig_spacing``,
``effective_spacing`` and ``num_slices`` per case, so the native *geometry
metadata* exists.  The native *ground truth* does not.  The stored mask was
destructively resized to 256x256 with ``order=0`` nearest neighbour
(``preprocess_acdc.py:80``); the discarded detail cannot be recovered.
Inverse-resizing a prediction back to ``orig_shape`` and comparing it against
an inverse-resized copy of that same downsampled mask measures the resampler,
not the model.

Therefore a truthful ``volume_native`` number requires **real native ground
truth supplied by the caller from the raw ACDC NIfTI files**.
:func:`evaluate_volume_native` raises rather than accept a missing native GT,
so it is not possible to publish a resized number under a native label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..audit.semantics import (
    METRIC_SPACE_SLICE_PROXY,
    METRIC_SPACE_VOLUME_NATIVE,
    METRIC_SPACE_VOLUME_RESIZED,
    classify_delta,
    empty_class_score,
    macro_mean,
    resolve_empty_policy,
    resolve_neutral_margin,
)
from ..data.common import to_depth_first


#: Foreground class ids used across ACDC reporting.
DEFAULT_CLASS_NAMES: dict[int, str] = {1: "RV", 2: "MYO", 3: "LV"}

__all__ = [
    "ACDC_PHASES",
    "COMPARISON_MODES",
    "DEFAULT_CLASS_NAMES",
    "METRIC_SPACE_SLICE_PROXY",
    "METRIC_SPACE_VOLUME_NATIVE",
    "METRIC_SPACE_VOLUME_RESIZED",
    "UNKNOWN_PHASE",
    "ComparisonMode",
    "VolumeInferenceResult",
    "build_25d_batch",
    "canonicalize_depth_first",
    "compare_initial_and_audited",
    "dice_block",
    "evaluate_comparison_modes",
    "evaluate_volume_native",
    "infer_patient_volume",
    "normalize_volume",
    "per_class_dice_with_policy",
    "reconstruct_volume",
    "resolve_class_names",
    "split_cases_by_phase",
    "to_native_geometry",
]


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
    accepted_count: torch.Tensor | None = None
    halt_turn: torch.Tensor | None = None
    num_attempted_turns: torch.Tensor | None = None
    final_active: torch.Tensor | None = None


def canonicalize_depth_first(volume: Any, depth_axis: int | None = None) -> np.ndarray:
    """Return a 3-D array in ``[Z,H,W]`` order without through-plane resize."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got shape {array.shape}")
    return to_depth_first(array, depth_axis=depth_axis).astype(np.float32, copy=False)


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


def reconstruct_volume(
    predictions: Any,
    num_slices: int | None = None,
    *,
    slice_indices: Any | None = None,
) -> torch.Tensor:
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
    if slice_indices is not None:
        indices = [int(value) for value in (slice_indices.tolist() if torch.is_tensor(slice_indices) else slice_indices)]
        if len(indices) != tensor.shape[0]:
            raise ValueError(f"Expected one slice index per prediction, got {len(indices)} for {tensor.shape[0]} predictions")
        if sorted(indices) != list(range(tensor.shape[0])):
            raise ValueError(f"Slice indices must be a unique complete range, got {indices}")
        order = torch.as_tensor(np.argsort(np.asarray(indices)), dtype=torch.long, device=tensor.device)
        tensor = tensor.index_select(0, order)
    if num_slices is not None and int(num_slices) != tensor.shape[0]:
        raise ValueError(f"Expected {num_slices} slices, got {tensor.shape[0]}")
    return tensor.long()


# ---------------------------------------------------------------------------
# Class naming, empty-class-aware Dice, and native geometry
# ---------------------------------------------------------------------------


def resolve_class_names(
    class_names: Mapping[int, str] | None = None,
    num_classes: int = 4,
) -> dict[int, str]:
    """Return ``{class_id: name}`` for the foreground classes.

    Defaults to the ACDC convention ``{1: "RV", 2: "MYO", 3: "LV"}``.  Any
    foreground class without an explicit name falls back to ``"class_<id>"``
    rather than being dropped, so a macro over ``num_classes`` is never
    silently narrowed by an incomplete name map.
    """

    provided = {int(key): str(value) for key, value in (class_names or DEFAULT_CLASS_NAMES).items()}
    return {
        cls: provided.get(cls, f"class_{cls}")
        for cls in range(1, int(num_classes))
    }


def per_class_dice_with_policy(
    prediction: Any,
    target: Any,
    *,
    num_classes: int = 4,
    empty_policy: str | None = None,
) -> dict[int, float]:
    """Per-class Dice with the shared empty-class policy applied.

    Exact semantics, per :mod:`self_audit.audit.semantics`: a class that is
    empty in **both** prediction and target takes
    :func:`~self_audit.audit.semantics.empty_class_score` (``nan`` under the
    default ``"exclude"`` policy, meaning "drop from the macro").  A class
    present in exactly one of them has a non-zero denominator and therefore
    scores ``0.0`` by the ordinary Dice formula; it is **never** excluded.
    """

    empty_score = empty_class_score(empty_policy)
    pred = _label_array(prediction)
    true = _label_array(target)
    if pred.shape != true.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {true.shape}")
    scores: dict[int, float] = {}
    for cls in range(1, int(num_classes)):
        p = pred == cls
        t = true == cls
        denom = int(p.sum()) + int(t.sum())
        scores[cls] = float(empty_score) if denom == 0 else float(2.0 * int((p & t).sum()) / denom)
    return scores


def _label_array(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if array.ndim == 4:
        array = array.argmax(axis=1)
    return array.astype(np.int64, copy=False)


def dice_block(
    prediction: Any,
    target: Any,
    *,
    metric_space: str,
    num_classes: int = 4,
    class_names: Mapping[int, str] | None = None,
    empty_policy: str | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one self-describing Dice block stamped with its metric space."""

    policy = resolve_empty_policy(empty_policy)
    names = resolve_class_names(class_names, num_classes=num_classes)
    per_class = per_class_dice_with_policy(
        prediction, target, num_classes=num_classes, empty_policy=policy
    )
    excluded = sorted(cls for cls, value in per_class.items() if not np.isfinite(value))
    return {
        "metric_space": str(metric_space),
        "num_classes": int(num_classes),
        "empty_policy": policy,
        "per_class_dice": {int(cls): float(value) for cls, value in per_class.items()},
        "per_class_dice_named": {
            names[cls]: float(value) for cls, value in per_class.items()
        },
        "excluded_classes": [int(cls) for cls in excluded],
        "excluded_classes_named": [names[cls] for cls in excluded],
        "macro_dice": macro_mean(list(per_class.values())),
        "class_names": {int(cls): name for cls, name in names.items()},
        "geometry": dict(geometry) if geometry is not None else None,
    }


def _normalize_geometry(geometry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Pass declared native-geometry metadata through, verbatim and unused.

    Only the three keys written by ``scripts/preprocess_acdc.py`` are carried.
    Nothing here is computed, inferred, or defaulted: geometry the caller did
    not supply stays absent.
    """

    if geometry is None:
        return None
    if not isinstance(geometry, Mapping):
        raise TypeError(f"geometry must be a mapping of declared metadata, got {type(geometry)!r}")
    declared: dict[str, Any] = {}
    for key in ("orig_shape", "orig_spacing", "effective_spacing", "num_slices"):
        if key in geometry and geometry[key] is not None:
            value = geometry[key]
            declared[key] = (
                list(value) if isinstance(value, (list, tuple, np.ndarray)) else value
            )
    declared["source"] = "declared_metadata"
    return declared


def _nearest_indices(in_size: int, out_size: int) -> np.ndarray:
    """Half-pixel-centred nearest-neighbour index map, matching ``skimage.resize(order=0)``."""

    if in_size <= 0 or out_size <= 0:
        raise ValueError(f"Sizes must be positive, got in={in_size} out={out_size}")
    centres = (np.arange(out_size, dtype=np.float64) + 0.5) * (in_size / out_size) - 0.5
    return np.clip(np.rint(centres).astype(np.int64), 0, in_size - 1)


def to_native_geometry(
    prediction_zhw: Any,
    orig_shape: Sequence[int],
    *,
    order: int = 0,
    shape_order: str = "hwz",
) -> np.ndarray:
    """Inverse-resize an integer label volume to the original in-plane grid.

    ``prediction_zhw`` is ``[Z,H,W]`` on the network grid (typically
    ``[Z,256,256]``).  The result is ``[Z, orig_H, orig_W]``: the through-plane
    axis is never resampled and slice order is preserved exactly.

    Only ``order=0`` (nearest neighbour) is permitted.  Interpolating across
    label values would invent labels that the model never predicted, so any
    other order raises.  The output label set is asserted to be a subset of the
    input label set.

    ``orig_shape`` may be 2 values ``(H, W)`` or 3 values.  With 3 values the
    default ``shape_order="hwz"`` matches what ``scripts/preprocess_acdc.py``
    writes into ``metadata.json`` (``img_data.shape`` is ``[H,W,Z]``); pass
    ``shape_order="zhw"`` for a depth-first triple.  The depth entry must equal
    the prediction's ``Z`` -- a mismatch raises instead of being transposed
    into silence.
    """

    if int(order) != 0:
        raise ValueError(
            f"to_native_geometry only supports order=0 (nearest neighbour); got order={order!r}. "
            "Interpolating a label map across class ids fabricates labels."
        )
    labels = _label_array(prediction_zhw)
    if labels.ndim != 3:
        raise ValueError(f"Expected a [Z,H,W] label volume, got shape {labels.shape}")
    depth = int(labels.shape[0])

    values = [int(v) for v in orig_shape]
    if len(values) == 2:
        out_h, out_w = values
    elif len(values) == 3:
        key = str(shape_order).lower()
        if key == "hwz":
            out_h, out_w, declared_depth = values
        elif key == "zhw":
            declared_depth, out_h, out_w = values
        else:
            raise ValueError(f"shape_order must be 'hwz' or 'zhw', got {shape_order!r}")
        if declared_depth != depth:
            raise ValueError(
                f"orig_shape {values} under shape_order={key!r} declares depth {declared_depth}, "
                f"but the prediction has {depth} slices. preprocess_acdc.py writes orig_shape as "
                "[H,W,Z]; pass shape_order='zhw' if yours is depth-first. Refusing to guess."
            )
    else:
        raise ValueError(f"orig_shape must have 2 or 3 entries, got {values}")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"orig_shape must be positive, got (H={out_h}, W={out_w})")

    rows = _nearest_indices(int(labels.shape[1]), out_h)
    cols = _nearest_indices(int(labels.shape[2]), out_w)
    native = labels[:, rows, :][:, :, cols]

    source_labels = set(np.unique(labels).tolist())
    output_labels = set(np.unique(native).tolist())
    if not output_labels.issubset(source_labels):
        raise AssertionError(
            f"to_native_geometry introduced new label values {sorted(output_labels - source_labels)}; "
            f"input labels were {sorted(source_labels)}"
        )
    assert native.shape == (depth, out_h, out_w)
    return native.astype(labels.dtype, copy=False)


def _resolve_physical_spacing(spacing: Any, ndim: int) -> tuple[float, ...] | None:
    """Return a validated ``(z, y, x)`` spacing, or ``None`` when unavailable.

    There is no default.  A caller without real physical spacing gets ``None``
    and the surface metrics are omitted entirely rather than reported in
    fabricated millimetres.
    """

    if spacing is None:
        return None
    values = tuple(float(v) for v in spacing)
    if len(values) != int(ndim):
        raise ValueError(f"spacing must have {ndim} values in (z, y, x) order, got {values}")
    if not all(np.isfinite(v) and v > 0.0 for v in values):
        raise ValueError(f"spacing must be finite and positive, got {values}")
    return values


def _surface_distances(mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray | None:
    """Distance field to the surface of ``mask``; ``None`` when ``mask`` is empty."""

    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    eroded = ndimage.binary_erosion(mask, border_value=0)
    surface = mask & ~eroded
    if not surface.any():
        surface = mask
    return ndimage.distance_transform_edt(~surface, sampling=spacing)


def _hd95_assd(
    pred: np.ndarray, true: np.ndarray, spacing: tuple[float, ...]
) -> tuple[float, float]:
    """``(HD95, ASSD)`` in physical units for one binary class."""

    if not pred.any() and not true.any():
        return float("nan"), float("nan")
    if not pred.any() or not true.any():
        return float("inf"), float("inf")
    from scipy import ndimage

    pred_field = _surface_distances(pred, spacing)
    true_field = _surface_distances(true, spacing)
    pred_surface = pred & ~ndimage.binary_erosion(pred, border_value=0)
    true_surface = true & ~ndimage.binary_erosion(true, border_value=0)
    if not pred_surface.any():
        pred_surface = pred
    if not true_surface.any():
        true_surface = true
    p_to_t = true_field[pred_surface]
    t_to_p = pred_field[true_surface]
    both = np.concatenate([p_to_t, t_to_p])
    return float(np.percentile(both, 95)), float(both.mean())


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
    accepted_counts: list[torch.Tensor] = []
    halt_turns: list[torch.Tensor] = []
    attempted_counts: list[torch.Tensor] = []
    final_active_values: list[torch.Tensor] = []
    try:
        for start in range(0, inputs.shape[0], max(int(batch_size), 1)):
            batch = inputs[start : start + max(int(batch_size), 1)].to(target_device)
            logits, initial_logits, detail = _model_predict(model, batch, mode, tau_accept, t_max)
            predictions.append(logits.argmax(dim=1).cpu())
            initial_predictions.append(initial_logits.argmax(dim=1).cpu())
            details.append(detail)
            if isinstance(detail, dict):
                for values, destination in (
                    (detail.get("accepted_count"), accepted_counts),
                    (detail.get("halt_turn"), halt_turns),
                    (detail.get("num_attempted_turns"), attempted_counts),
                    (detail.get("final_active", detail.get("active_mask")), final_active_values),
                ):
                    if torch.is_tensor(values):
                        destination.append(values.detach().cpu())
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
    return VolumeInferenceResult(
        prediction,
        initial_prediction,
        accepted,
        halted,
        details,
        torch.cat(accepted_counts) if accepted_counts else None,
        torch.cat(halt_turns) if halt_turns else None,
        torch.cat(attempted_counts) if attempted_counts else None,
        torch.cat(final_active_values) if final_active_values else None,
    )


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
    empty_policy: str | None = None,
    neutral_margin: float | None = None,
    num_classes: int = 4,
    class_names: Mapping[int, str] | None = None,
    geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all requested comparison modes, including GT-only oracle analysis.

    The oracle path is intentionally isolated here: it is an analysis helper
    and never part of ``infer_patient_volume`` or deployable model inference.

    **Metric space.** Everything returned here is
    ``METRIC_SPACE_VOLUME_RESIZED``.  The ground truth is nearest-resized down
    to the ``image_size`` network grid before scoring (see the ``F.interpolate``
    call below), so these numbers describe the model on the preprocessed grid.
    They are *not* native-geometry Dice; for that see
    :func:`evaluate_volume_native`, which requires raw-NIfTI ground truth.

    ``geometry`` is optional **declared metadata only** (``orig_shape``,
    ``orig_spacing``, ``effective_spacing``, ``num_slices`` as recorded by
    ``scripts/preprocess_acdc.py``).  It is carried through verbatim into every
    metric block and nothing is computed from it at this level.  Geometry that
    is not supplied is not invented.

    Return shape: ``{mode: {...}}`` for each of the four comparison modes, plus
    one non-mode key ``"evaluation_meta"``.  Iterate :data:`COMPARISON_MODES`
    rather than ``results.keys()``.

    Per-mode keys ``"inference"`` and ``"metrics"`` are unchanged from before
    this revision; ``"metric_space"``, ``"volume_dice"`` and
    ``"macro_dice_delta_vs_initial"`` are new.
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
    oracle_predictions: list[torch.Tensor] = []
    oracle_initial: list[torch.Tensor] = []
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
    results["oracle_accept"] = {"inference": oracle_result}

    policy = resolve_empty_policy(empty_policy)
    margin = resolve_neutral_margin(neutral_margin)
    declared_geometry = _normalize_geometry(geometry)
    names = resolve_class_names(class_names, num_classes=num_classes)

    for mode in COMPARISON_MODES:
        prediction = results[mode]["inference"].prediction
        results[mode]["metrics"] = metric_function(prediction, resized_ground_truth)
        results[mode]["metric_space"] = METRIC_SPACE_VOLUME_RESIZED
        results[mode]["volume_dice"] = dice_block(
            prediction,
            resized_ground_truth,
            metric_space=METRIC_SPACE_VOLUME_RESIZED,
            num_classes=num_classes,
            class_names=names,
            empty_policy=policy,
            geometry=declared_geometry,
        )

    baseline = results["initial_only"]["volume_dice"]["macro_dice"]
    for mode in COMPARISON_MODES:
        delta = float(results[mode]["volume_dice"]["macro_dice"] - baseline)
        results[mode]["macro_dice_delta_vs_initial"] = delta
        results[mode]["transition_class"] = (
            int(classify_delta(delta, margin)) if np.isfinite(delta) else None
        )

    results["evaluation_meta"] = {
        "metric_space": METRIC_SPACE_VOLUME_RESIZED,
        "grid": [int(v) for v in tuple(target_tensor.shape)],
        "image_size": int(image_size),
        "empty_policy": policy,
        "neutral_margin": margin,
        "num_classes": int(num_classes),
        "class_names": {int(cls): name for cls, name in names.items()},
        "geometry": declared_geometry,
        "modes": list(COMPARISON_MODES),
        "ground_truth_source": "preprocessed_resized",
        "native_dice_available": False,
        "native_dice_note": (
            "volume_native Dice is not derivable from preprocessed_data/: the stored mask was "
            "destructively nearest-resized to the network grid. Use evaluate_volume_native with "
            "raw-NIfTI ground truth."
        ),
    }
    return results


def evaluate_volume_native(
    prediction_zhw: Any,
    native_ground_truth_zhw: Any = None,
    *,
    orig_shape: Sequence[int] | None = None,
    spacing: Sequence[float] | None = None,
    empty_policy: str | None = None,
    num_classes: int = 4,
    class_names: Mapping[int, str] | None = None,
    shape_order: str = "hwz",
    affine: Any | None = None,
    target_affine: Any | None = None,
    strict_physical: bool = False,
) -> dict[str, Any]:
    """Per-class and macro Dice at **native acquisition geometry**.

    ``native_ground_truth_zhw`` must be the real native label volume read from
    the raw ACDC NIfTI files.  It is mandatory: passing ``None`` raises
    :class:`ValueError`.  The 256x256 mask in ``preprocessed_data/`` is *not* a
    substitute -- it was destructively downsampled, so inverse-resizing it back
    up would score the resampler rather than the model.

    ``prediction_zhw`` is the network-grid ``[Z,H,W]`` prediction.  When
    ``orig_shape`` is supplied the prediction is inverse-resized with
    :func:`to_native_geometry` (nearest neighbour only); otherwise it must
    already match the native ground-truth shape.

    ``spacing`` is an optional real physical ``(z, y, x)`` voxel size in mm.
    In unverified paths without verified continuous 3D coordinate transform
    provenance, native-mm surface distance metrics (HD95/ASSD) are suppressed
    to prevent false physical millimeter claims from discrete inverse-resizing.
    ``strict_physical=True`` enforces fail-closed validation and raises
    ``ValueError`` (DEFERRED pending raw NIfTI transform chain).
    Malformed, single-sided, or spacing-inconsistent affines are rejected.
    """

    if native_ground_truth_zhw is None:
        raise ValueError(
            "evaluate_volume_native requires real native ground truth (raw ACDC NIfTI labels at "
            "the original acquisition geometry). The 256x256 mask in preprocessed_data/ is NOT a "
            "substitute: it was destructively nearest-resized by scripts/preprocess_acdc.py, so "
            "inverse-resizing it measures the resampler, not the model. Refusing to return a "
            f"metric_space={METRIC_SPACE_VOLUME_NATIVE!r} number without it."
        )

    target = _label_array(native_ground_truth_zhw)
    if target.ndim != 3:
        raise ValueError(f"native ground truth must be [Z,H,W], got shape {target.shape}")

    prediction = _label_array(prediction_zhw)
    if orig_shape is not None:
        prediction = to_native_geometry(prediction, orig_shape, order=0, shape_order=shape_order)
    elif prediction.shape != target.shape:
        prediction = to_native_geometry(
            prediction, (int(target.shape[1]), int(target.shape[2])), order=0
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"Native prediction/ground-truth shape mismatch: {prediction.shape} vs {target.shape}"
        )

    # Validate physical spacing if provided
    physical = _resolve_physical_spacing(spacing, target.ndim) if spacing is not None else None

    # Geometry and affine validation — fail closed against attempted bypasses
    if affine is not None or target_affine is not None:
        if affine is None:
            raise ValueError(
                "Single target_affine supplied without paired affine; cannot establish transform alignment."
            )
        if target_affine is None:
            raise ValueError(
                "Single affine supplied without paired target_affine; cannot establish transform alignment."
            )

        aff_arr = np.asarray(affine, dtype=np.float64)
        tgt_aff_arr = np.asarray(target_affine, dtype=np.float64)

        if aff_arr.shape != (4, 4):
            raise ValueError(f"affine must be a 4x4 matrix, got shape {aff_arr.shape}")
        if tgt_aff_arr.shape != (4, 4):
            raise ValueError(f"target_affine must be a 4x4 matrix, got shape {tgt_aff_arr.shape}")
        if not np.all(np.isfinite(aff_arr)) or not np.all(np.isfinite(tgt_aff_arr)):
            raise ValueError("affine matrices must contain only finite numbers")

        if not np.allclose(aff_arr, tgt_aff_arr, atol=1e-3):
            raise ValueError(
                "Image/prediction affine does not match target ground-truth affine in native evaluation."
            )

        # Non-axis-aligned / sheared grids
        spatial_block = aff_arr[:3, :3]
        diag_block = np.diag(np.diag(spatial_block))
        if not np.allclose(spatial_block, diag_block, atol=1e-3):
            raise ValueError(
                "Non-axis-aligned or sheared physical grids are not supported by spacing-only "
                "distance transforms; native evaluation rejected/deferred."
            )

        # Spacing-affine consistency check
        if physical is not None:
            col_norms = np.linalg.norm(spatial_block, axis=0)
            if not np.allclose(sorted(physical), sorted(col_norms), atol=1e-2):
                raise ValueError(
                    f"Supplied spacing {physical} is inconsistent with affine voxel dimensions {tuple(col_norms)}"
                )

    if strict_physical:
        raise ValueError(
            "strict_physical evaluation is DEFERRED pending real transform/grid provenance "
            "(NIfTI transform chain / continuous 3D coordinate resampling); "
            "cannot emit verified native-mm metrics from discrete inverse-resize."
        )

    policy = resolve_empty_policy(empty_policy)
    names = resolve_class_names(class_names, num_classes=num_classes)
    block = dice_block(
        prediction,
        target,
        metric_space=METRIC_SPACE_VOLUME_NATIVE,
        num_classes=num_classes,
        class_names=names,
        empty_policy=policy,
        geometry={"orig_shape": list(orig_shape)} if orig_shape is not None else None,
    )
    block["native_shape"] = [int(v) for v in target.shape]
    block["ground_truth_source"] = "caller_supplied_native"
    block["native_reconstruction_verification"] = "DEFERRED"
    block["native_mm_available"] = False
    block["native_mm_note"] = (
        "Native-mm surface metrics (HD95, ASSD) are suppressed because native physical "
        "alignment has not been verified from raw transform provenance. Discrete in-plane "
        "inverse resize on labels cannot establish physical millimeter precision."
    )
    block["spacing_known"] = False
    if physical is not None:
        block["declared_spacing"] = list(physical)

    block["geometry_verification"] = {
        "status": "DEFERRED",
        "checked": {
            "shape_compatible": bool(prediction.shape == target.shape),
            "spacing_provided": bool(physical is not None),
            "affine_provided": bool(affine is not None and target_affine is not None),
            "affine_matched": bool(
                affine is not None
                and target_affine is not None
                and np.allclose(np.asarray(affine), np.asarray(target_affine), atol=1e-3)
            ),
        },
        "verified_evidence": False,
        "limitations": [
            "In-plane nearest-neighbor resize used without continuous coordinate resampling",
            "Physical/native alignment verification DEFERRED pending raw acquisition NIfTI header provenance",
        ],
    }
    return block


#: Phase suffixes recognised in an ACDC case id (``patientXXX_ED`` / ``_ES``).
ACDC_PHASES = ("ED", "ES")
UNKNOWN_PHASE = "unknown"


def split_cases_by_phase(case_ids: Any) -> dict[str, list[str]]:
    """Group ACDC case ids into ``{"ED": [...], "ES": [...], "unknown": [...]}``.

    The phase is read from the case-id suffix written by
    ``scripts/preprocess_acdc.py`` (``patientXXX_ED`` / ``patientXXX_ES``).  A
    case whose suffix is not recognised goes to ``"unknown"``; the phase is
    never guessed from anything else.  All three keys are always present so a
    caller can report ED and ES separately without a KeyError.
    """

    groups: dict[str, list[str]] = {"ED": [], "ES": [], UNKNOWN_PHASE: []}
    for case_id in case_ids:
        name = str(case_id)
        suffix = name.rsplit("_", 1)[-1].upper() if "_" in name else ""
        groups[suffix if suffix in ACDC_PHASES else UNKNOWN_PHASE].append(name)
    return groups


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
