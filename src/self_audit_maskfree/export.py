"""Versioned native-grid prediction export and immutable freeze manifests (W7).

Scientific role
---------------
This module is the boundary between "a model produced a draft partition for a
2-D sampling unit" and "a compared method has a frozen, hashable, geometrically
honest volume prediction on disk". Nothing here selects, scores, tunes or
learns. It moves arrays, it reverses the grid transform that the data layer
applied on the way in, and it records exactly what it could and could not
establish about the native geometry.

Three stages
------------
1. :func:`export_prediction` writes one sampling unit (labels ``[H,W]``,
   probabilities ``[4,H,W]``, validity ``[H,W]``, semantic alternatives) on the
   *stored model grid*, with a sidecar carrying the record fields needed to
   reverse the grid transform later. Per-unit shards are append-only.
2. :func:`assemble_volume` collects one volume's units by their native slice
   index, stacks them into ``[4,Z,H,W]`` / ``[Z,H,W]``, applies the explicit
   reverse shape and axis transform when - and only when - the inverse geometry
   is verifiable, and writes the volume.
3. :func:`freeze_predictions` hashes every written file and every compared
   method's checkpoint into one immutable manifest; :func:`validate_freeze`
   re-checks those hashes. Downstream verification and the isolated reference
   evaluator both refuse to run without a validated freeze.

Geometry honesty rules (not negotiable here)
--------------------------------------------
* A volume is labelled ``grid="native"`` only when the source was a NIfTI whose
  affine survived the data layer, the stored-to-native axis permutation is a
  real permutation, and the permuted reverse-interpolated shape equals the
  recorded native shape. Every one of those is checked, not assumed.
* An ``.npy`` source has no native affine. It is exported ``grid="stored"``
  with ``native_export_available=False`` and a reason. It is never relabelled
  native because it happens to have been resized back to the right numbers.
* A slice that no unit produced is written with ``validity=0`` and recorded in
  ``missing_slice_indices``. It is not silently converted to background.

What a freeze manifest is NOT
-----------------------------
It is a tamper-evident record of *which bytes were compared*. It says nothing
about whether those predictions are anatomically correct, and it contains no
reference masks. :data:`REQUIRED_COMPARISONS` is the enumeration a *final*
freeze is expected to cover; a preflight or smoke freeze is allowed to be
incomplete and is then explicitly marked ``completeness.complete = false``
rather than being quietly accepted as a full comparison.

Ownership: W7. The dataclasses in :mod:`contracts` are coordinator-owned and are
consumed, never redefined. The record keys this module reads are documented in
``docs/maskfree150.md`` and in :data:`RECORD_KEYS`; changes to them are
negotiated with W3/W5 through the coordinator, not by editing their files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .contracts import SEMANTIC_ORDER, VERSION

__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "FREEZE_SCHEMA_VERSION",
    "NUM_CLASSES",
    "RECORD_KEYS",
    "REQUIRED_COMPARISONS",
    "ExportError",
    "FreezeError",
    "assemble_all",
    "assemble_volume",
    "comparison_completeness",
    "export_prediction",
    "freeze_predictions",
    "normalize_record",
    "resolve_native_target",
    "verified_freeze_session",
    "validate_freeze",
]

EXPORT_SCHEMA_VERSION = "maskfree150.export.v1"
FREEZE_SCHEMA_VERSION = "maskfree150.freeze.v1"
NUM_CLASSES = len(SEMANTIC_ORDER)

# A verified session is intentionally process-local and object-identity-bound.
# It is never serialized into the freeze manifest: the entry/exit full-hash
# checks are the only authority, and a copied or reloaded mapping cannot use
# the fast path. ContextVar keeps nested verification calls isolated without a
# mutable global bypass switch.
_ACTIVE_FREEZE_SESSION: ContextVar[dict[str, Any] | None] = ContextVar(
    "maskfree_active_freeze_session", default=None
)

#: Method names a *final* freeze is expected to enumerate. Mirrored by
#: ``configs/maskfree_experiments.yaml``; ``tests/test_maskfree_export.py``
#: asserts the two stay identical. A freeze missing any of these is recorded as
#: incomplete, which is a reportable research outcome, not an export bug.
REQUIRED_COMPARISONS: tuple[str, ...] = (
    "E0_intensity_grouping",
    "E1_cuts_inspired_control",
    "E2_confidence_consistency",
    "E3_fitting_score",
    "E4_selection_evidence",
    "E5_evidence_plus_challenge",
    "control_random_same_budget",
    "control_shuffled_evidence",
    "control_all_background",
    "control_random_spatial_masks",
    "control_class_permutation",
    "control_excessive_partition",
    "control_random_edit_search",
    "student_no_audit",
    "student_audited",
)

#: Canonical record keys consumed by this module, AFTER normalization. ``True``
#: means required for a native export; a missing required key degrades the unit
#: to a stored-grid export with an explicit reason, it never raises mid-run and
#: never fabricates geometry.
#:
#: The data layer (W3) emits its own spelling - ``depth`` for the slice count and
#: one ``inverse_transform`` sub-record describing the reverse grid operation.
#: :func:`normalize_record` maps that authoritative shape onto these keys, so
#: both spellings work and W3 remains the single source of truth for geometry.
RECORD_KEYS: dict[str, bool] = {
    "study_id": True,
    "patient_id": True,
    "unit_id": True,
    # ``volume_id`` is the W3 acquisition/volume identity.  It is deliberately
    # optional for compatibility with the first record shape, where
    # ``study_id`` was the only grouping key.  A volume id, when present, is
    # always preferred for assembly; a patient id is never used as a volume
    # key.
    "volume_id": False,
    "acquisition_id": False,
    "dataset": True,
    "split": True,
    "slice_index": True,
    "num_slices": True,
    "source_format": True,
    "native_shape": True,
    "native_slice_shape": True,
    "depth_axis": True,
    "stored_to_native_axes": True,
    "native_geometry": True,
    "source_path": False,
    "protocol": False,
    "frame_index": False,
    "partition_id": False,
    "manifest_id": False,
    "cohort_provenance": False,
    "spacing_valid": False,
    "spacing_mm": False,
    "spatial_unit": False,
    "xyzt_units": False,
}

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._+-]")


class ExportError(RuntimeError):
    """Raised when an export request is internally inconsistent."""


class FreezeError(RuntimeError):
    """Raised when a freeze would overwrite, diverge from, or contradict a prior one."""


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _safe_component(value: Any, field: str) -> str:
    text = str(value)
    if not text or text in (".", ".."):
        raise ExportError(f"{field} is not a usable path component: {value!r}")
    cleaned = _SAFE_NAME.sub("_", text)
    if os.sep in text or (os.altsep and os.altsep in text):
        raise ExportError(f"{field} must not contain a path separator: {value!r}")
    return cleaned


def _record_volume_id(record: Mapping[str, Any]) -> str:
    """Return the acquisition/volume grouping identity from a W3 record.

    ``volume_id`` is the current data-layer field.  ``acquisition_id`` and then
    ``study_id`` are compatibility fallbacks for manifests produced before the
    volume identity was made explicit.  In particular, ``patient_id`` is never
    a fallback: one patient may legitimately have more than one acquisition.
    """
    raw = record.get("volume_id") or record.get("acquisition_id") or record.get("study_id")
    return _safe_component(raw, "record.volume_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    from .progress import current_progress
    digest = hashlib.sha256()
    with current_progress().stage("export.sha256", path=str(path)):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_numpy(tensor: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        array = tensor
    else:
        raise ExportError(f"{name} must be a torch.Tensor or numpy array, got {type(tensor)!r}")
    return array


# --------------------------------------------------------------------------- #
# geometry resolution
# --------------------------------------------------------------------------- #
#: Native axis orders this module can invert. The value is the permutation that
#: turns a stored ``(Z, H, W)`` stack into that native order via ``np.transpose``.
_NATIVE_AXIS_ORDERS: dict[str, tuple[int, int, int]] = {
    "HWZ": (1, 2, 0),
    "ZHW": (0, 1, 2),
    "HZW": (1, 0, 2),
}


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map a W3 unit record onto the canonical keys in :data:`RECORD_KEYS`.

    W3 is the authority on geometry, and its record says ``depth`` where this
    module says ``num_slices`` and packs the reverse grid operation into one
    ``inverse_transform`` sub-record. Rather than asking W3 to rename fields -
    or, worse, reconstructing the transform here from separate pieces - this
    reads W3's own description and translates it.

    A record that already uses the canonical spelling passes through unchanged,
    which is what the exporter's own fixtures and any future producer use.
    Anything unresolvable is left absent, so :func:`resolve_native_target`
    reports it as a missing key instead of guessing.
    """
    canonical = dict(record)
    # W3 calls the complete acquisition/volume identity ``volume_id``.  Older
    # manifests only exposed ``acquisition_id`` or ``study_id``; these fallbacks
    # preserve their behaviour without ever collapsing two different patients
    # merely because they share a patient label.
    if canonical.get("volume_id") is None:
        canonical["volume_id"] = canonical.get("acquisition_id") or canonical.get("study_id")
    if canonical.get("num_slices") is None:
        canonical["num_slices"] = canonical.get("depth")
    transform = record.get("inverse_transform")
    if isinstance(transform, Mapping):
        if canonical.get("slice_index") is None:
            canonical["slice_index"] = transform.get("slice_index")
        if transform.get("native_shape") is not None:
            canonical["native_shape"] = transform.get("native_shape")
        if transform.get("depth_axis") is not None:
            canonical["depth_axis"] = transform.get("depth_axis")

        inverse_resize = transform.get("inverse_resize")
        if isinstance(inverse_resize, Mapping) and inverse_resize.get("output_hw") is not None:
            canonical["native_slice_shape"] = inverse_resize.get("output_hw")
        elif canonical.get("native_slice_shape") is None:
            canonical["native_slice_shape"] = record.get("native_hw")

        # Prefer an explicit permutation when W3 provides one.  The named
        # order is retained as a readable fallback for older manifests.
        direct_axes = transform.get("stored_to_native_axes")
        order = str(transform.get("native_axis_order") or "")
        axes = direct_axes if direct_axes is not None else _NATIVE_AXIS_ORDERS.get(order)
        if axes is not None:
            canonical["stored_to_native_axes"] = list(axes)
        elif canonical.get("stored_to_native_axes") is None:
            # A 4-D native array (HWZT) needs time assembly, which volume
            # assembly does not do. Leave the transform unresolved and say why.
            canonical["stored_to_native_axes"] = None
            canonical["native_axis_order_unsupported"] = order or "<absent>"

        # W3 states native availability twice: a geometry status string and the
        # export decision itself. A native claim needs BOTH.  Accept both W3's
        # string status and the mapping used by the original W7 fixtures.
        raw_geometry = record.get("native_geometry")
        if isinstance(raw_geometry, Mapping):
            geometry_status = bool(raw_geometry.get("available"))
            affine = raw_geometry.get("affine", record.get("native_affine"))
            spacing = raw_geometry.get("spacing", record.get("zooms"))
            reason = raw_geometry.get("reason") or record.get("native_geometry_reason")
            spacing_mm = raw_geometry.get("spacing_mm", record.get("spacing_mm"))
            spatial_unit = raw_geometry.get("spatial_unit", record.get("spatial_unit"))
            xyzt_units = raw_geometry.get("xyzt_units", record.get("xyzt_units"))
            spacing_valid = raw_geometry.get("spacing_valid", record.get("spacing_valid"))
        else:
            geometry_status = raw_geometry == "available"
            affine = record.get("native_affine")
            spacing = record.get("zooms")
            reason = record.get("native_geometry_reason")
            spacing_mm = record.get("spacing_mm")
            spatial_unit = record.get("spatial_unit")
            xyzt_units = record.get("xyzt_units")
            spacing_valid = record.get("spacing_valid")
        export_flag = transform.get("native_grid_export", record.get("native_grid_export"))
        geometry_available = geometry_status and bool(export_flag)
        if geometry_status and not export_flag:
            reason = reason or "W3 resolved export_grid=stored for this unit"
        canonical["native_geometry"] = {
            "available": geometry_available,
            "affine": affine,
            "spacing": spacing,
            "spacing_mm": spacing_mm,
            "spatial_unit": spatial_unit,
            "xyzt_units": xyzt_units,
            "spacing_valid": spacing_valid,
            "reason": reason,
        }
    elif isinstance(canonical.get("native_geometry"), str):
        # Permit direct W3-shaped records in focused callers that omit the
        # inverse sub-record.  The geometry is still accepted only when all
        # canonical shape/axis fields are independently present and verified.
        status = canonical["native_geometry"] == "available"
        canonical["native_geometry"] = {
            "available": status and bool(canonical.get("native_grid_export", status)),
            "affine": canonical.get("native_affine"),
            "spacing": canonical.get("zooms"),
            "spacing_mm": canonical.get("spacing_mm"),
            "spatial_unit": canonical.get("spatial_unit"),
            "xyzt_units": canonical.get("xyzt_units"),
            "spacing_valid": canonical.get("spacing_valid"),
            "reason": canonical.get("native_geometry_reason"),
        }
    return canonical


def _is_permutation(axes: Any) -> bool:
    try:
        values = [int(a) for a in axes]
    except (TypeError, ValueError):
        return False
    return sorted(values) == [0, 1, 2]


def _affine_ok(affine: Any) -> bool:
    if affine is None:
        return False
    try:
        matrix = np.asarray(affine, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return False
    return bool(abs(np.linalg.det(matrix[:3, :3])) > 1e-12)


def resolve_native_target(record: Mapping[str, Any]) -> dict[str, Any]:
    """Decide whether this record supports a *verified* native-grid export.

    Returns a dict with ``native_export_available``, ``reason``, and, when
    available, the exact reverse transform: ``native_slice_shape``,
    ``num_slices``, ``stored_to_native_axes``, ``native_shape``, ``affine``.

    The check is deliberately conservative and fully mechanical. In particular
    it verifies that permuting the reverse-interpolated stored volume shape by
    ``stored_to_native_axes`` reproduces ``native_shape`` exactly - i.e. the
    claimed inverse transform is arithmetically consistent with the claimed
    native array - instead of trusting the field.
    """
    record = normalize_record(record)
    if record.get("native_axis_order_unsupported"):
        return {"native_export_available": False,
                "reason": "unsupported native axis order "
                          f"{record['native_axis_order_unsupported']!r}; volume assembly "
                          "covers 3-D natives only, a 4-D cine native needs time assembly"}
    missing = [key for key, required in RECORD_KEYS.items()
               if required and record.get(key) is None]
    if missing:
        return {"native_export_available": False,
                "reason": f"record missing required native keys: {sorted(missing)}"}

    source_format = str(record["source_format"]).lower()
    if source_format not in ("nifti", "nii", "nii.gz"):
        return {"native_export_available": False,
                "reason": f"source_format={source_format!r} carries no native affine; "
                          "stored-grid export only"}

    geometry = record.get("native_geometry") or {}
    if not isinstance(geometry, Mapping) or not geometry.get("available"):
        reason = "native_geometry unavailable"
        if isinstance(geometry, Mapping) and geometry.get("reason"):
            reason = f"{reason}: {geometry['reason']}"
        return {"native_export_available": False, "reason": reason}

    affine = geometry.get("affine")
    if not _affine_ok(affine):
        return {"native_export_available": False,
                "reason": "native affine absent, non-finite or singular"}

    axes = record["stored_to_native_axes"]
    if not _is_permutation(axes):
        return {"native_export_available": False,
                "reason": f"stored_to_native_axes={axes!r} is not a permutation of (0,1,2)"}
    axes = tuple(int(a) for a in axes)

    try:
        native_shape = tuple(int(v) for v in record["native_shape"])
        slice_shape = tuple(int(v) for v in record["native_slice_shape"])
        num_slices = int(record["num_slices"])
        depth_axis = int(record["depth_axis"])
    except (TypeError, ValueError) as exc:
        return {"native_export_available": False,
                "reason": f"native shape fields are not integral: {exc}"}

    if len(native_shape) != 3 or len(slice_shape) != 2:
        return {"native_export_available": False,
                "reason": "native_shape must be rank 3 and native_slice_shape rank 2"}
    if min(*slice_shape, num_slices) <= 0:
        return {"native_export_available": False, "reason": "non-positive native extent"}
    if not 0 <= depth_axis < 3:
        return {"native_export_available": False,
                "reason": f"depth_axis={depth_axis} outside the native rank"}

    stored_shape = (num_slices, slice_shape[0], slice_shape[1])
    permuted = tuple(stored_shape[a] for a in axes)
    if permuted != native_shape:
        return {"native_export_available": False,
                "reason": f"inverse transform inconsistent: permute({stored_shape}, {axes})"
                          f" = {permuted} != native_shape {native_shape}"}
    if native_shape[depth_axis] != num_slices:
        return {"native_export_available": False,
                "reason": f"depth_axis {depth_axis} extent {native_shape[depth_axis]}"
                          f" != num_slices {num_slices}"}

    return {
        "native_export_available": True,
        "reason": None,
        "native_shape": list(native_shape),
        "native_slice_shape": list(slice_shape),
        "num_slices": num_slices,
        "depth_axis": depth_axis,
        "stored_to_native_axes": list(axes),
        "affine": np.asarray(affine, dtype=np.float64).tolist(),
        "spacing": _jsonable(geometry.get("spacing")),
        "spacing_mm": _jsonable(geometry.get("spacing_mm")),
        "spatial_unit": geometry.get("spatial_unit"),
        "xyzt_units": _jsonable(geometry.get("xyzt_units")),
        "spacing_valid": geometry.get("spacing_valid"),
    }


# --------------------------------------------------------------------------- #
# stage 1: per-unit export
# --------------------------------------------------------------------------- #
def _validate_unit_arrays(labels: np.ndarray, probabilities: np.ndarray,
                          validity: np.ndarray) -> tuple[int, int]:
    if labels.ndim != 2:
        raise ExportError(f"labels must be [H,W], got {labels.shape}")
    height, width = labels.shape
    if height <= 0 or width <= 0:
        raise ExportError(f"labels must have positive spatial dimensions, got {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
        raise ExportError(
            f"labels must use an integer dtype in the fixed semantic order, got {labels.dtype}")
    if probabilities.shape != (NUM_CLASSES, height, width):
        raise ExportError(
            f"probabilities must be [{NUM_CLASSES},H,W]={NUM_CLASSES, height, width}, "
            f"got {probabilities.shape}")
    if validity.shape != (height, width):
        raise ExportError(f"validity must be [H,W], got {validity.shape}")
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ExportError(
            f"labels outside the fixed semantic order {SEMANTIC_ORDER}: "
            f"[{int(labels.min())}, {int(labels.max())}]")
    if not np.isfinite(probabilities).all() or not np.isfinite(validity).all():
        raise ExportError("non-finite probabilities or validity")
    if probabilities.min() < -1e-6:
        raise ExportError("negative probabilities")
    probability_sum = probabilities.sum(axis=0)
    if not np.all(np.abs(probability_sum - 1.0) <= 1e-3):
        deviation = float(np.max(np.abs(probability_sum - 1.0)))
        raise ExportError(
            f"probabilities must sum to 1 per pixel before export (max deviation {deviation:.6g})")
    if validity.min() < -1e-6 or validity.max() > 1.0 + 1e-6:
        raise ExportError("validity outside [0,1]")
    return height, width


def export_prediction(output_dir: str | Path, *, record: Mapping[str, Any],
                      prediction_name: str, labels: torch.Tensor,
                      probabilities: torch.Tensor, validity: torch.Tensor,
                      alternatives: Sequence[Mapping[str, Any]] | None,
                      checkpoint_id: str, version: str) -> dict[str, Any]:
    """Write one sampling unit's draft on the stored model grid.

    ``labels`` ``[H,W]`` long in the fixed named order ``0=BG,1=RV,2=MYO,3=LV``,
    ``probabilities`` ``[4,H,W]`` float, ``validity`` ``[H,W]`` float in [0,1],
    ``alternatives`` the unit's semantic/partition alternatives as recorded by
    the auditor. ``checkpoint_id`` and ``version`` are the model identity and
    the pseudo-label/bank version; both are carried into the freeze manifest so
    that a comparison cannot silently mix label versions.

    Returns the shard descriptor (paths, resolved geometry decision, identity).
    Writes are atomic per file. Re-exporting the same unit with identical
    content is idempotent; re-exporting it with *different* content raises,
    because a compared method's unit must not change under the same identity.
    """
    root = Path(output_dir)
    name = _safe_component(prediction_name, "prediction_name")
    study_id = _safe_component(record.get("study_id"), "record.study_id")
    volume_id = _record_volume_id(record)
    unit_id = _safe_component(record.get("unit_id"), "record.unit_id")

    labels_np = _as_numpy(labels, "labels")
    # Reject fractional labels before any conversion.  The exporter is allowed
    # to choose nearest interpolation during native assembly, but it must never
    # silently turn a model bug such as 1.7 into semantic class 1 at the unit
    # boundary.
    if not np.issubdtype(labels_np.dtype, np.integer) or np.issubdtype(labels_np.dtype, np.bool_):
        raise ExportError(f"labels must use an integer dtype, got {labels_np.dtype}")
    labels_np = labels_np.astype(np.int16)
    probs_np = _as_numpy(probabilities, "probabilities").astype(np.float32)
    validity_np = _as_numpy(validity, "validity").astype(np.float32)
    height, width = _validate_unit_arrays(labels_np, probs_np, validity_np)

    record = normalize_record(record)
    slice_identity = "record"
    if record.get("slice_index") is None and record.get("num_slices") is None:
        # A record with no slice identity at all - a full-input deployment record,
        # for instance - is treated as a single-slice volume. It is marked, and it
        # cannot claim a native grid, because without a slice identity there is no
        # verified inverse transform. Two such units under one study still collide
        # loudly on slice_index 0 rather than silently overwriting each other.
        record["slice_index"] = 0
        record["num_slices"] = 1
        slice_identity = "defaulted: record carried no slice_index/num_slices"
    try:
        slice_index = int(record["slice_index"])
        num_slices = int(record["num_slices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError(f"record needs integral slice_index and num_slices: {exc}") from exc
    if not 0 <= slice_index < num_slices:
        raise ExportError(
            f"slice_index {slice_index} outside [0,{num_slices}) for unit {unit_id}")

    geometry = resolve_native_target(record)
    if slice_identity != "record":
        geometry = {"native_export_available": False,
                    "reason": "no slice identity in the record, so no verified inverse transform"}
    unit_dir = root / "predictions" / name / "units" / volume_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    array_path = unit_dir / f"{unit_id}.npz"
    sidecar_path = unit_dir / f"{unit_id}.json"

    sidecar = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "contract_version": VERSION,
        "prediction_name": prediction_name,
        "checkpoint_id": str(checkpoint_id),
        "version": str(version),
        "grid": "stored",
        "stored_slice_shape": [height, width],
        "semantic_order": {str(k): v for k, v in SEMANTIC_ORDER.items()},
        "slice_identity": slice_identity,
        "geometry": geometry,
        "alternatives": _jsonable(list(alternatives or [])),
        "record": {key: _jsonable(record.get(key)) for key in RECORD_KEYS
                   if record.get(key) is not None},
        "written_utc": _utc_now(),
    }

    payload_hash = _canonical_hash({
        "labels": hashlib.sha256(labels_np.tobytes()).hexdigest(),
        "probabilities": hashlib.sha256(probs_np.tobytes()).hexdigest(),
        "validity": hashlib.sha256(validity_np.tobytes()).hexdigest(),
        "sidecar": {k: v for k, v in sidecar.items() if k != "written_utc"},
    })
    sidecar["payload_hash"] = payload_hash

    if sidecar_path.exists():
        previous = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if previous.get("payload_hash") != payload_hash:
            raise ExportError(
                f"unit {unit_id} of prediction {prediction_name!r} already exported with "
                f"different content (payload {previous.get('payload_hash')} vs {payload_hash}); "
                "a frozen comparison arm must not be rewritten in place")
        return _shard_descriptor(root, sidecar, array_path, sidecar_path, study_id, volume_id,
                                 slice_index)

    # np.savez_compressed appends ".npz" to a path that lacks it, so the staging
    # file is written through an explicit handle and then atomically renamed.
    tmp = array_path.with_name(array_path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, labels=labels_np, probabilities=probs_np,
                            validity=validity_np)
    os.replace(tmp, array_path)
    _write_json(sidecar_path, sidecar)
    return _shard_descriptor(root, sidecar, array_path, sidecar_path, study_id, volume_id,
                             slice_index)


def _shard_descriptor(root: Path, sidecar: Mapping[str, Any], array_path: Path,
                      sidecar_path: Path, study_id: str, volume_id: str,
                      slice_index: int) -> dict[str, Any]:
    return {
        "kind": "unit",
        "prediction_name": sidecar["prediction_name"],
        "study_id": study_id,
        "volume_id": volume_id,
        "unit_id": sidecar["record"].get("unit_id"),
        "slice_index": slice_index,
        "grid": "stored",
        "native_export_available": bool(sidecar["geometry"].get("native_export_available")),
        "reason": sidecar["geometry"].get("reason"),
        "dataset": sidecar["record"].get("dataset"),
        "protocol": sidecar["record"].get("protocol"),
        "partition_id": sidecar["record"].get("partition_id"),
        "manifest_id": sidecar["record"].get("manifest_id"),
        "checkpoint_id": sidecar["checkpoint_id"],
        "version": sidecar["version"],
        "payload_hash": sidecar["payload_hash"],
        "files": [str(array_path.relative_to(root)), str(sidecar_path.relative_to(root))],
        "root": str(root),
    }


# --------------------------------------------------------------------------- #
# stage 2: volume assembly and the explicit reverse transform
# --------------------------------------------------------------------------- #
def _resize_slice_stack(stack: np.ndarray, target_hw: tuple[int, int], mode: str) -> np.ndarray:
    """Resize ``[C,Z,h,w] -> [C,Z,H,W]`` slicewise with an explicit mode."""
    channels, depth, src_h, src_w = stack.shape
    if (src_h, src_w) == target_hw:
        return stack
    tensor = torch.from_numpy(np.ascontiguousarray(stack)).reshape(channels * depth, 1, src_h, src_w)
    if mode == "nearest":
        resized = torch.nn.functional.interpolate(tensor.float(), size=target_hw, mode="nearest")
    else:
        resized = torch.nn.functional.interpolate(tensor.float(), size=target_hw,
                                                  mode="bilinear", align_corners=False)
    return resized.reshape(channels, depth, *target_hw).numpy()


def _nifti_image(array: np.ndarray, affine: np.ndarray, geometry: Mapping[str, Any]) -> Any:
    """Create a NIfTI while retaining the source spatial unit when supported."""
    import nibabel as nib  # noqa: PLC0415 - optional dependency

    image = nib.Nifti1Image(array, affine)
    unit = geometry.get("spatial_unit")
    if unit is None:
        xyzt = geometry.get("xyzt_units")
        if isinstance(xyzt, Mapping):
            unit = xyzt.get("space") or xyzt.get("spatial")
        elif isinstance(xyzt, (list, tuple)) and xyzt:
            unit = xyzt[0]
    aliases = {
        "millimeter": "mm", "millimetre": "mm", "millimeters": "mm",
        "millimetres": "mm", "micrometer": "micron", "micrometers": "micron",
        "nanometers": "nanometer",
    }
    if unit is not None:
        unit = aliases.get(str(unit).strip().lower(), str(unit).strip().lower())
        # nibabel rejects unknown strings. Keep the original value in the
        # sidecar and leave the header's unit unknown rather than changing the
        # affine or inventing a conversion.
        try:
            # nibabel's public spelling is ``xyz`` in current releases; older
            # releases exposed the same positional pair. Keep the source unit
            # whenever either supported API accepts it.
            image.header.set_xyzt_units(xyz=unit)
        except (KeyError, TypeError, ValueError):
            try:
                image.header.set_xyzt_units(unit, None)
            except (KeyError, TypeError, ValueError):
                # Unknown source units remain in the sidecar; never alter the
                # affine or silently relabel them as millimetres.
                pass
    return image


def _collect_units(root: Path, prediction_name: str, volume_id: str) -> list[dict[str, Any]]:
    unit_dir = root / "predictions" / prediction_name / "units" / volume_id
    if not unit_dir.is_dir():
        raise ExportError(f"no exported units for {prediction_name!r}/{volume_id!r} under {root}")
    sidecars = sorted(unit_dir.glob("*.json"))
    if not sidecars:
        raise ExportError(f"no unit sidecars under {unit_dir}")
    units: list[dict[str, Any]] = []
    for path in sidecars:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        array_path = path.with_suffix(".npz")
        if not array_path.exists():
            raise ExportError(f"unit sidecar {path} has no array shard {array_path}")
        sidecar["_array_path"] = array_path
        units.append(sidecar)
    return units


def _require_single(values: Iterable[Any], field: str, prediction_name: str) -> Any:
    unique = {json.dumps(_jsonable(v), sort_keys=True, default=str) for v in values}
    if len(unique) != 1:
        raise ExportError(
            f"units of prediction {prediction_name!r} disagree on {field}: {sorted(unique)}")
    return json.loads(next(iter(unique)))


def assemble_volume(output_dir: str | Path, *, prediction_name: str,
                    study_id: str | None = None, volume_id: str | None = None,
                    write_nifti: bool = True) -> dict[str, Any]:
    """Assemble one volume's exported units, reversing the grid transform.

    Stacking order is the record's ``slice_index``, not filesystem order. Slices
    no unit produced are written with ``validity = 0``, ``labels = 0`` and a
    uniform probability vector, and are listed in ``missing_slice_indices``;
    they are an explicit unresolved gap, not background evidence.

    The reverse transform is applied only when :func:`resolve_native_target`
    verified it for every unit of the volume: probabilities bilinear then
    renormalized, labels nearest, validity bilinear (a weighted support, so it
    interpolates like a weight), then the stored-to-native axis permutation.
    Otherwise the volume stays on the stored grid, is labelled ``grid="stored"``
    with ``native_export_available=False`` and a reason, and is written as
    ``.npy`` - never as a native-grid NIfTI.
    """
    root = Path(output_dir)
    name = _safe_component(prediction_name, "prediction_name")
    if study_id is None and volume_id is None:
        raise ExportError("assemble_volume requires volume_id (study_id is a legacy alias)")
    requested_volume = _safe_component(volume_id or study_id, "volume_id")
    if study_id is not None and volume_id is not None:
        legacy_study = _safe_component(study_id, "study_id")
        if requested_volume != _safe_component(volume_id, "volume_id"):
            raise ExportError("study_id and volume_id identify different output volumes")
        # The explicit volume id controls the on-disk grouping.  ``study_id``
        # remains a metadata field and is checked after the units are loaded.
        _ = legacy_study
    from .progress import current_progress
    with current_progress().stage("export.load_volume_shards", method=name, volume_id=requested_volume,
                                  path=str(root)):
        units = _collect_units(root, name, requested_volume)
    study = _safe_component(study_id or units[0]["record"].get("study_id"), "study_id")

    actual_volume_ids = {
        _record_volume_id({**u.get("record", {}), "volume_id": u.get("volume_id", requested_volume)})
        for u in units
    }
    if actual_volume_ids != {requested_volume}:
        raise ExportError(
            f"units of prediction {name!r} disagree on volume_id: "
            f"expected {requested_volume!r}, got {sorted(actual_volume_ids)}")

    checkpoint_id = _require_single((u["checkpoint_id"] for u in units), "checkpoint_id", name)
    version = _require_single((u["version"] for u in units), "version", name)
    stored_hw = tuple(_require_single((u["stored_slice_shape"] for u in units),
                                      "stored_slice_shape", name))
    geometry = _require_single((u["geometry"] for u in units), "geometry", name)
    def _unit_num_slices(unit: Mapping[str, Any]) -> Any:
        record = unit.get("record", {})
        value = record.get("num_slices", record.get("depth"))
        if value is None:
            transform = record.get("inverse_transform")
            if isinstance(transform, Mapping):
                native_shape = transform.get("native_shape")
                depth_axis = transform.get("depth_axis")
                try:
                    if native_shape is not None and depth_axis is not None:
                        value = native_shape[int(depth_axis)]
                except (TypeError, ValueError, IndexError):
                    value = None
        return value

    num_slices = int(_require_single((_unit_num_slices(u) for u in units),
                                     "num_slices", name))

    by_index: dict[int, dict[str, Any]] = {}
    for unit in units:
        index = int(unit["record"].get("slice_index", unit.get("slice_index")))
        if index in by_index:
            raise ExportError(
                f"duplicate slice_index {index} in prediction {name!r} study {study!r}; "
                "the same acquisition slice must not be exported twice")
        by_index[index] = unit

    height, width = int(stored_hw[0]), int(stored_hw[1])
    labels = np.zeros((num_slices, height, width), dtype=np.int16)
    probabilities = np.full((NUM_CLASSES, num_slices, height, width),
                            1.0 / NUM_CLASSES, dtype=np.float32)
    validity = np.zeros((num_slices, height, width), dtype=np.float32)
    alternatives: dict[str, Any] = {}

    for index in range(num_slices):
        unit = by_index.get(index)
        if unit is None:
            continue
        with np.load(unit["_array_path"]) as payload:
            labels[index] = payload["labels"]
            probabilities[:, index] = payload["probabilities"]
            validity[index] = payload["validity"]
        if unit.get("alternatives"):
            alternatives[str(index)] = unit["alternatives"]

    missing = sorted(set(range(num_slices)) - set(by_index))

    native_available = bool(geometry.get("native_export_available"))
    reason = geometry.get("reason")
    grid = "stored"
    affine = None
    if native_available:
        target_hw = (int(geometry["native_slice_shape"][0]), int(geometry["native_slice_shape"][1]))
        probabilities = _resize_slice_stack(probabilities, target_hw, "bilinear")
        total = probabilities.sum(axis=0, keepdims=True)
        probabilities = np.divide(probabilities, np.maximum(total, 1e-8)).astype(np.float32)
        labels = _resize_slice_stack(labels[None].astype(np.float32), target_hw,
                                     "nearest")[0].round().astype(np.int16)
        validity = _resize_slice_stack(validity[None], target_hw, "bilinear")[0]
        validity = np.clip(validity, 0.0, 1.0).astype(np.float32)

        axes = tuple(int(a) for a in geometry["stored_to_native_axes"])
        labels = np.ascontiguousarray(np.transpose(labels, axes))
        validity = np.ascontiguousarray(np.transpose(validity, axes))
        probabilities = np.ascontiguousarray(
            np.transpose(probabilities, (0, *(a + 1 for a in axes))))
        expected = tuple(int(v) for v in geometry["native_shape"])
        if labels.shape != expected:
            raise ExportError(
                f"reverse transform produced {labels.shape}, record claims native {expected}")
        grid = "native"
        affine = np.asarray(geometry["affine"], dtype=np.float64)

    volume_dir = root / "predictions" / name / "volumes" / requested_volume
    volume_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    nifti_written = False
    if grid == "native" and write_nifti:
        try:
            import nibabel as nib  # noqa: PLC0415 - optional at import time on purpose
        except ImportError:
            reason = "nibabel unavailable; native arrays written as .npy with verified geometry"
        else:
            nib.save(_nifti_image(labels.astype(np.int16), affine, geometry),
                     str(volume_dir / "labels.nii.gz"))
            nib.save(_nifti_image(validity.astype(np.float32), affine, geometry),
                     str(volume_dir / "validity.nii.gz"))
            files += ["labels.nii.gz", "validity.nii.gz"]
            nifti_written = True

    if not nifti_written:
        np.save(volume_dir / "labels.npy", labels)
        np.save(volume_dir / "validity.npy", validity)
        files += ["labels.npy", "validity.npy"]
    np.save(volume_dir / "probabilities.npy", probabilities)
    files.append("probabilities.npy")

    sidecar = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "contract_version": VERSION,
        "prediction_name": prediction_name,
        "study_id": study,
        "volume_id": requested_volume,
        "patient_id": _require_single((u["record"].get("patient_id") for u in units),
                                      "patient_id", name),
        "dataset": _require_single((u["record"].get("dataset") for u in units), "dataset", name),
        "split": _require_single((u["record"].get("split") for u in units), "split", name),
        "protocol": _require_single((u["record"].get("protocol") for u in units), "protocol", name),
        "partition_id": _require_single((u["record"].get("partition_id") for u in units),
                                        "partition_id", name),
        "manifest_id": _require_single((u["record"].get("manifest_id") for u in units),
                                       "manifest_id", name),
        "checkpoint_id": checkpoint_id,
        "version": version,
        "grid": grid,
        "native_export_available": native_available,
        "native_export_reason": reason,
        "volume_shape": list(labels.shape),
        "probabilities_shape": list(probabilities.shape),
        "stored_slice_shape": [height, width],
        "semantic_order": {str(k): v for k, v in SEMANTIC_ORDER.items()},
        "label_interpolation": "nearest" if grid == "native" else "none",
        "probability_interpolation": "bilinear+renormalized" if grid == "native" else "none",
        "validity_interpolation": "bilinear" if grid == "native" else "none",
        "affine": affine.tolist() if affine is not None else None,
        "spacing": geometry.get("spacing"),
        "spacing_mm": geometry.get("spacing_mm"),
        "spacing_valid": geometry.get("spacing_valid"),
        "spatial_unit": geometry.get("spatial_unit"),
        "xyzt_units": geometry.get("xyzt_units"),
        "num_slices": num_slices,
        "exported_slice_indices": sorted(by_index),
        "missing_slice_indices": missing,
        "complete_volume": not missing,
        "written_utc": _utc_now(),
    }
    _write_json(volume_dir / "alternatives.json",
                {"study_id": study, "volume_id": requested_volume,
                 "prediction_name": prediction_name,
                 "by_slice_index": alternatives})
    files.append("alternatives.json")
    _write_json(volume_dir / "volume.json", sidecar)
    files.append("volume.json")

    descriptor = dict(sidecar)
    descriptor["kind"] = "volume"
    descriptor["files"] = [str((volume_dir / f).relative_to(root)) for f in files]
    descriptor["root"] = str(root)
    return descriptor


def assemble_all(output_dir: str | Path, *, prediction_name: str,
                 write_nifti: bool = True) -> list[dict[str, Any]]:
    """Assemble every study exported under ``prediction_name``."""
    root = Path(output_dir)
    name = _safe_component(prediction_name, "prediction_name")
    unit_root = root / "predictions" / name / "units"
    if not unit_root.is_dir():
        raise ExportError(f"prediction {prediction_name!r} has no exported units under {root}")
    studies = sorted(p.name for p in unit_root.iterdir() if p.is_dir())
    return [assemble_volume(root, prediction_name=name, volume_id=volume,
                            write_nifti=write_nifti)
            for volume in studies]


# --------------------------------------------------------------------------- #
# stage 3: immutable freeze
# --------------------------------------------------------------------------- #
def _as_name_list(value: Any) -> list[str]:
    """Turn a scalar/sequence/mapping into stable names without splitting strings."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return sorted({str(name) for name in value})
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return sorted({str(name) for name in value})


def _artifact_names(value: Any) -> list[str]:
    """Names used in completeness blocks for path-like artifact inputs."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return sorted({str(name) for name in value})
    if isinstance(value, (str, Path)):
        return [Path(value).name]
    return sorted({Path(item).name for item in value})


def _jsonable_unit_requirements(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(method): _as_name_list(unit_ids)
                for method, unit_ids in sorted(value.items(), key=lambda item: str(item[0]))}
    return _as_name_list(value)


def _expected_unit_map(required_unit_ids: Any, methods: Sequence[str]) -> dict[str, set[str]] | None:
    if required_unit_ids is None:
        return None
    if isinstance(required_unit_ids, Mapping):
        return {str(method): set(_as_name_list(unit_ids))
                for method, unit_ids in required_unit_ids.items()}
    expected = set(_as_name_list(required_unit_ids))
    return {str(method): set(expected) for method in methods}


def _unit_completeness(
    method_units: Mapping[str, Iterable[str]] | None,
    required_unit_ids: Any,
    methods: Sequence[str],
) -> dict[str, Any]:
    present = {str(method): sorted({str(unit) for unit in units})
               for method, units in (method_units or {}).items()}
    expected = _expected_unit_map(required_unit_ids, methods)
    # If the caller did not know the expected unit list, still detect a partial
    # method set when at least one method has a different observed set.  This
    # catches a missing slice for one method while leaving a single-method smoke
    # explicitly marked as unchecked.
    if expected is None and present:
        observed = [set(units) for units in present.values()]
        if len(observed) >= 2:
            union = set().union(*observed)
            expected = {method: set(union) for method in methods if method in present}
    if expected is None:
        return {
            "checked": False,
            "required": [],
            "present_by_method": present,
            "missing_by_method": {},
            "extra_by_method": {},
            "complete": True,
        }
    missing = {
        method: sorted(expected.get(method, set()) - set(present.get(method, [])))
        for method in expected
        if expected.get(method, set()) - set(present.get(method, []))
    }
    extra = {
        method: sorted(set(present.get(method, [])) - expected.get(method, set()))
        for method in set(present) | set(expected)
        if set(present.get(method, [])) - expected.get(method, set())
    }
    required_union = sorted(set().union(*(expected.get(method, set()) for method in expected)))
    return {
        "checked": True,
        "required": required_union,
        "required_by_method": {method: sorted(values) for method, values in expected.items()},
        "present_by_method": present,
        "missing_by_method": missing,
        "extra_by_method": extra,
        "complete": not missing and not extra,
    }


def comparison_completeness(
    present: Iterable[str],
    *,
    required_methods: Iterable[str] | None = None,
    method_units: Mapping[str, Iterable[str]] | None = None,
    required_unit_ids: Any = None,
    present_checkpoints: Iterable[str] | None = None,
    required_checkpoints: Iterable[str] | None = None,
    present_nuisance_files: Iterable[str] | None = None,
    required_nuisance_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Describe whether a freeze covers the requested methods and artifacts.

    Method completeness remains the historical top-level shape used by the
    launcher and focused tests.  The nested blocks are authoritative for final
    runs: a caller can supply the frozen data unit list, required checkpoint
    names, and nuisance files, while a preflight may leave those requirements
    unchecked and be reported as incomplete at the method level.
    """
    required = sorted({str(name) for name in
                       (REQUIRED_COMPARISONS if required_methods is None else required_methods)})
    seen = sorted({str(name) for name in present})
    missing = [name for name in required if name not in seen]
    extra = [name for name in seen if name not in required]

    required_ckpt = sorted({str(name) for name in (required_checkpoints or [])})
    present_ckpt = sorted({str(name) for name in (present_checkpoints or [])})
    missing_ckpt = [name for name in required_ckpt if name not in present_ckpt]
    checkpoints = {
        "checked": bool(required_ckpt),
        "required": required_ckpt,
        "present": present_ckpt,
        "missing": missing_ckpt,
        "complete": not missing_ckpt,
    }

    required_nuisance = sorted({str(name) for name in (required_nuisance_files or [])})
    present_nuisance = sorted({str(name) for name in (present_nuisance_files or [])})
    missing_nuisance = [name for name in required_nuisance if name not in present_nuisance]
    nuisance = {
        "checked": bool(required_nuisance),
        "required": required_nuisance,
        "present": present_nuisance,
        "missing": missing_nuisance,
        "complete": not missing_nuisance,
    }
    units = _unit_completeness(method_units, required_unit_ids, required)
    return {
        "required": required,
        "present": seen,
        "missing": missing,
        "extra": extra,
        "complete": not missing and not units["missing_by_method"] and
        not units["extra_by_method"] and checkpoints["complete"] and nuisance["complete"],
        "units": units,
        "checkpoints": checkpoints,
        "nuisance_files": nuisance,
    }


def _resolve_path(root: Path, raw: str | Path) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _file_entry(root: Path, relative: str | Path) -> dict[str, Any]:
    path = _resolve_path(root, relative)
    if not path.is_file():
        raise FreezeError(f"freeze references a missing file: {path}")
    try:
        stored_path = str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise FreezeError(f"freeze file escapes output root: {path}") from exc
    return {"path": stored_path, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _checkpoint_entries(root: Path, checkpoint_paths: Any) -> list[dict[str, Any]]:
    if checkpoint_paths is None:
        return []
    if isinstance(checkpoint_paths, Mapping):
        items = list(checkpoint_paths.items())
    else:
        items = [(Path(p).name, p) for p in checkpoint_paths]
    entries = []
    for label, raw in sorted(items, key=lambda kv: str(kv[0])):
        path = _resolve_path(root, raw)
        if not path.is_file():
            raise FreezeError(f"checkpoint {label!r} not found at {path}")
        entries.append({"name": str(label), "path": str(path.resolve()),
                        "sha256": _sha256(path), "bytes": path.stat().st_size})
    return entries


def _nuisance_entries(root: Path, nuisance_files: Any) -> list[dict[str, Any]]:
    if nuisance_files is None:
        return []
    if isinstance(nuisance_files, Mapping):
        items = list(nuisance_files.items())
    elif isinstance(nuisance_files, (str, Path)):
        items = [(Path(nuisance_files).name, nuisance_files)]
    else:
        items = [(Path(path).name, path) for path in nuisance_files]
    entries = []
    for label, raw in sorted(items, key=lambda kv: str(kv[0])):
        path = _resolve_path(root, raw)
        if not path.is_file():
            raise FreezeError(f"nuisance file {label!r} not found at {path}")
        entries.append({"name": str(label), "path": str(path.resolve()),
                        "sha256": _sha256(path), "bytes": path.stat().st_size})
    return entries


def _entry_value(entry: Mapping[str, Any], field: str) -> Any:
    """Read lineage fields from descriptors and legacy nested W3 records."""
    value = entry.get(field)
    if value is not None:
        return value
    nested = entry.get("record")
    if isinstance(nested, Mapping):
        return nested.get(field)
    return None


def _entry_unit_key(entry: Mapping[str, Any]) -> str | None:
    """Stable unit coverage key used by the final freeze completeness gate."""
    kind = str(entry.get("kind", "volume"))
    if kind == "unit" and entry.get("unit_id") is not None:
        return str(entry["unit_id"])
    value = entry.get("unit_id") or entry.get("volume_id") or entry.get("study_id")
    return str(value) if value is not None else None


def freeze_predictions(output_dir: str | Path, entries: Sequence[Mapping[str, Any]],
                       checkpoint_paths: Any, *, dataset: str, protocol: str,
                       epoch: int = 149,
                       required_methods: Iterable[str] | None = None,
                       required_unit_ids: Any = None,
                       required_checkpoints: Iterable[str] | Mapping[str, Any] | None = None,
                       required_nuisance_files: Any = None,
                       nuisance_files: Any = None) -> dict[str, Any]:
    """Hash every compared prediction and checkpoint into one immutable manifest.

    ``entries`` are the descriptors returned by :func:`assemble_volume` (or
    :func:`assemble_all`). ``checkpoint_paths`` maps a component name to a
    checkpoint file. The manifest enumerates the whole compared method set,
    records the data/observation lineage every entry agreed on, and reports
    completeness against :data:`REQUIRED_COMPARISONS`.  Finalization may
    additionally provide ``required_unit_ids``, ``required_checkpoints`` and
    ``required_nuisance_files``; those requirements are recorded in the
    manifest and enforced by ``validate_freeze(..., require_complete=True)``.

    Writing is once-only. Re-freezing byte-identical content returns the
    existing manifest; re-freezing *different* content raises
    :class:`FreezeError` rather than replacing a manifest other artifacts and
    reports already cite.
    """
    root = Path(output_dir)
    if not entries:
        raise FreezeError("refusing to freeze an empty comparison set")

    if required_nuisance_files is not None and nuisance_files is not None:
        if _artifact_names(required_nuisance_files) != _artifact_names(nuisance_files):
            raise FreezeError("required_nuisance_files and nuisance_files disagree")
    nuisance_spec = (required_nuisance_files if required_nuisance_files is not None
                     else nuisance_files)

    # A mapping passed as ``required_checkpoints`` can also be the checkpoint
    # source when the caller has no separate ``checkpoint_paths`` argument.
    checkpoint_source = checkpoint_paths
    if isinstance(required_checkpoints, Mapping) and checkpoint_paths is None:
        checkpoint_source = required_checkpoints
    if required_checkpoints is None:
        if isinstance(checkpoint_source, Mapping):
            required_checkpoint_names = _as_name_list(checkpoint_source)
        elif checkpoint_source is None:
            required_checkpoint_names = []
        else:
            required_checkpoint_names = [Path(p).name for p in checkpoint_source]
    else:
        required_checkpoint_names = _as_name_list(required_checkpoints)

    lineage_fields = ("dataset", "protocol", "partition_id", "manifest_id", "version")
    lineage: dict[str, Any] = {}
    for field in lineage_fields:
        # Per-unit shards do not carry every lineage field; only the entries that
        # state a value have to agree. A field nobody states stays None and is
        # reported as None, not invented.
        values = {json.dumps(_jsonable(_entry_value(e, field)), sort_keys=True, default=str)
                  for e in entries if _entry_value(e, field) is not None}
        # Role partitions are study/volume scoped by design. Every other
        # lineage identity must agree globally; a single partition id is kept
        # as a scalar for compatibility, while multiple ids are explicit.
        if field != "partition_id" and len(values) > 1:
            raise FreezeError(
                f"entries disagree on {field}: {sorted(values)}; a freeze must describe "
                "one dataset/protocol/partition/label-version identity")
        if not values:
            lineage[field] = None
        elif field == "partition_id" and len(values) > 1:
            lineage[field] = sorted(json.loads(value) for value in values)
        else:
            lineage[field] = json.loads(next(iter(values)))

    if lineage["dataset"] is not None and dataset is not None \
            and lineage["dataset"] != dataset:
        raise FreezeError(f"entries carry dataset={lineage['dataset']!r}, freeze says {dataset!r}")
    if lineage["protocol"] is not None and protocol is not None \
            and lineage["protocol"] != protocol:
        raise FreezeError(
            f"entries carry protocol={lineage['protocol']!r}, freeze says {protocol!r}")

    # The trainer streams per-unit shards as it finalizes and hands that list
    # straight to the freeze. Assembly therefore happens here, at finalization,
    # exactly as the contract specifies - the caller does not have to know about
    # a separate assembly step, and a freeze can never enumerate units whose
    # volume was never built. Entries that are already volume descriptors pass
    # through untouched, so an explicit assemble_volume/assemble_all caller keeps
    # working.
    unit_entries = [e for e in entries if e.get("kind") == "unit"]
    resolved: list[Mapping[str, Any]] = [e for e in entries if e.get("kind") != "unit"]
    if unit_entries:
        volumes: dict[tuple[str, str], None] = {}
        for entry in unit_entries:
            volume = (entry.get("volume_id") or _entry_value(entry, "volume_id") or
                      entry.get("study_id") or _entry_value(entry, "study_id"))
            if volume is None:
                raise FreezeError(
                    f"unit entry {entry.get('unit_id')!r} has no volume_id; refusing to group by patient")
            volumes[(str(entry["prediction_name"]), str(volume))] = None
        for prediction_name, volume_id in volumes:
            resolved.append(assemble_volume(root, prediction_name=prediction_name,
                                            volume_id=volume_id))
        # The per-unit shards stay in the freeze too: they are the bytes the
        # volume was built from, and a verifier must be able to re-derive it.
        resolved.extend(unit_entries)
    entries = resolved

    predictions = []
    for entry in sorted(entries, key=lambda e: (str(e.get("prediction_name")),
                                                str(e.get("volume_id") or e.get("study_id")),
                                                str(e.get("kind")),
                                                str(e.get("unit_id")))):
        files = [_file_entry(root, rel) for rel in entry.get("files", [])]
        if not files:
            raise FreezeError(
                f"entry {entry.get('prediction_name')!r}/{entry.get('study_id')!r} lists no files")
        predictions.append({
            "kind": entry.get("kind", "volume"),
            "unit_id": entry.get("unit_id"),
            "prediction_name": entry.get("prediction_name"),
            "study_id": entry.get("study_id"),
            "volume_id": entry.get("volume_id") or entry.get("study_id"),
            "patient_id": entry.get("patient_id"),
            "split": entry.get("split"),
            "dataset": _entry_value(entry, "dataset"),
            "protocol": _entry_value(entry, "protocol"),
            "partition_id": _entry_value(entry, "partition_id"),
            "manifest_id": _entry_value(entry, "manifest_id"),
            "checkpoint_id": entry.get("checkpoint_id"),
            "version": entry.get("version"),
            "grid": entry.get("grid"),
            "native_export_available": entry.get("native_export_available"),
            "native_export_reason": entry.get("native_export_reason") or entry.get("reason"),
            "volume_shape": entry.get("volume_shape"),
            "complete_volume": entry.get("complete_volume"),
            "missing_slice_indices": entry.get("missing_slice_indices"),
            "files": files,
        })

    checkpoints = _checkpoint_entries(root, checkpoint_source)
    nuisance = _nuisance_entries(root, nuisance_spec)

    # partition_ids and the flat path->sha256 `files` map are what the data
    # layer's own freeze-receipt gate requires before it will open a sealed
    # O_verify observation, and `manifest_id` there means the DATA manifest, not
    # this freeze. There is exactly one freeze manifest and one validate_freeze;
    # this shape satisfies both readers rather than spawning a parallel receipt.
    partition_ids = sorted({str(_entry_value(e, "partition_id")) for e in entries
                            if _entry_value(e, "partition_id") is not None})
    if lineage["partition_id"] is not None and not partition_ids:
        if isinstance(lineage["partition_id"], list):
            partition_ids = [str(value) for value in lineage["partition_id"]]
        else:
            partition_ids = [str(lineage["partition_id"])]

    method_units: dict[str, set[str]] = {}
    for entry in entries:
        method = entry.get("prediction_name")
        if method is None:
            continue
        method_name = str(method)
        # Per-unit descriptors are the authoritative coverage identity.  A
        # volume descriptor is used only when the caller supplied volumes
        # directly and no per-unit list exists for that method; otherwise it
        # would appear as a spurious extra unit in the strict final gate.
        if str(entry.get("kind", "volume")) == "unit":
            unit_key = _entry_unit_key(entry)
            if unit_key is not None:
                method_units.setdefault(method_name, set()).add(unit_key)
    if not unit_entries:
        for entry in entries:
            method = entry.get("prediction_name")
            unit_key = _entry_unit_key(entry)
            if method is not None and unit_key is not None:
                method_units.setdefault(str(method), set()).add(unit_key)

    completeness = comparison_completeness(
        (str(p["prediction_name"]) for p in predictions),
        required_methods=required_methods,
        method_units=method_units,
        required_unit_ids=required_unit_ids,
        present_checkpoints=(checkpoint["name"] for checkpoint in checkpoints),
        required_checkpoints=required_checkpoint_names,
        present_nuisance_files=(entry["name"] for entry in nuisance),
        required_nuisance_files=_artifact_names(nuisance_spec),
    )

    files_map: dict[str, str] = {}
    for prediction in predictions:
        for entry in prediction["files"]:
            files_map[str((root / entry["path"]).resolve())] = entry["sha256"]
    for checkpoint in checkpoints:
        files_map[str(Path(checkpoint["path"]).resolve())] = checkpoint["sha256"]
    for nuisance_file in nuisance:
        files_map[str(Path(nuisance_file["path"]).resolve())] = nuisance_file["sha256"]

    body = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "contract_version": VERSION,
        "dataset": dataset if dataset is not None else lineage["dataset"],
        "manifest_id": lineage["manifest_id"],   # the DATA manifest this run used
        "partition_ids": partition_ids,
        "protocol": protocol if protocol is not None else lineage["protocol"],
        "epoch": int(epoch),
        "semantic_order": {str(k): v for k, v in SEMANTIC_ORDER.items()},
        "lineage": lineage,
        "compared_methods": sorted({str(p["prediction_name"]) for p in predictions}),
        "method_units": {method: sorted(units) for method, units in sorted(method_units.items())},
        "required_unit_ids": _jsonable_unit_requirements(required_unit_ids),
        "required_checkpoints": required_checkpoint_names,
        "required_nuisance_files": _artifact_names(nuisance_spec),
        "completeness": completeness,
        "predictions": predictions,
        "checkpoints": checkpoints,
        "nuisance_files": nuisance,
        "files": files_map,
    }
    manifest = dict(body)
    manifest["freeze_id"] = _canonical_hash(body)
    manifest["created_utc"] = _utc_now()
    manifest["root"] = str(root)

    path = root / "freeze_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("freeze_id") == manifest["freeze_id"]:
            return previous
        raise FreezeError(
            f"{path} already holds freeze {previous.get('freeze_id')}; refusing to overwrite "
            f"with divergent freeze {manifest['freeze_id']}. Use a fresh run directory.")
    _write_json(path, manifest)
    return manifest


def _validate_manifest_completeness(payload: Mapping[str, Any], *, require_complete: bool) -> None:
    """Validate the manifest's own coverage claims before it is used as a receipt."""
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise FreezeError("freeze manifest has no prediction entries")
    predicted_methods = sorted({str(item.get("prediction_name")) for item in predictions})
    declared_methods = sorted({str(name) for name in payload.get("compared_methods", [])})
    if predicted_methods != declared_methods:
        raise FreezeError(
            f"freeze compared_methods disagree with predictions: {declared_methods} vs "
            f"{predicted_methods}")

    completeness = payload.get("completeness")
    if not isinstance(completeness, Mapping):
        raise FreezeError("freeze manifest has no completeness block")
    required_methods = completeness.get("required", list(REQUIRED_COMPARISONS))
    if not isinstance(required_methods, (list, tuple)):
        raise FreezeError("freeze completeness.required must be a list")
    nested_units = completeness.get("units") or {}
    method_units = payload.get("method_units") or nested_units.get("present_by_method") or {}
    required_by_method = nested_units.get("required_by_method") if isinstance(nested_units, Mapping) else None
    required_units = required_by_method if isinstance(required_by_method, Mapping) else None
    nested_checkpoints = completeness.get("checkpoints") or {}
    nested_nuisance = completeness.get("nuisance_files") or {}
    recomputed = comparison_completeness(
        declared_methods,
        required_methods=required_methods,
        method_units=method_units,
        required_unit_ids=required_units,
        present_checkpoints=(entry.get("name") for entry in payload.get("checkpoints", [])),
        required_checkpoints=nested_checkpoints.get("required", []),
        present_nuisance_files=(entry.get("name") for entry in payload.get("nuisance_files", [])),
        required_nuisance_files=nested_nuisance.get("required", []),
    )
    for key in ("required", "present", "missing", "extra", "complete"):
        if completeness.get(key) != recomputed.get(key):
            raise FreezeError(
                f"freeze completeness.{key} disagrees with its prediction/artifact entries")
    if require_complete:
        if not completeness.get("complete"):
            raise FreezeError(
                "freeze is incomplete: "
                f"missing methods={completeness.get('missing')}, "
                f"missing units={(nested_units or {}).get('missing_by_method', {})}, "
                f"missing checkpoints={(nested_checkpoints or {}).get('missing', [])}, "
                f"missing nuisance files={(nested_nuisance or {}).get('missing', [])}")
        incomplete_volumes = [
            f"{item.get('prediction_name')}/{item.get('volume_id') or item.get('study_id')}"
            for item in predictions
            if str(item.get("kind", "volume")) == "volume" and item.get("complete_volume") is not True
        ]
        if incomplete_volumes:
            raise FreezeError(
                "freeze contains incomplete volumes: " + ", ".join(sorted(incomplete_volumes)))


def _freeze_payload_and_root(
    manifest: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], dict[str, Any], Path]:
    """Load a manifest while retaining the caller's exact mapping identity."""
    if isinstance(manifest, (str, Path)):
        path = Path(manifest)
        if path.is_dir():
            path = path / "freeze_manifest.json"
        if not path.is_file():
            raise FreezeError(f"freeze manifest not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise FreezeError("freeze manifest must be a JSON object")
        root = Path(payload.get("root") or path.parent)
        return payload, payload, root
    if not isinstance(manifest, Mapping):
        raise FreezeError("freeze manifest must be a mapping, path, or run directory")
    payload = dict(manifest)
    root = Path(payload.get("root") or ".")
    return manifest, payload, root


@contextmanager
def verified_freeze_session(
    manifest: Mapping[str, Any] | str | Path,
    *,
    require_complete: bool = False,
    strict: bool | None = None,
):
    """Verify a freeze once, then permit identity-bound read-only reuse.

    Entry and exit always hash every prediction, checkpoint, nuisance and flat
    file. During the context, ``validate_freeze`` called with the exact mapping
    yielded here checks object identity and the admitted root, then skips the
    repeated manifest/corpus scan. A path or a copied dictionary is deliberately
    not eligible for that fast path. Exit validation is authoritative for any
    in-session mapping or byte mutation and raises :class:`FreezeError`.
    """
    effective_require_complete = bool(strict) if strict is not None else bool(require_complete)
    source, payload, root = _freeze_payload_and_root(manifest)
    # Ensure entry validation is a real full-hash pass, even when a caller nests
    # sessions or reuses a mapping admitted by an outer session.
    entry_token = _ACTIVE_FREEZE_SESSION.set(None)
    try:
        validate_freeze(source, require_complete=effective_require_complete)
    finally:
        _ACTIVE_FREEZE_SESSION.reset(entry_token)

    declared = payload.get("freeze_id")
    if not isinstance(declared, str) or not declared:
        raise FreezeError("manifest has no freeze_id")
    session = {
        "manifest": source,
        "root": root.resolve(),
        "freeze_id": declared,
        "require_complete": effective_require_complete,
    }
    session_token = _ACTIVE_FREEZE_SESSION.set(session)
    try:
        # A mapping caller receives the same object it supplied. A path caller
        # receives the one object loaded above, so downstream code can pass it
        # back to validate_freeze without triggering an O(N^2) corpus rehash.
        yield source
    finally:
        _ACTIVE_FREEZE_SESSION.reset(session_token)
        # Exit validation is intentionally outside the active context so it
        # re-hashes the complete corpus and detects any in-session mutation.
        exit_token = _ACTIVE_FREEZE_SESSION.set(None)
        try:
            validate_freeze(source, require_complete=effective_require_complete)
        finally:
            _ACTIVE_FREEZE_SESSION.reset(exit_token)


def validate_freeze(
    manifest: Mapping[str, Any] | str | Path,
    *,
    require_complete: bool = False,
    strict: bool | None = None,
) -> None:
    """Verify a freeze: manifest integrity, then every physical file and checkpoint.

    Raises :class:`FreezeError` on the first discrepancy. Returns ``None`` on
    success. This checks *bytes*, not correctness: a validated freeze means the
    compared predictions are exactly the ones that were frozen, nothing more.
    ``require_complete=True`` additionally makes the method/unit/checkpoint/
    nuisance completeness block a hard gate for a final run.  The default stays
    permissive for bounded preflight/smoke freezes, which are explicitly marked
    incomplete in their manifest. ``strict`` is an alias accepted for callers
    that use that spelling.
    """
    if strict is not None:
        require_complete = bool(strict)
    source, payload, root = _freeze_payload_and_root(manifest)

    # The session is process-local and identity-bound. Entry validation has
    # already checked the complete manifest and its files; exit validation
    # repeats that full pass. Inner calls on the exact yielded object therefore
    # need only confirm root identity and the session's cached strictness. This
    # keeps a verification loop linear in its frozen corpus size, while any
    # copied/reloaded mapping follows the normal full validator below.
    active = _ACTIVE_FREEZE_SESSION.get()
    if active is not None and source is active.get("manifest"):
        if root.resolve() != active["root"]:
            raise FreezeError("active verified freeze session root changed")
        if require_complete and not active.get("require_complete", False):
            raise FreezeError(
                "active verified freeze session was opened without require_complete=True")
        return

    declared = payload.get("freeze_id")
    if not declared:
        raise FreezeError("manifest has no freeze_id")
    body = {k: v for k, v in payload.items()
            if k not in ("freeze_id", "created_utc", "root")}
    recomputed = _canonical_hash(body)
    if recomputed != declared:
        raise FreezeError(
            f"freeze manifest body was modified: freeze_id {declared} != recomputed {recomputed}")

    if payload.get("semantic_order") != {str(k): v for k, v in SEMANTIC_ORDER.items()}:
        raise FreezeError("freeze semantic_order does not match the fixed contract")
    _validate_manifest_completeness(payload, require_complete=require_complete)

    # A manifest carries both structured prediction/checkpoint sections and a
    # flat path->digest index for the data firewall. Reuse the digest computed
    # for a path within this one validation call so a normal full validation is
    # O(number of distinct files), while still checking every reference and
    # preserving the mandatory full pass on entry/exit of a verified session.
    hashed_paths: dict[Path, str] = {}

    def checked_hash(path: Path) -> str:
        key = path.resolve()
        actual = hashed_paths.get(key)
        if actual is None:
            actual = _sha256(path)
            hashed_paths[key] = actual
        return actual

    for prediction in payload.get("predictions", []):
        for entry in prediction.get("files", []):
            file_path = root / entry["path"]
            if not file_path.is_file():
                raise FreezeError(f"frozen file missing: {file_path}")
            if file_path.stat().st_size != entry["bytes"]:
                raise FreezeError(f"frozen file size changed: {file_path}")
            actual = checked_hash(file_path)
            if actual != entry["sha256"]:
                raise FreezeError(
                    f"frozen file content changed: {file_path} sha256 {actual} "
                    f"!= {entry['sha256']}")

    for checkpoint in payload.get("checkpoints", []):
        file_path = Path(checkpoint["path"])
        if not file_path.is_file():
            raise FreezeError(f"frozen checkpoint missing: {file_path}")
        actual = checked_hash(file_path)
        if actual != checkpoint["sha256"]:
            raise FreezeError(
                f"frozen checkpoint changed: {file_path} sha256 {actual} "
                f"!= {checkpoint['sha256']}")

    for nuisance in payload.get("nuisance_files", []):
        file_path = Path(nuisance["path"])
        if not file_path.is_file():
            raise FreezeError(f"frozen nuisance file missing: {file_path}")
        actual = checked_hash(file_path)
        if actual != nuisance["sha256"]:
            raise FreezeError(
                f"frozen nuisance file changed: {file_path} sha256 {actual} "
                f"!= {nuisance['sha256']}")

    # The flat map the data layer reads. It covers the same bytes by absolute
    # path; checking it too means the two readers can never disagree about
    # whether a freeze is intact.
    for raw_path, expected in (payload.get("files") or {}).items():
        file_path = Path(raw_path)
        if not file_path.is_file():
            raise FreezeError(f"frozen file missing: {file_path}")
        actual = checked_hash(file_path)
        if actual != expected:
            raise FreezeError(
                f"frozen file content changed: {file_path} sha256 {actual} != {expected}")
