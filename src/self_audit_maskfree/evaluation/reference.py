"""Isolated reference evaluator - the ONLY module allowed to read manual masks.

Why this module is separate
---------------------------
Everything else in ``self_audit_maskfree`` is structurally unable to read a
segmentation mask. This module can, and therefore obeys four hard rules:

1. **It runs after the freeze, never before.** Its entry point refuses to
   compute a single number until every prediction file and every checkpoint
   listed in the freeze manifest has been re-hashed and matched.
2. **It imports nothing from training.** No trainer, no config, no producer, no
   auditor. The import graph is the enforcement: a cycle back into training
   would be visible here as an import statement, and there is none.
3. **It returns nothing to training.** Its only outputs are report files. There
   is no API that hands a decision, threshold, checkpoint choice, or label back
   into the pipeline.
4. **It never relabels per case.** The class mapping is the frozen named order
   ``0=BG, 1=RV, 2=MYO, 3=LV``. A cluster-matched number may only appear under
   :func:`oracle_cluster_matching_diagnostic`, which is a diagnostic and is
   reported as one.

Absent reference masks are an *availability* state, never a zero. If no masks
are configured, this module reports every reference metric as unavailable with a
reason and the image-only pipeline result stands on its own.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts import SEMANTIC_ORDER, VERSION
from .freeze import FreezeValidationError, load_freeze_manifest, validate_freeze_manifest
from .metrics import (
    COVERAGE_LEVELS,
    FOREGROUND_CLASSES,
    MetricContractError,
    coverage_error_curve,
    dice_iou_volume,
    edit_attribution,
    evidence_quality_association,
    metric_row,
    paired_patient_bootstrap,
    patient_macro,
    surface_metrics,
    write_csv,
    write_json,
    write_markdown,
)

#: Prediction names treated as the two lockstep student arms for the paired test.
DEFAULT_STUDENT_ARMS = ("student_audited", "student_no_audit")
CANONICAL_REFERENCE_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
_ROLE_TO_ID = {name: class_id for class_id, name in SEMANTIC_ORDER.items()}
REFERENCE_SPLITS = frozenset({"train", "dev", "test", "all", "mixed"})


class ReferenceConfigError(ValueError):
    """Raised when the isolated evaluator's configuration is unusable."""


def _parse_reference_label_map(raw: Mapping[Any, Any]) -> dict[int, int]:
    """Validate one fixed raw-integer to named-class mapping."""
    if not isinstance(raw, Mapping):
        raise ReferenceConfigError("reference_label_map must map raw integer ids to BG/RV/MYO/LV")
    parsed: dict[int, int] = {}
    for raw_id, role in raw.items():
        try:
            source = int(raw_id)
        except (TypeError, ValueError) as error:
            raise ReferenceConfigError(
                f"reference_label_map key {raw_id!r} is not an integer"
            ) from error
        if isinstance(role, str):
            target = _ROLE_TO_ID.get(role.upper())
            if target is None:
                raise ReferenceConfigError(
                    f"reference_label_map value {role!r} is not one of BG/RV/MYO/LV"
                )
        else:
            try:
                target = int(role)
            except (TypeError, ValueError) as error:
                raise ReferenceConfigError(
                    f"reference_label_map value {role!r} is not a named class"
                ) from error
        if source in parsed:
            raise ReferenceConfigError(f"duplicate raw class id {source} in reference_label_map")
        parsed[source] = target
    if set(parsed.values()) != set(SEMANTIC_ORDER):
        raise ReferenceConfigError(
            "reference_label_map must be a bijection covering BG, RV, MYO and LV"
        )
    if len(parsed) != len(SEMANTIC_ORDER):
        raise ReferenceConfigError(
            "reference_label_map must contain exactly four raw ids including background"
        )
    return parsed


@dataclass
class ReferenceCase:
    """One manual reference volume and its geometry status.

    A patient can have several frozen volumes (for example ED/ES or multiple
    cine times).  ``volume_key``/``volume_id``/``frame_index`` identify the
    exact reference for one of them.  A keyless case is only valid when that
    patient has exactly one frozen volume; applying one mask to several volumes
    is rejected as an ambiguous mapping.
    """

    patient_id: str
    mask_path: str
    volume_key: str | None = None
    volume_id: str | None = None
    study_id: str | None = None
    frame_index: int | None = None
    # For a sparse 4-D native annotation (for example M&Ms ``_sa_gt``), this
    # identifies the mask frame that is scored.  It is deliberately separate
    # from ``frame_index``: the latter belongs to the frozen image record and
    # the former is proof that the matching annotation frame was selected.
    mask_frame_index: int | None = None
    spacing: tuple[float, float, float] | None = None
    geometry_valid: bool = False
    note: str | None = None

    def resolved_volume_key(self) -> str:
        if self.volume_key is not None:
            return str(self.volume_key)
        if self.volume_id is not None:
            return str(self.volume_id)
        if self.frame_index is not None:
            return f"frame:{int(self.frame_index)}"
        if self.study_id is not None:
            return f"study:{self.study_id}"
        return "default"


@dataclass
class ReferenceConfig:
    """Explicit, separate configuration for reference evaluation.

    This never comes from the training config. A run that has no legitimate
    masks simply has no reference config, and every reference metric is then
    reported unavailable.
    """

    dataset: str
    split: str
    protocol: str
    cases: list[ReferenceCase] = field(default_factory=list)
    reference_root: str | None = None
    epoch: int | None = None
    provenance: str = "unspecified"
    initial_prediction: str = "E1_cuts_inspired_control"
    final_prediction: str = "E5"
    student_arms: tuple[str, str] = DEFAULT_STUDENT_ARMS
    # Raw reference integer -> frozen named class id.  A mapping is recorded in
    # every report and is applied once at load time; no per-case permutation is
    # ever inferred.  Configured cases must supply this mapping explicitly.
    reference_label_map: dict[int, int] | None = None
    coverage_prediction: str | None = None
    evidence_records: list[dict[str, Any]] = field(default_factory=list)
    bootstrap_iterations: int = 2000
    seed: int = 42
    oracle_diagnostic: bool = False
    # The final evaluator remains strict by default.  The isolated epoch
    # evaluator may set this only for native sparse annotations (for example
    # ED/ES masks in an all-frame image-only development freeze), where every
    # unmatched frozen volume is recorded explicitly as unavailable.
    allow_unmapped_volumes: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReferenceConfig":
        data = dict(payload)
        if "dataset" not in data:
            raise ReferenceConfigError("reference config must name a dataset")
        raw_cases = data.pop("cases", [])
        cases: list[ReferenceCase] = []
        for entry in raw_cases:
            if not isinstance(entry, Mapping):
                raise ReferenceConfigError("each reference case must be an object")
            spacing = entry.get("spacing")
            cases.append(
                ReferenceCase(
                    patient_id=str(entry["patient_id"]),
                    mask_path=str(entry["mask_path"]),
                    volume_key=(
                        None
                        if entry.get("volume_key") is None
                        else str(entry.get("volume_key"))
                    ),
                    volume_id=(
                        None
                        if entry.get("volume_id") is None
                        else str(entry.get("volume_id"))
                    ),
                    study_id=(
                        None
                        if entry.get("study_id") is None
                        else str(entry.get("study_id"))
                    ),
                    frame_index=(
                        None
                        if entry.get("frame_index") is None
                        else int(entry.get("frame_index"))
                    ),
                    mask_frame_index=(
                        None
                        if entry.get("mask_frame_index") is None
                        else int(entry.get("mask_frame_index"))
                    ),
                    spacing=None if spacing is None else tuple(float(v) for v in spacing),
                    geometry_valid=bool(entry.get("geometry_valid", False)),
                    note=entry.get("note"),
                )
            )
        arms = data.pop("student_arms", DEFAULT_STUDENT_ARMS)
        raw_label_map = data.pop("reference_label_map", None)
        if raw_cases and raw_label_map is None:
            raise ReferenceConfigError(
                "reference_label_map is required when reference cases are configured; "
                "declare the fixed raw integer -> BG/RV/MYO/LV bijection before evaluation"
            )
        label_map = _parse_reference_label_map(raw_label_map) if raw_label_map is not None else None
        data.setdefault("split", "test")
        split = data["split"]
        if not isinstance(split, str) or split.strip().lower() not in REFERENCE_SPLITS:
            raise ReferenceConfigError(
                "reference config split must be one of train, dev, test, all or mixed"
            )
        data["split"] = split.strip().lower()
        known = {field_name for field_name in cls.__dataclass_fields__ if field_name != "cases"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ReferenceConfigError(f"unknown reference config keys: {', '.join(unknown)}")
        data.setdefault("protocol", "unknown")
        if not isinstance(data.get("allow_unmapped_volumes", False), bool):
            raise ReferenceConfigError("allow_unmapped_volumes must be a boolean")
        return cls(
            cases=cases,
            student_arms=tuple(arms),
            reference_label_map=label_map,
            **data,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["student_arms"] = list(self.student_arms)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise ReferenceConfigError(f"reference config not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # lazy: only the isolated evaluator needs it

            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ReferenceConfigError("reference config must be a mapping")
        return cls.from_mapping(payload)


# ---------------------------------------------------------------------------
# volume loading
# ---------------------------------------------------------------------------
def load_volume(path: str | Path) -> np.ndarray:
    """Load a label or validity volume from .npy or NIfTI, without guessing geometry."""
    volume_path = Path(path)
    if not volume_path.is_file():
        raise FileNotFoundError(f"volume not found: {volume_path}")
    suffix = "".join(volume_path.suffixes[-2:]).lower()
    if volume_path.suffix.lower() == ".npy":
        return np.load(volume_path)
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        import nibabel  # lazy: only the isolated evaluator needs it

        return np.asarray(nibabel.load(str(volume_path)).dataobj)
    raise ReferenceConfigError(f"unsupported volume format: {volume_path.name}")


def _spacing_to_mm(values: Any, unit: Any) -> tuple[float, float, float] | None:
    """Convert a three-axis spacing declaration to millimetres."""
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        return None
    normalized = str(unit or "").strip().lower()
    scale = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "cm": 10.0,
        "centimeter": 10.0,
        "centimeters": 10.0,
        "m": 1000.0,
        "meter": 1000.0,
        "meters": 1000.0,
        "um": 0.001,
        "µm": 0.001,
        "micron": 0.001,
        "microns": 0.001,
    }.get(normalized)
    if scale is None:
        return None
    return tuple(float(value * scale) for value in array)


def _affine_array(value: Any) -> np.ndarray | None:
    """Return a finite 4x4 affine or None when no actual affine is available."""
    if value is None:
        return None
    try:
        affine = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return None
    return affine


def _load_volume_with_geometry(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one volume and retain only geometry read from that volume itself."""
    volume_path = Path(path)
    suffix = "".join(volume_path.suffixes[-2:]).lower()
    if volume_path.suffix.lower() == ".npy":
        return np.asarray(np.load(volume_path)), {
            "path": str(volume_path),
            "format": "npy",
            "affine": None,
            "spacing_mm": None,
            "spacing_valid": False,
            "native_file": False,
            "spatial_unit": None,
        }
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        import nibabel  # lazy: only the isolated evaluator needs it

        image = nibabel.load(str(volume_path))
        header = image.header
        units = header.get_xyzt_units()[0] if hasattr(header, "get_xyzt_units") else None
        zooms = header.get_zooms()[:3]
        spacing_mm = _spacing_to_mm(zooms, units)
        return np.asarray(image.dataobj), {
            "path": str(volume_path),
            "format": "nifti",
            "affine": _affine_array(image.affine),
            "spacing_mm": spacing_mm,
            "spacing_valid": spacing_mm is not None,
            "native_file": True,
            "spatial_unit": units,
        }
    raise ReferenceConfigError(f"unsupported volume format: {volume_path.name}")


def _sidecar_geometry(
    entry: Mapping[str, Any],
    base: Path,
) -> dict[str, Any]:
    """Read frozen volume geometry metadata from the hashed sidecar, if present."""
    sidecar_path = _entry_path(entry, "volume_path", "volume", "sidecar_path", "sidecar")
    if sidecar_path is None:
        payload: Mapping[str, Any] = {}
    else:
        resolved = Path(sidecar_path)
        if not resolved.is_absolute():
            resolved = base / resolved
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise FreezeValidationError(
                f"frozen volume sidecar is unreadable: {resolved}: {error}"
            ) from error
        if not isinstance(raw, Mapping):
            raise FreezeValidationError(f"frozen volume sidecar must be an object: {resolved}")
        payload = raw
    geometry = dict(payload)
    nested = entry.get("geometry")
    if isinstance(nested, Mapping):
        geometry.update(nested)
    for key in (
        "grid",
        "native_export_available",
        "native_export_reason",
        "affine",
        "spacing",
        "spacing_mm",
        "spacing_valid",
        "spatial_unit",
        "xyzt_units",
    ):
        if entry.get(key) is not None:
            geometry[key] = entry[key]
    return geometry


def _prediction_geometry(
    entry: Mapping[str, Any],
    labels_geometry: Mapping[str, Any],
    base: Path,
) -> dict[str, Any]:
    """Combine actual label-file geometry with the frozen export declaration."""
    declared = _sidecar_geometry(entry, base)
    actual_affine = _affine_array(labels_geometry.get("affine"))
    declared_affine = _affine_array(declared.get("affine"))
    if actual_affine is not None and declared_affine is not None and not np.allclose(
        actual_affine, declared_affine, rtol=1e-5, atol=1e-5
    ):
        raise MetricContractError(
            f"frozen prediction {entry.get('prediction_name')!r} sidecar affine does not "
            "match the actual label NIfTI affine"
        )
    declared_spacing = declared.get("spacing_mm")
    if declared_spacing is not None:
        # A field named ``spacing_mm`` is still validated as data rather than
        # trusted by name.  This keeps malformed sidecars unavailable instead
        # of allowing a later NumPy comparison to fail ambiguously.
        declared_spacing = _spacing_to_mm(declared_spacing, "mm")
    if declared_spacing is None:
        declared_spacing = _spacing_to_mm(
            declared.get("spacing"),
            declared.get("spatial_unit") or declared.get("xyzt_units"),
        )
    actual_spacing = labels_geometry.get("spacing_mm")
    if declared_spacing is not None and actual_spacing is not None:
        if not np.allclose(
            np.asarray(declared_spacing), np.asarray(actual_spacing), rtol=1e-5, atol=1e-5
        ):
            raise MetricContractError(
                f"frozen prediction {entry.get('prediction_name')!r} sidecar spacing does not "
                "match the actual label NIfTI spacing"
            )
    grid = str(declared.get("grid") or "").strip().lower()
    native_declared = bool(declared.get("native_export_available") is True)
    spacing_valid = bool(declared.get("spacing_valid") is True)
    reasons: list[str] = []
    if grid != "native":
        reasons.append("frozen prediction is not labelled native")
    if not native_declared:
        reasons.append("frozen native_export_available is false")
    if actual_affine is None:
        reasons.append("actual prediction label file has no NIfTI affine")
    if not spacing_valid:
        reasons.append("frozen spacing_valid is false or missing")
    if declared_spacing is None:
        reasons.append("frozen spacing cannot be converted to millimetres")
    return {
        "grid": grid,
        "native_export_available": native_declared,
        "affine": actual_affine,
        "spacing_mm": declared_spacing,
        "spacing_valid": spacing_valid and declared_spacing is not None,
        "native_geometry_valid": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "sidecar": declared,
    }


def _assert_grid_compatibility(
    name: str,
    patient: str,
    volume_key: str,
    prediction_geometry: Mapping[str, Any],
    reference_geometry: Mapping[str, Any],
) -> None:
    """Reject same-shape predictions and references whose affine cannot match."""
    prediction_affine = _affine_array(prediction_geometry.get("affine"))
    reference_affine = _affine_array(reference_geometry.get("affine"))
    prediction_is_native = prediction_geometry.get("grid") == "native"
    reference_is_native = bool(reference_geometry.get("native_file"))
    if prediction_is_native and prediction_affine is None:
        raise MetricContractError(
            f"{name}/{patient}/{volume_key}: frozen native prediction has no actual NIfTI affine"
        )
    if reference_is_native and reference_affine is None:
        raise MetricContractError(
            f"{name}/{patient}/{volume_key}: native reference has no finite affine"
        )
    if (prediction_affine is None) != (reference_affine is None):
        raise MetricContractError(
            f"{name}/{patient}/{volume_key}: prediction/reference affine availability differs; "
            "same-shaped arrays cannot establish a shared grid"
        )
    if prediction_affine is not None and not np.allclose(
        prediction_affine, reference_affine, rtol=1e-5, atol=1e-5
    ):
        raise MetricContractError(
            f"{name}/{patient}/{volume_key}: prediction/reference native affines differ; "
            "overlap metrics refuse a same-shape but misregistered grid"
        )


def _surface_gate(
    prediction_geometry: Mapping[str, Any],
    reference_geometry: Mapping[str, Any],
    case: ReferenceCase,
) -> tuple[bool, tuple[float, float, float] | None, str]:
    """Determine whether physical surface distances have defensible millimetres."""
    if not case.geometry_valid:
        return False, None, "reference case did not declare valid native geometry"
    if not bool(prediction_geometry.get("native_geometry_valid")):
        return False, None, (
            "frozen prediction geometry is not native/valid: "
            + str(prediction_geometry.get("reason") or "missing geometry evidence")
        )
    if not bool(reference_geometry.get("native_file")):
        return False, None, "reference mask is not an actual native NIfTI volume"
    reference_spacing = reference_geometry.get("spacing_mm")
    prediction_spacing = prediction_geometry.get("spacing_mm")
    if reference_spacing is None or prediction_spacing is None:
        return False, None, "prediction/reference spacing is not valid convertible millimetres"
    if not np.allclose(
        np.asarray(reference_spacing), np.asarray(prediction_spacing), rtol=1e-5, atol=1e-5
    ):
        raise MetricContractError(
            "prediction/reference native spacing differs; physical surfaces are not comparable"
        )
    if case.spacing is not None:
        configured = _spacing_to_mm(case.spacing, "mm")
        if configured is None:
            raise ReferenceConfigError("reference case spacing must be three positive finite millimetres")
        if not np.allclose(
            np.asarray(configured), np.asarray(reference_spacing), rtol=1e-5, atol=1e-5
        ):
            raise ReferenceConfigError(
                "reference case spacing disagrees with the actual reference NIfTI spacing"
            )
    return True, tuple(float(value) for value in prediction_spacing), "native geometry and spacing verified"


def _entry_path(entry: Mapping[str, Any], *keys: str) -> str | None:
    """Find a named artifact inside a frozen entry, whatever shape W7 wrote.

    Two shapes are accepted: an explicit ``labels_path``-style key, or W7's
    ``files`` list, in which the artifact is identified by its basename stem
    (``labels.nii.gz``, ``validity.npy``). NIfTI is preferred over ``.npy`` when
    both were written, because that is the native-grid export.
    """
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping):
            nested = value.get("path")
            if isinstance(nested, str) and nested:
                return nested
    stems = {key.split("_")[0] for key in keys}
    files = entry.get("files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        matches: list[str] = []
        for item in files:
            path = item.get("path") if isinstance(item, Mapping) else item
            if not isinstance(path, str):
                continue
            name = Path(path).name
            if any(name in {f"{stem}.npy", f"{stem}.json"} or name.startswith(f"{stem}.nii")
                   for stem in stems):
                matches.append(path)
        for path in matches:  # prefer the native-grid NIfTI when both exist
            if ".nii" in Path(path).name:
                return path
        if matches:
            return matches[0]
    return None


def _entry_volume_key(entry: Mapping[str, Any], *, labels_path: str | None = None) -> str:
    """Return the frozen volume/frame key without collapsing patient entries."""
    for key in ("volume_key", "volume_id", "reference_volume_key", "reference_key"):
        value = entry.get(key)
        if value is not None and str(value):
            return str(value)
    for key in ("frame_index", "frame_id", "time_index", "time_id"):
        value = entry.get(key)
        if value is not None and str(value):
            return f"frame:{value}"
    study = entry.get("study_id")
    patient = entry.get("patient_id")
    if study is not None and patient is not None and str(study) != str(patient):
        return f"study:{study}"
    # W7 volume descriptors currently identify a unique acquisition through
    # their frozen label path.  This fallback keeps two same-patient volumes
    # distinct even when an older manifest omitted frame metadata.
    if labels_path:
        return f"path:{labels_path}"
    return "default"


def _entry_split(entry: Mapping[str, Any]) -> str | None:
    """Return the split recorded on a frozen volume or its source record."""
    value = entry.get("split")
    if value is None and isinstance(entry.get("record"), Mapping):
        value = entry["record"].get("split")
    if value is None:
        return None
    return str(value).strip().lower()


def index_frozen_predictions(
    manifest: Mapping[str, Any],
    *,
    split: str | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Group entries as ``{method: {patient: {volume_or_frame: entry}}}``.

    The manifest is W7's artifact; this reader is deliberately tolerant about
    which key carries the path, and deliberately strict about identity: an entry
    without a prediction name or a patient id cannot be attributed and is
    rejected rather than quietly dropped.  Volume/frame keys are retained so a
    second ED/ES or cine-time entry never overwrites the first one.
    """
    requested_split: str | None = None
    if split is not None:
        requested_split = str(split).strip().lower()
        if requested_split not in REFERENCE_SPLITS:
            raise ReferenceConfigError(
                "reference prediction split must be one of train, dev, test, all or mixed"
            )
    entries = manifest.get("methods") or manifest.get("predictions") or manifest.get("entries")
    if isinstance(entries, Mapping):
        flattened: list[Mapping[str, Any]] = []
        for name, value in entries.items():
            if isinstance(value, Mapping) and "patient_id" in value:
                flattened.append({**value, "prediction_name": name})
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    flattened.append({**item, "prediction_name": name})
            else:
                raise FreezeValidationError(f"unreadable freeze entry for {name!r}")
        entries = flattened
    if not isinstance(entries, Sequence):
        raise FreezeValidationError("freeze manifest lists no prediction entries")

    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise FreezeValidationError("freeze prediction entries must be objects")
        if entry.get("kind") == "unit":
            # Per-unit shards are the bytes a volume was assembled from. They are
            # frozen and hashed, but reference metrics are defined on whole
            # native volumes, so only volume entries are evaluated here.
            continue
        entry_split = _entry_split(entry)
        if requested_split not in (None, "all", "mixed") and entry_split != requested_split:
            continue
        name = entry.get("prediction_name") or entry.get("name") or entry.get("method")
        patient = entry.get("patient_id") or entry.get("study_id") or entry.get("case_id")
        if name is None or patient is None:
            raise FreezeValidationError(
                "every frozen prediction entry needs a prediction name and a patient identity"
            )
        normalized = dict(entry)
        labels_path = _entry_path(
            normalized, "labels_path", "labels", "path", "prediction_path"
        )
        volume_key = _entry_volume_key(normalized, labels_path=labels_path)
        method = grouped.setdefault(str(name), {})
        patient_entries = method.setdefault(str(patient), {})
        if volume_key in patient_entries:
            raise FreezeValidationError(
                f"duplicate frozen volume key {str(name)!r}/{str(patient)!r}/{volume_key!r}; "
                "reference evaluation refuses to overwrite an acquisition"
            )
        normalized["_volume_key"] = volume_key
        patient_entries[volume_key] = normalized
    return grouped


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def oracle_cluster_matching_diagnostic(
    predictions: Mapping[str, np.ndarray],
    references: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Best single global class permutation, as a labelled diagnostic only.

    One permutation is chosen for the whole cohort - never per case - and the
    result is returned under this function's name so it cannot be mistaken for
    the named-mapping metric. It must never drive selection, thresholds or
    checkpoints; it exists to show how much of a low score is a naming failure
    rather than a partition failure.
    """
    shared = sorted(set(predictions) & set(references))
    if not shared:
        return {
            "available": False,
            "reason": "no patient has both a prediction and a reference",
            "diagnostic": True,
        }
    best: dict[str, Any] | None = None
    for permutation in permutations(range(4)):
        mapping = np.array(permutation, dtype=np.int64)
        per_patient: dict[str, dict[int, dict[str, Any]]] = {}
        for patient in shared:
            remapped = mapping[np.asarray(predictions[patient], dtype=np.int64)]
            per_patient[patient] = dice_iou_volume(remapped, references[patient])
        aggregate = patient_macro(per_patient, key="dice")
        value = aggregate["macro_mean"]
        if value is None:
            continue
        if best is None or value > best["macro_dice"]:
            best = {
                "permutation": list(permutation),
                "macro_dice": float(value),
                "patients_counted": aggregate["patients_counted"],
            }
    if best is None:
        return {
            "available": False,
            "reason": "no permutation produced a defined macro Dice",
            "diagnostic": True,
        }
    return {
        "available": True,
        "diagnostic": True,
        "reason": None,
        **best,
        "warning": (
            "oracle_cluster_matching_diagnostic: a global oracle permutation chosen with the "
            "reference. Not a reported accuracy and never used for any selection."
        ),
    }


# ---------------------------------------------------------------------------
# the evaluator
# ---------------------------------------------------------------------------
def _reference_volume_key(case: ReferenceCase) -> str:
    """Canonical key for one configured reference case."""
    return case.resolved_volume_key()


def _reference_catalog(
    config: ReferenceConfig,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    """Load configured masks and index them by patient and explicit volume key."""
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[str] = []
    if config.reference_label_map is None and config.cases:
        raise ReferenceConfigError(
            "reference_label_map is required when reference cases are configured; "
            "no canonical mapping may be inferred from native or synthetic masks"
        )
    label_map = (
        CANONICAL_REFERENCE_LABEL_MAP
        if config.reference_label_map is None
        else _parse_reference_label_map(config.reference_label_map)
    )
    for case in config.cases:
        key = _reference_volume_key(case)
        patient = str(case.patient_id)
        if key in catalog.setdefault(patient, {}):
            raise ReferenceConfigError(
                f"duplicate reference mapping for patient {patient!r} and volume {key!r}"
            )
        mask_path = Path(case.mask_path)
        if not mask_path.is_absolute() and config.reference_root:
            mask_path = Path(config.reference_root) / mask_path
        if not mask_path.is_file():
            missing.append(f"{patient}/{key}")
            continue
        raw_reference, reference_geometry = _load_volume_with_geometry(mask_path)
        if case.mask_frame_index is not None:
            index = case.mask_frame_index
            if raw_reference.ndim != 4 or not 0 <= index < raw_reference.shape[3]:
                raise ReferenceConfigError(
                    f"reference mask_frame_index {index} is invalid for {mask_path}: {raw_reference.shape}"
                )
            raw_reference = raw_reference[..., index]
        if raw_reference.ndim != 3:
            raise ReferenceConfigError(
                f"reference mask {mask_path} must be a 3-D volume, got shape {raw_reference.shape}"
            )
        if case.geometry_valid and not reference_geometry.get("native_file"):
            raise ReferenceConfigError(
                f"reference case {patient}/{key} declares geometry_valid but its mask is not "
                "an actual NIfTI volume with native geometry"
            )
        if case.spacing is not None:
            configured_spacing = _spacing_to_mm(case.spacing, "mm")
            if configured_spacing is None:
                raise ReferenceConfigError(
                    f"reference case {patient}/{key} spacing must be three positive finite millimetres"
                )
            actual_spacing = reference_geometry.get("spacing_mm")
            if actual_spacing is not None and not np.allclose(
                np.asarray(configured_spacing), np.asarray(actual_spacing), rtol=1e-5, atol=1e-5
            ):
                raise ReferenceConfigError(
                    f"reference case {patient}/{key} spacing disagrees with its actual NIfTI header"
                )
        if not np.issubdtype(raw_reference.dtype, np.integer):
            if not np.all(np.equal(np.mod(raw_reference, 1), 0)):
                raise ReferenceConfigError(
                    f"reference mask {mask_path} contains non-integer class labels"
                )
            raw_reference = raw_reference.astype(np.int64)
        unknown = sorted(set(np.unique(raw_reference).tolist()) - set(label_map))
        if unknown:
            raise ReferenceConfigError(
                f"reference mask {mask_path} contains raw ids absent from reference_label_map: "
                + ", ".join(map(str, unknown))
            )
        reference = np.zeros(raw_reference.shape, dtype=np.int64)
        for raw_id, class_id in label_map.items():
            reference[raw_reference == raw_id] = class_id
        catalog[patient][key] = {
            "volume_key": key,
            "reference": reference,
            "case": case,
            "geometry": reference_geometry,
        }
    return catalog, missing


def _entry_reference_key(entry: Mapping[str, Any], volume_key: str) -> str:
    """Read an explicit per-volume reference key when W7 supplied one."""
    for key in ("reference_volume_key", "reference_key", "volume_key", "volume_id"):
        value = entry.get(key)
        if value is not None and str(value):
            return str(value)
    for key in ("frame_index", "frame_id", "time_index", "time_id"):
        value = entry.get(key)
        if value is not None and str(value):
            return f"frame:{value}"
    study = entry.get("study_id")
    patient = entry.get("patient_id")
    if study is not None and patient is not None and str(study) != str(patient):
        return f"study:{study}"
    return volume_key


def _resolve_reference(
    patient: str,
    volume_key: str,
    entry: Mapping[str, Any],
    catalog: Mapping[str, Mapping[str, Mapping[str, Any]]],
    patient_volume_count: int,
) -> dict[str, Any] | None:
    """Resolve exactly one mask for a frozen volume, rejecting ambiguity."""
    candidates = dict(catalog.get(patient, {}))
    if not candidates:
        return None
    requested = _entry_reference_key(entry, volume_key)
    if requested in candidates:
        return candidates[requested]
    # A keyless reference is only safe for a patient with one frozen volume.
    if "default" in candidates:
        if patient_volume_count == 1:
            return candidates["default"]
        raise ReferenceConfigError(
            f"ambiguous reference mask for patient {patient!r}: {patient_volume_count} "
            "frozen volumes require an explicit volume_key/reference_key"
        )
    if len(candidates) == 1 and patient_volume_count == 1:
        # A single explicitly keyed reference may describe a single volume even
        # when an older manifest did not preserve that key.
        return next(iter(candidates.values()))
    raise ReferenceConfigError(
        f"no unambiguous reference mapping for patient {patient!r}, volume {volume_key!r}; "
        "provide a matching volume_key/reference_key for every frozen volume"
    )


def _aggregate_volume_metrics(
    per_volume: Mapping[str, Mapping[str, Mapping[int, Mapping[str, Any]]]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Average per-volume class scores within each patient before patient macro."""
    aggregated: dict[str, dict[int, dict[str, Any]]] = {}
    for patient, volumes in per_volume.items():
        classes_map: dict[int, dict[str, Any]] = {}
        for class_id in FOREGROUND_CLASSES:
            defined: list[float] = []
            both_empty = 0
            one_empty = 0
            for scores in volumes.values():
                score = scores.get(class_id)
                if score is None or not score.get("defined", False):
                    both_empty += 1
                    continue
                defined.append(float(score["dice"]))
                one_empty += int(bool(score.get("one_empty", False)))
            if defined:
                value = float(np.mean(defined))
                classes_map[class_id] = {
                    "class_id": class_id,
                    "class_name": SEMANTIC_ORDER[class_id],
                    "dice": value,
                    "iou": float(
                        np.mean([
                            float(scores[class_id]["iou"])
                            for scores in volumes.values()
                            if scores.get(class_id, {}).get("defined", False)
                        ])
                    ),
                    "defined": True,
                    "both_empty": False,
                    "one_empty": bool(one_empty),
                    "volume_count": len(defined),
                    "both_empty_volume_count": both_empty,
                    "reason": None,
                }
            else:
                classes_map[class_id] = {
                    "class_id": class_id,
                    "class_name": SEMANTIC_ORDER[class_id],
                    "dice": None,
                    "iou": None,
                    "defined": False,
                    "both_empty": True,
                    "one_empty": False,
                    "volume_count": 0,
                    "both_empty_volume_count": both_empty,
                    "reason": "class absent from every volume for this patient",
                }
        aggregated[patient] = classes_map
    return aggregated


def _foreground_mean(scores: Mapping[int, Mapping[str, Any]] | None) -> float | None:
    """Mean defined foreground Dice for one volume, preserving empty classes."""
    if not scores:
        return None
    values = [
        float(entry["dice"])
        for class_id, entry in scores.items()
        if class_id in FOREGROUND_CLASSES and entry.get("defined", False)
    ]
    return float(np.mean(values)) if values else None


def _sum_edit_attribution(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine per-volume edit counts into one patient-level attribution."""
    count_keys = (
        "fix", "regress", "preserved_correct", "preserved_wrong", "changed_pixels",
        "evaluated_pixels", "ignored_pixels", "initially_correct_pixels",
    )
    totals = {key: int(sum(int(value.get(key, 0)) for value in values)) for key in count_keys}
    evaluated = totals["evaluated_pixels"]
    initially_correct = totals["initially_correct_pixels"]
    return {
        **totals,
        "net_quality_change": totals["fix"] - totals["regress"],
        "net_quality_rate": (
            (totals["fix"] - totals["regress"]) / evaluated if evaluated else None
        ),
        "harm_rate": totals["regress"] / initially_correct if initially_correct else None,
        "harm_rate_reason": None if initially_correct else "no initially correct evaluated pixel",
        "initial_accuracy": initially_correct / evaluated if evaluated else None,
        "final_accuracy": (
            (totals["fix"] + totals["preserved_correct"]) / evaluated if evaluated else None
        ),
        "available": evaluated > 0,
        "reason": None if evaluated else "no evaluated pixel after validity masking",
    }


def _flatten_volume_mapping(
    volumes: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    references: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Flatten patient/volume maps for the global oracle diagnostic only."""
    predictions: dict[str, np.ndarray] = {}
    truth: dict[str, np.ndarray] = {}
    for patient, per_volume in volumes.items():
        for volume_key, prediction in per_volume.items():
            if volume_key in references.get(patient, {}):
                identity = f"{patient}/{volume_key}"
                predictions[identity] = prediction
                truth[identity] = references[patient][volume_key]
    return predictions, truth


def _freeze_lineage_value(manifest: Mapping[str, Any], field: str) -> Any:
    """Read a freeze-level lineage value, including the W7 lineage block."""
    value = manifest.get(field)
    if value is None and isinstance(manifest.get("lineage"), Mapping):
        value = manifest["lineage"].get(field)
    return value


def _same_lineage_value(field: str, expected: Any, actual: Any) -> bool:
    """Compare provenance without allowing a stringified epoch to drift."""
    if field == "epoch":
        if isinstance(expected, bool) or isinstance(actual, bool):
            return False
        try:
            return int(expected) == int(actual)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(actual)


def _validate_freeze_provenance(
    manifest: Mapping[str, Any],
    config: ReferenceConfig,
) -> None:
    """Require the reference config to identify exactly the frozen run."""
    for field in ("dataset", "protocol", "epoch"):
        expected = getattr(config, field)
        actual = _freeze_lineage_value(manifest, field)
        if expected is None or actual is None or not _same_lineage_value(field, expected, actual):
            raise ReferenceConfigError(
                f"reference config {field}={expected!r} does not match frozen "
                f"manifest {field}={actual!r}"
            )

    predictions = manifest.get("predictions")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        return
    for entry in predictions:
        if not isinstance(entry, Mapping):
            continue
        for field in ("dataset", "protocol", "epoch"):
            actual = entry.get(field)
            if actual is not None and not _same_lineage_value(field, getattr(config, field), actual):
                raise ReferenceConfigError(
                    f"frozen prediction entry {field}={actual!r} disagrees with "
                    f"reference config {field}={getattr(config, field)!r}"
                )


def _grouped_entry_count(grouped: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> int:
    """Count frozen volume entries in a method/patient/volume index."""
    return sum(
        len(volume_entries)
        for patient_entries in grouped.values()
        for volume_entries in patient_entries.values()
    )


def _grouped_splits(
    grouped: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> list[str]:
    """List split labels actually present on frozen volume entries."""
    return sorted(
        {
            split
            for patient_entries in grouped.values()
            for volume_entries in patient_entries.values()
            for entry in volume_entries.values()
            if (split := _entry_split(entry)) is not None
        }
    )


def evaluate_reference(
    freeze_manifest: Mapping[str, Any] | str | Path,
    config: ReferenceConfig,
    *,
    output_dir: str | Path,
    freeze_root: str | Path | None = None,
) -> dict[str, Any]:
    """Score every frozen prediction against manual masks and write the reports.

    The freeze is validated first, for *all* predictions, so a partial or edited
    export cannot be evaluated. Every produced number carries dataset, split,
    protocol, epoch, checkpoint, population, count, availability, unit and
    contract version.
    """
    manifest = (
        load_freeze_manifest(freeze_manifest)
        if isinstance(freeze_manifest, (str, Path))
        else freeze_manifest
    )
    # The isolated evaluator must enforce the producer's complete comparison
    # contract before opening any reference, including absent configured masks.
    from ..export import validate_freeze
    try:
        validate_freeze(manifest, require_complete=True)
    except Exception as error:
        raise FreezeValidationError(str(error)) from error
    grouped_all = index_frozen_predictions(manifest)
    receipt = validate_freeze_manifest(
        manifest, root=freeze_root, required_methods=sorted(grouped_all)
    )
    _validate_freeze_provenance(manifest, config)
    grouped = index_frozen_predictions(manifest, split=config.split)
    frozen_volume_count = _grouped_entry_count(grouped_all)
    evaluated_volume_count = _grouped_entry_count(grouped)
    frozen_splits = _grouped_splits(grouped_all)

    output_path = Path(output_dir)
    base = Path(freeze_root) if freeze_root is not None else Path(receipt["root"])

    # Reference masks are indexed per patient *and* per explicit volume key.
    # This prevents an ED mask, for example, from being silently reused for ES.
    reference_catalog, missing_reference = _reference_catalog(config)

    rows: list[dict[str, Any]] = []
    provenance = {
        "dataset": config.dataset,
        "split": config.split,
        "protocol": config.protocol,
        "epoch": config.epoch,
    }

    def row(name: str, value: float | None, *, unit: str, checkpoint: str, population: str,
            count: int, available: bool = True, reason: str | None = None, **extra: Any) -> None:
        rows.append(
            metric_row(name, value, unit=unit, checkpoint=checkpoint, population=population,
                       count=count, available=available, reason=reason, **provenance, **extra)
        )

    if not grouped:
        reason = f"no frozen volume entry matches requested split {config.split!r}"
        report = {
            "contract_version": VERSION,
            "available": False,
            "reason": reason,
            "freeze_receipt": receipt,
            "config": config.to_dict(),
            "predictions": [],
            "frozen_predictions": sorted(grouped_all),
            "frozen_volume_count": frozen_volume_count,
            "evaluated_volume_count": evaluated_volume_count,
            "frozen_splits": frozen_splits,
            "rows": rows,
            "missing_reference_patients": [],
            "reference_label_map": (
                {}
                if config.reference_label_map is None
                else {str(raw): int(class_id) for raw, class_id in config.reference_label_map.items()}
            ),
            "reference_label_map_explicit": config.reference_label_map is not None,
            "note": "reference metrics are unavailable for the requested frozen split",
        }
        _write_reports(output_path, report, rows, config)
        return report

    if not any(reference_catalog.values()):
        reason = (
            "no readable manual reference volume was configured"
            if not config.cases
            else f"configured reference volumes missing on disk: {', '.join(missing_reference)}"
        )
        for name in sorted(grouped):
            row(f"{name}.dice_macro", None, unit="dice", checkpoint=name,
                population="patient", count=0, available=False, reason=reason)
        report = {
            "contract_version": VERSION,
            "available": False,
            "reason": reason,
            "freeze_receipt": receipt,
            "config": config.to_dict(),
            "predictions": sorted(grouped),
            "frozen_predictions": sorted(grouped_all),
            "frozen_volume_count": frozen_volume_count,
            "evaluated_volume_count": evaluated_volume_count,
            "frozen_splits": frozen_splits,
            "rows": rows,
            "missing_reference_patients": missing_reference,
            "reference_label_map": {
                str(raw): int(class_id)
                for raw, class_id in (
                    CANONICAL_REFERENCE_LABEL_MAP
                    if config.reference_label_map is None
                    else config.reference_label_map
                ).items()
            },
            "reference_label_map_explicit": config.reference_label_map is not None,
            "note": "reference metrics are unavailable; they are never zero and never invented",
        }
        _write_reports(output_path, report, rows, config)
        return report

    overlap: dict[str, dict[str, Any]] = {}
    # All three maps retain patient -> volume/frame -> value.  Patient-level
    # summaries are derived only after every volume has been scored.
    volumes: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    validity_volumes: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    prediction_geometries: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    reference_geometries: dict[str, dict[str, dict[str, Any]]] = {
        patient: {
            key: value["geometry"] for key, value in per_patient.items()
        }
        for patient, per_patient in reference_catalog.items()
    }
    references: dict[str, dict[str, np.ndarray]] = {
        patient: {
            key: value["reference"] for key, value in per_patient.items()
        }
        for patient, per_patient in reference_catalog.items()
    }
    geometry: dict[str, dict[str, ReferenceCase]] = {
        patient: {
            key: value["case"] for key, value in per_patient.items()
        }
        for patient, per_patient in reference_catalog.items()
    }
    volume_geometry: dict[str, dict[str, dict[str, ReferenceCase]]] = {}
    for name, per_patient_entries in sorted(grouped.items()):
        per_volume: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        volumes[name] = {}
        validity_volumes[name] = {}
        prediction_geometries[name] = {}
        skipped: list[str] = []
        for patient, volume_entries in sorted(per_patient_entries.items()):
            patient_volume_count = len(volume_entries)
            for volume_key, entry in sorted(volume_entries.items()):
                if (config.allow_unmapped_volumes and
                        _entry_reference_key(entry, volume_key) not in reference_catalog.get(patient, {})):
                    skipped.append(f"{patient}/{volume_key}")
                    continue
                reference_entry = _resolve_reference(
                    patient, volume_key, entry, reference_catalog, patient_volume_count
                )
                if reference_entry is None:
                    skipped.append(f"{patient}/{volume_key}")
                    continue
                # Materialize the resolved key for downstream per-volume
                # aggregation and diagnostics.  The catalog may have stored a
                # keyless ``default`` case that safely maps to this one volume.
                references.setdefault(patient, {})[volume_key] = reference_entry["reference"]
                geometry.setdefault(patient, {})[volume_key] = reference_entry["case"]
                reference_geometries.setdefault(patient, {})[volume_key] = reference_entry[
                    "geometry"
                ]
                labels_path = _entry_path(
                    entry, "labels_path", "labels", "path", "prediction_path"
                )
                if labels_path is None:
                    raise FreezeValidationError(
                        f"frozen entry for {name}/{patient}/{volume_key} has no label volume path"
                    )
                resolved = Path(labels_path)
                if not resolved.is_absolute():
                    resolved = base / resolved
                prediction_raw, labels_geometry = _load_volume_with_geometry(resolved)
                prediction = np.asarray(prediction_raw).astype(np.int64)
                reference_volume = reference_entry["reference"]
                if prediction.shape != reference_volume.shape:
                    raise MetricContractError(
                        f"{name}/{patient}/{volume_key}: prediction {prediction.shape} and "
                        f"reference {reference_volume.shape} are on different grids; no silent "
                        "resize is allowed"
                    )
                prediction_geometry = _prediction_geometry(entry, labels_geometry, base)
                _assert_grid_compatibility(
                    name,
                    patient,
                    volume_key,
                    prediction_geometry,
                    reference_entry["geometry"],
                )
                volumes[name].setdefault(patient, {})[volume_key] = prediction
                prediction_geometries[name].setdefault(patient, {})[volume_key] = (
                    prediction_geometry
                )
                volume_geometry.setdefault(name, {}).setdefault(patient, {})[volume_key] = (
                    reference_entry["case"]
                )
                per_volume.setdefault(patient, {})[volume_key] = dice_iou_volume(
                    prediction, reference_volume
                )
                validity_path = _entry_path(entry, "validity_path", "validity")
                if validity_path is not None:
                    validity_resolved = Path(validity_path)
                    if not validity_resolved.is_absolute():
                        validity_resolved = base / validity_resolved
                    validity_volumes[name].setdefault(patient, {})[volume_key] = np.asarray(
                        load_volume(validity_resolved)
                    )

        per_patient = _aggregate_volume_metrics(per_volume)
        dice_aggregate = patient_macro(per_patient, key="dice")
        iou_aggregate = patient_macro(per_patient, key="iou")
        overlap[name] = {
            "per_volume": per_volume,
            "per_patient": per_patient,
            "dice": dice_aggregate,
            "iou": iou_aggregate,
            "patients_without_reference": skipped,
            "volumes_counted": sum(len(entries) for entries in per_volume.values()),
        }
        checkpoint = name
        for candidate_volumes in per_patient_entries.values():
            for candidate_entry in candidate_volumes.values():
                recorded = candidate_entry.get("checkpoint_id")
                if isinstance(recorded, str) and recorded:
                    checkpoint = recorded
                    break
            if checkpoint != name:
                break
        row(f"{name}.dice_macro", dice_aggregate["macro_mean"], unit="dice", checkpoint=checkpoint,
            population="patient", count=dice_aggregate["patients_counted"],
            available=dice_aggregate["available"], reason=dice_aggregate["reason"],
            patients_undefined=dice_aggregate["patients_undefined"],
            patients_without_reference=skipped)
        row(f"{name}.iou_macro", iou_aggregate["macro_mean"], unit="iou", checkpoint=checkpoint,
            population="patient", count=iou_aggregate["patients_counted"],
            available=iou_aggregate["available"], reason=iou_aggregate["reason"])
        for class_id in FOREGROUND_CLASSES:
            entry = dice_aggregate["per_class"][class_id]
            row(f"{name}.dice.{SEMANTIC_ORDER[class_id]}", entry["mean"], unit="dice",
                checkpoint=checkpoint, population="patient", count=entry["count"],
                available=entry["available"], reason=entry["reason"],
                both_empty_excluded=entry["both_empty_excluded"],
                one_empty_errors=entry["one_empty_errors"])

    # ---- surface metrics, only where native geometry is valid ---------------
    surfaces: dict[str, list[dict[str, Any]]] = {}
    for name, prediction_volumes in volumes.items():
        entries: list[dict[str, Any]] = []
        for patient, per_volume in sorted(prediction_volumes.items()):
            for volume_key, prediction in sorted(per_volume.items()):
                case = volume_geometry[name][patient][volume_key]
                prediction_geometry = prediction_geometries[name][patient][volume_key]
                reference_geometry = reference_geometries[patient][volume_key]
                surface_available, surface_spacing, surface_reason = _surface_gate(
                    prediction_geometry, reference_geometry, case
                )
                for class_id in FOREGROUND_CLASSES:
                    result = surface_metrics(
                        prediction, references[patient][volume_key], class_id=class_id,
                        spacing=surface_spacing,
                        geometry_valid=surface_available,
                    )
                    if not surface_available:
                        result["reason"] = surface_reason
                    result["patient_id"] = patient
                    result["volume_key"] = volume_key
                    entries.append(result)
        surfaces[name] = entries
        for metric_name in ("hd95", "assd"):
            values = [e[metric_name] for e in entries if e["available"] and e[metric_name] is not None]
            undefined = sum(1 for e in entries if e["undefined_case"])
            units = {e["unit"] for e in entries if e["available"]}
            row(f"{name}.{metric_name}", float(np.mean(values)) if values else None,
                unit=(units.pop() if len(units) == 1 else "unavailable"),
                checkpoint=name, population="volume-class", count=len(values),
                available=bool(values),
                reason=None if values else "no class had valid native geometry and two non-empty surfaces",
                undefined_cases=undefined)

    # ---- true edit attribution ---------------------------------------------
    attribution: dict[str, Any]
    initial_name = config.initial_prediction
    final_name = config.final_prediction
    if initial_name in volumes and final_name in volumes:
        shared_patients = sorted(
            set(volumes[initial_name]) & set(volumes[final_name]) & set(references)
        )
        per_volume_attribution: dict[str, dict[str, dict[str, Any]]] = {}
        per_patient_attribution: dict[str, dict[str, Any]] = {}
        for patient in shared_patients:
            shared_volumes = sorted(
                set(volumes[initial_name][patient])
                & set(volumes[final_name][patient])
                & set(references[patient])
            )
            volume_values = {
                volume_key: edit_attribution(
                    volumes[initial_name][patient][volume_key],
                    volumes[final_name][patient][volume_key],
                    references[patient][volume_key],
                )
                for volume_key in shared_volumes
            }
            if volume_values:
                per_volume_attribution[patient] = volume_values
                per_patient_attribution[patient] = _sum_edit_attribution(
                    list(volume_values.values())
                )
        totals = {
            key: int(sum(entry[key] for entry in per_patient_attribution.values()))
            for key in ("fix", "regress", "preserved_correct", "preserved_wrong",
                        "changed_pixels", "evaluated_pixels", "ignored_pixels",
                        "initially_correct_pixels")
        }
        attribution = {
            "initial": initial_name,
            "final": final_name,
            "per_volume": per_volume_attribution,
            "per_patient": per_patient_attribution,
            "totals": totals,
            "net_quality_change": totals["fix"] - totals["regress"],
            "harm_rate": (
                totals["regress"] / totals["initially_correct_pixels"]
                if totals["initially_correct_pixels"]
                else None
            ),
            "available": bool(per_patient_attribution),
            "reason": (
                None
                if per_patient_attribution
                else "no volume had both predictions and an explicit reference"
            ),
        }
        row("edit_attribution.net_quality_change",
            float(totals["fix"] - totals["regress"]) if per_patient_attribution else None,
            unit="pixels", checkpoint=f"{initial_name}->{final_name}", population="pixel",
            count=totals["evaluated_pixels"], available=bool(per_patient_attribution),
            reason=None if per_patient_attribution else "no paired volume",
            fix=totals["fix"], regress=totals["regress"])
        row("edit_attribution.harm_rate", attribution["harm_rate"], unit="fraction",
            checkpoint=f"{initial_name}->{final_name}", population="pixel",
            count=totals["initially_correct_pixels"],
            available=attribution["harm_rate"] is not None,
            reason=None if attribution["harm_rate"] is not None else "no initially correct pixel")
    else:
        attribution = {
            "available": False,
            "reason": (
                f"edit attribution needs both {initial_name!r} and {final_name!r} in the freeze"
            ),
        }
        row("edit_attribution.net_quality_change", None, unit="pixels",
            checkpoint=f"{initial_name}->{final_name}", population="pixel", count=0,
            available=False, reason=attribution["reason"])

    # ---- evidence gain versus true quality gain -----------------------------
    association: dict[str, Any]
    if config.evidence_records and initial_name in overlap and final_name in overlap:
        initial_patient_dice = overlap[initial_name]["dice"]["per_patient_foreground"]
        final_patient_dice = overlap[final_name]["dice"]["per_patient_foreground"]
        evidence_gain: list[float] = []
        quality_gain: list[float] = []
        used: list[str] = []
        for record in config.evidence_records:
            patient = str(record.get("patient_id", record.get("study_id", "")))
            requested_volume = record.get("volume_key", record.get("volume_id"))
            gain = record.get("evidence_gain")
            if requested_volume is not None:
                volume_key = str(requested_volume)
                initial_volume = overlap[initial_name]["per_volume"].get(patient, {}).get(volume_key)
                final_volume = overlap[final_name]["per_volume"].get(patient, {}).get(volume_key)
                initial_value = _foreground_mean(initial_volume)
                final_value = _foreground_mean(final_volume)
            else:
                volume_key = None
                initial_value = initial_patient_dice.get(patient)
                final_value = final_patient_dice.get(patient)
            if initial_value is not None and final_value is not None and gain is not None:
                evidence_gain.append(float(gain))
                quality_gain.append(float(final_value - initial_value))
                used.append(patient if volume_key is None else f"{patient}/{volume_key}")
        if evidence_gain:
            association = evidence_quality_association(evidence_gain, quality_gain)
            association["patients"] = used
        else:
            association = {
                "available": False,
                "reason": "no evidence record matched a patient with both predictions scored",
            }
    else:
        association = {
            "available": False,
            "reason": "no evidence records supplied, or the compared predictions are not frozen",
        }
    row("evidence_quality.spearman", association.get("spearman"), unit="rho",
        checkpoint=f"{initial_name}->{final_name}", population="patient",
        count=int(association.get("pairs", 0) or 0),
        available=association.get("spearman") is not None,
        reason=None if association.get("spearman") is not None else association.get("reason",
              "insufficient paired support"))
    row("evidence_quality.auroc", association.get("auroc"), unit="auroc",
        checkpoint=f"{initial_name}->{final_name}", population="patient",
        count=int(association.get("positives", 0) or 0) + int(association.get("negatives", 0) or 0),
        available=association.get("auroc") is not None,
        reason=None if association.get("auroc") is not None else association.get("reason",
              "insufficient ranking support"))

    # ---- coverage versus error ----------------------------------------------
    coverage_name = config.coverage_prediction or final_name
    coverage: dict[str, Any]
    if coverage_name in volumes and validity_volumes.get(coverage_name):
        curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for patient, per_volume in sorted(volumes[coverage_name].items()):
            for volume_key, prediction in sorted(per_volume.items()):
                validity = validity_volumes[coverage_name].get(patient, {}).get(volume_key)
                if validity is None or volume_key not in references.get(patient, {}):
                    continue
                curves.setdefault(patient, {})[volume_key] = coverage_error_curve(
                    prediction, references[patient][volume_key], validity, levels=COVERAGE_LEVELS
                )
        coverage = {
            "prediction": coverage_name,
            "per_volume": curves,
            # Kept as an explicit alias for older report readers.  Each value is
            # still a volume curve; no slices are pooled as independent patients.
            "per_patient": curves,
            "available": bool(curves),
            "reason": None if curves else "no validity volume was frozen for this prediction",
        }
        for level in COVERAGE_LEVELS:
            values = [
                entry["error_rate"]
                for patient_curves in curves.values()
                for volume_curve in patient_curves.values()
                for entry in volume_curve
                if entry["coverage_level"] == level and entry["available"]
            ]
            row(f"coverage.error_rate@{int(level * 100)}",
                float(np.mean(values)) if values else None, unit="fraction",
                checkpoint=coverage_name, population="volume", count=len(values),
                available=bool(values),
                reason=None if values else "no patient produced this coverage level",
                coverage_level=level)
    else:
        coverage = {
            "prediction": coverage_name,
            "available": False,
            "reason": "no frozen validity volume for the coverage prediction",
        }
        for level in COVERAGE_LEVELS:
            row(f"coverage.error_rate@{int(level * 100)}", None, unit="fraction",
                checkpoint=coverage_name, population="volume", count=0, available=False,
                reason=coverage["reason"], coverage_level=level)

    # ---- paired patient bootstrap between the two student arms --------------
    audited_name, unaudited_name = config.student_arms
    if audited_name in overlap and unaudited_name in overlap:
        bootstrap = paired_patient_bootstrap(
            overlap[audited_name]["dice"]["per_patient_foreground"],
            overlap[unaudited_name]["dice"]["per_patient_foreground"],
            iterations=config.bootstrap_iterations,
            seed=config.seed,
        )
    else:
        bootstrap = {
            "available": False,
            "reason": f"both student arms ({audited_name}, {unaudited_name}) must be frozen",
            "count": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
        }
    row("students.dice_difference", bootstrap.get("mean_difference") if bootstrap.get("available") else None, unit="dice",
        checkpoint=f"{audited_name}-{unaudited_name}", population="patient",
        count=int(bootstrap.get("count", 0)), available=bool(bootstrap.get("available")),
        reason=None if bootstrap.get("available") else bootstrap.get("reason"),
        ci_low=bootstrap.get("ci_low"), ci_high=bootstrap.get("ci_high"),
        bootstrap_unit="patient")

    diagnostic: dict[str, Any] | None = None
    if config.oracle_diagnostic and final_name in volumes:
        flat_predictions, flat_references = _flatten_volume_mapping(
            volumes[final_name], references
        )
        diagnostic = oracle_cluster_matching_diagnostic(flat_predictions, flat_references)

    report = {
        "contract_version": VERSION,
        "available": True,
        "reason": None,
        "freeze_receipt": receipt,
        "config": config.to_dict(),
        "predictions": sorted(grouped),
        "frozen_predictions": sorted(grouped_all),
        "frozen_volume_count": frozen_volume_count,
        "evaluated_volume_count": evaluated_volume_count,
        "frozen_splits": frozen_splits,
        "overlap": overlap,
        "surfaces": surfaces,
        "edit_attribution": attribution,
        "evidence_quality": association,
        "coverage": coverage,
        "student_bootstrap": bootstrap,
        "oracle_cluster_matching_diagnostic": diagnostic,
        "missing_reference_patients": missing_reference,
        "rows": rows,
        "reference_label_map": {
            str(raw): int(class_id)
            for raw, class_id in (
                CANONICAL_REFERENCE_LABEL_MAP
                if config.reference_label_map is None
                else config.reference_label_map
            ).items()
        },
        "reference_label_map_explicit": config.reference_label_map is not None,
        "note": (
            "Reference metrics computed after a validated freeze. Draft-label quality and "
            "student quality are reported as separate predictions; no number here fed back "
            "into training, selection or checkpointing."
        ),
    }
    _write_reports(output_path, report, rows, config)
    return report


def _write_reports(
    output_dir: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    config: ReferenceConfig,
) -> None:
    """Emit JSON, CSV and Markdown side by side so no format is the only record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "reference_metrics.json", dict(report))
    write_csv(output_dir / "reference_metrics.csv", rows)
    write_markdown(
        output_dir / "reference_metrics.md",
        f"Isolated reference evaluation - {config.dataset} ({config.split})",
        rows,
        notes=[
            "Computed only after the prediction freeze was validated by file hash.",
            "Both-empty classes are excluded and counted; one-empty classes count as errors.",
            "Unavailable metrics carry a reason and a null value; they are never zero.",
            "No number in this report influenced training, selection or checkpointing.",
        ],
    )


__all__ = [
    "ReferenceCase",
    "ReferenceConfig",
    "ReferenceConfigError",
    "evaluate_reference",
    "index_frozen_predictions",
    "load_volume",
    "oracle_cluster_matching_diagnostic",
]
