"""Patient-level physiological confidence weights for Q-3D code distillation.

This module only consumes teacher probabilities and public native geometry.
It never reads labels or ED/ES annotations from an image-only patient.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..candidate_a3d.geometry import VolumeGeometry3D, canonical_geometry


_CACHE_SCHEMA = "self_audit.candidate_q3d.physiology_cache.v1"
_COHORT_WEIGHTING_SCHEMA = "self_audit.candidate_q3d.cohort_weighting.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_RV, _CLASS_MYO, _CLASS_LV = 1, 2, 3


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha256(value: str | None, name: str) -> str | None:
    return None if value is None else _require_sha256(value, name)


def file_sha256(path: str | Path) -> str:
    """Hash a checkpoint or codebook without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _physio_options(config: Mapping[str, Any] | None) -> dict[str, Any]:
    root: Mapping[str, Any] = {}
    if config is None:
        section: Mapping[str, Any] = {}
    elif not isinstance(config, Mapping):
        raise TypeError("Q-3D physiology config must be a nested mapping")
    else:
        root = config
        sections = [name for name in ("physio", "physiology") if name in config]
        if len(sections) > 1:
            raise ValueError("config cannot define both physio and physiology sections")
        raw = config.get(sections[0], {}) if sections else config
        if not isinstance(raw, Mapping):
            raise ValueError("config.physio must be a mapping")
        section = raw

    tau_sv = float(section.get("tau_sv", section.get("tau_stroke_volume", 0.15)))
    tau_m = float(section.get("tau_m", section.get("tau_myocardium", 0.10)))
    if not math.isfinite(tau_sv) or tau_sv <= 0 or not math.isfinite(tau_m) or tau_m <= 0:
        raise ValueError("physiology tau_sv and tau_m must be finite and positive")
    stroke_balance = section.get("stroke_balance", True)
    has_rv = section.get("has_rv")
    if has_rv is None:
        dataset = root.get("dataset", {})
        class_mapping = dataset.get("class_mapping") if isinstance(dataset, Mapping) else None
        if isinstance(class_mapping, Mapping):
            has_rv = "RV" in {str(value).upper() for value in class_mapping.values()}
            has_rv = has_rv or "RV" in {str(key).upper() for key in class_mapping}
        elif isinstance(class_mapping, (list, tuple)):
            has_rv = "RV" in {str(value).upper() for value in class_mapping}
        else:
            has_rv = True
    if type(stroke_balance) is not bool or type(has_rv) is not bool:
        raise ValueError("physiology stroke_balance and has_rv must be booleans")
    if stroke_balance and not has_rv:
        raise ValueError("physiology.stroke_balance requires an RV class")
    minimum_lv_ed_ml = float(section.get("minimum_lv_ed_ml", 10.0))
    ef_range_percent = section.get("ef_range_percent", [5.0, 90.0])
    certainty_threshold = float(section.get("certainty_threshold", 0.7))
    if not math.isfinite(minimum_lv_ed_ml) or minimum_lv_ed_ml < 0:
        raise ValueError("physiology.minimum_lv_ed_ml must be finite and nonnegative")
    if (not isinstance(ef_range_percent, (tuple, list)) or len(ef_range_percent) != 2
            or not all(math.isfinite(float(value)) for value in ef_range_percent)
            or not 0 <= float(ef_range_percent[0]) < float(ef_range_percent[1]) <= 100):
        raise ValueError("physiology.ef_range_percent must be an ordered range within [0,100]")
    if not math.isfinite(certainty_threshold) or not 0 <= certainty_threshold <= 1:
        raise ValueError("physiology.certainty_threshold must be in [0,1]")
    return {"tau_sv": tau_sv, "tau_m": tau_m,
            "stroke_balance": stroke_balance, "has_rv": has_rv,
            "minimum_lv_ed_ml": minimum_lv_ed_ml,
            "ef_range_percent": [float(value) for value in ef_range_percent],
            "certainty_threshold": certainty_threshold}


def _ratio(a: float, b: float) -> float | None:
    denominator = (a + b) / 2.0
    if denominator == 0.0 or not math.isfinite(denominator):
        return 0.0 if a == b == 0.0 else None
    value = abs(a - b) / denominator
    return float(value) if math.isfinite(value) else None


def patient_physiology_from_probabilities(
    probabilities: Tensor | np.ndarray,
    geometry: Mapping[str, Any],
    *,
    patient_id: str,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute ED/ES, ventricular residuals and ``w_pat`` from a cine.

    ``probabilities`` must be ``[T,4,D,H,W]`` in BG/RV/MYO/LV order.  ED and
    ES are selected only from predicted LV volumes; no metadata is accepted.
    """

    if protocol not in {"A", "B"}:
        raise ValueError("Q-3D protocol must be A or B")
    teacher_sha256 = _require_sha256(teacher_sha256, "teacher_sha256")
    codebook_sha256 = _optional_sha256(codebook_sha256, "codebook_sha256")
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("patient_id must be a non-empty TRAIN identifier")
    runtime_geometry = VolumeGeometry3D.from_mapping(canonical_geometry(geometry))
    options = _physio_options(config)

    if isinstance(probabilities, Tensor):
        probs = probabilities.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        probs = np.asarray(probabilities, dtype=np.float64)
    expected = (probs.shape[0], 4, *runtime_geometry.shape) if probs.ndim == 5 else None
    if (expected is None or probs.shape != expected or probs.shape[0] < 1
            or not np.isfinite(probs).all() or (probs < 0).any()):
        raise ValueError("cine probabilities must be finite nonnegative [T,4,D,H,W]")
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("teacher probabilities must sum to one across the four classes")

    dz, dy, dx = runtime_geometry.spacing_zyx
    voxel_ml = float(dz * dy * dx / 1000.0)
    volumes = probs.sum(axis=(2, 3, 4)) * voxel_ml
    lv = volumes[:, _CLASS_LV]
    ed = int(np.argmax(lv))
    es = int(np.argmin(lv))
    ed_lv, es_lv = float(lv[ed]), float(lv[es])
    lv_sv = ed_lv - es_lv
    ef_lv = lv_sv / ed_lv if ed_lv > 0 else None

    reasons: list[str] = []
    if lv_sv <= 0.0:
        reasons.append("lv_stroke_volume_nonpositive")
    if ed_lv < options["minimum_lv_ed_ml"]:
        reasons.append("lv_ed_volume_below_10_ml")
    ef_range = options["ef_range_percent"]
    ef_percent = 100.0 * ef_lv if ef_lv is not None else None
    if ef_percent is None or not ef_range[0] <= ef_percent <= ef_range[1]:
        reasons.append("lv_ef_outside_5_to_90_percent")

    rv_sv: float | None = None
    r_sv: float | None = None
    if options["has_rv"]:
        rv = volumes[:, _CLASS_RV]
        rv_sv = float(rv[ed] - rv[es])
        r_sv = _ratio(lv_sv, rv_sv)
        if options["stroke_balance"] and r_sv is None:
            reasons.append("stroke_volume_residual_undefined")

    myo = volumes[:, _CLASS_MYO]
    r_m = _ratio(float(myo[ed]), float(myo[es]))
    if r_m is None:
        reasons.append("myocardial_volume_residual_undefined")

    w_sv = 1.0
    if options["stroke_balance"] and r_sv is not None:
        w_sv = math.exp(-0.5 * (r_sv / options["tau_sv"]) ** 2)
    w_m = (math.exp(-0.5 * (r_m / options["tau_m"]) ** 2)
           if r_m is not None else 0.0)
    w_pat = 0.0 if reasons else float(w_sv * w_m)
    if not math.isfinite(w_pat):
        w_pat = 0.0
        reasons.append("nonfinite_physiology_weight")

    return {
        "id": patient_id,
        "protocol": protocol,
        "teacher_sha256": teacher_sha256,
        "codebook_sha256": codebook_sha256,
        "frame_indices": list(range(int(probs.shape[0]))),
        "volumes_ml_by_frame": {
            "rv": [float(value) for value in volumes[:, _CLASS_RV]],
            "myo": [float(value) for value in myo],
            "lv": [float(value) for value in lv],
        },
        "ed_index": ed,
        "es_index": es,
        "lv_stroke_volume_ml": float(lv_sv),
        "rv_stroke_volume_ml": rv_sv,
        "ef_lv": float(ef_lv) if ef_lv is not None else None,
        "r_sv": float(r_sv) if r_sv is not None else None,
        "r_m": float(r_m) if r_m is not None else None,
        "w_pat": w_pat,
        "stroke_balance": options["stroke_balance"],
        "tau_sv": options["tau_sv"],
        "tau_m": options["tau_m"],
        "guard_reasons": reasons,
    }


def physiology_cell_weights(
    region_logits: Tensor,
    patient_weights: Tensor | list[float] | tuple[float, ...],
    *,
    cell_shape_zyx: tuple[int, int, int] = (1, 16, 16),
    threshold: float = 0.7,
) -> tuple[Tensor, dict[str, float]]:
    """Compute w_cell × w_pat on the student S4 grid from native logits.

    Each cell receives the average of the teacher's maximum class probability
    over its native voxels; cells below the certainty threshold receive zero.
    Partial edge cells average only the voxels present in the native volume.
    """

    if (not isinstance(region_logits, Tensor) or region_logits.ndim != 5
            or region_logits.shape[1] != 4 or not torch.isfinite(region_logits).all()):
        raise ValueError("region_logits must be finite [B,4,D,H,W]")
    if (not isinstance(cell_shape_zyx, (tuple, list)) or len(cell_shape_zyx) != 3
            or any(type(value) is not int or value < 1 for value in cell_shape_zyx)):
        raise ValueError("cell_shape_zyx must contain positive integer strides")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("cell certainty threshold must be in [0,1]")
    patient = torch.as_tensor(patient_weights, dtype=torch.float32,
                              device=region_logits.device).reshape(-1).detach()
    if patient.shape[0] != region_logits.shape[0] or not torch.isfinite(patient).all() \
            or bool((patient < 0).any()) or bool((patient > 1).any()):
        raise ValueError("patient_weights must contain one finite [0,1] value per batch item")
    confidence = torch.softmax(region_logits.detach().float(), dim=1).amax(dim=1, keepdim=True)
    pooled = F.avg_pool3d(confidence, kernel_size=tuple(cell_shape_zyx),
                          stride=tuple(cell_shape_zyx), ceil_mode=True,
                          count_include_pad=False)
    cell = torch.where(pooled >= float(threshold), pooled, torch.zeros_like(pooled))
    weights = cell * patient.view(-1, 1, 1, 1, 1)
    diagnostics = {
        "mean_cell_confidence": float(pooled.mean().detach().cpu()),
        "fraction_cells_above_threshold": float((pooled >= float(threshold)).float().mean().detach().cpu()),
        "mean_w_cell": float(cell.mean().detach().cpu()),
        "mean_w": float(weights.mean().detach().cpu()),
    }
    return weights, diagnostics


def physiology_weight_for_patient(
    teacher: torch.nn.Module,
    data: Any,
    patient_id: str,
    *,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run a frozen teacher on every image-only cine frame for one U patient."""

    unlabeled_patients = tuple(getattr(data, "unlabeled_patients", getattr(data, "uids", ())))
    patient_for_key = getattr(data, "unlabeled_patient_for_key", None)
    if patient_for_key is None:
        patient_for_key = lambda key: data.uids[int(key[0])]
    if patient_id not in set(unlabeled_patients):
        raise ValueError("physiology cache may only be computed for image-only TRAIN U patients")
    keys = [key for key in data.unlabeled_frame_keys()
            if patient_for_key(key) == patient_id]
    if not keys:
        raise ValueError(f"image-only TRAIN patient {patient_id!r} has no cine frames")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise ValueError("physiology snapshot teacher must be frozen before cache computation")
    try:
        was_training = bool(teacher.training)
        teacher.eval()
    except AttributeError as exc:
        raise TypeError("teacher must be an eval-capable torch module") from exc

    probs_by_frame: list[Tensor] = []
    common_geometry: dict[str, Any] | None = None
    try:
        with torch.no_grad():
            for key in keys:
                image, geometry, metadata = data.load_unlabeled_frame(key)
                if set(metadata) != {"patient", "frame"} or metadata["patient"] != patient_id:
                    raise ValueError("image-only frame metadata must not expose ED/ES annotations")
                canonical = canonical_geometry(geometry)
                if common_geometry is None:
                    common_geometry = canonical
                elif canonical != common_geometry:
                    raise ValueError("native geometry must remain fixed across one cine")
                output = teacher(image.to(device), geometry=canonical)
                if not isinstance(output, Mapping) or "region_logits" not in output:
                    raise ValueError("teacher extraction must return region_logits")
                logits = output["region_logits"]
                expected_shape = (1, 4, *VolumeGeometry3D.from_mapping(canonical).shape)
                if (not isinstance(logits, Tensor) or tuple(logits.shape) != expected_shape
                        or not torch.isfinite(logits).all()):
                    raise ValueError(f"teacher region_logits must be finite with shape {expected_shape}")
                probs_by_frame.append(torch.softmax(logits.float(), dim=1)[0].cpu())
    finally:
        teacher.train(was_training)

    if common_geometry is None:
        raise ValueError("no native geometry was loaded for image-only patient")
    probabilities = torch.stack(probs_by_frame, dim=0)
    return patient_physiology_from_probabilities(
        probabilities, common_geometry, patient_id=patient_id,
        teacher_sha256=teacher_sha256, codebook_sha256=codebook_sha256,
        protocol=protocol, config=config,
    )


def compute_physiology_cache_rows(
    teacher: torch.nn.Module,
    data: Any,
    *,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    config: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, Any]]:
    """Compute one physiology row per image-only TRAIN patient, and no others."""

    patients = tuple(getattr(data, "unlabeled_patients", getattr(data, "uids", ())))
    if not patients:
        raise ValueError("Q-3D physiology cache requires an image-only TRAIN U cohort")
    return [
        physiology_weight_for_patient(
            teacher, data, patient, teacher_sha256=teacher_sha256,
            codebook_sha256=codebook_sha256, protocol=protocol,
            config=config, device=device,
        )
        for patient in patients
    ]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _source_patient_ids(source_identity: Mapping[str, Any], protocol: str) -> set[str]:
    if source_identity.get("protocol") != protocol:
        raise ValueError("source identity protocol differs from physiology cache protocol")
    records = source_identity.get("unlabeled_patients")
    if not isinstance(records, list) or not records:
        raise ValueError("source identity must enumerate the image-only TRAIN U cohort")
    ids = [record.get("id") for record in records if isinstance(record, Mapping)]
    if (len(ids) != len(records) or any(not isinstance(pid, str) or not pid for pid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("source identity has malformed or duplicate U patient records")
    return set(ids)


def _validated_cache_rows(
    patients: Any,
) -> list[dict[str, Any]]:
    """Copy and validate the rows used by cache statistics and serialization.

    ``w_pat`` is the only row value used for cohort weighting.  It is still
    validated here (rather than only at the call sites) so the statistics in
    ``cohort_weighting`` can never be computed from a different set of rows
    than the rows written to or read from the cache.  Boolean values are
    rejected even though Python considers them integers.
    """

    if not isinstance(patients, list):
        raise ValueError("physiology cache patients must be a list")
    if not patients:
        raise ValueError("physiology cache requires at least one patient row")
    rows: list[dict[str, Any]] = []
    ids: list[str] = []
    for index, raw_row in enumerate(patients):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"physiology cache patient row {index} is not a mapping")
        row = dict(raw_row)
        patient_id = row.get("id")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("physiology cache patient IDs must be nonempty strings")
        weight = row.get("w_pat")
        if (isinstance(weight, bool) or not isinstance(weight, Real)
                or not math.isfinite(float(weight)) or not 0.0 <= float(weight) <= 1.0):
            raise ValueError(
                f"physiology cache patient row {patient_id!r} w_pat must be finite and in [0,1]"
            )
        # Normalize numeric scalar subclasses before json.dump.  This also
        # makes the exact rows used in the summary the exact rows persisted.
        row["w_pat"] = float(weight)
        ids.append(patient_id)
        rows.append(row)
    if len(set(ids)) != len(ids):
        raise ValueError("physiology cache patient rows are malformed or duplicated")
    return rows


def summarize_physiology_cohort(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Return raw U-cohort ``w_pat`` statistics and the frozen gate result.

    This helper deliberately reports the source weights, not normalized
    ``omega`` values.  The L-arm implementation of :func:`cohort_omega` is
    used only as a gate-free, numerically stable source-statistics calculator;
    callers choose an arm and construct normalized omega exactly once when a
    cache is consumed.  A zero-mean or low-ESS cohort is therefore valid audit
    output with ``gate_passed=False`` rather than a serialization error.
    """

    if not isinstance(rows, list) or not rows:
        raise ValueError("physiology cohort summary requires nonempty patient rows")
    validated = _validated_cache_rows(rows)
    patient_ids = sorted(row["id"] for row in validated)
    weights = {row["id"]: row["w_pat"] for row in validated}

    # Import lazily to keep this data/cache module usable by standalone tools
    # and to avoid imposing a losses import during module initialization.
    from .losses import cohort_omega

    raw = cohort_omega(weights, "L")
    mean_w_pat = float(raw["mean_w_pat"])
    ess = float(raw["ess"])
    ess_fraction = float(raw["ess_fraction"])
    return {
        "schema": _COHORT_WEIGHTING_SCHEMA,
        "patient_ids": patient_ids,
        "count": len(patient_ids),
        "mean_w_pat": mean_w_pat,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "gate_passed": bool(mean_w_pat > 0.0 and ess_fraction >= 0.25),
    }


def _validate_cohort_weighting(
    metadata: Any,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate cohort metadata against the exact cached patient rows."""

    if not isinstance(metadata, Mapping):
        raise ValueError(
            "physiology cache cohort_weighting metadata is missing; regenerate the cache"
        )
    required = {
        "schema", "patient_ids", "count", "mean_w_pat", "ess",
        "ess_fraction", "gate_passed",
    }
    if set(metadata) != required:
        raise ValueError(
            "physiology cache cohort_weighting metadata is stale or malformed; "
            "regenerate the cache"
        )
    if metadata.get("schema") != _COHORT_WEIGHTING_SCHEMA:
        raise ValueError(
            "physiology cache cohort_weighting schema mismatch; regenerate the cache"
        )
    patient_ids = metadata.get("patient_ids")
    expected = sorted(row["id"] for row in _validated_cache_rows(rows))
    if (not isinstance(patient_ids, list)
            or any(not isinstance(patient_id, str) for patient_id in patient_ids)
            or patient_ids != expected):
        raise ValueError(
            "physiology cache cohort_weighting patient_ids differ from cached rows; "
            "regenerate the cache"
        )
    count = metadata.get("count")
    if type(count) is not int or count != len(expected):
        raise ValueError(
            "physiology cache cohort_weighting count differs from cached rows; "
            "regenerate the cache"
        )
    for field in ("mean_w_pat", "ess", "ess_fraction"):
        value = metadata.get(field)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value))):
            raise ValueError(
                f"physiology cache cohort_weighting {field} is not finite; regenerate the cache"
            )
    gate_passed = metadata.get("gate_passed")
    if type(gate_passed) is not bool:
        raise ValueError(
            "physiology cache cohort_weighting gate_passed is not boolean; regenerate the cache"
        )

    expected_metadata = summarize_physiology_cohort(rows)
    for field in ("mean_w_pat", "ess", "ess_fraction"):
        # The expected values are emitted by the same canonical JSON encoder
        # used by save.  Exact equality catches a resealed forged metadata
        # value while avoiding an arbitrary tolerance at the gate boundary.
        if float(metadata[field]) != expected_metadata[field]:
            raise ValueError(
                f"physiology cache cohort_weighting {field} does not match cached rows; "
                "regenerate the cache"
            )
    if bool(gate_passed) != expected_metadata["gate_passed"]:
        raise ValueError(
            "physiology cache cohort_weighting gate_passed does not match cached rows; "
            "regenerate the cache"
        )
    return dict(metadata)


def save_physiology_cache(
    path: str | Path,
    patients: list[Mapping[str, Any]],
    *,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    refreshed_epoch: int = 0,
    config: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any],
) -> str:
    """Write an atomic SHA-sealed cache bound to teacher, codebook and protocol."""

    teacher_sha256 = _require_sha256(teacher_sha256, "teacher_sha256")
    codebook_sha256 = _optional_sha256(codebook_sha256, "codebook_sha256")
    if protocol not in {"A", "B"}:
        raise ValueError("Q-3D cache protocol must be A or B")
    if type(refreshed_epoch) is not int or refreshed_epoch < 0:
        raise ValueError("refreshed_epoch must be a nonnegative integer")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise ValueError("source_identity must bind the cache to a nonempty TRAIN manifest/cohort")
    expected_ids = _source_patient_ids(source_identity, protocol)
    options = _physio_options(config)
    rows = _validated_cache_rows(patients)
    ids = [row["id"] for row in rows]
    if set(ids) != expected_ids:
        raise ValueError("physiology rows must cover exactly the source image-only TRAIN U cohort")
    for row in rows:
        if (row.get("protocol") != protocol or row.get("teacher_sha256") != teacher_sha256
                or row.get("codebook_sha256") != codebook_sha256):
            raise ValueError("physiology row identity differs from cache protocol/checkpoints")
    rows = sorted(rows, key=lambda row: row["id"])
    cohort_weighting = summarize_physiology_cohort(rows)
    payload: dict[str, Any] = {
        "schema": _CACHE_SCHEMA,
        "protocol": protocol,
        "teacher_sha256": teacher_sha256,
        "codebook_sha256": codebook_sha256,
        "physio_config": options,
        "physio_config_sha256": hashlib.sha256(_canonical_json(options)).hexdigest(),
        "source_identity": dict(source_identity),
        "source_identity_sha256": hashlib.sha256(_canonical_json(source_identity)).hexdigest(),
        "refreshed_epoch": refreshed_epoch,
        "cohort_weighting": cohort_weighting,
        "patients": rows,
    }
    checksum = hashlib.sha256(_canonical_json(payload)).hexdigest()
    document = {"payload": payload, "sha256": checksum}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", delete=False) as stream:
            temp_name = stream.name
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    return checksum


def load_physiology_cache(
    path: str | Path,
    *,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    config: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify bytes, schema, and protocol identity before exposing cached rows."""

    teacher_sha256 = _require_sha256(teacher_sha256, "teacher_sha256")
    codebook_sha256 = _optional_sha256(codebook_sha256, "codebook_sha256")
    if protocol not in {"A", "B"}:
        raise ValueError("Q-3D cache protocol must be A or B")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise ValueError("source_identity must bind the cache to a nonempty TRAIN manifest/cohort")
    expected_ids = _source_patient_ids(source_identity, protocol)
    options = _physio_options(config)
    config_sha = hashlib.sha256(_canonical_json(options)).hexdigest()
    source_sha = hashlib.sha256(_canonical_json(source_identity)).hexdigest()
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"payload", "sha256"}:
        raise ValueError("physiology cache envelope is malformed")
    payload = document["payload"]
    if not isinstance(payload, dict) or payload.get("schema") != _CACHE_SCHEMA:
        raise ValueError("physiology cache schema mismatch")
    actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if document["sha256"] != actual:
        raise ValueError("physiology cache SHA-256 mismatch")
    if (payload.get("protocol") != protocol or payload.get("teacher_sha256") != teacher_sha256
            or payload.get("codebook_sha256") != codebook_sha256):
        raise ValueError("physiology cache protocol, teacher or codebook identity mismatch")
    try:
        payload_config_sha = hashlib.sha256(
            _canonical_json(payload["physio_config"])
        ).hexdigest()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("physiology cache config binding is malformed; regenerate the cache") from exc
    if payload.get("physio_config_sha256") != payload_config_sha:
        raise ValueError("physiology cache config binding hash mismatch; regenerate the cache")
    if payload_config_sha != config_sha:
        raise ValueError("physiology cache tau/gate settings differ from the requested config")
    try:
        payload_source_sha = hashlib.sha256(
            _canonical_json(payload["source_identity"])
        ).hexdigest()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("physiology cache source binding is malformed; regenerate the cache") from exc
    if payload.get("source_identity_sha256") != payload_source_sha:
        raise ValueError("physiology cache source binding hash mismatch; regenerate the cache")
    if payload_source_sha != source_sha:
        raise ValueError("physiology cache source manifest/cohort identity mismatch")
    patients = payload.get("patients")
    rows = _validated_cache_rows(patients)
    ids = [row["id"] for row in rows]
    if set(ids) != expected_ids:
        raise ValueError("physiology cache patients differ from the image-only TRAIN U cohort")
    for row in rows:
        if (row.get("protocol") != protocol or row.get("teacher_sha256") != teacher_sha256
                or row.get("codebook_sha256") != codebook_sha256):
            raise ValueError("physiology patient row checkpoint or protocol identity mismatch")
    _validate_cohort_weighting(payload.get("cohort_weighting"), rows)
    return document


def physiology_cache_needs_refresh(
    cache: Mapping[str, Any] | None,
    *,
    teacher_sha256: str,
    codebook_sha256: str | None = None,
    protocol: str,
    epoch: int,
    online_teacher: bool = False,
    refresh_interval: int = 10,
    config: Mapping[str, Any] | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether stage P must be recomputed for the current snapshot."""

    teacher_sha256 = _require_sha256(teacher_sha256, "teacher_sha256")
    codebook_sha256 = _optional_sha256(codebook_sha256, "codebook_sha256")
    if protocol not in {"A", "B"} or type(epoch) is not int or epoch < 0:
        raise ValueError("protocol must be A/B and epoch must be a nonnegative integer")
    if type(refresh_interval) is not int or refresh_interval < 1:
        raise ValueError("refresh_interval must be a positive integer")
    if not isinstance(cache, Mapping):
        return True
    payload = cache.get("payload")
    if not isinstance(payload, Mapping):
        return True
    try:
        actual_sha = hashlib.sha256(_canonical_json(payload)).hexdigest()
    except (TypeError, ValueError):
        return True
    if cache.get("sha256") != actual_sha:
        return True
    if (payload.get("protocol") != protocol or payload.get("teacher_sha256") != teacher_sha256
            or payload.get("codebook_sha256") != codebook_sha256):
        return True
    if not isinstance(source_identity, Mapping) or not source_identity:
        return True
    try:
        expected_ids = _source_patient_ids(source_identity, protocol)
        cached_ids = {row.get("id") for row in payload.get("patients", []) if isinstance(row, Mapping)}
    except (TypeError, ValueError):
        return True
    if cached_ids != expected_ids:
        return True
    options = _physio_options(config)
    try:
        payload_config_sha = hashlib.sha256(
            _canonical_json(payload["physio_config"])
        ).hexdigest()
        payload_source_sha = hashlib.sha256(
            _canonical_json(payload["source_identity"])
        ).hexdigest()
    except (KeyError, TypeError, ValueError):
        return True
    if payload.get("physio_config_sha256") != payload_config_sha:
        return True
    if payload_config_sha != hashlib.sha256(_canonical_json(options)).hexdigest():
        return True
    if payload.get("source_identity_sha256") != payload_source_sha:
        return True
    if payload_source_sha != hashlib.sha256(_canonical_json(source_identity)).hexdigest():
        return True
    try:
        rows = _validated_cache_rows(payload.get("patients"))
        if {row["id"] for row in rows} != expected_ids:
            return True
        _validate_cohort_weighting(payload.get("cohort_weighting"), rows)
    except (TypeError, ValueError):
        return True
    if not online_teacher:
        return False
    last = payload.get("refreshed_epoch")
    return type(last) is not int or epoch - last >= refresh_interval


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    if denom == 0.0:
        return None
    return float(np.dot(ra, rb) / denom)


def physiology_dev_diagnostic(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Report-only DEV association of cached weights with teacher RV metrics.

    The function has no cache-writing or parameter-selection behavior.  Input
    rows must already be paired by patient after the frozen teacher evaluation.
    """

    if not isinstance(rows, list) or not rows:
        raise ValueError("DEV physiology diagnostic requires paired patient rows")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(pid, str) or not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("DEV physiology diagnostic patient IDs must be unique")
    names = ("w_pat", "rv_dice", "rv_hd95")
    columns: dict[str, np.ndarray] = {}
    for name in names:
        values = np.asarray([row.get(name) for row in rows], dtype=np.float64)
        if values.shape != (len(rows),) or not np.isfinite(values).all():
            raise ValueError(f"DEV physiology diagnostic requires finite paired {name} values")
        columns[name] = values
    return {
        "status": "REPORT_ONLY",
        "patient_count": len(rows),
        "spearman_w_pat_rv_dice": _spearman(columns["w_pat"], columns["rv_dice"]),
        "spearman_w_pat_rv_hd95": _spearman(columns["w_pat"], columns["rv_hd95"]),
        "used_for_tau_selection": False,
    }


__all__ = [
    "file_sha256", "patient_physiology_from_probabilities",
    "physiology_cell_weights", "physiology_weight_for_patient",
    "compute_physiology_cache_rows", "save_physiology_cache",
    "load_physiology_cache", "physiology_cache_needs_refresh",
    "summarize_physiology_cohort",
    "physiology_dev_diagnostic",
]
