"""TRAIN-only, hash-bound fitting of the compact-to-teacher feature projector.

The fitted object is a 1x1 affine map.  Compact bottleneck features are first
bilinearly aligned to the teacher's stride-4 pre-refine[0] grid, then projected.
The alignment fit uses image-derived features only; this module has no label or
mask-payload input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


ALIGNMENT_SCHEMA = "self_audit.gradient_bridge_alignment.v1"
ALIGNMENT_FIT_LOSS = "patient_balanced_pre_refine0_feature_mse"
ALIGNMENT_FEATURE_TAP = "CompactStudent.encoder[-1]"
ALIGNMENT_TARGET_TAP = "NativeStudent.pre_refine[0] input"
ALIGNMENT_MODE = "bilinear_align_corners_false_then_1x1_affine"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _tensor_sha256(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for value in tensors:
        tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _matrix_sha256(value: Tensor) -> str:
    matrix = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(matrix.shape)).encode("ascii"))
    digest.update(matrix.numpy().tobytes())
    return digest.hexdigest()


def _normalise_identities(identities: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(identities, Mapping):
        raise ValueError("bridge identities must be a mapping")
    aliases = {
        "student_sha256": ("student_sha256", "student_payload_sha256", "student_checkpoint_sha256"),
        "teacher_sha256": ("teacher_sha256", "teacher_payload_sha256", "teacher_checkpoint_sha256"),
        "train_manifest_sha256": ("train_manifest_sha256", "train_sha256"),
    }
    result: dict[str, str] = {}
    for canonical, names in aliases.items():
        supplied = [identities[name] for name in names if name in identities]
        if supplied and any(value != supplied[0] for value in supplied[1:]):
            raise ValueError(f"conflicting aliases for bridge identity {canonical}")
        value = supplied[0] if supplied else None
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"bridge identity {canonical} must be a lowercase SHA-256")
        result[canonical] = value
    for name, value in identities.items():
        if name in {alias for names in aliases.values() for alias in names}:
            continue
        if not isinstance(name, str) or not name or not isinstance(value, (str, int, float, bool)):
            raise ValueError("additional bridge identities must be named JSON scalars")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite bridge identity: {name}")
        result[name] = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
    return dict(sorted(result.items()))


def _normalise_expected_identities(identities: Mapping[str, Any]) -> dict[str, str]:
    """Normalize any nonempty subset of identity keys for strict comparison."""
    if not isinstance(identities, Mapping) or not identities:
        raise ValueError("expected bridge identities must be a nonempty mapping")
    aliases = {
        "student_sha256": ("student_sha256", "student_payload_sha256", "student_checkpoint_sha256"),
        "teacher_sha256": ("teacher_sha256", "teacher_payload_sha256", "teacher_checkpoint_sha256"),
        "train_manifest_sha256": ("train_manifest_sha256", "train_sha256"),
    }
    alias_names = {alias for names in aliases.values() for alias in names}
    result: dict[str, str] = {}
    for canonical, names in aliases.items():
        supplied = [identities[name] for name in names if name in identities]
        if supplied:
            if any(value != supplied[0] for value in supplied[1:]):
                raise ValueError(f"conflicting aliases for bridge identity {canonical}")
            if not isinstance(supplied[0], str) or _SHA256.fullmatch(supplied[0]) is None:
                raise ValueError(f"expected bridge identity {canonical} must be a lowercase SHA-256")
            result[canonical] = supplied[0]
    for name, value in identities.items():
        if name in alias_names:
            continue
        if not isinstance(name, str) or not name or not isinstance(value, (str, int, float, bool)):
            raise ValueError("expected bridge identities must be named JSON scalars")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite expected bridge identity: {name}")
        result[name] = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class BridgeAlignmentConfig:
    """Locked, deterministic fit budget and objective for P."""

    max_source_ids: int
    max_train_patients: int
    max_pixels: int
    ridge: float = 1e-4
    fit_loss: str = ALIGNMENT_FIT_LOSS
    alignment_mode: str = ALIGNMENT_MODE
    seed: int = 0

    def validate(self) -> None:
        for name in ("max_source_ids", "max_train_patients", "max_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_train_patients < 2:
            raise ValueError("bridge fit requires a budget of at least two TRAIN patients")
        if self.max_source_ids < self.max_train_patients:
            raise ValueError("max_source_ids must cover at least one source per TRAIN patient")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.ridge) not in (float, int) or not math.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be a positive finite scalar")
        if self.fit_loss != ALIGNMENT_FIT_LOSS:
            raise ValueError("bridge fit loss is locked to patient-balanced feature MSE")
        if self.alignment_mode != ALIGNMENT_MODE:
            raise ValueError("bridge spatial alignment is locked to bilinear align_corners=False then 1x1")


@dataclass(frozen=True)
class BridgeFitSample:
    """One image-only TRAIN activation pair.  No ground-truth field is accepted."""

    student_h: Tensor
    teacher_pre_refine0: Tensor
    valid_mask: Tensor
    patient_id: str
    source_id: str
    split: str


@dataclass(frozen=True)
class BridgeAlignmentArtifact:
    """Verified projector weights plus the identities and fit receipt."""

    metadata: Mapping[str, Any]
    projector_weight: Tensor
    projector_bias: Tensor
    artifact_sha256: str

    def verify(self, expected_identities: Mapping[str, Any] | None = None) -> None:
        if not isinstance(self.metadata, Mapping):
            raise ValueError("bridge alignment metadata must be a mapping")
        weight = self.projector_weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
        bias = self.projector_bias.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if weight.ndim != 4 or weight.shape[-2:] != (1, 1) or bias.shape != (weight.shape[0],):
            raise ValueError("bridge alignment projector must be a 1x1 affine convolution")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("bridge alignment contains non-finite projector weights")
        payload = _artifact_payload(self.metadata, weight, bias)
        observed = hashlib.sha256(_canonical(payload)).hexdigest()
        if observed != self.artifact_sha256:
            raise ValueError("bridge alignment artifact hash mismatch")
        if self.metadata.get("schema") != ALIGNMENT_SCHEMA:
            raise ValueError("unsupported bridge alignment schema")
        if (self.metadata.get("fit_split") != "TRAIN"
                or self.metadata.get("feature_tap") != ALIGNMENT_FEATURE_TAP
                or self.metadata.get("target_tap") != ALIGNMENT_TARGET_TAP
                or self.metadata.get("loss") != ALIGNMENT_FIT_LOSS
                or self.metadata.get("alignment") != ALIGNMENT_MODE
                or self.metadata.get("raw_detail_conditioning") is not False):
            raise ValueError("bridge alignment does not match the locked TRAIN-only fit graph")
        if weight.shape[1] != self.metadata.get("student_channels") or weight.shape[0] != self.metadata.get("teacher_channels"):
            raise ValueError("bridge projector shape does not match recorded feature channels")
        if self.metadata.get("projector_sha256") != _tensor_sha256(weight, bias):
            raise ValueError("bridge projector tensor hash mismatch")
        identities = self.metadata.get("identities")
        if not isinstance(identities, Mapping):
            raise ValueError("bridge alignment identities are missing")
        recorded = _normalise_identities(identities)
        if expected_identities is not None:
            expected = _normalise_expected_identities(expected_identities)
            for name, value in expected.items():
                if recorded.get(name) != value:
                    raise ValueError(f"bridge alignment identity mismatch: {name}")
        patients = self.metadata.get("train_patient_ids")
        sources = self.metadata.get("train_source_ids")
        source_order = self.metadata.get("train_source_order")
        observations = self.metadata.get("fit_observations")
        budget = self.metadata.get("budget")
        if (not isinstance(patients, list) or not isinstance(sources, list)
                or not isinstance(source_order, list) or not isinstance(observations, list)
                or len(observations) != len(source_order) or not isinstance(budget, Mapping)):
            raise ValueError("bridge alignment lacks unique TRAIN source identities and budget")
        if (any(not isinstance(value, str) or not value.strip() for value in patients + sources + source_order)
                or not all(isinstance(row, Mapping) for row in observations)):
            raise ValueError("bridge alignment source and patient identities must be nonempty strings")
        if (patients != sorted(set(patients)) or len(patients) < 2
                or sources != sorted(set(sources)) or len(sources) < 2
                or sorted(source_order) != sources or len(source_order) != len(set(source_order))):
            raise ValueError("bridge alignment source and patient identifiers must be unique and ordered")
        if type(budget.get("seed")) is not int or budget["seed"] < 0:
            raise ValueError("invalid bridge alignment fit seed")
        for used_name, maximum_name in (("used_source_ids", "max_source_ids"),
                                        ("used_train_patients", "max_train_patients"),
                                        ("used_pixels", "max_pixels")):
            used, maximum = budget.get(used_name), budget.get(maximum_name)
            if type(used) is not int or type(maximum) is not int or used <= 0 or maximum <= 0 or used > maximum:
                raise ValueError(f"invalid bridge alignment budget field: {used_name}")
        if budget["max_train_patients"] < 2 or budget["max_source_ids"] < 2:
            raise ValueError("bridge alignment budgets must permit at least two TRAIN sources and patients")
        observed_sources: list[str] = []
        observed_patients: set[str] = set()
        observed_pixels = 0
        for row in observations:
            source_id, patient_id = row.get("source_id"), row.get("patient_id")
            student_hw, teacher_hw = row.get("student_feature_hw"), row.get("teacher_feature_hw")
            valid_pixels, used_pixels = row.get("valid_pixels"), row.get("used_pixels")
            if (not isinstance(source_id, str) or not source_id.strip()
                    or not isinstance(patient_id, str) or not patient_id.strip()
                    or not isinstance(student_hw, list) or len(student_hw) != 2
                    or not isinstance(teacher_hw, list) or len(teacher_hw) != 2
                    or any(type(value) is not int or value <= 0 for value in student_hw + teacher_hw)
                    or type(valid_pixels) is not int or valid_pixels < 0
                    or valid_pixels > teacher_hw[0] * teacher_hw[1]
                    or type(used_pixels) is not int or used_pixels < 0 or used_pixels > valid_pixels):
                raise ValueError("bridge alignment fit observation has invalid source geometry or pixel counts")
            observed_sources.append(source_id)
            if used_pixels > 0:
                observed_patients.add(patient_id)
            observed_pixels += used_pixels
        if (observed_sources != source_order or sorted(observed_sources) != sources
                or sorted(observed_patients) != patients
                or budget["used_source_ids"] != len(observations)
                or budget["used_train_patients"] != len(observed_patients)
                or budget["used_pixels"] != observed_pixels
                or hashlib.sha256(_canonical(sources)).hexdigest() != self.metadata.get("train_source_ids_sha256")
                or hashlib.sha256(_canonical(source_order)).hexdigest() != self.metadata.get("train_source_order_sha256")
                or hashlib.sha256(_canonical(observations)).hexdigest() != self.metadata.get("fit_observations_sha256")):
            raise ValueError("bridge alignment source identities do not match the recorded fit budget")
        metrics = self.metadata.get("fit_metrics")
        required_metrics = {
            "patient_balanced_feature_mse", "ridge_objective", "residual_covariance_trace",
            "residual_covariance_min_eigenvalue", "residual_mean_squared_sum", "input_covariance_trace",
            "input_covariance_min_eigenvalue", "cross_covariance_frobenius", "target_covariance_trace", "fit_pixels",
        }
        if (not isinstance(metrics, Mapping) or not required_metrics.issubset(metrics)
                or any(not isinstance(metrics[name], (int, float)) or not math.isfinite(metrics[name])
                       for name in required_metrics)):
            raise ValueError("bridge alignment fit metrics must be finite")
        ridge = self.metadata.get("ridge")
        if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("bridge alignment ridge value is invalid")
        channels_out = weight.shape[0]
        mse = (float(metrics["residual_covariance_trace"]) + float(metrics["residual_mean_squared_sum"])) / channels_out
        weight_by_y = weight[:, :, 0, 0].double().T
        objective = mse + float(ridge) * float(weight_by_y.square().sum().item()) / channels_out
        if (metrics["patient_balanced_feature_mse"] < 0 or metrics["ridge_objective"] < 0
                or metrics["residual_covariance_trace"] < -1e-8
                or metrics["residual_covariance_min_eigenvalue"] < -1e-8
                or metrics["input_covariance_trace"] < -1e-8
                or metrics["input_covariance_min_eigenvalue"] < -1e-8
                or metrics["cross_covariance_frobenius"] < 0
                or metrics["target_covariance_trace"] < -1e-8
                or type(metrics["fit_pixels"]) is not int
                or metrics["fit_pixels"] != budget["used_pixels"]
                or not math.isclose(float(metrics["patient_balanced_feature_mse"]), mse, rel_tol=1e-6, abs_tol=1e-9)
                or not math.isclose(float(metrics["ridge_objective"]), objective, rel_tol=1e-6, abs_tol=1e-9)):
            raise ValueError("bridge alignment residual/covariance metrics are inconsistent with the fitted projector")
        covariance_hashes = self.metadata.get("covariance_sha256")
        required_covariances = {"input", "input_target", "target", "residual"}
        if (not isinstance(covariance_hashes, Mapping) or set(covariance_hashes) != required_covariances
                or any(not isinstance(value, str) or _SHA256.fullmatch(value) is None
                       for value in covariance_hashes.values())):
            raise ValueError("bridge alignment covariance receipt is incomplete")


def _artifact_payload(metadata: Mapping[str, Any], weight: Tensor, bias: Tensor) -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_SCHEMA,
        "metadata": dict(metadata),
        "projector_weight": weight.tolist(),
        "projector_bias": bias.tolist(),
    }


def _new_artifact(metadata: Mapping[str, Any], weight: Tensor, bias: Tensor) -> BridgeAlignmentArtifact:
    weight = weight.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    bias = bias.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    payload = _artifact_payload(metadata, weight, bias)
    artifact = BridgeAlignmentArtifact(
        metadata=dict(metadata),
        projector_weight=weight,
        projector_bias=bias,
        artifact_sha256=hashlib.sha256(_canonical(payload)).hexdigest(),
    )
    artifact.verify()
    return artifact


def _pixel_stats(sample: BridgeFitSample, channels_in: int, channels_out: int, limit: int) -> tuple[Tensor, ...] | None:
    if sample.split != "TRAIN":
        raise ValueError("bridge projector fitting accepts TRAIN samples only")
    if not isinstance(sample.patient_id, str) or not sample.patient_id.strip():
        raise ValueError("bridge fit patient_id is required")
    if not isinstance(sample.source_id, str) or not sample.source_id.strip():
        raise ValueError("bridge fit source_id is required")
    h, target, valid = sample.student_h, sample.teacher_pre_refine0, sample.valid_mask
    if not isinstance(h, Tensor) or h.ndim != 4 or h.shape[0] != 1 or h.shape[1] != channels_in:
        raise ValueError("student_h must be [1,C_in,H,W] with a fixed channel count")
    if not isinstance(target, Tensor) or target.ndim != 4 or target.shape[0] != 1 or target.shape[1] != channels_out:
        raise ValueError("teacher_pre_refine0 must be [1,C_out,H,W] with a fixed channel count")
    if not isinstance(valid, Tensor) or valid.dtype is not torch.bool or valid.shape != (1, *target.shape[-2:]):
        raise ValueError("valid_mask must be boolean [1,H_teacher,W_teacher]")
    if not torch.isfinite(h).all() or not torch.isfinite(target).all():
        raise FloatingPointError("non-finite bridge alignment features")
    if limit <= 0:
        return None
    aligned = F.interpolate(h.detach(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    mask = valid.detach().to(device=aligned.device)[0].reshape(-1)
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return None
    if indices.numel() > limit:
        # Deterministic, evenly spaced row-major support selection.
        positions = torch.div(torch.arange(limit, device=indices.device) * indices.numel(), limit, rounding_mode="floor")
        indices = indices[positions]
    x = aligned[0].permute(1, 2, 0).reshape(-1, channels_in).index_select(0, indices).to(device="cpu", dtype=torch.float64)
    y = target.detach()[0].permute(1, 2, 0).reshape(-1, channels_out).index_select(0, indices.to(target.device)).to(device="cpu", dtype=torch.float64)
    return (
        x.sum(dim=0), y.sum(dim=0), x.T @ x, x.T @ y, (y * y).sum(dim=0),
        torch.tensor(x.shape[0], dtype=torch.float64),
    )


def fit_bridge_alignment(
    samples: Iterable[BridgeFitSample],
    config: BridgeAlignmentConfig,
    identities: Mapping[str, Any],
    output_path: str | os.PathLike[str] | None = None,
) -> BridgeAlignmentArtifact:
    """Fit P by patient-balanced ridge feature regression under an explicit budget.

    Samples are consumed in caller-locked order, at most ``max_source_ids`` and
    ``max_pixels``. Each source ID is unique. At least two distinct TRAIN
    patients must contribute valid features. The fit uses no labels and never
    initializes P randomly.
    """
    config.validate()
    normalized_identities = _normalise_identities(identities)
    iterator = iter(samples)
    source_ids: list[str] = []
    fit_observations: list[dict[str, Any]] = []
    patient_ids: set[str] = set()
    by_patient: dict[str, list[Tensor]] = {}
    channels_in: int | None = None
    channels_out: int | None = None
    remaining_pixels = config.max_pixels

    while len(source_ids) < config.max_source_ids and remaining_pixels > 0:
        try:
            sample = next(iterator)
        except StopIteration:
            break
        if not isinstance(sample, BridgeFitSample):
            raise TypeError("bridge fit input must yield BridgeFitSample values")
        if sample.split != "TRAIN":
            raise ValueError("bridge projector fitting accepts TRAIN samples only")
        if sample.source_id in source_ids:
            raise ValueError(f"duplicate bridge fit source_id: {sample.source_id}")
        if sample.patient_id not in patient_ids and len(patient_ids) >= config.max_train_patients:
            break
        if channels_in is None:
            if sample.student_h.ndim != 4 or sample.teacher_pre_refine0.ndim != 4:
                raise ValueError("bridge fit features must be BCHW tensors")
            channels_in, channels_out = int(sample.student_h.shape[1]), int(sample.teacher_pre_refine0.shape[1])
        assert channels_out is not None
        stats = _pixel_stats(sample, channels_in, channels_out, remaining_pixels)
        source_ids.append(sample.source_id)
        patient_ids.add(sample.patient_id)
        used_sample_pixels = 0 if stats is None else int(stats[-1].item())
        fit_observations.append({
            "source_id": sample.source_id,
            "patient_id": sample.patient_id,
            "student_feature_hw": [int(value) for value in sample.student_h.shape[-2:]],
            "teacher_feature_hw": [int(value) for value in sample.teacher_pre_refine0.shape[-2:]],
            "valid_pixels": int(sample.valid_mask.sum().item()),
            "used_pixels": used_sample_pixels,
        })
        if stats is None:
            continue
        remaining_pixels -= used_sample_pixels
        if sample.patient_id not in by_patient:
            by_patient[sample.patient_id] = [torch.zeros_like(value) for value in stats]
        for index, value in enumerate(stats):
            by_patient[sample.patient_id][index] += value

    if channels_in is None or channels_out is None or len(by_patient) < 2:
        raise ValueError("bridge alignment needs valid image features from at least two TRAIN patients")

    # Equal patient weight; each patient's pixels contribute their within-patient mean.
    count_patients = len(by_patient)
    means = [sum((stats[i] / stats[-1] for stats in by_patient.values()), torch.zeros_like(next(iter(by_patient.values()))[i])) / count_patients
             for i in range(5)]
    mean_x, mean_y, mean_xx, mean_xy, mean_yy = means
    cov_xx = mean_xx - torch.outer(mean_x, mean_x)
    cov_xy = mean_xy - torch.outer(mean_x, mean_y)
    cov_yy = mean_yy - torch.outer(mean_y, mean_y)
    cov_xx = 0.5 * (cov_xx + cov_xx.T)
    cov_yy = 0.5 * (cov_yy + cov_yy.T)
    system = cov_xx + float(config.ridge) * torch.eye(channels_in, dtype=torch.float64)
    weight_x_by_y = torch.linalg.solve(system, cov_xy)
    bias = mean_y - mean_x @ weight_x_by_y
    # Conv2d stores [C_out,C_in,1,1].
    weight = weight_x_by_y.T.contiguous().reshape(channels_out, channels_in, 1, 1).to(torch.float32)
    bias32 = bias.to(torch.float32)
    if not torch.isfinite(weight).all() or not torch.isfinite(bias32).all():
        raise FloatingPointError("non-finite fitted bridge projector")
    fitted_weight_x_by_y = weight[:, :, 0, 0].double().T
    fitted_bias = bias32.double()
    residual_mean = mean_y - mean_x @ fitted_weight_x_by_y - fitted_bias
    residual_covariance = (
        cov_yy - fitted_weight_x_by_y.T @ cov_xy - cov_xy.T @ fitted_weight_x_by_y
        + fitted_weight_x_by_y.T @ cov_xx @ fitted_weight_x_by_y
    )
    residual_covariance = 0.5 * (residual_covariance + residual_covariance.T)
    residual_trace = torch.trace(residual_covariance).clamp_min(0.0)
    residual_mean_squared_sum = residual_mean.square().sum()
    mse = (residual_trace + residual_mean_squared_sum) / channels_out
    objective = mse + float(config.ridge) * (fitted_weight_x_by_y.square().sum() / channels_out)
    used_pixels = sum(row["used_pixels"] for row in fit_observations)
    sorted_patients = sorted(by_patient)
    sorted_sources = sorted(source_ids)
    input_eigenvalues = torch.linalg.eigvalsh(cov_xx)
    residual_eigenvalues = torch.linalg.eigvalsh(residual_covariance)
    target_trace = torch.trace(cov_yy)
    covariance_sha256 = {
        "input": _matrix_sha256(cov_xx),
        "input_target": _matrix_sha256(cov_xy),
        "target": _matrix_sha256(cov_yy),
        "residual": _matrix_sha256(residual_covariance),
    }
    fit_metrics = {
        "patient_balanced_feature_mse": float(mse.item()),
        "ridge_objective": float(objective.item()),
        "residual_covariance_trace": max(0.0, float(residual_trace.item())),
        "residual_covariance_min_eigenvalue": max(0.0, float(residual_eigenvalues.min().item())),
        "residual_mean_squared_sum": float(residual_mean_squared_sum.item()),
        "input_covariance_trace": max(0.0, float(torch.trace(cov_xx).item())),
        "input_covariance_min_eigenvalue": max(0.0, float(input_eigenvalues.min().item())),
        "cross_covariance_frobenius": float(torch.linalg.vector_norm(cov_xy).item()),
        "target_covariance_trace": max(0.0, float(target_trace.item())),
        "fit_pixels": used_pixels,
    }
    if any(not math.isfinite(value) for value in fit_metrics.values()):
        raise FloatingPointError("non-finite bridge alignment residual or covariance metric")
    metadata = {
        "schema": ALIGNMENT_SCHEMA,
        "fit_split": "TRAIN",
        "feature_tap": ALIGNMENT_FEATURE_TAP,
        "target_tap": ALIGNMENT_TARGET_TAP,
        "bridge_suffix": "NativeStudent.refine[0] -> bilinear upsample -> fuse(zero_detail) -> head",
        "raw_detail_conditioning": False,
        "student_channels": channels_in,
        "teacher_channels": channels_out,
        "loss": ALIGNMENT_FIT_LOSS,
        "ridge": float(config.ridge),
        "alignment": ALIGNMENT_MODE,
        "budget": {
            "max_source_ids": config.max_source_ids,
            "max_train_patients": config.max_train_patients,
            "max_pixels": config.max_pixels,
            "seed": config.seed,
            "used_source_ids": len(source_ids),
            "used_train_patients": len(sorted_patients),
            "used_pixels": used_pixels,
        },
        "fit_metrics": fit_metrics,
        "covariance_sha256": covariance_sha256,
        "train_patient_ids": sorted_patients,
        "train_source_ids": sorted_sources,
        "train_source_order": list(source_ids),
        "fit_observations": fit_observations,
        "train_source_ids_sha256": hashlib.sha256(_canonical(sorted_sources)).hexdigest(),
        "train_source_order_sha256": hashlib.sha256(_canonical(source_ids)).hexdigest(),
        "fit_observations_sha256": hashlib.sha256(_canonical(fit_observations)).hexdigest(),
        "identities": normalized_identities,
        "projector_sha256": _tensor_sha256(weight, bias32),
    }
    artifact = _new_artifact(metadata, weight, bias32)
    if output_path is not None:
        save_bridge_alignment(artifact, output_path)
    return artifact


def save_bridge_alignment(artifact: BridgeAlignmentArtifact, path: str | os.PathLike[str]) -> Path:
    """Write a complete verified JSON artifact once; existing files are never replaced."""
    artifact.verify()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _artifact_payload(
        artifact.metadata,
        artifact.projector_weight.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        artifact.projector_bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    )
    payload["artifact_sha256"] = artifact.artifact_sha256
    encoded = _canonical(payload) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".bridge-alignment-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return destination


def load_bridge_alignment(
    path: str | os.PathLike[str],
    expected_identities: Mapping[str, Any] | None = None,
) -> BridgeAlignmentArtifact:
    """Load an alignment only after checking its payload hash and identities."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != ALIGNMENT_SCHEMA:
        raise ValueError("invalid bridge alignment artifact schema")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("schema") != ALIGNMENT_SCHEMA:
        raise ValueError("bridge alignment metadata missing or inconsistent")
    weight = torch.tensor(payload.get("projector_weight"), dtype=torch.float32)
    bias = torch.tensor(payload.get("projector_bias"), dtype=torch.float32)
    artifact = BridgeAlignmentArtifact(
        metadata=metadata,
        projector_weight=weight,
        projector_bias=bias,
        artifact_sha256=payload.get("artifact_sha256", ""),
    )
    artifact.verify()
    if metadata.get("projector_sha256") != _tensor_sha256(weight, bias):
        raise ValueError("bridge projector tensor hash mismatch")
    if expected_identities is not None:
        expected = _normalise_expected_identities(expected_identities)
        recorded = _normalise_identities(metadata.get("identities", {}))
        for name, value in expected.items():
            if recorded.get(name) != value:
                raise ValueError(f"bridge alignment identity mismatch: {name}")
    return artifact


__all__ = [
    "ALIGNMENT_FIT_LOSS",
    "ALIGNMENT_MODE",
    "BridgeAlignmentArtifact",
    "BridgeAlignmentConfig",
    "BridgeFitSample",
    "fit_bridge_alignment",
    "load_bridge_alignment",
    "save_bridge_alignment",
]
