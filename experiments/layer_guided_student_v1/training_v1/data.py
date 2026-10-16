"""Frozen historical-bank adapter for the opt-in Astra training path.

This module deliberately keeps the historical bank's arrays as source data.  It
does not turn historical ``Q`` into a conditional ``Z`` target, and it never
maps ``UNKNOWN`` to background.  The only semantic weighting policy exposed by
the adapter is the frozen historical reliability multiplied by the declared
per-class permission of the source-bank argmax.

The adapter performs one complete inventory/hash verification when it is
constructed.  Subsequent patient reads recheck the manifest pin and the
patient payload immediately before and after the read.  ``reverify`` is the
explicit end-of-run whole-bank check; it is intentionally not called for each
minibatch.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from integration.bank import PRODUCER_HASHES as REVIEWED_PRODUCER_HASHES
from integration.geometry import (
    GridGeometry,
    TeacherRegistration,
    normalize_cine,
    slice_contexts,
    transport_density,
)
from torch import Tensor

from self_audit_guidance.io import digest, load_bank_patient, verify

POLICY = "frozen_historical_q_reliability_v1"
ONTOLOGY = ("BG", "RV", "MYO", "LV")
FEATURE_DIM = 64
UNKNOWN = 255

# The old bank adapter reviewed teacher/pipeline/io.  Cardinal-v1 guidance also
# depends on the exact source that constructs R0/Q and their named priors.  This
# is the current main snapshot's reviewed core.py digest, recorded in the
# published cardinal-v1 bank manifests.
CARDINAL_CORE_SHA256 = (
    "1b44186c6bc9492b00710859abc0e4638593ee1f15d0797b3a3c1dbb92566ed0"
)
REVIEWED_CARDINAL_PRODUCER_HASHES = {
    **REVIEWED_PRODUCER_HASHES,
    "core.py": CARDINAL_CORE_SHA256,
}
PRODUCER_HASHES = REVIEWED_CARDINAL_PRODUCER_HASHES
HISTORICAL_POLICY = POLICY

_PATIENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field} must be a reviewed SHA256")
    return value


def _safe_patient_id(value: Any) -> str:
    if not isinstance(value, str) or not _PATIENT_RE.fullmatch(value) or ".." in value:
        raise ValueError("unsafe patient ID")
    return value


def _strict_patient_ids(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{field} must be a nonempty patient list")
    result = [_safe_patient_id(pid) for pid in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{field} contains duplicate patient IDs")
    return result


def _validate_splits(manifest: Mapping[str, Any], selected: Sequence[str]) -> None:
    splits = manifest.get("locked_splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("locked train/val/test patient splits are required")
    normalized: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        normalized[name] = _strict_patient_ids(
            splits[name], f"locked_splits.{name}", nonempty=name == "train"
        )
    if any(
        set(normalized[left]) & set(normalized[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("TRAIN/VAL/TEST patient leakage")
    if not set(selected) <= set(normalized["train"]):
        raise ValueError("selected bank patients must be a subset of TRAIN")


def _validate_affine(value: Any) -> tuple[tuple[float, ...], ...]:
    try:
        affine = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("image affine must be a finite 4x4 matrix") from exc
    if len(affine) != 4 or any(len(row) != 4 for row in affine):
        raise ValueError("image affine must be a finite 4x4 matrix")
    if not np.isfinite(np.asarray(affine, dtype=np.float64)).all():
        raise ValueError("image affine must be a finite 4x4 matrix")
    return affine


def _normalize_indices(indices: Any, *, shape: Sequence[int]) -> list[tuple[int, int]]:
    if isinstance(indices, Tensor):
        indices = indices.detach().cpu().tolist()
    elif isinstance(indices, np.ndarray):
        indices = indices.tolist()
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError("nonempty temporal/spatial indices are required")
    result: list[tuple[int, int]] = []
    for pair in indices:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or type(pair[0]) is not int
            or type(pair[1]) is not int
        ):
            raise ValueError("indices must contain integer (time, slice) pairs")
        t, z = pair
        if not (0 <= t < shape[0] and 0 <= z < shape[1]):
            raise ValueError("temporal/spatial index outside the cine")
        result.append((t, z))
    return result


def _report_class_pixels(report: Mapping[str, Any], pid: str) -> list[int]:
    values = report.get("class_pixels")
    if (
        not isinstance(values, list)
        or len(values) != len(ONTOLOGY)
        or any(type(value) is not int or value < 0 for value in values)
    ):
        raise ValueError(
            f"{pid}: patient_reports.class_pixels must contain four counts"
        )
    return list(values)


def _actual_class_pixels(labels: np.ndarray) -> list[int]:
    if labels.ndim < 1 or not np.isin(labels, [0, 1, 2, 3, UNKNOWN]).all():
        raise ValueError("bank labels contain an unsupported class or UNKNOWN value")
    return [int((labels == class_id).sum()) for class_id in range(len(ONTOLOGY))]


class FrozenTrainingBank:
    """Read-only, hash-pinned view of a frozen historical TRAIN guidance bank."""

    def __init__(self, path: str | Path, expected_manifest_sha256: str):
        supplied_path = Path(path)
        if supplied_path.is_symlink():
            raise ValueError("frozen TRAIN bank path symlink refused")
        self.path = supplied_path.resolve()
        if not self.path.is_dir() or self.path.is_symlink():
            raise ValueError("frozen TRAIN bank directory is required")
        expected = _require_digest(expected_manifest_sha256, "expected_manifest_sha256")
        frozen = self.path / "FROZEN.json"
        if frozen.is_symlink() or not frozen.is_file():
            raise ValueError("frozen TRAIN bank needs a regular FROZEN.json")
        actual = digest(frozen)
        if actual != expected:
            raise ValueError("bank manifest differs from the locked input")

        # This is the sole complete verification at construction.  Per-patient
        # reads below check only their own payload plus the manifest/inventory
        # pin; reverify() is the explicit end-of-run whole-bank check.
        manifest = verify(self.path, "guidance_bank")
        if digest(frozen) != expected:
            raise ValueError("bank manifest changed during initialization")
        if manifest.get("split") != "train":
            raise ValueError("FrozenTrainingBank accepts TRAIN banks only")

        selected = _strict_patient_ids(
            manifest.get("patients"), "patients", nonempty=True
        )
        _validate_splits(manifest, selected)
        self._validate_provenance(manifest)
        self._validate_inventory(manifest, selected)

        reports = manifest.get("patient_reports")
        if not isinstance(reports, dict) or set(reports) != set(selected):
            raise ValueError(
                "patient_reports must cover exactly the selected TRAIN patients"
            )
        class_patients = [
            sum(
                _report_class_pixels(reports[pid], pid)[class_id] > 0
                for pid in selected
            )
            for class_id in range(len(ONTOLOGY))
        ]

        config = manifest.get("config")
        if not isinstance(config, dict) or config.get("feature_dim") != FEATURE_DIM:
            raise ValueError("historical bank must declare feature_dim=64")

        self.manifest = copy.deepcopy(manifest)
        self.sha256 = expected
        self.patients = list(selected)
        self.allowed_classes = [count >= 2 for count in class_patients]
        self.class_patients = list(class_patients)
        self.policy = POLICY
        self.orientation_contract = {
            "policy": "cardinal_v1",
            "basis": "explicit_field_and_pinned_producer"
            if "orientation_policy" in manifest
            else "historical_schema_pinned_producer_without_policy_field",
            "anatomical_orientation_certified": False,
        }
        self._inventory = frozenset(p.name for p in self.path.iterdir())
        self._payload_hashes = {
            pid: _require_digest(manifest["files"][f"{pid}.npz"], f"{pid}.npz")
            for pid in selected
        }

    @staticmethod
    def _validate_provenance(manifest: Mapping[str, Any]) -> None:
        teacher = manifest.get("teacher")
        if not isinstance(teacher, dict) or teacher.get("frozen") is not True:
            raise ValueError("frozen teacher provenance required")
        _require_digest(teacher.get("checkpoint_sha256"), "teacher.checkpoint_sha256")
        source_id = teacher.get("source")
        provider, commit = (
            teacher.get("provider"),
            teacher.get("source_commit_expected"),
        )
        if not (isinstance(source_id, str) and source_id.strip()) and not (
            isinstance(provider, str)
            and provider.strip()
            and isinstance(commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", commit)
        ):
            raise ValueError(
                "teacher needs source or provider plus pinned source_commit_expected"
            )
        if teacher.get("segmentation_finetune") is not False:
            raise ValueError("historical teacher must not be segmentation-finetuned")

        source = manifest.get("source_sha256")
        if not isinstance(source, dict):
            raise ValueError("reviewed source producer hashes are required")
        for name, expected in REVIEWED_CARDINAL_PRODUCER_HASHES.items():
            if source.get(name) != expected:
                raise ValueError(
                    f"unreviewed or mismatched cardinal_v1 producer: {name}"
                )
        # The published cardinal-v1 producer predates this metadata field. Its
        # exact four source hashes above define the legacy interpretation. A
        # conflicting field is rejected; absent metadata is recorded explicitly.
        if (
            "orientation_policy" in manifest
            and manifest["orientation_policy"] != "cardinal_v1"
        ):
            raise ValueError(
                "FrozenTrainingBank requires the reviewed cardinal_v1 bank"
            )

    @staticmethod
    def _validate_inventory(
        manifest: Mapping[str, Any], selected: Sequence[str]
    ) -> None:
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("frozen bank file inventory is required")
        payloads = {
            name[:-4]
            for name in files
            if isinstance(name, str) and name.endswith(".npz")
        }
        if payloads != set(selected):
            raise ValueError(
                "bank payload inventory must match selected TRAIN patients"
            )
        image_records = manifest.get("image_records")
        if not isinstance(image_records, dict) or set(image_records) != set(selected):
            raise ValueError(
                "image_records must cover exactly the selected TRAIN patients"
            )
        for pid in selected:
            name = f"{pid}.npz"
            if name not in files or Path(name).name != name:
                raise ValueError(f"missing strict patient payload: {name}")

    def _assert_pinned_manifest(self) -> None:
        frozen = self.path / "FROZEN.json"
        if frozen.is_symlink() or not frozen.is_file() or digest(frozen) != self.sha256:
            raise ValueError("bank manifest changed while reading")
        current_inventory = frozenset(p.name for p in self.path.iterdir())
        if current_inventory != self._inventory:
            raise ValueError("bank file inventory changed while reading")

    def reverify(self) -> dict[str, Any]:
        """Rehash the complete frozen inventory at the end of a run."""

        self._assert_pinned_manifest()
        manifest = verify(self.path, "guidance_bank")
        if digest(self.path / "FROZEN.json") != self.sha256:
            raise ValueError("bank manifest changed during final verification")
        if manifest != self.manifest:
            raise ValueError("bank manifest contents differ from the pinned manifest")
        return manifest

    def _patient_payload(self, pid: str) -> dict[str, np.ndarray]:
        pid = _safe_patient_id(pid)
        if pid not in self.patients:
            raise ValueError("only selected TRAIN patients are allowed")
        self._assert_pinned_manifest()
        payload = self.path / f"{pid}.npz"
        expected = self._payload_hashes[pid]
        if payload.is_symlink() or not payload.is_file() or digest(payload) != expected:
            raise ValueError(f"patient payload hash mismatch: {pid}")
        data = load_bank_patient(self.path, pid)
        if digest(payload) != expected:
            raise ValueError(f"patient payload changed while reading: {pid}")
        self._assert_pinned_manifest()
        return data

    def load_patient(self, patient_id: str) -> dict[str, Any]:
        """Load one immutable patient and construct its registered geometry."""

        pid = _safe_patient_id(patient_id)
        data = self._patient_payload(pid)
        meta = self.manifest["image_records"][pid]
        image = data["image"]
        features = data["features"]
        expected_image_shape = tuple(reversed(meta.get("shape_xyzt", ())))
        if image.ndim != 4 or tuple(image.shape) != expected_image_shape:
            raise ValueError(f"{pid}: native image shape does not match image_records")
        if features.ndim != 5 or features.shape[:2] != image.shape[:2]:
            raise ValueError(f"{pid}: teacher feature grid does not match image")
        if features.shape[-1] != FEATURE_DIM:
            raise ValueError(f"{pid}: teacher feature dimension must be 64")

        counts = _actual_class_pixels(data["labels"])
        expected_counts = _report_class_pixels(
            self.manifest["patient_reports"][pid], pid
        )
        if counts != expected_counts:
            raise ValueError(
                f"{pid}: patient_reports class support differs from payload labels"
            )

        image_sha = _require_digest(meta.get("image_sha256"), f"{pid}.image_sha256")
        affine = _validate_affine(meta.get("affine"))
        normalized, normalization = normalize_cine(torch.from_numpy(image), image_sha)
        producer = tuple(sorted(REVIEWED_CARDINAL_PRODUCER_HASHES.items()))
        registration = TeacherRegistration(
            tuple(normalized.shape[-2:]),
            tuple(features.shape[-3:-1]),
            image_sha,
            affine,
            self.sha256,
            producer,
        )
        geometry = GridGeometry(
            tuple(normalized.shape[-2:]),
            tuple(features.shape[-3:-1]),
            image_sha,
            affine,
            meta.get("spatial_unit", "unknown"),
            registration,
        )
        # Keep cached float16 features on CPU; cast only the selected batch.
        teacher_features = torch.from_numpy(features).permute(0, 1, 4, 2, 3)
        return {
            "patient_id": pid,
            "data": data,
            "image": data["image"],
            "features": data["features"],
            "q": data["q"],
            "r0": data["r0"],
            "reliability": data["reliability"],
            "labels": data["labels"],
            "normalized_cine": normalized,
            "cine": normalized,
            "teacher_features": teacher_features,
            "geometry": geometry,
            "normalization": normalization,
            "allowed_classes": self.allowed_classes,
            "bank_manifest_sha256": self.sha256,
            "policy": POLICY,
        }

    def batch(
        self,
        patient: Mapping[str, Any] | str,
        indices: Sequence[Sequence[int]],
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Build one student batch without converting Q into named Z evidence."""

        if isinstance(patient, str):
            patient = self.load_patient(patient)
        if not isinstance(patient, Mapping):
            raise TypeError("patient must be a loaded patient mapping or patient ID")
        if patient.get("bank_manifest_sha256") != self.sha256:
            raise ValueError("patient was not loaded from this frozen bank")
        normalized = patient.get("normalized_cine")
        teacher_all = patient.get("teacher_features")
        if not isinstance(normalized, Tensor) or not isinstance(teacher_all, Tensor):
            raise ValueError(
                "patient must contain normalized cine and teacher features"
            )
        index_list = _normalize_indices(indices, shape=normalized.shape[:2])
        selected = list(zip(*index_list))
        times, slices = selected
        target_device = torch.device(device)
        images = slice_contexts(normalized, index_list).to(target_device)
        teacher_features = teacher_all[times, slices].to(
            target_device, dtype=torch.float32
        )
        data = patient.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("patient raw arrays are required for batching")
        q = torch.from_numpy(np.ascontiguousarray(data["q"][times, slices])).float()
        q = q.permute(0, 3, 1, 2).contiguous().to(target_device)
        reliability = (
            torch.from_numpy(np.ascontiguousarray(data["reliability"][times, slices]))
            .float()
            .to(target_device)
        )
        allowed = torch.as_tensor(
            self.allowed_classes, dtype=torch.bool, device=target_device
        )
        q_argmax = q.argmax(dim=1)
        weights = reliability * allowed[q_argmax].to(reliability.dtype)
        semantic_active_mask = (q_argmax != 0) & (weights > 0)
        return {
            "patient_id": patient["patient_id"],
            "indices": index_list,
            "images": images,
            "teacher_features": teacher_features,
            "q": q,
            "q_argmax": q_argmax,
            "weights": weights,
            "geometry": patient["geometry"],
            "semantic_active_mask": semantic_active_mask,
            "semantic_active": bool(semantic_active_mask.any().item()),
            "policy": POLICY,
            "named_mass_status": "NOT_CONDITIONAL_Z",
        }


def _validate_batch(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, Any]:
    if not isinstance(batch, Mapping):
        raise TypeError("batch mapping required")
    q = batch.get("q")
    weights = batch.get("weights")
    teacher = batch.get("teacher_features")
    geometry = batch.get("geometry")
    if (
        not isinstance(q, Tensor)
        or not isinstance(weights, Tensor)
        or not isinstance(teacher, Tensor)
    ):
        raise ValueError("batch must contain tensor q, weights and teacher_features")
    if not isinstance(geometry, GridGeometry):
        raise ValueError("batch must contain registered GridGeometry")
    if q.ndim != 4 or q.shape[1] != len(ONTOLOGY):
        raise ValueError("q must be Bx4xHt xWt")
    if weights.shape != (q.shape[0], q.shape[2], q.shape[3]):
        raise ValueError("weights must be BxHt xWt")
    if (
        teacher.ndim != 4
        or teacher.shape[0] != q.shape[0]
        or teacher.shape[1] != FEATURE_DIM
    ):
        raise ValueError("teacher_features must be Bx64xHt xWt")
    if (
        tuple(q.shape[-2:]) != geometry.teacher_hw
        or tuple(teacher.shape[-2:]) != geometry.teacher_hw
    ):
        raise ValueError("batch tensors do not match registered teacher grid")
    if (
        not torch.is_floating_point(q)
        or not torch.is_floating_point(weights)
        or not torch.is_floating_point(teacher)
    ):
        raise ValueError("q, weights and teacher_features must be floating point")
    if (
        not torch.isfinite(q).all()
        or not torch.isfinite(weights).all()
        or not torch.isfinite(teacher).all()
    ):
        raise ValueError("batch tensors must be finite")
    if (q < 0).any() or (weights < 0).any():
        raise ValueError("q and weights must be nonnegative")
    if not torch.allclose(q.sum(dim=1), torch.ones_like(weights), atol=2e-5, rtol=0):
        raise ValueError("q must be a simplex distribution")
    return q, weights, teacher, geometry


def prepare_targets(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Transport historical Q/reliability and teacher features to student grids.

    ``q`` is treated as a stored historical distribution.  Its weighted mass
    and positive support are transported together, then normalized only where
    transported support is positive.  No reliability threshold is introduced,
    and no semantic source certificate is claimed.
    """

    q, weights, teacher, geometry = _validate_batch(batch)
    mass = q * weights[:, None]
    support = transport_density(weights[:, None], geometry, stride=4)[:, 0]
    transported_mass = transport_density(mass, geometry, stride=4)
    positive = support > 0
    target = torch.full_like(transported_mass, 1.0 / len(ONTOLOGY))
    target = torch.where(
        positive[:, None],
        transported_mass
        / support[:, None].clamp_min(torch.finfo(transported_mass.dtype).tiny),
        target,
    )
    target = target / target.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(target.dtype).tiny
    )

    feature_mass = transport_density(teacher, geometry, stride=16)
    feature_coverage = transport_density(
        torch.ones_like(teacher[:, :1]), geometry, stride=16
    )[:, 0]
    feature_valid = feature_coverage >= 1.0 - 1e-6
    feature_valid &= feature_coverage > 0
    feature_target = feature_mass / feature_coverage[:, None].clamp_min(
        torch.finfo(feature_mass.dtype).tiny
    )
    feature_target = torch.where(
        feature_valid[:, None], feature_target, torch.zeros_like(feature_target)
    )

    return {
        "target": target,
        "coarse_target": target,
        "weights": support,
        "coarse_weights": support,
        "support": support,
        "valid": positive,
        "transported_mass": transported_mass,
        "feature_target": feature_target,
        "feature_targets": feature_target,
        "aligned_teacher_features": feature_target,
        "teacher_target": feature_target,
        "feature_coverage": feature_coverage,
        "feature_valid": feature_valid,
        "feature_weights": feature_valid,
        "geometry": geometry,
        "policy": POLICY,
        "named_mass_status": "NOT_CONDITIONAL_Z",
        "semantic_active": batch.get("semantic_active"),
        "semantic_active_mask": batch.get("semantic_active_mask"),
    }


__all__ = [
    "CARDINAL_CORE_SHA256",
    "FEATURE_DIM",
    "HISTORICAL_POLICY",
    "ONTOLOGY",
    "POLICY",
    "PRODUCER_HASHES",
    "REVIEWED_CARDINAL_PRODUCER_HASHES",
    "FrozenTrainingBank",
    "prepare_targets",
]
