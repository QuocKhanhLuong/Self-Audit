"""Image-only unit loading: fitting views, selection views, sealed verification.

A *unit* is one acquired Z slice of one source frame, together with its two
neighbouring slices as context. Multi-frame sources are enumerated by discovery
for every acquired frame; each frame has its own stable volume identity. The
loader builds all three observation roles on the native grid, zeroes every
withheld intensity *before* any resampling, and then resamples each role with a
masked normalized convolution so no output pixel ever mixes two roles.

Order matters and is not negotiable:

``partition on native grid -> normalize from fit pixels only -> zero withheld
-> resample per role``

Normalizing first would let a withheld intensity move the mean and standard
deviation that the fitting view is expressed in. Resampling first would let a
withheld intensity bleed across the block boundary. Both are leaks, and the
selection/verification mutation test in ``tests/test_maskfree_data.py`` exists
to keep this order honest.

``ImageOnlyDataset`` has no path to verification intensities at all. The sealed
role is reachable only through :func:`load_verification_unit`, which demands a
validated freeze receipt naming this manifest and this unit's partition.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..contracts import FittingView, ScoringView, TrainingUnit
from .discovery import SCHEMA_VERSION
from .firewall import assert_image_only
from .geometry import (
    inverse_transform_record,
    masked_resize,
    nearest_resize_mask,
    read_slice_stack,
)
from .partition import RolePartition, build_partition

#: Percentile clip computed on fitting pixels only.
CLIP_PERCENTILES = (0.5, 99.5)
#: Floor on the fit standard deviation, in raw intensity units.
STD_FLOOR = 1e-6

#: Keys the authoritative W7 freeze manifest carries before sealed observations
#: are released.  The data layer checks identity fields after W7 has verified
#: the manifest body and every artifact; it never reimplements a hash gate.
REQUIRED_FREEZE_KEYS = (
    "freeze_id",
    "dataset",
    "manifest_id",
    "partition_ids",
    "files",
    "completeness",
)


class UnitNotFoundError(KeyError):
    """Raised when a unit id is absent from the manifest."""


class FreezeReceiptError(PermissionError):
    """Raised when sealed verification data is requested without a valid freeze."""


class VerificationAccessError(PermissionError):
    """Raised when training-side code reaches for verification observations."""


_RECORD_INDICES: OrderedDict[int, tuple[Any, Any, dict[str, dict[str, Any]]]] = OrderedDict()


def _record_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bounded index for frozen manifests, also used by deployment/verification."""
    key = id(manifest)
    records = manifest["records"]
    cached = _RECORD_INDICES.get(key)
    if cached is not None and cached[0] is manifest and cached[1] is records:
        if len(cached[2]) != len(records):
            raise ValueError("frozen data manifest record list changed")
        _RECORD_INDICES.move_to_end(key)
        return cached[2]
    index = {record["unit_id"]: record for record in records}
    if len(index) != len(records):
        raise ValueError("duplicate unit IDs in data manifest")
    _RECORD_INDICES[key] = (manifest, records, index)
    while len(_RECORD_INDICES) > 4:
        _RECORD_INDICES.popitem(last=False)
    return index


def _get_record(manifest: dict[str, Any], unit_id: str) -> dict[str, Any]:
    try:
        return _record_index(manifest)[unit_id]
    except KeyError as exc:
        raise UnitNotFoundError(f"unit {unit_id!r} is not in manifest {manifest.get('manifest_id')}") from exc


def _native_stack(record: dict[str, Any]) -> torch.Tensor:
    from ..progress import current_progress
    with current_progress().stage("data.decode_slice_stack", path=record["path"],
                                  unit_id=record["unit_id"], slice_index=record["slice_index"],
                                  frame_index=record["frame_index"]):
        stack = read_slice_stack(
            record["path"],
            depth_axis=int(record["depth_axis"]),
            slice_index=int(record["slice_index"]),
            frame_index=record["frame_index"],
            frame_axis=record.get("frame_axis"),
        )
    return torch.from_numpy(np.ascontiguousarray(stack)).to(torch.float32)


def _fit_statistics(stack: torch.Tensor, fit_mask: torch.Tensor) -> dict[str, float]:
    """Clip percentiles and mean/std from fitting pixels only, across context."""
    values = stack[:, fit_mask]
    if values.numel() == 0:
        raise ValueError("fitting support is empty; cannot derive normalization")
    flat = values.reshape(-1).to(torch.float64)
    low = float(torch.quantile(flat, CLIP_PERCENTILES[0] / 100.0))
    high = float(torch.quantile(flat, CLIP_PERCENTILES[1] / 100.0))
    if not high > low:
        high = low + STD_FLOOR
    clipped = flat.clamp(low, high)
    mean = float(clipped.mean())
    std = max(float(clipped.std(unbiased=False)), STD_FLOOR)
    return {
        "percentile_low": low,
        "percentile_high": high,
        "mean": mean,
        "std": std,
        "source": "fit_pixels_only",
        "percentiles": list(CLIP_PERCENTILES),
        "std_floor": STD_FLOOR,
        "n_fit_pixels_used": int(values.numel()),
    }


def _normalize(stack: torch.Tensor, stats: dict[str, float]) -> torch.Tensor:
    clipped = stack.clamp(stats["percentile_low"], stats["percentile_high"])
    return (clipped - stats["mean"]) / stats["std"]


def _resize_role(
    normalized: torch.Tensor,
    role_mask: torch.Tensor,
    out_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero outside the role, then resample values and mask consistently."""
    zeroed = normalized * role_mask.to(normalized.dtype)
    resized, support_from_values = masked_resize(zeroed, role_mask, out_hw)
    support = nearest_resize_mask(role_mask, out_hw) & support_from_values
    resized = torch.where(support.unsqueeze(0), resized, torch.zeros_like(resized))
    return resized, support


def _resize_support(role_mask: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Resize a role support mask without reading any values from that role."""
    zeros = torch.zeros((1, *role_mask.shape), dtype=torch.float32)
    _, pooled_support = masked_resize(zeros, role_mask, out_hw)
    return nearest_resize_mask(role_mask, out_hw) & pooled_support


def _build_unit_tensors(
    record: dict[str, Any],
    partition: RolePartition,
    image_size: int,
    *,
    include_verify: bool = False,
) -> dict[str, Any]:
    stack = _native_stack(record)
    height, width = stack.shape[1], stack.shape[2]
    if (height, width) != tuple(record["native_hw"]):
        raise ValueError(
            f"unit {record['unit_id']} loaded {height}x{width}, manifest says {record['native_hw']}"
        )

    from ..progress import current_progress
    with current_progress().stage("data.fit_normalization", unit_id=record["unit_id"]):
        stats = _fit_statistics(stack, partition.fit)
    normalized = _normalize(stack, stats)
    out_hw = (int(image_size), int(image_size))

    context, fit_support = _resize_role(normalized, partition.fit, out_hw)
    select_values, select_support = _resize_role(
        normalized, partition.select, out_hw
    )
    # Support geometry is safe to derive before the freeze.  The verify values
    # themselves are deliberately not materialised unless the caller has
    # already passed the authoritative W7 freeze gate.
    verify_support = _resize_support(partition.verify, out_hw)
    verify_values = None
    if include_verify:
        verify_values, verify_support_from_values = _resize_role(
            normalized, partition.verify, out_hw
        )
        verify_support = verify_support & verify_support_from_values

    # Independent nearest resizing of three native masks can round two roles on
    # to the same output pixel. Resolve with a fixed priority so the fitting
    # support can never claim a pixel that any withheld role also claims.
    select_support = select_support & ~verify_support
    fit_support = fit_support & ~(verify_support | select_support)
    context = torch.where(fit_support.unsqueeze(0), context, torch.zeros_like(context))
    select_values = torch.where(
        select_support.unsqueeze(0), select_values, torch.zeros_like(select_values)
    )
    if verify_values is not None:
        verify_values = torch.where(
            verify_support.unsqueeze(0), verify_values, torch.zeros_like(verify_values)
        )

    if not fit_support.any():
        raise ValueError(f"unit {record['unit_id']} has empty fitting support after resampling")

    return {
        "stats": stats,
        "context": context,
        "fit_support": fit_support,
        "select_values": select_values,
        "select_support": select_support,
        "verify_values": verify_values,
        "verify_support": verify_support,
        "native_hw": (int(height), int(width)),
        "unit_hw": out_hw,
    }


def _unit_record(
    record: dict[str, Any],
    manifest: dict[str, Any],
    partition: RolePartition,
    built: dict[str, Any],
    image_size: int,
) -> dict[str, Any]:
    """Whitelisted per-unit metadata. No intensities, no masks, no verify values."""
    return {
        "unit_id": record["unit_id"],
        "volume_id": record.get("volume_id", record["unit_id"]),
        "study_id": record["study_id"],
        "patient_id": record["patient_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "manifest_id": manifest["manifest_id"],
        "protocol": manifest["resolved_protocol"],
        "protocol_reason": manifest["protocol_reason"],
        "partition_id": partition.partition_id,
        "partition_spec": partition.spec,
        "image_size": int(image_size),
        "native_hw": list(built["native_hw"]),
        "depth": record["depth"],
        "num_slices": record.get("num_slices", record["depth"]),
        "depth_axis": record["depth_axis"],
        "slice_index": record["slice_index"],
        "frame_index": record["frame_index"],
        "frame_selection_rule": record["frame_selection_rule"],
        "source_format": record["source_format"],
        "native_geometry": record["native_geometry"],
        "native_geometry_reason": record["native_geometry_reason"],
        "spacing_valid": record["spacing_valid"],
        "spacing": record.get("spacing_mm") if record.get("spacing_valid") else None,
        "orientation": record["orientation"],
        "zooms": record["zooms"],
        "native_affine": record["native_affine"],
        "export_grid": record["export_grid"],
        "native_grid_export": record["native_grid_export"],
        "inverse_transform": inverse_transform_record(
            native_shape=record.get("native_shape", [int(s) for s in record["shape"][:3]]),
            depth_axis=int(record["depth_axis"]),
            slice_index=int(record["slice_index"]),
            frame_index=record["frame_index"],
            native_hw=built["native_hw"],
            unit_hw=built["unit_hw"],
            native_grid_export=bool(record["native_grid_export"]),
            frame_axis=record.get("frame_axis"),
            full_native_shape=record["shape"] if len(record["shape"]) == 4 else None,
        ),
        "normalization": built["stats"],
        "support_counts": {
            "fit": int(built["fit_support"].sum()),
            "select": int(built["select_support"].sum()),
            "verify_reserved": int(built["verify_support"].sum()),
            "native": partition.counts(),
        },
        "capabilities": {
            "fitting": True,
            "selection": True,
            "verification": False,
            "verification_reason": "sealed until predictions are frozen",
        },
        "cohort_provenance": record["cohort_provenance"],
        "acquisition_id": record["acquisition_id"],
        "mask_inputs_used": False,
        "deployment_only": False,
    }


def build_training_unit(
    manifest: dict[str, Any],
    unit_id: str,
    *,
    image_size: int = 128,
    seed: int = 42,
) -> TrainingUnit:
    """Assemble the fitting view and the selection scoring view for one unit."""
    return _build_training_unit_from_record(
        manifest,
        _get_record(manifest, unit_id),
        image_size=image_size,
        seed=seed,
    )


def _build_training_unit_from_record(
    manifest: dict[str, Any],
    record: dict[str, Any],
    *,
    image_size: int,
    seed: int,
) -> TrainingUnit:
    """Build a unit from an already indexed record.

    ``ImageOnlyDataset`` keeps the manifest's frozen record list and passes the
    selected mapping here. This avoids rebuilding a ``unit_id -> record`` map
    over every Z/T item while preserving the public lookup API above.
    """
    height, width = int(record["native_hw"][0]), int(record["native_hw"][1])
    partition = build_partition(height, width, study_id=record["study_id"], seed=seed)
    if partition.partition_id != record["partition_id"]:
        raise ValueError(
            f"partition drift for {record['unit_id']}: manifest {record['partition_id']} vs "
            f"rebuilt {partition.partition_id}"
        )
    built = _build_unit_tensors(record, partition, image_size)
    unit_record = _unit_record(record, manifest, partition, built, image_size)
    # TrainingUnit.record is also handed to W7 export. Keep the full source
    # fingerprint there for resume/export lineage, while the fitting view gets
    # the separate whitelist-safe metadata mapping above.
    export_record = dict(unit_record)
    source_hash = record.get("source_hash", record.get("source_fingerprint"))
    if source_hash is not None:
        export_record["source_hash"] = source_hash
    # W7 sidecars retain the declared physical-unit lineage. These fields are
    # export metadata only; they are intentionally absent from fitting metadata
    # unless the coordinator explicitly whitelists a safe derived value.
    for key in ("spacing_mm", "spatial_unit", "xyzt_units"):
        if record.get(key) is not None:
            export_record[key] = record[key]

    fitting = FittingView(
        image=built["context"][1:2].clone(),
        support=built["fit_support"].clone(),
        context=built["context"].clone(),
        study_id=record["study_id"],
        unit_id=record["unit_id"],
        protocol=manifest["resolved_protocol"],
        metadata=unit_record,
        partition_id=partition.partition_id,
    )
    fitting.validate()

    selection = ScoringView(
        image=built["select_values"][1:2].clone(),
        support=built["select_support"].clone(),
        study_id=record["study_id"],
        unit_id=record["unit_id"],
        role="select",
        partition_id=partition.partition_id,
    )
    selection.validate()
    return TrainingUnit(fitting=fitting, selection=selection, record=export_record)


class ImageOnlyDataset:
    """Frozen list of study units for one split. No mask path, no verify path."""

    def __init__(
        self,
        manifest: dict[str, Any],
        split: str = "train",
        image_size: int = 128,
        seed: int = 42,
    ) -> None:
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema {manifest.get('schema_version')!r}")
        if split not in ("train", "dev", "test"):
            raise ValueError(f"split must be train|dev|test, got {split!r}")
        self.manifest = manifest
        self.split = split
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.dataset = manifest["dataset"]
        self.manifest_id = manifest["manifest_id"]
        self.protocol = manifest["resolved_protocol"]
        # Frozen sampling unit list: order is manifest order, not filesystem order.
        self._records = [r for r in manifest["records"] if r["split"] == split]
        self.unit_ids: list[str] = [r["unit_id"] for r in self._records]
        self.patient_ids: list[str] = sorted({r["patient_id"] for r in self._records})

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TrainingUnit:
        record = self._records[index]
        return _build_training_unit_from_record(
            self.manifest,
            record,
            image_size=self.image_size,
            seed=self.seed,
        )

    def unit_index(self, unit_id: str) -> int:
        return self.unit_ids.index(unit_id)

    def partition_ids(self) -> dict[str, str]:
        return {r["unit_id"]: r["partition_id"] for r in self._records}

    def fingerprint(self) -> str:
        """Hash of the frozen unit list; a trainer stores this in its resume state."""
        payload = json.dumps(
            {
                "manifest_id": self.manifest_id,
                "split": self.split,
                "image_size": self.image_size,
                "seed": self.seed,
                "units": [[r["unit_id"], r["partition_id"]] for r in self._records],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

    def load_verification_unit(self, *args: Any, **kwargs: Any) -> ScoringView:
        raise VerificationAccessError(
            "ImageOnlyDataset has no verification access; call "
            "self_audit_maskfree.data.load_verification_unit with a freeze receipt"
        )


def validate_freeze_receipt(
    freeze_manifest: dict[str, Any] | str | Path,
    *,
    manifest: dict[str, Any],
    partition_id: str,
) -> dict[str, Any]:
    """Validate a complete W7 freeze before releasing sealed observations.

    W7 owns the freeze schema, canonical body identity and every physical
    prediction/checkpoint/nuisance hash.  Calling that complete validator is
    the only physical gate here; the data layer then checks that the validated
    receipt belongs to this data manifest and role partition.  There is no
    arbitrary ``files``-only receipt fallback.
    """
    authoritative_input: dict[str, Any] | Path
    if isinstance(freeze_manifest, (str, Path)):
        path = Path(freeze_manifest)
        if not path.exists():
            raise FreezeReceiptError(f"freeze receipt {path} does not exist")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FreezeReceiptError(f"could not read freeze receipt {path}: {exc}") from exc
        authoritative_input = path
    elif isinstance(freeze_manifest, dict):
        receipt = dict(freeze_manifest)
        # Preserve the caller's object for W7's identity/session cache. The
        # shallow copy above is only for the identity reads below.
        authoritative_input = freeze_manifest
    else:
        raise FreezeReceiptError("freeze receipt must be a mapping or a path to one")

    # W7 is the source of truth for completeness and tamper evidence.  In
    # particular, do not hash receipt.files here: that would create a second,
    # weaker validator which could accidentally release a partial comparison.
    try:
        from ..export import validate_freeze as validate_w7_freeze

        validate_w7_freeze(authoritative_input, require_complete=True)
    except Exception as exc:  # W7 exposes FreezeError; wrap its public failure.
        raise FreezeReceiptError(
            f"W7 complete freeze validation failed: {type(exc).__name__}: {exc}"
        ) from exc

    missing = [key for key in REQUIRED_FREEZE_KEYS if key not in receipt]
    if missing:
        # This is defensive for a mocked/alternate W7 implementation.  The
        # normal path has already proved these fields through validate_freeze.
        raise FreezeReceiptError(f"validated freeze receipt is missing required keys: {missing}")
    completeness = receipt["completeness"]
    if not isinstance(completeness, dict) or completeness.get("complete") is not True:
        raise FreezeReceiptError("W7 freeze does not prove complete method coverage")
    for name in ("units", "checkpoints", "nuisance_files"):
        block = completeness.get(name)
        if not isinstance(block, dict) or block.get("checked") is not True:
            raise FreezeReceiptError(f"W7 freeze has no complete {name} proof")
        if block.get("complete") is not True:
            raise FreezeReceiptError(f"W7 freeze has incomplete {name} proof")

    if receipt["dataset"] != manifest["dataset"]:
        raise FreezeReceiptError(
            f"freeze receipt dataset {receipt['dataset']!r} != manifest {manifest['dataset']!r}"
        )
    if receipt["manifest_id"] != manifest["manifest_id"]:
        raise FreezeReceiptError(
            "freeze receipt was produced against a different data manifest"
        )
    if partition_id not in set(receipt["partition_ids"]):
        raise FreezeReceiptError(
            f"partition {partition_id} is not covered by the freeze receipt; sealed "
            "observations stay sealed"
        )
    files = receipt["files"]
    if not isinstance(files, dict) or not files:
        raise FreezeReceiptError("W7 freeze lists no frozen artifacts")
    return receipt


def load_verification_unit(
    manifest: dict[str, Any],
    unit_id: str,
    freeze_manifest: dict[str, Any] | str | Path,
    *,
    image_size: int = 128,
    seed: int = 42,
) -> ScoringView:
    """Release sealed ``O_verify`` observations, only behind a valid freeze.

    Returns a ``ScoringView(role='verify')`` on the identical partition the unit
    trained under. There is no tuning surface here: the caller gets observations
    once, after everything it could have changed is already hashed.
    """
    record = _get_record(manifest, unit_id)
    height, width = int(record["native_hw"][0]), int(record["native_hw"][1])
    partition = build_partition(height, width, study_id=record["study_id"], seed=seed)
    if partition.partition_id != record["partition_id"]:
        raise FreezeReceiptError(
            f"partition drift for {unit_id}; refusing to unseal verification data"
        )
    validate_freeze_receipt(freeze_manifest, manifest=manifest, partition_id=partition.partition_id)

    built = _build_unit_tensors(record, partition, image_size, include_verify=True)
    view = ScoringView(
        image=built["verify_values"][1:2].clone(),
        support=built["verify_support"].clone(),
        study_id=record["study_id"],
        unit_id=record["unit_id"],
        role="verify",
        partition_id=partition.partition_id,
    )
    view.validate()
    return view


def load_full_input(
    manifest: dict[str, Any],
    unit_id: str,
    *,
    image_size: int = 128,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Whole-image context for deployment export only. Never for training.

    The returned tensor has seen every pixel, including the selection and
    verification observations. A prediction made from it is a conventional
    full-input inference result and carries no claim of having predicted
    withheld intensities.
    """
    record = _get_record(manifest, unit_id)
    stack = _native_stack(record)
    full_support = torch.ones(stack.shape[1:], dtype=torch.bool)
    stats = _fit_statistics(stack, full_support)
    stats["source"] = "full_image_deployment_only"
    normalized = _normalize(stack, stats)
    out_hw = (int(image_size), int(image_size))
    context, _ = masked_resize(normalized, full_support, out_hw)

    deployment_record = {
        "unit_id": record["unit_id"],
        "volume_id": record.get("volume_id", record["unit_id"]),
        "study_id": record["study_id"],
        "patient_id": record["patient_id"],
        "dataset": record["dataset"],
        "split": record["split"],
        "manifest_id": manifest["manifest_id"],
        "protocol": manifest["resolved_protocol"],
        "partition_id": record["partition_id"],
        "image_size": int(image_size),
        "native_hw": list(record["native_hw"]),
        "slice_index": record["slice_index"],
        "frame_index": record["frame_index"],
        "depth_axis": record["depth_axis"],
        "depth": record["depth"],
        "num_slices": record.get("num_slices", record["depth"]),
        "source_format": record["source_format"],
        "native_geometry": record["native_geometry"],
        "native_geometry_reason": record["native_geometry_reason"],
        "spacing_valid": record["spacing_valid"],
        "spacing": record.get("spacing_mm") if record.get("spacing_valid") else None,
        "zooms": record["zooms"],
        "orientation": record["orientation"],
        "export_grid": record["export_grid"],
        "native_grid_export": record["native_grid_export"],
        "native_affine": record["native_affine"],
        "source_hash": record.get("source_hash", record.get("source_fingerprint")),
        "spacing_mm": record.get("spacing_mm") if record.get("spacing_valid") else None,
        "spatial_unit": record.get("spatial_unit"),
        "xyzt_units": record.get("xyzt_units"),
        "inverse_transform": inverse_transform_record(
            native_shape=record.get("native_shape", [int(s) for s in record["shape"][:3]]),
            depth_axis=int(record["depth_axis"]),
            slice_index=int(record["slice_index"]),
            frame_index=record["frame_index"],
            native_hw=(int(record["native_hw"][0]), int(record["native_hw"][1])),
            unit_hw=out_hw,
            native_grid_export=bool(record["native_grid_export"]),
            frame_axis=record.get("frame_axis"),
            full_native_shape=record["shape"] if len(record["shape"]) == 4 else None,
        ),
        "normalization": stats,
        "manifest_kind": "deployment_full_input",
        "deployment_only": True,
        "withheld_intensity_prediction_claim": False,
        "training_use_permitted": False,
        "mask_inputs_used": False,
    }
    return context, deployment_record


def assert_no_mask_tree_read(paths: list[str | Path]) -> None:
    """Explicit assertion helper for firewall tests and operator scripts."""
    for path in paths:
        assert_image_only(path)
