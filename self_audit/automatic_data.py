"""Data loading, complete support volume extraction, and feature caching for automatic v1.

Enforces:
- Complete labeled TRAIN-L frame volumes (all Z slices), never random audit slices.
- Native per-patient/volume feature extraction avoiding batch-padding boundary distortion.
- Padding with explicit boolean geometric validity for controller concatenation.
- Bounded support voxel count and padded cache bytes failing closed if exceeded.
- Conservative controller workspace estimation accounting for features, base logits,
  probabilities, audit weights, targets, candidate responses, probability JVP buffers,
  trial logits/probs, and host peak memory.
- SupportFeatureCache fingerprinting covering model weights, model non-tensor config (window/preprocessing),
  executed cache contents (features, labels, valid_mask), patient_ids, frame_ids,
  probe samples, and immutable deep-copied support records.
- Parity verification for every unique native BCHW shape/chunk with explicit validity masks,
  leftover batch sizes, positive finite tolerance, and try/finally mode preservation.
- Strict target label lineage tracking with zero U/query/VAL/TEST mask access.
- Rejection of malformed frame IDs, aliases, and out-of-bounds indices BEFORE any mask I/O.
- Total native voxel budget validation BEFORE any mask I/O across all support patients.
- Explicit image normalization in [0, 1] and finiteness checks.
- Explicit all-ignored support volume policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import TrainingData
from .model import digest

DEFAULT_MAX_CACHE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GiB
DEFAULT_MAX_CONTROLLER_WORKSPACE_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GiB


def hash_tensor(t: Tensor) -> str:
    """Deterministic hash of tensor shape, dtype, and executed byte content."""
    t_cpu = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t_cpu.shape).encode("ascii"))
    h.update(str(t_cpu.dtype).encode("ascii"))
    h.update(t_cpu.numpy().tobytes())
    return h.hexdigest()


def estimate_padded_cache_bytes(
    b_total: int,
    max_h: int,
    max_w: int,
    d: int = 32,
) -> int:
    """Conservative byte size of padded cache tensors (features float32, labels int64, valid_mask bool).

    features: b_total * d * max_h * max_w * 4 bytes (128 bytes/voxel for d=32)
    labels:   b_total * max_h * max_w * 8 bytes (8 bytes/voxel)
    valid:    b_total * max_h * max_w * 1 byte (1 byte/voxel)
    """
    if b_total < 1 or max_h < 1 or max_w < 1:
        raise ValueError(f"b_total, max_h, max_w must be positive integers, got b_total={b_total}, max_h={max_h}, max_w={max_w}")
    if d < 1:
        raise ValueError(f"d must be positive integer, got d={d}")
    padded_voxels = b_total * max_h * max_w
    return padded_voxels * (d * 4 + 8 + 1)


def estimate_controller_workspace_bytes(
    b_total: int,
    max_h: int,
    max_w: int,
    m: int = 4,
    d: int = 32,
    c: int = 4,
) -> int:
    """Conservative estimate of controller workspace for m directions, d features, c classes.

    Conservatively accounts for:
    - Cached features (float32): b_total * d * max_h * max_w * 4 bytes
    - Labels (int64) and validity mask (bool): b_total * max_h * max_w * 9 bytes
    - Base logits and base softmax probabilities (float32): 2 * b_total * c * max_h * max_w * 4 bytes
    - Audit weights and targets (float64, stacked): 2 * b_total * c * max_h * max_w * 8 bytes
    - Candidate responses dz (list + stacked dz): 2 * m * b_total * c * max_h * max_w * 4 bytes
    - Probability JVP buffers (dp float32, dp float64 projection, and intermediate products):
        (2 * m * c * 4 + m * c * 8) * b_total * max_h * max_w bytes
    - Combined direction response (dz_comb): b_total * c * max_h * max_w * 4 bytes
    - Trial logits, trial probabilities, and actual forward logits: 3 * b_total * c * max_h * max_w * 4 bytes
    - Trial predictions, margins, and reduction temporaries: (c * 4 + 8) * b_total * max_h * max_w bytes
    - Adds 25% headroom for Gram matrices, reduction moments, host peak memory, and allocator fragmentation.
    """
    if b_total < 1 or max_h < 1 or max_w < 1:
        raise ValueError(f"b_total, max_h, max_w must be positive integers, got b_total={b_total}, max_h={max_h}, max_w={max_w}")
    if m < 0 or d < 1 or c < 1:
        raise ValueError(f"m >= 0, d >= 1, c >= 1 required, got m={m}, d={d}, c={c}")

    padded_voxels = b_total * max_h * max_w
    per_voxel_bytes = (
        (d * 4)                       # cached features float32
        + 8                           # labels int64
        + 1                           # valid mask bool
        + (c * 4)                     # base logits float32
        + (c * 4)                     # base probabilities float32
        + (c * 8)                     # audit weights float64
        + (c * 8)                     # audit targets float64
        + (2 * m * c * 4)             # candidate responses dz_list + stacked dz
        + (2 * m * c * 4 + m * c * 8) # probability JVP: dp float32, dp float64 (v), and intermediate product
        + (c * 4)                     # combined direction response dz_comb
        + (3 * c * 4)                 # trial logits, trial probs (p_cached), actual forward logits
        + (c * 4 + 8)                 # trial margins, predictions, host/reduction temporaries
    )
    return int(math.ceil(padded_voxels * per_voxel_bytes * 1.25))


class SupportVolumeRecord:
    """Metadata-complete support volume record for one frame of a patient.

    Enforces complete Z and spatial geometry, tracked content hashes,
    immutable independent snapshots, and comprehensive mutation detection across
    all metadata, source/mask hashes, and tensor data.
    """

    def __init__(
        self,
        patient_id: str,
        frame_id: int,
        images: Tensor,  # [Z, 3, H, W]
        labels: Tensor,  # [Z, H, W]
        image_sha256: str,
        mask_sha256: str,
        *,
        source_path: str = "",
        mask_path: str = "",
        view_metadata: dict[str, Any] | None = None,
    ):
        if not isinstance(patient_id, str) or not patient_id:
            raise ValueError("patient_id must be a non-empty string")
        if not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"frame_id must be a nonnegative integer, got {frame_id}")
        if not isinstance(images, Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be a 4D Tensor [Z, 3, H, W], got shape {getattr(images, 'shape', None)}")
        if not torch.isfinite(images).all():
            raise ValueError("images tensor must contain only finite values")
        if not isinstance(labels, Tensor) or labels.ndim != 3:
            raise ValueError(f"labels must be a 3D Tensor [Z, H, W], got shape {getattr(labels, 'shape', None)}")
        if labels.shape[0] != images.shape[0] or labels.shape[1] != images.shape[2] or labels.shape[2] != images.shape[3]:
            raise ValueError(f"shape mismatch between images {images.shape} and labels {labels.shape}")
        if not isinstance(image_sha256, str) or len(image_sha256) != 64:
            raise ValueError("image_sha256 must be a 64-character hex string")
        if not isinstance(mask_sha256, str) or len(mask_sha256) != 64:
            raise ValueError("mask_sha256 must be a 64-character hex string")

        self.patient_id = patient_id
        self.frame_id = frame_id
        self.images = images
        self.labels = labels
        self.image_sha256 = str(image_sha256)
        self.mask_sha256 = str(mask_sha256)
        self.source_path = str(source_path)
        self.mask_path = str(mask_path)
        self.view_metadata = copy.deepcopy(dict(view_metadata or {}))

        # Track initial tensor version, shape, and executed content hash
        self._initial_patient_id = str(patient_id)
        self._initial_frame_id = int(frame_id)
        self._initial_image_sha256 = str(image_sha256)
        self._initial_mask_sha256 = str(mask_sha256)
        self._initial_source_path = str(source_path)
        self._initial_mask_path = str(mask_path)
        self._initial_view_metadata = copy.deepcopy(self.view_metadata)
        self._initial_shape = list(images.shape)
        self._images_version = images._version
        self._labels_version = labels._version
        self._images_hash = hash_tensor(images)
        self._labels_hash = hash_tensor(labels)

    def __deepcopy__(self, memo: dict[int, Any]) -> SupportVolumeRecord:
        """Create an independent deep copy of the support volume record with cloned tensors."""
        copied = SupportVolumeRecord(
            patient_id=self.patient_id,
            frame_id=self.frame_id,
            images=self.images.clone(),
            labels=self.labels.clone(),
            image_sha256=self.image_sha256,
            mask_sha256=self.mask_sha256,
            source_path=self.source_path,
            mask_path=self.mask_path,
            view_metadata=copy.deepcopy(self.view_metadata),
        )
        memo[id(self)] = copied
        return copied

    def compute_content_fingerprint(self) -> dict[str, Any]:
        """Compute deterministic fingerprint of current actual metadata and tensor content."""
        return {
            "patient_id": self.patient_id,
            "frame_id": self.frame_id,
            "images_hash": hash_tensor(self.images) if isinstance(self.images, Tensor) else "",
            "labels_hash": hash_tensor(self.labels) if isinstance(self.labels, Tensor) else "",
            "image_sha256": str(self.image_sha256),
            "mask_sha256": str(self.mask_sha256),
            "source_path": str(self.source_path),
            "mask_path": str(self.mask_path),
            "shape": list(self.images.shape) if isinstance(self.images, Tensor) else [],
            "view_metadata": copy.deepcopy(self.view_metadata),
        }

    @property
    def content_fingerprint(self) -> dict[str, Any]:
        """Return an independent deep-copied snapshot of the current content fingerprint."""
        return copy.deepcopy(self.compute_content_fingerprint())

    @property
    def num_slices(self) -> int:
        return self.images.shape[0]

    @property
    def height(self) -> int:
        return self.images.shape[2]

    @property
    def width(self) -> int:
        return self.images.shape[3]

    @property
    def total_voxels(self) -> int:
        return self.num_slices * self.height * self.width

    def verify_integrity(self) -> bool:
        """Verify that metadata, source hashes, and executed tensor contents have not been mutated."""
        if not isinstance(self.patient_id, str) or self.patient_id != self._initial_patient_id:
            return False
        if not isinstance(self.frame_id, int) or self.frame_id != self._initial_frame_id:
            return False
        if self.image_sha256 != self._initial_image_sha256:
            return False
        if self.mask_sha256 != self._initial_mask_sha256:
            return False
        if self.source_path != self._initial_source_path:
            return False
        if self.mask_path != self._initial_mask_path:
            return False
        if self.view_metadata != self._initial_view_metadata:
            return False
        if not isinstance(self.images, Tensor) or self.images._version != self._images_version:
            return False
        if list(self.images.shape) != self._initial_shape:
            return False
        if not isinstance(self.labels, Tensor) or self.labels._version != self._labels_version:
            return False
        if hash_tensor(self.images) != self._images_hash:
            return False
        if hash_tensor(self.labels) != self._labels_hash:
            return False
        return True


def extract_complete_support_volumes(
    data: TrainingData,
    support_patient_ids: Sequence[str] | None,
    *,
    max_support_voxels: int = 10_000_000,
    allow_all_ignored: bool = False,
) -> list[SupportVolumeRecord]:
    """Extract complete labeled frame volumes (all Z slices) for declared support patients.

    Never samples a random slice subset. Fails closed if voxels exceed max_support_voxels.
    Zero U/query/VAL/TEST masks are opened.
    Malformed frame IDs, aliases, out-of-bounds indices, and total native voxel budget
    fail BEFORE ANY mask I/O.
    Enforces explicit all-ignored support volume policy (rejects frames where all voxels are 255 unless explicitly allowed).
    Validates finite normalized cine images in [0, 1].
    """
    if not isinstance(max_support_voxels, int) or max_support_voxels <= 0:
        raise ValueError(f"max_support_voxels must be a positive integer, got {max_support_voxels}")

    if support_patient_ids is None:
        support_patient_ids = data.lids

    if not support_patient_ids:
        raise ValueError("support_patient_ids must be non-empty")

    declared_set = set(support_patient_ids)
    if len(declared_set) != len(support_patient_ids):
        raise ValueError("duplicate patient IDs in support declaration")

    # Check for any unlabeled/query/eval overlap
    if declared_set & set(data.uids):
        raise ValueError("support patient overlaps with unlabeled pool")
    eval_ids = set(data.manifest.get("evaluation_ids", []))
    if declared_set & eval_ids:
        raise ValueError("support patient overlaps with evaluation set")

    # Verify all support patients are in the labeled set of TrainingData
    labeled_map = {r["id"]: r for r in data.labeled}
    missing = declared_set - set(labeled_map.keys())
    if missing:
        raise ValueError(f"declared support patients not in labeled training pool: {sorted(missing)}")

    # Check authorized total budget from manifest
    label_budget = data.manifest.get("label_budget")
    if label_budget is not None and len(declared_set) > label_budget:
        raise ValueError(f"declared support patient count {len(declared_set)} exceeds authorized label budget {label_budget}")

    # Pass 1: validate frame IDs, aliases, bounds, image normalization, and native voxels budget
    # across ALL declared patients BEFORE any mask I/O
    total_native_voxels = 0
    patient_cine_map: dict[str, Any] = {}
    patient_frames_map: dict[str, list[int]] = {}

    for pid in sorted(support_patient_ids):
        r = labeled_map[pid]
        if not r.get("masks"):
            raise ValueError(f"patient {pid} has no labeled masks in manifest")

        # Load cine image
        cine = data.array(r["image"], r["image_sha256"])
        if cine.ndim != 4:
            raise ValueError(f"patient {pid} image has ndim {cine.ndim}, expected 4 (T, Z, H, W)")
        T, Z, H, W = cine.shape
        if min(cine.shape) < 1:
            raise ValueError(f"patient {pid} image has empty dimension: {cine.shape}")

        # Point 5: Verify image normalization and finiteness
        if not np.isfinite(cine).all():
            raise ValueError(f"patient {pid} image contains non-finite values (NaN or Inf)")
        if cine.min() < 0.0 or cine.max() > 1.0:
            raise ValueError(f"patient {pid} image values not normalized in [0, 1]: min={cine.min()}, max={cine.max()}")

        patient_cine_map[pid] = cine

        # Validate all frame strings, canonical formatting, bounds, and mask specs
        valid_fids: list[int] = []
        for frame_str, mask_info in r["masks"].items():
            if not isinstance(frame_str, str):
                raise ValueError(f"patient {pid} frame ID {frame_str!r} is not a string")
            try:
                fid = int(frame_str)
            except (ValueError, TypeError):
                raise ValueError(f"patient {pid} has malformed non-integer frame ID: {frame_str!r}")
            if str(fid) != frame_str:
                raise ValueError(f"patient {pid} has non-canonical frame ID alias: {frame_str!r}")
            if fid < 0 or fid >= T:
                raise ValueError(f"patient {pid} frame ID {fid} out of bounds for temporal length T={T}")

            if not isinstance(mask_info, dict) or set(mask_info.keys()) != {"path", "sha256"}:
                raise ValueError(f"patient {pid} frame {fid} invalid mask info specification")
            if not isinstance(mask_info["path"], str) or not mask_info["path"]:
                raise ValueError(f"patient {pid} frame {fid} empty mask path")
            if not isinstance(mask_info["sha256"], str) or len(mask_info["sha256"]) != 64:
                raise ValueError(f"patient {pid} frame {fid} invalid mask sha256")
            valid_fids.append(fid)

        patient_frames_map[pid] = sorted(valid_fids)
        patient_voxels = len(valid_fids) * (Z * H * W)
        total_native_voxels += patient_voxels

    # Enforce native voxels budget BEFORE any mask I/O
    if total_native_voxels > max_support_voxels:
        raise ValueError(
            f"support volume total voxels {total_native_voxels} exceed declared limit {max_support_voxels}; "
            "failing closed before mask I/O"
        )

    # Pass 2: perform mask I/O only after all frame metadata, normalization, and bounds are verified
    volume_records: list[SupportVolumeRecord] = []
    for pid in sorted(support_patient_ids):
        r = labeled_map[pid]
        cine = patient_cine_map[pid]
        T, Z, H, W = cine.shape

        for fid in patient_frames_map[pid]:
            mask_info = r["masks"][str(fid)]
            mask = data.array(mask_info["path"], mask_info["sha256"])
            if mask.shape != (Z, H, W):
                raise ValueError(f"patient {pid} frame {fid} shape mismatch: mask {mask.shape} vs cine {(Z, H, W)}")
            if not np.isin(mask, [0, 1, 2, 3, 255]).all():
                raise ValueError(f"patient {pid} frame {fid} contains invalid class labels outside {{0, 1, 2, 3, 255}}")

            valid_count = int(np.count_nonzero(mask != 255))
            if valid_count == 0 and not allow_all_ignored:
                raise ValueError(
                    f"support frame {fid} for patient {pid} has all-ignored labels (all {mask.size} voxels == 255); "
                    "rejected by explicit all-ignored support policy"
                )

            # Register label read into TrainingData accounting
            data.label_reads.add(pid)

            # Build 3-channel context for all Z slices: [Z, 3, H, W]
            slice_contexts = []
            slice_labels = []
            for z in range(Z):
                ctx = data.context(cine, fid, z)  # [1, 3, H, W]
                slice_contexts.append(ctx)
                lbl = torch.from_numpy(np.array(mask[z], dtype=np.int64, copy=True))[None]  # [1, H, W]
                slice_labels.append(lbl)

            images = torch.cat(slice_contexts, dim=0)  # [Z, 3, H, W]
            labels = torch.cat(slice_labels, dim=0)    # [Z, H, W]

            volume_records.append(SupportVolumeRecord(
                patient_id=pid,
                frame_id=fid,
                images=images,
                labels=labels,
                image_sha256=r["image_sha256"],
                mask_sha256=mask_info["sha256"],
                source_path=str(data.resolve(r["image"])),
                mask_path=str(data.resolve(mask_info["path"])),
                view_metadata={"T": T, "Z": Z, "H": H, "W": W},
            ))

    return volume_records


def compute_model_feature_fingerprint(model: nn.Module) -> str:
    """Deterministic hash of all feature-producing parameters and buffers (excluding head)."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if name.startswith("head."):
            continue
        h.update(name.encode("ascii"))
        h.update(hash_tensor(p).encode("ascii"))
    for name, b in sorted(model.named_buffers()):
        if name.startswith("head."):
            continue
        h.update(name.encode("ascii"))
        h.update(hash_tensor(b).encode("ascii"))
    return h.hexdigest()


def get_model_feature_config(model: nn.Module) -> dict[str, Any]:
    """Capture relevant non-tensor configuration of the model feature extraction path."""
    cfg: dict[str, Any] = {
        "model_class": type(model).__name__,
        "training": bool(getattr(model, "training", False)),
    }
    window = getattr(model, "window", None)
    if window is None:
        cfg["window"] = {"mode": "none"}
    else:
        cfg["window"] = {
            "mode": getattr(window, "mode", type(window).__name__),
            "embed_dim": getattr(window, "embed_dim", None),
            "offset_limit": getattr(window, "offset_limit", None),
            "initial_radius": getattr(window, "initial_radius", None),
            "radius_min": getattr(window, "radius_min", None),
            "radius_max": getattr(window, "radius_max", None),
        }
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        cfg["encoder"] = {
            "class": type(encoder).__name__,
            "channels": list(getattr(encoder, "channels", [])),
        }
    input_adapter = getattr(model, "input_adapter", None)
    if input_adapter is not None:
        cfg["input_adapter"] = {
            "class": type(input_adapter).__name__,
        }
    return cfg


class SupportFeatureCache:
    """Cached features on complete fixed support with strict fingerprinting and parity verification."""

    def __init__(
        self,
        features: Tensor,        # [B_total, 32, max_H, max_W]
        labels: Tensor,          # [B_total, max_H, max_W]
        valid_mask: Tensor,      # [B_total, max_H, max_W] bool
        patient_ids: list[str],
        frame_ids: list[int],
        fingerprint: dict[str, Any],
        probe_samples: list[dict[str, Any]],
        records: list[SupportVolumeRecord] | None = None,
    ):
        self.features = features
        self.labels = labels
        self.valid_mask = valid_mask
        self.patient_ids = list(patient_ids)
        self.frame_ids = list(frame_ids)
        self.fingerprint = copy.deepcopy(fingerprint)
        # Deepcopy and clone probe samples to avoid aliased references
        self.probe_samples = [
            {
                "x": p["x"].clone(),
                "f": p["f"].clone(),
                "valid_mask": p["valid_mask"].clone() if p.get("valid_mask") is not None else None,
                "patient_id": str(p["patient_id"]),
                "frame_id": int(p["frame_id"]),
                "H": int(p["H"]),
                "W": int(p["W"]),
                "B": int(p["B"]) if "B" in p else int(p.get("Z", 1)),
                "Z": int(p.get("Z", p.get("B", 1))),
            }
            for p in probe_samples
        ]
        self.probe_sample = self.probe_samples[0] if self.probe_samples else {}
        self.records = list(records) if records is not None else []

        # Version and executed content hash snapshots
        self._features_version = features._version
        self._labels_version = labels._version
        self._valid_mask_version = valid_mask._version
        self._features_hash = hash_tensor(features)
        self._labels_hash = hash_tensor(labels)
        self._valid_mask_hash = hash_tensor(valid_mask)

        # Snapshots of patient_ids and frame_ids
        self._initial_patient_ids = list(self.patient_ids)
        self._initial_frame_ids = list(self.frame_ids)

        # Snapshot hashes of probe tensors
        self._probe_hashes = [
            (
                hash_tensor(p["x"]),
                hash_tensor(p["f"]),
                hash_tensor(p["valid_mask"]) if p.get("valid_mask") is not None else "",
                p["patient_id"],
                p["frame_id"],
                p["H"],
                p["W"],
                p["B"],
            )
            for p in self.probe_samples
        ]

    def verify_fingerprint(
        self,
        model: nn.Module,
        records: Sequence[SupportVolumeRecord] | None = None,
    ) -> bool:
        """Verify model parameters, non-tensor config, cache contents, metadata, and support records identity."""
        # 1. Model feature parameters and buffers
        current_fp = compute_model_feature_fingerprint(model)
        if current_fp != self.fingerprint.get("feature_parameters_sha256"):
            return False

        # 2. Model non-tensor configuration
        current_cfg = get_model_feature_config(model)
        if current_cfg != self.fingerprint.get("model_config"):
            return False

        # 3. Cache tensor version and executed content checks
        if not isinstance(self.features, Tensor) or self.features._version != self._features_version:
            return False
        if not isinstance(self.labels, Tensor) or self.labels._version != self._labels_version:
            return False
        if not isinstance(self.valid_mask, Tensor) or self.valid_mask._version != self._valid_mask_version:
            return False

        if hash_tensor(self.features) != self._features_hash:
            return False
        if hash_tensor(self.labels) != self._labels_hash:
            return False
        if hash_tensor(self.valid_mask) != self._valid_mask_hash:
            return False

        # 4. Cache metadata checks (patient_ids and frame_ids)
        if not isinstance(self.patient_ids, list) or self.patient_ids != self._initial_patient_ids:
            return False
        if not isinstance(self.frame_ids, list) or self.frame_ids != self._initial_frame_ids:
            return False
        if self.patient_ids != self.fingerprint.get("patient_ids"):
            return False
        if self.frame_ids != self.fingerprint.get("frame_ids"):
            return False

        # 5. Probe samples integrity checks
        if not isinstance(self.probe_samples, list) or len(self.probe_samples) != len(self._probe_hashes):
            return False
        for probe, orig in zip(self.probe_samples, self._probe_hashes):
            if not isinstance(probe.get("x"), Tensor) or hash_tensor(probe["x"]) != orig[0]:
                return False
            if not isinstance(probe.get("f"), Tensor) or hash_tensor(probe["f"]) != orig[1]:
                return False
            vm = probe.get("valid_mask")
            vm_hash = hash_tensor(vm) if isinstance(vm, Tensor) else ""
            if vm_hash != orig[2]:
                return False
            if probe.get("patient_id") != orig[3] or probe.get("frame_id") != orig[4]:
                return False
            if probe.get("H") != orig[5] or probe.get("W") != orig[6] or probe.get("B") != orig[7]:
                return False

        # 6. Support records checks
        check_records = records if records is not None else self.records
        expected_rec_fps = self.fingerprint.get("records_fingerprints", [])
        if len(check_records) != len(expected_rec_fps):
            return False
        for rec, exp_fp in zip(check_records, expected_rec_fps):
            if not isinstance(rec, SupportVolumeRecord):
                return False
            if not rec.verify_integrity():
                return False
            # Recompute current actual metadata fingerprint and verify match against snapshot
            if rec.compute_content_fingerprint() != exp_fp:
                return False

        return True

    def verify_parity(self, model: nn.Module, *, tolerance: float = 1e-5) -> float:
        """Verify that head forward on cached features matches full end-to-end model forward.

        Checks every unique native BCHW shape/chunk sample recorded during extraction.
        Fails closed on nonfinite logits/features (NaN > tolerance is False).
        Preserves model training state and RNG even on failure via try/finally.
        """
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"tolerance must be a finite positive float, got {tolerance}")

        was_training = model.training
        model.eval()
        try:
            dev = next(model.parameters()).device
            max_diff = 0.0
            with torch.no_grad():
                fork_devices = [dev] if dev.type == "cuda" else []
                with torch.random.fork_rng(devices=fork_devices):
                    for probe in self.probe_samples:
                        probe_x = probe["x"].to(dev)
                        probe_f = probe["f"].to(dev)
                        valid_mask = probe.get("valid_mask")
                        if valid_mask is not None:
                            valid_mask = valid_mask.to(dev)

                        # Full forward vs head-on-cache forward
                        direct_logits = model(probe_x, valid_mask=valid_mask)
                        cached_logits = model.head(probe_f)

                        # Nonfinite checks FAIL CLOSED
                        if not torch.isfinite(direct_logits).all():
                            raise RuntimeError(
                                f"Nonfinite direct logits detected in parity check for patient {probe.get('patient_id')}; failing closed"
                            )
                        if not torch.isfinite(cached_logits).all():
                            raise RuntimeError(
                                f"Nonfinite cached logits detected in parity check for patient {probe.get('patient_id')}; failing closed"
                            )

                        diff = (direct_logits - cached_logits).abs()
                        if not torch.isfinite(diff).all():
                            raise RuntimeError("Nonfinite discrepancy between direct and cached logits; failing closed")

                        d = float(diff.max().item())
                        if not np.isfinite(d) or d > tolerance:
                            raise RuntimeError(
                                f"Feature cache parity check failed: max discrepancy {d:.3e} exceeds tolerance {tolerance:.3e}. "
                                "Feature representation or head state is inconsistent."
                            )
                        if d > max_diff:
                            max_diff = d

            return max_diff
        finally:
            if was_training:
                model.train(True)


def compute_support_feature_cache(
    model: nn.Module,
    records: list[SupportVolumeRecord],
    *,
    device: str | torch.device = "cpu",
    max_extract_batch: int = 8,
    tolerance: float = 1e-5,
    max_cache_bytes: int | None = DEFAULT_MAX_CACHE_BYTES,
    max_controller_workspace_bytes: int | None = DEFAULT_MAX_CONTROLLER_WORKSPACE_BYTES,
) -> SupportFeatureCache:
    """Extract features for all support volumes natively, then pad for controller concatenation.

    Native extraction preserves exact spatial context without batch-padding artifacts.
    Enforces padded memory bounds and controller workspace estimate before large allocations.
    Verifies parity across every unique native BCHW shape/chunk sample including leftover batch sizes.
    """
    if not records:
        raise ValueError("records must be non-empty to compute support feature cache")
    if not isinstance(max_extract_batch, int) or max_extract_batch < 1:
        raise ValueError(f"max_extract_batch must be a positive integer, got {max_extract_batch}")
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(f"tolerance must be a finite positive float, got {tolerance}")

    # Geometry planning across all support volumes
    b_total = sum(rec.num_slices for rec in records)
    max_H = max(rec.height for rec in records)
    max_W = max(rec.width for rec in records)

    padded_cache_bytes = estimate_padded_cache_bytes(b_total, max_H, max_W)
    controller_workspace_estimate = estimate_controller_workspace_bytes(b_total, max_H, max_W)

    # Enforce finite declared bounds before any large allocations
    limit_cache_bytes = max_cache_bytes if max_cache_bytes is not None else DEFAULT_MAX_CACHE_BYTES
    limit_workspace_bytes = (
        max_controller_workspace_bytes if max_controller_workspace_bytes is not None else DEFAULT_MAX_CONTROLLER_WORKSPACE_BYTES
    )

    if not isinstance(limit_cache_bytes, (int, float)) or not math.isfinite(limit_cache_bytes) or limit_cache_bytes <= 0:
        raise ValueError(f"max_cache_bytes must be a finite positive number, got {max_cache_bytes}")
    if not isinstance(limit_workspace_bytes, (int, float)) or not math.isfinite(limit_workspace_bytes) or limit_workspace_bytes <= 0:
        raise ValueError(f"max_controller_workspace_bytes must be a finite positive number, got {max_controller_workspace_bytes}")

    if padded_cache_bytes > limit_cache_bytes:
        raise ValueError(
            f"padded cache bytes {padded_cache_bytes} ({padded_cache_bytes / (1024*1024):.2f} MiB) "
            f"exceed declared max_cache_bytes {limit_cache_bytes} ({limit_cache_bytes / (1024*1024):.2f} MiB); "
            "failing closed before allocation"
        )
    if controller_workspace_estimate > limit_workspace_bytes:
        raise ValueError(
            f"controller workspace estimate {controller_workspace_estimate} ({controller_workspace_estimate / (1024*1024):.2f} MiB) "
            f"exceeds declared max_controller_workspace_bytes {limit_workspace_bytes} ({limit_workspace_bytes / (1024*1024):.2f} MiB); "
            "failing closed before allocation"
        )

    model.eval()
    dev = torch.device(device)

    # Fingerprint feature-producing parameters and non-tensor config
    feature_fp = compute_model_feature_fingerprint(model)
    model_cfg = get_model_feature_config(model)

    extracted_frames: list[dict[str, Any]] = []
    probe_samples: list[dict[str, Any]] = []
    seen_bchw_shapes: set[tuple[int, int, int, int]] = set()

    inputs_hasher = hashlib.sha256()

    with torch.no_grad():
        for rec in records:
            if not rec.verify_integrity():
                raise RuntimeError(f"Support record for patient {rec.patient_id} failed integrity verification before extraction")

            inputs_hasher.update(rec.patient_id.encode("utf-8"))
            inputs_hasher.update(str(rec.frame_id).encode("utf-8"))
            inputs_hasher.update(rec.image_sha256.encode("ascii"))
            inputs_hasher.update(rec.mask_sha256.encode("ascii"))
            inputs_hasher.update(rec.source_path.encode("utf-8"))
            inputs_hasher.update(rec.mask_path.encode("utf-8"))

            z_dim = rec.num_slices
            h_dim = rec.height
            w_dim = rec.width

            frame_feats_list = []
            for start in range(0, z_dim, max_extract_batch):
                end = min(start + max_extract_batch, z_dim)
                batch_size = end - start
                batch_x = rec.images[start:end].to(dev)

                f_chunk = model.features(batch_x, valid_mask=None)
                if not torch.isfinite(f_chunk).all():
                    raise RuntimeError(f"Nonfinite features extracted for patient {rec.patient_id}; failing closed")
                frame_feats_list.append(f_chunk.cpu())

                # Record probe sample for EVERY unique BCHW shape including leftover batch sizes
                bchw_key = (batch_size, 3, h_dim, w_dim)
                if bchw_key not in seen_bchw_shapes:
                    seen_bchw_shapes.add(bchw_key)
                    probe_samples.append({
                        "x": rec.images[start:end].clone(),
                        "f": f_chunk.cpu().clone(),
                        "valid_mask": torch.ones((batch_size, h_dim, w_dim), dtype=torch.bool),
                        "patient_id": rec.patient_id,
                        "frame_id": rec.frame_id,
                        "B": batch_size,
                        "H": h_dim,
                        "W": w_dim,
                        "Z": batch_size,
                    })

            frame_features = torch.cat(frame_feats_list, dim=0)

            extracted_frames.append({
                "patient_id": rec.patient_id,
                "frame_id": rec.frame_id,
                "features": frame_features,
                "labels": rec.labels,
                "H": h_dim,
                "W": w_dim,
                "Z": z_dim,
            })

    # Pad features, labels, and validity mask to common [max_H, max_W] grid
    padded_features_list = []
    padded_labels_list = []
    valid_mask_list = []
    patient_ids_list: list[str] = []
    frame_ids_list: list[int] = []

    for item in extracted_frames:
        z = item["Z"]
        h = item["H"]
        w = item["W"]
        f = item["features"]
        l = item["labels"]

        if h == max_H and w == max_W:
            pad_f = f
            pad_l = l
            val = torch.ones((z, max_H, max_W), dtype=torch.bool)
        else:
            pad_f = torch.zeros((z, 32, max_H, max_W), dtype=f.dtype)
            pad_f[:, :, :h, :w] = f
            pad_l = torch.full((z, max_H, max_W), 255, dtype=l.dtype)
            pad_l[:, :h, :w] = l
            val = torch.zeros((z, max_H, max_W), dtype=torch.bool)
            val[:, :h, :w] = True

        padded_features_list.append(pad_f)
        padded_labels_list.append(pad_l)
        valid_mask_list.append(val)
        patient_ids_list.extend([item["patient_id"]] * z)
        frame_ids_list.extend([item["frame_id"]] * z)

    total_features = torch.cat(padded_features_list, dim=0)
    total_labels = torch.cat(padded_labels_list, dim=0)
    total_valid = torch.cat(valid_mask_list, dim=0)

    # Immutable independent snapshot of records fingerprints (deep-copied)
    records_fps = copy.deepcopy([r.compute_content_fingerprint() for r in records])

    fingerprint = {
        "feature_parameters_sha256": feature_fp,
        "model_config": copy.deepcopy(model_cfg),
        "support_inputs_sha256": inputs_hasher.hexdigest(),
        "records_fingerprints": records_fps,
        "patient_ids": list(patient_ids_list),
        "frame_ids": list(frame_ids_list),
        "cache_features_hash": hash_tensor(total_features),
        "cache_labels_hash": hash_tensor(total_labels),
        "cache_valid_mask_hash": hash_tensor(total_valid),
        "total_slices": total_features.shape[0],
        "max_H": max_H,
        "max_W": max_W,
        "padded_cache_bytes": padded_cache_bytes,
        "controller_workspace_estimate_bytes": controller_workspace_estimate,
    }

    cache = SupportFeatureCache(
        features=total_features,
        labels=total_labels,
        valid_mask=total_valid,
        patient_ids=patient_ids_list,
        frame_ids=frame_ids_list,
        fingerprint=fingerprint,
        probe_samples=probe_samples,
        records=records,
    )

    cache.verify_parity(model, tolerance=tolerance)
    return cache
