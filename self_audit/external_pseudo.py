"""Hash-pinned, image-only TRAIN pseudo probabilities for the A1 control."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F

from .model import digest


class ExternalPseudoCache:
    """Read native `[T,Z,4,Y,X]` probabilities without loading a teacher.

    Cache records are validated against the TRAIN image manifest before any
    sample is requested. Per-sample crop geometry is copied from TrainingData,
    so targets never resize or choose their own crop.
    """

    def __init__(self, manifest_path, expected_sha256, training_data):
        self.path = Path(manifest_path).resolve()
        if digest(self.path) != expected_sha256:
            raise ValueError("external pseudo cache manifest SHA-256 mismatch")
        cache = json.loads(self.path.read_text())
        allowed = {"schema", "role", "synthetic", "class_map", "probability_layout", "evaluation_ids",
                   "records", "bank_sha256", "provenance"}
        if (cache.get("schema") != "self_audit.external_pseudo.v1"
                or cache.get("role") != "training"
                or cache.get("synthetic") is not bool(training_data.manifest["synthetic"])
                or cache.get("class_map") != ["BG", "RV", "MYO", "LV"]
                or cache.get("probability_layout") != "TZCYX"
                or not {"schema", "role", "synthetic", "class_map", "probability_layout", "evaluation_ids", "records"}.issubset(cache)
                or set(cache) - allowed):
            raise ValueError("invalid external pseudo cache identity/layout")
        evaluation_ids = cache.get("evaluation_ids")
        if not isinstance(evaluation_ids, list) or set(evaluation_ids) & set(training_data.lids + training_data.uids):
            raise ValueError("external pseudo cache evaluation IDs overlap TRAIN")
        train = {r["id"]: r for r in training_data.labeled + training_data.unlabeled}
        records = cache.get("records")
        if not isinstance(records, list):
            raise ValueError("external pseudo cache records must be a list")
        self.arrays = {}
        self.supports = {}
        seen = set()
        geometry_by_patient = cache.get("provenance", {}).get("native_geometry_by_patient", {})
        if not isinstance(geometry_by_patient, dict):
            raise ValueError("external pseudo cache must declare per-patient native geometry")
        for record in records:
            required = {"id", "image_sha256", "probabilities", "probabilities_sha256", "shape"}
            if not isinstance(record, dict) or set(record) != required:
                raise ValueError("invalid external pseudo cache record")
            pid = record["id"]
            if pid in seen:
                raise ValueError("duplicate patient in external pseudo cache")
            seen.add(pid)
            if pid not in train or record["image_sha256"] != train[pid]["image_sha256"]:
                raise ValueError("external pseudo cache must match a TRAIN image")
            path = Path(record["probabilities"])
            path = path if path.is_absolute() else self.path.parent / path
            if digest(path) != record["probabilities_sha256"]:
                raise ValueError("external pseudo probability payload hash mismatch")
            probabilities = np.load(path, mmap_mode="r", allow_pickle=False)
            image = training_data.array(train[pid]["image"], train[pid]["image_sha256"])
            expected_shape = (image.shape[0], image.shape[1], 4, image.shape[2], image.shape[3])
            if (probabilities.dtype != np.float32 or probabilities.shape != expected_shape
                    or record["shape"] != list(expected_shape)):
                raise ValueError("external pseudo probabilities must match native TRAIN T,Z,4,Y,X")
            geometry = geometry_by_patient.get(pid)
            geometry_keys = {"shape_tzyx", "affine_xyz_to_world_mm", "probability_shape_tzcyx", "axis_mapping"}
            if (not isinstance(geometry, dict) or not geometry_keys.issubset(geometry)
                    or set(geometry) != geometry_keys | {"valid_support"}):
                raise ValueError("external pseudo cache is missing explicit per-patient affine geometry")
            affine = np.asarray(geometry["affine_xyz_to_world_mm"], dtype=np.float64)
            if (geometry["shape_tzyx"] != list(image.shape)
                    or geometry["probability_shape_tzcyx"] != list(expected_shape)
                    or geometry["axis_mapping"] != "native_NIfTI_XYZT_to_prepared_TZYX=transpose(3,2,1,0); probability_TZCYX_plane=class,Y,X from the same native affine XYZ grid"
                    or affine.shape != (4, 4) or not np.isfinite(affine).all()
                    or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12):
                raise ValueError("external pseudo native affine/axis mapping is invalid")
            support_info = geometry.get("valid_support")
            if (not isinstance(support_info, dict)
                    or set(support_info) != {"path", "sha256", "shape_tzyx"}
                    or support_info["shape_tzyx"] != list(image.shape)):
                raise ValueError("invalid geometric support metadata")
            support_path = Path(support_info["path"])
            support_path = support_path if support_path.is_absolute() else self.path.parent / support_path
            if digest(support_path) != support_info["sha256"]:
                raise ValueError("external pseudo geometric support hash mismatch")
            support_array = np.load(support_path, mmap_mode="r", allow_pickle=False)
            if support_array.dtype != np.bool_ or support_array.shape != image.shape:
                raise ValueError("geometric support must be bool native TZYX")
            support = support_array
            # Validate all values once, without changing/normalizing teacher output.
            for start in range(0, expected_shape[0] * expected_shape[1], 16):
                flat = probabilities.reshape(-1, 4, *expected_shape[-2:])
                flat_support = support.reshape(-1, *expected_shape[-2:])
                block = np.asarray(flat[start:start + 16])
                if not np.isfinite(block).all() or np.any(block < 0):
                    raise ValueError("external pseudo probabilities must be finite and nonnegative")
                sums = block.sum(axis=1)
                valid_block = np.asarray(flat_support[start:start + 16])
                if valid_block.any() and not np.allclose(sums[valid_block], 1.0, rtol=0, atol=2e-4):
                    raise ValueError("external pseudo class probabilities must sum to one on support")
                if (~valid_block).any() and not np.all(block.transpose(0, 2, 3, 1)[~valid_block] == 0):
                    raise ValueError("external pseudo probabilities must be all-zero outside support")
            self.arrays[pid] = probabilities
            self.supports[pid] = support
        if not set(training_data.uids).issubset(seen):
            raise ValueError("external pseudo cache must cover every unlabeled TRAIN patient")

    def sample_batch(self, metadata, *, device):
        samples = metadata.get("samples", [metadata])
        targets, supports = [], []
        for sample in samples:
            pid, frame, z = sample["patient"], int(sample["frame"]), int(sample["slice"])
            if pid not in self.arrays:
                raise ValueError("external pseudo sample is absent from cache")
            plane = np.array(self.arrays[pid][frame, z], dtype=np.float32, copy=True)
            target = torch.from_numpy(plane)
            support = torch.from_numpy(np.array(self.supports[pid][frame, z], dtype=bool, copy=True))
            crop = sample.get("crop")
            if crop is not None:
                h, w = crop["original_hw"]
                if target.shape[-2:] != (h, w):
                    raise ValueError("cached probability plane and crop source geometry differ")
                left, right, top, bottom = crop["pad_lrtb"]
                if left or right or top or bottom:
                    target = F.pad(target[None], (left, right, top, bottom), mode="constant", value=0)[0]
                    support = F.pad(support[None, None].float(), (left, right, top, bottom), value=0)[0, 0].bool()
                y0, x0, size = int(crop["top"]), int(crop["left"]), int(crop["size"])
                target = target[..., y0:y0 + size, x0:x0 + size]
                support = support[y0:y0 + size, x0:x0 + size]
                if target.shape[-2:] != (size, size):
                    raise ValueError("cached probability crop falls outside native/padded support")
            targets.append(target)
            supports.append(support)
        return torch.stack(targets).to(device=device), torch.stack(supports).to(device=device)
