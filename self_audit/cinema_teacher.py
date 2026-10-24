"""Image-only adapter for the pinned CineMA M&Ms SAX segmentation teacher.

The public M&Ms preprocessing centers a crop using an LV annotation.  That
path is unavailable for unlabeled images, so this adapter instead uses a
full-FOV, image-only intensity domain and records that deviation.  Volumes and
features are in XYZ order; affines map XYZ voxel indices to physical space.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import types
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from torch import nn


SOURCE_COMMIT = "c10daa1d93f0ea28d8b9ad9206b0f673d25805c1"
HF_REVISION = "b1251ee50423bceeca84c080782fc3bc7756dea6"
CONFIG_SHA256 = "aab63af2dc5b1592fd76faa209b37d84b846c7bcf09639a6fb8f7469a31fbea1"
WEIGHTS_SHA256 = "ee11e5392cc89f0a03f7cd2c171e87de0771b839d414835cb59044077679acb8"
TARGET_SPACING_XYZ = (1.0, 1.0, 10.0)
PATCH_SIZE_XYZ = (192, 192, 16)
CLASS_MAP = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}


@dataclass(frozen=True)
class TeacherPrediction:
    """Teacher outputs remapped to native XYZ voxel space."""

    probabilities: np.ndarray  # [4, X, Y, Z], float32
    features: np.ndarray  # [32, X, Y, Z], float32
    valid_mask: np.ndarray  # [X, Y, Z], bool; native centers covered by teacher grid
    geometry: dict[str, Any]
    lineage: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def verify_source_checkout(source_root: str | Path, expected_commit: str = SOURCE_COMMIT) -> str:
    root = Path(source_root).resolve()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify CineMA git checkout at {root}") from exc
    if commit != expected_commit:
        raise ValueError(f"CineMA source commit mismatch: expected {expected_commit}, got {commit}")
    if dirty:
        raise ValueError("CineMA source checkout must be clean for pinned inference")
    return commit


def _import_pinned_get_model(source_root: Path):
    source = str(source_root.resolve())
    expected_module = (source_root / "cinema" / "segmentation" / "convunetr.py").resolve()
    if not expected_module.is_file():
        raise ValueError(f"pinned CineMA source is missing {expected_module}")
    # Python's module cache takes precedence over sys.path. Reject an earlier
    # package import from a different checkout instead of silently using it.
    for name, module in tuple(sys.modules.items()):
        if name == "cinema" or name.startswith("cinema."):
            _verify_module_origin(source_root, name, module)
    if source not in sys.path:
        sys.path.insert(0, source)
    else:
        sys.path.remove(source)
        sys.path.insert(0, source)
    try:
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from cinema.segmentation.convunetr import get_model
    except ImportError as exc:
        raise ImportError(
            "Pinned CineMA loading requires its source dependencies, including OmegaConf and safetensors"
        ) from exc
    module = sys.modules.get("cinema.segmentation.convunetr")
    _verify_module_origin(source_root, "cinema.segmentation.convunetr", module)
    if Path(module.__file__).resolve() != expected_module:
        raise ValueError("imported CineMA ConvUNetR module does not match pinned source path")
    return OmegaConf, load_file, get_model


def _verify_module_origin(source_root: Path, name: str, module: types.ModuleType | None) -> None:
    if module is None:
        return
    root = source_root.resolve()
    module_file = getattr(module, "__file__", None)
    paths = [module_file] if module_file else list(getattr(module, "__path__", ()))
    if not paths:
        raise ValueError(f"cannot verify origin of cached module {name}")
    for path in paths:
        try:
            resolved = Path(path).resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise ValueError(f"cached module {name} is outside pinned CineMA source root")


def _load_pinned_model(source_root: Path, config_path: Path, weights_path: Path) -> nn.Module:
    OmegaConf, load_file, get_model = _import_pinned_get_model(source_root)
    config = OmegaConf.load(str(config_path))
    model = get_model(config)
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    return model


def _disable_grad_checkpointing(model: nn.Module) -> None:
    for module in model.modules():
        setter = getattr(module, "set_grad_ckpt", None)
        if callable(setter):
            setter(False)


def _validate_geometry(volume: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volume.ndim != 3 or any(d <= 0 for d in volume.shape):
        raise ValueError("volume_xyz must be a non-empty 3D XYZ array")
    if not np.isfinite(volume).all():
        raise ValueError("volume_xyz must contain only finite image intensities")
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine_xyz must be a finite 4x4 native-index-to-physical transform")
    if not np.allclose(affine[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError("affine_xyz must be an affine homogeneous transform")
    matrix = affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if np.any(spacing <= 0) or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("affine_xyz must have invertible, non-zero voxel axes")
    direction = matrix / spacing[None, :]
    # The spacing-grid construction retains oblique directions and flips. A
    # sheared grid has no unique orthonormal direction/spacing decomposition.
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("sheared affines are unsupported; orthogonal oblique/flipped axes are supported")
    with np.errstate(over="ignore", invalid="ignore"):
        volume32 = volume.astype(np.float32, copy=False)
    if not np.isfinite(volume32).all():
        raise ValueError("volume_xyz overflows float32 image representation")
    return volume32, affine, spacing


def _resample_to_spacing(
    volume: np.ndarray, affine: np.ndarray, spacing: np.ndarray, target_spacing: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    direction = affine[:3, :3] / spacing[None, :]
    target = np.asarray(target_spacing, dtype=np.float64)
    # Align the leading voxel-cell boundary. The direction matrix is essential
    # here for oblique axes and axis flips.
    target_affine = affine.copy()
    target_affine[:3, :3] = direction * target[None, :]
    target_affine[:3, 3] = affine[:3, 3] + direction @ ((target - spacing) / 2.0)
    target_shape = tuple(max(1, int(math.ceil(n * old / new))) for n, old, new in zip(volume.shape, spacing, target))
    output_to_input = np.linalg.inv(affine) @ target_affine
    resampled = ndimage.affine_transform(
        volume,
        output_to_input[:3, :3],
        offset=output_to_input[:3, 3],
        output_shape=target_shape,
        order=1,
        mode="nearest",
        prefilter=False,
    ).astype(np.float32, copy=False)
    info = {
        "native_shape_xyz": list(volume.shape),
        "native_affine_xyz": affine.tolist(),
        "native_spacing_xyz_mm": spacing.tolist(),
        "resampled_shape_xyz": list(target_shape),
        "resampled_affine_xyz": target_affine.tolist(),
        "target_spacing_xyz_mm": target.tolist(),
        "direction_xyz": direction.tolist(),
        "origin_shift_mm": (target_affine[:3, 3] - affine[:3, 3]).tolist(),
        "origin_shift_rule": "D @ ((target_spacing_xyz - native_spacing_xyz) / 2)",
        "grid_transform_native_to_resampled": (np.linalg.inv(target_affine) @ affine).tolist(),
        "resampling_shape_policy": "ceil(native_shape * native_spacing / target_spacing)",
        "resampling_boundary_policy": "nearest edge extension inside half-voxel cell support; unsupported target centers masked",
    }
    return resampled, target_affine, info


def _normalize_full_fov(volume: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Image-only percentile clip, min/max scale, uint8-floor quantization."""
    domain = volume if valid_mask is None else volume[valid_mask]
    if domain.size == 0:
        raise ValueError("teacher grid has no target centers covered by native physical FOV")
    low, high = np.percentile(domain, (0.95, 99.5))
    clipped = np.clip(volume, low, high)
    lo, hi = float(clipped.min()), float(clipped.max())
    if hi <= lo:
        scaled = np.zeros_like(clipped, dtype=np.float32)
    else:
        scaled = ((clipped - lo) / (hi - lo)).astype(np.float32)
    # Upstream converts scaled intensity to uint8 (floor), then rescales to the
    # unit interval for inference. Domain here is the unlabeled full 3D FOV.
    quantized = np.floor(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)
    normalized = quantized.astype(np.float32) / 255.0
    if valid_mask is not None:
        normalized[~valid_mask] = 0.0
    return normalized


def _axis_support_mask(shape: tuple[int, int, int], index_transform: np.ndarray, input_shape: tuple[int, int, int]):
    """Support mask from separable native/teacher grids, respecting voxel cells."""
    matrix = index_transform[:3, :3]
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-6):
        raise ValueError("grid resampling bridge must preserve separable XYZ axes")
    axes = []
    for dim, length in enumerate(shape):
        coordinates = matrix[dim, dim] * np.arange(length) + index_transform[dim, 3]
        axes.append((coordinates >= -0.5) & (coordinates <= input_shape[dim] - 0.5))
    return axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]


def _tile_starts(length: int, patch: int) -> list[int]:
    if length <= patch:
        return [0]
    stride = max(1, patch // 2)
    starts = list(range(0, length - patch + 1, stride))
    final = length - patch
    if starts[-1] != final:
        starts.append(final)
    return starts


def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5 or logits.shape[1] != 4:
        raise ValueError(f"expected model logits [B,4,X,Y,Z], got {tuple(logits.shape)}")
    return logits.softmax(dim=1)


class _FeatureTap:
    def __init__(self, module: nn.Module):
        self.value: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("decoder_dict['sax'] hook must return a tensor")
        self.value = output
        return None

    def close(self):
        self.handle.remove()


class CineMATeacher:
    """Frozen, pinned CineMA SAX teacher with native-grid probability/features.

    Passing ``model`` is an explicit synthetic-test seam: it skips external
    source/config/weight checks and sets lineage status to ``SYNTHETIC``. It
    cannot be used to create real-teacher provenance.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        *,
        source_root: str | Path | None = None,
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
        max_workspace_bytes: int = 16 * 1024**3,
    ):
        self.device = torch.device(device)
        self._synthetic = model is not None
        if max_workspace_bytes <= 0:
            raise ValueError("max_workspace_bytes must be positive")
        self.max_workspace_bytes = int(max_workspace_bytes)
        if model is None:
            if source_root is None or config_path is None or weights_path is None:
                raise ValueError("real teacher requires source_root, config_path, and weights_path")
            self.source_root = Path(source_root).resolve()
            self.config_path, self.weights_path = Path(config_path), Path(weights_path)
            self.source_commit = verify_source_checkout(self.source_root, SOURCE_COMMIT)
            self.config_sha256 = verify_sha256(self.config_path, CONFIG_SHA256, "CineMA config")
            self.weights_sha256 = verify_sha256(self.weights_path, WEIGHTS_SHA256, "CineMA weights")
            model = _load_pinned_model(self.source_root, self.config_path, self.weights_path)
        else:
            self.source_root = Path(source_root).resolve() if source_root else None
            self.source_commit = None
            self.config_sha256 = None
            self.weights_sha256 = None
        self.model = model.to(self.device)
        _disable_grad_checkpointing(self.model)
        self.model.eval().requires_grad_(False)
        self._validate_model_contract()

    @property
    def synthetic(self) -> bool:
        """Whether this instance uses a test-only injected model."""
        return self._synthetic

    def _validate_model_contract(self) -> None:
        if not hasattr(self.model, "decoder_dict") or "sax" not in self.model.decoder_dict:
            raise ValueError("CineMA model must expose decoder_dict['sax'] for the feature tap")
        if not hasattr(self.model, "pred_head_dict") or "sax" not in self.model.pred_head_dict:
            raise ValueError("CineMA model must expose pred_head_dict['sax']")
        head = self.model.pred_head_dict["sax"]
        if isinstance(head, nn.Conv3d) and (head.in_channels != 32 or head.out_channels != 4):
            raise ValueError("expected SAX pred_head Conv3d(32, 4, kernel_size=1)")

    def predict(
        self,
        volume_xyz: np.ndarray | torch.Tensor,
        affine_xyz: np.ndarray,
        *,
        timepoint: str | int,
    ) -> TeacherPrediction:
        # Keep the external module frozen even if a caller toggled it since
        # construction; inference_mode alone does not disable dropout.
        self.model.eval().requires_grad_(False)
        _disable_grad_checkpointing(self.model)
        raw = volume_xyz.detach().cpu().numpy() if isinstance(volume_xyz, torch.Tensor) else np.asarray(volume_xyz)
        raw, affine, spacing = _validate_geometry(raw, affine_xyz)
        expected_target_shape = tuple(
            max(1, int(math.ceil(n * old / new)))
            for n, old, new in zip(raw.shape, spacing, TARGET_SPACING_XYZ)
        )
        padded_shape = tuple(max(n, p) for n, p in zip(expected_target_shape, PATCH_SIZE_XYZ))
        padded_voxels = math.prod(padded_shape)
        native_voxels = math.prod(raw.shape)
        tile_voxels = math.prod(PATCH_SIZE_XYZ)
        # Preflight before resampling allocates a potentially much larger grid.
        estimated_workspace = (
            padded_voxels * (4 + 4 + 4 + 4 + 4 + 20 + 128)
            + native_voxels * (4 * 4 + 32 * 4 + 1 + 4 + 4 + 4)
            + tile_voxels * (4 + 16 + 16 + 16 + 128)
        )
        if estimated_workspace > self.max_workspace_bytes:
            raise MemoryError(
                f"teacher workspace estimate {estimated_workspace} bytes exceeds configured cap "
                f"{self.max_workspace_bytes} bytes"
            )
        target_image, target_affine, geometry = _resample_to_spacing(
            raw, affine, spacing, TARGET_SPACING_XYZ
        )
        target_shape = target_image.shape
        if target_shape != expected_target_shape:
            raise RuntimeError("resampling shape differs from preflight estimate")
        target_to_native = np.linalg.inv(affine) @ target_affine
        target_support = _axis_support_mask(target_shape, target_to_native, raw.shape)
        normalized = _normalize_full_fov(target_image, target_support)
        shape = normalized.shape
        padded = np.zeros(padded_shape, dtype=np.float32)
        padded[tuple(slice(0, n) for n in shape)] = normalized

        pred_sum = np.zeros((4, *padded_shape), dtype=np.float32)
        feat_sum = np.zeros((32, *padded_shape), dtype=np.float32)
        weight_sum = np.zeros(padded_shape, dtype=np.float32)
        starts_by_axis = [_tile_starts(n, p) for n, p in zip(padded_shape, PATCH_SIZE_XYZ)]
        tile_starts = []
        hook_off_max_delta = 0.0
        head_max_delta = 0.0
        hook_off_checked = False
        tap = _FeatureTap(self.model.decoder_dict["sax"])
        try:
            with torch.inference_mode():
                for x in starts_by_axis[0]:
                    for y in starts_by_axis[1]:
                        for z in starts_by_axis[2]:
                            starts = (x, y, z)
                            tile_starts.append(list(starts))
                            tile_np = padded[
                                x : x + PATCH_SIZE_XYZ[0],
                                y : y + PATCH_SIZE_XYZ[1],
                                z : z + PATCH_SIZE_XYZ[2],
                            ]
                            image = torch.from_numpy(tile_np[None, None]).to(self.device)
                            tap.value = None
                            output = self.model({"sax": image})
                            if not isinstance(output, dict) or "sax" not in output:
                                raise ValueError("CineMA forward must return logits in output['sax']")
                            logits = output["sax"]
                            features = tap.value
                            if features is None:
                                raise RuntimeError("decoder feature hook did not capture a value")
                            if features.shape != (1, 32, *PATCH_SIZE_XYZ):
                                raise ValueError(f"expected decoder features [1,32,192,192,16], got {tuple(features.shape)}")
                            if tuple(logits.shape) != (1, 4, *PATCH_SIZE_XYZ):
                                raise ValueError(f"expected logits [1,4,192,192,16], got {tuple(logits.shape)}")
                            if not torch.isfinite(features).all():
                                raise RuntimeError("CineMA decoder features contain non-finite values")
                            if not torch.isfinite(logits).all():
                                raise RuntimeError("CineMA logits contain non-finite values")
                            head_logits = self.model.pred_head_dict["sax"](features)
                            if not torch.isfinite(head_logits).all():
                                raise RuntimeError("CineMA native prediction head produced non-finite logits")
                            head_delta = float((logits - head_logits).abs().max().item())
                            head_max_delta = max(head_max_delta, head_delta)
                            if not torch.allclose(logits, head_logits, atol=1e-5, rtol=1e-5):
                                raise RuntimeError("decoder hook parity failed: native head(features) != forward logits")
                            if not hook_off_checked:
                                tap.close()
                                no_hook_output = self.model({"sax": image})
                                no_hook_logits = no_hook_output["sax"]
                                if not torch.isfinite(no_hook_logits).all():
                                    raise RuntimeError("hook-off CineMA logits contain non-finite values")
                                hook_off_max_delta = float((logits - no_hook_logits).abs().max().item())
                                if not torch.allclose(logits, no_hook_logits, atol=1e-5, rtol=1e-5):
                                    raise RuntimeError("hook-on/off native logits parity failed")
                                hook_off_checked = True
                                tap = _FeatureTap(self.model.decoder_dict["sax"])
                            probs = _softmax_logits(logits)[0].float().cpu().numpy()
                            feats = features[0].float().cpu().numpy()
                            region = tuple(slice(s, s + p) for s, p in zip(starts, PATCH_SIZE_XYZ))
                            pred_sum[(slice(None), *region)] += probs
                            feat_sum[(slice(None), *region)] += feats
                            weight_sum[region] += 1.0
        finally:
            tap.close()

        np.divide(pred_sum, np.maximum(weight_sum[None], 1.0), out=pred_sum)
        np.divide(feat_sum, np.maximum(weight_sum[None], 1.0), out=feat_sum)
        crop = tuple(slice(0, n) for n in shape)
        target_probs = pred_sum[(slice(None), *crop)]
        target_features = feat_sum[(slice(None), *crop)]
        target_probs *= target_support[None]
        target_features *= target_support[None]

        # Inverse physical-coordinate mapping maps every native voxel center
        # through native->world->teacher-grid, preserving oblique directions.
        native_to_target = np.linalg.inv(target_affine) @ affine
        native_coverage = ndimage.affine_transform(
            target_support,
            native_to_target[:3, :3],
            native_to_target[:3, 3],
            output_shape=raw.shape,
            output=np.float32,
            order=1,
            mode="nearest",
            prefilter=False,
        )
        native_grid_bounds = _axis_support_mask(raw.shape, native_to_target, shape)
        native_valid_mask = (native_coverage > 1e-6) & native_grid_bounds
        if not native_valid_mask.any():
            raise RuntimeError("native volume has no voxel centers covered by resampled teacher grid")
        del pred_sum, feat_sum, weight_sum, padded, normalized, target_image
        native_probs = np.empty((4, *raw.shape), dtype=np.float32)
        for channel in range(4):
            native_probs[channel] = ndimage.affine_transform(
                target_probs[channel], native_to_target[:3, :3], native_to_target[:3, 3],
                output_shape=raw.shape, order=1, mode="nearest", prefilter=False,
            )
        np.maximum(native_probs, 0.0, out=native_probs)
        native_features = np.empty((32, *raw.shape), dtype=np.float32)
        for channel in range(32):
            native_features[channel] = ndimage.affine_transform(
                target_features[channel], native_to_target[:3, :3], native_to_target[:3, 3],
                output_shape=raw.shape, order=1, mode="nearest", prefilter=False,
            )
        np.divide(
            native_probs,
            np.maximum(native_coverage[None], 1e-8),
            out=native_probs,
            where=native_valid_mask[None],
        )
        np.divide(
            native_features,
            np.maximum(native_coverage[None], 1e-8),
            out=native_features,
            where=native_valid_mask[None],
        )
        native_probs *= native_valid_mask[None]
        native_features *= native_valid_mask[None]
        if not np.isfinite(native_probs).all() or not np.isfinite(native_features).all():
            raise RuntimeError("native remapping produced non-finite teacher output")
        probability_mass = native_probs.sum(axis=0, dtype=np.float32)
        np.maximum(probability_mass, 1e-8, out=probability_mass)
        np.divide(native_probs, probability_mass[None], out=native_probs)

        geometry.update({
            "patch_size_xyz": list(PATCH_SIZE_XYZ),
            "overlap_fraction": 0.5,
            "tile_starts_xyz": tile_starts,
            "end_padding_xyz": [p - n for p, n in zip(padded_shape, shape)],
            "inverse_resample": "native index -> physical via native affine -> teacher grid",
            "native_output_shape_xyz": list(raw.shape),
            "native_output_affine_xyz": affine.tolist(),
            "target_grid_support_fraction": float(target_support.mean()),
            "native_grid_coverage_fraction": float(native_valid_mask.mean()),
            "native_grid_valid_voxels": int(native_valid_mask.sum()),
            "native_grid_support_mean": float(native_coverage.mean()),
            "native_grid_partial_support_voxels": int(((native_coverage > 1e-6) & (native_coverage < 1 - 1e-6)).sum()),
            "workspace_estimate_bytes": estimated_workspace,
            "workspace_cap_bytes": self.max_workspace_bytes,
            "workspace_estimate_scope": "arrays and tile outputs; model-internal activations excluded",
        })
        lineage = {
            "status": "SYNTHETIC" if self._synthetic else "PINNED_SOURCE",
            "teacher": "CineMA M&Ms SAX segmentation",
            "source_commit": self.source_commit,
            "hf_revision": HF_REVISION,
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "timepoint": str(timepoint),
            "input_provenance": "not independently classified by adapter",
            "input_volume_sha256": hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest(),
            "class_map": dict(CLASS_MAP),
            "input_domain": "full-FOV image-only resampled 3D volume; no masks or GT crop read",
            "normalization": "p0.95/p99.5 clip; full-volume min/max to [0,1]; floor(uint8*255); rescale [0,1]",
            "normalization_deviation": "image-only full-FOV statistics replace upstream LV-GT-centered crop statistics",
            "quantization": "uint8 floor at 255 levels, rescaled to [0,1]",
            "feature_tap": "decoder_dict['sax'] output before pred_head_dict['sax']",
            "feature_logit_parity_checked": True,
            "hook_on_off_parity_checked": hook_off_checked,
            "hook_on_off_max_abs_delta": hook_off_max_delta,
            "head_feature_max_abs_delta": head_max_delta,
            "inference_mode": True,
            "model_training": self.model.training,
            "all_parameters_frozen": all(not p.requires_grad for p in self.model.parameters()),
            "optimizer_created": False,
            "gradient_checkpointing": False,
            "inference_device": str(self.device),
            "inference_dtype": "float32",
        }
        return TeacherPrediction(native_probs, native_features, native_valid_mask, geometry, lineage)
