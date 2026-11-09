"""Dual-branch teacher for Candidate A.

The active v2 model exposes a four-stage native encoder, but its final head is
not a suitable teacher for Candidate A: it has no separate region/boundary
processing and it does not predict a signed-distance field.  This module keeps
the encoder and input protocol while adding an explicit shared pyramid and two
independent processing paths.

DINOv3 is treated as an encoder initialization only.  It is deliberately not
reported as a pretrained segmentation teacher.  Native checkpoint loading is
provided through an already hash-verified ``native_state`` plus metadata so the
caller can keep manifest/payload verification in the canonical checkpoint
loader.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..losses import supervised_loss
from ..model import CenterDifference, Residual, TinyEncoder, build_backbone


CLASS_NAMES = ("BG", "RV", "MYO", "LV")


def _as_config(modelconfig: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Return a defensive flat model configuration copy."""

    if isinstance(modelconfig, (str, Path)):
        config = json.loads(Path(modelconfig).read_text())
    elif isinstance(modelconfig, Mapping):
        config = copy.deepcopy(dict(modelconfig))
    else:
        raise TypeError("modelconfig must be a mapping or JSON path")
    if isinstance(config.get("model"), Mapping) and "backbone" not in config:
        nested = copy.deepcopy(dict(config["model"]))
        nested.update({k: copy.deepcopy(v) for k, v in config.items() if k != "model"})
        config = nested
    return config


def _canonical_mapping(value: Any, *, field: str) -> tuple[str, ...]:
    """Validate and canonicalize a four-class mapping.

    A different order is rejected rather than silently remapping a native
    checkpoint.  Both ordered names and ``{"0": "BG", ...}`` are accepted.
    """

    if isinstance(value, Mapping):
        try:
            value = [value[str(i)] if str(i) in value else value[i] for i in range(4)]
        except (KeyError, TypeError):
            raise ValueError(f"{field} must contain class IDs 0..3") from None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be a four-class ordered mapping")
    names = tuple(str(v).strip().upper() for v in value)
    if names != CLASS_NAMES:
        raise ValueError(f"{field} must use the fixed order {list(CLASS_NAMES)}")
    return names


def _labels_bhw(labels: Tensor, *, classes: int) -> Tensor:
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W] or [B,1,H,W]")
    if labels.dtype.is_floating_point:
        if not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()):
            raise ValueError("labels must contain finite integer IDs")
        labels = labels.long()
    else:
        labels = labels.long()
    allowed = (labels == 255) | ((labels >= 0) & (labels < classes))
    if not bool(allowed.all()):
        raise ValueError("labels contain an unknown class ID")
    return labels


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


class _ContextBranch(nn.Module):
    """Region path with genuinely broad receptive fields."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            Residual(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.path(x)


class _BoundaryBranch(nn.Module):
    """Boundary path with a separate local/detail stream."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            Residual(channels),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            Residual(channels),
        )

    def forward(self, shared: Tensor, raw: Tensor) -> Tensor:
        local = self.spatial(shared)
        # Keep boundary detail at the native grid.  The shared path is brought
        # up to that grid before fusion, so the detail skip is not first
        # compressed to the 1/4-resolution shared pyramid.
        local = F.interpolate(local, size=raw.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail(raw)
        return self.fuse(local + detail)


class DualBranchTeacher(nn.Module):
    """Shared native encoder followed by region and boundary branches.

    Input is the active v2 2.5-D tensor ``[B,3,H,W]`` (previous, current and
    next slice), normalized by the same ``CenterDifference`` adapter as the
    native student.  Outputs are cropped back to the original spatial shape:

    ``region_logits``: ``[B,4,H,W]``
    ``boundary_logits``: ``[B,4,H,W]``
    ``sdf``: ``[B,4,H,W]`` in ``[-1,1]``
    """

    def __init__(
        self,
        encoder: nn.Module,
        input_adapter: nn.Module,
        *,
        widths: tuple[int, int, int, int],
        num_classes: int = 4,
        class_mapping: tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("Candidate A requires exactly four classes")
        if len(widths) != 4 or any(type(w) is not int or w <= 0 or w % 8 for w in widths):
            raise ValueError("teacher widths must be four positive multiples of 8")
        self.encoder = encoder
        self.input_adapter = input_adapter
        self.widths = tuple(int(w) for w in widths)
        self.num_classes = int(num_classes)
        self.class_mapping = _canonical_mapping(class_mapping, field="class_mapping")
        self.segmentation_ready = False
        self._encoder_frozen = False

        channels = tuple(int(c) for c in getattr(encoder, "channels", ()))
        if len(channels) != 4:
            raise ValueError("teacher requires a four-stage native encoder")
        self.encoder_channels = channels
        self.shared_lateral = nn.ModuleList(
            nn.Conv2d(cin, cout, 1) for cin, cout in zip(channels, self.widths)
        )
        self.shared_up = nn.ModuleList(
            nn.Conv2d(self.widths[i + 1], self.widths[i], 1) for i in range(3)
        )
        self.shared_refine = nn.ModuleList(
            nn.Sequential(Residual(width), Residual(width)) for width in self.widths
        )

        # The branch modules are deliberately disjoint.  Region uses two
        # dilated context convolutions; boundary owns both a local path and a
        # raw-detail path before its own fusion block.
        self.region_branch = _ContextBranch(self.widths[0])
        self.boundary_branch = _BoundaryBranch(self.widths[0])
        self.region_head = nn.Conv2d(self.widths[0], num_classes, 1)
        self.boundary_head = nn.Conv2d(self.widths[0], num_classes, 1)
        self.sdf_head = nn.Conv2d(self.widths[0], num_classes, 1)

    def _validate_input(self, x: Tensor) -> tuple[int, int]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("teacher expects finite normalized Bx3xHxW input")
        _require_finite("teacher input", x)
        if bool((x < 0).any()) or bool((x > 1).any()):
            raise ValueError("teacher expects input values in [0,1]")
        return int(x.shape[-2]), int(x.shape[-1])

    def _shared_features(self, x: Tensor) -> tuple[Tensor, Tensor, tuple[int, int]]:
        height, width = self._validate_input(x)
        pad = (0, (-width) % 32, 0, (-height) % 32)
        normalized = F.pad(self.input_adapter(x), pad, mode="replicate")
        raw = F.pad(x, pad, mode="replicate")
        stages = self.encoder(normalized)
        if len(stages) != 4:
            raise ValueError("teacher encoder must return four stages")
        for i, (feature, channels) in enumerate(zip(stages, self.encoder_channels)):
            expected = (normalized.shape[-2] // (4 * 2**i), normalized.shape[-1] // (4 * 2**i))
            if feature.shape[1] != channels or tuple(feature.shape[-2:]) != expected:
                raise ValueError("incorrect native encoder stride or channel count")

        shared = self.shared_refine[3](self.shared_lateral[3](stages[3]))
        for i in (2, 1, 0):
            skip = self.shared_lateral[i](stages[i])
            shared = self.shared_refine[i](
                skip + self.shared_up[i](F.interpolate(shared, skip.shape[-2:], mode="bilinear", align_corners=False))
            )
        return shared, raw, (height, width)

    def train(self, mode: bool = True) -> "DualBranchTeacher":
        """Preserve warm-up encoder freezing across caller ``train()`` calls."""

        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        shared, raw, original_shape = self._shared_features(x)
        region = self.region_branch(shared)
        boundary = self.boundary_branch(shared, raw)

        height, width = original_shape
        region_logits = F.interpolate(self.region_head(region), raw.shape[-2:], mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(self.boundary_head(boundary), raw.shape[-2:], mode="bilinear", align_corners=False)
        sdf = torch.tanh(F.interpolate(self.sdf_head(boundary), raw.shape[-2:], mode="bilinear", align_corners=False))
        return {
            "region_logits": region_logits[..., :height, :width],
            "boundary_logits": boundary_logits[..., :height, :width],
            "sdf": sdf[..., :height, :width],
        }


def _copy_native_module_state(
    module: nn.Module,
    native_state: Mapping[str, Tensor],
    prefix: str,
    *,
    label: str,
) -> None:
    """Copy one compatible module from a legacy native state dict."""

    source = {key[len(prefix):]: value for key, value in native_state.items() if key.startswith(prefix)}
    expected = module.state_dict()
    if set(source) != set(expected):
        missing = sorted(set(expected) - set(source))
        extra = sorted(set(source) - set(expected))
        raise ValueError(f"native checkpoint {label} keys mismatch; missing={missing}, extra={extra}")
    for key, value in expected.items():
        if not isinstance(source[key], Tensor) or tuple(source[key].shape) != tuple(value.shape):
            raise ValueError(f"native checkpoint {label}.{key} shape mismatch")
    module.load_state_dict(source, strict=True)


def _load_native_initialization(
    teacher: DualBranchTeacher,
    native_state: Mapping[str, Tensor],
    native_metadata: Mapping[str, Any],
    *,
    class_mapping: tuple[str, ...],
    mapping_explicit: bool,
) -> dict[str, Any]:
    """Load shared modules from an already hash-verified legacy native state.

    The native v2 checkpoint has no ontology field in its historical manifest.
    Therefore the caller must provide ``class_mapping`` explicitly for this
    path; the canonical default is intentionally not enough to certify a
    checkpoint with ambiguous metadata.
    """

    if not isinstance(native_state, Mapping) or not native_state:
        raise ValueError("native_state must be a non-empty state dictionary")
    if native_metadata.get("model_class") != "NativeStudent":
        raise ValueError("native checkpoint model_class must be NativeStudent")
    declared_mapping = native_metadata.get("class_mapping")
    if not mapping_explicit and declared_mapping is None:
        raise ValueError("native checkpoint lacks class mapping; pass explicit class_mapping")
    if declared_mapping is not None and _canonical_mapping(declared_mapping, field="native class_mapping") != class_mapping:
        raise ValueError("native checkpoint class mapping conflicts with configuration")
    head_weight = native_state.get("head.weight")
    head_bias = native_state.get("head.bias")
    if not isinstance(head_weight, Tensor) or not isinstance(head_bias, Tensor):
        raise ValueError("native checkpoint must expose the four-class segmentation head")
    if head_weight.ndim != 4 or head_weight.shape[0] != 4 or head_bias.ndim != 1 or head_bias.shape[0] != 4:
        raise ValueError("native checkpoint segmentation head must have exactly four classes")

    _copy_native_module_state(teacher.encoder, native_state, "encoder.", label="encoder")
    _copy_native_module_state(teacher.input_adapter, native_state, "input_adapter.", label="input_adapter")
    _copy_native_module_state(teacher.shared_lateral, native_state, "lateral.", label="lateral")
    _copy_native_module_state(teacher.shared_up, native_state, "up.", label="up")
    _copy_native_module_state(teacher.shared_refine, native_state, "refine.", label="refine")
    return {
        "model_class": "NativeStudent",
        "state_scope": "shared_encoder_and_native_projection",
        "class_mapping": list(class_mapping),
        "ontology_assertion": "explicit_config_required;historical_native_manifest_has_no_mapping",
        "hash_verification": "performed_by_caller_before_build_teacher",
    }


def build_teacher(
    modelconfig: Mapping[str, Any] | str | Path,
    *,
    synthetic: bool = False,
    native_state: Mapping[str, Tensor] | None = None,
    native_metadata: Mapping[str, Any] | None = None,
) -> tuple[DualBranchTeacher, dict[str, Any]]:
    """Build Candidate-A teacher and return ``(teacher, provenance)``.

    ``build_backbone`` performs existing hash-pinned DINOv3 validation.  A
    teacher branches are initialized from the verified backbone and require
    supervised Candidate-A pretraining before joint student updates.
    ``native_state`` is an already verified legacy native state.  It initializes
    shared encoder/projection modules only, while the two Candidate-A branches
    and heads remain new trainable parameters.
    """

    config = _as_config(modelconfig)
    classes = int(config.get("num_classes", 4))
    if classes != 4:
        raise ValueError("Candidate A requires num_classes=4")
    config_mapping = config.get("class_mapping")
    mapping_explicit = config_mapping is not None
    class_mapping = (
        _canonical_mapping(config_mapping, field="class_mapping")
        if config_mapping is not None
        else CLASS_NAMES
    )

    encoder, backbone_meta = build_backbone(config, synthetic=synthetic)
    widths = (8, 16, 24, 32) if tuple(encoder.channels) == TinyEncoder.channels else (96, 128, 192, 256)
    teacher = DualBranchTeacher(
        encoder,
        CenterDifference(),
        widths=widths,
        num_classes=classes,
        class_mapping=class_mapping,
    )

    native_meta: dict[str, Any] | None = None
    if native_state is not None:
        if native_metadata is None:
            raise ValueError("native_metadata is required with native_state")
        native_meta = _load_native_initialization(
            teacher,
            native_state,
            native_metadata,
            class_mapping=class_mapping,
            mapping_explicit=mapping_explicit,
        )

    provenance = {
        "model_class": "DualBranchTeacher",
        "architecture": "shared_native_encoder_region_context_boundary_detail_sdf",
        "num_classes": classes,
        "class_mapping": list(class_mapping),
        "class_mapping_source": "explicit_config" if mapping_explicit else "active_v2_fixed_ontology",
        "backbone_pretrained": bool(backbone_meta.get("pretrained", False)),
        "segmentation_pretrained": False,
        "pretrained_claim": "backbone_initialization_only;segmentation_training_required",
        "backbone": backbone_meta,
        "native_initialization": native_meta,
        "synthetic": bool(synthetic),
    }
    teacher.provenance = copy.deepcopy(provenance)
    return teacher, provenance


def teacher_loss(
    outputs: Mapping[str, Tensor],
    y: Tensor,
    sdf_targets: Tensor,
    lambda_distance: float,
) -> Tensor:
    """Compute CE+Dice for both branches plus normalized SDF L1.

    ``sdf_targets`` is ``[B,4,H,W]`` in the same normalized ``[-1,1]`` units
    as the teacher output.  Geometry and absent-class conventions belong to
    the target builder; this loss only checks finite tensors and ignores label
    ``255`` consistently.
    """

    if set(outputs) != {"region_logits", "boundary_logits", "sdf"}:
        raise ValueError("teacher outputs must contain region_logits, boundary_logits and sdf")
    region_logits = outputs["region_logits"]
    boundary_logits = outputs["boundary_logits"]
    sdf = outputs["sdf"]
    if region_logits.ndim != 4 or boundary_logits.shape != region_logits.shape:
        raise ValueError("teacher segmentation logits must have identical [B,4,H,W] shapes")
    if region_logits.shape[1] != 4 or sdf.shape != region_logits.shape:
        raise ValueError("teacher logits and sdf must have shape [B,4,H,W]")
    labels = _labels_bhw(y, classes=4)
    expected_labels = (region_logits.shape[0], region_logits.shape[2], region_logits.shape[3])
    if tuple(labels.shape) != expected_labels:
        raise ValueError("teacher labels do not match output spatial shape")
    if not isinstance(sdf_targets, Tensor) or sdf_targets.shape != sdf.shape:
        raise ValueError("sdf_targets must have the same shape as teacher sdf")
    _require_finite("region_logits", region_logits)
    _require_finite("boundary_logits", boundary_logits)
    _require_finite("sdf", sdf)
    _require_finite("sdf_targets", sdf_targets)
    if bool((sdf < -1.00001).any()) or bool((sdf > 1.00001).any()):
        raise ValueError("teacher sdf must be normalized to [-1, 1]")
    if bool((sdf_targets < -1.00001).any()) or bool((sdf_targets > 1.00001).any()):
        raise ValueError("sdf_targets must be normalized to [-1, 1]")
    if isinstance(lambda_distance, bool) or not isinstance(lambda_distance, (int, float)):
        raise ValueError("lambda_distance must be a non-negative finite scalar")
    if lambda_distance < 0 or not torch.isfinite(torch.tensor(float(lambda_distance))):
        raise ValueError("lambda_distance must be a non-negative finite scalar")

    region_seg = supervised_loss(region_logits, labels)
    boundary_seg = supervised_loss(boundary_logits, labels)
    valid = (labels != 255).unsqueeze(1).expand_as(sdf)
    if not bool(valid.any()):
        raise ValueError("teacher distance loss has no valid support")
    distance = (sdf - sdf_targets).abs().masked_select(valid).mean()
    total = region_seg + boundary_seg + float(lambda_distance) * distance
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite teacher loss")
    return total


def set_teacher_stage(teacher: DualBranchTeacher, stage: str, freeze: bool = False) -> DualBranchTeacher:
    """Set trainability for pretrain/warmup/joint runtime stages.

    Warm-up freezes only the inherited backbone encoder; input adaptation,
    shared projection and both new branches remain trainable.  Pretrain and joint
    unfreeze the complete teacher unless ``freeze=True`` is used for the
    frozen-teacher comparison.
    """

    if not isinstance(teacher, DualBranchTeacher):
        raise TypeError("teacher must be a DualBranchTeacher")
    aliases = {"teacher_pretrain": "pretrain", "teacher_warmup": "warmup"}
    stage = aliases.get(stage, stage)
    if stage not in {"pretrain", "warmup", "joint"}:
        raise ValueError("stage must be pretrain/teacher_pretrain, warmup/teacher_warmup or joint")
    if freeze:
        teacher._encoder_frozen = True
        teacher.requires_grad_(False)
        teacher.eval()
        return teacher

    teacher._encoder_frozen = False
    teacher.train()
    teacher.requires_grad_(True)
    frozen_encoder = stage == "warmup"
    teacher._encoder_frozen = frozen_encoder
    teacher.encoder.requires_grad_(not frozen_encoder)
    if frozen_encoder:
        teacher.encoder.eval()
    return teacher


__all__ = [
    "CLASS_NAMES",
    "DualBranchTeacher",
    "build_teacher",
    "set_teacher_stage",
    "teacher_loss",
]
