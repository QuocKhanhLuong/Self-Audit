"""FUSE's supervised teacher wrapper around the strict Q3D ResEnc teacher."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn.modules.utils import _triple

from self_audit.candidate_q3d.pretrained import build_q3d_teacher
from self_audit.candidate_q3d.losses import q3d_teacher_loss
from self_audit.candidate_q3d.engine import _sdf_targets, _schedule_lr


class FuseTeacher(nn.Module):
    """Expose Q3D outputs and the first five live encoder features in one pass."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Keep this exact type: downstream artifacts and checks rely on it.
        from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher

        if not isinstance(model, Q3DResEncTeacher):
            raise TypeError("FUSE teacher model must be Q3DResEncTeacher")
        self.model = model
        self.feature_channels = tuple(int(c) for c in model.encoder_channels[:5])
        if len(self.feature_channels) != 5:
            raise ValueError("Q3D teacher must expose at least five encoder stages")
        self.feature_phases = self._live_feature_phases()
        self._joint_frozen = False

    def _live_feature_phases(self):
        encoder = self.model.encoder
        stages = getattr(encoder, "stages", None)
        strides = getattr(encoder, "strides", None)
        if stages is None or strides is None or len(stages) != 6 or len(strides) != 6:
            raise ValueError("ResEnc teacher must expose six live stages and strides")
        result = []
        cumulative = [1, 1, 1]
        offset = [0.0, 0.0, 0.0]
        stem = getattr(encoder, "stem", None)
        stem_convs = [module for module in stem.modules() if isinstance(module, nn.Conv3d)]
        if len(stem_convs) != 1 or tuple(int(v) for v in stem_convs[0].stride) != (1, 1, 1):
            raise ValueError("ResEnc stem must expose one unit-stride live Conv3d")
        self._accumulate_conv_phase(stem_convs[0], cumulative, offset)
        for i, (stage, stride) in enumerate(zip(stages, strides)):
            # The first residual block's conv1 is the live main path; skip
            # projections do not define the residual block's output phase.
            blocks = getattr(stage, "blocks", None)
            first = blocks[0] if blocks is not None and len(blocks) else stage
            path = getattr(first, "conv1", None)
            conv = getattr(path, "conv", path)
            if not isinstance(conv, nn.Conv3d):
                raise ValueError(f"ResEnc stage {i} has no live main-path Conv3d")
            live_stride = tuple(int(v) for v in conv.stride)
            declared = tuple(int(v) for v in stride)
            if live_stride != declared:
                raise ValueError(f"ResEnc stage {i} live convolution stride disagrees with declaration")
            self._accumulate_conv_phase(conv, cumulative, offset)
            if i < 5:
                result.append((tuple(cumulative), tuple(offset)))
        expected = ((1, 1, 1), (1, 2, 2), (1, 4, 4), (2, 8, 8), (2, 16, 16))
        if tuple(item[0] for item in result) != expected:
            raise ValueError("FUSE teacher feature phases do not match the locked ResEnc schedule")
        locked = self.model.phase_offsets
        # The upstream Q3D recipe has explicit measured/locked values at
        # stride8/16. Validate them against live convolution derivation.
        for index, key in ((3, "encoder_stride8"), (4, "encoder_stride16")):
            locked_offset = tuple(float(v) for v in locked.get(key, (0.0, 0.0, 0.0)))
            if tuple(result[index][1]) != locked_offset:
                raise ValueError(f"live ResEnc phase at stage {index} disagrees with Q3D {key}")
        return tuple(result)

    @staticmethod
    def _accumulate_conv_phase(conv: nn.Conv3d, step: list[int], offset: list[float]) -> None:
        if isinstance(conv.padding, str):
            raise ValueError("string Conv3d padding is unsupported for phase derivation")
        kernel, padding, dilation = _triple(conv.kernel_size), _triple(conv.padding), _triple(conv.dilation)
        stride = tuple(int(v) for v in conv.stride)
        for axis, value in enumerate(stride):
            offset[axis] += ((kernel[axis] - 1) * dilation[axis] / 2.0 - padding[axis]) * step[axis]
            step[axis] *= value

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load complete FUSE teacher weights through Q3D's alias-safe loader."""
        if not isinstance(state_dict, Mapping):
            raise TypeError("FUSE teacher state_dict must be a mapping")
        prefix = "model."
        if any(not isinstance(key, str) or not key.startswith(prefix) for key in state_dict):
            raise ValueError("FUSE teacher checkpoint keys must all use the model. prefix")
        model_state = {key[len(prefix):]: value for key, value in state_dict.items()}
        # Q3D's model loader checks shared encoder aliases before delegating
        # strict key/shape handling to nn.Module.
        return self.model.load_state_dict(model_state, strict=strict, assign=assign)

    def forward(self, image: Tensor, geometry: Mapping[str, Any], *, return_features: bool = True):
        captured: list[Tensor] = []
        handles = []
        if return_features:
            for i, stage in enumerate(self.model.encoder.stages[:5]):
                def save(_module, _inputs, output, index=i):
                    if not isinstance(output, Tensor):
                        raise TypeError(f"ResEnc stage {index} output must be a tensor")
                    captured.append(output)
                handles.append(stage.register_forward_hook(save))
        try:
            outputs = self.model(image, geometry=geometry, return_features=False)
        finally:
            for handle in handles:
                handle.remove()
        if return_features:
            if len(captured) != 5:
                raise RuntimeError("ResEnc feature hooks did not capture exactly five stages")
            outputs = dict(outputs)
            outputs["features"] = tuple(captured)
        return outputs

    def freeze_for_joint(self) -> "FuseTeacher":
        for parameter in self.parameters():
            parameter.grad = None
        self.requires_grad_(False)
        self._joint_frozen = True
        super().train(False)
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        if self._joint_frozen and mode:
            # Parent engine .train() calls must never thaw a frozen T snapshot.
            super().train(False)
            self.model.eval()
            return self
        super().train(mode)
        return self


def build_teacher(config: Mapping[str, Any], *, synthetic: bool = False) -> tuple[FuseTeacher, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("FUSE teacher config must be a mapping")
    seed = config.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("FUSE teacher config requires a nonnegative integer root seed")
    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping) or not isinstance(model_cfg.get("teacher"), Mapping):
        raise ValueError("FUSE config requires model.teacher mapping")
    q3d_config = {"seed": seed, "model": {"teacher": model_cfg["teacher"]}}
    model, provenance = build_q3d_teacher(q3d_config, synthetic=synthetic)
    return FuseTeacher(model), provenance


def teacher_parameter_groups(teacher: FuseTeacher, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(teacher, FuseTeacher):
        raise TypeError("teacher must be FuseTeacher")
    training = config.get("training", {})
    encoder_lr = float(training.get("teacher_lr_encoder", 1e-4))
    head_lr = float(training.get("teacher_lr_head", 1e-3))
    if not all(math.isfinite(x) and x > 0 for x in (encoder_lr, head_lr)):
        raise ValueError("teacher learning rates must be finite and positive")
    encoder_ids = {id(p) for p in teacher.model.encoder.parameters()}
    seen: set[int] = set()
    enc, heads = [], []
    for p in teacher.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (enc if id(p) in encoder_ids else heads).append(p)
    return [
        {"name": "encoder", "params": enc, "lr": encoder_lr, "base_lr": encoder_lr},
        {"name": "heads", "params": heads, "lr": head_lr, "base_lr": head_lr},
    ]



class _SingleGroupOptimizer:
    def __init__(self, group):
        self.param_groups = [group]

def configure_teacher_step(teacher: FuseTeacher, optimizer: torch.optim.Optimizer, update: int,
                           total_updates: int, config: Mapping[str, Any]) -> dict[str, Any]:
    if total_updates <= 0 or update < 0 or update >= total_updates:
        raise ValueError("teacher update must be within a positive update budget")
    t_a = max(1, round(total_updates * 0.05))
    t_b = total_updates - t_a
    phase = "T-a" if update < t_a else "T-b"
    training = config.get("training", {})
    warmup_fraction = float(training.get("warmup_fraction", 0.05))
    for group in optimizer.param_groups:
        is_encoder = group.get("name") == "encoder"
        for parameter in group["params"]:
            parameter.requires_grad_(phase == "T-b" if is_encoder else True)
        if phase == "T-a":
            lr = float(training.get("teacher_lr_head", group.get("base_lr", 1e-3)))
        else:
            base = float(training.get("teacher_lr_encoder" if is_encoder else "teacher_lr_head", group.get("base_lr", 1e-3)))
            _schedule_lr(
                _SingleGroupOptimizer(group), step=update - t_a,
                total=max(1, t_b), base_lrs=[base],
                warmup_fraction=warmup_fraction,
            )
            lr = float(group["lr"])
        group["lr"] = lr
    teacher.train(True)
    teacher.model.encoder.train(phase == "T-b")
    return {"phase": phase, "update": update, "total_updates": total_updates,
            "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups}}


def teacher_loss(outputs: Mapping[str, Tensor], target: Tensor, geometry: Mapping[str, Any]) -> Tensor:
    """Apply the existing Q3D supervised region/boundary/SDF teacher objective."""
    if not isinstance(outputs, Mapping) or not isinstance(target, Tensor):
        raise TypeError("teacher_loss requires outputs mapping and target tensor")
    labels = target[:, 0] if target.ndim == 5 and target.shape[1] == 1 else target
    sdf_target = _sdf_targets(labels, geometry, radius_mm=5.0).to(device=labels.device)
    loss, _ = q3d_teacher_loss(outputs, labels, sdf_target, lambda_distance=1.0,
                               radius_mm=5.0, near_mm=2.5, alpha=0.5)
    return loss


__all__ = ["FuseTeacher", "build_teacher", "teacher_parameter_groups",
           "configure_teacher_step", "teacher_loss"]
