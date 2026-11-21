"""PG student built as a strict, no-mimic FUSE student subclass."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..candidate_fuse.student import FuseStudent, _shape
from ..candidate_q3d.alignment import GridPhase


PG_STUDENT_CHANNELS = (16, 32, 64, 128, 256)
PG_TEACHER_CHANNELS = (32, 64, 128, 256, 512)
PG_TEACHER_PHASES = (
    ((1, 1, 1), (0.0, 0.0, 0.0)),
    ((1, 2, 2), (0.0, 0.0, 0.0)),
    ((1, 4, 4), (0.0, 0.0, 0.0)),
    ((2, 8, 8), (0.0, 0.0, 0.0)),
    ((2, 16, 16), (0.0, 0.0, 0.0)),
)


class PGStudent(FuseStudent):
    """FUSE's encoder/decoder with PG-only dropout, stage3 and code branch.

    ``include_mimic`` is deliberately hard-disabled.  The inherited fusion
    path and teacher feature keyword arguments remain unchanged so the PG
    engine can share the FUSE batch/evaluation machinery.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        teacher_channels: tuple[int, ...] | list[int] | None = None,
        teacher_phases: tuple[GridPhase, ...] | list[GridPhase] | None = None,
    ) -> None:
        model = config.get("model", config)
        student = model.get("student", {}) if isinstance(model, Mapping) else {}
        channels = tuple(int(v) for v in student.get("channels", PG_STUDENT_CHANNELS))
        if channels != PG_STUDENT_CHANNELS:
            raise ValueError("PGStudent requires channels [16,32,64,128,256]")
        if int(student.get("num_classes", 4)) != 4:
            raise ValueError("PGStudent requires four classes")
        resolved_teacher_channels = tuple(teacher_channels or PG_TEACHER_CHANNELS)
        resolved_teacher_phases = tuple(teacher_phases or PG_TEACHER_PHASES)
        if resolved_teacher_channels != PG_TEACHER_CHANNELS or len(resolved_teacher_phases) != 5:
            raise ValueError("PGStudent requires five MedNeXt-L teacher channels/phases")
        super().__init__(
            config,
            teacher_channels=resolved_teacher_channels,
            teacher_phases=resolved_teacher_phases,
            include_fusion=True,
            include_mimic=False,
        )
        self.code_branch: nn.Module | None = None

    def attach_code_branch(self, branch: nn.Module) -> "PGStudent":
        """Attach the separately-owned train-only CBKD branch."""

        if not isinstance(branch, nn.Module):
            raise TypeError("PGStudent code_branch must be a torch module")
        self.code_branch = branch
        return self

    @staticmethod
    def _skip_dropout(
        value: Tensor, probability: float, training: bool,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if not training or probability <= 0.0:
            return value
        if generator is None:
            return F.dropout3d(value, p=probability, training=True)
        keep = torch.rand(
            (value.shape[0], value.shape[1], 1, 1, 1),
            device=value.device, generator=generator,
        ) >= probability
        return value * keep.to(value.dtype) / (1.0 - probability)

    def _decode_pg(
        self,
        bottleneck: Tensor,
        skips: tuple[Tensor, ...],
        native_hw: tuple[int, int],
        skip_dropout: float,
        dropout_generator: torch.Generator | None = None,
    ) -> Tensor:
        # Every decode input, including the bottleneck and all four skips, is
        # treated as a skip signal.  This is FP-only in the engine by passing
        # zero for labeled/clean paths and a nonzero probability for FP.
        x = self._skip_dropout(bottleneck, skip_dropout, self.training, dropout_generator)
        for reduce, block, skip in zip(
            self.decoder_reduce, self.decoder_blocks, (skips[0], skips[1], skips[2], skips[3])
        ):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = block(reduce(x) + self._skip_dropout(skip, skip_dropout, self.training, dropout_generator))
        return x[..., : native_hw[0], : native_hw[1]]

    def _validate_skip_dropout(self, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("skip_dropout must be a finite scalar in [0,1]")
        value = float(value)
        if not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("skip_dropout must be a finite scalar in [0,1]")
        return value

    def forward(
        self,
        image: Tensor,
        geometry: Any,
        *,
        teacher_features: tuple[Tensor, ...] | list[Tensor] | None = None,
        keep_mask: Tensor | None = None,
        skip_dropout: float = 0.0,
        return_stage3: bool = False,
        return_code: bool = False,
        dropout_generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        probability = self._validate_skip_dropout(skip_dropout)
        if image.ndim != 5 or image.shape[1] != 1 or not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError("PG input must be finite floating [B,1,D,H,W]")
        native = _shape(geometry, image)
        x = image[..., : native[1], : native[2]]
        if torch.any(x < 0) or torch.any(x > 1):
            raise ValueError("PG image must be normalized to [0,1]")
        x, native_hw = self._pad(x)
        any_kept = teacher_features is not None and bool((keep_mask != 0).any()) if keep_mask is not None else False
        if teacher_features is not None:
            if len(teacher_features) != 5 or keep_mask is None:
                raise ValueError("teacher features require five levels and keep_mask")
            if tuple(keep_mask.shape) != (image.shape[0], 1, 1, 1, 1):
                raise ValueError("keep_mask must have shape [B,1,1,1,1]")
            if not torch.isfinite(keep_mask).all() or not torch.all((keep_mask == 0) | (keep_mask == 1)):
                raise ValueError("keep_mask must contain only finite binary 0/1 values")
        elif keep_mask is not None:
            raise ValueError("keep_mask cannot be supplied without teacher features")

        stages: list[Tensor] = []
        gates: list[float | None] = []

        def fuse(index: int, feature: Tensor) -> Tensor:
            if teacher_features is None or not any_kept:
                gates.append(0.0 if teacher_features is None else None)
                return feature
            from ..candidate_fuse.fusion import align_feature

            aligned = align_feature(
                teacher_features[index], native, self.teacher_phases[index], self.phases[index], feature.shape[-3:]
            )
            output, gate = self.fusions[index](feature, aligned, keep_mask)
            gates.append(float(gate.detach().mean().item()))
            return output

        f0 = self.stem[0](x)
        f0 = fuse(0, f0)
        f0 = self.stem[1](f0)
        stages.append(f0)
        s1 = fuse(1, self.stage1[0](f0))
        s1 = self.stage1[1](s1)
        stages.append(s1)
        s2 = fuse(2, self.stage2[0](s1))
        s2 = self.stage2[1](s2)
        stages.append(s2)
        s3 = fuse(3, self.stage3.down(s2))
        s3 = self._epa_blocks(s3, s2.shape[-3:], self.stage3, 8, native)
        stages.append(s3)
        s4 = fuse(4, self.stage4.down(s3))
        s4 = self._epa_blocks(s4, s3.shape[-3:], self.stage4, 16, native)
        stages.append(s4)

        decoded = self._decode_pg(s4, (s3, s2, s1, f0), native_hw, probability, dropout_generator)
        logits = self.logits_head(decoded)[..., : native[1], : native[2]]
        output: dict[str, Any] = {
            "logits": logits,
            "features": tuple(stages),
            "gate_means": gates if teacher_features is not None else [0.0] * 5,
            # PG forbids FUSE mimic; retain the scalar key for engine shape
            # compatibility, but it is permanently a detached zero.
            "mimic_loss": logits.sum() * 0.0,
        }
        if return_stage3:
            output["stage3"] = stages[3]
        if self.training and return_code and self.code_branch is not None:
            valid = self._valid_centers(native, self.phases[3], stages[3].shape[-3:], stages[3].device)
            valid = valid.unsqueeze(0).expand(stages[3].shape[0], -1, -1, -1)
            output["code_branch"] = self.code_branch(stages[3], valid_mask=valid)
        return output


def build_student(
    config: Mapping[str, Any],
    *,
    teacher_channels: tuple[int, ...] | list[int] | None = None,
    teacher_phases: tuple[GridPhase, ...] | list[GridPhase] | None = None,
) -> PGStudent:
    """Engine-facing constructor with the same resolved config contract (``student.arch = "aniso"`` selects :class:`AnisoStudent`)."""

    if str(config.get("student", {}).get("arch", "unetrpp_lite")) == "aniso":
        from .aniso_student import AnisoStudent

        return AnisoStudent(config)
    return PGStudent(config, teacher_channels=teacher_channels, teacher_phases=teacher_phases)


# Historical worker code probes this stable name while the public class remains
# PGStudent; the alias does not alter FuseStudent or any old track.
StudentPG = PGStudent


__all__ = [
    "PGStudent", "StudentPG", "build_student", "PG_STUDENT_CHANNELS", "PG_TEACHER_CHANNELS", "PG_TEACHER_PHASES",
]
