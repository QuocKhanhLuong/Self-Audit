"""Teacher feature alignment and gated fusion for Candidate FUSE."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from ..candidate_q3d.alignment import GridPhase, align_teacher_to_student4


class GatedFusion(nn.Module):
    """Project a detached teacher feature and gate it into a student feature."""

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        if student_channels <= 0 or teacher_channels <= 0:
            raise ValueError("fusion channel counts must be positive")
        self.project = nn.Conv3d(teacher_channels, student_channels, 1)
        self.gate = nn.Conv3d(student_channels * 2, student_channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, student: Tensor, teacher: Tensor, keep_mask: Tensor) -> tuple[Tensor, Tensor]:
        if student.ndim != 5 or teacher.ndim != 5:
            raise ValueError("fusion features must be [B,C,D,H,W]")
        expected = (student.shape[0], 1, 1, 1, 1)
        if tuple(keep_mask.shape) != expected:
            raise ValueError(f"keep_mask must have shape {expected}")
        if keep_mask.device != student.device:
            raise ValueError("keep_mask and student features must share a device")
        projected = self.project(teacher.detach())
        if projected.shape[2:] != student.shape[2:]:
            raise ValueError("aligned teacher and student fusion grids must match")
        gate = torch.sigmoid(self.gate(torch.cat((student, projected), dim=1)))
        return student + keep_mask.to(student.dtype) * gate * projected, gate


def align_feature(teacher: Tensor, native_shape: Sequence[int], teacher_phase: GridPhase,
                  student_phase: GridPhase, target_shape: Sequence[int]) -> Tensor:
    """Sample native cell centers, then extend only the high-side internal pad."""
    aligned = align_teacher_to_student4(teacher, native_shape, teacher_phase, student_phase)
    target = tuple(int(v) for v in target_shape)
    if len(target) != 3 or any(a > b for a, b in zip(aligned.shape[-3:], target)):
        raise ValueError("student grid cannot be smaller than the native phase grid")
    # Student convolution padding is high-side only. Repeat endpoint values on
    # that internal-only extension; native sampled centers remain untouched.
    if tuple(aligned.shape[-3:]) != target:
        aligned = torch.nn.functional.pad(aligned, (0, target[2]-aligned.shape[-1],
            0, target[1]-aligned.shape[-2], 0, target[0]-aligned.shape[-3]), mode="replicate")
    return aligned.detach()


__all__ = ["GatedFusion", "GridPhase", "align_feature", "align_teacher_to_student4"]
