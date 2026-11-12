"""SOFTWARE_ONLY engine-level agreement of feature and target cell centers."""

import torch

from self_audit.candidate_q3d.config import canonical_config
from self_audit.candidate_q3d.engine import (
    _build_models, _features_to_student_grid, _pooled_teacher_targets,
)


def test_engine_feature_and_class_confidence_targets_share_native_centers():
    from self_audit.candidate_q3d.teacher_features import teacher_feature_phase

    config = canonical_config("v1", k=2)
    config["runtime"].update(synthetic=True, device="cpu")
    config["model"]["student"]["stage_channels"] = [4] * 5
    teacher, student, _tokenizer, _ = _build_models(
        config, device=torch.device("cpu"), teacher=None, student=None, tokenizer=None,
    )
    source_phase = teacher_feature_phase(teacher, "encoder_stride8", synthetic=True)
    student_phase = teacher._q3d_student_phase
    native = (17, 49, 49)
    step, offset = source_phase
    axes = [torch.arange((length + stride - 1) // stride, dtype=torch.float64) * stride + origin
            for length, stride, origin in zip(native, step, offset)]
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    feature = (2 * z + y + 3 * x)[None, None].requires_grad_()
    aligned = _features_to_student_grid(
        feature, native, "encoder_stride8", teacher_phase=source_phase, student_phase=student_phase,
    )
    target_axes = [torch.arange((length + stride - 1) // stride, dtype=torch.float64) * stride + origin
                   for length, stride, origin in zip(native, *student_phase)]
    sz, sy, sx = torch.meshgrid(*target_axes, indexing="ij")
    torch.testing.assert_close(aligned[0, 0], 2 * sz + sy + 3 * sx, rtol=0, atol=1e-12)
    assert not aligned.requires_grad

    probabilities = torch.full((1, 4, *native), 1 / 30)
    probabilities[:, 3, ..., :17] = 0.9
    probabilities[:, 0, ..., 17:] = 0.9
    fractions, classes, confidence = _pooled_teacher_targets(
        probabilities.log(), target_hw=(4, 4), student_phase=student_phase,
    )
    # The cell at x=16 covers [8,24] with half endpoint voxels. Its
    # LV mass has 8.5 voxels of 0.9 and 7.5 voxels of 1/30.
    expected_lv = (8.5 * 0.9 + 7.5 / 30) / 16
    torch.testing.assert_close(fractions[0, 3, 8, 1, 1], torch.tensor(expected_lv))
    assert classes[0, 8, 1, 1] == 3
    torch.testing.assert_close(confidence, torch.full_like(confidence, 0.9))
