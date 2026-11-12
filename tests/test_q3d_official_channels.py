"""Pin Q-3D feature widths to the official MedNeXt-v1 channel contract.

The fast contract tests need no clone, weights, or network.  The optional
integration test runs only when ``Q3D_OFFICIAL_MEDNEXT_SOURCE`` names an
already available, clean checkout at the pinned upstream commit; it uses
random initialization and a small CPU input as software evidence only.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from self_audit.candidate_a3d.teacher import build_teacher3d, set_teacher_stage
from self_audit.candidate_q3d.alignment import align_teacher_to_student4
from self_audit.candidate_q3d.config import (
    UPSTREAM_MEDNEXT_COMMIT,
    canonical_config,
    load_config,
)
from self_audit.candidate_q3d.engine import (
    _pooled_teacher_targets,
    _student_config,
)
from self_audit.candidate_q3d.losses import feature_kd_loss
from self_audit.candidate_q3d.student import CandidateQ3DStudent
from self_audit.candidate_q3d.teacher_features import (
    forward_teacher_with_features,
    teacher_feature_source,
)


# Pinned from official MedNeXt v1 L k3 source at UPSTREAM_MEDNEXT_COMMIT:
# stride4=down_block_1/decoder stride4 (128), stride8=dec_block_3 (256),
# and stride16=bottleneck (512). Keep independent of config.py's lookup.
OFFICIAL_MEDNEXT_CHANNELS = {4: 128, 8: 256, 16: 512}


def test_portable_q3d_recipes_and_c1_head_match_official_channel_contract() -> None:
    config_dir = Path("configs/candidate_q3d")
    for recipe in sorted(config_dir.glob("*.json")):
        config = load_config(recipe)
        teacher = config["model"]["teacher"]
        if teacher["architecture"] == "mednext_l_k3":
            stride = teacher["feature_stride"]
            assert teacher["feature_channels"] == OFFICIAL_MEDNEXT_CHANNELS[stride]

    canonical_sources = {4: ("v1", 4), 8: ("c1", None), 16: ("s1", None)}
    for stride, channels in OFFICIAL_MEDNEXT_CHANNELS.items():
        arm, declared_stride = canonical_sources[stride]
        config = canonical_config(arm, source_stride=declared_stride)
        assert config["model"]["teacher"]["feature_stride"] == stride
        assert config["model"]["teacher"]["feature_channels"] == channels

    c1 = load_config(config_dir / "c1.json")
    assert c1["runtime"]["synthetic"] is False
    assert c1["model"]["teacher"]["feature_stride"] == 8
    student_config = _student_config(c1)
    student = CandidateQ3DStudent(student_config)
    assert student_config["teacher"]["feature_channels"] == OFFICIAL_MEDNEXT_CHANNELS[8]
    assert student.continuous_feature_head is not None
    assert student.continuous_feature_head.out_channels == OFFICIAL_MEDNEXT_CHANNELS[8]


def test_c1_feature_head_and_loss_match_bounded_official_source_forward() -> None:
    source_value = os.environ.get("Q3D_OFFICIAL_MEDNEXT_SOURCE")
    if not source_value:
        pytest.skip("set Q3D_OFFICIAL_MEDNEXT_SOURCE to an existing pinned official checkout")
    source_repo = Path(source_value).expanduser().resolve()
    if not source_repo.is_dir():
        pytest.fail(f"Q3D_OFFICIAL_MEDNEXT_SOURCE does not exist: {source_repo}")

    native_shape = (6, 17, 19)
    geometry = {
        "shape": native_shape,
        "spacing_zyx": (1.0, 1.0, 1.0),
        "affine_zyx": torch.eye(4, dtype=torch.float64),
        "context": "full_volume",
    }
    config = canonical_config("c1")
    assert config["runtime"]["synthetic"] is False
    config["model"]["teacher"]["source_repo"] = str(source_repo)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            teacher, provenance = build_teacher3d(
                {
                    "backbone": "mednext_l_k3",
                    "input_channels": 1,
                    "source_repo": str(source_repo),
                    "source_commit": UPSTREAM_MEDNEXT_COMMIT,
                    "weights": None,
                    "weights_sha256": None,
                },
                synthetic=False,
            )
            assert provenance["quality"] == "UNTRAINED_RANDOM_INIT"
            assert provenance["synthetic"] is False
            set_teacher_stage(teacher, "joint", freeze=True)

            source = teacher_feature_source(teacher, "stride8")
            assert source["channels"] == OFFICIAL_MEDNEXT_CHANNELS[8]
            assert source["module_name"] in {"dec_block_3", "dec_block_3.7"}

            student_config = _student_config(config, teacher)
            student = CandidateQ3DStudent(student_config)
            assert student_config["teacher"]["feature_channels"] == source["channels"]
            assert student.continuous_feature_head is not None
            assert student.continuous_feature_head.out_channels == source["channels"]

            images = torch.rand((1, 1, *native_shape), dtype=torch.float32)
            teacher_outputs = forward_teacher_with_features(
                teacher, images, geometry=geometry, source="stride8",
            )
            assert teacher_outputs["feature"].shape == (1, 256, 4, 4, 4)
            aligned = align_teacher_to_student4(
                teacher_outputs["feature"], native_shape, teacher_stride=8,
            )
            assert aligned.shape == (1, 256, 6, 2, 2)
            student.train()
            student_outputs = student(images, geometry=geometry)
            student_feature = student_outputs["feature_kd"]
            assert student_feature is not None
            assert student_feature.shape == aligned.shape

            _probs, classes, confidence = _pooled_teacher_targets(
                teacher_outputs["region_logits"],
                target_hw=tuple(student_feature.shape[-2:]),
            )
            loss, _stats = feature_kd_loss(aligned, student_feature, classes, confidence)
            assert torch.isfinite(loss)
            loss.backward()
            student_gradients = [parameter.grad for parameter in student.parameters()
                                 if parameter.requires_grad and parameter.grad is not None]
            assert student_gradients
            assert all(torch.isfinite(gradient).all() for gradient in student_gradients)
            assert all(parameter.grad is None for parameter in teacher.parameters())
    finally:
        torch.set_num_threads(previous_threads)
