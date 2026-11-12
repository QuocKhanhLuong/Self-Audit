from __future__ import annotations

import math

import pytest
import torch

from self_audit.candidate_q3d.alignment import align_teacher_to_student4
from self_audit.candidate_q3d.teacher_features import (
    forward_teacher_with_features,
    teacher_feature_source,
)
from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher
from self_audit.candidate_q3d.tokenizer import TeacherTokenizer
from scripts.q3d_preflight import build_report


def _padded_size(size: int) -> int:
    return max(32, ((size + 15) // 16) * 16)


def _phase(stride: int) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    return (stride, stride, stride), (0.0, 0.0, 0.0)


@pytest.mark.parametrize("depth", [6, 9, 10, 13, 20])
def test_alignment_shapes_constant_features_and_detach(depth: int) -> None:
    height, width = 35, 41
    for stride in (4, 8, 16):
        source_shape = tuple(_padded_size(dim) // stride for dim in (depth, height, width))
        source = torch.full((2, 3, *source_shape), 2.75, requires_grad=True)
        aligned = align_teacher_to_student4(
            source, (depth, height, width), _phase(stride),
            ((1, 16, 16), (0.0, 0.0, 0.0)),
        )
        assert aligned.shape == (2, 3, depth, math.ceil(height / 16), math.ceil(width / 16))
        assert not aligned.requires_grad
        torch.testing.assert_close(aligned, torch.full_like(aligned, 2.75))


@pytest.mark.parametrize("stride", [4, 8, 16])
def test_alignment_linear_cell_centers_are_exact_for_full_cells(stride: int) -> None:
    depth, height, width = 20, 32, 48
    source_shape = tuple(_padded_size(dim) // stride for dim in (depth, height, width))
    z = torch.arange(source_shape[0], dtype=torch.float64) * stride
    y = torch.arange(source_shape[1], dtype=torch.float64) * stride
    x = torch.arange(source_shape[2], dtype=torch.float64) * stride
    source = (z[:, None, None] + y[None, :, None] + x[None, None, :])[None, None]

    aligned = align_teacher_to_student4(
        source, (depth, height, width), _phase(stride),
        ((1, 16, 16), (0.0, 0.0, 0.0)),
    )
    expected_z = torch.arange(depth, dtype=torch.float64).clamp(max=(source_shape[0] - 1) * stride)
    expected_y = torch.arange(height // 16, dtype=torch.float64) * 16
    expected_x = torch.arange(width // 16, dtype=torch.float64) * 16
    expected = (
        expected_z[:, None, None]
        + expected_y[None, :, None]
        + expected_x[None, None, :]
    )[None, None]
    torch.testing.assert_close(aligned, expected, atol=1e-5, rtol=0)


def test_alignment_stride8_samples_native_centers_at_odd_edges() -> None:
    depth, height, width = 10, 17, 25
    stride = 8
    source = torch.zeros(
        1, 1, _padded_size(depth) // stride,
        _padded_size(height) // stride, _padded_size(width) // stride,
    )
    # S4 centers are native indices 0 and 16, so the final y/x target reads
    # teacher feature index 2 rather than averaging an offset 2x2 block.
    source[..., 2, 2] = 6
    aligned = align_teacher_to_student4(
        source, (depth, height, width), _phase(stride),
        ((1, 16, 16), (0.0, 0.0, 0.0)),
    )
    assert aligned.shape == (1, 1, depth, 2, 2)
    torch.testing.assert_close(aligned[..., 1, 1], torch.full((1, 1, depth), 6.0))


def test_alignment_accepts_single_z_feature_when_native_depth_is_covered() -> None:
    source = torch.full((1, 1, 1, 4, 4), 2.5)
    aligned = align_teacher_to_student4(
        source, (6, 17, 17), _phase(8), ((1, 16, 16), (0.0, 0.0, 0.0)),
    )
    assert aligned.shape == (1, 1, 6, 2, 2)
    torch.testing.assert_close(aligned, torch.full_like(aligned, 2.5))


def test_alignment_rejects_partial_inplane_pooling_cells() -> None:
    source = torch.ones(1, 1, 1, 2, 4)
    with pytest.raises(ValueError, match="does not cover native_shape"):
        align_teacher_to_student4(
            source, (6, 17, 17), _phase(8), ((1, 16, 16), (0.0, 0.0, 0.0)),
        )


def test_alignment_rejects_uncovered_or_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="step values"):
        align_teacher_to_student4(
            torch.ones(1, 1, 4, 4, 4), (10, 17, 17),
            ((0, 8, 8), (0.0, 0.0, 0.0)), ((1, 16, 16), (0.0, 0.0, 0.0)),
        )
    with pytest.raises(ValueError, match="does not cover native_shape"):
        align_teacher_to_student4(
            torch.ones(1, 1, 1, 4, 4), (10, 17, 17), _phase(8),
            ((1, 16, 16), (0.0, 0.0, 0.0)),
        )


def _small_tokenizer() -> TeacherTokenizer:
    return TeacherTokenizer(
        3,
        {
            "q3d": {
                "tokenizer": {
                    "codebook_size": 8,
                    "dimension": 4,
                    "adapter_hidden": 8,
                    "initialization_volumes": 2,
                    "initialization_points_per_volume": 64,
                    "history_window": 2,
                    "min_perplexity": 0.0,
                    "max_dead_code_fraction": 1.0,
                    "min_class_codes": 1,
                },
            },
        },
    )


def test_tokenizer_kmeans_init_losses_gradient_boundary_and_gate() -> None:
    torch.manual_seed(7)
    tokenizer = _small_tokenizer()
    feature = torch.randn(1, 3, 2, 4, 4)
    class_probs = torch.softmax(torch.randn(1, 4, 2, 4, 4), dim=1)

    first = tokenizer(feature, class_probs)
    assert first["initialized"] is False
    assert first["pi"] is None
    assert int(tokenizer.initialization_volume_count.item()) == 1
    assert tokenizer.check_gate()["status"] == "NOT_READY"
    with pytest.raises(RuntimeError, match="before PASS gate"):
        tokenizer.freeze()

    output = tokenizer(feature + 0.1, class_probs)
    assert output["initialized"] is True
    assert output["pi"].shape == (1, 8, 2, 4, 4)
    assert output["codes"].shape == (1, 2, 4, 4)
    assert output["quantized"].shape == (1, 4, 2, 4, 4)
    assert tokenizer.C0.shape == (8, 4)
    torch.testing.assert_close(tokenizer.C0.norm(dim=1), torch.ones(8), atol=1e-5, rtol=0)
    assert "sha256" in tokenizer.artifact_metadata()
    assert "C0" not in dict(tokenizer.named_parameters())

    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in tokenizer.adapter.parameters())
    assert tokenizer.C0.grad is None
    for _ in range(2):
        tokenizer.update_codebook(output["u"].detach(), output["codes"])
    metrics = tokenizer.metrics(codes=output["codes"], class_probs=class_probs, snapshot_id="fixture")
    assert metrics["ema_updates"] == 2
    assert metrics["perplexity"] is not None
    assert metrics["dead_code_fraction"] is not None
    assert metrics["class_code_coverage"] is not None
    assert tokenizer.metrics(codes=output["codes"], class_probs=class_probs, snapshot_id="fixture")["churn"] == 0.0
    gate = tokenizer.check_gate(metrics)
    assert gate["status"] in {"PASS", "NO_GO"}
    assert gate["passed"] == (gate["status"] == "PASS")

    gate_codes = torch.arange(8).reshape(1, 2, 2, 2)
    gate_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).reshape(1, 2, 2, 2)
    gate_probs = torch.nn.functional.one_hot(gate_labels, num_classes=4)
    gate_probs = gate_probs.permute(0, 4, 1, 2, 3).float()
    tokenizer.metrics(codes=gate_codes, class_probs=gate_probs, snapshot_id="gate-1")
    gate_metrics = tokenizer.metrics(codes=gate_codes, class_probs=gate_probs, snapshot_id="gate-2")
    assert tokenizer.check_gate(gate_metrics)["status"] == "PASS"
    frozen_metadata = tokenizer.freeze()
    assert frozen_metadata["frozen"] is True
    assert frozen_metadata["train_heads_present"] is False
    assert tokenizer.reconstructor is None and tokenizer.anatomy_head is None
    assert not any(name.startswith(("reconstructor.", "anatomy_head.")) for name in tokenizer.state_dict())

    expected_parameters = {name: value.detach().clone() for name, value in tokenizer.named_parameters()}
    expected_buffers = {name: value.detach().clone() for name, value in tokenizer.named_buffers()}
    reloaded = _small_tokenizer()
    reloaded.load_state_dict(tokenizer.state_dict())
    assert reloaded.codebook_frozen
    assert reloaded.reconstructor is None and reloaded.anatomy_head is None
    assert not reloaded.training
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    assert set(dict(reloaded.named_parameters())) == set(expected_parameters)
    assert set(dict(reloaded.named_buffers())) == set(expected_buffers)
    for name, expected in expected_parameters.items():
        assert torch.equal(dict(reloaded.named_parameters())[name], expected)
    for name, expected in expected_buffers.items():
        assert torch.equal(dict(reloaded.named_buffers())[name], expected)
    reloaded.train()
    assert not reloaded.training
    assert all(not module.training for module in reloaded.modules())


def test_direct_canonical_tokenizer_config_is_consumed_and_unknown_keys_fail() -> None:
    tokenizer = TeacherTokenizer(
        3,
        {
            "codebook_size": 8,
            "dimension": 5,
            "adapter_hidden": 7,
            "usage_window_updates": 3,
            "anatomy_loss_weight": 0.2,
            "minimum_perplexity": 11.0,
            "maximum_dead_code_fraction": 0.4,
            "minimum_class_codes": 2,
            "dead_code_fraction": 0.007,
        },
    )
    assert tokenizer.embedding_dim == 5
    assert tokenizer.hidden_channels == 7
    assert tokenizer.history_window == 3
    assert tokenizer.anatomy_weight == 0.2
    assert tokenizer.min_perplexity == 11.0
    assert tokenizer.max_dead_code_fraction == 0.4
    assert tokenizer.min_class_codes == 2
    assert tokenizer.dead_code_threshold == 0.007
    assert tokenizer.initialization_volumes == 200
    assert tokenizer.initialization_points_per_volume == 1024
    with pytest.raises(ValueError, match="unknown tokenizer config keys"):
        TeacherTokenizer(3, {"codebook_size": 8, "mystery_option": True})


def test_class_code_coverage_uses_soft_mass_over_persisted_cohort_window() -> None:
    tokenizer = _small_tokenizer()
    first_codes = torch.zeros(1, 1, 1, 6, dtype=torch.long)
    first_probs = torch.tensor([0.1, 0.6, 0.1, 0.2]).view(1, 4, 1, 1, 1).expand(
        1, 4, 1, 1, 6,
    ).clone()
    first = tokenizer.metrics(
        codes=first_codes, class_probs=first_probs, cohort_id="cohort-a",
    )
    assert first["class_code_coverage"]["RV"] == 1
    assert first["class_mass_cohort_count"] == 1

    resumed = _small_tokenizer()
    resumed.load_state_dict(tokenizer.state_dict())
    second_codes = torch.tensor([[[[0, 1]]]], dtype=torch.long)
    # Code 0 remains RV-major after combining both cohorts. Code 1 has RV as
    # its plurality class (0.36 > 0.34) but no class owns over half its mass.
    second_probs = torch.tensor([[[[[0.05, 0.10]]], [[[0.05, 0.36]]],
                                  [[[0.80, 0.34]]], [[[0.10, 0.20]]]]])
    metrics = resumed.metrics(
        codes=second_codes, class_probs=second_probs, cohort_id="cohort-b",
    )
    assert metrics["class_mass_cohort_count"] == 2
    assert metrics["class_code_coverage"] == {"BG": 0, "RV": 1, "MYO": 0, "LV": 0}
    assert resumed._class_mass_cohort_ids == ["cohort-a", "cohort-b"]

    repeated = resumed.metrics(
        codes=second_codes, class_probs=second_probs, cohort_id="cohort-b",
    )
    assert repeated["class_mass_cohort_count"] == 2
    assert repeated["class_code_coverage"] == metrics["class_code_coverage"]


def test_dead_code_restart_generator_state_resumes_exactly() -> None:
    options = {
        "codebook_size": 8,
        "dimension": 4,
        "adapter_hidden": 8,
        "usage_window_updates": 1,
        "dead_code_fraction": 0.001,
        "initialization_volumes": 1,
        "initialization_points_per_volume": 8,
    }
    tokenizer = TeacherTokenizer(4, options)
    torch.manual_seed(33)
    centers = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    tokenizer.C0.copy_(centers)
    tokenizer.ema_counts.fill_(1.0)
    tokenizer.ema_sums.copy_(centers)
    tokenizer.is_initialized.fill_(True)
    u = torch.nn.functional.normalize(torch.randn(1, 4, 1, 1, 8), dim=1)
    codes = torch.zeros(1, 1, 1, 8, dtype=torch.long)

    tokenizer.update_codebook(u, codes)
    resumed = TeacherTokenizer(4, options)
    resumed.load_state_dict(tokenizer.state_dict())
    tokenizer.update_codebook(u.flip(-1), codes)
    resumed.update_codebook(u.flip(-1), codes)
    assert torch.equal(tokenizer.C0, resumed.C0)
    assert torch.equal(tokenizer._restart_generator.get_state(), resumed._restart_generator.get_state())


def _small_resenc_teacher() -> Q3DResEncTeacher:
    """Small six-stage ResEnc fixture that preserves S8/S16 phase semantics."""
    teacher = Q3DResEncTeacher(
        synthetic=True,
        strides=(
            (1, 1, 1), (2, 2, 2), (2, 2, 2),
            (2, 2, 2), (2, 2, 2), (1, 1, 1),
        ),
    )
    return teacher.eval()


def _resenc_geometry() -> dict[str, object]:
    return {
        "shape": (17, 17, 19),
        "spacing_zyx": (5.0, 1.2, 1.2),
        "affine_zyx": torch.eye(4),
        "context": "full_volume",
    }


def test_teacher_feature_capture_uses_one_native_resenc_forward() -> None:
    torch.manual_seed(11)
    teacher = _small_resenc_teacher()
    image = torch.rand(1, 1, 17, 17, 19)
    geometry = _resenc_geometry()
    calls = 0
    original = teacher.forward_with_features

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    teacher.forward_with_features = counted
    source = teacher_feature_source(teacher, "encoder_stride8", synthetic=True)
    assert source["feature_index"] == 3
    assert source["module_name"] == "encoder.stages.3"
    assert source["channels"] == 16
    assert source["capture_api"] == "forward_with_features"
    assert source["module"] is None
    captured = forward_teacher_with_features(
        teacher, image, geometry=geometry, source="encoder_stride8", synthetic=True,
    )
    assert calls == 1
    assert set(captured) == {"region_logits", "boundary_logits", "sdf", "feature"}
    assert captured["region_logits"].shape == (1, 4, 17, 17, 19)
    assert captured["feature"].shape == (1, 16, 4, 4, 4)
    assert captured["feature"].requires_grad is False


def test_resenc_feature_source_selects_declared_encoder_stages_only() -> None:
    teacher = _small_resenc_teacher()
    stride8 = teacher_feature_source(teacher, "stride8", synthetic=True)
    stride16 = teacher_feature_source(teacher, "encoder_stride16", synthetic=True)
    assert stride8["source"] == "encoder_stride8"
    assert stride8["phase"] == ((8, 8, 8), (0.0, 0.0, 0.0))
    assert stride16["source"] == "encoder_stride16"
    assert stride16["feature_index"] == 4
    assert stride16["channels"] == 20
    with pytest.raises(ValueError, match="encoder_stride8 or encoder_stride16"):
        teacher_feature_source(teacher, "decoder_stride8", synthetic=True)


def test_p0_preflight_reports_resenc_source_and_patient_diagnostics_scope() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = build_report(root / "configs/candidate_q3d/v1.json")
    assert report["config_validation"]["status"] == "PASS"
    assert report["status"] == "NOT_READY"
    checkpoint_status = report["teacher_feature"]["checkpoint"]["status"]
    source_status = report["teacher_feature"]["source_status"]
    if checkpoint_status == "HASH_VERIFIED_FILE_PRESENT":
        assert source_status == "ACTUAL_RESENC_MODULES_INSPECTED"
        assert report["teacher_feature"]["weights"]["status"] == "HASH_VERIFIED_FILE_PRESENT"
    elif source_status == "PHASE0_RECEIPT_ONLY":
        assert checkpoint_status == "NOT_RUN_CHECKPOINT_MISSING"
        assert report["teacher_feature"]["weights"]["status"] == "NOT_RUN_CHECKPOINT_MISSING"
    else:
        assert source_status == "NOT_RUN"
        assert report["teacher_feature"]["weights"]["status"] in {
            "NOT_RUN_WEIGHTS_FILE_MISSING", "NOT_RUN_NO_WEIGHTS_PATH_OR_SHA",
            "NOT_RUN_CHECKPOINT_MISSING",
        }
    assert report["datasets"]["acdc"]["status"] == "NOT_RUN"
    assert report["datasets"]["mnms"]["status"] == "NOT_RUN"
    assert report["clinical_gates"]["status"] == "NOT_RUN"
    assert report["near_far_sdf"]["q3d_teacher_loss"]["uses_balanced_near_far"] is True
    assert report["near_far_sdf"]["legacy_a3d_teacher_loss_uses_near_far"] is False
    assert report["student_parameters"]["status"] == "PASS"
    assert report["student_parameters"]["within_budget"] is True
