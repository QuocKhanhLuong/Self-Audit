from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import scripts.q3d_preflight as preflight


def _conv(kernel, stride, padding):
    return nn.Conv3d(
        1, 1, kernel_size=kernel, stride=stride, padding=padding,
        bias=False,
    )


class _ResEncEncoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = _conv(3, 1, 1)
        self.stages = nn.ModuleList([
            nn.ModuleList([_conv(3, 1, 1)]),
            nn.ModuleList([_conv(3, 2, 1), _conv(3, 1, 1)]),
            nn.ModuleList([_conv(3, 2, 1), _conv(3, 1, 1)]),
            nn.ModuleList([_conv(3, 2, 1), _conv(3, 1, 1)]),
            nn.ModuleList([_conv(3, 2, 1), _conv(3, 1, 1)]),
            nn.ModuleList([_conv(3, 2, 1), _conv(3, 1, 1)]),
        ])


class _ResEncTeacherFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _ResEncEncoderFixture()


class _WrappedConv(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.conv = _conv(3, stride, 1)


class _ResEncPhaseFixture(nn.Module):
    def __init__(self):
        super().__init__()
        stem = nn.Module()
        stem.convs = nn.ModuleList([_WrappedConv((1, 1, 1))])
        self.stem = stem
        stages = []
        for stage_index in range(4):
            block = nn.Module()
            stride = (1, 1, 1) if stage_index == 0 else (2, 2, 2)
            block.conv1 = _WrappedConv(stride)
            block.conv2 = _WrappedConv((1, 1, 1))
            if stage_index > 0:
                block.skip = nn.Sequential(
                    nn.AvgPool3d(kernel_size=stride, stride=stride, padding=0),
                    _WrappedConv((1, 1, 1)),
                )
            stage = nn.Module()
            stage.blocks = nn.ModuleList([block])
            stages.append(stage)
        self.stages = nn.ModuleList(stages)


def _toy_student_for_phase():
    slot = lambda stride: SimpleNamespace(
        conv=_conv((1, 3, 3), stride, (0, 1, 1)),
    )
    return SimpleNamespace(
        stem=[slot((1, 1, 1))],
        stage1=[slot((1, 2, 2))],
        stage2=[slot((1, 2, 2))],
        stage3=SimpleNamespace(down=slot((1, 2, 2))),
        stage4=SimpleNamespace(down=slot((1, 2, 2))),
    )


class _LOnlyDataFixture:
    def __init__(self, labels: torch.Tensor):
        self.labels = labels
        self.unlabeled_accessed = False

    def labeled_frame_keys(self):
        return [(0, 0)]

    def load_labeled_frame(self, key):
        assert key == (0, 0)
        return None, self.labels[None], {}, {"patient": "train_L_001", "frame": 0}

    def unlabeled_frame_keys(self):
        self.unlabeled_accessed = True
        raise AssertionError("preflight must not access U frame keys")

    def load_unlabeled_frame(self, key):
        self.unlabeled_accessed = True
        raise AssertionError("preflight must not read U frames or masks")


def test_phase_json_preserves_fractional_offsets() -> None:
    phase_json = preflight._phase_json(((1, 16, 16), (0, 0.5, -0.25)))
    assert phase_json == {
        "step_zyx": [1, 16, 16],
        "offset_zyx": [0, 0.5, -0.25],
    }
    assert preflight._phase_tuple_from_json(phase_json) == (
        (1, 16, 16), (0.0, 0.5, -0.25),
    )


def test_erf_defaults_are_center_aligned_for_locked_strides() -> None:
    defaults = preflight._erf_geometry_defaults({
        "encoder_strides": [
            [1, 1, 1], [2, 2, 2], [1, 2, 2],
            [1, 2, 2], [1, 2, 2], [1, 2, 2],
        ],
    })

    assert defaults == {
        "input_shape": (33, 97, 97),
        "input_center": (16, 48, 48),
        "source_step": (2, 8, 8),
        "source_cell": (8, 6, 6),
    }


def test_resenc_inventory_reports_actual_encoder_and_cpu_diagnostics() -> None:
    class _API:
        @staticmethod
        def geometry_diagnostics(**_kwargs):
            return {"status": "MEASURED", "native_grid": "declared"}

        @staticmethod
        def phase_probe(**_kwargs):
            return {"status": "PASS", "locked_offset": [0, 0, 0]}

        @staticmethod
        def erf_diagnostics(**_kwargs):
            return {"status": "MEASURED", "samples": 16, "threshold": 2.0}

    inventory = preflight._resenc_network_inventory(
        _ResEncTeacherFixture(), api=_API, config={"model": {}},
        checkpoint_inventory={"path": "/tmp/checkpoint", "sha256": "a" * 64},
    )

    assert inventory["status"] == "ACTUAL_RESENC_MODULES_INSPECTED"
    assert inventory["architecture"] == "ResEncL"
    assert inventory["encoder_conv_count"] == 12
    assert inventory["geometry"]["status"] == "MEASURED"
    assert inventory["phase"]["status"] == "PASS"
    assert inventory["erf"] == {"status": "MEASURED", "samples": 16, "threshold": 2.0}


def test_resenc_inventory_fails_closed_without_encoder_convolutions() -> None:
    with pytest.raises(ValueError, match="no Conv3d"):
        preflight._resenc_network_inventory(
            SimpleNamespace(encoder=nn.Identity()), api=SimpleNamespace(),
            config={"model": {}},
            checkpoint_inventory={"path": "/tmp/checkpoint", "sha256": "a" * 64},
        )


def test_resenc_inventory_does_not_call_declared_phase_a_pass() -> None:
    teacher = _ResEncTeacherFixture()
    teacher.feature_phases = {"encoder_stride8": ((8, 8, 8), (0.0, 0.0, 0.0))}

    inventory = preflight._resenc_network_inventory(
        teacher, api=SimpleNamespace(), config={"model": {}},
        checkpoint_inventory={"path": "/tmp/checkpoint", "sha256": "a" * 64},
    )

    assert inventory["phase"]["status"] == "DECLARED_PHASE_ONLY"


def test_resenc_phase_inventory_reports_residual_skip_offset() -> None:
    phase = preflight._resenc_phase_inventory(_ResEncPhaseFixture())

    assert phase is not None
    assert phase["status"] == "THEORETICAL_SKIP_OFFSET_REPORTED"
    assert phase["main_path"] == {
        "step_zyx": [8, 8, 8], "offset_zyx": [0, 0, 0],
    }
    assert phase["residual_skip_path"] == {
        "step_zyx": [8, 8, 8], "offset_zyx": [3.5, 3.5, 3.5],
    }
    assert phase["max_abs_residual_skip_offset_voxels"] == pytest.approx(3.5)


def test_resenc_inventory_executes_worker_owned_erf_api(monkeypatch) -> None:
    teacher = _ResEncTeacherFixture()
    teacher.feature_phases = {"encoder_stride8": ((8, 8, 8), (0.0, 0.0, 0.0))}
    teacher.cumulative_stride = (32, 32, 32)
    calls = {}

    class _ERFAPI:
        @staticmethod
        def run_resenc_erf(teacher, *, samples, seed, input_shape):
            calls.update({"teacher": teacher, "samples": samples, "seed": seed,
                          "input_shape": input_shape})
            return {
                "status": "PASS_MAIN_PHASE",
                "summary": {"mean_centroid_offset_voxels": [0.0, 0.0, 0.0]},
            }

    monkeypatch.setattr(preflight, "_resenc_erf_api", lambda: _ERFAPI)
    inventory = preflight._resenc_network_inventory(
        teacher, api=SimpleNamespace(), config={"model": {"teacher": {}}},
        checkpoint_inventory={"path": "/tmp/checkpoint", "sha256": "a" * 64},
        diagnostic_options={
            "run_erf": True, "erf_samples": 16, "erf_threshold": 2.0,
            "input_shape": (33, 97, 97), "seed": 42,
        },
    )

    assert calls == {
        "teacher": teacher, "samples": 16, "seed": 42,
        "input_shape": (33, 97, 97),
    }
    assert inventory["erf"]["status"] == "PASS_MAIN_PHASE"
    assert inventory["phase"]["status"] == "EMPIRICAL_MAIN_PHASE_SELECTED"


def test_preflight_uses_repo_receipt_without_promoting_missing_weights() -> None:
    result = preflight._teacher_diagnostics({
        "model": {
            "teacher": {
                "architecture": "nnfoundation_resenc_l",
                "encoder_strides": [
                    [1, 1, 1], [2, 2, 2], [1, 2, 2],
                    [1, 2, 2], [1, 2, 2], [1, 2, 2],
                ],
            },
            "student": {"input_channels": 1},
        },
    })

    assert result["source_status"] == "PHASE0_RECEIPT_ONLY"
    assert result["checkpoint"]["status"] == "NOT_RUN_CHECKPOINT_MISSING"
    assert result["checkpoint"]["strict_encoder"]["status"] == "PASS_RECEIPT_ONLY"
    assert result["resenc_source_inventory"]["status"] == "PHASE0_RECEIPT_VERIFIED"
    assert result["resenc_source_inventory"]["network"]["erf"]["status"] == "PASS_RECEIPT_ONLY"


def test_teacher_weights_runtime_binding_is_resolved(monkeypatch, tmp_path) -> None:
    bound = tmp_path / "checkpoint.pth"
    bound.write_bytes(b"fixture")
    seen = {}

    def inventory(path, expected_sha256, inspection_path):
        seen["path"] = path
        return {
            "status": "NOT_RUN_CHECKPOINT_MISSING",
            "strict_encoder": {"status": "NOT_RUN"},
        }

    monkeypatch.setenv("Q3D_NNFOUNDATION_WEIGHTS", str(bound))
    monkeypatch.setattr(preflight, "_resenc_checkpoint_inventory", inventory)
    result = preflight._teacher_diagnostics({
        "model": {
            "teacher": {
                "architecture": "nnfoundation_resenc_l",
                "pretrained": {"weights_path": "${Q3D_NNFOUNDATION_WEIGHTS}"},
            },
        },
    })

    assert seen["path"] == bound.resolve()
    assert result["weights"]["path"] == str(bound.resolve())


def test_run_erf_never_silently_uses_receipt_for_missing_checkpoint() -> None:
    result = preflight._teacher_diagnostics({
        "model": {
            "teacher": {
                "architecture": "nnfoundation_resenc_l",
                "encoder_strides": [
                    [1, 1, 1], [2, 2, 2], [1, 2, 2],
                    [1, 2, 2], [1, 2, 2], [1, 2, 2],
                ],
            },
            "student": {"input_channels": 1},
        },
    }, run_erf=True)

    assert result["source_status"] == "NOT_RUN"
    assert "checkpoint hash" in result["source_reason"]
    assert result["resenc_source_inventory"].get("network") is None


def test_l_gt_stage4_coverage_reads_l_only_and_opens_myo_review(monkeypatch) -> None:
    import self_audit.candidate_a3d.data as a3d_data
    from self_audit.candidate_q3d.alignment import student_grid_phase

    labels = torch.zeros((1, 160, 160), dtype=torch.long)
    labels[:, 72:88, 72:88] = 2
    data = _LOnlyDataFixture(labels)
    monkeypatch.setattr(a3d_data, "CandidateA3DData", lambda *args, **kwargs: data)
    phase = student_grid_phase(_toy_student_for_phase())

    report = preflight._ground_truth_stage4_coverage(
        Path("synthetic-train-manifest.json"), "0" * 64, phase,
    )

    assert report["status"] == "MEASURED_TRAIN_L_GROUND_TRUTH"
    assert report["label_scope"] == "L only; no U, DEV or TEST labels"
    assert report["class_counts"] == {"BG": 99, "RV": 0, "MYO": 1, "LV": 0}
    assert report["class_fractions"]["MYO"] == pytest.approx(0.01)
    assert report["myo_review"]["status"] == "OPEN"
    assert report["myo_review"]["threshold_changed"] is False
    assert report["tie_break_class_id"] == 0
    assert data.unlabeled_accessed is False


def test_stage4_argmax_ties_choose_lowest_class_id() -> None:
    probabilities = torch.tensor([[[[[0.5]]], [[[0.5]]], [[[0.0]]], [[[0.0]]]]])
    assert preflight._gt_probabilities_to_counts(probabilities) == [1, 0, 0, 0]


def test_resenc_checkpoint_inventory_is_fail_closed_on_missing_file(tmp_path) -> None:
    inventory = preflight._resenc_checkpoint_inventory(
        tmp_path / "missing.pth", preflight.RESENC_CHECKPOINT_SHA256,
        tmp_path / "missing-inspection.json",
    )
    assert inventory["status"] == "NOT_RUN_CHECKPOINT_MISSING"


def test_resenc_checkpoint_inventory_requires_strict_encoder_receipt(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(preflight, "_sha256", lambda _path: preflight.RESENC_CHECKPOINT_SHA256)
    inventory = preflight._resenc_checkpoint_inventory(
        checkpoint, preflight.RESENC_CHECKPOINT_SHA256, inspection,
    )
    assert inventory["status"] == "STRICT_ENCODER_INVENTORY_FAILED"
    assert inventory["strict_encoder"]["status"] == "FAIL"


def test_resenc_checkpoint_inventory_rejects_non_object_receipt(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"fixture")
    inspection = tmp_path / "inspection.json"
    inspection.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(preflight, "_sha256", lambda _path: preflight.RESENC_CHECKPOINT_SHA256)

    inventory = preflight._resenc_checkpoint_inventory(
        checkpoint, preflight.RESENC_CHECKPOINT_SHA256, inspection,
    )

    assert inventory["status"] == "HASH_VERIFIED_FILE_PRESENT"
    assert inventory["strict_encoder"] == {
        "status": "INVALID_INSPECTION",
        "reason": "strict checkpoint inspection receipt root is not an object",
    }


def test_report_marks_real_l_gt_coverage_not_run_without_train_manifest(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(preflight, "_teacher_diagnostics", lambda _config, **_kwargs: {
        "source_status": "NOT_RUN",
        "resenc_source_inventory": {"status": "PHASE0_CHECKPOINT_AND_ACTUAL_MODULES_INSPECTED"},
        "checkpoint": {"status": "NOT_RUN_CHECKPOINT_MISSING"},
    })
    monkeypatch.setattr(preflight, "_student_parameter_diagnostics", lambda _config: {
        "status": "PASS",
        "stage4_grid_phase_status": "PASS",
        "stage4_grid_phase": {"step_zyx": [1, 16, 16], "offset_zyx": [0, 0, 0]},
    })
    monkeypatch.setattr(preflight, "_near_far_diagnostics", lambda _config: {})

    report = preflight.build_report(root / "configs/candidate_q3d/v1.json")

    assert report["datasets"]["acdc"]["status"] == "NOT_RUN"
    assert report["datasets"]["acdc"]["labeled_train_stage4_gt_coverage"]["status"] == "NOT_RUN"
    assert report["datasets"]["acdc"]["labeled_train_stage4_gt_coverage"]["reason"] in {
        "TRAIN manifest path or SHA-256 is not declared",
        "Q-3D recipe failed validation",
    }
    assert report["datasets"]["acdc"]["labeled_train_stage4_gt_coverage"]["label_scope"] == \
        "L only; no U, DEV or TEST labels"
    assert report["datasets"]["mnms"]["status"] == "NOT_RUN"
    assert report["clinical_gates"]["status"] == "NOT_RUN"
