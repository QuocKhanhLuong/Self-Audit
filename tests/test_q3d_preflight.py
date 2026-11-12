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


class _PinnedTopologyFixture(nn.Module):
    def __init__(self, *, bad_residual: bool = False):
        super().__init__()
        self.stem = _conv(1, 1, 0)
        for index in range(4):
            block = nn.Module()
            block.conv1 = _conv(3, 2, 1)
            block.conv2 = _conv(1, 1, 0)
            block.conv3 = _conv(1, 1, 0)
            block.res_conv = _conv(2 if bad_residual and index == 2 else 1, 2, 0)
            setattr(self, f"down_{index}", block)
        self.outside_block_checkpointing = True
        self.dec_block_3 = nn.Sequential(*[nn.Identity() for _ in range(8)])
        self.dummy_tensor = nn.Parameter(torch.zeros(1))

    def iterative_checkpoint(self, sequential_block, x):
        for l in sequential_block:
            x = torch.utils.checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x


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


def test_mednext_inventory_reports_actual_main_and_residual_phases(monkeypatch) -> None:
    from self_audit.candidate_q3d import alignment

    model = _PinnedTopologyFixture()
    teacher = SimpleNamespace(encoder=SimpleNamespace(model=model))
    x_res_3_phase = alignment.grid_phase([
        model.stem,
        model.down_0.conv1, model.down_0.conv2, model.down_0.conv3,
        model.down_1.conv1, model.down_1.conv2, model.down_1.conv3,
        model.down_2.conv1, model.down_2.conv2, model.down_2.conv3,
    ])
    monkeypatch.setattr(alignment, "teacher_grid_phase", lambda *args, **kwargs: x_res_3_phase)

    inventory = preflight._mednext_network_inventory(
        teacher, source_repo=Path(__file__).resolve().parents[1],
    )

    assert inventory["stem"]["kernel_zyx"] == [1, 1, 1]
    assert inventory["stem"]["stride_zyx"] == [1, 1, 1]
    assert inventory["stem"]["padding_zyx"] == [0, 0, 0]
    assert len(inventory["down_blocks"]) == 4
    assert inventory["down_blocks"][0]["main"]["downsample"]["kernel_zyx"] == [3, 3, 3]
    assert inventory["down_blocks"][0]["main"]["downsample"]["stride_zyx"] == [2, 2, 2]
    assert inventory["down_blocks"][0]["main"]["downsample"]["phase_native"] == {
        "step_zyx": [2, 2, 2], "offset_zyx": [0, 0, 0],
    }
    assert inventory["down_blocks"][0]["residual"]["kernel_zyx"] == [1, 1, 1]
    assert inventory["down_blocks"][0]["residual"]["stride_zyx"] == [2, 2, 2]
    assert inventory["down_blocks"][0]["phase_match"] is True
    assert inventory["x_res_3_phase_native"] == {
        "step_zyx": [8, 8, 8], "offset_zyx": [0, 0, 0],
    }
    assert inventory["iterative_checkpoint"]["outside_block_checkpointing"] is True
    assert inventory["iterative_checkpoint"]["dec_block_3_child_count"] == 8
    assert inventory["iterative_checkpoint"]["hook_target_for_dec_block_3_output"] == "dec_block_3.7"


def test_mednext_inventory_fails_closed_when_residual_phase_differs(monkeypatch) -> None:
    from self_audit.candidate_q3d import alignment

    model = _PinnedTopologyFixture(bad_residual=True)
    teacher = SimpleNamespace(encoder=SimpleNamespace(model=model))
    monkeypatch.setattr(alignment, "teacher_grid_phase", lambda *args, **kwargs: ((8, 8, 8), (0, 0, 0)))

    with pytest.raises(ValueError, match="down_2 main and residual phases differ"):
        preflight._mednext_network_inventory(
            teacher, source_repo=Path(__file__).resolve().parents[1],
        )


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


def test_pinned_source_missing_is_not_run(monkeypatch, tmp_path) -> None:
    missing = Path("/tmp") / f"q3d-missing-preflight-source-{tmp_path.name}"
    monkeypatch.setattr(preflight, "PINNED_MEDNEXT_SOURCE", missing)

    source, reason = preflight._verified_pinned_source()

    assert source is None
    assert reason == "pinned MedNeXt source checkout is unavailable"


def test_report_marks_real_l_gt_coverage_not_run_without_train_manifest(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(preflight, "_teacher_diagnostics", lambda _config: {
        "source_status": "NOT_RUN",
        "official_source_inventory": {"status": "PINNED_SOURCE_AND_ACTUAL_MODULES_INSPECTED"},
        "weights": {"status": "NOT_RUN_NO_WEIGHTS_PATH_OR_SHA"},
    })
    monkeypatch.setattr(preflight, "_student_parameter_diagnostics", lambda _config: {
        "status": "PASS",
        "stage4_grid_phase_status": "PASS",
        "stage4_grid_phase": {"step_zyx": [1, 16, 16], "offset_zyx": [0, 0, 0]},
    })
    monkeypatch.setattr(preflight, "_near_far_diagnostics", lambda _config: {})

    report = preflight.build_report(root / "configs/candidate_q3d/v1.json")

    assert report["datasets"]["acdc"]["status"] == "NOT_RUN"
    assert report["datasets"]["acdc"]["labeled_train_stage4_gt_coverage"] == {
        "status": "NOT_RUN",
        "reason": "TRAIN manifest path or SHA-256 is not declared",
        "label_scope": "L only; no U, DEV or TEST labels",
    }
    assert report["datasets"]["mnms"]["status"] == "NOT_RUN"
    assert report["clinical_gates"]["status"] == "NOT_RUN"
