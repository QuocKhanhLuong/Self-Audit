from __future__ import annotations

import pytest

from self_audit.candidate_q3d.erf import run_resenc_erf
from self_audit.candidate_q3d.teacher_resenc import Q3DResEncTeacher


def _teacher() -> Q3DResEncTeacher:
    return Q3DResEncTeacher(synthetic=True).eval()


def test_resenc_erf_runs_sixteen_samples_on_same_encoder_prefix() -> None:
    result = run_resenc_erf(_teacher(), samples=16, seed=42, input_shape=(9, 17, 17))
    assert result["schema"] == "self_audit.q3d_resenc.erf_probe.v1"
    assert result["status"] in {"PASS_MAIN_PHASE", "MEASURED_OFFSET_REQUIRED"}
    assert result["compute"]["samples"] == 16
    assert result["compute"]["random_zscore_input"] is True
    assert result["compute"]["patient_data"] == "NOT_USED"
    assert result["compute"]["padding_mode"] == "replicate"
    assert result["model"]["stage_index"] == 3
    assert result["model"]["stage_name"] == "encoder.stages[3]"
    assert result["model"]["stage_channels"] == 16
    assert result["model"]["cumulative_stride_zyx"] == [8, 8, 8]
    assert len(result["samples"]) == 16
    assert all(row["channel_aggregation"] == "signed_mean_all_stage_channels" for row in result["samples"])
    assert all("signed_mean_cell_activation" in row for row in result["samples"])
    assert all(row["channel_count"] == 16 for row in result["samples"])
    assert len(result["summary"]["mean_centroid_offset_voxels"]) == 3
    assert len(result["summary"]["proposed_phase_offset_zyx"]) == 3
    assert result["proposed_offset_zyx"] == result["summary"]["proposed_phase_offset_zyx"]
    assert result["selected_phase"]["step_zyx"] == [8, 8, 8]


def test_resenc_erf_ignores_configured_phase_correction_for_nominal_comparison() -> None:
    teacher = Q3DResEncTeacher(
        synthetic=True,
        phase_offsets={"encoder_stride8": (1.0, 2.0, 3.0)},
    ).eval()
    result = run_resenc_erf(teacher, samples=16, seed=42, input_shape=(9, 17, 17))
    assert result["model"]["configured_phase"][1] == (1.0, 2.0, 3.0)
    assert result["model"]["phase_reference"].startswith("theoretical main convolution path")
    assert result["samples"][0]["nominal_main_phase_center_voxels"] == [8.0, 8.0, 8.0]


@pytest.mark.parametrize("samples", [0, 1, 15])
def test_resenc_erf_requires_minimum_sample_count(samples: int) -> None:
    with pytest.raises(ValueError, match="at least 16"):
        run_resenc_erf(_teacher(), samples=samples, input_shape=(9, 17, 17))


def test_resenc_erf_does_not_leave_parameter_gradients() -> None:
    teacher = _teacher()
    run_resenc_erf(teacher, samples=16, seed=42, input_shape=(9, 17, 17))
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_resenc_erf_rejects_non_integer_native_shape() -> None:
    with pytest.raises(ValueError, match="positive D,H,W"):
        run_resenc_erf(_teacher(), samples=16, input_shape=(9.5, 17, 17))
