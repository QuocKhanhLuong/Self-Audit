import pytest

from self_audit.volume.config import SCHEMA, default_volume_config, validate_volume_config


def test_defaults_are_bounded_and_explicit():
    c = validate_volume_config(default_volume_config())
    assert c["schema"] == SCHEMA
    assert c["training"]["steps"] <= 8
    assert c["training"]["method"] == "ema_full"
    assert c["prediction_model"] == "student"


def test_unknown_fields_and_long_run_fail_closed():
    c = default_volume_config()
    c["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown volume config"):
        validate_volume_config(c)
    c = default_volume_config()
    c["training"]["steps"] = 9
    with pytest.raises(ValueError, match="requires explicit"):
        validate_volume_config(c)
    assert validate_volume_config(c, allow_full_run=True)["training"]["steps"] == 9


def test_amp_is_explicit_opt_in():
    c = default_volume_config()
    c["training"]["precision"] = "amp"
    with pytest.raises(ValueError, match="AMP"):
        validate_volume_config(c)
    assert validate_volume_config(c, allow_full_run=True)["training"]["precision"] == "amp"


def test_unsupported_aliases_and_model_architecture_overlap_fail_closed():
    c = default_volume_config()
    c["optimization"]["recipe"] = "scheduler_only"
    with pytest.raises(ValueError, match="scheduler_only"):
        validate_volume_config(c)
    c = default_volume_config()
    c["training"]["precision"] = "amp_bf16"
    with pytest.raises(ValueError, match="amp_bf16"):
        validate_volume_config(c)
    c = default_volume_config()
    c["model"]["architecture"] = "hybrid"
    with pytest.raises(ValueError, match="architecture fields"):
        validate_volume_config(c)


def test_correction_protocol_is_explicit_and_validated():
    c = default_volume_config()
    c["correction"] = {"protocol": "interactive_extra_supervision", "mode": "image_guided", "dataset_role": "FIXTURE"}
    normalized = validate_volume_config(c)
    assert normalized["correction"]["protocol"] == "interactive_extra_supervision"
    c["correction"]["protocol"] = "infer_from_filename"
    with pytest.raises(ValueError, match="correction.protocol"):
        validate_volume_config(c)


def test_pseudo_view_and_student_mode_are_explicit_policy_fields():
    c = default_volume_config()
    c["training"].update({"pseudo_view": "strong_intensity", "pseudo_student_mode": "eval"})
    normalized = validate_volume_config(c)
    assert normalized["training"]["pseudo_view"] == "strong_intensity"
    assert normalized["training"]["pseudo_student_mode"] == "eval"
    c["training"]["pseudo_view"] = "hidden_augmentation"
    with pytest.raises(ValueError, match="pseudo_view"):
        validate_volume_config(c)


def test_unsupported_active_controls_fail_closed():
    c = default_volume_config()
    c["regions"]["detached"] = False
    with pytest.raises(ValueError, match="detached=false"):
        validate_volume_config(c)
    c = default_volume_config()
    c["temporal"]["window"] = "arbitrary"
    with pytest.raises(ValueError, match="temporal.window"):
        validate_volume_config(c)
    c = default_volume_config()
    c["geometry"]["spacing_xyz_mm"] = [1.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="native_z"):
        validate_volume_config(c)
    c = default_volume_config()
    c["training"]["method"] = "supervised"
    c["regions"].update({"mode": "image", "weight": 0.1})
    with pytest.raises(ValueError, match="unlabeled EMA"):
        validate_volume_config(c)


def test_configured_geometry_requires_operational_spacing_and_fov():
    c = default_volume_config()
    c["geometry"]["resampling"] = "configured"
    with pytest.raises(ValueError, match="spacing_xyz_mm"):
        validate_volume_config(c)
    c["geometry"]["spacing_xyz_mm"] = [1.0, 1.5, 8.0]
    normalized = validate_volume_config(c)
    assert normalized["geometry"]["resampling"] == "configured"
    assert normalized["geometry"]["spacing_xyz_mm"] == [1.0, 1.5, 8.0]

    c["geometry"]["resampling"] = "configured_physical_fov"
    c["geometry"]["fov_xyz_mm"] = None
    with pytest.raises(ValueError, match="fov_xyz_mm"):
        validate_volume_config(c)
    c["geometry"]["fov_xyz_mm"] = [96.0, 96.0, 24.0]
    normalized = validate_volume_config(c)
    assert normalized["geometry"]["fov_xyz_mm"] == [96.0, 96.0, 24.0]


@pytest.mark.parametrize("spacing", [[0.0, 1.0, 1.0], [1.0, float("inf"), 1.0], [1.0, 1.0, float("nan")]])
def test_configured_geometry_rejects_nonfinite_or_nonpositive_spacing(spacing):
    c = default_volume_config()
    c["geometry"].update({"resampling": "configured", "spacing_xyz_mm": spacing})
    with pytest.raises(ValueError, match="geometry.spacing_xyz_mm"):
        validate_volume_config(c)


def test_cli_stop_target_does_not_shorten_planned_horizon():
    c = default_volume_config()
    c["training"]["steps"] = 4
    assert validate_volume_config(c, stop_steps=2)["training"]["steps"] == 4
