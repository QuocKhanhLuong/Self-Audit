"""Runtime schema and native-geometry safeguards for diagnostic B."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from self_audit.gradient_bridge_metrics import GATE_SCHEMA
from self_audit.gradient_bridge_runtime import (
    SCHEMA,
    _assert_a_b_match,
    _feature_support,
    _fit_valid_support,
    _seeded_patient_records,
    validate_config,
)


_SHA = "a" * 64


def _config(*, profile="fixture", confidence=0.0):
    return {
        "schema": SCHEMA,
        "profile": profile,
        "synthetic_fixture": profile == "fixture",
        "device": "cpu",
        "threads": 1,
        "batch_size": 2,
        "normalization": "prepared",
        "teacher": {
            "path": "teacher",
            "manifest_sha256": _SHA,
            "payload_sha256": "b" * 64,
            "runtime_root": ".",
            "runtime_source_sha256": "c" * 64,
        },
        "student": {
            "path": "student",
            "manifest_sha256": "d" * 64,
            "payload_sha256": "e" * 64,
            "runtime_root": ".",
            "runtime_source_sha256": "f" * 64,
        },
        "train_manifest": {"path": "train.json", "sha256": "1" * 64},
        "val_manifest": {"path": "val.json", "sha256": "2" * 64},
        "parity": {"logit_atol": 1e-6, "logit_rtol": 1e-5},
        "fit_p": {
            "seed": 42,
            "max_source_ids": 4,
            "max_train_patients": 2,
            "max_pixels": 256,
            "ridge": 1e-4,
            "fit_loss": "patient_balanced_pre_refine0_feature_mse",
            "alignment_mode": "bilinear_align_corners_false_then_1x1_affine",
            "output_file": "alignment.json",
        },
        "kd": {
            "temperature": 1.0,
            "coefficient": 1.0,
            "confidence": confidence,
            "class_weights": None,
            "denominator": "valid_pixel_count",
        },
        "directions": {
            "rho_primary": 0.01,
            "rho_sensitivity": [0.0, 0.01],
            "norm_epsilon": 1e-8,
            "shuffle_seed": 17,
        },
        "gate": {
            "schema": GATE_SCHEMA,
            "minimum_fg_absolute_delta": 0.0,
            "bootstrap": {"n_resamples": 10, "seed": 17},
            "confidence_rule": {
                "method": "paired_percentile",
                "level": 0.95,
                "minimum_defined_fraction": 1.0,
                "acceptance": "lower_bound_strictly_above_minimum",
            },
            "loo_rule": {"method": "all_deltas_at_least_minimum", "require_all_defined": True},
        },
    }


def test_runtime_schema_accepts_explicit_fixture_science_config():
    assert validate_config(_config())["schema"] == SCHEMA


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["kd"].update(confidence=None), "explicitly selected"),
        (lambda value: value["directions"].update(rho_primary=None), "rho_primary"),
        (lambda value: value["gate"]["bootstrap"].update(n_resamples=None), "n_resamples"),
    ],
)
def test_unfilled_template_thresholds_fail_closed(mutate, match):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        validate_config(config)


def test_historical_profile_keeps_the_pinned_plain_kd_objective():
    config = _config(profile="historical")
    config["synthetic_fixture"] = False
    config["kd"]["confidence"] = 0.6
    with pytest.raises(ValueError, match="historical B8/wide plain-KD"):
        validate_config(config)


def test_patient_order_uses_seeded_order_and_keeps_each_identity_once():
    rows = [{"id": f"train-{i}"} for i in range(8)]
    first = _seeded_patient_records(rows, 42)
    assert first == _seeded_patient_records(rows, 42)
    assert {row["id"] for row in first} == {row["id"] for row in rows}
    with pytest.raises(ValueError, match="duplicate TRAIN patient"):
        _seeded_patient_records([{"id": "same"}, {"id": "same"}], 42)


def test_alignment_support_false_pads_internal_geometry_before_downsample():
    native = torch.ones((32, 31), dtype=torch.bool)
    support = _fit_valid_support(native, (32, 31), (8, 8))
    assert support.shape == (1, 8, 8)
    assert support[0, :, :-1].all()
    assert not support[0, :, -1].any()


def test_prediction_feature_support_uses_padded_hw_and_full_cell_support():
    valid = torch.ones((1, 32, 31), dtype=torch.bool)
    feature = torch.ones((1, 4, 1, 1))
    context = SimpleNamespace(
        h=feature,
        valid_mask=valid,
        input_hw=(32, 31),
        padded_hw=(32, 32),
    )
    support = _feature_support(context)
    assert support.shape == (1, 1, 1)
    assert not support.any()


def test_stage_a_and_b_must_share_pair_cohort_and_preprocessing():
    config = _config()
    a_config = {
        "teacher": config["teacher"],
        "student": config["student"],
        "val_manifest": config["val_manifest"],
        "synthetic_fixture": True,
        "normalization": "prepared",
        "parity": config["parity"],
        "device": "cpu",
    }
    identity = _assert_a_b_match({"config": a_config}, config)
    assert identity["evaluation_manifest_sha256"] == config["val_manifest"]["sha256"]

    bad = dict(a_config)
    bad["student"] = {**config["student"], "payload_sha256": "0" * 64}
    with pytest.raises(ValueError, match="student.payload_sha256"):
        _assert_a_b_match({"config": bad}, config)

    bad = dict(a_config)
    bad["val_manifest"] = {"sha256": "9" * 64}
    with pytest.raises(ValueError, match="evaluation manifest"):
        _assert_a_b_match({"config": bad}, config)


def test_historical_a_template_without_fixture_flag_matches_historical_b():
    template_path = Path(__file__).resolve().parents[1] / "configs" / "gradient_bridge_historical_val18.template.json"
    a_config = json.loads(template_path.read_text())
    # Simulate the older persisted A config while retaining its checked-in
    # historical model, cohort, preprocessing, parity, and device identities.
    a_config.pop("synthetic_fixture", None)
    assert "synthetic_fixture" not in a_config
    config = _config(profile="historical")
    config["device"] = a_config["device"]
    config["parity"] = dict(a_config["parity"])
    config["val_manifest"]["sha256"] = a_config["val_manifest"]["sha256"]
    for role in ("teacher", "student"):
        config[role].update({
            key: a_config[role][key]
            for key in ("manifest_sha256", "payload_sha256", "runtime_source_sha256")
        })

    identity = _assert_a_b_match({"config": a_config}, config)
    assert identity["synthetic_fixture"] is False
    assert identity["device"] == "cuda"
    assert identity["parity"] == a_config["parity"]

    bad = dict(config)
    bad["parity"] = {**config["parity"], "logit_atol": 1e-4}
    with pytest.raises(ValueError, match="parity config"):
        _assert_a_b_match({"config": a_config}, bad)
    bad = dict(config)
    bad["device"] = "cpu"
    with pytest.raises(ValueError, match="locked device"):
        _assert_a_b_match({"config": a_config}, bad)
